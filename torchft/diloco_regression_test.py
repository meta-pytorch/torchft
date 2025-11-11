# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import logging
import os
import sys
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, overload, Tuple
from unittest import skipIf, TestCase

import torch
from parameterized import parameterized
from torch import nn, optim

from torchft._test.diloco_trainer import DiLoCoTrainer, MultiModel
from torchft._torchft import LighthouseServer
from torchft.local_sgd import DiLoCo
from torchft.manager import Manager
from torchft.manager_integ_test import EventInjector, EventInjectorEvent, Runner

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def handle_fixture(
    fixture_filename: str,
    results: list[list[Dict[str, Dict[int, Dict[str, List[float]]]]]],
) -> Optional[list[list[Dict[str, Dict[str, Dict[str, List[float]]]]]]]:
    """
    Handle reading from or writing to a fixture file.

    Args:
        fixture_filename: The name of the fixture file (without path)
        results: The results to write to the fixture file if in write mode

    Returns:
        The fixture data when reading, None when writing
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.dirname(script_directory)

    fixture_path = os.path.join(root_directory, "test_fixtures", fixture_filename)

    write_fixture = os.environ.get("WRITE_FIXTURE", "false").lower() in ("true")

    if write_fixture:
        # Write results to fixture file
        logger.info(f"Writing fixture to {fixture_path}")
        with open(fixture_path, "w+") as f:
            json.dump(results, f, indent=2)
        return None

    # Read fixture file and return the data
    assert os.path.exists(fixture_path), f"Fixture file {fixture_path} does not exist"
    logger.info(f"Validating against fixture at {fixture_path}")
    with open(fixture_path, "r") as f:
        fixture_data = json.load(f)

    return fixture_data


class MockLinear(nn.Module):
    """
    A mock linear layer with deterministic parameter updates.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # Initialize with specific values to make tracking easier
        self.weight = nn.Parameter(torch.ones(out_features, in_features))

        # Fixed gradients for deterministic updates
        self.weight_grad_value = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We don't actually do a forward pass, this should not be called
        raise


class MockModel(MultiModel):
    """
    A mock model with deterministic parameter updates.
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 4, n_layers: int = 1) -> None:
        super().__init__()

        for _ in range(n_layers):
            # We don't care about matching dimensionality, we're not going to pass any
            # input through the model
            self.layers.append(MockLinear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We don't actually do a forward pass, this should not be called
        raise


class MockOptimizer(optim.Optimizer):
    """
    A mock optimizer with deterministic parameter updates.
    """

    from typing import Iterator

    def __init__(self, params: Iterator[torch.nn.Parameter], lr: float = 0.1) -> None:
        defaults = dict(lr=lr)
        super(MockOptimizer, self).__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Apply a fixed update rule: subtract lr * grad
                p.data.add_(p.grad.data, alpha=-group["lr"])


class MockDiLoCoTrainer(DiLoCoTrainer):
    """
    A customized DiLoCoTrainer that uses mock components for deterministic parameter updates.
    """

    def __init__(
        self,
        rank: int,
        store_port: int,
        device: torch.device,
        runner: Runner,
        model_state_dict: dict[str, Any],
        n_fragments: int,
        diloco_args: dict[str, Any],
        inner_lr: float = 1,
        outer_lr: float = 2,
        quorum_barrier: Optional[threading.Barrier] = None,
    ) -> None:
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

        # Call parent constructor
        super().__init__(
            rank, store_port, device, runner, model_state_dict, n_fragments, diloco_args
        )

        self.quorum_barrier = quorum_barrier

    def setup_model(self) -> MockModel:
        """Set up the mock model and move it to the device."""
        model = MockModel(in_dim=1, out_dim=1, n_layers=self.n_fragments)
        model.load_state_dict(self.model_state_dict)
        model.to(self.device)
        return model

    def setup_inner_optimizer(self) -> torch.optim.Optimizer:
        """Set up the mock inner optimizer."""
        return MockOptimizer(self.model.parameters(), lr=self.inner_lr)

    def setup_outer_optimizers(self) -> list[torch.optim.Optimizer]:
        """Set up mock outer optimizers."""
        outer_optimizers = []
        for i in range(self.n_fragments):
            outer_optimizers.append(
                MockOptimizer(self.model.layers[i].parameters(), lr=self.outer_lr)
            )
        return outer_optimizers

    def train_loop(self) -> Dict[str, Any]:
        """Run the training loop with mocked components."""
        # Ensure sync_every is set in diloco_args
        if "sync_every" not in self.diloco_args:
            self.diloco_args["sync_every"] = 2

        parameter_history = {"history": {}, "global_parameter_history": {}}

        with DiLoCo(
            self.manager,
            [layer for layer in self.model.layers],
            self.inner_optimizer,
            self.outer_optimizers,
            backup_device=self.device,
            **self.diloco_args,
        ) as self.diloco:
            if self.quorum_barrier is not None:
                self.manager.start_quorum()
                self.manager.wait_quorum()
                assert self.quorum_barrier is not None
                self.quorum_barrier.wait()
                assert self.manager.should_commit()
                assert self.manager.should_commit()

            local_step = 0
            manager_steps = set()
            while True:
                # Capture parameters before each step
                step_params = {}
                for name, param in self.model.named_parameters():
                    step_params[name] = param.data.clone().detach().cpu().tolist()
                parameter_history["history"][local_step] = step_params

                manager_curr_step = self.manager.current_step()

                if manager_curr_step == 7:
                    break

                if manager_curr_step not in manager_steps:
                    # Store the manager state dict, converting to the right type
                    state_dict = copy.deepcopy(self.manager._manager_state_dict())
                    user_state_dict = cast(dict[str, object], state_dict["user"])
                    parameter_history["global_parameter_history"][local_step] = {}

                    for i in range(self.n_fragments):
                        value = cast(
                            dict[str, dict[str, torch.Tensor]],
                            user_state_dict[f"StreamingDiLoCoFragment_{i}"],
                        )
                        parameter_history["global_parameter_history"][local_step][
                            f"layers.{i}.weight"
                        ] = (
                            value["original_parameters"]["weight"]
                            .data.clone()
                            .detach()
                            .cpu()
                            .tolist()
                        )

                    manager_steps.add(manager_curr_step)

                # For each parameter, set a deterministic gradient
                for _, layer in enumerate(self.model.layers):
                    if isinstance(layer, MockLinear):
                        # Set fixed gradients
                        layer.weight.grad = (
                            torch.ones_like(layer.weight) * layer.weight_grad_value
                        )

                # Step with deterministic updates
                self.inner_optimizer.step()

                self.runner.event_injector.check(self.rank, self.manager.current_step())
                local_step += 1

        return parameter_history


def mock_diloco_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
    train_loop_args: dict[str, Any] = {},
) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    Training loop with mocked components for deterministic parameter updates.
    Uses MockDiLoCoTrainer to handle the training process.
    """
    model_state_dict = train_loop_args.get("model_state_dict", {})
    n_fragments = train_loop_args.get("n_fragments", 1)
    diloco_args = train_loop_args.get("diloco_args", {})
    quorum_barrier = train_loop_args.get("quorum_barrier", None)

    with ExitStack() as stack:
        trainer = MockDiLoCoTrainer(
            rank,
            store_port,
            device,
            runner,
            model_state_dict,
            n_fragments,
            diloco_args,
            quorum_barrier=quorum_barrier,
        )
        stack.callback(trainer.manager.shutdown)
        return trainer.train_loop()
    return {}


class DiLoCoMockedUpdateTest(TestCase):
    @parameterized.expand(
        [
            # Format: (use_cuda, n_fragments, fragment_sync_delay, fragment_update_alpha)
            (False, 2, 0, 0),  # 2 fragments, no delay, 0% mixing
            (False, 2, 0, 0.5),  # 2 fragments, no delay, 50% mixing
            (False, 2, 0, 1),  # 2 fragments, no delay, 100% mixing
            (False, 2, 1, 0),  # 2 fragments, with delay, 0% mixing
            (False, 2, 1, 0.5),  # 2 fragments, with delay, 50% mixing
            (False, 2, 1, 1),  # 2 fragments, with delay, 100% mixing
        ]
    )
    def test_diloco_mocked_updates(
        self,
        use_cuda: bool,
        n_fragments: int,
        fragment_sync_delay: int,
        fragment_update_alpha: float,
    ) -> None:
        """
        Test that validates the model parameters are correctly updated by DiLoCo
        using mocked components for deterministic updates with different configurations:
        - n_fragments: Number of model fragments (1 or 2)
        - fragment_sync_delay: Delay between preparing and syncing fragments (0 or 1)
        - fragment_update_alpha: Controls mixing of local and global parameters (0.0, 0.5, or 1.0)
        """
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(bind="[::]:0", min_replicas=2)
        sync_every = 6
        num_replicas = 2
        futures = []

        torch.manual_seed(42)
        # Initialize the model with the specified number of fragments
        # Create a proper state_dict for the model to avoid load_state_dict errors
        temp_model = MockModel(in_dim=1, out_dim=1, n_layers=n_fragments)
        model_state_dict = temp_model.state_dict()
        quorum_barrier = threading.Barrier(num_replicas)

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                event_injector = EventInjector()
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=mock_diloco_train_loop,
                    use_cuda=use_cuda,
                    train_loop_args={
                        "quorum_barrier": quorum_barrier,
                        "n_fragments": n_fragments,
                        "model_state_dict": model_state_dict,
                        "diloco_args": {
                            "sync_every": sync_every,
                            "fragment_sync_delay": fragment_sync_delay,
                            "fragment_update_alpha": fragment_update_alpha,
                        },
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            for fut in as_completed(futures):
                continue

            results = []
            for fut in futures:
                results.append(fut.result())

        lighthouse.shutdown()

        # Check results against fixture or validate parameter updates
        compared_with_fixture = self._check_against_fixture(results)

        if not compared_with_fixture:
            # If no fixture comparison was done, validate parameters directly
            self._validate_parameter_updates(
                results[0][0],
                n_fragments,
                sync_every,
                fragment_sync_delay,
                fragment_update_alpha,
            )

    @parameterized.expand(
        [
            # Format: (use_cuda, n_fragments, fragment_sync_delay, fragment_update_alpha)
            (False, 2, 0, 0),  # 2 fragments, no delay, 0% mixing
        ]
    )
    def test_diloco_mocked_failure_recovery(
        self,
        use_cuda: bool,
        n_fragments: int,
        fragment_sync_delay: int,
        fragment_update_alpha: float,
    ) -> None:
        """
        Test that validates DiLoCo can recover from a replica failure.
        One replica is set to fail at step 2, and the test verifies that
        the system recovers and parameters are correctly synchronized after recovery.
        """
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(bind="[::]:0", min_replicas=2)
        sync_every = 6
        num_replicas = 2
        futures = []

        # Create event injectors - make the second replica fail at step 2
        event_injectors = [
            EventInjector(),  # First replica runs normally
            EventInjector().fail_at(0, 2),  # Second replica fails at step 2
        ]

        torch.manual_seed(42)
        # Initialize the model with the specified number of fragments
        temp_model = MockModel(in_dim=1, out_dim=1, n_layers=n_fragments)
        model_state_dict = temp_model.state_dict()

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, event_injector in zip(range(num_replicas), event_injectors):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=mock_diloco_train_loop,
                    use_cuda=use_cuda,
                    train_loop_args={
                        "n_fragments": n_fragments,
                        "model_state_dict": model_state_dict,
                        "diloco_args": {
                            "sync_every": sync_every,
                            "fragment_sync_delay": fragment_sync_delay,
                            "fragment_update_alpha": fragment_update_alpha,
                        },
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            # Wait for all futures to complete
            for fut in as_completed(futures):
                continue

            results = []
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"Error in replica: {e}")
                    raise

        lighthouse.shutdown()

        # Check results against fixture or validate failure recovery
        compared_with_fixture = self._check_against_fixture(results)

        if not compared_with_fixture:
            # Verify that the failure was injected
            self.assertEqual(
                event_injectors[1].count[EventInjectorEvent.Failure],
                1,
                "Expected one failure event to be injected",
            )

            # Verify that both replicas have the same global parameters at the end
            # Extract the global parameter history from both replicas
            rep0_global = results[0][0]["global_parameter_history"]
            rep1_global = results[1][0]["global_parameter_history"]

            # Get the last step in both histories
            last_step_rep0 = max(int(step) for step in rep0_global.keys())
            last_step_rep1 = max(int(step) for step in rep1_global.keys())

            # Compare the global parameters at the last step
            for param_name in rep0_global[last_step_rep0].keys():
                rep0_param = torch.tensor(rep0_global[last_step_rep0][param_name])
                rep1_param = torch.tensor(rep1_global[last_step_rep1][param_name])

                self.assertTrue(
                    torch.allclose(rep0_param, rep1_param, rtol=1e-5, atol=1e-5),
                    f"Global parameters don't match at the end: {rep0_param} vs {rep1_param} for {param_name}",
                )

    def _check_against_fixture(
        self, results: list[list[Dict[str, Dict[int, Dict[str, List[float]]]]]]
    ) -> bool:
        """
        Check test results against fixture data.

        Args:
            results: The test results to check against fixture

        Returns:
            bool: True if comparison with fixture was performed, False otherwise
        """
        # Handle fixture reading/writing
        fixture_data = handle_fixture(f"{self.id()}.json", results)

        # If no fixture exists we can't compare results
        if fixture_data is None:
            return False

        # Compare fixture data with current results
        for replica_idx, (fixture_history, current_history) in enumerate(
            zip(fixture_data, results)
        ):
            fixture_history = fixture_history[0]["history"]
            current_history = current_history[0]["history"]
            for step, fixture_params in fixture_history.items():
                for param_name, fixture_values in fixture_params.items():
                    current_values = current_history[int(step)][param_name]
                    # Convert to tensors for comparison with tolerance
                    fixture_tensor = torch.tensor(fixture_values)
                    current_tensor = torch.tensor(current_values)
                    self.assertTrue(
                        torch.allclose(
                            fixture_tensor, current_tensor, rtol=1e-5, atol=1e-5
                        ),
                        f"{fixture_tensor} is not the same as {current_tensor} for {param_name} at step {step}",
                    )

        return True

    def _validate_parameter_updates(
        self,
        parameter_history: Dict[str, Dict[int, Dict[str, List[float]]]],
        n_fragments: int,
        sync_every: int,
        fragment_sync_delay: int,
        fragment_update_alpha: float,
    ) -> None:
        """
        Validate that model parameters are updated as expected according to DiLoCo algorithm.
        Validates both regular steps (inner optimizer updates) and sync steps (outer optimizer updates).

        Args:
            history: Parameter history for a replica
            num_replicas: Total number of replicas
            n_fragments: Number of model fragments
            sync_every: How often to sync parameters
            fragment_sync_delay: Delay between preparing and syncing fragments
            fragment_update_alpha: Controls mixing of local and global parameters
        """
        # Sync happens every sync_every steps for each fragment
        sync_every_fragment = sync_every // n_fragments

        history = parameter_history["history"]
        global_parameter_history = parameter_history["global_parameter_history"]

        # For each step in history, validate parameter updates
        for step in range(1, 16):  # Skip step 0 (initial state)
            for fragment_param_name in history[step].keys():
                # Get current parameters
                fragment_idx = int(fragment_param_name.split(".")[1]) + 1
                current_params = torch.tensor(history[step][fragment_param_name])

                # Determine if this is a sync step for this fragment
                # In DiLoCo, fragments are synced in a round-robin fashion
                # Fragment i is synced at steps: i*sync_every_fragment + k*sync_every
                # where k is a non-negative integer
                is_sync_step = (
                    step - fragment_idx * sync_every_fragment
                ) % sync_every == 0

                if is_sync_step:
                    # This is a sync step for this fragment
                    # Find the previous sync step for this fragment
                    prev_sync_step = max(step - sync_every, 0)

                    # Find the prepare step for this fragment (when pseudogradients were calculated)
                    prepare_step = step - fragment_sync_delay

                    # Parameters at the previous sync step (global parameters before update)
                    prev_sync_params = torch.tensor(
                        global_parameter_history[prev_sync_step][fragment_param_name]
                    )

                    # Parameters at the prepare step (before allreduce)
                    prepare_params = (
                        torch.tensor(history[prepare_step - 1][fragment_param_name]) - 2
                    )  # inner_lr (1) * weight_grad_value (2)

                    # Calculate pseudogradient (difference between global and local params)
                    pseudogradient = prev_sync_params - prepare_params

                    # After allreduce, pseudogradient is averaged across replicas
                    # In our mock setup, all replicas have the same gradient, so no averaging is needed
                    averaged_pseudogradient = pseudogradient

                    # Outer optimizer applies this pseudogradient with its learning rate
                    outer_lr = 2

                    # Calculate expected global parameters after outer optimizer update
                    expected_global_params = (
                        prev_sync_params - outer_lr * averaged_pseudogradient
                    )

                    prev_params = torch.tensor(history[step - 1][fragment_param_name])
                    local_params = (
                        prev_params - 2
                    )  # inner_lr (1) * weight_grad_value (2)

                    # lerp: result = global_params * fragment_update_alpha + local_params * (1 - fragment_update_alpha)
                    expected_params = (
                        local_params * fragment_update_alpha
                        + expected_global_params * (1 - fragment_update_alpha)
                    )

                    # Validate synced parameters
                    self.assertTrue(
                        torch.allclose(
                            current_params, expected_params, rtol=1e-5, atol=1e-5
                        ),
                        f"Parameters at sync step {step} for fragment {fragment_param_name} "
                        f"don't match expected: {current_params} vs {expected_params}. "
                        f"{prepare_params=}, {prev_sync_params=}, {pseudogradient=}, {averaged_pseudogradient=}, {expected_global_params=}",
                    )
                else:
                    # Get previous parameters
                    prev_params = torch.tensor(history[step - 1][fragment_param_name])

                    # Regular step (inner optimizer update)
                    # In our mock setup, each step parameters change by -lr * grad = -1 * 2 = -2
                    expected_params = (
                        prev_params - 2
                    )  # inner_lr (1) * weight_grad_value (2)

                    # Validate synced parameters
                    self.assertTrue(
                        torch.allclose(
                            current_params, expected_params, rtol=1e-5, atol=1e-5
                        ),
                        f"Parameters at sync step {step} for fragment {fragment_param_name} "
                        f"don't match expected: {current_params} vs {expected_params}. ",
                    )


if __name__ == "__main__":
    import unittest

    unittest.main()
