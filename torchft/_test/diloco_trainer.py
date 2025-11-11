# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
from datetime import timedelta
from typing import Any, Dict

import torch
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor

from torchft.local_sgd import DiLoCo
from torchft.manager import Manager
from torchft.manager_integ_test import MyModel, Runner
from torchft.process_group import (
    FakeProcessGroupWrapper,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)

logger: logging.Logger = logging.getLogger(__name__)


class MultiModel(torch.nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 4, n_layers: int = 1) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()

    def get_rand_inputs(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        raise

    def get_rand_labels(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        raise


class MultiMyModel(MultiModel):
    def __init__(self, in_dim: int = 3, out_dim: int = 4, n_layers: int = 1) -> None:
        super().__init__()
        self.in_dim = in_dim

        for _ in range(n_layers):
            self.layers.append(MyModel(in_dim, out_dim))
            in_dim, out_dim = out_dim, in_dim

        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_rand_inputs(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        return torch.rand(batch_size, self.in_dim, device=device)

    def get_rand_labels(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        return torch.randint(self.out_dim, (batch_size,), device=device)


class DiLoCoTrainer:
    """
    A class that encapsulates the DiLoCo training process.
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
    ) -> None:
        """
        Initialize the DiLoCoTrainer.

        Args:
            rank: The rank of the current process.
            store_port: The port for the store.
            device: The device to use for training.
            runner: The runner instance.
            train_loop_args: Additional arguments for the training loop.
        """
        self.rank: int = rank
        self.store_port: int = store_port
        self.device: torch.device = device
        self.runner: Runner = runner

        # Extract arguments from train_loop_args
        self.model_state_dict: Dict[str, Any] = model_state_dict
        self.n_fragments: int = n_fragments
        self.diloco_args: dict[str, Any] = diloco_args

        # Initialize components
        self.model: MultiModel = self.setup_model()
        self.inner_optimizer: torch.optim.Optimizer = self.setup_inner_optimizer()
        self.outer_optimizers: list[torch.optim.Optimizer] = (
            self.setup_outer_optimizers()
        )

        self.pg: FakeProcessGroupWrapper = self.setup_pg()
        # Set up the process group for the event injector
        self.runner.event_injector.set_pg(self.pg)

        self.manager: Manager = self.setup_manager()

        self.device_mesh: None | DeviceMesh = None
        self.setup_distributed()

        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.diloco: DiLoCo | None = None

    def setup_model(self) -> MultiModel:
        """Set up the model and move it to the device."""
        model = MultiMyModel(2, 3, self.n_fragments)
        model.load_state_dict(self.model_state_dict)
        model.to(self.device)
        return model

    def setup_inner_optimizer(self) -> torch.optim.Optimizer:
        """Set up the inner optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )

    def setup_outer_optimizers(self) -> list[torch.optim.Optimizer]:
        """Set up outer optimizers."""
        # Setup inner optimizer
        # Create one outer optimizer per fragment
        outer_optimizers = []
        for _, layers in enumerate(self.model.layers):
            outer_optimizers.append(
                torch.optim.SGD(
                    layers.parameters(), lr=0.7, momentum=0.9, nesterov=True
                )
            )
        return outer_optimizers

    def setup_pg(self) -> FakeProcessGroupWrapper:
        if self.device.type == "cuda":
            return FakeProcessGroupWrapper(ProcessGroupBabyNCCL())
        else:
            return FakeProcessGroupWrapper(
                ProcessGroupGloo(timeout=timedelta(seconds=10))
            )

    def setup_manager(self) -> Manager:
        """Set up the process group and manager."""
        print(
            f"worker {self.runner.replica_id=} {self.rank=} {self.runner.world_size=} starting"
        )

        # Create manager with all arguments passed directly
        return Manager(
            pg=self.pg,
            min_replica_size=2,
            use_async_quorum=False,
            load_state_dict=self.load_state_dict,
            state_dict=self.state_dict,
            replica_id=str(self.runner.replica_id),
            store_addr="localhost",
            store_port=self.store_port,
            rank=self.rank,
            world_size=self.runner.world_size,
            lighthouse_addr=self.runner.lighthouse_address,
            port=19530 + self.runner.replica_id,
            connect_timeout=timedelta(seconds=10),
            quorum_timeout=timedelta(seconds=10),
            timeout=timedelta(seconds=10),
            **self.runner.manager_args,  # type: ignore
        )

    def setup_distributed(self) -> None:
        """Set up distributed training."""
        # Initialize default group for device mesh to work
        if not torch.distributed.is_initialized():
            # TODO: remove this try-except once pytorch is updated to 2.8.0 and can use localhost:0
            try:
                torch.distributed.init_process_group(
                    init_method="tcp://localhost:0",
                    rank=self.rank,
                    world_size=self.runner.world_size,
                )
            except ValueError:
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "0"
                os.environ["WORLD_SIZE"] = str(self.runner.world_size)
                os.environ["RANK"] = str(self.rank)

        self.device_mesh = DeviceMesh(
            self.device.type,
            torch.arange(self.runner.world_size),
        )

        # Convert model parameters to DTensor
        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param = DTensor.from_local(
                        param,
                        device_mesh=self.device_mesh,
                    )

    def load_state_dict(self, state_dict: Dict[str, Dict[str, object]]) -> None:
        """
        Load the state dictionary.

        Args:
            state_dict: The state dictionary to load.
        """
        assert self.diloco is not None

        self.model.load_state_dict(state_dict["model"])
        self.model.to(self.device)

        self.inner_optimizer.load_state_dict(state_dict["inner_optim"])

    def state_dict(self) -> Dict[str, Dict[str, object]]:
        """
        Get the state dictionary.

        Returns:
            The state dictionary.
        """
        assert self.diloco is not None

        return {
            "model": self.model.state_dict(),
            "inner_optim": self.inner_optimizer.state_dict(),
        }

    def train_loop(self) -> dict[str, Any]:
        """Run the training loop."""
        # Ensure sync_every is set in diloco_args
        all_state_dicts = {}

        if "sync_every" not in self.diloco_args:
            self.diloco_args["sync_every"] = 2

        with DiLoCo(
            self.manager,
            [layer for layer in self.model.layers],
            self.inner_optimizer,
            self.outer_optimizers,
            backup_device=self.device,
            **self.diloco_args,
        ) as self.diloco:
            while True:
                self.runner.event_injector.check(self.rank, self.manager.current_step())

                manager_curr_step = self.manager.current_step()
                if manager_curr_step not in all_state_dicts:
                    # Store the manager state dict, converting to the right type
                    all_state_dicts[manager_curr_step] = copy.deepcopy(
                        self.manager._manager_state_dict()
                    )

                batch_size = 1
                inputs = self.model.get_rand_inputs(batch_size, device=self.device)
                labels = self.model.get_rand_labels(batch_size, device=self.device)

                out = self.model(inputs)
                loss = self.criterion(out, labels)

                self.inner_optimizer.zero_grad()
                loss.backward()
                self.inner_optimizer.step()

                # after 4 model updates then break
                if self.manager.current_step() >= 4:
                    break

        return all_state_dicts
