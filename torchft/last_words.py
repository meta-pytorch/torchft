# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Training-thread snapshots and single-decision updates for emergency saves."""

from __future__ import annotations

import copy
from inspect import signature
from threading import Event
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from .manager import Manager
from .utils import synchronize


def snapshot_to_cpu(state):
    leaves, spec = tree_flatten(state)
    copied = []
    for value in leaves:
        if isinstance(value, DTensor):
            value = value.to_local()
        copied.append(
            value.detach().to(device="cpu", copy=True) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        )
    return tree_unflatten(copied, spec)


@torch.no_grad()
def restore_tensor_leaves(snapshot, live_state):
    """Copy local shards into existing tensors while preserving DTensor placement."""
    saved_leaves, saved_spec = tree_flatten(snapshot)
    live_leaves, live_spec = tree_flatten(live_state)
    if saved_spec != live_spec:
        raise ValueError("TorchFT recovery state structure does not match the current model")
    # Validate every leaf before changing any live state.
    for saved, live in zip(saved_leaves, live_leaves):
        if isinstance(saved, torch.Tensor) != isinstance(live, torch.Tensor):
            raise ValueError("TorchFT recovery tensor type mismatch")
        if isinstance(live, torch.Tensor):
            target = live.to_local() if isinstance(live, DTensor) else live
            if saved.shape != target.shape or saved.dtype != target.dtype:
                raise ValueError("TorchFT recovery tensor shape or dtype mismatch")
    restored = []
    for saved, live in zip(saved_leaves, live_leaves):
        if isinstance(live, torch.Tensor):
            if isinstance(live, DTensor) or live.device.type != "cpu":
                target = live.to_local() if isinstance(live, DTensor) else live
                target.copy_(saved)
                restored.append(live)
            else:
                # Optimizers can take ownership of CPU step counters on load.
                restored.append(saved.clone())
        else:
            restored.append(copy.deepcopy(saved))
    return tree_unflatten(restored, live_spec)


class TorchFTCommitTimeoutError(RuntimeError):
    """A commit result is unknown; torchrun must restart the whole replica."""


class TorchFTQuorumStoppedError(RuntimeError):
    """Terminal saving interrupted CPU quorum before an optimizer vote."""


class LastWordsManager(Manager):
    """Manager with explicit application-state and device lifecycle boundaries."""

    def __init__(
        self,
        *args,
        synchronize_device=None,
        process_group_usable=None,
        barrier_after_initial_quorum=False,
        **kwargs,
    ):
        arguments = signature(Manager.__init__).bind(self, *args, **kwargs)
        if arguments.arguments.get("use_async_quorum", False):
            raise ValueError("Last words requires synchronous quorum and healing")
        arguments.arguments["use_async_quorum"] = False
        super().__init__(*arguments.args[1:], **arguments.kwargs)
        self._synchronize_device = synchronize_device or synchronize
        self._process_group_usable = process_group_usable or (lambda: True)
        self._barrier_after_initial_quorum = barrier_after_initial_quorum
        self.commit_authorized = False
        self.commit_decided = False
        self.commit_unknown = False
        self._initial_quorum_barrier_pending = True
        self._healing_snapshot: dict[str, Any] | None = None
        self._snapshot_requested = Event()
        self._snapshot_ready = Event()
        self._snapshot_error: BaseException | None = None
        self.last_words_request: Callable[[str], None] | None = None
        self.last_words_stop_requested: Callable[[], bool] | None = None
        self.last_words_terminal_stop_requested: Callable[[], bool] | None = None
        self.last_words_restore_incomplete = False
        self._last_words_initial_sync_complete = False
        self.elastic_enabled = True
        self.training_boundary = None
        self.after_heal = None

    def begin_step(self):
        self.commit_authorized = False
        self.commit_decided = False
        self.commit_unknown = False
        if self._quorum_future is not None and not self._quorum_future.done():
            # A late healing request must finish before the snapshot events reset.
            self.wait_quorum()
        if self._quorum_id != -1 and not self._process_group_usable():
            # A failed configure may have recorded the quorum ID without creating a PG.
            self._quorum_id = -1
        self._healing_snapshot = None
        self._snapshot_error = None
        self._snapshot_requested.clear()
        self._snapshot_ready.clear()
        if self._initial_quorum_barrier_pending:
            if self._group_world_size > 1:
                dist.barrier()
            self._initial_quorum_barrier_pending = False
        if getattr(self, "elastic_enabled", True) and getattr(self, "last_words_request", None) is None:
            self.start_quorum()
        else:
            self.start_quorum(
                allow_heal=getattr(self, "elastic_enabled", False) or not self._last_words_initial_sync_complete
            )
        self._last_words_initial_sync_complete = True
        if getattr(self, "_barrier_after_initial_quorum", False) and not getattr(
            self, "_initial_quorum_completed", False
        ):
            if self._group_world_size > 1:
                dist.barrier()
            self._initial_quorum_completed = True
        boundary = getattr(self, "training_boundary", None)
        if boundary is not None:
            boundary()

    def wait_quorum(self):
        assert self._quorum_future is not None
        while not self._quorum_future.done():
            stop_requested = getattr(self, "last_words_stop_requested", None)
            terminal_stop = getattr(self, "last_words_terminal_stop_requested", None)
            elastic = getattr(self, "elastic_enabled", True)
            if (not elastic and stop_requested is not None and stop_requested()) or (
                elastic and terminal_stop is not None and terminal_stop()
            ):
                error = TorchFTQuorumStoppedError("last words stop requested while waiting for quorum")
                if (
                    not self._last_words_initial_sync_complete
                    or self._pending_state_dict is not None
                    or self._recovery_event is not None
                    or self.last_words_restore_incomplete
                    or self.commit_unknown
                    or self.commit_authorized
                ):
                    self.last_words_restore_incomplete = True
                # Unblock any snapshot requester without exporting state after
                # terminal stop. A late PG configure is fenced independently.
                self._snapshot_error = error
                self._snapshot_ready.set()
                raise error
            if self._snapshot_requested.wait(0.01) and not self._snapshot_ready.is_set():
                try:
                    self._healing_snapshot = self._export_snapshot()
                except BaseException as error:
                    self._snapshot_error = error
                finally:
                    self._snapshot_ready.set()
        return super().wait_quorum()

    def should_commit(self, timeout=None):
        if getattr(self, "commit_decided", False):
            raise RuntimeError("TorchFT commit may only be voted once per attempt")
        self.commit_decided = True
        try:
            accepted = super().should_commit(timeout)
        except TimeoutError as error:
            self.commit_unknown = True
            self.report_error(error)
            self._checkpoint_transport.disallow_checkpoint()
            raise TorchFTCommitTimeoutError("TorchFT commit RPC timed out; restart the entire replica") from error
        except BaseException:
            self.commit_unknown = True
            raise
        self.commit_authorized = accepted
        return accepted

    def discard_interrupted_step(self, error):
        if self.commit_authorized:
            raise RuntimeError("Cannot discard an authorized optimizer update; restart the replica") from error
        # Device restart invalidates the old recovery stream event. Quorum was
        # applied synchronously before forward; no pending healing is lost here.
        self._recovery_event = None
        self.report_error(error)
        try:
            if getattr(self, "commit_unknown", False):
                raise TorchFTCommitTimeoutError("Cannot discard an attempt with an unknown commit result") from error
            future = getattr(self, "_quorum_future", None)
            if future is None or (future.done() and future.exception() is not None):
                # No commit RPC can have been issued for a failed quorum future.
                # Asking should_commit would rethrow that failure before voting.
                self.commit_decided = True
                self._commit_failures += 1
                self._checkpoint_transport.disallow_checkpoint()
                self._quorum_future = None
                return
            if not getattr(self, "commit_decided", False) and self.should_commit():
                raise RuntimeError("An interrupted TorchFT attempt was unexpectedly authorized") from error
        finally:
            # Rebuild a destroyed communicator even if membership is unchanged.
            self._quorum_id = -1

    def _export_snapshot(self):
        return {
            "user": snapshot_to_cpu({key: fn() for key, fn in self._user_state_dicts.items()}),
            "torchft": self.state_dict(),
        }

    def set_last_words_hooks(
        self,
        *,
        request,
        stop_requested=None,
        terminal_stop_requested=None,
        training_boundary=None,
        after_heal=None,
        elastic_enabled=True,
    ):
        """Connect application policy without handing it Manager's internals.

        Bind once, before the first quorum. Recovery completion callbacks execute
        on the training thread after the supplied device synchronization hook.
        """
        if self._quorum_future is not None:
            raise RuntimeError("Bind last-words hooks before the first quorum")
        self.last_words_request = request
        self.last_words_stop_requested = stop_requested
        self.last_words_terminal_stop_requested = terminal_stop_requested
        self.training_boundary = training_boundary
        self.after_heal = after_heal
        self.elastic_enabled = elastic_enabled

    @property
    def restoration_incomplete(self):
        """Whether partial restoration prevents saving the application state."""
        return self.last_words_restore_incomplete

    def quorum_members(self):
        return list(self.quorum_snapshot().members)

    def _apply_pending_state_dict(self):
        if (
            getattr(self, "last_words_request", None) is not None
            and not getattr(self, "elastic_enabled", False)
            and self._last_words_initial_sync_complete
            and self._pending_state_dict is not None
        ):
            request = self.last_words_request
            assert request is not None
            request("Runtime replica healing is unsupported with last words")
            raise RuntimeError("Runtime replica healing is unsupported with last words")
        if self._recovery_event is not None:
            self._recovery_event.synchronize()
            self._recovery_event = None
        received_state = self._pending_state_dict is not None
        if getattr(self, "last_words_request", None) is not None:
            # Keep this latched on any partial tensor/state restoration failure.
            self.last_words_restore_incomplete = True
        super()._apply_pending_state_dict()
        if getattr(self, "last_words_request", None) is not None:
            if not received_state:
                raise RuntimeError("Healing did not provide a complete checkpoint; replica is ineligible")
            if self.last_words_restore_incomplete:
                self._synchronize_device()
                after_heal = getattr(self, "after_heal", None)
                if after_heal is not None:
                    after_heal()
            self.last_words_restore_incomplete = False
            self._last_words_initial_sync_complete = True

    def register_state_dict_fn(self, key, load_state_dict, state_dict):
        def load_from_cpu(snapshot):
            load_state_dict(restore_tensor_leaves(snapshot, state_dict()))

        super().register_state_dict_fn(key, load_from_cpu, state_dict)

    def _manager_state_dict(self):
        self._snapshot_requested.set()
        if not self._snapshot_ready.wait(self._quorum_timeout.total_seconds()):
            raise TimeoutError("Training thread did not prepare the TorchFT healing state")
        if self._snapshot_error is not None:
            raise RuntimeError("TorchFT healing state export failed") from self._snapshot_error
        if self._healing_snapshot is None:
            raise RuntimeError("TorchFT healing state was not prepared")
        return self._healing_snapshot
