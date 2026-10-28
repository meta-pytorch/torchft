# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fault tolerant FSDP
===================

Fully sharded training (no replication) that survives host failures. Each
step's sharded state is copied asynchronously to pinned CPU memory and
replicated to one other host with ``torch.distributed._transport``. When a
host fails, a lighthouse quorum assigns a hot spare to its slot, the
``nccl2`` process group is reconfigured in place, and every rank restores the
newest snapshot step that all ranks hold.

Add :class:`FaultTolerance` to an existing FSDP training loop; see
:mod:`torchft.fsdp.fault_tolerance` for an example.

- :mod:`torchft.fsdp.fault_tolerance`: training loop integration.
- :mod:`torchft.fsdp.membership`: lighthouse quorum and slot assignment.
- :mod:`torchft.fsdp.snapshot`: async pinned CPU snapshots and replication.
- :mod:`torchft.fsdp.coordinator`: TCPStore master plus lighthouse.

Requires a PyTorch build where TCPStore client operations honor the store
timeout (https://github.com/pytorch/pytorch/pull/200384); otherwise a stalled store connection can hang
recovery.
"""

from torchft.fsdp.fault_tolerance import FaultTolerance, FTFSDPConfig
from torchft.fsdp.membership import (
    RestartRequiredError,
    TrainingFinishedError,
    UnrecoverableError,
)

__all__ = [
    "FaultTolerance",
    "FTFSDPConfig",
    "RestartRequiredError",
    "TrainingFinishedError",
    "UnrecoverableError",
]
