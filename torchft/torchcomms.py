# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchComms Process Group
=========================

This module provides a shim layer that wraps a ``torchcomms.TorchComm``
communicator as a ``torchft.process_group.ProcessGroup``, enabling
fault-tolerant reconfiguration via the TorchComm ``reconfigure()`` API.

On each :meth:`configure` call, init-handles are exchanged via the store
and ``TorchComm.reconfigure()`` is called.  The same ``TorchComm`` object
is reused across reconfigurations.
"""

import logging
from datetime import timedelta
from typing import List, Optional, Union

import torch

# pyre-fixme[21]: Could not find a module corresponding to import `torchcomms`.
import torchcomms
from torch.distributed.distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
    Work,
)
from torchft.process_group import create_store_client, ProcessGroup

logger: logging.Logger = logging.getLogger(__name__)


class _TorchCommsWork(Work):
    """
    Wraps a ``torchcomms.TorchWork`` to satisfy the c10d ``Work`` interface.

    Warning: this class only supports futures partially. CPU futures require the
    user to call wait() for the future to resolve. GPU futures are automatically
    resolved to match stream semantics.
    """

    def __init__(
        self,
        device: torch.device,
        # pyre-fixme[11]: Annotation `TorchWork` is not defined as a type.
        work: torchcomms.TorchWork | None = None,
        value: object | None = None,
    ) -> None:
        super().__init__()
        self._work: Optional[torchcomms.TorchWork] = work
        is_cpu = device.type == "cpu"
        self._fut: torch.futures.Future[object] = torch.futures.Future(
            # pyre-fixme[6]: Expected `Optional[List[Union[int, str, device]]]`
            devices=[] if is_cpu else [device]
        )
        self._value = value
        # Resolve the future immediately so that DDP's
        # get_future().then(callback) pattern works without calling wait().
        self._fut.set_result(self._value)

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        if self._work is not None:
            self._work.wait()
        return True

    def get_future(self) -> torch.futures.Future[object]:
        return self._fut

    def is_completed(self) -> bool:
        if self._work is None:
            return True
        return self._work.is_completed()


_REDUCE_OP_MAP: dict[ReduceOp, object] = {
    ReduceOp.SUM: torchcomms.ReduceOp.SUM,
    ReduceOp.PRODUCT: torchcomms.ReduceOp.PRODUCT,
    ReduceOp.MIN: torchcomms.ReduceOp.MIN,
    ReduceOp.MAX: torchcomms.ReduceOp.MAX,
    ReduceOp.BAND: torchcomms.ReduceOp.BAND,
    ReduceOp.BOR: torchcomms.ReduceOp.BOR,
    ReduceOp.BXOR: torchcomms.ReduceOp.BXOR,
    ReduceOp.AVG: torchcomms.ReduceOp.AVG,
}


def _to_torchcomms_reduce_op(
    opts: Union[
        ReduceOp, AllreduceOptions, AllreduceCoalescedOptions, ReduceScatterOptions
    ],
    # pyre-fixme[11]: Annotation `ReduceOp` is not defined as a type.
) -> "torchcomms.ReduceOp":
    """Convert a c10d reduce-op (or options carrying one) to ``torchcomms.ReduceOp``."""
    if isinstance(
        opts, (AllreduceOptions, AllreduceCoalescedOptions, ReduceScatterOptions)
    ):
        op = opts.reduceOp
    elif isinstance(opts, ReduceOp):
        op = opts
    else:
        op = ReduceOp.SUM
    return _REDUCE_OP_MAP.get(op, torchcomms.ReduceOp.SUM)


class ProcessGroupTorchComms(ProcessGroup):
    """
    A :class:`~torchft.process_group.ProcessGroup` backed by a
    ``torchcomms.TorchComm`` communicator.

    The caller creates a ``TorchComm`` with ``enable_reconfigure=True`` and
    hands it to this wrapper.  Each :meth:`configure` call exchanges
    init-handles via the store and calls ``TorchComm.reconfigure()``.  The
    same ``TorchComm`` object is reused across reconfigurations.

    Args:
        comm: a ``torchcomms.TorchComm`` instance (with
            ``enable_reconfigure=True``).
        timeout: timeout for store operations and communicator creation.
    """

    def __init__(
        self,
        # pyre-fixme[11]: Annotation `TorchComm` is not defined as a type.
        comm: torchcomms.TorchComm,
        timeout: timedelta = timedelta(seconds=60),
    ) -> None:
        super().__init__(0, 1)
        self._comm: Optional[torchcomms.TorchComm] = comm
        self._timeout = timeout
        self._world_size: int = 1
        self._backend: str = str(comm.get_backend())
        self._device: torch.device = comm.get_device()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure(
        self,
        store_addr: str,
        replica_id: str,
        rank: int,
        world_size: int,
        quorum_id: Optional[int] = None,
        group_rank: Optional[int] = None,
        group_world_size: Optional[int] = None,
        global_ranks: Optional[list[int]] = None,
    ) -> None:
        assert self._comm is not None

        # Reconfigure path: exchange init-handles via the store.
        store = create_store_client(store_addr, timeout=self._timeout)

        handle: str = self._comm.get_init_handle()

        # Publish our handle so every other rank can read it.
        store.set(f"torchcomms_init_handle/{rank}", handle)

        # Collect handles from all ranks (ordered list → rank = index).
        all_handles: list[str] = []
        for i in range(world_size):
            key = f"torchcomms_init_handle/{i}"
            store.wait([key])
            all_handles.append(store.get(key).decode("utf-8"))

        work = self._comm.reconfigure(
            uuid=quorum_id,
            init_handles=all_handles,
            timeout=self._timeout,
        )
        work.wait()

        self._world_size = world_size

    @property
    def comm(self) -> "torchcomms.TorchComm":
        """Return the underlying communicator (must be configured first)."""
        assert self._comm is not None, "ProcessGroup not configured"
        return self._comm

    def size(self) -> int:
        return self._world_size

    def getBackendName(self) -> str:
        return f"torchcomms:{self._backend}"

    def abort(self) -> None:
        if self._comm is not None:
            try:
                self._comm.finalize()
            except Exception:
                logger.debug("finalize failed during abort", exc_info=True)
            self._comm = None

    def shutdown(self) -> None:
        self.abort()

    # ------------------------------------------------------------------
    # Collective operations
    # ------------------------------------------------------------------

    def allreduce(
        self,
        tensors: List[torch.Tensor],
        opts: Union[AllreduceOptions, ReduceOp],
    ) -> Work:
        assert len(tensors) == 1, f"expected 1 tensor, got {len(tensors)}"
        op = _to_torchcomms_reduce_op(opts)
        work = self.comm.all_reduce(tensors[0], op, async_op=True)
        return _TorchCommsWork(self._device, work, tensors)

    def allreduce_coalesced(
        self,
        tensors: List[torch.Tensor],
        opts: AllreduceCoalescedOptions,
    ) -> Work:
        assert len(tensors) == 1, f"expected 1 tensor, got {len(tensors)}"
        op = _to_torchcomms_reduce_op(opts)
        work = self.comm.all_reduce(tensors[0], op, async_op=True)
        return _TorchCommsWork(self._device, work, tensors)

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        assert len(output_tensors) == 1, f"expected 1 tensor, got {len(output_tensors)}"
        assert len(input_tensor) == 1, f"expected 1 tensor, got {len(input_tensor)}"
        work = self.comm.all_gather(output_tensors[0], input_tensor[0], async_op=True)
        return _TorchCommsWork(self._device, work, output_tensors)

    def allgather_into_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        assert len(output_tensors) == 1, f"expected 1 tensor, got {len(output_tensors)}"
        assert len(input_tensors) == 1, f"expected 1 tensor, got {len(input_tensors)}"
        work = self.comm.all_gather_single(
            output_tensors[0], input_tensors[0], async_op=True
        )
        return _TorchCommsWork(self._device, work, output_tensors)

    def broadcast(
        self,
        tensor_list: List[torch.Tensor],
        opts: BroadcastOptions,
    ) -> Work:
        assert len(tensor_list) == 1, f"expected 1 tensor, got {len(tensor_list)}"
        root = opts.rootRank
        work = self.comm.broadcast(tensor_list[0], root, async_op=True)
        return _TorchCommsWork(self._device, work, tensor_list)

    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: ReduceScatterOptions,
    ) -> Work:
        assert len(output_tensors) == 1, f"expected 1 tensor, got {len(output_tensors)}"
        assert len(input_tensors) == 1, f"expected 1 tensor, got {len(input_tensors)}"
        op = _to_torchcomms_reduce_op(opts)
        work = self.comm.reduce_scatter(
            output_tensors[0], input_tensors[0], op, async_op=True
        )
        return _TorchCommsWork(self._device, work, output_tensors)

    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> Work:
        assert len(output_tensors) == 1, f"expected 1 tensor, got {len(output_tensors)}"
        assert len(input_tensors) == 1, f"expected 1 tensor, got {len(input_tensors)}"
        op = _to_torchcomms_reduce_op(opts)
        work = self.comm.reduce_scatter_single(
            output_tensors[0], input_tensors[0], op, async_op=True
        )
        return _TorchCommsWork(self._device, work, output_tensors)

    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts: AllToAllOptions,
    ) -> Work:
        work = self.comm.all_to_all_single(output_buffer, input_buffer, async_op=True)
        return _TorchCommsWork(self._device, work, output_buffer)

    def barrier(self, opts: Optional[BarrierOptions] = None) -> Work:
        work = self.comm.barrier(async_op=True)
        return _TorchCommsWork(self._device, work)

    def send(self, tensors: List[torch.Tensor], dst_rank: int, tag: int) -> Work:
        assert len(tensors) == 1, f"expected 1 tensor, got {len(tensors)}"
        work = self.comm.send(tensors[0], dst_rank, async_op=True)
        return _TorchCommsWork(self._device, work, tensors)

    def recv(self, tensors: List[torch.Tensor], src_rank: int, tag: int) -> Work:
        assert len(tensors) == 1, f"expected 1 tensor, got {len(tensors)}"
        work = self.comm.recv(tensors[0], src_rank, async_op=True)
        return _TorchCommsWork(self._device, work, tensors)

    def __repr__(self) -> str:
        return (
            f"ProcessGroupTorchComms(backend={self._backend!r}, "
            f"device={self._device}, configured={self._comm is not None})"
        )
