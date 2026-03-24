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

For backends that do not support ``reconfigure()`` (e.g. Gloo), wrap the
configuration in a :class:`ReconfigurableTorchComm` and pass it to
:class:`ProcessGroupTorchComms`.  The PG will delegate ``configure()`` to
the wrapper, which tears down and recreates the communicator each time.
"""

import logging
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Dict, Generator, List, Optional, Set, Union

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

    # pyre-fixme[11]: Annotation `TorchWork` is not defined as a type.
    def __init__(
        self,
        device: torch.device,
        work: torchcomms.TorchWork | None = None,
        value: object | None = None,
    ) -> None:
        super().__init__()
        self._work: Optional[torchcomms.TorchWork] = work
        is_cpu = device.type == "cpu"
        self._fut: torch.futures.Future[object] = torch.futures.Future(
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


class _CompletedWork:
    """A minimal work object that is already completed (for synchronous ops)."""

    def wait(self) -> None:
        pass

    def is_completed(self) -> bool:
        return True


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

    Alternatively, pass a :class:`ReconfigurableTorchComm` which will be
    delegated to for ``configure()`` calls (tear-down-and-recreate path).

    Args:
        comm: a ``torchcomms.TorchComm`` instance (with
            ``enable_reconfigure=True``) or a
            :class:`ReconfigurableTorchComm` wrapper.
        timeout: timeout for store operations and communicator creation.
    """

    def __init__(
        self,
        # pyre-fixme[11]: Annotation `TorchComm` is not defined as a type.
        comm: Union[torchcomms.TorchComm, "ReconfigurableTorchComm"],
        timeout: timedelta = timedelta(seconds=60),
    ) -> None:
        super().__init__(0, 1)
        self._comm: Optional[Union[torchcomms.TorchComm, "ReconfigurableTorchComm"]] = (
            comm
        )
        self._timeout = timeout
        self._configure_count: int = 0
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

        # For ReconfigurableTorchComm, pass the store address so that
        # reconfigure() can bootstrap a fresh communicator via new_comm().
        if isinstance(self._comm, ReconfigurableTorchComm):
            self._comm._store_addr = store_addr

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
            uuid=self._configure_count,
            init_handles=all_handles,
            timeout=self._timeout,
        )
        work.wait()

        self._world_size = world_size
        self._configure_count += 1

    @property
    def comm(self) -> Union["torchcomms.TorchComm", "ReconfigurableTorchComm"]:
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
    # Work wrapping and context hooks
    # ------------------------------------------------------------------

    def _wrap_work(
        self,
        # pyre-fixme[11]: Annotation `TorchWork` is not defined as a type.
        work: Optional[torchcomms.TorchWork],
        value: object = None,
    ) -> Work:
        """Wrap a ``torchcomms.TorchWork`` into a c10d ``Work``.

        Subclasses can override this to add timeout handling or error tracking.
        """
        return _TorchCommsWork(self._device, work, value)

    @contextmanager
    def _run_context(self) -> Generator[None, None, None]:
        """Context manager wrapping every collective call.

        Subclasses can override this to add timeout / error-handling logic
        around collective operations.
        """
        yield

    # ------------------------------------------------------------------
    # Collective operations
    # ------------------------------------------------------------------

    def allreduce(
        self,
        tensors: List[torch.Tensor],
        opts: Union[AllreduceOptions, ReduceOp],
    ) -> Work:
        op = _to_torchcomms_reduce_op(opts)
        with self._run_context():
            work = None
            for t in tensors:
                work = self.comm.all_reduce(t, op, async_op=True)
            return self._wrap_work(work, tensors)

    def allreduce_coalesced(
        self,
        tensors: List[torch.Tensor],
        opts: AllreduceCoalescedOptions,
    ) -> Work:
        op = _to_torchcomms_reduce_op(opts)
        with self._run_context():
            work = None
            for t in tensors:
                work = self.comm.all_reduce(t, op, async_op=True)
            return self._wrap_work(work, tensors)

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        with self._run_context():
            work = None
            for out_list, inp in zip(output_tensors, input_tensor):
                work = self.comm.all_gather(out_list, inp, async_op=True)
            return self._wrap_work(work, output_tensors)

    def allgather_into_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        with self._run_context():
            work = None
            for out, inp in zip(output_tensors, input_tensors):
                work = self.comm.all_gather_single(out, inp, async_op=True)
            return self._wrap_work(work, output_tensors)

    def broadcast(
        self,
        tensor_list: List[torch.Tensor],
        opts: BroadcastOptions,
    ) -> Work:
        root = opts.rootRank
        with self._run_context():
            work = None
            for t in tensor_list:
                work = self.comm.broadcast(t, root, async_op=True)
            return self._wrap_work(work, tensor_list)

    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: ReduceScatterOptions,
    ) -> Work:
        op = _to_torchcomms_reduce_op(opts)
        with self._run_context():
            work = None
            for out, inp_list in zip(output_tensors, input_tensors):
                work = self.comm.reduce_scatter(out, inp_list, op, async_op=True)
            return self._wrap_work(work, output_tensors)

    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> Work:
        op = _to_torchcomms_reduce_op(opts)
        with self._run_context():
            work = None
            for out, inp in zip(output_tensors, input_tensors):
                work = self.comm.reduce_scatter_single(out, inp, op, async_op=True)
            return self._wrap_work(work, output_tensors)

    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts: AllToAllOptions,
    ) -> Work:
        with self._run_context():
            work = self.comm.all_to_all_single(
                output_buffer, input_buffer, async_op=True
            )
            return self._wrap_work(work, output_buffer)

    def barrier(self, opts: Optional[BarrierOptions] = None) -> Work:
        with self._run_context():
            work = self.comm.barrier(async_op=True)
            return self._wrap_work(work)

    def send(self, tensors: List[torch.Tensor], dst_rank: int, tag: int) -> Work:
        with self._run_context():
            work = None
            for t in tensors:
                work = self.comm.send(t, dst_rank, async_op=True)
            return self._wrap_work(work, tensors)

    def recv(self, tensors: List[torch.Tensor], src_rank: int, tag: int) -> Work:
        with self._run_context():
            work = None
            for t in tensors:
                work = self.comm.recv(t, src_rank, async_op=True)
            return self._wrap_work(work, tensors)

    def __repr__(self) -> str:
        return (
            f"ProcessGroupTorchComms(backend={self._backend!r}, "
            f"device={self._device}, configured={self._comm is not None})"
        )


class ReconfigurableTorchComm:
    """
    A standalone wrapper that implements the ``TorchComm`` interface for
    backends that do not support ``TorchComm.reconfigure()`` (e.g. Gloo).

    Each :meth:`reconfigure` call finalizes the current communicator and
    creates a fresh one via ``torchcomms.new_comm``.

    Pass an instance to :class:`ProcessGroupTorchComms` which will call
    ``get_init_handle()`` and ``reconfigure()`` on it just like a real
    ``TorchComm``.

    Args:
        backend: the torchcomms backend name (e.g. ``"gloo"``).
        device: the torch device for the communicator.
        timeout: default timeout for communicator creation.
        store_addr: store address (``host:port/prefix``) used by
            ``new_comm`` to bootstrap the communicator.
    """

    def __init__(
        self,
        backend: str,
        device: torch.device,
        timeout: timedelta = timedelta(seconds=60),
        store_addr: str = "",
    ) -> None:
        self._backend: str = backend
        self._device: torch.device = device
        self._timeout: timedelta = timeout
        self._store_addr: str = store_addr
        self._comm: Optional[torchcomms.TorchComm] = None
        self._configure_count: int = 0

    def get_init_handle(self) -> str:
        """Return a dummy init handle (unused — ``new_comm`` does its own bootstrap)."""
        return ""

    def reconfigure(
        self,
        uuid: int,
        init_handles: Union[List[str], Set[str]],
        timeout: Optional[timedelta] = None,
        hints: Optional[Dict[str, str]] = None,
    ) -> "torchcomms.TorchWork":
        """Finalize the old comm (if any) and create a fresh one.

        The *init_handles* are unused — ``torchcomms.new_comm`` performs its
        own bootstrap via the store and environment variables.
        """
        if self._comm is not None:
            try:
                self._comm.finalize()
            except Exception:
                logger.debug("finalize failed during reconfigure", exc_info=True)
            self._comm = None

        effective_timeout = timeout if timeout is not None else self._timeout
        store = create_store_client(self._store_addr, timeout=effective_timeout)

        # Parse host:port from store_addr (format: host:port/prefix).
        host, _, rest = self._store_addr.partition(":")
        port_str, _, _ = rest.partition("/")

        # Derive rank / world_size from the init_handles list.
        if isinstance(init_handles, set):
            world_size = len(init_handles)
            rank = 0  # set-mode: backend determines rank assignment
        else:
            world_size = len(init_handles)
            # Our handle is "" — find our position in the list.
            rank = init_handles.index("")

        # ``torchcomms.new_comm`` reads env vars for rank discovery.
        env_keys = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        old_env = {k: os.environ.get(k) for k in env_keys}
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = port_str

        try:
            name = f"torchft_{self._configure_count}"
            self._comm = torchcomms.new_comm(
                backend=self._backend,
                device=self._device,
                name=name,
                store=store,
                timeout=effective_timeout,
            )
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

        self._configure_count += 1

        # Return a completed TorchWork (new_comm is synchronous).
        return _CompletedWork()

    def finalize(self) -> None:
        """Finalize (shut down) the underlying communicator."""
        if self._comm is not None:
            self._comm.finalize()
            self._comm = None

    def get_backend(self) -> str:
        return self._backend

    def get_device(self) -> torch.device:
        return self._device

    # -- Forward collective operations to the underlying comm --

    def all_reduce(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.all_reduce(*args, **kwargs)

    def all_gather(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.all_gather(*args, **kwargs)

    def all_gather_single(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.all_gather_single(*args, **kwargs)

    def broadcast(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.broadcast(*args, **kwargs)

    def reduce_scatter(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.reduce_scatter(*args, **kwargs)

    def reduce_scatter_single(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.reduce_scatter_single(*args, **kwargs)

    def all_to_all_single(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.all_to_all_single(*args, **kwargs)

    def barrier(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.barrier(*args, **kwargs)

    def send(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.send(*args, **kwargs)

    def recv(
        self, *args: object, **kwargs: object
    ) -> "torchcomms.TorchWork":
        assert self._comm is not None, "not configured"
        return self._comm.recv(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ReconfigurableTorchComm(backend={self._backend!r}, "
            f"device={self._device}, configured={self._comm is not None})"
        )
