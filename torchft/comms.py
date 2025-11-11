# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchComm Integration Library
===============================

This module provides a wrapper around torchcomms that is used for cross replica
communication within torchft.

This uses torchcomms directly while providing a compatible interface for
reconfiguration and collective operations.

Usage:
    # For Gloo backend
    comm_gloo = TorchCommGloo(timeout=timedelta(seconds=60))
    comm_gloo.configure(store_addr, replica_id, rank, world_size)

    # For NCCL backend
    comm_nccl = TorchCommNCCL(timeout=timedelta(seconds=60))
    comm_nccl.configure(store_addr, replica_id, rank, world_size)
"""

import logging
import os
import warnings
from contextlib import contextmanager
from datetime import timedelta
from typing import Dict, Generator, List, Optional, Union

import torch
import torch.distributed as dist
import torchcomms
from torch.distributed.distributed_c10d import AllreduceOptions, ReduceOp
from torchft.futures import context_timeout, stream_timeout
from torchft.process_group import (
    create_store_client,
    TORCHFT_TRIGGER_FR_ON_ABORT,
    trigger_nccl_fr_trace_through_pipe,
)

logger: logging.Logger = logging.getLogger(__name__)


class TorchWork(dist._Work):
    """
    Timeout wrapper for TorchWork that wraps TorchWork objects to
    add timeout handling for wait operations.

    Args:
        comm: The TorchComm instance to abort on timeout
        work: The TorchWork object to wrap
        timeout: The timeout duration for operations
    """

    def __init__(
        self,
        comm: "TorchComm",
        work: torchcomms.TorchWork,
        value: object,
        timeout: timedelta,
    ) -> None:
        super().__init__()
        self._comm: "TorchComm" = comm
        self._work: torchcomms.TorchWork = work
        self._value: object = value
        self._timeout: timedelta = timeout

        self._fut: torch.futures.Future[object] = torch.futures.Future()
        self._fut.set_result(self._value)

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        """
        Wait for the work to complete with timeout handling.

        Args:
            timeout: Optional timeout override
        """
        async_timeout = timeout or self._timeout
        with self._stream_timeout(self._comm, async_timeout):
            if self._work is not None:
                self._work.wait()

            # Always use cuda stream for timeout to avoid ProcessGroupNCCL
            # watchdog firing and crashing the process.
            if timeout is not None:
                torch.cuda.synchronize()

        return True

    def get_future(
        self,
    ) -> torch.futures.Future[object]:
        return self._fut

    def is_completed(self) -> bool:
        """Check if the work is completed."""
        return self._work.is_completed() if self._work is not None else True

    def block_current_stream(self, timeout: Optional[timedelta] = None) -> None:
        raise NotImplementedError("The method is not supposed to be called")

    def synchronize(self) -> None:
        raise NotImplementedError("The method is not supposed to be called")

    @classmethod
    @contextmanager
    def _stream_timeout(
        cls, comm: "TorchComm", timeout: timedelta
    ) -> Generator[None, None, None]:
        """
        Set a timeout on the CUDA stream for the given comm.

        This does not hold a reference to self to avoid holding the work
        object/tensors longer than necessary.

        Args:
            comm: The TorchComm to call abort on.
            timeout: The timeout to set on the CUDA stream.
        """

        def callback() -> None:
            logger.error(f"aborting after {timeout}!")
            comm.abort()

        # make sure .wait() can be cancelled if it blocks i.e. in barrier
        with context_timeout(callback, timeout):
            yield

        # Cancel work if the cuda stream doesn't complete
        stream_timeout(callback, timeout)


class TorchComm:
    """
    Base wrapper for torchcomms providing a process group-like interface.

    This provides the common implementation for both Gloo and NCCL backends
    using torchcomms as the underlying communication library.

    Args:
        backend: torchcomms backend name (e.g., "gloo", "nccl")
        timeout: default timeout for operations
        device: torch device to use (e.g., "cpu", "cuda")
    """

    def __init__(
        self,
        backend: str,
        timeout: timedelta,
        device: torch.device,
    ) -> None:
        self._backend = backend
        self._timeout = timeout
        self._device = device
        self._comm: Optional[torchcomms.TorchComm] = None
        self._replica_id: Optional[str] = None
        self._rank: Optional[int] = None
        self._world_size: Optional[int] = None
        self._quorum_id: Optional[int] = None
        self._group_rank: Optional[int] = None
        self._group_world_size: Optional[int] = None
        self._global_ranks: Optional[List[int]] = None
        self._errored: Optional[Exception] = None

        self.errors_logger: logging.Logger = logging.getLogger("torchft_errors")

    def _wrap_work(self, work: torchcomms.TorchWork, value: object) -> TorchWork:
        """
        Wrap work object to allow intercepting wait/synchronization.

        Subclasses can override this to provide custom work wrapping,
        such as adding timeouts or error handling.

        Args:
            work: The work object to wrap
            value: The tensor or value associated with this work

        Returns:
            The wrapped work object (or original if no wrapping needed)
        """
        return TorchWork(self, work, value, self._timeout)

    @contextmanager
    def _run_context(self) -> Generator[None, None, None]:
        """
        Context manager for running collective operations.

        Subclasses can override this to provide custom behavior around
        collective operations, such as timeout management or error handling.

        Yields:
            None
        """
        yield

    def configure(
        self,
        store_addr: str,
        replica_id: str,
        rank: int,
        world_size: int,
        quorum_id: Optional[int] = None,
        group_rank: Optional[int] = None,
        group_world_size: Optional[int] = None,
        global_ranks: Optional[List[int]] = None,
    ) -> None:
        """
        Reconfigure the communication group with new parameters.

        Args:
            store_addr: address of the store to use (host:port/prefix)
            replica_id: the replica_id for this group
            rank: rank of this process
            world_size: world size of this communication group
            quorum_id: current quorum's identifier
            group_rank: local rank within the replica group
            group_world_size: the number of ranks within a replica
            global_ranks: the global ranks part of this group
        """
        self._replica_id = replica_id
        self._rank = rank
        self._world_size = world_size
        self._quorum_id = quorum_id
        self._group_rank = group_rank
        self._group_world_size = group_world_size
        self._global_ranks = global_ranks

        # Shutdown existing comm if present
        if self._comm is not None:
            self.shutdown()

        store = create_store_client(store_addr, self._timeout)

        # Build communication name and hints
        comm_name = f"torchft_{replica_id}_q{quorum_id}_r{rank}"
        hints: Dict[str, str] = {}

        # TODO: unused currently but this can be used to set metadata for
        # flight recorder
        if self._global_ranks:
            hints["global_ranks"] = ",".join(str(r) for r in self._global_ranks)
        if self._group_rank is not None and self._group_world_size is not None:
            hints["group_name"] = (
                f"torchft_quorum_{self._quorum_id}_"
                f"rank_{self._group_rank % self._group_world_size}"
            )

        # Set the ranks properly for the cross replica process group
        os.environ["TORCHCOMM_RANK"] = str(rank)
        os.environ["TORCHCOMM_SIZE"] = str(world_size)

        # Create torchcomms communicator
        self._comm = torchcomms.new_comm(
            backend=self._backend,
            device=self._device,
            abort_process_on_timeout_or_error=False,
            timeout=self._timeout,
            store=store,
            name=comm_name,
            hints=hints,
        )

        self._errored = None

    def shutdown(self) -> None:
        """Shutdown the communication group."""
        if self._comm is not None:
            self._comm.finalize()
            self._comm = None

    def abort(self) -> None:
        """
        Abort the communication group with error logging.

        This logs the error before shutting down the communicator.
        """
        self._errored = RuntimeError("aborted")

        self.errors_logger.info(
            "",
            extra={
                "job_id": os.environ.get("JOB_ID", "unknown"),
                "replica_id": self._replica_id,
                "rank": self._rank,
                "quorum_id": self._quorum_id,
                "error": "torchcomm_abort",
            },
        )

        # Trigger NCCL flight recorder trace if enabled
        if (
            os.environ.get(TORCHFT_TRIGGER_FR_ON_ABORT, "0") == "1"
            and self._rank is not None
        ):
            trigger_nccl_fr_trace_through_pipe(self._rank)

        self.shutdown()

    def errored(self) -> Optional[Exception]:
        """Check if an error has occurred (torchcomms compatible method)."""
        return self._errored

    # Collective operations - all operations use async_op=True and return TorchWork
    # Users should call .wait() on the returned work object to synchronize

    def allgather(
        self,
        output_tensors: List[torch.Tensor],
        input_tensor: torch.Tensor,
    ) -> TorchWork:
        """
        Gather tensors from all ranks into output_tensors.

        Args:
            output_tensors: List of output tensors, one per rank
            input_tensor: Input tensor to gather from this rank

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.all_gather(
                tensor_list=output_tensors,
                tensor=input_tensor,
                async_op=True,
            )
            return self._wrap_work(work, input_tensor)

    def allgather_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> TorchWork:
        """
        Gather tensors from all ranks into a single output tensor.

        Args:
            output: Output tensor (size: world_size * input.size())
            input: Input tensor to gather from this rank

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.all_gather_single(
                output=output,
                input=input,
                async_op=True,
            )
            return self._wrap_work(work, input)

    def allreduce(
        self,
        tensors: list[torch.Tensor],
        opts: Union[AllreduceOptions, ReduceOp, torchcomms.ReduceOp],
    ) -> TorchWork:
        """
        Reduce tensor across all ranks.

        Args:
            tensor: Tensor to reduce (in-place)
            op: Reduction operation (default: SUM)

        Returns:
            TorchWork object that can be waited on
        """
        assert len(tensors) == 1

        if isinstance(opts, ReduceOp):
            if opts == ReduceOp.SUM:
                tc_opts = torchcomms.ReduceOp.SUM
            elif opts == ReduceOp.AVG:
                tc_opts = torchcomms.ReduceOp.AVG
            else:
                raise AssertionError("unsupported reduce op")
        elif isinstance(opts, AllreduceOptions):
            if opts.reduceOp == ReduceOp.SUM:
                tc_opts = torchcomms.ReduceOp.SUM
            elif opts.reduceOp == ReduceOp.AVG:
                tc_opts = torchcomms.ReduceOp.AVG
            else:
                raise AssertionError("unsupported reduce op")
        elif isinstance(opts, torchcomms.ReduceOp):
            tc_opts = opts
        else:
            raise AssertionError("unsupported reduce option type")

        with self._run_context():
            assert self._comm is not None
            work = self._comm.all_reduce(
                tensor=tensors[0],
                op=tc_opts,
                async_op=True,
            )
            return self._wrap_work(work, tensors[0])

    def alltoall_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> TorchWork:
        """
        All-to-all scatter/gather operation with single tensors.

        Args:
            output: Output tensor
            input: Input tensor

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.all_to_all_single(
                output=output,
                input=input,
                async_op=True,
            )
            return self._wrap_work(work, input)

    def alltoall_v_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> TorchWork:
        """
        All-to-all scatter/gather operation with variable sizes.

        Args:
            output: Output tensor
            input: Input tensor
            output_split_sizes: Sizes for splitting output
            input_split_sizes: Sizes for splitting input

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.all_to_all_v_single(
                output=output,
                input=input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                async_op=True,
            )
            return self._wrap_work(work, input)

    def barrier(self) -> TorchWork:
        """
        Synchronize all processes.

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.barrier(async_op=True)
            return self._wrap_work(work, None)

    def broadcast(
        self,
        tensor: torch.Tensor,
        root: int,
    ) -> TorchWork:
        """
        Broadcast tensor from root to all other ranks.

        Args:
            tensor: Tensor to broadcast
            root: Root rank

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.broadcast(
                tensor=tensor,
                root=root,
                async_op=True,
            )
            return self._wrap_work(work, tensor)

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: torchcomms.ReduceOp = torchcomms.ReduceOp.SUM,
    ) -> TorchWork:
        """
        Reduce and scatter tensors across all ranks.

        Args:
            output: Output tensor
            input_list: List of input tensors
            op: Reduction operation (default: SUM)

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.reduce_scatter(
                output=output,
                input_list=input_list,
                op=op,
                async_op=True,
            )
            return self._wrap_work(work, output)

    def reduce_scatter_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        op: torchcomms.ReduceOp = torchcomms.ReduceOp.SUM,
    ) -> TorchWork:
        """
        Reduce and scatter with single tensors.

        Args:
            output: Output tensor
            input: Input tensor
            op: Reduction operation (default: SUM)

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.reduce_scatter_single(
                output=output,
                input=input,
                op=op,
                async_op=True,
            )
            return self._wrap_work(work, output)

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
    ) -> TorchWork:
        """
        Send tensor to destination rank.

        Args:
            tensor: Tensor to send
            dst: Destination rank

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.send(
                tensor=tensor,
                dst=dst,
                async_op=True,
            )
            return self._wrap_work(work, tensor)

    def recv(
        self,
        tensor: torch.Tensor,
        src: int,
    ) -> TorchWork:
        """
        Receive tensor from source rank.

        Args:
            tensor: Tensor to receive into
            src: Source rank

        Returns:
            TorchWork object that can be waited on
        """
        with self._run_context():
            assert self._comm is not None
            work = self._comm.recv(
                tensor=tensor,
                src=src,
                async_op=True,
            )
            return self._wrap_work(work, tensor)


class TorchCommGloo(TorchComm):
    """
    Gloo backend wrapper for torchcomms.

    This provides a drop-in replacement for ProcessGroupGloo using torchcomms.

    Args:
        timeout: Default timeout for operations (default: 60 seconds)

    Example:
        comm = TorchCommGloo()
        comm.configure(store_addr="localhost:1234/prefix", replica_id="r0",
                      rank=0, world_size=4)
        tensor = torch.randn(10)
        work = comm.allreduce(tensor)
        work.wait()
    """

    def __init__(self, timeout: timedelta = timedelta(seconds=60)) -> None:
        super().__init__(
            backend="gloo",
            timeout=timeout,
            device=torch.device("cpu"),
        )

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: torchcomms.ReduceOp = torchcomms.ReduceOp.SUM,
    ) -> TorchWork:
        """
        Gloo backend does not support reduce_scatter.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError("Gloo backend does not support reduce_scatter")

    def reduce_scatter_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        op: torchcomms.ReduceOp = torchcomms.ReduceOp.SUM,
    ) -> TorchWork:
        """
        Gloo backend does not support reduce_scatter_single.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError("Gloo backend does not support reduce_scatter_single")


class TorchCommNCCL(TorchComm):
    """
    NCCL backend wrapper for torchcomms.

    This provides a drop-in replacement for ProcessGroupNCCL using torchcomms.

    If you are using a supported version of NCCL (NCCL >= 2.26, torch >= 2.7)
    this will attempt to use ncclCommAbort to recover from any timeouts.

    Args:
        timeout: Default timeout for operations (default: 60 seconds)

    Example:
        comm = TorchCommNCCL()
        comm.configure(store_addr="localhost:1234/prefix", replica_id="r0",
                      rank=0, world_size=4)
        tensor = torch.randn(10).cuda()
        work = comm.allreduce(tensor)
        work.wait()
    """

    def __init__(self, timeout: timedelta = timedelta(seconds=60.0)) -> None:
        super().__init__(
            backend="nccl",
            timeout=timeout,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        self._use_abort: bool = torch.cuda.nccl.version() >= (2, 25)

        NONBLOCKING_TIMEOUT_ENV = "TORCH_NCCL_NONBLOCKING_TIMEOUT"
        if NONBLOCKING_TIMEOUT_ENV not in os.environ:
            warnings.warn(
                f"{NONBLOCKING_TIMEOUT_ENV} is not set, defaulting to {timeout}. "
                "If any nonblocking NCCL operations have already run this may "
                "result in the default timeout of 30 minutes and hangs on error.",
                stacklevel=2,
            )
            os.environ[NONBLOCKING_TIMEOUT_ENV] = str(timeout.total_seconds())

    @contextmanager
    def _run_context(self) -> Generator[None, None, None]:
        """
        Context manager for running collective operations with timeout.

        Yields:
            None
        """
        if not self._use_abort:
            yield

        timeout: timedelta = self._timeout

        def callback() -> None:
            logger.error(f"aborting after {timeout}!")
            self.abort()

        # when running in blocking mode we need to make sure collectives can timeout
        with context_timeout(callback, timeout):
            yield
