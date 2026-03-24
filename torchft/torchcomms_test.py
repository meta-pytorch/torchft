# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta
from typing import Dict, TYPE_CHECKING
from unittest import skipUnless, TestCase
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
from torch import nn
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
)
from torch.distributed import ProcessGroup as BaseProcessGroup, TCPStore

try:
    # pyre-fixme[21]: Could not find a module corresponding to import `torchcomms`.
    import torchcomms

    # pyre-fixme[21]: Could not find a module corresponding to import `torchcomms`.
    import torchcomms._comms_mccl

    TORCHCOMMS_AVAILABLE = True
except ImportError:
    TORCHCOMMS_AVAILABLE = False

if TYPE_CHECKING:
    from torchft.torchcomms import ProcessGroupTorchComms


def _dummy_init_pg() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )


def _test_torchcomms_pg(
    pg: BaseProcessGroup,
    example_tensor: torch.Tensor = torch.randn((2, 3), dtype=torch.float32),
) -> Dict[str, dist._Work]:
    """Test collective operations on a ProcessGroupTorchComms instance."""
    shape: torch.Size = example_tensor.shape
    dtype: torch.dtype = example_tensor.dtype
    world_size = dist.get_world_size(pg)

    input_tensor = example_tensor.clone()
    output_tensors = [[torch.empty_like(input_tensor) for _ in range(world_size)]]
    tensor_list = [torch.empty_like(input_tensor)]

    def check_tensors(arg: object) -> None:
        if isinstance(arg, torch.Tensor):
            assert arg.dtype == dtype, f"dtype mismatch: {arg.dtype} != {dtype}"
            assert arg.shape == shape, f"shape mismatch: {arg.shape} != {shape}"
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                check_tensors(item)

    collectives = [
        ("allreduce", ([input_tensor], AllreduceOptions())),
        ("allreduce", ([input_tensor], ReduceOp.SUM)),
        ("allreduce_coalesced", ([input_tensor], AllreduceCoalescedOptions())),
        ("allgather", (output_tensors, [input_tensor], AllgatherOptions())),
        # TODO: mccl std::bad_alloc
        (
            "allgather_into_tensor_coalesced",
            (output_tensors[0], [input_tensor], AllgatherOptions()),
        ),
        (
            "alltoall_base",
            (
                output_tensors[0][0],
                input_tensor,
                [input_tensor.shape[0]],
                [input_tensor.shape[0]],
                AllToAllOptions(),
            ),
        ),
        ("barrier", (BarrierOptions(),)),
        # TODO: mccl std::bad_alloc
        # ("broadcast", (tensor_list, BroadcastOptions())),
        # ("broadcast_one", (input_tensor, 0)),
        (
            "reduce_scatter",
            (output_tensors[0], [[input_tensor]], ReduceScatterOptions()),
        ),
        (
            "reduce_scatter_tensor_coalesced",
            (output_tensors[0], [input_tensor], ReduceScatterOptions()),
        ),
    ]

    works: Dict[str, dist._Work] = {}
    for coll_str, args in collectives:
        try:
            coll = getattr(pg, coll_str)
            work = coll(*args)
            works[coll_str] = work
            work.wait()
            # Verify get_future() returns a resolved future
            fut = work.get_future()
            assert fut.done(), f"future for {coll_str} should be done after wait()"
            check_tensors(args)
        except Exception as e:
            print(f"{coll_str}: {e}")
            if "not implemented" in str(e):
                continue
            raise
    return works


_mccl_comm_counter = 0


@skipUnless(TORCHCOMMS_AVAILABLE, "torchcomms not installed")
@skipUnless(torch.cuda.is_available(), "CUDA not available")
class ProcessGroupTorchCommsMcclTest(TestCase):
    """Tests for ProcessGroupTorchComms with mccl backend."""

    def setUp(self) -> None:
        _dummy_init_pg()
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    def _make_pg(self) -> "ProcessGroupTorchComms":
        from torchft.torchcomms import ProcessGroupTorchComms

        global _mccl_comm_counter
        name = f"test_mccl_{_mccl_comm_counter}"
        _mccl_comm_counter += 1

        comm = torchcomms.new_comm(
            backend="mccl",
            device=torch.device("cuda:0"),
            name=name,
            enable_reconfigure=True,
        )
        return ProcessGroupTorchComms(comm=comm, timeout=timedelta(seconds=60))

    def test_mccl_apis(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        pg = self._make_pg()

        store_addr = f"localhost:{store.port}/prefix"
        pg.configure(store_addr, "0", 0, 1, quorum_id=0)

        self.assertEqual(pg.size(), 1)
        self.assertEqual(pg.getBackendName(), "torchcomms:mccl")
        self.assertIsNotNone(pg.comm)

        _test_torchcomms_pg(pg, torch.tensor([2], dtype=torch.float32, device="cuda:0"))

    def test_reconfigure(self) -> None:
        """Verify that calling configure() twice works."""
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        pg = self._make_pg()

        # First configure
        store_addr = f"localhost:{store.port}/prefix1"
        pg.configure(store_addr, "0", 0, 1, quorum_id=0)
        self.assertEqual(pg.size(), 1)

        # Run a collective
        t = torch.tensor([1.0], device="cuda:0")
        work = pg.allreduce([t], AllreduceOptions())
        work.wait()
        self.assertEqual(t.item(), 1.0)

        # Reconfigure
        store_addr2 = f"localhost:{store.port}/prefix2"
        pg.configure(store_addr2, "0", 0, 1, quorum_id=1)
        self.assertEqual(pg.size(), 1)

        # Run another collective after reconfigure
        t2 = torch.tensor([2.0], device="cuda:0")
        work2 = pg.allreduce([t2], AllreduceOptions())
        work2.wait()
        self.assertEqual(t2.item(), 2.0)

    def test_shutdown(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        pg = self._make_pg()
        store_addr = f"localhost:{store.port}/prefix"
        pg.configure(store_addr, "0", 0, 1, quorum_id=0)
        pg.shutdown()
        self.assertIsNone(pg._comm)

    def test_abort(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        pg = self._make_pg()
        store_addr = f"localhost:{store.port}/prefix"
        pg.configure(store_addr, "0", 0, 1, quorum_id=0)
        pg.abort()
        self.assertIsNone(pg._comm)

    def test_ddp_forward_backward(self) -> None:
        """Integration test: run a DDP forward/backward pass over the PG."""
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        pg = self._make_pg()
        store_addr = f"localhost:{store.port}/prefix"
        pg.configure(store_addr, "0", 0, 1, quorum_id=0)

        m = nn.Linear(3, 4).cuda()
        try:
            m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        except Exception as e:
            if "std::bad_alloc" in str(e):
                self.skipTest("mccl doesn't support broadcast")
            else:
                raise

        out = m(torch.rand(2, 3, device="cuda:0"))
        loss = out.sum()
        loss.backward()

        # Verify gradients were computed and synced via the PG
        for p in m.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


@skipUnless(TORCHCOMMS_AVAILABLE, "torchcomms not installed")
class ProcessGroupTorchCommsTest(TestCase):
    """Tests for ProcessGroupTorchComms (reconfigure path)."""

    def setUp(self) -> None:
        _dummy_init_pg()

    def test_reconfigure_handle_exchange(self) -> None:
        """Test the reconfigure code path with a mocked TorchComm."""
        from torchft.torchcomms import ProcessGroupTorchComms

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        # Build a mock comm whose get_init_handle succeeds (reconfigure path).
        mock_comm = MagicMock()
        mock_comm.get_init_handle.return_value = "handle_rank0"
        mock_comm.get_backend.return_value = "nccl"
        mock_comm.get_device.return_value = torch.device("cpu")

        mock_work = MagicMock()
        mock_comm.reconfigure.return_value = mock_work

        pg = ProcessGroupTorchComms(comm=mock_comm, timeout=timedelta(seconds=60))

        store_addr = f"localhost:{store.port}/prefix"
        pg.configure(store_addr, "0", 0, 1, quorum_id=0)

        # reconfigure should have been called with uuid=0 and the handle list.
        mock_comm.reconfigure.assert_called_once()
        call_kwargs = mock_comm.reconfigure.call_args
        self.assertEqual(call_kwargs.kwargs["uuid"], 0)
        self.assertEqual(call_kwargs.kwargs["init_handles"], ["handle_rank0"])
        mock_work.wait.assert_called_once()

        # Second configure with a different quorum_id.
        mock_comm.reconfigure.reset_mock()
        mock_work.reset_mock()
        store_addr2 = f"localhost:{store.port}/prefix2"
        pg.configure(store_addr2, "0", 0, 1, quorum_id=1)

        call_kwargs2 = mock_comm.reconfigure.call_args
        self.assertEqual(call_kwargs2.kwargs["uuid"], 1)

    def test_work_is_completed(self) -> None:
        """Test _TorchCommsWork.is_completed() delegation."""
        from torchft.torchcomms import _TorchCommsWork

        device = torch.device("cpu")

        # None work is always completed
        w = _TorchCommsWork(device, None)
        self.assertTrue(w.is_completed())
        self.assertTrue(w.wait())

        # Delegated work
        mock_work = MagicMock()
        mock_work.is_completed.return_value = False
        w = _TorchCommsWork(device, mock_work)
        self.assertFalse(w.is_completed())

        mock_work.is_completed.return_value = True
        self.assertTrue(w.is_completed())

    def test_work_get_future(self) -> None:
        """Test _TorchCommsWork.get_future() returns a future."""
        from torchft.torchcomms import _TorchCommsWork

        device = torch.device("cpu")

        # With a mock work object, wait() resolves the CPU future
        value = torch.tensor([1.0])
        mock_work = MagicMock()
        w = _TorchCommsWork(device, mock_work, value=value)
        w.wait()
        fut = w.get_future()
        self.assertTrue(fut.done())
        self.assertIs(fut.value(), value)
