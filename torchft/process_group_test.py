# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import io
import multiprocessing
import os
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Dict, List, Tuple, cast
from unittest import TestCase, skipUnless
from unittest.mock import Mock

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch import nn
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    BroadcastOptions,
    ReduceOp,
    _resolve_process_group,
)
from torch.distributed import (
    ReduceOp,
    TCPStore,
    Work,
    _functional_collectives,
    get_world_size,
)
from torch.distributed.device_mesh import init_device_mesh

from torchft.manager import Manager
from torchft.process_group import (
    ErrorSwallowingProcessGroupWrapper,
    ManagedProcessGroup,
    ProcessGroup,
    ProcessGroupBabyGloo,
    ProcessGroupBabyNCCL,
    ProcessGroupDummy,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ProcessGroupWrapper,
    _DummyWork,
    _ErrorSwallowingWork,
    _ManagedWork,
    extend_device_mesh,
    ft_init_device_mesh,
)


def dummy_init_pg() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )


def run_collective(
    pg: ProcessGroup,
    collective: str,
    example_tensor: torch.Tensor = torch.randn((2, 3), dtype=torch.float32),
) -> List[Work]:
    """Run a single collective."""
    shape: torch.Size = example_tensor.shape
    dtype: torch.dtype = example_tensor.dtype

    input_tensor = example_tensor.clone()
    output_tensors = [
        [torch.empty_like(input_tensor) for _ in range(get_world_size(pg))]
    ]
    tensor_list = [torch.empty_like(input_tensor)]

    if collective == "allreduce":
        works = [
            pg.allreduce([input_tensor], AllreduceOptions()),
            pg.allreduce([input_tensor], ReduceOp.SUM),
        ]
        input_tensors = input_tensor
    elif collective == "allgather":
        works = [pg.allgather(output_tensors, [input_tensor], AllgatherOptions())]
        input_tensors = (output_tensors, input_tensor)
    elif collective == "broadcast":
        print("tensor_list: ", tensor_list, " and type: ", type(tensor_list))
        works = [pg.broadcast(tensor_list, BroadcastOptions())]
        input_tensors = tensor_list
    elif collective  == "broadcast_one":
        works = [pg.broadcast_one(input_tensor, 0)]
        input_tensors = input_tensor
    else:
        raise ValueError(f"Unsupported collective: {collective}.")

    def check_tensors(input_tensors: Any) -> None:  # pyre-ignore[2]
        """Recursively check tensors for input_tensors shape and dtype."""
        if isinstance(input_tensors, torch.Tensor):
            assert input_tensors.dtype == dtype, f"Output dtype mismatch: {input_tensors.dtype} != {dtype}"
            assert input_tensors.shape == shape, f"Output shape mismatch: {input_tensors.shape} != {shape}"
        elif isinstance(input_tensors, (list, tuple)):
            for item in input_tensors:
                check_tensors(item)

    for work in works:
        work.wait()
        fut = work.get_future()
        fut.wait()
        # Check that all tensor arguments have the input_tensors shapes and dtypes
        check_tensors(input_tensors)

    print(works)
    return works


@skipUnless(torch.cuda.is_available(), "needs CUDA")
class NCCLTests(TestCase):
    collectives = [
        "allreduce",
        "allgather",
        "broadcast",
        "broadcast_one",
    ]

    def setUp(self) -> None:
        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store_addr = f"localhost:{self.store.port}/prefix"

    @parameterized.expand(collectives)
    def test_nccl(self, collective: str) -> None:
        device = "cuda"

        pg = ProcessGroupNCCL()
        pg.configure(self.store_addr, 0, 1)

        self.assertEqual(pg.size(), 1)

        run_collective(
            pg=pg,
            collective=collective,
            example_tensor=torch.tensor([2], device=device),
        )

        m = nn.Linear(3, 4).to(device)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3, device=device))

        # reconfigure
        store_addr = f"localhost:{self.store.port}/prefix2"
        pg.configure(store_addr, 0, 1)

        run_collective(
            pg=pg,
            collective=collective,
            example_tensor=torch.tensor([2], device=device),
        )

        torch.cuda.synchronize()

    @parameterized.expand(collectives)
    def test_baby_nccl_apis(self, collective: str) -> None:
        # set to 1 if more than >=2 gpus
        device_id = 1 % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

        pg = ProcessGroupBabyNCCL(timeout=timedelta(seconds=10))
        try:
            pg.configure(self.store_addr, 0, 1)

            run_collective(
                pg=pg,
                collective=collective,
                example_tensor=torch.randn((2, 3), device="cuda"),
            )

            torch.cuda.synchronize()

            # force collection to ensure no BabyWork objects remain
            gc.collect()

            self.assertEqual(pg.num_active_work(), 0)
        finally:
            pg.shutdown()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipUnless(torch.cuda.device_count() >= 2, "need two CUDA devices")
    def test_baby_nccl_2gpu(self) -> None:
        def run(rank: int) -> Tuple[ProcessGroupBabyNCCL, torch.Tensor, Work]:
            a = ProcessGroupBabyNCCL(
                timeout=timedelta(seconds=10.0),
            )
            a.configure(self.store_addr, rank, 2)
            self.assertEqual(a.size(), 2)

            # We test using set_device to ensure stream device is correct.
            torch.cuda.set_device(rank)
            at = torch.tensor([rank + 1], device="cuda")

            a_work = a.allreduce([at], ReduceOp.SUM)
            return a, at, a_work

        with ThreadPoolExecutor(max_workers=2) as executor:
            a_fut = executor.submit(run, 0)
            b_fut = executor.submit(run, 1)

        a, at, a_work = a_fut.result()
        b, bt, b_work = b_fut.result()

        try:
            a_work.wait()
            b_work.get_future().wait()
            torch.testing.assert_close(at.cpu(), bt.cpu())
            torch.cuda.synchronize()
        finally:
            # cleanup - first ensure that babywork is deleted before shutting down PGs
            # note futures must be deleted as they hold references to babywork
            del a_fut
            del b_fut
            del a_work
            del b_work
            gc.collect()
            b.shutdown()
            a.shutdown()


class GlooTests(TestCase):
    collectives = ["allreduce", "allgather", "broadcast", "broadcast_one"]

    def setUp(self) -> None:
        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store_addr = f"localhost:{self.store.port}/prefix"

    @parameterized.expand(collectives)
    def test_gloo(self, collective: str) -> None:
        pg = ProcessGroupGloo()
        pg.configure(self.store_addr, 0, 1)

        self.assertEqual(pg.size(), 1)
        run_collective(pg=pg, collective=collective)
        m = nn.Linear(3, 4)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3))

    @parameterized.expand(collectives)
    def test_baby_gloo_apis(self, collective: str) -> None:
        pg = ProcessGroupBabyGloo(timeout=timedelta(seconds=10))
        pg.configure(self.store_addr, 0, 1)

        run_collective(pg=pg, collective=collective)

        # force collection to ensure no BabyWork objects remain
        gc.collect()

        self.assertEqual(pg.num_active_work(), 0)
        pg.shutdown()

    def test_timeout(self) -> None:
        pg = ProcessGroupGloo(timeout=timedelta(seconds=0.01))
        with self.assertRaisesRegex(
            RuntimeError, "(timeout after 10ms|Socket Timeout)"
        ):
            pg.configure(self.store_addr, 0, 2)

    def test_baby_gloo(self) -> None:
        def run(rank: int) -> Tuple[torch.Tensor, Work]:
            a = ProcessGroupBabyGloo()
            a.configure(self.store_addr, rank, 2)

            self.assertEqual(a.size(), 2)

            at = torch.tensor([rank + 1])

            a_work = a.allreduce([at], ReduceOp.SUM)
            return at, a_work

        with ThreadPoolExecutor(max_workers=2) as executor:
            a_fut = executor.submit(run, 0)
            b_fut = executor.submit(run, 1)

        at, a_work = a_fut.result()
        bt, b_work = b_fut.result()

        a_work.wait()
        fut = b_work.get_future()

        fut.wait()

        torch.testing.assert_close(at, torch.tensor([3]))
        torch.testing.assert_close(bt, torch.tensor([3]))

    def test_baby_gloo_timeout(self) -> None:
        a = ProcessGroupBabyGloo(timeout=timedelta(seconds=0.01))
        with self.assertRaisesRegex(TimeoutError, "timed out after 0.01 seconds"):
            a.configure(self.store_addr, 0, 2)

    def test_reconfigure_process_group_baby_gloo(self) -> None:
        a = ProcessGroupBabyGloo()
        a.configure(self.store_addr, 0, 1)
        future_thread_1 = a._future_thread
        future_queue_1 = a._future_queue
        p_1 = a._p

        store_addr = f"localhost:{self.store.port}/prefix2"
        a.configure(store_addr, 0, 1)
        future_thread_2 = a._future_thread
        future_queue_2 = a._future_queue
        p_2 = a._p

        self.assertNotEqual(future_thread_1, future_thread_2)
        self.assertNotEqual(future_queue_1, future_queue_2)
        self.assertNotEqual(p_1, p_2)

        assert future_thread_1 is not None
        self.assertFalse(future_thread_1.is_alive())
        assert future_queue_1 is not None
        self.assertTrue(future_queue_1.closed())
        assert p_1 is not None
        self.assertFalse(p_1.is_alive())

        assert future_thread_2 is not None
        self.assertTrue(future_thread_2.is_alive())
        assert future_queue_2 is not None
        self.assertFalse(future_queue_2.closed())
        assert p_2 is not None
        self.assertTrue(p_2.is_alive())

    def test_device_mesh(self) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(0)
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)

        mesh_1d = init_device_mesh("cpu", mesh_shape=(1,), mesh_dim_names=("tp",))

        pg = ProcessGroupGloo()
        pg.register("test_device_mesh")
        pg.configure(self.store_addr, 0, 1)

        mesh_2d = extend_device_mesh(mesh_1d, pg)
        mesh_2d.get_group("dp")
        assert mesh_2d.ndim == 2

        pg.unregister()

    def test_functional_collectives(self) -> None:
        dummy_init_pg()
        pg = ProcessGroupGloo().register("test_func_col")
        pg.configure(self.store_addr, 0, 1)

        self.assertEqual(pg.group_name, str(dist.get_pg_count() - 1))

        self.assertIs(_resolve_process_group(pg.group_name), pg)

        try:
            t = torch.zeros(10)
            _functional_collectives.all_reduce(t, "sum", pg).wait()
        finally:
            pg.unregister()


class DummyTests(TestCase):
    collectives = ["allreduce", "allgather", "broadcast", "broadcast_one"]

    def test_dummy(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        m = nn.Linear(3, 4)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3))

    def test_process_group_wrapper(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        wrapper = ProcessGroupWrapper(pg)
        self.assertIs(wrapper.parent, pg)

        wrapper.configure("addr", 0, 1)
        self.assertEqual(pg.configure_count, 1)

        self.assertEqual(repr(wrapper), "ProcessGroupWrapper(pg=ProcessGroupDummy())")

    def test_error_swallowing_process_group_wrapper(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        wrapper = ErrorSwallowingProcessGroupWrapper(pg)
        self.assertIs(wrapper.parent, pg)

        works = run_collective(pg=wrapper, collective="allreduce")
        self.assertIsInstance(works[0], _ErrorSwallowingWork)

        err = RuntimeError("test")
        wrapper.report_error(err)
        self.assertEqual(wrapper.error(), err)

        works = run_collective(pg=wrapper, collective="allreduce")
        for work in works:
            self.assertIsInstance(work, _DummyWork)

    def test_managed_process_group(self) -> None:
        """Test the interaction between a ManagedProcessGroup and a Manager."""
        manager = Mock(spec=Manager)
        manager.errored.return_value = None
        manager._pg = ProcessGroupDummy(0, 1)
        pg = ManagedProcessGroup(manager)
        manager.num_participants.return_value = 123

        self.assertEqual(pg.size(), 123)

        works = run_collective(pg=pg, collective="allreduce")
        self.assertIsInstance(works[0], _ManagedWork)

        # No errors occurred during collective
        self.assertEqual(manager.report_error.call_count, 0)
        self.assertEqual(manager.wrap_future.call_count, 2)
        self.assertEqual(manager.wait_quorum.call_count, 2)


class DeviceMeshTest(TestCase):
    @staticmethod
    def _test_init_device_mesh(world_size: int, rank: int) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(12346)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(4)

        testcase = TestCase()

        manager = Mock(spec=Manager)
        # Even though we only have 4 workers, we can still initialize (2, 4) mesh.
        # That's because the replicate group is NOT phystically created in the
        # real mesh but is virtually added to the mesh via ManagedDeviceMesh.
        device_mesh = ft_init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, world_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
            replicate_dim=0,
            manager=manager,
        )

        testcase.assertTrue(
            isinstance(device_mesh.get_group("dp_replicate"), ManagedProcessGroup)
        )
        testcase.assertTrue(
            not isinstance(device_mesh.get_group("dp_shard"), ManagedProcessGroup)
        )
        replicate_group = device_mesh.get_group("dp_replicate")
        testcase.assertEqual(
            cast(ManagedProcessGroup, replicate_group)._manager, manager
        )
        replicate_mesh = device_mesh["dp_replicate"]
        testcase.assertEqual(replicate_mesh.get_group(), replicate_group)
        flatten_mesh = device_mesh._flatten("dp")
        manager.num_participants.return_value = 1
        testcase.assertEqual(flatten_mesh.size(), world_size)
        testcase.assertEqual(flatten_mesh.get_local_rank(), dist.get_rank())

    def test_init_device_mesh(self) -> None:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                future = executor.submit(self._test_init_device_mesh, 4, i)
                futures.append(future)
