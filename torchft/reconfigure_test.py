# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import List
from unittest import skipUnless, TestCase
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parameterized import parameterized
from torchft.manager import Manager
from torchft.process_group import (
    _rendezvous,
    ProcessGroupGloo,
    reconfigure_with_store,
    ReconfigureOptions,
)


HAS_RECONFIGURE: bool = hasattr(dist.ProcessGroup, "reconfigure")


def _resize_worker(rank: int, port: int, backend: str) -> None:
    timeout = timedelta(seconds=15)
    device = torch.device("cpu")
    if backend == "nccl2":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    store = dist.TCPStore("localhost", port, is_master=False, timeout=timeout)
    # Native Gloo requires a shared bootstrap store and distinct initial ranks.
    # enable_reconfigure avoids connecting communicators at startup.
    dist.init_process_group(
        backend,
        store=store,
        rank=rank,
        world_size=3,
        timeout=timeout,
        enable_reconfigure=True,
    )
    pg = dist.group.WORLD
    assert pg is not None
    name = pg.group_name
    try:
        for generation, members in enumerate(([0, 1, 2], [2, 0], [1, 2, 0])):
            if rank in members:
                reconfigure_with_store(
                    pg,
                    f"localhost:{port}/resize/{generation}",
                    members.index(rank),
                    len(members),
                    timeout,
                )
                assert pg is dist.group.WORLD
                assert pg.group_name == name
                assert dist.get_rank(pg) == members.index(rank)
                assert dist.get_world_size(pg) == len(members)
                tensor = torch.tensor([float(rank + 1)], device=device)
                dist.all_reduce(tensor, group=pg)
                torch.testing.assert_close(
                    tensor.cpu(),
                    torch.tensor([float(sum(peer + 1 for peer in members))]),
                )
            store.set(f"done/{generation}/{rank}", "1")
            store.wait([f"done/{generation}/{peer}" for peer in range(3)], timeout)

        # Aborting must not discard the native PG or its reconfigure handle.
        pg.abort()
        reconfigure_with_store(pg, f"localhost:{port}/recovery", rank, 3, timeout)
        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor, group=pg)
        torch.testing.assert_close(tensor.cpu(), torch.tensor([3.0]))
    finally:
        dist.destroy_process_group()


class ReconfigureTest(TestCase):
    @parameterized.expand([("gloo",), ("nccl2",)])
    @skipUnless(HAS_RECONFIGURE, "requires native process group reconfiguration")
    def test_resize_and_recover(self, backend: str) -> None:
        if backend == "nccl2" and (
            not dist.is_nccl_available() or torch.cuda.device_count() < 3
        ):
            self.skipTest("requires NCCL and three GPUs")
        store = dist.TCPStore("localhost", 0, is_master=True, wait_for_workers=False)
        mp.spawn(_resize_worker, args=(store.port, backend), nprocs=3)

    @patch("torchft.process_group.create_store_client")
    def test_order_and_timeout(self, create_store: Mock) -> None:
        store = create_store.return_value
        store.multi_get.return_value = [b"42", b"peer", b"self"]
        pg = Mock()
        pg.get_reconfigure_handle.return_value = "self"
        timeout = timedelta(seconds=7)
        reconfigure_with_store(pg, "host:123/prefix", 1, 2, timeout)
        create_store.assert_called_once_with("host:123/prefix", timeout)
        store.set.assert_called_once_with("handle/1", "self")
        store.wait.assert_called_once_with(["uuid", "handle/0", "handle/1"], timeout)
        (opts,) = pg.reconfigure.call_args.args
        self.assertEqual(opts.uuid, 42)
        self.assertEqual(opts.handles, ["peer", "self"])
        self.assertEqual(opts.timeout, timeout)
        pg.reconfigure.return_value.wait.assert_called_once_with()

    def test_missing_peer(self) -> None:
        store = dist.TCPStore("localhost", 0, is_master=True, wait_for_workers=False)
        pg = Mock()
        pg.get_reconfigure_handle.return_value = "self"
        with self.assertRaises(dist.DistStoreError):
            reconfigure_with_store(
                pg,
                f"localhost:{store.port}/missing",
                0,
                2,
                timedelta(milliseconds=100),
            )
        pg.reconfigure.assert_not_called()

    def test_failed_work_and_new_uuid(self) -> None:
        store = dist.TCPStore("localhost", 0, is_master=True, wait_for_workers=False)
        pg = Mock()
        pg.get_reconfigure_handle.return_value = "self"
        pg.reconfigure.return_value.wait.side_effect = RuntimeError("failed work")
        for generation in range(2):
            with self.assertRaisesRegex(RuntimeError, "failed work"):
                reconfigure_with_store(
                    pg,
                    f"localhost:{store.port}/{generation}",
                    0,
                    1,
                    timedelta(seconds=1),
                )
        ids = [call.args[0].uuid for call in pg.reconfigure.call_args_list]
        self.assertNotEqual(ids[0], ids[1])

    def test_manager_reconfigure_pg(self) -> None:
        # Only the reconfigure path is under test; skip Manager server setup.
        manager = Manager.__new__(Manager)
        manager._pg = pg = Mock()
        manager._timeout = timeout = timedelta(seconds=7)
        manager._reconfigure_pg(42, ["a", "b"])
        (opts,) = pg.reconfigure.call_args.args
        self.assertEqual(opts.uuid, 42)
        self.assertEqual(opts.handles, ["a", "b"])
        self.assertEqual(opts.timeout, timeout)
        pg.reconfigure.return_value.wait.assert_called_once_with()

    def test_rendezvous(self) -> None:
        pgs = [ProcessGroupGloo() for _ in range(2)]
        pgs[0].set_rank_info("0", 1, 2, 3)
        pgs[1].set_rank_info("1", 1, 2, 5)
        opts = ReconfigureOptions()
        opts.uuid = 7
        handles = [pg.get_reconfigure_handle() for pg in pgs]
        opts.handles = handles
        store = json.loads(handles[0])["store"]

        rendezvous = _rendezvous(opts, pgs[1]._handle_id())
        self.assertEqual(rendezvous.store_addr, f"{store}/torchft/7")
        self.assertEqual(rendezvous.rank, 1)
        self.assertEqual(rendezvous.world_size, 2)
        self.assertEqual(rendezvous.global_ranks, [3, 5])
        self.assertTrue(pgs[0].supports_reconfigure)
        self.assertNotEqual(handles[0], handles[1])

        # Handles identify the group across rank info changes.
        pgs[0].set_rank_info("0", 1, 2, None)
        opts.handles = [
            pgs[1].get_reconfigure_handle(),
            pgs[0].get_reconfigure_handle(),
        ]
        rendezvous = _rendezvous(opts, pgs[0]._handle_id())
        self.assertEqual(rendezvous.rank, 1)
        self.assertIsNone(rendezvous.global_ranks)

    def test_torchft_process_group(self) -> None:
        store: dist.TCPStore = dist.TCPStore(
            "localhost", 0, is_master=True, wait_for_workers=False
        )
        timeout: timedelta = timedelta(seconds=10)
        pgs: List[ProcessGroupGloo] = [
            ProcessGroupGloo(timeout=timeout) for _ in range(2)
        ]

        def run(rank: int) -> torch.Tensor:
            tensor = torch.zeros(1)
            for generation in range(2):
                reconfigure_with_store(
                    pgs[rank],
                    f"localhost:{store.port}/torchft/{generation}",
                    rank,
                    2,
                    timeout,
                )
                tensor = torch.tensor([float(rank + 1)])
                pgs[rank].allreduce([tensor], dist.ReduceOp.SUM).wait()
            return tensor

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, range(2)))
        for tensor in results:
            torch.testing.assert_close(tensor, torch.tensor([3.0]))
        for pg in pgs:
            pg.shutdown()
