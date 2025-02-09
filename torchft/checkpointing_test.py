# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest import skipUnless, TestCase
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
from torch.distributed import TCPStore
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor

from torchft.checkpointing import CheckpointServer, PGTransport
from torchft.process_group import ProcessGroupBabyNCCL, ProcessGroupGloo


class TestCheckpointing(TestCase):
    def test_checkpoint_server(self) -> None:
        expected = {"state": "dict"}
        state_dict_fn = MagicMock()
        state_dict_fn.return_value = expected
        server = CheckpointServer(
            timeout=timedelta(seconds=10),
        )

        server.send_checkpoint(
            dst_ranks=[],
            step=1234,
            state_dict=expected,
            timeout=timedelta(seconds=10),
        )

        metadata = server.metadata()

        out = server.recv_checkpoint(
            src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=10)
        )
        self.assertEqual(out, expected)

        # test timeout
        with self.assertRaisesRegex(urllib.error.URLError, r"urlopen error"):
            server.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=0.0)
            )

        # test mismatch case
        server.send_checkpoint(
            dst_ranks=[],
            step=2345,
            state_dict=expected,
            timeout=timedelta(seconds=10),
        )

        with self.assertRaisesRegex(urllib.error.HTTPError, r"Error 400"):
            server.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=10)
            )

        server.shutdown()

    def test_checkpoint_server_locking(self) -> None:
        server = CheckpointServer(
            timeout=timedelta(seconds=10),
        )

        # server should start up in a disallowed state this will block incoming
        # requests until allow_checkpoint is called
        self.assertTrue(server._checkpoint_lock.locked())
        self.assertTrue(server._disallowed)
        self.assertEqual(server._step, -1)

        # allow requests
        server.allow_checkpoint(1)

        self.assertFalse(server._checkpoint_lock.locked())
        self.assertFalse(server._disallowed)
        self.assertEqual(server._step, 1)

        # duplicate allow/disallow is fine
        server.allow_checkpoint(2)
        self.assertEqual(server._step, 2)

        server.disallow_checkpoint()
        server.disallow_checkpoint()
        self.assertTrue(server._checkpoint_lock.locked())
        self.assertTrue(server._disallowed)

        server.shutdown()

    def test_timed_acquire(self) -> None:
        lock = threading.Lock()

        with _timed_acquire(lock, timedelta(seconds=10)):
            self.assertTrue(lock.locked())

        self.assertFalse(lock.locked())

        lock.acquire()

        with self.assertRaisesRegex(
            TimeoutError, r"timed out acquiring lock after 0.0"
        ):
            with _timed_acquire(lock, timedelta(seconds=0.0)):
                pass

        self.assertTrue(lock.locked())

    def _test_pg_transport(self, backend: str, device: str) -> None:
        dist.init_process_group(
            backend=backend, rank=0, world_size=1, store=dist.HashStore()
        )
        device_mesh = DeviceMesh("cpu", 1)

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"

        timeout = timedelta(seconds=5)

        def sender(device: str) -> object:
            if backend == "gloo":
                a = ProcessGroupGloo(timeout=timeout)
            elif backend == "nccl":
                device = f"{device}:0"
                a = ProcessGroupBabyNCCL(timeout=timeout)
            else:
                raise ValueError(f"unknown backend: {backend}")

            a.configure(store_addr, 0, 2)

            print("send configured")

            tensor = torch.randn(4, 4)
            dtensor = distribute_tensor(tensor, device_mesh, [])

            state_dict = {
                "tensors": {
                    "float32": torch.tensor([1, 2, 3], dtype=torch.float32),
                    "strided": torch.rand(10, dtype=torch.float32)[1::2],
                    "uint16": torch.tensor([1, 2, 3], dtype=torch.uint16),
                    "dtensor": dtensor,
                },
                "non_tensors": "blah",
            }

            transport = PGTransport(a, timeout=timeout, device=device)
            transport.send_checkpoint(
                dst_ranks=[1],
                step=123,
                state_dict=state_dict,
                timeout=timeout,
            )
            transport.disallow_checkpoint()

            return state_dict

        def receiver(device: str) -> object:
            if backend == "gloo":
                a = ProcessGroupGloo(timeout=timeout)
            elif backend == "nccl":
                # torch.cuda.set_device(1)
                device = f"{device}:1"
                a = ProcessGroupBabyNCCL(timeout=timeout)
            else:
                raise ValueError(f"unknown backend: {backend}")

            a.configure(store_addr, 1, 2)

            print("recv configured")

            transport = PGTransport(a, timeout=timeout, device=device)
            state_dict = transport.recv_checkpoint(
                src_rank=0, metadata="blah", step=123, timeout=timeout
            )

            return state_dict

        with ThreadPoolExecutor(max_workers=2) as executor:
            send_fut = executor.submit(sender, device)
            recv_fut = executor.submit(receiver, device)

            send_state_dict = send_fut.result()
            recv_state_dict = recv_fut.result()

        for k, a in send_state_dict["tensors"].items():
            b = recv_state_dict["tensors"][k]

            if isinstance(a, DTensor):
                torch.testing.assert_close(b._local_tensor.cpu(), a._local_tensor.cpu())
                self.assertEqual(b._spec, a._spec)
            else:
                torch.testing.assert_close(b.cpu(), a.cpu())
        self.assertEqual(recv_state_dict["non_tensors"], send_state_dict["non_tensors"])

        dist.destroy_process_group()

    def test_pg_transport_gloo(self) -> None:
        self._test_pg_transport("gloo", "cpu")

    @skipUnless(torch.cuda.device_count() >= 2, "need two CUDA devices")
    def test_pg_transport_baby_nccl(self) -> None:
        self._test_pg_transport("nccl", "cuda")
