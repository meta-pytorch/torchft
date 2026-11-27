# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Optional
from unittest import mock, skipIf, skipUnless, TestCase

import torch
from parameterized import parameterized
from torch.distributed import TCPStore
from torchft.checkpointing.pg_transport import PGTransport
from torchft.checkpointing.transport import CheckpointTransport
from torchft.checkpointing.transport_test import (
    assertStateDictEqual,
    make_state_dict,
    run_multi_recovery_test,
    TIMEOUT_REGEX,
)
from torchft.process_group import (
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
    reconfigure_with_store,
)


class PGTransportTest(TestCase):
    @skipIf(sys.platform == "darwin", "not passing on mac")
    def test_pg_transport_gloo(self) -> None:
        store: TCPStore = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device: torch.device = torch.device("cpu")

        def init(rank: int, world_size: int) -> CheckpointTransport[dict[str, object]]:
            pg = ProcessGroupGloo()
            reconfigure_with_store(
                pg, f"localhost:{store.port}/prefix", rank, world_size
            )

            return PGTransport[dict[str, object]](
                pg, timeout=timedelta(seconds=10), device=device
            )

        run_multi_recovery_test(self, init, device=device)

    @skipUnless(torch.cuda.device_count() >= 3, "need three CUDA devices")
    def test_pg_transport_baby_nccl(self) -> None:
        store: TCPStore = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device: torch.device = torch.device("cuda")
        timeout: timedelta = timedelta(seconds=10)

        def init(rank: int, world_size: int) -> CheckpointTransport[dict[str, object]]:
            torch.cuda.set_device(rank)

            pg = ProcessGroupBabyNCCL(timeout=timeout)
            reconfigure_with_store(
                pg, f"localhost:{store.port}/prefix", rank, world_size
            )

            return PGTransport[dict[str, object]](pg, timeout=timeout, device=device)

        run_multi_recovery_test(self, init, device=device)

    @skipUnless(torch.cuda.device_count() >= 3, "need three CUDA devices")
    def test_pg_transport_baby_nccl_inplace(self) -> None:
        store: TCPStore = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device: torch.device = torch.device("cuda")
        timeout: timedelta = timedelta(seconds=10)

        def state_dict() -> dict[str, object]:
            return make_state_dict(device)

        def init(rank: int, world_size: int) -> CheckpointTransport[dict[str, object]]:
            torch.cuda.set_device(rank)

            pg = ProcessGroupBabyNCCL(timeout=timeout)
            reconfigure_with_store(
                pg, f"localhost:{store.port}/prefix", rank, world_size
            )

            return PGTransport[dict[str, object]](
                pg,
                timeout=timeout,
                device=device,
                state_dict=state_dict,
            )

        run_multi_recovery_test(self, init, device=device)


@skipIf(sys.platform == "darwin", "not passing on mac")
class PGTransportLiveTest(TestCase):
    """Tests for the chunked receiver's live-apply path (set_live_state_dict)."""

    def setUp(self) -> None:
        self.store = TCPStore(
            host_name="127.0.0.1", port=0, is_master=True, wait_for_workers=False
        )

        def init(rank: int) -> ProcessGroupGloo:
            pg = ProcessGroupGloo(timeout=timedelta(seconds=10))
            reconfigure_with_store(pg, f"127.0.0.1:{self.store.port}/live", rank, 2)
            return pg

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.pgs = list(executor.map(init, range(2)))
        for pg in self.pgs:
            self.addCleanup(pg.shutdown)

    def _roundtrip(
        self,
        expected: dict[str, object],
        live: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        sender = PGTransport[dict[str, object]](
            self.pgs[0], timeout=timedelta(seconds=10), device=torch.device("cpu")
        )
        receiver = PGTransport[dict[str, object]](
            self.pgs[1], timeout=timedelta(seconds=10), device=torch.device("cpu")
        )
        if live is not None:
            receiver.set_live_state_dict(lambda: live)

        with ThreadPoolExecutor(max_workers=2) as executor:
            sent = executor.submit(
                sender.send_checkpoint, [1], 1, expected, None
            )
            got = receiver.recv_checkpoint(0, sender.metadata(), 1, None)
            sent.result(timeout=10)
        return got

    def test_live_apply_roundtrip(self) -> None:
        device: torch.device = torch.device("cpu")
        live = make_state_dict(device)
        expected = make_state_dict(device)

        got = self._roundtrip(expected, live)

        assertStateDictEqual(self, got, expected)
        # Live D2D apply: returned leaves must be the very same live tensors.
        self.assertIs(got["rank"], live["rank"])
        self.assertIs(got["dtensor"], live["dtensor"])

    def test_live_apply_missing_leaf_fallback(self) -> None:
        device: torch.device = torch.device("cpu")
        live = make_state_dict(device)
        del live["rank"]  # receiver tree has no "rank": must fall back to fresh buffer
        expected = make_state_dict(device)

        got = self._roundtrip(expected, live)

        assertStateDictEqual(self, got, expected)
        self.assertNotIsInstance(got["rank"], type(None))
        self.assertIsNot(got["rank"], live.get("rank"))
        # Other leaves still land in the live tensors.
        self.assertIs(got["dtensor"], live["dtensor"])

    def test_chunked_multi_chunk(self) -> None:
        """A state dict larger than one chunk must be split and reassembled."""
        expected: dict[str, object] = {
            "a": torch.randn(512 * 1024),
            "b": torch.randn(512 * 1024),
            "c": torch.randn(512 * 1024),
        }
        with mock.patch.dict(os.environ, {"TORCHFT_RECV_CHUNK_BYTES": str(1024 * 1024)}):
            got = self._roundtrip(expected)
        assertStateDictEqual(self, got, expected)

    def test_recv_chunk_bytes_validation(self) -> None:
        with mock.patch.dict(
            os.environ, {"TORCHFT_RECV_CHUNK_BYTES": str(1024 * 1024)}
        ):
            self.assertEqual(PGTransport._recv_chunk_bytes(), 1024 * 1024)
        with mock.patch.dict(os.environ, {"TORCHFT_RECV_CHUNK_BYTES": "0"}):
            with self.assertRaisesRegex(ValueError, "TORCHFT_RECV_CHUNK_BYTES"):
                PGTransport._recv_chunk_bytes()
        with mock.patch.dict(
            os.environ, {"TORCHFT_RECV_CHUNK_BYTES": str(17 * 1024 * 1024 * 1024)}
        ):
            with self.assertRaisesRegex(ValueError, "TORCHFT_RECV_CHUNK_BYTES"):
                PGTransport._recv_chunk_bytes()
        with mock.patch.dict(os.environ, {"TORCHFT_RECV_CHUNK_BYTES": "abc"}):
            with self.assertRaises(ValueError):
                PGTransport._recv_chunk_bytes()
        # Unset falls back to the 1GB default.
        with mock.patch.dict(os.environ, {"TORCHFT_RECV_CHUNK_BYTES": ""}):
            del os.environ["TORCHFT_RECV_CHUNK_BYTES"]
            self.assertEqual(PGTransport._recv_chunk_bytes(), 1000 * 1024 * 1024)

    def test_live_shape_mismatch_guard(self) -> None:
        device: torch.device = torch.device("cpu")
        expected: dict[str, object] = {
            "rank": torch.tensor([1, 2, 3], device=device),
        }
        live: dict[str, object] = {
            "rank": torch.tensor([1, 2], device=device),  # wrong shape
        }
        with self.assertRaisesRegex(AssertionError, "shape mismatch"):
            self._roundtrip(expected, live)

    def test_live_dtype_mismatch_guard(self) -> None:
        device: torch.device = torch.device("cpu")
        expected: dict[str, object] = {
            "rank": torch.tensor([1, 2, 3], dtype=torch.float32, device=device),
        }
        live: dict[str, object] = {
            "rank": torch.tensor([1, 2, 3], dtype=torch.int64, device=device),
        }
        with self.assertRaisesRegex(AssertionError, "dtype mismatch"):
            self._roundtrip(expected, live)


@skipIf(sys.platform == "darwin", "not passing on mac")
class PGTransportTimeoutTest(TestCase):
    def setUp(self) -> None:
        self.store = TCPStore(
            host_name="127.0.0.1", port=0, is_master=True, wait_for_workers=False
        )

        def init(rank: int) -> ProcessGroupGloo:
            pg = ProcessGroupGloo(timeout=timedelta(seconds=10))
            reconfigure_with_store(pg, f"127.0.0.1:{self.store.port}/timeout", rank, 2)
            return pg

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.pgs = list(executor.map(init, range(2)))
        for pg in self.pgs:
            self.addCleanup(pg.shutdown)

    @parameterized.expand(
        [
            ("send", "omitted"),
            ("send", "none"),
            ("send", "override"),
            ("recv", "omitted"),
            ("recv", "none"),
            ("recv", "override"),
        ]
    )
    def test_stalled_peer(self, direction: str, mode: str) -> None:
        short_timeout = timedelta(milliseconds=100)
        transport = PGTransport[dict[str, object]](
            self.pgs[0],
            timeout=timedelta(seconds=10) if mode == "override" else short_timeout,
            device=torch.device("cpu"),
        )
        kwargs = (
            {}
            if mode == "omitted"
            else {"timeout": short_timeout if mode == "override" else None}
        )
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, TIMEOUT_REGEX):
            if direction == "send":
                transport.send_checkpoint([1], 1, {"tensor": torch.arange(4)}, **kwargs)
            else:
                transport.recv_checkpoint(1, "<n/a>", 1, **kwargs)
        # Distinguish the transport deadline from the ten-second PG timeout.
        self.assertLess(time.monotonic() - start, 5)

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    def test_default_timeout_roundtrip(self, inplace: bool, tensors: bool) -> None:
        expected: dict[str, object] = {"step": 1}
        destination: dict[str, object] = {}
        if tensors:
            expected["tensor"] = torch.arange(4)
            destination["tensor"] = torch.zeros(4, dtype=torch.int64)
        sender = PGTransport[dict[str, object]](
            self.pgs[0], timeout=timedelta(seconds=5), device=torch.device("cpu")
        )
        receiver = PGTransport[dict[str, object]](
            self.pgs[1],
            timeout=timedelta(seconds=5),
            device=torch.device("cpu"),
            state_dict=(lambda: destination) if inplace else None,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            sent = executor.submit(sender.send_checkpoint, [1], 1, expected)
            got = receiver.recv_checkpoint(0, sender.metadata(), 1, timeout=None)
            sent.result(timeout=10)
        assertStateDictEqual(self, got, expected)
        if tensors and inplace:
            torch.testing.assert_close(destination["tensor"], expected["tensor"])

    @parameterized.expand([("send", 0), ("send", 5), ("recv", 0), ("recv", 5)])
    def test_per_call_timeout_precedence(self, first: str, seconds: int) -> None:
        transports: list[PGTransport[dict[str, object]]] = [
            PGTransport[dict[str, object]](
                pg, timeout=timedelta(milliseconds=20), device=torch.device("cpu")
            )
            for pg in self.pgs
        ]
        expected: dict[str, object] = {"tensor": torch.arange(4)}

        def send() -> None:
            transports[0].send_checkpoint(
                [1], 1, expected, timeout=timedelta(seconds=seconds)
            )

        def recv() -> dict[str, object]:
            return transports[1].recv_checkpoint(
                0, transports[0].metadata(), 1, timeout=timedelta(seconds=seconds)
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            if first == "send":
                sent = executor.submit(send)
                time.sleep(0.1)
                got = recv()
                sent.result(timeout=10)
            else:
                received = executor.submit(recv)
                time.sleep(0.1)
                send()
                got = received.result(timeout=10)
        assertStateDictEqual(self, got, expected)
