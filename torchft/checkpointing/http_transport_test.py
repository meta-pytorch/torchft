# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import urllib.error
from datetime import timedelta
from typing import Any, Dict
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from parameterized import parameterized

from torchft.checkpointing.http_transport import HTTPTransport
from torchft.checkpointing.http_transport_bench import main as bench_main


class TestHTTPTransport(TestCase):
    def assertStateDictEqual(self, a: Dict[str, object], b: Dict[str, object]) -> None:
        for k, v1 in a.items():
            v2 = b[k]
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                torch.testing.assert_close(v1.cpu(), v2.cpu())
            else:
                self.assertEqual(v1, v2)

    @parameterized.expand(
        [
            ("no chunks", 0),
            ("chunked", 3),
        ]
    )
    def test_checkpoint_server(self, name: str, num_chunks: int) -> None:
        expected: Dict[str, object] = {
            "state": "dict",
            "tensor": torch.rand(5, 2),
            "cuda": torch.rand(
                2, 3, device="cuda" if torch.cuda.is_available() else "cpu"
            ),
        }
        state_dict_fn = MagicMock()
        state_dict_fn.return_value = expected
        server = HTTPTransport(
            timeout=timedelta(seconds=10),
            num_chunks=num_chunks,
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
        self.assertStateDictEqual(out, expected)

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

        with self.assertRaisesRegex(
            urllib.error.HTTPError, r"Error 400.*serving 2345 but got step=1234"
        ):
            server.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=10)
            )

        server.shutdown()

    def test_checkpoint_server_locking(self) -> None:
        server = HTTPTransport(
            timeout=timedelta(seconds=10),
            num_chunks=0,
        )

        # server should start up in a disallowed state this will block incoming
        # requests until allow_checkpoint is called
        self.assertTrue(server._checkpoint_lock.w_locked())
        self.assertTrue(server._disallowed)
        self.assertEqual(server._step, -1)

        # allow requests
        server.allow_checkpoint(1)

        self.assertFalse(server._checkpoint_lock.w_locked())
        self.assertFalse(server._disallowed)
        self.assertEqual(server._step, 1)

        # duplicate allow/disallow is fine
        server.allow_checkpoint(2)
        self.assertEqual(server._step, 2)

        server.disallow_checkpoint()
        server.disallow_checkpoint()
        self.assertTrue(server._checkpoint_lock.w_locked())
        self.assertTrue(server._disallowed)

        server.shutdown()

    def test_benchmark(self) -> None:
        bench_main(
            [
                "--chunk-size=10",
                "--num-chunks=0",
                "--total-size=100",
                "--device=cpu",
            ]
        )
