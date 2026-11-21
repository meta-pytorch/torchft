# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import urllib.error
from collections import OrderedDict
from datetime import timedelta
from typing import Dict
from unittest import skipUnless, TestCase
from unittest.mock import MagicMock

import torch
from parameterized import parameterized
from torch.utils._pytree import tree_leaves, tree_structure
from torchft.checkpointing.http_transport import _unflatten_skeleton, HTTPTransport
from torchft.checkpointing.http_transport_bench import main as bench_main
from torchft.checkpointing.transport import CheckpointTransport
from torchft.checkpointing.transport_test import run_multi_recovery_test


class _Custom:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Custom)


class TestHTTPTransport(TestCase):
    @parameterized.expand(
        [
            ("no chunks", 0, True),
            ("chunked", 3, True),
            ("no chunks unsafe", 0, False),
            ("chunked unsafe", 3, False),
        ]
    )
    def test_checkpoint_server(
        self, name: str, num_chunks: int, weights_only: bool
    ) -> None:
        expected: Dict[str, object] = {
            "state": "dict",
            "tensor": torch.rand(5, 2),
            "cuda": torch.rand(
                2, 3, device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            "nested": OrderedDict(
                a={0: torch.rand(3), 1: 1.5},
                b=[torch.rand(1), (2, None)],
            ),
        }
        state_dict_fn = MagicMock()
        state_dict_fn.return_value = expected
        server = HTTPTransport(
            timeout=timedelta(seconds=10),
            num_chunks=num_chunks,
            weights_only=weights_only,
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
        self.assertEqual(tree_structure(out), tree_structure(expected))
        for a, b in zip(tree_leaves(out), tree_leaves(expected)):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a.cpu(), b.cpu())
            else:
                self.assertEqual(a, b)

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

    @parameterized.expand([("no chunks", 0), ("chunked", 3)])
    def test_weights_only(self, name: str, num_chunks: int) -> None:
        state_dict: Dict[str, object] = {"tensor": torch.rand(2), "custom": _Custom()}

        def roundtrip(**kwargs: bool) -> Dict[str, object]:
            server = HTTPTransport(
                timeout=timedelta(seconds=10), num_chunks=num_chunks, **kwargs
            )
            try:
                server.send_checkpoint(
                    dst_ranks=[],
                    step=1,
                    state_dict=state_dict,
                    timeout=timedelta(seconds=10),
                )
                return server.recv_checkpoint(
                    src_rank=0,
                    metadata=server.metadata(),
                    step=1,
                    timeout=timedelta(seconds=10),
                )
            finally:
                server.shutdown()

        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported global"):
            roundtrip()
        self.assertEqual(roundtrip(weights_only=False)["custom"], _Custom())

    def test_unflatten_skeleton(self) -> None:
        skeleton = {"a": [1, (0, ())], "b": OrderedDict(c=2), "d": {}}
        self.assertEqual(
            _unflatten_skeleton(skeleton, ["x", "y", "z"]),
            {"a": ["y", ("x", ())], "b": OrderedDict(c="z"), "d": {}},
        )

        # shared containers would cause exponential traversal
        shared: object = [0]
        for _ in range(64):
            shared = [shared, shared]
        with self.assertRaisesRegex(ValueError, "shared container"):
            _unflatten_skeleton(shared, [1])

        for bad, values in [
            ([0, 0], [1, 2]),
            ([True], [1]),
            ([1], [1]),
            ([-1], [1]),
            ([0], [1, 2]),
            (["a"], [1]),
        ]:
            with self.assertRaisesRegex(ValueError, "bad leaf indices"):
                _unflatten_skeleton(bad, values)

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

    def test_multi_http_transport_cpu(self) -> None:
        device = torch.device("cpu")

        def init(rank: int, world_size: int) -> CheckpointTransport[Dict[str, object]]:
            return HTTPTransport(
                timeout=timedelta(seconds=10),
                num_chunks=0,
            )

        run_multi_recovery_test(self, init, device=device)

    @skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_multi_http_transport_cuda(self) -> None:
        device = torch.device("cuda")

        def init(rank: int, world_size: int) -> CheckpointTransport[Dict[str, object]]:
            return HTTPTransport(
                timeout=timedelta(seconds=10),
                num_chunks=0,
            )

        run_multi_recovery_test(self, init, device=device)

    def test_benchmark(self) -> None:
        bench_main(
            [
                "--chunk-size=10",
                "--num-chunks=0",
                "--total-size=100",
                "--device=cpu",
            ]
        )
