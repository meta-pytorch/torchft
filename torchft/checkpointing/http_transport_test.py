# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import urllib.error
from datetime import timedelta
from typing import Dict
from unittest import skipUnless, TestCase
from unittest.mock import MagicMock, patch

import torch
from parameterized import parameterized
from torchft.checkpointing.http_transport import _to_cpu, HTTPTransport
from torchft.checkpointing.http_transport_bench import main as bench_main
from torchft.checkpointing.transport import CheckpointTransport
from torchft.checkpointing.transport_test import (
    assertStateDictEqual,
    run_multi_recovery_test,
)

# The PrivateUse1 backend's current name: "privateuseone", or e.g. "neuron"
# once an out-of-tree backend such as torch-neuronx has renamed it.
_PRIVATEUSE1: str = torch._C._get_privateuse1_backend_name()


def _accelerator_or_cpu() -> torch.device:
    accelerator = torch.accelerator.current_accelerator()
    if torch.accelerator.is_available() and accelerator is not None:
        return accelerator
    return torch.device("cpu")


class TestHTTPTransport(TestCase):
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
            "accelerator": torch.rand(2, 3, device=_accelerator_or_cpu()),
        }
        # Staging copies run on the transport's own stream, so finish writing
        # the tensors first, as the Manager does before every reconfigure.
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
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
        assertStateDictEqual(self, out, expected)

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

    def test_multi_http_transport_cpu(self) -> None:
        device = torch.device("cpu")

        def init(rank: int, world_size: int) -> CheckpointTransport[Dict[str, object]]:
            return HTTPTransport(
                timeout=timedelta(seconds=10),
                num_chunks=0,
            )

        run_multi_recovery_test(self, init, device=device)

    @skipUnless(torch.accelerator.is_available(), "no accelerator is available")
    def test_multi_http_transport_accelerator(self) -> None:
        device = _accelerator_or_cpu()

        def init(rank: int, world_size: int) -> CheckpointTransport[Dict[str, object]]:
            return HTTPTransport(
                timeout=timedelta(seconds=10),
                num_chunks=0,
            )

        run_multi_recovery_test(self, init, device=device)

    def test_to_cpu_stages_any_non_cpu_tensor(self) -> None:
        # A tensor on an arbitrary (e.g. out-of-tree) accelerator must be
        # staged to CPU; TorchFT must not need to know the device type.
        staged = torch.rand(2, 3)
        accelerator_tensor = MagicMock(spec=torch.Tensor)
        accelerator_tensor.device = torch.device(_PRIVATEUSE1)
        accelerator_tensor.cpu.return_value = staged
        cpu_tensor = torch.rand(2)

        out = _to_cpu([accelerator_tensor, cpu_tensor, "not a tensor"], False)

        accelerator_tensor.cpu.assert_called_once_with()
        self.assertIs(out[0], staged)
        self.assertIs(out[1], cpu_tensor)
        self.assertEqual(out[2], "not a tensor")

    def test_no_staging_stream_when_streams_cannot_synchronize(self) -> None:
        # e.g. MPS: Stream.synchronize raises, so staging uses the current stream.
        with patch(
            "torchft.checkpointing.http_transport.synchronizable_stream",
            return_value=None,
        ):
            server = HTTPTransport(timeout=timedelta(seconds=10), num_chunks=0)
        self.assertIsNone(server._stream)
        expected: Dict[str, object] = {"tensor": torch.rand(5, 2)}
        server.send_checkpoint(
            dst_ranks=[], step=1, state_dict=expected, timeout=timedelta(seconds=10)
        )
        out = server.recv_checkpoint(
            src_rank=0,
            metadata=server.metadata(),
            step=1,
            timeout=timedelta(seconds=10),
        )
        assertStateDictEqual(self, out, expected)
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
