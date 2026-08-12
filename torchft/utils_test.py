# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from unittest import skipUnless, TestCase

import torch
from torchft.utils import get_stream_context, record_event, synchronize


class UtilsTest(TestCase):
    def test_get_stream_context_none(self) -> None:
        """A None stream yields a nullcontext regardless of accelerator."""
        ctx = get_stream_context(None)
        self.assertIsInstance(ctx, nullcontext)

        with ctx:
            pass

    @skipUnless(torch.accelerator.is_available(), "needs an accelerator")
    def test_get_stream_context_accelerator(self) -> None:
        """A real stream yields a usable device stream context."""
        acc = torch.accelerator.current_accelerator()
        stream = torch.Stream(acc)

        with get_stream_context(stream):
            t = torch.ones(4, device=acc) * 2
            self.assertEqual(t.sum().item(), 8)

        synchronize()

    def test_synchronize_without_accelerator(self) -> None:
        """synchronize() is a no-op rather than an error when no accelerator exists."""
        if torch.accelerator.is_available():
            self.skipTest("accelerator present; covered by the accelerator test")

        synchronize()

    @skipUnless(torch.accelerator.is_available(), "needs an accelerator")
    def test_record_event(self) -> None:
        """record_event() records on the current stream for the active accelerator."""
        record_event()
        synchronize()


@skipUnless(torch.cuda.is_available(), "needs CUDA")
class UtilsCudaTest(TestCase):
    def test_get_stream_context_returns_cuda_context(self) -> None:
        stream = torch.Stream(torch.device("cuda"))
        ctx = get_stream_context(stream)
        self.assertIsInstance(ctx, torch.cuda.StreamContext)


@skipUnless(torch.xpu.is_available(), "needs XPU")
class UtilsXpuTest(TestCase):
    def test_get_stream_context_returns_xpu_context(self) -> None:
        stream = torch.Stream(torch.device("xpu"))
        ctx = get_stream_context(stream)
        self.assertIsInstance(ctx, torch.xpu.StreamContext)

    def test_stream_context_runs_work_on_xpu(self) -> None:
        stream = torch.Stream(torch.device("xpu"))

        with get_stream_context(stream):
            t = torch.arange(8, device="xpu", dtype=torch.float32)
            out = t * 3
            # synchronize() waits on the current stream, so it has to run
            # inside the context while `stream` is still current.
            synchronize()

        self.assertEqual(out.sum().item(), 84.0)
