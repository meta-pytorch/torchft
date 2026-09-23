# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import torch
from torchft.utils import record_event


class TestRecordEvent(TestCase):
    def test_returns_the_event_it_recorded(self) -> None:
        """Regression test: `record_event` used to return None, so the process
        receiving the event had nothing to wait on."""
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        event = record_event()

        self.assertIsNotNone(event, "record_event must return the event it recorded")
        if not isinstance(event, torch.cuda.Event):
            raise AssertionError(f"Expected a CUDA event, got {type(event)}")
        # a recorded interprocess event must be able to produce an IPC handle
        self.assertIsNotNone(event.ipc_handle())
        event.synchronize()
        self.assertTrue(event.query())
