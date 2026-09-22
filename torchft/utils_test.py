# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from torchft.utils import get_stream_context, record_event, synchronize


class UtilsTest(TestCase):
    @patch("torchft.utils.torch.accelerator.is_available", return_value=False)
    def test_stream_context_without_accelerator_returns_nullcontext(
        self, _is_available: MagicMock
    ) -> None:
        self.assertIsInstance(get_stream_context(MagicMock()), nullcontext)

    @patch("torchft.utils.torch.accelerator.current_accelerator")
    @patch("torchft.utils.torch.accelerator.is_available", return_value=True)
    def test_stream_context_uses_current_accelerator_module(
        self, _is_available: MagicMock, current_accelerator: MagicMock
    ) -> None:
        current_accelerator.return_value = SimpleNamespace(type="test_accel")
        stream = MagicMock()
        context = MagicMock()
        accelerator_module = MagicMock()
        accelerator_module.stream.return_value = context

        with patch.object(torch, "test_accel", accelerator_module, create=True):
            self.assertIs(get_stream_context(stream), context)

        accelerator_module.stream.assert_called_once_with(stream)

    @patch("torchft.utils.torch.accelerator.current_accelerator")
    @patch("torchft.utils.torch.accelerator.is_available", return_value=True)
    def test_record_event_uses_current_accelerator_module(
        self, _is_available: MagicMock, current_accelerator: MagicMock
    ) -> None:
        current_accelerator.return_value = SimpleNamespace(type="test_accel")
        accelerator_module = MagicMock()
        event = accelerator_module.Event.return_value

        with patch.object(torch, "test_accel", accelerator_module, create=True):
            record_event()

        accelerator_module.Event.assert_called_once_with()
        accelerator_module.current_stream.return_value.record_event.assert_called_once_with(
            event
        )

    @patch("torchft.utils.torch.accelerator.current_accelerator")
    @patch("torchft.utils.torch.accelerator.is_available", return_value=True)
    def test_record_event_keeps_cuda_interprocess_event(
        self, _is_available: MagicMock, current_accelerator: MagicMock
    ) -> None:
        current_accelerator.return_value = SimpleNamespace(type="cuda")

        with patch.object(torch.cuda, "Event") as event_cls:
            with patch.object(torch.cuda, "current_stream") as current_stream:
                record_event()

        event_cls.assert_called_once_with(interprocess=True)
        current_stream.return_value.record_event.assert_called_once_with(
            event_cls.return_value
        )

    @patch("torchft.utils.torch.accelerator.current_accelerator")
    @patch("torchft.utils.torch.accelerator.is_available", return_value=True)
    def test_synchronize_uses_current_accelerator_stream(
        self, _is_available: MagicMock, current_accelerator: MagicMock
    ) -> None:
        current_accelerator.return_value = SimpleNamespace(type="test_accel")
        accelerator_module = MagicMock()

        with patch.object(torch, "test_accel", accelerator_module, create=True):
            synchronize()

        accelerator_module.current_stream.return_value.synchronize.assert_called_once_with()
