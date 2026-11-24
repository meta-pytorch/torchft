# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Generator, Optional
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from torchft.utils import (
    _events_cross_processes,
    accelerator_events_cross_processes,
    current_accelerator_type,
    get_stream_context,
    is_accelerator_tensor,
    record_event,
    synchronizable_stream,
    synchronize,
)

# The PrivateUse1 backend's current name: "privateuseone", or e.g. "neuron"
# once an out-of-tree backend such as torch-neuronx has renamed it.
_PRIVATEUSE1: str = torch._C._get_privateuse1_backend_name()


@contextmanager
def _mock_accelerator(
    device_type: Optional[str], module: Optional[object] = None
) -> Generator[None, None, None]:
    """
    Pretends that ``device_type`` is the current accelerator (or that no
    accelerator is available if it is None) and installs ``module`` as
    ``torch.<device_type>``.
    """
    with (
        patch(
            "torchft.utils.torch.accelerator.is_available",
            return_value=device_type is not None,
        ),
        patch(
            "torchft.utils.torch.accelerator.current_accelerator",
            return_value=(
                SimpleNamespace(type=device_type) if device_type is not None else None
            ),
        ),
    ):
        if module is None:
            yield
        else:
            with patch.object(torch, str(device_type), module, create=True):
                yield


class TestRecordEvent(TestCase):
    def test_returns_the_event_it_recorded(self) -> None:
        """Regression test: `record_event` used to return None, so the process
        receiving the event had nothing to wait on."""
        if not torch.accelerator.is_available():
            self.skipTest("needs an accelerator")
        event = record_event()

        self.assertIsNotNone(event, "record_event must return the event it recorded")
        if torch.cuda.is_available():
            if not isinstance(event, torch.cuda.Event):
                raise AssertionError(f"Expected a CUDA event, got {type(event)}")
            # a recorded interprocess event must be able to produce an IPC handle
            self.assertIsNotNone(event.ipc_handle())
        event.synchronize()
        self.assertTrue(event.query())

    def test_uses_current_accelerator_module(self) -> None:
        accelerator_module = MagicMock()
        event = accelerator_module.Event.return_value

        with _mock_accelerator("test_accel", accelerator_module):
            self.assertIs(record_event(), event)

        accelerator_module.Event.assert_called_once_with()
        accelerator_module.current_stream.return_value.record_event.assert_called_once_with(
            event
        )

    def test_keeps_cuda_interprocess_event(self) -> None:
        with (
            _mock_accelerator("cuda"),
            patch.object(torch.cuda, "Event") as event_cls,
            patch.object(torch.cuda, "current_stream") as current_stream,
        ):
            self.assertIs(record_event(), event_cls.return_value)

        event_cls.assert_called_once_with(interprocess=True)
        current_stream.return_value.record_event.assert_called_once_with(
            event_cls.return_value
        )

    def test_falls_back_to_generic_event_and_stream(self) -> None:
        # A device module without Event/current_stream uses torch.Event and
        # torch.accelerator.current_stream.
        accelerator_module = SimpleNamespace()
        with (
            _mock_accelerator("test_accel", accelerator_module),
            patch("torchft.utils.torch.Event") as event_cls,
            patch("torchft.utils.torch.accelerator.current_stream") as current_stream,
        ):
            self.assertIs(record_event(), event_cls.return_value)

        event_cls.assert_called_once_with(device="test_accel")
        current_stream.return_value.record_event.assert_called_once_with(
            event_cls.return_value
        )

    def test_requires_accelerator(self) -> None:
        with _mock_accelerator(None):
            with self.assertRaisesRegex(RuntimeError, "requires an accelerator"):
                record_event()


class TestStreamContext(TestCase):
    def test_without_accelerator_returns_nullcontext(self) -> None:
        with _mock_accelerator(None):
            self.assertIsInstance(get_stream_context(MagicMock()), nullcontext)

    def test_without_stream_returns_nullcontext(self) -> None:
        with _mock_accelerator("test_accel", MagicMock()):
            self.assertIsInstance(get_stream_context(None), nullcontext)

    def test_uses_current_accelerator_module(self) -> None:
        stream = MagicMock()
        context = MagicMock()
        accelerator_module = MagicMock()
        accelerator_module.stream.return_value = context

        with _mock_accelerator("test_accel", accelerator_module):
            self.assertIs(get_stream_context(stream), context)

        accelerator_module.stream.assert_called_once_with(stream)

    def test_falls_back_to_generic_stream_context(self) -> None:
        # torch.Stream is itself a context manager; use it when the device
        # module has no stream() helper.
        stream = torch.Stream()
        with _mock_accelerator("test_accel", SimpleNamespace()):
            self.assertIs(get_stream_context(stream), stream)


class TestSynchronize(TestCase):
    def test_uses_current_accelerator_stream(self) -> None:
        accelerator_module = MagicMock()

        with _mock_accelerator("test_accel", accelerator_module):
            synchronize()

        accelerator_module.current_stream.return_value.synchronize.assert_called_once_with()

    def test_without_accelerator_is_noop(self) -> None:
        with (
            _mock_accelerator(None),
            patch("torchft.utils.torch.accelerator.current_stream") as current_stream,
        ):
            synchronize()

        current_stream.assert_not_called()


class TestAcceleratorDetection(TestCase):
    def test_current_accelerator_type(self) -> None:
        with _mock_accelerator("test_accel"):
            self.assertEqual(current_accelerator_type(), "test_accel")
        with _mock_accelerator(None):
            self.assertIsNone(current_accelerator_type())

    def test_is_accelerator_tensor(self) -> None:
        accelerator_tensor = MagicMock(spec=torch.Tensor)
        accelerator_tensor.device = torch.device(_PRIVATEUSE1)

        with _mock_accelerator(_PRIVATEUSE1):
            self.assertTrue(is_accelerator_tensor(accelerator_tensor))
            self.assertFalse(is_accelerator_tensor(torch.zeros(1)))
            self.assertFalse(is_accelerator_tensor("not a tensor"))
        with _mock_accelerator(None):
            self.assertFalse(is_accelerator_tensor(accelerator_tensor))


class TestAcceleratorEventsCrossProcesses(TestCase):
    def setUp(self) -> None:
        # The answer is cached per device type; the tests reuse type names.
        _events_cross_processes.cache_clear()

    def test_cuda_events_have_a_reducer(self) -> None:
        with _mock_accelerator("cuda"):
            self.assertTrue(accelerator_events_cross_processes())

    def test_module_event_without_a_reducer(self) -> None:
        class _Event:
            pass

        with _mock_accelerator("test_accel", SimpleNamespace(Event=_Event)):
            self.assertFalse(accelerator_events_cross_processes())

    def test_generic_event_cannot_be_pickled(self) -> None:
        # No device module: record_event falls back to a plain torch.Event,
        # which torch.multiprocessing cannot pickle: torch 2.14 has no reducer
        # for it (TypeError), and newer torch pickles only interprocess events
        # (RuntimeError).
        with _mock_accelerator("test_accel"):
            self.assertFalse(accelerator_events_cross_processes())
        from multiprocessing.reduction import ForkingPickler

        import torch.multiprocessing  # noqa: F401

        try:
            event = torch.Event()
        except RuntimeError:
            self.skipTest("cannot create a torch.Event here")
        with self.assertRaises((TypeError, RuntimeError)):
            ForkingPickler.dumps(event)

    def test_no_accelerator(self) -> None:
        with _mock_accelerator(None):
            self.assertFalse(accelerator_events_cross_processes())


class TestSynchronizableStream(TestCase):
    def test_no_accelerator(self) -> None:
        with _mock_accelerator(None):
            self.assertIsNone(synchronizable_stream())

    def test_stream_that_synchronizes(self) -> None:
        stream = MagicMock()
        with (
            _mock_accelerator("test_accel"),
            patch("torchft.utils.torch.Stream", return_value=stream),
        ):
            self.assertIs(synchronizable_stream(), stream)
        stream.synchronize.assert_called_once_with()

    def test_stream_that_cannot_synchronize(self) -> None:
        # MPS: "Backend doesn't support synchronizing streams."
        stream = MagicMock()
        stream.synchronize.side_effect = RuntimeError(
            "Backend doesn't support synchronizing streams."
        )
        with (
            _mock_accelerator("mps"),
            patch("torchft.utils.torch.Stream", return_value=stream),
        ):
            self.assertIsNone(synchronizable_stream())
