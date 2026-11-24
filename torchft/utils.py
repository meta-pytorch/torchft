# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for TorchFT.

The helpers in this module dispatch through :mod:`torch.accelerator` so that
any accelerator registered with PyTorch (CUDA, XPU, or an out-of-tree
``PrivateUse1`` backend) is supported without TorchFT naming the device.
"""

import functools
import io
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Optional

import torch


def current_accelerator_type() -> Optional[str]:
    """
    Returns the device type of the current accelerator (e.g. ``"cuda"``), or
    ``None`` if no accelerator is available.
    """
    if not torch.accelerator.is_available():
        return None
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        return None
    return accelerator.type


def is_accelerator_tensor(obj: object) -> bool:
    """
    Returns true if ``obj`` is a tensor on the current accelerator.
    """
    if not isinstance(obj, torch.Tensor):
        return False
    device_type = current_accelerator_type()
    return device_type is not None and obj.device.type == device_type


def _accelerator_module(device_type: str) -> Optional[Any]:
    """
    Returns the ``torch.<device_type>`` device module for the accelerator, if
    one is registered. In-tree accelerators provide e.g. ``torch.cuda``;
    out-of-tree backends register theirs via ``torch._register_device_module``.
    """
    return getattr(torch, device_type, None)


def _current_stream(device_type: str) -> Any:
    """
    Returns the current stream of the accelerator, preferring the device
    module's ``current_stream()`` (which may return a device-specific stream
    subclass) and falling back to :func:`torch.accelerator.current_stream`.
    """
    module = _accelerator_module(device_type)
    if module is not None and hasattr(module, "current_stream"):
        return module.current_stream()
    return torch.accelerator.current_stream()


def get_stream_context(
    stream: Optional[torch.Stream],
) -> AbstractContextManager[Any]:
    """
    Get the appropriate stream context for the given stream.

    This function provides a unified way to handle stream contexts across
    accelerator types. The device module's ``stream()`` context manager is
    used when available (e.g. ``torch.cuda.stream``); otherwise the generic
    :class:`torch.Stream` context manager is used.

    Args:
        stream: The stream to create a context for. If None, returns nullcontext.

    Returns:
        The appropriate stream context for the accelerator type, or nullcontext
        if stream is None or no accelerator is available.
    """
    if stream is None:
        return nullcontext()

    device_type = current_accelerator_type()
    if device_type is None:
        return nullcontext()

    module = _accelerator_module(device_type)
    if module is not None and hasattr(module, "stream"):
        return module.stream(stream)
    if isinstance(stream, AbstractContextManager):
        return stream
    return nullcontext()


def _event_type(device_type: str) -> type:
    """The event class :func:`record_event` creates for ``device_type``."""
    if device_type == "cuda":
        return torch.cuda.Event
    module = _accelerator_module(device_type)
    if module is not None and hasattr(module, "Event"):
        return module.Event
    return torch.Event


def accelerator_events_cross_processes() -> bool:
    """
    Whether the current accelerator's events can be sent to another process.

    An event crosses a process boundary only if ``torch.multiprocessing`` has
    a reducer for its type. In torch 2.14 only ``torch.cuda.Event`` has one;
    the generic :class:`torch.Event` cannot be pickled.
    """
    device_type = current_accelerator_type()
    if device_type is None:
        return False
    return _events_cross_processes(device_type)


@functools.cache
def _events_cross_processes(device_type: str) -> bool:
    from multiprocessing.reduction import ForkingPickler

    import torch.multiprocessing  # noqa: F401  registers torch's reducers

    return _event_type(device_type) in ForkingPickler(io.BytesIO()).dispatch_table


def synchronizable_stream() -> Optional[torch.Stream]:
    """
    A new stream on the current accelerator, or None if there is no accelerator
    or its streams can't be synchronized (for example MPS, where
    ``Stream.synchronize`` raises "Backend doesn't support synchronizing
    streams").
    """
    if not torch.accelerator.is_available():
        return None
    try:
        stream = torch.Stream(torch.accelerator.current_accelerator())
        stream.synchronize()
    except (RuntimeError, NotImplementedError):
        return None
    return stream


def record_event() -> Any:
    """
    Record an event in the current stream of the current accelerator and
    return it.

    This function provides a unified way to record events across accelerator
    types. Callers send the returned event to another process so it can wait
    on this stream's work. On CUDA an interprocess event is created so that it
    can be shared over IPC; other accelerators use the event type provided by
    their device module, or :class:`torch.Event` if there is none.

    Raises:
        RuntimeError: if no accelerator is available.
    """
    device_type = current_accelerator_type()
    if device_type is None:
        raise RuntimeError("record_event requires an accelerator")

    event_type = _event_type(device_type)
    if device_type == "cuda":
        event = torch.cuda.Event(interprocess=True)
    elif event_type is torch.Event:
        event = torch.Event(device=device_type)
    else:
        event = event_type()

    _current_stream(device_type).record_event(event)
    return event


def synchronize() -> None:
    """
    This function provides a unified way to synchronize the current stream
    across accelerator types. It is a no-op if no accelerator is available.
    """
    device_type = current_accelerator_type()
    if device_type is not None:
        _current_stream(device_type).synchronize()
