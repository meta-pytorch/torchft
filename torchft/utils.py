# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for TorchFT.
"""

import os
import socket
from contextlib import nullcontext
from typing import cast, Optional, Union

import torch

HOSTNAME_ENV: str = "TORCHFT_HOSTNAME"


def get_stream_context(
    stream: Optional[torch.Stream],
) -> Union[torch.cuda.StreamContext, torch.xpu.StreamContext, nullcontext[None]]:
    """
    Get the appropriate stream context for the given stream.

    This function provides a unified way to handle stream contexts across different
    accelerator types (CUDA, XPU).

    Args:
        stream: The stream to create a context for. If None, returns nullcontext.

    Returns:
        The appropriate stream context for the accelerator type, or nullcontext
        if stream is None or no accelerator is available.
    """
    if stream is not None:
        if torch.cuda.is_available():
            return torch.cuda.stream(cast(torch.cuda.Stream, stream))
        elif torch.xpu.is_available():
            return torch.xpu.stream(cast(torch.xpu.Stream, stream))
        else:
            return nullcontext()
    else:
        return nullcontext()


def record_event() -> Union[torch.cuda.Event, torch.xpu.Event]:
    """
    Record an event in the current stream and return it.

    This function provides a unified way to record events across different
    accelerator types (CUDA, XPU). Callers send the returned event to another
    process so it can wait on this stream's work.
    """
    if torch.xpu.is_available():
        xpu_event = torch.xpu.Event()
        torch.xpu.current_stream().record_event(xpu_event)
        return xpu_event

    cuda_event = torch.cuda.Event(interprocess=True)
    torch.cuda.current_stream().record_event(cuda_event)
    return cuda_event


def synchronize() -> None:
    """
    This function provides a unified way to synchronize current stream across different
    accelerator types (CUDA, XPU).
    """
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    elif torch.xpu.is_available():
        torch.xpu.current_stream().synchronize()


def get_hostname(hostname: Optional[str] = None) -> str:
    """
    Returns the unbracketed hostname or IP address to advertise to peers.

    Resolution priority:
        1. Explicit non-empty ``hostname`` argument.
        2. ``TORCHFT_HOSTNAME`` environment variable.
        3. ``socket.gethostname()``.

    Surrounding IPv6 brackets (e.g. ``[::1]``) are stripped so the returned
    value can be passed directly to APIs such as ``TCPStore`` and formatted
    with :func:`format_host_port`.
    """
    raw = (hostname or "").strip().strip("[]")
    env = os.environ.get(HOSTNAME_ENV, "").strip().strip("[]")
    return raw or env or socket.gethostname()


def format_host_port(host: str, port: int) -> str:
    """
    Formats a host and port as ``host:port``, wrapping IPv6 literals in brackets.
    """
    bare_host = host.strip().strip("[]")
    if ":" in bare_host:
        return f"[{bare_host}]:{port}"
    return f"{bare_host}:{port}"
