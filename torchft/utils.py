# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for TorchFT.
"""

from contextlib import nullcontext
from typing import Any, ContextManager, Optional

import torch


def get_stream_context(
    stream: Optional[torch.Stream],
) -> ContextManager[None]:
    """
    Get the appropriate stream context for the given stream.

    This function provides a unified way to handle stream contexts across different
    accelerator types (CUDA, XPU, etc.).

    Args:
        stream: The stream to create a context for. If None, returns nullcontext.

    Returns:
        The appropriate stream context for the accelerator type, or nullcontext
        if stream is None or no accelerator is available.
    """
    if stream is not None:
        device_type = stream.device.type
        if device_type == "cuda" and torch.cuda.is_available():
            # pyre-fixme[6]: Expected `Optional[streams.Stream]` but got `_C.Stream`
            return torch.cuda.stream(stream)
        elif device_type == "xpu" and torch.xpu.is_available():
            # pyre-fixme[6]: Expected `Optional[streams.Stream]` but got `_C.Stream`
            return torch.xpu.stream(stream)
        else:
            return nullcontext()
    else:
        return nullcontext()


def record_event() -> None:
    """
    Record an event in the current stream.

    This function provides a unified way to record events across different
    accelerator types.
    """
    if torch.accelerator.is_available():
        stream = torch.accelerator.current_stream()
        if stream.device.type == "cuda":
            stream.record_event(torch.cuda.Event(interprocess=True))
        elif stream.device.type == "xpu":
            stream.record_event(torch.xpu.Event())
        else:
            # For other accelerator types, try generic approach
            pass


def synchronize() -> None:
    """
    This function provides a unified way to synchronize current stream across different
    accelerator types.
    """
    if torch.accelerator.is_available():
        torch.accelerator.current_stream().synchronize()
