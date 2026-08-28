# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for TorchFT.
"""

from contextlib import nullcontext
from typing import Any, Optional

import torch


def get_stream_context(
    stream: Optional[torch.Stream],
) -> Any:
    """
    Get the appropriate stream context for the given stream.

    This function provides a unified way to handle stream contexts across
    accelerator types.

    Args:
        stream: The stream to create a context for. If None, returns nullcontext.

    Returns:
        The appropriate stream context for the accelerator type, or nullcontext
        if stream is None or no accelerator is available.
    """
    if stream is None or not torch.accelerator.is_available():
        return nullcontext()

    accelerator = torch.accelerator.current_accelerator().type
    accelerator_module = getattr(torch, accelerator, None)
    if accelerator_module is None or not hasattr(accelerator_module, "stream"):
        return nullcontext()
    return accelerator_module.stream(stream)


def record_event() -> None:
    """
    Record an event in the current stream.

    This function provides a unified way to record events across accelerator
    types.
    """
    if not torch.accelerator.is_available():
        return

    accelerator = torch.accelerator.current_accelerator().type
    accelerator_module = getattr(torch, accelerator)
    event = (
        accelerator_module.Event(interprocess=True)
        if accelerator == "cuda"
        else accelerator_module.Event()
    )
    accelerator_module.current_stream().record_event(event)


def synchronize() -> None:
    """
    This function provides a unified way to synchronize the current stream
    across accelerator types.
    """
    if torch.accelerator.is_available():
        accelerator = torch.accelerator.current_accelerator().type
        getattr(torch, accelerator).current_stream().synchronize()
