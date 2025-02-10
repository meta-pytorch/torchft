# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Generic, List, TypeVar

T = TypeVar("T")


class CheckpointTransport(Generic[T], ABC):
    @abstractmethod
    def metadata(self) -> str:
        """
        Returns a string that will be used by the remote CheckpointTransport to fetch the checkpoint.
        """
        ...

    @abstractmethod
    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        """
        Sends the checkpoint, only called when there is a rank that is behind.

        This may be async.

        Args:
            dst_ranks: the ranks to send to
            step: the step number to send
            state_dict: the state dict to send
            timeout: the timeout to wait for the checkpoint to be sent
        """
        ...

    def disallow_checkpoint(self) -> None:
        """
        Called after send_checkpoint to wait for the checkpoint to be sent.

        Once this returns, the state_dict may be mutated so no further data should be sent.
        """
        ...

    @abstractmethod
    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        """
        Receives the checkpoint from the given rank.

        Args:
            src_rank: the rank to receive the checkpoint from
            metadata: the metadata returned by the remote CheckpointTransport
            step: the step number to receive
            timeout: the timeout to wait for the checkpoint
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """
        Called to shutdown the checkpoint transport.

        Args:
            wait: whether to wait for the transport to shutdown
        """
