# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Immutable observations exchanged at completed training boundaries."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class QuorumSnapshot:
    """One completed, healthy quorum and its native TorchFT progress.

    This observation does not authorize an optimizer update or checkpoint save.
    The training owner must separately establish a complete application state.
    """

    quorum_id: int
    members: Tuple[int, ...]
    step: int
    batches_committed: int
