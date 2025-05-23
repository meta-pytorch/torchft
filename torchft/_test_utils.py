# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def any_nan(ts: list[torch.Tensor]) -> bool:
    for t in ts:
        if torch.isnan(t).any():
            return True
    return False


def combine_views(
    views: list[list[tuple[int, ...]]],
    combinations: list[list[tuple[int, ...]]],
    tmp: list[tuple[int, ...]],
    i: int,
) -> None:
    if i == len(views):
        combinations.append(tmp.copy())
        return

    for j in range(len(views[i])):
        tmp.append(views[i][j])
        combine_views(views, combinations, tmp, i + 1)
        tmp.pop()


def gen_views(inp: torch.Tensor) -> list[tuple[int, ...]]:
    size = inp.numel()

    views = []
    for m in range(1 if size % 2 == 0 else 2, size):
        if size % m == 0:
            views.append((m, size // m))

    return views


def gen_splits(inp: torch.Tensor, split_size: int) -> list[list[tuple[int, ...]]]:
    views = []

    for split in torch.split(inp, split_size):
        views.append(gen_views(split))

    combinations = []
    combine_views(views, combinations, [], 0)

    return combinations
