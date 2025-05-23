# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def any_nan(ts: list[torch.Tensor]) -> bool:
    """
    Check if any tensor in the list contains NaN values.

    Args:
        ts: List of tensors to check for NaN values

    Returns:
        True if any tensor contains NaN values, False otherwise
    """
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
    """
    Recursively generate all possible combinations of views from a list of
    lists of views.

    This function uses backtracking to generate all possible combinations by
    selecting each list in the input. The results are stored in the
    combinations list.

    Args:
        views: A list of lists, where each inner list contains possible view
            shapes (tuples)
        combinations: Output list where all combinations will be stored
        tmp: Temporary list to build the current combination
        i: Current index in the views list being processed

    Returns:
        None. Results are stored in the combinations list passed as
        an argument.
    """
    if i == len(views):
        combinations.append(tmp.copy())
        return

    for j in range(len(views[i])):
        tmp.append(views[i][j])
        combine_views(views, combinations, tmp, i + 1)
        tmp.pop()


def gen_views(inp: torch.Tensor) -> list[tuple[int, ...]]:
    """
    Generate all possible 2D views (shapes) for a tensor with a given number
    of elements.

    This function finds all pairs of integers (m, n) such that m * n equals the
    total number of elements in the input tensor. These pairs represent possible
    2D shapes that the tensor can be reshaped into.

    Args:
        inp: Input tensor

    Returns:
        A list of tuples, where each tuple (m, n) represents a possible 2D shape
        such that m * n equals the total number of elements in the input tensor
    """
    size = inp.numel()

    views = []
    for m in range(1 if size % 2 == 0 else 2, size):
        if size % m == 0:
            views.append((m, size // m))

    return views


def gen_splits(inp: torch.Tensor, split_size: int) -> list[list[tuple[int, ...]]]:
    """
    Split a tensor into chunks and generate all possible combinations of views.

    This function first splits the input tensor into chunks of the specified size,
    then generates all possible 2D views for each chunk, and finally computes all
    possible combinations of these views across all chunks.

    Args:
        inp: Input tensor to be split
        split_size: Size of each chunk

    Returns:
        A list of lists, where each inner list contains a combination of view
        shapes, one for each chunk of the input tensor
    """
    views = []

    for split in torch.split(inp, split_size):
        views.append(gen_views(split))

    combinations = []
    combine_views(views, combinations, [], 0)

    return combinations
