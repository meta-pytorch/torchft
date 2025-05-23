# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Callable
from unittest import TestCase, skipUnless

import torch
from parameterized import parameterized
from torch import cuda
from torch.distributed import AllreduceOptions, ReduceOp
from torch.distributed.distributed_c10d import ReduceOp

from torchft import _test_utils
from torchft.process_group import ProcessGroup
from torchft.process_group_test import MultiPgBaseTest

try:
    # pyre-ignore[21]: Could not find a module corresponding to import `triton`
    import triton
except ImportError:
    pass
else:
    from torchft.collectives import allreduce_quantized

    @skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "2 CUDA devices are required for this test",
    )
    class QuantizedAllReduceTest(MultiPgBaseTest):
        BACKEND = "nccl"
        WORLD_SIZE = 2

        def _run_parallel_collectives(
            self, collective: Callable[[ProcessGroup, int, str], None]
        ) -> None:
            futures = []
            for rank in range(self.WORLD_SIZE):
                pg = self.pg_pool[rank]
                device = f"cuda:{rank}"
                fut = self.executor.submit(collective, pg, rank, device)
                futures.append(fut)

            self._collect(futures)

        def _run_collective(
            self,
            pg: ProcessGroup,
            rank: int,
            device: str,
            tensors_num: int,
            tensor_size: int,
            multiplier: float,
            tolerance: float,
        ) -> None:
            cuda.set_device(device)
            inp = (
                torch.rand(
                    tensors_num * tensor_size,
                    dtype=torch.float32,
                    device=device,
                )
                * multiplier
            )
            for split in _test_utils.gen_splits(inp, tensor_size):
                actual = inp.clone()
                expected = inp.clone()
                tensors = [
                    i.view(*s)
                    for s, i in zip(
                        split,
                        torch.split(actual, tensor_size),
                    )
                ]

                fut = allreduce_quantized(tensors, ReduceOp.AVG, pg)
                fut.wait()

                work = pg.allreduce([expected], ReduceOp.AVG)
                work.get_future().wait()

                diff = torch.abs((expected - actual).div(expected))
                mean_diff = diff.mean().item()

                if mean_diff > tolerance:
                    raise AssertionError(f"Results not within tolerance {tolerance}")

        END_TO_END_CONFIGS: list[tuple[int, float]] = [
            (ts, m)
            for ts in [128, 512, 1024, 2048, 4096]
            for m in [1.0, 10.0, 100.0, 1000.0]
        ]

        @parameterized.expand(END_TO_END_CONFIGS)
        def test_collective(self, tensor_size: int, multiplier: float) -> None:
            self._run_parallel_collectives(
                lambda pg, rank, device: self._run_collective(
                    pg,
                    rank,
                    device,
                    3,
                    tensor_size,
                    multiplier,
                    3.0,
                )
            )
