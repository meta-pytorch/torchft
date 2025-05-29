# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase, skipUnless

import torch
from parameterized import parameterized

from torchft import _test_utils

torch.set_printoptions(precision=4, sci_mode=False)

DEVICE = "cuda"

try:
    # pyre-fixme[21]: Could not find a module corresponding to import `triton`
    import triton
except ImportError:
    pass
else:
    from torchft.quantization import (
        fused_dequantize_from_fp8,
        fused_quantize_into_fp8,
        fused_reduce_fp8,
    )

    @skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for this test",
    )
    class QuantizationTest(TestCase):

        def run_test(
            self,
            world_size: int,
            tensors_num: int,
            tensor_size: int,
            multiplier: float,
            tolerance: float,
        ) -> None:
            inp = (
                torch.rand(
                    tensors_num * tensor_size,
                    dtype=torch.float32,
                    device="cuda",
                )
                * multiplier
            )

            for split in _test_utils.gen_splits(inp, tensor_size):
                inputs = inp.clone()
                outputs = torch.empty_like(inputs)

                reshaped_inputs = []
                reshaped_outputs = []
                for s, i, o in zip(
                    split,
                    torch.split(inputs, tensor_size),
                    torch.split(outputs, tensor_size),
                ):
                    reshaped_inputs.append(i.view(*s))
                    reshaped_outputs.append(o.view(*s))

                quant = fused_quantize_into_fp8(reshaped_inputs, world_size)
                quant_slices = torch.split(quant, quant.numel() // world_size)

                quant_final = torch.empty_like(quant)
                quant_final_slices = torch.split(
                    quant_final, quant_final.numel() // world_size
                )

                for rank in range(world_size):
                    r = (rank) % world_size
                    quant_copy = torch.empty_like(quant)
                    quant_copy_slices = torch.split(
                        quant_copy, quant_copy.numel() // world_size
                    )
                    for other in range(world_size):
                        quant_copy_slices[other].copy_(quant_slices[r])

                    fused_reduce_fp8(reshaped_inputs, quant_copy, world_size, r)

                    quant_final_slices[r].copy_(quant_copy_slices[r])

                fused_dequantize_from_fp8(reshaped_outputs, quant_final, world_size)

                self.assertFalse(_test_utils.any_nan(reshaped_outputs))

                diff = torch.abs((inputs - outputs).div(inputs))
                mean_diff = diff.mean().item()
                self.assertLessEqual(
                    mean_diff, tolerance, f"Results not within tolerance {tolerance}"
                )

        END_TO_END_CONFIGS: list[tuple[int, float]] = [
            (ts, m)
            for ts in [128, 256, 512, 1024, 2048, 4096]
            for m in [1.0, 10.0, 100.0, 1000.0]
        ]

        @parameterized.expand(END_TO_END_CONFIGS)
        def test_end_to_end(self, tensor_size: int, multiplier: float) -> None:
            self.run_test(
                world_size=2,
                tensors_num=4,
                tensor_size=tensor_size,
                multiplier=multiplier,
                tolerance=3.0,
            )
