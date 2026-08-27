# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import skipUnless, TestCase

import torch
from parameterized import parameterized
from torch.distributed import ReduceOp
from torchft import _test_utils

torch.set_printoptions(precision=4, sci_mode=False)

# Resolved per-accelerator so the same test body covers CUDA and XPU.
DEVICE: str = (
    str(torch.accelerator.current_accelerator())
    if torch.accelerator.is_available()
    else "cpu"
)

try:
    # pyre-fixme[21]: Could not find a module corresponding to import `triton`
    import triton
except ImportError:
    pass
else:
    from torchft.quantization import (
        _supports_native_fp8,
        fused_dequantize_from_fp8,
        fused_quantize_into_fp8,
        fused_reduce_fp8,
    )

    @skipUnless(
        torch.accelerator.is_available(),
        "An accelerator (CUDA or XPU) is required for this test",
    )
    class QuantizationTest(TestCase):
        def run_test(
            self,
            world_size: int,
            tensors_num: int,
            tensor_size: int,
            multiplier: float,
            tolerance: float,
            reduce_op: ReduceOp,
            type: torch.dtype,
        ) -> None:
            inp = (
                torch.rand(
                    tensors_num * tensor_size,
                    dtype=type,
                    device=DEVICE,
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

                    fused_reduce_fp8(
                        reshaped_inputs, quant_copy, world_size, r, reduce_op
                    )

                    quant_final_slices[r].copy_(quant_copy_slices[r])

                fused_dequantize_from_fp8(reshaped_outputs, quant_final, world_size)

                self.assertFalse(_test_utils.any_nan(reshaped_outputs))

                if reduce_op == ReduceOp.SUM:
                    inputs.mul_(world_size)

                diff = torch.abs(
                    (inputs - outputs).div(inputs.to(torch.float32) + 0.0000001)
                )
                mean_diff = diff.mean().item()
                self.assertLessEqual(
                    mean_diff, tolerance, f"Results not within tolerance {tolerance}"
                )

        END_TO_END_CONFIGS: list[tuple[int, float, ReduceOp, torch.dtype]] = [
            (ts, m, o, t)
            for ts in [128, 512, 4096]
            for m in [1.0, 100.0, 1000.0]
            for o in [ReduceOp.AVG, ReduceOp.SUM]
            for t in [torch.float32, torch.float16, torch.bfloat16]
        ]

        @parameterized.expand(END_TO_END_CONFIGS)
        def test_end_to_end(
            self,
            tensor_size: int,
            multiplier: float,
            reduce_op: ReduceOp,
            type: torch.dtype,
        ) -> None:
            self.run_test(
                world_size=2,
                tensors_num=3,
                tensor_size=tensor_size,
                multiplier=multiplier,
                tolerance=0.05,
                reduce_op=reduce_op,
                type=type,
            )

    @skipUnless(torch.xpu.is_available(), "XPU is required for this test")
    class QuantizationXpuTest(TestCase):
        def test_no_native_fp8_on_xpu(self) -> None:
            """XPU has no native FP8, so the kernels must take the int8 path."""
            self.assertFalse(_supports_native_fp8())

        @parameterized.expand(
            [
                (torch.float32,),
                (torch.float16,),
                (torch.bfloat16,),
            ]
        )
        def test_quantize_dequantize_roundtrip(self, dtype: torch.dtype) -> None:
            """The int8 fallback round-trips XPU tensors within tolerance."""
            world_size = 2
            tensor_size = 512
            inp = torch.rand(2 * tensor_size, dtype=dtype, device="xpu")

            for split in _test_utils.gen_splits(inp, tensor_size):
                inputs = inp.clone()
                outputs = torch.empty_like(inputs)

                tensors_in = [
                    i.view(*s) for s, i in zip(split, torch.split(inputs, tensor_size))
                ]
                tensors_out = [
                    o.view(*s) for s, o in zip(split, torch.split(outputs, tensor_size))
                ]

                quant = fused_quantize_into_fp8(tensors_in, world_size)
                fused_dequantize_from_fp8(tensors_out, quant, world_size)

                torch.xpu.synchronize()

                self.assertFalse(_test_utils.any_nan(tensors_out))
                diff = torch.abs(
                    (inputs - outputs).div(inputs.to(torch.float32) + 0.0000001)
                )
                self.assertLessEqual(diff.mean().item(), 0.05)
