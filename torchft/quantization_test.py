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

DEVICE = "cuda"

try:
    import triton
except ImportError:
    pass
else:
    from torchft.quantization import (
        _fused_kernel_dequantize_from_fp8,
        _fused_kernel_quantize_into_fp8,
        _fused_kernel_reduce_fp8,
        _get_fp8_max,
        _get_fp8_type,
        _prepare_quantize_fp8,
        BLOCK_SIZE_T,
        fused_dequantize_from_fp8,
        fused_quantize_into_fp8,
        fused_reduce_fp8,
    )

    @skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for this test",
    )
    class QuantizationTest(TestCase):
        def test_large_index_metadata(self) -> None:
            small = torch.ones((1, 1), device=DEVICE, dtype=torch.float16)
            large = small.expand(1_048_576, 2048)
            _, shapes, strides, offsets, _, _, _, _ = _prepare_quantize_fp8(
                [large, small], 1
            )
            self.assertEqual(shapes.dtype, torch.int64)
            self.assertEqual(strides.dtype, torch.int64)
            self.assertEqual(offsets.tolist(), [0, 1_048_576 * 2052])

            # The individual strides fit in int32 but their combined address
            # does not. Meta tensors allow testing without allocating 4 GiB.
            for stride, expected in (
                (2**31 - 2, torch.int32),
                (2**31 - 1, torch.int64),
                (2**31, torch.int64),
            ):
                tensor = torch.empty_strided((2, 2), (stride, 1), device="meta")
                _, shapes, _, _, _, _, _, _ = _prepare_quantize_fp8([tensor], 1)
                self.assertEqual(shapes.dtype, expected)

            # The per-rank buffer is below 2 GiB, while all ranks combined
            # exceed it. The rank multiplication also needs a wide index.
            tensor = torch.empty((1, 1), device="meta").expand(3, 2**30)
            _, shapes, _, _, _, _, _, _ = _prepare_quantize_fp8([tensor], 3)
            self.assertEqual(shapes.dtype, torch.int64)

        def test_non_contiguous_inputs(self) -> None:
            inputs = [
                torch.randn(8, 12, device=DEVICE)[:, ::2],
                torch.randn(6, 8, device=DEVICE).t(),
            ]
            outputs = [torch.empty_like(value) for value in inputs]
            encoded = fused_quantize_into_fp8(inputs, 3)
            fused_dequantize_from_fp8(outputs, encoded, 3)
            for expected, actual in zip(inputs, outputs):
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)

        def test_large_rank_offsets(self) -> None:
            # A rank's offset can exceed 2 GiB even when the per-rank stride
            # fits in int32. Only a few bytes of this buffer are touched.
            stride = 2**30 + 16
            if torch.cuda.mem_get_info()[0] < 3 * 2**30:
                self.skipTest("requires more than 3 GiB of free GPU memory")
            buffer = torch.empty(2 * stride + 8, dtype=torch.uint8, device=DEVICE)
            inputs = torch.tensor([[1.0], [3.0], [5.0]], device=DEVICE)
            outputs = torch.empty_like(inputs)
            ptrs, shapes, strides, offsets, dtype, _, _, _ = _prepare_quantize_fp8(
                [inputs], 3
            )
            prepared = (ptrs, shapes.long(), strides.long(), offsets.long(), dtype)
            ptrs, shapes, strides, offsets, dtype, _, _, _ = _prepare_quantize_fp8(
                [outputs], 3
            )
            prepared_out = (ptrs, shapes.long(), strides.long(), offsets.long(), dtype)

            _fused_kernel_quantize_into_fp8[(3, 1)](
                *prepared[:5],
                buffer,
                stride,
                3,
                WIDE_INDEX=True,
                BLOCK_SIZE=BLOCK_SIZE_T,
                TL_FP8_TYPE=_get_fp8_type(),
                TL_FP8_MAX=_get_fp8_max(),
            )
            _fused_kernel_reduce_fp8[(1, 1)](
                prepared[1],
                prepared[3],
                buffer,
                stride,
                3,
                2,
                3.0,
                WIDE_INDEX=True,
                BLOCK_SIZE=BLOCK_SIZE_T,
                TL_FP8_TYPE=_get_fp8_type(),
                TL_FP8_MAX=_get_fp8_max(),
            )
            _fused_kernel_dequantize_from_fp8[(3, 1)](
                *prepared_out[:5],
                buffer,
                stride,
                3,
                WIDE_INDEX=True,
                BLOCK_SIZE=BLOCK_SIZE_T,
                TL_FP8_TYPE=_get_fp8_type(),
            )
            torch.testing.assert_close(
                outputs[2], inputs.mean(dim=0), rtol=0.02, atol=0.04
            )

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

        # pyrefly: ignore [bad-assignment]
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
