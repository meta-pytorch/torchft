# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import torch
import torch.cuda as cuda

# pyre-ignore[21]: Could not find a module corresponding to import `triton`
import triton

# pyre-ignore[21]: Could not find a module corresponding to import `triton.language`
import triton.language as tl

# pyre-ignore[21]: Could not find a module corresponding to import `triton.runtime`
import triton.runtime as tr
from torch.distributed import ReduceOp

SCALE_DTYPE: torch.dtype = torch.float32
SCALE_DTYPE_BYTES: int = 4
SCALE_TL_DTYPE = tl.float32
SCALE_TL_DTYPE_BYTES = tl.constexpr(4)

BLOCK_SIZE_T: int = 2048


# pyre-ignore[11]: Annotation `tl.constexpr` is not defined
def _get_fp8_max() -> tl.constexpr:
    if cuda.get_device_capability() >= (9, 0):
        return tl.constexpr(448.0)
    else:
        return tl.constexpr(127)


def _get_fp8_type() -> tl.constexpr:
    if cuda.get_device_capability() >= (9, 0):
        return tl.constexpr(tl.float8e4nv)
    else:
        return tl.constexpr(tl.int8)


@triton.jit
# pyre-ignore[11]: Annotation `tl.tensor` is not defined
def _kernel_calculate_scale(row_max, TL_FP8_MAX: tl.constexpr) -> tl.tensor:
    row_scale = TL_FP8_MAX / row_max
    is_inf = row_scale == float("inf")
    row_scale = tl.where(is_inf, 1.0, row_scale)
    return row_scale


@triton.jit
def _fused_kernel_quantize_into_fp8(
    i_ptrs,
    i_shapes,
    i_strides,
    i_offsets,
    i_dtype,
    o_ptr,
    o_size_bytes_per_rank,
    all_reduce_size,
    BLOCK_SIZE: tl.constexpr,
    TL_FP8_TYPE: tl.constexpr,
    TL_FP8_MAX: tl.constexpr,
):
    """
    Kernel to quantize a set of input tensors into fp8. The input tensors are
    expected to be two-dimensional and the output tensor is expected to be
    one-dimensional. The output tensor is expected to be large enough to hold
    the quantized input and scales for all input tensors. The quantized input
    and scales are interleaved in the output tensor. The quantized input
    is stored as fp8 and the scales are stored as fp32.

    Args:
        i_ptrs: Pointers to the input tensors to be quantized
        i_shapes: Shapes of the input tensors to be quantized
        i_strides: Strides of the input tensors to be quantized
        i_offsets: Offsets of the output tensors for each input tensor
        i_dtype: Dummy tensor that carries the dtype of the input tensors
        o_ptr: Pointer to the output tensor for the quantized input and scales
        o_size_bytes_per_rank: Size in bytes in the output tensor per rank
        all_reduce_size: Size of the all-reduce group
        BLOCK_SIZE: Block size for the quantization
        NUM_SM: Number of SMs to use for the quantization
    """
    # Index of the row in the input tensor
    i_row_idx = tl.program_id(0)
    # Index of the input tensor
    i_idx = tl.program_id(1)

    # Number of rows and colums in the input tensor
    i_rows_num = tl.load(i_shapes + i_idx * 2)
    if i_row_idx >= i_rows_num:
        return
    i_cols_num = tl.load(i_shapes + i_idx * 2 + 1)

    # Stride to advance by a single row and column in the input tensor
    # assume contiguous tensors
    i_row_stride = tl.load(i_strides + i_idx * 2)
    i_col_stride = tl.load(i_strides + i_idx * 2 + 1)

    # Pointer to the input tensor
    i_ptr = tl.load(i_ptrs + i_idx).to(i_dtype.dtype)

    # Number of the rows in the input tensor that are processed by a single
    # rank
    i_row_slice_size = tl.cdiv(i_rows_num, all_reduce_size)
    # Index of the row slice in the input tensor
    i_row_slice_idx = i_row_idx // i_row_slice_size

    # Size in bytes of a single input tensor row quantized and written to the
    # output tensor
    o_row_size_bytes = (
        tl.cdiv(i_cols_num, SCALE_TL_DTYPE_BYTES) + 1
    ) * SCALE_TL_DTYPE_BYTES

    # Pointer to the output tensor where
    o_offset = (
        o_size_bytes_per_rank * i_row_slice_idx
        + tl.load(i_offsets + i_idx)
        + (i_row_idx % i_row_slice_size) * o_row_size_bytes
    )
    # Pointer to the output tensor where the scale and quantized row will
    # be written
    o_curr_ptr = o_ptr + o_offset
    o_scale_ptr = o_curr_ptr.to(tl.pointer_type(SCALE_TL_DTYPE))
    o_quant_ptr = (o_curr_ptr + SCALE_TL_DTYPE_BYTES).to(tl.pointer_type(TL_FP8_TYPE))  # type: ignore

    # Compute maximum for the current row block by block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_maxes = tl.full((BLOCK_SIZE,), 0, dtype=tl.float32)
    for i_b in range(0, tl.cdiv(i_cols_num, BLOCK_SIZE)):
        i_row_block = tl.load(
            i_ptr + i_row_idx * i_row_stride + col_offsets * i_col_stride,
            mask=col_offsets < i_cols_num,
            other=0.0,
        )
        col_maxes = tl.maximum(tl.abs(i_row_block), col_maxes)
        col_offsets += BLOCK_SIZE

    # Compute and store scale for the current row
    i_row_max = tl.max(col_maxes)
    i_row_scale = _kernel_calculate_scale(i_row_max, TL_FP8_MAX)
    tl.store(o_scale_ptr, i_row_scale)

    # Scale and quantize current row block by block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for i_b in range(0, tl.cdiv(i_cols_num, BLOCK_SIZE)):
        i_row_block = tl.load(
            i_ptr + i_row_idx * i_row_stride + col_offsets * i_col_stride,
            mask=col_offsets < i_cols_num,
            other=0.0,
        )
        scaled_row_block = i_row_block * i_row_scale
        quantized_row_block = scaled_row_block.to(TL_FP8_TYPE)
        tl.store(
            o_quant_ptr + col_offsets,
            quantized_row_block,
            mask=col_offsets < i_cols_num,
        )
        col_offsets += BLOCK_SIZE


@triton.jit
def _fused_kernel_dequantize_from_fp8(
    i_ptrs,
    i_shapes,
    i_strides,
    i_offsets,
    i_dtype,
    o_ptr,
    o_size_bytes_per_rank,
    all_reduce_size,
    BLOCK_SIZE: tl.constexpr,
    TL_FP8_TYPE: tl.constexpr,
) -> None:
    """
    Kernel to dequantize a set of input tensors from fp8. The input tensors
    are expected to be of the same shape as those passed to the quantization.
    The result of the dequantization is stored in the input tensors.

    Args:
        i_ptrs: Pointers to the input tensors to be dequantized into
        i_shapes: Shapes of the input tensors to be dequantized into
        i_strides: Strides of the input tensors to be dequantized into
        i_offsets: Offsets of the output tensors for each input tensor
        i_dtype: Dummy tensor that carries the dtype of the input tensors
        o_ptr: Pointer to the tensor that contains output of the quantization
            or local reduction
        o_size_bytes_per_rank: Size in bytes in the output tensor per rank
        all_reduce_size: Size of the all-reduce group
        BLOCK_SIZE: Block size for the dequantization
    """
    # Index of the row in the input tensor
    i_row_idx = tl.program_id(0)
    # Index of the input tensor
    i_idx = tl.program_id(1)

    # Number of rows and colums in the input tensor
    i_rows_num = tl.load(i_shapes + i_idx * 2)
    if i_row_idx >= i_rows_num:
        return
    i_cols_num = tl.load(i_shapes + i_idx * 2 + 1)

    # Stride to advance by a single row and column in the input tensor
    # assume contiguous tensors
    i_row_stride = tl.load(i_strides + i_idx * 2)
    i_col_stride = tl.load(i_strides + i_idx * 2 + 1)

    # Pointer to the input tensor
    i_ptr = tl.load(i_ptrs + i_idx).to(i_dtype.dtype)

    # Number of the rows in the input tensor that are processed by a single
    # rank
    i_row_slice_size = tl.cdiv(i_rows_num, all_reduce_size)
    # Index of the row slice in the input tensor
    i_row_slice_idx = i_row_idx // i_row_slice_size

    # Size in bytes of a single input tensor row quantized and written to the
    # output tensor
    o_row_size_bytes = (
        tl.cdiv(i_cols_num, SCALE_TL_DTYPE_BYTES) + 1
    ) * SCALE_TL_DTYPE_BYTES

    # Pointer to the output tensor where
    o_offset = (
        o_size_bytes_per_rank * i_row_slice_idx
        + tl.load(i_offsets + i_idx)
        + (i_row_idx % i_row_slice_size) * o_row_size_bytes
    )
    # Pointer to the output tensor where the scale and quantized row will be
    # written
    o_curr_ptr = o_ptr + o_offset
    o_scale_ptr = o_curr_ptr.to(tl.pointer_type(SCALE_TL_DTYPE))
    o_quant_ptr = (o_curr_ptr + SCALE_TL_DTYPE_BYTES).to(tl.pointer_type(TL_FP8_TYPE))  # type: ignore

    # Load row scale
    i_row_scale = tl.load(o_scale_ptr)

    # Dequantize and store current row block by block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for i_b in range(0, tl.cdiv(i_cols_num, BLOCK_SIZE)):
        i_quant_row_block = tl.load(
            o_quant_ptr + col_offsets,
            mask=col_offsets < i_cols_num,
            other=0.0,
        )

        i_dequant_row_block = (
            i_quant_row_block.to(i_dtype.dtype.element_ty) / i_row_scale
        )
        tl.store(
            i_ptr + i_row_idx * i_row_stride + col_offsets * i_col_stride,
            i_dequant_row_block,
            mask=col_offsets < i_cols_num,
        )
        col_offsets += BLOCK_SIZE


@triton.jit
def _fused_kernel_reduce_fp8(
    i_shapes,
    i_offsets,
    o_ptr,
    o_size_bytes_per_rank,
    all_reduce_size,
    all_reduce_rank,
    division_factor,
    BLOCK_SIZE: tl.constexpr,
    TL_FP8_TYPE: tl.constexpr,
    TL_FP8_MAX: tl.constexpr,
) -> None:
    """
    Reduces rows of the output tensor for the given rank. The output tensor
    is expected to be holding quantized rows and scales for all ranks. The
    quantized rows are dequantized, averaged and quantized again. The result
    is stored in the output tensor for the given rank. After the reduction
    the row correspoding to the current rank can be shared with other ranks.

    Args:
        i_shapes: Shapes of the input tensors to be reduced, used to compute
            the number and length of rows
        i_offsets: Offsets of the output tensors for each input tensor
        o_ptr: Pointer to the tensor that contains output of the quantization
            of all ranks for a row the corresponding to the current rank
        o_size_bytes_per_rank: Size in bytes in the output tensor per rank
        all_reduce_size: Size of the all-reduce group
        all_reduce_rank: Rank in the all-reduce group
        division_factor: Division factor for the reduction result
        BLOCK_SIZE: Block size for the reduction
        NUM_SM: Number of SMs to use for the reduction
    """
    # Index of the row in the input tensor
    i_row_block_idx = tl.program_id(0)
    # Index of the input tensor
    i_idx = tl.program_id(1)

    # Number of rows and colums in the input tensor
    i_rows_num = tl.load(i_shapes + i_idx * 2)
    if i_row_block_idx >= tl.cdiv(i_rows_num, all_reduce_size):
        return
    i_cols_num = tl.load(i_shapes + i_idx * 2 + 1)

    # Size in bytes of a single input tensor row quantized and written to the
    # output tensor
    o_row_size_bytes = (
        tl.cdiv(i_cols_num, SCALE_TL_DTYPE_BYTES) + 1
    ) * SCALE_TL_DTYPE_BYTES

    # Pointer to the output tensor where
    o_offset = tl.load(i_offsets + i_idx) + i_row_block_idx * o_row_size_bytes

    o_row_block_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Compute scaling factor the reduced row
    o_row_max = 0.0
    for o_b in range(0, tl.cdiv(i_cols_num, BLOCK_SIZE)):
        o_row_block_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        col_offsets_mask = col_offsets < i_cols_num
        # Load blocks of quantized rows, dequantize and accumulate
        o_row_block_acc = _fused_kernel_accumulate_block(
            o_row_block_acc,
            o_ptr + o_offset,
            all_reduce_size,
            all_reduce_rank,
            o_size_bytes_per_rank,
            col_offsets,
            col_offsets_mask,
            TL_FP8_TYPE,
        )

        # Compute maximum across accumulated blocks
        o_row_block_max = tl.max(tl.abs(o_row_block_acc))
        o_row_max = tl.maximum(o_row_block_max, o_row_max)

        col_offsets += BLOCK_SIZE

    # Compute scaling factor for the reduced row
    o_row_scale = _kernel_calculate_scale(o_row_max / division_factor, TL_FP8_MAX)

    o_rank_row_ptr = o_ptr + all_reduce_rank * o_size_bytes_per_rank + o_offset
    o_rank_scale_ptr = o_rank_row_ptr.to(tl.pointer_type(SCALE_TL_DTYPE))
    o_rank_quant_ptr = (o_rank_row_ptr + SCALE_TL_DTYPE_BYTES).to(
        tl.pointer_type(TL_FP8_TYPE)  # type: ignore
    )

    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Reduce the row in blocks and write them out
    for o_b in range(0, tl.cdiv(i_cols_num, BLOCK_SIZE)):
        o_row_block_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        col_offsets_mask = col_offsets < i_cols_num
        # Load blocks of quantized rows, dequantize and accumulate
        o_row_block_acc = _fused_kernel_accumulate_block(
            o_row_block_acc,
            o_ptr + o_offset,
            all_reduce_size,
            all_reduce_rank,
            o_size_bytes_per_rank,
            col_offsets,
            col_offsets_mask,
            TL_FP8_TYPE,
        )
        o_row_block_acc = o_row_block_acc * o_row_scale / division_factor
        o_quant_row_block_acc = o_row_block_acc.to(TL_FP8_TYPE)
        tl.store(
            o_rank_quant_ptr + col_offsets,
            o_quant_row_block_acc,
            mask=col_offsets_mask,
        )

        col_offsets += BLOCK_SIZE

    # Write reduced row scale
    tl.store(o_rank_scale_ptr, o_row_scale)


@triton.jit
def _fused_kernel_accumulate_block(
    o_row_block_acc,
    o_ptr,
    o_row_num,
    o_row_start,
    o_row_stride,
    col_offsets,
    col_mask,
    TL_FP8_TYPE: tl.constexpr,
) -> tl.tensor:
    """
    Sums up blocks of quantized rows. The blocks are loaded from the output
    tensor, dequantized and accumulated into the row block accumulator.

    Args:
        o_row_block_acc: Row block accumulator
        o_ptr: Pointer to the output tensor
        o_row_num: Number of rows in the output tensor
        o_row_start: Start row index in the output tensor, used to ensure that
            accumulation happens in the correct order
        o_row_stride: Stride to advance by a single row in the output tensor
        col_offsets: Column offsets for the block of quantized rows
        col_mask: Column mask for the block of quantized rows, used to prevent
            going out of bounds
    """
    # Load blocks of quantized rows, dequantize and accumulate
    for o_row_idx in range(o_row_num):
        # Start with the row that corresponds to the current rank
        o_row_idx_wrapped = (o_row_idx + o_row_start) % o_row_num

        o_row_ptr = o_ptr + o_row_idx_wrapped * o_row_stride

        # Load row scale and block of quantized row
        o_scale_ptr = o_row_ptr.to(tl.pointer_type(tl.float32))
        o_quant_ptr = (o_row_ptr + SCALE_TL_DTYPE_BYTES).to(
            tl.pointer_type(TL_FP8_TYPE)  # type: ignore
        )

        o_row_scale = tl.load(o_scale_ptr)
        # Ensure that we do not divide by zero when reducing "padding" rows
        o_row_scale = tl.where(o_row_scale == 0.0, 1.0, o_row_scale)
        o_row_quant_block = tl.load(
            o_quant_ptr + col_offsets,
            mask=col_mask,
            other=0.0,
        )

        o_row_block_acc += o_row_quant_block.to(tl.float32) / o_row_scale

    return o_row_block_acc


def _prepare_quantize_fp8(
    inputs: list[torch.Tensor], all_reduce_group_size: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.device,
]:
    """
    Prepares the inputs for the quantization, dequantization and reduction kernels.

    Args:
        inputs: List of input tensors to be quantized, dequantized or reduced
        all_reduce_group_size: Size of the all-reduce group

    Returns:
        d_i_ptrs: Pointers to the input tensors
        d_i_shapes: Shapes of the input tensors
        d_i_strides: Row strides of the input tensors
        d_i_offsets: Offsets into the output tensor for each rank for each input
            tensor.
        d_i_dtype: The type of the input tensors
        output_size: Size of the output tensor in bytes including necessary padding
        i_max_row_num: Maximum number of rows in the input tensors
        device: Device of the input tensors
    """

    i_num = len(inputs)
    assert i_num > 0, "At least one input tensor is required"
    device = inputs[0].device
    dtype = inputs[0].dtype
    for i in range(1, i_num):
        assert (
            inputs[i].device == inputs[i - 1].device
        ), "All inputs must be on the same device"
        assert (
            inputs[i].dtype == inputs[i - 1].dtype
        ), "All inputs must be on the same dtype"

    assert dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], "Only fp32, fp16 and bf16 are supported"
    i_ptrs = []
    i_shapes = []
    i_strides = []
    i_offsets = []
    output_size = 0
    i_max_row_num = 0
    output_size_per_rank = 0
    for i in range(i_num):
        if len(inputs[i].shape) == 1:
            inputs[i] = inputs[i].unsqueeze(1)
        assert len(inputs[i].shape) == 2, "Only 2D tensors are supported"
        i_ptrs.append(inputs[i].data_ptr())
        i_m, i_n = inputs[i].shape
        i_shapes.append([i_m, i_n])
        i_m_stride, i_n_stride = inputs[i].stride()
        i_strides.append([i_m_stride, i_n_stride])
        i_m_padded = triton.cdiv(i_m, all_reduce_group_size) * all_reduce_group_size
        i_max_row_num = max(i_max_row_num, i_m_padded)

        i_n_padded = (
            i_m_padded * (triton.cdiv(i_n, SCALE_DTYPE_BYTES) + 1) * SCALE_DTYPE_BYTES
        )
        i_offsets.append(output_size_per_rank)
        output_size_per_rank += i_n_padded // all_reduce_group_size
        output_size += i_n_padded

    d_i_ptrs = torch.empty(i_num, dtype=torch.int64, device=device)
    d_i_ptrs.copy_(torch.tensor(i_ptrs), non_blocking=True)

    d_i_shapes = torch.empty(i_num, 2, dtype=torch.int32, device=device)
    d_i_shapes.copy_(torch.tensor(i_shapes, dtype=torch.int32), non_blocking=True)

    d_i_strides = torch.empty(i_num, 2, dtype=torch.int32, device=device)
    d_i_strides.copy_(torch.tensor(i_strides, dtype=torch.int32), non_blocking=True)

    d_i_offsets = torch.empty(i_num, dtype=torch.int32, device=device)
    d_i_offsets.copy_(torch.tensor(i_offsets, dtype=torch.int32), non_blocking=True)

    d_i_dtype = torch.empty(1, dtype=dtype, device=device)

    return (
        d_i_ptrs,
        d_i_shapes,
        d_i_strides,
        d_i_offsets,
        d_i_dtype,
        output_size,
        i_max_row_num,
        device,
    )


def fused_quantize_into_fp8(
    inputs: list[torch.Tensor], all_reduce_group_size: int
) -> torch.Tensor:
    """
    Quantizes a set of input tensors into fp8 where each row of each input
    tensor is quantized individually. The result is stored in the output tensor.
    Note that quantized rows and their scales are interleaved in the output
    tensor. Conceptually the output tensor consists one row per rank in the all
    reduce group. Each output row contains subset (input tensor rows are
    divided by the all group size and padded if needed) of quantized rows from
    the input tensors and their scales. The quantized rows are encoded as fp32
    scale followed by fp8 values followed by padding to ensure aligned memory
    access.

    Args:
        inputs: List of input tensors to be quantized
        all_reduce_group_size: Size of the all-reduce group

    Returns:
        output: Output tensor that contains quantized rows and scales for all
            ranks.
    """

    (
        d_i_ptrs,
        d_i_shapes,
        d_i_strides,
        d_i_offsets,
        d_i_dtype,
        output_size,
        i_max_row_num,
        device,
    ) = _prepare_quantize_fp8(inputs, all_reduce_group_size)

    # Allocate output tensor in scale dtype so that we can store scales by
    # doing pointer arithmetic and do not get misaligned memory access.
    output = torch.zeros(
        output_size // SCALE_DTYPE_BYTES,
        device=device,
        dtype=SCALE_DTYPE,
    ).view(torch.uint8)

    grid = (i_max_row_num, len(inputs))
    _fused_kernel_quantize_into_fp8[grid](
        d_i_ptrs,
        d_i_shapes,
        d_i_strides,
        d_i_offsets,
        d_i_dtype,
        output,
        output_size // all_reduce_group_size,
        all_reduce_group_size,
        BLOCK_SIZE=BLOCK_SIZE_T,  # type: ignore
        TL_FP8_TYPE=_get_fp8_type(),
        TL_FP8_MAX=_get_fp8_max(),
    )

    return output


def fused_dequantize_from_fp8(
    inputs: list[torch.Tensor], output: torch.Tensor, all_reduce_group_size: int
) -> None:
    """
    Dequantizes a set of input tensors from fp8 stored in the output tensor.
    The input tensors are expected to be of the same shape as those passed to
    the quantization. The result of the dequantization is stored in the input
    tensors. Note that quantized rows and their scales are interleaved in the
    output tensor. Conceptually the output tensor consists one row per rank in
    the all reduce group. Each output row contains subset (input tensor rows are
    divided by the all group size and padded if needed) of quantized rows from
    the input tensors and their scales.

    Args:
        inputs: List of input tensors to be dequantized into
        output: Output tensor that contains quantized rows and scales for all
            ranks.
        all_reduce_group_size: Size of the all-reduce group
    """
    (
        d_i_ptrs,
        d_i_shapes,
        d_i_strides,
        d_i_offsets,
        d_i_dtype,
        output_size,
        i_max_row_num,
        device,
    ) = _prepare_quantize_fp8(inputs, all_reduce_group_size)

    assert output.shape[0] == output_size, "Output size does not match"

    grid = (i_max_row_num, len(inputs))
    _fused_kernel_dequantize_from_fp8[grid](
        d_i_ptrs,
        d_i_shapes,
        d_i_strides,
        d_i_offsets,
        d_i_dtype,
        output,
        output_size // all_reduce_group_size,
        all_reduce_group_size,
        BLOCK_SIZE=BLOCK_SIZE_T,  # type: ignore
        TL_FP8_TYPE=_get_fp8_type(),
    )


def fused_reduce_fp8(
    inputs: list[torch.Tensor],
    output: torch.Tensor,
    all_reduce_group_size: int,
    all_reduce_rank: int,
    reduce_op: ReduceOp = ReduceOp.SUM,
) -> None:
    """
    Reduces rows of the output tensor for the given rank. The output tensor
    is expected to be holding quantized rows and scales for all ranks. The
    quantized rows are dequantized, averaged and quantized again. The result
    is stored in the output tensor for the given rank. After the reduction
    the row correspoding to the current rank can be shared with other
    ranks.

    Args:
        inputs: List of input tensors to be reduced
        output: Output tensor that contains quantized rows and scales for
            all ranks.
        all_reduce_group_size: Size of the all-reduce group
        all_reduce_rank: Rank in the all-reduce group
    """

    (
        d_i_ptrs,
        d_i_shapes,
        d_i_strides,
        d_i_offsets,
        d_i_dtype,
        output_size,
        i_max_row_num,
        device,
    ) = _prepare_quantize_fp8(inputs, all_reduce_group_size)

    assert output.shape[0] == output_size, "Output size does not match"

    grid = (i_max_row_num // all_reduce_group_size, len(inputs))
    _fused_kernel_reduce_fp8[grid](
        d_i_shapes,
        d_i_offsets,
        output,
        output_size // all_reduce_group_size,
        all_reduce_group_size,
        all_reduce_rank,
        1.0 if reduce_op == ReduceOp.SUM else float(all_reduce_group_size),
        BLOCK_SIZE=BLOCK_SIZE_T,  # type: ignore
        TL_FP8_TYPE=_get_fp8_type(),
        TL_FP8_MAX=_get_fp8_max(),
    )
