//! CUDA activation kernel wrappers.
#![allow(clippy::too_many_arguments)]

use crate::cuda::bindings;
use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::memory::DevicePtr;

fn to_i32_len(op: &'static str, value: usize) -> CudaResult<i32> {
    i32::try_from(value).map_err(|_| CudaError::SizeOverflow {
        op,
        count: value,
        elem_size: 1,
    })
}

fn validate_softmax_dims(op: &'static str, rows: usize, cols: usize) -> CudaResult<()> {
    if rows > 0 && cols == 0 {
        return Err(CudaError::InvalidInput {
            op,
            message: "cols must be greater than zero when rows is non-zero",
        });
    }
    Ok(())
}

fn validate_positive_dim(op: &'static str, name: &'static str, value: usize) -> CudaResult<()> {
    if value == 0 {
        return Err(CudaError::InvalidInput { op, message: name });
    }
    Ok(())
}

fn validate_len(op: &'static str, actual: usize, expected: usize) -> CudaResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(CudaError::SizeMismatch {
            op,
            expected,
            actual,
        })
    }
}

fn checked_mul2(op: &'static str, a: usize, b: usize) -> CudaResult<usize> {
    a.checked_mul(b).ok_or(CudaError::SizeOverflow {
        op,
        count: a,
        elem_size: b,
    })
}

fn checked_mul3(op: &'static str, a: usize, b: usize, c: usize) -> CudaResult<usize> {
    a.checked_mul(b)
        .and_then(|v| v.checked_mul(c))
        .ok_or(CudaError::SizeOverflow {
            op,
            count: a,
            elem_size: b.saturating_mul(c),
        })
}

#[cfg(cuda)]
pub fn relu_inplace(data: &DevicePtr<f64>) -> CudaResult<()> {
    crate::cuda::init()?;
    let size = to_i32_len("cuda::kernels::relu_inplace(size)", data.len())?;
    if size == 0 {
        return Ok(());
    }

    let status = unsafe {
        bindings::cuda_relu(
            std::ptr::null_mut(),
            size,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::relu_inplace",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn gelu_inplace(data: &DevicePtr<f64>) -> CudaResult<()> {
    crate::cuda::init()?;
    let size = to_i32_len("cuda::kernels::gelu_inplace(size)", data.len())?;
    if size == 0 {
        return Ok(());
    }

    let status = unsafe {
        bindings::cuda_gelu(
            std::ptr::null_mut(),
            size,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::gelu_inplace",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn softmax_inplace(data: &DevicePtr<f64>, rows: usize, cols: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_inplace(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_inplace",
            expected,
            actual: data.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_inplace", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::softmax_inplace(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_inplace(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax(
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_inplace",
            code: status as u32,
        })
    }
}

/// Threshold for using the small-batch cooperative kernel.
/// When rows <= this value, the 2D block kernel (256x4 threads per block)
/// provides better GPU utilization than one block per row.
const SOFTMAX_SMALL_BATCH_THRESHOLD: usize = 16;

#[cfg(cuda)]
pub fn softmax_inplace_auto(data: &DevicePtr<f64>, rows: usize, cols: usize) -> CudaResult<()> {
    if rows <= SOFTMAX_SMALL_BATCH_THRESHOLD && rows > 1 {
        softmax_small_batch_inplace(data, rows, cols)
    } else {
        softmax_inplace(data, rows, cols)
    }
}

#[cfg(cuda)]
pub fn softmax_small_batch_inplace(
    data: &DevicePtr<f64>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_small_batch_inplace(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_small_batch_inplace",
            expected,
            actual: data.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_small_batch_inplace", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::softmax_small_batch_inplace(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_small_batch_inplace(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax_small_batch(
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_small_batch_inplace",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn softmax_causal_inplace(data: &DevicePtr<f64>, rows: usize, cols: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_causal_inplace(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_causal_inplace",
            expected,
            actual: data.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_causal_inplace", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::softmax_causal_inplace(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_causal_inplace(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax_causal(
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_causal_inplace",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn log_softmax(
    logits: &DevicePtr<f64>,
    out: &DevicePtr<f64>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::log_softmax(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != logits.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax(logits)",
            expected,
            actual: logits.len(),
        });
    }
    if expected != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax(out)",
            expected,
            actual: out.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::log_softmax", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::log_softmax(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::log_softmax(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_log_softmax(
            std::ptr::null(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            logits.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::log_softmax",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn softmax_backward(
    out: &DevicePtr<f64>,
    grad_out: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_backward(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_backward(out)",
            expected,
            actual: out.len(),
        });
    }
    if expected != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_backward(grad_out)",
            expected,
            actual: grad_out.len(),
        });
    }
    if expected != input_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_backward(input_grad)",
            expected,
            actual: input_grad.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_backward", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::softmax_backward(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_backward(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            out.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn log_softmax_backward(
    out: &DevicePtr<f64>,
    grad_out: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::log_softmax_backward(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_backward(out)",
            expected,
            actual: out.len(),
        });
    }
    if expected != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_backward(grad_out)",
            expected,
            actual: grad_out.len(),
        });
    }
    if expected != input_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_backward(input_grad)",
            expected,
            actual: input_grad.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::log_softmax_backward", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::log_softmax_backward(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::log_softmax_backward(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_log_softmax_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            out.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::log_softmax_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rope_inplace(
    data: &DevicePtr<f64>,
    cos_cache: &DevicePtr<f64>,
    sin_cache: &DevicePtr<f64>,
    seq_len: usize,
    dim: usize,
    total_batches: usize,
    start_pos: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let half_dim = dim / 2;
    let expected_data_len = total_batches
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::rope_inplace(data_len)",
            count: total_batches,
            elem_size: seq_len * dim,
        })?;
    if expected_data_len != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_inplace(data)",
            expected: expected_data_len,
            actual: data.len(),
        });
    }
    let expected_cache_len =
        (seq_len + start_pos)
            .checked_mul(half_dim)
            .ok_or(CudaError::SizeOverflow {
                op: "cuda::kernels::rope_inplace(cache_len)",
                count: seq_len + start_pos,
                elem_size: half_dim,
            })?;
    if expected_cache_len > cos_cache.len() || expected_cache_len > sin_cache.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_inplace(cache)",
            expected: expected_cache_len,
            actual: cos_cache.len().min(sin_cache.len()),
        });
    }
    if !dim.is_multiple_of(2) {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::rope_inplace",
            message: "dim must be even for RoPE",
        });
    }

    let seq_len_i32 = to_i32_len("cuda::kernels::rope_inplace(seq_len)", seq_len)?;
    let dim_i32 = to_i32_len("cuda::kernels::rope_inplace(dim)", dim)?;
    let total_batches_i32 =
        to_i32_len("cuda::kernels::rope_inplace(total_batches)", total_batches)?;
    let start_pos_i32 = to_i32_len("cuda::kernels::rope_inplace(start_pos)", start_pos)?;

    let status = unsafe {
        bindings::cuda_rope(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            seq_len_i32,
            dim_i32,
            total_batches_i32,
            start_pos_i32,
            data.as_raw() as *mut std::os::raw::c_int,
            cos_cache.as_raw() as *mut std::os::raw::c_int,
            sin_cache.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rope_inplace",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rope_backward(
    grad_out: &DevicePtr<f64>,
    cos_cache: &DevicePtr<f64>,
    sin_cache: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    seq_len: usize,
    dim: usize,
    total_batches: usize,
    start_pos: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let half_dim = dim / 2;
    let expected_data_len = total_batches
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::rope_backward(data_len)",
            count: total_batches,
            elem_size: seq_len * dim,
        })?;
    if expected_data_len != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_backward(grad_out)",
            expected: expected_data_len,
            actual: grad_out.len(),
        });
    }
    if expected_data_len != input_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_backward(input_grad)",
            expected: expected_data_len,
            actual: input_grad.len(),
        });
    }
    let expected_cache_len =
        (seq_len + start_pos)
            .checked_mul(half_dim)
            .ok_or(CudaError::SizeOverflow {
                op: "cuda::kernels::rope_backward(cache_len)",
                count: seq_len + start_pos,
                elem_size: half_dim,
            })?;
    if expected_cache_len > cos_cache.len() || expected_cache_len > sin_cache.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_backward(cache)",
            expected: expected_cache_len,
            actual: cos_cache.len().min(sin_cache.len()),
        });
    }
    if !dim.is_multiple_of(2) {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::rope_backward",
            message: "dim must be even for RoPE",
        });
    }

    let seq_len_i32 = to_i32_len("cuda::kernels::rope_backward(seq_len)", seq_len)?;
    let dim_i32 = to_i32_len("cuda::kernels::rope_backward(dim)", dim)?;
    let total_batches_i32 =
        to_i32_len("cuda::kernels::rope_backward(total_batches)", total_batches)?;
    let start_pos_i32 = to_i32_len("cuda::kernels::rope_backward(start_pos)", start_pos)?;

    let status = unsafe {
        bindings::cuda_rope_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            seq_len_i32,
            dim_i32,
            total_batches_i32,
            start_pos_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            cos_cache.as_raw() as *mut std::os::raw::c_int,
            sin_cache.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rope_backward",
            code: status as u32,
        })
    }
}

/// Attention weighted sum: out[row, d] = sum_k(attn[row, k] * values[k, d])
/// where attn_weights is [rows x cols] and values is [cols x head_dim]
/// output is [rows x head_dim]
#[cfg(cuda)]
pub fn attention_weighted_sum(
    attn_weights: &DevicePtr<f64>,
    values: &DevicePtr<f64>,
    output: &DevicePtr<f64>,
    rows: usize,
    cols: usize,
    head_dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;

    // Validate dimensions
    let expected_attn_len = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::attention_weighted_sum(attn_weights_len)",
        count: rows,
        elem_size: cols,
    })?;
    if expected_attn_len != attn_weights.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::attention_weighted_sum(attn_weights)",
            expected: expected_attn_len,
            actual: attn_weights.len(),
        });
    }

    let expected_values_len = cols.checked_mul(head_dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::attention_weighted_sum(values_len)",
        count: cols,
        elem_size: head_dim,
    })?;
    if expected_values_len != values.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::attention_weighted_sum(values)",
            expected: expected_values_len,
            actual: values.len(),
        });
    }

    let expected_output_len = rows.checked_mul(head_dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::attention_weighted_sum(output_len)",
        count: rows,
        elem_size: head_dim,
    })?;
    if expected_output_len != output.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::attention_weighted_sum(output)",
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    if cols == 0 || head_dim == 0 {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::attention_weighted_sum",
            message: "cols and head_dim must be greater than zero",
        });
    }

    let rows_i32 = to_i32_len("cuda::kernels::attention_weighted_sum(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::attention_weighted_sum(cols)", cols)?;
    let head_dim_i32 = to_i32_len("cuda::kernels::attention_weighted_sum(head_dim)", head_dim)?;

    let status = unsafe {
        bindings::cuda_attention_weighted_sum(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            head_dim_i32,
            attn_weights.as_raw() as *mut std::os::raw::c_int,
            values.as_raw() as *mut std::os::raw::c_int,
            output.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::attention_weighted_sum",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn relu_backward(
    input: &DevicePtr<f64>,
    grad_out: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::relu_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_relu_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::relu_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn gelu_backward(
    input: &DevicePtr<f64>,
    grad_out: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::gelu_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_gelu_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::gelu_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn add_backward(
    grad_out: &DevicePtr<f64>,
    a_grad: &DevicePtr<f64>,
    b_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::add_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_add_backward(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::add_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sub_backward(
    grad_out: &DevicePtr<f64>,
    a_grad: &DevicePtr<f64>,
    b_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::sub_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_sub_backward(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sub_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn mul_backward(
    grad_out: &DevicePtr<f64>,
    a_data: &DevicePtr<f64>,
    b_data: &DevicePtr<f64>,
    a_grad: &DevicePtr<f64>,
    b_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::mul_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_mul_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_data.as_raw() as *mut std::os::raw::c_int,
            b_data.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::mul_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn div_backward(
    grad_out: &DevicePtr<f64>,
    a_data: &DevicePtr<f64>,
    b_data: &DevicePtr<f64>,
    a_grad: &DevicePtr<f64>,
    b_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::div_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_div_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_data.as_raw() as *mut std::os::raw::c_int,
            b_data.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::div_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn acc_buffer(dst: &DevicePtr<f64>, src: &DevicePtr<f64>, size: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::acc_buffer(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_acc_buffer(
            std::ptr::null_mut(),
            std::ptr::null(),
            size_i32,
            dst.as_raw() as *mut std::os::raw::c_int,
            src.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::acc_buffer",
            code: status as u32,
        })
    }
}

macro_rules! define_elementwise_forward {
    ($name:ident, $binding:ident, $op_name:expr) => {
        #[cfg(cuda)]
        pub fn $name(
            a: &DevicePtr<f64>,
            b: &DevicePtr<f64>,
            out: &DevicePtr<f64>,
            size: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let size_i32 = to_i32_len(concat!("cuda::kernels::", $op_name, "(size)"), size)?;
            if size == 0 {
                return Ok(());
            }
            let status = unsafe {
                crate::cuda::bindings::$binding(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    size_i32,
                    a.as_raw() as *mut std::os::raw::c_int,
                    b.as_raw() as *mut std::os::raw::c_int,
                    out.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: concat!("cuda::kernels::", $op_name),
                    code: status as u32,
                })
            }
        }
    };
}

define_elementwise_forward!(add_forward, cuda_add_forward, "add_forward");
define_elementwise_forward!(sub_forward, cuda_sub_forward, "sub_forward");
define_elementwise_forward!(mul_forward, cuda_mul_forward, "mul_forward");
define_elementwise_forward!(div_forward, cuda_div_forward, "div_forward");

#[cfg(cuda)]
pub fn fill(data: &DevicePtr<f64>, value: f64) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::fill(size)", data.len())?;
    if data.len() == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_fill(
            std::ptr::null_mut(),
            value,
            size_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::fill",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sumsq_accum(input: &DevicePtr<f64>, accum: &DevicePtr<f64>, size: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::sumsq_accum(input)", input.len(), size)?;
    validate_len("cuda::kernels::sumsq_accum(accum)", accum.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::sumsq_accum(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_sumsq_accum(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            accum.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sumsq_accum",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sum_accum(
    input: &DevicePtr<f64>,
    accum: &DevicePtr<f64>,
    size: usize,
    scale: f64,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::sum_accum(input)", input.len(), size)?;
    validate_len("cuda::kernels::sum_accum(accum)", accum.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::sum_accum(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_sum_accum(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            scale,
            input.as_raw() as *mut std::os::raw::c_int,
            accum.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sum_accum",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn clip_coef_from_sumsq(
    sumsq: &DevicePtr<f64>,
    coef: &DevicePtr<f64>,
    max_norm: f64,
    eps: f64,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::clip_coef_from_sumsq(sumsq)", sumsq.len(), 1)?;
    validate_len("cuda::kernels::clip_coef_from_sumsq(coef)", coef.len(), 1)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_clip_coef_from_sumsq(
            std::ptr::null(),
            std::ptr::null_mut(),
            max_norm,
            eps,
            sumsq.as_raw() as *mut std::os::raw::c_int,
            coef.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::clip_coef_from_sumsq",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn scale_inplace_by_scalar(
    data: &DevicePtr<f64>,
    scalar: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::scale_inplace_by_scalar(data)",
        data.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::scale_inplace_by_scalar(scalar)",
        scalar.len(),
        1,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::scale_inplace_by_scalar(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_scale_inplace_by_scalar(
            std::ptr::null_mut(),
            std::ptr::null(),
            size_i32,
            data.as_raw() as *mut std::os::raw::c_int,
            scalar.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::scale_inplace_by_scalar",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn add_scalar(
    data: &DevicePtr<f64>,
    scalar: &DevicePtr<f64>,
    scale: f64,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::add_scalar(data)", data.len(), size)?;
    validate_len("cuda::kernels::add_scalar(scalar)", scalar.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::add_scalar(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_add_scalar(
            std::ptr::null_mut(),
            std::ptr::null(),
            scale,
            size_i32,
            data.as_raw() as *mut std::os::raw::c_int,
            scalar.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::add_scalar",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn lerp_inplace(
    target: &DevicePtr<f64>,
    source: &DevicePtr<f64>,
    tau: f64,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::lerp_inplace(target)", target.len(), size)?;
    validate_len("cuda::kernels::lerp_inplace(source)", source.len(), size)?;
    let size_i32 = to_i32_len("cuda::kernels::lerp_inplace(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_lerp_inplace(
            std::ptr::null_mut(),
            std::ptr::null(),
            tau,
            size_i32,
            target.as_raw() as *mut std::os::raw::c_int,
            source.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::lerp_inplace",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn double_dqn_target(
    q_next_eval: &DevicePtr<f64>,
    q_next_target: &DevicePtr<f64>,
    rewards: &DevicePtr<f64>,
    dones: &DevicePtr<f64>,
    out: &DevicePtr<f64>,
    batch: usize,
    actions: usize,
    gamma: f64,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::double_dqn_target(actions)",
        "actions must be greater than zero",
        actions,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target(eval)",
        q_next_eval.len(),
        checked_mul2("cuda::kernels::double_dqn_target(eval)", batch, actions)?,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target(target)",
        q_next_target.len(),
        checked_mul2("cuda::kernels::double_dqn_target(target)", batch, actions)?,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target(rewards)",
        rewards.len(),
        batch,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target(dones)",
        dones.len(),
        batch,
    )?;
    validate_len("cuda::kernels::double_dqn_target(out)", out.len(), batch)?;
    let batch_i32 = to_i32_len("cuda::kernels::double_dqn_target(batch)", batch)?;
    let actions_i32 = to_i32_len("cuda::kernels::double_dqn_target(actions)", actions)?;
    if batch == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_double_dqn_target(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            batch_i32,
            actions_i32,
            gamma,
            q_next_eval.as_raw() as *mut std::os::raw::c_int,
            q_next_target.as_raw() as *mut std::os::raw::c_int,
            rewards.as_raw() as *mut std::os::raw::c_int,
            dones.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::double_dqn_target",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn abs_diff(
    a: &DevicePtr<f64>,
    b: &DevicePtr<f64>,
    out: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::abs_diff(a)", a.len(), size)?;
    validate_len("cuda::kernels::abs_diff(b)", b.len(), size)?;
    validate_len("cuda::kernels::abs_diff(out)", out.len(), size)?;
    let size_i32 = to_i32_len("cuda::kernels::abs_diff(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_abs_diff(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            a.as_raw() as *mut std::os::raw::c_int,
            b.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::abs_diff",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn select_last_token(
    input: &DevicePtr<f64>,
    out: &DevicePtr<f64>,
    batch: usize,
    seq: usize,
    dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::select_last_token(seq)",
        "seq must be greater than zero",
        seq,
    )?;
    validate_positive_dim(
        "cuda::kernels::select_last_token(dim)",
        "dim must be greater than zero",
        dim,
    )?;
    let input_len = batch
        .checked_mul(seq)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::select_last_token(input_len)",
            count: batch,
            elem_size: seq.saturating_mul(dim),
        })?;
    let out_len = batch.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::select_last_token(out_len)",
        count: batch,
        elem_size: dim,
    })?;
    validate_len(
        "cuda::kernels::select_last_token(input)",
        input.len(),
        input_len,
    )?;
    validate_len("cuda::kernels::select_last_token(out)", out.len(), out_len)?;
    if batch == 0 {
        return Ok(());
    }
    let batch_i32 = to_i32_len("cuda::kernels::select_last_token(batch)", batch)?;
    let seq_i32 = to_i32_len("cuda::kernels::select_last_token(seq)", seq)?;
    let dim_i32 = to_i32_len("cuda::kernels::select_last_token(dim)", dim)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_select_last_token(
            std::ptr::null(),
            std::ptr::null_mut(),
            batch_i32,
            seq_i32,
            dim_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::select_last_token",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn select_last_token_backward(
    grad_out: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    batch: usize,
    seq: usize,
    dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::select_last_token_backward(seq)",
        "seq must be greater than zero",
        seq,
    )?;
    validate_positive_dim(
        "cuda::kernels::select_last_token_backward(dim)",
        "dim must be greater than zero",
        dim,
    )?;
    let grad_len = batch.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::select_last_token_backward(grad_len)",
        count: batch,
        elem_size: dim,
    })?;
    let input_len = batch
        .checked_mul(seq)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::select_last_token_backward(input_len)",
            count: batch,
            elem_size: seq.saturating_mul(dim),
        })?;
    validate_len(
        "cuda::kernels::select_last_token_backward(grad_out)",
        grad_out.len(),
        grad_len,
    )?;
    validate_len(
        "cuda::kernels::select_last_token_backward(input_grad)",
        input_grad.len(),
        input_len,
    )?;
    if batch == 0 {
        return Ok(());
    }
    let batch_i32 = to_i32_len("cuda::kernels::select_last_token_backward(batch)", batch)?;
    let seq_i32 = to_i32_len("cuda::kernels::select_last_token_backward(seq)", seq)?;
    let dim_i32 = to_i32_len("cuda::kernels::select_last_token_backward(dim)", dim)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_select_last_token_backward(
            std::ptr::null(),
            std::ptr::null_mut(),
            batch_i32,
            seq_i32,
            dim_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::select_last_token_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn index_select(input: &DevicePtr<f64>, out: &DevicePtr<f64>, idx: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    if idx >= input.len() {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::index_select",
            message: "idx out of bounds",
        });
    }
    validate_len("cuda::kernels::index_select(out)", out.len(), 1)?;
    let idx_i32 = to_i32_len("cuda::kernels::index_select(idx)", idx)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_index_select(
            std::ptr::null(),
            std::ptr::null_mut(),
            idx_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::index_select",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn index_select_backward(
    grad_out: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    idx: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::index_select_backward(grad_out)",
        grad_out.len(),
        1,
    )?;
    if idx >= input_grad.len() {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::index_select_backward",
            message: "idx out of bounds",
        });
    }
    let idx_i32 = to_i32_len("cuda::kernels::index_select_backward(idx)", idx)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_index_select_backward(
            std::ptr::null(),
            std::ptr::null_mut(),
            idx_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::index_select_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn argmax(input: &DevicePtr<f64>, out_idx: &DevicePtr<i32>, size: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::argmax(size)",
        "size must be greater than zero",
        size,
    )?;
    validate_len("cuda::kernels::argmax(input)", input.len(), size)?;
    validate_len("cuda::kernels::argmax(out_idx)", out_idx.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::argmax(size)", size)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_argmax(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out_idx.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::argmax",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn exp(input: &DevicePtr<f64>, out: &DevicePtr<f64>, size: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::exp(input)", input.len(), size)?;
    validate_len("cuda::kernels::exp(out)", out.len(), size)?;
    let size_i32 = to_i32_len("cuda::kernels::exp(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_exp(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::exp",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn exp_backward(
    exp_out: &DevicePtr<f64>,
    grad_out: &DevicePtr<f64>,
    input_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::exp_backward(exp_out)", exp_out.len(), size)?;
    validate_len(
        "cuda::kernels::exp_backward(grad_out)",
        grad_out.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::exp_backward(input_grad)",
        input_grad.len(),
        size,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::exp_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_exp_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            exp_out.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::exp_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn weighted_mse_loss(
    pred: &DevicePtr<f64>,
    target: &DevicePtr<f64>,
    weights: &DevicePtr<f64>,
    out: &DevicePtr<f64>,
    weight_sum: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::weighted_mse_loss(pred)", pred.len(), size)?;
    validate_len(
        "cuda::kernels::weighted_mse_loss(target)",
        target.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_loss(weights)",
        weights.len(),
        size,
    )?;
    validate_len("cuda::kernels::weighted_mse_loss(out)", out.len(), 1)?;
    validate_len(
        "cuda::kernels::weighted_mse_loss(weight_sum)",
        weight_sum.len(),
        1,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::weighted_mse_loss(size)", size)?;
    if size == 0 {
        fill(out, 0.0)?;
        fill(weight_sum, 0.0)?;
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_weighted_mse_loss(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            pred.as_raw() as *mut std::os::raw::c_int,
            target.as_raw() as *mut std::os::raw::c_int,
            weights.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
            weight_sum.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::weighted_mse_loss",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn weighted_mse_backward(
    pred: &DevicePtr<f64>,
    target: &DevicePtr<f64>,
    weights: &DevicePtr<f64>,
    weight_sum: &DevicePtr<f64>,
    grad_out: &DevicePtr<f64>,
    pred_grad: &DevicePtr<f64>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::weighted_mse_backward(pred)",
        pred.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward(target)",
        target.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward(weights)",
        weights.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward(weight_sum)",
        weight_sum.len(),
        1,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward(grad_out)",
        grad_out.len(),
        1,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward(pred_grad)",
        pred_grad.len(),
        size,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::weighted_mse_backward(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_weighted_mse_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            pred.as_raw() as *mut std::os::raw::c_int,
            target.as_raw() as *mut std::os::raw::c_int,
            weights.as_raw() as *mut std::os::raw::c_int,
            weight_sum.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            pred_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::weighted_mse_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn adam_step(
    params: &DevicePtr<f64>,
    grads: &DevicePtr<f64>,
    m: &DevicePtr<f64>,
    v: &DevicePtr<f64>,
    size: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    bias_correction1: f64,
    bias_correction2: f64,
    clip_coef: f64,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::adam_step(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_adam_step(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            clip_coef,
            params.as_raw() as *mut std::os::raw::c_int,
            grads.as_raw() as *mut std::os::raw::c_int,
            m.as_raw() as *mut std::os::raw::c_int,
            v.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::adam_step",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rmsnorm_forward(
    x: &DevicePtr<f64>,
    weight: &DevicePtr<f64>,
    out: &DevicePtr<f64>,
    dim: usize,
    eps: f64,
    num_rows: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let dim_i32 = to_i32_len("cuda::kernels::rmsnorm_forward(dim)", dim)?;
    let num_rows_i32 = to_i32_len("cuda::kernels::rmsnorm_forward(num_rows)", num_rows)?;
    let expected_len = num_rows.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::rmsnorm_forward(len)",
        count: num_rows,
        elem_size: dim,
    })?;
    if expected_len != x.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_forward(x)",
            expected: expected_len,
            actual: x.len(),
        });
    }
    if expected_len != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_forward(out)",
            expected: expected_len,
            actual: out.len(),
        });
    }
    if weight.len() != dim {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_forward(weight)",
            expected: dim,
            actual: weight.len(),
        });
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_rmsnorm_forward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            dim_i32,
            eps,
            num_rows_i32,
            x.as_raw() as *mut std::os::raw::c_int,
            weight.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rmsnorm_forward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rmsnorm_backward(
    grad_out: &DevicePtr<f64>,
    x: &DevicePtr<f64>,
    weight: &DevicePtr<f64>,
    x_grad: &DevicePtr<f64>,
    w_grad: &DevicePtr<f64>,
    dim: usize,
    eps: f64,
    num_rows: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let dim_i32 = to_i32_len("cuda::kernels::rmsnorm_backward(dim)", dim)?;
    let num_rows_i32 = to_i32_len("cuda::kernels::rmsnorm_backward(num_rows)", num_rows)?;
    let expected_len = num_rows.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::rmsnorm_backward(len)",
        count: num_rows,
        elem_size: dim,
    })?;
    if expected_len != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward(grad_out)",
            expected: expected_len,
            actual: grad_out.len(),
        });
    }
    if expected_len != x.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward(x)",
            expected: expected_len,
            actual: x.len(),
        });
    }
    if expected_len != x_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward(x_grad)",
            expected: expected_len,
            actual: x_grad.len(),
        });
    }
    if weight.len() != dim {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward(weight)",
            expected: dim,
            actual: weight.len(),
        });
    }
    if w_grad.len() != dim {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward(w_grad)",
            expected: dim,
            actual: w_grad.len(),
        });
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_rmsnorm_backward(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            dim_i32,
            eps,
            num_rows_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            x.as_raw() as *mut std::os::raw::c_int,
            weight.as_raw() as *mut std::os::raw::c_int,
            x_grad.as_raw() as *mut std::os::raw::c_int,
            w_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rmsnorm_backward",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sparse_matvec(
    x: &DevicePtr<f64>,
    w: &DevicePtr<f64>,
    mask: &DevicePtr<u8>,
    y: &DevicePtr<f64>,
    num_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let num_rows_i32 = to_i32_len("cuda::kernels::sparse_matvec(num_rows)", num_rows)?;
    let in_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec(in_dim)", in_dim)?;
    let out_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec(out_dim)", out_dim)?;
    if num_rows == 0 || in_dim == 0 || out_dim == 0 {
        return Ok(());
    }
    let status = unsafe {
        bindings::cuda_sparse_matvec(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            num_rows_i32,
            in_dim_i32,
            out_dim_i32,
            x.as_raw() as *mut std::os::raw::c_int,
            w.as_raw() as *mut std::os::raw::c_int,
            mask.as_raw() as *mut std::os::raw::c_int,
            y.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sparse_matvec",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sparse_matvec_bias(
    x: &DevicePtr<f64>,
    w: &DevicePtr<f64>,
    mask: &DevicePtr<u8>,
    bias: &DevicePtr<f64>,
    y: &DevicePtr<f64>,
    num_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let num_rows_i32 = to_i32_len("cuda::kernels::sparse_matvec_bias(num_rows)", num_rows)?;
    let in_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec_bias(in_dim)", in_dim)?;
    let out_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec_bias(out_dim)", out_dim)?;
    if num_rows == 0 || in_dim == 0 || out_dim == 0 {
        return Ok(());
    }
    let status = unsafe {
        bindings::cuda_sparse_matvec_bias(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            num_rows_i32,
            in_dim_i32,
            out_dim_i32,
            x.as_raw() as *mut std::os::raw::c_int,
            w.as_raw() as *mut std::os::raw::c_int,
            mask.as_raw() as *mut std::os::raw::c_int,
            bias.as_raw() as *mut std::os::raw::c_int,
            y.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sparse_matvec_bias",
            code: status as u32,
        })
    }
}

// =============================================================================
// F32 kernel wrappers
// =============================================================================

#[cfg(cuda)]
pub fn relu_inplace_f32(data: &DevicePtr<f32>) -> CudaResult<()> {
    crate::cuda::init()?;
    let size = to_i32_len("cuda::kernels::relu_inplace_f32(size)", data.len())?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        bindings::cuda_relu_f32(
            std::ptr::null_mut(),
            size,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::relu_inplace_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn gelu_inplace_f32(data: &DevicePtr<f32>) -> CudaResult<()> {
    crate::cuda::init()?;
    let size = to_i32_len("cuda::kernels::gelu_inplace_f32(size)", data.len())?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        bindings::cuda_gelu_f32(
            std::ptr::null_mut(),
            size,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::gelu_inplace_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn softmax_inplace_f32(data: &DevicePtr<f32>, rows: usize, cols: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_inplace_f32(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_inplace_f32",
            expected,
            actual: data.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_inplace_f32", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }
    let rows_i32 = to_i32_len("cuda::kernels::softmax_inplace_f32(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_inplace_f32(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax_f32(
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_inplace_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn softmax_inplace_auto_f32(data: &DevicePtr<f32>, rows: usize, cols: usize) -> CudaResult<()> {
    if rows <= SOFTMAX_SMALL_BATCH_THRESHOLD && rows > 1 {
        softmax_small_batch_inplace_f32(data, rows, cols)
    } else {
        softmax_inplace_f32(data, rows, cols)
    }
}

#[cfg(cuda)]
pub fn softmax_small_batch_inplace_f32(
    data: &DevicePtr<f32>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_small_batch_inplace_f32(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_small_batch_inplace_f32",
            expected,
            actual: data.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_small_batch_inplace_f32", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }
    let rows_i32 = to_i32_len("cuda::kernels::softmax_small_batch_inplace_f32(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_small_batch_inplace_f32(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax_small_batch_f32(
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_small_batch_inplace_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn softmax_causal_inplace_f32(
    data: &DevicePtr<f32>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_causal_inplace_f32(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_causal_inplace_f32",
            expected,
            actual: data.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_causal_inplace_f32", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }
    let rows_i32 = to_i32_len("cuda::kernels::softmax_causal_inplace_f32(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_causal_inplace_f32(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax_causal_f32(
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_causal_inplace_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn log_softmax_f32(
    logits: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::log_softmax_f32(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != logits.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_f32(logits)",
            expected,
            actual: logits.len(),
        });
    }
    if expected != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_f32(out)",
            expected,
            actual: out.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::log_softmax_f32", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }
    let rows_i32 = to_i32_len("cuda::kernels::log_softmax_f32(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::log_softmax_f32(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_log_softmax_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            logits.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::log_softmax_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn softmax_backward_f32(
    out: &DevicePtr<f32>,
    grad_out: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::softmax_backward_f32(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_backward_f32(out)",
            expected,
            actual: out.len(),
        });
    }
    if expected != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_backward_f32(grad_out)",
            expected,
            actual: grad_out.len(),
        });
    }
    if expected != input_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::softmax_backward_f32(input_grad)",
            expected,
            actual: input_grad.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::softmax_backward_f32", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::softmax_backward_f32(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::softmax_backward_f32(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_softmax_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            out.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::softmax_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn log_softmax_backward_f32(
    out: &DevicePtr<f32>,
    grad_out: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    rows: usize,
    cols: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let expected = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::log_softmax_backward_f32(rows*cols)",
        count: rows,
        elem_size: cols,
    })?;
    if expected != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_backward_f32(out)",
            expected,
            actual: out.len(),
        });
    }
    if expected != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_backward_f32(grad_out)",
            expected,
            actual: grad_out.len(),
        });
    }
    if expected != input_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::log_softmax_backward_f32(input_grad)",
            expected,
            actual: input_grad.len(),
        });
    }
    validate_softmax_dims("cuda::kernels::log_softmax_backward_f32", rows, cols)?;
    if expected == 0 {
        return Ok(());
    }

    let rows_i32 = to_i32_len("cuda::kernels::log_softmax_backward_f32(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::log_softmax_backward_f32(cols)", cols)?;
    let status = unsafe {
        bindings::cuda_log_softmax_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            out.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::log_softmax_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rope_inplace_f32(
    data: &DevicePtr<f32>,
    cos_cache: &DevicePtr<f32>,
    sin_cache: &DevicePtr<f32>,
    seq_len: usize,
    dim: usize,
    total_batches: usize,
    start_pos: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let half_dim = dim / 2;
    let expected_data_len = total_batches
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::rope_inplace_f32(data_len)",
            count: total_batches,
            elem_size: seq_len * dim,
        })?;
    if expected_data_len != data.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_inplace_f32(data)",
            expected: expected_data_len,
            actual: data.len(),
        });
    }
    let expected_cache_len =
        (seq_len + start_pos)
            .checked_mul(half_dim)
            .ok_or(CudaError::SizeOverflow {
                op: "cuda::kernels::rope_inplace_f32(cache_len)",
                count: seq_len + start_pos,
                elem_size: half_dim,
            })?;
    if expected_cache_len > cos_cache.len() || expected_cache_len > sin_cache.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_inplace_f32(cache)",
            expected: expected_cache_len,
            actual: cos_cache.len().min(sin_cache.len()),
        });
    }
    if !dim.is_multiple_of(2) {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::rope_inplace_f32",
            message: "dim must be even for RoPE",
        });
    }

    let seq_len_i32 = to_i32_len("cuda::kernels::rope_inplace_f32(seq_len)", seq_len)?;
    let dim_i32 = to_i32_len("cuda::kernels::rope_inplace_f32(dim)", dim)?;
    let total_batches_i32 = to_i32_len(
        "cuda::kernels::rope_inplace_f32(total_batches)",
        total_batches,
    )?;
    let start_pos_i32 = to_i32_len("cuda::kernels::rope_inplace_f32(start_pos)", start_pos)?;

    let status = unsafe {
        bindings::cuda_rope_f32(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            seq_len_i32,
            dim_i32,
            total_batches_i32,
            start_pos_i32,
            data.as_raw() as *mut std::os::raw::c_int,
            cos_cache.as_raw() as *mut std::os::raw::c_int,
            sin_cache.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rope_inplace_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rope_backward_f32(
    grad_out: &DevicePtr<f32>,
    cos_cache: &DevicePtr<f32>,
    sin_cache: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    seq_len: usize,
    dim: usize,
    total_batches: usize,
    start_pos: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let half_dim = dim / 2;
    let expected_data_len = total_batches
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::rope_backward_f32(data_len)",
            count: total_batches,
            elem_size: seq_len * dim,
        })?;
    if expected_data_len != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_backward_f32(grad_out)",
            expected: expected_data_len,
            actual: grad_out.len(),
        });
    }
    if expected_data_len != input_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_backward_f32(input_grad)",
            expected: expected_data_len,
            actual: input_grad.len(),
        });
    }
    let expected_cache_len =
        (seq_len + start_pos)
            .checked_mul(half_dim)
            .ok_or(CudaError::SizeOverflow {
                op: "cuda::kernels::rope_backward_f32(cache_len)",
                count: seq_len + start_pos,
                elem_size: half_dim,
            })?;
    if expected_cache_len > cos_cache.len() || expected_cache_len > sin_cache.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rope_backward_f32(cache)",
            expected: expected_cache_len,
            actual: cos_cache.len().min(sin_cache.len()),
        });
    }
    if !dim.is_multiple_of(2) {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::rope_backward_f32",
            message: "dim must be even for RoPE",
        });
    }

    let seq_len_i32 = to_i32_len("cuda::kernels::rope_backward_f32(seq_len)", seq_len)?;
    let dim_i32 = to_i32_len("cuda::kernels::rope_backward_f32(dim)", dim)?;
    let total_batches_i32 = to_i32_len(
        "cuda::kernels::rope_backward_f32(total_batches)",
        total_batches,
    )?;
    let start_pos_i32 = to_i32_len("cuda::kernels::rope_backward_f32(start_pos)", start_pos)?;

    let status = unsafe {
        bindings::cuda_rope_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            seq_len_i32,
            dim_i32,
            total_batches_i32,
            start_pos_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            cos_cache.as_raw() as *mut std::os::raw::c_int,
            sin_cache.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rope_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn attention_weighted_sum_f32(
    attn_weights: &DevicePtr<f32>,
    values: &DevicePtr<f32>,
    output: &DevicePtr<f32>,
    rows: usize,
    cols: usize,
    head_dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;

    let expected_attn_len = rows.checked_mul(cols).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::attention_weighted_sum_f32(attn_weights_len)",
        count: rows,
        elem_size: cols,
    })?;
    if expected_attn_len != attn_weights.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::attention_weighted_sum_f32(attn_weights)",
            expected: expected_attn_len,
            actual: attn_weights.len(),
        });
    }

    let expected_values_len = cols.checked_mul(head_dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::attention_weighted_sum_f32(values_len)",
        count: cols,
        elem_size: head_dim,
    })?;
    if expected_values_len != values.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::attention_weighted_sum_f32(values)",
            expected: expected_values_len,
            actual: values.len(),
        });
    }

    let expected_output_len = rows.checked_mul(head_dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::attention_weighted_sum_f32(output_len)",
        count: rows,
        elem_size: head_dim,
    })?;
    if expected_output_len != output.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::attention_weighted_sum_f32(output)",
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    if cols == 0 || head_dim == 0 {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::attention_weighted_sum_f32",
            message: "cols and head_dim must be greater than zero",
        });
    }

    let rows_i32 = to_i32_len("cuda::kernels::attention_weighted_sum_f32(rows)", rows)?;
    let cols_i32 = to_i32_len("cuda::kernels::attention_weighted_sum_f32(cols)", cols)?;
    let head_dim_i32 = to_i32_len(
        "cuda::kernels::attention_weighted_sum_f32(head_dim)",
        head_dim,
    )?;

    let status = unsafe {
        bindings::cuda_attention_weighted_sum_f32(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            rows_i32,
            cols_i32,
            head_dim_i32,
            attn_weights.as_raw() as *mut std::os::raw::c_int,
            values.as_raw() as *mut std::os::raw::c_int,
            output.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::attention_weighted_sum_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn relu_backward_f32(
    input: &DevicePtr<f32>,
    grad_out: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::relu_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_relu_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::relu_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn gelu_backward_f32(
    input: &DevicePtr<f32>,
    grad_out: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::gelu_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_gelu_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::gelu_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn add_backward_f32(
    grad_out: &DevicePtr<f32>,
    a_grad: &DevicePtr<f32>,
    b_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::add_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_add_backward_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::add_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sub_backward_f32(
    grad_out: &DevicePtr<f32>,
    a_grad: &DevicePtr<f32>,
    b_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::sub_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_sub_backward_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sub_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn mul_backward_f32(
    grad_out: &DevicePtr<f32>,
    a_data: &DevicePtr<f32>,
    b_data: &DevicePtr<f32>,
    a_grad: &DevicePtr<f32>,
    b_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::mul_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_mul_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_data.as_raw() as *mut std::os::raw::c_int,
            b_data.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::mul_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn div_backward_f32(
    grad_out: &DevicePtr<f32>,
    a_data: &DevicePtr<f32>,
    b_data: &DevicePtr<f32>,
    a_grad: &DevicePtr<f32>,
    b_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::div_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_div_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            a_data.as_raw() as *mut std::os::raw::c_int,
            b_data.as_raw() as *mut std::os::raw::c_int,
            a_grad.as_raw() as *mut std::os::raw::c_int,
            b_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::div_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn acc_buffer_f32(dst: &DevicePtr<f32>, src: &DevicePtr<f32>, size: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::acc_buffer_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_acc_buffer_f32(
            std::ptr::null_mut(),
            std::ptr::null(),
            size_i32,
            dst.as_raw() as *mut std::os::raw::c_int,
            src.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::acc_buffer_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn add_forward_f32(
    a: &DevicePtr<f32>,
    b: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::add_forward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_add_forward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            a.as_raw() as *mut std::os::raw::c_int,
            b.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::add_forward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sub_forward_f32(
    a: &DevicePtr<f32>,
    b: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::sub_forward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_sub_forward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            a.as_raw() as *mut std::os::raw::c_int,
            b.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sub_forward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn mul_forward_f32(
    a: &DevicePtr<f32>,
    b: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::mul_forward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_mul_forward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            a.as_raw() as *mut std::os::raw::c_int,
            b.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::mul_forward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn div_forward_f32(
    a: &DevicePtr<f32>,
    b: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::div_forward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_div_forward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            a.as_raw() as *mut std::os::raw::c_int,
            b.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::div_forward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn adam_step_f32(
    params: &DevicePtr<f32>,
    grads: &DevicePtr<f32>,
    m: &DevicePtr<f32>,
    v: &DevicePtr<f32>,
    size: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    clip_coef: f32,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::adam_step_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_adam_step_f32(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            clip_coef,
            params.as_raw() as *mut std::os::raw::c_int,
            grads.as_raw() as *mut std::os::raw::c_int,
            m.as_raw() as *mut std::os::raw::c_int,
            v.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::adam_step_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn fill_f32(data: &DevicePtr<f32>, value: f32) -> CudaResult<()> {
    crate::cuda::init()?;
    let size_i32 = to_i32_len("cuda::kernels::fill_f32(size)", data.len())?;
    if data.len() == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_fill_f32(
            std::ptr::null_mut(),
            value,
            size_i32,
            data.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::fill_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sumsq_accum_f32(
    input: &DevicePtr<f32>,
    accum: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::sumsq_accum_f32(input)", input.len(), size)?;
    validate_len("cuda::kernels::sumsq_accum_f32(accum)", accum.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::sumsq_accum_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_sumsq_accum_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            accum.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sumsq_accum_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sum_accum_f32(
    input: &DevicePtr<f32>,
    accum: &DevicePtr<f32>,
    size: usize,
    scale: f32,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::sum_accum_f32(input)", input.len(), size)?;
    validate_len("cuda::kernels::sum_accum_f32(accum)", accum.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::sum_accum_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_sum_accum_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            scale,
            input.as_raw() as *mut std::os::raw::c_int,
            accum.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sum_accum_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn clip_coef_from_sumsq_f32(
    sumsq: &DevicePtr<f32>,
    coef: &DevicePtr<f32>,
    max_norm: f32,
    eps: f32,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::clip_coef_from_sumsq_f32(sumsq)",
        sumsq.len(),
        1,
    )?;
    validate_len(
        "cuda::kernels::clip_coef_from_sumsq_f32(coef)",
        coef.len(),
        1,
    )?;
    let status = unsafe {
        crate::cuda::bindings::cuda_clip_coef_from_sumsq_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            max_norm,
            eps,
            sumsq.as_raw() as *mut std::os::raw::c_int,
            coef.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::clip_coef_from_sumsq_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn scale_inplace_by_scalar_f32(
    data: &DevicePtr<f32>,
    scalar: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::scale_inplace_by_scalar_f32(data)",
        data.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::scale_inplace_by_scalar_f32(scalar)",
        scalar.len(),
        1,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::scale_inplace_by_scalar_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_scale_inplace_by_scalar_f32(
            std::ptr::null_mut(),
            std::ptr::null(),
            size_i32,
            data.as_raw() as *mut std::os::raw::c_int,
            scalar.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::scale_inplace_by_scalar_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn add_scalar_f32(
    data: &DevicePtr<f32>,
    scalar: &DevicePtr<f32>,
    scale: f32,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::add_scalar_f32(data)", data.len(), size)?;
    validate_len("cuda::kernels::add_scalar_f32(scalar)", scalar.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::add_scalar_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_add_scalar_f32(
            std::ptr::null_mut(),
            std::ptr::null(),
            scale,
            size_i32,
            data.as_raw() as *mut std::os::raw::c_int,
            scalar.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::add_scalar_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn lerp_inplace_f32(
    target: &DevicePtr<f32>,
    source: &DevicePtr<f32>,
    tau: f32,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::lerp_inplace_f32(target)",
        target.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::lerp_inplace_f32(source)",
        source.len(),
        size,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::lerp_inplace_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_lerp_inplace_f32(
            std::ptr::null_mut(),
            std::ptr::null(),
            tau,
            size_i32,
            target.as_raw() as *mut std::os::raw::c_int,
            source.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::lerp_inplace_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn double_dqn_target_f32(
    q_next_eval: &DevicePtr<f32>,
    q_next_target: &DevicePtr<f32>,
    rewards: &DevicePtr<f32>,
    dones: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    batch: usize,
    actions: usize,
    gamma: f32,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::double_dqn_target_f32(actions)",
        "actions must be greater than zero",
        actions,
    )?;
    let q_len = checked_mul2("cuda::kernels::double_dqn_target_f32(q)", batch, actions)?;
    validate_len(
        "cuda::kernels::double_dqn_target_f32(eval)",
        q_next_eval.len(),
        q_len,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target_f32(target)",
        q_next_target.len(),
        q_len,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target_f32(rewards)",
        rewards.len(),
        batch,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target_f32(dones)",
        dones.len(),
        batch,
    )?;
    validate_len(
        "cuda::kernels::double_dqn_target_f32(out)",
        out.len(),
        batch,
    )?;
    let batch_i32 = to_i32_len("cuda::kernels::double_dqn_target_f32(batch)", batch)?;
    let actions_i32 = to_i32_len("cuda::kernels::double_dqn_target_f32(actions)", actions)?;
    if batch == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_double_dqn_target_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            batch_i32,
            actions_i32,
            gamma,
            q_next_eval.as_raw() as *mut std::os::raw::c_int,
            q_next_target.as_raw() as *mut std::os::raw::c_int,
            rewards.as_raw() as *mut std::os::raw::c_int,
            dones.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::double_dqn_target_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn abs_diff_f32(
    a: &DevicePtr<f32>,
    b: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::abs_diff_f32(a)", a.len(), size)?;
    validate_len("cuda::kernels::abs_diff_f32(b)", b.len(), size)?;
    validate_len("cuda::kernels::abs_diff_f32(out)", out.len(), size)?;
    let size_i32 = to_i32_len("cuda::kernels::abs_diff_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_abs_diff_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            a.as_raw() as *mut std::os::raw::c_int,
            b.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::abs_diff_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn select_last_token_f32(
    input: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    batch: usize,
    seq: usize,
    dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::select_last_token_f32(seq)",
        "seq must be greater than zero",
        seq,
    )?;
    validate_positive_dim(
        "cuda::kernels::select_last_token_f32(dim)",
        "dim must be greater than zero",
        dim,
    )?;
    let input_len = batch
        .checked_mul(seq)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::select_last_token_f32(input_len)",
            count: batch,
            elem_size: seq.saturating_mul(dim),
        })?;
    let out_len = batch.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::select_last_token_f32(out_len)",
        count: batch,
        elem_size: dim,
    })?;
    validate_len(
        "cuda::kernels::select_last_token_f32(input)",
        input.len(),
        input_len,
    )?;
    validate_len(
        "cuda::kernels::select_last_token_f32(out)",
        out.len(),
        out_len,
    )?;
    if batch == 0 {
        return Ok(());
    }
    let batch_i32 = to_i32_len("cuda::kernels::select_last_token_f32(batch)", batch)?;
    let seq_i32 = to_i32_len("cuda::kernels::select_last_token_f32(seq)", seq)?;
    let dim_i32 = to_i32_len("cuda::kernels::select_last_token_f32(dim)", dim)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_select_last_token_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            batch_i32,
            seq_i32,
            dim_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::select_last_token_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn select_last_token_backward_f32(
    grad_out: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    batch: usize,
    seq: usize,
    dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::select_last_token_backward_f32(seq)",
        "seq must be greater than zero",
        seq,
    )?;
    validate_positive_dim(
        "cuda::kernels::select_last_token_backward_f32(dim)",
        "dim must be greater than zero",
        dim,
    )?;
    let grad_len = batch.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::select_last_token_backward_f32(grad_len)",
        count: batch,
        elem_size: dim,
    })?;
    let input_len = batch
        .checked_mul(seq)
        .and_then(|v| v.checked_mul(dim))
        .ok_or(CudaError::SizeOverflow {
            op: "cuda::kernels::select_last_token_backward_f32(input_len)",
            count: batch,
            elem_size: seq.saturating_mul(dim),
        })?;
    validate_len(
        "cuda::kernels::select_last_token_backward_f32(grad_out)",
        grad_out.len(),
        grad_len,
    )?;
    validate_len(
        "cuda::kernels::select_last_token_backward_f32(input_grad)",
        input_grad.len(),
        input_len,
    )?;
    if batch == 0 {
        return Ok(());
    }
    let batch_i32 = to_i32_len(
        "cuda::kernels::select_last_token_backward_f32(batch)",
        batch,
    )?;
    let seq_i32 = to_i32_len("cuda::kernels::select_last_token_backward_f32(seq)", seq)?;
    let dim_i32 = to_i32_len("cuda::kernels::select_last_token_backward_f32(dim)", dim)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_select_last_token_backward_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            batch_i32,
            seq_i32,
            dim_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::select_last_token_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn index_select_f32(
    input: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    idx: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    if idx >= input.len() {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::index_select_f32",
            message: "idx out of bounds",
        });
    }
    validate_len("cuda::kernels::index_select_f32(out)", out.len(), 1)?;
    let idx_i32 = to_i32_len("cuda::kernels::index_select_f32(idx)", idx)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_index_select_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            idx_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::index_select_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn index_select_backward_f32(
    grad_out: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    idx: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::index_select_backward_f32(grad_out)",
        grad_out.len(),
        1,
    )?;
    if idx >= input_grad.len() {
        return Err(CudaError::InvalidInput {
            op: "cuda::kernels::index_select_backward_f32",
            message: "idx out of bounds",
        });
    }
    let idx_i32 = to_i32_len("cuda::kernels::index_select_backward_f32(idx)", idx)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_index_select_backward_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            idx_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::index_select_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn argmax_f32(input: &DevicePtr<f32>, out_idx: &DevicePtr<i32>, size: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_positive_dim(
        "cuda::kernels::argmax_f32(size)",
        "size must be greater than zero",
        size,
    )?;
    validate_len("cuda::kernels::argmax_f32(input)", input.len(), size)?;
    validate_len("cuda::kernels::argmax_f32(out_idx)", out_idx.len(), 1)?;
    let size_i32 = to_i32_len("cuda::kernels::argmax_f32(size)", size)?;
    let status = unsafe {
        crate::cuda::bindings::cuda_argmax_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out_idx.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::argmax_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn exp_f32(input: &DevicePtr<f32>, out: &DevicePtr<f32>, size: usize) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len("cuda::kernels::exp_f32(input)", input.len(), size)?;
    validate_len("cuda::kernels::exp_f32(out)", out.len(), size)?;
    let size_i32 = to_i32_len("cuda::kernels::exp_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_exp_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            input.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::exp_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn exp_backward_f32(
    exp_out: &DevicePtr<f32>,
    grad_out: &DevicePtr<f32>,
    input_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::exp_backward_f32(exp_out)",
        exp_out.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::exp_backward_f32(grad_out)",
        grad_out.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::exp_backward_f32(input_grad)",
        input_grad.len(),
        size,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::exp_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_exp_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            exp_out.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            input_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::exp_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn weighted_mse_loss_f32(
    pred: &DevicePtr<f32>,
    target: &DevicePtr<f32>,
    weights: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    weight_sum: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::weighted_mse_loss_f32(pred)",
        pred.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_loss_f32(target)",
        target.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_loss_f32(weights)",
        weights.len(),
        size,
    )?;
    validate_len("cuda::kernels::weighted_mse_loss_f32(out)", out.len(), 1)?;
    validate_len(
        "cuda::kernels::weighted_mse_loss_f32(weight_sum)",
        weight_sum.len(),
        1,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::weighted_mse_loss_f32(size)", size)?;
    if size == 0 {
        fill_f32(out, 0.0)?;
        fill_f32(weight_sum, 0.0)?;
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_weighted_mse_loss_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size_i32,
            pred.as_raw() as *mut std::os::raw::c_int,
            target.as_raw() as *mut std::os::raw::c_int,
            weights.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
            weight_sum.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::weighted_mse_loss_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn weighted_mse_backward_f32(
    pred: &DevicePtr<f32>,
    target: &DevicePtr<f32>,
    weights: &DevicePtr<f32>,
    weight_sum: &DevicePtr<f32>,
    grad_out: &DevicePtr<f32>,
    pred_grad: &DevicePtr<f32>,
    size: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    validate_len(
        "cuda::kernels::weighted_mse_backward_f32(pred)",
        pred.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward_f32(target)",
        target.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward_f32(weights)",
        weights.len(),
        size,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward_f32(weight_sum)",
        weight_sum.len(),
        1,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward_f32(grad_out)",
        grad_out.len(),
        1,
    )?;
    validate_len(
        "cuda::kernels::weighted_mse_backward_f32(pred_grad)",
        pred_grad.len(),
        size,
    )?;
    let size_i32 = to_i32_len("cuda::kernels::weighted_mse_backward_f32(size)", size)?;
    if size == 0 {
        return Ok(());
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_weighted_mse_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            size_i32,
            pred.as_raw() as *mut std::os::raw::c_int,
            target.as_raw() as *mut std::os::raw::c_int,
            weights.as_raw() as *mut std::os::raw::c_int,
            weight_sum.as_raw() as *mut std::os::raw::c_int,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            pred_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::weighted_mse_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rmsnorm_forward_f32(
    x: &DevicePtr<f32>,
    weight: &DevicePtr<f32>,
    out: &DevicePtr<f32>,
    dim: usize,
    eps: f32,
    num_rows: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let dim_i32 = to_i32_len("cuda::kernels::rmsnorm_forward_f32(dim)", dim)?;
    let num_rows_i32 = to_i32_len("cuda::kernels::rmsnorm_forward_f32(num_rows)", num_rows)?;
    let expected_len = num_rows.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::rmsnorm_forward_f32(len)",
        count: num_rows,
        elem_size: dim,
    })?;
    if expected_len != x.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_forward_f32(x)",
            expected: expected_len,
            actual: x.len(),
        });
    }
    if expected_len != out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_forward_f32(out)",
            expected: expected_len,
            actual: out.len(),
        });
    }
    if weight.len() != dim {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_forward_f32(weight)",
            expected: dim,
            actual: weight.len(),
        });
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_rmsnorm_forward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            dim_i32,
            eps,
            num_rows_i32,
            x.as_raw() as *mut std::os::raw::c_int,
            weight.as_raw() as *mut std::os::raw::c_int,
            out.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rmsnorm_forward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn rmsnorm_backward_f32(
    grad_out: &DevicePtr<f32>,
    x: &DevicePtr<f32>,
    weight: &DevicePtr<f32>,
    x_grad: &DevicePtr<f32>,
    w_grad: &DevicePtr<f32>,
    dim: usize,
    eps: f32,
    num_rows: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let dim_i32 = to_i32_len("cuda::kernels::rmsnorm_backward_f32(dim)", dim)?;
    let num_rows_i32 = to_i32_len("cuda::kernels::rmsnorm_backward_f32(num_rows)", num_rows)?;
    let expected_len = num_rows.checked_mul(dim).ok_or(CudaError::SizeOverflow {
        op: "cuda::kernels::rmsnorm_backward_f32(len)",
        count: num_rows,
        elem_size: dim,
    })?;
    if expected_len != grad_out.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward_f32(grad_out)",
            expected: expected_len,
            actual: grad_out.len(),
        });
    }
    if expected_len != x.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward_f32(x)",
            expected: expected_len,
            actual: x.len(),
        });
    }
    if expected_len != x_grad.len() {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward_f32(x_grad)",
            expected: expected_len,
            actual: x_grad.len(),
        });
    }
    if weight.len() != dim {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward_f32(weight)",
            expected: dim,
            actual: weight.len(),
        });
    }
    if w_grad.len() != dim {
        return Err(CudaError::SizeMismatch {
            op: "cuda::kernels::rmsnorm_backward_f32(w_grad)",
            expected: dim,
            actual: w_grad.len(),
        });
    }
    let status = unsafe {
        crate::cuda::bindings::cuda_rmsnorm_backward_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            dim_i32,
            eps,
            num_rows_i32,
            grad_out.as_raw() as *mut std::os::raw::c_int,
            x.as_raw() as *mut std::os::raw::c_int,
            weight.as_raw() as *mut std::os::raw::c_int,
            x_grad.as_raw() as *mut std::os::raw::c_int,
            w_grad.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::rmsnorm_backward_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sparse_matvec_f32(
    x: &DevicePtr<f32>,
    w: &DevicePtr<f32>,
    mask: &DevicePtr<u8>,
    y: &DevicePtr<f32>,
    num_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let num_rows_i32 = to_i32_len("cuda::kernels::sparse_matvec_f32(num_rows)", num_rows)?;
    let in_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec_f32(in_dim)", in_dim)?;
    let out_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec_f32(out_dim)", out_dim)?;
    if num_rows == 0 || in_dim == 0 || out_dim == 0 {
        return Ok(());
    }
    let status = unsafe {
        bindings::cuda_sparse_matvec_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            num_rows_i32,
            in_dim_i32,
            out_dim_i32,
            x.as_raw() as *mut std::os::raw::c_int,
            w.as_raw() as *mut std::os::raw::c_int,
            mask.as_raw() as *mut std::os::raw::c_int,
            y.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sparse_matvec_f32",
            code: status as u32,
        })
    }
}

#[cfg(cuda)]
pub fn sparse_matvec_bias_f32(
    x: &DevicePtr<f32>,
    w: &DevicePtr<f32>,
    mask: &DevicePtr<u8>,
    bias: &DevicePtr<f32>,
    y: &DevicePtr<f32>,
    num_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> CudaResult<()> {
    crate::cuda::init()?;
    let num_rows_i32 = to_i32_len("cuda::kernels::sparse_matvec_bias_f32(num_rows)", num_rows)?;
    let in_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec_bias_f32(in_dim)", in_dim)?;
    let out_dim_i32 = to_i32_len("cuda::kernels::sparse_matvec_bias_f32(out_dim)", out_dim)?;
    if num_rows == 0 || in_dim == 0 || out_dim == 0 {
        return Ok(());
    }
    let status = unsafe {
        bindings::cuda_sparse_matvec_bias_f32(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            num_rows_i32,
            in_dim_i32,
            out_dim_i32,
            x.as_raw() as *mut std::os::raw::c_int,
            w.as_raw() as *mut std::os::raw::c_int,
            mask.as_raw() as *mut std::os::raw::c_int,
            bias.as_raw() as *mut std::os::raw::c_int,
            y.as_raw() as *mut std::os::raw::c_int,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            op: "cuda::kernels::sparse_matvec_bias_f32",
            code: status as u32,
        })
    }
}

macro_rules! define_scale_wrappers {
    (
        $scale_fn:ident,
        $scale_binding:ident,
        $scale_backward_fn:ident,
        $scale_backward_binding:ident,
        $ty:ty,
        $op_scale:expr,
        $op_backward:expr
    ) => {
        #[cfg(cuda)]
        pub fn $scale_fn(
            input: &DevicePtr<$ty>,
            output: &DevicePtr<$ty>,
            scale: $ty,
            size: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            validate_len($op_scale, input.len(), size)?;
            validate_len($op_scale, output.len(), size)?;
            if size == 0 {
                return Ok(());
            }
            let size_i32 = to_i32_len(concat!($op_scale, "(size)"), size)?;
            let status = unsafe {
                bindings::$scale_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    scale,
                    size_i32,
                    input.as_raw() as *mut std::os::raw::c_int,
                    output.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_scale,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $scale_backward_fn(
            grad_out: &DevicePtr<$ty>,
            input_grad: &DevicePtr<$ty>,
            scale: $ty,
            size: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            validate_len($op_backward, grad_out.len(), size)?;
            validate_len($op_backward, input_grad.len(), size)?;
            if size == 0 {
                return Ok(());
            }
            let size_i32 = to_i32_len(concat!($op_backward, "(size)"), size)?;
            let status = unsafe {
                bindings::$scale_backward_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    scale,
                    size_i32,
                    grad_out.as_raw() as *mut std::os::raw::c_int,
                    input_grad.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_backward,
                    code: status as u32,
                })
            }
        }
    };
}

define_scale_wrappers!(
    scale,
    cuda_scale,
    scale_backward,
    cuda_scale_backward,
    f64,
    "cuda::kernels::scale",
    "cuda::kernels::scale_backward"
);
define_scale_wrappers!(
    scale_f32,
    cuda_scale_f32,
    scale_backward_f32,
    cuda_scale_backward_f32,
    f32,
    "cuda::kernels::scale_f32",
    "cuda::kernels::scale_backward_f32"
);

macro_rules! define_causal_mask_wrappers {
    (
        $mask_fn:ident,
        $mask_binding:ident,
        $mask_backward_fn:ident,
        $mask_backward_binding:ident,
        $ty:ty,
        $op_mask:expr,
        $op_backward:expr
    ) => {
        #[cfg(cuda)]
        pub fn $mask_fn(
            input: &DevicePtr<$ty>,
            output: &DevicePtr<$ty>,
            batches: usize,
            seq: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let expected = batches
                .checked_mul(seq)
                .and_then(|v| v.checked_mul(seq))
                .ok_or(CudaError::SizeOverflow {
                    op: $op_mask,
                    count: batches,
                    elem_size: seq.saturating_mul(seq),
                })?;
            validate_len($op_mask, input.len(), expected)?;
            validate_len($op_mask, output.len(), expected)?;
            if expected == 0 {
                return Ok(());
            }
            let batches_i32 = to_i32_len(concat!($op_mask, "(batches)"), batches)?;
            let seq_i32 = to_i32_len(concat!($op_mask, "(seq)"), seq)?;
            let status = unsafe {
                bindings::$mask_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    batches_i32,
                    seq_i32,
                    input.as_raw() as *mut std::os::raw::c_int,
                    output.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_mask,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $mask_backward_fn(
            grad_out: &DevicePtr<$ty>,
            input_grad: &DevicePtr<$ty>,
            batches: usize,
            seq: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let expected = batches
                .checked_mul(seq)
                .and_then(|v| v.checked_mul(seq))
                .ok_or(CudaError::SizeOverflow {
                    op: $op_backward,
                    count: batches,
                    elem_size: seq.saturating_mul(seq),
                })?;
            validate_len($op_backward, grad_out.len(), expected)?;
            validate_len($op_backward, input_grad.len(), expected)?;
            if expected == 0 {
                return Ok(());
            }
            let batches_i32 = to_i32_len(concat!($op_backward, "(batches)"), batches)?;
            let seq_i32 = to_i32_len(concat!($op_backward, "(seq)"), seq)?;
            let status = unsafe {
                bindings::$mask_backward_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    batches_i32,
                    seq_i32,
                    grad_out.as_raw() as *mut std::os::raw::c_int,
                    input_grad.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_backward,
                    code: status as u32,
                })
            }
        }
    };
}

define_causal_mask_wrappers!(
    causal_mask,
    cuda_causal_mask,
    causal_mask_backward,
    cuda_causal_mask_backward,
    f64,
    "cuda::kernels::causal_mask",
    "cuda::kernels::causal_mask_backward"
);
define_causal_mask_wrappers!(
    causal_mask_f32,
    cuda_causal_mask_f32,
    causal_mask_backward_f32,
    cuda_causal_mask_backward_f32,
    f32,
    "cuda::kernels::causal_mask_f32",
    "cuda::kernels::causal_mask_backward_f32"
);

macro_rules! define_concat_wrappers {
    (
        $concat_fn:ident,
        $concat_binding:ident,
        $concat_backward_fn:ident,
        $concat_backward_binding:ident,
        $ty:ty,
        $op_concat:expr,
        $op_backward:expr
    ) => {
        #[cfg(cuda)]
        pub fn $concat_fn(
            a: &DevicePtr<$ty>,
            b: &DevicePtr<$ty>,
            out: &DevicePtr<$ty>,
            rows: usize,
            a_dim: usize,
            b_dim: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let a_len = checked_mul2($op_concat, rows, a_dim)?;
            let b_len = checked_mul2($op_concat, rows, b_dim)?;
            let out_dim = a_dim.checked_add(b_dim).ok_or(CudaError::SizeOverflow {
                op: $op_concat,
                count: a_dim,
                elem_size: b_dim,
            })?;
            let out_len = checked_mul2($op_concat, rows, out_dim)?;
            validate_len($op_concat, a.len(), a_len)?;
            validate_len($op_concat, b.len(), b_len)?;
            validate_len($op_concat, out.len(), out_len)?;
            if out.len() == 0 {
                return Ok(());
            }
            let rows_i32 = to_i32_len(concat!($op_concat, "(rows)"), rows)?;
            let a_dim_i32 = to_i32_len(concat!($op_concat, "(a_dim)"), a_dim)?;
            let b_dim_i32 = to_i32_len(concat!($op_concat, "(b_dim)"), b_dim)?;
            let status = unsafe {
                bindings::$concat_binding(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    rows_i32,
                    a_dim_i32,
                    b_dim_i32,
                    a.as_raw() as *mut std::os::raw::c_int,
                    b.as_raw() as *mut std::os::raw::c_int,
                    out.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_concat,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $concat_backward_fn(
            grad_out: &DevicePtr<$ty>,
            a_grad: &DevicePtr<$ty>,
            b_grad: &DevicePtr<$ty>,
            rows: usize,
            a_dim: usize,
            b_dim: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let a_len = checked_mul2($op_backward, rows, a_dim)?;
            let b_len = checked_mul2($op_backward, rows, b_dim)?;
            let out_dim = a_dim.checked_add(b_dim).ok_or(CudaError::SizeOverflow {
                op: $op_backward,
                count: a_dim,
                elem_size: b_dim,
            })?;
            let out_len = checked_mul2($op_backward, rows, out_dim)?;
            validate_len($op_backward, grad_out.len(), out_len)?;
            validate_len($op_backward, a_grad.len(), a_len)?;
            validate_len($op_backward, b_grad.len(), b_len)?;
            if grad_out.len() == 0 {
                return Ok(());
            }
            let rows_i32 = to_i32_len(concat!($op_backward, "(rows)"), rows)?;
            let a_dim_i32 = to_i32_len(concat!($op_backward, "(a_dim)"), a_dim)?;
            let b_dim_i32 = to_i32_len(concat!($op_backward, "(b_dim)"), b_dim)?;
            let status = unsafe {
                bindings::$concat_backward_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    rows_i32,
                    a_dim_i32,
                    b_dim_i32,
                    grad_out.as_raw() as *mut std::os::raw::c_int,
                    a_grad.as_raw() as *mut std::os::raw::c_int,
                    b_grad.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_backward,
                    code: status as u32,
                })
            }
        }
    };
}

define_concat_wrappers!(
    concat_last_dim,
    cuda_concat_last_dim,
    concat_last_dim_backward,
    cuda_concat_last_dim_backward,
    f64,
    "cuda::kernels::concat_last_dim",
    "cuda::kernels::concat_last_dim_backward"
);
define_concat_wrappers!(
    concat_last_dim_f32,
    cuda_concat_last_dim_f32,
    concat_last_dim_backward_f32,
    cuda_concat_last_dim_backward_f32,
    f32,
    "cuda::kernels::concat_last_dim_f32",
    "cuda::kernels::concat_last_dim_backward_f32"
);

macro_rules! define_split_wrappers {
    (
        $split_fn:ident,
        $split_binding:ident,
        $split_backward_fn:ident,
        $split_backward_binding:ident,
        $ty:ty,
        $op_split:expr,
        $op_backward:expr
    ) => {
        #[cfg(cuda)]
        pub fn $split_fn(
            input: &DevicePtr<$ty>,
            output: &DevicePtr<$ty>,
            rows: usize,
            input_dim: usize,
            part_dim: usize,
            part_idx: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            validate_len(
                $op_split,
                input.len(),
                checked_mul2($op_split, rows, input_dim)?,
            )?;
            validate_len(
                $op_split,
                output.len(),
                checked_mul2($op_split, rows, part_dim)?,
            )?;
            if output.len() == 0 {
                return Ok(());
            }
            let rows_i32 = to_i32_len(concat!($op_split, "(rows)"), rows)?;
            let input_dim_i32 = to_i32_len(concat!($op_split, "(input_dim)"), input_dim)?;
            let part_dim_i32 = to_i32_len(concat!($op_split, "(part_dim)"), part_dim)?;
            let part_idx_i32 = to_i32_len(concat!($op_split, "(part_idx)"), part_idx)?;
            let status = unsafe {
                bindings::$split_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    rows_i32,
                    input_dim_i32,
                    part_dim_i32,
                    part_idx_i32,
                    input.as_raw() as *mut std::os::raw::c_int,
                    output.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_split,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $split_backward_fn(
            grad_out: &DevicePtr<$ty>,
            input_grad: &DevicePtr<$ty>,
            rows: usize,
            input_dim: usize,
            part_dim: usize,
            part_idx: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            validate_len(
                $op_backward,
                grad_out.len(),
                checked_mul2($op_backward, rows, part_dim)?,
            )?;
            validate_len(
                $op_backward,
                input_grad.len(),
                checked_mul2($op_backward, rows, input_dim)?,
            )?;
            if grad_out.len() == 0 {
                return Ok(());
            }
            let rows_i32 = to_i32_len(concat!($op_backward, "(rows)"), rows)?;
            let input_dim_i32 = to_i32_len(concat!($op_backward, "(input_dim)"), input_dim)?;
            let part_dim_i32 = to_i32_len(concat!($op_backward, "(part_dim)"), part_dim)?;
            let part_idx_i32 = to_i32_len(concat!($op_backward, "(part_idx)"), part_idx)?;
            let status = unsafe {
                bindings::$split_backward_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    rows_i32,
                    input_dim_i32,
                    part_dim_i32,
                    part_idx_i32,
                    grad_out.as_raw() as *mut std::os::raw::c_int,
                    input_grad.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_backward,
                    code: status as u32,
                })
            }
        }
    };
}

define_split_wrappers!(
    split_last_dim,
    cuda_split_last_dim,
    split_last_dim_backward,
    cuda_split_last_dim_backward,
    f64,
    "cuda::kernels::split_last_dim",
    "cuda::kernels::split_last_dim_backward"
);
define_split_wrappers!(
    split_last_dim_f32,
    cuda_split_last_dim_f32,
    split_last_dim_backward_f32,
    cuda_split_last_dim_backward_f32,
    f32,
    "cuda::kernels::split_last_dim_f32",
    "cuda::kernels::split_last_dim_backward_f32"
);

macro_rules! define_broadcast_batch_wrappers {
    (
        $broadcast_fn:ident,
        $broadcast_binding:ident,
        $broadcast_backward_fn:ident,
        $broadcast_backward_binding:ident,
        $ty:ty,
        $op_broadcast:expr,
        $op_backward:expr
    ) => {
        #[cfg(cuda)]
        pub fn $broadcast_fn(
            input: &DevicePtr<$ty>,
            output: &DevicePtr<$ty>,
            batch_size: usize,
            inner_len: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            validate_len($op_broadcast, input.len(), inner_len)?;
            validate_len(
                $op_broadcast,
                output.len(),
                checked_mul2($op_broadcast, batch_size, inner_len)?,
            )?;
            if output.len() == 0 {
                return Ok(());
            }
            let batch_i32 = to_i32_len(concat!($op_broadcast, "(batch_size)"), batch_size)?;
            let inner_i32 = to_i32_len(concat!($op_broadcast, "(inner_len)"), inner_len)?;
            let status = unsafe {
                bindings::$broadcast_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    batch_i32,
                    inner_i32,
                    input.as_raw() as *mut std::os::raw::c_int,
                    output.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_broadcast,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $broadcast_backward_fn(
            grad_out: &DevicePtr<$ty>,
            input_grad: &DevicePtr<$ty>,
            batch_size: usize,
            inner_len: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            validate_len(
                $op_backward,
                grad_out.len(),
                checked_mul2($op_backward, batch_size, inner_len)?,
            )?;
            validate_len($op_backward, input_grad.len(), inner_len)?;
            if inner_len == 0 {
                return Ok(());
            }
            let batch_i32 = to_i32_len(concat!($op_backward, "(batch_size)"), batch_size)?;
            let inner_i32 = to_i32_len(concat!($op_backward, "(inner_len)"), inner_len)?;
            let status = unsafe {
                bindings::$broadcast_backward_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    batch_i32,
                    inner_i32,
                    grad_out.as_raw() as *mut std::os::raw::c_int,
                    input_grad.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_backward,
                    code: status as u32,
                })
            }
        }
    };
}

define_broadcast_batch_wrappers!(
    broadcast_batch,
    cuda_broadcast_batch,
    broadcast_batch_backward,
    cuda_broadcast_batch_backward,
    f64,
    "cuda::kernels::broadcast_batch",
    "cuda::kernels::broadcast_batch_backward"
);
define_broadcast_batch_wrappers!(
    broadcast_batch_f32,
    cuda_broadcast_batch_f32,
    broadcast_batch_backward_f32,
    cuda_broadcast_batch_backward_f32,
    f32,
    "cuda::kernels::broadcast_batch_f32",
    "cuda::kernels::broadcast_batch_backward_f32"
);

macro_rules! define_transpose_wrappers {
    (
        $last_two_fn:ident,
        $last_two_binding:ident,
        $transpose_4d_fn:ident,
        $transpose_4d_binding:ident,
        $ty:ty,
        $op_last_two:expr,
        $op_4d:expr
    ) => {
        #[cfg(cuda)]
        pub fn $last_two_fn(
            input: &DevicePtr<$ty>,
            output: &DevicePtr<$ty>,
            outer: usize,
            rows: usize,
            cols: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let expected = outer
                .checked_mul(rows)
                .and_then(|v| v.checked_mul(cols))
                .ok_or(CudaError::SizeOverflow {
                    op: $op_last_two,
                    count: outer,
                    elem_size: rows.saturating_mul(cols),
                })?;
            validate_len($op_last_two, input.len(), expected)?;
            validate_len($op_last_two, output.len(), expected)?;
            if expected == 0 {
                return Ok(());
            }
            let outer_i32 = to_i32_len(concat!($op_last_two, "(outer)"), outer)?;
            let rows_i32 = to_i32_len(concat!($op_last_two, "(rows)"), rows)?;
            let cols_i32 = to_i32_len(concat!($op_last_two, "(cols)"), cols)?;
            let status = unsafe {
                bindings::$last_two_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    outer_i32,
                    rows_i32,
                    cols_i32,
                    input.as_raw() as *mut std::os::raw::c_int,
                    output.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_last_two,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $transpose_4d_fn(
            input: &DevicePtr<$ty>,
            output: &DevicePtr<$ty>,
            shape: [usize; 4],
            dim0: usize,
            dim1: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let expected = shape
                .iter()
                .try_fold(1usize, |acc, &v| acc.checked_mul(v))
                .ok_or(CudaError::SizeOverflow {
                    op: $op_4d,
                    count: shape[0],
                    elem_size: shape[1].saturating_mul(shape[2]).saturating_mul(shape[3]),
                })?;
            validate_len($op_4d, input.len(), expected)?;
            validate_len($op_4d, output.len(), expected)?;
            if expected == 0 {
                return Ok(());
            }
            let d0 = to_i32_len(concat!($op_4d, "(d0)"), shape[0])?;
            let d1 = to_i32_len(concat!($op_4d, "(d1)"), shape[1])?;
            let d2 = to_i32_len(concat!($op_4d, "(d2)"), shape[2])?;
            let d3 = to_i32_len(concat!($op_4d, "(d3)"), shape[3])?;
            let dim0_i32 = to_i32_len(concat!($op_4d, "(dim0)"), dim0)?;
            let dim1_i32 = to_i32_len(concat!($op_4d, "(dim1)"), dim1)?;
            let status = unsafe {
                bindings::$transpose_4d_binding(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    d0,
                    d1,
                    d2,
                    d3,
                    dim0_i32,
                    dim1_i32,
                    input.as_raw() as *mut std::os::raw::c_int,
                    output.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_4d,
                    code: status as u32,
                })
            }
        }
    };
}

define_transpose_wrappers!(
    transpose_last_two,
    cuda_transpose_last_two,
    transpose_4d,
    cuda_transpose_4d,
    f64,
    "cuda::kernels::transpose_last_two",
    "cuda::kernels::transpose_4d"
);
define_transpose_wrappers!(
    transpose_last_two_f32,
    cuda_transpose_last_two_f32,
    transpose_4d_f32,
    cuda_transpose_4d_f32,
    f32,
    "cuda::kernels::transpose_last_two_f32",
    "cuda::kernels::transpose_4d_f32"
);

macro_rules! define_batched_attention_wrappers {
    (
        $qk_fn:ident,
        $qk_binding:ident,
        $qk_backward_fn:ident,
        $qk_backward_binding:ident,
        $attn_backward_fn:ident,
        $attn_backward_binding:ident,
        $ty:ty,
        $op_qk:expr,
        $op_qk_backward:expr,
        $op_attn_backward:expr
    ) => {
        #[cfg(cuda)]
        pub fn $qk_fn(
            q: &DevicePtr<$ty>,
            k: &DevicePtr<$ty>,
            out: &DevicePtr<$ty>,
            batches: usize,
            seq: usize,
            dim: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let q_len = checked_mul3($op_qk, batches, seq, dim)?;
            let out_len = checked_mul3($op_qk, batches, seq, seq)?;
            validate_len($op_qk, q.len(), q_len)?;
            validate_len($op_qk, k.len(), q_len)?;
            validate_len($op_qk, out.len(), out_len)?;
            if out_len == 0 {
                return Ok(());
            }
            let batches_i32 = to_i32_len(concat!($op_qk, "(batches)"), batches)?;
            let seq_i32 = to_i32_len(concat!($op_qk, "(seq)"), seq)?;
            let dim_i32 = to_i32_len(concat!($op_qk, "(dim)"), dim)?;
            let status = unsafe {
                bindings::$qk_binding(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    batches_i32,
                    seq_i32,
                    dim_i32,
                    q.as_raw() as *mut std::os::raw::c_int,
                    k.as_raw() as *mut std::os::raw::c_int,
                    out.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_qk,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $qk_backward_fn(
            grad_out: &DevicePtr<$ty>,
            q: &DevicePtr<$ty>,
            k: &DevicePtr<$ty>,
            q_grad: &DevicePtr<$ty>,
            k_grad: &DevicePtr<$ty>,
            batches: usize,
            seq: usize,
            dim: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let q_len = checked_mul3($op_qk_backward, batches, seq, dim)?;
            let out_len = checked_mul3($op_qk_backward, batches, seq, seq)?;
            validate_len($op_qk_backward, grad_out.len(), out_len)?;
            validate_len($op_qk_backward, q.len(), q_len)?;
            validate_len($op_qk_backward, k.len(), q_len)?;
            validate_len($op_qk_backward, q_grad.len(), q_len)?;
            validate_len($op_qk_backward, k_grad.len(), q_len)?;
            if q_len == 0 {
                return Ok(());
            }
            let batches_i32 = to_i32_len(concat!($op_qk_backward, "(batches)"), batches)?;
            let seq_i32 = to_i32_len(concat!($op_qk_backward, "(seq)"), seq)?;
            let dim_i32 = to_i32_len(concat!($op_qk_backward, "(dim)"), dim)?;
            let status = unsafe {
                bindings::$qk_backward_binding(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    batches_i32,
                    seq_i32,
                    dim_i32,
                    grad_out.as_raw() as *mut std::os::raw::c_int,
                    q.as_raw() as *mut std::os::raw::c_int,
                    k.as_raw() as *mut std::os::raw::c_int,
                    q_grad.as_raw() as *mut std::os::raw::c_int,
                    k_grad.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_qk_backward,
                    code: status as u32,
                })
            }
        }

        #[cfg(cuda)]
        pub fn $attn_backward_fn(
            grad_out: &DevicePtr<$ty>,
            probs: &DevicePtr<$ty>,
            values: &DevicePtr<$ty>,
            probs_grad: &DevicePtr<$ty>,
            values_grad: &DevicePtr<$ty>,
            batches: usize,
            seq: usize,
            head_dim: usize,
        ) -> CudaResult<()> {
            crate::cuda::init()?;
            let probs_len = checked_mul3($op_attn_backward, batches, seq, seq)?;
            let values_len = checked_mul3($op_attn_backward, batches, seq, head_dim)?;
            validate_len($op_attn_backward, grad_out.len(), values_len)?;
            validate_len($op_attn_backward, probs.len(), probs_len)?;
            validate_len($op_attn_backward, values.len(), values_len)?;
            validate_len($op_attn_backward, probs_grad.len(), probs_len)?;
            validate_len($op_attn_backward, values_grad.len(), values_len)?;
            if values_len == 0 && probs_len == 0 {
                return Ok(());
            }
            let batches_i32 = to_i32_len(concat!($op_attn_backward, "(batches)"), batches)?;
            let seq_i32 = to_i32_len(concat!($op_attn_backward, "(seq)"), seq)?;
            let head_dim_i32 = to_i32_len(concat!($op_attn_backward, "(head_dim)"), head_dim)?;
            let status = unsafe {
                bindings::$attn_backward_binding(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    batches_i32,
                    seq_i32,
                    head_dim_i32,
                    grad_out.as_raw() as *mut std::os::raw::c_int,
                    probs.as_raw() as *mut std::os::raw::c_int,
                    values.as_raw() as *mut std::os::raw::c_int,
                    probs_grad.as_raw() as *mut std::os::raw::c_int,
                    values_grad.as_raw() as *mut std::os::raw::c_int,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(CudaError::Runtime {
                    op: $op_attn_backward,
                    code: status as u32,
                })
            }
        }
    };
}

define_batched_attention_wrappers!(
    batched_qk_scores,
    cuda_batched_qk_scores,
    batched_qk_scores_backward,
    cuda_batched_qk_scores_backward,
    attention_weighted_sum_backward,
    cuda_attention_weighted_sum_backward,
    f64,
    "cuda::kernels::batched_qk_scores",
    "cuda::kernels::batched_qk_scores_backward",
    "cuda::kernels::attention_weighted_sum_backward"
);
define_batched_attention_wrappers!(
    batched_qk_scores_f32,
    cuda_batched_qk_scores_f32,
    batched_qk_scores_backward_f32,
    cuda_batched_qk_scores_backward_f32,
    attention_weighted_sum_backward_f32,
    cuda_attention_weighted_sum_backward_f32,
    f32,
    "cuda::kernels::batched_qk_scores_f32",
    "cuda::kernels::batched_qk_scores_backward_f32",
    "cuda::kernels::attention_weighted_sum_backward_f32"
);

#[cfg(test)]
mod tests {
    use super::validate_softmax_dims;

    #[test]
    fn softmax_dim_validation_rejects_zero_cols_with_rows() {
        let err = validate_softmax_dims("test", 2, 0).unwrap_err();
        assert!(format!("{err}").contains("cols must be greater than zero"));
    }

    #[test]
    fn softmax_dim_validation_accepts_wide_rows() {
        // CUDA kernel now supports arbitrary column counts via shared memory reduction
        assert!(validate_softmax_dims("test", 1, 33).is_ok());
        assert!(validate_softmax_dims("test", 4, 1024).is_ok());
        assert!(validate_softmax_dims("test", 8, 4096).is_ok());
    }

    #[test]
    fn softmax_dim_validation_allows_empty_rows() {
        assert!(validate_softmax_dims("test", 0, 4096).is_ok());
    }
}
