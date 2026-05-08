//! CUDA activation kernel wrappers.
#![allow(clippy::too_many_arguments)]

use crate::cuda::bindings;
use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::memory::DevicePtr;

fn to_i32_len(op: &'static str, value: usize) -> CudaResult<i32> {
    i32::try_from(value).map_err(|_| CudaError::SizeOverflow {
        op,
        count: value,
        elem_size: std::mem::size_of::<f64>(),
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
