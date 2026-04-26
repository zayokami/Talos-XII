//! CUDA activation kernel wrappers.

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
