//! CUDA error model used across runtime, memory, stream, and cuBLAS wrappers.

use std::fmt::{Display, Formatter};

/// Unified CUDA result type.
pub type CudaResult<T> = Result<T, CudaError>;

/// Unified CUDA error type for diagnostics and propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// CUDA support was requested at runtime but the binary was built without CUDA.
    UnsupportedBuild { op: &'static str },
    /// Runtime API call failed with a CUDA driver/runtime error code.
    Runtime { op: &'static str, code: u32 },
    /// cuBLAS call failed with a cuBLAS status code.
    Blas { op: &'static str, code: i32 },
    /// Input shape/size does not meet API preconditions.
    InvalidInput {
        op: &'static str,
        message: &'static str,
    },
    /// Copy length mismatched between host and device buffers.
    SizeMismatch {
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Element count * element size overflowed usize.
    SizeOverflow {
        op: &'static str,
        count: usize,
        elem_size: usize,
    },
    /// No CUDA device is available.
    NoDevice { op: &'static str },
}

impl Display for CudaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBuild { op } => {
                write!(f, "{op}: CUDA feature is not enabled in this binary")
            }
            Self::Runtime { op, code } => write!(f, "{op}: CUDA runtime error code {code}"),
            Self::Blas { op, code } => write!(f, "{op}: cuBLAS error code {code}"),
            Self::InvalidInput { op, message } => write!(f, "{op}: invalid input ({message})"),
            Self::SizeMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{op}: size mismatch (expected {expected}, actual {actual})"
            ),
            Self::SizeOverflow {
                op,
                count,
                elem_size,
            } => write!(
                f,
                "{op}: size overflow (count={count}, elem_size={elem_size})"
            ),
            Self::NoDevice { op } => write!(f, "{op}: no CUDA device available"),
        }
    }
}

impl std::error::Error for CudaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_error_display_contains_operation() {
        let err = CudaError::Runtime {
            op: "cuMemAlloc",
            code: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("cuMemAlloc"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn cuda_error_size_mismatch_message() {
        let err = CudaError::SizeMismatch {
            op: "cuMemcpyHtoD",
            expected: 8,
            actual: 4,
        };
        let msg = err.to_string();
        assert!(msg.contains("expected 8"));
        assert!(msg.contains("actual 4"));
    }
}
