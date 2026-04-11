//! cuBLAS wrapper for GPU-accelerated BLAS operations
//!
//! Provides GPU matrix multiplication using NVIDIA cuBLAS library.

#[cfg(cuda)]
use crate::cuda::bindings::{
    cublasCreate_v2, cublasDestroy_v2, cublasDgemm_v2, cublasSetStream_v2,
    CUBLAS_OP_N, CUBLAS_OP_T,
};

/// cuBLAS context wrapper
#[cfg(cuda)]
pub struct Cublas {
    handle: usize,  // cublasHandle_t
}

#[cfg(cuda)]
impl Cublas {
    /// Create a new cuBLAS context
    pub fn new() -> Result<Self, ()> {
        unsafe {
            let mut handle: usize = 0;
            let result = cublasCreate_v2(&mut handle);
            if result != 0 {
                eprintln!("[CUDA] cublasCreate_v2 failed: {}", result);
                return Err(());
            }
            Ok(Cublas { handle })
        }
    }

    /// Set the CUDA stream for cuBLAS operations
    #[cfg(cuda)]
    pub fn set_stream(&mut self, stream: &crate::cuda::stream::CudaStream) -> Result<(), ()> {
        unsafe {
            let result = cublasSetStream_v2(self.handle, stream.as_raw());
            if result != 0 {
                eprintln!("[CUDA] cublasSetStream_v2 failed: {}", result);
                return Err(());
            }
        }
        Ok(())
    }

    /// Matrix multiplication: C = alpha * A * B + beta * C
    /// Uses row-major layout: A [m, k] * B [k, n] = C [m, n]
    #[cfg(cuda)]
    pub fn gemm(
        &mut self,
        transa: bool,  // transpose A
        transb: bool,  // transpose B
        m: i32,        // rows of A and C
        n: i32,        // columns of B and C
        k: i32,        // columns of A and rows of B
        alpha: f64,
        a: &[f64],     // A matrix data (row-major)
        lda: i32,      // leading dim of A
        b: &[f64],     // B matrix data (row-major)
        ldb: i32,      // leading dim of B
        beta: f64,
        c: &mut [f64], // C matrix data (row-major)
        ldc: i32,      // leading dim of C
    ) -> Result<(), ()> {
        // cuBLAS uses column-major, but we're using row-major
        // For row-major: C = A * B means column-major: C = B^T * A^T
        // So we swap A <-> B and transpose the operation

        let op_a = if transa {
            CUBLAS_OP_T
        } else {
            CUBLAS_OP_N
        };
        let op_b = if transb {
            CUBLAS_OP_T
        } else {
            CUBLAS_OP_N
        };

        unsafe {
            let result = cublasDgemm_v2(
                self.handle,
                op_b,   // swapped
                op_a,   // swapped
                n,      // swapped: columns of B (rows of C)
                m,      // swapped: rows of A (cols of C)
                k,      // K dimension unchanged
                &alpha,
                b.as_ptr(),
                ldb,
                a.as_ptr(),
                lda,
                &beta,
                c.as_mut_ptr(),
                ldc,
            );
            if result != 0 {
                eprintln!("[CUDA] cublasDgemm_v2 failed: {}", result);
                return Err(());
            }
        }
        Ok(())
    }
}

#[cfg(cuda)]
impl Drop for Cublas {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle);
        }
    }
}

// =============================================================================
// Stub implementations for non-CUDA builds
// =============================================================================

#[cfg(not(cuda))]
pub struct Cublas;

#[cfg(not(cuda))]
impl Cublas {
    pub fn new() -> Result<Self, ()> {
        Err(())
    }

    pub fn set_stream(&mut self, _stream: &crate::cuda::stream::CudaStream) -> Result<(), ()> {
        Err(())
    }

    pub fn gemm(
        &mut self,
        _transa: bool,
        _transb: bool,
        _m: i32,
        _n: i32,
        _k: i32,
        _alpha: f64,
        _a: &[f64],
        _lda: i32,
        _b: &[f64],
        _ldb: i32,
        _beta: f64,
        _c: &mut [f64],
        _ldc: i32,
    ) -> Result<(), ()> {
        Err(())
    }
}
