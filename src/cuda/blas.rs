//! cuBLAS wrapper for GPU-accelerated BLAS operations
//!
//! Provides GPU matrix multiplication using NVIDIA cuBLAS library.
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

#[cfg(cuda)]
use crate::cuda::bindings::{
    cublasCreate_v2, cublasDestroy_v2, cublasDgemm_v2, cublasGemmEx, cublasHandle_t,
    cublasSetStream_v2, cublasSgemm_v2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16BF, CUDA_R_32F,
};
use crate::cuda::error::{CudaError, CudaResult};
#[cfg(cuda)]
use std::cell::RefCell;

/// cuBLAS context wrapper
#[cfg(cuda)]
pub struct Cublas {
    handle: cublasHandle_t,
}

#[cfg(cuda)]
impl Cublas {
    /// Create a new cuBLAS context
    pub fn new() -> CudaResult<Self> {
        crate::cuda::init()?;
        unsafe {
            let mut handle: cublasHandle_t = std::ptr::null_mut();
            let result = cublasCreate_v2(&mut handle);
            if result != 0 {
                return Err(CudaError::Blas {
                    op: "cublasCreate_v2",
                    code: result,
                });
            }
            Ok(Cublas { handle })
        }
    }

    /// Set the CUDA stream for cuBLAS operations
    pub fn set_stream(&mut self, stream: &crate::cuda::stream::CudaStream) -> CudaResult<()> {
        unsafe {
            let result = cublasSetStream_v2(self.handle, stream.as_raw());
            if result != 0 {
                return Err(CudaError::Blas {
                    op: "cublasSetStream_v2",
                    code: result,
                });
            }
        }
        Ok(())
    }

    /// Matrix multiplication: C = alpha * A * B + beta * C
    /// Uses row-major layout: A [m, k] * B [k, n] = C [m, n]
    pub fn gemm(
        &mut self,
        transa: bool, // transpose A
        transb: bool, // transpose B
        m: i32,       // rows of A and C
        n: i32,       // columns of B and C
        k: i32,       // columns of A and rows of B
        alpha: f64,
        a: usize, // device ptr for A matrix (row-major)
        lda: i32, // leading dim of A
        b: usize, // device ptr for B matrix (row-major)
        ldb: i32, // leading dim of B
        beta: f64,
        c: usize, // device ptr for C matrix (row-major)
        ldc: i32, // leading dim of C
    ) -> CudaResult<()> {
        // cuBLAS uses column-major, but we're using row-major
        // For row-major: C = A * B means column-major: C = B^T * A^T
        // So we swap A <-> B and transpose the operation

        let op_a = if transa { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if transb { CUBLAS_OP_T } else { CUBLAS_OP_N };

        unsafe {
            let result = cublasDgemm_v2(
                self.handle,
                op_b, // swapped
                op_a, // swapped
                n,    // swapped: columns of B (rows of C)
                m,    // swapped: rows of A (cols of C)
                k,    // K dimension unchanged
                &alpha,
                b as *const f64,
                ldb,
                a as *const f64,
                lda,
                &beta,
                c as *mut f64,
                ldc,
            );
            if result != 0 {
                return Err(CudaError::Blas {
                    op: "cublasDgemm_v2",
                    code: result,
                });
            }
        }
        Ok(())
    }

    /// Matrix multiplication (single precision): C = alpha * A * B + beta * C
    pub fn gemm_f32(
        &mut self,
        transa: bool,
        transb: bool,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: usize,
        lda: i32,
        b: usize,
        ldb: i32,
        beta: f32,
        c: usize,
        ldc: i32,
    ) -> CudaResult<()> {
        let op_a = if transa { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if transb { CUBLAS_OP_T } else { CUBLAS_OP_N };

        unsafe {
            let result = cublasSgemm_v2(
                self.handle,
                op_b,
                op_a,
                n,
                m,
                k,
                &alpha,
                b as *const f32,
                ldb,
                a as *const f32,
                lda,
                &beta,
                c as *mut f32,
                ldc,
            );
            if result != 0 {
                return Err(CudaError::Blas {
                    op: "cublasSgemm_v2",
                    code: result,
                });
            }
        }
        Ok(())
    }

    /// BF16 Tensor Core GEMM with F32 accumulation and F32 output.
    ///
    /// Inputs are row-major BF16 matrices. Output is row-major F32 because the
    /// rest of the autograd engine uses F32 as the compute dtype for BF16 data.
    pub fn gemm_bf16_to_f32(
        &mut self,
        transa: bool,
        transb: bool,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: usize,
        lda: i32,
        b: usize,
        ldb: i32,
        beta: f32,
        c: usize,
        ldc: i32,
    ) -> CudaResult<()> {
        let op_a = if transa { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if transb { CUBLAS_OP_T } else { CUBLAS_OP_N };

        unsafe {
            let result = cublasGemmEx(
                self.handle,
                op_b,
                op_a,
                n,
                m,
                k,
                (&alpha as *const f32).cast(),
                b as *const std::ffi::c_void,
                CUDA_R_16BF,
                ldb,
                a as *const std::ffi::c_void,
                CUDA_R_16BF,
                lda,
                (&beta as *const f32).cast(),
                c as *mut std::ffi::c_void,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if result != 0 {
                return Err(CudaError::Blas {
                    op: "cublasGemmEx(BF16)",
                    code: result,
                });
            }
        }
        Ok(())
    }
}

#[cfg(cuda)]
thread_local! {
    static THREAD_CUBLAS: RefCell<Option<Cublas>> = const { RefCell::new(None) };
}

/// Thread-local cuBLAS GEMM entry.
///
/// CUDA docs recommend minimizing cublasCreate/cublasDestroy and using one handle
/// per host thread. This API reuses a handle in thread-local storage.
#[cfg(cuda)]
pub fn gemm_thread_local(
    transa: bool,
    transb: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: usize,
    lda: i32,
    b: usize,
    ldb: i32,
    beta: f64,
    c: usize,
    ldc: i32,
) -> CudaResult<()> {
    THREAD_CUBLAS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Cublas::new()?);
        }
        let cublas = slot.as_mut().ok_or(CudaError::InvalidInput {
            op: "cuda::blas::gemm_thread_local",
            message: "thread-local cublas handle unavailable",
        })?;
        cublas.gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    })
}

#[cfg(cuda)]
pub fn gemm_thread_local_f32(
    transa: bool,
    transb: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: usize,
    lda: i32,
    b: usize,
    ldb: i32,
    beta: f32,
    c: usize,
    ldc: i32,
) -> CudaResult<()> {
    THREAD_CUBLAS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Cublas::new()?);
        }
        let cublas = slot.as_mut().ok_or(CudaError::InvalidInput {
            op: "cuda::blas::gemm_thread_local_f32",
            message: "thread-local cublas handle unavailable",
        })?;
        cublas.gemm_f32(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    })
}

#[cfg(cuda)]
pub fn gemm_thread_local_bf16_to_f32(
    transa: bool,
    transb: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: usize,
    lda: i32,
    b: usize,
    ldb: i32,
    beta: f32,
    c: usize,
    ldc: i32,
) -> CudaResult<()> {
    THREAD_CUBLAS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Cublas::new()?);
        }
        let cublas = slot.as_mut().ok_or(CudaError::InvalidInput {
            op: "cuda::blas::gemm_thread_local_bf16_to_f32",
            message: "thread-local cublas handle unavailable",
        })?;
        cublas.gemm_bf16_to_f32(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    })
}

#[cfg(cuda)]
impl Drop for Cublas {
    fn drop(&mut self) {
        unsafe {
            let result = cublasDestroy_v2(self.handle);
            if result != 0 {
                eprintln!("[CUDA] cublasDestroy_v2 failed during drop: {}", result);
            }
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
    pub fn new() -> CudaResult<Self> {
        Err(CudaError::UnsupportedBuild {
            op: "cuda::blas::new",
        })
    }

    pub fn set_stream(&mut self, _stream: &crate::cuda::stream::CudaStream) -> CudaResult<()> {
        Err(CudaError::UnsupportedBuild {
            op: "cuda::blas::set_stream",
        })
    }

    pub fn gemm(
        &mut self,
        _transa: bool,
        _transb: bool,
        _m: i32,
        _n: i32,
        _k: i32,
        _alpha: f64,
        _a: usize,
        _lda: i32,
        _b: usize,
        _ldb: i32,
        _beta: f64,
        _c: usize,
        _ldc: i32,
    ) -> CudaResult<()> {
        Err(CudaError::UnsupportedBuild {
            op: "cuda::blas::gemm",
        })
    }

    pub fn gemm_f32(
        &mut self,
        _transa: bool,
        _transb: bool,
        _m: i32,
        _n: i32,
        _k: i32,
        _alpha: f32,
        _a: usize,
        _lda: i32,
        _b: usize,
        _ldb: i32,
        _beta: f32,
        _c: usize,
        _ldc: i32,
    ) -> CudaResult<()> {
        Err(CudaError::UnsupportedBuild {
            op: "cuda::blas::gemm_f32",
        })
    }

    pub fn gemm_bf16_to_f32(
        &mut self,
        _transa: bool,
        _transb: bool,
        _m: i32,
        _n: i32,
        _k: i32,
        _alpha: f32,
        _a: usize,
        _lda: i32,
        _b: usize,
        _ldb: i32,
        _beta: f32,
        _c: usize,
        _ldc: i32,
    ) -> CudaResult<()> {
        Err(CudaError::UnsupportedBuild {
            op: "cuda::blas::gemm_bf16_to_f32",
        })
    }
}

#[cfg(not(cuda))]
pub fn gemm_thread_local(
    _transa: bool,
    _transb: bool,
    _m: i32,
    _n: i32,
    _k: i32,
    _alpha: f64,
    _a: usize,
    _lda: i32,
    _b: usize,
    _ldb: i32,
    _beta: f64,
    _c: usize,
    _ldc: i32,
) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::blas::gemm_thread_local",
    })
}

#[cfg(not(cuda))]
pub fn gemm_thread_local_f32(
    _transa: bool,
    _transb: bool,
    _m: i32,
    _n: i32,
    _k: i32,
    _alpha: f32,
    _a: usize,
    _lda: i32,
    _b: usize,
    _ldb: i32,
    _beta: f32,
    _c: usize,
    _ldc: i32,
) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::blas::gemm_thread_local_f32",
    })
}

#[cfg(not(cuda))]
pub fn gemm_thread_local_bf16_to_f32(
    _transa: bool,
    _transb: bool,
    _m: i32,
    _n: i32,
    _k: i32,
    _alpha: f32,
    _a: usize,
    _lda: i32,
    _b: usize,
    _ldb: i32,
    _beta: f32,
    _c: usize,
    _ldc: i32,
) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::blas::gemm_thread_local_bf16_to_f32",
    })
}
