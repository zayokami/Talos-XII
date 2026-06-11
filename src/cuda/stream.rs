//! CUDA Stream management
//!
//! Provides asynchronous stream operations for CUDA kernels.
#![allow(dead_code)]

#[cfg(cuda)]
use crate::cuda::bindings::{
    cuStreamCreate, cuStreamDestroy, cuStreamSynchronize, CUstream, CUDA_SUCCESS,
};
use crate::cuda::error::{CudaError, CudaResult};

/// CUDA Stream wrapper
#[cfg(cuda)]
pub struct CudaStream {
    handle: CUstream,
}

#[cfg(cuda)]
impl CudaStream {
    /// Create a new CUDA stream
    pub fn new() -> CudaResult<Self> {
        unsafe {
            let mut handle: CUstream = std::ptr::null_mut();
            let result = cuStreamCreate(&mut handle, 0);
            if result != CUDA_SUCCESS {
                return Err(CudaError::Runtime {
                    op: "cuStreamCreate",
                    code: result,
                });
            }
            Ok(CudaStream { handle })
        }
    }

    /// Synchronize the stream (block until all operations complete)
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe {
            let result = cuStreamSynchronize(self.handle);
            if result != CUDA_SUCCESS {
                return Err(CudaError::Runtime {
                    op: "cuStreamSynchronize",
                    code: result,
                });
            }
        }
        Ok(())
    }

    /// Get raw handle
    pub fn as_raw(&self) -> CUstream {
        self.handle
    }
}

#[cfg(cuda)]
impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            let result = cuStreamDestroy(self.handle);
            if result != CUDA_SUCCESS {
                eprintln!("[CUDA] cuStreamDestroy failed during drop: {}", result);
            }
        }
    }
}

// =============================================================================
// Process-wide transfer stream
// =============================================================================

/// Raw handle of the process-wide transfer stream, stored as `usize` so the
/// `OnceLock` is `Sync`. States:
/// - unset: `cuda::init()` has not completed yet -> callers use the
///   synchronous `cudaMemcpy` path,
/// - `0`: stream creation failed -> permanently degraded to the synchronous
///   path (system behaves exactly as before async transfers existed),
/// - non-zero: valid `CUstream` handle.
///
/// The stream is created with flag 0 (`CU_STREAM_DEFAULT`), i.e. a *blocking*
/// stream: the legacy default stream and blocking streams implicitly
/// synchronize with each other in both directions. All custom kernels in this
/// crate launch on the legacy default stream, so "copy on the transfer
/// stream, kernel on the default stream" is naturally ordered.
///
/// The stream is intentionally never destroyed; it lives until process exit.
/// Destroying it from a `Drop` impl of a static would race with other
/// statics' teardown order (pinned pool, cuBLAS handles), so we let the OS
/// reclaim it.
#[cfg(cuda)]
static GLOBAL_TRANSFER_STREAM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Create the global transfer stream. Called once from the slow path of
/// `cuda::init()` (the initializing thread has the shared context current).
/// On failure the handle is recorded as 0 and the whole transfer subsystem
/// degrades to synchronous `cudaMemcpy`, with one warning logged.
#[cfg(cuda)]
pub(crate) fn init_global_transfer_stream() {
    GLOBAL_TRANSFER_STREAM.get_or_init(|| unsafe {
        let mut handle: CUstream = std::ptr::null_mut();
        let result = cuStreamCreate(&mut handle, 0);
        if result == CUDA_SUCCESS && !handle.is_null() {
            handle as usize
        } else {
            log::warn!(
                "[CUDA] cuStreamCreate for the global transfer stream failed (code {result}); \
                 transfers fall back to synchronous cudaMemcpy on the legacy default stream"
            );
            0
        }
    });
}

/// Raw handle of the global transfer stream, or `None` when async transfers
/// are unavailable (init not run yet, or stream creation failed). Callers
/// must fall back to the synchronous `cudaMemcpy` path on `None`.
#[cfg(cuda)]
pub fn global_transfer_stream() -> Option<CUstream> {
    match GLOBAL_TRANSFER_STREAM.get().copied() {
        Some(handle) if handle != 0 => Some(handle as CUstream),
        _ => None,
    }
}

// =============================================================================
// Stub implementations for non-CUDA builds
// =============================================================================

#[cfg(not(cuda))]
pub struct CudaStream;

#[cfg(not(cuda))]
impl CudaStream {
    pub fn new() -> CudaResult<Self> {
        Err(CudaError::UnsupportedBuild {
            op: "cuda::stream::new",
        })
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        Err(CudaError::UnsupportedBuild {
            op: "cuda::stream::synchronize",
        })
    }

    pub fn as_raw(&self) -> usize {
        0
    }
}

#[cfg(not(cuda))]
impl Default for CudaStream {
    fn default() -> Self {
        Self
    }
}
