//! CUDA Stream management
//!
//! Provides asynchronous stream operations for CUDA kernels.
#![allow(dead_code)]

#[cfg(cuda)]
use crate::cuda::bindings::{
    cuStreamCreate, cuStreamDestroy, cuStreamSynchronize, CUstream, CUDA_SUCCESS,
};

/// CUDA Stream wrapper
#[cfg(cuda)]
pub struct CudaStream {
    handle: CUstream,
}

#[cfg(cuda)]
impl CudaStream {
    /// Create a new CUDA stream
    pub fn new() -> Result<Self, ()> {
        unsafe {
            let mut handle: CUstream = std::ptr::null_mut();
            let result = cuStreamCreate(&mut handle, 0);
            if result != CUDA_SUCCESS {
                eprintln!("[CUDA] cuStreamCreate failed: {}", result);
                return Err(());
            }
            Ok(CudaStream { handle })
        }
    }

    /// Synchronize the stream (block until all operations complete)
    pub fn synchronize(&self) -> Result<(), ()> {
        unsafe {
            let result = cuStreamSynchronize(self.handle);
            if result != CUDA_SUCCESS {
                eprintln!("[CUDA] cuStreamSynchronize failed: {}", result);
                return Err(());
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
            cuStreamDestroy(self.handle);
        }
    }
}

// =============================================================================
// Stub implementations for non-CUDA builds
// =============================================================================

#[cfg(not(cuda))]
pub struct CudaStream;

#[cfg(not(cuda))]
impl CudaStream {
    pub fn new() -> Result<Self, ()> {
        Err(())
    }

    pub fn synchronize(&self) -> Result<(), ()> {
        Err(())
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
