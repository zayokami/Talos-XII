//! CUDA GPU acceleration module for Talos-XII
//!
//! This module provides CUDA GPU acceleration for neural network operations.
//! When the `cuda` feature is not enabled, all operations gracefully fall back to CPU.
#![allow(dead_code)]

#[cfg(cuda)]
pub mod bindings;
#[cfg(cuda)]
pub mod blas;
pub mod error;
#[cfg(cuda)]
pub mod kernels;
#[cfg(cuda)]
pub mod memory;
#[cfg(cuda)]
pub mod stream;

use self::error::{CudaError, CudaResult};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(cuda)]
use std::{ffi::c_char, ffi::CStr};

/// Indicates whether CUDA is available and initialized
static CUDA_INITIALIZED: AtomicBool = AtomicBool::new(false);
static CUDA_MATMUL_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_INIT: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_ALLOC: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_COPY: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_GEMM: AtomicU64 = AtomicU64::new(0);

/// Runtime observability counters for CUDA matmul routing.
#[derive(Debug, Clone, Copy)]
pub struct CudaRuntimeStats {
    pub matmul_attempts: u64,
    pub matmul_successes: u64,
    pub matmul_fallback_init: u64,
    pub matmul_fallback_alloc: u64,
    pub matmul_fallback_copy: u64,
    pub matmul_fallback_gemm: u64,
}

pub fn record_matmul_attempt() {
    CUDA_MATMUL_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_matmul_success() {
    CUDA_MATMUL_SUCCESSES.fetch_add(1, Ordering::Relaxed);
}

pub fn record_matmul_fallback(stage: &'static str) {
    match stage {
        "init" => {
            CUDA_MATMUL_FALLBACK_INIT.fetch_add(1, Ordering::Relaxed);
        }
        "alloc" => {
            CUDA_MATMUL_FALLBACK_ALLOC.fetch_add(1, Ordering::Relaxed);
        }
        "copy" => {
            CUDA_MATMUL_FALLBACK_COPY.fetch_add(1, Ordering::Relaxed);
        }
        "gemm" => {
            CUDA_MATMUL_FALLBACK_GEMM.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

pub fn runtime_stats() -> CudaRuntimeStats {
    CudaRuntimeStats {
        matmul_attempts: CUDA_MATMUL_ATTEMPTS.load(Ordering::Relaxed),
        matmul_successes: CUDA_MATMUL_SUCCESSES.load(Ordering::Relaxed),
        matmul_fallback_init: CUDA_MATMUL_FALLBACK_INIT.load(Ordering::Relaxed),
        matmul_fallback_alloc: CUDA_MATMUL_FALLBACK_ALLOC.load(Ordering::Relaxed),
        matmul_fallback_copy: CUDA_MATMUL_FALLBACK_COPY.load(Ordering::Relaxed),
        matmul_fallback_gemm: CUDA_MATMUL_FALLBACK_GEMM.load(Ordering::Relaxed),
    }
}

/// Device information
pub struct CudaDevice {
    pub id: usize,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
}

/// Initialize CUDA runtime
/// Returns Ok(()) if CUDA is available, Err(()) otherwise
#[cfg(cuda)]
pub fn init() -> CudaResult<()> {
    if CUDA_INITIALIZED.load(Ordering::SeqCst) {
        return Ok(());
    }

    unsafe {
        let init_result = bindings::cuInit(0);
        if init_result != bindings::CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuInit",
                code: init_result,
            });
        }

        let mut count: i32 = 0;
        let count_result = bindings::cuDeviceGetCount(&mut count);
        if count_result != bindings::CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuDeviceGetCount",
                code: count_result,
            });
        }
        if count <= 0 {
            return Err(CudaError::NoDevice {
                op: "cuDeviceGetCount",
            });
        }
    }

    CUDA_INITIALIZED.store(true, Ordering::SeqCst);
    Ok(())
}

#[cfg(not(cuda))]
pub fn init() -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild { op: "cuda::init" })
}

/// Check if CUDA is available
pub fn is_available() -> bool {
    #[cfg(cuda)]
    {
        init().is_ok()
    }
    #[cfg(not(cuda))]
    {
        false
    }
}

/// Get device information
#[cfg(cuda)]
pub fn get_device_info(device_id: usize) -> CudaResult<CudaDevice> {
    use bindings::*;

    init()?;

    unsafe {
        let mut device: CUdevice = 0;
        let get_result = cuDeviceGet(&mut device, device_id as i32);
        if get_result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuDeviceGet",
                code: get_result,
            });
        }

        let mut name = [0 as c_char; 256];
        let name_result = cuDeviceGetName(name.as_mut_ptr(), 256, device);
        if name_result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuDeviceGetName",
                code: name_result,
            });
        }

        let mut cc_major: i32 = 0;
        let mut cc_minor: i32 = 0;
        let cc_major_result = cuDeviceGetAttribute(
            &mut cc_major,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        );
        if cc_major_result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuDeviceGetAttribute(major)",
                code: cc_major_result,
            });
        }
        let cc_minor_result = cuDeviceGetAttribute(
            &mut cc_minor,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        );
        if cc_minor_result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuDeviceGetAttribute(minor)",
                code: cc_minor_result,
            });
        }

        let mut total_mem: usize = 0;
        let mem_result = cuDeviceTotalMem(&mut total_mem, device);
        if mem_result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuDeviceTotalMem",
                code: mem_result,
            });
        }

        let name_str = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();

        Ok(CudaDevice {
            id: device_id,
            name: name_str,
            compute_capability: (cc_major as u32, cc_minor as u32),
            total_memory: total_mem,
        })
    }
}

#[cfg(not(cuda))]
pub fn get_device_info(_device_id: usize) -> CudaResult<CudaDevice> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::get_device_info",
    })
}

/// Number of available CUDA devices
#[cfg(cuda)]
pub fn device_count() -> CudaResult<usize> {
    init()?;
    unsafe {
        let mut count: i32 = 0;
        let result = bindings::cuDeviceGetCount(&mut count);
        if result != bindings::CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuDeviceGetCount",
                code: result,
            });
        }
        if count <= 0 {
            return Err(CudaError::NoDevice {
                op: "cuDeviceGetCount",
            });
        }
        Ok(count as usize)
    }
}

#[cfg(not(cuda))]
pub fn device_count() -> CudaResult<usize> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::device_count",
    })
}
