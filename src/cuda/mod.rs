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
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(cuda)]
use std::{ffi::c_char, ffi::CStr};

/// Indicates whether CUDA is available and initialized
static CUDA_INITIALIZED: AtomicBool = AtomicBool::new(false);

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
