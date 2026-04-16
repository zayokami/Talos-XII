//! CUDA GPU acceleration module for Talos-XII
//!
//! This module provides CUDA GPU acceleration for neural network operations.
//! When the `cuda` feature is not enabled, all operations gracefully fall back to CPU.
#![allow(dead_code)]

#[cfg(cuda)]
pub mod bindings;
#[cfg(cuda)]
pub mod blas;
#[cfg(cuda)]
pub mod memory;
#[cfg(cuda)]
pub mod stream;

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
pub fn init() -> Result<(), ()> {
    if CUDA_INITIALIZED.load(Ordering::SeqCst) {
        return Ok(());
    }

    unsafe {
        let result = bindings::cuInit(0);
        if result != bindings::CUDA_SUCCESS {
            eprintln!("[CUDA] cuInit failed with error code: {}", result);
            return Err(());
        }
    }

    CUDA_INITIALIZED.store(true, Ordering::SeqCst);
    println!("[CUDA] Initialized successfully");
    Ok(())
}

#[cfg(not(cuda))]
pub fn init() -> Result<(), ()> {
    Err(())
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
pub fn get_device_info(device_id: usize) -> Result<CudaDevice, ()> {
    use bindings::*;

    init()?;

    unsafe {
        let mut device: CUdevice = 0;
        let result = cuDeviceGet(&mut device, device_id as i32);
        if result != CUDA_SUCCESS {
            return Err(());
        }

        let mut name = [0 as c_char; 256];
        cuDeviceGetName(name.as_mut_ptr(), 256, device);

        let mut cc_major: i32 = 0;
        let mut cc_minor: i32 = 0;
        cuDeviceGetAttribute(
            &mut cc_major,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        );
        cuDeviceGetAttribute(
            &mut cc_minor,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        );

        let mut total_mem: usize = 0;
        cuDeviceTotalMem(&mut total_mem, device);

        let name_str = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();

        println!(
            "[CUDA] Device {}: {} (CC {}.{}, {} MB)",
            device_id,
            name_str,
            cc_major,
            cc_minor,
            total_mem / (1024 * 1024)
        );

        Ok(CudaDevice {
            id: device_id,
            name: name_str,
            compute_capability: (cc_major as u32, cc_minor as u32),
            total_memory: total_mem,
        })
    }
}

#[cfg(not(cuda))]
pub fn get_device_info(_device_id: usize) -> Result<CudaDevice, ()> {
    Err(())
}

/// Number of available CUDA devices
#[cfg(cuda)]
pub fn device_count() -> usize {
    if init().is_err() {
        return 0;
    }
    unsafe {
        let mut count: i32 = 0;
        let result = bindings::cuDeviceGetCount(&mut count);
        if result != bindings::CUDA_SUCCESS {
            0
        } else {
            count as usize
        }
    }
}

#[cfg(not(cuda))]
pub fn device_count() -> usize {
    0
}
