//! CUDA memory management
//!
//! Provides GPU memory allocation, deallocation, and CPU<->GPU data transfer.
#![allow(dead_code)]

#[cfg(cuda)]
use crate::cuda::bindings::{cuMemAlloc, cuMemFree, cuMemcpyDtoH, cuMemcpyHtoD, CUDA_SUCCESS};
#[cfg(cuda)]
use std::ffi::c_void;

/// Opaque CUDA memory pointer wrapper
pub struct DevicePtr<T> {
    ptr: usize,
    size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> DevicePtr<T> {
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn as_raw(&self) -> usize {
        self.ptr
    }
}

impl<T> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                cuMemFree(self.ptr);
            }
        }
    }
}

/// Allocate GPU memory for `count` elements of type T
#[cfg(cuda)]
pub fn alloc<T>(count: usize) -> Result<DevicePtr<T>, ()> {
    let size_bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or(())?;
    let mut ptr: usize = 0;

    unsafe {
        let result = cuMemAlloc(&mut ptr, size_bytes);
        if result != CUDA_SUCCESS {
            eprintln!("[CUDA] cuMemAlloc failed: {}", result);
            return Err(());
        }
    }

    Ok(DevicePtr {
        ptr,
        size: count,
        _phantom: std::marker::PhantomData,
    })
}

/// Free GPU memory (automatically called when DevicePtr is dropped)
#[cfg(cuda)]
pub fn free<T>(_device: &DevicePtr<T>) -> Result<(), ()> {
    Ok(())
}

/// Copy data from host (CPU) to device (GPU) - synchronous
#[cfg(cuda)]
pub fn copy_h2d<T: Copy>(device: &DevicePtr<T>, host: &[T]) -> Result<(), ()> {
    if host.len() != device.size {
        eprintln!(
            "[CUDA] Size mismatch in H2D copy: host={}, device={}",
            host.len(),
            device.size
        );
        return Err(());
    }

    let size_bytes = host.len().checked_mul(std::mem::size_of::<T>()).ok_or(())?;
    unsafe {
        let result = cuMemcpyHtoD(device.ptr, host.as_ptr().cast::<c_void>(), size_bytes);
        if result != CUDA_SUCCESS {
            eprintln!("[CUDA] cuMemcpyHtoD failed: {}", result);
            return Err(());
        }
    }
    Ok(())
}

/// Copy data from device (GPU) to host (CPU) - synchronous
#[cfg(cuda)]
pub fn copy_d2h<T: Copy>(host: &mut [T], device: &DevicePtr<T>) -> Result<(), ()> {
    if host.len() != device.size {
        eprintln!(
            "[CUDA] Size mismatch in D2H copy: host={}, device={}",
            host.len(),
            device.size
        );
        return Err(());
    }

    let size_bytes = host.len().checked_mul(std::mem::size_of::<T>()).ok_or(())?;
    unsafe {
        let result = cuMemcpyDtoH(host.as_mut_ptr().cast::<c_void>(), device.ptr, size_bytes);
        if result != CUDA_SUCCESS {
            eprintln!("[CUDA] cuMemcpyDtoH failed: {}", result);
            return Err(());
        }
    }
    Ok(())
}

/// Copy data from device (GPU) to host (CPU) using raw pointers - synchronous
/// Does NOT free the GPU memory (caller manages lifetime)
#[cfg(cuda)]
pub unsafe fn copy_d2h_raw<T: Copy>(
    host: *mut T,
    device_ptr: usize,
    count: usize,
) -> Result<(), ()> {
    let size_bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or(())?;
    let result = cuMemcpyDtoH(host.cast::<c_void>(), device_ptr, size_bytes);
    if result != CUDA_SUCCESS {
        eprintln!("[CUDA] cuMemcpyDtoH (raw) failed: {}", result);
        return Err(());
    }
    Ok(())
}

// =============================================================================
// Stub implementations for non-CUDA builds
// =============================================================================

#[cfg(not(cuda))]
pub struct DevicePtr<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(cuda))]
impl<T> DevicePtr<T> {
    pub fn len(&self) -> usize {
        0
    }
    pub fn as_raw(&self) -> usize {
        0
    }
}

#[cfg(not(cuda))]
pub fn alloc<T>(_count: usize) -> Result<DevicePtr<T>, ()> {
    Err(())
}

#[cfg(not(cuda))]
pub fn free<T>(_device: &DevicePtr<T>) -> Result<(), ()> {
    Err(())
}

#[cfg(not(cuda))]
pub fn copy_h2d<T: Copy>(_device: &DevicePtr<T>, _host: &[T]) -> Result<(), ()> {
    Err(())
}

#[cfg(not(cuda))]
pub fn copy_d2h<T: Copy>(_host: &mut [T], _device: &DevicePtr<T>) -> Result<(), ()> {
    Err(())
}
