//! CUDA memory management
//!
//! Provides GPU memory allocation, deallocation, and CPU<->GPU data transfer.
#![allow(dead_code)]

#[cfg(cuda)]
use crate::cuda::bindings::{cuMemAlloc, cuMemFree, cuMemcpyDtoH, cuMemcpyHtoD, CUDA_SUCCESS};
use crate::cuda::error::{CudaError, CudaResult};
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
        #[cfg(cuda)]
        if self.ptr != 0 {
            unsafe {
                let result = cuMemFree(self.ptr);
                if result != CUDA_SUCCESS {
                    eprintln!("[CUDA] cuMemFree failed during drop: {}", result);
                }
            }
        }
    }
}

/// Allocate GPU memory for `count` elements of type T
#[cfg(cuda)]
pub fn alloc<T>(count: usize) -> CudaResult<DevicePtr<T>> {
    if count == 0 {
        return Err(CudaError::InvalidInput {
            op: "cuMemAlloc",
            message: "count must be greater than zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cuMemAlloc",
            count,
            elem_size,
        })?;
    let mut ptr: usize = 0;

    unsafe {
        let result = cuMemAlloc(&mut ptr, size_bytes);
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuMemAlloc",
                code: result,
            });
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
pub fn free<T>(_device: &DevicePtr<T>) -> CudaResult<()> {
    Ok(())
}

/// Copy data from host (CPU) to device (GPU) - synchronous
#[cfg(cuda)]
pub fn copy_h2d<T: Copy>(device: &DevicePtr<T>, host: &[T]) -> CudaResult<()> {
    if host.len() != device.size {
        return Err(CudaError::SizeMismatch {
            op: "cuMemcpyHtoD",
            expected: device.size,
            actual: host.len(),
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = host
        .len()
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cuMemcpyHtoD",
            count: host.len(),
            elem_size,
        })?;
    unsafe {
        let result = cuMemcpyHtoD(device.ptr, host.as_ptr().cast::<c_void>(), size_bytes);
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuMemcpyHtoD",
                code: result,
            });
        }
    }
    Ok(())
}

/// Copy data from device (GPU) to host (CPU) - synchronous
#[cfg(cuda)]
pub fn copy_d2h<T: Copy>(host: &mut [T], device: &DevicePtr<T>) -> CudaResult<()> {
    if host.len() != device.size {
        return Err(CudaError::SizeMismatch {
            op: "cuMemcpyDtoH",
            expected: device.size,
            actual: host.len(),
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = host
        .len()
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cuMemcpyDtoH",
            count: host.len(),
            elem_size,
        })?;
    unsafe {
        let result = cuMemcpyDtoH(host.as_mut_ptr().cast::<c_void>(), device.ptr, size_bytes);
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuMemcpyDtoH",
                code: result,
            });
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
) -> CudaResult<()> {
    if host.is_null() {
        return Err(CudaError::InvalidInput {
            op: "cuMemcpyDtoH",
            message: "host pointer must not be null",
        });
    }
    if device_ptr == 0 {
        return Err(CudaError::InvalidInput {
            op: "cuMemcpyDtoH",
            message: "device pointer must not be zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cuMemcpyDtoH",
            count,
            elem_size,
        })?;
    let result = cuMemcpyDtoH(host.cast::<c_void>(), device_ptr, size_bytes);
    if result != CUDA_SUCCESS {
        return Err(CudaError::Runtime {
            op: "cuMemcpyDtoH",
            code: result,
        });
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
pub fn alloc<T>(_count: usize) -> CudaResult<DevicePtr<T>> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::alloc",
    })
}

#[cfg(not(cuda))]
pub fn free<T>(_device: &DevicePtr<T>) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::free",
    })
}

#[cfg(not(cuda))]
pub fn copy_h2d<T: Copy>(_device: &DevicePtr<T>, _host: &[T]) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::copy_h2d",
    })
}

#[cfg(not(cuda))]
pub fn copy_d2h<T: Copy>(_host: &mut [T], _device: &DevicePtr<T>) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::copy_d2h",
    })
}
