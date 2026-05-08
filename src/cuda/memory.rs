//! CUDA memory management
//!
//! Provides GPU memory allocation, deallocation, and CPU<->GPU data transfer.
#![allow(dead_code)]

#[cfg(cuda)]
use crate::cuda::bindings::{
    cuMemAlloc, cuMemFree, cuMemcpyDtoD, cuMemcpyDtoH, cuMemcpyHtoD, CUDA_SUCCESS,
};
use crate::cuda::error::{CudaError, CudaResult};
#[cfg(cuda)]
use std::ffi::c_void;
use std::sync::Mutex;

/// Size-keyed GPU memory pool for reusing temporary allocations.
/// Keys are buffer size in bytes; values are lists of raw device pointers.
#[cfg(cuda)]
static GPU_BUFFER_POOL: std::sync::LazyLock<Mutex<std::collections::HashMap<usize, Vec<usize>>>> =
    std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

const MAX_POOL_ENTRIES_PER_SIZE: usize = 8;

/// Opaque CUDA memory pointer wrapper
pub struct DevicePtr<T> {
    ptr: usize,
    size: usize,
    pooled: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> DevicePtr<T> {
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn as_raw(&self) -> usize {
        self.ptr
    }

    /// Create a zero-sized device pointer (no GPU allocation)
    pub fn zero_sized() -> Self {
        DevicePtr {
            ptr: 0,
            size: 0,
            pooled: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        #[cfg(cuda)]
        if self.ptr != 0 {
            if self.pooled {
                let size_bytes = self.size * std::mem::size_of::<T>();
                if let Ok(mut pool) = GPU_BUFFER_POOL.lock() {
                    let entry = pool.entry(size_bytes).or_default();
                    if entry.len() < MAX_POOL_ENTRIES_PER_SIZE {
                        entry.push(self.ptr);
                        self.ptr = 0;
                        return;
                    }
                }
            }
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
    crate::cuda::init()?;
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
        pooled: false,
        _phantom: std::marker::PhantomData,
    })
}

/// Allocate GPU memory from pool if available, otherwise allocate fresh.
/// The returned buffer returns to the pool on Drop instead of being freed.
#[cfg(cuda)]
pub fn alloc_pooled<T>(count: usize) -> CudaResult<DevicePtr<T>> {
    crate::cuda::init()?;
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

    // Try pool first
    if let Ok(mut pool) = GPU_BUFFER_POOL.lock() {
        if let Some(vec) = pool.get_mut(&size_bytes) {
            if let Some(ptr) = vec.pop() {
                return Ok(DevicePtr {
                    ptr,
                    size: count,
                    pooled: true,
                    _phantom: std::marker::PhantomData,
                });
            }
        }
    }

    // Fall back to fresh allocation
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
        pooled: true,
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
    crate::cuda::init()?;
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
    crate::cuda::init()?;
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

/// Copy data from device (GPU) to device (GPU) - synchronous
#[cfg(cuda)]
pub fn copy_d2d<T: Copy>(dst: &DevicePtr<T>, src: &DevicePtr<T>) -> CudaResult<()> {
    crate::cuda::init()?;
    if dst.size != src.size {
        return Err(CudaError::SizeMismatch {
            op: "cuMemcpyDtoD",
            expected: dst.size,
            actual: src.size,
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = dst
        .size
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cuMemcpyDtoD",
            count: dst.size,
            elem_size,
        })?;
    unsafe {
        let result = cuMemcpyDtoD(dst.ptr, src.ptr, size_bytes);
        if result != CUDA_SUCCESS {
            return Err(CudaError::Runtime {
                op: "cuMemcpyDtoD",
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
    pub fn zero_sized() -> Self {
        DevicePtr {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(not(cuda))]
pub fn alloc<T>(_count: usize) -> CudaResult<DevicePtr<T>> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::alloc",
    })
}

#[cfg(not(cuda))]
pub fn alloc_pooled<T>(_count: usize) -> CudaResult<DevicePtr<T>> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::alloc_pooled",
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

#[cfg(not(cuda))]
pub fn copy_d2d<T: Copy>(_dst: &DevicePtr<T>, _src: &DevicePtr<T>) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::copy_d2d",
    })
}
