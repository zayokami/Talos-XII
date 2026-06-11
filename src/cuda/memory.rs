//! CUDA memory management
//!
//! Provides GPU memory allocation, deallocation, and CPU<->GPU data transfer.
#![allow(dead_code)]

#[cfg(cuda)]
use crate::cuda::bindings::{
    cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost,
    cudaMemcpyHostToDevice, CUDA_SUCCESS,
};
use crate::cuda::error::{CudaError, CudaResult};
#[cfg(cuda)]
use std::ffi::c_void;
#[cfg(cuda)]
use std::os::raw::c_int;
use std::sync::Mutex;

/// Size-keyed GPU memory pool for reusing temporary allocations.
/// Keys are buffer size in bytes; values are lists of raw device pointers.
///
/// Correctness invariants:
/// - Buffers handed out by `alloc_pooled` contain stale data from their
///   previous use. Callers must fully overwrite them (GEMM with beta=0,
///   full-range kernels, `copy_h2d`/`copy_d2d`) or `fill` them first.
/// - Recycling is safe without an explicit device sync only because every
///   kernel launch, cuBLAS call and memcpy in this crate runs on the legacy
///   default stream, so writes into a recycled buffer are stream-ordered
///   after any earlier reads of it. If non-default streams are ever
///   introduced, pool recycling must synchronize on buffer return.
#[cfg(cuda)]
static GPU_BUFFER_POOL: std::sync::LazyLock<Mutex<std::collections::HashMap<usize, Vec<usize>>>> =
    std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

const MAX_POOL_ENTRIES_PER_SIZE: usize = 8;

/// Opaque CUDA memory pointer wrapper
#[derive(Clone)]
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
                let result = cudaFree(self.ptr as *mut c_void);
                if result != CUDA_SUCCESS as c_int {
                    eprintln!("[CUDA] cudaFree failed during drop: {}", result);
                }
            }
        }
    }
}

/// Type-erased GPU buffer for mixed-dtype cache storage.
#[derive(Clone)]
pub enum CudaBuffer {
    BF16(DevicePtr<crate::dtype::bf16>),
    I8(DevicePtr<i8>),
    F32(DevicePtr<f32>),
    F64(DevicePtr<f64>),
}

impl CudaBuffer {
    pub fn len(&self) -> usize {
        match self {
            CudaBuffer::BF16(b) => b.len(),
            CudaBuffer::I8(b) => b.len(),
            CudaBuffer::F32(b) => b.len(),
            CudaBuffer::F64(b) => b.len(),
        }
    }

    pub fn as_raw(&self) -> usize {
        match self {
            CudaBuffer::BF16(b) => b.as_raw(),
            CudaBuffer::I8(b) => b.as_raw(),
            CudaBuffer::F32(b) => b.as_raw(),
            CudaBuffer::F64(b) => b.as_raw(),
        }
    }

    pub fn dtype(&self) -> crate::dtype::Dtype {
        match self {
            CudaBuffer::BF16(_) => crate::dtype::Dtype::BF16,
            CudaBuffer::I8(_) => crate::dtype::Dtype::I8,
            CudaBuffer::F32(_) => crate::dtype::Dtype::F32,
            CudaBuffer::F64(_) => crate::dtype::Dtype::F64,
        }
    }

    pub fn as_bf16(&self) -> Option<&DevicePtr<crate::dtype::bf16>> {
        match self {
            CudaBuffer::BF16(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_i8(&self) -> Option<&DevicePtr<i8>> {
        match self {
            CudaBuffer::I8(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&DevicePtr<f32>> {
        match self {
            CudaBuffer::F32(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&DevicePtr<f64>> {
        match self {
            CudaBuffer::F64(p) => Some(p),
            _ => None,
        }
    }
}

/// Allocate GPU memory for `count` elements of type T
#[cfg(cuda)]
pub fn alloc<T>(count: usize) -> CudaResult<DevicePtr<T>> {
    crate::cuda::init()?;
    if count == 0 {
        return Err(CudaError::InvalidInput {
            op: "cudaMalloc",
            message: "count must be greater than zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMalloc",
            count,
            elem_size,
        })?;
    let mut ptr: *mut c_void = std::ptr::null_mut();

    unsafe {
        let result = cudaMalloc(&mut ptr, size_bytes);
        if result != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMalloc",
                code: result as u32,
            });
        }
    }

    Ok(DevicePtr {
        ptr: ptr as usize,
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
            op: "cudaMalloc",
            message: "count must be greater than zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMalloc",
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
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        let result = cudaMalloc(&mut ptr, size_bytes);
        if result != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMalloc",
                code: result as u32,
            });
        }
    }

    Ok(DevicePtr {
        ptr: ptr as usize,
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

// TODO(perf): transfers use synchronous `cudaMemcpy` on the legacy default
// stream. Pinned staging + `cudaMemcpyAsync` on per-thread streams would
// allow compute/transfer overlap, but requires per-buffer stream/event
// tracking before the buffer pool and the cross-thread tensor caches can be
// recycled safely (see the pool invariants above). The typical transfers in
// this workload are tiny (latency-bound, not bandwidth-bound), so the
// expected gain does not currently justify that complexity.

// TODO(perf): pinned staging + async transfers.
// H2D/D2H below use synchronous `cudaMemcpy` on pageable host memory. A
// faster design is a pool of pinned staging buffers (`cudaMallocHost`, FFI
// already declared in bindings.rs) + `cudaMemcpyAsync` on a per-thread
// non-default stream with `cublasSetStream` bound to the same stream.
// This was deliberately NOT enabled yet: tensor/grad GPU buffers are cached
// in process-global maps (autograd/cuda_bridge.rs) and the size-bucketed
// GPU_BUFFER_POOL is shared across rayon threads. Today every operation is
// ordered by the legacy default stream, which is what makes cross-thread
// buffer reuse safe. Introducing per-thread streams without per-buffer event
// tracking would allow a pooled buffer released on stream A to be rewritten
// on stream B while A's kernel is still reading it. Correctness first.

/// Copy data from host (CPU) to device (GPU) - synchronous
#[cfg(cuda)]
pub fn copy_h2d<T: Copy>(device: &DevicePtr<T>, host: &[T]) -> CudaResult<()> {
    crate::cuda::init()?;
    if host.len() != device.size {
        return Err(CudaError::SizeMismatch {
            op: "cudaMemcpy(H2D)",
            expected: device.size,
            actual: host.len(),
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = host
        .len()
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(H2D)",
            count: host.len(),
            elem_size,
        })?;
    unsafe {
        let result = cudaMemcpy(
            device.ptr as *mut c_void,
            host.as_ptr().cast::<c_void>(),
            size_bytes,
            cudaMemcpyHostToDevice,
        );
        if result != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMemcpy(H2D)",
                code: result as u32,
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
            op: "cudaMemcpy(D2H)",
            expected: device.size,
            actual: host.len(),
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = host
        .len()
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(D2H)",
            count: host.len(),
            elem_size,
        })?;
    unsafe {
        let result = cudaMemcpy(
            host.as_mut_ptr().cast::<c_void>(),
            device.ptr as *const c_void,
            size_bytes,
            cudaMemcpyDeviceToHost,
        );
        if result != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMemcpy(D2H)",
                code: result as u32,
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
            op: "cudaMemcpy(D2D)",
            expected: dst.size,
            actual: src.size,
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = dst
        .size
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(D2D)",
            count: dst.size,
            elem_size,
        })?;
    unsafe {
        let result = cudaMemcpy(
            dst.ptr as *mut c_void,
            src.ptr as *const c_void,
            size_bytes,
            cudaMemcpyDeviceToDevice,
        );
        if result != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMemcpy(D2D)",
                code: result as u32,
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
            op: "cudaMemcpy(D2H)",
            message: "host pointer must not be null",
        });
    }
    if device_ptr == 0 {
        return Err(CudaError::InvalidInput {
            op: "cudaMemcpy(D2H)",
            message: "device pointer must not be zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(D2H)",
            count,
            elem_size,
        })?;
    let result = cudaMemcpy(
        host.cast::<c_void>(),
        device_ptr as *const c_void,
        size_bytes,
        cudaMemcpyDeviceToHost,
    );
    if result != CUDA_SUCCESS as c_int {
        return Err(CudaError::Runtime {
            op: "cudaMemcpy(D2H)",
            code: result as u32,
        });
    }
    Ok(())
}

// =============================================================================
// Stub implementations for non-CUDA builds
// =============================================================================

#[cfg(not(cuda))]
#[derive(Clone)]
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
