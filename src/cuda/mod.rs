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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(cuda)]
use std::{ffi::c_char, ffi::CStr};

/// Indicates whether CUDA is available and initialized
static CUDA_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Raw pointer to the primary CUDA context (stored as usize for Sync).
#[cfg(cuda)]
static CUDA_CONTEXT_PTR: AtomicUsize = AtomicUsize::new(0);

#[cfg(cuda)]
fn get_cuda_context() -> Option<bindings::CUcontext> {
    let ptr = CUDA_CONTEXT_PTR.load(Ordering::SeqCst);
    if ptr == 0 {
        None
    } else {
        Some(ptr as bindings::CUcontext)
    }
}
static CUDA_MATMUL_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_INIT: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_ALLOC: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_COPY: AtomicU64 = AtomicU64::new(0);
static CUDA_MATMUL_FALLBACK_GEMM: AtomicU64 = AtomicU64::new(0);
static CUDA_ACTIVATION_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static CUDA_ACTIVATION_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_ACTIVATION_FALLBACK_ALLOC: AtomicU64 = AtomicU64::new(0);
static CUDA_ACTIVATION_FALLBACK_COPY: AtomicU64 = AtomicU64::new(0);
static CUDA_ACTIVATION_FALLBACK_KERNEL: AtomicU64 = AtomicU64::new(0);
static CUDA_LOGSOFTMAX_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static CUDA_LOGSOFTMAX_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_LOGSOFTMAX_FALLBACK_ALLOC: AtomicU64 = AtomicU64::new(0);
static CUDA_LOGSOFTMAX_FALLBACK_COPY: AtomicU64 = AtomicU64::new(0);
static CUDA_LOGSOFTMAX_FALLBACK_KERNEL: AtomicU64 = AtomicU64::new(0);
static CUDA_BACKWARD_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static CUDA_BACKWARD_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_BACKWARD_FALLBACK_KERNEL: AtomicU64 = AtomicU64::new(0);
static CUDA_OPTIMIZER_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static CUDA_OPTIMIZER_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static CUDA_OPTIMIZER_FALLBACK_PARAM: AtomicU64 = AtomicU64::new(0);

/// Runtime observability counters for CUDA matmul routing.
#[derive(Debug, Clone, Copy)]
pub struct CudaRuntimeStats {
    pub matmul_attempts: u64,
    pub matmul_successes: u64,
    pub matmul_fallback_init: u64,
    pub matmul_fallback_alloc: u64,
    pub matmul_fallback_copy: u64,
    pub matmul_fallback_gemm: u64,
    pub activation_attempts: u64,
    pub activation_successes: u64,
    pub activation_fallback_alloc: u64,
    pub activation_fallback_copy: u64,
    pub activation_fallback_kernel: u64,
    pub log_softmax_attempts: u64,
    pub log_softmax_successes: u64,
    pub log_softmax_fallback_alloc: u64,
    pub log_softmax_fallback_copy: u64,
    pub log_softmax_fallback_kernel: u64,
    pub backward_attempts: u64,
    pub backward_successes: u64,
    pub backward_fallback_kernel: u64,
    pub optimizer_attempts: u64,
    pub optimizer_successes: u64,
    pub optimizer_fallback_param: u64,
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

pub fn record_activation_attempt() {
    CUDA_ACTIVATION_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_activation_success() {
    CUDA_ACTIVATION_SUCCESSES.fetch_add(1, Ordering::Relaxed);
}

pub fn record_activation_fallback(stage: &'static str) {
    match stage {
        "alloc" => {
            CUDA_ACTIVATION_FALLBACK_ALLOC.fetch_add(1, Ordering::Relaxed);
        }
        "copy" => {
            CUDA_ACTIVATION_FALLBACK_COPY.fetch_add(1, Ordering::Relaxed);
        }
        "kernel" => {
            CUDA_ACTIVATION_FALLBACK_KERNEL.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

pub fn record_log_softmax_attempt() {
    CUDA_LOGSOFTMAX_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_log_softmax_success() {
    CUDA_LOGSOFTMAX_SUCCESSES.fetch_add(1, Ordering::Relaxed);
}

pub fn record_log_softmax_fallback(stage: &'static str) {
    match stage {
        "alloc" => {
            CUDA_LOGSOFTMAX_FALLBACK_ALLOC.fetch_add(1, Ordering::Relaxed);
        }
        "copy" => {
            CUDA_LOGSOFTMAX_FALLBACK_COPY.fetch_add(1, Ordering::Relaxed);
        }
        "kernel" => {
            CUDA_LOGSOFTMAX_FALLBACK_KERNEL.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

pub fn record_backward_attempt() {
    CUDA_BACKWARD_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_backward_success() {
    CUDA_BACKWARD_SUCCESSES.fetch_add(1, Ordering::Relaxed);
}

pub fn record_backward_fallback() {
    CUDA_BACKWARD_FALLBACK_KERNEL.fetch_add(1, Ordering::Relaxed);
}

pub fn record_optimizer_attempt() {
    CUDA_OPTIMIZER_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_optimizer_success() {
    CUDA_OPTIMIZER_SUCCESSES.fetch_add(1, Ordering::Relaxed);
}

pub fn record_optimizer_fallback() {
    CUDA_OPTIMIZER_FALLBACK_PARAM.fetch_add(1, Ordering::Relaxed);
}

pub fn runtime_stats() -> CudaRuntimeStats {
    CudaRuntimeStats {
        matmul_attempts: CUDA_MATMUL_ATTEMPTS.load(Ordering::Relaxed),
        matmul_successes: CUDA_MATMUL_SUCCESSES.load(Ordering::Relaxed),
        matmul_fallback_init: CUDA_MATMUL_FALLBACK_INIT.load(Ordering::Relaxed),
        matmul_fallback_alloc: CUDA_MATMUL_FALLBACK_ALLOC.load(Ordering::Relaxed),
        matmul_fallback_copy: CUDA_MATMUL_FALLBACK_COPY.load(Ordering::Relaxed),
        matmul_fallback_gemm: CUDA_MATMUL_FALLBACK_GEMM.load(Ordering::Relaxed),
        activation_attempts: CUDA_ACTIVATION_ATTEMPTS.load(Ordering::Relaxed),
        activation_successes: CUDA_ACTIVATION_SUCCESSES.load(Ordering::Relaxed),
        activation_fallback_alloc: CUDA_ACTIVATION_FALLBACK_ALLOC.load(Ordering::Relaxed),
        activation_fallback_copy: CUDA_ACTIVATION_FALLBACK_COPY.load(Ordering::Relaxed),
        activation_fallback_kernel: CUDA_ACTIVATION_FALLBACK_KERNEL.load(Ordering::Relaxed),
        log_softmax_attempts: CUDA_LOGSOFTMAX_ATTEMPTS.load(Ordering::Relaxed),
        log_softmax_successes: CUDA_LOGSOFTMAX_SUCCESSES.load(Ordering::Relaxed),
        log_softmax_fallback_alloc: CUDA_LOGSOFTMAX_FALLBACK_ALLOC.load(Ordering::Relaxed),
        log_softmax_fallback_copy: CUDA_LOGSOFTMAX_FALLBACK_COPY.load(Ordering::Relaxed),
        log_softmax_fallback_kernel: CUDA_LOGSOFTMAX_FALLBACK_KERNEL.load(Ordering::Relaxed),
        backward_attempts: CUDA_BACKWARD_ATTEMPTS.load(Ordering::Relaxed),
        backward_successes: CUDA_BACKWARD_SUCCESSES.load(Ordering::Relaxed),
        backward_fallback_kernel: CUDA_BACKWARD_FALLBACK_KERNEL.load(Ordering::Relaxed),
        optimizer_attempts: CUDA_OPTIMIZER_ATTEMPTS.load(Ordering::Relaxed),
        optimizer_successes: CUDA_OPTIMIZER_SUCCESSES.load(Ordering::Relaxed),
        optimizer_fallback_param: CUDA_OPTIMIZER_FALLBACK_PARAM.load(Ordering::Relaxed),
    }
}

/// Device information
pub struct CudaDevice {
    pub id: usize,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
}

/// Initialize CUDA runtime and create a primary context.
/// Must be called (directly or transitively) before any driver-API operation.
/// Thread-safe: context is created once; on subsequent calls the same context
/// is pushed onto the current thread.
#[cfg(cuda)]
pub fn init() -> CudaResult<()> {
    if CUDA_INITIALIZED.load(Ordering::SeqCst) {
        // Context already created — ensure it is current on this thread
        // (CUDA contexts are thread-local).
        if let Some(ctx) = get_cuda_context() {
            unsafe {
                bindings::cuCtxSetCurrent(ctx);
            }
        }
        return Ok(());
    }

    unsafe {
        // 1. Initialize driver API (needed for cuCtxGetCurrent / cuCtxSetCurrent)
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

        // 2. Use CUDA Runtime API to create the primary context.
        //    This is the context shared by cuBLAS and our nvcc-compiled kernels.
        let rt_err = bindings::cudaSetDevice(0);
        if rt_err != 0 {
            return Err(CudaError::Runtime {
                op: "cudaSetDevice",
                code: rt_err as u32,
            });
        }
        // Force primary context creation with a dummy allocation
        let mut dummy: *mut std::ffi::c_void = std::ptr::null_mut();
        let rt_err = bindings::cudaMalloc(&mut dummy, 1);
        if rt_err != 0 {
            return Err(CudaError::Runtime {
                op: "cudaMalloc(dummy)",
                code: rt_err as u32,
            });
        }
        let _ = bindings::cudaFree(dummy);

        // 3. Retrieve the Runtime-created primary context via Driver API
        let mut ctx: bindings::CUcontext = std::ptr::null_mut();
        let get_result = bindings::cuCtxGetCurrent(&mut ctx);
        if get_result != bindings::CUDA_SUCCESS || ctx.is_null() {
            return Err(CudaError::Runtime {
                op: "cuCtxGetCurrent",
                code: get_result,
            });
        }

        CUDA_CONTEXT_PTR.store(ctx as usize, Ordering::SeqCst);

        // 4. cuBLAS warmup: verify cuBLAS can initialize in this context.
        //    If this fails, the user is missing cuBLAS dependencies (e.g.
        //    nvfatbin.dll, zlibwapi.dll) even if cublas64_12.dll is present.
        {
            let mut handle: crate::cuda::bindings::cublasHandle_t = std::ptr::null_mut();
            let cublas_result = crate::cuda::bindings::cublasCreate_v2(&mut handle);
            if cublas_result != 0 {
                eprintln!(
                    "[CUDA] CRITICAL: cuBLAS initialization test failed (code={}).",
                    cublas_result
                );
                eprintln!("[CUDA] This usually means a missing cuBLAS dependency DLL.");
                eprintln!("[CUDA] Ensure the following DLLs are next to talos_xii.exe:");
                eprintln!("  - cudart64_12.dll");
                eprintln!("  - cublas64_12.dll");
                eprintln!("  - cublasLt64_12.dll");
                eprintln!("[CUDA] Also check for hidden dependencies:");
                eprintln!("  - nvfatbin.dll   (in CUDA bin directory)");
                eprintln!("  - zlibwapi.dll   (in CUDA bin directory)");
                eprintln!("[CUDA] You can find these in your CUDA installation under:");
                eprintln!("  C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/bin/");
                return Err(CudaError::Blas {
                    op: "cublasCreate_v2 (warmup)",
                    code: cublas_result,
                });
            }
            let _ = crate::cuda::bindings::cublasDestroy_v2(handle);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_counters_are_monotonic() {
        let before = runtime_stats();
        record_activation_attempt();
        record_activation_fallback("kernel");
        record_activation_success();
        let after = runtime_stats();

        assert!(after.activation_attempts > before.activation_attempts);
        assert!(after.activation_fallback_kernel > before.activation_fallback_kernel);
        assert!(after.activation_successes > before.activation_successes);
    }

    #[test]
    fn log_softmax_counters_are_monotonic() {
        let before = runtime_stats();
        record_log_softmax_attempt();
        record_log_softmax_fallback("copy");
        record_log_softmax_success();
        let after = runtime_stats();

        assert!(after.log_softmax_attempts > before.log_softmax_attempts);
        assert!(after.log_softmax_fallback_copy > before.log_softmax_fallback_copy);
        assert!(after.log_softmax_successes > before.log_softmax_successes);
    }
}
