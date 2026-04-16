//! CUDA FFI bindings
//!
//! Direct FFI declarations for CUDA runtime and cuBLAS APIs.
//! These bindings are based on the CUDA driver API.
#![allow(dead_code, non_camel_case_types, non_snake_case)]

use std::os::raw::c_char;

// =============================================================================
// CUDA Types
// =============================================================================

pub type CUresult = u32;
pub type CUdevice = i32;
pub type CUstream = *mut std::ffi::c_void;
pub type CUdeviceptr = usize;
pub type CUcontext = *mut std::ffi::c_void;
pub type cublasHandle_t = *mut std::ffi::c_void;
pub type cublasStatus_t = i32;
pub type cublasOperation_t = u32;

// =============================================================================
// CUDA Constants
// =============================================================================

pub const CUDA_SUCCESS: CUresult = 0;

// cublasStatus_t
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

// cublasOperation_t
pub const CUBLAS_OP_N: cublasOperation_t = 111; // 'N' or 'n'
pub const CUBLAS_OP_T: cublasOperation_t = 114; // 'T' or 't'
pub const CUBLAS_OP_C: cublasOperation_t = 99; // 'C' or 'c'

// =============================================================================
// CUDA Driver API (libcuda.so / cuda.dll)
// =============================================================================

extern "C" {
    // Initialization
    pub fn cuInit(flags: u32) -> CUresult;

    // Device management
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: i32, device: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem(bytes: *mut usize, device: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(
        pi: *mut i32,
        attrib: CUdevice_attribute,
        device: CUdevice,
    ) -> CUresult;

    // Context management
    pub fn cuCtxCreate(ctx: *mut CUcontext, flags: u32, device: CUdevice) -> CUresult;
    pub fn cuCtxDestroy(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;

    // Memory management
    pub fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD(
        dst: CUdeviceptr,
        src: *const std::ffi::c_void,
        bytesize: usize,
    ) -> CUresult;
    pub fn cuMemcpyDtoH(dst: *mut std::ffi::c_void, src: CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemcpyDtoD(dst: CUdeviceptr, src: CUdeviceptr, bytesize: usize) -> CUresult;

    // Stream management
    pub fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult;
    pub fn cuStreamDestroy(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
}

// =============================================================================
// cuBLAS API (libcublas.so)
// =============================================================================

extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) -> cublasStatus_t;

    // SGEMM (single precision)
    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: *const f32,
        C: *mut f32,
        ldc: i32,
    ) -> cublasStatus_t;

    // DGEMM (double precision)
    pub fn cublasDgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f64,
        A: *const f64,
        lda: i32,
        B: *const f64,
        ldb: i32,
        beta: *const f64,
        C: *mut f64,
        ldc: i32,
    ) -> cublasStatus_t;
}

// =============================================================================
// Device Attributes (for cuDeviceGetAttribute)
// =============================================================================

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
}
