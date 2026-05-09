//! CUDA FFI bindings
//!
//! Direct FFI declarations for CUDA runtime and cuBLAS APIs.
//! These bindings are based on the CUDA driver API.
#![allow(dead_code, non_camel_case_types, non_snake_case)]

use std::os::raw::{c_char, c_int};

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
pub const CUBLAS_OP_N: cublasOperation_t = 0; // non-transpose
pub const CUBLAS_OP_T: cublasOperation_t = 1; // transpose
pub const CUBLAS_OP_C: cublasOperation_t = 2; // conjugate transpose

// =============================================================================
// CUDA Runtime API (libcudart.so / cudart.dll)
// =============================================================================

// Runtime memory copy directions
#[allow(non_upper_case_globals)]
pub const cudaMemcpyHostToHost: c_int = 0;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyHostToDevice: c_int = 1;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToHost: c_int = 2;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToDevice: c_int = 3;

extern "C" {
    pub fn cudaSetDevice(device: c_int) -> c_int;
    pub fn cudaGetLastError() -> c_int;
    pub fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    pub fn cudaFree(devPtr: *mut std::ffi::c_void) -> c_int;
    pub fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
}

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

    // Primary context (shared with CUDA Runtime / cuBLAS)
    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;

    // Context management
    pub fn cuCtxCreate(ctx: *mut CUcontext, flags: u32, device: CUdevice) -> CUresult;
    pub fn cuCtxDestroy(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxPushCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxPopCurrent(pctx: *mut CUcontext) -> CUresult;

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
// Custom CUDA kernels (compiled from cuda/*.cu)
// =============================================================================

extern "C" {
    #[link_name = "relu"]
    pub fn cuda_relu(h_data: *mut f64, size: c_int, d_data: *mut c_int) -> c_int;

    #[link_name = "gelu"]
    pub fn cuda_gelu(h_data: *mut f64, size: c_int, d_data: *mut c_int) -> c_int;

    #[link_name = "softmax"]
    pub fn cuda_softmax(data: *mut f64, rows: c_int, cols: c_int, d_data: *mut c_int) -> c_int;

    #[link_name = "softmax_causal"]
    pub fn cuda_softmax_causal(
        data: *mut f64,
        rows: c_int,
        cols: c_int,
        d_data: *mut c_int,
    ) -> c_int;

    #[link_name = "softmax_small_batch"]
    pub fn cuda_softmax_small_batch(
        data: *mut f64,
        rows: c_int,
        cols: c_int,
        d_data: *mut c_int,
    ) -> c_int;

    #[link_name = "log_softmax"]
    pub fn cuda_log_softmax(
        h_logits: *const f64,
        h_out: *mut f64,
        rows: c_int,
        cols: c_int,
        d_logits: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "cuda_rope"]
    pub fn cuda_rope(
        data: *mut f64,
        cos_cache: *const f64,
        sin_cache: *const f64,
        seq_len: c_int,
        dim: c_int,
        total_batches: c_int,
        start_pos: c_int,
        d_data: *mut c_int,
        d_cos_cache: *mut c_int,
        d_sin_cache: *mut c_int,
    ) -> c_int;

    #[link_name = "attention_weighted_sum"]
    pub fn cuda_attention_weighted_sum(
        h_attn: *mut f64,
        h_values: *mut f64,
        h_output: *mut f64,
        rows: c_int,
        cols: c_int,
        head_dim: c_int,
        d_attn: *mut c_int,
        d_values: *mut c_int,
        d_output: *mut c_int,
    ) -> c_int;

    // Backward kernels
    #[link_name = "relu_backward"]
    pub fn cuda_relu_backward(
        h_input: *const f64,
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        size: c_int,
        d_input: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "gelu_backward"]
    pub fn cuda_gelu_backward(
        h_input: *const f64,
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        size: c_int,
        d_input: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "add_backward"]
    pub fn cuda_add_backward(
        h_grad_out: *const f64,
        h_a_grad: *mut f64,
        h_b_grad: *mut f64,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "mul_backward"]
    pub fn cuda_mul_backward(
        h_grad_out: *const f64,
        h_a_data: *const f64,
        h_b_data: *const f64,
        h_a_grad: *mut f64,
        h_b_grad: *mut f64,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_data: *mut c_int,
        d_b_data: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "acc_buffer"]
    pub fn cuda_acc_buffer(
        h_dst: *mut f64,
        h_src: *const f64,
        size: c_int,
        d_dst: *mut c_int,
        d_src: *mut c_int,
    ) -> c_int;

    // Element-wise forward kernels
    #[link_name = "add_forward"]
    pub fn cuda_add_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "sub_forward"]
    pub fn cuda_sub_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "mul_forward"]
    pub fn cuda_mul_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "div_forward"]
    pub fn cuda_div_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "adam_step"]
    pub fn cuda_adam_step(
        h_params: *mut f64,
        h_grads: *const f64,
        h_m: *mut f64,
        h_v: *mut f64,
        size: c_int,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        bias_correction1: f64,
        bias_correction2: f64,
        clip_coef: f64,
        d_params: *mut c_int,
        d_grads: *mut c_int,
        d_m: *mut c_int,
        d_v: *mut c_int,
    ) -> c_int;

    #[link_name = "rmsnorm_forward"]
    pub fn cuda_rmsnorm_forward(
        h_x: *const f64,
        h_weight: *const f64,
        h_out: *mut f64,
        dim: c_int,
        eps: f64,
        num_rows: c_int,
        d_x: *mut c_int,
        d_weight: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "rmsnorm_backward"]
    pub fn cuda_rmsnorm_backward(
        h_grad_out: *const f64,
        h_x: *const f64,
        h_weight: *const f64,
        h_x_grad: *mut f64,
        h_w_grad: *mut f64,
        dim: c_int,
        eps: f64,
        num_rows: c_int,
        d_grad_out: *mut c_int,
        d_x: *mut c_int,
        d_weight: *mut c_int,
        d_x_grad: *mut c_int,
        d_w_grad: *mut c_int,
    ) -> c_int;
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
