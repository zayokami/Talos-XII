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
pub type cudaEvent_t = *mut std::ffi::c_void;
pub type CUdeviceptr = usize;
pub type CUcontext = *mut std::ffi::c_void;
pub type cublasHandle_t = *mut std::ffi::c_void;
pub type cublasStatus_t = i32;
pub type cublasOperation_t = u32;
pub type cudaDataType_t = i32;
pub type cublasComputeType_t = i32;
pub type cublasGemmAlgo_t = i32;

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
pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: cublasGemmAlgo_t = 99;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;
pub const CUDA_R_16BF: cudaDataType_t = 14;
pub const CUDA_R_32F: cudaDataType_t = 0;

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

/// cudaError_t returned by `cudaEventQuery` while the recorded work has not
/// completed yet. This is the expected "still running" answer, NOT a failure;
/// callers must treat it as "buffer busy" rather than an error.
#[allow(non_upper_case_globals)]
pub const cudaErrorNotReady: c_int = 600;

/// Flag for `cudaEventCreateWithFlags`: the event carries no timing data.
/// The pinned-staging transfer path only needs completion gating, and
/// timing-free events are cheaper to record and query.
#[allow(non_upper_case_globals)]
pub const cudaEventDisableTiming: u32 = 0x02;

extern "C" {
    pub fn cudaSetDevice(device: c_int) -> c_int;
    pub fn cudaGetLastError() -> c_int;
    pub fn cudaDeviceSynchronize() -> c_int;
    pub fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    pub fn cudaFree(devPtr: *mut std::ffi::c_void) -> c_int;
    pub fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
    // Pinned (page-locked) host memory + async copy, used by the
    // pinned-staging transfer path in cuda/memory.rs. Async copies only
    // overlap when the host buffer is pinned; on pageable memory
    // cudaMemcpyAsync silently degrades to a synchronous copy.
    pub fn cudaMallocHost(ptr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    pub fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> c_int;
    pub fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
        stream: CUstream,
    ) -> c_int;
    // Events for gating pinned-staging buffer reuse (CUDA Runtime API 12.x).
    // `cudaEventQuery` returns cudaSuccess (0) once the recorded work is done
    // and `cudaErrorNotReady` (600) while it is still in flight.
    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: u32) -> c_int;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: CUstream) -> c_int;
    pub fn cudaEventQuery(event: cudaEvent_t) -> c_int;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> c_int;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> c_int;
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

    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const std::ffi::c_void,
        A: *const std::ffi::c_void,
        Atype: cudaDataType_t,
        lda: i32,
        B: *const std::ffi::c_void,
        Btype: cudaDataType_t,
        ldb: i32,
        beta: *const std::ffi::c_void,
        C: *mut std::ffi::c_void,
        Ctype: cudaDataType_t,
        ldc: i32,
        computeType: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t;
}

// =============================================================================
// Custom CUDA kernels (compiled from cuda/*.cu)
// =============================================================================

extern "C" {
    // ReLU / GELU
    #[link_name = "relu_f64"]
    pub fn cuda_relu(h_data: *mut f64, size: c_int, d_data: *mut c_int) -> c_int;
    #[link_name = "relu_f32"]
    pub fn cuda_relu_f32(h_data: *mut f32, size: c_int, d_data: *mut c_int) -> c_int;

    #[link_name = "gelu_f64"]
    pub fn cuda_gelu(h_data: *mut f64, size: c_int, d_data: *mut c_int) -> c_int;
    #[link_name = "gelu_f32"]
    pub fn cuda_gelu_f32(h_data: *mut f32, size: c_int, d_data: *mut c_int) -> c_int;

    // Softmax
    #[link_name = "softmax_f64"]
    pub fn cuda_softmax(data: *mut f64, rows: c_int, cols: c_int, d_data: *mut c_int) -> c_int;
    #[link_name = "softmax_f32"]
    pub fn cuda_softmax_f32(data: *mut f32, rows: c_int, cols: c_int, d_data: *mut c_int) -> c_int;

    #[link_name = "softmax_causal_f64"]
    pub fn cuda_softmax_causal(
        data: *mut f64,
        rows: c_int,
        cols: c_int,
        d_data: *mut c_int,
    ) -> c_int;
    #[link_name = "softmax_causal_f32"]
    pub fn cuda_softmax_causal_f32(
        data: *mut f32,
        rows: c_int,
        cols: c_int,
        d_data: *mut c_int,
    ) -> c_int;

    #[link_name = "softmax_small_batch_f64"]
    pub fn cuda_softmax_small_batch(
        data: *mut f64,
        rows: c_int,
        cols: c_int,
        d_data: *mut c_int,
    ) -> c_int;
    #[link_name = "softmax_small_batch_f32"]
    pub fn cuda_softmax_small_batch_f32(
        data: *mut f32,
        rows: c_int,
        cols: c_int,
        d_data: *mut c_int,
    ) -> c_int;

    #[link_name = "log_softmax_f64"]
    pub fn cuda_log_softmax(
        h_logits: *const f64,
        h_out: *mut f64,
        rows: c_int,
        cols: c_int,
        d_logits: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "log_softmax_f32"]
    pub fn cuda_log_softmax_f32(
        h_logits: *const f32,
        h_out: *mut f32,
        rows: c_int,
        cols: c_int,
        d_logits: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "softmax_backward_f64"]
    pub fn cuda_softmax_backward(
        h_out: *const f64,
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        rows: c_int,
        cols: c_int,
        d_out: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "softmax_backward_f32"]
    pub fn cuda_softmax_backward_f32(
        h_out: *const f32,
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        rows: c_int,
        cols: c_int,
        d_out: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "log_softmax_backward_f64"]
    pub fn cuda_log_softmax_backward(
        h_out: *const f64,
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        rows: c_int,
        cols: c_int,
        d_out: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "log_softmax_backward_f32"]
    pub fn cuda_log_softmax_backward_f32(
        h_out: *const f32,
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        rows: c_int,
        cols: c_int,
        d_out: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    // RoPE
    #[link_name = "cuda_rope_f64"]
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
    #[link_name = "cuda_rope_f32"]
    pub fn cuda_rope_f32(
        data: *mut f32,
        cos_cache: *const f32,
        sin_cache: *const f32,
        seq_len: c_int,
        dim: c_int,
        total_batches: c_int,
        start_pos: c_int,
        d_data: *mut c_int,
        d_cos_cache: *mut c_int,
        d_sin_cache: *mut c_int,
    ) -> c_int;

    #[link_name = "cuda_rope_backward_f64"]
    pub fn cuda_rope_backward(
        h_grad_out: *const f64,
        h_cos_cache: *const f64,
        h_sin_cache: *const f64,
        h_input_grad: *mut f64,
        seq_len: c_int,
        dim: c_int,
        total_batches: c_int,
        start_pos: c_int,
        d_grad_out: *mut c_int,
        d_cos_cache: *mut c_int,
        d_sin_cache: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "cuda_rope_backward_f32"]
    pub fn cuda_rope_backward_f32(
        h_grad_out: *const f32,
        h_cos_cache: *const f32,
        h_sin_cache: *const f32,
        h_input_grad: *mut f32,
        seq_len: c_int,
        dim: c_int,
        total_batches: c_int,
        start_pos: c_int,
        d_grad_out: *mut c_int,
        d_cos_cache: *mut c_int,
        d_sin_cache: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    // Attention weighted sum
    #[link_name = "attention_weighted_sum_f64"]
    pub fn cuda_attention_weighted_sum(
        h_attn: *mut f64,
        h_values: *mut f64,
        h_output: *mut f64,
        batches: c_int,
        seq: c_int,
        head_dim: c_int,
        d_attn: *mut c_int,
        d_values: *mut c_int,
        d_output: *mut c_int,
    ) -> c_int;
    #[link_name = "attention_weighted_sum_f32"]
    pub fn cuda_attention_weighted_sum_f32(
        h_attn: *mut f32,
        h_values: *mut f32,
        h_output: *mut f32,
        batches: c_int,
        seq: c_int,
        head_dim: c_int,
        d_attn: *mut c_int,
        d_values: *mut c_int,
        d_output: *mut c_int,
    ) -> c_int;

    // Backward kernels
    #[link_name = "relu_backward_f64"]
    pub fn cuda_relu_backward(
        h_input: *const f64,
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        size: c_int,
        d_input: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "relu_backward_f32"]
    pub fn cuda_relu_backward_f32(
        h_input: *const f32,
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        size: c_int,
        d_input: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "gelu_backward_f64"]
    pub fn cuda_gelu_backward(
        h_input: *const f64,
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        size: c_int,
        d_input: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "gelu_backward_f32"]
    pub fn cuda_gelu_backward_f32(
        h_input: *const f32,
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        size: c_int,
        d_input: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "add_backward_f64"]
    pub fn cuda_add_backward(
        h_grad_out: *const f64,
        h_a_grad: *mut f64,
        h_b_grad: *mut f64,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "add_backward_f32"]
    pub fn cuda_add_backward_f32(
        h_grad_out: *const f32,
        h_a_grad: *mut f32,
        h_b_grad: *mut f32,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "sub_backward_f64"]
    pub fn cuda_sub_backward(
        h_grad_out: *const f64,
        h_a_grad: *mut f64,
        h_b_grad: *mut f64,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "sub_backward_f32"]
    pub fn cuda_sub_backward_f32(
        h_grad_out: *const f32,
        h_a_grad: *mut f32,
        h_b_grad: *mut f32,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "mul_backward_f64"]
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
    #[link_name = "mul_backward_f32"]
    pub fn cuda_mul_backward_f32(
        h_grad_out: *const f32,
        h_a_data: *const f32,
        h_b_data: *const f32,
        h_a_grad: *mut f32,
        h_b_grad: *mut f32,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_data: *mut c_int,
        d_b_data: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "div_backward_f64"]
    pub fn cuda_div_backward(
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
    #[link_name = "div_backward_f32"]
    pub fn cuda_div_backward_f32(
        h_grad_out: *const f32,
        h_a_data: *const f32,
        h_b_data: *const f32,
        h_a_grad: *mut f32,
        h_b_grad: *mut f32,
        size: c_int,
        d_grad_out: *mut c_int,
        d_a_data: *mut c_int,
        d_b_data: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "acc_buffer_f64"]
    pub fn cuda_acc_buffer(
        h_dst: *mut f64,
        h_src: *const f64,
        size: c_int,
        d_dst: *mut c_int,
        d_src: *mut c_int,
    ) -> c_int;
    #[link_name = "acc_buffer_f32"]
    pub fn cuda_acc_buffer_f32(
        h_dst: *mut f32,
        h_src: *const f32,
        size: c_int,
        d_dst: *mut c_int,
        d_src: *mut c_int,
    ) -> c_int;

    // Element-wise forward kernels
    #[link_name = "add_forward_f64"]
    pub fn cuda_add_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "add_forward_f32"]
    pub fn cuda_add_forward_f32(
        h_a: *const f32,
        h_b: *const f32,
        h_out: *mut f32,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "sub_forward_f64"]
    pub fn cuda_sub_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "sub_forward_f32"]
    pub fn cuda_sub_forward_f32(
        h_a: *const f32,
        h_b: *const f32,
        h_out: *mut f32,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "mul_forward_f64"]
    pub fn cuda_mul_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "mul_forward_f32"]
    pub fn cuda_mul_forward_f32(
        h_a: *const f32,
        h_b: *const f32,
        h_out: *mut f32,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "div_forward_f64"]
    pub fn cuda_div_forward(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "div_forward_f32"]
    pub fn cuda_div_forward_f32(
        h_a: *const f32,
        h_b: *const f32,
        h_out: *mut f32,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "adam_step_f64"]
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
    #[link_name = "adam_step_f32"]
    pub fn cuda_adam_step_f32(
        h_params: *mut f32,
        h_grads: *const f32,
        h_m: *mut f32,
        h_v: *mut f32,
        size: c_int,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        clip_coef: f32,
        d_params: *mut c_int,
        d_grads: *mut c_int,
        d_m: *mut c_int,
        d_v: *mut c_int,
    ) -> c_int;

    #[link_name = "rmsnorm_forward_f64"]
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
    #[link_name = "rmsnorm_forward_f32"]
    pub fn cuda_rmsnorm_forward_f32(
        h_x: *const f32,
        h_weight: *const f32,
        h_out: *mut f32,
        dim: c_int,
        eps: f32,
        num_rows: c_int,
        d_x: *mut c_int,
        d_weight: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "rmsnorm_backward_f64"]
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
    #[link_name = "rmsnorm_backward_f32"]
    pub fn cuda_rmsnorm_backward_f32(
        h_grad_out: *const f32,
        h_x: *const f32,
        h_weight: *const f32,
        h_x_grad: *mut f32,
        h_w_grad: *mut f32,
        dim: c_int,
        eps: f32,
        num_rows: c_int,
        d_grad_out: *mut c_int,
        d_x: *mut c_int,
        d_weight: *mut c_int,
        d_x_grad: *mut c_int,
        d_w_grad: *mut c_int,
    ) -> c_int;

    // Sparse matrix-vector multiplication kernels
    #[link_name = "cuda_sparse_matvec_f64"]
    pub fn cuda_sparse_matvec(
        h_x: *const f64,
        h_w: *const f64,
        h_mask: *const u8,
        h_y: *mut f64,
        num_rows: c_int,
        in_dim: c_int,
        out_dim: c_int,
        d_x: *mut c_int,
        d_w: *mut c_int,
        d_mask: *mut c_int,
        d_y: *mut c_int,
    ) -> c_int;
    #[link_name = "cuda_sparse_matvec_f32"]
    pub fn cuda_sparse_matvec_f32(
        h_x: *const f32,
        h_w: *const f32,
        h_mask: *const u8,
        h_y: *mut f32,
        num_rows: c_int,
        in_dim: c_int,
        out_dim: c_int,
        d_x: *mut c_int,
        d_w: *mut c_int,
        d_mask: *mut c_int,
        d_y: *mut c_int,
    ) -> c_int;

    #[link_name = "cuda_sparse_matvec_bias_f64"]
    pub fn cuda_sparse_matvec_bias(
        h_x: *const f64,
        h_w: *const f64,
        h_mask: *const u8,
        h_bias: *const f64,
        h_y: *mut f64,
        num_rows: c_int,
        in_dim: c_int,
        out_dim: c_int,
        d_x: *mut c_int,
        d_w: *mut c_int,
        d_mask: *mut c_int,
        d_bias: *mut c_int,
        d_y: *mut c_int,
    ) -> c_int;
    #[link_name = "cuda_sparse_matvec_bias_f32"]
    pub fn cuda_sparse_matvec_bias_f32(
        h_x: *const f32,
        h_w: *const f32,
        h_mask: *const u8,
        h_bias: *const f32,
        h_y: *mut f32,
        num_rows: c_int,
        in_dim: c_int,
        out_dim: c_int,
        d_x: *mut c_int,
        d_w: *mut c_int,
        d_mask: *mut c_int,
        d_bias: *mut c_int,
        d_y: *mut c_int,
    ) -> c_int;

    #[link_name = "scale_f64"]
    pub fn cuda_scale(
        h_in: *const f64,
        h_out: *mut f64,
        scale: f64,
        size: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "scale_f32"]
    pub fn cuda_scale_f32(
        h_in: *const f32,
        h_out: *mut f32,
        scale: f32,
        size: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "fill_f64"]
    pub fn cuda_fill(h_data: *mut f64, value: f64, size: c_int, d_data: *mut c_int) -> c_int;
    #[link_name = "fill_f32"]
    pub fn cuda_fill_f32(h_data: *mut f32, value: f32, size: c_int, d_data: *mut c_int) -> c_int;
    #[link_name = "sumsq_accum_f64"]
    pub fn cuda_sumsq_accum(
        h_in: *const f64,
        h_accum: *mut f64,
        size: c_int,
        d_in: *mut c_int,
        d_accum: *mut c_int,
    ) -> c_int;
    #[link_name = "sumsq_accum_f32"]
    pub fn cuda_sumsq_accum_f32(
        h_in: *const f32,
        h_accum: *mut f32,
        size: c_int,
        d_in: *mut c_int,
        d_accum: *mut c_int,
    ) -> c_int;
    #[link_name = "sum_accum_f64"]
    pub fn cuda_sum_accum(
        h_in: *const f64,
        h_accum: *mut f64,
        size: c_int,
        scale: f64,
        d_in: *mut c_int,
        d_accum: *mut c_int,
    ) -> c_int;
    #[link_name = "sum_accum_f32"]
    pub fn cuda_sum_accum_f32(
        h_in: *const f32,
        h_accum: *mut f32,
        size: c_int,
        scale: f32,
        d_in: *mut c_int,
        d_accum: *mut c_int,
    ) -> c_int;
    #[link_name = "clip_coef_from_sumsq_f64"]
    pub fn cuda_clip_coef_from_sumsq(
        h_sumsq: *const f64,
        h_coef: *mut f64,
        max_norm: f64,
        eps: f64,
        d_sumsq: *mut c_int,
        d_coef: *mut c_int,
    ) -> c_int;
    #[link_name = "clip_coef_from_sumsq_f32"]
    pub fn cuda_clip_coef_from_sumsq_f32(
        h_sumsq: *const f32,
        h_coef: *mut f32,
        max_norm: f32,
        eps: f32,
        d_sumsq: *mut c_int,
        d_coef: *mut c_int,
    ) -> c_int;
    #[link_name = "scale_inplace_by_scalar_f64"]
    pub fn cuda_scale_inplace_by_scalar(
        h_data: *mut f64,
        h_scalar: *const f64,
        size: c_int,
        d_data: *mut c_int,
        d_scalar: *mut c_int,
    ) -> c_int;
    #[link_name = "scale_inplace_by_scalar_f32"]
    pub fn cuda_scale_inplace_by_scalar_f32(
        h_data: *mut f32,
        h_scalar: *const f32,
        size: c_int,
        d_data: *mut c_int,
        d_scalar: *mut c_int,
    ) -> c_int;
    #[link_name = "add_scalar_f64"]
    pub fn cuda_add_scalar(
        h_data: *mut f64,
        h_scalar: *const f64,
        scale: f64,
        size: c_int,
        d_data: *mut c_int,
        d_scalar: *mut c_int,
    ) -> c_int;
    #[link_name = "add_scalar_f32"]
    pub fn cuda_add_scalar_f32(
        h_data: *mut f32,
        h_scalar: *const f32,
        scale: f32,
        size: c_int,
        d_data: *mut c_int,
        d_scalar: *mut c_int,
    ) -> c_int;
    #[link_name = "lerp_inplace_f64"]
    pub fn cuda_lerp_inplace(
        h_target: *mut f64,
        h_source: *const f64,
        tau: f64,
        size: c_int,
        d_target: *mut c_int,
        d_source: *mut c_int,
    ) -> c_int;
    #[link_name = "lerp_inplace_f32"]
    pub fn cuda_lerp_inplace_f32(
        h_target: *mut f32,
        h_source: *const f32,
        tau: f32,
        size: c_int,
        d_target: *mut c_int,
        d_source: *mut c_int,
    ) -> c_int;
    #[link_name = "double_dqn_target_f64"]
    pub fn cuda_double_dqn_target(
        h_eval: *const f64,
        h_target: *const f64,
        h_rewards: *const f64,
        h_dones: *const f64,
        h_out: *mut f64,
        batch: c_int,
        actions: c_int,
        gamma: f64,
        d_eval: *mut c_int,
        d_target: *mut c_int,
        d_rewards: *mut c_int,
        d_dones: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "double_dqn_target_f32"]
    pub fn cuda_double_dqn_target_f32(
        h_eval: *const f32,
        h_target: *const f32,
        h_rewards: *const f32,
        h_dones: *const f32,
        h_out: *mut f32,
        batch: c_int,
        actions: c_int,
        gamma: f32,
        d_eval: *mut c_int,
        d_target: *mut c_int,
        d_rewards: *mut c_int,
        d_dones: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "abs_diff_f64"]
    pub fn cuda_abs_diff(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "abs_diff_f32"]
    pub fn cuda_abs_diff_f32(
        h_a: *const f32,
        h_b: *const f32,
        h_out: *mut f32,
        size: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "per_store_transition_f32"]
    pub fn cuda_per_store_transition_f32(
        d_states: *mut f32,
        d_next_states: *mut f32,
        d_actions: *mut c_int,
        d_rewards: *mut f32,
        d_dones: *mut f32,
        d_priorities: *mut f32,
        d_state: *const f32,
        d_next_state: *const f32,
        idx: c_int,
        action: c_int,
        reward: f32,
        done: f32,
        priority: f32,
        dim: c_int,
    ) -> c_int;
    #[link_name = "per_store_transition_with_max_f32"]
    pub fn cuda_per_store_transition_with_max_f32(
        d_states: *mut f32,
        d_next_states: *mut f32,
        d_actions: *mut c_int,
        d_rewards: *mut f32,
        d_dones: *mut f32,
        d_priorities: *mut f32,
        d_max_priority: *const f32,
        d_state: *const f32,
        d_next_state: *const f32,
        idx: c_int,
        action: c_int,
        reward: f32,
        done: f32,
        alpha: f32,
        dim: c_int,
    ) -> c_int;
    #[link_name = "per_sample_f32"]
    pub fn cuda_per_sample_f32(
        d_states: *const f32,
        d_next_states: *const f32,
        d_actions: *const c_int,
        d_rewards: *const f32,
        d_dones: *const f32,
        d_priorities: *const f32,
        d_uniforms: *const f32,
        d_batch_states: *mut f32,
        d_batch_next_states: *mut f32,
        d_batch_action_mask: *mut f32,
        d_batch_rewards: *mut f32,
        d_batch_dones: *mut f32,
        d_batch_weights: *mut f32,
        d_batch_indices: *mut c_int,
        size: c_int,
        dim: c_int,
        actions_count: c_int,
        batch: c_int,
        beta: f32,
        total_priority: f32,
    ) -> c_int;
    #[link_name = "per_update_priorities_f32"]
    pub fn cuda_per_update_priorities_f32(
        d_priorities: *mut f32,
        d_indices: *const c_int,
        d_td_errors: *const f32,
        d_max_priority: *mut f32,
        batch: c_int,
        capacity: c_int,
        alpha: f32,
        epsilon: f32,
    ) -> c_int;
    #[link_name = "select_last_token_f64"]
    pub fn cuda_select_last_token(
        h_in: *const f64,
        h_out: *mut f64,
        batch: c_int,
        seq: c_int,
        dim: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "select_last_token_f32"]
    pub fn cuda_select_last_token_f32(
        h_in: *const f32,
        h_out: *mut f32,
        batch: c_int,
        seq: c_int,
        dim: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "select_last_token_backward_f64"]
    pub fn cuda_select_last_token_backward(
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        batch: c_int,
        seq: c_int,
        dim: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "select_last_token_backward_f32"]
    pub fn cuda_select_last_token_backward_f32(
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        batch: c_int,
        seq: c_int,
        dim: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "index_select_f64"]
    pub fn cuda_index_select(
        h_in: *const f64,
        h_out: *mut f64,
        idx: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "index_select_f32"]
    pub fn cuda_index_select_f32(
        h_in: *const f32,
        h_out: *mut f32,
        idx: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "index_select_backward_f64"]
    pub fn cuda_index_select_backward(
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        idx: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "index_select_backward_f32"]
    pub fn cuda_index_select_backward_f32(
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        idx: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "argmax_f64"]
    pub fn cuda_argmax(
        h_in: *const f64,
        h_out: *mut c_int,
        size: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "argmax_f32"]
    pub fn cuda_argmax_f32(
        h_in: *const f32,
        h_out: *mut c_int,
        size: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "exp_f64"]
    pub fn cuda_exp(
        h_in: *const f64,
        h_out: *mut f64,
        size: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "exp_f32"]
    pub fn cuda_exp_f32(
        h_in: *const f32,
        h_out: *mut f32,
        size: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "exp_backward_f64"]
    pub fn cuda_exp_backward(
        h_exp_out: *const f64,
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        size: c_int,
        d_exp_out: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "exp_backward_f32"]
    pub fn cuda_exp_backward_f32(
        h_exp_out: *const f32,
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        size: c_int,
        d_exp_out: *mut c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "weighted_mse_loss_f64"]
    pub fn cuda_weighted_mse_loss(
        h_pred: *const f64,
        h_target: *const f64,
        h_weights: *const f64,
        h_out: *mut f64,
        h_weight_sum: *mut f64,
        size: c_int,
        d_pred: *mut c_int,
        d_target: *mut c_int,
        d_weights: *mut c_int,
        d_out: *mut c_int,
        d_weight_sum: *mut c_int,
    ) -> c_int;
    #[link_name = "weighted_mse_loss_f32"]
    pub fn cuda_weighted_mse_loss_f32(
        h_pred: *const f32,
        h_target: *const f32,
        h_weights: *const f32,
        h_out: *mut f32,
        h_weight_sum: *mut f32,
        size: c_int,
        d_pred: *mut c_int,
        d_target: *mut c_int,
        d_weights: *mut c_int,
        d_out: *mut c_int,
        d_weight_sum: *mut c_int,
    ) -> c_int;
    #[link_name = "weighted_mse_backward_f64"]
    pub fn cuda_weighted_mse_backward(
        h_pred: *const f64,
        h_target: *const f64,
        h_weights: *const f64,
        h_weight_sum: *const f64,
        h_grad_out: *const f64,
        h_pred_grad: *mut f64,
        size: c_int,
        d_pred: *mut c_int,
        d_target: *mut c_int,
        d_weights: *mut c_int,
        d_weight_sum: *mut c_int,
        d_grad_out: *mut c_int,
        d_pred_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "weighted_mse_backward_f32"]
    pub fn cuda_weighted_mse_backward_f32(
        h_pred: *const f32,
        h_target: *const f32,
        h_weights: *const f32,
        h_weight_sum: *const f32,
        h_grad_out: *const f32,
        h_pred_grad: *mut f32,
        size: c_int,
        d_pred: *mut c_int,
        d_target: *mut c_int,
        d_weights: *mut c_int,
        d_weight_sum: *mut c_int,
        d_grad_out: *mut c_int,
        d_pred_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "scale_backward_f64"]
    pub fn cuda_scale_backward(
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        scale: f64,
        size: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "scale_backward_f32"]
    pub fn cuda_scale_backward_f32(
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        scale: f32,
        size: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "causal_mask_f64"]
    pub fn cuda_causal_mask(
        h_in: *const f64,
        h_out: *mut f64,
        batches: c_int,
        seq: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "causal_mask_f32"]
    pub fn cuda_causal_mask_f32(
        h_in: *const f32,
        h_out: *mut f32,
        batches: c_int,
        seq: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "causal_mask_backward_f64"]
    pub fn cuda_causal_mask_backward(
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        batches: c_int,
        seq: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "causal_mask_backward_f32"]
    pub fn cuda_causal_mask_backward_f32(
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        batches: c_int,
        seq: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "concat_last_dim_f64"]
    pub fn cuda_concat_last_dim(
        h_a: *const f64,
        h_b: *const f64,
        h_out: *mut f64,
        rows: c_int,
        a_dim: c_int,
        b_dim: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "concat_last_dim_f32"]
    pub fn cuda_concat_last_dim_f32(
        h_a: *const f32,
        h_b: *const f32,
        h_out: *mut f32,
        rows: c_int,
        a_dim: c_int,
        b_dim: c_int,
        d_a: *mut c_int,
        d_b: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "concat_last_dim_backward_f64"]
    pub fn cuda_concat_last_dim_backward(
        h_grad_out: *const f64,
        h_a_grad: *mut f64,
        h_b_grad: *mut f64,
        rows: c_int,
        a_dim: c_int,
        b_dim: c_int,
        d_grad_out: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "concat_last_dim_backward_f32"]
    pub fn cuda_concat_last_dim_backward_f32(
        h_grad_out: *const f32,
        h_a_grad: *mut f32,
        h_b_grad: *mut f32,
        rows: c_int,
        a_dim: c_int,
        b_dim: c_int,
        d_grad_out: *mut c_int,
        d_a_grad: *mut c_int,
        d_b_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "split_last_dim_f64"]
    pub fn cuda_split_last_dim(
        h_in: *const f64,
        h_out: *mut f64,
        rows: c_int,
        input_dim: c_int,
        part_dim: c_int,
        part_idx: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "split_last_dim_f32"]
    pub fn cuda_split_last_dim_f32(
        h_in: *const f32,
        h_out: *mut f32,
        rows: c_int,
        input_dim: c_int,
        part_dim: c_int,
        part_idx: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "split_last_dim_backward_f64"]
    pub fn cuda_split_last_dim_backward(
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        rows: c_int,
        input_dim: c_int,
        part_dim: c_int,
        part_idx: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "split_last_dim_backward_f32"]
    pub fn cuda_split_last_dim_backward_f32(
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        rows: c_int,
        input_dim: c_int,
        part_dim: c_int,
        part_idx: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "broadcast_batch_f64"]
    pub fn cuda_broadcast_batch(
        h_in: *const f64,
        h_out: *mut f64,
        batch_size: c_int,
        inner_len: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "broadcast_batch_f32"]
    pub fn cuda_broadcast_batch_f32(
        h_in: *const f32,
        h_out: *mut f32,
        batch_size: c_int,
        inner_len: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "broadcast_batch_backward_f64"]
    pub fn cuda_broadcast_batch_backward(
        h_grad_out: *const f64,
        h_input_grad: *mut f64,
        batch_size: c_int,
        inner_len: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "broadcast_batch_backward_f32"]
    pub fn cuda_broadcast_batch_backward_f32(
        h_grad_out: *const f32,
        h_input_grad: *mut f32,
        batch_size: c_int,
        inner_len: c_int,
        d_grad_out: *mut c_int,
        d_input_grad: *mut c_int,
    ) -> c_int;

    #[link_name = "transpose_last_two_f64"]
    pub fn cuda_transpose_last_two(
        h_in: *const f64,
        h_out: *mut f64,
        outer: c_int,
        rows: c_int,
        cols: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "transpose_last_two_f32"]
    pub fn cuda_transpose_last_two_f32(
        h_in: *const f32,
        h_out: *mut f32,
        outer: c_int,
        rows: c_int,
        cols: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "transpose_4d_f64"]
    pub fn cuda_transpose_4d(
        h_in: *const f64,
        h_out: *mut f64,
        d0: c_int,
        d1: c_int,
        d2: c_int,
        d3: c_int,
        dim0: c_int,
        dim1: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "transpose_4d_f32"]
    pub fn cuda_transpose_4d_f32(
        h_in: *const f32,
        h_out: *mut f32,
        d0: c_int,
        d1: c_int,
        d2: c_int,
        d3: c_int,
        dim0: c_int,
        dim1: c_int,
        d_in: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;

    #[link_name = "batched_qk_scores_f64"]
    pub fn cuda_batched_qk_scores(
        h_q: *const f64,
        h_k: *const f64,
        h_out: *mut f64,
        batches: c_int,
        seq: c_int,
        dim: c_int,
        d_q: *mut c_int,
        d_k: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "batched_qk_scores_f32"]
    pub fn cuda_batched_qk_scores_f32(
        h_q: *const f32,
        h_k: *const f32,
        h_out: *mut f32,
        batches: c_int,
        seq: c_int,
        dim: c_int,
        d_q: *mut c_int,
        d_k: *mut c_int,
        d_out: *mut c_int,
    ) -> c_int;
    #[link_name = "batched_qk_scores_backward_f64"]
    pub fn cuda_batched_qk_scores_backward(
        h_grad_out: *const f64,
        h_q: *const f64,
        h_k: *const f64,
        h_q_grad: *mut f64,
        h_k_grad: *mut f64,
        batches: c_int,
        seq: c_int,
        dim: c_int,
        d_grad_out: *mut c_int,
        d_q: *mut c_int,
        d_k: *mut c_int,
        d_q_grad: *mut c_int,
        d_k_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "batched_qk_scores_backward_f32"]
    pub fn cuda_batched_qk_scores_backward_f32(
        h_grad_out: *const f32,
        h_q: *const f32,
        h_k: *const f32,
        h_q_grad: *mut f32,
        h_k_grad: *mut f32,
        batches: c_int,
        seq: c_int,
        dim: c_int,
        d_grad_out: *mut c_int,
        d_q: *mut c_int,
        d_k: *mut c_int,
        d_q_grad: *mut c_int,
        d_k_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "attention_weighted_sum_backward_f64"]
    pub fn cuda_attention_weighted_sum_backward(
        h_grad_out: *const f64,
        h_probs: *const f64,
        h_values: *const f64,
        h_probs_grad: *mut f64,
        h_values_grad: *mut f64,
        batches: c_int,
        seq: c_int,
        head_dim: c_int,
        d_grad_out: *mut c_int,
        d_probs: *mut c_int,
        d_values: *mut c_int,
        d_probs_grad: *mut c_int,
        d_values_grad: *mut c_int,
    ) -> c_int;
    #[link_name = "attention_weighted_sum_backward_f32"]
    pub fn cuda_attention_weighted_sum_backward_f32(
        h_grad_out: *const f32,
        h_probs: *const f32,
        h_values: *const f32,
        h_probs_grad: *mut f32,
        h_values_grad: *mut f32,
        batches: c_int,
        seq: c_int,
        head_dim: c_int,
        d_grad_out: *mut c_int,
        d_probs: *mut c_int,
        d_values: *mut c_int,
        d_probs_grad: *mut c_int,
        d_values_grad: *mut c_int,
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
