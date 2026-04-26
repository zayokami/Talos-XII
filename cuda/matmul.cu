// matmul.cu - Matrix multiplication kernels (stub, actual GEMM uses cuBLAS via src/cuda/blas.rs)
#include "common.cu"

// Note: Custom GEMM kernels (naive, blocked, relu-fused, vector_add)
// were never wired into autograd and have been removed.
// The codebase uses cuBLAS via src/cuda/blas.rs for all GPU matmul operations.
// This file is kept to maintain the build.rs CUDA file list.

void matmul_stub_placeholder() {
    // Prevent empty translation unit warning
}
