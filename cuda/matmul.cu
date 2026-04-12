// matmul.cu - Matrix multiplication kernels
#include "common.cu"

// TILE_SIZE for shared memory blocking
#define TILE_SIZE 16

//==============================================================================
// Naive GEMM kernel (baseline)
// C = alpha * A * B + beta * C
// A: [m, k], B: [k, n], C: [m, n]
// Using row-major order
//==============================================================================
__global__ void gemm_naive_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int m, int n, int k,
    double alpha, double beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

//==============================================================================
// Shared memory blocked GEMM kernel
// Uses tiling to reduce global memory accesses
//==============================================================================
__global__ void gemm_blocked_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int m, int n, int k,
    double alpha, double beta
) {
    // Shared memory for tiles
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    double sum = 0.0;

    // Loop over tiles
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load A tile into shared memory
        if (row < m && (tile * TILE_SIZE + tx) < k) {
            As[ty][tx] = A[row * k + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        // Load B tile into shared memory
        if (col < n && (tile * TILE_SIZE + ty) < k) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        // Compute partial result
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < m && col < n) {
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

//==============================================================================
// FP64 GEMM using cuBLAS (wrapper for high performance)
// For row-major: C[m,n] = A[m,k] * B[k,n]
//
// cuBLAS is column-major: C_col[n,m] = B_col[k,n]^T * A_col[m,k]^T
// Therefore we call cublasDgemm with both matrices transposed:
// cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, alpha, B, n, A, k, beta, C, n)
//==============================================================================
extern "C" void cublas_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
) {
    // A is [m, k] row-major -> column-major shape [k, m], leading dim m
    // B is [k, n] row-major -> column-major shape [n, k], leading dim k
    // C is [m, n] row-major -> column-major shape [n, m], leading dim m
    //
    // For row-major C = A * B using cuBLAS column-major:
    // C_col = transpose(B) * transpose(A)
    // cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, alpha, B, n, A, m, beta, C, m)
    //
    // The transa/transb parameters are ignored since we always transpose.
    // We swap A and B, and swap m and n to achieve the row-major result.

    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_T,  // Transpose A (row-major A -> column-major A^T)
        CUBLAS_OP_T,  // Transpose B (row-major B -> column-major B^T)
        n, m, k,      // Dimensions swapped: (n,m) = (cols_C, rows_C) in column-major
        alpha,
        B, n,         // B is [k,n] row-major -> leading dim is n
        A, m,         // A is [m,k] row-major -> leading dim is m
        beta,
        C, m          // C is [m,n] row-major -> leading dim is m
    ));
}

//==============================================================================
// Host wrapper for naive GEMM
//==============================================================================
extern "C" void gemm_naive(
    const double* h_A, const double* h_B, double* h_C,
    int m, int n, int k,
    double alpha, double beta,
    int* d_A, int* d_B, int* d_C
) {
    double *dev_A = (double*)d_A;
    double *dev_B = (double*)d_B;
    double *dev_C = (double*)d_C;

    // Use 32x32 blocks for better occupancy
    dim3 block_dim(32, 32);
    dim3 grid_dim((n + 31) / 32, (m + 31) / 32);

    CUDA_LAUNCH(gemm_naive_kernel, grid_dim, block_dim, 0,
        dev_A, dev_B, dev_C, m, n, k, alpha, beta);

    CUDA_CHECK(cudaPeekAtLastError());
}

//==============================================================================
// GEMM with fused ReLU activation
//==============================================================================
__global__ void gemm_relu_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int m, int n, int k,
    double alpha, double beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        double val = alpha * sum + beta * C[row * n + col];
        C[row * n + col] = val > 0.0 ? val : 0.0;
    }
}

//==============================================================================
// Host wrapper for GEMM + ReLU
//==============================================================================
extern "C" void gemm_relu(
    const double* h_A, const double* h_B, double* h_C,
    int m, int n, int k,
    double alpha, double beta,
    int* d_A, int* d_B, int* d_C
) {
    double *dev_A = (double*)d_A;
    double *dev_B = (double*)d_B;
    double *dev_C = (double*)d_C;

    dim3 block_dim(32, 32);
    dim3 grid_dim((n + 31) / 32, (m + 31) / 32);

    CUDA_LAUNCH(gemm_relu_kernel, grid_dim, block_dim, 0,
        dev_A, dev_B, dev_C, m, n, k, alpha, beta);

    CUDA_CHECK(cudaPeekAtLastError());
}

//==============================================================================
// Vector add kernel (for bias addition)
//==============================================================================
__global__ void vector_add_kernel(
    double* __restrict__ C,
    const double* __restrict__ bias,
    int m, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;

    if (idx < total) {
        int col = idx % n;
        C[idx] += bias[col];
    }
}

//==============================================================================
// Host wrapper for bias addition
//==============================================================================
extern "C" void add_bias(
    double* h_C, const double* h_bias,
    int m, int n,
    int* d_C, int* d_bias
) {
    double* dev_C = (double*)d_C;
    const double* dev_bias = (const double*)d_bias;

    int total = m * n;
    dim3 grid_dim = compute_grid_1d(total, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(vector_add_kernel, grid_dim, block_dim, 0,
        dev_C, dev_bias, m, n);

    CUDA_CHECK(cudaPeekAtLastError());
}
