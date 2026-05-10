// sparse.cu - Sparse matrix-vector multiplication kernel for ACHF
// Uses dense+mask representation: weights are stored densely,
// but a uint8 mask indicates which elements are non-zero (static sparsity).
// This avoids CSR conversion overhead and works natively with f64.
#include "common.cu"

#define SPARSE_BLOCK 256

//==============================================================================
// sparse_matvec_f64 kernel
// Computes y[row, j] = sum_i x[row, i] * w[i, j] for mask[i, j] != 0
//
// x:     [num_rows, in_dim]  input matrix (row-major)
// w:     [in_dim, out_dim]   weight matrix (row-major)
// mask:  [in_dim, out_dim]   uint8 mask, 1 = active, 0 = skipped
// y:     [num_rows, out_dim] output matrix (row-major)
//==============================================================================
__global__ void sparse_matvec_f64(
    const double* __restrict__ x,
    const double* __restrict__ w,
    const uint8_t* __restrict__ mask,
    double* __restrict__ y,
    int num_rows, int in_dim, int out_dim
) {
    int row = blockIdx.y;
    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows || j >= out_dim) return;

    double sum = 0.0;
    const double* x_row = x + row * in_dim;
    for (int i = 0; i < in_dim; i++) {
        int idx = i * out_dim + j;
        if (mask[idx]) {
            sum += x_row[i] * w[idx];
        }
    }
    y[row * out_dim + j] = sum;
}

//==============================================================================
// sparse_matvec_bias_f64 kernel
// Same as above but adds bias vector: y += bias
//==============================================================================
__global__ void sparse_matvec_bias_f64(
    const double* __restrict__ x,
    const double* __restrict__ w,
    const uint8_t* __restrict__ mask,
    const double* __restrict__ bias,
    double* __restrict__ y,
    int num_rows, int in_dim, int out_dim
) {
    int row = blockIdx.y;
    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows || j >= out_dim) return;

    double sum = bias[j];
    const double* x_row = x + row * in_dim;
    for (int i = 0; i < in_dim; i++) {
        int idx = i * out_dim + j;
        if (mask[idx]) {
            sum += x_row[i] * w[idx];
        }
    }
    y[row * out_dim + j] = sum;
}

//==============================================================================
// Host wrapper
//==============================================================================
extern "C" int cuda_sparse_matvec(
    const double* x,
    const double* w,
    const uint8_t* mask,
    double* y,
    int num_rows, int in_dim, int out_dim,
    int* d_x, int* d_w, int* d_mask, int* d_y
) {
    const double* dev_x    = (const double*)d_x;
    const double* dev_w    = (const double*)d_w;
    const uint8_t* dev_m   = (const uint8_t*)d_mask;
    double* dev_y          = (double*)d_y;

    dim3 block(SPARSE_BLOCK);
    dim3 grid((out_dim + SPARSE_BLOCK - 1) / SPARSE_BLOCK, num_rows);

    sparse_matvec_f64<<<grid, block>>>(
        dev_x, dev_w, dev_m, dev_y, num_rows, in_dim, out_dim);

    return (int)cudaPeekAtLastError();
}

extern "C" int cuda_sparse_matvec_bias(
    const double* x,
    const double* w,
    const uint8_t* mask,
    const double* bias,
    double* y,
    int num_rows, int in_dim, int out_dim,
    int* d_x, int* d_w, int* d_mask, int* d_bias, int* d_y
) {
    const double* dev_x    = (const double*)d_x;
    const double* dev_w    = (const double*)d_w;
    const uint8_t* dev_m   = (const uint8_t*)d_mask;
    const double* dev_b    = (const double*)d_bias;
    double* dev_y          = (double*)d_y;

    dim3 block(SPARSE_BLOCK);
    dim3 grid((out_dim + SPARSE_BLOCK - 1) / SPARSE_BLOCK, num_rows);

    sparse_matvec_bias_f64<<<grid, block>>>(
        dev_x, dev_w, dev_m, dev_b, dev_y, num_rows, in_dim, out_dim);

    return (int)cudaPeekAtLastError();
}
