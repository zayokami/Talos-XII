// sparse.cu - Sparse matrix-vector multiplication kernel for ACHF
#include "common.cu"

#define SPARSE_BLOCK 256

template<typename T>
__global__ void sparse_matvec_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w,
    const uint8_t* __restrict__ mask,
    T* __restrict__ y,
    int num_rows, int in_dim, int out_dim
) {
    int row = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows || j >= out_dim) return;

    T sum = T(0.0);
    const T* x_row = x + row * in_dim;
    for (int i = 0; i < in_dim; i++) {
        int idx = i * out_dim + j;
        if (mask[idx]) {
            sum += x_row[i] * w[idx];
        }
    }
    y[row * out_dim + j] = sum;
}

template<typename T>
__global__ void sparse_matvec_bias_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w,
    const uint8_t* __restrict__ mask,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int num_rows, int in_dim, int out_dim
) {
    int row = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows || j >= out_dim) return;

    T sum = bias[j];
    const T* x_row = x + row * in_dim;
    for (int i = 0; i < in_dim; i++) {
        int idx = i * out_dim + j;
        if (mask[idx]) {
            sum += x_row[i] * w[idx];
        }
    }
    y[row * out_dim + j] = sum;
}

extern "C" int cuda_sparse_matvec_f64(
    const double* x, const double* w, const uint8_t* mask, double* y,
    int num_rows, int in_dim, int out_dim,
    int* d_x, int* d_w, int* d_mask, int* d_y
) {
    const double* dev_x = (const double*)d_x;
    const double* dev_w = (const double*)d_w;
    const uint8_t* dev_m = (const uint8_t*)d_mask;
    double* dev_y = (double*)d_y;
    dim3 block(SPARSE_BLOCK);
    dim3 grid((out_dim + SPARSE_BLOCK - 1) / SPARSE_BLOCK, num_rows);
    sparse_matvec_kernel<double><<<grid, block>>>(
        dev_x, dev_w, dev_m, dev_y, num_rows, in_dim, out_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int cuda_sparse_matvec_f32(
    const float* x, const float* w, const uint8_t* mask, float* y,
    int num_rows, int in_dim, int out_dim,
    int* d_x, int* d_w, int* d_mask, int* d_y
) {
    const float* dev_x = (const float*)d_x;
    const float* dev_w = (const float*)d_w;
    const uint8_t* dev_m = (const uint8_t*)d_mask;
    float* dev_y = (float*)d_y;
    dim3 block(SPARSE_BLOCK);
    dim3 grid((out_dim + SPARSE_BLOCK - 1) / SPARSE_BLOCK, num_rows);
    sparse_matvec_kernel<float><<<grid, block>>>(
        dev_x, dev_w, dev_m, dev_y, num_rows, in_dim, out_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int cuda_sparse_matvec_bias_f64(
    const double* x, const double* w, const uint8_t* mask, const double* bias, double* y,
    int num_rows, int in_dim, int out_dim,
    int* d_x, int* d_w, int* d_mask, int* d_bias, int* d_y
) {
    const double* dev_x = (const double*)d_x;
    const double* dev_w = (const double*)d_w;
    const uint8_t* dev_m = (const uint8_t*)d_mask;
    const double* dev_b = (const double*)d_bias;
    double* dev_y = (double*)d_y;
    dim3 block(SPARSE_BLOCK);
    dim3 grid((out_dim + SPARSE_BLOCK - 1) / SPARSE_BLOCK, num_rows);
    sparse_matvec_bias_kernel<double><<<grid, block>>>(
        dev_x, dev_w, dev_m, dev_b, dev_y, num_rows, in_dim, out_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int cuda_sparse_matvec_bias_f32(
    const float* x, const float* w, const uint8_t* mask, const float* bias, float* y,
    int num_rows, int in_dim, int out_dim,
    int* d_x, int* d_w, int* d_mask, int* d_bias, int* d_y
) {
    const float* dev_x = (const float*)d_x;
    const float* dev_w = (const float*)d_w;
    const uint8_t* dev_m = (const uint8_t*)d_mask;
    const float* dev_b = (const float*)d_bias;
    float* dev_y = (float*)d_y;
    dim3 block(SPARSE_BLOCK);
    dim3 grid((out_dim + SPARSE_BLOCK - 1) / SPARSE_BLOCK, num_rows);
    sparse_matvec_bias_kernel<float><<<grid, block>>>(
        dev_x, dev_w, dev_m, dev_b, dev_y, num_rows, in_dim, out_dim);
    return (int)cudaPeekAtLastError();
}
