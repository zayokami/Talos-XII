// rmsnorm.cu - CUDA RMSNorm kernels
#include "common.cu"

//==============================================================================
// RMSNorm forward kernel
// out[row, i] = x[row, i] / rms(row) * weight[i]
// rms(row) = sqrt(mean(x^2) + eps)
//==============================================================================
__global__ void rmsnorm_forward_kernel(
    const double* __restrict__ x,
    const double* __restrict__ weight,
    double* __restrict__ out,
    int dim,
    double eps,
    int num_rows
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    __shared__ double sdata[256];

    // Compute sum of squares for this row
    double sum_sq = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        double val = x[row * dim + i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    double rms = sqrt(sdata[0] / dim + eps);
    double inv_rms = 1.0 / rms;

    // Write output
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        out[idx] = x[idx] * inv_rms * weight[i];
    }
}

//==============================================================================
// RMSNorm backward kernel
// x_grad[row, i] += inv_rms * (dl_dxhat - x_hat * mean_dot)
// w_grad[i]      += sum_rows(grad_out[row, i] * x_hat)
//==============================================================================
__global__ void rmsnorm_backward_kernel(
    const double* __restrict__ grad_out,
    const double* __restrict__ x,
    const double* __restrict__ weight,
    double* __restrict__ x_grad,
    double* __restrict__ w_grad,
    int dim,
    double eps,
    int num_rows
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    __shared__ double sdata[256];
    __shared__ double shared_inv_rms;
    __shared__ double shared_mean_dot;

    // Compute rms for this row
    double sum_sq = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        double val = x[row * dim + i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    double rms = sqrt(sdata[0] / dim + eps);
    double inv_rms = 1.0 / rms;
    if (tid == 0) shared_inv_rms = inv_rms;
    __syncthreads();

    // Compute dot_sum = sum(grad_out * weight * x_hat)
    double dot = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        double g = grad_out[idx];
        double w = weight[i];
        double x_hat = x[idx] * inv_rms;
        dot += g * w * x_hat;
    }
    sdata[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) shared_mean_dot = sdata[0] / dim;
    __syncthreads();

    // Write x_grad
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        double g = grad_out[idx];
        double w = weight[i];
        double x_hat = x[idx] * inv_rms;
        double dl_dxhat = g * w;
        x_grad[idx] += inv_rms * (dl_dxhat - x_hat * shared_mean_dot);
    }

    // Accumulate w_grad using atomicAdd
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        double g = grad_out[idx];
        double x_hat = x[idx] * inv_rms;
        atomicAdd(&w_grad[i], g * x_hat);
    }
}

//==============================================================================
// Host wrappers
//==============================================================================
extern "C" int rmsnorm_forward(
    const double* h_x,
    const double* h_weight,
    double* h_out,
    int dim,
    double eps,
    int num_rows,
    int* d_x,
    int* d_weight,
    int* d_out
) {
    const double* dev_x = (const double*)d_x;
    const double* dev_weight = (const double*)d_weight;
    double* dev_out = (double*)d_out;

    dim3 grid_dim(num_rows);
    dim3 block_dim(256);

    CUDA_LAUNCH(rmsnorm_forward_kernel, grid_dim, block_dim, 0,
                dev_x, dev_weight, dev_out, dim, eps, num_rows);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int rmsnorm_backward(
    const double* h_grad_out,
    const double* h_x,
    const double* h_weight,
    double* h_x_grad,
    double* h_w_grad,
    int dim,
    double eps,
    int num_rows,
    int* d_grad_out,
    int* d_x,
    int* d_weight,
    int* d_x_grad,
    int* d_w_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_x = (const double*)d_x;
    const double* dev_weight = (const double*)d_weight;
    double* dev_x_grad = (double*)d_x_grad;
    double* dev_w_grad = (double*)d_w_grad;

    dim3 grid_dim(num_rows);
    dim3 block_dim(256);

    CUDA_LAUNCH(rmsnorm_backward_kernel, grid_dim, block_dim, 0,
                dev_grad_out, dev_x, dev_weight, dev_x_grad, dev_w_grad,
                dim, eps, num_rows);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
