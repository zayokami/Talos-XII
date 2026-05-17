// rmsnorm.cu - CUDA RMSNorm kernels
#include "common.cu"

template<typename T>
__global__ void rmsnorm_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int dim, T eps, int num_rows
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    __shared__ T sdata[256];

    T sum_sq = T(0.0);
    for (int i = tid; i < dim; i += blockDim.x) {
        T val = x[row * dim + i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    T rms = sqrt(sdata[0] / T(dim) + eps);
    T inv_rms = T(1.0) / rms;

    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        out[idx] = x[idx] * inv_rms * weight[i];
    }
}

template<typename T>
__global__ void rmsnorm_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ x_grad,
    T* __restrict__ w_grad,
    int dim, T eps, int num_rows
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    __shared__ T sdata[256];
    __shared__ T shared_inv_rms;
    __shared__ T shared_mean_dot;

    T sum_sq = T(0.0);
    for (int i = tid; i < dim; i += blockDim.x) {
        T val = x[row * dim + i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    T rms = sqrt(sdata[0] / T(dim) + eps);
    T inv_rms = T(1.0) / rms;
    if (tid == 0) shared_inv_rms = inv_rms;
    __syncthreads();

    T dot = T(0.0);
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        T g = grad_out[idx];
        T w = weight[i];
        T x_hat = x[idx] * inv_rms;
        dot += g * w * x_hat;
    }
    sdata[tid] = dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) shared_mean_dot = sdata[0] / T(dim);
    __syncthreads();

    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        T g = grad_out[idx];
        T w = weight[i];
        T x_hat = x[idx] * inv_rms;
        T dl_dxhat = g * w;
        x_grad[idx] += inv_rms * (dl_dxhat - x_hat * shared_mean_dot);
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        T g = grad_out[idx];
        T x_hat = x[idx] * inv_rms;
        atomicAdd(&w_grad[i], g * x_hat);
    }
}

extern "C" int rmsnorm_forward_f64(
    const double* h_x, const double* h_weight, double* h_out,
    int dim, double eps, int num_rows,
    int* d_x, int* d_weight, int* d_out
) {
    const double* dev_x = (const double*)d_x;
    const double* dev_weight = (const double*)d_weight;
    double* dev_out = (double*)d_out;
    dim3 grid_dim(num_rows);
    dim3 block_dim(256);
    CUDA_LAUNCH(rmsnorm_forward_kernel<double>, grid_dim, block_dim, 0,
                dev_x, dev_weight, dev_out, dim, eps, num_rows);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int rmsnorm_forward_f32(
    const float* h_x, const float* h_weight, float* h_out,
    int dim, float eps, int num_rows,
    int* d_x, int* d_weight, int* d_out
) {
    const float* dev_x = (const float*)d_x;
    const float* dev_weight = (const float*)d_weight;
    float* dev_out = (float*)d_out;
    dim3 grid_dim(num_rows);
    dim3 block_dim(256);
    CUDA_LAUNCH(rmsnorm_forward_kernel<float>, grid_dim, block_dim, 0,
                dev_x, dev_weight, dev_out, dim, eps, num_rows);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int rmsnorm_backward_f64(
    const double* h_grad_out, const double* h_x, const double* h_weight,
    double* h_x_grad, double* h_w_grad,
    int dim, double eps, int num_rows,
    int* d_grad_out, int* d_x, int* d_weight, int* d_x_grad, int* d_w_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_x = (const double*)d_x;
    const double* dev_weight = (const double*)d_weight;
    double* dev_x_grad = (double*)d_x_grad;
    double* dev_w_grad = (double*)d_w_grad;
    dim3 grid_dim(num_rows);
    dim3 block_dim(256);
    CUDA_LAUNCH(rmsnorm_backward_kernel<double>, grid_dim, block_dim, 0,
                dev_grad_out, dev_x, dev_weight, dev_x_grad, dev_w_grad,
                dim, eps, num_rows);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int rmsnorm_backward_f32(
    const float* h_grad_out, const float* h_x, const float* h_weight,
    float* h_x_grad, float* h_w_grad,
    int dim, float eps, int num_rows,
    int* d_grad_out, int* d_x, int* d_weight, int* d_x_grad, int* d_w_grad
) {
    const float* dev_grad_out = (const float*)d_grad_out;
    const float* dev_x = (const float*)d_x;
    const float* dev_weight = (const float*)d_weight;
    float* dev_x_grad = (float*)d_x_grad;
    float* dev_w_grad = (float*)d_w_grad;
    dim3 grid_dim(num_rows);
    dim3 block_dim(256);
    CUDA_LAUNCH(rmsnorm_backward_kernel<float>, grid_dim, block_dim, 0,
                dev_grad_out, dev_x, dev_weight, dev_x_grad, dev_w_grad,
                dim, eps, num_rows);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
