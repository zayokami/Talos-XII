// gelu.cu - GELU activation kernel
#include "common.cuh"

const double GELU_SQRT_2_OVER_PI = 0.7978845608028654;  // sqrt(2/pi)
const double GELU_C = 0.044715;

//==============================================================================
// GELU activation kernel
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//==============================================================================
__global__ void gelu_kernel(
    double* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        double x = data[idx];
        double x3 = x * x * x;
        double inner = GELU_SQRT_2_OVER_PI * (x + GELU_C * x3);
        data[idx] = 0.5 * x * (1.0 + tanh(inner));
    }
}

//==============================================================================
// GELU with inplace modification
//==============================================================================
__global__ void gelu_inplace_kernel(
    double* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        double x = data[idx];
        double x3 = x * x * x;
        double inner = GELU_SQRT_2_OVER_PI * (x + GELU_C * x3);
        data[idx] = 0.5 * x * (1.0 + tanh(inner));
    }
}

//==============================================================================
// GELU derivative kernel (for backward pass)
// dGELU/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh^2(inner)) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
//==============================================================================
__global__ void gelu_backward_kernel(
    const double* __restrict__ grad_output,
    const double* __restrict__ forward_input,
    double* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        double x = forward_input[idx];
        double x3 = x * x * x;
        double inner = GELU_SQRT_2_OVER_PI * (x + GELU_C * x3);
        double tanh_inner = tanh(inner);
        double sech_inner = 1.0 - tanh_inner * tanh_inner;  // 1 - tanh^2

        double d_gelu = 0.5 * (1.0 + tanh_inner) +
                        0.5 * x * sech_inner * GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_C * x * x);

        grad_input[idx] = grad_output[idx] * d_gelu;
    }
}

//==============================================================================
// ReLU kernel
//==============================================================================
__global__ void relu_kernel(
    double* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = fmax(0.0, data[idx]);
    }
}

//==============================================================================
// ReLU backward kernel
//==============================================================================
__global__ void relu_backward_kernel(
    const double* __restrict__ grad_output,
    const double* __restrict__ forward_input,
    double* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad_input[idx] = (forward_input[idx] > 0.0) ? grad_output[idx] : 0.0;
    }
}

//==============================================================================
// Host wrapper for GELU
//==============================================================================
extern "C" void gelu(
    double* h_data, int size,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(gelu_inplace_kernel, grid_dim, block_dim, 0, dev_data, size);

    CUDA_CHECK(cudaPeekAtLastError());
}

//==============================================================================
// Host wrapper for GELU backward
//==============================================================================
extern "C" void gelu_backward(
    const double* h_grad_out, const double* h_forward,
    double* h_grad_in, int size,
    int* d_grad_out, int* d_forward, int* d_grad_in
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_forward = (const double*)d_forward;
    double* dev_grad_in = (double*)d_grad_in;

    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(gelu_backward_kernel, grid_dim, block_dim, 0,
        dev_grad_out, dev_forward, dev_grad_in, size);

    CUDA_CHECK(cudaPeekAtLastError());
}

//==============================================================================
// Host wrapper for ReLU
//==============================================================================
extern "C" void relu(
    double* h_data, int size,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(relu_inplace_kernel, grid_dim, block_dim, 0, dev_data, size);

    CUDA_CHECK(cudaPeekAtLastError());
}
