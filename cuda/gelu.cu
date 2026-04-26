// gelu.cu - GELU activation kernel
#include "common.cu"

#define GELU_SQRT_2_OVER_PI 0.7978845608028654  // sqrt(2/pi)
#define GELU_C 0.044715

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
// Host wrapper for GELU
//==============================================================================
extern "C" int gelu(
    double* h_data, int size,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(gelu_kernel, grid_dim, block_dim, 0, dev_data, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

//==============================================================================
// Host wrapper for ReLU
//==============================================================================
extern "C" int relu(
    double* h_data, int size,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(relu_kernel, grid_dim, block_dim, 0, dev_data, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
