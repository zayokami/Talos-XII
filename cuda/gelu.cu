// gelu.cu - GELU activation kernel
#include "common.cu"

#define GELU_SQRT_2_OVER_PI 0.7978845608028654
#define GELU_C 0.044715

template<typename T>
__global__ void gelu_kernel(T* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = data[idx];
        T x3 = x * x * x;
        T inner = T(GELU_SQRT_2_OVER_PI) * (x + T(GELU_C) * x3);
        data[idx] = T(0.5) * x * (T(1.0) + tanh(inner));
    }
}

template<typename T>
__global__ void relu_kernel(T* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmax(T(0.0), data[idx]);
    }
}

extern "C" int gelu_f64(double* h_data, int size, int* d_data) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(gelu_kernel<double>, grid_dim, block_dim, 0, dev_data, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int gelu_f32(float* h_data, int size, int* d_data) {
    float* dev_data = (float*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(gelu_kernel<float>, grid_dim, block_dim, 0, dev_data, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int relu_f64(double* h_data, int size, int* d_data) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(relu_kernel<double>, grid_dim, block_dim, 0, dev_data, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int relu_f32(float* h_data, int size, int* d_data) {
    float* dev_data = (float*)d_data;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(relu_kernel<float>, grid_dim, block_dim, 0, dev_data, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
