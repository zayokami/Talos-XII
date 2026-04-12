// common.cu - Common CUDA utilities and macros
#ifndef COMMON_CU
#define COMMON_CU

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// CUDA check macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// cuBLAS check macro
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: code %d\n",               \
                    __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel launch macro with preferred grid size
#define CUDA_LAUNCH(kernel, grid, block, stream, ...)                         \
    kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)

// Host function to check GPU availability
inline int get_gpu_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

// Host function to get GPU name
inline void get_gpu_name(int device_id, char* name, int max_len) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    strncpy(name, prop.name, max_len - 1);
    name[max_len - 1] = '\0';
}

// Host function to allocate device memory
template<typename T>
inline T* device_alloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&ptr, count * sizeof(T)));
    return ptr;
}

// Host function to free device memory
template<typename T>
inline void device_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Host function to copy data H2D
template<typename T>
inline void copy_h2d(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
}

// Host function to copy data D2H
template<typename T>
inline void copy_d2h(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
}

// Host function to copy data D2D
template<typename T>
inline void copy_d2d(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream));
}

// Device function for getting thread index
__device__ inline int get_thread_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Device function for getting block index
__device__ inline int get_block_idx() {
    return blockIdx.x;
}

// Device function for getting global thread index
__device__ inline int get_global_thread_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Compute grid size for given data size
inline dim3 compute_grid_1d(size_t n, size_t block_size = 256) {
    dim3 grid;
    grid.x = (n + block_size - 1) / block_size;
    grid.y = 1;
    grid.z = 1;
    return grid;
}

// Compute grid size for 2D data
inline dim3 compute_grid_2d(size_t rows, size_t cols, size_t block_rows = 16, size_t block_cols = 16) {
    dim3 grid;
    grid.x = (cols + block_cols - 1) / block_cols;
    grid.y = (rows + block_rows - 1) / block_rows;
    grid.z = 1;
    return grid;
}

#endif // COMMON_CU
