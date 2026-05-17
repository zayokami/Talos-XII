// softmax.cu - Fused softmax kernel
#include "common.cu"

#define SOFTMAX_BLOCK 256

template<typename T>
__forceinline__ __device__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_xor_sync(0xffffffff, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

template<typename T>
__forceinline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

#define WARP_MAX(val) warp_reduce_max(val)
#define WARP_SUM(val) warp_reduce_sum(val)

template<typename T>
__global__ void softmax_kernel(T* __restrict__ data, int rows, int cols) {
    extern __shared__ char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    T thread_max = neg_inf<T>();
    for (int c = tid; c < cols; c += block_size) {
        T val = data[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    T warp_max = WARP_MAX(thread_max);
    sdata[tid] = (tid < 32) ? warp_max : neg_inf<T>();
    __syncthreads();

    for (int s = 4; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    T row_max = sdata[0];

    T thread_sum = T(0.0);
    for (int base = 0; base < cols; base += block_size) {
        int c = base + tid;
        if (c < cols) {
            T val = exp(data[row * cols + c] - row_max);
            data[row * cols + c] = val;
            thread_sum += val;
        }
    }

    T warp_sum = WARP_SUM(thread_sum);
    sdata[tid] = (tid < 32) ? warp_sum : T(0.0);
    __syncthreads();

    for (int s = 4; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    T row_sum = sdata[0];

    if (row_sum > T(0.0)) {
        for (int base = 0; base < cols; base += block_size) {
            int c = base + tid;
            if (c < cols) {
                data[row * cols + c] /= row_sum;
            }
        }
    }
}

#define SMALL_BATCH_BLOCK_Y 4

template<typename T>
__global__ void softmax_kernel_small_batch(T* __restrict__ data, int rows, int cols) {
    extern __shared__ char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int rows_per_block = blockDim.y;
    int start_row = blockIdx.x * rows_per_block;

    for (int ri = 0; ri < rows_per_block; ri++) {
        int row = start_row + ri;
        if (row >= rows) continue;

        T thread_max = neg_inf<T>();
        for (int c = tid; c < cols; c += block_size) {
            T val = data[row * cols + c];
            thread_max = (val > thread_max) ? val : thread_max;
        }

        T warp_max = WARP_MAX(thread_max);
        sdata[tid] = (tid < 32) ? warp_max : neg_inf<T>();
        __syncthreads();

        for (int s = 4; s >= 1; s >>= 1) {
            if (tid < s) {
                sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
            }
            __syncthreads();
        }
        T row_max = sdata[0];

        T thread_sum = T(0.0);
        for (int base = 0; base < cols; base += block_size) {
            int c = base + tid;
            if (c < cols) {
                T val = exp(data[row * cols + c] - row_max);
                data[row * cols + c] = val;
                thread_sum += val;
            }
        }

        T warp_sum = WARP_SUM(thread_sum);
        sdata[tid] = (tid < 32) ? warp_sum : T(0.0);
        __syncthreads();

        for (int s = 4; s >= 1; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        T row_sum = sdata[0];

        if (row_sum > T(0.0)) {
            for (int base = 0; base < cols; base += block_size) {
                int c = base + tid;
                if (c < cols) {
                    data[row * cols + c] /= row_sum;
                }
            }
        }
    }
}

template<typename T>
__global__ void softmax_causal_kernel(T* __restrict__ data, int rows, int cols) {
    extern __shared__ char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    int row_limit = (row < cols - 1) ? row : cols - 1;

    T thread_max = neg_inf<T>();
    for (int c = tid; c <= row_limit; c += block_size) {
        T val = data[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    T warp_max = WARP_MAX(thread_max);
    sdata[tid] = (tid < 32) ? warp_max : neg_inf<T>();
    __syncthreads();

    for (int s = 4; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    T row_max = sdata[0];

    T thread_sum = T(0.0);
    for (int base = 0; base <= row_limit; base += block_size) {
        int c = base + tid;
        if (c <= row_limit && c < cols) {
            T val = exp(data[row * cols + c] - row_max);
            data[row * cols + c] = val;
            thread_sum += val;
        }
    }

    T warp_sum = WARP_SUM(thread_sum);
    sdata[tid] = (tid < 32) ? warp_sum : T(0.0);
    __syncthreads();

    for (int s = 4; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    T row_sum = sdata[0];

    if (row_sum > T(0.0)) {
        for (int base = 0; base <= row_limit; base += block_size) {
            int c = base + tid;
            if (c <= row_limit && c < cols) {
                data[row * cols + c] /= row_sum;
            }
        }
    }
}

template<typename T>
__global__ void log_softmax_kernel(const T* __restrict__ logits, T* __restrict__ out, int rows, int cols) {
    extern __shared__ char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    T thread_max = neg_inf<T>();
    for (int c = tid; c < cols; c += block_size) {
        T val = logits[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    T warp_max = WARP_MAX(thread_max);
    sdata[tid] = (tid < 32) ? warp_max : neg_inf<T>();
    __syncthreads();

    for (int s = 4; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    T row_max = sdata[0];

    T thread_sum = T(0.0);
    for (int base = 0; base < cols; base += block_size) {
        int c = base + tid;
        if (c < cols) {
            thread_sum += exp(logits[row * cols + c] - row_max);
        }
    }

    T warp_sum = WARP_SUM(thread_sum);
    sdata[tid] = (tid < 32) ? warp_sum : T(0.0);
    __syncthreads();

    for (int s = 4; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    T row_sum = sdata[0];

    T log_sum = row_max + log(row_sum);
    for (int base = 0; base < cols; base += block_size) {
        int c = base + tid;
        if (c < cols) {
            out[row * cols + c] = logits[row * cols + c] - log_sum;
        }
    }
}

extern "C" int softmax_f64(double* data, int rows, int cols, int* d_data) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);
    softmax_kernel<double><<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int softmax_f32(float* data, int rows, int cols, int* d_data) {
    float* dev_data = (float*)d_data;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(float);
    softmax_kernel<float><<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int softmax_small_batch_f64(double* data, int rows, int cols, int* d_data) {
    double* dev_data = (double*)d_data;
    int grid_rows = (rows + SMALL_BATCH_BLOCK_Y - 1) / SMALL_BATCH_BLOCK_Y;
    dim3 grid_dim(grid_rows);
    dim3 block_dim(SOFTMAX_BLOCK, SMALL_BATCH_BLOCK_Y);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);
    softmax_kernel_small_batch<double><<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int softmax_small_batch_f32(float* data, int rows, int cols, int* d_data) {
    float* dev_data = (float*)d_data;
    int grid_rows = (rows + SMALL_BATCH_BLOCK_Y - 1) / SMALL_BATCH_BLOCK_Y;
    dim3 grid_dim(grid_rows);
    dim3 block_dim(SOFTMAX_BLOCK, SMALL_BATCH_BLOCK_Y);
    size_t shmem = SOFTMAX_BLOCK * sizeof(float);
    softmax_kernel_small_batch<float><<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int softmax_causal_f64(double* data, int rows, int cols, int* d_data) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);
    softmax_causal_kernel<double><<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int softmax_causal_f32(float* data, int rows, int cols, int* d_data) {
    float* dev_data = (float*)d_data;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(float);
    softmax_causal_kernel<float><<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int log_softmax_f64(const double* h_logits, double* h_out, int rows, int cols, int* d_logits, int* d_out) {
    const double* dev_logits = (const double*)d_logits;
    double* dev_out = (double*)d_out;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);
    log_softmax_kernel<double><<<grid_dim, block_dim, shmem, 0>>>(dev_logits, dev_out, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int log_softmax_f32(const float* h_logits, float* h_out, int rows, int cols, int* d_logits, int* d_out) {
    const float* dev_logits = (const float*)d_logits;
    float* dev_out = (float*)d_out;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(float);
    log_softmax_kernel<float><<<grid_dim, block_dim, shmem, 0>>>(dev_logits, dev_out, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
