// softmax.cu - Fused softmax kernel
#include "common.cu"

// Softmax block size - must be power of 2 for tree reduction
#define SOFTMAX_BLOCK 256

//==============================================================================
// Softmax forward kernel (row-wise softmax) - arbitrary column width
// Input:  logits [rows, cols]
// Output: probs [rows, cols] (in-place)
// Performs: probs[i,j] = exp(logits[i,j]) / sum_k(exp(logits[i,k]))
// Uses max-shift for numerical stability
// One block per row, threads cooperate via shared memory tree reduction
// Supports any column count via strided access and block-level sync
//==============================================================================
__global__ void softmax_kernel(
    double* __restrict__ data,
    int rows, int cols
) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    // ---- Pass 1: Find row max ----
    // Each thread computes local max across its assigned elements
    double thread_max = -HUGE_VAL;
    for (int c = tid; c < cols; c += block_size) {
        double val = data[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Tree reduction for max within block
    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid + s] > sdata[tid]) ? sdata[tid + s] : sdata[tid];
        }
        __syncthreads();
    }
    // Final warp-level reduction (within a warp, no sync needed)
    double row_max;
    if (tid < 32) {
        row_max = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_xor_sync(0xffffffff, row_max, offset);
            row_max = (other > row_max) ? other : row_max;
        }
        sdata[0] = row_max; // Broadcast to shared memory
    }
    __syncthreads();
    row_max = sdata[0];

    // ---- Pass 2: Compute exp(x - row_max) and sum ----
    double thread_sum = 0.0;
    for (int c = tid; c < cols; c += block_size) {
        double val = exp(data[row * cols + c] - row_max);
        data[row * cols + c] = val;
        thread_sum += val;
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    double row_sum;
    if (tid < 32) {
        row_sum = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
        }
        sdata[0] = row_sum;
    }
    __syncthreads();
    row_sum = sdata[0];

    // ---- Pass 3: Normalize ----
    if (row_sum > 0.0) {
        for (int c = tid; c < cols; c += block_size) {
            data[row * cols + c] /= row_sum;
        }
    }
}

//==============================================================================
// Softmax with causal mask (for autoregressive attention)
// Mask: positions where col > row should be 0 (future positions)
// In-place operation on data array
// Supports arbitrary column counts via shared memory tree reduction
//==============================================================================
__global__ void softmax_causal_kernel(
    double* __restrict__ data,
    int rows, int cols
) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    // ---- Pass 1: Find row max (only valid positions col <= row) ----
    double thread_max = -HUGE_VAL;
    for (int c = tid; c < cols; c += block_size) {
        if (c <= row) {
            double val = data[row * cols + c];
            thread_max = (val > thread_max) ? val : thread_max;
        }
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid + s] > sdata[tid]) ? sdata[tid + s] : sdata[tid];
        }
        __syncthreads();
    }
    double row_max;
    if (tid < 32) {
        row_max = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_xor_sync(0xffffffff, row_max, offset);
            row_max = (other > row_max) ? other : row_max;
        }
        sdata[0] = row_max;
    }
    __syncthreads();
    row_max = sdata[0];

    // ---- Pass 2: Compute exp(x - row_max) for valid, 0 for masked, sum ----
    double thread_sum = 0.0;
    for (int c = tid; c < cols; c += block_size) {
        if (c <= row) {
            double val = exp(data[row * cols + c] - row_max);
            data[row * cols + c] = val;
            thread_sum += val;
        } else {
            data[row * cols + c] = 0.0;
        }
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    double row_sum;
    if (tid < 32) {
        row_sum = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
        }
        sdata[0] = row_sum;
    }
    __syncthreads();
    row_sum = sdata[0];

    // ---- Pass 3: Normalize ----
    if (row_sum > 0.0) {
        for (int c = tid; c < cols; c += block_size) {
            data[row * cols + c] /= row_sum;
        }
    }
}

//==============================================================================
// Log-softmax kernel
// log_softmax[i,j] = logits[i,j] - log(sum_k(exp(logits[i,k])))
// Supports arbitrary column counts via shared memory tree reduction
//==============================================================================
__global__ void log_softmax_kernel(
    const double* __restrict__ logits,
    double* __restrict__ out,
    int rows, int cols
) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    // ---- Pass 1: Find row max ----
    double thread_max = -HUGE_VAL;
    for (int c = tid; c < cols; c += block_size) {
        double val = logits[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid + s] > sdata[tid]) ? sdata[tid + s] : sdata[tid];
        }
        __syncthreads();
    }
    double row_max;
    if (tid < 32) {
        row_max = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_xor_sync(0xffffffff, row_max, offset);
            row_max = (other > row_max) ? other : row_max;
        }
        sdata[0] = row_max;
    }
    __syncthreads();
    row_max = sdata[0];

    // ---- Pass 2: Compute exp(x - max) and sum ----
    double thread_sum = 0.0;
    for (int c = tid; c < cols; c += block_size) {
        thread_sum += exp(logits[row * cols + c] - row_max);
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    double row_sum;
    if (tid < 32) {
        row_sum = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
        }
        sdata[0] = row_sum;
    }
    __syncthreads();
    row_sum = sdata[0];

    // ---- Pass 3: Write log-softmax values ----
    double log_sum = row_max + log(row_sum);
    for (int c = tid; c < cols; c += block_size) {
        out[row * cols + c] = logits[row * cols + c] - log_sum;
    }
}

//==============================================================================
// Host wrapper for softmax
//==============================================================================
extern "C" int softmax(
    double* data,
    int rows, int cols,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);

    softmax_kernel<<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

//==============================================================================
// Host wrapper for causal softmax
//==============================================================================
extern "C" int softmax_causal(
    double* data,
    int rows, int cols,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);

    softmax_causal_kernel<<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

//==============================================================================
// Host wrapper for log-softmax
//==============================================================================
extern "C" int log_softmax(
    const double* h_logits, double* h_out,
    int rows, int cols,
    int* d_logits, int* d_out
) {
    const double* dev_logits = (const double*)d_logits;
    double* dev_out = (double*)d_out;

    dim3 grid_dim(rows);
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);

    log_softmax_kernel<<<grid_dim, block_dim, shmem, 0>>>(
        dev_logits, dev_out, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
