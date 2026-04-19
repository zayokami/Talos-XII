// softmax.cu - Fused softmax kernel
#include "common.cu"

//==============================================================================
// Softmax forward kernel (row-wise softmax)
// Input:  logits [rows, cols]
// Output: probs [rows, cols] (in-place)
// Performs: probs[i,j] = exp(logits[i,j]) / sum_k(exp(logits[i,k]))
// Uses max-shift for numerical stability
// One block per row, threads cooperate via warp shuffle reduction
//==============================================================================
__global__ void softmax_kernel(
    double* __restrict__ data,
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    // Step 1: Find row max using warp shuffle reduction
    // Each thread loads elements at stride block_size
    double thread_max = -HUGE_VAL;

    for (int c = tid; c < cols; c += block_size) {
        double val = data[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    // Warp-level reduction using __shfl_xor_sync
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xffffffff, thread_max, offset);
        thread_max = (other > thread_max) ? other : thread_max;
    }

    double row_max = thread_max;

    // Step 2: Compute exp(x - row_max) and sum using warp shuffle
    double thread_sum = 0.0;
    for (int c = tid; c < cols; c += block_size) {
        double val = exp(data[row * cols + c] - row_max);
        data[row * cols + c] = val;
        thread_sum += val;
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xffffffff, thread_sum, offset);
        thread_sum += other;
    }

    double row_sum = thread_sum;

    // Step 3: Normalize - one thread writes all values
    if (tid == 0 && row_sum > 0.0) {
        for (int c = 0; c < cols; c++) {
            data[row * cols + c] /= row_sum;
        }
    }
}

//==============================================================================
// Softmax with causal mask (for autoregressive attention)
// Mask: positions where col > row should be 0 (future positions)
// In-place operation on data array
//==============================================================================
__global__ void softmax_causal_kernel(
    double* __restrict__ data,
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    // Step 1: Find row max considering only valid (non-masked) positions
    // For causal mask, valid positions are col <= row
    double thread_max = -HUGE_VAL;

    for (int c = tid; c < cols; c += block_size) {
        if (c <= row) {
            double val = data[row * cols + c];
            thread_max = (val > thread_max) ? val : thread_max;
        }
    }

    // Warp-level reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xffffffff, thread_max, offset);
        thread_max = (other > thread_max) ? other : thread_max;
    }

    double row_max = thread_max;

    // Step 2: Compute exp(x - row_max) for valid positions, 0 for masked
    // Also compute thread-local sum
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

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xffffffff, thread_sum, offset);
        thread_sum += other;
    }

    double row_sum = thread_sum;

    // Step 3: Normalize - one thread writes all values
    if (tid == 0 && row_sum > 0.0) {
        for (int c = 0; c < cols; c++) {
            data[row * cols + c] /= row_sum;
        }
    }
}

//==============================================================================
// Log-softmax kernel
// log_softmax[i,j] = logits[i,j] - log(sum_k(exp(logits[i,k])))
//==============================================================================
__global__ void log_softmax_kernel(
    const double* __restrict__ logits,
    double* __restrict__ out,
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= rows) return;

    // Step 1: Find row max using warp shuffle
    double thread_max = -HUGE_VAL;
    for (int c = tid; c < cols; c += block_size) {
        double val = logits[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xffffffff, thread_max, offset);
        thread_max = (other > thread_max) ? other : thread_max;
    }

    double row_max = thread_max;

    // Step 2: Compute exp(x - max) and sum
    double thread_sum = 0.0;
    for (int c = tid; c < cols; c += block_size) {
        thread_sum += exp(logits[row * cols + c] - row_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xffffffff, thread_sum, offset);
        thread_sum += other;
    }

    double row_sum = thread_sum;
    double log_sum = row_max + log(row_sum);

    // Step 3: Write log-softmax values
    if (tid == 0) {
        for (int c = 0; c < cols; c++) {
            out[row * cols + c] = logits[row * cols + c] - log_sum;
        }
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
    dim3 block_dim(256);

    CUDA_LAUNCH(softmax_kernel, grid_dim, block_dim, 0, dev_data, rows, cols);
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
    dim3 block_dim(256);

    CUDA_LAUNCH(softmax_causal_kernel, grid_dim, block_dim, 0, dev_data, rows, cols);
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
    dim3 block_dim(256);

    CUDA_LAUNCH(log_softmax_kernel, grid_dim, block_dim, 0,
        dev_logits, dev_out, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
