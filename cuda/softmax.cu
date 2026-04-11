// softmax.cu - Fused softmax kernel
#include "common.cuh"

//==============================================================================
// Softmax forward kernel (row-wise softmax)
// Input:  logits [rows, cols]
// Output: probs [rows, cols]
// Performs: probs[i,j] = exp(logits[i,j]) / sum_k(exp(logits[i,k]))
// Uses max-shift for numerical stability
//==============================================================================
__global__ void softmax_kernel(
    double* __restrict__ data,
    int rows, int cols
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows) {
        // Step 1: Find max in this row (using shared memory reduction)
        __shared__ double max_vals[256];
        __shared__ double sum_vals[256];

        if (col < cols) {
            max_vals[col] = data[row * cols + col];
        } else {
            max_vals[col] = 0.0;  // Will be filtered out
        }

        // Load remaining cols if cols > 256
        for (int c = 256; c < cols; c += 256) {
            if (col < 256 && (c + col) < cols) {
                double val = data[row * cols + c + col];
                if (val > max_vals[col]) {
                    max_vals[col] = val;
                }
            }
        }

        __syncthreads();

        // Parallel reduction to find max
        int tid = col;
        if (tid < cols) {
            for (int s = 256 / 2; s > 0; s >>= 1) {
                if (tid + s < cols && tid < s) {
                    double v1 = max_vals[tid];
                    double v2 = max_vals[tid + s];
                    max_vals[tid] = (v1 > v2) ? v1 : v2;
                }
                __syncthreads();
            }
        }

        __syncthreads();

        double row_max = max_vals[0];

        // Step 2: Compute exp(x - max) and sum
        if (col < cols) {
            double val = exp(data[row * cols + col] - row_max);
            data[row * cols + col] = val;
            sum_vals[col] = val;
        } else {
            sum_vals[col] = 0.0;
        }

        __syncthreads();

        // Reduction for sum
        for (int s = 256 / 2; s > 0; s >>= 1) {
            if (tid + s < cols && tid < s) {
                sum_vals[tid] += sum_vals[tid + s];
            }
            __syncthreads();
        }

        double row_sum = sum_vals[0];

        // Step 3: Normalize
        if (col < cols) {
            data[row * cols + col] /= row_sum;
        }
    }
}

//==============================================================================
// Softmax with causal mask (for autoregressive attention)
// Mask: positions where col > row should be 0 (future positions)
//==============================================================================
__global__ void softmax_causal_kernel(
    double* __restrict__ data,
    int rows, int cols
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        // Apply causal mask - mask out future positions
        if (col > row) {
            data[row * cols + col] = 0.0;  // Future position
            return;
        }

        // Find max in this row (only considering unmasked positions)
        // We need to be careful about the mask here
    }

    // For simplicity, the causal mask is typically applied before softmax
    // by setting masked positions to -inf before calling softmax
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
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        // Reduction for max
        __shared__ double max_vals[256];

        if (col < cols) {
            max_vals[col] = logits[row * cols + col];
        }

        __syncthreads();

        // Parallel reduction
        for (int s = 128; s > 0; s >>= 1) {
            if (col < s && col + s < cols) {
                max_vals[col] = fmax(max_vals[col], max_vals[col + s]);
            }
            __syncthreads();
        }

        double row_max = max_vals[0];

        // Compute exp(x - max) and sum
        double sum_exp = 0.0;
        for (int c = 0; c < cols; c++) {
            sum_exp += exp(logits[row * cols + c] - row_max);
        }

        double log_sum = row_max + log(sum_exp);

        // Output log-softmax
        out[row * cols + col] = logits[row * cols + col] - log_sum;
    }
}

//==============================================================================
// Host wrapper for softmax
//==============================================================================
extern "C" void softmax(
    double* data,
    int rows, int cols,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    dim3 grid_dim(rows);  // One block per row
    dim3 block_dim(256);  // 256 threads per block

    CUDA_LAUNCH(softmax_kernel, grid_dim, block_dim, 0, dev_data, rows, cols);

    CUDA_CHECK(cudaPeekAtLastError());
}

//==============================================================================
// Host wrapper for log-softmax
//==============================================================================
extern "C" void log_softmax(
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

    CUDA_CHECK(cudaPeekAtLastError());
}
