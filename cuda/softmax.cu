// softmax.cu - Fused softmax kernel
#include "common.cu"

// Softmax block size - one block per row
#define SOFTMAX_BLOCK 256

//==============================================================================
// Warp-level reduction primitives (CUDA 9+, CC 7.0+)
// __reduce_max_sync / __reduce_add_sync perform butterfly reduction across
// the warp and return the result to all active threads.
// Falls back to manual __shfl_xor butterfly on older archs.
//==============================================================================
#if CUDA_ARCH_MAJOR >= 7
    #define WARP_MAX(val) __reduce_max_sync(0xffffffff, val)
    #define WARP_SUM(val) __reduce_add_sync(0xffffffff, val)
#else
    // Manual butterfly using __shfl_xor (works on all CC 3.0+)
    #define WARP_MAX(val) __warp_max_manual(val)
    #define WARP_SUM(val) __warp_sum_manual(val)
    __device__ double __warp_max_manual(double val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_xor_sync(0xffffffff, val, offset);
            val = (other > val) ? other : val;
        }
        return val;
    }
    __device__ double __warp_sum_manual(double val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        return val;
    }
#endif

//==============================================================================
// Softmax forward kernel (row-wise softmax) - arbitrary column width
// Input:  logits [rows, cols]
// Output: probs [rows, cols] (in-place)
// Performs: probs[i,j] = exp(logits[i,j]) / sum_k(exp(logits[i,k]))
// Uses max-shift for numerical stability
//
// Optimization summary:
// - Pass 1 (max): strided load into shared memory, tree reduction,
//   final warp-shuffle broadcast. Memory access is strided but all
//   threads participate equally (good GPU latency hiding).
// - Pass 2 (exp+sum): re-reads logits with exp, writes to output,
//   local sum accumulated per thread. Write-back uses contiguous
//   per-iteration pattern: each iteration writes block_size consecutive
//   doubles (coalesced within each group of block_size iterations).
// - Pass 3 (norm): in-place division with same access pattern as Pass 2.
// - Bank conflicts avoided: sequential shared memory access (stride=1).
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
    double thread_max = -HUGE_VAL;
    for (int c = tid; c < cols; c += block_size) {
        double val = data[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Tree reduction (stride-1 sequential → no bank conflict)
    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    // Warp-level reduction + broadcast
    double row_max;
    if (tid < 32) {
        row_max = WARP_MAX(sdata[tid]);
        if (tid == 0) sdata[0] = row_max;
    }
    __syncthreads();
    row_max = sdata[0];

    // ---- Pass 2: exp(x - max) + sum ----
    double thread_sum = 0.0;
    for (int base = 0; base < cols; base += block_size) {
        int c = base + tid;
        if (c < cols) {
            double val = exp(data[row * cols + c] - row_max);
            data[row * cols + c] = val;       // write-back (coalesced per iteration)
            sdata[tid] = val;                  // local accum in shared memory
        } else {
            sdata[tid] = 0.0;
        }
        __syncthreads();

        // Reduce within block (tree, stride-1)
        for (int s = block_size >> 1; s >= 1; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        thread_sum += sdata[0];
    }

    // Final warp-level sum reduction + broadcast
    double row_sum;
    if (tid < 32) {
        row_sum = (tid < block_size) ? thread_sum : 0.0;
        row_sum = WARP_SUM(row_sum);
        if (tid == 0) sdata[0] = row_sum;
    }
    __syncthreads();
    row_sum = sdata[0];

    // ---- Pass 3: Normalize ----
    if (row_sum > 0.0) {
        for (int base = 0; base < cols; base += block_size) {
            int c = base + tid;
            if (c < cols) {
                data[row * cols + c] /= row_sum;
            }
        }
    }
}

//==============================================================================
// Softmax kernel for small batches (rows < SM count)
// Uses 2D block: blockDim.x = 256 (cols), blockDim.y = 4 (multi-row per block)
// Each block processes 4 rows via grid-stride in blockIdx.x
// This improves GPU utilization when batch size is small (1-4 rows)
//==============================================================================
#define SMALL_BATCH_BLOCK_Y 4

__global__ void softmax_kernel_small_batch(
    double* __restrict__ data,
    int rows, int cols
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    // Rows per block = blockDim.y
    int rows_per_block = blockDim.y;
    // Grid-stride: each block processes rows_per_block rows, strided by gridDim.y
    int start_row = blockIdx.x * rows_per_block;
    int row_stride = gridDim.x * rows_per_block;

    for (int ri = 0; ri < rows_per_block; ri += 1) {
        int row = start_row + ri;
        if (row >= rows) continue;

        // ---- Pass 1: Find row max ----
        double thread_max = -HUGE_VAL;
        for (int c = tid; c < cols; c += block_size) {
            double val = data[row * cols + c];
            thread_max = (val > thread_max) ? val : thread_max;
        }
        sdata[tid] = thread_max;
        __syncthreads();

        for (int s = block_size >> 1; s > 32; s >>= 1) {
            if (tid < s) {
                sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
            }
            __syncthreads();
        }
        double row_max;
        if (tid < 32) {
            row_max = WARP_MAX(sdata[tid]);
            if (tid == 0) sdata[0] = row_max;
        }
        __syncthreads();
        row_max = sdata[0];

        // ---- Pass 2: exp(x - max) + sum ----
        double thread_sum = 0.0;
        for (int base = 0; base < cols; base += block_size) {
            int c = base + tid;
            if (c < cols) {
                double val = exp(data[row * cols + c] - row_max);
                data[row * cols + c] = val;
                sdata[tid] = val;
            } else {
                sdata[tid] = 0.0;
            }
            __syncthreads();

            for (int s = block_size >> 1; s >= 1; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            thread_sum += sdata[0];
        }

        double row_sum;
        if (tid < 32) {
            row_sum = (tid < block_size) ? thread_sum : 0.0;
            row_sum = WARP_SUM(row_sum);
            if (tid == 0) sdata[0] = row_sum;
        }
        __syncthreads();
        row_sum = sdata[0];

        // ---- Pass 3: Normalize ----
        if (row_sum > 0.0) {
            for (int base = 0; base < cols; base += block_size) {
                int c = base + tid;
                if (c < cols) {
                    data[row * cols + c] /= row_sum;
                }
            }
        }
    }
}

//==============================================================================
// Softmax with causal mask (for autoregressive attention)
// Mask: positions where col > row should be 0 (future positions)
// In-place operation on data array
//
// Optimization: precompute row_limit = row once, eliminating repeated
// conditionals inside hot loops.
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

    // Precompute causal boundary once (Fix 3: eliminates repeated conditionals)
    int row_limit = (row < cols - 1) ? row : cols - 1;

    // ---- Pass 1: Find row max (only valid positions col <= row) ----
    double thread_max = -HUGE_VAL;
    for (int c = tid; c <= row_limit; c += block_size) {
        double val = data[row * cols + c];
        thread_max = (val > thread_max) ? val : thread_max;
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (int s = block_size >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    double row_max;
    if (tid < 32) {
        row_max = WARP_MAX(sdata[tid]);
        if (tid == 0) sdata[0] = row_max;
    }
    __syncthreads();
    row_max = sdata[0];

    // ---- Pass 2: exp(x - max) + sum (causal mask) ----
    // Only iterate over valid positions (col <= row_limit)
    double thread_sum = 0.0;
    for (int base = 0; base <= row_limit; base += block_size) {
        int c = base + tid;
        if (c <= row_limit && c < cols) {
            double val = exp(data[row * cols + c] - row_max);
            data[row * cols + c] = val;
            sdata[tid] = val;
        } else {
            sdata[tid] = 0.0;
        }
        __syncthreads();

        for (int s = block_size >> 1; s >= 1; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        thread_sum += sdata[0];
    }

    double row_sum;
    if (tid < 32) {
        row_sum = (tid < block_size) ? thread_sum : 0.0;
        row_sum = WARP_SUM(row_sum);
        if (tid == 0) sdata[0] = row_sum;
    }
    __syncthreads();
    row_sum = sdata[0];

    // ---- Pass 3: Normalize (only valid positions col <= row_limit) ----
    if (row_sum > 0.0) {
        for (int base = 0; base <= row_limit; base += block_size) {
            int c = base + tid;
            if (c <= row_limit && c < cols) {
                data[row * cols + c] /= row_sum;
            }
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
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    double row_max;
    if (tid < 32) {
        row_max = WARP_MAX(sdata[tid]);
        if (tid == 0) sdata[0] = row_max;
    }
    __syncthreads();
    row_max = sdata[0];

    // ---- Pass 2: Compute exp(x - max) and sum ----
    double thread_sum = 0.0;
    for (int base = 0; base < cols; base += block_size) {
        int c = base + tid;
        if (c < cols) {
            sdata[tid] = exp(logits[row * cols + c] - row_max);
        } else {
            sdata[tid] = 0.0;
        }
        __syncthreads();

        for (int s = block_size >> 1; s >= 1; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        thread_sum += sdata[0];
    }

    double row_sum;
    if (tid < 32) {
        row_sum = (tid < block_size) ? thread_sum : 0.0;
        row_sum = WARP_SUM(row_sum);
        if (tid == 0) sdata[0] = row_sum;
    }
    __syncthreads();
    row_sum = sdata[0];

    // ---- Pass 3: Write log-softmax values ----
    double log_sum = row_max + log(row_sum);
    for (int base = 0; base < cols; base += block_size) {
        int c = base + tid;
        if (c < cols) {
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
    dim3 block_dim(SOFTMAX_BLOCK);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);

    softmax_kernel<<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

//==============================================================================
// Host wrapper for softmax (small batch, rows < SM count)
// Uses 2D block: blockDim.x=256, blockDim.y=SMALL_BATCH_BLOCK_Y
// Grid is sized so total blocks >= rows, with rows-per-block=Y dimension
//==============================================================================
extern "C" int softmax_small_batch(
    double* data,
    int rows, int cols,
    int* d_data
) {
    double* dev_data = (double*)d_data;
    // blocks_y = rows (one block per row conceptually), but with blockDim.y
    // threads cooperating per block, we need ceil(rows / SMALL_BATCH_BLOCK_Y) blocks
    int grid_rows = (rows + SMALL_BATCH_BLOCK_Y - 1) / SMALL_BATCH_BLOCK_Y;
    dim3 grid_dim(grid_rows);
    dim3 block_dim(SOFTMAX_BLOCK, SMALL_BATCH_BLOCK_Y);
    size_t shmem = SOFTMAX_BLOCK * sizeof(double);

    softmax_kernel_small_batch<<<grid_dim, block_dim, shmem, 0>>>(dev_data, rows, cols);
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
