// attention_output.cu - Weighted sum kernel for attention output
#include "common.cu"

#define ATTN_BLOCK 256
#define TILE_THRESHOLD 64

__device__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Simple kernel for small cols: each thread computes full dot product for its output element
__global__ void attention_weighted_sum_small_kernel(
    const double* __restrict__ attn_weights,
    const double* __restrict__ values,
    double* __restrict__ output,
    int rows, int cols, int head_dim
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
        double sum = 0.0;
        for (int k = 0; k < cols; k++) {
            sum += attn_weights[row * cols + k] * values[k * head_dim + d];
        }
        output[row * head_dim + d] = sum;
    }
}

// Tiled kernel for large cols: cache values in shared memory for better bandwidth
__global__ void attention_weighted_sum_tiled_kernel(
    const double* __restrict__ attn_weights,
    const double* __restrict__ values,
    double* __restrict__ output,
    int rows, int cols, int head_dim
) {
    extern __shared__ double s_values[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    double acc[4] = {0.0, 0.0, 0.0, 0.0};

    for (int tile_start = 0; tile_start < cols; tile_start += ATTN_BLOCK) {
        int tile_end = (tile_start + ATTN_BLOCK < cols) ? tile_start + ATTN_BLOCK : cols;

        // Cooperative load: each thread loads values[tile + tid] for all d
        for (int k = tile_start + tid; k < tile_end; k += ATTN_BLOCK) {
            #pragma unroll
            for (int di = 0; di < 4; di++) {
                int d = di * (head_dim / 4);
                if (d < head_dim) s_values[tid * head_dim + d] = values[k * head_dim + d];
            }
        }
        __syncthreads();

        // Accumulate tile
        for (int k = tile_start; k < tile_end; k++) {
            double w = attn_weights[row * cols + k];
            int local_idx = k - tile_start;
            #pragma unroll
            for (int di = 0; di < 4; di++) {
                int d = di * (head_dim / 4);
                if (d < head_dim) acc[di] += w * s_values[local_idx * head_dim + d];
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int di = 0; di < 4; di++) {
        int d = di * (head_dim / 4);
        if (d < head_dim && tid + d < head_dim) output[row * head_dim + tid + d] = acc[di];
    }
}

extern "C" int attention_weighted_sum(
    double* h_attn, double* h_values, double* h_output,
    int rows, int cols, int head_dim,
    int* d_attn, int* d_values, int* d_output
) {
    const double* dev_attn = (const double*)d_attn;
    const double* dev_values = (const double*)d_values;
    double* dev_output = (double*)d_output;

    if (cols <= TILE_THRESHOLD) {
        attention_weighted_sum_small_kernel<<<dim3(rows), dim3(ATTN_BLOCK), 0, 0>>>(
            dev_attn, dev_values, dev_output, rows, cols, head_dim);
    } else {
        size_t shmem = ATTN_BLOCK * head_dim * sizeof(double);
        attention_weighted_sum_tiled_kernel<<<dim3(rows), dim3(ATTN_BLOCK), shmem, 0>>>(
            dev_attn, dev_values, dev_output, rows, cols, head_dim);
    }
    return (int)cudaPeekAtLastError();
}