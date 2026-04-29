// rope.cu - Rotary Positional Encoding kernel
#include "common.cu"

#define ROPE_BLOCK 256

//==============================================================================
// RoPE kernel - applies rotary positional encoding to input tensor
// Input:  data [Batch, Seq, Heads, Dim] or [Batch, Seq, Dim] in batch-major layout
//         cos_cache, sin_cache: precomputed caches indexed by pos * (dim/2) + i
//         seq_len: sequence length per batch
//         dim: feature dimension (must be even)
//         total_batches: number of batches in the data
//         start_pos: starting position offset for inference
//
// Each thread handles one rotation pair (2 consecutive elements).
// Rotation: x[2i] = x[2i]*cos - x[2i+1]*sin; x[2i+1] = x[2i]*sin + x[2i+1]*cos
//==============================================================================
__global__ void rope_kernel(
    double* __restrict__ data,
    const double* __restrict__ cos_cache,
    const double* __restrict__ sin_cache,
    int seq_len,
    int dim,
    int total_batches,
    int start_pos
) {
    int idx = get_global_thread_idx();
    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;

    if (idx >= total_pairs) return;

    // Decode thread index to (batch, position, pair_index)
    int b = idx / pairs_per_batch;
    int remainder = idx % pairs_per_batch;
    int t = remainder / half_dim;
    int i = remainder % half_dim;

    int pos = start_pos + t;

    // Cache lookup: pos * (dim/2) + i
    int cache_idx = pos * half_dim + i;

    // Data index: b * (seq_len * dim) + t * dim + 2*i
    int data_idx = b * (seq_len * dim) + t * dim + 2 * i;

    double x1 = data[data_idx];
    double x2 = data[data_idx + 1];
    double cos_val = cos_cache[cache_idx];
    double sin_val = sin_cache[cache_idx];

    // Apply rotation
    data[data_idx] = x1 * cos_val - x2 * sin_val;
    data[data_idx + 1] = x1 * sin_val + x2 * cos_val;
}

//==============================================================================
// Host wrapper for RoPE
//==============================================================================
extern "C" int cuda_rope(
    double* data,
    const double* cos_cache,
    const double* sin_cache,
    int seq_len,
    int dim,
    int total_batches,
    int start_pos,
    int* d_data,
    int* d_cos_cache,
    int* d_sin_cache
) {
    double* dev_data = (double*)d_data;
    const double* dev_cos = (const double*)d_cos_cache;
    const double* dev_sin = (const double*)d_sin_cache;

    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;

    dim3 grid_dim(compute_grid_1d(total_pairs, ROPE_BLOCK));
    dim3 block_dim(ROPE_BLOCK);

    rope_kernel<<<grid_dim, block_dim, 0, 0>>>(
        dev_data, dev_cos, dev_sin, seq_len, dim, total_batches, start_pos);

    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
