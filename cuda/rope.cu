// rope.cu - Rotary Positional Encoding kernel
#include "common.cu"

#define ROPE_BLOCK 256

template<typename T>
__global__ void rope_kernel(
    T* __restrict__ data,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    int seq_len, int dim, int total_batches, int start_pos
) {
    int idx = get_global_thread_idx();
    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;

    if (idx >= total_pairs) return;

    int b = idx / pairs_per_batch;
    int remainder = idx % pairs_per_batch;
    int t = remainder / half_dim;
    int i = remainder % half_dim;

    int pos = start_pos + t;
    int cache_idx = pos * half_dim + i;
    int data_idx = b * (seq_len * dim) + t * dim + 2 * i;

    T x1 = data[data_idx];
    T x2 = data[data_idx + 1];
    T cos_val = cos_cache[cache_idx];
    T sin_val = sin_cache[cache_idx];

    data[data_idx] = x1 * cos_val - x2 * sin_val;
    data[data_idx + 1] = x1 * sin_val + x2 * cos_val;
}

template<typename T>
__global__ void rope_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    T* __restrict__ input_grad,
    int seq_len, int dim, int total_batches, int start_pos
) {
    int idx = get_global_thread_idx();
    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;

    if (idx >= total_pairs) return;

    int b = idx / pairs_per_batch;
    int remainder = idx % pairs_per_batch;
    int t = remainder / half_dim;
    int i = remainder % half_dim;

    int pos = start_pos + t;
    int cache_idx = pos * half_dim + i;
    int data_idx = b * (seq_len * dim) + t * dim + 2 * i;

    T g1 = grad_out[data_idx];
    T g2 = grad_out[data_idx + 1];
    T cos_val = cos_cache[cache_idx];
    T sin_val = sin_cache[cache_idx];

    input_grad[data_idx] += g1 * cos_val + g2 * sin_val;
    input_grad[data_idx + 1] += -g1 * sin_val + g2 * cos_val;
}

extern "C" int cuda_rope_f64(
    double* data, const double* cos_cache, const double* sin_cache,
    int seq_len, int dim, int total_batches, int start_pos,
    int* d_data, int* d_cos_cache, int* d_sin_cache
) {
    double* dev_data = (double*)d_data;
    const double* dev_cos = (const double*)d_cos_cache;
    const double* dev_sin = (const double*)d_sin_cache;
    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;
    dim3 grid_dim(compute_grid_1d(total_pairs, ROPE_BLOCK));
    dim3 block_dim(ROPE_BLOCK);
    rope_kernel<double><<<grid_dim, block_dim, 0, 0>>>(
        dev_data, dev_cos, dev_sin, seq_len, dim, total_batches, start_pos);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int cuda_rope_backward_f64(
    const double* h_grad_out, const double* h_cos_cache, const double* h_sin_cache,
    double* h_input_grad, int seq_len, int dim, int total_batches, int start_pos,
    int* d_grad_out, int* d_cos_cache, int* d_sin_cache, int* d_input_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_cos = (const double*)d_cos_cache;
    const double* dev_sin = (const double*)d_sin_cache;
    double* dev_input_grad = (double*)d_input_grad;
    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;
    dim3 grid_dim(compute_grid_1d(total_pairs, ROPE_BLOCK));
    dim3 block_dim(ROPE_BLOCK);
    rope_backward_kernel<double><<<grid_dim, block_dim, 0, 0>>>(
        dev_grad_out, dev_cos, dev_sin, dev_input_grad, seq_len, dim, total_batches, start_pos);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int cuda_rope_backward_f32(
    const float* h_grad_out, const float* h_cos_cache, const float* h_sin_cache,
    float* h_input_grad, int seq_len, int dim, int total_batches, int start_pos,
    int* d_grad_out, int* d_cos_cache, int* d_sin_cache, int* d_input_grad
) {
    const float* dev_grad_out = (const float*)d_grad_out;
    const float* dev_cos = (const float*)d_cos_cache;
    const float* dev_sin = (const float*)d_sin_cache;
    float* dev_input_grad = (float*)d_input_grad;
    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;
    dim3 grid_dim(compute_grid_1d(total_pairs, ROPE_BLOCK));
    dim3 block_dim(ROPE_BLOCK);
    rope_backward_kernel<float><<<grid_dim, block_dim, 0, 0>>>(
        dev_grad_out, dev_cos, dev_sin, dev_input_grad, seq_len, dim, total_batches, start_pos);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int cuda_rope_f32(
    float* data, const float* cos_cache, const float* sin_cache,
    int seq_len, int dim, int total_batches, int start_pos,
    int* d_data, int* d_cos_cache, int* d_sin_cache
) {
    float* dev_data = (float*)d_data;
    const float* dev_cos = (const float*)d_cos_cache;
    const float* dev_sin = (const float*)d_sin_cache;
    int half_dim = dim / 2;
    int pairs_per_batch = seq_len * half_dim;
    int total_pairs = total_batches * pairs_per_batch;
    dim3 grid_dim(compute_grid_1d(total_pairs, ROPE_BLOCK));
    dim3 block_dim(ROPE_BLOCK);
    rope_kernel<float><<<grid_dim, block_dim, 0, 0>>>(
        dev_data, dev_cos, dev_sin, seq_len, dim, total_batches, start_pos);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
