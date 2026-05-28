// tensor_ops.cu - Generic CUDA tensor layout and elementwise utility kernels
#include "common.cu"

#define TENSOR_OP_BLOCK 256

template<typename T>
__global__ void scale_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    T scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scale;
    }
}

template<typename T>
__global__ void fill_kernel(T* __restrict__ data, T value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

template<typename T>
__global__ void sumsq_accum_kernel(
    const T* __restrict__ input,
    T* __restrict__ accum,
    int size
) {
    __shared__ T partial[TENSOR_OP_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T local = T(0.0);
    if (idx < size) {
        T v = input[idx];
        local = v * v;
    }
    partial[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(accum, partial[0]);
    }
}

template<typename T>
__global__ void sum_accum_kernel(
    const T* __restrict__ input,
    T* __restrict__ accum,
    int size,
    T scale
) {
    __shared__ T partial[TENSOR_OP_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T local = T(0.0);
    if (idx < size) {
        local = input[idx];
    }
    partial[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(accum, partial[0] * scale);
    }
}

template<typename T>
__global__ void clip_coef_kernel(
    const T* __restrict__ sumsq,
    T* __restrict__ coef,
    T max_norm,
    T eps
) {
    T norm = sqrt(sumsq[0]);
    coef[0] = norm > max_norm ? max_norm / (norm + eps) : T(1.0);
}

template<typename T>
__global__ void scale_inplace_by_scalar_kernel(
    T* __restrict__ data,
    const T* __restrict__ scalar,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scalar[0];
    }
}

template<typename T>
__global__ void add_scalar_kernel(
    T* __restrict__ data,
    const T* __restrict__ scalar,
    T scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += scalar[0] * scale;
    }
}

template<typename T>
__global__ void lerp_inplace_kernel(
    T* __restrict__ target,
    const T* __restrict__ source,
    T tau,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        target[idx] = target[idx] * (T(1.0) - tau) + source[idx] * tau;
    }
}

template<typename T>
__global__ void double_dqn_target_kernel(
    const T* __restrict__ q_next_eval,
    const T* __restrict__ q_next_target,
    const T* __restrict__ rewards,
    const T* __restrict__ dones,
    T* __restrict__ targets,
    int batch,
    int actions,
    T gamma
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;

    int base = row * actions;
    int max_idx = 0;
    T max_val = q_next_eval[base];
    for (int action = 1; action < actions; action++) {
        T val = q_next_eval[base + action];
        if (val > max_val) {
            max_val = val;
            max_idx = action;
        }
    }

    T next_q = q_next_target[base + max_idx];
    targets[row] = rewards[row] + gamma * next_q * (T(1.0) - dones[row]);
}

template<typename T>
__global__ void abs_diff_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T diff = a[idx] - b[idx];
        out[idx] = diff < T(0.0) ? -diff : diff;
    }
}

template<typename T>
__global__ void select_last_token_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch,
    int seq,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;
    if (idx >= total) return;

    int b = idx / dim;
    int d = idx % dim;
    int in_idx = (b * seq + (seq - 1)) * dim + d;
    output[idx] = input[in_idx];
}

template<typename T>
__global__ void select_last_token_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int batch,
    int seq,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;
    if (idx >= total) return;

    int b = idx / dim;
    int d = idx % dim;
    int in_idx = (b * seq + (seq - 1)) * dim + d;
    input_grad[in_idx] += grad_out[idx];
}

template<typename T>
__global__ void index_select_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int idx
) {
    output[0] = input[idx];
}

template<typename T>
__global__ void index_select_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int idx
) {
    input_grad[idx] += grad_out[0];
}

template<typename T>
__global__ void argmax_kernel(
    const T* __restrict__ input,
    int* __restrict__ out_idx,
    int size
) {
    __shared__ T partial_vals[TENSOR_OP_BLOCK];
    __shared__ int partial_idxs[TENSOR_OP_BLOCK];
    int tid = threadIdx.x;
    T best = neg_inf<T>();
    int best_idx = 0;

    for (int idx = tid; idx < size; idx += blockDim.x) {
        T v = input[idx];
        if (v > best || (v == best && idx < best_idx)) {
            best = v;
            best_idx = idx;
        }
    }

    partial_vals[tid] = best;
    partial_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            T other_val = partial_vals[tid + stride];
            int other_idx = partial_idxs[tid + stride];
            if (other_val > partial_vals[tid] ||
                (other_val == partial_vals[tid] && other_idx < partial_idxs[tid])) {
                partial_vals[tid] = other_val;
                partial_idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[0] = partial_idxs[0];
    }
}

template<typename T>
__global__ void exp_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = exp(input[idx]);
    }
}

template<typename T>
__global__ void exp_backward_kernel(
    const T* __restrict__ exp_out,
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_grad[idx] += grad_out[idx] * exp_out[idx];
    }
}

template<typename T>
__global__ void weighted_mse_loss_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    const T* __restrict__ weights,
    T* __restrict__ loss_out,
    T* __restrict__ weight_sum_out,
    int size
) {
    __shared__ T partial_loss[TENSOR_OP_BLOCK];
    __shared__ T partial_weight[TENSOR_OP_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T loss = T(0.0);
    T weight_sum = T(0.0);
    if (idx < size) {
        T w = weights[idx];
        T diff = pred[idx] - target[idx];
        loss = w * diff * diff;
        weight_sum = w;
    }
    partial_loss[tid] = loss;
    partial_weight[tid] = weight_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_loss[tid] += partial_loss[tid + stride];
            partial_weight[tid] += partial_weight[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_out, partial_loss[0]);
        atomicAdd(weight_sum_out, partial_weight[0]);
    }
}

template<typename T>
__global__ void weighted_mse_finalize_kernel(
    T* __restrict__ loss_out,
    const T* __restrict__ weight_sum_out
) {
    T denom = weight_sum_out[0];
    if (denom < T(0.0)) {
        denom = -denom;
    }
    if (denom < T(1e-12)) {
        denom = T(1.0);
    } else {
        denom = weight_sum_out[0];
    }
    loss_out[0] = loss_out[0] / denom;
}

template<typename T>
__global__ void weighted_mse_backward_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    const T* __restrict__ weights,
    const T* __restrict__ weight_sum,
    const T* __restrict__ grad_out,
    T* __restrict__ pred_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    T denom = weight_sum[0];
    if (denom < T(0.0)) {
        denom = -denom;
    }
    if (denom < T(1e-12)) {
        denom = T(1.0);
    } else {
        denom = weight_sum[0];
    }
    T g = grad_out[0] * T(2.0) * weights[idx] * (pred[idx] - target[idx]) / denom;
    pred_grad[idx] += g;
}

template<typename T>
__global__ void scale_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    T scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_grad[idx] += grad_out[idx] * scale;
    }
}

template<typename T>
__global__ void causal_mask_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batches,
    int seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batches * seq * seq;
    if (idx >= total) return;

    int local = idx % (seq * seq);
    int row = local / seq;
    int col = local % seq;
    output[idx] = (col <= row) ? input[idx] : neg_inf<T>();
}

template<typename T>
__global__ void causal_mask_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int batches,
    int seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batches * seq * seq;
    if (idx >= total) return;

    int local = idx % (seq * seq);
    int row = local / seq;
    int col = local % seq;
    if (col <= row) {
        input_grad[idx] += grad_out[idx];
    }
}

template<typename T>
__global__ void concat_last_dim_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int rows,
    int a_dim,
    int b_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = a_dim + b_dim;
    int total = rows * stride;
    if (idx >= total) return;

    int col = idx % stride;
    int row = idx / stride;
    if (col < a_dim) {
        out[idx] = a[row * a_dim + col];
    } else {
        out[idx] = b[row * b_dim + (col - a_dim)];
    }
}

template<typename T>
__global__ void concat_last_dim_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ a_grad,
    T* __restrict__ b_grad,
    int rows,
    int a_dim,
    int b_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = a_dim + b_dim;
    int total = rows * stride;
    if (idx >= total) return;

    int col = idx % stride;
    int row = idx / stride;
    if (col < a_dim) {
        a_grad[row * a_dim + col] += grad_out[idx];
    } else {
        b_grad[row * b_dim + (col - a_dim)] += grad_out[idx];
    }
}

template<typename T>
__global__ void split_last_dim_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows,
    int input_dim,
    int part_dim,
    int part_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * part_dim;
    if (idx >= total) return;

    int row = idx / part_dim;
    int col = idx % part_dim;
    output[idx] = input[row * input_dim + part_idx * part_dim + col];
}

template<typename T>
__global__ void split_last_dim_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int rows,
    int input_dim,
    int part_dim,
    int part_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * part_dim;
    if (idx >= total) return;

    int row = idx / part_dim;
    int col = idx % part_dim;
    input_grad[row * input_dim + part_idx * part_dim + col] += grad_out[idx];
}

template<typename T>
__global__ void broadcast_batch_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int inner_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * inner_len;
    if (idx < total) {
        output[idx] = input[idx % inner_len];
    }
}

template<typename T>
__global__ void broadcast_batch_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int batch_size,
    int inner_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inner_len) return;

    T sum = T(0.0);
    for (int b = 0; b < batch_size; b++) {
        sum += grad_out[b * inner_len + idx];
    }
    input_grad[idx] += sum;
}

template<typename T>
__global__ void transpose_last_two_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outer,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * rows * cols;
    if (idx >= total) return;

    int local = idx % (rows * cols);
    int batch = idx / (rows * cols);
    int out_r = local / rows;
    int out_c = local % rows;
    output[idx] = input[batch * rows * cols + out_c * cols + out_r];
}

template<typename T>
__global__ void transpose_4d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int d0,
    int d1,
    int d2,
    int d3,
    int dim0,
    int dim1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (idx >= total) return;

    int shape[4] = {d0, d1, d2, d3};
    int out_shape[4] = {d0, d1, d2, d3};
    int tmp = out_shape[dim0];
    out_shape[dim0] = out_shape[dim1];
    out_shape[dim1] = tmp;

    int coords[4];
    int rem = idx;
    coords[0] = rem / (out_shape[1] * out_shape[2] * out_shape[3]);
    rem %= out_shape[1] * out_shape[2] * out_shape[3];
    coords[1] = rem / (out_shape[2] * out_shape[3]);
    rem %= out_shape[2] * out_shape[3];
    coords[2] = rem / out_shape[3];
    coords[3] = rem % out_shape[3];

    tmp = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = tmp;

    int old_idx = ((coords[0] * shape[1] + coords[1]) * shape[2] + coords[2]) * shape[3] + coords[3];
    output[idx] = input[old_idx];
}

template<typename T>
__global__ void batched_qk_scores_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    T* __restrict__ out,
    int batches,
    int seq,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batches * seq * seq;
    if (idx >= total) return;

    int col = idx % seq;
    int row = (idx / seq) % seq;
    int batch = idx / (seq * seq);
    int base = batch * seq * dim;
    T sum = T(0.0);
    for (int d = 0; d < dim; d++) {
        sum += q[base + row * dim + d] * k[base + col * dim + d];
    }
    out[idx] = sum;
}

template<typename T>
__global__ void batched_qk_scores_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ q,
    const T* __restrict__ k,
    T* __restrict__ q_grad,
    T* __restrict__ k_grad,
    int batches,
    int seq,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batches * seq * dim;
    if (idx >= total) return;

    int d = idx % dim;
    int row = (idx / dim) % seq;
    int batch = idx / (seq * dim);
    int q_base = batch * seq * dim;
    int out_base = batch * seq * seq;

    T q_sum = T(0.0);
    for (int j = 0; j < seq; j++) {
        q_sum += grad_out[out_base + row * seq + j] * k[q_base + j * dim + d];
    }
    q_grad[idx] += q_sum;

    T k_sum = T(0.0);
    for (int i = 0; i < seq; i++) {
        k_sum += grad_out[out_base + i * seq + row] * q[q_base + i * dim + d];
    }
    k_grad[idx] += k_sum;
}

template<typename T>
__global__ void attention_weighted_sum_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ probs,
    const T* __restrict__ values,
    T* __restrict__ probs_grad,
    T* __restrict__ values_grad,
    int batches,
    int seq,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_p = batches * seq * seq;
    int total_v = batches * seq * head_dim;

    if (idx < total_p) {
        int j = idx % seq;
        int i = (idx / seq) % seq;
        int batch = idx / (seq * seq);
        int out_base = batch * seq * head_dim;
        int v_base = batch * seq * head_dim;
        T sum = T(0.0);
        for (int d = 0; d < head_dim; d++) {
            sum += grad_out[out_base + i * head_dim + d] * values[v_base + j * head_dim + d];
        }
        probs_grad[idx] += sum;
    }

    if (idx < total_v) {
        int d = idx % head_dim;
        int j = (idx / head_dim) % seq;
        int batch = idx / (seq * head_dim);
        int out_base = batch * seq * head_dim;
        int p_base = batch * seq * seq;
        T sum = T(0.0);
        for (int i = 0; i < seq; i++) {
            sum += grad_out[out_base + i * head_dim + d] * probs[p_base + i * seq + j];
        }
        values_grad[idx] += sum;
    }
}

extern "C" int scale_f64(const double* h_in, double* h_out, double scale, int size, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    scale_kernel<double><<<grid, block>>>(dev_in, dev_out, scale, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int scale_f32(const float* h_in, float* h_out, float scale, int size, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    scale_kernel<float><<<grid, block>>>(dev_in, dev_out, scale, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int fill_f64(double* h_data, double value, int size, int* d_data) {
    double* dev_data = (double*)d_data;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    fill_kernel<double><<<grid, block>>>(dev_data, value, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int fill_f32(float* h_data, float value, int size, int* d_data) {
    float* dev_data = (float*)d_data;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    fill_kernel<float><<<grid, block>>>(dev_data, value, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int sumsq_accum_f64(const double* h_in, double* h_accum, int size, int* d_in, int* d_accum) {
    const double* dev_in = (const double*)d_in;
    double* dev_accum = (double*)d_accum;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    sumsq_accum_kernel<double><<<grid, block>>>(dev_in, dev_accum, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int sumsq_accum_f32(const float* h_in, float* h_accum, int size, int* d_in, int* d_accum) {
    const float* dev_in = (const float*)d_in;
    float* dev_accum = (float*)d_accum;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    sumsq_accum_kernel<float><<<grid, block>>>(dev_in, dev_accum, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int sum_accum_f64(const double* h_in, double* h_accum, int size, double scale, int* d_in, int* d_accum) {
    const double* dev_in = (const double*)d_in;
    double* dev_accum = (double*)d_accum;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    sum_accum_kernel<double><<<grid, block>>>(dev_in, dev_accum, size, scale);
    return (int)cudaPeekAtLastError();
}

extern "C" int sum_accum_f32(const float* h_in, float* h_accum, int size, float scale, int* d_in, int* d_accum) {
    const float* dev_in = (const float*)d_in;
    float* dev_accum = (float*)d_accum;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    sum_accum_kernel<float><<<grid, block>>>(dev_in, dev_accum, size, scale);
    return (int)cudaPeekAtLastError();
}

extern "C" int clip_coef_from_sumsq_f64(const double* h_sumsq, double* h_coef, double max_norm, double eps, int* d_sumsq, int* d_coef) {
    const double* dev_sumsq = (const double*)d_sumsq;
    double* dev_coef = (double*)d_coef;
    clip_coef_kernel<double><<<1, 1>>>(dev_sumsq, dev_coef, max_norm, eps);
    return (int)cudaPeekAtLastError();
}

extern "C" int clip_coef_from_sumsq_f32(const float* h_sumsq, float* h_coef, float max_norm, float eps, int* d_sumsq, int* d_coef) {
    const float* dev_sumsq = (const float*)d_sumsq;
    float* dev_coef = (float*)d_coef;
    clip_coef_kernel<float><<<1, 1>>>(dev_sumsq, dev_coef, max_norm, eps);
    return (int)cudaPeekAtLastError();
}

extern "C" int scale_inplace_by_scalar_f64(double* h_data, const double* h_scalar, int size, int* d_data, int* d_scalar) {
    double* dev_data = (double*)d_data;
    const double* dev_scalar = (const double*)d_scalar;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    scale_inplace_by_scalar_kernel<double><<<grid, block>>>(dev_data, dev_scalar, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int scale_inplace_by_scalar_f32(float* h_data, const float* h_scalar, int size, int* d_data, int* d_scalar) {
    float* dev_data = (float*)d_data;
    const float* dev_scalar = (const float*)d_scalar;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    scale_inplace_by_scalar_kernel<float><<<grid, block>>>(dev_data, dev_scalar, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int add_scalar_f64(double* h_data, const double* h_scalar, double scale, int size, int* d_data, int* d_scalar) {
    double* dev_data = (double*)d_data;
    const double* dev_scalar = (const double*)d_scalar;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    add_scalar_kernel<double><<<grid, block>>>(dev_data, dev_scalar, scale, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int add_scalar_f32(float* h_data, const float* h_scalar, float scale, int size, int* d_data, int* d_scalar) {
    float* dev_data = (float*)d_data;
    const float* dev_scalar = (const float*)d_scalar;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    add_scalar_kernel<float><<<grid, block>>>(dev_data, dev_scalar, scale, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int lerp_inplace_f64(double* h_target, const double* h_source, double tau, int size, int* d_target, int* d_source) {
    double* dev_target = (double*)d_target;
    const double* dev_source = (const double*)d_source;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    lerp_inplace_kernel<double><<<grid, block>>>(dev_target, dev_source, tau, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int lerp_inplace_f32(float* h_target, const float* h_source, float tau, int size, int* d_target, int* d_source) {
    float* dev_target = (float*)d_target;
    const float* dev_source = (const float*)d_source;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    lerp_inplace_kernel<float><<<grid, block>>>(dev_target, dev_source, tau, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int double_dqn_target_f64(const double* h_eval, const double* h_target, const double* h_rewards, const double* h_dones, double* h_out, int batch, int actions, double gamma, int* d_eval, int* d_target, int* d_rewards, int* d_dones, int* d_out) {
    const double* dev_eval = (const double*)d_eval;
    const double* dev_target = (const double*)d_target;
    const double* dev_rewards = (const double*)d_rewards;
    const double* dev_dones = (const double*)d_dones;
    double* dev_out = (double*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((batch + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    double_dqn_target_kernel<double><<<grid, block>>>(dev_eval, dev_target, dev_rewards, dev_dones, dev_out, batch, actions, gamma);
    return (int)cudaPeekAtLastError();
}

extern "C" int double_dqn_target_f32(const float* h_eval, const float* h_target, const float* h_rewards, const float* h_dones, float* h_out, int batch, int actions, float gamma, int* d_eval, int* d_target, int* d_rewards, int* d_dones, int* d_out) {
    const float* dev_eval = (const float*)d_eval;
    const float* dev_target = (const float*)d_target;
    const float* dev_rewards = (const float*)d_rewards;
    const float* dev_dones = (const float*)d_dones;
    float* dev_out = (float*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((batch + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    double_dqn_target_kernel<float><<<grid, block>>>(dev_eval, dev_target, dev_rewards, dev_dones, dev_out, batch, actions, gamma);
    return (int)cudaPeekAtLastError();
}

extern "C" int abs_diff_f64(const double* h_a, const double* h_b, double* h_out, int size, int* d_a, int* d_b, int* d_out) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    abs_diff_kernel<double><<<grid, block>>>(dev_a, dev_b, dev_out, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int abs_diff_f32(const float* h_a, const float* h_b, float* h_out, int size, int* d_a, int* d_b, int* d_out) {
    const float* dev_a = (const float*)d_a;
    const float* dev_b = (const float*)d_b;
    float* dev_out = (float*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    abs_diff_kernel<float><<<grid, block>>>(dev_a, dev_b, dev_out, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int select_last_token_f64(const double* h_in, double* h_out, int batch, int seq, int dim, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((batch * dim + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    select_last_token_kernel<double><<<grid, block>>>(dev_in, dev_out, batch, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int select_last_token_f32(const float* h_in, float* h_out, int batch, int seq, int dim, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((batch * dim + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    select_last_token_kernel<float><<<grid, block>>>(dev_in, dev_out, batch, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int select_last_token_backward_f64(const double* h_grad_out, double* h_input_grad, int batch, int seq, int dim, int* d_grad_out, int* d_input_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((batch * dim + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    select_last_token_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_input_grad, batch, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int select_last_token_backward_f32(const float* h_grad_out, float* h_input_grad, int batch, int seq, int dim, int* d_grad_out, int* d_input_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((batch * dim + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    select_last_token_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_input_grad, batch, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int index_select_f64(const double* h_in, double* h_out, int idx, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    index_select_kernel<double><<<1, 1>>>(dev_in, dev_out, idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int index_select_f32(const float* h_in, float* h_out, int idx, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    index_select_kernel<float><<<1, 1>>>(dev_in, dev_out, idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int index_select_backward_f64(const double* h_grad_out, double* h_input_grad, int idx, int* d_grad_out, int* d_input_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    index_select_backward_kernel<double><<<1, 1>>>(dev_grad_out, dev_input_grad, idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int index_select_backward_f32(const float* h_grad_out, float* h_input_grad, int idx, int* d_grad_out, int* d_input_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    index_select_backward_kernel<float><<<1, 1>>>(dev_grad_out, dev_input_grad, idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int argmax_f64(const double* h_in, int* h_out, int size, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    int* dev_out = (int*)d_out;
    argmax_kernel<double><<<1, TENSOR_OP_BLOCK>>>(dev_in, dev_out, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int argmax_f32(const float* h_in, int* h_out, int size, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    int* dev_out = (int*)d_out;
    argmax_kernel<float><<<1, TENSOR_OP_BLOCK>>>(dev_in, dev_out, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int exp_f64(const double* h_in, double* h_out, int size, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    exp_kernel<double><<<grid, block>>>(dev_in, dev_out, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int exp_f32(const float* h_in, float* h_out, int size, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    exp_kernel<float><<<grid, block>>>(dev_in, dev_out, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int exp_backward_f64(const double* h_exp_out, const double* h_grad_out, double* h_input_grad, int size, int* d_exp_out, int* d_grad_out, int* d_input_grad) {
    const double* dev_exp_out = (const double*)d_exp_out;
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    exp_backward_kernel<double><<<grid, block>>>(dev_exp_out, dev_grad_out, dev_input_grad, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int exp_backward_f32(const float* h_exp_out, const float* h_grad_out, float* h_input_grad, int size, int* d_exp_out, int* d_grad_out, int* d_input_grad) {
    const float* dev_exp_out = (const float*)d_exp_out;
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    exp_backward_kernel<float><<<grid, block>>>(dev_exp_out, dev_grad_out, dev_input_grad, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int weighted_mse_loss_f64(const double* h_pred, const double* h_target, const double* h_weights, double* h_out, double* h_weight_sum, int size, int* d_pred, int* d_target, int* d_weights, int* d_out, int* d_weight_sum) {
    const double* dev_pred = (const double*)d_pred;
    const double* dev_target = (const double*)d_target;
    const double* dev_weights = (const double*)d_weights;
    double* dev_out = (double*)d_out;
    double* dev_weight_sum = (double*)d_weight_sum;
    fill_kernel<double><<<1, 1>>>(dev_out, 0.0, 1);
    fill_kernel<double><<<1, 1>>>(dev_weight_sum, 0.0, 1);
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    weighted_mse_loss_kernel<double><<<grid, block>>>(dev_pred, dev_target, dev_weights, dev_out, dev_weight_sum, size);
    weighted_mse_finalize_kernel<double><<<1, 1>>>(dev_out, dev_weight_sum);
    return (int)cudaPeekAtLastError();
}

extern "C" int weighted_mse_loss_f32(const float* h_pred, const float* h_target, const float* h_weights, float* h_out, float* h_weight_sum, int size, int* d_pred, int* d_target, int* d_weights, int* d_out, int* d_weight_sum) {
    const float* dev_pred = (const float*)d_pred;
    const float* dev_target = (const float*)d_target;
    const float* dev_weights = (const float*)d_weights;
    float* dev_out = (float*)d_out;
    float* dev_weight_sum = (float*)d_weight_sum;
    fill_kernel<float><<<1, 1>>>(dev_out, 0.0f, 1);
    fill_kernel<float><<<1, 1>>>(dev_weight_sum, 0.0f, 1);
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    weighted_mse_loss_kernel<float><<<grid, block>>>(dev_pred, dev_target, dev_weights, dev_out, dev_weight_sum, size);
    weighted_mse_finalize_kernel<float><<<1, 1>>>(dev_out, dev_weight_sum);
    return (int)cudaPeekAtLastError();
}

extern "C" int weighted_mse_backward_f64(const double* h_pred, const double* h_target, const double* h_weights, const double* h_weight_sum, const double* h_grad_out, double* h_pred_grad, int size, int* d_pred, int* d_target, int* d_weights, int* d_weight_sum, int* d_grad_out, int* d_pred_grad) {
    const double* dev_pred = (const double*)d_pred;
    const double* dev_target = (const double*)d_target;
    const double* dev_weights = (const double*)d_weights;
    const double* dev_weight_sum = (const double*)d_weight_sum;
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_pred_grad = (double*)d_pred_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    weighted_mse_backward_kernel<double><<<grid, block>>>(dev_pred, dev_target, dev_weights, dev_weight_sum, dev_grad_out, dev_pred_grad, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int weighted_mse_backward_f32(const float* h_pred, const float* h_target, const float* h_weights, const float* h_weight_sum, const float* h_grad_out, float* h_pred_grad, int size, int* d_pred, int* d_target, int* d_weights, int* d_weight_sum, int* d_grad_out, int* d_pred_grad) {
    const float* dev_pred = (const float*)d_pred;
    const float* dev_target = (const float*)d_target;
    const float* dev_weights = (const float*)d_weights;
    const float* dev_weight_sum = (const float*)d_weight_sum;
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_pred_grad = (float*)d_pred_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    weighted_mse_backward_kernel<float><<<grid, block>>>(dev_pred, dev_target, dev_weights, dev_weight_sum, dev_grad_out, dev_pred_grad, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int scale_backward_f64(const double* h_grad_out, double* h_input_grad, double scale, int size, int* d_grad_out, int* d_input_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    scale_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_input_grad, scale, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int scale_backward_f32(const float* h_grad_out, float* h_input_grad, float scale, int size, int* d_grad_out, int* d_input_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((size + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    scale_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_input_grad, scale, size);
    return (int)cudaPeekAtLastError();
}

extern "C" int causal_mask_f64(const double* h_in, double* h_out, int batches, int seq, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    int total = batches * seq * seq;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    causal_mask_kernel<double><<<grid, block>>>(dev_in, dev_out, batches, seq);
    return (int)cudaPeekAtLastError();
}

extern "C" int causal_mask_f32(const float* h_in, float* h_out, int batches, int seq, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    int total = batches * seq * seq;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    causal_mask_kernel<float><<<grid, block>>>(dev_in, dev_out, batches, seq);
    return (int)cudaPeekAtLastError();
}

extern "C" int causal_mask_backward_f64(const double* h_grad_out, double* h_input_grad, int batches, int seq, int* d_grad_out, int* d_input_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    int total = batches * seq * seq;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    causal_mask_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_input_grad, batches, seq);
    return (int)cudaPeekAtLastError();
}

extern "C" int causal_mask_backward_f32(const float* h_grad_out, float* h_input_grad, int batches, int seq, int* d_grad_out, int* d_input_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    int total = batches * seq * seq;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    causal_mask_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_input_grad, batches, seq);
    return (int)cudaPeekAtLastError();
}

extern "C" int concat_last_dim_f64(const double* h_a, const double* h_b, double* h_out, int rows, int a_dim, int b_dim, int* d_a, int* d_b, int* d_out) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    int total = rows * (a_dim + b_dim);
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    concat_last_dim_kernel<double><<<grid, block>>>(dev_a, dev_b, dev_out, rows, a_dim, b_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int concat_last_dim_f32(const float* h_a, const float* h_b, float* h_out, int rows, int a_dim, int b_dim, int* d_a, int* d_b, int* d_out) {
    const float* dev_a = (const float*)d_a;
    const float* dev_b = (const float*)d_b;
    float* dev_out = (float*)d_out;
    int total = rows * (a_dim + b_dim);
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    concat_last_dim_kernel<float><<<grid, block>>>(dev_a, dev_b, dev_out, rows, a_dim, b_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int concat_last_dim_backward_f64(const double* h_grad_out, double* h_a_grad, double* h_b_grad, int rows, int a_dim, int b_dim, int* d_grad_out, int* d_a_grad, int* d_b_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_a_grad = (double*)d_a_grad;
    double* dev_b_grad = (double*)d_b_grad;
    int total = rows * (a_dim + b_dim);
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    concat_last_dim_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_a_grad, dev_b_grad, rows, a_dim, b_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int concat_last_dim_backward_f32(const float* h_grad_out, float* h_a_grad, float* h_b_grad, int rows, int a_dim, int b_dim, int* d_grad_out, int* d_a_grad, int* d_b_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_a_grad = (float*)d_a_grad;
    float* dev_b_grad = (float*)d_b_grad;
    int total = rows * (a_dim + b_dim);
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    concat_last_dim_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_a_grad, dev_b_grad, rows, a_dim, b_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int split_last_dim_f64(const double* h_in, double* h_out, int rows, int input_dim, int part_dim, int part_idx, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    int total = rows * part_dim;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    split_last_dim_kernel<double><<<grid, block>>>(dev_in, dev_out, rows, input_dim, part_dim, part_idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int split_last_dim_f32(const float* h_in, float* h_out, int rows, int input_dim, int part_dim, int part_idx, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    int total = rows * part_dim;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    split_last_dim_kernel<float><<<grid, block>>>(dev_in, dev_out, rows, input_dim, part_dim, part_idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int split_last_dim_backward_f64(const double* h_grad_out, double* h_input_grad, int rows, int input_dim, int part_dim, int part_idx, int* d_grad_out, int* d_input_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    int total = rows * part_dim;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    split_last_dim_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_input_grad, rows, input_dim, part_dim, part_idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int split_last_dim_backward_f32(const float* h_grad_out, float* h_input_grad, int rows, int input_dim, int part_dim, int part_idx, int* d_grad_out, int* d_input_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    int total = rows * part_dim;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    split_last_dim_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_input_grad, rows, input_dim, part_dim, part_idx);
    return (int)cudaPeekAtLastError();
}

extern "C" int broadcast_batch_f64(const double* h_in, double* h_out, int batch_size, int inner_len, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    int total = batch_size * inner_len;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    broadcast_batch_kernel<double><<<grid, block>>>(dev_in, dev_out, batch_size, inner_len);
    return (int)cudaPeekAtLastError();
}

extern "C" int broadcast_batch_f32(const float* h_in, float* h_out, int batch_size, int inner_len, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    int total = batch_size * inner_len;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    broadcast_batch_kernel<float><<<grid, block>>>(dev_in, dev_out, batch_size, inner_len);
    return (int)cudaPeekAtLastError();
}

extern "C" int broadcast_batch_backward_f64(const double* h_grad_out, double* h_input_grad, int batch_size, int inner_len, int* d_grad_out, int* d_input_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((inner_len + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    broadcast_batch_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_input_grad, batch_size, inner_len);
    return (int)cudaPeekAtLastError();
}

extern "C" int broadcast_batch_backward_f32(const float* h_grad_out, float* h_input_grad, int batch_size, int inner_len, int* d_grad_out, int* d_input_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((inner_len + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    broadcast_batch_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_input_grad, batch_size, inner_len);
    return (int)cudaPeekAtLastError();
}

extern "C" int transpose_last_two_f64(const double* h_in, double* h_out, int outer, int rows, int cols, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    int total = outer * rows * cols;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    transpose_last_two_kernel<double><<<grid, block>>>(dev_in, dev_out, outer, rows, cols);
    return (int)cudaPeekAtLastError();
}

extern "C" int transpose_last_two_f32(const float* h_in, float* h_out, int outer, int rows, int cols, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    int total = outer * rows * cols;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    transpose_last_two_kernel<float><<<grid, block>>>(dev_in, dev_out, outer, rows, cols);
    return (int)cudaPeekAtLastError();
}

extern "C" int transpose_4d_f64(const double* h_in, double* h_out, int d0, int d1, int d2, int d3, int dim0, int dim1, int* d_in, int* d_out) {
    const double* dev_in = (const double*)d_in;
    double* dev_out = (double*)d_out;
    int total = d0 * d1 * d2 * d3;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    transpose_4d_kernel<double><<<grid, block>>>(dev_in, dev_out, d0, d1, d2, d3, dim0, dim1);
    return (int)cudaPeekAtLastError();
}

extern "C" int transpose_4d_f32(const float* h_in, float* h_out, int d0, int d1, int d2, int d3, int dim0, int dim1, int* d_in, int* d_out) {
    const float* dev_in = (const float*)d_in;
    float* dev_out = (float*)d_out;
    int total = d0 * d1 * d2 * d3;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    transpose_4d_kernel<float><<<grid, block>>>(dev_in, dev_out, d0, d1, d2, d3, dim0, dim1);
    return (int)cudaPeekAtLastError();
}

extern "C" int batched_qk_scores_f64(const double* h_q, const double* h_k, double* h_out, int batches, int seq, int dim, int* d_q, int* d_k, int* d_out) {
    const double* dev_q = (const double*)d_q;
    const double* dev_k = (const double*)d_k;
    double* dev_out = (double*)d_out;
    int total = batches * seq * seq;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    batched_qk_scores_kernel<double><<<grid, block>>>(dev_q, dev_k, dev_out, batches, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int batched_qk_scores_f32(const float* h_q, const float* h_k, float* h_out, int batches, int seq, int dim, int* d_q, int* d_k, int* d_out) {
    const float* dev_q = (const float*)d_q;
    const float* dev_k = (const float*)d_k;
    float* dev_out = (float*)d_out;
    int total = batches * seq * seq;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    batched_qk_scores_kernel<float><<<grid, block>>>(dev_q, dev_k, dev_out, batches, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int batched_qk_scores_backward_f64(const double* h_grad_out, const double* h_q, const double* h_k, double* h_q_grad, double* h_k_grad, int batches, int seq, int dim, int* d_grad_out, int* d_q, int* d_k, int* d_q_grad, int* d_k_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_q = (const double*)d_q;
    const double* dev_k = (const double*)d_k;
    double* dev_q_grad = (double*)d_q_grad;
    double* dev_k_grad = (double*)d_k_grad;
    int total = batches * seq * dim;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    batched_qk_scores_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_q, dev_k, dev_q_grad, dev_k_grad, batches, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int batched_qk_scores_backward_f32(const float* h_grad_out, const float* h_q, const float* h_k, float* h_q_grad, float* h_k_grad, int batches, int seq, int dim, int* d_grad_out, int* d_q, int* d_k, int* d_q_grad, int* d_k_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    const float* dev_q = (const float*)d_q;
    const float* dev_k = (const float*)d_k;
    float* dev_q_grad = (float*)d_q_grad;
    float* dev_k_grad = (float*)d_k_grad;
    int total = batches * seq * dim;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    batched_qk_scores_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_q, dev_k, dev_q_grad, dev_k_grad, batches, seq, dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int attention_weighted_sum_backward_f64(const double* h_grad_out, const double* h_probs, const double* h_values, double* h_probs_grad, double* h_values_grad, int batches, int seq, int head_dim, int* d_grad_out, int* d_probs, int* d_values, int* d_probs_grad, int* d_values_grad) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_probs = (const double*)d_probs;
    const double* dev_values = (const double*)d_values;
    double* dev_probs_grad = (double*)d_probs_grad;
    double* dev_values_grad = (double*)d_values_grad;
    int total_p = batches * seq * seq;
    int total_v = batches * seq * head_dim;
    int total = total_p > total_v ? total_p : total_v;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    attention_weighted_sum_backward_kernel<double><<<grid, block>>>(dev_grad_out, dev_probs, dev_values, dev_probs_grad, dev_values_grad, batches, seq, head_dim);
    return (int)cudaPeekAtLastError();
}

extern "C" int attention_weighted_sum_backward_f32(const float* h_grad_out, const float* h_probs, const float* h_values, float* h_probs_grad, float* h_values_grad, int batches, int seq, int head_dim, int* d_grad_out, int* d_probs, int* d_values, int* d_probs_grad, int* d_values_grad) {
    const float* dev_grad_out = (const float*)d_grad_out;
    const float* dev_probs = (const float*)d_probs;
    const float* dev_values = (const float*)d_values;
    float* dev_probs_grad = (float*)d_probs_grad;
    float* dev_values_grad = (float*)d_values_grad;
    int total_p = batches * seq * seq;
    int total_v = batches * seq * head_dim;
    int total = total_p > total_v ? total_p : total_v;
    dim3 block(TENSOR_OP_BLOCK);
    dim3 grid((total + TENSOR_OP_BLOCK - 1) / TENSOR_OP_BLOCK);
    attention_weighted_sum_backward_kernel<float><<<grid, block>>>(dev_grad_out, dev_probs, dev_values, dev_probs_grad, dev_values_grad, batches, seq, head_dim);
    return (int)cudaPeekAtLastError();
}
