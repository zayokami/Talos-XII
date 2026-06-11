// attention_output.cu - Weighted sum kernel for attention output
#include "common.cu"

#define ATTN_BLOCK 256

template<typename T>
__global__ void attention_weighted_sum_kernel(
    const T* __restrict__ attn_weights,
    const T* __restrict__ values,
    T* __restrict__ output,
    int batches, int seq, int head_dim
) {
    int batch = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;

    if (batch >= batches || row >= seq) return;

    int attn_base = batch * seq * seq + row * seq;
    int values_base = batch * seq * head_dim;
    int output_base = values_base + row * head_dim;

    for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
        T sum = T(0.0);
        for (int k = 0; k < seq; k++) {
            sum += attn_weights[attn_base + k] * values[values_base + k * head_dim + d];
        }
        output[output_base + d] = sum;
    }
}

extern "C" int attention_weighted_sum_f64(
    double* h_attn, double* h_values, double* h_output,
    int batches, int seq, int head_dim,
    int* d_attn, int* d_values, int* d_output
) {
    const double* dev_attn = (const double*)d_attn;
    const double* dev_values = (const double*)d_values;
    double* dev_output = (double*)d_output;

    if (batches <= 0) {
        return (int)cudaSuccess;
    }
    if (seq <= 0 || head_dim <= 0 || dev_attn == nullptr || dev_values == nullptr || dev_output == nullptr) {
        return (int)cudaErrorInvalidValue;
    }

    (void)cudaGetLastError();
    attention_weighted_sum_kernel<double><<<dim3(batches, seq), dim3(ATTN_BLOCK), 0, 0>>>(
        dev_attn, dev_values, dev_output, batches, seq, head_dim);
    return (int)cudaGetLastError();
}

extern "C" int attention_weighted_sum_f32(
    float* h_attn, float* h_values, float* h_output,
    int batches, int seq, int head_dim,
    int* d_attn, int* d_values, int* d_output
) {
    const float* dev_attn = (const float*)d_attn;
    const float* dev_values = (const float*)d_values;
    float* dev_output = (float*)d_output;

    if (batches <= 0) {
        return (int)cudaSuccess;
    }
    if (seq <= 0 || head_dim <= 0 || dev_attn == nullptr || dev_values == nullptr || dev_output == nullptr) {
        return (int)cudaErrorInvalidValue;
    }

    (void)cudaGetLastError();
    attention_weighted_sum_kernel<float><<<dim3(batches, seq), dim3(ATTN_BLOCK), 0, 0>>>(
        dev_attn, dev_values, dev_output, batches, seq, head_dim);
    return (int)cudaGetLastError();
}
