// backward.cu - CUDA backward propagation kernels
#include "common.cu"

template<typename T>
__global__ void add_forward_kernel(
    const T* __restrict__ a, const T* __restrict__ b,
    T* __restrict__ out, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

template<typename T>
__global__ void sub_forward_kernel(
    const T* __restrict__ a, const T* __restrict__ b,
    T* __restrict__ out, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] - b[idx];
}

template<typename T>
__global__ void mul_forward_kernel(
    const T* __restrict__ a, const T* __restrict__ b,
    T* __restrict__ out, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] * b[idx];
}

template<typename T>
__global__ void div_forward_kernel(
    const T* __restrict__ a, const T* __restrict__ b,
    T* __restrict__ out, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T bv = b[idx];
        out[idx] = a[idx] / (bv != T(0.0) ? bv : T(1e-12));
    }
}

template<typename T>
__global__ void relu_backward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input[idx] > T(0.0)) {
            input_grad[idx] += grad_out[idx];
        }
    }
}

#define GELU_SQRT_2_OVER_PI 0.7978845608028654
#define GELU_C 0.044715

template<typename T>
__global__ void gelu_backward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ grad_out,
    T* __restrict__ input_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        T x2 = x * x;
        T x3 = x2 * x;
        T u = T(GELU_SQRT_2_OVER_PI) * (x + T(GELU_C) * x3);
        T tanh_u = tanh(u);
        T sech2_u = T(1.0) - tanh_u * tanh_u;
        T du_dx = T(GELU_SQRT_2_OVER_PI) * (T(1.0) + T(3.0) * T(GELU_C) * x2);
        T gelu_grad = T(0.5) * (T(1.0) + tanh_u) + T(0.5) * x * sech2_u * du_dx;
        input_grad[idx] += grad_out[idx] * gelu_grad;
    }
}

template<typename T>
__global__ void add_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ a_grad,
    T* __restrict__ b_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T g = grad_out[idx];
        a_grad[idx] += g;
        b_grad[idx] += g;
    }
}

template<typename T>
__global__ void sub_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ a_grad,
    T* __restrict__ b_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T g = grad_out[idx];
        a_grad[idx] += g;
        b_grad[idx] -= g;
    }
}

template<typename T>
__global__ void mul_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ a_data,
    const T* __restrict__ b_data,
    T* __restrict__ a_grad,
    T* __restrict__ b_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T g = grad_out[idx];
        a_grad[idx] += g * b_data[idx];
        b_grad[idx] += g * a_data[idx];
    }
}

template<typename T>
__global__ void div_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ a_data,
    const T* __restrict__ b_data,
    T* __restrict__ a_grad,
    T* __restrict__ b_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T g = grad_out[idx];
        T a = a_data[idx];
        T b = b_data[idx];
        T safe_b = (b != T(0.0)) ? b : T(1e-12);
        a_grad[idx] += g / safe_b;
        b_grad[idx] += g * (-a / (safe_b * safe_b));
    }
}

template<typename T>
__global__ void acc_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] += src[idx];
    }
}

template<typename T>
__global__ void adam_step_kernel(
    T* __restrict__ params,
    const T* __restrict__ grads,
    T* __restrict__ m,
    T* __restrict__ v,
    int size,
    T lr, T beta1, T beta2, T eps,
    T weight_decay, T bias_correction1, T bias_correction2, T clip_coef
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T g = grads[idx] * clip_coef;
        m[idx] = beta1 * m[idx] + (T(1.0) - beta1) * g;
        v[idx] = beta2 * v[idx] + (T(1.0) - beta2) * g * g;
        T m_hat = m[idx] / bias_correction1;
        T v_hat = v[idx] / bias_correction2;
        params[idx] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * params[idx]);
    }
}

#define BW_LAUNCH_F64(kernel, ...) \
    dim3 grid_dim = compute_grid_1d(size, 256); \
    dim3 block_dim(256); \
    CUDA_LAUNCH(kernel<double>, grid_dim, block_dim, 0, __VA_ARGS__); \
    cudaError_t err = cudaPeekAtLastError(); \
    return (int)err;

#define BW_LAUNCH_F32(kernel, ...) \
    dim3 grid_dim = compute_grid_1d(size, 256); \
    dim3 block_dim(256); \
    CUDA_LAUNCH(kernel<float>, grid_dim, block_dim, 0, __VA_ARGS__); \
    cudaError_t err = cudaPeekAtLastError(); \
    return (int)err;

extern "C" int relu_backward_f64(
    const double* h_input, const double* h_grad_out, double* h_input_grad,
    int size, int* d_input, int* d_grad_out, int* d_input_grad
) {
    const double* dev_input = (const double*)d_input;
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    BW_LAUNCH_F64(relu_backward_kernel, dev_input, dev_grad_out, dev_input_grad, size)
}

extern "C" int relu_backward_f32(
    const float* h_input, const float* h_grad_out, float* h_input_grad,
    int size, int* d_input, int* d_grad_out, int* d_input_grad
) {
    const float* dev_input = (const float*)d_input;
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    BW_LAUNCH_F32(relu_backward_kernel, dev_input, dev_grad_out, dev_input_grad, size)
}

extern "C" int gelu_backward_f64(
    const double* h_input, const double* h_grad_out, double* h_input_grad,
    int size, int* d_input, int* d_grad_out, int* d_input_grad
) {
    const double* dev_input = (const double*)d_input;
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;
    BW_LAUNCH_F64(gelu_backward_kernel, dev_input, dev_grad_out, dev_input_grad, size)
}

extern "C" int gelu_backward_f32(
    const float* h_input, const float* h_grad_out, float* h_input_grad,
    int size, int* d_input, int* d_grad_out, int* d_input_grad
) {
    const float* dev_input = (const float*)d_input;
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_input_grad = (float*)d_input_grad;
    BW_LAUNCH_F32(gelu_backward_kernel, dev_input, dev_grad_out, dev_input_grad, size)
}

extern "C" int add_backward_f64(
    const double* h_grad_out, double* h_a_grad, double* h_b_grad,
    int size, int* d_grad_out, int* d_a_grad, int* d_b_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_a_grad = (double*)d_a_grad;
    double* dev_b_grad = (double*)d_b_grad;
    BW_LAUNCH_F64(add_backward_kernel, dev_grad_out, dev_a_grad, dev_b_grad, size)
}

extern "C" int add_backward_f32(
    const float* h_grad_out, float* h_a_grad, float* h_b_grad,
    int size, int* d_grad_out, int* d_a_grad, int* d_b_grad
) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_a_grad = (float*)d_a_grad;
    float* dev_b_grad = (float*)d_b_grad;
    BW_LAUNCH_F32(add_backward_kernel, dev_grad_out, dev_a_grad, dev_b_grad, size)
}

extern "C" int sub_backward_f64(
    const double* h_grad_out, double* h_a_grad, double* h_b_grad,
    int size, int* d_grad_out, int* d_a_grad, int* d_b_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_a_grad = (double*)d_a_grad;
    double* dev_b_grad = (double*)d_b_grad;
    BW_LAUNCH_F64(sub_backward_kernel, dev_grad_out, dev_a_grad, dev_b_grad, size)
}

extern "C" int sub_backward_f32(
    const float* h_grad_out, float* h_a_grad, float* h_b_grad,
    int size, int* d_grad_out, int* d_a_grad, int* d_b_grad
) {
    const float* dev_grad_out = (const float*)d_grad_out;
    float* dev_a_grad = (float*)d_a_grad;
    float* dev_b_grad = (float*)d_b_grad;
    BW_LAUNCH_F32(sub_backward_kernel, dev_grad_out, dev_a_grad, dev_b_grad, size)
}

extern "C" int mul_backward_f64(
    const double* h_grad_out, const double* h_a_data, const double* h_b_data,
    double* h_a_grad, double* h_b_grad,
    int size, int* d_grad_out, int* d_a_data, int* d_b_data, int* d_a_grad, int* d_b_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_a_data = (const double*)d_a_data;
    const double* dev_b_data = (const double*)d_b_data;
    double* dev_a_grad = (double*)d_a_grad;
    double* dev_b_grad = (double*)d_b_grad;
    BW_LAUNCH_F64(mul_backward_kernel, dev_grad_out, dev_a_data, dev_b_data, dev_a_grad, dev_b_grad, size)
}

extern "C" int mul_backward_f32(
    const float* h_grad_out, const float* h_a_data, const float* h_b_data,
    float* h_a_grad, float* h_b_grad,
    int size, int* d_grad_out, int* d_a_data, int* d_b_data, int* d_a_grad, int* d_b_grad
) {
    const float* dev_grad_out = (const float*)d_grad_out;
    const float* dev_a_data = (const float*)d_a_data;
    const float* dev_b_data = (const float*)d_b_data;
    float* dev_a_grad = (float*)d_a_grad;
    float* dev_b_grad = (float*)d_b_grad;
    BW_LAUNCH_F32(mul_backward_kernel, dev_grad_out, dev_a_data, dev_b_data, dev_a_grad, dev_b_grad, size)
}

extern "C" int div_backward_f64(
    const double* h_grad_out, const double* h_a_data, const double* h_b_data,
    double* h_a_grad, double* h_b_grad,
    int size, int* d_grad_out, int* d_a_data, int* d_b_data, int* d_a_grad, int* d_b_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_a_data = (const double*)d_a_data;
    const double* dev_b_data = (const double*)d_b_data;
    double* dev_a_grad = (double*)d_a_grad;
    double* dev_b_grad = (double*)d_b_grad;
    BW_LAUNCH_F64(div_backward_kernel, dev_grad_out, dev_a_data, dev_b_data, dev_a_grad, dev_b_grad, size)
}

extern "C" int div_backward_f32(
    const float* h_grad_out, const float* h_a_data, const float* h_b_data,
    float* h_a_grad, float* h_b_grad,
    int size, int* d_grad_out, int* d_a_data, int* d_b_data, int* d_a_grad, int* d_b_grad
) {
    const float* dev_grad_out = (const float*)d_grad_out;
    const float* dev_a_data = (const float*)d_a_data;
    const float* dev_b_data = (const float*)d_b_data;
    float* dev_a_grad = (float*)d_a_grad;
    float* dev_b_grad = (float*)d_b_grad;
    BW_LAUNCH_F32(div_backward_kernel, dev_grad_out, dev_a_data, dev_b_data, dev_a_grad, dev_b_grad, size)
}

extern "C" int acc_buffer_f64(
    double* h_dst, const double* h_src, int size, int* d_dst, int* d_src
) {
    double* dev_dst = (double*)d_dst;
    const double* dev_src = (const double*)d_src;
    BW_LAUNCH_F64(acc_kernel, dev_dst, dev_src, size)
}

extern "C" int acc_buffer_f32(
    float* h_dst, const float* h_src, int size, int* d_dst, int* d_src
) {
    float* dev_dst = (float*)d_dst;
    const float* dev_src = (const float*)d_src;
    BW_LAUNCH_F32(acc_kernel, dev_dst, dev_src, size)
}

extern "C" int add_forward_f64(
    const double* h_a, const double* h_b, double* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    BW_LAUNCH_F64(add_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int add_forward_f32(
    const float* h_a, const float* h_b, float* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const float* dev_a = (const float*)d_a;
    const float* dev_b = (const float*)d_b;
    float* dev_out = (float*)d_out;
    BW_LAUNCH_F32(add_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int sub_forward_f64(
    const double* h_a, const double* h_b, double* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    BW_LAUNCH_F64(sub_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int sub_forward_f32(
    const float* h_a, const float* h_b, float* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const float* dev_a = (const float*)d_a;
    const float* dev_b = (const float*)d_b;
    float* dev_out = (float*)d_out;
    BW_LAUNCH_F32(sub_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int mul_forward_f64(
    const double* h_a, const double* h_b, double* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    BW_LAUNCH_F64(mul_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int mul_forward_f32(
    const float* h_a, const float* h_b, float* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const float* dev_a = (const float*)d_a;
    const float* dev_b = (const float*)d_b;
    float* dev_out = (float*)d_out;
    BW_LAUNCH_F32(mul_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int div_forward_f64(
    const double* h_a, const double* h_b, double* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    BW_LAUNCH_F64(div_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int div_forward_f32(
    const float* h_a, const float* h_b, float* h_out,
    int size, int* d_a, int* d_b, int* d_out
) {
    const float* dev_a = (const float*)d_a;
    const float* dev_b = (const float*)d_b;
    float* dev_out = (float*)d_out;
    BW_LAUNCH_F32(div_forward_kernel, dev_a, dev_b, dev_out, size)
}

extern "C" int adam_step_f64(
    double* h_params, const double* h_grads, double* h_m, double* h_v,
    int size, double lr, double beta1, double beta2, double eps,
    double weight_decay, double bias_correction1, double bias_correction2, double clip_coef,
    int* d_params, int* d_grads, int* d_m, int* d_v
) {
    double* dev_params = (double*)d_params;
    const double* dev_grads = (const double*)d_grads;
    double* dev_m = (double*)d_m;
    double* dev_v = (double*)d_v;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(adam_step_kernel<double>, grid_dim, block_dim, 0,
                dev_params, dev_grads, dev_m, dev_v, size,
                lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2, clip_coef);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int adam_step_f32(
    float* h_params, const float* h_grads, float* h_m, float* h_v,
    int size, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bias_correction1, float bias_correction2, float clip_coef,
    int* d_params, int* d_grads, int* d_m, int* d_v
) {
    float* dev_params = (float*)d_params;
    const float* dev_grads = (const float*)d_grads;
    float* dev_m = (float*)d_m;
    float* dev_v = (float*)d_v;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(adam_step_kernel<float>, grid_dim, block_dim, 0,
                dev_params, dev_grads, dev_m, dev_v, size,
                lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2, clip_coef);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
