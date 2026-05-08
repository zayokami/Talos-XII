// backward.cu - CUDA backward propagation kernels
#include "common.cu"

//==============================================================================
// Element-wise forward kernels
//==============================================================================

__global__ void add_forward_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_forward_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_forward_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_forward_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double bv = b[idx];
        out[idx] = a[idx] / (bv != 0.0 ? bv : 1e-12);
    }
}

//==============================================================================
// ReLU backward kernel
// d_input_grad[i] += (input[i] > 0.0) ? grad_out[i] : 0.0
//==============================================================================
__global__ void relu_backward_kernel(
    const double* __restrict__ input,
    const double* __restrict__ grad_out,
    double* __restrict__ input_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input[idx] > 0.0) {
            input_grad[idx] += grad_out[idx];
        }
    }
}

//==============================================================================
// GELU backward kernel
// GELU(x) = 0.5 * x * (1 + tanh(u)) where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// dGELU/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
// du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
//==============================================================================
#define GELU_SQRT_2_OVER_PI 0.7978845608028654
#define GELU_C 0.044715

__global__ void gelu_backward_kernel(
    const double* __restrict__ input,
    const double* __restrict__ grad_out,
    double* __restrict__ input_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = input[idx];
        double x2 = x * x;
        double x3 = x2 * x;
        double u = GELU_SQRT_2_OVER_PI * (x + GELU_C * x3);
        double tanh_u = tanh(u);
        double sech2_u = 1.0 - tanh_u * tanh_u;
        double du_dx = GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_C * x2);
        double gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
        input_grad[idx] += grad_out[idx] * gelu_grad;
    }
}

//==============================================================================
// Element-wise add backward kernel
// d_a_grad[i] += grad_out[i]
// d_b_grad[i] += grad_out[i]
//==============================================================================
__global__ void add_backward_kernel(
    const double* __restrict__ grad_out,
    double* __restrict__ a_grad,
    double* __restrict__ b_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double g = grad_out[idx];
        a_grad[idx] += g;
        b_grad[idx] += g;
    }
}

//==============================================================================
// Element-wise mul backward kernel
// d_a_grad[i] += grad_out[i] * b[i]
// d_b_grad[i] += grad_out[i] * a[i]
//==============================================================================
__global__ void mul_backward_kernel(
    const double* __restrict__ grad_out,
    const double* __restrict__ a_data,
    const double* __restrict__ b_data,
    double* __restrict__ a_grad,
    double* __restrict__ b_grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double g = grad_out[idx];
        a_grad[idx] += g * b_data[idx];
        b_grad[idx] += g * a_data[idx];
    }
}

//==============================================================================
// Element-wise accumulate kernel
// dst[i] += src[i]
//==============================================================================
__global__ void acc_kernel(
    double* __restrict__ dst,
    const double* __restrict__ src,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] += src[idx];
    }
}

//==============================================================================
// Host wrappers
//==============================================================================
extern "C" int relu_backward(
    const double* h_input,
    const double* h_grad_out,
    double* h_input_grad,
    int size,
    int* d_input,
    int* d_grad_out,
    int* d_input_grad
) {
    const double* dev_input = (const double*)d_input;
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;

    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(relu_backward_kernel, grid_dim, block_dim, 0,
                dev_input, dev_grad_out, dev_input_grad, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int gelu_backward(
    const double* h_input,
    const double* h_grad_out,
    double* h_input_grad,
    int size,
    int* d_input,
    int* d_grad_out,
    int* d_input_grad
) {
    const double* dev_input = (const double*)d_input;
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_input_grad = (double*)d_input_grad;

    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(gelu_backward_kernel, grid_dim, block_dim, 0,
                dev_input, dev_grad_out, dev_input_grad, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int add_backward(
    const double* h_grad_out,
    double* h_a_grad,
    double* h_b_grad,
    int size,
    int* d_grad_out,
    int* d_a_grad,
    int* d_b_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    double* dev_a_grad = (double*)d_a_grad;
    double* dev_b_grad = (double*)d_b_grad;

    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(add_backward_kernel, grid_dim, block_dim, 0,
                dev_grad_out, dev_a_grad, dev_b_grad, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int mul_backward(
    const double* h_grad_out,
    const double* h_a_data,
    const double* h_b_data,
    double* h_a_grad,
    double* h_b_grad,
    int size,
    int* d_grad_out,
    int* d_a_data,
    int* d_b_data,
    int* d_a_grad,
    int* d_b_grad
) {
    const double* dev_grad_out = (const double*)d_grad_out;
    const double* dev_a_data = (const double*)d_a_data;
    const double* dev_b_data = (const double*)d_b_data;
    double* dev_a_grad = (double*)d_a_grad;
    double* dev_b_grad = (double*)d_b_grad;

    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(mul_backward_kernel, grid_dim, block_dim, 0,
                dev_grad_out, dev_a_data, dev_b_data, dev_a_grad, dev_b_grad, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int acc_buffer(
    double* h_dst,
    const double* h_src,
    int size,
    int* d_dst,
    int* d_src
) {
    double* dev_dst = (double*)d_dst;
    const double* dev_src = (const double*)d_src;

    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);

    CUDA_LAUNCH(acc_kernel, grid_dim, block_dim, 0, dev_dst, dev_src, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int add_forward(
    const double* h_a, const double* h_b, double* h_out,
    int size,
    int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(add_forward_kernel, grid_dim, block_dim, 0, dev_a, dev_b, dev_out, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int sub_forward(
    const double* h_a, const double* h_b, double* h_out,
    int size,
    int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(sub_forward_kernel, grid_dim, block_dim, 0, dev_a, dev_b, dev_out, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int mul_forward(
    const double* h_a, const double* h_b, double* h_out,
    int size,
    int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(mul_forward_kernel, grid_dim, block_dim, 0, dev_a, dev_b, dev_out, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

extern "C" int div_forward(
    const double* h_a, const double* h_b, double* h_out,
    int size,
    int* d_a, int* d_b, int* d_out
) {
    const double* dev_a = (const double*)d_a;
    const double* dev_b = (const double*)d_b;
    double* dev_out = (double*)d_out;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(div_forward_kernel, grid_dim, block_dim, 0, dev_a, dev_b, dev_out, size);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}

//==============================================================================
// Adam optimizer step kernel
// params[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * params[i])
//==============================================================================
__global__ void adam_step_kernel(
    double* __restrict__ params,
    const double* __restrict__ grads,
    double* __restrict__ m,
    double* __restrict__ v,
    int size,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,
    double bias_correction2,
    double clip_coef
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double g = grads[idx] * clip_coef;
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;
        double m_hat = m[idx] / bias_correction1;
        double v_hat = v[idx] / bias_correction2;
        params[idx] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * params[idx]);
    }
}

extern "C" int adam_step(
    double* h_params,
    const double* h_grads,
    double* h_m,
    double* h_v,
    int size,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,
    double bias_correction2,
    double clip_coef,
    int* d_params,
    int* d_grads,
    int* d_m,
    int* d_v
) {
    double* dev_params = (double*)d_params;
    const double* dev_grads = (const double*)d_grads;
    double* dev_m = (double*)d_m;
    double* dev_v = (double*)d_v;
    dim3 grid_dim = compute_grid_1d(size, 256);
    dim3 block_dim(256);
    CUDA_LAUNCH(adam_step_kernel, grid_dim, block_dim, 0,
                dev_params, dev_grads, dev_m, dev_v, size,
                lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2, clip_coef);
    cudaError_t err = cudaPeekAtLastError();
    return (int)err;
}
