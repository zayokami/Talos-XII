#[cfg(cuda)]
use crate::autograd::cuda_grad_out_buffer;
use crate::autograd::{Context, Device, Tensor, PAR_THRESHOLD};
use crate::dtype::{Dtype, Storage};
use crate::simd::{vector_gelu, vector_relu};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

impl Tensor {
    pub fn relu(&self) -> Tensor {
        // GPU routing for float dtypes
        #[cfg(cuda)]
        if self.device == Device::Cuda && (self.dtype == Dtype::F32 || self.dtype == Dtype::F64) {
            return self.relu_cuda();
        }

        // Generic path for non-F64 dtypes
        if self.dtype != Dtype::F64 {
            return self.relu_generic();
        }

        // CPU path - inline implementation when cuda is disabled
        #[cfg(not(cuda))]
        {
            let self_data = self.data_f64();
            let len = self_data.len();
            let mut data = vec![0.0; len];
            if len >= PAR_THRESHOLD {
                data.par_chunks_mut(PAR_THRESHOLD)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * PAR_THRESHOLD;
                        vector_relu(chunk, &self_data[start..start + chunk.len()]);
                    });
            } else {
                vector_relu(&mut data, &self_data);
            }
            let parents = vec![self.clone()];

            Tensor {
                data: Storage::F64(Arc::new(RwLock::new(data))),
                grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
                shape: self.shape.clone(),
                device: Device::Cpu,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let input = &parents[0];
                        let input_data = input.data_f64();
                        let mut inp_grad = input.grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_data.par_iter())
                                .for_each(|((ig, &go), &val)| {
                                    if val > 0.0 {
                                        *ig += go;
                                    }
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                if input_data[i] > 0.0 {
                                    inp_grad[i] += grad_out_f64[i];
                                }
                            }
                        }
                    }),
                })),
            }
        }

        // CPU path - call fallback when cuda is enabled but device is CPU
        #[cfg(cuda)]
        self.relu_cpu_fallback()
    }

    /// Generic ReLU for non-F64 dtypes.
    fn relu_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        let mask: Vec<bool> = self_f32.iter().map(|&x| x > 0.0).collect();
        for i in 0..len {
            data[i] = if mask[i] { self_f32[i] } else { 0.0 };
        }
        let mask = Arc::new(mask);
        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    for i in 0..len {
                        if mask[i] {
                            inp_grad[i] += grad_out_f64[i];
                        }
                    }
                }),
            })),
        }
    }

    /// CPU fallback for ReLU activation
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn relu_cpu_fallback(&self) -> Tensor {
        if self.dtype == Dtype::F32 {
            return self.relu_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_chunks_mut(PAR_THRESHOLD)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * PAR_THRESHOLD;
                    vector_relu(chunk, &self_data[start..start + chunk.len()]);
                });
        } else {
            vector_relu(&mut data, &self_data);
        }
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &go), &val)| {
                                if val > 0.0 {
                                    *ig += go;
                                }
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            if input_data[i] > 0.0 {
                                inp_grad[i] += grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    /// GPU-accelerated ReLU activation using CUDA kernel
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn relu_cuda(&self) -> Tensor {
        use crate::cuda::kernels::{
            relu_backward, relu_backward_f32, relu_inplace, relu_inplace_f32,
        };
        use crate::cuda::memory::{alloc_pooled, copy_d2d, CudaBuffer};

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.relu_cpu_fallback(),
        };
        if len == 0 {
            return self.relu_cpu_fallback();
        }
        crate::cuda::record_activation_attempt();

        let d_src = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_activation_fallback(stage);
                log::warn!(
                    "[Autograd] CUDA prepare ReLU input failed ({}), using CPU",
                    err
                );
                return self.relu_cpu_fallback();
            }
        };
        let d_data = match self.dtype {
            Dtype::F32 => match alloc_pooled::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc ReLU buffer failed ({}), using CPU",
                        err
                    );
                    return self.relu_cpu_fallback();
                }
            },
            Dtype::F64 => match alloc_pooled::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc ReLU buffer failed ({}), using CPU",
                        err
                    );
                    return self.relu_cpu_fallback();
                }
            },
            _ => return self.relu_cpu_fallback(),
        };
        let copy_ok = match (self.dtype, &d_data, &*d_src) {
            (Dtype::F32, CudaBuffer::F32(dst), CudaBuffer::F32(src)) => copy_d2d(dst, src).is_ok(),
            (Dtype::F64, CudaBuffer::F64(dst), CudaBuffer::F64(src)) => copy_d2d(dst, src).is_ok(),
            _ => false,
        };
        if !copy_ok {
            crate::cuda::record_activation_fallback("copy");
            log::warn!("[Autograd] CUDA D2D ReLU input copy failed, using CPU");
            return self.relu_cpu_fallback();
        }

        let kernel_ok = match (self.dtype, &d_data) {
            (Dtype::F32, CudaBuffer::F32(b)) => relu_inplace_f32(b).is_ok(),
            (Dtype::F64, CudaBuffer::F64(b)) => relu_inplace(b).is_ok(),
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA ReLU kernel failed, using CPU");
            return self.relu_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let out_dtype = self.dtype;
        let grad_dtype = Tensor::grad_dtype_for(out_dtype);
        let out = Tensor {
            data: Tensor::empty_storage(out_dtype),
            grad: Storage::zeros(len, grad_dtype),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    #[cfg(cuda)]
                    if input.device == Device::Cuda {
                        if let Some(d_input) = input.cuda_cached_buffer() {
                            let d_grad_tmp = cuda_grad_out_buffer(grad_out);
                            if let Some(d_grad_tmp) = d_grad_tmp {
                                if let Some(d_in_grad) = input.cuda_grad_ensure_buffer() {
                                    match (&*d_input, &*d_grad_tmp, &*d_in_grad, input.dtype) {
                                        (
                                            CudaBuffer::F32(inp),
                                            CudaBuffer::F32(gt),
                                            CudaBuffer::F32(ig),
                                            Dtype::F32,
                                        ) => {
                                            if relu_backward_f32(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        (
                                            CudaBuffer::F64(inp),
                                            CudaBuffer::F64(gt),
                                            CudaBuffer::F64(ig),
                                            Dtype::F64,
                                        ) => {
                                            if relu_backward(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &go), &val)| {
                                if val > 0.0 {
                                    *ig += go;
                                }
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            if input_data[i] > 0.0 {
                                inp_grad[i] += grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_data);
        out
    }

    /// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Tensor {
        // GPU routing
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            return self.gelu_cuda();
        }
        if self.dtype != Dtype::F64 {
            return self.gelu_generic();
        }

        let self_data = self.data_f64();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_chunks_mut(PAR_THRESHOLD)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * PAR_THRESHOLD;
                    vector_gelu(chunk, &self_data[start..start + chunk.len()]);
                });
        } else {
            vector_gelu(&mut data, &self_data);
        }
        let parents = vec![self.clone()];

        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let c = 0.044715;

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &go), &x)| {
                                let x2 = x * x;
                                let x3 = x2 * x;
                                let u = sqrt_2_over_pi * (x + c * x3);
                                let tanh_u = u.tanh();
                                let sech2_u = 1.0 - tanh_u * tanh_u;
                                let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                                let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                                *ig += go * gelu_grad;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            let x = input_data[i];
                            let x2 = x * x;
                            let x3 = x2 * x;
                            let u = sqrt_2_over_pi * (x + c * x3);
                            let tanh_u = u.tanh();
                            let sech2_u = 1.0 - tanh_u * tanh_u;
                            let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                            let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                            inp_grad[i] += grad_out_f64[i] * gelu_grad;
                        }
                    }
                }),
            })),
        }
    }

    fn gelu_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        let c = 0.044715f32;
        for i in 0..len {
            let x = self_f32[i];
            let x3 = x * x * x;
            let inner = sqrt_2_over_pi * (x + c * x3);
            data[i] = 0.5 * x * (1.0 + inner.tanh());
        }
        let input_cache: Arc<Vec<f64>> = Arc::new(self_f32.iter().map(|&v| v as f64).collect());
        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    for i in 0..len {
                        let x = input_cache[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = (2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx =
                            (2.0f64 / std::f64::consts::PI).sqrt() * (1.0 + 3.0 * 0.044715 * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out_f64[i] * gelu_grad;
                    }
                }),
            })),
        }
    }

    /// GPU-accelerated GELU activation using CUDA kernel
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn gelu_cuda(&self) -> Tensor {
        use crate::cuda::kernels::{
            gelu_backward, gelu_backward_f32, gelu_inplace, gelu_inplace_f32,
        };
        use crate::cuda::memory::{alloc_pooled, copy_d2d, CudaBuffer};

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.gelu_cpu_fallback(),
        };
        if len == 0 {
            return self.gelu_cpu_fallback();
        }
        crate::cuda::record_activation_attempt();

        let d_src = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_activation_fallback(stage);
                log::warn!(
                    "[Autograd] CUDA prepare GELU input failed ({}), using CPU",
                    err
                );
                return self.gelu_cpu_fallback();
            }
        };
        let d_data = match self.dtype {
            Dtype::F32 => match alloc_pooled::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc GELU buffer failed ({}), using CPU",
                        err
                    );
                    return self.gelu_cpu_fallback();
                }
            },
            Dtype::F64 => match alloc_pooled::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc GELU buffer failed ({}), using CPU",
                        err
                    );
                    return self.gelu_cpu_fallback();
                }
            },
            _ => return self.gelu_cpu_fallback(),
        };
        let copy_ok = match (self.dtype, &d_data, &*d_src) {
            (Dtype::F32, CudaBuffer::F32(dst), CudaBuffer::F32(src)) => copy_d2d(dst, src).is_ok(),
            (Dtype::F64, CudaBuffer::F64(dst), CudaBuffer::F64(src)) => copy_d2d(dst, src).is_ok(),
            _ => false,
        };
        if !copy_ok {
            crate::cuda::record_activation_fallback("copy");
            log::warn!("[Autograd] CUDA D2D GELU input copy failed, using CPU");
            return self.gelu_cpu_fallback();
        }

        let kernel_ok = match (self.dtype, &d_data) {
            (Dtype::F32, CudaBuffer::F32(b)) => gelu_inplace_f32(b).is_ok(),
            (Dtype::F64, CudaBuffer::F64(b)) => gelu_inplace(b).is_ok(),
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA GELU kernel failed, using CPU");
            return self.gelu_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let out_dtype = self.dtype;
        let grad_dtype = Tensor::grad_dtype_for(out_dtype);
        let out = Tensor {
            data: Tensor::empty_storage(out_dtype),
            grad: Storage::zeros(len, grad_dtype),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    #[cfg(cuda)]
                    if input.device == Device::Cuda {
                        if let Some(d_input) = input.cuda_cached_buffer() {
                            let d_grad_tmp = cuda_grad_out_buffer(grad_out);
                            if let Some(d_grad_tmp) = d_grad_tmp {
                                if let Some(d_in_grad) = input.cuda_grad_ensure_buffer() {
                                    match (&*d_input, &*d_grad_tmp, &*d_in_grad, input.dtype) {
                                        (
                                            CudaBuffer::F32(inp),
                                            CudaBuffer::F32(gt),
                                            CudaBuffer::F32(ig),
                                            Dtype::F32,
                                        ) => {
                                            if gelu_backward_f32(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        (
                                            CudaBuffer::F64(inp),
                                            CudaBuffer::F64(gt),
                                            CudaBuffer::F64(ig),
                                            Dtype::F64,
                                        ) => {
                                            if gelu_backward(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                    let c = 0.044715;
                    for i in 0..inp_grad.len() {
                        let x = input_data[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = sqrt_2_over_pi * (x + c * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out_f64[i] * gelu_grad;
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_data);
        out
    }

    /// CPU fallback for GELU
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn gelu_cpu_fallback(&self) -> Tensor {
        if self.dtype == Dtype::F32 {
            return self.gelu_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        for i in 0..len {
            let x = self_data[i];
            let x3 = x * x * x;
            let inner = 0.7978845608028654 * (x + 0.044715 * x3);
            data[i] = 0.5 * x * (1.0 + inner.tanh());
        }
        let parents = vec![self.clone()];
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let c = 0.044715;

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    for i in 0..inp_grad.len() {
                        let x = input_data[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = sqrt_2_over_pi * (x + c * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out_f64[i] * gelu_grad;
                    }
                }),
            })),
        }
    }
}
