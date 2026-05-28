use crate::autograd::{Context, Tensor, PAR_THRESHOLD};
use crate::dtype::{Dtype, Storage};
use crate::simd::horizontal_sum;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

impl Tensor {
    /// Select a single element by index, producing a scalar Tensor with gradient support.
    pub fn index_select(&self, idx: usize) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.index_select_cuda(idx) {
            return out;
        }
        let self_data = self.data_f64();
        let val = self_data[idx];
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![val]))),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    if idx < inp_grad.len() {
                        inp_grad[idx] += grad_out_f64[0];
                    }
                }),
            })),
        }
    }

    pub fn sum(&self) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.sum_cuda(1.0) {
            return out;
        }

        if self.dtype != Dtype::F64 {
            return self.sum_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let sum_val: f64 = if len >= PAR_THRESHOLD {
            self_data.par_iter().sum()
        } else {
            horizontal_sum(&self_data)
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![sum_val]))),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    let g = grad_out_f64[0];
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad.par_iter_mut().for_each(|v| *v += g);
                    } else {
                        for v in inp_grad.iter_mut() {
                            *v += g;
                        }
                    }
                }),
            })),
        }
    }

    pub fn mean(&self) -> Tensor {
        #[cfg(cuda)]
        if self.numel() > 0 {
            if let Some(out) = self.sum_cuda(1.0 / self.numel() as f64) {
                return out;
            }
        }

        if self.dtype != Dtype::F64 {
            return self.mean_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let sum_val: f64 = if len >= PAR_THRESHOLD {
            self_data.par_iter().sum()
        } else {
            horizontal_sum(&self_data)
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![sum_val / len as f64]))),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    let g = grad_out_f64[0] / len as f64;
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad.par_iter_mut().for_each(|v| *v += g);
                    } else {
                        for v in inp_grad.iter_mut() {
                            *v += g;
                        }
                    }
                }),
            })),
        }
    }

    fn sum_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let sum_val: f32 = self_f32.iter().sum();
        Tensor {
            data: Storage::from_f32_vec(vec![sum_val], self.dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(self.dtype)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let g = grad_out_f64[0];
                    for v in inp_grad.iter_mut() {
                        *v += g;
                    }
                }),
            })),
        }
    }

    fn mean_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let len = self_f32.len();
        let mean_val = self_f32.iter().sum::<f32>() / len as f32;
        Tensor {
            data: Storage::from_f32_vec(vec![mean_val], self.dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(self.dtype)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let g = grad_out_f64[0] / len as f64;
                    for v in inp_grad.iter_mut() {
                        *v += g;
                    }
                }),
            })),
        }
    }
}
