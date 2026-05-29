use crate::autograd::{Context, Device, Tensor};
use crate::dtype::Storage;
use std::sync::Arc;

impl Tensor {
    // Generic element-wise binary ops for non-F64 dtypes.
    // -------------------------------------------------------------------------

    pub(super) fn assert_same_numel(&self, rhs: &Tensor, op: &'static str) {
        assert_eq!(
            self.numel(),
            rhs.numel(),
            "{} data length mismatch: left shape {:?} ({} elems), right shape {:?} ({} elems)",
            op,
            self.shape,
            self.numel(),
            rhs.shape,
            rhs.numel()
        );
    }

    pub(super) fn add_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "add_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] + rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        let mut grad = _parents[0].grad_write_compat();
                        for i in 0..out_len {
                            grad[i] += grad_out_f64[i] * 2.0;
                        }
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i];
                            rhs_grad[i] += grad_out_f64[i];
                        }
                    }
                }),
            })),
        }
    }

    pub(super) fn sub_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "sub_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] - rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        // d/dx (x - x) = 0, no-op
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i];
                            rhs_grad[i] -= grad_out_f64[i];
                        }
                    }
                }),
            })),
        }
    }

    pub(super) fn mul_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "mul_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] * rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        let rhs_cache: Arc<Vec<f64>> = Arc::new(rhs_f32.iter().map(|&v| v as f64).collect());
        let self_cache: Arc<Vec<f64>> = Arc::new(self_f32.iter().map(|&v| v as f64).collect());
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        let mut grad = _parents[0].grad_write_compat();
                        for i in 0..out_len {
                            grad[i] += grad_out_f64[i] * 2.0 * rhs_cache[i];
                        }
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i] * rhs_cache[i];
                            rhs_grad[i] += grad_out_f64[i] * self_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    pub(super) fn div_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "div_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] / rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        let rhs_cache: Arc<Vec<f64>> = Arc::new(rhs_f32.iter().map(|&v| v as f64).collect());
        let self_cache: Arc<Vec<f64>> = Arc::new(self_f32.iter().map(|&v| v as f64).collect());
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        // d/dx (x/x) = 0, no-op
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i] / rhs_cache[i];
                            rhs_grad[i] +=
                                grad_out_f64[i] * (-self_cache[i] / (rhs_cache[i] * rhs_cache[i]));
                        }
                    }
                }),
            })),
        }
    }
}
