use crate::autograd::{Context, Device, Tensor, TensorReadGuard, PAR_THRESHOLD};
use crate::dtype::{Dtype, Storage};
use crate::simd::{vector_add, vector_grad_acc, vector_mul, vector_sub};
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, RwLock};

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Add data length mismatch");
        if self.dtype != rhs.dtype {
            return self.add_generic(&rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.add_cuda(&rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.add_generic(&rhs);
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] + rhs_data[i]);
        } else {
            vector_add(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg += g);
                        } else {
                            vector_grad_acc(&mut rhs_grad, &grad_out_f64);
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Add<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Add data length mismatch");
        if self.dtype != rhs.dtype {
            return self.add_generic(rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.add_cuda(rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.add_generic(rhs);
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] + rhs_data[i]);
        } else {
            vector_add(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg += g);
                        } else {
                            vector_grad_acc(&mut rhs_grad, &grad_out_f64);
                        }
                    }
                }),
            })),
        }
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Sub data length mismatch");
        if self.dtype != rhs.dtype {
            return self.sub_generic(&rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.sub_cuda(&rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.sub_generic(&rhs);
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] - rhs_data[i]);
        } else {
            vector_sub(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg -= g);
                        } else {
                            for i in 0..len {
                                rhs_grad[i] -= grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Sub<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Sub data length mismatch");
        if self.dtype != rhs.dtype {
            return self.sub_generic(rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.sub_cuda(rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.sub_generic(rhs);
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] - rhs_data[i]);
        } else {
            vector_sub(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg -= g);
                        } else {
                            for i in 0..len {
                                rhs_grad[i] -= grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Mul data length mismatch");
        if self.dtype != rhs.dtype {
            return self.mul_generic(&rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.mul_cuda(&rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.mul_generic(&rhs);
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] * rhs_data[i]);
        } else {
            vector_mul(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            grad.par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(lhs_data.par_iter())
                                .zip(rhs_data.par_iter())
                                .for_each(|(((g, &go), &l), &r)| {
                                    *g += go * (l + r);
                                });
                        } else {
                            for i in 0..len {
                                grad[i] += grad_out_f64[i] * (lhs_data[i] + rhs_data[i]);
                            }
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                lhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(rhs_data.par_iter())
                                    .for_each(|((lg, &g), &r)| *lg += g * r);
                            } else {
                                for i in 0..len {
                                    lhs_grad[i] += grad_out_f64[i] * rhs_data[i];
                                }
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                rhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(lhs_data.par_iter())
                                    .for_each(|((rg, &g), &l)| *rg += g * l);
                            } else {
                                for i in 0..len {
                                    rhs_grad[i] += grad_out_f64[i] * lhs_data[i];
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Mul<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Mul data length mismatch");
        if self.dtype != rhs.dtype {
            return self.mul_generic(rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.mul_cuda(rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.mul_generic(rhs);
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] * rhs_data[i]);
        } else {
            vector_mul(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        for i in 0..len {
                            grad[i] += grad_out_f64[i] * (lhs_data[i] + rhs_data[i]);
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            for i in 0..len {
                                lhs_grad[i] += grad_out_f64[i] * rhs_data[i];
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            for i in 0..len {
                                rhs_grad[i] += grad_out_f64[i] * lhs_data[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Div data length mismatch");
        if self.dtype != rhs.dtype {
            return self.div_generic(&rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.div_cuda(&rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.div_generic(&rhs);
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data
                .par_iter()
                .zip(rhs_data.par_iter())
                .map(|(a, b)| a / b)
                .collect()
        } else {
            self_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a / b)
                .collect()
        };
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    const DIV_EPS: f64 = 1e-12;
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            grad.par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(lhs_data.par_iter())
                                .zip(rhs_data.par_iter())
                                .for_each(|(((g, &go), &l), &r)| {
                                    let safe_r = if r.abs() < DIV_EPS {
                                        r.signum() * DIV_EPS
                                    } else {
                                        r
                                    };
                                    *g += go / safe_r - go * l / (safe_r * safe_r);
                                });
                        } else {
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                grad[i] += grad_out_f64[i] / safe_r
                                    - grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                            }
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                lhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(rhs_data.par_iter())
                                    .for_each(|((lg, &g), &r)| {
                                        let safe_r = if r.abs() < DIV_EPS {
                                            r.signum() * DIV_EPS
                                        } else {
                                            r
                                        };
                                        *lg += g / safe_r;
                                    });
                            } else {
                                for i in 0..len {
                                    let r = rhs_data[i];
                                    let safe_r = if r.abs() < DIV_EPS {
                                        r.signum() * DIV_EPS
                                    } else {
                                        r
                                    };
                                    lhs_grad[i] += grad_out_f64[i] / safe_r;
                                }
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                rhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(lhs_data.par_iter())
                                    .zip(rhs_data.par_iter())
                                    .for_each(|(((rg, &g), &l), &r)| {
                                        let safe_r = if r.abs() < DIV_EPS {
                                            r.signum() * DIV_EPS
                                        } else {
                                            r
                                        };
                                        *rg -= g * l / (safe_r * safe_r);
                                    });
                            } else {
                                for i in 0..len {
                                    let r = rhs_data[i];
                                    let safe_r = if r.abs() < DIV_EPS {
                                        r.signum() * DIV_EPS
                                    } else {
                                        r
                                    };
                                    rhs_grad[i] -=
                                        grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Div<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.numel(), rhs.numel(), "Div data length mismatch");
        if self.dtype != rhs.dtype {
            return self.div_generic(rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.div_cuda(rhs) {
            return out;
        }
        if self.dtype == Dtype::F32 {
            return self.div_generic(rhs);
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let data: Vec<f64> = self_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(a, b)| a / b)
            .collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    const DIV_EPS: f64 = 1e-12;
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        for i in 0..len {
                            let r = rhs_data[i];
                            let safe_r = if r.abs() < DIV_EPS {
                                r.signum() * DIV_EPS
                            } else {
                                r
                            };
                            grad[i] += grad_out_f64[i] / safe_r
                                - grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                lhs_grad[i] += grad_out_f64[i] / safe_r;
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                rhs_grad[i] -= grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        if self.dtype != Dtype::F64 {
            let self_data = self.data_to_f32_vec();
            let len = self_data.len();
            let data: Vec<f32> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| -x).collect()
            } else {
                self_data.iter().map(|&x| -x).collect()
            };
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::from_f32_vec(data, self.dtype),
                grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
                shape: self.shape.clone(),
                device: Device::Cpu,
                dtype: self.dtype,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(|grad_out, parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = parents[0].grad_write_compat();
                        let len = grad_out_f64.len();
                        if len >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(ig, &g)| *ig -= g);
                        } else {
                            for i in 0..len {
                                inp_grad[i] -= grad_out_f64[i];
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_f64();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| -x).collect()
        } else {
            self_data.iter().map(|&x| -x).collect()
        };
        let parents = vec![self.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_f64();
                    let len = grad_out_f64.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .for_each(|(ig, &g)| *ig -= g);
                    } else {
                        for i in 0..len {
                            inp_grad[i] -= grad_out_f64[i];
                        }
                    }
                }),
            })),
        }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        -self.clone()
    }
}
