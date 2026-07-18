use crate::autograd::{Context, Device, Tensor, PAR_THRESHOLD};
use crate::dtype::{Dtype, Storage};
use rayon::prelude::*;
use std::sync::Arc;

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

impl Tensor {
    pub fn log(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.ln()).collect()
            } else {
                self_data.iter().map(|&x| x.ln()).collect()
            };
            let input_cache = Arc::new(self_data);
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        const LOG_GRAD_EPS: f64 = 1e-12;
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_cache.par_iter())
                                .for_each(|((ig, &g), &id)| {
                                    let safe = if id.abs() < LOG_GRAD_EPS {
                                        id.signum() * LOG_GRAD_EPS
                                    } else {
                                        id
                                    };
                                    *ig += g / safe;
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                let safe = if input_cache[i].abs() < LOG_GRAD_EPS {
                                    input_cache[i].signum() * LOG_GRAD_EPS
                                } else {
                                    input_cache[i]
                                };
                                inp_grad[i] += grad_out_f64[i] / safe;
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.ln()).collect()
        } else {
            self_data.iter().map(|&x| x.ln()).collect()
        };
        let parents = vec![self.clone()];

        let input_cache_f64: Arc<Vec<f64>> =
            Arc::new(self_data.iter().map(|&v| v as f64).collect());
        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    const LOG_GRAD_EPS: f64 = 1e-12;
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_cache_f64.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                let safe = if id.abs() < LOG_GRAD_EPS {
                                    id.signum() * LOG_GRAD_EPS
                                } else {
                                    id
                                };
                                *ig += g / safe;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            let safe = if input_cache_f64[i].abs() < LOG_GRAD_EPS {
                                input_cache_f64[i].signum() * LOG_GRAD_EPS
                            } else {
                                input_cache_f64[i]
                            };
                            inp_grad[i] += grad_out_f64[i] / safe;
                        }
                    }
                }),
            })),
        }
    }

    pub fn exp(&self) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.exp_cuda() {
            return out;
        }

        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.exp()).collect()
            } else {
                self_data.iter().map(|&x| x.exp()).collect()
            };
            let parents = vec![self.clone()];
            let exp_cache = Arc::new(data.clone());
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(exp_cache.par_iter())
                                .for_each(|((ig, &g), &cached)| {
                                    *ig += g * cached;
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] += grad_out_f64[i] * exp_cache[i];
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let mut data = vec![0.0f32; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.exp()).collect();
        } else {
            crate::simd::fast_exp_bulk_f32(&mut data, &self_data);
        }
        let parents = vec![self.clone()];
        let exp_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|&v| v as f64).collect());

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(exp_cache.par_iter())
                            .for_each(|((ig, &g), &cached)| {
                                *ig += g * cached;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * exp_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    /// Element-wise absolute value.
    /// Forward: |x|
    /// Backward: d/dx|x| = sign(x), where sign(0) = 0
    pub fn abs(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let mut data = vec![0.0f64; len];
            if len >= PAR_THRESHOLD {
                data = self_data.par_iter().map(|&x| x.abs()).collect();
            } else {
                for i in 0..len {
                    data[i] = self_data[i].abs();
                }
            }
            let sign_cache: Arc<Vec<f64>> = Arc::new(
                self_data
                    .iter()
                    .map(|&x| {
                        if x > 0.0 {
                            1.0
                        } else if x < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(sign_cache.par_iter())
                                .for_each(|((ig, &g), &s)| {
                                    *ig += g * s;
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] += grad_out_f64[i] * sign_cache[i];
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let mut data = vec![0.0f32; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.abs()).collect();
        } else {
            for i in 0..len {
                data[i] = self_data[i].abs();
            }
        }
        let sign_cache: Arc<Vec<f64>> = Arc::new(
            self_data
                .iter()
                .map(|&x| {
                    if x > 0.0 {
                        1.0f64
                    } else if x < 0.0 {
                        -1.0f64
                    } else {
                        0.0f64
                    }
                })
                .collect(),
        );
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(sign_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                *ig += g * s;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * sign_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    /// Element-wise exponentiation: x^exponent.
    /// Forward: x^n
    /// Backward: d/dx x^n = n * x^(n-1)
    pub fn pow(&self, exponent: f64) -> Tensor {
        let self_data = self.data_as_f64_vec();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.powf(exponent)).collect();
        } else {
            for i in 0..len {
                data[i] = self_data[i].powf(exponent);
            }
        }
        let exp = exponent;
        let pow_cache: Arc<Vec<f64>> =
            Arc::new(self_data.iter().map(|&x| x.powf(exp - 1.0)).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f64_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(pow_cache.par_iter())
                            .for_each(|((ig, &g), &cached)| {
                                *ig += g * exp * cached;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * exp * pow_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    pub fn sin(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.sin()).collect()
            } else {
                self_data.iter().map(|&x| x.sin()).collect()
            };
            let input_cache = Arc::new(self_data);
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_cache.par_iter())
                                .for_each(|((ig, &g), &id)| {
                                    *ig += g * id.cos();
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] += grad_out_f64[i] * input_cache[i].cos();
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.sin()).collect()
        } else {
            self_data.iter().map(|&x| x.sin()).collect()
        };
        let input_cache: Arc<Vec<f64>> = Arc::new(self_data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_cache.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig += g * id.cos();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * input_cache[i].cos();
                        }
                    }
                }),
            })),
        }
    }

    pub fn cos(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.cos()).collect()
            } else {
                self_data.iter().map(|&x| x.cos()).collect()
            };
            let input_cache = Arc::new(self_data);
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_cache.par_iter())
                                .for_each(|((ig, &g), &id)| {
                                    *ig -= g * id.sin();
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] -= grad_out_f64[i] * input_cache[i].sin();
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.cos()).collect()
        } else {
            self_data.iter().map(|&x| x.cos()).collect()
        };
        let input_cache: Arc<Vec<f64>> = Arc::new(self_data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_cache.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig -= g * id.sin();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] -= grad_out_f64[i] * input_cache[i].sin();
                        }
                    }
                }),
            })),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.sqrt_cuda() {
            return out;
        }

        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.sqrt()).collect()
            } else {
                self_data.iter().map(|&x| x.sqrt()).collect()
            };
            let sqrt_cache = data.clone();
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        let len = inp_grad.len();
                        if len >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(sqrt_cache.par_iter())
                                .for_each(|((ig, &g), &s)| {
                                    if s > 0.0 {
                                        *ig += g * 0.5 / s;
                                    }
                                });
                        } else {
                            for i in 0..len {
                                if sqrt_cache[i] > 0.0 {
                                    inp_grad[i] += grad_out_f64[i] * 0.5 / sqrt_cache[i];
                                }
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.sqrt()).collect()
        } else {
            self_data.iter().map(|&x| x.sqrt()).collect()
        };
        let sqrt_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let len = inp_grad.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(sqrt_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                if s > 0.0 {
                                    *ig += g * 0.5 / s;
                                }
                            });
                    } else {
                        for i in 0..len {
                            if sqrt_cache[i] > 0.0 {
                                inp_grad[i] += grad_out_f64[i] * 0.5 / sqrt_cache[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn tanh(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.tanh()).collect()
            } else {
                self_data.iter().map(|&x| x.tanh()).collect()
            };
            let out_cache = Arc::new(data.clone());
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: Device::Cpu,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(out_cache.par_iter())
                                .for_each(|((ig, &g), &y)| {
                                    *ig += g * (1.0 - y * y);
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                let y = out_cache[i];
                                inp_grad[i] += grad_out_f64[i] * (1.0 - y * y);
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.tanh()).collect()
        } else {
            self_data.iter().map(|&x| x.tanh()).collect()
        };
        let out_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(out_cache.par_iter())
                            .for_each(|((ig, &g), &y)| {
                                *ig += g * (1.0 - y * y);
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            let y = out_cache[i];
                            inp_grad[i] += grad_out_f64[i] * (1.0 - y * y);
                        }
                    }
                }),
            })),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| sigmoid_f64(x)).collect()
            } else {
                self_data.iter().map(|&x| sigmoid_f64(x)).collect()
            };
            let out_cache = Arc::new(data.clone());
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: Device::Cpu,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(out_cache.par_iter())
                                .for_each(|((ig, &g), &y)| {
                                    *ig += g * y * (1.0 - y);
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                let y = out_cache[i];
                                inp_grad[i] += grad_out_f64[i] * y * (1.0 - y);
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| sigmoid_f32(x)).collect()
        } else {
            self_data.iter().map(|&x| sigmoid_f32(x)).collect()
        };
        let out_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(out_cache.par_iter())
                            .for_each(|((ig, &g), &y)| {
                                *ig += g * y * (1.0 - y);
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            let y = out_cache[i];
                            inp_grad[i] += grad_out_f64[i] * y * (1.0 - y);
                        }
                    }
                }),
            })),
        }
    }

    pub fn clamp(&self, min: f64, max: f64) -> Tensor {
        self.clip(min, max)
    }

    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        let self_data = self.data_as_f64_vec();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.max(min).min(max)).collect()
        } else {
            self_data.iter().map(|&x| x.max(min).min(max)).collect()
        };
        let input_cache = Arc::new(self_data);
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f64_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let mut inp_grad = input.grad_write_compat();
                    let len = inp_grad.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_cache.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                if id >= min && id <= max {
                                    *ig += g;
                                }
                            });
                    } else {
                        for i in 0..len {
                            if input_cache[i] >= min && input_cache[i] <= max {
                                inp_grad[i] += grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}
