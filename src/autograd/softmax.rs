use crate::autograd::{Context, Device, Tensor};
use crate::dtype::{Dtype, Storage};
use crate::simd::softmax_exp_sum;
use std::sync::{Arc, RwLock};

impl Tensor {
    /// Softmax: exp(x_i) / sum_j(exp(x_j)) with numerical stability (shift by max)
    pub fn softmax(&self) -> Tensor {
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            return self.softmax_cuda();
        }
        if self.dtype != Dtype::F64 {
            return self.softmax_generic();
        }

        let self_data = self.data_f64();
        let len = self_data.len();
        let cols = self.shape.last().copied().unwrap_or(len.max(1));
        let rows = len.checked_div(cols).unwrap_or(0);
        assert!(
            rows.checked_mul(cols) == Some(len),
            "Softmax shape mismatch"
        );

        let mut data = vec![0.0; len];
        for row in 0..rows {
            let base = row * cols;
            data[base..base + cols].copy_from_slice(&self_data[base..base + cols]);
            let sum_exp = softmax_exp_sum(&mut data[base..base + cols]);
            for j in 0..cols {
                data[base + j] /= sum_exp;
            }
        }

        let softmax_cache: Arc<Vec<f64>> = Arc::new(data.to_vec());
        let parents = vec![self.clone()];
        let rows_cap = rows;
        let cols_cap = cols;

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
                    let mut inp_grad = parents[0].grad_write_compat();
                    for row in 0..rows_cap {
                        let base = row * cols_cap;
                        let mut sum_term = 0.0;
                        for j in 0..cols_cap {
                            let idx = base + j;
                            sum_term += grad_out_f64[idx] * softmax_cache[idx];
                        }
                        for j in 0..cols_cap {
                            let idx = base + j;
                            inp_grad[idx] += softmax_cache[idx] * (grad_out_f64[idx] - sum_term);
                        }
                    }
                }),
            })),
        }
    }

    fn softmax_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let len = self_f32.len();
        let cols = self.shape.last().copied().unwrap_or(len.max(1));
        let rows = len.checked_div(cols).unwrap_or(0);
        assert!(
            rows.checked_mul(cols) == Some(len),
            "Softmax shape mismatch"
        );

        let mut data = vec![0.0f32; len];
        for row in 0..rows {
            let base = row * cols;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..cols {
                if self_f32[base + j] > max_val {
                    max_val = self_f32[base + j];
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..cols {
                let e = (self_f32[base + j] - max_val).exp();
                data[base + j] = e;
                sum_exp += e;
            }
            for j in 0..cols {
                data[base + j] /= sum_exp;
            }
        }

        let softmax_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|&v| v as f64).collect());
        let rows_cap = rows;
        let cols_cap = cols;

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
                    for row in 0..rows_cap {
                        let base = row * cols_cap;
                        let mut sum_term = 0.0f64;
                        for j in 0..cols_cap {
                            let idx = base + j;
                            sum_term += grad_out_f64[idx] * softmax_cache[idx];
                        }
                        for j in 0..cols_cap {
                            let idx = base + j;
                            inp_grad[idx] += softmax_cache[idx] * (grad_out_f64[idx] - sum_term);
                        }
                    }
                }),
            })),
        }
    }

    /// Log-softmax along specified dimension: log(softmax(x)) along dim.
    /// Fused for efficiency with numerical stability.
    pub fn log_softmax_dim(&self, dim: usize) -> Tensor {
        assert!(dim < self.shape.len(), "dim out of bounds");
        #[cfg(cuda)]
        if self.device == Device::Cuda && dim == self.shape.len() - 1 {
            return self.log_softmax_cuda_last_dim();
        }

        let self_data = self.data_as_f64_vec();
        let shape = self.shape.clone();
        let dim_size = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();

        let mut data = vec![0.0; self_data.len()];
        let mut softmax_values = vec![0.0; self_data.len()];

        if inner == 1 {
            for outer_idx in 0..outer {
                let base = outer_idx * dim_size;
                let mut max_val = f64::NEG_INFINITY;
                for dim_idx in 0..dim_size {
                    let v = self_data[base + dim_idx];
                    if v > max_val {
                        max_val = v;
                    }
                }
                let mut sum_exp = 0.0;
                for dim_idx in 0..dim_size {
                    let e = (self_data[base + dim_idx] - max_val).exp();
                    softmax_values[base + dim_idx] = e;
                    sum_exp += e;
                }
                sum_exp = sum_exp.max(f64::MIN_POSITIVE);
                let log_sum_exp = sum_exp.ln() + max_val;
                for dim_idx in 0..dim_size {
                    let idx = base + dim_idx;
                    data[idx] = self_data[idx] - log_sum_exp;
                    softmax_values[idx] /= sum_exp;
                }
            }
        } else {
            for outer_idx in 0..outer {
                let base_outer = outer_idx * dim_size * inner;
                for inner_idx in 0..inner {
                    let mut max_val = f64::NEG_INFINITY;
                    for dim_idx in 0..dim_size {
                        let idx = base_outer + dim_idx * inner + inner_idx;
                        if self_data[idx] > max_val {
                            max_val = self_data[idx];
                        }
                    }

                    let mut sum_exp = 0.0;
                    for dim_idx in 0..dim_size {
                        let idx = base_outer + dim_idx * inner + inner_idx;
                        sum_exp += (self_data[idx] - max_val).exp();
                    }
                    sum_exp = sum_exp.max(f64::MIN_POSITIVE);
                    let log_sum_exp = sum_exp.ln() + max_val;

                    for dim_idx in 0..dim_size {
                        let idx = base_outer + dim_idx * inner + inner_idx;
                        let softmax_val = (self_data[idx] - max_val).exp() / sum_exp;
                        data[idx] = self_data[idx] - log_sum_exp;
                        softmax_values[idx] = softmax_val;
                    }
                }
            }
        }

        let softmax_cache: Arc<Vec<f64>> = Arc::new(softmax_values);
        let parents = vec![self.clone()];
        let dim_size_cap = dim_size;
        let inner_cap = inner;
        let outer_cap = outer;
        let softmax_cache_for_backward = softmax_cache.clone();

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(self_data.len(), Tensor::grad_dtype_for(Dtype::F64)),
            shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    if inner_cap == 1 {
                        for outer_idx in 0..outer_cap {
                            let base = outer_idx * dim_size_cap;
                            let mut sum_term = 0.0;
                            for dim_idx in 0..dim_size_cap {
                                sum_term += grad_out_f64[base + dim_idx];
                            }
                            for dim_idx in 0..dim_size_cap {
                                let idx = base + dim_idx;
                                inp_grad[idx] +=
                                    grad_out_f64[idx] - softmax_cache_for_backward[idx] * sum_term;
                            }
                        }
                    } else {
                        for outer_idx in 0..outer_cap {
                            let base_outer = outer_idx * dim_size_cap * inner_cap;
                            for inner_idx in 0..inner_cap {
                                let mut sum_term = 0.0;
                                for dim_idx in 0..dim_size_cap {
                                    let idx = base_outer + dim_idx * inner_cap + inner_idx;
                                    sum_term += grad_out_f64[idx];
                                }
                                for dim_idx in 0..dim_size_cap {
                                    let idx = base_outer + dim_idx * inner_cap + inner_idx;
                                    inp_grad[idx] += grad_out_f64[idx]
                                        - softmax_cache_for_backward[idx] * sum_term;
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }

    /// Fused log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    /// Single allocation instead of 6+ intermediate tensors.
    /// Uses log_softmax_dim internally for backward compatibility.
    pub fn log_softmax(&self) -> Tensor {
        self.log_softmax_dim(self.shape.len() - 1)
    }

    /// Layer Normalization (simplified): normalize over the last dimension.
    /// y = (x - mean) / sqrt(var + eps)
    /// No learnable gamma/beta parameters.
    pub fn layer_norm_simple(&self, eps: f64) -> Tensor {
        let self_data = self.data_as_f64_vec();
        let shape = self.shape.clone();
        let ndim = shape.len();
        assert!(ndim >= 1, "layer_norm requires at least 1D tensor");

        let last_dim = shape[ndim - 1];
        let outer_len = shape[..ndim - 1].iter().product();

        let mut mean = vec![0.0; outer_len];
        for (i, mean_elem) in mean.iter_mut().enumerate().take(outer_len) {
            let base = i * last_dim;
            let mut sum = 0.0;
            for j in 0..last_dim {
                sum += self_data[base + j];
            }
            *mean_elem = sum / last_dim as f64;
        }

        let mut var = vec![0.0; outer_len];
        let mut output = vec![0.0; self_data.len()];
        for i in 0..outer_len {
            let base = i * last_dim;
            let m = mean[i];
            let mut sum_sq = 0.0;
            for j in 0..last_dim {
                let diff = self_data[base + j] - m;
                sum_sq += diff * diff;
            }
            var[i] = sum_sq / last_dim as f64;
            let inv_std = 1.0 / (var[i] + eps).sqrt();
            let slice = &self_data[base..base + last_dim];
            let normalized = crate::simd::layer_norm(slice, m, inv_std, &[], &[]);
            output[base..base + last_dim].copy_from_slice(&normalized);
        }

        let input_data = Arc::new(self_data);
        let mean_arc = Arc::new(mean);
        let var_arc = Arc::new(var);
        let last_dim_f = last_dim as f64;
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f64_vec(output, self.dtype),
            grad: Storage::zeros(input_data.len(), Tensor::grad_dtype_for(self.dtype)),
            shape,
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();

                    for i in 0..outer_len {
                        let base = i * last_dim;
                        let m = mean_arc[i];
                        let v = var_arc[i];
                        let std = (v + eps).sqrt();
                        let std3 = std * std * std;

                        let mut g_sum = 0.0;
                        let mut g_diff_sum = 0.0;
                        for j in 0..last_dim {
                            let diff = input_data[base + j] - m;
                            g_sum += grad_out_f64[base + j];
                            g_diff_sum += grad_out_f64[base + j] * diff;
                        }

                        let dvar = -0.5 * g_diff_sum / std3;
                        let dmean = -g_sum / std + dvar * -2.0 * m / last_dim_f;

                        for j in 0..last_dim {
                            let diff = input_data[base + j] - m;
                            let dx = grad_out_f64[base + j] / std
                                + dvar * 2.0 * diff / last_dim_f
                                + dmean / last_dim_f;
                            inp_grad[base + j] += dx;
                        }
                    }
                }),
            })),
        }
    }
}
