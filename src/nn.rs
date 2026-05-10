use crate::autograd::{Context, Tensor};
use crate::dtype::Storage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        // Xavier initialization
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let weight = Tensor::rand(vec![in_features, out_features], -limit, limit, seed);
        let bias = if bias {
            Some(Tensor::zeros(vec![out_features]))
        } else {
            None
        };
        Linear {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn forward_inference(&self, input: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        self.forward_inference_into(input, &mut out);
        out
    }

    pub fn forward_inference_into(&self, input: &[f64], out: &mut Vec<f64>) {
        let in_dim = self.in_features;
        let out_dim = self.out_features;
        debug_assert!(
            in_dim > 0 && input.len().is_multiple_of(in_dim),
            "forward_inference: input length {} is not divisible by in_features {}",
            input.len(),
            in_dim
        );
        let num_rows = input.len() / in_dim;
        out.resize(num_rows * out_dim, 0.0);
        let w_data = self.weight.data_f64();
        let b_data = self.bias.as_ref().map(|b| b.data_f64());

        use crate::simd::add_scaled_row;

        // Disabled: nested parallelism causes thread pool oversubscription when called
        // from within an outer par_iter. External parallelism handles this better.
        let parallel_matvec = false;

        for r in 0..num_rows {
            let row_offset_in = r * in_dim;
            let row_offset_out = r * out_dim;

            if parallel_matvec {
                let out_row = &mut out[row_offset_out..row_offset_out + out_dim];
                let input_row = &input[row_offset_in..row_offset_in + in_dim];
                let n_chunks = rayon::current_num_threads().min(8);
                let chunk_size = out_dim.div_ceil(n_chunks);
                out_row
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let col_start = ci * chunk_size;
                        if let Some(b) = &b_data {
                            chunk.copy_from_slice(&b[col_start..col_start + chunk.len()]);
                        }
                        for i in 0..in_dim {
                            let scale = input_row[i];
                            if scale == 0.0 {
                                continue;
                            }
                            let w_slice = &w_data
                                [i * out_dim + col_start..i * out_dim + col_start + chunk.len()];
                            add_scaled_row(chunk, w_slice, scale);
                        }
                    });
            } else {
                let out_row = &mut out[row_offset_out..row_offset_out + out_dim];
                if let Some(b) = &b_data {
                    out_row.copy_from_slice(b);
                } else {
                    out_row.fill(0.0);
                }

                for i in 0..in_dim {
                    let scale = input[row_offset_in + i];
                    if scale == 0.0 {
                        continue;
                    }
                    let w_row = &w_data[i * out_dim..(i + 1) * out_dim];
                    add_scaled_row(out_row, w_row, scale);
                }
            }
        }
    }

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        if let Ok(t) = self.weight.to_cuda() {
            self.weight = t;
        }
        if let Some(ref mut b) = self.bias {
            if let Ok(t) = b.to_cuda() {
                *b = t;
            }
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Handle input flattening for N-D tensors (e.g. [Batch, Seq, Dim])
        let input_shape = &input.shape;
        let rank = input_shape.len();

        let (x_flat, is_flattened) = if rank > 2 {
            let batch_dim: usize = input_shape.iter().take(rank - 1).product();
            (input.reshape(vec![batch_dim, self.in_features]), true)
        } else {
            (input.clone(), false)
        };

        let out_flat = x_flat.matmul(&self.weight);

        let mut out = if is_flattened {
            let mut new_shape = input_shape[..rank - 1].to_vec();
            new_shape.push(self.out_features);
            out_flat.reshape(new_shape)
        } else {
            out_flat
        };

        if let Some(b) = &self.bias {
            // If out is [Batch, Out], broadcast b [Out] to [Batch, Out]
            // If out is [Batch, Seq, Out], flatten logic above handled matmul, but reshaping back
            // means out is 3D. We need to add bias to the last dim.
            // My Tensor::add currently requires exact shape match or simple broadcast.
            // Let's implement manual broadcast if needed, or rely on broadcast_to_batch if rank=2.

            if out.shape.len() == 2 && out.shape[0] > 1 {
                // Batch mode (common case)
                let batch_size = out.shape[0];
                let b_broadcast = b.broadcast_to_batch(batch_size);
                out = out + b_broadcast;
            } else if out.shape.len() > 2 {
                // N-D case: Flatten out again to add bias, then reshape back?
                // Or just iterate.
                // Let's flatten, add, reshape.
                let total_elements = out.shape.iter().product::<usize>();
                let batch_dim = total_elements / self.out_features;

                let out_flat = out.reshape(vec![batch_dim, self.out_features]);
                let b_broadcast = b.broadcast_to_batch(batch_dim);
                let res_flat = out_flat + b_broadcast;
                out = res_flat.reshape(out.shape.clone());
            } else {
                // Single vector or exact match
                if out.shape != b.shape {
                    if out.shape.len() == 2 && out.shape[0] == 1 {
                        let b_reshaped = b.reshape(vec![1, self.out_features]);
                        out = out + b_reshaped;
                    } else {
                        out = out + b.clone();
                    }
                } else {
                    out = out + b.clone();
                }
            }
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
    pub dim: usize,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, _seed: u64) -> Self {
        Self {
            weight: Tensor::new(vec![1.0; dim], vec![dim]),
            eps,
            dim,
        }
    }

    pub fn forward_inference(&self, input: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        self.forward_inference_into(input, &mut out);
        out
    }

    pub fn forward_inference_into(&self, input: &[f64], out: &mut Vec<f64>) {
        let dim = self.dim;
        let num_rows = input.len() / dim;
        out.resize(input.len(), 0.0);
        let w_data = self.weight.data_f64();

        for r in 0..num_rows {
            let base = r * dim;
            let mut sum_sq = 0.0;
            for i in 0..dim {
                let val = input[base + i];
                sum_sq += val * val;
            }
            let rms = (sum_sq / dim as f64 + self.eps).sqrt();
            for i in 0..dim {
                out[base + i] = (input[base + i] / rms) * w_data[i];
            }
        }
    }

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        if let Ok(t) = self.weight.to_cuda() {
            self.weight = t;
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [..., Dim]
        let shape = &x.shape;
        let last_dim = shape[shape.len() - 1];
        assert_eq!(last_dim, self.dim, "RMSNorm dim mismatch");

        let num_elements: usize = shape.iter().product();
        let num_rows = num_elements / self.dim;

        #[cfg(cuda)]
        if x.device == crate::autograd::Device::Cuda {
            if let Ok(d_x) = x.cuda_get_or_upload_buffer() {
                if let Ok(d_weight) = self.weight.cuda_get_or_upload_buffer() {
                    if let Ok(d_out) = crate::cuda::memory::alloc::<f64>(num_elements) {
                        let d_out = std::sync::Arc::new(d_out);
                        if crate::cuda::kernels::rmsnorm_forward(
                            &d_x, &d_weight, &d_out, self.dim, self.eps, num_rows,
                        )
                        .is_ok()
                        {
                            let parents = vec![x.clone(), self.weight.clone()];
                            let dim = self.dim;
                            let eps = self.eps;
                            let out = Tensor {
                                data: Storage::F64(Arc::new(RwLock::new(vec![0.0; num_elements]))),
                                grad: Arc::new(RwLock::new(vec![0.0; num_elements])),
                                shape: shape.clone(),
                                device: crate::autograd::Device::Cuda,
                                _ctx: Some(Arc::new(Context {
                                    parents,
                                    backward_op: Box::new(move |grad_out, parents| {
                                        let x_in = &parents[0];
                                        let w_in = &parents[1];

                                        #[cfg(cuda)]
                                        if x_in.device == crate::autograd::Device::Cuda {
                                            if let Ok(d_grad_tmp) =
                                                crate::cuda::memory::alloc_pooled::<f64>(
                                                    grad_out.len(),
                                                )
                                            {
                                                let d_grad_tmp = std::sync::Arc::new(d_grad_tmp);
                                                if crate::cuda::memory::copy_h2d(
                                                    &d_grad_tmp,
                                                    grad_out,
                                                )
                                                .is_ok()
                                                {
                                                    if let (Some(d_x), Some(d_weight)) = (
                                                        x_in.cuda_cached_buffer(),
                                                        w_in.cuda_cached_buffer(),
                                                    ) {
                                                        if let Some(d_x_grad) =
                                                            x_in.cuda_grad_ensure_buffer()
                                                        {
                                                            if let Some(d_w_grad) =
                                                                w_in.cuda_grad_ensure_buffer()
                                                            {
                                                                let _ = crate::cuda::kernels::rmsnorm_backward(
                                                                    &d_grad_tmp,
                                                                    &d_x,
                                                                    &d_weight,
                                                                    &d_x_grad,
                                                                    &d_w_grad,
                                                                    dim,
                                                                    eps,
                                                                    num_rows,
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }),
                                })),
                            };
                            out.cuda_set_cached_buffer(d_out);
                            return out;
                        }
                    }
                }
            }
            log::warn!("[RMSNorm] CUDA forward failed, falling back to CPU");
        }

        let x_data = x.data_f64();
        let w_data = self.weight.data_f64();

        let mut out_data = vec![0.0; num_elements];
        let mut rms_cache = vec![0.0; num_rows];
        let mut x_hat_cache = vec![0.0; num_elements];

        out_data
            .par_chunks_mut(self.dim)
            .zip(x_hat_cache.par_chunks_mut(self.dim))
            .zip(rms_cache.par_iter_mut())
            .enumerate()
            .for_each(|(r, ((out_row, x_hat_row), rms_ref))| {
                let base = r * self.dim;
                let mut sum_sq = 0.0;
                for i in 0..self.dim {
                    let val = x_data[base + i];
                    sum_sq += val * val;
                }
                let rms = (sum_sq / self.dim as f64 + self.eps).sqrt();
                *rms_ref = rms;

                for i in 0..self.dim {
                    let val = x_data[base + i];
                    let x_hat = val / rms;
                    x_hat_row[i] = x_hat;
                    out_row[i] = x_hat * w_data[i];
                }
            });

        let parents = vec![x.clone(), self.weight.clone()];
        let dim = self.dim;

        let rms_cache = Arc::new(rms_cache);
        let x_hat_cache = Arc::new(x_hat_cache);

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Arc::new(RwLock::new(vec![0.0; num_elements])),
            shape: shape.clone(),
            device: x.device,
            dtype: x.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let x_in = &parents[0];
                    let w_in = &parents[1];

                    let mut x_grad = x_in.grad.write().unwrap();
                    let mut w_grad = w_in.grad.write().unwrap();
                    let w_data = w_in.data_f64();

                    // 1. Calculate dL/dx parallel over rows
                    x_grad
                        .par_chunks_mut(dim)
                        .zip(grad_out.par_chunks(dim))
                        .enumerate()
                        .for_each(|(r, (x_g_row, g_out_row))| {
                            let base = r * dim;
                            let rms = rms_cache[r];
                            let inv_rms = 1.0 / rms;

                            let mut dot_sum = 0.0;
                            for i in 0..dim {
                                let g = g_out_row[i];
                                let w = w_data[i];
                                let dl_dxhat = g * w;
                                dot_sum += dl_dxhat * x_hat_cache[base + i];
                            }

                            let mean_dot = dot_sum / dim as f64;

                            for i in 0..dim {
                                let g = g_out_row[i];
                                let w = w_data[i];
                                let dl_dxhat = g * w;
                                let x_hat = x_hat_cache[base + i];
                                x_g_row[i] += inv_rms * (dl_dxhat - x_hat * mean_dot);
                            }
                        });

                    // 2. Accumulate weight gradient (reduction over batch)
                    let num_rows = grad_out.len() / dim;

                    // Parallelize over dimension (feature)
                    w_grad.par_iter_mut().enumerate().for_each(|(i, wg)| {
                        let mut sum = 0.0;
                        for r in 0..num_rows {
                            let base = r * dim;
                            sum += grad_out[base + i] * x_hat_cache[base + i];
                        }
                        *wg += sum;
                    });
                }),
            })),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
