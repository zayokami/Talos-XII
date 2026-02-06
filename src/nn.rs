use crate::autograd::{Context, Tensor};
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
        let in_dim = self.in_features;
        let out_dim = self.out_features;
        let num_rows = input.len() / in_dim;
        let mut out = vec![0.0; num_rows * out_dim];
        let w_data = self.weight.data.read().unwrap();
        let b_data = self.bias.as_ref().map(|b| b.data.read().unwrap());

        use crate::simd::add_scaled_row;

        for r in 0..num_rows {
            let row_offset_in = r * in_dim;
            let row_offset_out = r * out_dim;

            // Initialize with bias if present
            if let Some(b) = &b_data {
                let out_row = &mut out[row_offset_out..row_offset_out + out_dim];
                out_row.copy_from_slice(b);
            }

            for i in 0..in_dim {
                let scale = input[row_offset_in + i];
                if scale == 0.0 {
                    continue;
                } // Optimization for sparse inputs (ReLU)

                let w_row = &w_data[i * out_dim..(i + 1) * out_dim];
                let out_row = &mut out[row_offset_out..row_offset_out + out_dim];
                add_scaled_row(out_row, w_row, scale);
            }
        }
        out
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
        let dim = self.dim;
        let num_rows = input.len() / dim;
        let mut out = vec![0.0; input.len()];
        let w_data = self.weight.data.read().unwrap();

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
        out
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [..., Dim]
        let shape = &x.shape;
        let last_dim = shape[shape.len() - 1];
        assert_eq!(last_dim, self.dim, "RMSNorm dim mismatch");

        let x_data = x.data.read().unwrap();
        let w_data = self.weight.data.read().unwrap();

        let num_elements = x_data.len();
        let num_rows = num_elements / self.dim;

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
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; num_elements])),
            shape: shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let x_in = &parents[0];
                    let w_in = &parents[1];

                    let mut x_grad = x_in.grad.write().unwrap();
                    let mut w_grad = w_in.grad.write().unwrap();
                    let w_data = w_in.data.read().unwrap();

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
                    // Parallel accumulation for weight grad
                    // We can't write to w_grad in parallel directly without lock or reduction.
                    // Simple approach: calculate partial sums and reduce.
                    // Or serial sum for weight grad (it's small, size Dim).

                    // Serial accumulation for now (safer/easier)
                    // Optimization: transpose loop orders if dim is large?
                    // Here dim is small (e.g. 512), rows is large (Batch*Seq).
                    // We iterate cols then rows.

                    // Using a thread-local accumulator would be better.
                    // For now, let's keep it simple or use Rayon reduce.

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
