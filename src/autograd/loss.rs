#[cfg(cuda)]
use crate::autograd::cuda_grad_out_buffer;
use crate::autograd::{Context, Device, Tensor};
#[cfg(cuda)]
use crate::dtype::Dtype;
use crate::dtype::Storage;
use std::sync::Arc;

impl Tensor {
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        assert_eq!(self.numel(), target.numel(), "mse_loss size mismatch");
        assert!(self.numel() > 0, "mse_loss requires a non-empty tensor");
        let diff = self - target;
        let sq = &diff * &diff;
        sq.mean()
    }

    /// Weighted MSE: sum(w_i * (pred_i - target_i)^2) / sum(w_i).
    /// `weights` shape must broadcast to self/target shape along the batch dimension.
    /// Typical usage: pred=[B,1], target=[B,1], weights=[B,1].
    pub fn weighted_mse_loss(&self, target: &Tensor, weights: &Tensor) -> Tensor {
        assert!(
            self.numel() > 0,
            "weighted_mse_loss requires a non-empty tensor"
        );
        #[cfg(cuda)]
        if let Some(out) = self.weighted_mse_loss_cuda(target, weights) {
            return out;
        }
        let diff = self - target;
        let sq = &diff * &diff;
        let weighted = &sq * weights;
        let total = weighted.sum();
        let w_sum = weights.sum();

        let out_dtype = Tensor::binary_dtype(self.dtype, target.dtype);
        let out_dtype = Tensor::binary_dtype(out_dtype, weights.dtype);
        let total_val = total.data_as_f64_vec()[0];
        let w_sum_val = w_sum.data_as_f64_vec()[0];
        let denom = if w_sum_val.abs() < 1e-12 {
            1.0
        } else {
            w_sum_val
        };
        let result_val = total_val / denom;

        let parents = vec![total.clone(), w_sum.clone()];
        let denom_cap = denom;
        let numerator_cap = result_val;
        Tensor {
            data: Storage::from_f64_vec(vec![result_val], out_dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(out_dtype)),
            shape: vec![1],
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut total_grad = parents[0].grad_write_compat();
                    total_grad[0] += grad_out_f64[0] / denom_cap;
                    drop(total_grad);
                    let mut wsum_grad = parents[1].grad_write_compat();
                    wsum_grad[0] += grad_out_f64[0] * (-numerator_cap / denom_cap);
                }),
            })),
        }
    }

    /// L2 loss: sum(x^2) / 2.
    pub fn l2_loss(&self) -> Tensor {
        let input = self.data_as_f64_vec();
        let value = input.iter().map(|x| x * x).sum::<f64>() * 0.5;
        let len = input.len();
        let input_cache = Arc::new(input);
        let dtype = self.dtype;
        Tensor {
            data: Storage::from_f64_vec(vec![value], dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad = grad_out.to_f64_vec()[0];
                    let mut input_grad = parents[0].grad_write_compat();
                    for i in 0..len {
                        input_grad[i] += grad * input_cache[i];
                    }
                }),
            })),
        }
    }

    /// Smooth L1 loss with mean reduction.
    pub fn smooth_l1_loss(&self, target: &Tensor, beta: f64) -> Tensor {
        assert_eq!(self.numel(), target.numel(), "smooth_l1_loss size mismatch");
        assert!(beta > 0.0, "smooth_l1_loss beta must be positive");
        let input = self.data_as_f64_vec();
        let target_data = target.data_as_f64_vec();
        let len = input.len();
        assert!(len > 0, "smooth_l1_loss requires a non-empty tensor");
        let mut value = 0.0;
        for i in 0..len {
            let abs_diff = (input[i] - target_data[i]).abs();
            value += if abs_diff < beta {
                0.5 * abs_diff * abs_diff / beta
            } else {
                abs_diff - 0.5 * beta
            };
        }
        value /= len as f64;

        let input_cache = Arc::new(input);
        let target_cache = Arc::new(target_data);
        let dtype = Tensor::binary_dtype(self.dtype, target.dtype);
        Tensor {
            data: Storage::from_f64_vec(vec![value], dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), target.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let scale = grad_out.to_f64_vec()[0] / len as f64;
                    if !parents[0].grad.ptr_eq(&parents[1].grad) {
                        let mut input_grad = parents[0].grad_write_compat();
                        let mut target_grad = parents[1].grad_write_compat();
                        for i in 0..len {
                            let diff = input_cache[i] - target_cache[i];
                            let grad = if diff.abs() < beta {
                                diff / beta
                            } else {
                                diff.signum()
                            } * scale;
                            input_grad[i] += grad;
                            target_grad[i] -= grad;
                        }
                    }
                }),
            })),
        }
    }

    /// Numerically stable binary cross entropy with logits and mean reduction.
    pub fn sigmoid_cross_entropy_with_logits(&self, target: &Tensor) -> Tensor {
        assert_eq!(
            self.numel(),
            target.numel(),
            "sigmoid_cross_entropy_with_logits size mismatch"
        );
        let logits = self.data_as_f64_vec();
        let target_data = target.data_as_f64_vec();
        let len = logits.len();
        assert!(
            len > 0,
            "sigmoid_cross_entropy_with_logits requires a non-empty tensor"
        );
        let mut value = 0.0;
        for i in 0..len {
            let x = logits[i];
            value += x.max(0.0) - x * target_data[i] + (-x.abs()).exp().ln_1p();
        }
        value /= len as f64;

        let logits_cache = Arc::new(logits);
        let target_cache = Arc::new(target_data);
        let dtype = Tensor::binary_dtype(self.dtype, target.dtype);
        Tensor {
            data: Storage::from_f64_vec(vec![value], dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), target.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let scale = grad_out.to_f64_vec()[0] / len as f64;
                    if parents[0].grad.ptr_eq(&parents[1].grad) {
                        let mut grad_storage = parents[0].grad_write_compat();
                        for i in 0..len {
                            let x = logits_cache[i];
                            let probability = if x >= 0.0 {
                                1.0 / (1.0 + (-x).exp())
                            } else {
                                let exp_x = x.exp();
                                exp_x / (1.0 + exp_x)
                            };
                            grad_storage[i] += scale * (probability - target_cache[i] - x);
                        }
                    } else {
                        let mut logits_grad = parents[0].grad_write_compat();
                        let mut target_grad = parents[1].grad_write_compat();
                        for i in 0..len {
                            let x = logits_cache[i];
                            let probability = if x >= 0.0 {
                                1.0 / (1.0 + (-x).exp())
                            } else {
                                let exp_x = x.exp();
                                exp_x / (1.0 + exp_x)
                            };
                            logits_grad[i] += scale * (probability - target_cache[i]);
                            target_grad[i] -= scale * x;
                        }
                    }
                }),
            })),
        }
    }

    /// Softmax cross entropy for one-hot or probability targets on the last dimension.
    pub fn softmax_cross_entropy_with_logits(&self, target: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, target.shape,
            "softmax_cross_entropy_with_logits shape mismatch"
        );
        let cols = *self
            .shape
            .last()
            .expect("softmax_cross_entropy_with_logits requires rank >= 1");
        assert!(
            cols > 0,
            "softmax_cross_entropy_with_logits empty class dim"
        );
        let logits = self.data_as_f64_vec();
        let targets = target.data_as_f64_vec();
        let rows = logits.len() / cols;
        assert!(rows > 0, "softmax_cross_entropy_with_logits empty batch");
        let mut probabilities = vec![0.0; logits.len()];
        let mut log_probabilities = vec![0.0; logits.len()];
        let mut value = 0.0;

        for row in 0..rows {
            let base = row * cols;
            let max = logits[base..base + cols]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp = logits[base..base + cols]
                .iter()
                .map(|x| (x - max).exp())
                .sum::<f64>();
            let log_sum_exp = sum_exp.ln() + max;
            for col in 0..cols {
                let idx = base + col;
                log_probabilities[idx] = logits[idx] - log_sum_exp;
                probabilities[idx] = log_probabilities[idx].exp();
                value -= targets[idx] * log_probabilities[idx];
            }
        }
        value /= rows as f64;

        let probabilities = Arc::new(probabilities);
        let log_probabilities = Arc::new(log_probabilities);
        let targets = Arc::new(targets);
        let dtype = Tensor::binary_dtype(self.dtype, target.dtype);
        Tensor {
            data: Storage::from_f64_vec(vec![value], dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), target.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let scale = grad_out.to_f64_vec()[0] / rows as f64;
                    if parents[0].grad.ptr_eq(&parents[1].grad) {
                        let mut grad_storage = parents[0].grad_write_compat();
                        for i in 0..probabilities.len() {
                            grad_storage[i] +=
                                scale * (probabilities[i] - targets[i] - log_probabilities[i]);
                        }
                    } else {
                        let mut logits_grad = parents[0].grad_write_compat();
                        let mut target_grad = parents[1].grad_write_compat();
                        for i in 0..probabilities.len() {
                            logits_grad[i] += scale * (probabilities[i] - targets[i]);
                            target_grad[i] -= scale * log_probabilities[i];
                        }
                    }
                }),
            })),
        }
    }

    /// Cosine embedding loss over the last dimension with mean reduction.
    pub fn cosine_embedding_loss(&self, other: &Tensor, target: &Tensor, margin: f64) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "cosine_embedding_loss shape mismatch"
        );
        let cols = *self
            .shape
            .last()
            .expect("cosine_embedding_loss requires rank >= 1");
        assert!(cols > 0, "cosine_embedding_loss empty embedding dim");
        let lhs = self.data_as_f64_vec();
        let rhs = other.data_as_f64_vec();
        let rows = lhs.len() / cols;
        assert!(rows > 0, "cosine_embedding_loss empty batch");
        assert_eq!(
            target.numel(),
            rows,
            "cosine_embedding_loss target must contain one label per row"
        );
        let labels = target.data_as_f64_vec();
        let mut cosine = vec![0.0; rows];
        let mut lhs_norms = vec![0.0; rows];
        let mut rhs_norms = vec![0.0; rows];
        let mut value = 0.0;
        for row in 0..rows {
            let base = row * cols;
            let mut dot = 0.0;
            let mut lhs_sq = 0.0;
            let mut rhs_sq = 0.0;
            for col in 0..cols {
                let idx = base + col;
                dot += lhs[idx] * rhs[idx];
                lhs_sq += lhs[idx] * lhs[idx];
                rhs_sq += rhs[idx] * rhs[idx];
            }
            lhs_norms[row] = lhs_sq.max(1e-12).sqrt();
            rhs_norms[row] = rhs_sq.max(1e-12).sqrt();
            cosine[row] = dot / (lhs_norms[row] * rhs_norms[row]);
            value += if labels[row] > 0.0 {
                1.0 - cosine[row]
            } else {
                (cosine[row] - margin).max(0.0)
            };
        }
        value /= rows as f64;

        let lhs = Arc::new(lhs);
        let rhs = Arc::new(rhs);
        let labels = Arc::new(labels);
        let cosine = Arc::new(cosine);
        let lhs_norms = Arc::new(lhs_norms);
        let rhs_norms = Arc::new(rhs_norms);
        let dtype = Tensor::binary_dtype(self.dtype, other.dtype);
        Tensor {
            data: Storage::from_f64_vec(vec![value], dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), other.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let scale = grad_out.to_f64_vec()[0] / rows as f64;
                    if parents[0].grad.ptr_eq(&parents[1].grad) {
                        let mut grad_storage = parents[0].grad_write_compat();
                        for row in 0..rows {
                            let sign = if labels[row] > 0.0 {
                                -1.0
                            } else if cosine[row] > margin {
                                1.0
                            } else {
                                0.0
                            } * scale;
                            let base = row * cols;
                            for col in 0..cols {
                                let idx = base + col;
                                let lhs_cos_grad = rhs[idx] / (lhs_norms[row] * rhs_norms[row])
                                    - cosine[row] * lhs[idx] / (lhs_norms[row] * lhs_norms[row]);
                                let rhs_cos_grad = lhs[idx] / (lhs_norms[row] * rhs_norms[row])
                                    - cosine[row] * rhs[idx] / (rhs_norms[row] * rhs_norms[row]);
                                grad_storage[idx] += sign * (lhs_cos_grad + rhs_cos_grad);
                            }
                        }
                    } else {
                        let mut lhs_grad = parents[0].grad_write_compat();
                        let mut rhs_grad = parents[1].grad_write_compat();
                        for row in 0..rows {
                            let sign = if labels[row] > 0.0 {
                                -1.0
                            } else if cosine[row] > margin {
                                1.0
                            } else {
                                0.0
                            } * scale;
                            let base = row * cols;
                            for col in 0..cols {
                                let idx = base + col;
                                let lhs_cos_grad = rhs[idx] / (lhs_norms[row] * rhs_norms[row])
                                    - cosine[row] * lhs[idx] / (lhs_norms[row] * lhs_norms[row]);
                                let rhs_cos_grad = lhs[idx] / (lhs_norms[row] * rhs_norms[row])
                                    - cosine[row] * rhs[idx] / (rhs_norms[row] * rhs_norms[row]);
                                lhs_grad[idx] += sign * lhs_cos_grad;
                                rhs_grad[idx] += sign * rhs_cos_grad;
                            }
                        }
                    }
                }),
            })),
        }
    }

    #[cfg(cuda)]
    fn weighted_mse_loss_cuda(&self, target: &Tensor, weights: &Tensor) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda
            || target.device != Device::Cuda
            || weights.device != Device::Cuda
            || self.numel() != target.numel()
            || self.numel() != weights.numel()
            || self.dtype != target.dtype
            || self.dtype != weights.dtype
            || !matches!(self.dtype, Dtype::F32 | Dtype::F64)
        {
            return None;
        }

        let size = self.numel();
        let d_pred = self.cuda_get_or_upload_buffer().ok()?;
        let d_target = target.cuda_get_or_upload_buffer().ok()?;
        let d_weights = weights.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(1).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(1).ok()?),
            _ => return None,
        };
        let d_weight_sum = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(1).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(1).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let d_weight_sum = Arc::new(d_weight_sum);

        let ok = match (
            &*d_pred,
            &*d_target,
            &*d_weights,
            &*d_out,
            &*d_weight_sum,
            self.dtype,
        ) {
            (
                CudaBuffer::F32(pred),
                CudaBuffer::F32(target),
                CudaBuffer::F32(weights),
                CudaBuffer::F32(out),
                CudaBuffer::F32(weight_sum),
                Dtype::F32,
            ) => crate::cuda::kernels::weighted_mse_loss_f32(
                pred, target, weights, out, weight_sum, size,
            )
            .is_ok(),
            (
                CudaBuffer::F64(pred),
                CudaBuffer::F64(target),
                CudaBuffer::F64(weights),
                CudaBuffer::F64(out),
                CudaBuffer::F64(weight_sum),
                Dtype::F64,
            ) => crate::cuda::kernels::weighted_mse_loss(
                pred, target, weights, out, weight_sum, size,
            )
            .is_ok(),
            _ => false,
        };
        if !ok {
            return None;
        }

        let dtype = self.dtype;
        let weight_sum_for_backward = d_weight_sum.clone();
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), target.clone(), weights.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let pred = &parents[0];
                    let target = &parents[1];
                    let weights = &parents[2];
                    if pred.device == Device::Cuda
                        && target.device == Device::Cuda
                        && weights.device == Device::Cuda
                    {
                        if let (Some(d_grad_tmp), Ok(d_pred), Ok(d_target), Ok(d_weights)) = (
                            cuda_grad_out_buffer(grad_out),
                            pred.cuda_get_or_upload_buffer(),
                            target.cuda_get_or_upload_buffer(),
                            weights.cuda_get_or_upload_buffer(),
                        ) {
                            if let Some(d_pred_grad) = pred.cuda_grad_ensure_buffer() {
                                let ok = match (
                                    &*d_pred,
                                    &*d_target,
                                    &*d_weights,
                                    &*weight_sum_for_backward,
                                    &*d_grad_tmp,
                                    &*d_pred_grad,
                                    dtype,
                                ) {
                                    (
                                        CudaBuffer::F32(pred),
                                        CudaBuffer::F32(target),
                                        CudaBuffer::F32(weights),
                                        CudaBuffer::F32(weight_sum),
                                        CudaBuffer::F32(grad_out),
                                        CudaBuffer::F32(pred_grad),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::weighted_mse_backward_f32(
                                        pred, target, weights, weight_sum, grad_out, pred_grad,
                                        size,
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(pred),
                                        CudaBuffer::F64(target),
                                        CudaBuffer::F64(weights),
                                        CudaBuffer::F64(weight_sum),
                                        CudaBuffer::F64(grad_out),
                                        CudaBuffer::F64(pred_grad),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::weighted_mse_backward(
                                        pred, target, weights, weight_sum, grad_out, pred_grad,
                                        size,
                                    )
                                    .is_ok(),
                                    _ => false,
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }

                    let pred_data = pred.data_as_f64_vec();
                    let target_data = target.data_as_f64_vec();
                    let weights_data = weights.data_as_f64_vec();
                    let denom_raw = weights_data.iter().sum::<f64>();
                    let denom = if denom_raw.abs() < 1e-12 {
                        1.0
                    } else {
                        denom_raw
                    };
                    let grad_scalar = grad_out.to_f64_vec()[0];
                    let mut pred_grad = pred.grad_write_compat();
                    for i in 0..size {
                        pred_grad[i] +=
                            grad_scalar * 2.0 * weights_data[i] * (pred_data[i] - target_data[i])
                                / denom;
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }
}
