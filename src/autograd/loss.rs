#[cfg(cuda)]
use crate::autograd::cuda_grad_out_buffer;
use crate::autograd::{Context, Device, Tensor};
#[cfg(cuda)]
use crate::dtype::Dtype;
use crate::dtype::Storage;
use std::sync::Arc;

impl Tensor {
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        let diff = self - target;
        let sq = &diff * &diff;
        sq.mean()
    }

    /// Weighted MSE: sum(w_i * (pred_i - target_i)^2) / sum(w_i).
    /// `weights` shape must broadcast to self/target shape along the batch dimension.
    /// Typical usage: pred=[B,1], target=[B,1], weights=[B,1].
    pub fn weighted_mse_loss(&self, target: &Tensor, weights: &Tensor) -> Tensor {
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
