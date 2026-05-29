#[cfg(any(cuda, test))]
use crate::dtype::Dtype;
#[cfg(cuda)]
use crate::dtype::Storage;
#[cfg(cuda)]
use rayon::prelude::*;
#[cfg(cuda)]
use std::sync::Arc;

// --- Autograd Engine ---

mod activation;
mod binary_ops;
mod conv_pooling;
mod core;
#[cfg(cuda)]
mod cuda_bridge;
mod guards;
mod lifecycle;
mod loss;
mod matmul;
mod operators;
mod reductions;
mod serde_impl;
mod shape_ops;
mod softmax;
mod storage;
mod unary_ops;

#[cfg(cuda)]
pub(crate) use core::BackwardOp;
pub use core::{Context, Device, GradWriteCompat, Tensor};
#[cfg(cuda)]
pub(crate) use cuda_bridge::cuda_grad_out_buffer;
#[cfg(cuda)]
use cuda_bridge::cuda_sync_grad_to_host;
pub use guards::TensorReadGuard;

// Minimum element count to justify Rayon parallel dispatch.
// Below this, serial iteration is faster due to scheduling overhead.
pub(crate) const PAR_THRESHOLD: usize = 4096;

#[cfg(cuda)]
#[derive(Clone, Copy)]
enum CudaBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[cfg(cuda)]
pub(crate) fn cuda_clip_gradients_in_place(params: &[Tensor], max_norm: f64, eps: f64) -> bool {
    use crate::cuda::memory::{alloc, CudaBuffer};

    let dtype = params
        .iter()
        .find(|p| p.cuda_storage_len() > 0)
        .map(|p| p.grad.dtype());
    let Some(dtype) = dtype else {
        return true;
    };
    if !matches!(dtype, Dtype::F32 | Dtype::F64) {
        return false;
    }
    if params
        .iter()
        .any(|p| p.cuda_storage_len() > 0 && p.grad.dtype() != dtype)
    {
        return false;
    }

    match dtype {
        Dtype::F32 => {
            let sumsq = match alloc::<f32>(1) {
                Ok(buf) => buf,
                Err(_) => return false,
            };
            let coef = match alloc::<f32>(1) {
                Ok(buf) => buf,
                Err(_) => return false,
            };
            if crate::cuda::kernels::fill_f32(&sumsq, 0.0).is_err() {
                return false;
            }
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = match param.cuda_grad_get_or_upload_buffer() {
                    Ok(buf) => buf,
                    Err(_) => return false,
                };
                let CudaBuffer::F32(grad) = &*grad else {
                    return false;
                };
                if crate::cuda::kernels::sumsq_accum_f32(grad, &sumsq, len).is_err() {
                    return false;
                }
            }
            if crate::cuda::kernels::clip_coef_from_sumsq_f32(
                &sumsq,
                &coef,
                max_norm as f32,
                eps as f32,
            )
            .is_err()
            {
                return false;
            }
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = match param.cuda_grad_get_or_upload_buffer() {
                    Ok(buf) => buf,
                    Err(_) => return false,
                };
                let CudaBuffer::F32(grad) = &*grad else {
                    return false;
                };
                if crate::cuda::kernels::scale_inplace_by_scalar_f32(grad, &coef, len).is_err() {
                    return false;
                }
            }
            true
        }
        Dtype::F64 => {
            let sumsq = match alloc::<f64>(1) {
                Ok(buf) => buf,
                Err(_) => return false,
            };
            let coef = match alloc::<f64>(1) {
                Ok(buf) => buf,
                Err(_) => return false,
            };
            if crate::cuda::kernels::fill(&sumsq, 0.0).is_err() {
                return false;
            }
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = match param.cuda_grad_get_or_upload_buffer() {
                    Ok(buf) => buf,
                    Err(_) => return false,
                };
                let CudaBuffer::F64(grad) = &*grad else {
                    return false;
                };
                if crate::cuda::kernels::sumsq_accum(grad, &sumsq, len).is_err() {
                    return false;
                }
            }
            if crate::cuda::kernels::clip_coef_from_sumsq(&sumsq, &coef, max_norm, eps).is_err() {
                return false;
            }
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = match param.cuda_grad_get_or_upload_buffer() {
                    Ok(buf) => buf,
                    Err(_) => return false,
                };
                let CudaBuffer::F64(grad) = &*grad else {
                    return false;
                };
                if crate::cuda::kernels::scale_inplace_by_scalar(grad, &coef, len).is_err() {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

impl Tensor {
    #[cfg(cuda)]
    pub(crate) fn index_select_cuda(&self, idx: usize) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || idx >= self.numel() {
            return None;
        }
        if !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(1).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(1).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(out), Dtype::F32) => {
                crate::cuda::kernels::index_select_f32(input, out, idx).is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(out), Dtype::F64) => {
                crate::cuda::kernels::index_select(input, out, idx).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let dtype = self.dtype;
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (&*d_grad_tmp, &*d_input_grad, dtype) {
                                    (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                        crate::cuda::kernels::index_select_backward_f32(gt, ig, idx)
                                            .is_ok()
                                    }
                                    (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                        crate::cuda::kernels::index_select_backward(gt, ig, idx)
                                            .is_ok()
                                    }
                                    _ => false,
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = input.grad_write_compat();
                    inp_grad[idx] += grad_out_f64[0];
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn exp_cuda(&self) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(out), Dtype::F32) => {
                crate::cuda::kernels::exp_f32(input, out, len).is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(out), Dtype::F64) => {
                crate::cuda::kernels::exp(input, out, len).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let dtype = self.dtype;
        let d_out_for_backward = d_out.clone();
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (
                                    &*d_out_for_backward,
                                    &*d_grad_tmp,
                                    &*d_input_grad,
                                    dtype,
                                ) {
                                    (
                                        CudaBuffer::F32(exp_out),
                                        CudaBuffer::F32(gt),
                                        CudaBuffer::F32(ig),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::exp_backward_f32(
                                        exp_out,
                                        gt,
                                        ig,
                                        exp_out.len(),
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(exp_out),
                                        CudaBuffer::F64(gt),
                                        CudaBuffer::F64(ig),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::exp_backward(
                                        exp_out,
                                        gt,
                                        ig,
                                        exp_out.len(),
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

                    let grad_out_f64 = grad_out.to_f64_vec();
                    let exp_cache: Vec<f64> = match &*d_out_for_backward {
                        CudaBuffer::F32(buf) => {
                            let mut host = vec![0.0f32; buf.len()];
                            if crate::cuda::memory::copy_d2h(&mut host, buf).is_err() {
                                return;
                            }
                            host.into_iter().map(|v| v as f64).collect()
                        }
                        CudaBuffer::F64(buf) => {
                            let mut host = vec![0.0f64; buf.len()];
                            if crate::cuda::memory::copy_d2h(&mut host, buf).is_err() {
                                return;
                            }
                            host
                        }
                        CudaBuffer::BF16(_) | CudaBuffer::I8(_) => return,
                    };
                    let mut inp_grad = input.grad_write_compat();
                    for i in 0..inp_grad.len() {
                        inp_grad[i] += grad_out_f64[i] * exp_cache[i];
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn sum_cuda(&self, scale: f64) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(1).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(1).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(out), Dtype::F32) => {
                crate::cuda::kernels::fill_f32(out, 0.0).is_ok()
                    && crate::cuda::kernels::sum_accum_f32(input, out, len, scale as f32).is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(out), Dtype::F64) => {
                crate::cuda::kernels::fill(out, 0.0).is_ok()
                    && crate::cuda::kernels::sum_accum(input, out, len, scale).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let dtype = self.dtype;
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(dtype)),
            shape: vec![1],
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (&*d_grad_tmp, &*d_input_grad, dtype) {
                                    (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                        crate::cuda::kernels::add_scalar_f32(
                                            ig,
                                            gt,
                                            scale as f32,
                                            input.numel(),
                                        )
                                        .is_ok()
                                    }
                                    (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                        crate::cuda::kernels::add_scalar(
                                            ig,
                                            gt,
                                            scale,
                                            input.numel(),
                                        )
                                        .is_ok()
                                    }
                                    _ => false,
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }

                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = input.grad_write_compat();
                    let g = grad_out_f64[0] * scale;
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad.par_iter_mut().for_each(|v| *v += g);
                    } else {
                        for v in inp_grad.iter_mut() {
                            *v += g;
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn select_last_token_cuda(&self) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || self.shape.len() != 3 {
            return None;
        }
        if !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let batch = self.shape[0];
        let seq = self.shape[1];
        let dim = self.shape[2];
        if seq == 0 || dim == 0 {
            return None;
        }
        let out_len = batch.checked_mul(dim)?;
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(out_len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(out_len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(out), Dtype::F32) => {
                crate::cuda::kernels::select_last_token_f32(input, out, batch, seq, dim).is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(out), Dtype::F64) => {
                crate::cuda::kernels::select_last_token(input, out, batch, seq, dim).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let dtype = self.dtype;
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape: vec![batch, dim],
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (&*d_grad_tmp, &*d_input_grad, dtype) {
                                    (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                        crate::cuda::kernels::select_last_token_backward_f32(
                                            gt, ig, batch, seq, dim,
                                        )
                                        .is_ok()
                                    }
                                    (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                        crate::cuda::kernels::select_last_token_backward(
                                            gt, ig, batch, seq, dim,
                                        )
                                        .is_ok()
                                    }
                                    _ => false,
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = input.grad_write_compat();
                    for b in 0..batch {
                        let dst = (b * seq + (seq - 1)) * dim;
                        let src = b * dim;
                        for d in 0..dim {
                            inp_grad[dst + d] += grad_out_f64[src + d];
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }
}

#[cfg(cuda)]
#[allow(dead_code)]
impl Tensor {
    #[cfg(cuda)]
    pub(crate) fn scale_cuda(&self, scale: f64) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || (self.dtype != Dtype::F32 && self.dtype != Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(output), Dtype::F32) => {
                crate::cuda::kernels::scale_f32(input, output, scale as f32, len).is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(output), Dtype::F64) => {
                crate::cuda::kernels::scale(input, output, scale, len).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let parents = vec![self.clone()];
        let out = Tensor {
            data: Tensor::empty_storage(self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (&*d_grad_tmp, &*d_input_grad, input.dtype) {
                                    (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                        crate::cuda::kernels::scale_backward_f32(
                                            gt,
                                            ig,
                                            scale as f32,
                                            gt.len(),
                                        )
                                        .is_ok()
                                    }
                                    (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                        crate::cuda::kernels::scale_backward(
                                            gt,
                                            ig,
                                            scale,
                                            gt.len(),
                                        )
                                        .is_ok()
                                    }
                                    _ => false,
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = input.grad_write_compat();
                    inp_grad
                        .par_iter_mut()
                        .zip(grad_out_f64.par_iter())
                        .for_each(|(ig, &g)| *ig += g * scale);
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn causal_mask_cuda(&self, seq_len: usize) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || (self.dtype != Dtype::F32 && self.dtype != Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let seq_area = seq_len.checked_mul(seq_len)?;
        if seq_len == 0 || !len.is_multiple_of(seq_area) {
            return None;
        }
        let batches = len / seq_area;
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(output), Dtype::F32) => {
                crate::cuda::kernels::causal_mask_f32(input, output, batches, seq_len).is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(output), Dtype::F64) => {
                crate::cuda::kernels::causal_mask(input, output, batches, seq_len).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let parents = vec![self.clone()];
        let out = Tensor {
            data: Tensor::empty_storage(self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (&*d_grad_tmp, &*d_input_grad, input.dtype) {
                                    (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                        crate::cuda::kernels::causal_mask_backward_f32(
                                            gt, ig, batches, seq_len,
                                        )
                                        .is_ok()
                                    }
                                    (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                        crate::cuda::kernels::causal_mask_backward(
                                            gt, ig, batches, seq_len,
                                        )
                                        .is_ok()
                                    }
                                    _ => false,
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = input.grad_write_compat();
                    for (idx, (&g, ig)) in grad_out_f64.iter().zip(inp_grad.iter_mut()).enumerate()
                    {
                        let local = idx % (seq_len * seq_len);
                        let i = local / seq_len;
                        let j = local % seq_len;
                        if j <= i {
                            *ig += g;
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn concat_last_dim_cuda(a: &Tensor, b: &Tensor) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if a.device != Device::Cuda || b.device != Device::Cuda {
            return None;
        }
        if a.shape.len() != b.shape.len() || a.shape.is_empty() {
            return None;
        }
        if a.dtype != b.dtype || (a.dtype != Dtype::F32 && a.dtype != Dtype::F64) {
            return None;
        }
        if a.shape[..a.shape.len() - 1] != b.shape[..b.shape.len() - 1] {
            return None;
        }
        let a_dim = *a.shape.last()?;
        let b_dim = *b.shape.last()?;
        let rows = a.numel() / a_dim;
        let out_len = rows * (a_dim + b_dim);
        let d_a = a.cuda_get_or_upload_buffer().ok()?;
        let d_b = b.cuda_get_or_upload_buffer().ok()?;
        let d_out = match a.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(out_len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(out_len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_a, &*d_b, &*d_out, a.dtype) {
            (CudaBuffer::F32(da), CudaBuffer::F32(db), CudaBuffer::F32(out), Dtype::F32) => {
                crate::cuda::kernels::concat_last_dim_f32(da, db, out, rows, a_dim, b_dim).is_ok()
            }
            (CudaBuffer::F64(da), CudaBuffer::F64(db), CudaBuffer::F64(out), Dtype::F64) => {
                crate::cuda::kernels::concat_last_dim(da, db, out, rows, a_dim, b_dim).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let mut shape = a.shape.clone();
        *shape.last_mut()? = a_dim + b_dim;
        let dtype = a.dtype;
        let parents = vec![a.clone(), b.clone()];
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape,
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let a_in = &parents[0];
                    let b_in = &parents[1];
                    if a_in.device == Device::Cuda && b_in.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let (Some(d_a_grad), Some(d_b_grad)) = (
                                a_in.cuda_grad_ensure_buffer(),
                                b_in.cuda_grad_ensure_buffer(),
                            ) {
                                let ok = match (&*d_grad_tmp, &*d_a_grad, &*d_b_grad, dtype) {
                                    (
                                        CudaBuffer::F32(gt),
                                        CudaBuffer::F32(ag),
                                        CudaBuffer::F32(bg),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::concat_last_dim_backward_f32(
                                        gt, ag, bg, rows, a_dim, b_dim,
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(gt),
                                        CudaBuffer::F64(ag),
                                        CudaBuffer::F64(bg),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::concat_last_dim_backward(
                                        gt, ag, bg, rows, a_dim, b_dim,
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
                    let mut a_grad = a_in.grad_write_compat();
                    let mut b_grad = b_in.grad_write_compat();
                    let stride = a_dim + b_dim;
                    a_grad
                        .par_chunks_mut(a_dim)
                        .zip(b_grad.par_chunks_mut(b_dim))
                        .zip(grad_out_f64.par_chunks(stride))
                        .for_each(|((ag_row, bg_row), g_row)| {
                            for k in 0..a_dim {
                                ag_row[k] += g_row[k];
                            }
                            for k in 0..b_dim {
                                bg_row[k] += g_row[a_dim + k];
                            }
                        });
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn split_last_dim_cuda(&self, parts: usize) -> Option<Vec<Tensor>> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda
            || (self.dtype != Dtype::F32 && self.dtype != Dtype::F64)
            || self.shape.is_empty()
            || parts == 0
        {
            return None;
        }
        let input_dim = *self.shape.last()?;
        if !input_dim.is_multiple_of(parts) {
            return None;
        }
        let part_dim = input_dim / parts;
        let rows = self.numel() / input_dim;
        let d_input = self.cuda_get_or_upload_buffer().ok()?;
        let mut out = Vec::with_capacity(parts);
        for part_idx in 0..parts {
            let d_part = match self.dtype {
                Dtype::F32 => CudaBuffer::F32(alloc::<f32>(rows * part_dim).ok()?),
                Dtype::F64 => CudaBuffer::F64(alloc::<f64>(rows * part_dim).ok()?),
                _ => return None,
            };
            let d_part = Arc::new(d_part);
            let ok = match (&*d_input, &*d_part, self.dtype) {
                (CudaBuffer::F32(input), CudaBuffer::F32(part), Dtype::F32) => {
                    crate::cuda::kernels::split_last_dim_f32(
                        input, part, rows, input_dim, part_dim, part_idx,
                    )
                    .is_ok()
                }
                (CudaBuffer::F64(input), CudaBuffer::F64(part), Dtype::F64) => {
                    crate::cuda::kernels::split_last_dim(
                        input, part, rows, input_dim, part_dim, part_idx,
                    )
                    .is_ok()
                }
                _ => false,
            };
            if !ok {
                return None;
            }
            let mut shape = self.shape.clone();
            *shape.last_mut()? = part_dim;
            let input = self.clone();
            let dtype = self.dtype;
            let tensor = Tensor {
                data: Tensor::empty_storage(dtype),
                grad: Storage::zeros(rows * part_dim, Tensor::grad_dtype_for(dtype)),
                shape,
                device: Device::Cuda,
                dtype,
                _ctx: Some(Arc::new(Context {
                    parents: vec![input],
                    backward_op: Box::new(move |grad_out, parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let input = &parents[0];
                        if input.device == Device::Cuda {
                            if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                                if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                    let ok = match (&*d_grad_tmp, &*d_input_grad, dtype) {
                                        (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                            crate::cuda::kernels::split_last_dim_backward_f32(
                                                gt, ig, rows, input_dim, part_dim, part_idx,
                                            )
                                            .is_ok()
                                        }
                                        (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                            crate::cuda::kernels::split_last_dim_backward(
                                                gt, ig, rows, input_dim, part_dim, part_idx,
                                            )
                                            .is_ok()
                                        }
                                        _ => false,
                                    };
                                    if ok {
                                        return;
                                    }
                                }
                            }
                        }
                        let mut input_grad = input.grad_write_compat();
                        for r in 0..rows {
                            let src = r * part_dim;
                            let dst = r * input_dim + part_idx * part_dim;
                            for c in 0..part_dim {
                                input_grad[dst + c] += grad_out_f64[src + c];
                            }
                        }
                    }),
                })),
            };
            tensor.cuda_set_cached_buffer(d_part);
            out.push(tensor);
        }
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn broadcast_to_batch_cuda(&self, batch_size: usize) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || (self.dtype != Dtype::F32 && self.dtype != Dtype::F64) {
            return None;
        }
        let inner_len = self.numel();
        let out_len = inner_len.checked_mul(batch_size)?;
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(out_len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(out_len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(output), Dtype::F32) => {
                crate::cuda::kernels::broadcast_batch_f32(input, output, batch_size, inner_len)
                    .is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(output), Dtype::F64) => {
                crate::cuda::kernels::broadcast_batch(input, output, batch_size, inner_len).is_ok()
            }
            _ => false,
        };
        if !ok {
            return None;
        }

        let mut shape = vec![batch_size];
        shape.extend_from_slice(&self.shape);
        let dtype = self.dtype;
        let parents = vec![self.clone()];
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape,
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (&*d_grad_tmp, &*d_input_grad, dtype) {
                                    (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                        crate::cuda::kernels::broadcast_batch_backward_f32(
                                            gt, ig, batch_size, inner_len,
                                        )
                                        .is_ok()
                                    }
                                    (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                        crate::cuda::kernels::broadcast_batch_backward(
                                            gt, ig, batch_size, inner_len,
                                        )
                                        .is_ok()
                                    }
                                    _ => false,
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = input.grad_write_compat();
                    for chunk in grad_out_f64.chunks(inner_len) {
                        for (i, &g) in chunk.iter().enumerate() {
                            inp_grad[i] += g;
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn transpose_cuda(&self, dim0: usize, dim1: usize) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || (self.dtype != Dtype::F32 && self.dtype != Dtype::F64) {
            return None;
        }
        let rank = self.shape.len();
        if dim0 >= rank || dim1 >= rank || dim0 == dim1 {
            return None;
        }
        if rank != 2 && rank != 4 {
            return None;
        }
        let len = self.numel();
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc::<f64>(len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = if rank == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            match (&*d_in, &*d_out, self.dtype) {
                (CudaBuffer::F32(input), CudaBuffer::F32(output), Dtype::F32) => {
                    crate::cuda::kernels::transpose_last_two_f32(input, output, 1, rows, cols)
                        .is_ok()
                }
                (CudaBuffer::F64(input), CudaBuffer::F64(output), Dtype::F64) => {
                    crate::cuda::kernels::transpose_last_two(input, output, 1, rows, cols).is_ok()
                }
                _ => false,
            }
        } else {
            let shape = [self.shape[0], self.shape[1], self.shape[2], self.shape[3]];
            match (&*d_in, &*d_out, self.dtype) {
                (CudaBuffer::F32(input), CudaBuffer::F32(output), Dtype::F32) => {
                    crate::cuda::kernels::transpose_4d_f32(input, output, shape, dim0, dim1).is_ok()
                }
                (CudaBuffer::F64(input), CudaBuffer::F64(output), Dtype::F64) => {
                    crate::cuda::kernels::transpose_4d(input, output, shape, dim0, dim1).is_ok()
                }
                _ => false,
            }
        };
        if !ok {
            return None;
        }

        let dtype = self.dtype;
        let parents = vec![self.clone()];
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
            shape: new_shape,
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = if rank == 2 {
                                    let rows = input.shape[1];
                                    let cols = input.shape[0];
                                    match (&*d_grad_tmp, &*d_input_grad, dtype) {
                                        (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                            crate::cuda::kernels::transpose_last_two_f32(
                                                gt, ig, 1, rows, cols,
                                            )
                                            .is_ok()
                                        }
                                        (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                            crate::cuda::kernels::transpose_last_two(
                                                gt, ig, 1, rows, cols,
                                            )
                                            .is_ok()
                                        }
                                        _ => false,
                                    }
                                } else {
                                    let mut shape = [
                                        input.shape[0],
                                        input.shape[1],
                                        input.shape[2],
                                        input.shape[3],
                                    ];
                                    shape.swap(dim0, dim1);
                                    match (&*d_grad_tmp, &*d_input_grad, dtype) {
                                        (CudaBuffer::F32(gt), CudaBuffer::F32(ig), Dtype::F32) => {
                                            crate::cuda::kernels::transpose_4d_f32(
                                                gt, ig, shape, dim0, dim1,
                                            )
                                            .is_ok()
                                        }
                                        (CudaBuffer::F64(gt), CudaBuffer::F64(ig), Dtype::F64) => {
                                            crate::cuda::kernels::transpose_4d(
                                                gt, ig, shape, dim0, dim1,
                                            )
                                            .is_ok()
                                        }
                                        _ => false,
                                    }
                                };
                                if ok {
                                    return;
                                }
                            }
                        }
                    }

                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = input.grad_write_compat();
                    if rank == 2 {
                        let rows = input.shape[0];
                        let cols = input.shape[1];
                        for r in 0..rows {
                            for c in 0..cols {
                                inp_grad[r * cols + c] += grad_out_f64[c * rows + r];
                            }
                        }
                    } else {
                        let shape = &input.shape;
                        for (idx, &g) in grad_out_f64.iter().enumerate() {
                            let mut coords = [0usize; 4];
                            let mut tmp = idx;
                            let out_shape = {
                                let mut s = shape.clone();
                                s.swap(dim0, dim1);
                                s
                            };
                            coords[0] = tmp / (out_shape[1] * out_shape[2] * out_shape[3]);
                            tmp %= out_shape[1] * out_shape[2] * out_shape[3];
                            coords[1] = tmp / (out_shape[2] * out_shape[3]);
                            tmp %= out_shape[2] * out_shape[3];
                            coords[2] = tmp / out_shape[3];
                            coords[3] = tmp % out_shape[3];
                            coords.swap(dim0, dim1);
                            let old_idx = ((coords[0] * shape[1] + coords[1]) * shape[2]
                                + coords[2])
                                * shape[3]
                                + coords[3];
                            inp_grad[old_idx] += g;
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    /// GPU-accelerated Softmax activation using CUDA kernel
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn softmax_cuda(&self) -> Tensor {
        use crate::cuda::kernels::{softmax_inplace_auto, softmax_inplace_auto_f32};
        use crate::cuda::memory::{alloc, copy_d2d, CudaBuffer};

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.softmax_cpu_fallback(),
        };
        if len == 0 {
            return self.softmax_cpu_fallback();
        }
        let cols = self.shape.last().copied().unwrap_or(len.max(1));
        if cols == 0 {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA Softmax invalid last dimension (cols=0), using CPU");
            return self.softmax_cpu_fallback();
        }
        let rows = len / cols;
        if rows.checked_mul(cols) != Some(len) {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!(
                "[Autograd] CUDA Softmax shape mismatch (len={}, rows={}, cols={}), using CPU",
                len,
                rows,
                cols
            );
            return self.softmax_cpu_fallback();
        }
        crate::cuda::record_activation_attempt();

        let d_src = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_activation_fallback(stage);
                log::warn!(
                    "[Autograd] CUDA prepare Softmax input failed ({}), using CPU",
                    err
                );
                return self.softmax_cpu_fallback();
            }
        };
        let d_data = match self.dtype {
            Dtype::F32 => match alloc::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc Softmax buffer failed ({}), using CPU",
                        err
                    );
                    return self.softmax_cpu_fallback();
                }
            },
            Dtype::F64 => match alloc::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc Softmax buffer failed ({}), using CPU",
                        err
                    );
                    return self.softmax_cpu_fallback();
                }
            },
            _ => return self.softmax_cpu_fallback(),
        };
        let copy_ok = match (self.dtype, &d_data, &*d_src) {
            (Dtype::F32, CudaBuffer::F32(dst), CudaBuffer::F32(src)) => copy_d2d(dst, src).is_ok(),
            (Dtype::F64, CudaBuffer::F64(dst), CudaBuffer::F64(src)) => copy_d2d(dst, src).is_ok(),
            _ => false,
        };
        if !copy_ok {
            crate::cuda::record_activation_fallback("copy");
            log::warn!("[Autograd] CUDA D2D Softmax input copy failed, using CPU");
            return self.softmax_cpu_fallback();
        }

        let kernel_ok = match (self.dtype, &d_data) {
            (Dtype::F32, CudaBuffer::F32(b)) => softmax_inplace_auto_f32(b, rows, cols).is_ok(),
            (Dtype::F64, CudaBuffer::F64(b)) => softmax_inplace_auto(b, rows, cols).is_ok(),
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA Softmax kernel failed, using CPU");
            return self.softmax_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let rows_cap = rows;
        let cols_cap = cols;
        let d_data_for_backward = d_data.clone();
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
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (
                                    &*d_data_for_backward,
                                    &*d_grad_tmp,
                                    &*d_input_grad,
                                    input.dtype,
                                ) {
                                    (
                                        CudaBuffer::F32(out),
                                        CudaBuffer::F32(gt),
                                        CudaBuffer::F32(ig),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::softmax_backward_f32(
                                        out, gt, ig, rows_cap, cols_cap,
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(out),
                                        CudaBuffer::F64(gt),
                                        CudaBuffer::F64(ig),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::softmax_backward(
                                        out, gt, ig, rows_cap, cols_cap,
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

                    let grad_out_f64 = grad_out.to_f64_vec();
                    let out_data: Vec<f64> = match &*d_data_for_backward {
                        CudaBuffer::F32(b) => {
                            let mut cpu = vec![0.0f32; b.len()];
                            if crate::cuda::memory::copy_d2h(&mut cpu, b).is_ok() {
                                cpu.iter().map(|&v| v as f64).collect()
                            } else {
                                return;
                            }
                        }
                        CudaBuffer::F64(b) => {
                            let mut cpu = vec![0.0f64; b.len()];
                            if crate::cuda::memory::copy_d2h(&mut cpu, b).is_ok() {
                                cpu
                            } else {
                                return;
                            }
                        }
                        CudaBuffer::BF16(_) | CudaBuffer::I8(_) => return,
                    };

                    let mut inp_grad = parents[0].grad_write_compat();
                    for row in 0..rows_cap {
                        let base = row * cols_cap;
                        let mut sum_term = 0.0;
                        for j in 0..cols_cap {
                            let idx = base + j;
                            sum_term += grad_out_f64[idx] * out_data[idx];
                        }
                        for j in 0..cols_cap {
                            let idx = base + j;
                            inp_grad[idx] += out_data[idx] * (grad_out_f64[idx] - sum_term);
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_data);
        out
    }

    /// CPU fallback for Softmax
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn softmax_cpu_fallback(&self) -> Tensor {
        let cpu_view = Tensor {
            data: self.data.clone(),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: self._ctx.clone(),
        };
        cpu_view.softmax()
    }

    /// GPU-accelerated causal softmax (forward only, no materialization).
    /// Causal mask is applied inside the kernel (no separate mask step).
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub(crate) fn softmax_causal_cuda(&self) -> Tensor {
        use crate::cuda::kernels::{softmax_causal_inplace, softmax_causal_inplace_f32};
        use crate::cuda::memory::{alloc, copy_d2d, CudaBuffer};

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.softmax_cpu_fallback(),
        };
        if len == 0 {
            return self.softmax_cpu_fallback();
        }
        let cols = self.shape.last().copied().unwrap_or(len.max(1));
        if cols == 0 {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA CausalSoftmax invalid last dimension (cols=0), using CPU");
            return self.softmax_cpu_fallback();
        }
        let rows = len / cols;
        if rows.checked_mul(cols) != Some(len) {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!(
                "[Autograd] CUDA CausalSoftmax shape mismatch (len={}, rows={}, cols={}), using CPU",
                len, rows, cols
            );
            return self.softmax_cpu_fallback();
        }
        crate::cuda::record_activation_attempt();

        let d_src = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_activation_fallback(stage);
                log::warn!(
                    "[Autograd] CUDA prepare CausalSoftmax input failed ({}), using CPU",
                    err
                );
                return self.softmax_cpu_fallback();
            }
        };
        let d_data = match self.dtype {
            Dtype::F32 => match alloc::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc CausalSoftmax buffer failed ({}), using CPU",
                        err
                    );
                    return self.softmax_cpu_fallback();
                }
            },
            Dtype::F64 => match alloc::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc CausalSoftmax buffer failed ({}), using CPU",
                        err
                    );
                    return self.softmax_cpu_fallback();
                }
            },
            _ => return self.softmax_cpu_fallback(),
        };
        let copy_ok = match (self.dtype, &d_data, &*d_src) {
            (Dtype::F32, CudaBuffer::F32(dst), CudaBuffer::F32(src)) => copy_d2d(dst, src).is_ok(),
            (Dtype::F64, CudaBuffer::F64(dst), CudaBuffer::F64(src)) => copy_d2d(dst, src).is_ok(),
            _ => false,
        };
        if !copy_ok {
            crate::cuda::record_activation_fallback("copy");
            log::warn!("[Autograd] CUDA D2D CausalSoftmax input copy failed, using CPU");
            return self.softmax_cpu_fallback();
        }

        let kernel_ok = match (self.dtype, &d_data) {
            (Dtype::F32, CudaBuffer::F32(b)) => softmax_causal_inplace_f32(b, rows, cols).is_ok(),
            (Dtype::F64, CudaBuffer::F64(b)) => softmax_causal_inplace(b, rows, cols).is_ok(),
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA CausalSoftmax kernel failed, using CPU");
            return self.softmax_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let rows_cap = rows;
        let cols_cap = cols;
        let d_data_for_backward = d_data.clone();
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
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (
                                    &*d_data_for_backward,
                                    &*d_grad_tmp,
                                    &*d_input_grad,
                                    input.dtype,
                                ) {
                                    (
                                        CudaBuffer::F32(out),
                                        CudaBuffer::F32(gt),
                                        CudaBuffer::F32(ig),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::softmax_backward_f32(
                                        out, gt, ig, rows_cap, cols_cap,
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(out),
                                        CudaBuffer::F64(gt),
                                        CudaBuffer::F64(ig),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::softmax_backward(
                                        out, gt, ig, rows_cap, cols_cap,
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

                    let grad_out_f64 = grad_out.to_f64_vec();
                    let out_data: Vec<f64> = match &*d_data_for_backward {
                        CudaBuffer::F32(b) => {
                            let mut cpu = vec![0.0f32; b.len()];
                            if crate::cuda::memory::copy_d2h(&mut cpu, b).is_ok() {
                                cpu.iter().map(|&v| v as f64).collect()
                            } else {
                                return;
                            }
                        }
                        CudaBuffer::F64(b) => {
                            let mut cpu = vec![0.0f64; b.len()];
                            if crate::cuda::memory::copy_d2h(&mut cpu, b).is_ok() {
                                cpu
                            } else {
                                return;
                            }
                        }
                        CudaBuffer::BF16(_) | CudaBuffer::I8(_) => return,
                    };

                    let mut inp_grad = parents[0].grad_write_compat();
                    for row in 0..rows_cap {
                        let base = row * cols_cap;
                        let mut sum_term = 0.0;
                        for j in 0..cols_cap {
                            let idx = base + j;
                            sum_term += grad_out_f64[idx] * out_data[idx];
                        }
                        for j in 0..cols_cap {
                            let idx = base + j;
                            inp_grad[idx] += out_data[idx] * (grad_out_f64[idx] - sum_term);
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_data);
        out
    }

    /// GPU-accelerated RoPE forward.
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub(crate) fn rope_cuda(
        &self,
        cos_cache: &[f64],
        sin_cache: &[f64],
        seq_len: usize,
        dim: usize,
        total_batches: usize,
        start_pos: usize,
    ) -> Tensor {
        use crate::cuda::kernels::{rope_inplace, rope_inplace_f32};
        use crate::cuda::memory::{alloc, copy_d2d, copy_h2d, CudaBuffer};

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.clone(),
        };
        if len == 0 {
            return self.clone();
        }
        let expected = total_batches * seq_len * dim;
        if len != expected {
            return self.clone();
        }
        crate::cuda::record_activation_attempt();

        let d_src = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_activation_fallback(stage);
                log::warn!("[Autograd] CUDA RoPE upload failed ({}), using CPU", err);
                return self.clone();
            }
        };
        let d_data = match self.dtype {
            Dtype::F32 => match alloc::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(_) => {
                    crate::cuda::record_activation_fallback("alloc");
                    return self.clone();
                }
            },
            Dtype::F64 => match alloc::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(_) => {
                    crate::cuda::record_activation_fallback("alloc");
                    return self.clone();
                }
            },
            _ => return self.clone(),
        };
        let copy_ok = match (self.dtype, &d_data, &*d_src) {
            (Dtype::F32, CudaBuffer::F32(dst), CudaBuffer::F32(src)) => copy_d2d(dst, src).is_ok(),
            (Dtype::F64, CudaBuffer::F64(dst), CudaBuffer::F64(src)) => copy_d2d(dst, src).is_ok(),
            _ => false,
        };
        if !copy_ok {
            crate::cuda::record_activation_fallback("copy");
            return self.clone();
        }

        let kernel_ok = match (self.dtype, &d_data) {
            (Dtype::F32, CudaBuffer::F32(b)) => {
                let cos_f32: Vec<f32> = cos_cache.iter().map(|&v| v as f32).collect();
                let sin_f32: Vec<f32> = sin_cache.iter().map(|&v| v as f32).collect();
                let d_cos = match alloc::<f32>(cos_f32.len()) {
                    Ok(buf) => buf,
                    Err(_) => {
                        crate::cuda::record_activation_fallback("alloc_cos");
                        return self.clone();
                    }
                };
                let d_sin = match alloc::<f32>(sin_f32.len()) {
                    Ok(buf) => buf,
                    Err(_) => {
                        crate::cuda::record_activation_fallback("alloc_sin");
                        return self.clone();
                    }
                };
                if copy_h2d(&d_cos, &cos_f32).is_err() || copy_h2d(&d_sin, &sin_f32).is_err() {
                    crate::cuda::record_activation_fallback("copy_cos");
                    return self.clone();
                }
                rope_inplace_f32(b, &d_cos, &d_sin, seq_len, dim, total_batches, start_pos).is_ok()
            }
            (Dtype::F64, CudaBuffer::F64(b)) => {
                let d_cos = match alloc::<f64>(cos_cache.len()) {
                    Ok(buf) => buf,
                    Err(_) => {
                        crate::cuda::record_activation_fallback("alloc_cos");
                        return self.clone();
                    }
                };
                let d_sin = match alloc::<f64>(sin_cache.len()) {
                    Ok(buf) => buf,
                    Err(_) => {
                        crate::cuda::record_activation_fallback("alloc_sin");
                        return self.clone();
                    }
                };
                if copy_h2d(&d_cos, cos_cache).is_err() || copy_h2d(&d_sin, sin_cache).is_err() {
                    crate::cuda::record_activation_fallback("copy_cos");
                    return self.clone();
                }
                rope_inplace(b, &d_cos, &d_sin, seq_len, dim, total_batches, start_pos).is_ok()
            }
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA RoPE kernel failed, using CPU");
            return self.clone();
        }
        crate::cuda::record_activation_success();

        let d_data = Arc::new(d_data);
        let d_cos_for_backward;
        let d_sin_for_backward;
        let parents = vec![self.clone()];
        let dim_cap = dim;
        let cos_cache_for_backward = Arc::new(cos_cache.to_vec());
        let sin_cache_for_backward = Arc::new(sin_cache.to_vec());
        let seq_len_cap = seq_len;
        let total_batches_cap = total_batches;
        let start_pos_cap = start_pos;
        let out_dtype = self.dtype;
        let grad_dtype = Tensor::grad_dtype_for(out_dtype);

        match self.dtype {
            Dtype::F32 => {
                let cos_f32: Vec<f32> = cos_cache.iter().map(|&v| v as f32).collect();
                let sin_f32: Vec<f32> = sin_cache.iter().map(|&v| v as f32).collect();
                let Ok(d_cos) = alloc::<f32>(cos_f32.len()) else {
                    return self.clone();
                };
                let Ok(d_sin) = alloc::<f32>(sin_f32.len()) else {
                    return self.clone();
                };
                if copy_h2d(&d_cos, &cos_f32).is_err() || copy_h2d(&d_sin, &sin_f32).is_err() {
                    return self.clone();
                }
                d_cos_for_backward = Arc::new(CudaBuffer::F32(d_cos));
                d_sin_for_backward = Arc::new(CudaBuffer::F32(d_sin));
            }
            Dtype::F64 => {
                let Ok(d_cos) = alloc::<f64>(cos_cache.len()) else {
                    return self.clone();
                };
                let Ok(d_sin) = alloc::<f64>(sin_cache.len()) else {
                    return self.clone();
                };
                if copy_h2d(&d_cos, cos_cache).is_err() || copy_h2d(&d_sin, sin_cache).is_err() {
                    return self.clone();
                }
                d_cos_for_backward = Arc::new(CudaBuffer::F64(d_cos));
                d_sin_for_backward = Arc::new(CudaBuffer::F64(d_sin));
            }
            _ => return self.clone(),
        }

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
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (
                                    &*d_grad_tmp,
                                    &*d_cos_for_backward,
                                    &*d_sin_for_backward,
                                    &*d_input_grad,
                                    input.dtype,
                                ) {
                                    (
                                        CudaBuffer::F32(gt),
                                        CudaBuffer::F32(cos),
                                        CudaBuffer::F32(sin),
                                        CudaBuffer::F32(ig),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::rope_backward_f32(
                                        gt,
                                        cos,
                                        sin,
                                        ig,
                                        seq_len_cap,
                                        dim_cap,
                                        total_batches_cap,
                                        start_pos_cap,
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(gt),
                                        CudaBuffer::F64(cos),
                                        CudaBuffer::F64(sin),
                                        CudaBuffer::F64(ig),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::rope_backward(
                                        gt,
                                        cos,
                                        sin,
                                        ig,
                                        seq_len_cap,
                                        dim_cap,
                                        total_batches_cap,
                                        start_pos_cap,
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
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    let half_dim = dim_cap / 2;
                    for b in 0..total_batches_cap {
                        for t in 0..seq_len_cap {
                            let pos = start_pos_cap + t;
                            let cache_idx = pos * half_dim;
                            if cache_idx + half_dim > cos_cache_for_backward.len()
                                || cache_idx + half_dim > sin_cache_for_backward.len()
                            {
                                continue;
                            }
                            let base_idx = b * (seq_len_cap * dim_cap) + t * dim_cap;
                            for i in 0..half_dim {
                                let c = cos_cache_for_backward[cache_idx + i];
                                let s = sin_cache_for_backward[cache_idx + i];
                                let g1 = grad_out_f64[base_idx + 2 * i];
                                let g2 = grad_out_f64[base_idx + 2 * i + 1];
                                inp_grad[base_idx + 2 * i] += g1 * c + g2 * s;
                                inp_grad[base_idx + 2 * i + 1] += -g1 * s + g2 * c;
                            }
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_data);
        out
    }

    /// GPU-accelerated log-softmax for the last dimension.
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn log_softmax_cuda_last_dim(&self) -> Tensor {
        use crate::cuda::kernels::log_softmax_f32;
        use crate::cuda::memory::{alloc, CudaBuffer};

        let shape = self.shape.clone();
        let dim_size = *shape.last().unwrap_or(&1);
        if dim_size == 0 {
            return self.log_softmax_last_dim_cpu_fallback();
        }

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.log_softmax_last_dim_cpu_fallback(),
        };
        let num_slices = len / dim_size;
        if len == 0 || num_slices == 0 {
            return self.log_softmax_last_dim_cpu_fallback();
        }
        crate::cuda::record_log_softmax_attempt();

        let d_in = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_log_softmax_fallback(stage);
                log::warn!(
                    "[Autograd] CUDA prepare LogSoftmax input failed ({}), using CPU",
                    err
                );
                return self.log_softmax_last_dim_cpu_fallback();
            }
        };
        let d_out = match self.dtype {
            Dtype::F32 => match alloc::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(err) => {
                    crate::cuda::record_log_softmax_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc LogSoftmax output failed ({}), using CPU",
                        err
                    );
                    return self.log_softmax_last_dim_cpu_fallback();
                }
            },
            Dtype::F64 => match alloc::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(err) => {
                    crate::cuda::record_log_softmax_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc LogSoftmax output failed ({}), using CPU",
                        err
                    );
                    return self.log_softmax_last_dim_cpu_fallback();
                }
            },
            _ => return self.log_softmax_last_dim_cpu_fallback(),
        };
        let d_out = Arc::new(d_out);

        let kernel_ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(i), CudaBuffer::F32(o), Dtype::F32) => {
                log_softmax_f32(i, o, num_slices, dim_size).is_ok()
            }
            (CudaBuffer::F64(i), CudaBuffer::F64(o), Dtype::F64) => {
                crate::cuda::kernels::log_softmax(i, o, num_slices, dim_size).is_ok()
            }
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_log_softmax_fallback("kernel");
            log::warn!("[Autograd] CUDA LogSoftmax kernel failed, using CPU");
            return self.log_softmax_last_dim_cpu_fallback();
        }

        crate::cuda::record_log_softmax_success();

        let parents = vec![self.clone()];
        let dim_size_cap = dim_size;
        let num_slices_cap = num_slices;
        let d_out_for_backward = d_out.clone();
        let out_dtype = self.dtype;
        let grad_dtype = Tensor::grad_dtype_for(out_dtype);

        let out = Tensor {
            data: Tensor::empty_storage(out_dtype),
            grad: Storage::zeros(len, grad_dtype),
            shape,
            device: Device::Cuda,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    if input.device == Device::Cuda {
                        if let Some(d_grad_tmp) = cuda_grad_out_buffer(grad_out) {
                            if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                let ok = match (
                                    &*d_out_for_backward,
                                    &*d_grad_tmp,
                                    &*d_input_grad,
                                    input.dtype,
                                ) {
                                    (
                                        CudaBuffer::F32(out),
                                        CudaBuffer::F32(gt),
                                        CudaBuffer::F32(ig),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::log_softmax_backward_f32(
                                        out,
                                        gt,
                                        ig,
                                        num_slices_cap,
                                        dim_size_cap,
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(out),
                                        CudaBuffer::F64(gt),
                                        CudaBuffer::F64(ig),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::log_softmax_backward(
                                        out,
                                        gt,
                                        ig,
                                        num_slices_cap,
                                        dim_size_cap,
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

                    let grad_out_f64 = grad_out.to_f64_vec();
                    let log_softmax_data: Vec<f64> = match &*d_out_for_backward {
                        CudaBuffer::F32(b) => {
                            let mut cpu = vec![0.0f32; b.len()];
                            if crate::cuda::memory::copy_d2h(&mut cpu, b).is_ok() {
                                cpu.iter().map(|&v| v as f64).collect()
                            } else {
                                return;
                            }
                        }
                        CudaBuffer::F64(b) => {
                            let mut cpu = vec![0.0f64; b.len()];
                            if crate::cuda::memory::copy_d2h(&mut cpu, b).is_ok() {
                                cpu
                            } else {
                                return;
                            }
                        }
                        CudaBuffer::BF16(_) | CudaBuffer::I8(_) => return,
                    };
                    let mut inp_grad = parents[0].grad_write_compat();
                    for slice_idx in 0..num_slices_cap {
                        let base = slice_idx * dim_size_cap;
                        let mut slice_sum = 0.0;
                        for j in 0..dim_size_cap {
                            slice_sum += grad_out_f64[base + j];
                        }
                        for j in 0..dim_size_cap {
                            let idx = base + j;
                            inp_grad[idx] +=
                                grad_out_f64[idx] - log_softmax_data[idx].exp() * slice_sum;
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        out
    }

    #[cfg(cuda)]
    fn log_softmax_last_dim_cpu_fallback(&self) -> Tensor {
        let cpu_view = Tensor {
            data: self.data.clone(),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: self._ctx.clone(),
        };
        cpu_view.log_softmax_dim(cpu_view.shape.len() - 1)
    }

    // --- GPU element-wise helpers ---

    #[cfg(cuda)]
    #[allow(clippy::type_complexity)]
    fn elementwise_op_cuda(
        &self,
        rhs: &Tensor,
        op: CudaBinaryOp,
        forward_f32: fn(
            &crate::cuda::memory::DevicePtr<f32>,
            &crate::cuda::memory::DevicePtr<f32>,
            &crate::cuda::memory::DevicePtr<f32>,
            usize,
        ) -> crate::cuda::error::CudaResult<()>,
        forward_f64: fn(
            &crate::cuda::memory::DevicePtr<f64>,
            &crate::cuda::memory::DevicePtr<f64>,
            &crate::cuda::memory::DevicePtr<f64>,
            usize,
        ) -> crate::cuda::error::CudaResult<()>,
        backward_f32: Option<
            fn(
                &crate::cuda::memory::DevicePtr<f32>,
                &crate::cuda::memory::DevicePtr<f32>,
                &crate::cuda::memory::DevicePtr<f32>,
                &crate::cuda::memory::DevicePtr<f32>,
                &crate::cuda::memory::DevicePtr<f32>,
                usize,
            ) -> crate::cuda::error::CudaResult<()>,
        >,
        backward_f64: Option<
            fn(
                &crate::cuda::memory::DevicePtr<f64>,
                &crate::cuda::memory::DevicePtr<f64>,
                &crate::cuda::memory::DevicePtr<f64>,
                &crate::cuda::memory::DevicePtr<f64>,
                &crate::cuda::memory::DevicePtr<f64>,
                usize,
            ) -> crate::cuda::error::CudaResult<()>,
        >,
    ) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, CudaBuffer};

        if self.device != Device::Cuda || rhs.device != Device::Cuda {
            return None;
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        if out_dtype != Dtype::F32 && out_dtype != Dtype::F64 {
            return None;
        }
        let len = self.numel();
        let d_a = self.cuda_get_or_upload_buffer().ok()?;
        let d_b = rhs.cuda_get_or_upload_buffer().ok()?;
        let d_out = match out_dtype {
            Dtype::F32 => alloc::<f32>(len).ok().map(CudaBuffer::F32),
            Dtype::F64 => alloc::<f64>(len).ok().map(CudaBuffer::F64),
            _ => None,
        }?;
        let d_out = std::sync::Arc::new(d_out);

        let fw_ok = match (&*d_a, &*d_b, &*d_out, out_dtype) {
            (CudaBuffer::F32(a), CudaBuffer::F32(b), CudaBuffer::F32(o), Dtype::F32) => {
                forward_f32(a, b, o, len).is_ok()
            }
            (CudaBuffer::F64(a), CudaBuffer::F64(b), CudaBuffer::F64(o), Dtype::F64) => {
                forward_f64(a, b, o, len).is_ok()
            }
            _ => false,
        };
        if !fw_ok {
            return None;
        }

        let parents = vec![self.clone(), rhs.clone()];
        let out = Tensor {
            data: Tensor::empty_storage(out_dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let a = &parents[0];
                    let b = &parents[1];
                    let mut gpu_backward_ok = false;
                    #[cfg(cuda)]
                    if a.device == Device::Cuda && b.device == Device::Cuda {
                        if let (Some(d_a), Some(d_b)) =
                            (a.cuda_cached_buffer(), b.cuda_cached_buffer())
                        {
                            let grad_dtype = grad_out.dtype();
                            let d_grad_tmp = cuda_grad_out_buffer(grad_out);
                            if let Some(d_grad_tmp) = d_grad_tmp {
                                if let Some(d_a_grad) = a.cuda_grad_ensure_buffer() {
                                    if let Some(d_b_grad) = b.cuda_grad_ensure_buffer() {
                                        match (
                                            &*d_grad_tmp,
                                            &*d_a,
                                            &*d_b,
                                            &*d_a_grad,
                                            &*d_b_grad,
                                            grad_dtype,
                                        ) {
                                            (
                                                CudaBuffer::F32(gt),
                                                CudaBuffer::F32(a_buf),
                                                CudaBuffer::F32(b_buf),
                                                CudaBuffer::F32(ag),
                                                CudaBuffer::F32(bg),
                                                Dtype::F32,
                                            ) => {
                                                if let Some(bk) = backward_f32 {
                                                    gpu_backward_ok = bk(
                                                        gt,
                                                        a_buf,
                                                        b_buf,
                                                        ag,
                                                        bg,
                                                        grad_out_f64.len(),
                                                    )
                                                    .is_ok();
                                                }
                                            }
                                            (
                                                CudaBuffer::F64(gt),
                                                CudaBuffer::F64(a_buf),
                                                CudaBuffer::F64(b_buf),
                                                CudaBuffer::F64(ag),
                                                CudaBuffer::F64(bg),
                                                Dtype::F64,
                                            ) => {
                                                if let Some(bk) = backward_f64 {
                                                    gpu_backward_ok = bk(
                                                        gt,
                                                        a_buf,
                                                        b_buf,
                                                        ag,
                                                        bg,
                                                        grad_out_f64.len(),
                                                    )
                                                    .is_ok();
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if gpu_backward_ok {
                        return;
                    }

                    let mut a_grad = a.grad_write_compat();
                    let mut b_grad = b.grad_write_compat();
                    match op {
                        CudaBinaryOp::Add => {
                            for i in 0..grad_out_f64.len() {
                                a_grad[i] += grad_out_f64[i];
                                b_grad[i] += grad_out_f64[i];
                            }
                        }
                        CudaBinaryOp::Sub => {
                            for i in 0..grad_out_f64.len() {
                                a_grad[i] += grad_out_f64[i];
                                b_grad[i] -= grad_out_f64[i];
                            }
                        }
                        CudaBinaryOp::Mul => {
                            let a_data = a.data_as_f64_vec();
                            let b_data = b.data_as_f64_vec();
                            for i in 0..grad_out_f64.len() {
                                a_grad[i] += grad_out_f64[i] * b_data[i];
                                b_grad[i] += grad_out_f64[i] * a_data[i];
                            }
                        }
                        CudaBinaryOp::Div => {
                            let a_data = a.data_as_f64_vec();
                            let b_data = b.data_as_f64_vec();
                            for i in 0..grad_out_f64.len() {
                                let safe_b = if b_data[i] != 0.0 { b_data[i] } else { 1e-12 };
                                a_grad[i] += grad_out_f64[i] / safe_b;
                                b_grad[i] += grad_out_f64[i] * (-a_data[i] / (safe_b * safe_b));
                            }
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(super) fn add_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            CudaBinaryOp::Add,
            crate::cuda::kernels::add_forward_f32,
            crate::cuda::kernels::add_forward,
            Some(|grad_out, _a_data, _b_data, a_grad, b_grad, size| {
                crate::cuda::kernels::add_backward_f32(grad_out, a_grad, b_grad, size)
            }),
            Some(|grad_out, _a_data, _b_data, a_grad, b_grad, size| {
                crate::cuda::kernels::add_backward(grad_out, a_grad, b_grad, size)
            }),
        )
    }

    #[cfg(cuda)]
    pub(super) fn sub_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            CudaBinaryOp::Sub,
            crate::cuda::kernels::sub_forward_f32,
            crate::cuda::kernels::sub_forward,
            Some(|grad_out, _a_data, _b_data, a_grad, b_grad, size| {
                crate::cuda::kernels::sub_backward_f32(grad_out, a_grad, b_grad, size)
            }),
            Some(|grad_out, _a_data, _b_data, a_grad, b_grad, size| {
                crate::cuda::kernels::sub_backward(grad_out, a_grad, b_grad, size)
            }),
        )
    }

    #[cfg(cuda)]
    pub(super) fn mul_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            CudaBinaryOp::Mul,
            crate::cuda::kernels::mul_forward_f32,
            crate::cuda::kernels::mul_forward,
            Some(crate::cuda::kernels::mul_backward_f32),
            Some(crate::cuda::kernels::mul_backward),
        )
    }

    #[cfg(cuda)]
    pub(super) fn div_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            CudaBinaryOp::Div,
            crate::cuda::kernels::div_forward_f32,
            crate::cuda::kernels::div_forward,
            Some(crate::cuda::kernels::div_backward_f32),
            Some(crate::cuda::kernels::div_backward),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file_path(prefix: &str) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("{}_{}_{}.bin", prefix, std::process::id(), now))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_broadcast_scalar() {
        let t = Tensor::new_f32(vec![5.0], vec![1]);
        let b = t.broadcast(vec![2, 2]);
        assert_eq!(b.shape, vec![2, 2]);
        let data = b.data_as_f64_vec();
        assert_eq!(*data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_from_mmap_rejects_trailing_bytes() {
        let path = temp_file_path("autograd_mmap_invalid");
        std::fs::write(&path, [0u8; 9]).unwrap();

        let result = Tensor::from_mmap(&path, vec![1]);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_mmap_roundtrip_small_tensor() {
        let path = temp_file_path("autograd_mmap_roundtrip");
        let tensor = Tensor::new_f32(vec![1.25, -2.5], vec![2]);
        tensor.save_binary(&path).unwrap();

        let loaded = Tensor::from_mmap(&path, vec![2]).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.shape, vec![2]);
        let data = loaded.data_as_f64_vec();
        assert_eq!(*data, vec![1.25, -2.5]);
    }

    #[test]
    fn test_bf16_serde_roundtrip_preserves_dtype() {
        let tensor = Tensor::new_bf16(vec![1.25, -2.5, 3.5], vec![3]);
        let buf = crate::binary_codec::to_vec(&tensor).unwrap();
        let decoded: Tensor = crate::binary_codec::from_slice(&buf).unwrap();

        assert_eq!(decoded.dtype, Dtype::BF16);
        assert_eq!(decoded.shape, vec![3]);
        let data = decoded.data_to_f32_vec();
        assert!((data[0] - 1.25).abs() < 0.01);
        assert!((data[1] + 2.5).abs() < 0.01);
        assert!((data[2] - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_log_softmax_dim_zero_normalization() {
        let t = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = t.log_softmax_dim(0);
        let d = out.data_as_f64_vec();

        for col in 0..3 {
            let p0 = d[col].exp();
            let p1 = d[3 + col].exp();
            let sum = p0 + p1;
            assert!((sum - 1.0).abs() < 1e-5, "column {} sum={}", col, sum);
        }
    }

    #[test]
    fn test_log_softmax_dim_one_normalization() {
        let t = Tensor::new_f32(vec![1.0, -1.0, 2.0, 0.0, 3.0, 4.0], vec![2, 3]);
        let out = t.log_softmax_dim(1);
        let d = out.data_as_f64_vec();

        for row in 0..2 {
            let base = row * 3;
            let sum = d[base..base + 3].iter().map(|v| v.exp()).sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-5, "row {} sum={}", row, sum);
        }
    }

    #[test]
    fn test_softmax_last_dim_row_normalization_cpu() {
        let t = Tensor::new_f32(vec![1.0, -1.0, 2.0, 0.0, 3.0, 4.0], vec![2, 3]);
        let out = t.softmax();
        let d = out.data_as_f64_vec();

        for row in 0..2 {
            let base = row * 3;
            let sum = d[base..base + 3].iter().sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-5, "row {} sum={}", row, sum);
        }
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_softmax_last_dim_row_normalization() {
        if crate::cuda::init().is_err() {
            return;
        }

        let t = Tensor::new_f32(vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0], vec![2, 3]);
        let t_cuda = match t.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let out = t_cuda.softmax();
        let d = out.data_as_f64_vec();

        for row in 0..2 {
            let base = row * 3;
            let sum = d[base..base + 3].iter().sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-5, "row {} sum={}", row, sum);
        }
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_log_softmax_last_dim_matches_cpu() {
        if crate::cuda::init().is_err() {
            return;
        }

        let t = Tensor::new_f32(vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0], vec![2, 3]);
        let t_cuda = match t.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let cpu = t.log_softmax_dim(1);
        let cuda = t_cuda.log_softmax_dim(1);

        let cpu_d = cpu.data_as_f64_vec();
        let cuda_d = cuda.data_as_f64_vec();
        assert_eq!(cpu_d.len(), cuda_d.len());
        for i in 0..cpu_d.len() {
            assert!(
                (cpu_d[i] - cuda_d[i]).abs() < 1e-5,
                "idx {} cpu={} cuda={}",
                i,
                cpu_d[i],
                cuda_d[i]
            );
        }
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_relu_refreshes_cached_input_after_host_mutation() {
        if crate::cuda::init().is_err() {
            return;
        }

        let t = Tensor::new_f32(vec![-3.0, 2.0], vec![2]);
        let t_cuda = match t.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };

        {
            let mut data = t_cuda.data_write_f32();
            data[0] = 5.0;
            data[1] = -4.0;
        }

        let out = t_cuda.relu();
        let d = out.data_as_f64_vec();
        assert_eq!(d.as_slice(), &[5.0, 0.0]);
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_softmax_wide_last_dim_matches_cpu() {
        if crate::cuda::init().is_err() {
            return;
        }

        let cols = 64usize;
        let mut values = Vec::with_capacity(cols * 2);
        for i in 0..(cols * 2) {
            values.push((i as f64 * 0.125) - 4.0);
        }

        let t = Tensor::new_f32(values, vec![2, cols]);
        let t_cuda = match t.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };

        let cpu = t.softmax();
        let cuda = t_cuda.softmax();
        let cpu_d = cpu.data_as_f64_vec();
        let cuda_d = cuda.data_as_f64_vec();
        assert_eq!(cpu_d.len(), cuda_d.len());
        for i in 0..cpu_d.len() {
            assert!(
                (cpu_d[i] - cuda_d[i]).abs() < 1e-5,
                "idx {} cpu={} cuda={}",
                i,
                cpu_d[i],
                cuda_d[i]
            );
        }
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_softmax_small_batch_rows_match_cpu() {
        if crate::cuda::init().is_err() {
            return;
        }

        for rows in [2usize, 3, 5] {
            let cols = 7usize;
            let values: Vec<f64> = (0..rows * cols)
                .map(|i| (i as f64 * 0.37).sin() * 3.0)
                .collect();
            let t = Tensor::new_f32(values, vec![rows, cols]);
            let t_cuda = match t.to_cuda() {
                Ok(tensor) => tensor,
                Err(_) => return,
            };
            let cpu = t.softmax();
            let cuda = t_cuda.softmax();
            let cpu_d = cpu.data_as_f64_vec();
            let cuda_d = cuda.data_as_f64_vec();
            for i in 0..cpu_d.len() {
                assert!(
                    (cpu_d[i] - cuda_d[i]).abs() < 1e-5,
                    "rows={} idx={} cpu={} cuda={}",
                    rows,
                    i,
                    cpu_d[i],
                    cuda_d[i]
                );
            }
        }
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_device_only_reshape_transpose_roundtrip() {
        if crate::cuda::init().is_err() {
            return;
        }

        let t = Tensor::new_f32((0..24).map(|v| v as f64).collect(), vec![2, 3, 4]);
        let t_cuda = match t.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let reshaped = t_cuda.reshape(vec![2, 3, 2, 2]);
        assert_eq!(reshaped.device, Device::Cuda);
        let transposed = reshaped.transpose(1, 2);
        assert_eq!(transposed.device, Device::Cuda);
        let data = transposed.data_as_f64_vec();
        assert_eq!(data.len(), 24);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[23], 23.0);
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_div_backward_matches_cpu() {
        if crate::cuda::init().is_err() {
            return;
        }

        let a = Tensor::new_f32(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
        let b = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let cpu = (&a / &b).sum();
        cpu.backward();
        let a_cpu_grad = a.grad_to_f64_vec();
        let b_cpu_grad = b.grad_to_f64_vec();

        let a_cuda = match Tensor::new_f32(vec![2.0, 4.0, 6.0, 8.0], vec![4]).to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let b_cuda = match Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]).to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let cuda = (&a_cuda / &b_cuda).sum();
        cuda.backward();
        let a_cuda_grad = a_cuda.grad_to_f64_vec();
        let b_cuda_grad = b_cuda.grad_to_f64_vec();

        for i in 0..4 {
            assert!((a_cpu_grad[i] - a_cuda_grad[i]).abs() < 1e-5);
            assert!((b_cpu_grad[i] - b_cuda_grad[i]).abs() < 1e-5);
        }
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_softmax_log_softmax_backward_matches_cpu() {
        if crate::cuda::init().is_err() {
            return;
        }

        let values = vec![0.2, -1.0, 2.0, 0.7, -0.3, 1.4];
        let cpu_in = Tensor::new_f32(values.clone(), vec![2, 3]);
        cpu_in.softmax().sum().backward();
        let cpu_grad = cpu_in.grad_to_f64_vec();

        let cuda_in = match Tensor::new_f32(values.clone(), vec![2, 3]).to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        cuda_in.softmax().sum().backward();
        let cuda_grad = cuda_in.grad_to_f64_vec();
        for i in 0..cpu_grad.len() {
            assert!((cpu_grad[i] - cuda_grad[i]).abs() < 1e-5);
        }

        let cpu_in = Tensor::new_f32(values.clone(), vec![2, 3]);
        cpu_in.log_softmax_dim(1).sum().backward();
        let cpu_grad = cpu_in.grad_to_f64_vec();
        let cuda_in = match Tensor::new_f32(values, vec![2, 3]).to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        cuda_in.log_softmax_dim(1).sum().backward();
        let cuda_grad = cuda_in.grad_to_f64_vec();
        for i in 0..cpu_grad.len() {
            assert!((cpu_grad[i] - cuda_grad[i]).abs() < 1e-5);
        }
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_bf16_matmul_tensor_core_path() {
        if crate::cuda::init().is_err() {
            return;
        }

        let a = Tensor::with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Dtype::BF16);
        let b = Tensor::with_dtype(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Dtype::BF16);
        let a_cuda = match a.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let b_cuda = match b.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let out = a_cuda.matmul(&b_cuda);
        assert_eq!(out.device, Device::Cuda);
        assert_eq!(out.dtype, Dtype::F32);
        let data = out.data_as_f64_vec();
        assert!((data[0] - 19.0).abs() < 1e-2);
        assert!((data[3] - 50.0).abs() < 1e-2);
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_detach_preserves_device() {
        if crate::cuda::init().is_err() {
            return;
        }

        let t = Tensor::new_f32(vec![1.0, 2.0], vec![2]);
        let t_cuda = match t.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let detached = t_cuda.detach();
        assert_eq!(detached.device, Device::Cuda);
    }

    #[cfg(cuda)]
    #[test]
    fn test_cuda_grad_cache_is_separate_from_data_cache() {
        if crate::cuda::init().is_err() {
            return;
        }

        let a = match Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let b = match Tensor::new_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };
        let out = (&a + &b).sum();
        out.backward();

        let grad = a.grad_to_f64_vec();
        assert_eq!(grad.len(), 4);
        assert_eq!(a.data_as_f64_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_matmul_performance() {
        use std::time::Instant;
        let size = 1024;
        println!("Initializing {}x{} tensors...", size, size);
        let a = Tensor::rand(vec![size, size], -1.0, 1.0, 42);
        let b = Tensor::rand(vec![size, size], -1.0, 1.0, 123);

        println!("Starting MatMul...");
        let start = Instant::now();
        let _c = a.matmul(&b);
        let duration = start.elapsed();
        println!("MatMul {}x{} took: {:.2?}", size, size, duration);
    }

    #[test]
    fn test_conv2d_simple() {
        // Input: 1x1x3x3
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );

        // Weight: 1x1x2x2 (all ones)
        // [[1, 1],
        //  [1, 1]]
        let weight = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);

        // Output should be 2x2
        // [1+2+4+5, 2+3+5+6] = [12, 16]
        // [4+5+7+8, 5+6+8+9] = [24, 28]

        let out = input.conv2d(&weight, 1, 0);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        let data = out.data_as_f64_vec();
        assert_eq!(*data, vec![12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn test_max_pool2d_simple() {
        // Input: 1x1x4x4
        let data: Vec<f64> = (0..16).map(|x| x as f64).collect();
        let input = Tensor::new(data, vec![1, 1, 4, 4]);

        // Kernel 2, Stride 2
        // [[0, 1, 2, 3],
        //  [4, 5, 6, 7],
        //  [8, 9, 10, 11],
        //  [12,13, 14, 15]]
        //
        // Pool 2x2 s=2:
        // [max(0,1,4,5)=5, max(2,3,6,7)=7]
        // [max(8,9,12,13)=13, max(10,11,14,15)=15]

        let out = input.max_pool2d(2, 2, 0);
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        let d = out.data_as_f64_vec();
        assert_eq!(*d, vec![5.0, 7.0, 13.0, 15.0]);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Dtype-dispatch tests (F32 / BF16)
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_f32_matmul_forward_backward() {
        let a = Tensor::with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Dtype::F32);
        let b = Tensor::with_dtype(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Dtype::F32);

        let c = a.matmul(&b);
        assert_eq!(c.dtype, Dtype::F32);
        assert_eq!(c.shape, vec![2, 2]);

        // Expected: [[19, 22], [43, 50]]
        let c_f64 = c.data_as_f64_vec();
        assert!((c_f64[0] - 19.0).abs() < 1e-4);
        assert!((c_f64[1] - 22.0).abs() < 1e-4);
        assert!((c_f64[2] - 43.0).abs() < 1e-4);
        assert!((c_f64[3] - 50.0).abs() < 1e-4);

        // Backward
        c.sum().backward();
        let a_grad = a.grad_to_f64_vec();
        let _b_grad = b.grad_to_f64_vec();
        // dL/da = ones * b^T
        assert!((a_grad[0] - 11.0).abs() < 1e-4);
        assert!((a_grad[1] - 15.0).abs() < 1e-4);
        assert!((a_grad[2] - 11.0).abs() < 1e-4);
        assert!((a_grad[3] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_bf16_matmul() {
        let a = Tensor::with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Dtype::BF16);
        let b = Tensor::with_dtype(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Dtype::BF16);

        let c = a.matmul(&b);
        assert_eq!(c.dtype, Dtype::BF16);
        assert_eq!(c.shape, vec![2, 2]);

        let c_f64 = c.data_as_f64_vec();
        assert!((c_f64[0] - 19.0).abs() < 1e-2); // BF16 has lower precision
    }

    #[test]
    fn test_f32_elementwise_ops() {
        let a = Tensor::with_dtype(vec![1.0, 2.0, 3.0], vec![3], Dtype::F32);
        let b = Tensor::with_dtype(vec![4.0, 5.0, 6.0], vec![3], Dtype::F32);

        let add_out = &a + &b;
        assert_eq!(add_out.dtype, Dtype::F32);
        assert_eq!(add_out.data_as_f64_vec(), vec![5.0, 7.0, 9.0]);

        let sub_out = &a - &b;
        assert_eq!(sub_out.dtype, Dtype::F32);
        assert_eq!(sub_out.data_as_f64_vec(), vec![-3.0, -3.0, -3.0]);

        let mul_out = &a * &b;
        assert_eq!(mul_out.dtype, Dtype::F32);
        assert_eq!(mul_out.data_as_f64_vec(), vec![4.0, 10.0, 18.0]);

        let div_out = &b / &a;
        assert_eq!(div_out.dtype, Dtype::F32);
        assert_eq!(div_out.data_as_f64_vec(), vec![4.0, 2.5, 2.0]);
    }

    #[test]
    fn test_mixed_dtype_with_f64_promotes_to_f64() {
        let a = Tensor::with_dtype(vec![1.0, 2.0], vec![2], Dtype::F32);
        let b = Tensor::with_dtype(vec![3.0, 4.0], vec![2], Dtype::F64);

        let c = &a + &b;
        assert_eq!(c.dtype, Dtype::F64);
        assert_eq!(c.data_as_f64_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_mixed_f32_bf16_promotes_to_f32() {
        let a = Tensor::with_dtype(vec![1.0, 2.0], vec![2], Dtype::F32);
        let b = Tensor::with_dtype(vec![3.0, 4.0], vec![2], Dtype::BF16);

        let c = &a + &b;
        assert_eq!(c.dtype, Dtype::F32);
        assert_eq!(c.data_as_f64_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_f32_relu_softmax_sum() {
        let a = Tensor::with_dtype(vec![-1.0, 2.0, -3.0, 4.0], vec![4], Dtype::F32);

        let r = a.relu();
        assert_eq!(r.dtype, Dtype::F32);
        assert_eq!(r.data_as_f64_vec(), vec![0.0, 2.0, 0.0, 4.0]);

        let s = r.sum();
        assert_eq!(s.dtype, Dtype::F32);
        assert_eq!(s.data_as_f64_vec(), vec![6.0]);

        let m = r.mean();
        assert_eq!(m.dtype, Dtype::F32);
        assert_eq!(m.data_as_f64_vec(), vec![1.5]);

        let sm = r.softmax();
        assert_eq!(sm.dtype, Dtype::F32);
        let sm_f64 = sm.data_as_f64_vec();
        let sum_sm: f64 = sm_f64.iter().sum();
        assert!((sum_sm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_reshape_broadcast_transpose() {
        let a = Tensor::with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Dtype::F32);

        let r = a.reshape(vec![4]);
        assert_eq!(r.dtype, Dtype::F32);
        assert_eq!(r.data_as_f64_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        let b = a.broadcast(vec![2, 2]);
        assert_eq!(b.dtype, Dtype::F32);
        assert_eq!(b.shape, vec![2, 2]);

        let t = a.transpose2d();
        assert_eq!(t.dtype, Dtype::F32);
        assert_eq!(t.data_as_f64_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_f32_gelu_exp_log() {
        let a = Tensor::with_dtype(vec![0.0, 1.0, 2.0], vec![3], Dtype::F32);

        let g = a.gelu();
        assert_eq!(g.dtype, Dtype::F32);
        // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
        let g_f64 = g.data_as_f64_vec();
        assert!(g_f64[0].abs() < 1e-6);
        assert!((g_f64[1] - 0.841).abs() < 1e-3);

        let e = a.exp();
        assert_eq!(e.dtype, Dtype::F32);
        let e_f64 = e.data_as_f64_vec();
        assert!((e_f64[0] - 1.0).abs() < 1e-4);
        assert!((e_f64[1] - std::f64::consts::E).abs() < 1e-3);

        let l = Tensor::with_dtype(vec![1.0, 2.0, 3.0], vec![3], Dtype::F32).log();
        assert_eq!(l.dtype, Dtype::F32);
        let l_f64 = l.data_as_f64_vec();
        assert!(l_f64[0].abs() < 1e-6);
        assert!((l_f64[1] - 0.693).abs() < 1e-3);
    }

    #[test]
    fn test_f32_backward_elementwise() {
        let a = Tensor::with_dtype(vec![2.0, 3.0], vec![2], Dtype::F32);
        let b = Tensor::with_dtype(vec![4.0, 5.0], vec![2], Dtype::F32);

        let c = (&a * &b).sum();
        c.backward();

        let a_grad = a.grad_to_f64_vec();
        let b_grad = b.grad_to_f64_vec();
        // d(ab)/da = b
        assert!((a_grad[0] - 4.0).abs() < 1e-4);
        assert!((a_grad[1] - 5.0).abs() < 1e-4);
        // d(ab)/db = a
        assert!((b_grad[0] - 2.0).abs() < 1e-4);
        assert!((b_grad[1] - 3.0).abs() < 1e-4);
    }
}
