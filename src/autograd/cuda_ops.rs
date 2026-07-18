use super::{cuda_grad_out_buffer, Context, Device, Tensor, PAR_THRESHOLD};
use crate::dtype::{Dtype, Storage};
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(cuda)]
#[derive(Clone, Copy)]
enum CudaBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[cfg(cuda)]
pub(crate) fn cuda_clip_gradients_in_place(
    params: &[Tensor],
    max_norm: f64,
    eps: f64,
) -> crate::cuda::error::CudaResult<()> {
    use crate::cuda::error::CudaError;
    use crate::cuda::memory::{alloc_pooled, CudaBuffer};

    let dtype = params
        .iter()
        .find(|p| p.cuda_storage_len() > 0)
        .map(|p| p.grad.dtype());
    let Some(dtype) = dtype else {
        return Ok(());
    };
    if !matches!(dtype, Dtype::F32 | Dtype::F64) {
        return Err(CudaError::InvalidInput {
            op: "cuda_clip_gradients_in_place",
            message: "gradient dtype must be f32 or f64",
        });
    }
    if params
        .iter()
        .any(|p| p.cuda_storage_len() > 0 && p.grad.dtype() != dtype)
    {
        return Err(CudaError::InvalidInput {
            op: "cuda_clip_gradients_in_place",
            message: "all gradient buffers must use the same dtype",
        });
    }

    match dtype {
        Dtype::F32 => {
            let sumsq = alloc_pooled::<f32>(1)?;
            let coef = alloc_pooled::<f32>(1)?;
            crate::cuda::kernels::fill_f32(&sumsq, 0.0)?;
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = param
                    .cuda_grad_get_or_upload_buffer()
                    .map_err(|(_, error)| error)?;
                let CudaBuffer::F32(grad) = &*grad else {
                    return Err(CudaError::InvalidInput {
                        op: "cuda_clip_gradients_in_place",
                        message: "f32 gradient expected",
                    });
                };
                crate::cuda::kernels::sumsq_accum_f32(grad, &sumsq, len)?;
            }
            crate::cuda::kernels::clip_coef_from_sumsq_f32(
                &sumsq,
                &coef,
                max_norm as f32,
                eps as f32,
            )?;
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = param
                    .cuda_grad_get_or_upload_buffer()
                    .map_err(|(_, error)| error)?;
                let CudaBuffer::F32(grad) = &*grad else {
                    return Err(CudaError::InvalidInput {
                        op: "cuda_clip_gradients_in_place",
                        message: "f32 gradient expected",
                    });
                };
                crate::cuda::kernels::scale_inplace_by_scalar_f32(grad, &coef, len)?;
            }
            Ok(())
        }
        Dtype::F64 => {
            let sumsq = alloc_pooled::<f64>(1)?;
            let coef = alloc_pooled::<f64>(1)?;
            crate::cuda::kernels::fill(&sumsq, 0.0)?;
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = param
                    .cuda_grad_get_or_upload_buffer()
                    .map_err(|(_, error)| error)?;
                let CudaBuffer::F64(grad) = &*grad else {
                    return Err(CudaError::InvalidInput {
                        op: "cuda_clip_gradients_in_place",
                        message: "f64 gradient expected",
                    });
                };
                crate::cuda::kernels::sumsq_accum(grad, &sumsq, len)?;
            }
            crate::cuda::kernels::clip_coef_from_sumsq(&sumsq, &coef, max_norm, eps)?;
            for param in params {
                let len = param.cuda_storage_len();
                if len == 0 {
                    continue;
                }
                let grad = param
                    .cuda_grad_get_or_upload_buffer()
                    .map_err(|(_, error)| error)?;
                let CudaBuffer::F64(grad) = &*grad else {
                    return Err(CudaError::InvalidInput {
                        op: "cuda_clip_gradients_in_place",
                        message: "f64 gradient expected",
                    });
                };
                crate::cuda::kernels::scale_inplace_by_scalar(grad, &coef, len)?;
            }
            Ok(())
        }
        _ => unreachable!("dtype validated above"),
    }
}

impl Tensor {
    #[cfg(cuda)]
    pub(crate) fn index_select_cuda(&self, idx: usize) -> Option<Tensor> {
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

        if self.device != Device::Cuda || idx >= self.numel() {
            return None;
        }
        if !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(1).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(1).ok()?),
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

        if self.device != Device::Cuda || !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(len).ok()?),
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
    pub(crate) fn sqrt_cuda(&self) -> Option<Tensor> {
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

        if self.device != Device::Cuda || !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(len).ok()?),
            _ => return None,
        };
        let d_out = Arc::new(d_out);
        let ok = match (&*d_in, &*d_out, self.dtype) {
            (CudaBuffer::F32(input), CudaBuffer::F32(out), Dtype::F32) => {
                crate::cuda::kernels::sqrt_f32(input, out, len).is_ok()
            }
            (CudaBuffer::F64(input), CudaBuffer::F64(out), Dtype::F64) => {
                crate::cuda::kernels::sqrt(input, out, len).is_ok()
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
                                        CudaBuffer::F32(sqrt_out),
                                        CudaBuffer::F32(gt),
                                        CudaBuffer::F32(ig),
                                        Dtype::F32,
                                    ) => crate::cuda::kernels::sqrt_backward_f32(
                                        sqrt_out,
                                        gt,
                                        ig,
                                        sqrt_out.len(),
                                    )
                                    .is_ok(),
                                    (
                                        CudaBuffer::F64(sqrt_out),
                                        CudaBuffer::F64(gt),
                                        CudaBuffer::F64(ig),
                                        Dtype::F64,
                                    ) => crate::cuda::kernels::sqrt_backward(
                                        sqrt_out,
                                        gt,
                                        ig,
                                        sqrt_out.len(),
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
                    let sqrt_cache: Vec<f64> = match &*d_out_for_backward {
                        CudaBuffer::F32(buffer) => {
                            let mut host = vec![0.0f32; buffer.len()];
                            if crate::cuda::memory::copy_d2h(&mut host, buffer).is_err() {
                                return;
                            }
                            host.into_iter().map(|value| value as f64).collect()
                        }
                        CudaBuffer::F64(buffer) => {
                            let mut host = vec![0.0f64; buffer.len()];
                            if crate::cuda::memory::copy_d2h(&mut host, buffer).is_err() {
                                return;
                            }
                            host
                        }
                        CudaBuffer::BF16(_) | CudaBuffer::I8(_) => return,
                    };
                    let mut input_grad = input.grad_write_compat();
                    for index in 0..input_grad.len() {
                        if sqrt_cache[index] > 0.0 {
                            input_grad[index] += grad_out_f64[index] * 0.5 / sqrt_cache[index];
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn sum_cuda(&self, scale: f64) -> Option<Tensor> {
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

        if self.device != Device::Cuda || !matches!(self.dtype, Dtype::F32 | Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(1).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(1).ok()?),
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

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
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(out_len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(out_len).ok()?),
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

        if self.device != Device::Cuda || (self.dtype != Dtype::F32 && self.dtype != Dtype::F64) {
            return None;
        }
        let len = self.numel();
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(len).ok()?),
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

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
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(len).ok()?),
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

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
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(out_len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(out_len).ok()?),
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
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let stride = a_dim + b_dim;

                    // Local buffers avoid holding a_grad and b_grad guards at
                    // once (deadlocks if a_in and b_in alias the same node).
                    let mut a_grad_local = vec![0.0f64; a_in.numel()];
                    let mut b_grad_local = vec![0.0f64; b_in.numel()];
                    a_grad_local
                        .par_chunks_mut(a_dim)
                        .zip(b_grad_local.par_chunks_mut(b_dim))
                        .zip(grad_out_f64.par_chunks(stride))
                        .for_each(|((ag_row, bg_row), g_row)| {
                            for k in 0..a_dim {
                                ag_row[k] += g_row[k];
                            }
                            for k in 0..b_dim {
                                bg_row[k] += g_row[a_dim + k];
                            }
                        });

                    a_in.grad_add_slice(&a_grad_local);
                    b_in.grad_add_slice(&b_grad_local);
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    pub(crate) fn split_last_dim_cuda(&self, parts: usize) -> Option<Vec<Tensor>> {
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

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
                Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(rows * part_dim).ok()?),
                Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(rows * part_dim).ok()?),
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
                        let grad_out_f64 = grad_out.to_f64_vec();
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

        if self.device != Device::Cuda || (self.dtype != Dtype::F32 && self.dtype != Dtype::F64) {
            return None;
        }
        let inner_len = self.numel();
        let out_len = inner_len.checked_mul(batch_size)?;
        let d_in = self.cuda_get_or_upload_buffer().ok()?;
        let d_out = match self.dtype {
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(out_len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(out_len).ok()?),
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

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
            Dtype::F32 => CudaBuffer::F32(alloc_pooled::<f32>(len).ok()?),
            Dtype::F64 => CudaBuffer::F64(alloc_pooled::<f64>(len).ok()?),
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
    pub(super) fn softmax_cuda(&self) -> Tensor {
        use crate::cuda::kernels::{softmax_inplace_auto, softmax_inplace_auto_f32};
        use crate::cuda::memory::{alloc_pooled, copy_d2d, CudaBuffer};

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
            Dtype::F32 => match alloc_pooled::<f32>(len) {
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
            Dtype::F64 => match alloc_pooled::<f64>(len) {
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
        use crate::cuda::memory::{alloc_pooled, copy_d2d, CudaBuffer};

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
            Dtype::F32 => match alloc_pooled::<f32>(len) {
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
            Dtype::F64 => match alloc_pooled::<f64>(len) {
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
        use crate::cuda::memory::{alloc_pooled, copy_d2d, copy_h2d, CudaBuffer};

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
            Dtype::F32 => match alloc_pooled::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(_) => {
                    crate::cuda::record_activation_fallback("alloc");
                    return self.clone();
                }
            },
            Dtype::F64 => match alloc_pooled::<f64>(len) {
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
                let d_cos = match alloc_pooled::<f32>(cos_f32.len()) {
                    Ok(buf) => buf,
                    Err(_) => {
                        crate::cuda::record_activation_fallback("alloc_cos");
                        return self.clone();
                    }
                };
                let d_sin = match alloc_pooled::<f32>(sin_f32.len()) {
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
                let d_cos = match alloc_pooled::<f64>(cos_cache.len()) {
                    Ok(buf) => buf,
                    Err(_) => {
                        crate::cuda::record_activation_fallback("alloc_cos");
                        return self.clone();
                    }
                };
                let d_sin = match alloc_pooled::<f64>(sin_cache.len()) {
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
                let Ok(d_cos) = alloc_pooled::<f32>(cos_f32.len()) else {
                    return self.clone();
                };
                let Ok(d_sin) = alloc_pooled::<f32>(sin_f32.len()) else {
                    return self.clone();
                };
                if copy_h2d(&d_cos, &cos_f32).is_err() || copy_h2d(&d_sin, &sin_f32).is_err() {
                    return self.clone();
                }
                d_cos_for_backward = Arc::new(CudaBuffer::F32(d_cos));
                d_sin_for_backward = Arc::new(CudaBuffer::F32(d_sin));
            }
            Dtype::F64 => {
                let Ok(d_cos) = alloc_pooled::<f64>(cos_cache.len()) else {
                    return self.clone();
                };
                let Ok(d_sin) = alloc_pooled::<f64>(sin_cache.len()) else {
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
    pub(super) fn log_softmax_cuda_last_dim(&self) -> Tensor {
        use crate::cuda::kernels::log_softmax_f32;
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

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
            Dtype::F32 => match alloc_pooled::<f32>(len) {
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
            Dtype::F64 => match alloc_pooled::<f64>(len) {
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
        use crate::cuda::memory::{alloc_pooled, CudaBuffer};

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
            Dtype::F32 => alloc_pooled::<f32>(len).ok().map(CudaBuffer::F32),
            Dtype::F64 => alloc_pooled::<f64>(len).ok().map(CudaBuffer::F64),
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
                                                    gpu_backward_ok =
                                                        bk(gt, a_buf, b_buf, ag, bg, len).is_ok();
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
                                                    gpu_backward_ok =
                                                        bk(gt, a_buf, b_buf, ag, bg, len).is_ok();
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

                    let grad_out_f64 = grad_out.to_f64_vec();

                    // Local buffers avoid holding a_grad and b_grad guards at
                    // once (deadlocks if a and b alias, e.g. `x * x`).
                    let mut a_grad = vec![0.0f64; grad_out_f64.len()];
                    let mut b_grad = vec![0.0f64; grad_out_f64.len()];
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

                    a.grad_add_slice(&a_grad);
                    b.grad_add_slice(&b_grad);
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
