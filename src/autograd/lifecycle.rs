use crate::autograd::Tensor;

#[cfg(cuda)]
use crate::autograd::{BackwardOp, Context, Device};
#[cfg(cuda)]
use crate::dtype::{Dtype, Storage};
#[cfg(cuda)]
use std::sync::Arc;

impl Tensor {
    pub fn backward(&self) {
        // Topological sort — iterative DFS post-order to avoid stack overflow
        // on deep computation graphs (e.g. multi-layer transformers).
        let mut visited = std::collections::HashSet::new();
        let mut topo = Vec::new();
        let mut stack = vec![(self, false)];
        while let Some((t, post)) = stack.pop() {
            if post {
                topo.push(t.clone());
            } else {
                let id = t.grad.id();
                if visited.insert(id) {
                    // Push self back as "post" so it is emitted after all parents.
                    stack.push((t, true));
                    if let Some(ctx) = &t._ctx {
                        for parent in &ctx.parents {
                            stack.push((parent, false));
                        }
                    }
                }
            }
        }

        // Seed gradient of this tensor to 1.0
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            self.cuda_grad_remove_cached_buffer();
        }
        self.grad.fill_f64(1.0);
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            if let Some(d_grad) = self.cuda_grad_zero_buffer() {
                if d_grad.len() > 0 {
                    match &*d_grad {
                        crate::cuda::memory::CudaBuffer::BF16(_) => {}
                        crate::cuda::memory::CudaBuffer::I8(_) => {}
                        crate::cuda::memory::CudaBuffer::F32(buf) => {
                            let ones = vec![1.0f32; d_grad.len()];
                            let _ = crate::cuda::memory::copy_h2d(buf, &ones);
                        }
                        crate::cuda::memory::CudaBuffer::F64(buf) => {
                            let ones = vec![1.0f64; d_grad.len()];
                            let _ = crate::cuda::memory::copy_h2d(buf, &ones);
                        }
                    }
                }
            }
        }

        // Backprop
        for t in topo.iter().rev() {
            if let Some(ctx) = &t._ctx {
                // GPU-aware backward ops read from GPU buffers directly;
                // we no longer force materialization of all parents to CPU.
                (ctx.backward_op)(&t.grad, &ctx.parents);
            }
        }
    }

    // Explicitly clear the graph history to free memory
    pub fn clear_graph(&mut self) {
        self._ctx = None;
    }

    pub fn zero_grad(&self) {
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            self.cuda_grad_remove_cached_buffer();
        }
        self.grad.zero();
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            if let Some(d_grad) = self.cuda_grad_zero_buffer() {
                let _ = d_grad;
            }
        }
    }

    /// Copy tensor data to CUDA GPU and keep host data lazy until materialized.
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub fn to_cuda(&self) -> crate::cuda::error::CudaResult<Tensor> {
        if let Err(err) = crate::cuda::init() {
            log::error!("[Tensor] CUDA runtime unavailable: {err}");
            return Err(err);
        }

        let tensor = Tensor {
            data: Storage::zeros(0, self.dtype),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: self.dtype,
            _ctx: self._ctx.clone(),
        };

        let len = self.numel();
        if len > 0 {
            if let Err((_, err)) = self.cuda_get_or_upload_buffer() {
                log::warn!("[Tensor] CUDA upload failed: {err}");
                return Err(err);
            }
            if let Some(buffer) = self.cuda_cached_buffer() {
                tensor.cuda_set_cached_buffer(buffer);
            }
        }

        Ok(tensor)
    }

    #[cfg(cuda)]
    #[inline]
    pub(crate) fn cuda_storage_len(&self) -> usize {
        let host_len = self.data.len();
        if host_len > 0 {
            host_len
        } else {
            self.numel()
        }
    }

    #[cfg(cuda)]
    pub(crate) fn cuda_device_tensor(
        data: Arc<crate::cuda::memory::CudaBuffer>,
        shape: Vec<usize>,
        dtype: Dtype,
        parents: Vec<Tensor>,
        backward_op: BackwardOp,
    ) -> Tensor {
        let len: usize = shape.iter().product();
        let out = Tensor {
            data: Tensor::empty_storage(dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
            shape,
            device: Device::Cuda,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op,
            })),
        };
        out.cuda_set_cached_buffer(data);
        out
    }

    /// Copy tensor data from CUDA GPU back to CPU.
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub fn from_cuda(&self) -> crate::cuda::error::CudaResult<Vec<f64>> {
        use crate::cuda::memory::CudaBuffer;

        if self.device != Device::Cuda {
            return Err(crate::cuda::error::CudaError::InvalidInput {
                op: "Tensor::from_cuda",
                message: "tensor is not on CUDA device",
            });
        }

        if let Some(buffer) = self.cuda_cached_buffer() {
            match &*buffer {
                CudaBuffer::BF16(b) => {
                    let mut host = vec![crate::dtype::bf16::default(); buffer.len()];
                    crate::cuda::memory::copy_d2h(&mut host, b)?;
                    return Ok(host.iter().map(|&v| v.to_f64()).collect());
                }
                CudaBuffer::I8(b) => {
                    let mut host = vec![0i8; buffer.len()];
                    crate::cuda::memory::copy_d2h(&mut host, b)?;
                    return Ok(host.iter().map(|&v| v as f64).collect());
                }
                CudaBuffer::F32(b) => {
                    let mut host = vec![0.0f32; buffer.len()];
                    crate::cuda::memory::copy_d2h(&mut host, b)?;
                    return Ok(host.iter().map(|&v| v as f64).collect());
                }
                CudaBuffer::F64(b) => {
                    let mut host = vec![0.0f64; buffer.len()];
                    crate::cuda::memory::copy_d2h(&mut host, b)?;
                    return Ok(host);
                }
            }
        }

        Ok(self.data_f64().clone())
    }
}
