use crate::autograd::Tensor;
use crate::dtype::{Dtype, Storage};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type CudaTensorBufferMap = HashMap<usize, Arc<crate::cuda::memory::CudaBuffer>>;
static CUDA_TENSOR_BUFFER_CACHE: OnceLock<Mutex<CudaTensorBufferMap>> = OnceLock::new();
static CUDA_GRAD_BUFFER_CACHE: OnceLock<Mutex<CudaTensorBufferMap>> = OnceLock::new();

pub(super) fn cuda_tensor_buffer_cache() -> &'static Mutex<CudaTensorBufferMap> {
    CUDA_TENSOR_BUFFER_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn cuda_grad_buffer_cache() -> &'static Mutex<CudaTensorBufferMap> {
    CUDA_GRAD_BUFFER_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl Drop for Tensor {
    fn drop(&mut self) {
        let is_last = match &self.data {
            Storage::F64(v) => Arc::strong_count(v) == 1,
            Storage::F32(v) => Arc::strong_count(v) == 1,
            Storage::BF16(v) => Arc::strong_count(v) == 1,
            Storage::I8(v) => Arc::strong_count(v) == 1,
        };
        if is_last {
            let key = match &self.data {
                Storage::F64(v) => Arc::as_ptr(v) as usize,
                Storage::F32(v) => Arc::as_ptr(v) as usize,
                Storage::BF16(v) => Arc::as_ptr(v) as usize,
                Storage::I8(v) => Arc::as_ptr(v) as usize,
            };
            if let Some(cache) = CUDA_TENSOR_BUFFER_CACHE.get() {
                if let Ok(mut map) = cache.lock() {
                    map.remove(&key);
                }
            }
        }

        let grad_is_last = match &self.grad {
            Storage::F64(v) => Arc::strong_count(v) == 1,
            Storage::F32(v) => Arc::strong_count(v) == 1,
            Storage::BF16(v) => Arc::strong_count(v) == 1,
            Storage::I8(v) => Arc::strong_count(v) == 1,
        };
        if grad_is_last {
            let key = match &self.grad {
                Storage::F64(v) => Arc::as_ptr(v) as usize,
                Storage::F32(v) => Arc::as_ptr(v) as usize,
                Storage::BF16(v) => Arc::as_ptr(v) as usize,
                Storage::I8(v) => Arc::as_ptr(v) as usize,
            };
            if let Some(cache) = CUDA_GRAD_BUFFER_CACHE.get() {
                if let Ok(mut map) = cache.lock() {
                    map.remove(&key);
                }
            }
        }
    }
}

pub(crate) fn cuda_grad_out_buffer(
    grad_out: &Storage,
) -> Option<Arc<crate::cuda::memory::CudaBuffer>> {
    use crate::cuda::memory::{alloc_pooled, copy_h2d, CudaBuffer};

    let len = grad_out.len();
    if let Ok(map) = cuda_grad_buffer_cache().lock() {
        if let Some(buffer) = map.get(&grad_out.id()) {
            if buffer.len() == len && buffer.dtype() == grad_out.dtype() {
                return Some(buffer.clone());
            }
        }
    }
    if len == 0 {
        let buffer = match grad_out.dtype() {
            Dtype::F32 => CudaBuffer::F32(crate::cuda::memory::DevicePtr::zero_sized()),
            Dtype::F64 => CudaBuffer::F64(crate::cuda::memory::DevicePtr::zero_sized()),
            Dtype::BF16 => CudaBuffer::BF16(crate::cuda::memory::DevicePtr::zero_sized()),
            Dtype::I8 => CudaBuffer::I8(crate::cuda::memory::DevicePtr::zero_sized()),
        };
        return Some(Arc::new(buffer));
    }
    match grad_out.dtype() {
        Dtype::F32 => {
            let buffer = alloc_pooled::<f32>(len).ok()?;
            copy_h2d(&buffer, &grad_out.to_f32_vec()).ok()?;
            Some(Arc::new(CudaBuffer::F32(buffer)))
        }
        Dtype::F64 => {
            let buffer = alloc_pooled::<f64>(len).ok()?;
            copy_h2d(&buffer, &grad_out.to_f64_vec()).ok()?;
            Some(Arc::new(CudaBuffer::F64(buffer)))
        }
        _ => None,
    }
}

pub(super) fn cuda_sync_grad_to_host(tensor: &Tensor) -> bool {
    use crate::cuda::memory::{copy_d2h, CudaBuffer};

    let Some(buffer) = tensor.cuda_grad_cached_buffer() else {
        return false;
    };
    match (&tensor.grad, &*buffer) {
        (Storage::F32(storage), CudaBuffer::F32(device)) => {
            let mut host = vec![0.0f32; device.len()];
            if copy_d2h(&mut host, device).is_err() {
                return false;
            }
            *storage.write().unwrap() = host;
            true
        }
        (Storage::F64(storage), CudaBuffer::F64(device)) => {
            let mut host = vec![0.0f64; device.len()];
            if copy_d2h(&mut host, device).is_err() {
                return false;
            }
            *storage.write().unwrap() = host;
            true
        }
        _ => false,
    }
}
