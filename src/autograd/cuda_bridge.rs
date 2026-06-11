use crate::autograd::{Device, Tensor};
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

#[allow(dead_code)]
impl Tensor {
    fn cuda_cache_key(&self) -> usize {
        match &self.data {
            Storage::F64(v) => Arc::as_ptr(v) as usize,
            Storage::F32(v) => Arc::as_ptr(v) as usize,
            Storage::BF16(v) => Arc::as_ptr(v) as usize,
            Storage::I8(v) => Arc::as_ptr(v) as usize,
        }
    }

    pub(crate) fn cuda_cached_buffer(&self) -> Option<Arc<crate::cuda::memory::CudaBuffer>> {
        let key = self.cuda_cache_key();
        let cache = cuda_tensor_buffer_cache();
        let map = cache.lock().ok()?;
        map.get(&key).cloned()
    }

    pub(crate) fn cuda_set_cached_buffer(&self, buffer: Arc<crate::cuda::memory::CudaBuffer>) {
        let key = self.cuda_cache_key();
        if let Ok(mut map) = cuda_tensor_buffer_cache().lock() {
            map.insert(key, buffer);
        }
    }

    pub(super) fn cuda_remove_cached_buffer(&self) {
        let key = self.cuda_cache_key();
        if let Ok(mut map) = cuda_tensor_buffer_cache().lock() {
            map.remove(&key);
        }
    }

    pub(crate) fn cuda_clear_host_data_preserve_cache(&self) {
        match &self.data {
            Storage::F64(v) => v.write().unwrap().clear(),
            Storage::F32(v) => v.write().unwrap().clear(),
            Storage::BF16(v) => v.write().unwrap().clear(),
            Storage::I8(v) => v.write().unwrap().clear(),
        }
    }

    pub(crate) fn cuda_lerp_in_place_from(&self, source: &Tensor, tau: f64) -> bool {
        use crate::cuda::memory::CudaBuffer;

        if self.device != Device::Cuda
            || source.device != Device::Cuda
            || self.dtype != source.dtype
            || self.numel() != source.numel()
            || !matches!(self.dtype, Dtype::F32 | Dtype::F64)
        {
            return false;
        }
        let len = self.numel();
        let target = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err(_) => return false,
        };
        let source = match source.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err(_) => return false,
        };
        let ok = match (&*target, &*source, self.dtype) {
            (CudaBuffer::F32(target), CudaBuffer::F32(source), Dtype::F32) => {
                crate::cuda::kernels::lerp_inplace_f32(target, source, tau as f32, len).is_ok()
            }
            (CudaBuffer::F64(target), CudaBuffer::F64(source), Dtype::F64) => {
                crate::cuda::kernels::lerp_inplace(target, source, tau, len).is_ok()
            }
            _ => false,
        };
        if ok {
            self.cuda_clear_host_data_preserve_cache();
        }
        ok
    }

    // --- GPU gradient buffer cache (keyed by grad Arc pointer) ---

    fn cuda_grad_cache_key(&self) -> usize {
        match &self.grad {
            Storage::F64(v) => Arc::as_ptr(v) as usize,
            Storage::F32(v) => Arc::as_ptr(v) as usize,
            Storage::BF16(v) => Arc::as_ptr(v) as usize,
            Storage::I8(v) => Arc::as_ptr(v) as usize,
        }
    }

    pub(super) fn cuda_grad_cached_buffer(&self) -> Option<Arc<crate::cuda::memory::CudaBuffer>> {
        let key = self.cuda_grad_cache_key();
        let cache = cuda_grad_buffer_cache();
        let map = cache.lock().ok()?;
        map.get(&key).cloned()
    }

    fn cuda_grad_set_cached_buffer(&self, buffer: Arc<crate::cuda::memory::CudaBuffer>) {
        let key = self.cuda_grad_cache_key();
        if let Ok(mut map) = cuda_grad_buffer_cache().lock() {
            map.insert(key, buffer);
        }
    }

    pub(super) fn cuda_grad_remove_cached_buffer(&self) {
        let key = self.cuda_grad_cache_key();
        if let Ok(mut map) = cuda_grad_buffer_cache().lock() {
            map.remove(&key);
        }
    }

    /// Upload CPU grad to GPU, returning the GPU buffer.
    pub(crate) fn cuda_grad_get_or_upload_buffer(
        &self,
    ) -> Result<Arc<crate::cuda::memory::CudaBuffer>, (&'static str, crate::cuda::error::CudaError)>
    {
        use crate::cuda::memory::{alloc_pooled, copy_h2d, CudaBuffer};

        let grad_dtype = self.grad.dtype();
        let len = self.grad.len();
        if let Some(buffer) = self.cuda_grad_cached_buffer() {
            if buffer.len() == len && buffer.dtype() == grad_dtype {
                return Ok(buffer);
            }
            self.cuda_grad_remove_cached_buffer();
        }

        if len == 0 {
            let buffer = match self.dtype {
                Dtype::BF16 => CudaBuffer::BF16(crate::cuda::memory::DevicePtr::zero_sized()),
                Dtype::I8 => CudaBuffer::I8(crate::cuda::memory::DevicePtr::zero_sized()),
                Dtype::F32 => CudaBuffer::F32(crate::cuda::memory::DevicePtr::zero_sized()),
                Dtype::F64 => CudaBuffer::F64(crate::cuda::memory::DevicePtr::zero_sized()),
            };
            return Ok(Arc::new(buffer));
        }

        let buffer = match grad_dtype {
            Dtype::F32 => {
                let host = self.grad_to_f32_vec();
                let device = match alloc_pooled::<f32>(len) {
                    Ok(buf) => buf,
                    Err(err) => return Err(("alloc", err)),
                };
                if let Err(err) = copy_h2d(&device, &host) {
                    return Err(("copy", err));
                }
                CudaBuffer::F32(device)
            }
            Dtype::F64 => {
                let host = self.grad_to_f64_vec();
                let device = match alloc_pooled::<f64>(len) {
                    Ok(buf) => buf,
                    Err(err) => return Err(("alloc", err)),
                };
                if let Err(err) = copy_h2d(&device, &host) {
                    return Err(("copy", err));
                }
                CudaBuffer::F64(device)
            }
            _ => {
                return Err((
                    "alloc",
                    crate::cuda::error::CudaError::UnsupportedBuild {
                        op: "cuda grad upload for non-float dtype",
                    },
                ))
            }
        };

        let buffer = Arc::new(buffer);
        self.cuda_grad_set_cached_buffer(buffer.clone());
        Ok(buffer)
    }

    /// Ensure a zero-initialized GPU grad buffer exists for this tensor.
    pub(crate) fn cuda_grad_ensure_buffer(&self) -> Option<Arc<crate::cuda::memory::CudaBuffer>> {
        self.cuda_grad_get_or_upload_buffer().ok()
    }

    pub(crate) fn cuda_grad_zero_buffer(&self) -> Option<Arc<crate::cuda::memory::CudaBuffer>> {
        let buffer = self.cuda_grad_ensure_buffer()?;
        let len = buffer.len();
        if len == 0 {
            return Some(buffer);
        }
        match &*buffer {
            crate::cuda::memory::CudaBuffer::F32(b) => {
                let _ = crate::cuda::kernels::fill_f32(b, 0.0);
            }
            crate::cuda::memory::CudaBuffer::F64(b) => {
                let _ = crate::cuda::kernels::fill(b, 0.0);
            }
            crate::cuda::memory::CudaBuffer::BF16(_) | crate::cuda::memory::CudaBuffer::I8(_) => {}
        }
        Some(buffer)
    }

    /// Materialize GPU data to CPU if this tensor lives on GPU but has empty CPU data.
    pub(super) fn cuda_materialize(&self) {
        use crate::cuda::memory::{copy_d2h, CudaBuffer};

        if self.device != Device::Cuda {
            return;
        }
        if !self.data.is_empty() {
            return;
        }

        if let Some(buffer) = self.cuda_cached_buffer() {
            match &*buffer {
                CudaBuffer::BF16(b) => {
                    let mut data = vec![crate::dtype::bf16::default(); b.len()];
                    if let Err(err) = copy_d2h(&mut data, b) {
                        log::warn!(
                            "[Tensor] CUDA materialize D2H failed ({}), data remains empty",
                            err
                        );
                        return;
                    }
                    if let Storage::BF16(v) = &self.data {
                        *v.write().unwrap() = data;
                    }
                }
                CudaBuffer::I8(b) => {
                    let mut data = vec![0i8; b.len()];
                    if let Err(err) = copy_d2h(&mut data, b) {
                        log::warn!(
                            "[Tensor] CUDA materialize D2H failed ({}), data remains empty",
                            err
                        );
                        return;
                    }
                    if let Storage::I8(v) = &self.data {
                        *v.write().unwrap() = data;
                    }
                }
                CudaBuffer::F32(b) => {
                    let mut data = vec![0.0f32; b.len()];
                    if let Err(err) = copy_d2h(&mut data, b) {
                        log::warn!(
                            "[Tensor] CUDA materialize D2H failed ({}), data remains empty",
                            err
                        );
                        return;
                    }
                    if let Storage::F32(v) = &self.data {
                        *v.write().unwrap() = data;
                    }
                }
                CudaBuffer::F64(b) => {
                    let mut data = vec![0.0f64; b.len()];
                    if let Err(err) = copy_d2h(&mut data, b) {
                        log::warn!(
                            "[Tensor] CUDA materialize D2H failed ({}), data remains empty",
                            err
                        );
                        return;
                    }
                    if let Storage::F64(v) = &self.data {
                        *v.write().unwrap() = data;
                    }
                }
            }
        }
    }

    pub(crate) fn cuda_get_or_upload_buffer(
        &self,
    ) -> Result<Arc<crate::cuda::memory::CudaBuffer>, (&'static str, crate::cuda::error::CudaError)>
    {
        use crate::cuda::memory::{alloc_pooled, copy_h2d, CudaBuffer};

        let host_len = self.data.len();
        let len = if host_len > 0 { host_len } else { self.numel() };
        let expected_len = self.numel();
        if len != expected_len {
            return Err((
                "alloc",
                crate::cuda::error::CudaError::SizeMismatch {
                    op: "cuda upload length",
                    expected: expected_len,
                    actual: len,
                },
            ));
        }

        if let Some(buffer) = self.cuda_cached_buffer() {
            if buffer.len() == len && buffer.dtype() == self.dtype {
                return Ok(buffer);
            }
            self.cuda_remove_cached_buffer();
        }

        if len == 0 {
            let buffer = match self.dtype {
                Dtype::BF16 => CudaBuffer::BF16(crate::cuda::memory::DevicePtr::zero_sized()),
                Dtype::I8 => CudaBuffer::I8(crate::cuda::memory::DevicePtr::zero_sized()),
                Dtype::F32 => CudaBuffer::F32(crate::cuda::memory::DevicePtr::zero_sized()),
                Dtype::F64 => CudaBuffer::F64(crate::cuda::memory::DevicePtr::zero_sized()),
            };
            return Ok(Arc::new(buffer));
        }
        if host_len == 0 {
            return Err((
                "copy",
                crate::cuda::error::CudaError::InvalidInput {
                    op: "cuda upload",
                    message: "host data is empty and no CUDA buffer is cached",
                },
            ));
        }

        let buffer = match self.dtype {
            Dtype::BF16 => {
                let host = self.data_bf16();
                let device = match alloc_pooled::<crate::dtype::bf16>(len) {
                    Ok(buf) => buf,
                    Err(err) => return Err(("alloc", err)),
                };
                if let Err(err) = copy_h2d(&device, &host) {
                    return Err(("copy", err));
                }
                CudaBuffer::BF16(device)
            }
            Dtype::I8 => {
                let device = match alloc_pooled::<i8>(len) {
                    Ok(buf) => buf,
                    Err(err) => return Err(("alloc", err)),
                };
                if let Storage::I8(v) = &self.data {
                    let host = v.read().unwrap();
                    if let Err(err) = copy_h2d(&device, &host) {
                        return Err(("copy", err));
                    }
                }
                CudaBuffer::I8(device)
            }
            Dtype::F32 => {
                let host = self.data_f32();
                let device = match alloc_pooled::<f32>(len) {
                    Ok(buf) => buf,
                    Err(err) => return Err(("alloc", err)),
                };
                if let Err(err) = copy_h2d(&device, &host) {
                    return Err(("copy", err));
                }
                CudaBuffer::F32(device)
            }
            Dtype::F64 => {
                let host = self.data_f64();
                let device = match alloc_pooled::<f64>(len) {
                    Ok(buf) => buf,
                    Err(err) => return Err(("alloc", err)),
                };
                if let Err(err) = copy_h2d(&device, &host) {
                    return Err(("copy", err));
                }
                CudaBuffer::F64(device)
            }
        };

        let buffer = Arc::new(buffer);
        self.cuda_set_cached_buffer(buffer.clone());
        Ok(buffer)
    }
}
