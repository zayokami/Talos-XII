use crate::dtype::{Dtype, Storage};
use crate::simd::{
    add_scaled_row, dot_product, horizontal_sum, prefetch_read_l1, softmax_exp_sum, vector_fma,
    vector_gelu, vector_relu,
};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::sync::{Arc, RwLock};
#[cfg(cuda)]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

// --- Autograd Engine ---

mod core;
mod guards;
mod operators;
mod serde_impl;

#[cfg(cuda)]
pub(crate) use core::BackwardOp;
pub use core::{Context, Device, GradWriteCompat, Tensor};
pub use guards::TensorReadGuard;

// Minimum element count to justify Rayon parallel dispatch.
// Below this, serial iteration is faster due to scheduling overhead.
pub(crate) const PAR_THRESHOLD: usize = 4096;

#[cfg(cuda)]
type CudaTensorBufferMap = HashMap<usize, Arc<crate::cuda::memory::CudaBuffer>>;
#[cfg(cuda)]
static CUDA_TENSOR_BUFFER_CACHE: OnceLock<Mutex<CudaTensorBufferMap>> = OnceLock::new();
#[cfg(cuda)]
static CUDA_GRAD_BUFFER_CACHE: OnceLock<Mutex<CudaTensorBufferMap>> = OnceLock::new();

#[cfg(cuda)]
fn cuda_tensor_buffer_cache() -> &'static Mutex<CudaTensorBufferMap> {
    CUDA_TENSOR_BUFFER_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(cuda)]
fn cuda_grad_buffer_cache() -> &'static Mutex<CudaTensorBufferMap> {
    CUDA_GRAD_BUFFER_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(cuda)]
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

#[cfg(cuda)]
#[derive(Clone, Copy)]
enum CudaBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[cfg(cuda)]
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

#[cfg(cuda)]
fn cuda_sync_grad_to_host(tensor: &Tensor) -> bool {
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
    #[inline]
    #[cfg_attr(not(cuda), allow(dead_code))]
    pub(crate) fn empty_storage(dtype: Dtype) -> Storage {
        Storage::zeros(0, dtype)
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn from_mmap(path: &str, shape: Vec<usize>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let bytes = &mmap[..];
        let elem_size = std::mem::size_of::<f64>();
        if bytes.len() % elem_size != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Mmap size is not a multiple of f64 element size",
            ));
        }

        let expected_len: usize = shape.iter().product();
        let expected_bytes = expected_len.checked_mul(elem_size).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Mmap shape size overflow")
        })?;
        if bytes.len() != expected_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Mmap size mismatch",
            ));
        }

        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f64>()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Mmap pointer is not aligned for f64 data",
            ));
        }

        // Zero-copy cast (unsafe but fast)
        let slice =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, expected_len) };
        let data = slice.to_vec(); // Copy to Vec (mmap loading is still faster than JSON parsing!)

        Ok(Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(expected_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: None,
        })
    }

    pub fn save_binary(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = File::create(path)?;
        let data = self.data_as_f64_vec();
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
        file.write_all(bytes)?;
        Ok(())
    }

    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::with_dtype(data, shape, Dtype::F64)
    }

    /// Create an F32 tensor from f64 data (auto-converts to f32).
    pub fn new_f32(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::with_dtype(data, shape, Dtype::F32)
    }

    /// Create a BF16 tensor from f64 data (auto-converts through f32).
    pub fn new_bf16(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::with_dtype(data, shape, Dtype::BF16)
    }

    /// Create a tensor with explicit dtype.
    pub fn with_dtype(data: Vec<f64>, shape: Vec<usize>, dtype: Dtype) -> Self {
        let len = data.len();
        assert_eq!(
            len,
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );
        match dtype {
            Dtype::F64 => Tensor {
                data: Storage::F64(Arc::new(RwLock::new(data))),
                grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
                shape,
                device: Device::Cpu,
                dtype,
                _ctx: None,
            },
            // Other dtypes: convert data and store in appropriate storage
            Dtype::F32 => {
                let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Tensor {
                    data: Storage::F32(Arc::new(RwLock::new(f32_data))),
                    grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
                    shape,
                    device: Device::Cpu,
                    dtype,
                    _ctx: None,
                }
            }
            Dtype::BF16 => {
                let bf16_data: Vec<crate::dtype::bf16> = data
                    .iter()
                    .map(|&v| crate::dtype::bf16::from_f64(v))
                    .collect();
                Tensor {
                    data: Storage::BF16(Arc::new(RwLock::new(bf16_data))),
                    grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
                    shape,
                    device: Device::Cpu,
                    dtype,
                    _ctx: None,
                }
            }
            Dtype::I8 => {
                let i8_data: Vec<i8> = data.iter().map(|&v| v as i8).collect();
                Tensor {
                    data: Storage::I8(Arc::new(RwLock::new(i8_data))),
                    grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
                    shape,
                    device: Device::Cpu,
                    dtype,
                    _ctx: None,
                }
            }
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>();
        Tensor::new(vec![0.0; len], shape)
    }

    /// Zeros with F32 dtype.
    pub fn zeros_f32(shape: Vec<usize>) -> Self {
        Self::zeros_with_dtype(shape, Dtype::F32)
    }

    /// Zeros with BF16 dtype.
    pub fn zeros_bf16(shape: Vec<usize>) -> Self {
        Self::zeros_with_dtype(shape, Dtype::BF16)
    }

    /// Zeros with explicit dtype.
    pub fn zeros_with_dtype(shape: Vec<usize>, dtype: Dtype) -> Self {
        let len = shape.iter().product::<usize>();
        match dtype {
            Dtype::F64 => Tensor::new(vec![0.0; len], shape),
            Dtype::F32 => Tensor {
                data: Storage::F32(Arc::new(RwLock::new(vec![0.0f32; len]))),
                grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
                shape,
                device: Device::Cpu,
                dtype,
                _ctx: None,
            },
            Dtype::BF16 => Tensor {
                data: Storage::BF16(Arc::new(RwLock::new(vec![
                    crate::dtype::bf16::default();
                    len
                ]))),
                grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
                shape,
                device: Device::Cpu,
                dtype,
                _ctx: None,
            },
            Dtype::I8 => Tensor {
                data: Storage::I8(Arc::new(RwLock::new(vec![0i8; len]))),
                grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
                shape,
                device: Device::Cpu,
                dtype,
                _ctx: None,
            },
        }
    }

    /// Fill tensor with a scalar value in-place.
    pub fn fill_(&mut self, value: f64) -> &mut Self {
        self.data.fill_f64(value);
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            let keep_cache = if let Some(buffer) = self.cuda_cached_buffer() {
                match &*buffer {
                    crate::cuda::memory::CudaBuffer::F32(buf) => {
                        crate::cuda::kernels::fill_f32(buf, value as f32).is_ok()
                    }
                    crate::cuda::memory::CudaBuffer::F64(buf) => {
                        crate::cuda::kernels::fill(buf, value).is_ok()
                    }
                    crate::cuda::memory::CudaBuffer::BF16(_)
                    | crate::cuda::memory::CudaBuffer::I8(_) => false,
                }
            } else {
                true
            };
            if !keep_cache {
                self.cuda_remove_cached_buffer();
            }
        }
        self
    }

    // -------------------------------------------------------------------------
    // Dtype-aware lock access helpers
    // -------------------------------------------------------------------------

    /// Read F64 data lock. Panics if not F64 or poisoned.
    #[inline]
    pub fn data_f64(&self) -> std::sync::RwLockReadGuard<'_, Vec<f64>> {
        #[cfg(cuda)]
        self.cuda_materialize();
        match &self.data {
            Storage::F64(v) => v.read().unwrap(),
            _ => panic!("Expected F64 storage, got {:?}", self.dtype),
        }
    }

    /// Read F32 data lock. Panics if not F32 or poisoned.
    #[inline]
    pub fn data_f32(&self) -> std::sync::RwLockReadGuard<'_, Vec<f32>> {
        #[cfg(cuda)]
        self.cuda_materialize();
        match &self.data {
            Storage::F32(v) => v.read().unwrap(),
            _ => panic!("Expected F32 storage, got {:?}", self.dtype),
        }
    }

    /// Write F64 data lock. Panics if not F64 or poisoned.
    #[inline]
    pub fn data_write_f64(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f64>> {
        #[cfg(cuda)]
        self.cuda_materialize();
        #[cfg(cuda)]
        self.cuda_remove_cached_buffer();
        match &self.data {
            Storage::F64(v) => v.write().unwrap(),
            _ => panic!("Expected F64 storage, got {:?}", self.dtype),
        }
    }

    /// Write F32 data lock. Panics if not F32 or poisoned.
    #[inline]
    pub fn data_write_f32(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f32>> {
        #[cfg(cuda)]
        self.cuda_materialize();
        #[cfg(cuda)]
        self.cuda_remove_cached_buffer();
        match &self.data {
            Storage::F32(v) => v.write().unwrap(),
            _ => panic!("Expected F32 storage, got {:?}", self.dtype),
        }
    }

    /// Read BF16 data lock. Panics if not BF16 or poisoned.
    #[inline]
    pub fn data_bf16(&self) -> std::sync::RwLockReadGuard<'_, Vec<crate::dtype::bf16>> {
        #[cfg(cuda)]
        self.cuda_materialize();
        match &self.data {
            Storage::BF16(v) => v.read().unwrap(),
            _ => panic!("Expected BF16 storage, got {:?}", self.dtype),
        }
    }

    /// Write BF16 data lock. Panics if not BF16 or poisoned.
    #[inline]
    pub fn data_write_bf16(&self) -> std::sync::RwLockWriteGuard<'_, Vec<crate::dtype::bf16>> {
        #[cfg(cuda)]
        self.cuda_materialize();
        #[cfg(cuda)]
        self.cuda_remove_cached_buffer();
        match &self.data {
            Storage::BF16(v) => v.write().unwrap(),
            _ => panic!("Expected BF16 storage, got {:?}", self.dtype),
        }
    }

    /// Return a detached copy of this tensor converted to `dtype`.
    #[inline]
    pub fn to_dtype(&self, dtype: Dtype) -> Self {
        Tensor::with_dtype(self.data_as_f64_vec(), self.shape.clone(), dtype)
    }

    /// Return a detached BF16 copy of this tensor for inference/storage.
    #[inline]
    pub fn to_bf16(&self) -> Self {
        self.to_dtype(Dtype::BF16)
    }

    /// Read F64 data lock, returning `None` if poisoned.
    #[inline]
    pub fn data_read_safe(&self) -> Option<std::sync::RwLockReadGuard<'_, Vec<f64>>> {
        match &self.data {
            Storage::F64(v) => v.read().ok(),
            _ => None,
        }
    }

    /// Read F64 data lock, resilient to poisoning.
    #[inline]
    pub fn data_f64_poison_resilient(&self) -> std::sync::RwLockReadGuard<'_, Vec<f64>> {
        match &self.data {
            Storage::F64(v) => v.read().unwrap_or_else(|e| e.into_inner()),
            _ => panic!("Expected F64 storage, got {:?}", self.dtype),
        }
    }

    /// Read the data lock. Panics if poisoned (use in hot paths where the lock
    /// can never be poisoned under normal operation).
    #[inline]
    pub fn data_read(&self) -> std::sync::RwLockReadGuard<'_, Vec<f64>> {
        self.data_f64()
    }

    /// Write the data lock. Panics if poisoned.
    #[inline]
    pub fn data_write(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f64>> {
        self.data_write_f64()
    }

    /// Read tensor data as Vec<f64>. For F64 storage this clones; for other
    /// dtypes it converts element-wise.
    #[inline]
    pub fn data_as_f64_vec(&self) -> Vec<f64> {
        #[cfg(cuda)]
        self.cuda_materialize();
        self.data.to_f64_vec()
    }

    /// Read data as Vec<f32>, converting from native dtype as needed.
    #[inline]
    pub fn data_to_f32_vec(&self) -> Vec<f32> {
        #[cfg(cuda)]
        self.cuda_materialize();
        self.data.to_f32_vec()
    }

    /// Output dtype for a binary op.
    ///
    /// F64 is preserved only when explicitly present. Mixed F32/BF16 compute
    /// promotes to F32 so model paths do not silently fall back to F64.
    #[inline]
    pub fn binary_dtype(a: Dtype, b: Dtype) -> Dtype {
        if a == b {
            a
        } else if a == Dtype::F64 || b == Dtype::F64 {
            Dtype::F64
        } else {
            Dtype::F32
        }
    }

    /// Determine grad dtype from data dtype.
    /// F64 data -> F64 grad (backward compat).
    /// All other dtypes -> F32 grad (native compute precision).
    #[inline]
    pub fn grad_dtype_for(data_dtype: Dtype) -> Dtype {
        match data_dtype {
            Dtype::F64 => Dtype::F64,
            _ => Dtype::F32,
        }
    }

    // --- Grad access helpers (typed) ---

    /// Read F32 grad lock. Panics if grad is not F32.
    #[inline]
    pub fn grad_read_f32(&self) -> std::sync::RwLockReadGuard<'_, Vec<f32>> {
        #[cfg(cuda)]
        let _ = cuda_sync_grad_to_host(self);
        match &self.grad {
            Storage::F32(v) => v.read().unwrap(),
            _ => panic!("Expected F32 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Write F32 grad lock. Panics if grad is not F32.
    #[inline]
    pub fn grad_write_f32(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f32>> {
        #[cfg(cuda)]
        {
            let _ = cuda_sync_grad_to_host(self);
            self.cuda_grad_remove_cached_buffer();
        }
        match &self.grad {
            Storage::F32(v) => v.write().unwrap(),
            _ => panic!("Expected F32 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Read F64 grad lock. Panics if grad is not F64.
    #[inline]
    pub fn grad_read_f64(&self) -> std::sync::RwLockReadGuard<'_, Vec<f64>> {
        #[cfg(cuda)]
        let _ = cuda_sync_grad_to_host(self);
        match &self.grad {
            Storage::F64(v) => v.read().unwrap(),
            _ => panic!("Expected F64 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Write F64 grad lock. Panics if grad is not F64.
    #[inline]
    pub fn grad_write_f64(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f64>> {
        #[cfg(cuda)]
        {
            let _ = cuda_sync_grad_to_host(self);
            self.cuda_grad_remove_cached_buffer();
        }
        match &self.grad {
            Storage::F64(v) => v.write().unwrap(),
            _ => panic!("Expected F64 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Convert grad to Vec<f32>.
    #[inline]
    pub fn grad_to_f32_vec(&self) -> Vec<f32> {
        #[cfg(cuda)]
        let _ = cuda_sync_grad_to_host(self);
        self.grad.to_f32_vec()
    }

    /// Convert grad to Vec<f64>.
    #[inline]
    pub fn grad_to_f64_vec(&self) -> Vec<f64> {
        #[cfg(cuda)]
        let _ = cuda_sync_grad_to_host(self);
        self.grad.to_f64_vec()
    }

    /// Accumulate f32 slice into grad.
    #[inline]
    pub fn grad_accumulate_f32(&self, slice: &[f32]) {
        #[cfg(cuda)]
        {
            let _ = cuda_sync_grad_to_host(self);
            self.cuda_grad_remove_cached_buffer();
        }
        self.grad.accumulate_f32(slice);
    }

    /// Accumulate f64 slice into grad.
    #[inline]
    pub fn grad_accumulate_f64(&self, slice: &[f64]) {
        #[cfg(cuda)]
        {
            let _ = cuda_sync_grad_to_host(self);
            self.cuda_grad_remove_cached_buffer();
        }
        self.grad.accumulate_f64(slice);
    }

    /// Get a write-compatible gradient guard.
    /// For F64 grad, returns direct RwLockWriteGuard.
    /// For F32 grad, returns a temporary f64 buffer that flushes back on Drop.
    #[inline]
    pub fn grad_write_compat(&self) -> GradWriteCompat<'_> {
        #[cfg(cuda)]
        {
            let _ = cuda_sync_grad_to_host(self);
            self.cuda_grad_remove_cached_buffer();
        }
        match &self.grad {
            Storage::F64(v) => GradWriteCompat::F64(v.write().unwrap()),
            Storage::F32(v) => {
                let len = v.read().unwrap().len();
                GradWriteCompat::F32Buffer(vec![0.0; len], &self.grad)
            }
            _ => panic!(
                "grad_write_compat: unsupported grad dtype {:?}",
                self.grad.dtype()
            ),
        }
    }

    /// Generate a tensor with uniformly distributed random values in [min, max).
    pub fn rand(shape: Vec<usize>, min: f64, max: f64, seed: u64) -> Self {
        use crate::rng::Rng;
        let len = shape.iter().product::<usize>();
        let mut rng = Rng::from_seed(seed);
        let data: Vec<f64> = (0..len)
            .map(|_| {
                let r = rng.next_f64();
                min + r * (max - min)
            })
            .collect();
        Tensor::new(data, shape)
    }

    /// Generate an F32 tensor with uniformly distributed random values in [min, max).
    pub fn rand_f32(shape: Vec<usize>, min: f32, max: f32, seed: u64) -> Self {
        use crate::rng::Rng;
        let len = shape.iter().product::<usize>();
        let mut rng = Rng::from_seed(seed);
        let data: Vec<f32> = (0..len)
            .map(|_| {
                let r = rng.next_f64() as f32;
                min + r * (max - min)
            })
            .collect();
        Tensor {
            data: Storage::F32(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F32)),
            shape,
            device: Device::Cpu,
            dtype: Dtype::F32,
            _ctx: None,
        }
    }

    /// Generate a tensor with normally distributed random values (mean=0, std=1).
    pub fn randn(shape: Vec<usize>, seed: u64) -> Self {
        use crate::rng::Rng;
        let len = shape.iter().product::<usize>();
        let mut rng = Rng::from_seed(seed);
        let data: Vec<f64> = (0..len).map(|_| rng.next_f64_normal()).collect();
        Tensor::new(data, shape)
    }

    /// Generate an F32 tensor with normally distributed random values (mean=0, std=1).
    pub fn randn_f32(shape: Vec<usize>, seed: u64) -> Self {
        use crate::rng::Rng;
        let len = shape.iter().product::<usize>();
        let mut rng = Rng::from_seed(seed);
        let data: Vec<f32> = (0..len).map(|_| rng.next_f64_normal() as f32).collect();
        Tensor {
            data: Storage::F32(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F32)),
            shape,
            device: Device::Cpu,
            dtype: Dtype::F32,
            _ctx: None,
        }
    }

    pub fn detach(&self) -> Tensor {
        let grad_len = self.grad.len();
        Tensor {
            data: self.data.clone(),
            grad: Storage::zeros(grad_len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            _ctx: None,
        }
    }

    // Create a new leaf tensor with same data (copy)
    pub fn item(&self) -> f32 {
        assert_eq!(self.shape.iter().product::<usize>(), 1);
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            use crate::cuda::memory::CudaBuffer;
            if let Some(buffer) = self.cuda_cached_buffer() {
                match &*buffer {
                    CudaBuffer::F32(b) if b.len() == 1 => {
                        let mut host = [0.0_f32; 1];
                        if crate::cuda::memory::copy_d2h(&mut host, b).is_ok() {
                            return host[0];
                        }
                    }
                    CudaBuffer::F64(b) if b.len() == 1 => {
                        let mut host = [0.0_f64; 1];
                        if crate::cuda::memory::copy_d2h(&mut host, b).is_ok() {
                            return host[0] as f32;
                        }
                    }
                    CudaBuffer::BF16(b) if b.len() == 1 => {
                        let mut host = [crate::dtype::bf16::default(); 1];
                        if crate::cuda::memory::copy_d2h(&mut host, b).is_ok() {
                            return host[0].to_f32();
                        }
                    }
                    CudaBuffer::I8(b) if b.len() == 1 => {
                        let mut host = [0_i8; 1];
                        if crate::cuda::memory::copy_d2h(&mut host, b).is_ok() {
                            return host[0] as f32;
                        }
                    }
                    _ => {}
                }
            }
        }
        self.data_to_f32_vec()[0]
    }

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

    // Operations

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() <= 2 && other.shape.len() == 2);

        let (m, k) = if self.shape.len() == 1 {
            (1, self.shape[0])
        } else {
            (self.shape[0], self.shape[1])
        };
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "MatMul dimension mismatch");

        // GPU routing: keep CUDA tensors on GPU for supported GEMM dtypes.
        #[cfg(cuda)]
        {
            let use_gpu = self.device == Device::Cuda
                && other.device == Device::Cuda
                && matches!(
                    (self.dtype, other.dtype),
                    (Dtype::F32, Dtype::F32)
                        | (Dtype::F64, Dtype::F64)
                        | (Dtype::BF16, Dtype::BF16)
                );
            if use_gpu {
                return self.matmul_cuda(other, m, k, n);
            }
        }

        let out_dtype = Tensor::binary_dtype(self.dtype, other.dtype);
        if self.dtype != Dtype::F64 || other.dtype != Dtype::F64 {
            return self.matmul_generic(other, m, k, n, out_dtype);
        }

        let mut out_data = vec![0.0; m * n];

        // Use batch lock acquisition to reduce lock overhead
        {
            let guards = TensorReadGuard::new(&[self, other]);
            let lhs_data = guards.get(0);
            let rhs_data = guards.get(1);

            // Heuristic for parallelization overhead
            // A simple matmul has M*N*K ops.
            // Rayon overhead is small but significant for tiny matrices.
            let ops = m * n * k;

            if ops < 32768 {
                for r in 0..m {
                    let out_row = &mut out_data[r * n..(r + 1) * n];
                    for i in 0..k {
                        let scale = lhs_data[r * k + i];
                        if scale == 0.0 {
                            continue;
                        }
                        if i + 2 < k {
                            prefetch_read_l1(rhs_data[(i + 2) * n..].as_ptr());
                        }
                        let rhs_row = &rhs_data[i * n..(i + 1) * n];
                        add_scaled_row(out_row, rhs_row, scale);
                    }
                }
            } else if (2..=4).contains(&m) && n >= 512 {
                let n_chunks = rayon::current_num_threads().min(8);
                let chunk_size = n.div_ceil(n_chunks);
                for r in 0..m {
                    let out_row = &mut out_data[r * n..(r + 1) * n];
                    out_row
                        .par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(ci, chunk)| {
                            let col_start = ci * chunk_size;
                            for i in 0..k {
                                let scale = lhs_data[r * k + i];
                                if scale == 0.0 {
                                    continue;
                                }
                                let rhs_slice =
                                    &rhs_data[i * n + col_start..i * n + col_start + chunk.len()];
                                add_scaled_row(chunk, rhs_slice, scale);
                            }
                        });
                }
            } else if m == 1 {
                let out_row = &mut out_data[..n];
                for i in 0..k {
                    let scale = lhs_data[i];
                    if scale == 0.0 {
                        continue;
                    }
                    let rhs_row = &rhs_data[i * n..(i + 1) * n];
                    add_scaled_row(out_row, rhs_row, scale);
                }
            } else {
                out_data
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(r, out_row)| {
                        for i in 0..k {
                            let scale = lhs_data[r * k + i];
                            if scale == 0.0 {
                                continue;
                            }
                            if i + 2 < k {
                                prefetch_read_l1(rhs_data[(i + 2) * n..].as_ptr());
                            }
                            let rhs_row = &rhs_data[i * n..(i + 1) * n];
                            add_scaled_row(out_row, rhs_row, scale);
                        }
                    });
            }
        }

        let out_shape = if self.shape.len() == 1 {
            vec![n]
        } else {
            vec![m, n]
        };

        let parents = vec![self.clone(), other.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(m * n, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];

                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);

                    // dL/dLHS = grad_out * RHS^T
                    {
                        let mut lhs_grad = lhs.grad_write_compat();
                        let ops = m * k * n;
                        if ops < 32768 {
                            for r in 0..m {
                                let grad_out_row_start = r * n;
                                let lhs_grad_row_start = r * k;
                                for i in 0..k {
                                    let rhs_row_start = i * n;
                                    let grad_row =
                                        &grad_out_f64[grad_out_row_start..grad_out_row_start + n];
                                    let rhs_row = &rhs_data[rhs_row_start..rhs_row_start + n];
                                    lhs_grad[lhs_grad_row_start + i] +=
                                        dot_product(grad_row, rhs_row);
                                }
                            }
                        } else if (2..=4).contains(&m) && k >= 64 {
                            for r in 0..m {
                                let grad_row = &grad_out_f64[r * n..(r + 1) * n];
                                let lhs_row = &mut lhs_grad[r * k..(r + 1) * k];
                                lhs_row.par_iter_mut().enumerate().for_each(|(i, val)| {
                                    let rhs_row = &rhs_data[i * n..i * n + n];
                                    *val += dot_product(grad_row, rhs_row);
                                });
                            }
                        } else if m == 1 {
                            let grad_row = &grad_out_f64[..n];
                            for i in 0..k {
                                let rhs_row = &rhs_data[i * n..i * n + n];
                                lhs_grad[i] += dot_product(grad_row, rhs_row);
                            }
                        } else {
                            lhs_grad
                                .par_chunks_mut(k)
                                .enumerate()
                                .for_each(|(r, lhs_row)| {
                                    let grad_out_row_start = r * n;
                                    let grad_row =
                                        &grad_out_f64[grad_out_row_start..grad_out_row_start + n];
                                    for (i, lhs_val) in lhs_row.iter_mut().enumerate().take(k) {
                                        let rhs_row_start = i * n;
                                        let rhs_row = &rhs_data[rhs_row_start..rhs_row_start + n];
                                        *lhs_val += dot_product(grad_row, rhs_row);
                                    }
                                });
                        }
                    }

                    // dL/dRHS = LHS^T * grad_out
                    // RHS_grad[i, :] += sum_r ( LHS[r, i] * grad_out_f64[r, :] )
                    {
                        let mut rhs_grad = rhs.grad_write_compat();
                        let ops = k * n * m;
                        if ops < 32768 {
                            // Serial
                            // Iterate over output rows (i)
                            for i in 0..k {
                                let rhs_grad_row_start = i * n;
                                let rhs_row =
                                    &mut rhs_grad[rhs_grad_row_start..rhs_grad_row_start + n];
                                for r in 0..m {
                                    let scale = lhs_data[r * k + i];
                                    if scale == 0.0 {
                                        continue;
                                    }
                                    let grad_out_row_start = r * n;
                                    let grad_row =
                                        &grad_out_f64[grad_out_row_start..grad_out_row_start + n];
                                    add_scaled_row(rhs_row, grad_row, scale);
                                }
                            }
                        } else {
                            rhs_grad
                                .par_chunks_mut(n)
                                .enumerate()
                                .for_each(|(i, rhs_row)| {
                                    for r in 0..m {
                                        let scale = lhs_data[r * k + i];
                                        if scale == 0.0 {
                                            continue;
                                        }
                                        let grad_out_row_start = r * n;
                                        let grad_row = &grad_out_f64
                                            [grad_out_row_start..grad_out_row_start + n];
                                        add_scaled_row(rhs_row, grad_row, scale);
                                    }
                                });
                        }
                    }
                }),
            })),
        }
    }

    /// Generic matmul for non-F64 dtypes (F32, BF16, I8, or mixed).
    /// Converts inputs to f32, computes, then converts output to the target dtype.
    fn matmul_generic(
        &self,
        other: &Tensor,
        m: usize,
        k: usize,
        n: usize,
        out_dtype: Dtype,
    ) -> Tensor {
        let lhs_f32 = self.data_to_f32_vec();
        let rhs_f32 = other.data_to_f32_vec();

        let mut out_data = vec![0.0f32; m * n];
        for r in 0..m {
            for i in 0..k {
                let scale = lhs_f32[r * k + i];
                if scale == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out_data[r * n + j] += scale * rhs_f32[i * n + j];
                }
            }
        }

        let out_shape = if self.shape.len() == 1 {
            vec![n]
        } else {
            vec![m, n]
        };

        let lhs_cache: Arc<Vec<f64>> = Arc::new(lhs_f32.iter().map(|&v| v as f64).collect());
        let rhs_cache: Arc<Vec<f64>> = Arc::new(rhs_f32.iter().map(|&v| v as f64).collect());

        Tensor {
            data: Storage::from_f32_vec(out_data, out_dtype),
            grad: Storage::zeros(m * n, Tensor::grad_dtype_for(out_dtype)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), other.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    // dL/dLHS = grad_out * RHS^T
                    let mut lhs_grad = _parents[0].grad_write_compat();
                    for r in 0..m {
                        for i in 0..k {
                            let mut sum = 0.0f64;
                            for j in 0..n {
                                sum += grad_out_f64[r * n + j] * rhs_cache[i * n + j];
                            }
                            lhs_grad[r * k + i] += sum;
                        }
                    }

                    // dL/dRHS = LHS^T * grad_out
                    let mut rhs_grad = _parents[1].grad_write_compat();
                    for i in 0..k {
                        for j in 0..n {
                            let mut sum = 0.0f64;
                            for r in 0..m {
                                sum += lhs_cache[r * k + i] * grad_out_f64[r * n + j];
                            }
                            rhs_grad[i * n + j] += sum;
                        }
                    }
                }),
            })),
        }
    }

    pub fn relu(&self) -> Tensor {
        // GPU routing for float dtypes
        #[cfg(cuda)]
        if self.device == Device::Cuda && (self.dtype == Dtype::F32 || self.dtype == Dtype::F64) {
            return self.relu_cuda();
        }

        // Generic path for non-F64 dtypes
        if self.dtype != Dtype::F64 {
            return self.relu_generic();
        }

        // CPU path - inline implementation when cuda is disabled
        #[cfg(not(cuda))]
        {
            let self_data = self.data_f64();
            let len = self_data.len();
            let mut data = vec![0.0; len];
            if len >= PAR_THRESHOLD {
                data.par_chunks_mut(PAR_THRESHOLD)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * PAR_THRESHOLD;
                        vector_relu(chunk, &self_data[start..start + chunk.len()]);
                    });
            } else {
                vector_relu(&mut data, &self_data);
            }
            let parents = vec![self.clone()];

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
                        let input = &parents[0];
                        let input_data = input.data_f64();
                        let mut inp_grad = input.grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_data.par_iter())
                                .for_each(|((ig, &go), &val)| {
                                    if val > 0.0 {
                                        *ig += go;
                                    }
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                if input_data[i] > 0.0 {
                                    inp_grad[i] += grad_out_f64[i];
                                }
                            }
                        }
                    }),
                })),
            }
        }

        // CPU path - call fallback when cuda is enabled but device is CPU
        #[cfg(cuda)]
        self.relu_cpu_fallback()
    }

    /// Generic ReLU for non-F64 dtypes.
    fn relu_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        let mask: Vec<bool> = self_f32.iter().map(|&x| x > 0.0).collect();
        for i in 0..len {
            data[i] = if mask[i] { self_f32[i] } else { 0.0 };
        }
        let mask = Arc::new(mask);
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
                    for i in 0..len {
                        if mask[i] {
                            inp_grad[i] += grad_out_f64[i];
                        }
                    }
                }),
            })),
        }
    }

    /// CPU fallback for ReLU activation
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn relu_cpu_fallback(&self) -> Tensor {
        if self.dtype == Dtype::F32 {
            return self.relu_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_chunks_mut(PAR_THRESHOLD)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * PAR_THRESHOLD;
                    vector_relu(chunk, &self_data[start..start + chunk.len()]);
                });
        } else {
            vector_relu(&mut data, &self_data);
        }
        let parents = vec![self.clone()];

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
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &go), &val)| {
                                if val > 0.0 {
                                    *ig += go;
                                }
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            if input_data[i] > 0.0 {
                                inp_grad[i] += grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    /// GPU-accelerated ReLU activation using CUDA kernel
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn relu_cuda(&self) -> Tensor {
        use crate::cuda::kernels::{
            relu_backward, relu_backward_f32, relu_inplace, relu_inplace_f32,
        };
        use crate::cuda::memory::{alloc, copy_d2d, CudaBuffer};

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.relu_cpu_fallback(),
        };
        if len == 0 {
            return self.relu_cpu_fallback();
        }
        crate::cuda::record_activation_attempt();

        let d_src = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_activation_fallback(stage);
                log::warn!(
                    "[Autograd] CUDA prepare ReLU input failed ({}), using CPU",
                    err
                );
                return self.relu_cpu_fallback();
            }
        };
        let d_data = match self.dtype {
            Dtype::F32 => match alloc::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc ReLU buffer failed ({}), using CPU",
                        err
                    );
                    return self.relu_cpu_fallback();
                }
            },
            Dtype::F64 => match alloc::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc ReLU buffer failed ({}), using CPU",
                        err
                    );
                    return self.relu_cpu_fallback();
                }
            },
            _ => return self.relu_cpu_fallback(),
        };
        let copy_ok = match (self.dtype, &d_data, &*d_src) {
            (Dtype::F32, CudaBuffer::F32(dst), CudaBuffer::F32(src)) => copy_d2d(dst, src).is_ok(),
            (Dtype::F64, CudaBuffer::F64(dst), CudaBuffer::F64(src)) => copy_d2d(dst, src).is_ok(),
            _ => false,
        };
        if !copy_ok {
            crate::cuda::record_activation_fallback("copy");
            log::warn!("[Autograd] CUDA D2D ReLU input copy failed, using CPU");
            return self.relu_cpu_fallback();
        }

        let kernel_ok = match (self.dtype, &d_data) {
            (Dtype::F32, CudaBuffer::F32(b)) => relu_inplace_f32(b).is_ok(),
            (Dtype::F64, CudaBuffer::F64(b)) => relu_inplace(b).is_ok(),
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA ReLU kernel failed, using CPU");
            return self.relu_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
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
                    #[cfg(cuda)]
                    if input.device == Device::Cuda {
                        if let Some(d_input) = input.cuda_cached_buffer() {
                            let d_grad_tmp = cuda_grad_out_buffer(grad_out);
                            if let Some(d_grad_tmp) = d_grad_tmp {
                                if let Some(d_in_grad) = input.cuda_grad_ensure_buffer() {
                                    match (&*d_input, &*d_grad_tmp, &*d_in_grad, input.dtype) {
                                        (
                                            CudaBuffer::F32(inp),
                                            CudaBuffer::F32(gt),
                                            CudaBuffer::F32(ig),
                                            Dtype::F32,
                                        ) => {
                                            if relu_backward_f32(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        (
                                            CudaBuffer::F64(inp),
                                            CudaBuffer::F64(gt),
                                            CudaBuffer::F64(ig),
                                            Dtype::F64,
                                        ) => {
                                            if relu_backward(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &go), &val)| {
                                if val > 0.0 {
                                    *ig += go;
                                }
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            if input_data[i] > 0.0 {
                                inp_grad[i] += grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_data);
        out
    }

    /// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Tensor {
        // GPU routing
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            return self.gelu_cuda();
        }
        if self.dtype != Dtype::F64 {
            return self.gelu_generic();
        }

        let self_data = self.data_f64();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_chunks_mut(PAR_THRESHOLD)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * PAR_THRESHOLD;
                    vector_gelu(chunk, &self_data[start..start + chunk.len()]);
                });
        } else {
            vector_gelu(&mut data, &self_data);
        }
        let parents = vec![self.clone()];

        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let c = 0.044715;

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
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &go), &x)| {
                                let x2 = x * x;
                                let x3 = x2 * x;
                                let u = sqrt_2_over_pi * (x + c * x3);
                                let tanh_u = u.tanh();
                                let sech2_u = 1.0 - tanh_u * tanh_u;
                                let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                                let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                                *ig += go * gelu_grad;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            let x = input_data[i];
                            let x2 = x * x;
                            let x3 = x2 * x;
                            let u = sqrt_2_over_pi * (x + c * x3);
                            let tanh_u = u.tanh();
                            let sech2_u = 1.0 - tanh_u * tanh_u;
                            let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                            let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                            inp_grad[i] += grad_out_f64[i] * gelu_grad;
                        }
                    }
                }),
            })),
        }
    }

    fn gelu_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        let c = 0.044715f32;
        for i in 0..len {
            let x = self_f32[i];
            let x3 = x * x * x;
            let inner = sqrt_2_over_pi * (x + c * x3);
            data[i] = 0.5 * x * (1.0 + inner.tanh());
        }
        let input_cache: Arc<Vec<f64>> = Arc::new(self_f32.iter().map(|&v| v as f64).collect());
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
                    for i in 0..len {
                        let x = input_cache[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = (2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx =
                            (2.0f64 / std::f64::consts::PI).sqrt() * (1.0 + 3.0 * 0.044715 * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out_f64[i] * gelu_grad;
                    }
                }),
            })),
        }
    }

    pub fn log(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.ln()).collect()
            } else {
                self_data.iter().map(|&x| x.ln()).collect()
            };
            let input_cache = Arc::new(self_data);
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        const LOG_GRAD_EPS: f64 = 1e-12;
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_cache.par_iter())
                                .for_each(|((ig, &g), &id)| {
                                    let safe = if id.abs() < LOG_GRAD_EPS {
                                        id.signum() * LOG_GRAD_EPS
                                    } else {
                                        id
                                    };
                                    *ig += g / safe;
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                let safe = if input_cache[i].abs() < LOG_GRAD_EPS {
                                    input_cache[i].signum() * LOG_GRAD_EPS
                                } else {
                                    input_cache[i]
                                };
                                inp_grad[i] += grad_out_f64[i] / safe;
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.ln()).collect()
        } else {
            self_data.iter().map(|&x| x.ln()).collect()
        };
        let parents = vec![self.clone()];

        let input_cache_f64: Arc<Vec<f64>> =
            Arc::new(self_data.iter().map(|&v| v as f64).collect());
        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    const LOG_GRAD_EPS: f64 = 1e-12;
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_cache_f64.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                let safe = if id.abs() < LOG_GRAD_EPS {
                                    id.signum() * LOG_GRAD_EPS
                                } else {
                                    id
                                };
                                *ig += g / safe;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            let safe = if input_cache_f64[i].abs() < LOG_GRAD_EPS {
                                input_cache_f64[i].signum() * LOG_GRAD_EPS
                            } else {
                                input_cache_f64[i]
                            };
                            inp_grad[i] += grad_out_f64[i] / safe;
                        }
                    }
                }),
            })),
        }
    }

    pub fn exp(&self) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.exp_cuda() {
            return out;
        }

        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.exp()).collect()
            } else {
                self_data.iter().map(|&x| x.exp()).collect()
            };
            let parents = vec![self.clone()];
            let exp_cache = Arc::new(data.clone());
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(exp_cache.par_iter())
                                .for_each(|((ig, &g), &cached)| {
                                    *ig += g * cached;
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] += grad_out_f64[i] * exp_cache[i];
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let mut data = vec![0.0f32; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.exp()).collect();
        } else {
            crate::simd::fast_exp_bulk_f32(&mut data, &self_data);
        }
        let parents = vec![self.clone()];
        let exp_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|&v| v as f64).collect());

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(exp_cache.par_iter())
                            .for_each(|((ig, &g), &cached)| {
                                *ig += g * cached;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * exp_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    /// Element-wise absolute value.
    /// Forward: |x|
    /// Backward: d/dx|x| = sign(x), where sign(0) = 0
    pub fn abs(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let mut data = vec![0.0f64; len];
            if len >= PAR_THRESHOLD {
                data = self_data.par_iter().map(|&x| x.abs()).collect();
            } else {
                for i in 0..len {
                    data[i] = self_data[i].abs();
                }
            }
            let sign_cache: Arc<Vec<f64>> = Arc::new(
                self_data
                    .iter()
                    .map(|&x| {
                        if x > 0.0 {
                            1.0
                        } else if x < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(sign_cache.par_iter())
                                .for_each(|((ig, &g), &s)| {
                                    *ig += g * s;
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] += grad_out_f64[i] * sign_cache[i];
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let mut data = vec![0.0f32; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.abs()).collect();
        } else {
            for i in 0..len {
                data[i] = self_data[i].abs();
            }
        }
        let sign_cache: Arc<Vec<f64>> = Arc::new(
            self_data
                .iter()
                .map(|&x| {
                    if x > 0.0 {
                        1.0f64
                    } else if x < 0.0 {
                        -1.0f64
                    } else {
                        0.0f64
                    }
                })
                .collect(),
        );
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(sign_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                *ig += g * s;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * sign_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    /// Element-wise exponentiation: x^exponent.
    /// Forward: x^n
    /// Backward: d/dx x^n = n * x^(n-1)
    pub fn pow(&self, exponent: f64) -> Tensor {
        let self_data = self.data_as_f64_vec();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.powf(exponent)).collect();
        } else {
            for i in 0..len {
                data[i] = self_data[i].powf(exponent);
            }
        }
        // Cache forward result and exponent for backward
        let exp = exponent;
        let pow_cache: Arc<Vec<f64>> =
            Arc::new(self_data.iter().map(|&x| x.powf(exp - 1.0)).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f64_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(pow_cache.par_iter())
                            .for_each(|((ig, &g), &cached)| {
                                *ig += g * exp * cached;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * exp * pow_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    /// Softmax: exp(x_i) / sum_j(exp(x_j)) with numerical stability (shift by max)
    pub fn softmax(&self) -> Tensor {
        // GPU routing
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
                    // d/dx softmax = softmax * (grad_out - sum(grad_out * softmax))
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

        // Row-major arbitrary-dim indexing:
        // idx = outer_idx * (dim_size * inner) + dim_idx * inner + inner_idx
        if inner == 1 {
            // Fast path: contiguous rows (common case for last-dim softmax)
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

        // Compute mean over last dimension
        let mut mean = vec![0.0; outer_len];
        for (i, mean_elem) in mean.iter_mut().enumerate().take(outer_len) {
            let base = i * last_dim;
            let mut sum = 0.0;
            for j in 0..last_dim {
                sum += self_data[base + j];
            }
            *mean_elem = sum / last_dim as f64;
        }

        // Compute variance and normalized output
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
            // inv_std = 1/sqrt(var + eps) precomputed to avoid sqrt in inner loop
            let inv_std = 1.0 / (var[i] + eps).sqrt();
            let slice = &self_data[base..base + last_dim];
            let normalized = crate::simd::layer_norm(slice, m, inv_std, &[], &[]);
            output[base..base + last_dim].copy_from_slice(&normalized);
        }

        // Store input data in Arc so backward pass can access it
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

                        // Compute aggregated gradients
                        let mut g_sum = 0.0;
                        let mut g_diff_sum = 0.0;
                        for j in 0..last_dim {
                            let diff = input_data[base + j] - m;
                            g_sum += grad_out_f64[base + j];
                            g_diff_sum += grad_out_f64[base + j] * diff;
                        }

                        // dvar = sum grad_out * (x - mean) * -0.5 / std^3
                        let dvar = -0.5 * g_diff_sum / std3;
                        // dmean = -sum(grad_out) / std + dvar * -2 * mean / N
                        let dmean = -g_sum / std + dvar * -2.0 * m / last_dim_f;

                        // dx_j = grad_out_j / std + dvar * 2 * (x_j - m) / N + dmean / N
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

    /// Select a single element by index, producing a scalar Tensor with gradient support.
    pub fn index_select(&self, idx: usize) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.index_select_cuda(idx) {
            return out;
        }
        let self_data = self.data_f64();
        let val = self_data[idx];
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![val]))),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    if idx < inp_grad.len() {
                        inp_grad[idx] += grad_out_f64[0];
                    }
                }),
            })),
        }
    }

    pub fn sum(&self) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.sum_cuda(1.0) {
            return out;
        }

        if self.dtype != Dtype::F64 {
            return self.sum_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let sum_val: f64 = if len >= PAR_THRESHOLD {
            self_data.par_iter().sum()
        } else {
            horizontal_sum(&self_data)
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![sum_val]))),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    let g = grad_out_f64[0];
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad.par_iter_mut().for_each(|v| *v += g);
                    } else {
                        for v in inp_grad.iter_mut() {
                            *v += g;
                        }
                    }
                }),
            })),
        }
    }

    pub fn mean(&self) -> Tensor {
        #[cfg(cuda)]
        if self.numel() > 0 {
            if let Some(out) = self.sum_cuda(1.0 / self.numel() as f64) {
                return out;
            }
        }

        if self.dtype != Dtype::F64 {
            return self.mean_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let sum_val: f64 = if len >= PAR_THRESHOLD {
            self_data.par_iter().sum()
        } else {
            horizontal_sum(&self_data)
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![sum_val / len as f64]))),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    let g = grad_out_f64[0] / len as f64;
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad.par_iter_mut().for_each(|v| *v += g);
                    } else {
                        for v in inp_grad.iter_mut() {
                            *v += g;
                        }
                    }
                }),
            })),
        }
    }

    fn sum_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let sum_val: f32 = self_f32.iter().sum();
        Tensor {
            data: Storage::from_f32_vec(vec![sum_val], self.dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(self.dtype)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let g = grad_out_f64[0];
                    for v in inp_grad.iter_mut() {
                        *v += g;
                    }
                }),
            })),
        }
    }

    fn mean_generic(&self) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let len = self_f32.len();
        let mean_val = self_f32.iter().sum::<f32>() / len as f32;
        Tensor {
            data: Storage::from_f32_vec(vec![mean_val], self.dtype),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(self.dtype)),
            shape: vec![1],
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let g = grad_out_f64[0] / len as f64;
                    for v in inp_grad.iter_mut() {
                        *v += g;
                    }
                }),
            })),
        }
    }

    pub fn transpose2d(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose requires 2D tensor");
        #[cfg(cuda)]
        if let Some(out) = self.transpose_cuda(0, 1) {
            return out;
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        if self.dtype != Dtype::F64 {
            return self.transpose2d_generic();
        }
        let self_data = self.data_f64();
        let mut out_data = vec![0.0; self_data.len()];
        for r in 0..rows {
            for c in 0..cols {
                out_data[c * rows + r] = self_data[r * cols + c];
            }
        }
        let parents = vec![self.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(rows * cols, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![cols, rows],
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let mut inp_grad = input.grad_write_compat();
                    for r in 0..rows {
                        for c in 0..cols {
                            inp_grad[r * cols + c] += grad_out_f64[c * rows + r];
                        }
                    }
                }),
            })),
        }
    }

    fn transpose2d_generic(&self) -> Tensor {
        let rows = self.shape[0];
        let cols = self.shape[1];
        let self_f32 = self.data_to_f32_vec();
        let mut out_data = vec![0.0f32; self_f32.len()];
        for r in 0..rows {
            for c in 0..cols {
                out_data[c * rows + r] = self_f32[r * cols + c];
            }
        }
        Tensor {
            data: Storage::from_f32_vec(out_data, self.dtype),
            grad: Storage::zeros(rows * cols, Tensor::grad_dtype_for(self.dtype)),
            shape: vec![cols, rows],
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    for r in 0..rows {
                        for c in 0..cols {
                            inp_grad[r * cols + c] += grad_out_f64[c * rows + r];
                        }
                    }
                }),
            })),
        }
    }

    /// Check if two shapes are broadcast-compatible.
    /// Dimensions are compared from the right; each dimension must either match or be 1.
    fn broadcastable_shapes(old: &[usize], new: &[usize]) -> bool {
        let old_len = old.len();
        let new_len = new.len();
        let max_len = old_len.max(new_len);

        for i in 0..max_len {
            // Compute offset into old/new when aligned from the right
            // i=0 is the rightmost dimension
            let old_offset = i as isize - (max_len as isize - old_len as isize);
            let new_offset = i as isize - (max_len as isize - new_len as isize);

            let old_dim = if old_offset < 0 {
                1
            } else {
                old[old_len - 1 - old_offset as usize]
            };
            let new_dim = if new_offset < 0 {
                1
            } else {
                new[new_len - 1 - new_offset as usize]
            };

            if old_dim != new_dim && old_dim != 1 && new_dim != 1 {
                return false;
            }
        }
        true
    }

    pub fn broadcast(&self, new_shape: Vec<usize>) -> Tensor {
        let old_shape = self.shape.clone();
        let old_len = old_shape.len();
        let new_len = new_shape.len();

        assert!(
            Self::broadcastable_shapes(&old_shape, &new_shape),
            "Shapes {:?} and {:?} are not broadcast-compatible",
            old_shape,
            new_shape
        );

        let max_len = old_len.max(new_len);
        let total_elements: usize = new_shape.iter().product();

        let self_data = self.data_as_f64_vec();
        let old_data = &self_data;

        let mut new_data = Vec::with_capacity(total_elements);

        for linear_idx in 0..total_elements {
            let mut old_linear_idx = 0usize;
            let mut multiplier = 1usize;

            for dim in 0..max_len {
                let old_dim = if dim < max_len - old_len {
                    1
                } else {
                    old_shape[old_len - 1 - (dim - (max_len - old_len))]
                };
                let new_dim = if dim < max_len - new_len {
                    1
                } else {
                    new_shape[new_len - 1 - (dim - (max_len - new_len))]
                };

                let pos = (linear_idx / multiplier) % new_dim;
                if old_dim != 1 {
                    old_linear_idx += pos * multiplier;
                }
                multiplier *= new_dim;
            }

            new_data.push(old_data[old_linear_idx]);
        }

        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f64_vec(new_data, self.dtype),
            grad: Storage::zeros(total_elements, Tensor::grad_dtype_for(self.dtype)),
            shape: new_shape.clone(),
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let total_elements = grad_out_f64.len();
                    let old_shape = old_shape.clone();
                    let old_len = old_shape.len();
                    let new_len = new_shape.len();
                    let max_len = old_len.max(new_len);

                    #[allow(clippy::needless_range_loop)]
                    for linear_idx in 0..total_elements {
                        let mut old_linear_idx = 0usize;
                        let mut multiplier = 1usize;

                        for dim in 0..max_len {
                            let old_dim = if dim < max_len - old_len {
                                1
                            } else {
                                old_shape[old_len - 1 - (dim - (max_len - old_len))]
                            };
                            let new_dim = if dim < max_len - new_len {
                                1
                            } else {
                                new_shape[new_len - 1 - (dim - (max_len - new_len))]
                            };

                            let pos = (linear_idx / multiplier) % new_dim;
                            if old_dim != 1 {
                                old_linear_idx += pos * multiplier;
                            }
                            multiplier *= new_dim;
                        }

                        inp_grad[old_linear_idx] += grad_out_f64[linear_idx];
                    }
                }),
            })),
        }
    }

    pub fn broadcast_to_batch(&self, batch_size: usize) -> Tensor {
        #[cfg(cuda)]
        if let Some(out) = self.broadcast_to_batch_cuda(batch_size) {
            return out;
        }
        let self_data = self.data_as_f64_vec();
        let len = self_data.len();
        let mut new_data = Vec::with_capacity(len * batch_size);
        for _ in 0..batch_size {
            new_data.extend_from_slice(&self_data);
        }

        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(&self.shape);

        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f64_vec(new_data, self.dtype),
            grad: Storage::zeros(len * batch_size, Tensor::grad_dtype_for(self.dtype)),
            shape: new_shape,
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let chunk_size = inp_grad.len();
                    for chunk in grad_out_f64.chunks(chunk_size) {
                        for (i, &g) in chunk.iter().enumerate() {
                            inp_grad[i] += g;
                        }
                    }
                }),
            })),
        }
    }

    pub fn sin(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.sin()).collect()
            } else {
                self_data.iter().map(|&x| x.sin()).collect()
            };
            let input_cache = Arc::new(self_data);
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_cache.par_iter())
                                .for_each(|((ig, &g), &id)| {
                                    *ig += g * id.cos();
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] += grad_out_f64[i] * input_cache[i].cos();
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.sin()).collect()
        } else {
            self_data.iter().map(|&x| x.sin()).collect()
        };
        let input_cache: Arc<Vec<f64>> = Arc::new(self_data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_cache.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig += g * id.cos();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f64[i] * input_cache[i].cos();
                        }
                    }
                }),
            })),
        }
    }

    pub fn cos(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.cos()).collect()
            } else {
                self_data.iter().map(|&x| x.cos()).collect()
            };
            let input_cache = Arc::new(self_data);
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(input_cache.par_iter())
                                .for_each(|((ig, &g), &id)| {
                                    *ig -= g * id.sin();
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                inp_grad[i] -= grad_out_f64[i] * input_cache[i].sin();
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.cos()).collect()
        } else {
            self_data.iter().map(|&x| x.cos()).collect()
        };
        let input_cache: Arc<Vec<f64>> = Arc::new(self_data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_cache.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig -= g * id.sin();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] -= grad_out_f64[i] * input_cache[i].sin();
                        }
                    }
                }),
            })),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        if self.dtype == Dtype::F64 {
            let self_data = self.data_as_f64_vec();
            let len = self_data.len();
            let data: Vec<f64> = if len >= PAR_THRESHOLD {
                self_data.par_iter().map(|&x| x.sqrt()).collect()
            } else {
                self_data.iter().map(|&x| x.sqrt()).collect()
            };
            let sqrt_cache = data.clone();
            let parents = vec![self.clone()];
            return Tensor {
                data: Storage::f64(data),
                grad: Storage::zeros(len, Dtype::F64),
                shape: self.shape.clone(),
                device: self.device,
                dtype: Dtype::F64,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        let mut inp_grad = _parents[0].grad_write_compat();
                        let len = inp_grad.len();
                        if len >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(sqrt_cache.par_iter())
                                .for_each(|((ig, &g), &s)| {
                                    if s > 0.0 {
                                        *ig += g * 0.5 / s;
                                    }
                                });
                        } else {
                            for i in 0..len {
                                if sqrt_cache[i] > 0.0 {
                                    inp_grad[i] += grad_out_f64[i] * 0.5 / sqrt_cache[i];
                                }
                            }
                        }
                    }),
                })),
            };
        }

        let self_data = self.data_to_f32_vec();
        let len = self_data.len();
        let data: Vec<f32> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.sqrt()).collect()
        } else {
            self_data.iter().map(|&x| x.sqrt()).collect()
        };
        let sqrt_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|&v| v as f64).collect());
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let len = inp_grad.len();
                    // Reuse cached sqrt values: d/dx sqrt(x) = 0.5 / sqrt(x)
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(sqrt_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                if s > 0.0 {
                                    *ig += g * 0.5 / s;
                                }
                            });
                    } else {
                        for i in 0..len {
                            if sqrt_cache[i] > 0.0 {
                                inp_grad[i] += grad_out_f64[i] * 0.5 / sqrt_cache[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let shape = &self.shape;
        let rank = shape.len();
        assert!(dim0 < rank && dim1 < rank);
        #[cfg(cuda)]
        if let Some(out) = self.transpose_cuda(dim0, dim1) {
            return out;
        }

        let self_data = self.data_as_f64_vec();
        assert!(
            rank <= 8,
            "transpose: rank > 8 not supported by stack-allocated coords"
        );

        let mut new_shape = shape.clone();
        new_shape.swap(dim0, dim1);

        let len = self_data.len();
        let mut new_data = vec![0.0; len];

        let mut strides = [0usize; 8];
        strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let mut new_strides = [0usize; 8];
        new_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        for (i, value) in new_data.iter_mut().enumerate().take(len) {
            let mut temp = i;
            let mut coords = [0usize; 8];
            for d in 0..rank {
                coords[d] = temp / new_strides[d];
                temp %= new_strides[d];
            }

            coords.swap(dim0, dim1);

            let mut old_idx = 0;
            for d in 0..rank {
                old_idx += coords[d] * strides[d];
            }

            *value = self_data[old_idx];
        }

        let parents = vec![self.clone()];
        let dim0_cap = dim0;
        let dim1_cap = dim1;
        let cap_strides = strides;
        let cap_new_strides = new_strides;
        let cap_rank = rank;

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(new_data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: new_shape,
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let mut inp_grad = input.grad_write_compat();

                    for (i, &grad_val) in grad_out_f64.iter().enumerate() {
                        let mut temp = i;
                        let mut coords = [0usize; 8];
                        for d in 0..cap_rank {
                            coords[d] = temp / cap_new_strides[d];
                            temp %= cap_new_strides[d];
                        }

                        coords.swap(dim0_cap, dim1_cap);

                        let mut old_idx = 0;
                        for d in 0..cap_rank {
                            old_idx += coords[d] * cap_strides[d];
                        }

                        inp_grad[old_idx] += grad_val;
                    }
                }),
            })),
        }
    }

    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        let self_data = self.data_f64();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.max(min).min(max)).collect()
        } else {
            self_data.iter().map(|&x| x.max(min).min(max)).collect()
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    let len = inp_grad.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                if id >= min && id <= max {
                                    *ig += g;
                                }
                            });
                    } else {
                        for i in 0..len {
                            if input_data[i] >= min && input_data[i] <= max {
                                inp_grad[i] += grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let len: usize = new_shape.iter().product::<usize>();
        assert_eq!(len, self.numel(), "Reshape dimension mismatch");

        // Zero-copy: share the same data Arc, only change shape metadata
        let parents = vec![self.clone()];

        let out = Tensor {
            data: self.data.clone(),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: new_shape,
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_compat();
                    let len = grad_out_f64.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .for_each(|(ig, &g)| *ig += g);
                    } else {
                        for i in 0..len {
                            inp_grad[i] += grad_out_f64[i];
                        }
                    }
                }),
            })),
        };
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            if let Some(buffer) = self.cuda_cached_buffer() {
                out.cuda_set_cached_buffer(buffer);
            }
        }
        out
    }

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

    // Winograd F(2x2, 3x3) implementation
    // Input tile: 4x4, Output tile: 2x2
    fn winograd_conv2d_3x3(&self, weight: &Tensor, padding: usize) -> Tensor {
        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, _, _, _) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );
        // h_out, w_out calculation for stride 1, kernel 3
        let h_out = h_in + 2 * padding - 2;
        let w_out = w_in + 2 * padding - 2;

        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len: usize = out_shape.iter().product();
        let mut out_data = vec![0.0; out_len];

        // Standard Winograd F(2,3) matrices. Hardcoded for speed.
        // G (4x3), B^T (4x4), A^T (2x4)

        // We compute U = G * g * G^T per [k, c] 3x3 block.

        let u_len = c_out * c_in * 16;
        let mut u_data = vec![0.0; u_len]; // [C_out, C_in, 4, 4]

        {
            let weight_data = weight.data_f64();

            // Precompute U. This transforms the kernel into Winograd domain.
            u_data
                .par_chunks_mut(16)
                .enumerate()
                .for_each(|(idx, u_block)| {
                    // idx corresponds to (k * c_in + c)
                    let k = idx / c_in;
                    let c = idx % c_in;

                    // Read 3x3 weight
                    let w_base = (k * c_in + c) * 9;
                    let g00 = weight_data[w_base];
                    let g01 = weight_data[w_base + 1];
                    let g02 = weight_data[w_base + 2];
                    let g10 = weight_data[w_base + 3];
                    let g11 = weight_data[w_base + 4];
                    let g12 = weight_data[w_base + 5];
                    let g20 = weight_data[w_base + 6];
                    let g21 = weight_data[w_base + 7];
                    let g22 = weight_data[w_base + 8];

                    // Compute U = G * g * G^T
                    // Unrolled manually to avoid allocation

                    // Tmp = g * G^T
                    let t00 = g00;
                    let t01 = 0.5 * (g00 + g01 + g02);
                    let t02 = 0.5 * (g00 - g01 + g02);
                    let t03 = g02;

                    let t10 = g10;
                    let t11 = 0.5 * (g10 + g11 + g12);
                    let t12 = 0.5 * (g10 - g11 + g12);
                    let t13 = g12;

                    let t20 = g20;
                    let t21 = 0.5 * (g20 + g21 + g22);
                    let t22 = 0.5 * (g20 - g21 + g22);
                    let t23 = g22;

                    // U = G * Tmp
                    u_block[0] = t00;
                    u_block[4] = 0.5 * (t00 + t10 + t20);
                    u_block[8] = 0.5 * (t00 - t10 + t20);
                    u_block[12] = t20;

                    u_block[1] = t01;
                    u_block[5] = 0.5 * (t01 + t11 + t21);
                    u_block[9] = 0.5 * (t01 - t11 + t21);
                    u_block[13] = t21;

                    u_block[2] = t02;
                    u_block[6] = 0.5 * (t02 + t12 + t22);
                    u_block[10] = 0.5 * (t02 - t12 + t22);
                    u_block[14] = t22;

                    u_block[3] = t03;
                    u_block[7] = 0.5 * (t03 + t13 + t23);
                    u_block[11] = 0.5 * (t03 - t13 + t23);
                    u_block[15] = t23;
                });
        }

        {
            let input_data = self.data_f64();

            // Output is computed in 2x2 blocks (tiles).
            let n_tiles_h = h_out.div_ceil(2);
            let n_tiles_w = w_out.div_ceil(2);
            let n_tiles = n_tiles_h * n_tiles_w;

            let out_plane_len = h_out * w_out;

            out_data
                .par_chunks_mut(c_out * out_plane_len)
                .enumerate()
                .for_each(|(b, out_batch)| {
                    // We could parallelize over tiles, but that requires atomic writes to output or careful locking.
                    // Easier to parallelize over Output Channels (C_out) since they are independent.

                    // First, transform input image into V domain: V = B^T d B.
                    // This is shared across all C_out, so we do it once per batch item.
                    // V: [Tiles, C_in, 4, 4]
                    let mut v_data = vec![0.0; n_tiles * c_in * 16];

                    // Parallelize V computation over (Tile, C_in)
                    v_data
                        .par_chunks_mut(16)
                        .enumerate()
                        .for_each(|(idx, v_block)| {
                            let tile_idx = idx / c_in;
                            let c = idx % c_in;

                            let th = tile_idx / n_tiles_w;
                            let tw = tile_idx % n_tiles_w;

                            let h_start = (th * 2) as isize - padding as isize;
                            let w_start = (tw * 2) as isize - padding as isize;

                            // Read 4x4 input tile d
                            let mut d = [0.0; 16];
                            for i in 0..4 {
                                for j in 0..4 {
                                    let ih = h_start + i as isize;
                                    let iw = w_start + j as isize;
                                    if ih >= 0
                                        && ih < h_in as isize
                                        && iw >= 0
                                        && iw < w_in as isize
                                    {
                                        d[i * 4 + j] = input_data[((b * c_in + c) * h_in
                                            + ih as usize)
                                            * w_in
                                            + iw as usize];
                                    }
                                }
                            }

                            // Compute V = B^T * d * B
                            // 1. Tmp = B^T * d
                            let mut tmp = [0.0; 16];
                            for j in 0..4 {
                                let d0 = d[j];
                                let d1 = d[4 + j];
                                let d2 = d[8 + j];
                                let d3 = d[12 + j];
                                tmp[j] = d0 - d2;
                                tmp[4 + j] = d1 + d2;
                                tmp[8 + j] = d2 - d1;
                                tmp[12 + j] = d1 - d3;
                            }

                            // 2. V = Tmp * B
                            for i in 0..4 {
                                // row i
                                let t0 = tmp[i * 4];
                                let t1 = tmp[i * 4 + 1];
                                let t2 = tmp[i * 4 + 2];
                                let t3 = tmp[i * 4 + 3];
                                v_block[i * 4] = t0 - t2;
                                v_block[i * 4 + 1] = t1 + t2;
                                v_block[i * 4 + 2] = t2 - t1;
                                v_block[i * 4 + 3] = t1 - t3;
                            }
                        });

                    // Now Compute M = U * V and Y = A^T M A
                    // This part is specific to each C_out.
                    out_batch
                        .par_chunks_mut(out_plane_len)
                        .enumerate()
                        .for_each(|(k, out_plane)| {
                            for t in 0..n_tiles {
                                let th = t / n_tiles_w;
                                let tw = t % n_tiles_w;

                                // M = Sum_c (U[k,c] .* V[t,c])
                                let mut m = [0.0; 16];
                                for c in 0..c_in {
                                    let u_ptr = &u_data[((k * c_in + c) * 16)..];
                                    let v_ptr = &v_data[((t * c_in + c) * 16)..];
                                    // Element-wise mul. Hot path!
                                    vector_fma(&mut m, &u_ptr[0..16], &v_ptr[0..16]);
                                }

                                // Y = A^T * m * A
                                // 1. Tmp = A^T * m
                                let mut tmp = [0.0; 8];
                                for j in 0..4 {
                                    let m0 = m[j];
                                    let m1 = m[4 + j];
                                    let m2 = m[8 + j];
                                    let m3 = m[12 + j];
                                    tmp[j] = m0 + m1 + m2;
                                    tmp[4 + j] = m1 - m2 - m3;
                                }

                                // 2. Y = Tmp * A
                                let t00 = tmp[0];
                                let t01 = tmp[1];
                                let t02 = tmp[2];
                                let t03 = tmp[3];
                                let t10 = tmp[4];
                                let t11 = tmp[5];
                                let t12 = tmp[6];
                                let t13 = tmp[7];

                                let y00 = t00 + t01 + t02;
                                let y01 = t01 - t02 - t03;
                                let y10 = t10 + t11 + t12;
                                let y11 = t11 - t12 - t13;

                                // Scatter write to output
                                let oh_base = th * 2;
                                let ow_base = tw * 2;

                                if oh_base < h_out && ow_base < w_out {
                                    out_plane[oh_base * w_out + ow_base] = y00;
                                }
                                if oh_base < h_out && ow_base + 1 < w_out {
                                    out_plane[oh_base * w_out + ow_base + 1] = y01;
                                }
                                if oh_base + 1 < h_out && ow_base < w_out {
                                    out_plane[(oh_base + 1) * w_out + ow_base] = y10;
                                }
                                if oh_base + 1 < h_out && ow_base + 1 < w_out {
                                    out_plane[(oh_base + 1) * w_out + ow_base + 1] = y11;
                                }
                            }
                        });
                });
        }

        // Backward pass: Use standard Im2Col gradient computation.
        // Winograd F(2x2, 3x3) is mathematically equivalent to standard conv2d,
        // so standard backward produces correct gradients for the forward result.

        let parents = vec![self.clone(), weight.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                // Using standard Im2Col backward pass logic.
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let weight = &parents[1];
                    // Use batch lock for better performance
                    let guards = TensorReadGuard::new(&[input, weight]);
                    let input_data = guards.get(0);
                    let weight_data = guards.get(1);

                    let (n, c_in, h_in, w_in) = (
                        input.shape[0],
                        input.shape[1],
                        input.shape[2],
                        input.shape[3],
                    );
                    let (c_out, _, k_h, k_w) = (
                        weight.shape[0],
                        weight.shape[1],
                        weight.shape[2],
                        weight.shape[3],
                    );

                    // Winograd F(2x2, 3x3) only supports stride=1.
                    // Stride and padding are captured; stride is hardcoded to 1.
                    let stride = 1;
                    let h_out = h_in + 2 * padding - 2;
                    let w_out = w_in + 2 * padding - 2; // k_h=3, k_w=3

                    // dL/dInput (Standard Col2Im)
                    {
                        let mut input_grad = input.grad_write_compat();
                        input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(
                            |(idx, in_plane)| {
                                let b = idx / c_in;
                                let c = idx % c_in;

                                for ih in 0..h_in {
                                    let oh_min = (ih + padding).saturating_sub(k_h - 1) / stride;
                                    let oh_max = ((ih + padding) / stride).min(h_out - 1);

                                    for iw in 0..w_in {
                                        let mut sum = 0.0;
                                        let ow_min =
                                            (iw + padding).saturating_sub(k_w - 1) / stride;
                                        let ow_max = ((iw + padding) / stride).min(w_out - 1);

                                        if oh_min <= oh_max && ow_min <= ow_max {
                                            for oh in oh_min..=oh_max {
                                                for ow in ow_min..=ow_max {
                                                    let kh = ih as isize - (oh * stride) as isize
                                                        + padding as isize;
                                                    let kw = iw as isize - (ow * stride) as isize
                                                        + padding as isize;

                                                    if kh >= 0
                                                        && kh < k_h as isize
                                                        && kw >= 0
                                                        && kw < k_w as isize
                                                    {
                                                        for k in 0..c_out {
                                                            let g = grad_out_f64[((b * c_out + k)
                                                                * h_out
                                                                + oh)
                                                                * w_out
                                                                + ow];
                                                            let w = weight_data[((k * c_in + c)
                                                                * k_h
                                                                + kh as usize)
                                                                * k_w
                                                                + kw as usize];
                                                            sum += g * w;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        in_plane[ih * w_in + iw] += sum;
                                    }
                                }
                            },
                        );
                    }

                    // dL/dWeight
                    {
                        let mut weight_grad = weight.grad_write_compat();
                        weight_grad.par_chunks_mut(k_h * k_w).enumerate().for_each(
                            |(idx, w_plane)| {
                                let k = idx / c_in;
                                let c = idx % c_in;

                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let mut sum = 0.0;
                                        for b in 0..n {
                                            for oh in 0..h_out {
                                                for ow in 0..w_out {
                                                    let h_in_idx = (oh * stride) as isize
                                                        - padding as isize
                                                        + kh as isize;
                                                    let w_in_idx = (ow * stride) as isize
                                                        - padding as isize
                                                        + kw as isize;

                                                    if h_in_idx >= 0
                                                        && h_in_idx < h_in as isize
                                                        && w_in_idx >= 0
                                                        && w_in_idx < w_in as isize
                                                    {
                                                        let val_in = input_data[((b * c_in + c)
                                                            * h_in
                                                            + h_in_idx as usize)
                                                            * w_in
                                                            + w_in_idx as usize];
                                                        let g_val = grad_out_f64[((b * c_out + k)
                                                            * h_out
                                                            + oh)
                                                            * w_out
                                                            + ow];
                                                        sum += val_in * g_val;
                                                    }
                                                }
                                            }
                                        }
                                        w_plane[kh * k_w + kw] += sum;
                                    }
                                }
                            },
                        );
                    }
                }),
            })),
        }
    }

    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        assert_eq!(weight.shape.len(), 4, "Weight must be 4D (OIHW)");

        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, c_in_k, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );

        assert_eq!(
            c_in, c_in_k,
            "Input channels must match weight input channels"
        );

        // Use Winograd F(2x2, 3x3) for 3x3 kernel with stride 1
        if k_h == 3 && k_w == 3 && stride == 1 {
            return self.winograd_conv2d_3x3(weight, padding);
        }

        let h_out = (h_in + 2 * padding - k_h) / stride + 1;
        let w_out = (w_in + 2 * padding - k_w) / stride + 1;

        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len: usize = out_shape.iter().product();
        let mut out_data = vec![0.0; out_len];

        let k_len = c_in * k_h * k_w;
        let out_plane_len = h_out * w_out;

        {
            let input_data = self.data_f64();
            let weight_data = weight.data_f64();

            // Standard Im2Col implementation. Memory hungry but fast.
            // Parallelize over Batch
            out_data
                .par_chunks_mut(c_out * out_plane_len)
                .enumerate()
                .for_each(|(b, out_batch)| {
                    // Im2Col: Input (C_in, H, W) -> Cols (K_len, Out_len)
                    let mut cols = vec![0.0; k_len * out_plane_len];

                    // Parallelize filling cols (by kernel rows)
                    cols.par_chunks_mut(out_plane_len)
                        .enumerate()
                        .for_each(|(k_idx, col_row)| {
                            let c = k_idx / (k_h * k_w);
                            let rem = k_idx % (k_h * k_w);
                            let kh = rem / k_w;
                            let kw = rem % k_w;

                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let h_in_idx =
                                        (oh * stride) as isize - padding as isize + kh as isize;
                                    let w_in_idx =
                                        (ow * stride) as isize - padding as isize + kw as isize;

                                    if h_in_idx >= 0
                                        && h_in_idx < h_in as isize
                                        && w_in_idx >= 0
                                        && w_in_idx < w_in as isize
                                    {
                                        col_row[oh * w_out + ow] =
                                            input_data[((b * c_in + c) * h_in + h_in_idx as usize)
                                                * w_in
                                                + w_in_idx as usize];
                                    }
                                }
                            }
                        });

                    // GEMM: Weight (C_out, K_len) * Cols (K_len, Out_len) -> Out (C_out, Out_len)
                    // out_batch is already slice of size C_out * Out_len

                    // Iterate over output rows (C_out)
                    out_batch
                        .par_chunks_mut(out_plane_len)
                        .enumerate()
                        .for_each(|(out_c, out_row)| {
                            // For each output channel, dot product weight row with all cols
                            // weight row start: out_c * k_len
                            let w_row_start = out_c * k_len;
                            let w_row = &weight_data[w_row_start..w_row_start + k_len];

                            for i in 0..out_plane_len {
                                let mut sum = 0.0;
                                // This inner loop is the hot path.
                                // Vectorization potential here.
                                for k in 0..k_len {
                                    sum += w_row[k] * cols[k * out_plane_len + i];
                                }
                                out_row[i] = sum;
                            }
                        });
                });
        }

        let parents = vec![self.clone(), weight.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let weight = &parents[1];
                    // Use batch lock for better performance
                    let guards = TensorReadGuard::new(&[input, weight]);
                    let input_data = guards.get(0);
                    let weight_data = guards.get(1);

                    // dL/dInput
                    {
                        let mut input_grad = input.grad_write_compat();
                        // Parallel over Input (N, C_in)
                        input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(
                            |(idx, in_plane)| {
                                let b = idx / c_in;
                                let c = idx % c_in;

                                // Optimized Col2Im (Transposed Conv)
                                for ih in 0..h_in {
                                    // Pre-calculate bounds to avoid inner loop checks
                                    let oh_min = (ih + padding).saturating_sub(k_h - 1) / stride;
                                    let oh_max = ((ih + padding) / stride).min(h_out - 1);

                                    for iw in 0..w_in {
                                        let mut sum = 0.0;
                                        let ow_min =
                                            (iw + padding).saturating_sub(k_w - 1) / stride;
                                        let ow_max = ((iw + padding) / stride).min(w_out - 1);

                                        // Check if range is valid (could be empty if padding is small/large)
                                        if oh_min <= oh_max && ow_min <= ow_max {
                                            for oh in oh_min..=oh_max {
                                                for ow in ow_min..=ow_max {
                                                    // ih = oh*s - p + kh => kh = ih - oh*s + p
                                                    let kh = ih as isize - (oh * stride) as isize
                                                        + padding as isize;
                                                    let kw = iw as isize - (ow * stride) as isize
                                                        + padding as isize;

                                                    if kh >= 0
                                                        && kh < k_h as isize
                                                        && kw >= 0
                                                        && kw < k_w as isize
                                                    {
                                                        // Should always be true given bounds, but stride check needed?
                                                        // If we iterate oh, ow, kh is determined.

                                                        for k in 0..c_out {
                                                            let g = grad_out_f64[((b * c_out + k)
                                                                * h_out
                                                                + oh)
                                                                * w_out
                                                                + ow];
                                                            let w = weight_data[((k * c_in + c)
                                                                * k_h
                                                                + kh as usize)
                                                                * k_w
                                                                + kw as usize];
                                                            sum += g * w;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        in_plane[ih * w_in + iw] += sum;
                                    }
                                }
                            },
                        );
                    }

                    // dL/dWeight
                    {
                        let mut weight_grad = weight.grad_write_compat();
                        // dWeight = grad_out * Input_Cols^T
                        // Implemented via manual accumulation over batch

                        // Direct loop is safer for memory.

                        // Parallel over Weight (C_out, C_in, KH, KW)
                        weight_grad.par_chunks_mut(k_h * k_w).enumerate().for_each(
                            |(idx, w_plane)| {
                                let k = idx / c_in;
                                let c = idx % c_in;

                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let mut sum = 0.0;
                                        for b in 0..n {
                                            for oh in 0..h_out {
                                                for ow in 0..w_out {
                                                    let h_in_idx = (oh * stride) as isize
                                                        - padding as isize
                                                        + kh as isize;
                                                    let w_in_idx = (ow * stride) as isize
                                                        - padding as isize
                                                        + kw as isize;

                                                    if h_in_idx >= 0
                                                        && h_in_idx < h_in as isize
                                                        && w_in_idx >= 0
                                                        && w_in_idx < w_in as isize
                                                    {
                                                        let val_in = input_data[((b * c_in + c)
                                                            * h_in
                                                            + h_in_idx as usize)
                                                            * w_in
                                                            + w_in_idx as usize];
                                                        let g_val = grad_out_f64[((b * c_out + k)
                                                            * h_out
                                                            + oh)
                                                            * w_out
                                                            + ow];
                                                        sum += val_in * g_val;
                                                    }
                                                }
                                            }
                                        }
                                        w_plane[kh * k_w + kw] += sum;
                                    }
                                }
                            },
                        );
                    }
                }),
            })),
        }
    }

    pub fn max_pool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);

        let h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

        let out_shape = vec![n, c, h_out, w_out];
        let out_len: usize = out_shape.iter().product();
        let mut out_data = vec![0.0; out_len];

        {
            let input_data = self.data_f64();
            // Parallelize over (N, C)
            out_data
                .par_chunks_mut(h_out * w_out)
                .enumerate()
                .for_each(|(idx, out_plane)| {
                    let b = idx / c;
                    let ch = idx % c;

                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let h_start = (oh * stride) as isize - padding as isize;
                            let w_start = (ow * stride) as isize - padding as isize;

                            let mut max_val = f64::NEG_INFINITY;

                            // Optimization: Optimized bounds
                            let kh_start = if h_start < 0 { (-h_start) as usize } else { 0 };
                            let kw_start = if w_start < 0 { (-w_start) as usize } else { 0 };
                            let kh_end = if h_start + kernel_size as isize > h_in as isize {
                                (h_in as isize - h_start) as usize
                            } else {
                                kernel_size
                            };
                            let kw_end = if w_start + kernel_size as isize > w_in as isize {
                                (w_in as isize - w_start) as usize
                            } else {
                                kernel_size
                            };

                            // Inner loops now guaranteed valid
                            for kh in kh_start..kh_end {
                                for kw in kw_start..kw_end {
                                    let h_in_idx = (h_start + kh as isize) as usize;
                                    let w_in_idx = (w_start + kw as isize) as usize;
                                    let val = input_data
                                        [((b * c + ch) * h_in + h_in_idx) * w_in + w_in_idx];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                            out_plane[oh * w_out + ow] = max_val;
                        }
                    }
                });
        }

        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut input_grad = input.grad_write_compat();

                    // Parallelize over Input (N, C)
                    input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(
                        |(idx, in_plane)| {
                            let b = idx / c;
                            let ch = idx % c;

                            for ih in 0..h_in {
                                for iw in 0..w_in {
                                    let mut grad_sum = 0.0;
                                    let val_in = input_data[((b * c + ch) * h_in + ih) * w_in + iw];

                                    // Determine possible output windows
                                    // ih = oh*s - p + kh  => oh*s = ih + p - kh
                                    // oh_min occurs when kh is max (k-1) -> oh*s = ih + p - (k-1)
                                    // oh_max occurs when kh is min (0)   -> oh*s = ih + p

                                    let oh_min =
                                        (ih + padding).saturating_sub(kernel_size - 1) / stride;
                                    let oh_max = ((ih + padding) / stride).min(h_out - 1);
                                    let ow_min =
                                        (iw + padding).saturating_sub(kernel_size - 1) / stride;
                                    let ow_max = ((iw + padding) / stride).min(w_out - 1);

                                    if oh_min <= oh_max && ow_min <= ow_max {
                                        for oh in oh_min..=oh_max {
                                            for ow in ow_min..=ow_max {
                                                // Check stride alignment effectively handled by division/range but:
                                                // We need to check if ih is actually in the window for this oh.
                                                // The range calculation above is necessary but not sufficient if stride > 1?
                                                // Actually integer division handles "floor".
                                                // Let's verify: oh*s <= ih+p < oh*s + k
                                                // oh*s - p <= ih < oh*s - p + k

                                                let h_start =
                                                    (oh * stride) as isize - padding as isize;
                                                let w_start =
                                                    (ow * stride) as isize - padding as isize;

                                                if (ih as isize) >= h_start
                                                    && (ih as isize)
                                                        < h_start + kernel_size as isize
                                                    && (iw as isize) >= w_start
                                                    && (iw as isize)
                                                        < w_start + kernel_size as isize
                                                {
                                                    // Re-find max
                                                    let mut max_val = f64::NEG_INFINITY;

                                                    // Optimized bounds for inner search
                                                    let kh_start = if h_start < 0 {
                                                        (-h_start) as usize
                                                    } else {
                                                        0
                                                    };
                                                    let kw_start = if w_start < 0 {
                                                        (-w_start) as usize
                                                    } else {
                                                        0
                                                    };
                                                    let kh_end = if h_start + kernel_size as isize
                                                        > h_in as isize
                                                    {
                                                        (h_in as isize - h_start) as usize
                                                    } else {
                                                        kernel_size
                                                    };
                                                    let kw_end = if w_start + kernel_size as isize
                                                        > w_in as isize
                                                    {
                                                        (w_in as isize - w_start) as usize
                                                    } else {
                                                        kernel_size
                                                    };

                                                    for kh in kh_start..kh_end {
                                                        for kw in kw_start..kw_end {
                                                            let h_k =
                                                                (h_start + kh as isize) as usize;
                                                            let w_k =
                                                                (w_start + kw as isize) as usize;
                                                            let v = input_data[((b * c + ch)
                                                                * h_in
                                                                + h_k)
                                                                * w_in
                                                                + w_k];
                                                            if v > max_val {
                                                                max_val = v;
                                                            }
                                                        }
                                                    }

                                                    if (val_in - max_val).abs() < 1e-6 {
                                                        grad_sum += grad_out_f64[((b * c + ch)
                                                            * h_out
                                                            + oh)
                                                            * w_out
                                                            + ow];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    in_plane[ih * w_in + iw] += grad_sum;
                                }
                            }
                        },
                    );
                }),
            })),
        }
    }

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
                    // f = total / w_sum
                    // df/d(total) = 1 / w_sum
                    let mut total_grad = parents[0].grad_write_compat();
                    total_grad[0] += grad_out_f64[0] / denom_cap;
                    drop(total_grad);
                    // df/d(w_sum) = -total / w_sum^2 = -f / w_sum
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

    // Generic element-wise binary ops for non-F64 dtypes.
    // -------------------------------------------------------------------------

    pub(super) fn assert_same_numel(&self, rhs: &Tensor, op: &'static str) {
        assert_eq!(
            self.numel(),
            rhs.numel(),
            "{} data length mismatch: left shape {:?} ({} elems), right shape {:?} ({} elems)",
            op,
            self.shape,
            self.numel(),
            rhs.shape,
            rhs.numel()
        );
    }

    pub(super) fn add_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "add_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] + rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        let mut grad = _parents[0].grad_write_compat();
                        for i in 0..out_len {
                            grad[i] += grad_out_f64[i] * 2.0;
                        }
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i];
                            rhs_grad[i] += grad_out_f64[i];
                        }
                    }
                }),
            })),
        }
    }

    pub(super) fn sub_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "sub_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] - rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        // d/dx (x - x) = 0, no-op
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i];
                            rhs_grad[i] -= grad_out_f64[i];
                        }
                    }
                }),
            })),
        }
    }

    pub(super) fn mul_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "mul_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] * rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        let rhs_cache: Arc<Vec<f64>> = Arc::new(rhs_f32.iter().map(|&v| v as f64).collect());
        let self_cache: Arc<Vec<f64>> = Arc::new(self_f32.iter().map(|&v| v as f64).collect());
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        let mut grad = _parents[0].grad_write_compat();
                        for i in 0..out_len {
                            grad[i] += grad_out_f64[i] * 2.0 * rhs_cache[i];
                        }
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i] * rhs_cache[i];
                            rhs_grad[i] += grad_out_f64[i] * self_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    pub(super) fn div_generic(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "div_generic");
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let out_len = self_f32.len();
        let mut data = vec![0.0f32; out_len];
        for i in 0..out_len {
            data[i] = self_f32[i] / rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        let rhs_cache: Arc<Vec<f64>> = Arc::new(rhs_f32.iter().map(|&v| v as f64).collect());
        let self_cache: Arc<Vec<f64>> = Arc::new(self_f32.iter().map(|&v| v as f64).collect());
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let same_grad = _parents[0].grad.ptr_eq(&_parents[1].grad);
                    if same_grad {
                        // d/dx (x/x) = 0, no-op
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_compat();
                        let mut rhs_grad = _parents[1].grad_write_compat();
                        for i in 0..out_len {
                            lhs_grad[i] += grad_out_f64[i] / rhs_cache[i];
                            rhs_grad[i] +=
                                grad_out_f64[i] * (-self_cache[i] / (rhs_cache[i] * rhs_cache[i]));
                        }
                    }
                }),
            })),
        }
    }
}

#[cfg(cuda)]
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

    fn cuda_remove_cached_buffer(&self) {
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

    fn cuda_grad_cached_buffer(&self) -> Option<Arc<crate::cuda::memory::CudaBuffer>> {
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

    fn cuda_grad_remove_cached_buffer(&self) {
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
        use crate::cuda::memory::{alloc, copy_h2d, CudaBuffer};

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
                let device = match alloc::<f32>(len) {
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
                let device = match alloc::<f64>(len) {
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
                let zeros = vec![0.0f32; len];
                let _ = crate::cuda::memory::copy_h2d(b, &zeros);
            }
            crate::cuda::memory::CudaBuffer::F64(b) => {
                let zeros = vec![0.0f64; len];
                let _ = crate::cuda::memory::copy_h2d(b, &zeros);
            }
            crate::cuda::memory::CudaBuffer::BF16(_) | crate::cuda::memory::CudaBuffer::I8(_) => {}
        }
        Some(buffer)
    }

    /// Materialize GPU data to CPU if this tensor lives on GPU but has empty CPU data.
    #[cfg(cuda)]
    fn cuda_materialize(&self) {
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
        use crate::cuda::memory::{alloc, copy_h2d, CudaBuffer};

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
                if host_len > 0 {
                    match (&*buffer, self.dtype) {
                        (CudaBuffer::BF16(b), Dtype::BF16) => {
                            let host = self.data_bf16();
                            if let Err(err) = copy_h2d(b, &host) {
                                self.cuda_remove_cached_buffer();
                                return Err(("copy", err));
                            }
                        }
                        (CudaBuffer::I8(b), Dtype::I8) => {
                            if let Storage::I8(v) = &self.data {
                                let host = v.read().unwrap();
                                if let Err(err) = copy_h2d(b, &host) {
                                    self.cuda_remove_cached_buffer();
                                    return Err(("copy", err));
                                }
                            }
                        }
                        (CudaBuffer::F32(b), Dtype::F32) => {
                            let host = self.data_f32();
                            if let Err(err) = copy_h2d(b, &host) {
                                self.cuda_remove_cached_buffer();
                                return Err(("copy", err));
                            }
                        }
                        (CudaBuffer::F64(b), Dtype::F64) => {
                            let host = self.data_f64();
                            if let Err(err) = copy_h2d(b, &host) {
                                self.cuda_remove_cached_buffer();
                                return Err(("copy", err));
                            }
                        }
                        _ => {
                            self.cuda_remove_cached_buffer();
                        }
                    }
                }
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
                let device = match alloc::<crate::dtype::bf16>(len) {
                    Ok(buf) => buf,
                    Err(err) => return Err(("alloc", err)),
                };
                if let Err(err) = copy_h2d(&device, &host) {
                    return Err(("copy", err));
                }
                CudaBuffer::BF16(device)
            }
            Dtype::I8 => {
                let device = match alloc::<i8>(len) {
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
                let device = match alloc::<f32>(len) {
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
                let device = match alloc::<f64>(len) {
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

    /// GPU-accelerated matrix multiplication using CUDA cuBLAS
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn matmul_cuda(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
        use crate::cuda::blas::gemm_thread_local;
        use crate::cuda::memory::alloc;

        crate::cuda::record_matmul_attempt();

        let d_a = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, _err)) => {
                crate::cuda::record_matmul_fallback(stage);
                return self.matmul_cpu_fallback(other, m, k, n);
            }
        };
        let d_b = match other.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, _err)) => {
                crate::cuda::record_matmul_fallback(stage);
                return self.matmul_cpu_fallback(other, m, k, n);
            }
        };

        let out_dtype = Tensor::binary_dtype(self.dtype, other.dtype);
        let supported_forward = matches!(
            (self.dtype, other.dtype),
            (Dtype::F32, Dtype::F32) | (Dtype::F64, Dtype::F64) | (Dtype::BF16, Dtype::BF16)
        );
        if !supported_forward {
            crate::cuda::record_matmul_fallback("gemm");
            return self.matmul_cpu_fallback(other, m, k, n);
        }
        let compute_out_dtype = if self.dtype == Dtype::BF16 && other.dtype == Dtype::BF16 {
            Dtype::F32
        } else {
            out_dtype
        };
        let grad_dtype = Tensor::grad_dtype_for(compute_out_dtype);

        let d_c = match compute_out_dtype {
            Dtype::F32 => match alloc::<f32>(m * n) {
                Ok(buf) => crate::cuda::memory::CudaBuffer::F32(buf),
                Err(_err) => {
                    crate::cuda::record_matmul_fallback("alloc");
                    return self.matmul_cpu_fallback(other, m, k, n);
                }
            },
            _ => match alloc::<f64>(m * n) {
                Ok(buf) => crate::cuda::memory::CudaBuffer::F64(buf),
                Err(_err) => {
                    crate::cuda::record_matmul_fallback("alloc");
                    return self.matmul_cpu_fallback(other, m, k, n);
                }
            },
        };
        let d_c = Arc::new(d_c);
        let (m_i32, n_i32, k_i32) =
            match (i32::try_from(m), i32::try_from(n), i32::try_from(k)) {
                (Ok(mv), Ok(nv), Ok(kv)) => (mv, nv, kv),
                _ => {
                    crate::cuda::record_matmul_fallback("gemm");
                    log::warn!(
                    "[Autograd] CUDA GEMM dimensions overflow i32 (m={}, n={}, k={}), using CPU",
                    m, n, k
                );
                    return self.matmul_cpu_fallback(other, m, k, n);
                }
            };

        let gemm_ok = match (self.dtype, other.dtype, compute_out_dtype) {
            (Dtype::BF16, Dtype::BF16, Dtype::F32) => {
                crate::cuda::blas::gemm_thread_local_bf16_to_f32(
                    false,
                    false,
                    m_i32,
                    n_i32,
                    k_i32,
                    1.0f32,
                    d_a.as_raw(),
                    k_i32,
                    d_b.as_raw(),
                    n_i32,
                    0.0f32,
                    d_c.as_raw(),
                    n_i32,
                )
                .is_ok()
            }
            (Dtype::F32, Dtype::F32, Dtype::F32) => crate::cuda::blas::gemm_thread_local_f32(
                false,
                false,
                m_i32,
                n_i32,
                k_i32,
                1.0f32,
                d_a.as_raw(),
                k_i32,
                d_b.as_raw(),
                n_i32,
                0.0f32,
                d_c.as_raw(),
                n_i32,
            )
            .is_ok(),
            (Dtype::F64, Dtype::F64, Dtype::F64) => gemm_thread_local(
                false,
                false,
                m_i32,
                n_i32,
                k_i32,
                1.0,
                d_a.as_raw(),
                k_i32,
                d_b.as_raw(),
                n_i32,
                0.0,
                d_c.as_raw(),
                n_i32,
            )
            .is_ok(),
            _ => false,
        };

        if !gemm_ok {
            crate::cuda::record_matmul_fallback("gemm");
            return self.matmul_cpu_fallback(other, m, k, n);
        }

        crate::cuda::record_matmul_success();

        let out_shape = if self.shape.len() == 1 {
            vec![n]
        } else {
            vec![m, n]
        };
        let parents = vec![self.clone(), other.clone()];
        let out = Tensor {
            data: Tensor::empty_storage(compute_out_dtype),
            grad: Storage::zeros(m * n, grad_dtype),
            shape: out_shape,
            device: Device::Cuda,
            dtype: compute_out_dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];

                    #[cfg(cuda)]
                    if lhs.device == Device::Cuda && rhs.device == Device::Cuda {
                        crate::cuda::record_backward_attempt();
                        let mut gpu_backward_ok = false;
                        if let (Some(d_lhs), Some(d_rhs)) =
                            (lhs.cuda_cached_buffer(), rhs.cuda_cached_buffer())
                        {
                            let grad_dtype = grad_out.dtype();
                            let d_grad_tmp = cuda_grad_out_buffer(grad_out);
                            if let Some(d_grad_tmp) = d_grad_tmp {
                                let lhs_ok = if let Some(d_lhs_grad) = lhs.cuda_grad_ensure_buffer()
                                {
                                    match (&*d_grad_tmp, &*d_rhs, &*d_lhs_grad, grad_dtype) {
                                        (
                                            crate::cuda::memory::CudaBuffer::F32(gt),
                                            crate::cuda::memory::CudaBuffer::F32(r),
                                            crate::cuda::memory::CudaBuffer::F32(lg),
                                            Dtype::F32,
                                        ) => crate::cuda::blas::gemm_thread_local_f32(
                                            false,
                                            true,
                                            m as i32,
                                            k as i32,
                                            n as i32,
                                            1.0f32,
                                            gt.as_raw(),
                                            n as i32,
                                            r.as_raw(),
                                            n as i32,
                                            1.0f32,
                                            lg.as_raw(),
                                            k as i32,
                                        )
                                        .is_ok(),
                                        (
                                            crate::cuda::memory::CudaBuffer::F64(gt),
                                            crate::cuda::memory::CudaBuffer::F64(r),
                                            crate::cuda::memory::CudaBuffer::F64(lg),
                                            Dtype::F64,
                                        ) => gemm_thread_local(
                                            false,
                                            true,
                                            m as i32,
                                            k as i32,
                                            n as i32,
                                            1.0,
                                            gt.as_raw(),
                                            n as i32,
                                            r.as_raw(),
                                            n as i32,
                                            1.0,
                                            lg.as_raw(),
                                            k as i32,
                                        )
                                        .is_ok(),
                                        _ => false,
                                    }
                                } else {
                                    false
                                };

                                let rhs_ok = if let Some(d_rhs_grad) = rhs.cuda_grad_ensure_buffer()
                                {
                                    match (&*d_grad_tmp, &*d_lhs, &*d_rhs_grad, grad_dtype) {
                                        (
                                            crate::cuda::memory::CudaBuffer::F32(gt),
                                            crate::cuda::memory::CudaBuffer::F32(l),
                                            crate::cuda::memory::CudaBuffer::F32(rg),
                                            Dtype::F32,
                                        ) => crate::cuda::blas::gemm_thread_local_f32(
                                            true,
                                            false,
                                            k as i32,
                                            n as i32,
                                            m as i32,
                                            1.0f32,
                                            l.as_raw(),
                                            k as i32,
                                            gt.as_raw(),
                                            n as i32,
                                            1.0f32,
                                            rg.as_raw(),
                                            n as i32,
                                        )
                                        .is_ok(),
                                        (
                                            crate::cuda::memory::CudaBuffer::F64(gt),
                                            crate::cuda::memory::CudaBuffer::F64(l),
                                            crate::cuda::memory::CudaBuffer::F64(rg),
                                            Dtype::F64,
                                        ) => gemm_thread_local(
                                            true,
                                            false,
                                            k as i32,
                                            n as i32,
                                            m as i32,
                                            1.0,
                                            l.as_raw(),
                                            k as i32,
                                            gt.as_raw(),
                                            n as i32,
                                            1.0,
                                            rg.as_raw(),
                                            n as i32,
                                        )
                                        .is_ok(),
                                        _ => false,
                                    }
                                } else {
                                    false
                                };

                                if lhs_ok && rhs_ok {
                                    gpu_backward_ok = true;
                                }
                            }
                        }
                        if gpu_backward_ok {
                            crate::cuda::record_backward_success();
                            return;
                        } else {
                            crate::cuda::record_backward_fallback();
                        }
                    }

                    // CPU backward path
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);

                    {
                        let mut lhs_grad = lhs.grad_write_compat();
                        for r in 0..m {
                            for i in 0..k {
                                lhs_grad[r * k + i] += dot_product(
                                    &grad_out_f64[r * n..r * n + n],
                                    &rhs_data[i * n..i * n + n],
                                );
                            }
                        }
                    }

                    {
                        let mut rhs_grad = rhs.grad_write_compat();
                        for i in 0..k {
                            for j in 0..n {
                                for r in 0..m {
                                    rhs_grad[i * n + j] +=
                                        lhs_data[r * k + i] * grad_out_f64[r * n + j];
                                }
                            }
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_c);
        out
    }

    /// CPU fallback for matmul
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn matmul_cpu_fallback(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
        let out_dtype = Tensor::binary_dtype(self.dtype, other.dtype);
        if out_dtype == Dtype::F32 {
            return self.matmul_generic(other, m, k, n, out_dtype);
        }
        let mut out_data = vec![0.0; m * n];
        let guards = TensorReadGuard::new(&[self, other]);
        let lhs_data = guards.get(0);
        let rhs_data = guards.get(1);

        for r in 0..m {
            let out_row = &mut out_data[r * n..(r + 1) * n];
            for i in 0..k {
                let scale = lhs_data[r * k + i];
                if scale == 0.0 {
                    continue;
                }
                let rhs_row = &rhs_data[i * n..(i + 1) * n];
                add_scaled_row(out_row, rhs_row, scale);
            }
        }

        let out_shape = if self.shape.len() == 1 {
            vec![n]
        } else {
            vec![m, n]
        };
        let parents = vec![self.clone(), other.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(m * n, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);

                    {
                        // dL/dLHS
                        let mut lhs_grad = lhs.grad_write_compat();
                        for r in 0..m {
                            for i in 0..k {
                                lhs_grad[r * k + i] += dot_product(
                                    &grad_out_f64[r * n..r * n + n],
                                    &rhs_data[i * n..i * n + n],
                                );
                            }
                        }
                    }

                    {
                        // dL/dRHS
                        let mut rhs_grad = rhs.grad_write_compat();
                        for i in 0..k {
                            for j in 0..n {
                                for r in 0..m {
                                    rhs_grad[i * n + j] +=
                                        lhs_data[r * k + i] * grad_out_f64[r * n + j];
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }

    /// GPU-accelerated GELU activation using CUDA kernel
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn gelu_cuda(&self) -> Tensor {
        use crate::cuda::kernels::{
            gelu_backward, gelu_backward_f32, gelu_inplace, gelu_inplace_f32,
        };
        use crate::cuda::memory::{alloc, copy_d2d, CudaBuffer};

        let len = match self.dtype {
            Dtype::F32 | Dtype::F64 => self.numel(),
            _ => return self.gelu_cpu_fallback(),
        };
        if len == 0 {
            return self.gelu_cpu_fallback();
        }
        crate::cuda::record_activation_attempt();

        let d_src = match self.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((stage, err)) => {
                crate::cuda::record_activation_fallback(stage);
                log::warn!(
                    "[Autograd] CUDA prepare GELU input failed ({}), using CPU",
                    err
                );
                return self.gelu_cpu_fallback();
            }
        };
        let d_data = match self.dtype {
            Dtype::F32 => match alloc::<f32>(len) {
                Ok(buf) => CudaBuffer::F32(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc GELU buffer failed ({}), using CPU",
                        err
                    );
                    return self.gelu_cpu_fallback();
                }
            },
            Dtype::F64 => match alloc::<f64>(len) {
                Ok(buf) => CudaBuffer::F64(buf),
                Err(err) => {
                    crate::cuda::record_activation_fallback("alloc");
                    log::warn!(
                        "[Autograd] CUDA alloc GELU buffer failed ({}), using CPU",
                        err
                    );
                    return self.gelu_cpu_fallback();
                }
            },
            _ => return self.gelu_cpu_fallback(),
        };
        let copy_ok = match (self.dtype, &d_data, &*d_src) {
            (Dtype::F32, CudaBuffer::F32(dst), CudaBuffer::F32(src)) => copy_d2d(dst, src).is_ok(),
            (Dtype::F64, CudaBuffer::F64(dst), CudaBuffer::F64(src)) => copy_d2d(dst, src).is_ok(),
            _ => false,
        };
        if !copy_ok {
            crate::cuda::record_activation_fallback("copy");
            log::warn!("[Autograd] CUDA D2D GELU input copy failed, using CPU");
            return self.gelu_cpu_fallback();
        }

        let kernel_ok = match (self.dtype, &d_data) {
            (Dtype::F32, CudaBuffer::F32(b)) => gelu_inplace_f32(b).is_ok(),
            (Dtype::F64, CudaBuffer::F64(b)) => gelu_inplace(b).is_ok(),
            _ => false,
        };
        if !kernel_ok {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA GELU kernel failed, using CPU");
            return self.gelu_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
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
                    #[cfg(cuda)]
                    if input.device == Device::Cuda {
                        if let Some(d_input) = input.cuda_cached_buffer() {
                            let d_grad_tmp = cuda_grad_out_buffer(grad_out);
                            if let Some(d_grad_tmp) = d_grad_tmp {
                                if let Some(d_in_grad) = input.cuda_grad_ensure_buffer() {
                                    match (&*d_input, &*d_grad_tmp, &*d_in_grad, input.dtype) {
                                        (
                                            CudaBuffer::F32(inp),
                                            CudaBuffer::F32(gt),
                                            CudaBuffer::F32(ig),
                                            Dtype::F32,
                                        ) => {
                                            if gelu_backward_f32(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        (
                                            CudaBuffer::F64(inp),
                                            CudaBuffer::F64(gt),
                                            CudaBuffer::F64(ig),
                                            Dtype::F64,
                                        ) => {
                                            if gelu_backward(inp, gt, ig, gt.len()).is_ok() {
                                                return;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                    let c = 0.044715;
                    for i in 0..inp_grad.len() {
                        let x = input_data[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = sqrt_2_over_pi * (x + c * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out_f64[i] * gelu_grad;
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_data);
        out
    }

    /// CPU fallback for GELU
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn gelu_cpu_fallback(&self) -> Tensor {
        if self.dtype == Dtype::F32 {
            return self.gelu_generic();
        }
        let self_data = self.data_f64();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        for i in 0..len {
            let x = self_data[i];
            let x3 = x * x * x;
            let inner = 0.7978845608028654 * (x + 0.044715 * x3);
            data[i] = 0.5 * x * (1.0 + inner.tanh());
        }
        let parents = vec![self.clone()];
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let c = 0.044715;

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
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut inp_grad = input.grad_write_compat();
                    for i in 0..inp_grad.len() {
                        let x = input_data[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = sqrt_2_over_pi * (x + c * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out_f64[i] * gelu_grad;
                    }
                }),
            })),
        }
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
