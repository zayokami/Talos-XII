use crate::autograd::{Device, GradWriteCompat, Tensor};
use crate::dtype::{Dtype, Storage};
use memmap2::Mmap;
use std::fs::File;
use std::sync::{Arc, RwLock};

#[cfg(cuda)]
use super::cuda_sync_grad_to_host;

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
}
