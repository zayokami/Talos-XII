use crate::dtype::{Dtype, Storage};
use crate::simd::{
    add_scaled_row, dot_product, horizontal_sum, prefetch_read_l1, softmax_exp_sum, vector_add,
    vector_fma, vector_gelu, vector_grad_acc, vector_mul, vector_relu, vector_sub,
};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fs::File;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, RwLock};
#[cfg(cuda)]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

// --- Device enumeration for CPU/GPU placement ---

/// Device where tensor data resides
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Device {
    /// CPU (default)
    #[default]
    Cpu,
    /// CUDA GPU (when cuda feature is enabled)
    #[cfg(cuda)]
    Cuda,
}

// --- Autograd Engine ---

// Minimum element count to justify Rayon parallel dispatch.
// Below this, serial iteration is faster due to scheduling overhead.
const PAR_THRESHOLD: usize = 4096;

#[cfg(cuda)]
type CudaTensorBufferMap = HashMap<usize, Arc<crate::cuda::memory::DevicePtr<f64>>>;
#[cfg(cuda)]
static CUDA_TENSOR_BUFFER_CACHE: OnceLock<Mutex<CudaTensorBufferMap>> = OnceLock::new();

#[cfg(cuda)]
fn cuda_tensor_buffer_cache() -> &'static Mutex<CudaTensorBufferMap> {
    CUDA_TENSOR_BUFFER_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Storage, // Typed storage (F64, F32, BF16, I8)
    pub grad: Storage, // Typed gradients
    pub shape: Vec<usize>,
    pub device: Device,             // Device where tensor resides
    pub dtype: Dtype,               // Element data type
    pub _ctx: Option<Arc<Context>>, // Keeps the graph alive
}

#[cfg(cuda)]
impl Drop for Tensor {
    fn drop(&mut self) {
        // Remove cache entry when this is the last owner of tensor data.
        if Arc::strong_count(&self.data) == 1 {
            let key = Arc::as_ptr(&self.data) as usize;
            if let Some(cache) = CUDA_TENSOR_BUFFER_CACHE.get() {
                if let Ok(mut map) = cache.lock() {
                    map.remove(&key);
                }
            }
        }
    }
}

/// Batch lock acquisition helper for reducing lock overhead in parallel operations
pub struct TensorReadGuard<'a> {
    guards: Vec<std::sync::RwLockReadGuard<'a, Vec<f64>>>,
}

impl<'a> TensorReadGuard<'a> {
    /// Acquire read locks for multiple tensors at once.
    /// If any lock is poisoned, logs a warning and skips that tensor's data
    /// (the guard will contain `None` for that index and `get` will panic).
    pub fn new(tensors: &[&'a Tensor]) -> Self {
        let guards: Vec<_> = tensors
            .iter()
            .map(|t| {
                t.data_read_safe().unwrap_or_else(|| {
                    log::warn!(
                        target: "resilience",
                        "TensorReadGuard: data lock poisoned, proceeding with possibly stale data"
                    );
                    // We still need a guard. In the poisoned case we recover the inner data
                    // by ignoring the poison error. This is safe for neural-network weights.
                    t.data_f64_poison_resilient()
                })
            })
            .collect();
        TensorReadGuard { guards }
    }

    /// Get data by index
    pub fn get(&self, idx: usize) -> &Vec<f64> {
        &self.guards[idx]
    }
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let data = self.data_f64();
        let mut state = serializer.serialize_struct("Tensor", 3)?;
        state.serialize_field("data", &*data)?;
        state.serialize_field("shape", &self.shape)?;
        state.serialize_field("dtype", &self.dtype)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TensorData {
            data: Vec<f64>,
            shape: Vec<usize>,
            #[serde(default)]
            dtype: Dtype,
        }

        let helper = TensorData::deserialize(deserializer)?;
        Ok(Tensor::with_dtype(helper.data, helper.shape, helper.dtype))
    }
}

type BackwardOp = Box<dyn Fn(&Storage, &Vec<Tensor>) + Send + Sync>;

pub struct Context {
    pub parents: Vec<Tensor>,
    pub backward_op: BackwardOp, // receives grad_output Storage, parents
}

/// Write guard that transparently handles F64 and F32 gradients.
/// For F64 grad, writes directly. For F32 grad, accumulates into a temporary
/// f64 buffer and flushes back to F32 storage on Drop.
pub enum GradWriteCompat<'a> {
    F64(std::sync::RwLockWriteGuard<'a, Vec<f64>>),
    F32Buffer(Vec<f64>, &'a Storage),
}

impl<'a> std::ops::Deref for GradWriteCompat<'a> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        match self {
            GradWriteCompat::F64(g) => g,
            GradWriteCompat::F32Buffer(buf, _) => buf,
        }
    }
}

impl<'a> std::ops::DerefMut for GradWriteCompat<'a> {
    fn deref_mut(&mut self) -> &mut [f64] {
        match self {
            GradWriteCompat::F64(g) => g,
            GradWriteCompat::F32Buffer(buf, _) => buf,
        }
    }
}

impl<'a> Drop for GradWriteCompat<'a> {
    fn drop(&mut self) {
        if let GradWriteCompat::F32Buffer(buf, storage) = self {
            storage.accumulate_f64_slice(buf);
        }
    }
}

impl Tensor {
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
        let len = data.len();
        assert_eq!(
            len,
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: None,
        }
    }

    /// Create an F32 tensor from f64 data (auto-converts to f32).
    pub fn new_f32(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::with_dtype(data, shape, Dtype::F32)
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
        {
            let mut data = self.data_write_f64();
            for d in data.iter_mut() {
                *d = value;
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
        match &self.data {
            Storage::F64(v) => v.read().unwrap(),
            _ => panic!("Expected F64 storage, got {:?}", self.dtype),
        }
    }

    /// Write F64 data lock. Panics if not F64 or poisoned.
    #[inline]
    pub fn data_write_f64(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f64>> {
        match &self.data {
            Storage::F64(v) => v.write().unwrap(),
            _ => panic!("Expected F64 storage, got {:?}", self.dtype),
        }
    }

    /// Write F32 data lock. Panics if not F32 or poisoned.
    #[inline]
    pub fn data_write_f32(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f32>> {
        match &self.data {
            Storage::F32(v) => v.write().unwrap(),
            _ => panic!("Expected F32 storage, got {:?}", self.dtype),
        }
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
        self.data.to_f64_vec()
    }

    /// Read data as Vec<f32>, converting from native dtype as needed.
    #[inline]
    pub fn data_to_f32_vec(&self) -> Vec<f32> {
        self.data.to_f32_vec()
    }

    /// Output dtype for a binary op: match if same, otherwise promote to F64.
    #[inline]
    pub fn binary_dtype(a: Dtype, b: Dtype) -> Dtype {
        if a == b {
            a
        } else {
            Dtype::F64
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
        match &self.grad {
            Storage::F32(v) => v.read().unwrap(),
            _ => panic!("Expected F32 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Write F32 grad lock. Panics if grad is not F32.
    #[inline]
    pub fn grad_write_f32(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f32>> {
        match &self.grad {
            Storage::F32(v) => v.write().unwrap(),
            _ => panic!("Expected F32 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Read F64 grad lock. Panics if grad is not F64.
    #[inline]
    pub fn grad_read_f64(&self) -> std::sync::RwLockReadGuard<'_, Vec<f64>> {
        match &self.grad {
            Storage::F64(v) => v.read().unwrap(),
            _ => panic!("Expected F64 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Write F64 grad lock. Panics if grad is not F64.
    #[inline]
    pub fn grad_write_f64(&self) -> std::sync::RwLockWriteGuard<'_, Vec<f64>> {
        match &self.grad {
            Storage::F64(v) => v.write().unwrap(),
            _ => panic!("Expected F64 grad, got {:?}", self.grad.dtype()),
        }
    }

    /// Convert grad to Vec<f32>.
    #[inline]
    pub fn grad_to_f32_vec(&self) -> Vec<f32> {
        self.grad.to_f32_vec()
    }

    /// Convert grad to Vec<f64>.
    #[inline]
    pub fn grad_to_f64_vec(&self) -> Vec<f64> {
        self.grad.to_f64_vec()
    }

    /// Accumulate f32 slice into grad.
    #[inline]
    pub fn grad_accumulate_f32(&self, slice: &[f32]) {
        self.grad.accumulate_f32(slice);
    }

    /// Accumulate f64 slice into grad.
    #[inline]
    pub fn grad_accumulate_f64(&self, slice: &[f64]) {
        self.grad.accumulate_f64(slice);
    }

    /// Get a write-compatible gradient guard.
    /// For F64 grad, returns direct RwLockWriteGuard.
    /// For F32 grad, returns a temporary f64 buffer that flushes back on Drop.
    #[inline]
    pub fn grad_write_compat(&self) -> GradWriteCompat<'_> {
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

    /// Generate a tensor with normally distributed random values (mean=0, std=1).
    pub fn randn(shape: Vec<usize>, seed: u64) -> Self {
        use crate::rng::Rng;
        let len = shape.iter().product::<usize>();
        let mut rng = Rng::from_seed(seed);
        let data: Vec<f64> = (0..len).map(|_| rng.next_f64_normal()).collect();
        Tensor::new(data, shape)
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
    pub fn item(&self) -> f64 {
        assert_eq!(self.shape.iter().product::<usize>(), 1);
        self.data_f64()[0]
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
        self.grad.fill_f64(1.0);
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            if let Some(d_grad) = self.cuda_grad_ensure_buffer() {
                if d_grad.len() > 0 {
                    let ones = vec![1.0_f64; d_grad.len()];
                    let _ = crate::cuda::memory::copy_h2d(&d_grad, &ones);
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
        self.grad.zero();
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            if let Some(d_grad) = self.cuda_grad_ensure_buffer() {
                // cuda_grad_ensure_buffer already zeros the buffer
                let _ = d_grad;
            }
        }
    }

    /// Copy tensor data to CUDA GPU.
    /// Current implementation keeps data on host while marking device intent.
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub fn to_cuda(&self) -> crate::cuda::error::CudaResult<Tensor> {
        if let Err(err) = crate::cuda::init() {
            log::error!("[Tensor] CUDA runtime unavailable: {err}");
            return Err(err);
        }

        let tensor = Tensor {
            data: self.data.clone(),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
            device: Device::Cuda,
            _ctx: self._ctx.clone(),
        };

        let len = tensor.data_f64().len();
        if len > 0 {
            if let Err((_, err)) = tensor.cuda_get_or_upload_buffer() {
                log::warn!("[Tensor] CUDA upload failed: {err}");
                return Err(err);
            }
        }

        Ok(tensor)
    }

    /// Copy tensor data from CUDA GPU back to CPU.
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub fn from_cuda(&self) -> crate::cuda::error::CudaResult<Vec<f64>> {
        if self.device != Device::Cuda {
            return Err(crate::cuda::error::CudaError::InvalidInput {
                op: "Tensor::from_cuda",
                message: "tensor is not on CUDA device",
            });
        }

        if let Some(buffer) = self.cuda_cached_buffer() {
            let mut host = vec![0.0; buffer.len()];
            crate::cuda::memory::copy_d2h(&mut host, &buffer)?;
            return Ok(host);
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

        // GPU routing: use CUDA if both tensors are on GPU and large enough
        #[cfg(cuda)]
        {
            let ops = m * n * k;
            let use_gpu =
                ops >= 32768 && self.device == Device::Cuda && other.device == Device::Cuda;
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

        let lhs_cache = Arc::new(lhs_f32);
        let rhs_cache = Arc::new(rhs_f32);

        Tensor {
            data: Storage::from_f32_vec(out_data, out_dtype),
            grad: Storage::zeros(m * n, Tensor::grad_dtype_for(out_dtype)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), other.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    // dL/dLHS = grad_out * RHS^T
                    let mut lhs_grad = _parents[0].grad_write_f32();
                    for r in 0..m {
                        for i in 0..k {
                            let mut sum = 0.0f32;
                            for j in 0..n {
                                sum += grad_out_f32[r * n + j] * rhs_cache[i * n + j];
                            }
                            lhs_grad[r * k + i] += sum;
                        }
                    }

                    // dL/dRHS = LHS^T * grad_out
                    let mut rhs_grad = _parents[1].grad_write_f32();
                    for i in 0..k {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for r in 0..m {
                                sum += lhs_cache[r * k + i] * grad_out_f32[r * n + j];
                            }
                            rhs_grad[i * n + j] += sum;
                        }
                    }
                }),
            })),
        }
    }

    pub fn relu(&self) -> Tensor {
        // Generic path for non-F64 dtypes
        if self.dtype != Dtype::F64 {
            return self.relu_generic();
        }

        // GPU routing
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            return self.relu_cuda();
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    for i in 0..len {
                        if mask[i] {
                            inp_grad[i] += grad_out_f32[i];
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
        use crate::cuda::kernels::relu_inplace;
        use crate::cuda::memory::{alloc, copy_d2d};

        let len = self.data_f64().len();
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
        let d_data = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(err) => {
                crate::cuda::record_activation_fallback("alloc");
                log::warn!(
                    "[Autograd] CUDA alloc ReLU buffer failed ({}), using CPU",
                    err
                );
                return self.relu_cpu_fallback();
            }
        };
        if let Err(err) = copy_d2d(&d_data, &d_src) {
            crate::cuda::record_activation_fallback("copy");
            log::warn!(
                "[Autograd] CUDA D2D ReLU input copy failed ({}), using CPU",
                err
            );
            return self.relu_cpu_fallback();
        }

        if let Err(err) = relu_inplace(&d_data) {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA ReLU kernel failed ({}), using CPU", err);
            return self.relu_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let out = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![0.0; len]))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    #[cfg(cuda)]
                    if input.device == Device::Cuda {
                        if let Some(d_input) = input.cuda_cached_buffer() {
                            if let Ok(d_grad_tmp) =
                                crate::cuda::memory::alloc_pooled::<f64>(grad_out_f64.len())
                            {
                                let d_grad_tmp = std::sync::Arc::new(d_grad_tmp);
                                if crate::cuda::memory::copy_h2d(&d_grad_tmp, grad_out).is_ok() {
                                    if let Some(d_in_grad) = input.cuda_grad_ensure_buffer() {
                                        let _ = crate::cuda::kernels::relu_backward(
                                            &d_input,
                                            &d_grad_tmp,
                                            &d_in_grad,
                                            grad_out_f64.len(),
                                        );
                                    }
                                }
                            }
                        }
                    }
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
        let input_cache = Arc::new(self_f32);
        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    for i in 0..len {
                        let x = input_cache[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = sqrt_2_over_pi * (x + c * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out_f32[i] * gelu_grad;
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
        let input_cache = Arc::new(self_data);
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    const LOG_GRAD_EPS: f32 = 1e-12;
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f32.par_iter())
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
                            inp_grad[i] += grad_out_f32[i] / safe;
                        }
                    }
                }),
            })),
        }
    }

    pub fn exp(&self) -> Tensor {
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
        let exp_cache = Arc::new(data.clone());

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f32.par_iter())
                            .zip(exp_cache.par_iter())
                            .for_each(|((ig, &g), &cached)| {
                                *ig += g * cached;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f32[i] * exp_cache[i];
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
        let sign_cache: Arc<Vec<f32>> = Arc::new(
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

        Tensor {
            data: Storage::from_f32_vec(data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f32.par_iter())
                            .zip(sign_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                *ig += g * s;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f32[i] * sign_cache[i];
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

        let softmax_cache: Arc<Vec<f32>> = Arc::new(data.to_vec());
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    for row in 0..rows_cap {
                        let base = row * cols_cap;
                        let mut sum_term = 0.0f32;
                        for j in 0..cols_cap {
                            let idx = base + j;
                            sum_term += grad_out_f32[idx] * softmax_cache[idx];
                        }
                        for j in 0..cols_cap {
                            let idx = base + j;
                            inp_grad[idx] += softmax_cache[idx] * (grad_out_f32[idx] - sum_term);
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
                                let idx = base + dim_idx;
                                sum_term += grad_out_f64[idx] * softmax_cache_for_backward[idx];
                            }
                            for dim_idx in 0..dim_size_cap {
                                let idx = base + dim_idx;
                                inp_grad[idx] += softmax_cache_for_backward[idx]
                                    * (grad_out_f64[idx] - sum_term);
                            }
                        }
                    } else {
                        for outer_idx in 0..outer_cap {
                            let base_outer = outer_idx * dim_size_cap * inner_cap;
                            for inner_idx in 0..inner_cap {
                                let mut sum_term = 0.0;
                                for dim_idx in 0..dim_size_cap {
                                    let idx = base_outer + dim_idx * inner_cap + inner_idx;
                                    sum_term += grad_out_f64[idx] * softmax_cache_for_backward[idx];
                                }
                                for dim_idx in 0..dim_size_cap {
                                    let idx = base_outer + dim_idx * inner_cap + inner_idx;
                                    inp_grad[idx] += softmax_cache_for_backward[idx]
                                        * (grad_out_f64[idx] - sum_term);
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    let g = grad_out_f32[0];
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    let g = grad_out_f32[0] / len as f32;
                    for v in inp_grad.iter_mut() {
                        *v += g;
                    }
                }),
            })),
        }
    }

    pub fn transpose2d(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose requires 2D tensor");
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    for r in 0..rows {
                        for c in 0..cols {
                            inp_grad[r * cols + c] += grad_out_f32[c * rows + r];
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
        let input_cache = Arc::new(self_data);
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f32.par_iter())
                            .zip(input_cache.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig += g * id.cos();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out_f32[i] * input_cache[i].cos();
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
        let input_cache = Arc::new(self_data);
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f32.par_iter())
                            .zip(input_cache.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig -= g * id.sin();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] -= grad_out_f32[i] * input_cache[i].sin();
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
        let sqrt_cache = data.clone();
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
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let mut inp_grad = _parents[0].grad_write_f32();
                    let len = inp_grad.len();
                    // Reuse cached sqrt values: d/dx sqrt(x) = 0.5 / sqrt(x)
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f32.par_iter())
                            .zip(sqrt_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                if s > 0.0 {
                                    *ig += g * 0.5 / s;
                                }
                            });
                    } else {
                        for i in 0..len {
                            if sqrt_cache[i] > 0.0 {
                                inp_grad[i] += grad_out_f32[i] * 0.5 / sqrt_cache[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let self_data = self.data_as_f64_vec();
        let shape = &self.shape;
        let rank = shape.len();
        assert!(dim0 < rank && dim1 < rank);
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
        assert_eq!(len, self.data.len(), "Reshape dimension mismatch");

        // Zero-copy: share the same data Arc, only change shape metadata
        let parents = vec![self.clone()];

        Tensor {
            data: self.data.clone(),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
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
        }
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
        let diff = self - target;
        let sq = &diff * &diff;
        let weighted = &sq * weights;
        let total = weighted.sum();
        let w_sum = weights.sum();

        let w_sum_data = w_sum.data_f64();
        let denom = if w_sum_data[0].abs() < 1e-12 {
            1.0
        } else {
            w_sum_data[0]
        };
        drop(w_sum_data);

        let total_data = total.data_f64();
        let result_val = total_data[0] / denom;
        drop(total_data);

        let parents = vec![total.clone(), w_sum.clone()];
        let denom_cap = denom;
        let numerator_cap = result_val;
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![result_val]))),
            grad: Storage::zeros(1, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![1],
            device: Device::Cpu,
            dtype: Dtype::F64,
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

    // Generic element-wise binary ops for non-F64 dtypes.
    // -------------------------------------------------------------------------

    fn add_generic(&self, rhs: &Tensor) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        for i in 0..len {
            data[i] = self_f32[i] + rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let same_grad = if let (Storage::F32(a), Storage::F32(b)) =
                        (&_parents[0].grad, &_parents[1].grad)
                    {
                        std::sync::Arc::ptr_eq(a, b)
                    } else {
                        false
                    };
                    if same_grad {
                        let mut grad = _parents[0].grad_write_f32();
                        for i in 0..len {
                            grad[i] += grad_out_f32[i] * 2.0;
                        }
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_f32();
                        let mut rhs_grad = _parents[1].grad_write_f32();
                        for i in 0..len {
                            lhs_grad[i] += grad_out_f32[i];
                            rhs_grad[i] += grad_out_f32[i];
                        }
                    }
                }),
            })),
        }
    }

    fn sub_generic(&self, rhs: &Tensor) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        for i in 0..len {
            data[i] = self_f32[i] - rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let same_grad = if let (Storage::F32(a), Storage::F32(b)) =
                        (&_parents[0].grad, &_parents[1].grad)
                    {
                        std::sync::Arc::ptr_eq(a, b)
                    } else {
                        false
                    };
                    if same_grad {
                        // d/dx (x - x) = 0, no-op
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_f32();
                        let mut rhs_grad = _parents[1].grad_write_f32();
                        for i in 0..len {
                            lhs_grad[i] += grad_out_f32[i];
                            rhs_grad[i] -= grad_out_f32[i];
                        }
                    }
                }),
            })),
        }
    }

    fn mul_generic(&self, rhs: &Tensor) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        for i in 0..len {
            data[i] = self_f32[i] * rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        let rhs_cache = Arc::new(rhs_f32);
        let self_cache = Arc::new(self_f32);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let same_grad = if let (Storage::F32(a), Storage::F32(b)) =
                        (&_parents[0].grad, &_parents[1].grad)
                    {
                        std::sync::Arc::ptr_eq(a, b)
                    } else {
                        false
                    };
                    if same_grad {
                        let mut grad = _parents[0].grad_write_f32();
                        for i in 0..len {
                            grad[i] += grad_out_f32[i] * 2.0 * rhs_cache[i];
                        }
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_f32();
                        let mut rhs_grad = _parents[1].grad_write_f32();
                        for i in 0..len {
                            lhs_grad[i] += grad_out_f32[i] * rhs_cache[i];
                            rhs_grad[i] += grad_out_f32[i] * self_cache[i];
                        }
                    }
                }),
            })),
        }
    }

    fn div_generic(&self, rhs: &Tensor) -> Tensor {
        let self_f32 = self.data_to_f32_vec();
        let rhs_f32 = rhs.data_to_f32_vec();
        let len = self_f32.len();
        let mut data = vec![0.0f32; len];
        for i in 0..len {
            data[i] = self_f32[i] / rhs_f32[i];
        }
        let out_dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);
        let rhs_cache = Arc::new(rhs_f32);
        let self_cache = Arc::new(self_f32);
        Tensor {
            data: Storage::from_f32_vec(data, out_dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(out_dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: out_dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let same_grad = if let (Storage::F32(a), Storage::F32(b)) =
                        (&_parents[0].grad, &_parents[1].grad)
                    {
                        std::sync::Arc::ptr_eq(a, b)
                    } else {
                        false
                    };
                    if same_grad {
                        // d/dx (x/x) = 0, no-op
                    } else {
                        let mut lhs_grad = _parents[0].grad_write_f32();
                        let mut rhs_grad = _parents[1].grad_write_f32();
                        for i in 0..len {
                            lhs_grad[i] += grad_out_f32[i] / rhs_cache[i];
                            rhs_grad[i] +=
                                grad_out_f32[i] * (-self_cache[i] / (rhs_cache[i] * rhs_cache[i]));
                        }
                    }
                }),
            })),
        }
    }
}

// Operator overloads

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Add shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.add_generic(&rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.add_cuda(&rhs) {
            return out;
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] + rhs_data[i]);
        } else {
            vector_add(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg += g);
                        } else {
                            vector_grad_acc(&mut rhs_grad, &grad_out_f64);
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Add<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Add shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.add_generic(rhs);
        }
        #[cfg(cuda)]
        if let Some(out) = self.add_cuda(rhs) {
            return out;
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] + rhs_data[i]);
        } else {
            vector_add(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg += g);
                        } else {
                            vector_grad_acc(&mut rhs_grad, &grad_out_f64);
                        }
                    }
                }),
            })),
        }
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Sub shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.sub_generic(&rhs);
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] - rhs_data[i]);
        } else {
            vector_sub(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg -= g);
                        } else {
                            for i in 0..len {
                                rhs_grad[i] -= grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Sub<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Sub shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.sub_generic(rhs);
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] - rhs_data[i]);
        } else {
            vector_sub(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let len = grad_out_f64.len();
                    {
                        let mut lhs_grad = parents[0].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, &grad_out_f64);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .for_each(|(rg, &g)| *rg -= g);
                        } else {
                            for i in 0..len {
                                rhs_grad[i] -= grad_out_f64[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Mul shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.mul_generic(&rhs);
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] * rhs_data[i]);
        } else {
            vector_mul(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            grad.par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(lhs_data.par_iter())
                                .zip(rhs_data.par_iter())
                                .for_each(|(((g, &go), &l), &r)| {
                                    *g += go * (l + r);
                                });
                        } else {
                            for i in 0..len {
                                grad[i] += grad_out_f64[i] * (lhs_data[i] + rhs_data[i]);
                            }
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                lhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(rhs_data.par_iter())
                                    .for_each(|((lg, &g), &r)| *lg += g * r);
                            } else {
                                for i in 0..len {
                                    lhs_grad[i] += grad_out_f64[i] * rhs_data[i];
                                }
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                rhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(lhs_data.par_iter())
                                    .for_each(|((rg, &g), &l)| *rg += g * l);
                            } else {
                                for i in 0..len {
                                    rhs_grad[i] += grad_out_f64[i] * lhs_data[i];
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Mul<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Mul shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.mul_generic(rhs);
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data.par_iter_mut()
                .enumerate()
                .for_each(|(i, d)| *d = self_data[i] * rhs_data[i]);
        } else {
            vector_mul(&mut data, self_data, rhs_data);
        }
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        for i in 0..len {
                            grad[i] += grad_out_f64[i] * (lhs_data[i] + rhs_data[i]);
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            for i in 0..len {
                                lhs_grad[i] += grad_out_f64[i] * rhs_data[i];
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            for i in 0..len {
                                rhs_grad[i] += grad_out_f64[i] * lhs_data[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Div shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.div_generic(&rhs);
        }
        let guards = TensorReadGuard::new(&[&self, &rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data
                .par_iter()
                .zip(rhs_data.par_iter())
                .map(|(a, b)| a / b)
                .collect()
        } else {
            self_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a / b)
                .collect()
        };
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    const DIV_EPS: f64 = 1e-12;
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        if len >= PAR_THRESHOLD {
                            grad.par_iter_mut()
                                .zip(grad_out_f64.par_iter())
                                .zip(lhs_data.par_iter())
                                .zip(rhs_data.par_iter())
                                .for_each(|(((g, &go), &l), &r)| {
                                    let safe_r = if r.abs() < DIV_EPS {
                                        r.signum() * DIV_EPS
                                    } else {
                                        r
                                    };
                                    *g += go / safe_r - go * l / (safe_r * safe_r);
                                });
                        } else {
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                grad[i] += grad_out_f64[i] / safe_r
                                    - grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                            }
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                lhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(rhs_data.par_iter())
                                    .for_each(|((lg, &g), &r)| {
                                        let safe_r = if r.abs() < DIV_EPS {
                                            r.signum() * DIV_EPS
                                        } else {
                                            r
                                        };
                                        *lg += g / safe_r;
                                    });
                            } else {
                                for i in 0..len {
                                    let r = rhs_data[i];
                                    let safe_r = if r.abs() < DIV_EPS {
                                        r.signum() * DIV_EPS
                                    } else {
                                        r
                                    };
                                    lhs_grad[i] += grad_out_f64[i] / safe_r;
                                }
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            if len >= PAR_THRESHOLD {
                                rhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out_f64.par_iter())
                                    .zip(lhs_data.par_iter())
                                    .zip(rhs_data.par_iter())
                                    .for_each(|(((rg, &g), &l), &r)| {
                                        let safe_r = if r.abs() < DIV_EPS {
                                            r.signum() * DIV_EPS
                                        } else {
                                            r
                                        };
                                        *rg -= g * l / (safe_r * safe_r);
                                    });
                            } else {
                                for i in 0..len {
                                    let r = rhs_data[i];
                                    let safe_r = if r.abs() < DIV_EPS {
                                        r.signum() * DIV_EPS
                                    } else {
                                        r
                                    };
                                    rhs_grad[i] -=
                                        grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'b> Div<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &'b Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Div shape mismatch");
        if self.dtype != Dtype::F64 || rhs.dtype != Dtype::F64 {
            return self.div_generic(rhs);
        }
        let guards = TensorReadGuard::new(&[self, rhs]);
        let self_data = guards.get(0);
        let rhs_data = guards.get(1);
        let len = self_data.len();
        let data: Vec<f64> = self_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(a, b)| a / b)
            .collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out_f64.len();
                    const DIV_EPS: f64 = 1e-12;
                    if lhs.grad.id() == rhs.grad.id() {
                        let mut grad = lhs.grad_write_f64();
                        for i in 0..len {
                            let r = rhs_data[i];
                            let safe_r = if r.abs() < DIV_EPS {
                                r.signum() * DIV_EPS
                            } else {
                                r
                            };
                            grad[i] += grad_out_f64[i] / safe_r
                                - grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad_write_f64();
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                lhs_grad[i] += grad_out_f64[i] / safe_r;
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad_write_f64();
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                rhs_grad[i] -= grad_out_f64[i] * lhs_data[i] / (safe_r * safe_r);
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        let self_data = self.data_f64();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| -x).collect()
        } else {
            self_data.iter().map(|&x| -x).collect()
        };
        let parents = vec![self.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_f64();
                    let len = grad_out_f64.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out_f64.par_iter())
                            .for_each(|(ig, &g)| *ig -= g);
                    } else {
                        for i in 0..len {
                            inp_grad[i] -= grad_out_f64[i];
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
        Arc::as_ptr(&self.data) as usize
    }

    pub(crate) fn cuda_cached_buffer(&self) -> Option<Arc<crate::cuda::memory::DevicePtr<f64>>> {
        let key = self.cuda_cache_key();
        let cache = cuda_tensor_buffer_cache();
        let map = cache.lock().ok()?;
        map.get(&key).cloned()
    }

    pub(crate) fn cuda_set_cached_buffer(&self, buffer: Arc<crate::cuda::memory::DevicePtr<f64>>) {
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

    // --- GPU gradient buffer cache (keyed by grad Arc pointer) ---

    fn cuda_grad_cache_key(&self) -> usize {
        Arc::as_ptr(&self.grad) as usize
    }

    fn cuda_grad_cached_buffer(&self) -> Option<Arc<crate::cuda::memory::DevicePtr<f64>>> {
        let key = self.cuda_grad_cache_key();
        let cache = cuda_tensor_buffer_cache();
        let map = cache.lock().ok()?;
        map.get(&key).cloned()
    }

    fn cuda_grad_set_cached_buffer(&self, buffer: Arc<crate::cuda::memory::DevicePtr<f64>>) {
        let key = self.cuda_grad_cache_key();
        if let Ok(mut map) = cuda_tensor_buffer_cache().lock() {
            map.insert(key, buffer);
        }
    }

    fn cuda_grad_remove_cached_buffer(&self) {
        let key = self.cuda_grad_cache_key();
        if let Ok(mut map) = cuda_tensor_buffer_cache().lock() {
            map.remove(&key);
        }
    }

    /// Upload CPU grad to GPU, returning the GPU buffer.
    pub(crate) fn cuda_grad_get_or_upload_buffer(
        &self,
    ) -> Result<
        Arc<crate::cuda::memory::DevicePtr<f64>>,
        (&'static str, crate::cuda::error::CudaError),
    > {
        use crate::cuda::memory::{alloc, copy_h2d};

        let host = self.grad_to_f64_vec();
        let len = host.len();
        if let Some(buffer) = self.cuda_grad_cached_buffer() {
            if buffer.len() == len {
                if !host.is_empty() {
                    if let Err(err) = copy_h2d(&buffer, &host) {
                        self.cuda_grad_remove_cached_buffer();
                        return Err(("copy", err));
                    }
                }
                return Ok(buffer);
            }
            self.cuda_grad_remove_cached_buffer();
        }

        if len == 0 {
            return Ok(Arc::new(crate::cuda::memory::DevicePtr::zero_sized()));
        }

        let device = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(err) => return Err(("alloc", err)),
        };
        if let Err(err) = copy_h2d(&device, &host) {
            return Err(("copy", err));
        }

        let device = Arc::new(device);
        self.cuda_grad_set_cached_buffer(device.clone());
        Ok(device)
    }

    /// Ensure a zero-initialized GPU grad buffer exists for this tensor.
    pub(crate) fn cuda_grad_ensure_buffer(
        &self,
    ) -> Option<Arc<crate::cuda::memory::DevicePtr<f64>>> {
        use crate::cuda::memory::{alloc, copy_h2d};

        let len = self.grad.len();
        if len == 0 {
            return Some(Arc::new(crate::cuda::memory::DevicePtr::zero_sized()));
        }
        if let Some(buffer) = self.cuda_grad_cached_buffer() {
            if buffer.len() == len {
                // Zero out existing buffer
                let zeros = vec![0.0_f64; len];
                let _ = copy_h2d(&buffer, &zeros);
                return Some(buffer);
            }
            self.cuda_grad_remove_cached_buffer();
        }
        let device = alloc::<f64>(len).ok()?;
        let zeros = vec![0.0_f64; len];
        let _ = copy_h2d(&device, &zeros);
        let device = Arc::new(device);
        self.cuda_grad_set_cached_buffer(device.clone());
        Some(device)
    }

    /// Materialize GPU data to CPU if this tensor lives on GPU but has empty CPU data.
    /// This is needed because GPU operations may skip the D2H copy, leaving `data` empty.
    /// Lazy materialization ensures backward pass can read the data.
    #[cfg(cuda)]
    fn cuda_materialize(&self) {
        use crate::cuda::memory::copy_d2h;

        // If device is CPU or data is non-empty, nothing to do
        if self.device != Device::Cuda {
            return;
        }
        let len = self.data_f64().len();
        if len > 0 {
            return;
        }

        // Data is empty but we're on GPU - try to materialize from cached GPU buffer
        if let Some(buffer) = self.cuda_cached_buffer() {
            let mut data = vec![0.0; buffer.len()];
            if let Err(err) = copy_d2h(&mut data, &buffer) {
                log::warn!(
                    "[Tensor] CUDA materialize D2H failed ({}), data remains empty",
                    err
                );
                return;
            }
            let mut self_data = self.data_write_f64();
            *self_data = data;
        }
    }

    pub(crate) fn cuda_get_or_upload_buffer(
        &self,
    ) -> Result<
        Arc<crate::cuda::memory::DevicePtr<f64>>,
        (&'static str, crate::cuda::error::CudaError),
    > {
        use crate::cuda::memory::{alloc, copy_h2d};

        let host = self.data_f64();
        let len = host.len();
        if let Some(buffer) = self.cuda_cached_buffer() {
            if buffer.len() == len {
                // GPU buffer exists - check if host data is non-empty
                if !host.is_empty() {
                    if let Err(err) = copy_h2d(&buffer, &host) {
                        self.cuda_remove_cached_buffer();
                        return Err(("copy", err));
                    }
                }
                return Ok(buffer);
            }
            self.cuda_remove_cached_buffer();
        }

        if len == 0 {
            // Zero-length tensors don't need GPU memory
            return Ok(Arc::new(crate::cuda::memory::DevicePtr::zero_sized()));
        }

        let device = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(err) => return Err(("alloc", err)),
        };
        if let Err(err) = copy_h2d(&device, &host) {
            return Err(("copy", err));
        }
        drop(host);

        let device = Arc::new(device);
        self.cuda_set_cached_buffer(device.clone());
        Ok(device)
    }

    /// GPU-accelerated matrix multiplication using CUDA cuBLAS
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn matmul_cuda(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
        use crate::cuda::blas::gemm_thread_local;
        use crate::cuda::error::CudaError;
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
        let d_c = match alloc::<f64>(m * n) {
            Ok(buf) => buf,
            Err(_err) => {
                crate::cuda::record_matmul_fallback("alloc");
                return self.matmul_cpu_fallback(other, m, k, n);
            }
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

        if let Err(err) = gemm_thread_local(
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
        ) {
            let stage = match &err {
                CudaError::Blas { op, .. } if *op == "cublasCreate_v2" => "init",
                _ => "gemm",
            };
            crate::cuda::record_matmul_fallback(stage);
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
            data: Storage::F64(Arc::new(RwLock::new(vec![0.0; m * n]))),
            grad: Storage::zeros(m * n, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cuda,
            dtype: Dtype::F64,
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
                            if let Ok(d_grad_tmp) =
                                crate::cuda::memory::alloc_pooled::<f64>(grad_out_f64.len())
                            {
                                let d_grad_tmp = std::sync::Arc::new(d_grad_tmp);
                                if crate::cuda::memory::copy_h2d(&d_grad_tmp, grad_out).is_ok() {
                                    use crate::cuda::blas::gemm_thread_local;

                                    // dL/dLHS = grad_out_f64[m,n] * rhs^T[n,k]  --> m x k
                                    let lhs_ok =
                                        if let Some(d_lhs_grad) = lhs.cuda_grad_ensure_buffer() {
                                            gemm_thread_local(
                                                false,
                                                true,
                                                m as i32,
                                                k as i32,
                                                n as i32,
                                                1.0,
                                                d_grad_tmp.as_raw(),
                                                n as i32,
                                                d_rhs.as_raw(),
                                                n as i32,
                                                1.0,
                                                d_lhs_grad.as_raw(),
                                                k as i32,
                                            )
                                            .is_ok()
                                        } else {
                                            false
                                        };

                                    // dL/dRHS = lhs^T[k,m] * grad_out_f64[m,n]  --> k x n
                                    let rhs_ok =
                                        if let Some(d_rhs_grad) = rhs.cuda_grad_ensure_buffer() {
                                            gemm_thread_local(
                                                true,
                                                false,
                                                k as i32,
                                                n as i32,
                                                m as i32,
                                                1.0,
                                                d_lhs.as_raw(),
                                                k as i32,
                                                d_grad_tmp.as_raw(),
                                                n as i32,
                                                1.0,
                                                d_rhs_grad.as_raw(),
                                                n as i32,
                                            )
                                            .is_ok()
                                        } else {
                                            false
                                        };

                                    if lhs_ok && rhs_ok {
                                        gpu_backward_ok = true;
                                    }
                                }
                            }
                        }
                        if gpu_backward_ok {
                            crate::cuda::record_backward_success();
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
        use crate::cuda::kernels::gelu_inplace;
        use crate::cuda::memory::{alloc, copy_d2d};

        let len = self.data_f64().len();
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
        let d_data = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(err) => {
                crate::cuda::record_activation_fallback("alloc");
                log::warn!(
                    "[Autograd] CUDA alloc GELU buffer failed ({}), using CPU",
                    err
                );
                return self.gelu_cpu_fallback();
            }
        };
        if let Err(err) = copy_d2d(&d_data, &d_src) {
            crate::cuda::record_activation_fallback("copy");
            log::warn!(
                "[Autograd] CUDA D2D GELU input copy failed ({}), using CPU",
                err
            );
            return self.gelu_cpu_fallback();
        }

        if let Err(err) = gelu_inplace(&d_data) {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA GELU kernel failed ({}), using CPU", err);
            return self.gelu_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let out = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![0.0; len]))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    #[cfg(cuda)]
                    if input.device == Device::Cuda {
                        if let Some(d_input) = input.cuda_cached_buffer() {
                            if let Ok(d_grad_tmp) =
                                crate::cuda::memory::alloc_pooled::<f64>(grad_out_f64.len())
                            {
                                let d_grad_tmp = std::sync::Arc::new(d_grad_tmp);
                                if crate::cuda::memory::copy_h2d(&d_grad_tmp, grad_out).is_ok() {
                                    if let Some(d_in_grad) = input.cuda_grad_ensure_buffer() {
                                        let _ = crate::cuda::kernels::gelu_backward(
                                            &d_input,
                                            &d_grad_tmp,
                                            &d_in_grad,
                                            grad_out_f64.len(),
                                        );
                                    }
                                }
                            }
                        }
                    }
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
        use crate::cuda::kernels::softmax_inplace_auto;
        use crate::cuda::memory::{alloc, copy_d2d};

        let len = self.data_f64().len();
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
        let d_data = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(err) => {
                crate::cuda::record_activation_fallback("alloc");
                log::warn!(
                    "[Autograd] CUDA alloc Softmax buffer failed ({}), using CPU",
                    err
                );
                return self.softmax_cpu_fallback();
            }
        };

        if let Err(err) = copy_d2d(&d_data, &d_src) {
            crate::cuda::record_activation_fallback("copy");
            log::warn!(
                "[Autograd] CUDA D2D Softmax input copy failed ({}), using CPU",
                err
            );
            return self.softmax_cpu_fallback();
        }

        if let Err(err) = softmax_inplace_auto(&d_data, rows, cols) {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA Softmax kernel failed ({}), using CPU", err);
            return self.softmax_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let rows_cap = rows;
        let cols_cap = cols;
        let out = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![0.0; len]))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    // Backward needs softmax output for: grad_in = softmax * (grad_out - sum(grad_out * softmax))
                    // The softmax output lives on GPU (self's cached buffer).
                    // We materialize it here so the backward computation can proceed on CPU.
                    let out_data: Vec<f64> = if let Some(buf) = parents[0].cuda_cached_buffer() {
                        let mut cpu = vec![0.0; buf.len()];
                        if crate::cuda::memory::copy_d2h(&mut cpu, &buf).is_ok() {
                            cpu
                        } else {
                            // Fallback: zero gradient on materialization failure
                            return;
                        }
                    } else {
                        return;
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
            _ctx: self._ctx.clone(),
        };
        cpu_view.softmax()
    }

    /// GPU-accelerated causal softmax (forward only, no materialization).
    /// Causal mask is applied inside the kernel (no separate mask step).
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub(crate) fn softmax_causal_cuda(&self) -> Tensor {
        use crate::cuda::kernels::softmax_causal_inplace;
        use crate::cuda::memory::{alloc, copy_d2d};

        let len = self.data_f64().len();
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
        let d_data = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(err) => {
                crate::cuda::record_activation_fallback("alloc");
                log::warn!(
                    "[Autograd] CUDA alloc CausalSoftmax buffer failed ({}), using CPU",
                    err
                );
                return self.softmax_cpu_fallback();
            }
        };

        if let Err(err) = copy_d2d(&d_data, &d_src) {
            crate::cuda::record_activation_fallback("copy");
            log::warn!(
                "[Autograd] CUDA D2D CausalSoftmax input copy failed ({}), using CPU",
                err
            );
            return self.softmax_cpu_fallback();
        }

        if let Err(err) = softmax_causal_inplace(&d_data, rows, cols) {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!(
                "[Autograd] CUDA CausalSoftmax kernel failed ({}), using CPU",
                err
            );
            return self.softmax_cpu_fallback();
        }

        let d_data = Arc::new(d_data);
        crate::cuda::record_activation_success();

        let parents = vec![self.clone()];
        let rows_cap = rows;
        let cols_cap = cols;
        let out = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![0.0; len]))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    // Causal softmax backward: grad_in = out * (grad_out - sum(grad_out * out))
                    // Masked positions (j > i) have out = 0, so grad_in = 0 for them.
                    let out_data: Vec<f64> = if let Some(buf) = parents[0].cuda_cached_buffer() {
                        let mut cpu = vec![0.0; buf.len()];
                        if crate::cuda::memory::copy_d2h(&mut cpu, &buf).is_ok() {
                            cpu
                        } else {
                            return;
                        }
                    } else {
                        return;
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
                            // Causal mask: j > row positions were -inf in forward,
                            // their output is 0, so gradient contribution is also 0.
                            // The formula handles this naturally since out_data[idx] = 0.
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
        use crate::cuda::kernels::rope_inplace;
        use crate::cuda::memory::{alloc, copy_d2d, copy_h2d};

        let len = self.data_f64().len();
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
        let d_data = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(_err) => {
                crate::cuda::record_activation_fallback("alloc");
                return self.clone();
            }
        };

        if let Err(_err) = copy_d2d(&d_data, &d_src) {
            crate::cuda::record_activation_fallback("copy");
            return self.clone();
        }

        // Upload cos/sin caches
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
        if let Err(_e) = copy_h2d(&d_cos, cos_cache) {
            crate::cuda::record_activation_fallback("copy_cos");
            return self.clone();
        }
        if let Err(_e) = copy_h2d(&d_sin, sin_cache) {
            crate::cuda::record_activation_fallback("copy_sin");
            return self.clone();
        }

        if let Err(err) = rope_inplace(
            &d_data,
            &d_cos,
            &d_sin,
            seq_len,
            dim,
            total_batches,
            start_pos,
        ) {
            crate::cuda::record_activation_fallback("kernel");
            log::warn!("[Autograd] CUDA RoPE kernel failed ({}), using CPU", err);
            return self.clone();
        }
        crate::cuda::record_activation_success();

        let d_data = Arc::new(d_data);
        let parents = vec![self.clone()];
        let dim_cap = dim;
        let cos_cache = cos_cache.to_vec();
        let sin_cache = sin_cache.to_vec();
        let seq_len_cap = seq_len;
        let total_batches_cap = total_batches;
        let start_pos_cap = start_pos;

        let out = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![0.0; len]))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |_grad_out, parents| {
                    let cos_cache = cos_cache.clone();
                    let sin_cache = sin_cache.clone();
                    let grad_out_data: Vec<f64> = if let Some(buf) = parents[0].cuda_cached_buffer()
                    {
                        let mut cpu = vec![0.0; buf.len()];
                        if crate::cuda::memory::copy_d2h(&mut cpu, &buf).is_ok() {
                            cpu
                        } else {
                            return;
                        }
                    } else {
                        return;
                    };
                    let mut inp_grad = parents[0].grad_write_f64();
                    let half_dim = dim_cap / 2;
                    for b in 0..total_batches_cap {
                        for t in 0..seq_len_cap {
                            let pos = start_pos_cap + t;
                            if pos * half_dim >= cos_cache.len() {
                                continue;
                            }
                            let cache_idx = pos * half_dim;
                            let base_idx = b * (seq_len_cap * dim_cap) + t * dim_cap;
                            for i in 0..half_dim {
                                let c = cos_cache[cache_idx + i];
                                let s = sin_cache[cache_idx + i];
                                let g1 = grad_out_data[base_idx + 2 * i];
                                let g2 = grad_out_data[base_idx + 2 * i + 1];
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
        use crate::cuda::kernels::log_softmax as cuda_log_softmax;
        use crate::cuda::memory::{alloc, copy_d2h};

        let shape = self.shape.clone();
        let dim_size = *shape.last().unwrap_or(&1);
        if dim_size == 0 {
            return self.log_softmax_last_dim_cpu_fallback();
        }

        let len = self.data_f64().len();
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
        let d_out = match alloc::<f64>(len) {
            Ok(buf) => buf,
            Err(err) => {
                crate::cuda::record_log_softmax_fallback("alloc");
                log::warn!(
                    "[Autograd] CUDA alloc LogSoftmax output failed ({}), using CPU",
                    err
                );
                return self.log_softmax_last_dim_cpu_fallback();
            }
        };
        let d_out = Arc::new(d_out);

        if let Err(err) = cuda_log_softmax(&d_in, &d_out, num_slices, dim_size) {
            crate::cuda::record_log_softmax_fallback("kernel");
            log::warn!(
                "[Autograd] CUDA LogSoftmax kernel failed ({}), using CPU",
                err
            );
            return self.log_softmax_last_dim_cpu_fallback();
        }

        let mut data = vec![0.0; len];
        if let Err(err) = copy_d2h(&mut data, &d_out) {
            crate::cuda::record_log_softmax_fallback("copy");
            log::warn!("[Autograd] CUDA D2H LogSoftmax failed ({}), using CPU", err);
            return self.log_softmax_last_dim_cpu_fallback();
        }
        crate::cuda::record_log_softmax_success();

        let softmax_cache: Arc<Vec<f64>> = Arc::new(data.iter().map(|v| v.exp()).collect());
        let parents = vec![self.clone()];
        let dim_size_cap = dim_size;
        let num_slices_cap = num_slices;
        let softmax_cache_for_backward = softmax_cache.clone();

        let out = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape,
            device: Device::Cuda,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut sum_terms = vec![0.0; num_slices_cap];
                    for (slice_idx, sum_term) in sum_terms.iter_mut().enumerate() {
                        let base = slice_idx * dim_size_cap;
                        let mut slice_sum = 0.0;
                        for j in 0..dim_size_cap {
                            let idx = base + j;
                            slice_sum += grad_out_f64[idx] * softmax_cache_for_backward[idx];
                        }
                        *sum_term = slice_sum;
                    }

                    let mut inp_grad = parents[0].grad_write_compat();
                    #[allow(clippy::needless_range_loop)]
                    for slice_idx in 0..num_slices_cap {
                        let base = slice_idx * dim_size_cap;
                        for j in 0..dim_size_cap {
                            let idx = base + j;
                            inp_grad[idx] += softmax_cache_for_backward[idx]
                                * (grad_out_f64[idx] - sum_terms[slice_idx]);
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
        forward_kernel: fn(
            &crate::cuda::memory::DevicePtr<f64>,
            &crate::cuda::memory::DevicePtr<f64>,
            &crate::cuda::memory::DevicePtr<f64>,
            usize,
        ) -> crate::cuda::error::CudaResult<()>,
        backward_kernel: Option<
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
        use crate::cuda::memory::alloc;

        if self.device != Device::Cuda || rhs.device != Device::Cuda {
            return None;
        }
        let len = self.data_f64().len();
        let d_a = self.cuda_get_or_upload_buffer().ok()?;
        let d_b = rhs.cuda_get_or_upload_buffer().ok()?;
        let d_out = alloc::<f64>(len).ok()?;
        let d_out = std::sync::Arc::new(d_out);

        if forward_kernel(&d_a, &d_b, &d_out, len).is_err() {
            return None;
        }

        let parents = vec![self.clone(), rhs.clone()];
        let out = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(vec![0.0; len]))),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: self.shape.clone(),
            device: Device::Cuda,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let a = &parents[0];
                    let b = &parents[1];
                    #[cfg(cuda)]
                    if a.device == Device::Cuda && b.device == Device::Cuda {
                        if let (Some(d_a), Some(d_b)) =
                            (a.cuda_cached_buffer(), b.cuda_cached_buffer())
                        {
                            if let Ok(d_grad_tmp) =
                                crate::cuda::memory::alloc_pooled::<f64>(grad_out_f64.len())
                            {
                                let d_grad_tmp = std::sync::Arc::new(d_grad_tmp);
                                if crate::cuda::memory::copy_h2d(&d_grad_tmp, grad_out).is_ok() {
                                    if let Some(d_a_grad) = a.cuda_grad_ensure_buffer() {
                                        if let Some(d_b_grad) = b.cuda_grad_ensure_buffer() {
                                            if let Some(bk) = backward_kernel {
                                                let _ = bk(
                                                    &d_grad_tmp,
                                                    &d_a,
                                                    &d_b,
                                                    &d_a_grad,
                                                    &d_b_grad,
                                                    grad_out_f64.len(),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // CPU backward fallback
                    let mut a_grad = a.grad_write_compat();
                    let mut b_grad = b.grad_write_compat();
                    for i in 0..grad_out_f64.len() {
                        a_grad[i] += grad_out_f64[i];
                        b_grad[i] += grad_out_f64[i];
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }

    #[cfg(cuda)]
    fn add_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            crate::cuda::kernels::add_forward,
            None, // Add backward is simple accumulation handled by CPU fallback
        )
    }

    #[cfg(cuda)]
    fn sub_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            crate::cuda::kernels::sub_forward,
            None, // Sub backward is same as Add: accumulate grad_out
        )
    }

    #[cfg(cuda)]
    fn mul_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            crate::cuda::kernels::mul_forward,
            Some(crate::cuda::kernels::mul_backward),
        )
    }

    #[cfg(cuda)]
    fn div_cuda(&self, rhs: &Tensor) -> Option<Tensor> {
        self.elementwise_op_cuda(
            rhs,
            crate::cuda::kernels::div_forward,
            None, // TODO: implement div_backward kernel
        )
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        -self.clone()
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
    fn test_mixed_dtype_promotes_to_f64() {
        let a = Tensor::with_dtype(vec![1.0, 2.0], vec![2], Dtype::F32);
        let b = Tensor::with_dtype(vec![3.0, 4.0], vec![2], Dtype::F64);

        let c = &a + &b;
        assert_eq!(c.dtype, Dtype::F64);
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
