use crate::simd::{
    add_scaled_row, dot_product, horizontal_sum, prefetch_read_l1, vector_add, vector_fma,
    vector_gelu, vector_grad_acc, vector_mul, vector_relu, vector_sub,
};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fs::File;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, RwLock};

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

#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<RwLock<Vec<f64>>>, // Read-write access (backward pass needs write)
    pub grad: Arc<RwLock<Vec<f64>>>,
    pub shape: Vec<usize>,
    pub device: Device,             // Device where tensor resides
    pub _ctx: Option<Arc<Context>>, // Keeps the graph alive
}

/// Batch lock acquisition helper for reducing lock overhead in parallel operations
pub struct TensorReadGuard<'a> {
    guards: Vec<std::sync::RwLockReadGuard<'a, Vec<f64>>>,
}

impl<'a> TensorReadGuard<'a> {
    /// Acquire read locks for multiple tensors at once
    pub fn new(tensors: &[&'a Tensor]) -> Self {
        let guards: Vec<_> = tensors.iter().map(|t| t.data.read().unwrap()).collect();
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
        let data = self.data.read().unwrap();
        let mut state = serializer.serialize_struct("Tensor", 2)?;
        state.serialize_field("data", &*data)?;
        state.serialize_field("shape", &self.shape)?;
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
        }

        let helper = TensorData::deserialize(deserializer)?;
        Ok(Tensor::new(helper.data, helper.shape))
    }
}

type BackwardOp = Box<dyn Fn(&Vec<f64>, &Vec<Tensor>) + Send + Sync>;

pub struct Context {
    pub parents: Vec<Tensor>,
    pub backward_op: BackwardOp, // receives grad_output, parents
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; expected_len])),
            shape,
            device: Device::Cpu,
            _ctx: None,
        })
    }

    pub fn save_binary(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = File::create(path)?;
        let data = self.data.read().unwrap();
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape,
            device: Device::Cpu,
            _ctx: None,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>();
        Tensor::new(vec![0.0; len], shape)
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
        let grad_len = self.grad.read().unwrap().len();
        Tensor {
            data: self.data.clone(),
            grad: Arc::new(RwLock::new(vec![0.0; grad_len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: None,
        }
    }

    // Create a new leaf tensor with same data (copy)
    pub fn item(&self) -> f64 {
        assert_eq!(self.shape.iter().product::<usize>(), 1);
        self.data.read().unwrap()[0]
    }

    pub fn backward(&self) {
        // Topological sort
        let mut visited = std::collections::HashSet::new();
        let mut topo = Vec::new();
        fn build_topo(
            t: &Tensor,
            visited: &mut std::collections::HashSet<usize>,
            topo: &mut Vec<Tensor>,
        ) {
            // Use pointer address of grad RwLock as ID
            let id = Arc::as_ptr(&t.grad) as usize;
            if !visited.contains(&id) {
                visited.insert(id);
                if let Some(ctx) = &t._ctx {
                    for parent in &ctx.parents {
                        build_topo(parent, visited, topo);
                    }
                }
                topo.push(t.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);

        // Seed gradient of this tensor to 1.0
        {
            let mut g = self.grad.write().unwrap();
            for v in g.iter_mut() {
                *v = 1.0;
            }
        }

        // Backprop
        for t in topo.iter().rev() {
            if let Some(ctx) = &t._ctx {
                let grad = t.grad.read().unwrap();
                (ctx.backward_op)(&grad, &ctx.parents);
            }
        }
    }

    // Explicitly clear the graph history to free memory
    pub fn clear_graph(&mut self) {
        self._ctx = None;
    }

    pub fn zero_grad(&self) {
        let mut g = self.grad.write().unwrap();
        for v in g.iter_mut() {
            *v = 0.0;
        }
    }

    /// Copy tensor data to CUDA GPU.
    /// Current implementation keeps data on host while marking device intent.
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub fn to_cuda(&self) -> Result<Tensor, ()> {
        if crate::cuda::init().is_err() {
            eprintln!("[Tensor] CUDA runtime unavailable");
            return Err(());
        }

        Ok(Tensor {
            data: self.data.clone(),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
            device: Device::Cuda,
            _ctx: self._ctx.clone(),
        })
    }

    /// Copy tensor data from CUDA GPU back to CPU.
    #[cfg(cuda)]
    #[allow(dead_code)]
    pub fn from_cuda(&self) -> Result<Vec<f64>, ()> {
        if self.device != Device::Cuda {
            return Err(());
        }
        Ok(self.data.read().unwrap().clone())
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
            if ops >= 32768 && self.device == Device::Cuda && other.device == Device::Cuda {
                // GPU path
                return self.matmul_cuda(other, m, k, n);
            }
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
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; m * n])),
            shape: out_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];

                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);

                    // dL/dLHS = grad_out * RHS^T
                    {
                        let mut lhs_grad = lhs.grad.write().unwrap();
                        let ops = m * k * n;
                        if ops < 32768 {
                            for r in 0..m {
                                let grad_out_row_start = r * n;
                                let lhs_grad_row_start = r * k;
                                for i in 0..k {
                                    let rhs_row_start = i * n;
                                    let grad_row =
                                        &grad_out[grad_out_row_start..grad_out_row_start + n];
                                    let rhs_row = &rhs_data[rhs_row_start..rhs_row_start + n];
                                    lhs_grad[lhs_grad_row_start + i] +=
                                        dot_product(grad_row, rhs_row);
                                }
                            }
                        } else if (2..=4).contains(&m) && k >= 64 {
                            for r in 0..m {
                                let grad_row = &grad_out[r * n..(r + 1) * n];
                                let lhs_row = &mut lhs_grad[r * k..(r + 1) * k];
                                lhs_row.par_iter_mut().enumerate().for_each(|(i, val)| {
                                    let rhs_row = &rhs_data[i * n..i * n + n];
                                    *val += dot_product(grad_row, rhs_row);
                                });
                            }
                        } else if m == 1 {
                            let grad_row = &grad_out[..n];
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
                                        &grad_out[grad_out_row_start..grad_out_row_start + n];
                                    for (i, lhs_val) in lhs_row.iter_mut().enumerate().take(k) {
                                        let rhs_row_start = i * n;
                                        let rhs_row = &rhs_data[rhs_row_start..rhs_row_start + n];
                                        *lhs_val += dot_product(grad_row, rhs_row);
                                    }
                                });
                        }
                    }

                    // dL/dRHS = LHS^T * grad_out
                    // RHS_grad[i, :] += sum_r ( LHS[r, i] * grad_out[r, :] )
                    {
                        let mut rhs_grad = rhs.grad.write().unwrap();
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
                                        &grad_out[grad_out_row_start..grad_out_row_start + n];
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
                                        let grad_row =
                                            &grad_out[grad_out_row_start..grad_out_row_start + n];
                                        add_scaled_row(rhs_row, grad_row, scale);
                                    }
                                });
                        }
                    }
                }),
            })),
        }
    }

    pub fn relu(&self) -> Tensor {
        // GPU routing
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            return self.relu_cuda();
        }

        // CPU path - inline implementation when cuda is disabled
        #[cfg(not(cuda))]
        {
            let self_data = self.data.read().unwrap();
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
                data: Arc::new(RwLock::new(data)),
                grad: Arc::new(RwLock::new(vec![0.0; len])),
                shape: self.shape.clone(),
                device: Device::Cpu,
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, parents| {
                        let input = &parents[0];
                        let input_data = input.data.read().unwrap();
                        let mut inp_grad = input.grad.write().unwrap();
                        if inp_grad.len() >= PAR_THRESHOLD {
                            inp_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .zip(input_data.par_iter())
                                .for_each(|((ig, &go), &val)| {
                                    if val > 0.0 {
                                        *ig += go;
                                    }
                                });
                        } else {
                            for i in 0..inp_grad.len() {
                                if input_data[i] > 0.0 {
                                    inp_grad[i] += grad_out[i];
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

    /// CPU fallback for ReLU activation
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn relu_cpu_fallback(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &go), &val)| {
                                if val > 0.0 {
                                    *ig += go;
                                }
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            if input_data[i] > 0.0 {
                                inp_grad[i] += grad_out[i];
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
        self.relu_cpu_fallback()
    }

    /// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Tensor {
        // GPU routing
        #[cfg(cuda)]
        if self.device == Device::Cuda {
            return self.gelu_cuda();
        }

        let self_data = self.data.read().unwrap();
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
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
                            inp_grad[i] += grad_out[i] * gelu_grad;
                        }
                    }
                }),
            })),
        }
    }

    pub fn log(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.ln()).collect()
        } else {
            self_data.iter().map(|&x| x.ln()).collect()
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    // d(ln x)/dx = 1/x; clamp denominator to avoid inf when x -> 0
                    const LOG_GRAD_EPS: f64 = 1e-12;
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(input_data.par_iter())
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
                            let safe = if input_data[i].abs() < LOG_GRAD_EPS {
                                input_data[i].signum() * LOG_GRAD_EPS
                            } else {
                                input_data[i]
                            };
                            inp_grad[i] += grad_out[i] / safe;
                        }
                    }
                }),
            })),
        }
    }

    pub fn exp(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.exp()).collect();
        } else {
            crate::simd::fast_exp_bulk(&mut data, &self_data);
        }
        let parents = vec![self.clone()];
        // Cache forward result for backward (d/dx e^x = e^x)
        let exp_cache = Arc::new(data.clone());

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(exp_cache.par_iter())
                            .for_each(|((ig, &g), &cached)| {
                                *ig += g * cached;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out[i] * exp_cache[i];
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
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let mut data = vec![0.0; len];
        if len >= PAR_THRESHOLD {
            data = self_data.par_iter().map(|&x| x.abs()).collect();
        } else {
            for i in 0..len {
                data[i] = self_data[i].abs();
            }
        }
        // Cache forward result for backward (sign function needs original values)
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

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(sign_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                *ig += g * s;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out[i] * sign_cache[i];
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
        let self_data = self.data.read().unwrap();
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(pow_cache.par_iter())
                            .for_each(|((ig, &g), &cached)| {
                                *ig += g * exp * cached;
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out[i] * exp * pow_cache[i];
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

        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let max_val = self_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_shifted = vec![0.0; len];
        for i in 0..len {
            exp_shifted[i] = (self_data[i] - max_val).exp();
        }
        let sum_exp: f64 = exp_shifted.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let mut data = vec![0.0; len];
        for i in 0..len {
            data[i] = exp_shifted[i] / sum_exp;
        }

        let softmax_cache: Arc<Vec<f64>> = Arc::new(data.to_vec());
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    // d/dx softmax = softmax * (grad_out - sum(grad_out * softmax))
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let sum_term: f64 = grad_out
                        .iter()
                        .zip(softmax_cache.iter())
                        .map(|(&g, &s)| g * s)
                        .sum();
                    for i in 0..inp_grad.len() {
                        inp_grad[i] += softmax_cache[i] * (grad_out[i] - sum_term);
                    }
                }),
            })),
        }
    }

    /// Log-softmax along specified dimension: log(softmax(x)) along dim.
    /// Fused for efficiency with numerical stability.
    pub fn log_softmax_dim(&self, dim: usize) -> Tensor {
        assert!(dim < self.shape.len(), "dim out of bounds");
        let self_data = self.data.read().unwrap();
        let shape = &self.shape;
        let rank = shape.len();

        // Compute strides for multi-dimensional indexing
        let mut strides = vec![0usize; rank];
        strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let dim_size = shape[dim];
        let num_slices: usize = self_data.len() / dim_size;

        let mut data = vec![0.0; self_data.len()];
        let mut softmax_values = vec![0.0; self_data.len()];

        // Forward pass: compute softmax per slice
        for slice_idx in 0..num_slices {
            // Compute max in this slice
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..dim_size {
                let linear_idx = slice_idx * dim_size + j;
                if self_data[linear_idx] > max_val {
                    max_val = self_data[linear_idx];
                }
            }

            // Compute sum of exp(x - max) in this slice
            let mut sum_exp = 0.0;
            for j in 0..dim_size {
                let linear_idx = slice_idx * dim_size + j;
                sum_exp += (self_data[linear_idx] - max_val).exp();
            }
            sum_exp = sum_exp.max(f64::MIN_POSITIVE);
            let log_sum_exp = sum_exp.ln() + max_val;

            // Compute output and cache softmax values
            for j in 0..dim_size {
                let linear_idx = slice_idx * dim_size + j;
                let softmax_val = (self_data[linear_idx] - max_val).exp() / sum_exp;
                data[linear_idx] = self_data[linear_idx] - log_sum_exp;
                softmax_values[linear_idx] = softmax_val;
            }
        }

        let softmax_cache: Arc<Vec<f64>> = Arc::new(softmax_values);
        let parents = vec![self.clone()];
        let dim_size_cap = dim_size;
        let num_slices_cap = num_slices;
        let softmax_cache_for_backward = softmax_cache.clone();

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    // Precompute sum_term per slice: sum(grad_out * softmax)
                    let mut sum_terms = vec![0.0; num_slices_cap];
                    for (slice_idx, sum_term) in sum_terms.iter_mut().enumerate() {
                        let base = slice_idx * dim_size_cap;
                        let mut slice_sum = 0.0;
                        for j in 0..dim_size_cap {
                            let idx = base + j;
                            slice_sum += grad_out[idx] * softmax_cache_for_backward[idx];
                        }
                        *sum_term = slice_sum;
                    }

                    let mut inp_grad = parents[0].grad.write().unwrap();
                    // Manually compute slice_idx and idx to avoid needless iteration
                    #[allow(clippy::needless_range_loop)]
                    for slice_idx in 0..num_slices_cap {
                        let base = slice_idx * dim_size_cap;
                        for j in 0..dim_size_cap {
                            let idx = base + j;
                            inp_grad[idx] += softmax_cache_for_backward[idx]
                                * (grad_out[idx] - sum_terms[slice_idx]);
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
        let self_data = self.data.read().unwrap();
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
            let std = (var[i] + eps).sqrt();
            for j in 0..last_dim {
                output[base + j] = (self_data[base + j] - m) / std;
            }
        }

        // Store input data in Arc so backward pass can access it
        let input_data = (*self_data).clone();
        let mean_arc = Arc::new(mean);
        let var_arc = Arc::new(var);
        let last_dim_f = last_dim as f64;
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(output)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();

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
                            g_sum += grad_out[base + j];
                            g_diff_sum += grad_out[base + j] * diff;
                        }

                        // dvar = sum grad_out * (x - mean) * -0.5 / std^3
                        let dvar = -0.5 * g_diff_sum / std3;
                        // dmean = -sum(grad_out) / std + dvar * -2 * mean / N
                        let dmean = -g_sum / std + dvar * -2.0 * m / last_dim_f;

                        // dx_j = grad_out_j / std + dvar * 2 * (x_j - m) / N + dmean / N
                        for j in 0..last_dim {
                            let diff = input_data[base + j] - m;
                            let dx = grad_out[base + j] / std
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
        let self_data = self.data.read().unwrap();
        let val = self_data[idx];
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(vec![val])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    if idx < inp_grad.len() {
                        inp_grad[idx] += grad_out[0];
                    }
                }),
            })),
        }
    }

    pub fn sum(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let sum_val: f64 = if len >= PAR_THRESHOLD {
            self_data.par_iter().sum()
        } else {
            horizontal_sum(&self_data)
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(vec![sum_val])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let g = grad_out[0];
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
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let sum_val: f64 = if len >= PAR_THRESHOLD {
            self_data.par_iter().sum()
        } else {
            self_data.iter().sum()
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(vec![sum_val / len as f64])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let g = grad_out[0] / len as f64;
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

    pub fn transpose2d(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose requires 2D tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let self_data = self.data.read().unwrap();
        let mut out_data = vec![0.0; self_data.len()];
        for r in 0..rows {
            for c in 0..cols {
                out_data[c * rows + r] = self_data[r * cols + c];
            }
        }
        let parents = vec![self.clone()];
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; rows * cols])),
            shape: vec![cols, rows],
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();
                    for r in 0..rows {
                        for c in 0..cols {
                            inp_grad[r * cols + c] += grad_out[c * rows + r];
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

        let self_data = self.data.read().unwrap();
        let old_data = &*self_data;

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
                // Input position is pos when old_dim > 1 (broadcast replicates along dim=1)
                if old_dim != 1 {
                    old_linear_idx += pos * multiplier;
                }
                multiplier *= new_dim;
            }

            new_data.push(old_data[old_linear_idx]);
        }

        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; total_elements])),
            shape: new_shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let total_elements = grad_out.len();
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

                        inp_grad[old_linear_idx] += grad_out[linear_idx];
                    }
                }),
            })),
        }
    }

    pub fn broadcast_to_batch(&self, batch_size: usize) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let mut new_data = Vec::with_capacity(len * batch_size);
        for _ in 0..batch_size {
            new_data.extend_from_slice(&self_data);
        }

        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(&self.shape);

        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; len * batch_size])),
            shape: new_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    // Sum gradients across batch dimension
                    // grad_out is (Batch, D...)
                    // inp_grad is (D...)
                    let chunk_size = inp_grad.len();

                    // Parallel accumulation could be tricky without extra buffer,
                    // but simple serial sum over batch chunks is likely fast enough compared to matmul
                    for chunk in grad_out.chunks(chunk_size) {
                        for (i, &g) in chunk.iter().enumerate() {
                            inp_grad[i] += g;
                        }
                    }
                }),
            })),
        }
    }

    pub fn sin(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.sin()).collect()
        } else {
            self_data.iter().map(|&x| x.sin()).collect()
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig += g * id.cos();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] += grad_out[i] * input_data[i].cos();
                        }
                    }
                }),
            })),
        }
    }

    pub fn cos(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.cos()).collect()
        } else {
            self_data.iter().map(|&x| x.cos()).collect()
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    if inp_grad.len() >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                *ig -= g * id.sin();
                            });
                    } else {
                        for i in 0..inp_grad.len() {
                            inp_grad[i] -= grad_out[i] * input_data[i].sin();
                        }
                    }
                }),
            })),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.sqrt()).collect()
        } else {
            self_data.iter().map(|&x| x.sqrt()).collect()
        };
        let sqrt_cache = data.clone();
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();
                    let len = inp_grad.len();
                    // Reuse cached sqrt values: d/dx sqrt(x) = 0.5 / sqrt(x)
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(sqrt_cache.par_iter())
                            .for_each(|((ig, &g), &s)| {
                                if s > 0.0 {
                                    *ig += g * 0.5 / s;
                                }
                            });
                    } else {
                        for i in 0..len {
                            if sqrt_cache[i] > 0.0 {
                                inp_grad[i] += grad_out[i] * 0.5 / sqrt_cache[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let self_data = self.data.read().unwrap();
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
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: new_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();

                    for (i, &grad_val) in grad_out.iter().enumerate() {
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
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| x.max(min).min(max)).collect()
        } else {
            self_data.iter().map(|&x| x.max(min).min(max)).collect()
        };
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    let len = inp_grad.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(input_data.par_iter())
                            .for_each(|((ig, &g), &id)| {
                                if id >= min && id <= max {
                                    *ig += g;
                                }
                            });
                    } else {
                        for i in 0..len {
                            if input_data[i] >= min && input_data[i] <= max {
                                inp_grad[i] += grad_out[i];
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let len: usize = new_shape.iter().product::<usize>();
        {
            let d = self.data.read().unwrap();
            assert_eq!(len, d.len(), "Reshape dimension mismatch");
        }

        // Zero-copy: share the same data Arc, only change shape metadata
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::clone(&self.data),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: new_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let len = grad_out.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .for_each(|(ig, &g)| *ig += g);
                    } else {
                        for i in 0..len {
                            inp_grad[i] += grad_out[i];
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
            let weight_data = weight.data.read().unwrap();

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
            let input_data = self.data.read().unwrap();

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
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; out_len])),
            shape: out_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                // Using standard Im2Col backward pass logic.
                backward_op: Box::new(move |grad_out, parents| {
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
                        let mut input_grad = input.grad.write().unwrap();
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
                                                            let g = grad_out[((b * c_out + k)
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
                        let mut weight_grad = weight.grad.write().unwrap();
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
                                                        let g_val = grad_out[((b * c_out + k)
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
            let input_data = self.data.read().unwrap();
            let weight_data = weight.data.read().unwrap();

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
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; out_len])),
            shape: out_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let weight = &parents[1];
                    // Use batch lock for better performance
                    let guards = TensorReadGuard::new(&[input, weight]);
                    let input_data = guards.get(0);
                    let weight_data = guards.get(1);

                    // dL/dInput
                    {
                        let mut input_grad = input.grad.write().unwrap();
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
                                                            let g = grad_out[((b * c_out + k)
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
                        let mut weight_grad = weight.grad.write().unwrap();
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
                                                        let g_val = grad_out[((b * c_out + k)
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
            let input_data = self.data.read().unwrap();
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
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; out_len])),
            shape: out_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut input_grad = input.grad.write().unwrap();

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
                                                        grad_sum +=
                                                            grad_out[((b * c + ch) * h_out + oh)
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

        let w_sum_data = w_sum.data.read().unwrap();
        let denom = if w_sum_data[0].abs() < 1e-12 {
            1.0
        } else {
            w_sum_data[0]
        };
        drop(w_sum_data);

        let total_data = total.data.read().unwrap();
        let result_val = total_data[0] / denom;
        drop(total_data);

        let parents = vec![total.clone(), w_sum.clone()];
        let denom_cap = denom;
        let numerator_cap = result_val;
        Tensor {
            data: Arc::new(RwLock::new(vec![result_val])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    // f = total / w_sum
                    // df/d(total) = 1 / w_sum
                    let mut total_grad = parents[0].grad.write().unwrap();
                    total_grad[0] += grad_out[0] / denom_cap;
                    drop(total_grad);
                    // df/d(w_sum) = -total / w_sum^2 = -f / w_sum
                    let mut wsum_grad = parents[1].grad.write().unwrap();
                    wsum_grad[0] += grad_out[0] * (-numerator_cap / denom_cap);
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let len = grad_out.len();
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, grad_out);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(rg, &g)| *rg += g);
                        } else {
                            vector_grad_acc(&mut rhs_grad, grad_out);
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let len = grad_out.len();
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, grad_out);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(rg, &g)| *rg += g);
                        } else {
                            vector_grad_acc(&mut rhs_grad, grad_out);
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let len = grad_out.len();
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, grad_out);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(rg, &g)| *rg -= g);
                        } else {
                            for i in 0..len {
                                rhs_grad[i] -= grad_out[i];
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let len = grad_out.len();
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            lhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(lg, &g)| *lg += g);
                        } else {
                            vector_grad_acc(&mut lhs_grad, grad_out);
                        }
                    }
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            rhs_grad
                                .par_iter_mut()
                                .zip(grad_out.par_iter())
                                .for_each(|(rg, &g)| *rg -= g);
                        } else {
                            for i in 0..len {
                                rhs_grad[i] -= grad_out[i];
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out.len();
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            grad.par_iter_mut()
                                .zip(grad_out.par_iter())
                                .zip(lhs_data.par_iter())
                                .zip(rhs_data.par_iter())
                                .for_each(|(((g, &go), &l), &r)| {
                                    *g += go * (l + r);
                                });
                        } else {
                            for i in 0..len {
                                grad[i] += grad_out[i] * (lhs_data[i] + rhs_data[i]);
                            }
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            if len >= PAR_THRESHOLD {
                                lhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out.par_iter())
                                    .zip(rhs_data.par_iter())
                                    .for_each(|((lg, &g), &r)| *lg += g * r);
                            } else {
                                for i in 0..len {
                                    lhs_grad[i] += grad_out[i] * rhs_data[i];
                                }
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            if len >= PAR_THRESHOLD {
                                rhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out.par_iter())
                                    .zip(lhs_data.par_iter())
                                    .for_each(|((rg, &g), &l)| *rg += g * l);
                            } else {
                                for i in 0..len {
                                    rhs_grad[i] += grad_out[i] * lhs_data[i];
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out.len();
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        for i in 0..len {
                            grad[i] += grad_out[i] * (lhs_data[i] + rhs_data[i]);
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            for i in 0..len {
                                lhs_grad[i] += grad_out[i] * rhs_data[i];
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            for i in 0..len {
                                rhs_grad[i] += grad_out[i] * lhs_data[i];
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out.len();
                    const DIV_EPS: f64 = 1e-12;
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        if len >= PAR_THRESHOLD {
                            grad.par_iter_mut()
                                .zip(grad_out.par_iter())
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
                                grad[i] += grad_out[i] / safe_r
                                    - grad_out[i] * lhs_data[i] / (safe_r * safe_r);
                            }
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            if len >= PAR_THRESHOLD {
                                lhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out.par_iter())
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
                                    lhs_grad[i] += grad_out[i] / safe_r;
                                }
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            if len >= PAR_THRESHOLD {
                                rhs_grad
                                    .par_iter_mut()
                                    .zip(grad_out.par_iter())
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
                                    rhs_grad[i] -= grad_out[i] * lhs_data[i] / (safe_r * safe_r);
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);
                    let len = grad_out.len();
                    const DIV_EPS: f64 = 1e-12;
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        for i in 0..len {
                            let r = rhs_data[i];
                            let safe_r = if r.abs() < DIV_EPS {
                                r.signum() * DIV_EPS
                            } else {
                                r
                            };
                            grad[i] += grad_out[i] / safe_r
                                - grad_out[i] * lhs_data[i] / (safe_r * safe_r);
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                lhs_grad[i] += grad_out[i] / safe_r;
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            for i in 0..len {
                                let r = rhs_data[i];
                                let safe_r = if r.abs() < DIV_EPS {
                                    r.signum() * DIV_EPS
                                } else {
                                    r
                                };
                                rhs_grad[i] -= grad_out[i] * lhs_data[i] / (safe_r * safe_r);
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
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            self_data.par_iter().map(|&x| -x).collect()
        } else {
            self_data.iter().map(|&x| -x).collect()
        };
        let parents = vec![self.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let len = grad_out.len();
                    if len >= PAR_THRESHOLD {
                        inp_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .for_each(|(ig, &g)| *ig -= g);
                    } else {
                        for i in 0..len {
                            inp_grad[i] -= grad_out[i];
                        }
                    }
                }),
            })),
        }
    }
}

#[cfg(cuda)]
impl Tensor {
    /// GPU-accelerated matrix multiplication using CUDA cuBLAS
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn matmul_cuda(&self, other: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
        self.matmul_cpu_fallback(other, m, k, n)
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
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; m * n])),
            shape: out_shape,
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let guards = TensorReadGuard::new(&[lhs, rhs]);
                    let lhs_data = guards.get(0);
                    let rhs_data = guards.get(1);

                    {
                        // dL/dLHS
                        let mut lhs_grad = lhs.grad.write().unwrap();
                        for r in 0..m {
                            for i in 0..k {
                                lhs_grad[r * k + i] += dot_product(
                                    &grad_out[r * n..r * n + n],
                                    &rhs_data[i * n..i * n + n],
                                );
                            }
                        }
                    }

                    {
                        // dL/dRHS
                        let mut rhs_grad = rhs.grad.write().unwrap();
                        for i in 0..k {
                            for j in 0..n {
                                for r in 0..m {
                                    rhs_grad[i * n + j] +=
                                        lhs_data[r * k + i] * grad_out[r * n + j];
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
        self.gelu_cpu_fallback()
    }

    /// CPU fallback for GELU
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn gelu_cpu_fallback(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
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
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    for i in 0..inp_grad.len() {
                        let x = input_data[i];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let u = sqrt_2_over_pi * (x + c * x3);
                        let tanh_u = u.tanh();
                        let sech2_u = 1.0 - tanh_u * tanh_u;
                        let du_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x2);
                        let gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2_u * du_dx;
                        inp_grad[i] += grad_out[i] * gelu_grad;
                    }
                }),
            })),
        }
    }

    /// GPU-accelerated Softmax activation using CUDA kernel
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn softmax_cuda(&self) -> Tensor {
        self.softmax_cpu_fallback()
    }

    /// CPU fallback for Softmax
    #[cfg(cuda)]
    #[allow(dead_code)]
    fn softmax_cpu_fallback(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let max_val = self_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f64 = self_data
            .iter()
            .map(|&x| (x - max_val).exp())
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        let data: Vec<f64> = self_data
            .iter()
            .map(|&x| (x - max_val).exp() / sum_exp)
            .collect();

        let softmax_cache: Arc<Vec<f64>> = Arc::new(data.clone());
        let parents = vec![self.clone()];

        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: self.shape.clone(),
            device: Device::Cpu,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let sum_term: f64 = grad_out
                        .iter()
                        .zip(softmax_cache.iter())
                        .map(|(&g, &s)| g * s)
                        .sum();
                    for i in 0..inp_grad.len() {
                        inp_grad[i] += softmax_cache[i] * (grad_out[i] - sum_term);
                    }
                }),
            })),
        }
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
        let t = Tensor::new(vec![5.0], vec![1]);
        let b = t.broadcast(vec![2, 2]);
        assert_eq!(b.shape, vec![2, 2]);
        let data = b.data.read().unwrap();
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
        let tensor = Tensor::new(vec![1.25, -2.5], vec![2]);
        tensor.save_binary(&path).unwrap();

        let loaded = Tensor::from_mmap(&path, vec![2]).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.shape, vec![2]);
        let data = loaded.data.read().unwrap();
        assert_eq!(*data, vec![1.25, -2.5]);
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
        let data = out.data.read().unwrap();
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
        let d = out.data.read().unwrap();
        assert_eq!(*d, vec![5.0, 7.0, 13.0, 15.0]);
    }
}
