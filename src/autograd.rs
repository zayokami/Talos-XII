use std::sync::{Arc, RwLock};
use std::ops::{Add, Sub, Mul, Div, Neg};
use rayon::prelude::*;
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::ser::SerializeStruct;
use std::fs::File;
use memmap2::Mmap;
use crate::simd::{add_scaled_row, dot_product, vector_fma};

// --- Autograd Engine ---

// TensorData removed as we are using direct mmap-to-vec for stability for now.


#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<RwLock<Vec<f64>>>, // Kept as Vec<f64> for now to minimize refactoring pain, but we can load from mmap
    pub grad: Arc<RwLock<Vec<f64>>>,
    pub shape: Vec<usize>,
    pub _ctx: Option<Arc<Context>>, // Keeps the graph alive
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

pub struct Context {
    pub parents: Vec<Tensor>,
    pub backward_op: Box<dyn Fn(&Vec<f64>, &Vec<Tensor>) + Send + Sync>, // receives grad_output, parents
}

impl Tensor {
    pub fn from_mmap(path: &str, shape: Vec<usize>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let bytes = &mmap[..];
        let len = bytes.len() / std::mem::size_of::<f64>();
        let expected_len: usize = shape.iter().product();
        if len != expected_len {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Mmap size mismatch"));
        }
        
        // Zero-copy cast (unsafe but fast)
        let slice = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, len) };
        let data = slice.to_vec(); // We still copy to Vec because RwLock needs ownership. 
        // True zero-copy requires changing Tensor.data to Cow<'a, [f64]> or similar which is a huge refactor.
        // But mmap loading is still much faster than JSON parsing!
        
        Ok(Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape,
            _ctx: None,
        })
    }
    
    pub fn save_binary(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = File::create(path)?;
        let data = self.data.read().unwrap();
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
        file.write_all(bytes)?;
        Ok(())
    }

    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let len = data.len();
        assert_eq!(len, shape.iter().product::<usize>(), "Data length must match shape");
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape,
            _ctx: None,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>();
        Tensor::new(vec![0.0; len], shape)
    }
    
    pub fn rand(shape: Vec<usize>, min: f64, max: f64, seed: u64) -> Self {
        let len = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(len);
        let mut x = seed;
        for _ in 0..len {
             x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
             let r = (x >> 33) as f64 / u32::MAX as f64;
             data.push(min + r * (max - min));
        }
        Tensor::new(data, shape)
    }

    pub fn detach(&self) -> Tensor {
        Tensor {
            data: self.data.clone(), // Share data
            grad: self.grad.clone(), // Share grad? Or new grad? 
            // Typically detach() creates a new Tensor that shares data but has NO graph history.
            // But if we modify it, should it affect original? Yes.
            // PyTorch detach: shares storage, requires_grad=False.
            // Here we just want to break the graph.
            shape: self.shape.clone(),
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
        fn build_topo(t: &Tensor, visited: &mut std::collections::HashSet<usize>, topo: &mut Vec<Tensor>) {
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
            for v in g.iter_mut() { *v = 1.0; }
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
        for v in g.iter_mut() { *v = 0.0; }
    }
    
    // Operations
    
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() <= 2 && other.shape.len() == 2);
        
        let (m, k) = if self.shape.len() == 1 { (1, self.shape[0]) } else { (self.shape[0], self.shape[1]) };
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "MatMul dimension mismatch");
        
        let mut out_data = vec![0.0; m * n];
        
        {
            let lhs_data = self.data.read().unwrap();
            let rhs_data = other.data.read().unwrap();
            
            // Heuristic for parallelization overhead
            // A simple matmul has M*N*K ops.
            // Rayon overhead is small but significant for tiny matrices.
            let ops = m * n * k;
            
            if ops < 32768 { // Serial path for small matrices (e.g. 64x64x5)
                for r in 0..m {
                    let out_row_start = r * n;
                    for i in 0..k {
                        let scale = lhs_data[r * k + i];
                        let rhs_row_start = i * n;
                        for c in 0..n {
                            out_data[out_row_start + c] += scale * rhs_data[rhs_row_start + c];
                        }
                    }
                }
            } else { // Parallel path
                out_data.par_chunks_mut(n).enumerate().for_each(|(r, out_row)| {
                    // Optimized Row-Accumulation (Cache Friendly)
                    // C[r, :] += A[r, i] * B[i, :]
                    for i in 0..k {
                        let scale = lhs_data[r * k + i];
                        let rhs_row_start = i * n;
                        // Inner loop: sequential access on B and C -> SIMD friendly
                        for c in 0..n {
                            out_row[c] += scale * rhs_data[rhs_row_start + c];
                        }
                    }
                });
            }
        }
        
        let out_shape = if self.shape.len() == 1 { vec![n] } else { vec![m, n] };
        
        let parents = vec![self.clone(), other.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; m * n])),
            shape: out_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    
                    let lhs_data = lhs.data.read().unwrap();
                    let rhs_data = rhs.data.read().unwrap();
                    
                    // dL/dLHS = grad_out * RHS^T
                    {
                        let mut lhs_grad = lhs.grad.write().unwrap();
                        let ops = m * k * n;
                        if ops < 32768 {
                             // Serial
                             for r in 0..m {
                                 let grad_out_row_start = r * n;
                                 let lhs_grad_row_start = r * k;
                                 for i in 0..k {
                                     let rhs_row_start = i * n;
                                     let grad_row = &grad_out[grad_out_row_start..grad_out_row_start + n];
                                     let rhs_row = &rhs_data[rhs_row_start..rhs_row_start + n];
                                     lhs_grad[lhs_grad_row_start + i] += dot_product(grad_row, rhs_row);
                                 }
                             }
                        } else {
                            lhs_grad.par_chunks_mut(k).enumerate().for_each(|(r, lhs_row)| {
                                let grad_out_row_start = r * n;
                                let grad_row = &grad_out[grad_out_row_start..grad_out_row_start + n];
                                for i in 0..k {
                                    let rhs_row_start = i * n;
                                    let rhs_row = &rhs_data[rhs_row_start..rhs_row_start + n];
                                    lhs_row[i] += dot_product(grad_row, rhs_row);
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
                                 let rhs_row = &mut rhs_grad[rhs_grad_row_start..rhs_grad_row_start + n];
                                 for r in 0..m {
                                     let scale = lhs_data[r * k + i];
                                     if scale == 0.0 { continue; }
                                     let grad_out_row_start = r * n;
                                     let grad_row = &grad_out[grad_out_row_start..grad_out_row_start + n];
                                     add_scaled_row(rhs_row, grad_row, scale);
                                 }
                             }
                         } else {
                            rhs_grad.par_chunks_mut(n).enumerate().for_each(|(i, rhs_row)| {
                                for r in 0..m {
                                    let scale = lhs_data[r * k + i];
                                    if scale == 0.0 { continue; }
                                    let grad_out_row_start = r * n;
                                    let grad_row = &grad_out[grad_out_row_start..grad_out_row_start + n];
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
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).zip(input_data.par_iter()).for_each(|((ig, &go), &val)| {
                        if val > 0.0 {
                            *ig += go;
                        }
                    });
                }),
            })),
        }
    }

    pub fn log(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().map(|&x| x.ln()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).zip(input_data.par_iter()).for_each(|((ig, &g), &id)| {
                        *ig += g / id;
                    });
                }),
            })),
        }
    }

    pub fn exp(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().map(|&x| x.exp()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    // d/dx(e^x) = e^x
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).zip(input_data.par_iter()).for_each(|((ig, &g), &id)| {
                        *ig += g * id.exp();
                    });
                }),
            })),
        }
    }

    pub fn sum(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let sum_val: f64 = self_data.par_iter().sum();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(vec![sum_val])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let g = grad_out[0];
                    inp_grad.par_iter_mut().for_each(|v| *v += g);
                }),
            })),
        }
    }

    pub fn mean(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let sum_val: f64 = self_data.par_iter().sum();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(vec![sum_val / len as f64])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let g = grad_out[0] / len as f64;
                    inp_grad.par_iter_mut().for_each(|v| *v += g);
                }),
            })),
        }
    }

    pub fn broadcast(&self, new_shape: Vec<usize>) -> Tensor {
        assert_eq!(self.shape, vec![1], "Broadcast only supported for scalar (1) to N");
        let len = new_shape.iter().product();
        let val = self.data.read().unwrap()[0];
        let data = vec![val; len];
        
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: new_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let sum_grad: f64 = grad_out.par_iter().sum();
                    inp_grad[0] += sum_grad;
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
        let data: Vec<f64> = self_data.par_iter().map(|&x| x.sin()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).zip(input_data.par_iter()).for_each(|((ig, &g), &id)| {
                        *ig += g * id.cos();
                    });
                }),
            })),
        }
    }

    pub fn cos(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().map(|&x| x.cos()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).zip(input_data.par_iter()).for_each(|((ig, &g), &id)| {
                        *ig -= g * id.sin();
                    });
                }),
            })),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().map(|&x| x.sqrt()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).zip(input_data.par_iter()).for_each(|((ig, &g), &id)| {
                        if id > 0.0 {
                            *ig += g * 0.5 / id.sqrt();
                        }
                    });
                }),
            })),
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        // Full data copy transpose for simplicity
        let self_data = self.data.read().unwrap();
        let shape = &self.shape;
        let rank = shape.len();
        assert!(dim0 < rank && dim1 < rank);
        
        let mut new_shape = shape.clone();
        new_shape.swap(dim0, dim1);
        
        let len = self_data.len();
        let mut new_data = vec![0.0; len];
        
        // Calculate strides
        let mut strides = vec![1; rank];
        for i in (0..rank-1).rev() {
            strides[i] = strides[i+1] * shape[i+1];
        }
        
        let mut new_strides = vec![1; rank];
        for i in (0..rank-1).rev() {
            new_strides[i] = new_strides[i+1] * new_shape[i+1];
        }
        
        // Iterate and copy
        // This is generic N-dim transpose
        for i in 0..len {
            // Unravel index 'i' based on new_shape
            let mut temp = i;
            let mut coords = vec![0; rank];
            for d in 0..rank {
                coords[d] = temp / new_strides[d];
                temp %= new_strides[d];
            }
            
            // Swap coords to get old coordinates
            coords.swap(dim0, dim1);
            
            // Ravel coords based on old shape (strides)
            let mut old_idx = 0;
            for d in 0..rank {
                old_idx += coords[d] * strides[d];
            }
            
            new_data[i] = self_data[old_idx];
        }
        
        let parents = vec![self.clone()];
        let dim0_cap = dim0;
        let dim1_cap = dim1;
        
        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: new_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();
                    
                    // Transpose backward is transpose of grad
                    // We need to map grad_out (which is in transposed shape) back to input shape
                    
                    let shape = &input.shape; // Old shape
                    let rank = shape.len();
                    
                    // Recompute strides (captured closure would be better but expensive to copy vecs)
                    let mut strides = vec![1; rank];
                    for i in (0..rank-1).rev() {
                        strides[i] = strides[i+1] * shape[i+1];
                    }
                    
                    // New shape (transposed)
                    let mut new_shape = shape.clone();
                    new_shape.swap(dim0_cap, dim1_cap);
                    let mut new_strides = vec![1; rank];
                    for i in (0..rank-1).rev() {
                        new_strides[i] = new_strides[i+1] * new_shape[i+1];
                    }
                    
                    for i in 0..grad_out.len() {
                        // i is index in grad_out (transposed layout)
                        // We want to find corresponding index in inp_grad (original layout)
                        
                        // Unravel i using new_strides
                        let mut temp = i;
                        let mut coords = vec![0; rank];
                        for d in 0..rank {
                            coords[d] = temp / new_strides[d];
                            temp %= new_strides[d];
                        }
                        
                        // Swap coords back
                        coords.swap(dim0_cap, dim1_cap);
                        
                        // Ravel using strides
                        let mut old_idx = 0;
                        for d in 0..rank {
                            old_idx += coords[d] * strides[d];
                        }
                        
                        inp_grad[old_idx] += grad_out[i];
                    }
                }),
            })),
        }
    }
    
    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().map(|&x| x.max(min).min(max)).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).zip(input_data.par_iter()).for_each(|((ig, &g), &id)| {
                        if id >= min && id <= max {
                            *ig += g;
                        }
                    });
                }),
            })),
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len: usize = new_shape.iter().product::<usize>();
        assert_eq!(len, self_data.len(), "Reshape dimension mismatch");
        
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(self_data.clone())),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: new_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).for_each(|(ig, &g)| *ig += g);
                }),
            })),
        }
    }
    
    // Winograd F(2x2, 3x3) implementation
    // Input tile: 4x4, Output tile: 2x2
    fn winograd_conv2d_3x3(&self, weight: &Tensor, padding: usize) -> Tensor {
        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, _, _, _) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
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
            u_data.par_chunks_mut(16).enumerate().for_each(|(idx, u_block)| {
                // idx corresponds to (k * c_in + c)
                let k = idx / c_in;
                let c = idx % c_in;
                
                // Read 3x3 weight
                let w_base = (k * c_in + c) * 9;
                let g00 = weight_data[w_base + 0]; let g01 = weight_data[w_base + 1]; let g02 = weight_data[w_base + 2];
                let g10 = weight_data[w_base + 3]; let g11 = weight_data[w_base + 4]; let g12 = weight_data[w_base + 5];
                let g20 = weight_data[w_base + 6]; let g21 = weight_data[w_base + 7]; let g22 = weight_data[w_base + 8];
                
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
                u_block[0]  = t00;
                u_block[4]  = 0.5 * (t00 + t10 + t20);
                u_block[8]  = 0.5 * (t00 - t10 + t20);
                u_block[12] = t20;
                
                u_block[1]  = t01;
                u_block[5]  = 0.5 * (t01 + t11 + t21);
                u_block[9]  = 0.5 * (t01 - t11 + t21);
                u_block[13] = t21;
                
                u_block[2]  = t02;
                u_block[6]  = 0.5 * (t02 + t12 + t22);
                u_block[10] = 0.5 * (t02 - t12 + t22);
                u_block[14] = t22;
                
                u_block[3]  = t03;
                u_block[7]  = 0.5 * (t03 + t13 + t23);
                u_block[11] = 0.5 * (t03 - t13 + t23);
                u_block[15] = t23;
            });
        }
        
        {
            let input_data = self.data.read().unwrap();
            
            // Output is computed in 2x2 blocks (tiles).
            let n_tiles_h = (h_out + 1) / 2;
            let n_tiles_w = (w_out + 1) / 2;
            let n_tiles = n_tiles_h * n_tiles_w;
            
            let out_plane_len = h_out * w_out;
            
            out_data.par_chunks_mut(c_out * out_plane_len).enumerate().for_each(|(b, out_batch)| {
                // We could parallelize over tiles, but that requires atomic writes to output or careful locking.
                // Easier to parallelize over Output Channels (C_out) since they are independent.
                
                // First, transform input image into V domain: V = B^T d B.
                // This is shared across all C_out, so we do it once per batch item.
                // V: [Tiles, C_in, 4, 4]
                let mut v_data = vec![0.0; n_tiles * c_in * 16];
                
                // Parallelize V computation over (Tile, C_in)
                v_data.par_chunks_mut(16).enumerate().for_each(|(idx, v_block)| {
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
                            if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                d[i*4 + j] = input_data[((b * c_in + c) * h_in + ih as usize) * w_in + iw as usize];
                            }
                        }
                    }
                    
                    // Compute V = B^T * d * B
                    // 1. Tmp = B^T * d
                    let mut tmp = [0.0; 16];
                    for j in 0..4 { // col j
                        let d0 = d[0*4 + j]; let d1 = d[1*4 + j]; let d2 = d[2*4 + j]; let d3 = d[3*4 + j];
                        tmp[0*4 + j] = d0 - d2;
                        tmp[1*4 + j] = d1 + d2;
                        tmp[2*4 + j] = d2 - d1;
                        tmp[3*4 + j] = d1 - d3;
                    }
                    
                    // 2. V = Tmp * B
                    for i in 0..4 { // row i
                        let t0 = tmp[i*4 + 0]; let t1 = tmp[i*4 + 1]; let t2 = tmp[i*4 + 2]; let t3 = tmp[i*4 + 3];
                        v_block[i*4 + 0] = t0 - t2;
                        v_block[i*4 + 1] = t1 + t2;
                        v_block[i*4 + 2] = t2 - t1;
                        v_block[i*4 + 3] = t1 - t3;
                    }
                });
                
                // Now Compute M = U * V and Y = A^T M A
                // This part is specific to each C_out.
                out_batch.par_chunks_mut(out_plane_len).enumerate().for_each(|(k, out_plane)| {
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
                             let m0 = m[0*4 + j]; let m1 = m[1*4 + j]; let m2 = m[2*4 + j]; let m3 = m[3*4 + j];
                             tmp[0*4 + j] = m0 + m1 + m2;
                             tmp[1*4 + j] = m1 - m2 - m3;
                         }
                         
                         // 2. Y = Tmp * A
                         let t00 = tmp[0]; let t01 = tmp[1]; let t02 = tmp[2]; let t03 = tmp[3];
                         let t10 = tmp[4]; let t11 = tmp[5]; let t12 = tmp[6]; let t13 = tmp[7];
                         
                         let y00 = t00 + t01 + t02;
                         let y01 = t01 - t02 - t03;
                         let y10 = t10 + t11 + t12;
                         let y11 = t11 - t12 - t13;
                         
                         // Scatter write to output
                         let oh_base = th * 2;
                         let ow_base = tw * 2;
                         
                         if oh_base < h_out && ow_base < w_out { out_plane[oh_base * w_out + ow_base] = y00; }
                         if oh_base < h_out && ow_base + 1 < w_out { out_plane[oh_base * w_out + ow_base + 1] = y01; }
                         if oh_base + 1 < h_out && ow_base < w_out { out_plane[(oh_base + 1) * w_out + ow_base] = y10; }
                         if oh_base + 1 < h_out && ow_base + 1 < w_out { out_plane[(oh_base + 1) * w_out + ow_base + 1] = y11; }
                     }
                });
            });
        }
        
        // Backward pass: Just use the standard implementation.
        // Gradients don't care about the forward algorithm.
        
        let parents = vec![self.clone(), weight.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; out_len])),
            shape: out_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                // Using standard Im2Col backward pass logic.
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let weight = &parents[1];
                    let input_data = input.data.read().unwrap();
                    let weight_data = weight.data.read().unwrap();
                    
                    let (n, c_in, h_in, w_in) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
                    let (c_out, _, k_h, k_w) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
                    
                    // Stride is 1, Padding is `padding` (captured)
                    let stride = 1;
                    let h_out = h_in + 2 * padding - 2;
                    let w_out = w_in + 2 * padding - 2; // k_h=3, k_w=3
                    
                    // dL/dInput (Standard Col2Im)
                    {
                        let mut input_grad = input.grad.write().unwrap();
                         input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(|(idx, in_plane)| {
                            let b = idx / c_in;
                            let c = idx % c_in;
                            
                            for ih in 0..h_in {
                                let oh_min = (ih + padding).saturating_sub(k_h - 1) / stride;
                                let oh_max = ((ih + padding) / stride).min(h_out - 1);
                                
                                for iw in 0..w_in {
                                    let mut sum = 0.0;
                                    let ow_min = (iw + padding).saturating_sub(k_w - 1) / stride;
                                    let ow_max = ((iw + padding) / stride).min(w_out - 1);
                                    
                                    if oh_min <= oh_max && ow_min <= ow_max {
                                        for oh in oh_min..=oh_max {
                                            for ow in ow_min..=ow_max {
                                                 let kh = ih as isize - (oh * stride) as isize + padding as isize;
                                                 let kw = iw as isize - (ow * stride) as isize + padding as isize;
                                                 
                                                 if kh >= 0 && kh < k_h as isize && kw >= 0 && kw < k_w as isize {
                                                     for k in 0..c_out {
                                                         let g = grad_out[((b * c_out + k) * h_out + oh) * w_out + ow];
                                                         let w = weight_data[((k * c_in + c) * k_h + kh as usize) * k_w + kw as usize];
                                                         sum += g * w;
                                                     }
                                                 }
                                            }
                                        }
                                    }
                                    in_plane[ih * w_in + iw] += sum;
                                }
                            }
                         });
                    }
                    
                    // dL/dWeight
                    {
                        let mut weight_grad = weight.grad.write().unwrap();
                        weight_grad.par_chunks_mut(k_h * k_w).enumerate().for_each(|(idx, w_plane)| {
                             let k = idx / c_in;
                             let c = idx % c_in;
                             
                             for kh in 0..k_h {
                                 for kw in 0..k_w {
                                     let mut sum = 0.0;
                                     for b in 0..n {
                                         for oh in 0..h_out {
                                             for ow in 0..w_out {
                                                let h_in_idx = (oh * stride) as isize - padding as isize + kh as isize;
                                                let w_in_idx = (ow * stride) as isize - padding as isize + kw as isize;
                                                
                                                if h_in_idx >= 0 && h_in_idx < h_in as isize && w_in_idx >= 0 && w_in_idx < w_in as isize {
                                                    let val_in = input_data[((b * c_in + c) * h_in + h_in_idx as usize) * w_in + w_in_idx as usize];
                                                    let g_val = grad_out[((b * c_out + k) * h_out + oh) * w_out + ow];
                                                    sum += val_in * g_val;
                                                }
                                             }
                                         }
                                     }
                                     w_plane[kh * k_w + kw] += sum;
                                 }
                             }
                        });
                    }
                }),
            })),
        }
    }

    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        assert_eq!(weight.shape.len(), 4, "Weight must be 4D (OIHW)");
        
        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, c_in_k, k_h, k_w) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
        
        assert_eq!(c_in, c_in_k, "Input channels must match weight input channels");
        
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
            out_data.par_chunks_mut(c_out * out_plane_len).enumerate().for_each(|(b, out_batch)| {
                 // Im2Col: Input (C_in, H, W) -> Cols (K_len, Out_len)
                 let mut cols = vec![0.0; k_len * out_plane_len];
                 
                 // Parallelize filling cols (by kernel rows)
                 cols.par_chunks_mut(out_plane_len).enumerate().for_each(|(k_idx, col_row)| {
                     let c = k_idx / (k_h * k_w);
                     let rem = k_idx % (k_h * k_w);
                     let kh = rem / k_w;
                     let kw = rem % k_w;
                     
                     for oh in 0..h_out {
                         for ow in 0..w_out {
                             let h_in_idx = (oh * stride) as isize - padding as isize + kh as isize;
                             let w_in_idx = (ow * stride) as isize - padding as isize + kw as isize;
                             
                             if h_in_idx >= 0 && h_in_idx < h_in as isize && w_in_idx >= 0 && w_in_idx < w_in as isize {
                                 col_row[oh * w_out + ow] = input_data[((b * c_in + c) * h_in + h_in_idx as usize) * w_in + w_in_idx as usize];
                             }
                         }
                     }
                 });

                 // GEMM: Weight (C_out, K_len) * Cols (K_len, Out_len) -> Out (C_out, Out_len)
                 // out_batch is already slice of size C_out * Out_len
                 
                 // Iterate over output rows (C_out)
                 out_batch.par_chunks_mut(out_plane_len).enumerate().for_each(|(out_c, out_row)| {
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
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let weight = &parents[1];
                    let input_data = input.data.read().unwrap();
                    let weight_data = weight.data.read().unwrap();
                    
                    // dL/dInput
                    {
                        let mut input_grad = input.grad.write().unwrap();
                        // Parallel over Input (N, C_in)
                         input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(|(idx, in_plane)| {
                            let b = idx / c_in;
                            let c = idx % c_in;
                            
                            // Optimized Col2Im (Transposed Conv)
                            for ih in 0..h_in {
                                // Pre-calculate bounds to avoid inner loop checks
                                let oh_min = (ih + padding).saturating_sub(k_h - 1) / stride;
                                let oh_max = ((ih + padding) / stride).min(h_out - 1);
                                
                                for iw in 0..w_in {
                                    let mut sum = 0.0;
                                    let ow_min = (iw + padding).saturating_sub(k_w - 1) / stride;
                                    let ow_max = ((iw + padding) / stride).min(w_out - 1);

                                    // Check if range is valid (could be empty if padding is small/large)
                                    if oh_min <= oh_max && ow_min <= ow_max {
                                        for oh in oh_min..=oh_max {
                                            for ow in ow_min..=ow_max {
                                                 // ih = oh*s - p + kh => kh = ih - oh*s + p
                                                 let kh = ih as isize - (oh * stride) as isize + padding as isize;
                                                 let kw = iw as isize - (ow * stride) as isize + padding as isize;
                                                 
                                                 if kh >= 0 && kh < k_h as isize && kw >= 0 && kw < k_w as isize {
                                                     // Should always be true given bounds, but stride check needed?
                                                     // If we iterate oh, ow, kh is determined.
                                                     
                                                     for k in 0..c_out {
                                                         let g = grad_out[((b * c_out + k) * h_out + oh) * w_out + ow];
                                                         let w = weight_data[((k * c_in + c) * k_h + kh as usize) * k_w + kw as usize];
                                                         sum += g * w;
                                                     }
                                                 }
                                            }
                                        }
                                    }
                                    in_plane[ih * w_in + iw] += sum;
                                }
                            }
                         });
                    }
                    
                    // dL/dWeight
                    {
                        let mut weight_grad = weight.grad.write().unwrap();
                        // dWeight = grad_out * Input_Cols^T
                        // Implemented via manual accumulation over batch
                        
                        // Direct loop is safer for memory.
                        
                        // Parallel over Weight (C_out, C_in, KH, KW)
                        weight_grad.par_chunks_mut(k_h * k_w).enumerate().for_each(|(idx, w_plane)| {
                             let k = idx / c_in;
                             let c = idx % c_in;
                             
                             for kh in 0..k_h {
                                 for kw in 0..k_w {
                                     let mut sum = 0.0;
                                     for b in 0..n {
                                         for oh in 0..h_out {
                                             for ow in 0..w_out {
                                                let h_in_idx = (oh * stride) as isize - padding as isize + kh as isize;
                                                let w_in_idx = (ow * stride) as isize - padding as isize + kw as isize;
                                                
                                                if h_in_idx >= 0 && h_in_idx < h_in as isize && w_in_idx >= 0 && w_in_idx < w_in as isize {
                                                    let val_in = input_data[((b * c_in + c) * h_in + h_in_idx as usize) * w_in + w_in_idx as usize];
                                                    let g_val = grad_out[((b * c_out + k) * h_out + oh) * w_out + ow];
                                                    sum += val_in * g_val;
                                                }
                                             }
                                         }
                                     }
                                     w_plane[kh * k_w + kw] += sum;
                                 }
                             }
                        });
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
             out_data.par_chunks_mut(h_out * w_out).enumerate().for_each(|(idx, out_plane)| {
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
                         let kh_end = if h_start + kernel_size as isize > h_in as isize { (h_in as isize - h_start) as usize } else { kernel_size };
                         let kw_end = if w_start + kernel_size as isize > w_in as isize { (w_in as isize - w_start) as usize } else { kernel_size };
                         
                         // Inner loops now guaranteed valid
                         for kh in kh_start..kh_end {
                             for kw in kw_start..kw_end {
                                 let h_in_idx = (h_start + kh as isize) as usize;
                                 let w_in_idx = (w_start + kw as isize) as usize;
                                 let val = input_data[((b * c + ch) * h_in + h_in_idx) * w_in + w_in_idx];
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
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut input_grad = input.grad.write().unwrap();
                    
                    // Parallelize over Input (N, C)
                    input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(|(idx, in_plane)| {
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
                                
                                let oh_min = (ih + padding).saturating_sub(kernel_size - 1) / stride;
                                let oh_max = ((ih + padding) / stride).min(h_out - 1);
                                let ow_min = (iw + padding).saturating_sub(kernel_size - 1) / stride;
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
                                            
                                            let h_start = (oh * stride) as isize - padding as isize;
                                            let w_start = (ow * stride) as isize - padding as isize;
                                            
                                            if (ih as isize) >= h_start && (ih as isize) < h_start + kernel_size as isize &&
                                               (iw as isize) >= w_start && (iw as isize) < w_start + kernel_size as isize {
                                                
                                                // Re-find max
                                                let mut max_val = f64::NEG_INFINITY;
                                                
                                                // Optimized bounds for inner search
                                                let kh_start = if h_start < 0 { (-h_start) as usize } else { 0 };
                                                let kw_start = if w_start < 0 { (-w_start) as usize } else { 0 };
                                                let kh_end = if h_start + kernel_size as isize > h_in as isize { (h_in as isize - h_start) as usize } else { kernel_size };
                                                let kw_end = if w_start + kernel_size as isize > w_in as isize { (w_in as isize - w_start) as usize } else { kernel_size };
                                                
                                                for kh in kh_start..kh_end {
                                                    for kw in kw_start..kw_end {
                                                         let h_k = (h_start + kh as isize) as usize;
                                                         let w_k = (w_start + kw as isize) as usize;
                                                         let v = input_data[((b * c + ch) * h_in + h_k) * w_in + w_k];
                                                         if v > max_val { max_val = v; }
                                                    }
                                                }
                                                
                                                if (val_in - max_val).abs() < 1e-6 {
                                                    grad_sum += grad_out[((b * c + ch) * h_out + oh) * w_out + ow];
                                                }
                                            }
                                        }
                                    }
                                }
                                in_plane[ih * w_in + iw] += grad_sum;
                            }
                        }
                    });
                }),
            })),
         }
    }

    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        let diff = self - target;
        let sq = &diff * &diff;
        sq.mean()
    }
}

// Operator overloads

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Add shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().zip(rhs_data.par_iter()).map(|(a, b)| a + b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    // Scope 1: LHS
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        lhs_grad.par_iter_mut().zip(grad_out.par_iter()).for_each(|(lg, &g)| *lg += g);
                    }
                    // Scope 2: RHS
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        rhs_grad.par_iter_mut().zip(grad_out.par_iter()).for_each(|(rg, &g)| *rg += g);
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor { self.clone() + rhs.clone() }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Sub shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().zip(rhs_data.par_iter()).map(|(a, b)| a - b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    // Scope 1: LHS
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        lhs_grad.par_iter_mut().zip(grad_out.par_iter()).for_each(|(lg, &g)| *lg += g);
                    }
                    // Scope 2: RHS
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        rhs_grad.par_iter_mut().zip(grad_out.par_iter()).for_each(|(rg, &g)| *rg -= g);
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor { self.clone() - rhs.clone() }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        // Broadcast support? For now assuming same shape or scalar broadcast
        // Simplifying to same shape for now to avoid complexity
        assert_eq!(self.shape, rhs.shape, "Mul shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().zip(rhs_data.par_iter()).map(|(a, b)| a * b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let lhs_data = lhs.data.read().unwrap();
                    let rhs_data = rhs.data.read().unwrap();
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        grad.par_iter_mut().zip(grad_out.par_iter()).zip(lhs_data.par_iter()).zip(rhs_data.par_iter())
                            .for_each(|(((g, &go), &l), &r)| {
                                *g += go * (l + r);
                            });
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            lhs_grad.par_iter_mut().zip(grad_out.par_iter()).zip(rhs_data.par_iter())
                                .for_each(|((lg, &g), &r)| *lg += g * r);
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            rhs_grad.par_iter_mut().zip(grad_out.par_iter()).zip(lhs_data.par_iter())
                                .for_each(|((rg, &g), &l)| *rg += g * l);
                        }
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor { self.clone() * rhs.clone() }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Div shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().zip(rhs_data.par_iter()).map(|(a, b)| a / b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let lhs_data = lhs.data.read().unwrap();
                    let rhs_data = rhs.data.read().unwrap();
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        grad.par_iter_mut().zip(grad_out.par_iter()).zip(lhs_data.par_iter()).zip(rhs_data.par_iter())
                            .for_each(|(((g, &go), &l), &r)| {
                                *g += go / r - go * l / (r * r);
                            });
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            lhs_grad.par_iter_mut().zip(grad_out.par_iter()).zip(rhs_data.par_iter())
                                .for_each(|((lg, &g), &r)| *lg += g / r);
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            rhs_grad.par_iter_mut().zip(grad_out.par_iter()).zip(lhs_data.par_iter()).zip(rhs_data.par_iter())
                                .for_each(|(((rg, &g), &l), &r)| *rg -= g * l / (r * r));
                        }
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn div(self, rhs: &'b Tensor) -> Tensor { self.clone() / rhs.clone() }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.par_iter().map(|&x| -x).collect();
        let parents = vec![self.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    inp_grad.par_iter_mut().zip(grad_out.par_iter()).for_each(|(ig, &g)| *ig -= g);
                }),
            })),
        }
    }
}

impl<'a> Neg for &'a Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor { -self.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_scalar() {
        let t = Tensor::new(vec![5.0], vec![1]);
        let b = t.broadcast(vec![2, 2]);
        assert_eq!(b.shape, vec![2, 2]);
        let data = b.data.read().unwrap();
        assert_eq!(*data, vec![5.0, 5.0, 5.0, 5.0]);
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
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![1, 1, 3, 3]);
        
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
