use crate::autograd::{Context, Device, Tensor, PAR_THRESHOLD};
use crate::dtype::{Dtype, Storage};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

impl Tensor {
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
            device: self.device,
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
}
