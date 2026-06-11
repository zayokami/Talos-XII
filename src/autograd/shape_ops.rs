use crate::autograd::{Context, Device, Tensor, PAR_THRESHOLD};
use crate::dtype::{Dtype, Storage};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

fn checked_product(shape: &[usize], op: &'static str) -> usize {
    shape.iter().copied().fold(1usize, |acc, dim| {
        acc.checked_mul(dim)
            .unwrap_or_else(|| panic!("{op} element count overflow"))
    })
}

fn checked_mul(lhs: usize, rhs: usize, op: &'static str) -> usize {
    lhs.checked_mul(rhs)
        .unwrap_or_else(|| panic!("{op} index arithmetic overflow"))
}

fn checked_add(lhs: usize, rhs: usize, op: &'static str) -> usize {
    lhs.checked_add(rhs)
        .unwrap_or_else(|| panic!("{op} index arithmetic overflow"))
}

fn checked_strides(shape: &[usize], op: &'static str) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = checked_mul(strides[dim + 1], shape[dim + 1], op);
    }
    strides
}

fn pad_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut padded = vec![1usize; rank.saturating_sub(shape.len())];
    padded.extend_from_slice(shape);
    padded
}

#[derive(Clone)]
struct BroadcastIndexPlan {
    old_padded: Vec<usize>,
    new_padded: Vec<usize>,
    old_strides: Vec<usize>,
    new_strides: Vec<usize>,
}

impl BroadcastIndexPlan {
    fn new(old_shape: &[usize], new_shape: &[usize]) -> Self {
        let rank = old_shape.len().max(new_shape.len());
        let old_padded = pad_shape(old_shape, rank);
        let new_padded = pad_shape(new_shape, rank);
        let old_strides = checked_strides(&old_padded, "broadcast");
        let new_strides = checked_strides(&new_padded, "broadcast");
        Self {
            old_padded,
            new_padded,
            old_strides,
            new_strides,
        }
    }

    fn source_index(&self, linear_idx: usize) -> usize {
        let mut source_idx = 0usize;
        for dim in 0..self.old_padded.len() {
            if self.old_padded[dim] == 1 {
                continue;
            }
            let coord = (linear_idx / self.new_strides[dim]) % self.new_padded[dim];
            source_idx = checked_add(
                source_idx,
                checked_mul(coord, self.old_strides[dim], "broadcast"),
                "broadcast",
            );
        }
        source_idx
    }
}

impl Tensor {
    /// Concatenate tensors along an existing dimension.
    pub fn concat(tensors: &[Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "concat requires at least one tensor");
        let rank = tensors[0].shape.len();
        assert!(dim < rank, "concat dim out of bounds");
        let mut shape = tensors[0].shape.clone();
        let mut dtype = tensors[0].dtype;
        shape[dim] = 0;
        for tensor in tensors {
            assert_eq!(tensor.shape.len(), rank, "concat rank mismatch");
            for axis in 0..rank {
                if axis != dim {
                    assert_eq!(
                        tensor.shape[axis], tensors[0].shape[axis],
                        "concat shape mismatch at dim {}",
                        axis
                    );
                }
            }
            shape[dim] = shape[dim]
                .checked_add(tensor.shape[dim])
                .unwrap_or_else(|| panic!("concat output dimension overflow"));
            dtype = Tensor::binary_dtype(dtype, tensor.dtype);
        }

        let inner = checked_product(&shape[dim + 1..], "concat");
        let outer = checked_product(&shape[..dim], "concat");
        let out_len = checked_product(&shape, "concat");
        let chunk_lens: Vec<usize> = tensors
            .iter()
            .map(|tensor| checked_mul(tensor.shape[dim], inner, "concat"))
            .collect();
        let mut prefix_lens = Vec::with_capacity(chunk_lens.len());
        let mut total_chunk_len = 0usize;
        for &chunk_len in &chunk_lens {
            prefix_lens.push(total_chunk_len);
            total_chunk_len = checked_add(total_chunk_len, chunk_len, "concat");
        }
        let mut output = Vec::with_capacity(out_len);
        let input_data: Vec<Vec<f64>> = tensors.iter().map(Tensor::data_as_f64_vec).collect();
        for outer_idx in 0..outer {
            for (data, &chunk_len) in input_data.iter().zip(chunk_lens.iter()) {
                let base = checked_mul(outer_idx, chunk_len, "concat");
                output.extend_from_slice(&data[base..base + chunk_len]);
            }
        }

        let parents = tensors.to_vec();
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape,
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    for (tensor_idx, parent) in parents.iter().enumerate() {
                        let mut input_grad = parent.grad_write_compat();
                        let chunk_len = chunk_lens[tensor_idx];
                        for outer_idx in 0..outer {
                            let output_base = checked_add(
                                checked_mul(outer_idx, total_chunk_len, "concat"),
                                prefix_lens[tensor_idx],
                                "concat",
                            );
                            let input_base = checked_mul(outer_idx, chunk_len, "concat");
                            for offset in 0..chunk_len {
                                input_grad[input_base + offset] += grad_out[output_base + offset];
                            }
                        }
                    }
                }),
            })),
        }
    }

    /// Split a tensor along a dimension. Split sizes must cover the whole dimension.
    pub fn split(&self, dim: usize, sizes: Vec<usize>) -> Vec<Tensor> {
        assert!(dim < self.shape.len(), "split dim out of bounds");
        assert!(!sizes.is_empty(), "split requires at least one output");
        let total_size = sizes
            .iter()
            .copied()
            .fold(0usize, |acc, size| checked_add(acc, size, "split"));
        assert_eq!(
            total_size, self.shape[dim],
            "split sizes must sum to the selected dimension"
        );
        let input = self.data_as_f64_vec();
        let inner = checked_product(&self.shape[dim + 1..], "split");
        let outer = checked_product(&self.shape[..dim], "split");
        let input_dim_stride = checked_mul(self.shape[dim], inner, "split");
        let mut offset_dim = 0usize;

        sizes
            .into_iter()
            .map(|size| {
                let output_dim_stride = checked_mul(size, inner, "split");
                let output_len = checked_mul(outer, output_dim_stride, "split");
                let mut output = Vec::with_capacity(output_len);
                for outer_idx in 0..outer {
                    let input_base = checked_add(
                        checked_mul(outer_idx, input_dim_stride, "split"),
                        checked_mul(offset_dim, inner, "split"),
                        "split",
                    );
                    output.extend_from_slice(&input[input_base..input_base + output_dim_stride]);
                }

                let mut shape = self.shape.clone();
                shape[dim] = size;
                let dtype = self.dtype;
                let captured_offset = offset_dim;
                offset_dim = checked_add(offset_dim, size, "split");
                Tensor {
                    data: Storage::from_f64_vec(output, dtype),
                    grad: Storage::zeros(output_len, Tensor::grad_dtype_for(dtype)),
                    shape,
                    device: Device::Cpu,
                    dtype,
                    _ctx: Some(Arc::new(Context {
                        parents: vec![self.clone()],
                        backward_op: Box::new(move |grad_out, parents| {
                            let grad_out = grad_out.to_f64_vec();
                            let mut input_grad = parents[0].grad_write_compat();
                            for outer_idx in 0..outer {
                                let input_base = checked_add(
                                    checked_mul(outer_idx, input_dim_stride, "split"),
                                    checked_mul(captured_offset, inner, "split"),
                                    "split",
                                );
                                let output_base =
                                    checked_mul(outer_idx, output_dim_stride, "split");
                                for offset in 0..output_dim_stride {
                                    input_grad[input_base + offset] +=
                                        grad_out[output_base + offset];
                                }
                            }
                        }),
                    })),
                }
            })
            .collect()
    }

    /// Positive-stride slicing across all dimensions.
    pub fn strided_slice(&self, begin: Vec<usize>, end: Vec<usize>, strides: Vec<usize>) -> Tensor {
        let rank = self.shape.len();
        assert_eq!(begin.len(), rank, "strided_slice begin rank mismatch");
        assert_eq!(end.len(), rank, "strided_slice end rank mismatch");
        assert_eq!(strides.len(), rank, "strided_slice strides rank mismatch");
        let mut output_shape = Vec::with_capacity(rank);
        for dim in 0..rank {
            assert!(strides[dim] > 0, "strided_slice strides must be positive");
            assert!(
                begin[dim] <= end[dim] && end[dim] <= self.shape[dim],
                "strided_slice bounds invalid at dim {}",
                dim
            );
            output_shape.push((end[dim] - begin[dim]).div_ceil(strides[dim]));
        }

        let input_strides = checked_strides(&self.shape, "strided_slice");
        let output_strides = checked_strides(&output_shape, "strided_slice");

        let input = self.data_as_f64_vec();
        let out_len = checked_product(&output_shape, "strided_slice");
        let mut output = vec![0.0; out_len];
        let mut input_indices = vec![0usize; out_len];
        for output_idx in 0..out_len {
            let mut remainder = output_idx;
            let mut input_idx = 0usize;
            for dim in 0..rank {
                let coord = remainder / output_strides[dim];
                remainder %= output_strides[dim];
                let stepped = checked_add(
                    begin[dim],
                    checked_mul(coord, strides[dim], "strided_slice"),
                    "strided_slice",
                );
                input_idx = checked_add(
                    input_idx,
                    checked_mul(stepped, input_strides[dim], "strided_slice"),
                    "strided_slice",
                );
            }
            output[output_idx] = input[input_idx];
            input_indices[output_idx] = input_idx;
        }
        let input_indices = Arc::new(input_indices);
        let dtype = self.dtype;
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape: output_shape,
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_grad = parents[0].grad_write_compat();
                    for (output_idx, &input_idx) in input_indices.iter().enumerate() {
                        input_grad[input_idx] += grad_out[output_idx];
                    }
                }),
            })),
        }
    }

    pub fn flatten(&self) -> Tensor {
        self.reshape(vec![self.numel()])
    }

    pub fn transpose2d(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose requires 2D tensor");
        #[cfg(cuda)]
        if let Some(out) = self.transpose_cuda(0, 1) {
            return out;
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let len = checked_product(&[rows, cols], "transpose2d");
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
            grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F64)),
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
        let len = checked_product(&[rows, cols], "transpose2d");
        let self_f32 = self.data_to_f32_vec();
        let mut out_data = vec![0.0f32; self_f32.len()];
        for r in 0..rows {
            for c in 0..cols {
                out_data[c * rows + r] = self_f32[r * cols + c];
            }
        }
        Tensor {
            data: Storage::from_f32_vec(out_data, self.dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(self.dtype)),
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
        let rank = old.len().max(new.len());
        let old_padded = pad_shape(old, rank);
        let new_padded = pad_shape(new, rank);
        for (&old_dim, &new_dim) in old_padded.iter().zip(new_padded.iter()) {
            if old_dim != new_dim && old_dim != 1 && new_dim != 1 {
                return false;
            }
        }
        true
    }

    pub fn broadcast(&self, new_shape: Vec<usize>) -> Tensor {
        let old_shape = self.shape.clone();

        assert!(
            Self::broadcastable_shapes(&old_shape, &new_shape),
            "Shapes {:?} and {:?} are not broadcast-compatible",
            old_shape,
            new_shape
        );

        let total_elements = checked_product(&new_shape, "broadcast");

        let self_data = self.data_as_f64_vec();
        let old_data = &self_data;
        let plan = BroadcastIndexPlan::new(&old_shape, &new_shape);
        let backward_plan = plan.clone();

        let mut new_data = Vec::with_capacity(total_elements);

        for linear_idx in 0..total_elements {
            let old_linear_idx = plan.source_index(linear_idx);
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

                    #[allow(clippy::needless_range_loop)]
                    for linear_idx in 0..total_elements {
                        let old_linear_idx = backward_plan.source_index(linear_idx);
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
        let out_len = checked_mul(len, batch_size, "broadcast_to_batch");
        let mut new_data = Vec::with_capacity(out_len);
        for _ in 0..batch_size {
            new_data.extend_from_slice(&self_data);
        }

        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(&self.shape);

        let parents = vec![self.clone()];

        Tensor {
            data: Storage::from_f64_vec(new_data, self.dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(self.dtype)),
            shape: new_shape,
            device: self.device,
            dtype: self.dtype,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, _parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = _parents[0].grad_write_compat();
                    let chunk_size = inp_grad.len();
                    if chunk_size == 0 {
                        return;
                    }
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

        let len = checked_product(shape, "transpose");
        assert_eq!(len, self_data.len(), "transpose shape/data length mismatch");
        let mut new_data = vec![0.0; len];

        let mut strides = [0usize; 8];
        let stride_vec = checked_strides(shape, "transpose");
        strides[..rank].copy_from_slice(&stride_vec);

        let mut new_strides = [0usize; 8];
        let new_stride_vec = checked_strides(&new_shape, "transpose");
        new_strides[..rank].copy_from_slice(&new_stride_vec);

        for (i, value) in new_data.iter_mut().enumerate().take(len) {
            let mut temp = i;
            let mut coords = [0usize; 8];
            for d in 0..rank {
                coords[d] = temp / new_strides[d];
                temp %= new_strides[d];
            }

            coords.swap(dim0, dim1);

            let mut old_idx = 0usize;
            for d in 0..rank {
                old_idx = checked_add(
                    old_idx,
                    checked_mul(coords[d], strides[d], "transpose"),
                    "transpose",
                );
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

                        let mut old_idx = 0usize;
                        for d in 0..cap_rank {
                            old_idx = checked_add(
                                old_idx,
                                checked_mul(coords[d], cap_strides[d], "transpose"),
                                "transpose",
                            );
                        }

                        inp_grad[old_idx] += grad_val;
                    }
                }),
            })),
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let len = checked_product(&new_shape, "reshape");
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
                    #[cfg(cuda)]
                    {
                        let input = &parents[0];
                        if input.device == Device::Cuda {
                            if let Some(d_grad_tmp) =
                                crate::autograd::cuda_grad_out_buffer(grad_out)
                            {
                                if let Some(d_input_grad) = input.cuda_grad_ensure_buffer() {
                                    let ok =
                                        match (&*d_grad_tmp, &*d_input_grad, input.grad.dtype()) {
                                            (
                                                crate::cuda::memory::CudaBuffer::F32(src),
                                                crate::cuda::memory::CudaBuffer::F32(dst),
                                                Dtype::F32,
                                            ) => {
                                                crate::cuda::kernels::acc_buffer_f32(dst, src, len)
                                                    .is_ok()
                                            }
                                            (
                                                crate::cuda::memory::CudaBuffer::F64(src),
                                                crate::cuda::memory::CudaBuffer::F64(dst),
                                                Dtype::F64,
                                            ) => crate::cuda::kernels::acc_buffer(dst, src, len)
                                                .is_ok(),
                                            _ => false,
                                        };
                                    if ok {
                                        return;
                                    }
                                }
                            }
                        }
                    }

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
