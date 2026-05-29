#[cfg(cuda)]
use crate::autograd::cuda_grad_out_buffer;
use crate::autograd::{Context, Device, Tensor, TensorReadGuard};
use crate::dtype::{Dtype, Storage};
use crate::simd::{add_scaled_row, dot_product, prefetch_read_l1};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() <= 2 && other.shape.len() == 2);

        let (m, k) = if self.shape.len() == 1 {
            (1, self.shape[0])
        } else {
            (self.shape[0], self.shape[1])
        };
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "MatMul dimension mismatch");

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
        {
            let guards = TensorReadGuard::new(&[self, other]);
            let lhs_data = guards.get(0);
            let rhs_data = guards.get(1);
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

                    {
                        let mut rhs_grad = rhs.grad_write_compat();
                        let ops = k * n * m;
                        if ops < 32768 {
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

        if out_dtype == Dtype::F64 {
            let lhs_cache: Arc<Vec<f64>> = Arc::new(lhs_f32.iter().map(|&v| v as f64).collect());
            let rhs_cache: Arc<Vec<f64>> = Arc::new(rhs_f32.iter().map(|&v| v as f64).collect());

            return Tensor {
                data: Storage::from_f32_vec(out_data, out_dtype),
                grad: Storage::zeros(m * n, Tensor::grad_dtype_for(out_dtype)),
                shape: out_shape,
                device: Device::Cpu,
                dtype: out_dtype,
                _ctx: Some(Arc::new(Context {
                    parents: vec![self.clone(), other.clone()],
                    backward_op: Box::new(move |grad_out, _parents| {
                        let grad_out_f64 = grad_out.to_f64_vec();
                        {
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
                        }

                        {
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
                        }
                    }),
                })),
            };
        }

        let lhs_cache: Arc<Vec<f32>> = Arc::new(lhs_f32);
        let rhs_cache: Arc<Vec<f32>> = Arc::new(rhs_f32);

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
                    {
                        let mut lhs_grad = _parents[0].grad_write_f32();
                        for r in 0..m {
                            for i in 0..k {
                                let mut sum = 0.0f64;
                                for j in 0..n {
                                    sum += grad_out_f32[r * n + j] as f64
                                        * rhs_cache[i * n + j] as f64;
                                }
                                lhs_grad[r * k + i] += sum as f32;
                            }
                        }
                    }

                    {
                        let mut rhs_grad = _parents[1].grad_write_f32();
                        for i in 0..k {
                            for j in 0..n {
                                let mut sum = 0.0f64;
                                for r in 0..m {
                                    sum += lhs_cache[r * k + i] as f64
                                        * grad_out_f32[r * n + j] as f64;
                                }
                                rhs_grad[i * n + j] += sum as f32;
                            }
                        }
                    }
                }),
            })),
        }
    }

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
        let (m_i32, n_i32, k_i32) = match (i32::try_from(m), i32::try_from(n), i32::try_from(k)) {
            (Ok(mv), Ok(nv), Ok(kv)) => (mv, nv, kv),
            _ => {
                crate::cuda::record_matmul_fallback("gemm");
                log::warn!(
                    "[Autograd] CUDA GEMM dimensions overflow i32 (m={}, n={}, k={}), using CPU",
                    m,
                    n,
                    k
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
        }
    }
}
