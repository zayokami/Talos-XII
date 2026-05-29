#[cfg(test)]
use crate::dtype::Dtype;

// --- Autograd Engine ---

mod activation;
mod binary_ops;
mod conv_pooling;
mod core;
#[cfg(cuda)]
mod cuda_bridge;
#[cfg(cuda)]
mod cuda_ops;
mod guards;
mod lifecycle;
mod loss;
mod matmul;
mod operators;
mod reductions;
mod serde_impl;
mod shape_ops;
mod softmax;
mod storage;
mod unary_ops;

#[cfg(cuda)]
pub(crate) use core::BackwardOp;
pub use core::{Context, Device, GradWriteCompat, Tensor};
#[cfg(cuda)]
pub(crate) use cuda_bridge::cuda_grad_out_buffer;
#[cfg(cuda)]
use cuda_bridge::cuda_sync_grad_to_host;
#[cfg(cuda)]
pub(crate) use cuda_ops::cuda_clip_gradients_in_place;
pub use guards::TensorReadGuard;

// Minimum element count to justify Rayon parallel dispatch.
// Below this, serial iteration is faster due to scheduling overhead.
pub(crate) const PAR_THRESHOLD: usize = 4096;

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
