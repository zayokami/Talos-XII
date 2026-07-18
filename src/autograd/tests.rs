use super::*;
use crate::dtype::Dtype;

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

fn assert_panics<F>(f: F)
where
    F: FnOnce(),
{
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
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
fn test_broadcast_right_aligned_shape_forward_backward() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let b = t.broadcast(vec![3, 4]);
    assert_eq!(b.shape, vec![3, 4]);
    assert_eq!(
        *b.data_as_f64_vec(),
        vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    );

    b.sum().backward();
    assert_eq!(t.grad_to_f64_vec(), vec![4.0, 4.0, 4.0]);
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
fn test_softmax_dim_zero_normalization() {
    let t = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = t.softmax_dim(0);
    let d = out.data_as_f64_vec();

    for col in 0..3 {
        let sum = d[col] + d[3 + col];
        assert!((sum - 1.0).abs() < 1e-5, "column {} sum={}", col, sum);
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

#[test]
fn test_extra_unary_binary_and_loss_ops() {
    let x = Tensor::new(vec![-1.0, 0.0, 2.0, 4.0], vec![4]);
    assert_eq!(
        x.relu6().data_as_f64_vec().as_slice(),
        &[0.0, 0.0, 2.0, 4.0]
    );
    assert_eq!(
        x.sign().data_as_f64_vec().as_slice(),
        &[-1.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(
        x.square().data_as_f64_vec().as_slice(),
        &[1.0, 0.0, 4.0, 16.0]
    );

    let rhs = Tensor::new(vec![0.5, 0.0, 3.0, 1.0], vec![4]);
    assert_eq!(
        x.maximum(&rhs).data_as_f64_vec().as_slice(),
        &[0.5, 0.0, 3.0, 4.0]
    );
    assert_eq!(
        x.equal(&rhs).data_as_f64_vec().as_slice(),
        &[0.0, 1.0, 0.0, 0.0]
    );

    let loss_input = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]);
    let loss = loss_input.l2_loss();
    assert!((loss.item() - 7.0).abs() < 1e-9);
    loss.backward();
    assert_eq!(loss_input.grad_to_f64_vec(), vec![1.0, -2.0, 3.0]);
}

#[test]
fn test_reduce_normalize_and_shape_ops() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(
        x.reduce_sum_dim(1, false).data_as_f64_vec().as_slice(),
        &[6.0, 15.0]
    );
    assert_eq!(
        x.reduce_mean_dim(0, false).data_as_f64_vec().as_slice(),
        &[2.5, 3.5, 4.5]
    );
    assert_eq!(x.reduce_max_dim(1, true).shape, vec![2, 1]);

    let norm = x.l2_normalize(1, 1e-12);
    let data = norm.data_as_f64_vec();
    for row in 0..2 {
        let base = row * 3;
        let sum_sq = data[base..base + 3].iter().map(|v| v * v).sum::<f64>();
        assert!((sum_sq - 1.0).abs() < 1e-9);
    }

    let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
    let cat = Tensor::concat(&[a.clone(), b.clone()], 0);
    assert_eq!(cat.shape, vec![2, 2]);
    assert_eq!(cat.data_as_f64_vec().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    let split = cat.split(0, vec![1, 1]);
    assert_eq!(split[0].data_as_f64_vec().as_slice(), &[1.0, 2.0]);
    assert_eq!(split[1].data_as_f64_vec().as_slice(), &[3.0, 4.0]);

    let sliced = x.strided_slice(vec![0, 0], vec![2, 3], vec![1, 2]);
    assert_eq!(sliced.shape, vec![2, 2]);
    assert_eq!(sliced.data_as_f64_vec().as_slice(), &[1.0, 3.0, 4.0, 6.0]);
}

#[test]
fn test_pooling_and_extended_conv_shapes() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let avg = x.avg_pool2d(2, 1, 0, false);
    assert_eq!(avg.shape, vec![1, 1, 1, 1]);
    assert!((avg.item() - 2.5).abs() < 1e-9);

    let trans_weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    let transposed = x.conv2d_transpose(&trans_weight, 1, 0);
    assert_eq!(transposed.shape, vec![1, 1, 2, 2]);
    assert_eq!(
        transposed.data_as_f64_vec().as_slice(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let depthwise_weight = Tensor::new(vec![2.0], vec![1, 1, 1, 1]);
    let depthwise = x.depthwise_conv2d(&depthwise_weight, 1, 0);
    assert_eq!(depthwise.shape, vec![1, 1, 2, 2]);
    assert_eq!(
        depthwise.data_as_f64_vec().as_slice(),
        &[2.0, 4.0, 6.0, 8.0]
    );

    let x3 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 2, 2]);
    let w3 = Tensor::new(vec![1.0], vec![1, 1, 1, 1, 1]);
    let conv3 = x3.conv3d(&w3, 1, 0);
    assert_eq!(conv3.shape, vec![1, 1, 1, 2, 2]);
    assert_eq!(conv3.data_as_f64_vec().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_conv_pool_invalid_boundaries_are_rejected() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let w = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    assert_panics(|| {
        let _ = x.conv2d(&w, 0, 0);
    });

    let large_w = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
    assert_panics(|| {
        let _ = x.conv2d(&large_w, 1, 0);
    });

    assert_panics(|| {
        let _ = x.max_pool2d(0, 1, 0);
    });
    assert_panics(|| {
        let _ = x.max_pool2d(2, 0, 0);
    });
    assert_panics(|| {
        let _ = x.max_pool2d(2, 1, 8);
    });
    assert_panics(|| {
        let _ = x.avg_pool2d(2, 1, 8, false);
    });

    assert_panics(|| {
        let _ = x.conv2d_transpose(&w, 1, 8);
    });

    let depthwise_w = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
    assert_panics(|| {
        let _ = x.depthwise_conv2d(&depthwise_w, 1, 0);
    });

    let x3 = Tensor::new(vec![1.0], vec![1, 1, 1, 1, 1]);
    let w3 = Tensor::new(vec![1.0; 8], vec![1, 1, 2, 2, 2]);
    assert_panics(|| {
        let _ = x3.conv3d(&w3, 1, 0);
    });
}

#[test]
fn test_autograd_alias_and_empty_boundaries_do_not_deadlock_or_nan() {
    let x = Tensor::new(vec![0.5, -1.0], vec![1, 2]);
    x.smooth_l1_loss(&x, 1.0).backward();
    x.sigmoid_cross_entropy_with_logits(&x).backward();
    x.softmax_cross_entropy_with_logits(&x).backward();

    let labels = Tensor::new(vec![1.0], vec![1]);
    x.cosine_embedding_loss(&x, &labels, 0.0).backward();

    let prelu_x = Tensor::new(vec![-2.0, 3.0], vec![2]);
    prelu_x.prelu(&prelu_x).sum().backward();

    let bias_x = Tensor::new(vec![1.0, 2.0], vec![2]);
    bias_x.bias_add(&bias_x).sum().backward();

    let norm_x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    norm_x.batch_norm2d(&norm_x, &norm_x, 1e-5).sum().backward();

    let conv2d_t = Tensor::new(vec![1.0], vec![1, 1, 1, 1]);
    conv2d_t.conv2d_transpose(&conv2d_t, 1, 0).sum().backward();
    conv2d_t.depthwise_conv2d(&conv2d_t, 1, 0).sum().backward();

    let conv3d_t = Tensor::new(vec![1.0], vec![1, 1, 1, 1, 1]);
    conv3d_t.conv3d(&conv3d_t, 1, 0).sum().backward();

    assert_panics(|| {
        let _ = Tensor::new(vec![], vec![0]).mse_loss(&Tensor::new(vec![], vec![0]));
    });
    assert_panics(|| {
        let _ = Tensor::new(vec![], vec![1, 0]).l2_normalize(1, 1e-12);
    });
    assert_panics(|| {
        let _ = Tensor::new(vec![], vec![1, 0]).layer_norm_simple(1e-5);
    });
    assert_panics(|| {
        let _ = Tensor::new(vec![1.0], vec![1]).layer_norm_simple(0.0);
    });

    assert_panics(|| {
        let _ = Tensor::new(vec![], vec![0, 2]).cosine_embedding_loss(
            &Tensor::new(vec![], vec![0, 2]),
            &Tensor::new(vec![], vec![0]),
            0.0,
        );
    });
    assert_panics(|| {
        let _ = Tensor::new(vec![], vec![1, 0]).group_norm(1, 1e-5);
    });
    assert_panics(|| {
        let _ = Tensor::new(vec![1.0], vec![]).log_softmax();
    });
    assert_panics(|| {
        let _ = Tensor::new(vec![], vec![2, 0]).softmax_dim(1);
    });

    let empty = Tensor::new(vec![], vec![0]);
    empty.broadcast_to_batch(2).sum().backward();
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
fn test_cuda_broadcast_backward_materializes_cached_gradient() {
    if crate::cuda::init().is_err() {
        return;
    }

    let matrix = Tensor::new_f32(vec![1.0; 6], vec![2, 3])
        .to_cuda()
        .expect("CUDA matrix upload");
    let bias = Tensor::new_f32(vec![0.5, 1.0, 1.5], vec![3])
        .to_cuda()
        .expect("CUDA bias upload");
    let expanded_bias = bias.broadcast(vec![2, 3]);

    (&matrix + &expanded_bias).sum().backward();

    assert_eq!(bias.grad_to_f64_vec(), vec![2.0, 2.0, 2.0]);
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

#[cfg(cuda)]
#[test]
fn test_cuda_fill_invalidates_cpu_tensor_cache() {
    if crate::cuda::init().is_err() {
        return;
    }

    let mut tensor = Tensor::new_f32(vec![1.0, 2.0], vec![2]);
    let first_cuda = match tensor.to_cuda() {
        Ok(tensor) => tensor,
        Err(_) => return,
    };
    assert_eq!(first_cuda.data_as_f64_vec(), vec![1.0, 2.0]);

    tensor.fill_(3.0);
    let second_cuda = match tensor.to_cuda() {
        Ok(tensor) => tensor,
        Err(_) => return,
    };
    assert_eq!(second_cuda.data_as_f64_vec(), vec![3.0, 3.0]);
}

#[cfg(cuda)]
#[test]
fn test_cuda_neg_preserves_device_and_backward() {
    if crate::cuda::init().is_err() {
        return;
    }

    let input = match Tensor::new_f32(vec![1.0, -2.0, 3.0], vec![3]).to_cuda() {
        Ok(tensor) => tensor,
        Err(_) => return,
    };
    let neg = -input.clone();
    assert_eq!(neg.device, Device::Cuda);
    assert_eq!(neg.data_as_f64_vec(), vec![-1.0, 2.0, -3.0]);

    neg.sum().backward();
    assert_eq!(input.grad_to_f64_vec(), vec![-1.0, -1.0, -1.0]);
}

#[cfg(cuda)]
#[test]
fn test_cuda_reshape_backward_matches_view_layout() {
    if crate::cuda::init().is_err() {
        return;
    }

    let input = match Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).to_cuda() {
        Ok(tensor) => tensor,
        Err(_) => return,
    };
    let reshaped = input.reshape(vec![4]);
    assert_eq!(reshaped.device, Device::Cuda);

    reshaped.sum().backward();
    assert_eq!(input.grad_to_f64_vec(), vec![1.0, 1.0, 1.0, 1.0]);
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

    let neg_out = -&a;
    assert_eq!(neg_out.dtype, Dtype::F32);
    assert_eq!(neg_out.data_as_f64_vec(), vec![-1.0, -2.0, -3.0]);

    neg_out.sum().backward();
    assert_eq!(a.grad_to_f64_vec(), vec![-1.0, -1.0, -1.0]);
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
fn test_f32_index_select_forward_backward() {
    let a = Tensor::with_dtype(vec![1.0, 2.0, 3.0], vec![3], Dtype::F32);
    let selected = a.index_select(1);
    assert_eq!(selected.dtype, Dtype::F32);
    assert_eq!(selected.data_as_f64_vec(), vec![2.0]);

    selected.backward();

    assert_eq!(a.grad_to_f64_vec(), vec![0.0, 1.0, 0.0]);
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
fn test_f32_transposed_weight_matmul_backward() {
    let input = Tensor::with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Dtype::F32);
    let weight = Tensor::with_dtype(vec![3.0, 4.0], vec![1, 2], Dtype::F32);
    let transposed = weight.transpose(0, 1);

    assert_eq!(transposed.data.dtype(), Dtype::F32);
    assert_eq!(transposed.grad.dtype(), Dtype::F32);

    input.matmul(&transposed).sum().backward();

    assert_eq!(input.grad_to_f64_vec(), vec![3.0, 4.0, 3.0, 4.0]);
    assert_eq!(weight.grad_to_f64_vec(), vec![4.0, 6.0]);
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
fn test_f32_more_unary_math_ops() {
    let a = Tensor::with_dtype(vec![-1.0, 0.0, 2.0], vec![3], Dtype::F32);
    let tanh = a.tanh();
    assert_eq!(tanh.dtype, Dtype::F32);
    let tanh_data = tanh.data_as_f64_vec();
    for (actual, expected) in
        tanh_data
            .iter()
            .zip([(-1.0_f64).tanh(), 0.0_f64.tanh(), 2.0_f64.tanh()])
    {
        assert!((actual - expected).abs() < 1e-5);
    }
    tanh.sum().backward();
    let a_grad = a.grad_to_f64_vec();
    for (actual, y) in a_grad.iter().zip(tanh_data.iter()) {
        assert!((actual - (1.0 - y * y)).abs() < 1e-5);
    }

    let b = Tensor::with_dtype(vec![-1.0, 0.0, 2.0], vec![3], Dtype::F32);
    let sigmoid = b.sigmoid();
    assert_eq!(sigmoid.dtype, Dtype::F32);
    let sigmoid_data = sigmoid.data_as_f64_vec();
    for (actual, x) in sigmoid_data.iter().zip([-1.0_f64, 0.0, 2.0]) {
        let expected = 1.0 / (1.0 + (-x).exp());
        assert!((actual - expected).abs() < 1e-5);
    }
    sigmoid.sum().backward();
    let b_grad = b.grad_to_f64_vec();
    for (actual, y) in b_grad.iter().zip(sigmoid_data.iter()) {
        assert!((actual - y * (1.0 - y)).abs() < 1e-5);
    }

    let c = Tensor::with_dtype(vec![-2.0, 0.0, 0.5, 3.0], vec![4], Dtype::F32);
    let clamped = c.clamp(-0.5, 1.0);
    assert_eq!(clamped.dtype, Dtype::F32);
    assert_eq!(clamped.data_as_f64_vec(), vec![-0.5, 0.0, 0.5, 1.0]);
    clamped.sum().backward();
    assert_eq!(c.grad_to_f64_vec(), vec![0.0, 1.0, 1.0, 0.0]);

    let d = Tensor::with_dtype(vec![1.0, 4.0, 9.0], vec![3], Dtype::F32);
    assert_eq!(d.sqrt().data_as_f64_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(d.pow(2.0).data_as_f64_vec(), vec![1.0, 16.0, 81.0]);
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

#[test]
fn test_backward_with_explicit_gradient_seed() {
    let input = Tensor::new_f32(vec![1.0, 2.0], vec![2]);
    let output = &input * &input;
    let gradient = Tensor::new_f32(vec![0.5, 2.0], vec![2]);

    output.backward_with_gradient(Some(&gradient)).unwrap();

    let actual = input.grad_to_f32_vec();
    assert!((actual[0] - 1.0).abs() < 1e-6);
    assert!((actual[1] - 8.0).abs() < 1e-6);
}

#[test]
fn test_repeated_backward_resets_intermediate_gradients() {
    let input = Tensor::new_f32(vec![2.0], vec![1]);
    let squared = &input * &input;
    let loss = squared.sum();

    loss.backward();
    loss.backward();

    let actual = input.grad_to_f32_vec();
    assert!((actual[0] - 8.0).abs() < 1e-6);
}

#[test]
fn test_f64_item_preserves_precision() {
    let value = 1.123_456_789_012_345_f64;
    let tensor = Tensor::new(vec![value], Vec::new());

    assert_eq!(tensor.item(), value);
}
