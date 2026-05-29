use crate::achf::AchfLayer;
use crate::autograd::Tensor;
#[cfg(test)]
use crate::autograd::TensorReadGuard;
use crate::config::{AchfConfig, Config};
use crate::env_net::EnvNet;
use crate::neural::{NeuralLuckOptimizer, DIM};
use crate::nn::{Linear, Module};
use crate::rng::Rng;
use crate::sim::{build_features, env_net_env, prob_6, PullState};
use crate::training_metrics::{StepSnapshot, TrainingMetrics, TrainingMetricsSink};
use std::cell::RefCell;
use std::collections::VecDeque;

// DQN Hyperparameters
const GAMMA: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const BUFFER_CAPACITY: usize = 10000;
const EPSILON_START: f64 = 1.0;
const EPSILON_END: f64 = 0.1;
const EPSILON_DECAY: usize = 50000;
const LEARNING_RATE: f64 = 0.001;
const TRAIN_FREQ: usize = 10;
const LOG_FREQ: usize = 100;
const DEFAULT_DQN_HIDDEN: usize = 1024;
use crate::utils::{create_bar, ACTIONS, ACTION_SPACE, EPISODE_MAX_PULLS};

// PER Hyperparameters (Schaul et al. 2016)
const PER_ALPHA: f64 = 0.6;
const PER_BETA_START: f64 = 0.4;
const PER_BETA_END: f64 = 1.0;
const PER_EPSILON: f64 = 1e-6;

// Constant tensor data for reuse - avoids per-step allocation
const ONES_5_1_DATA: [f64; 5] = [1.0; 5];

#[cfg(test)]
fn double_dqn_target_cpu(
    q_next_eval: &Tensor,
    q_next_target: &Tensor,
    rewards_tensor: &Tensor,
    dones_tensor: &Tensor,
    gamma: f64,
    target_vals: &mut Vec<f64>,
) -> Tensor {
    target_vals.clear();

    let guards = TensorReadGuard::new(&[q_next_eval, q_next_target]);
    let q_next_eval_data = guards.get(0);
    let q_next_target_data = guards.get(1);
    let rewards = rewards_tensor.data_as_f64_vec();
    let dones = dones_tensor.data_as_f64_vec();
    let batch = q_next_eval.shape.first().copied().unwrap_or(BATCH_SIZE);
    let actions = q_next_eval.shape.get(1).copied().unwrap_or(ACTION_SPACE);

    for i in 0..batch {
        let start = i * actions;
        let end = start + actions;
        let row_eval = &q_next_eval_data[start..end];
        let mut max_idx = 0;
        let mut max_val = f64::NEG_INFINITY;
        for (k, &v) in row_eval.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = k;
            }
        }
        let next_q_val = q_next_target_data[start + max_idx];
        target_vals.push(rewards[i] + gamma * next_q_val * (1.0 - dones[i]));
    }

    Tensor::new_f32(std::mem::take(target_vals), vec![batch, 1])
}

fn double_dqn_target_from_q_values_cpu(
    q_next_eval_data: &[f32],
    q_next_target_data: &[f32],
    rewards: &[f32],
    dones: &[f32],
    gamma: f64,
    target_vals: &mut Vec<f64>,
) -> Tensor {
    assert_eq!(rewards.len(), dones.len());
    let batch = rewards.len();
    assert_eq!(q_next_eval_data.len(), batch * ACTION_SPACE);
    assert_eq!(q_next_target_data.len(), batch * ACTION_SPACE);

    target_vals.clear();
    target_vals.reserve(batch);

    for i in 0..batch {
        let start = i * ACTION_SPACE;
        let end = start + ACTION_SPACE;
        let row_eval = &q_next_eval_data[start..end];
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (k, &v) in row_eval.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = k;
            }
        }
        let next_q_val = q_next_target_data[start + max_idx] as f64;
        target_vals.push(rewards[i] as f64 + gamma * next_q_val * (1.0 - dones[i] as f64));
    }

    Tensor::new_f32(std::mem::take(target_vals), vec![batch, 1])
}

fn double_dqn_target_inference_cpu(
    policy_net: &DuelingQNetwork,
    target_net: &DuelingQNetwork,
    batch_next_state: &Tensor,
    rewards_tensor: &Tensor,
    dones_tensor: &Tensor,
    gamma: f64,
    target_vals: &mut Vec<f64>,
) -> Tensor {
    let next_states = batch_next_state.data_to_f32_vec();
    assert!(
        next_states.len().is_multiple_of(DIM),
        "DQN next-state batch length {} is not divisible by feature dim {}",
        next_states.len(),
        DIM
    );
    let q_next_eval = policy_net.forward_inference_batch_values(&next_states);
    let q_next_target = target_net.forward_inference_batch_values(&next_states);
    let rewards = rewards_tensor.data_to_f32_vec();
    let dones = dones_tensor.data_to_f32_vec();

    double_dqn_target_from_q_values_cpu(
        &q_next_eval,
        &q_next_target,
        &rewards,
        &dones,
        gamma,
        target_vals,
    )
}

#[cfg(cuda)]
fn double_dqn_target_cuda(
    q_next_eval: &Tensor,
    q_next_target: &Tensor,
    rewards: &Tensor,
    dones: &Tensor,
    gamma: f64,
) -> Option<Tensor> {
    use crate::autograd::Device;
    use crate::cuda::memory::{alloc, CudaBuffer};
    use crate::dtype::{Dtype, Storage};

    if q_next_eval.device != Device::Cuda
        || q_next_target.device != Device::Cuda
        || rewards.device != Device::Cuda
        || dones.device != Device::Cuda
        || q_next_eval.dtype != q_next_target.dtype
        || q_next_eval.dtype != rewards.dtype
        || q_next_eval.dtype != dones.dtype
        || !matches!(q_next_eval.dtype, Dtype::F32 | Dtype::F64)
        || q_next_eval.shape.len() != 2
        || q_next_eval.shape != q_next_target.shape
    {
        return None;
    }
    let batch = q_next_eval.shape[0];
    let actions = q_next_eval.shape[1];
    if actions == 0 || rewards.numel() != batch || dones.numel() != batch {
        return None;
    }

    let d_eval = q_next_eval.cuda_get_or_upload_buffer().ok()?;
    let d_target = q_next_target.cuda_get_or_upload_buffer().ok()?;
    let d_rewards = rewards.cuda_get_or_upload_buffer().ok()?;
    let d_dones = dones.cuda_get_or_upload_buffer().ok()?;
    let d_out = match q_next_eval.dtype {
        Dtype::F32 => CudaBuffer::F32(alloc::<f32>(batch).ok()?),
        Dtype::F64 => CudaBuffer::F64(alloc::<f64>(batch).ok()?),
        _ => return None,
    };
    let d_out = std::sync::Arc::new(d_out);
    let ok = match (
        &*d_eval,
        &*d_target,
        &*d_rewards,
        &*d_dones,
        &*d_out,
        q_next_eval.dtype,
    ) {
        (
            CudaBuffer::F32(eval),
            CudaBuffer::F32(target),
            CudaBuffer::F32(rewards),
            CudaBuffer::F32(dones),
            CudaBuffer::F32(out),
            Dtype::F32,
        ) => crate::cuda::kernels::double_dqn_target_f32(
            eval,
            target,
            rewards,
            dones,
            out,
            batch,
            actions,
            gamma as f32,
        )
        .is_ok(),
        (
            CudaBuffer::F64(eval),
            CudaBuffer::F64(target),
            CudaBuffer::F64(rewards),
            CudaBuffer::F64(dones),
            CudaBuffer::F64(out),
            Dtype::F64,
        ) => crate::cuda::kernels::double_dqn_target(
            eval, target, rewards, dones, out, batch, actions, gamma,
        )
        .is_ok(),
        _ => false,
    };
    if !ok {
        return None;
    }

    let out = Tensor {
        data: Tensor::empty_storage(q_next_eval.dtype),
        grad: Storage::zeros(batch, Tensor::grad_dtype_for(q_next_eval.dtype)),
        shape: vec![batch, 1],
        device: Device::Cuda,
        dtype: q_next_eval.dtype,
        _ctx: None,
    };
    out.cuda_set_cached_buffer(d_out);
    Some(out)
}

#[cfg(cuda)]
fn abs_diff_cuda(a: &Tensor, b: &Tensor) -> Option<Tensor> {
    use crate::autograd::Device;
    use crate::cuda::memory::{alloc, CudaBuffer};
    use crate::dtype::{Dtype, Storage};

    if a.device != Device::Cuda
        || b.device != Device::Cuda
        || a.dtype != b.dtype
        || a.numel() != b.numel()
        || !matches!(a.dtype, Dtype::F32 | Dtype::F64)
    {
        return None;
    }
    let len = a.numel();
    let d_a = a.cuda_get_or_upload_buffer().ok()?;
    let d_b = b.cuda_get_or_upload_buffer().ok()?;
    let d_out = match a.dtype {
        Dtype::F32 => CudaBuffer::F32(alloc::<f32>(len).ok()?),
        Dtype::F64 => CudaBuffer::F64(alloc::<f64>(len).ok()?),
        _ => return None,
    };
    let d_out = std::sync::Arc::new(d_out);
    let ok = match (&*d_a, &*d_b, &*d_out, a.dtype) {
        (CudaBuffer::F32(lhs), CudaBuffer::F32(rhs), CudaBuffer::F32(out), Dtype::F32) => {
            crate::cuda::kernels::abs_diff_f32(lhs, rhs, out, len).is_ok()
        }
        (CudaBuffer::F64(lhs), CudaBuffer::F64(rhs), CudaBuffer::F64(out), Dtype::F64) => {
            crate::cuda::kernels::abs_diff(lhs, rhs, out, len).is_ok()
        }
        _ => false,
    };
    if !ok {
        return None;
    }

    let out = Tensor {
        data: Tensor::empty_storage(a.dtype),
        grad: Storage::zeros(len, Tensor::grad_dtype_for(a.dtype)),
        shape: a.shape.clone(),
        device: Device::Cuda,
        dtype: a.dtype,
        _ctx: None,
    };
    out.cuda_set_cached_buffer(d_out);
    Some(out)
}

#[cfg(cuda)]
fn argmax_cuda_index(values: &Tensor) -> Option<usize> {
    use crate::autograd::Device;
    use crate::cuda::memory::{alloc, copy_d2h, CudaBuffer};
    use crate::dtype::Dtype;

    if values.device != Device::Cuda || values.numel() == 0 {
        return None;
    }
    let d_values = values.cuda_get_or_upload_buffer().ok()?;
    let d_idx = alloc::<i32>(1).ok()?;
    let ok = match (&*d_values, values.dtype) {
        (CudaBuffer::F32(v), Dtype::F32) => {
            crate::cuda::kernels::argmax_f32(v, &d_idx, values.numel()).is_ok()
        }
        (CudaBuffer::F64(v), Dtype::F64) => {
            crate::cuda::kernels::argmax(v, &d_idx, values.numel()).is_ok()
        }
        _ => false,
    };
    if !ok {
        return None;
    }
    let mut host = [0_i32; 1];
    copy_d2h(&mut host, &d_idx).ok()?;
    usize::try_from(host[0]).ok()
}

// --- Layers ---
// Linear layer is now imported from crate::nn

// --- Dueling Q-Network ---
// Feature Extractor (from NeuralLuckOptimizer) -> Hidden -> Value + Advantage

/// Dueling Q-Network for discrete luck action selection.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DuelingQNetwork {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    val_head: Linear,
    adv_head: Linear,
    achf: Option<AchfLayer>,
}

impl Module for DuelingQNetwork {
    fn forward(&self, state: &Tensor) -> Tensor {
        self.forward_impl(state)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.l1.parameters());
        p.extend(self.l2.parameters());
        p.extend(self.l3.parameters());
        p.extend(self.val_head.parameters());
        p.extend(self.adv_head.parameters());
        if let Some(achf) = &self.achf {
            p.extend(achf.parameters());
        }
        p
    }
}

impl DuelingQNetwork {
    pub fn new_with_config(config: &Config, seed: u64) -> Self {
        let hidden = config.model_hidden_dim;
        let l1 = Linear::new(DIM, hidden, true, seed);
        let l2 = Linear::new(hidden, hidden, true, seed.wrapping_add(1));
        let l3 = Linear::new(hidden, hidden, true, seed.wrapping_add(2));
        let val_head = Linear::new(hidden, 1, true, seed.wrapping_add(3));
        let adv_head = Linear::new(hidden, ACTION_SPACE, true, seed.wrapping_add(4));
        let achf_layer = if config.achf.enabled && config.achf.apply_dqn {
            Some(AchfLayer::new(
                hidden,
                hidden,
                true,
                config.achf.clone(),
                seed.wrapping_add(500),
            ))
        } else {
            None
        };

        DuelingQNetwork {
            l1,
            l2,
            l3,
            val_head,
            adv_head,
            achf: achf_layer,
        }
    }

    /// Convenience constructor using compact hard-coded defaults.
    pub fn new(seed: u64, achf: &AchfConfig) -> Self {
        let l1 = Linear::new(DIM, DEFAULT_DQN_HIDDEN, true, seed);
        let l2 = Linear::new(
            DEFAULT_DQN_HIDDEN,
            DEFAULT_DQN_HIDDEN,
            true,
            seed.wrapping_add(1),
        );
        let l3 = Linear::new(
            DEFAULT_DQN_HIDDEN,
            DEFAULT_DQN_HIDDEN,
            true,
            seed.wrapping_add(2),
        );
        let val_head = Linear::new(DEFAULT_DQN_HIDDEN, 1, true, seed.wrapping_add(3));
        let adv_head = Linear::new(DEFAULT_DQN_HIDDEN, ACTION_SPACE, true, seed.wrapping_add(4));
        let achf_layer = if achf.enabled && achf.apply_dqn {
            Some(AchfLayer::new(
                DEFAULT_DQN_HIDDEN,
                DEFAULT_DQN_HIDDEN,
                true,
                achf.clone(),
                seed.wrapping_add(500),
            ))
        } else {
            None
        };

        DuelingQNetwork {
            l1,
            l2,
            l3,
            val_head,
            adv_head,
            achf: achf_layer,
        }
    }

    pub fn forward_impl(&self, state: &Tensor) -> Tensor {
        // state: (Batch, 8) or (8)
        let x = self.l1.forward(state).relu();
        let x = self.l2.forward(&x).relu();
        let x = if let Some(achf) = &self.achf {
            achf.forward(&x).relu()
        } else {
            self.l3.forward(&x).relu()
        };

        let val = self.val_head.forward(&x); // (Batch, 1) or (1)
        let adv = self.adv_head.forward(&x); // (Batch, 5) or (5)

        // Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))

        if state.shape.len() == 2 {
            // Batch Mode

            // val is (B, 1). Expand to (B, 5).
            // Multiply by ones(1, 5) -> (B, 5)
            // MatMul: (B, 1) x (1, 5) -> (B, 5)
            let ones_1_5 = Tensor::new_f32(vec![1.0; 5], vec![1, 5]);
            #[cfg(cuda)]
            let ones_1_5 = if state.device == crate::autograd::Device::Cuda {
                ones_1_5.to_cuda().unwrap_or(ones_1_5)
            } else {
                ones_1_5
            };
            let val_expanded = val.matmul(&ones_1_5);

            // Mean Adv: (B, 5) -> (B, 1)
            // Multiply by ones(5, 1) / 5.0
            let ones_5_1 = Tensor::new_f32(vec![0.2; 5], vec![5, 1]); // 1/5 = 0.2
            #[cfg(cuda)]
            let ones_5_1 = if state.device == crate::autograd::Device::Cuda {
                ones_5_1.to_cuda().unwrap_or(ones_5_1)
            } else {
                ones_5_1
            };
            let mean_adv = adv.matmul(&ones_5_1); // (B, 1)
            let mean_adv_expanded = mean_adv.matmul(&ones_1_5); // (B, 5)

            // Result: val + adv - mean
            val_expanded + adv - mean_adv_expanded
        } else {
            // Single Mode
            let mean_adv_scalar = adv.mean(); // (1)
            let val_expanded = val.broadcast(vec![ACTION_SPACE]); // (5)
            let mean_adv_broadcast = mean_adv_scalar.broadcast(vec![ACTION_SPACE]); // (5)

            val_expanded + adv - mean_adv_broadcast
        }
    }

    pub fn forward(&self, state: &Tensor) -> Tensor {
        self.forward_impl(state)
    }

    pub fn achf_config(&self) -> AchfConfig {
        self.achf
            .as_ref()
            .map(|achf| achf.config.clone())
            .unwrap_or_default()
    }

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        self.l1.to_cuda();
        self.l2.to_cuda();
        self.l3.to_cuda();
        self.val_head.to_cuda();
        self.adv_head.to_cuda();
        if let Some(ref mut achf) = self.achf {
            achf.to_cuda();
        }
    }

    pub fn update_achf_after_backward(&self) {
        if let Some(achf) = &self.achf {
            achf.update_after_backward();
        }
    }

    pub fn freeze_achf_for_inference(&mut self) {
        if let Some(achf) = &mut self.achf {
            achf.freeze_for_inference();
        }
    }

    pub fn prune_achf(&mut self, threshold: f64) {
        if let Some(achf) = &mut self.achf {
            achf.prune(threshold);
        }
    }

    pub fn achf_cache_stats(&self) -> Option<crate::achf::AchfCacheStats> {
        self.achf.as_ref().map(|achf| achf.cache_stats())
    }

    pub fn snapshot_achf(&self) -> Option<crate::achf::AchfStateSnapshot> {
        self.achf.as_ref().map(|achf| achf.snapshot_state())
    }

    pub fn param_count(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape.iter().product::<usize>())
            .sum()
    }

    pub fn achf_orthogonal_penalty(&self) -> Option<Tensor> {
        self.achf
            .as_ref()
            .and_then(|achf| achf.orthogonal_penalty())
    }

    pub fn to_inference_bf16(&self) -> Self {
        let mut out = self.clone();
        out.l1 = self.l1.to_inference_bf16();
        out.l2 = self.l2.to_inference_bf16();
        out.l3 = self.l3.to_inference_bf16();
        out.val_head = self.val_head.to_inference_bf16();
        out.adv_head = self.adv_head.to_inference_bf16();
        out.achf = self.achf.as_ref().map(AchfLayer::to_inference_bf16);
        out
    }

    fn forward_inference_batch_values(&self, states: &[f32]) -> Vec<f32> {
        if states.is_empty() {
            return Vec::new();
        }
        assert!(
            states.len().is_multiple_of(DIM),
            "DQN batch inference input length {} is not divisible by feature dim {}",
            states.len(),
            DIM
        );
        let batch = states.len() / DIM;
        let mut h1 = vec![0.0f32; batch * self.l1.out_features];
        let mut h2 = vec![0.0f32; batch * self.l2.out_features];
        let mut h3 = vec![0.0f32; batch * self.l3.out_features];
        let mut val = vec![0.0f32; batch * self.val_head.out_features];
        let mut adv = vec![0.0f32; batch * self.adv_head.out_features];

        self.l1.forward_inference_into(states, &mut h1);
        for v in &mut h1 {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        self.l2.forward_inference_into(&h1, &mut h2);
        for v in &mut h2 {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        if let Some(achf) = &self.achf {
            let out = achf.forward_inference_residual(&h2);
            h3.resize(out.len(), 0.0);
            h3.copy_from_slice(&out);
        } else {
            h3.resize(h2.len(), 0.0);
            self.l3.forward_inference_into(&h2, &mut h3);
        }
        for v in &mut h3 {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        self.val_head.forward_inference_into(&h3, &mut val);
        self.adv_head.forward_inference_into(&h3, &mut adv);

        let mut q_values = vec![0.0f32; batch * ACTION_SPACE];
        for b in 0..batch {
            let val_b = val[b];
            let adv_row = &adv[b * ACTION_SPACE..(b + 1) * ACTION_SPACE];
            let mean_adv = adv_row.iter().sum::<f32>() / ACTION_SPACE as f32;
            for (i, &a) in adv_row.iter().enumerate() {
                q_values[b * ACTION_SPACE + i] = val_b + a - mean_adv;
            }
        }
        q_values
    }

    #[cfg(test)]
    pub fn forward_inference_batch(&self, state: &Tensor) -> Tensor {
        let states = state.data_to_f32_vec();
        let q_values = self.forward_inference_batch_values(&states);
        let batch = if state.shape.len() == 2 {
            state.shape[0]
        } else {
            1
        };
        let shape = if batch == 1 {
            vec![ACTION_SPACE]
        } else {
            vec![batch, ACTION_SPACE]
        };
        Tensor::new_f32(q_values.into_iter().map(|v| v as f64).collect(), shape)
    }

    // Copy weights
    pub fn load_state_dict(&mut self, other: &Self) {
        fn copy_tensor(dst: &mut Tensor, src: &Tensor) {
            match (dst.dtype, src.dtype) {
                (crate::dtype::Dtype::F32, crate::dtype::Dtype::F32) => {
                    let src_data = src.data_f32().clone();
                    let mut dst_data = dst.data_write_f32();
                    *dst_data = src_data;
                }
                (crate::dtype::Dtype::BF16, crate::dtype::Dtype::BF16) => {
                    let src_data = src.data_bf16().clone();
                    let mut dst_data = dst.data_write_bf16();
                    *dst_data = src_data;
                }
                (crate::dtype::Dtype::F64, crate::dtype::Dtype::F64) => {
                    let src_data = src.data_f64().clone();
                    let mut dst_data = dst.data_write_f64();
                    *dst_data = src_data;
                }
                _ => {
                    let src_data = src.data_as_f64_vec();
                    if dst.dtype == crate::dtype::Dtype::F32 {
                        let mut dst_data = dst.data_write_f32();
                        *dst_data = src_data.iter().map(|&v| v as f32).collect();
                    } else if dst.dtype == crate::dtype::Dtype::BF16 {
                        let mut dst_data = dst.data_write_bf16();
                        *dst_data = src_data
                            .iter()
                            .map(|&v| crate::dtype::bf16::from_f64(v))
                            .collect();
                    } else {
                        let mut dst_data = dst.data_write_f64();
                        *dst_data = src_data;
                    }
                }
            }
        }

        let copy_linear = |dst: &mut Linear, src: &Linear| {
            copy_tensor(&mut dst.weight, &src.weight);
            if let (Some(db), Some(sb)) = (&mut dst.bias, &src.bias) {
                copy_tensor(db, sb);
            }
        };

        copy_linear(&mut self.l1, &other.l1);
        copy_linear(&mut self.l2, &other.l2);
        copy_linear(&mut self.l3, &other.l3);
        copy_linear(&mut self.val_head, &other.val_head);
        copy_linear(&mut self.adv_head, &other.adv_head);
        if let (Some(dst), Some(src)) = (&mut self.achf, &other.achf) {
            dst.load_state_dict(src);
        }
    }

    pub fn soft_update(&mut self, source: &Self, tau: f64) {
        fn interpolate(target: &mut Tensor, source: &Tensor, tau: f64) {
            #[cfg(cuda)]
            if target.cuda_lerp_in_place_from(source, tau) {
                return;
            }

            let tau_f32 = tau as f32;
            match (target.dtype, source.dtype) {
                (crate::dtype::Dtype::F32, crate::dtype::Dtype::F32) => {
                    let mut t_data = target.data_write_f32();
                    let s_data = source.data_f32();
                    for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                        *t = *t * (1.0 - tau_f32) + *s * tau_f32;
                    }
                }
                (crate::dtype::Dtype::BF16, crate::dtype::Dtype::BF16) => {
                    let mut t_data = target.data_write_bf16();
                    let s_data = source.data_bf16();
                    for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                        let tv = t.to_f32();
                        let sv = s.to_f32();
                        *t = crate::dtype::bf16::from_f32(tv * (1.0 - tau_f32) + sv * tau_f32);
                    }
                }
                (crate::dtype::Dtype::F64, crate::dtype::Dtype::F64) => {
                    let mut t_data = target.data_write_f64();
                    let s_data = source.data_f64();
                    for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                        *t = *t * (1.0 - tau) + *s * tau;
                    }
                }
                _ => {
                    let t_data = target.data_as_f64_vec();
                    let s_data = source.data_as_f64_vec();
                    let new_data: Vec<f64> = t_data
                        .iter()
                        .zip(s_data.iter())
                        .map(|(t, s)| t * (1.0 - tau) + s * tau)
                        .collect();
                    if target.dtype == crate::dtype::Dtype::F32 {
                        let mut dst = target.data_write_f32();
                        *dst = new_data.iter().map(|&v| v as f32).collect();
                    } else if target.dtype == crate::dtype::Dtype::BF16 {
                        let mut dst = target.data_write_bf16();
                        *dst = new_data
                            .iter()
                            .map(|&v| crate::dtype::bf16::from_f64(v))
                            .collect();
                    } else {
                        let mut dst = target.data_write_f64();
                        *dst = new_data;
                    }
                }
            }
        }

        let update_linear = |dst: &mut Linear, src: &Linear| {
            interpolate(&mut dst.weight, &src.weight, tau);
            if let (Some(db), Some(sb)) = (&mut dst.bias, &src.bias) {
                interpolate(db, sb, tau);
            }
        };

        update_linear(&mut self.l1, &source.l1);
        update_linear(&mut self.l2, &source.l2);
        update_linear(&mut self.l3, &source.l3);
        update_linear(&mut self.val_head, &source.val_head);
        update_linear(&mut self.adv_head, &source.adv_head);
        if let (Some(dst), Some(src)) = (&mut self.achf, &source.achf) {
            dst.soft_update(src, tau);
        }
    }

    pub fn predict_action(&self, state: &Tensor) -> (usize, f32) {
        let q_values = self.forward(state);
        #[cfg(cuda)]
        if let Some(max_idx) = argmax_cuda_index(&q_values) {
            return (max_idx, ACTIONS[max_idx] as f32);
        }
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        let q_data = q_values.data_to_f32_vec();
        for (i, &val) in q_data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        (max_idx, ACTIONS[max_idx] as f32)
    }

    /// Zero-allocation inference: compute Q-values from a raw feature slice
    /// using `Linear::forward_inference_into`, bypassing the autograd `Tensor` graph.
    ///
    /// This function uses thread-local scratch buffers to avoid allocations in hot paths.
    /// Uses RepCache to avoid recomputing Q-values for previously seen states.
    pub fn predict_action_fast(&self, state: &[f32]) -> (usize, f32) {
        struct Scratch {
            h1: Vec<f32>,
            h2: Vec<f32>,
            h3: Vec<f32>,
            val: Vec<f32>,
            adv: Vec<f32>,
        }

        // RepCache: bounded hash map for state -> Q-values
        struct RepCache {
            entries: std::collections::HashMap<u64, [f32; ACTION_SPACE]>,
        }

        impl RepCache {
            // FNV-1a hash, same as ACHF input_hash
            fn state_hash(x: &[f32]) -> u64 {
                let mut h: u64 = 0xcbf29ce484222325;
                let step = (x.len() / 64).clamp(1, 8);
                for i in (0..x.len()).step_by(step) {
                    let quantized = (x[i] * 10000.0).round() as i64;
                    let bytes = quantized.to_le_bytes();
                    for &b in &bytes {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                }
                h
            }

            fn get(&self, hash: u64) -> Option<&[f32; ACTION_SPACE]> {
                self.entries.get(&hash).map(|q| unsafe {
                    // Safe: ACTION_SPACE is constant and we're just reinterpreting
                    &*(q.as_slice() as *const [f32] as *const [_; ACTION_SPACE])
                })
            }

            fn insert(&mut self, hash: u64, q_values: [f32; ACTION_SPACE]) {
                if self.entries.len() >= 1024 {
                    if let Some(key) = self.entries.keys().next().copied() {
                        self.entries.remove(&key);
                    }
                }
                self.entries.insert(hash, q_values);
            }
        }

        thread_local! {
            static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch {
                h1: Vec::new(),
                h2: Vec::new(),
                h3: Vec::new(),
                val: Vec::new(),
                adv: Vec::new(),
            }) };
        }
        thread_local! {
            static REP_CACHE: RefCell<RepCache> = RefCell::new(RepCache {
                entries: std::collections::HashMap::with_capacity(1024),
            });
        }

        let state_hash = RepCache::state_hash(state);

        // Check RepCache first (cache hit path)
        let cache_hit = REP_CACHE.with(|cache| cache.borrow().get(state_hash).is_some());
        if cache_hit {
            return REP_CACHE.with(|cache| {
                let binding = cache.borrow();
                let q_ref = binding.get(state_hash).unwrap();
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &q) in q_ref.iter().enumerate() {
                    if q > max_val {
                        max_val = q;
                        max_idx = i;
                    }
                }
                (max_idx, ACTIONS[max_idx] as f32)
            });
        }

        // Cache miss: compute forward pass
        SCRATCH.with(|scratch| {
            let mut s = scratch.borrow_mut();
            let Scratch {
                h1,
                h2,
                h3,
                val,
                adv,
            } = &mut *s;

            self.l1.forward_inference_into(state, h1);
            for v in h1.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }

            self.l2.forward_inference_into(h1, h2);
            for v in h2.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }

            if let Some(achf) = &self.achf {
                let out = achf.forward_inference_residual(h2);
                h3.resize(out.len(), 0.0);
                h3.copy_from_slice(&out);
            } else {
                h3.resize(h2.len(), 0.0);
                self.l3.forward_inference_into(h2, h3);
            }
            for v in h3.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }

            self.val_head.forward_inference_into(h3, val);
            self.adv_head.forward_inference_into(h3, adv);
            if adv.len() != ACTION_SPACE {
                return (0, ACTIONS[0] as f32);
            }

            let mean_adv: f32 = adv.iter().sum::<f32>() / ACTION_SPACE as f32;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            let base = val.first().copied().unwrap_or(0.0);
            let mut q_values = [0.0f32; ACTION_SPACE];
            for (i, &a) in adv.iter().enumerate() {
                let q = base + a - mean_adv;
                q_values[i] = q;
                if q > max_val {
                    max_val = q;
                    max_idx = i;
                }
            }

            // Store in RepCache
            REP_CACHE.with(|cache| {
                cache.borrow_mut().insert(state_hash, q_values);
            });

            (max_idx, ACTIONS[max_idx] as f32)
        })
    }
}

// --- Optimizer ---

struct Adam {
    params: Vec<Tensor>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl Adam {
    fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.data.len()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.data.len()]).collect();
        Adam {
            params,
            m,
            v,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        }
    }

    fn step(&mut self) {
        self.t += 1;
        // Global gradient clipping (max_norm = 1.0) using f32 grad
        let mut total_norm_sq = 0.0f32;
        for param in &self.params {
            let grad = param.grad_to_f32_vec();
            for &g in grad.iter() {
                total_norm_sq += g * g;
            }
        }
        let total_norm = (total_norm_sq as f64).sqrt();
        let clip_coef = if total_norm > 1.0 {
            1.0 / total_norm
        } else {
            1.0
        };
        let clip = clip_coef as f32;

        let bc1 = (1.0 - self.beta1.powi(self.t as i32)) as f32;
        let bc2 = (1.0 - self.beta2.powi(self.t as i32)) as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let wd = self.weight_decay as f32;
        let b1 = self.beta1 as f32;
        let b2 = self.beta2 as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = param.grad_to_f32_vec();
            if param.dtype == crate::dtype::Dtype::F64 {
                let mut data = param.data_write_f64();
                for j in 0..data.len() {
                    let g = grad[j] * clip;
                    self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * g;
                    self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * g * g;
                    let m_hat = self.m[i][j] / bc1;
                    let v_hat = self.v[i][j] / bc2;
                    let update = lr * (m_hat / (v_hat.sqrt() + eps) + wd * data[j] as f32);
                    data[j] -= update as f64;
                }
            } else {
                let mut data = param.data_write_f32();
                for j in 0..data.len() {
                    let g = grad[j] * clip;
                    self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * g;
                    self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * g * g;
                    let m_hat = self.m[i][j] / bc1;
                    let v_hat = self.v[i][j] / bc2;
                    data[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * data[j]);
                }
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

#[cfg(cuda)]
struct GpuAdam {
    params: Vec<Tensor>,
    m: Vec<std::sync::Arc<crate::cuda::memory::CudaBuffer>>,
    v: Vec<std::sync::Arc<crate::cuda::memory::CudaBuffer>>,
    t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

#[cfg(cuda)]
impl GpuAdam {
    fn new(params: Vec<Tensor>, lr: f64) -> Option<Self> {
        use crate::cuda::memory::CudaBuffer;
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            let len = p.cuda_storage_len();
            if p.device != crate::autograd::Device::Cuda {
                return None;
            }
            let (d_m, d_v) = match p.dtype {
                crate::dtype::Dtype::F32 => {
                    let dm = crate::cuda::memory::alloc::<f32>(len).ok()?;
                    let dv = crate::cuda::memory::alloc::<f32>(len).ok()?;
                    let zeros = vec![0.0f32; len];
                    crate::cuda::memory::copy_h2d(&dm, &zeros).ok()?;
                    crate::cuda::memory::copy_h2d(&dv, &zeros).ok()?;
                    (CudaBuffer::F32(dm), CudaBuffer::F32(dv))
                }
                crate::dtype::Dtype::F64 => {
                    let dm = crate::cuda::memory::alloc::<f64>(len).ok()?;
                    let dv = crate::cuda::memory::alloc::<f64>(len).ok()?;
                    let zeros = vec![0.0f64; len];
                    crate::cuda::memory::copy_h2d(&dm, &zeros).ok()?;
                    crate::cuda::memory::copy_h2d(&dv, &zeros).ok()?;
                    (CudaBuffer::F64(dm), CudaBuffer::F64(dv))
                }
                _ => return None,
            };
            m.push(std::sync::Arc::new(d_m));
            v.push(std::sync::Arc::new(d_v));
        }
        Some(GpuAdam {
            params,
            m,
            v,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        })
    }

    fn step(&mut self) {
        use crate::cuda::memory::CudaBuffer;
        self.t += 1;
        crate::cuda::record_optimizer_attempt();
        if !crate::autograd::cuda_clip_gradients_in_place(&self.params, 1.0, 1e-6) {
            crate::cuda::record_optimizer_fallback();
            return;
        }

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        let mut all_ok = true;
        for (i, param) in self.params.iter().enumerate() {
            let len = param.cuda_storage_len();
            if len == 0 {
                continue;
            }
            let d_params = match param.cuda_get_or_upload_buffer() {
                Ok(buf) => buf,
                Err(_) => {
                    crate::cuda::record_optimizer_fallback();
                    all_ok = false;
                    continue;
                }
            };
            let d_grads = match param.cuda_grad_get_or_upload_buffer() {
                Ok(buf) => buf,
                Err(_) => {
                    crate::cuda::record_optimizer_fallback();
                    all_ok = false;
                    continue;
                }
            };
            let d_m = self.m[i].clone();
            let d_v = self.v[i].clone();

            let step_ok = match (param.dtype, &*d_params, &*d_grads, &*d_m, &*d_v) {
                (
                    crate::dtype::Dtype::F32,
                    CudaBuffer::F32(p),
                    CudaBuffer::F32(g),
                    CudaBuffer::F32(mbuf),
                    CudaBuffer::F32(vbuf),
                ) => crate::cuda::kernels::adam_step_f32(
                    p,
                    g,
                    mbuf,
                    vbuf,
                    len,
                    self.lr as f32,
                    self.beta1 as f32,
                    self.beta2 as f32,
                    self.eps as f32,
                    self.weight_decay as f32,
                    bias_correction1 as f32,
                    bias_correction2 as f32,
                    1.0,
                )
                .is_ok(),
                (
                    crate::dtype::Dtype::F64,
                    CudaBuffer::F64(p),
                    CudaBuffer::F64(g),
                    CudaBuffer::F64(mbuf),
                    CudaBuffer::F64(vbuf),
                ) => crate::cuda::kernels::adam_step(
                    p,
                    g,
                    mbuf,
                    vbuf,
                    len,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    bias_correction1,
                    bias_correction2,
                    1.0,
                )
                .is_ok(),
                _ => false,
            };
            if !step_ok {
                crate::cuda::record_optimizer_fallback();
                all_ok = false;
            }
        }
        if all_ok {
            crate::cuda::record_optimizer_success();
        }
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

enum Optimizer {
    Cpu(Adam),
    #[cfg(cuda)]
    Gpu(GpuAdam),
}

impl Optimizer {
    fn step(&mut self) {
        match self {
            Optimizer::Cpu(o) => o.step(),
            #[cfg(cuda)]
            Optimizer::Gpu(o) => o.step(),
        }
    }

    fn zero_grad(&self) {
        match self {
            Optimizer::Cpu(o) => o.zero_grad(),
            #[cfg(cuda)]
            Optimizer::Gpu(o) => o.zero_grad(),
        }
    }
}

// --- SumTree for O(log N) proportional PER sampling ---

struct SumTree {
    capacity: usize,
    tree: Vec<f64>,
    data: Vec<Option<Experience>>,
    write_pos: usize,
    size: usize,
}

impl SumTree {
    fn new(capacity: usize) -> Self {
        SumTree {
            capacity,
            tree: vec![0.0; 2 * capacity],
            data: (0..capacity).map(|_| None).collect(),
            write_pos: 0,
            size: 0,
        }
    }

    fn total_priority(&self) -> f64 {
        self.tree[1]
    }

    fn add(&mut self, priority: f64, exp: Experience) {
        let idx = self.write_pos;
        self.data[idx] = Some(exp);
        self.update(idx, priority);
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    fn update(&mut self, data_idx: usize, priority: f64) {
        let mut tree_idx = data_idx + self.capacity;
        self.tree[tree_idx] = priority;
        while tree_idx > 1 {
            tree_idx >>= 1;
            self.tree[tree_idx] = self.tree[tree_idx * 2] + self.tree[tree_idx * 2 + 1];
        }
    }

    // Retrieve the leaf whose cumulative sum covers `value`.
    fn get(&self, mut value: f64) -> (usize, f64) {
        let mut idx = 1;
        while idx < self.capacity {
            let left = idx * 2;
            let right = left + 1;
            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                idx = right;
            }
        }
        let data_idx = idx - self.capacity;
        (data_idx, self.tree[idx])
    }
}

// --- Replay Buffer (SumTree-backed PER) ---

/// Transition tuple for DQN replay buffer.
#[derive(Clone)]
pub struct Experience {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

struct PERSample {
    experiences: Vec<Experience>,
    indices: Vec<usize>,
    is_weights: Vec<f64>,
    #[cfg(cuda)]
    cuda_batch: Option<CudaPERSample>,
}

struct ReplayBuffer {
    tree: SumTree,
    alpha: f64,
    max_priority: f64,
    #[cfg(cuda)]
    cuda: Option<CudaReplayMirror>,
}

#[cfg(cuda)]
struct CudaPERSample {
    states: Tensor,
    next_states: Tensor,
    action_mask: Tensor,
    rewards: Tensor,
    dones: Tensor,
    is_weights: Tensor,
    indices: crate::cuda::memory::DevicePtr<i32>,
}

#[cfg(cuda)]
struct CudaReplayMirror {
    capacity: usize,
    dim: usize,
    states: crate::cuda::memory::DevicePtr<f32>,
    next_states: crate::cuda::memory::DevicePtr<f32>,
    actions: crate::cuda::memory::DevicePtr<i32>,
    rewards: crate::cuda::memory::DevicePtr<f32>,
    dones: crate::cuda::memory::DevicePtr<f32>,
    priorities: crate::cuda::memory::DevicePtr<f32>,
    max_priority: crate::cuda::memory::DevicePtr<f32>,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        ReplayBuffer {
            tree: SumTree::new(capacity),
            alpha: PER_ALPHA,
            max_priority: 1.0,
            #[cfg(cuda)]
            cuda: CudaReplayMirror::new(capacity, DIM).ok(),
        }
    }

    fn push(&mut self, exp: Experience) {
        let priority = self.max_priority.powf(self.alpha);
        #[cfg(cuda)]
        let idx = self.tree.write_pos;
        #[cfg(cuda)]
        if let Some(cuda) = self.cuda.as_mut() {
            if cuda.push(idx, &exp, self.alpha).is_err() {
                self.cuda = None;
            }
        }
        self.tree.add(priority, exp);
    }

    /// Proportional PER sampling with importance-sampling weights.
    /// `beta` controls IS correction strength (annealed from PER_BETA_START to PER_BETA_END).
    fn sample(&self, rng: &mut Rng, batch_size: usize, beta: f64) -> PERSample {
        assert!(batch_size > 0, "batch_size must be > 0");
        assert!(self.tree.size > 0, "cannot sample from empty buffer");

        #[cfg(cuda)]
        if let Some(cuda_batch) = self
            .cuda
            .as_ref()
            .and_then(|cuda| cuda.sample(rng, self.tree.size, batch_size, beta))
        {
            return PERSample {
                experiences: Vec::new(),
                indices: Vec::new(),
                is_weights: Vec::new(),
                cuda_batch: Some(cuda_batch),
            };
        }

        let total = self.tree.total_priority();

        // If all priorities are zero (degenerate), fall back to uniform sampling
        if total <= 0.0 {
            let mut experiences = Vec::with_capacity(batch_size);
            let mut indices = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let idx = rng.next_u64_bounded(self.tree.size as u64) as usize;
                if let Some(exp) = &self.tree.data[idx] {
                    experiences.push(exp.clone());
                    indices.push(idx);
                }
            }
            let is_weights = vec![1.0; experiences.len()];
            return PERSample {
                experiences,
                indices,
                is_weights,
                #[cfg(cuda)]
                cuda_batch: None,
            };
        }

        let segment = total / batch_size as f64;
        let n = self.tree.size as f64;

        let mut experiences = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut priorities = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let lo = segment * i as f64;
            let hi = segment * (i + 1) as f64;
            let value = lo + rng.next_f64() * (hi - lo);
            let (data_idx, priority) = self.tree.get(value.min(total - 1e-12));

            if let Some(exp) = &self.tree.data[data_idx] {
                experiences.push(exp.clone());
                indices.push(data_idx);
                priorities.push(priority);
            } else {
                // Fallback: resample up to 3 times to find a non-empty slot
                for _ in 0..3 {
                    let fallback_val = rng.next_f64() * total;
                    let (fb_idx, fb_pri) = self.tree.get(fallback_val);
                    if let Some(exp) = &self.tree.data[fb_idx] {
                        experiences.push(exp.clone());
                        indices.push(fb_idx);
                        priorities.push(fb_pri);
                        break;
                    }
                }
            }
        }

        // IS weights: w_i = (N * P(i))^{-beta}, normalized by max(w)
        let mut is_weights = Vec::with_capacity(priorities.len());
        let mut max_weight = f64::NEG_INFINITY;
        for &p in &priorities {
            let prob = (p / total).max(1e-12);
            let w = (n * prob).powf(-beta);
            if w > max_weight {
                max_weight = w;
            }
            is_weights.push(w);
        }
        if max_weight > 0.0 {
            for w in &mut is_weights {
                *w /= max_weight;
            }
        }

        PERSample {
            experiences,
            indices,
            is_weights,
            #[cfg(cuda)]
            cuda_batch: None,
        }
    }

    fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        let capacity = self.tree.capacity;
        for (&idx, &td) in indices.iter().zip(td_errors.iter()) {
            if idx >= capacity {
                continue;
            }
            let clipped_td = if td.is_finite() { td.abs() } else { 1.0 };
            let priority = (clipped_td + PER_EPSILON).powf(self.alpha);
            self.tree.update(idx, priority);
            if clipped_td + PER_EPSILON > self.max_priority {
                self.max_priority = clipped_td + PER_EPSILON;
            }
        }
    }

    fn len(&self) -> usize {
        self.tree.size
    }

    #[cfg(cuda)]
    fn update_priorities_cuda(
        &mut self,
        indices: &crate::cuda::memory::DevicePtr<i32>,
        td_errors: &Tensor,
        batch_size: usize,
    ) -> bool {
        self.cuda
            .as_mut()
            .is_some_and(|cuda| cuda.update_priorities(indices, td_errors, batch_size, self.alpha))
    }
}

#[cfg(cuda)]
impl CudaReplayMirror {
    fn new(capacity: usize, dim: usize) -> crate::cuda::error::CudaResult<Self> {
        use crate::cuda::memory::{alloc, copy_h2d};

        let states = alloc::<f32>(capacity * dim)?;
        let next_states = alloc::<f32>(capacity * dim)?;
        let actions = alloc::<i32>(capacity)?;
        let rewards = alloc::<f32>(capacity)?;
        let dones = alloc::<f32>(capacity)?;
        let priorities = alloc::<f32>(capacity)?;
        let max_priority = alloc::<f32>(1)?;

        let zeros = vec![0.0_f32; capacity * dim];
        copy_h2d(&states, &zeros)?;
        copy_h2d(&next_states, &zeros)?;
        copy_h2d(&actions, &vec![0_i32; capacity])?;
        copy_h2d(&rewards, &vec![0.0_f32; capacity])?;
        copy_h2d(&dones, &vec![0.0_f32; capacity])?;
        copy_h2d(&priorities, &vec![0.0_f32; capacity])?;
        copy_h2d(&max_priority, &[1.0_f32])?;

        Ok(Self {
            capacity,
            dim,
            states,
            next_states,
            actions,
            rewards,
            dones,
            priorities,
            max_priority,
        })
    }

    fn push(
        &mut self,
        idx: usize,
        exp: &Experience,
        alpha: f64,
    ) -> crate::cuda::error::CudaResult<()> {
        use crate::cuda::memory::{alloc, copy_h2d};

        if idx >= self.capacity || exp.state.len() != self.dim || exp.next_state.len() != self.dim {
            return Err(crate::cuda::error::CudaError::InvalidInput {
                op: "CudaReplayMirror::push",
                message: "transition shape/index mismatch",
            });
        }

        let state = alloc::<f32>(self.dim)?;
        let next_state = alloc::<f32>(self.dim)?;
        let state_f32: Vec<f32> = exp.state.iter().map(|&v| v as f32).collect();
        let next_state_f32: Vec<f32> = exp.next_state.iter().map(|&v| v as f32).collect();
        copy_h2d(&state, &state_f32)?;
        copy_h2d(&next_state, &next_state_f32)?;

        crate::cuda::kernels::per_store_transition_with_max_f32(
            &self.states,
            &self.next_states,
            &self.actions,
            &self.rewards,
            &self.dones,
            &self.priorities,
            &self.max_priority,
            &state,
            &next_state,
            idx,
            exp.action,
            exp.reward as f32,
            if exp.done { 1.0 } else { 0.0 },
            alpha as f32,
            self.capacity,
            self.dim,
        )
    }

    fn sample(
        &self,
        rng: &mut Rng,
        size: usize,
        batch_size: usize,
        beta: f64,
    ) -> Option<CudaPERSample> {
        use crate::autograd::Device;
        use crate::cuda::memory::{alloc, copy_h2d, CudaBuffer};
        use crate::dtype::{Dtype, Storage};
        use std::sync::{Arc, RwLock};

        if size == 0 || size > self.capacity || batch_size == 0 {
            return None;
        }

        let uniforms = alloc::<f32>(batch_size).ok()?;
        let host_uniforms: Vec<f32> = (0..batch_size).map(|_| rng.next_f64() as f32).collect();
        copy_h2d(&uniforms, &host_uniforms).ok()?;

        let batch_states = alloc::<f32>(batch_size * self.dim).ok()?;
        let batch_next_states = alloc::<f32>(batch_size * self.dim).ok()?;
        let batch_action_mask = alloc::<f32>(batch_size * ACTION_SPACE).ok()?;
        let batch_rewards = alloc::<f32>(batch_size).ok()?;
        let batch_dones = alloc::<f32>(batch_size).ok()?;
        let batch_weights = alloc::<f32>(batch_size).ok()?;
        let batch_indices = alloc::<i32>(batch_size).ok()?;

        crate::cuda::kernels::per_sample_f32(
            &self.states,
            &self.next_states,
            &self.actions,
            &self.rewards,
            &self.dones,
            &self.priorities,
            &uniforms,
            &batch_states,
            &batch_next_states,
            &batch_action_mask,
            &batch_rewards,
            &batch_dones,
            &batch_weights,
            &batch_indices,
            size,
            self.capacity,
            self.dim,
            ACTION_SPACE,
            batch_size,
            beta as f32,
            0.0,
        )
        .ok()?;

        fn tensor_from_f32_device(
            device: crate::cuda::memory::DevicePtr<f32>,
            shape: Vec<usize>,
        ) -> Tensor {
            let len = shape.iter().product();
            let tensor = Tensor {
                data: Storage::F32(Arc::new(RwLock::new(Vec::new()))),
                grad: Storage::zeros(len, Tensor::grad_dtype_for(Dtype::F32)),
                shape,
                device: Device::Cuda,
                dtype: Dtype::F32,
                _ctx: None,
            };
            tensor.cuda_set_cached_buffer(Arc::new(CudaBuffer::F32(device)));
            tensor
        }

        Some(CudaPERSample {
            states: tensor_from_f32_device(batch_states, vec![batch_size, self.dim]),
            next_states: tensor_from_f32_device(batch_next_states, vec![batch_size, self.dim]),
            action_mask: tensor_from_f32_device(batch_action_mask, vec![batch_size, ACTION_SPACE]),
            rewards: tensor_from_f32_device(batch_rewards, vec![batch_size, 1]),
            dones: tensor_from_f32_device(batch_dones, vec![batch_size, 1]),
            is_weights: tensor_from_f32_device(batch_weights, vec![batch_size, 1]),
            indices: batch_indices,
        })
    }

    fn update_priorities(
        &mut self,
        indices: &crate::cuda::memory::DevicePtr<i32>,
        td_errors: &Tensor,
        batch_size: usize,
        alpha: f64,
    ) -> bool {
        use crate::autograd::Device;
        use crate::cuda::memory::CudaBuffer;
        use crate::dtype::Dtype;

        if td_errors.device != Device::Cuda
            || td_errors.dtype != Dtype::F32
            || td_errors.numel() != batch_size
        {
            return false;
        }
        let Ok(td_buf) = td_errors.cuda_get_or_upload_buffer() else {
            return false;
        };
        let CudaBuffer::F32(td) = &*td_buf else {
            return false;
        };
        crate::cuda::kernels::per_update_priorities_f32(
            &self.priorities,
            indices,
            td,
            &self.max_priority,
            batch_size,
            self.capacity,
            alpha as f32,
            PER_EPSILON as f32,
        )
        .is_ok()
    }
}

fn dqn_cpu_batch_tensors(
    per_sample: &PERSample,
    scratch: &mut DqnTrainerScratch,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    scratch.reset();

    for exp in &per_sample.experiences {
        scratch.states_vec.extend_from_slice(&exp.state);
        scratch.next_states_vec.extend_from_slice(&exp.next_state);
        let mask_start = scratch.actions_vec.len();
        scratch.actions_vec.resize(mask_start + ACTION_SPACE, 0.0);
        scratch.actions_vec[mask_start + exp.action] = 1.0;
        scratch.rewards_vec.push(exp.reward);
        scratch.dones_vec.push(if exp.done { 1.0 } else { 0.0 });
    }

    let batch_state = Tensor::new_f32(
        std::mem::take(&mut scratch.states_vec),
        vec![BATCH_SIZE, DIM],
    );
    let batch_next_state = Tensor::new_f32(
        std::mem::take(&mut scratch.next_states_vec),
        vec![BATCH_SIZE, DIM],
    );
    let batch_mask = Tensor::new_f32(
        std::mem::take(&mut scratch.actions_vec),
        vec![BATCH_SIZE, ACTION_SPACE],
    );
    let rewards_tensor = Tensor::new_f32(
        std::mem::take(&mut scratch.rewards_vec),
        vec![BATCH_SIZE, 1],
    );
    let dones_tensor = Tensor::new_f32(std::mem::take(&mut scratch.dones_vec), vec![BATCH_SIZE, 1]);
    let is_weights_tensor = Tensor::new_f32(per_sample.is_weights.clone(), vec![BATCH_SIZE, 1]);

    #[cfg(cuda)]
    {
        let batch_state = match batch_state.to_cuda() {
            Ok(t) => t,
            Err(_) => batch_state,
        };
        let batch_next_state = match batch_next_state.to_cuda() {
            Ok(t) => t,
            Err(_) => batch_next_state,
        };
        let batch_mask = match batch_mask.to_cuda() {
            Ok(t) => t,
            Err(_) => batch_mask,
        };
        let rewards_tensor = match rewards_tensor.to_cuda() {
            Ok(t) => t,
            Err(_) => rewards_tensor,
        };
        let dones_tensor = match dones_tensor.to_cuda() {
            Ok(t) => t,
            Err(_) => dones_tensor,
        };
        let is_weights_tensor = match is_weights_tensor.to_cuda() {
            Ok(t) => t,
            Err(_) => is_weights_tensor,
        };
        (
            batch_state,
            batch_next_state,
            batch_mask,
            rewards_tensor,
            dones_tensor,
            is_weights_tensor,
        )
    }

    #[cfg(not(cuda))]
    {
        (
            batch_state,
            batch_next_state,
            batch_mask,
            rewards_tensor,
            dones_tensor,
            is_weights_tensor,
        )
    }
}

// --- Training Loop ---

/// Train a DQN agent using Double DQN with Prioritized Experience Replay.
pub fn train_dqn(
    _initial_model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
) -> DuelingQNetwork {
    train_dqn_impl(
        _initial_model,
        rng,
        env_net,
        config,
        TrainingMetricsSink::noop(),
    )
}

fn train_dqn_impl(
    _initial_model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    mut metrics: impl TrainingMetrics,
) -> DuelingQNetwork {
    println!("\n[DQN] Initializing Double Dueling DQN Training...");

    let mut policy_net = DuelingQNetwork::new_with_config(config, rng.next_u64());
    let mut target_net = DuelingQNetwork::new_with_config(config, rng.next_u64());
    target_net.load_state_dict(&policy_net); // Sync weights

    #[cfg(cuda)]
    policy_net.to_cuda();
    #[cfg(cuda)]
    {
        target_net.to_cuda();
    }
    let params = policy_net.parameters();
    #[cfg(cuda)]
    let mut optimizer = GpuAdam::new(params, LEARNING_RATE)
        .map(Optimizer::Gpu)
        .unwrap_or_else(|| Optimizer::Cpu(Adam::new(policy_net.parameters(), LEARNING_RATE)));
    #[cfg(not(cuda))]
    let mut optimizer = Optimizer::Cpu(Adam::new(params, LEARNING_RATE));
    let mut replay_buffer = ReplayBuffer::new(BUFFER_CAPACITY);

    let total_steps = if config.fast_init { 5_000 } else { 50_000 };
    let mut epsilon = EPSILON_START;

    let mut state_struct = PullState {
        pity_6: 0,
        total_pulls_in_pool: 0,
        has_obtained_up: false,
        streak_4_star: 0,
        loss_streak: 0,
    };
    let (mut env_noise, mut env_bias) = env_net_env(env_net, rng, 0, 0, 0, 0);
    let mut pulls_done = 0;

    let mut episode_reward = 0.0;
    let mut episode_count = 0;
    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(51);

    let beta_anneal_steps = total_steps as f64;
    let snapshot_every = (total_steps / 200).max(1);
    let mut last_train_loss = 0.0_f64;
    let mut pending_train_loss: Option<Tensor> = None;

    // Pre-allocated scratch buffers to avoid per-step heap allocations
    let mut scratch = DqnTrainerScratch::new();
    let ones_5_1 = Tensor::new_f32(ONES_5_1_DATA.to_vec(), vec![5, 1]);
    #[cfg(cuda)]
    let ones_5_1 = match ones_5_1.to_cuda() {
        Ok(t) => t,
        Err(_) => ones_5_1,
    };

    let pb = create_bar(total_steps as u64, "DQN Training");

    for step in 0..total_steps {
        if step <= 5 || step % 10 == 0 {
            // silence
        }
        // 1. Build State
        let current_state_raw = build_features(
            state_struct.pity_6,
            pulls_done,
            env_noise,
            state_struct.streak_4_star,
            env_bias,
            state_struct.loss_streak,
            config,
        )
        .to_vec();

        let current_state_tensor = Tensor::new_f32(current_state_raw.clone(), vec![1, DIM]);
        #[cfg(cuda)]
        let current_state_tensor = match current_state_tensor.to_cuda() {
            Ok(t) => t,
            Err(_) => current_state_tensor,
        };

        // 2. Select Action
        let action = if rng.next_f64() < epsilon {
            rng.next_u64_bounded(ACTION_SPACE as u64) as usize
        } else {
            let q_values = policy_net.forward(&current_state_tensor);
            #[cfg(cuda)]
            if let Some(max_idx) = argmax_cuda_index(&q_values) {
                max_idx
            } else {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0;
                let q_data = q_values.data_as_f64_vec();
                for (i, &val) in q_data.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                max_idx
            }
            #[cfg(not(cuda))]
            {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0;
                let q_data = q_values.data_as_f64_vec();
                for (i, &val) in q_data.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                max_idx
            }
        };

        // 3. Step Environment
        let luck_modifier = ACTIONS[action];
        let base_prob_6 = prob_6(state_struct.pity_6, config);
        let final_prob_6 = (base_prob_6 + luck_modifier).clamp(0.0, 1.0);

        let r = rng.next_f64();
        let mut is_six = false;
        let mut is_up = false;

        state_struct.pity_6 += 1;
        state_struct.total_pulls_in_pool += 1;

        let big_pity_gate = if config.big_pity_requires_not_up {
            !state_struct.has_obtained_up
        } else {
            true
        };
        #[allow(clippy::if_same_then_else)]
        if config.up_pity_soft > 0
            && state_struct.total_pulls_in_pool == config.up_pity_soft
            && big_pity_gate
        {
            is_six = true;
            is_up = true;
            state_struct.pity_6 = 0;
            state_struct.streak_4_star = 0;
            state_struct.loss_streak = 0;
            state_struct.has_obtained_up = true;
        } else if config.big_pity_cumulative > 0
            && state_struct.total_pulls_in_pool == config.big_pity_cumulative
            && big_pity_gate
        {
            is_six = true;
            is_up = true;
            state_struct.pity_6 = 0;
            state_struct.streak_4_star = 0;
            state_struct.loss_streak = 0;
            state_struct.has_obtained_up = true;
        } else if r < final_prob_6 {
            is_six = true;
            state_struct.pity_6 = 0;
            state_struct.streak_4_star = 0;
            if config.up_rate > 0.0 && !config.up_six.is_empty() {
                if rng.next_f64() < config.up_rate {
                    is_up = true;
                    state_struct.loss_streak = 0;
                    state_struct.has_obtained_up = true;
                } else {
                    state_struct.loss_streak += 1;
                }
            }
        } else if config.always_5_star
            || (config.five_star_pity > 0
                && state_struct.streak_4_star >= config.five_star_pity - 1)
            || r < (final_prob_6 + config.prob_5_base).min(1.0)
        {
            state_struct.streak_4_star = 0;
        } else {
            state_struct.streak_4_star += 1;
        }
        pulls_done += 1;

        let reward = crate::utils::compute_reward_dqn(is_six, is_up, state_struct.loss_streak);

        episode_reward += reward;

        let next_state_raw = build_features(
            state_struct.pity_6,
            pulls_done,
            env_noise,
            state_struct.streak_4_star,
            env_bias,
            state_struct.loss_streak,
            config,
        )
        .to_vec();

        let done = is_up || pulls_done >= EPISODE_MAX_PULLS;

        replay_buffer.push(Experience {
            state: current_state_raw,
            action,
            reward,
            next_state: next_state_raw,
            done,
        });

        // 4. Train
        if replay_buffer.len() > BATCH_SIZE && step % TRAIN_FREQ == 0 {
            let beta = PER_BETA_START
                + (PER_BETA_END - PER_BETA_START) * (step as f64 / beta_anneal_steps);
            let start_train = std::time::Instant::now();
            let per_sample = replay_buffer.sample(rng, BATCH_SIZE, beta);
            let sample_time = start_train.elapsed();

            let start_forward = std::time::Instant::now();
            optimizer.zero_grad();
            #[cfg(cuda)]
            let (
                batch_state,
                batch_next_state,
                batch_mask,
                rewards_tensor,
                dones_tensor,
                is_weights_tensor,
            ) = if let Some(cuda_batch) = per_sample.cuda_batch.as_ref() {
                (
                    cuda_batch.states.clone(),
                    cuda_batch.next_states.clone(),
                    cuda_batch.action_mask.clone(),
                    cuda_batch.rewards.clone(),
                    cuda_batch.dones.clone(),
                    cuda_batch.is_weights.clone(),
                )
            } else {
                dqn_cpu_batch_tensors(&per_sample, &mut scratch)
            };
            #[cfg(not(cuda))]
            let (
                batch_state,
                batch_next_state,
                batch_mask,
                rewards_tensor,
                dones_tensor,
                is_weights_tensor,
            ) = dqn_cpu_batch_tensors(&per_sample, &mut scratch);

            // 2. Policy Forward
            let q_values = policy_net.forward(&batch_state); // (B, 5)

            // Select Action Q-Values: (B, 5) * (B, 5) -> (B, 5) [one non-zero per row]
            // Sum across dim 1 to get (B, 1)
            // MatMul by ones(5, 1) -> (B, 1)
            let q_actions = (&q_values * &batch_mask).matmul(&ones_5_1); // (B, 1)

            // 3. Compute Targets (Double DQN)
            #[cfg(cuda)]
            let target_tensor = if batch_next_state.device == crate::autograd::Device::Cuda
                && q_values.device == crate::autograd::Device::Cuda
                && crate::cuda::is_available()
            {
                let q_next_eval = policy_net.forward(&batch_next_state).detach(); // (B, 5)
                let q_next_target = target_net.forward(&batch_next_state).detach(); // (B, 5)
                double_dqn_target_cuda(
                    &q_next_eval,
                    &q_next_target,
                    &rewards_tensor,
                    &dones_tensor,
                    GAMMA,
                )
                .unwrap_or_else(|| {
                    double_dqn_target_inference_cpu(
                        &policy_net,
                        &target_net,
                        &batch_next_state,
                        &rewards_tensor,
                        &dones_tensor,
                        GAMMA,
                        &mut scratch.target_vals,
                    )
                })
            } else {
                double_dqn_target_inference_cpu(
                    &policy_net,
                    &target_net,
                    &batch_next_state,
                    &rewards_tensor,
                    &dones_tensor,
                    GAMMA,
                    &mut scratch.target_vals,
                )
            };
            #[cfg(not(cuda))]
            let target_tensor = double_dqn_target_inference_cpu(
                &policy_net,
                &target_net,
                &batch_next_state,
                &rewards_tensor,
                &dones_tensor,
                GAMMA,
                &mut scratch.target_vals,
            );

            // IS-weighted loss: w_i * (q - target)^2, normalized
            let mut loss = q_actions.weighted_mse_loss(&target_tensor, &is_weights_tensor);
            if let Some(reg) = policy_net.achf_orthogonal_penalty() {
                loss = loss + reg;
            }

            pending_train_loss = Some(loss.detach());
            let forward_time = start_forward.elapsed();

            let start_backward = std::time::Instant::now();
            loss.backward();
            policy_net.update_achf_after_backward();
            let backward_time = start_backward.elapsed();

            let start_opt = std::time::Instant::now();
            optimizer.step();
            let opt_time = start_opt.elapsed();

            // Write back per-sample TD errors for priority update
            {
                #[cfg(cuda)]
                {
                    if let Some(cuda_batch) = per_sample.cuda_batch.as_ref() {
                        let td_tensor = abs_diff_cuda(&q_actions, &target_tensor);
                        if let Some(td_tensor) = td_tensor {
                            let _ = replay_buffer.update_priorities_cuda(
                                &cuda_batch.indices,
                                &td_tensor,
                                BATCH_SIZE,
                            );
                        }
                    } else {
                        let td_tensor = abs_diff_cuda(&q_actions, &target_tensor);
                        let td_errors = if let Some(td_tensor) = td_tensor {
                            td_tensor.data_as_f64_vec()
                        } else {
                            let q_data = q_actions.data_as_f64_vec();
                            let t_data = target_tensor.data_as_f64_vec();
                            q_data
                                .iter()
                                .zip(t_data.iter())
                                .map(|(&q, &t)| (q - t).abs())
                                .collect()
                        };
                        scratch.td_errors.clear();
                        scratch.td_errors.extend(td_errors);
                        replay_buffer.update_priorities(&per_sample.indices, &scratch.td_errors);
                    }
                }
                #[cfg(not(cuda))]
                {
                    let q_data = q_actions.data_as_f64_vec();
                    let t_data = target_tensor.data_as_f64_vec();
                    let td_errors: Vec<f64> = q_data
                        .iter()
                        .zip(t_data.iter())
                        .map(|(&q, &t)| (q - t).abs())
                        .collect();
                    scratch.td_errors.clear();
                    scratch.td_errors.extend(td_errors);
                    replay_buffer.update_priorities(&per_sample.indices, &scratch.td_errors);
                }
            }

            // Soft Update Target Network
            target_net.soft_update(&policy_net, 0.005);

            if step % LOG_FREQ == 0 {
                println!(
                    "[Perf] Step {}: Sample={:?} Fwd={:?} Bwd={:?} Opt={:?}",
                    step, sample_time, forward_time, backward_time, opt_time
                );
            }
        }

        // Removed hard update logic (step % TARGET_UPDATE_FREQ == 0)
        // if step % TARGET_UPDATE_FREQ == 0 { ... }

        if epsilon > EPSILON_END {
            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY as f64;
        }

        if done {
            episode_count += 1;
            recent_rewards.push_back(episode_reward);
            if recent_rewards.len() > 50 {
                recent_rewards.pop_front();
            }

            state_struct = PullState {
                pity_6: 0,
                total_pulls_in_pool: 0,
                has_obtained_up: false,
                streak_4_star: 0,
                loss_streak: 0,
            };
            let new_env = env_net_env(env_net, rng, 0, 0, 0, 0);
            env_noise = new_env.0;
            env_bias = new_env.1;
            pulls_done = 0;
            episode_reward = 0.0;
        }

        if step % LOG_FREQ == 0 {
            let avg_r = if recent_rewards.is_empty() {
                0.0
            } else {
                recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64
            };
            pb.set_position(step as u64);
            pb.set_message(format!(
                "Ep: {} | Avg R: {:.2} | Eps: {:.3}",
                episode_count, avg_r, epsilon
            ));
        }

        if metrics.is_enabled() && step % snapshot_every == 0 {
            if let Some(loss) = pending_train_loss.take() {
                last_train_loss = loss.item() as f64;
            }
            let avg_r = if recent_rewards.is_empty() {
                0.0
            } else {
                recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64
            };
            metrics.emit_achf_snapshot(step, last_train_loss, avg_r, policy_net.snapshot_achf());
        }

        if config.achf.cache_log_interval_steps > 0
            && step % config.achf.cache_log_interval_steps == 0
        {
            if let Some(stats) = policy_net.achf_cache_stats() {
                if stats.calls > 0 {
                    println!("\n{}", crate::utils::format_achf_stats(&stats));
                }
            }
        }
    }
    pb.finish_with_message("DQN Training Complete.");
    policy_net.prune_achf(config.achf.prune_threshold);
    policy_net.freeze_achf_for_inference();
    policy_net
}

/// Train a DQN agent with optional metrics collection for benchmarking.
pub fn train_dqn_with_metrics(
    initial_model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    metrics_tx: Option<std::sync::mpsc::Sender<StepSnapshot>>,
) -> DuelingQNetwork {
    train_dqn_impl(
        initial_model,
        rng,
        env_net,
        config,
        TrainingMetricsSink::from(metrics_tx),
    )
}

/// Scratch buffers for DQN training to avoid per-step heap allocations.
struct DqnTrainerScratch {
    states_vec: Vec<f64>,
    next_states_vec: Vec<f64>,
    actions_vec: Vec<f64>,
    rewards_vec: Vec<f64>,
    dones_vec: Vec<f64>,
    target_vals: Vec<f64>,
    td_errors: Vec<f64>,
}

impl DqnTrainerScratch {
    fn new() -> Self {
        Self {
            states_vec: Vec::with_capacity(BATCH_SIZE * DIM),
            next_states_vec: Vec::with_capacity(BATCH_SIZE * DIM),
            actions_vec: Vec::with_capacity(BATCH_SIZE * ACTION_SPACE),
            rewards_vec: Vec::with_capacity(BATCH_SIZE),
            dones_vec: Vec::with_capacity(BATCH_SIZE),
            target_vals: Vec::with_capacity(BATCH_SIZE),
            td_errors: Vec::with_capacity(BATCH_SIZE),
        }
    }

    fn reset(&mut self) {
        self.states_vec = Vec::with_capacity(BATCH_SIZE * DIM);
        self.next_states_vec = Vec::with_capacity(BATCH_SIZE * DIM);
        self.actions_vec = Vec::with_capacity(BATCH_SIZE * ACTION_SPACE);
        self.rewards_vec = Vec::with_capacity(BATCH_SIZE);
        self.dones_vec = Vec::with_capacity(BATCH_SIZE);
        self.target_vals = Vec::with_capacity(BATCH_SIZE);
        self.td_errors = Vec::with_capacity(BATCH_SIZE);
    }
}

/// Incremental DQN trainer for online learning during interactive mode.
pub struct OnlineDqnTrainer {
    policy: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: Optimizer,
    replay_buffer: ReplayBuffer,
    steps_done: usize,
    scratch: DqnTrainerScratch,
}

impl OnlineDqnTrainer {
    pub fn from_policy(policy: DuelingQNetwork, seed: u64) -> Self {
        let achf = policy.achf_config();
        let mut target = DuelingQNetwork::new(seed, &achf);
        target.load_state_dict(&policy);
        #[cfg(cuda)]
        let policy = {
            let mut p = policy;
            p.to_cuda();
            p
        };
        #[cfg(cuda)]
        {
            target.to_cuda();
        }
        let params = policy.parameters();
        #[cfg(cuda)]
        let optimizer = GpuAdam::new(params, LEARNING_RATE)
            .map(Optimizer::Gpu)
            .unwrap_or_else(|| Optimizer::Cpu(Adam::new(policy.parameters(), LEARNING_RATE)));
        #[cfg(not(cuda))]
        let optimizer = Optimizer::Cpu(Adam::new(params, LEARNING_RATE));
        Self {
            policy,
            target,
            optimizer,
            replay_buffer: ReplayBuffer::new(BUFFER_CAPACITY),
            steps_done: 0,
            scratch: DqnTrainerScratch::new(),
        }
    }

    pub fn push(&mut self, exp: Experience) {
        self.replay_buffer.push(exp);
    }

    pub fn train_step(&mut self, rng: &mut Rng) -> bool {
        if self.replay_buffer.len() < BATCH_SIZE {
            return false;
        }
        // Beta anneals linearly from PER_BETA_START toward PER_BETA_END
        let beta = (PER_BETA_START
            + (PER_BETA_END - PER_BETA_START) * (self.steps_done as f64 / EPSILON_DECAY as f64))
            .min(PER_BETA_END);
        let per_sample = self.replay_buffer.sample(rng, BATCH_SIZE, beta);
        self.optimizer.zero_grad();

        #[cfg(cuda)]
        let (
            batch_state,
            batch_next_state,
            batch_mask,
            rewards_tensor,
            dones_tensor,
            is_weights_tensor,
        ) = if let Some(cuda_batch) = per_sample.cuda_batch.as_ref() {
            (
                cuda_batch.states.clone(),
                cuda_batch.next_states.clone(),
                cuda_batch.action_mask.clone(),
                cuda_batch.rewards.clone(),
                cuda_batch.dones.clone(),
                cuda_batch.is_weights.clone(),
            )
        } else {
            dqn_cpu_batch_tensors(&per_sample, &mut self.scratch)
        };
        #[cfg(not(cuda))]
        let (
            batch_state,
            batch_next_state,
            batch_mask,
            rewards_tensor,
            dones_tensor,
            is_weights_tensor,
        ) = dqn_cpu_batch_tensors(&per_sample, &mut self.scratch);

        let q_values = self.policy.forward(&batch_state);
        let ones_5_1 = Tensor::new_f32(ONES_5_1_DATA.to_vec(), vec![5, 1]);
        #[cfg(cuda)]
        let ones_5_1 = match ones_5_1.to_cuda() {
            Ok(t) => t,
            Err(_) => ones_5_1,
        };
        let q_actions = (&q_values * &batch_mask).matmul(&ones_5_1);

        #[cfg(cuda)]
        let target_tensor = if batch_next_state.device == crate::autograd::Device::Cuda
            && q_values.device == crate::autograd::Device::Cuda
            && crate::cuda::is_available()
        {
            let q_next_eval = self.policy.forward(&batch_next_state).detach();
            let q_next_target = self.target.forward(&batch_next_state).detach();
            double_dqn_target_cuda(
                &q_next_eval,
                &q_next_target,
                &rewards_tensor,
                &dones_tensor,
                GAMMA,
            )
            .unwrap_or_else(|| {
                double_dqn_target_inference_cpu(
                    &self.policy,
                    &self.target,
                    &batch_next_state,
                    &rewards_tensor,
                    &dones_tensor,
                    GAMMA,
                    &mut self.scratch.target_vals,
                )
            })
        } else {
            double_dqn_target_inference_cpu(
                &self.policy,
                &self.target,
                &batch_next_state,
                &rewards_tensor,
                &dones_tensor,
                GAMMA,
                &mut self.scratch.target_vals,
            )
        };
        #[cfg(not(cuda))]
        let target_tensor = double_dqn_target_inference_cpu(
            &self.policy,
            &self.target,
            &batch_next_state,
            &rewards_tensor,
            &dones_tensor,
            GAMMA,
            &mut self.scratch.target_vals,
        );
        let mut loss = q_actions.weighted_mse_loss(&target_tensor, &is_weights_tensor);
        if let Some(reg) = self.policy.achf_orthogonal_penalty() {
            loss = loss + reg;
        }
        loss.backward();
        self.policy.update_achf_after_backward();
        self.optimizer.step();

        // Write back per-sample TD errors for priority update
        {
            #[cfg(cuda)]
            {
                if let Some(cuda_batch) = per_sample.cuda_batch.as_ref() {
                    if let Some(td_tensor) = abs_diff_cuda(&q_actions, &target_tensor) {
                        let _ = self.replay_buffer.update_priorities_cuda(
                            &cuda_batch.indices,
                            &td_tensor,
                            BATCH_SIZE,
                        );
                    }
                } else {
                    let td_tensor = abs_diff_cuda(&q_actions, &target_tensor);
                    let td_errors: Vec<f64> = if let Some(td_tensor) = td_tensor {
                        td_tensor.data_as_f64_vec()
                    } else {
                        let q_data = q_actions.data_as_f64_vec();
                        let t_data = target_tensor.data_as_f64_vec();
                        q_data
                            .iter()
                            .zip(t_data.iter())
                            .map(|(&q, &t)| (q - t).abs())
                            .collect()
                    };
                    self.replay_buffer
                        .update_priorities(&per_sample.indices, &td_errors);
                }
            }
            #[cfg(not(cuda))]
            {
                let q_data = q_actions.data_as_f64_vec();
                let t_data = target_tensor.data_as_f64_vec();
                let td_errors: Vec<f64> = q_data
                    .iter()
                    .zip(t_data.iter())
                    .map(|(&q, &t)| (q - t).abs())
                    .collect();
                self.replay_buffer
                    .update_priorities(&per_sample.indices, &td_errors);
            }
        }

        self.target.soft_update(&self.policy, 0.005);
        self.steps_done += 1;
        true
    }

    pub fn sync_to(&self, shared: &std::sync::RwLock<DuelingQNetwork>) {
        for attempt in 0..3u64 {
            if let Ok(mut guard) = shared.try_write() {
                guard.load_state_dict(&self.policy);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1 + attempt));
        }
        if let Ok(mut guard) = shared.write() {
            guard.load_state_dict(&self.policy);
        }
    }

    pub fn policy(&self) -> &DuelingQNetwork {
        &self.policy
    }

    pub fn steps_done(&self) -> usize {
        self.steps_done
    }

    pub fn buffer_len(&self) -> usize {
        self.replay_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_dqn_config() -> Config {
        Config {
            model_hidden_dim: 16,
            model_num_layers: 1,
            achf: AchfConfig {
                enabled: false,
                ..AchfConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn dqn_batch_inference_matches_autograd_forward_without_graph() {
        let config = small_dqn_config();
        let dqn = DuelingQNetwork::new_with_config(&config, 123);
        let values: Vec<f64> = (0..2 * DIM)
            .map(|i| i as f64 / (2 * DIM) as f64 - 0.25)
            .collect();
        let states = Tensor::new_f32(values, vec![2, DIM]);

        let fast = dqn.forward_inference_batch(&states);
        let slow = dqn.forward(&states).detach();

        assert!(fast._ctx.is_none());
        assert_eq!(fast.shape, vec![2, ACTION_SPACE]);
        let fast_data = fast.data_to_f32_vec();
        let slow_data = slow.data_to_f32_vec();
        assert_eq!(fast_data.len(), slow_data.len());
        for (idx, (&actual, &expected)) in fast_data.iter().zip(slow_data.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "idx {idx}: inference={actual}, autograd={expected}"
            );
        }
    }

    #[test]
    fn double_dqn_target_inference_matches_tensor_target_without_graph() {
        let config = small_dqn_config();
        let policy = DuelingQNetwork::new_with_config(&config, 7);
        let target = DuelingQNetwork::new_with_config(&config, 11);
        let batch = 3;
        let values: Vec<f64> = (0..batch * DIM)
            .map(|i| ((i % DIM) as f64 - 8.0) / 32.0)
            .collect();
        let next_states = Tensor::new_f32(values, vec![batch, DIM]);
        let rewards = Tensor::new_f32(vec![1.0, -0.5, 0.25], vec![batch, 1]);
        let dones = Tensor::new_f32(vec![0.0, 1.0, 0.0], vec![batch, 1]);

        let q_next_eval = policy.forward(&next_states).detach();
        let q_next_target = target.forward(&next_states).detach();
        let mut expected_vals = Vec::new();
        let expected = double_dqn_target_cpu(
            &q_next_eval,
            &q_next_target,
            &rewards,
            &dones,
            GAMMA,
            &mut expected_vals,
        );

        let mut actual_vals = Vec::new();
        let actual = double_dqn_target_inference_cpu(
            &policy,
            &target,
            &next_states,
            &rewards,
            &dones,
            GAMMA,
            &mut actual_vals,
        );

        assert!(actual._ctx.is_none());
        let actual_data = actual.data_to_f32_vec();
        let expected_data = expected.data_to_f32_vec();
        for (idx, (&actual, &expected)) in actual_data.iter().zip(expected_data.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-4,
                "idx {idx}: inference_target={actual}, tensor_target={expected}"
            );
        }
    }
}
