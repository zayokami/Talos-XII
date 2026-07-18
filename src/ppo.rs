use crate::autograd::Tensor;
use crate::config::{AchfConfig, ComputeDevice, Config};
use crate::env_net::EnvNet;
use crate::gacha_env::{step_pull, GachaAction};
use crate::neural::DIM;
use crate::nn::{Linear, Module};
use crate::policy_eval::{evaluate_ppo_policy, format_policy_eval};
use crate::rng::Rng;
use crate::sim::{build_features_with_luck_budget, env_net_env, PpoExperience, PullState};
use crate::training_error::TrainingError;
use crate::training_metrics::{StepSnapshot, TrainingMetrics, TrainingMetricsSink};
use crate::transformer::{KVCache, LuckTransformer};
use crate::utils::{
    compute_reward_ppo_breakdown, create_bar, normalize_slice, sum_f64, sum_sq_diff,
    RewardBreakdown, ACTIONS, ACTION_SPACE, EPISODE_MAX_PULLS,
};
use crate::worker::GoodJobWorker;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// --- PPO Components ---

const CLIP_EPSILON: f64 = 0.2;
const GAMMA: f64 = 0.99;
const GAE_LAMBDA: f64 = 0.95;
const KL_CHECK_INTERVAL: usize = 4;
const VALUE_COEF: f64 = 0.5;
const ENTROPY_COEF: f64 = 0.01;
pub(crate) const EARLY_UP_BONUS_THRESHOLD_1: usize = 80;
pub(crate) const EARLY_UP_BONUS_THRESHOLD_2: usize = 50;

#[derive(Default)]
struct CachedStepScratch {
    last: Vec<f32>,
    logits: Vec<f32>,
    value: Vec<f32>,
}

thread_local! {
    static CACHED_STEP_SCRATCH: RefCell<CachedStepScratch> =
        RefCell::new(CachedStepScratch::default());
}

// Softmax + categorical sample over fixed ACTION_SPACE logits.
// Returns (action_index, log_prob_of_action).
// If top_k > 0, only the top_k logits are kept (others set to -inf).
#[inline]
fn softmax_probs(logits: &[f32], top_k: usize) -> [f32; ACTION_SPACE] {
    // Find max for numerical stability
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_l {
            max_l = v;
        }
    }

    // Top-k truncation: set non-top-k logits to -inf
    let mut probs = [0.0f32; ACTION_SPACE];
    if top_k > 0 && top_k < ACTION_SPACE {
        // Find top_k values
        let mut sorted_logits = logits.to_vec();
        sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted_logits[top_k.saturating_sub(1)];

        // Compute softmax with top-k masking
        let mut sum_exp = 0.0f32;
        for (i, prob) in probs.iter_mut().enumerate() {
            if logits[i] < threshold {
                *prob = 0.0f32; // masked out
            } else {
                *prob = (logits[i] - max_l).exp();
                sum_exp += *prob;
            }
        }
        if sum_exp > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum_exp;
            }
        } else {
            let uniform = 1.0 / ACTION_SPACE as f32;
            probs.fill(uniform);
        }
    } else {
        // Full softmax (top_k disabled or >= ACTION_SPACE)
        let mut sum_exp = 0.0f32;
        for (i, prob) in probs.iter_mut().enumerate() {
            *prob = (logits[i] - max_l).exp();
            sum_exp += *prob;
        }
        for prob in probs.iter_mut() {
            *prob /= sum_exp;
        }
    }

    probs
}

#[inline]
fn softmax_sample(logits: &[f32], top_k: usize) -> (usize, f32) {
    if logits.len() != ACTION_SPACE {
        let fallback_prob = 1.0 / ACTION_SPACE as f32;
        return (0, fallback_prob.ln());
    }

    let probs = softmax_probs(logits, top_k);

    // Categorical sampling
    let mut r = rand::random::<f32>();
    let mut idx = ACTION_SPACE - 1;
    for (i, &p) in probs.iter().enumerate() {
        if r < p {
            idx = i;
            break;
        }
        r -= p;
    }
    let log_prob = probs[idx].max(f32::MIN_POSITIVE).ln();
    (idx, log_prob)
}

#[inline]
fn argmax_action(logits: &[f32]) -> usize {
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &value) in logits.iter().enumerate().take(ACTION_SPACE) {
        if value > max_val {
            max_val = value;
            max_idx = idx;
        }
    }
    max_idx
}

/// Actor-Critic model combining a LuckTransformer backbone with action/value heads.
#[derive(Clone, Serialize, Deserialize)]
pub struct ActorCritic {
    pub backbone: LuckTransformer,
    pub actor_head: Linear,
    pub critic_head: Linear,
}

impl ActorCritic {
    pub fn uses_cuda(&self) -> bool {
        #[cfg(cuda)]
        {
            self.parameters()
                .first()
                .is_some_and(|parameter| parameter.device == crate::autograd::Device::Cuda)
        }
        #[cfg(not(cuda))]
        {
            false
        }
    }

    pub fn new_with_config(config: &Config, seed: u64) -> Self {
        let backbone = LuckTransformer::new_with_config(config, seed);
        let actor_head = Linear::new(
            config.model_hidden_dim,
            ACTION_SPACE,
            true,
            seed.wrapping_add(100),
        );
        let critic_head = Linear::new(config.model_hidden_dim, 1, true, seed.wrapping_add(200));

        ActorCritic {
            backbone,
            actor_head,
            critic_head,
        }
    }

    /// Test-friendly constructor with configurable size.
    /// Production now uses a compact default; tests can still override size explicitly.
    #[allow(dead_code)]
    pub fn new(seed: u64, achf: &AchfConfig, hidden_dim: usize, num_layers: usize) -> Self {
        let backbone = LuckTransformer::new_compat(DIM, hidden_dim, true, num_layers, seed, achf);
        let actor_head = Linear::new(hidden_dim, ACTION_SPACE, true, seed.wrapping_add(100));
        let critic_head = Linear::new(hidden_dim, 1, true, seed.wrapping_add(200));

        ActorCritic {
            backbone,
            actor_head,
            critic_head,
        }
    }

    fn forward_backbone(&self, state: &Tensor, pity: &[usize]) -> Tensor {
        let x = if state.shape.len() == 1 {
            state.reshape(vec![1, 1, state.shape[0]])
        } else if state.shape.len() == 2 {
            state.reshape(vec![1, state.shape[0], state.shape[1]])
        } else {
            state.clone()
        };
        let seq = self.backbone.forward(&x, pity);
        self.backbone.last_token(&seq)
    }

    fn squeeze_head(t: Tensor) -> Tensor {
        if t.shape.len() == 2 && t.shape[0] == 1 {
            t.reshape(vec![t.shape[1]])
        } else {
            t
        }
    }

    #[allow(dead_code)]
    pub fn forward_actor(&self, state: &Tensor, pity: &[usize]) -> Tensor {
        let last = self.forward_backbone(state, pity);
        Self::squeeze_head(self.actor_head.forward(&last))
    }

    #[allow(dead_code)]
    pub fn forward_critic(&self, state: &Tensor, pity: &[usize]) -> Tensor {
        let last = self.forward_backbone(state, pity);
        Self::squeeze_head(self.critic_head.forward(&last))
    }

    pub fn forward_actor_critic(&self, state: &Tensor, pity: &[usize]) -> (Tensor, Tensor) {
        let last = self.forward_backbone(state, pity);
        let logits = Self::squeeze_head(self.actor_head.forward(&last));
        let value = Self::squeeze_head(self.critic_head.forward(&last));
        (logits, value)
    }

    /// Batched forward for training efficiency.
    /// `states` should be `[Batch, Seq, Dim]`.
    pub fn forward_actor_critic_batch(&self, states: &Tensor) -> (Tensor, Tensor) {
        let seq = self.backbone.forward(states, &[]);
        let last = self.backbone.last_token(&seq);
        let logits = self.actor_head.forward(&last);
        let value = self.critic_head.forward(&last);
        (logits, value)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.backbone.parameters();
        p.extend(self.actor_head.parameters());
        p.extend(self.critic_head.parameters());
        p
    }

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        self.backbone.to_cuda();
        self.actor_head.to_cuda();
        self.critic_head.to_cuda();
    }

    #[cfg(cuda)]
    pub fn try_to_cuda(&mut self) -> crate::cuda::error::CudaResult<()> {
        use crate::autograd::Device;
        use crate::cuda::error::CudaError;

        crate::cuda::init()?;
        let mut candidate = self.clone();
        candidate.to_cuda();
        if candidate
            .parameters()
            .iter()
            .any(|parameter| parameter.device != Device::Cuda)
        {
            return Err(CudaError::InvalidInput {
                op: "ActorCritic::try_to_cuda",
                message: "one or more parameters failed to migrate to CUDA",
            });
        }
        *self = candidate;
        Ok(())
    }

    pub fn to_inference_bf16(&self) -> Self {
        Self {
            backbone: self.backbone.to_inference_bf16(),
            actor_head: self.actor_head.to_inference_bf16(),
            critic_head: self.critic_head.to_inference_bf16(),
        }
    }

    pub fn load_state_dict(&mut self, other: &Self) {
        self.backbone.load_state_dict(&other.backbone);
        self.actor_head.load_state_dict(&other.actor_head);
        self.critic_head.load_state_dict(&other.critic_head);
    }

    pub fn update_achf_after_backward(&self) {
        self.backbone.update_achf_after_backward();
    }

    pub fn refresh_achf_after_optimizer_step(&mut self) {
        self.backbone.refresh_achf_after_optimizer_step();
    }

    pub fn freeze_achf_for_inference(&mut self) {
        self.backbone.freeze_achf_for_inference();
    }

    pub fn begin_achf_candidate_calibration(&mut self) -> usize {
        self.backbone.begin_achf_candidate_calibration()
    }

    pub fn set_achf_candidate_calibration_training(&self) {
        self.backbone.set_achf_candidate_calibration_training();
    }

    pub fn set_achf_candidate_calibration_validation(&self) {
        self.backbone.set_achf_candidate_calibration_validation();
    }

    pub fn achf_candidate_calibration_parameters(&self) -> Vec<Tensor> {
        self.backbone.achf_candidate_calibration_parameters()
    }

    pub fn achf_candidate_calibration_parameter_masks(&self) -> Vec<Option<Vec<u8>>> {
        self.backbone.achf_candidate_calibration_parameter_masks()
    }

    pub fn achf_candidate_calibration_records(&self) -> Vec<crate::achf::AchfCandidateCalibration> {
        self.backbone.achf_candidate_calibration_records()
    }

    pub fn take_achf_candidate_calibration_loss(&self) -> Option<Tensor> {
        self.backbone.take_achf_candidate_calibration_loss()
    }

    pub fn enforce_achf_candidate_masks(&mut self) {
        self.backbone.enforce_achf_candidate_masks();
    }

    pub fn record_achf_candidate_calibration_checkpoint(&mut self, step: usize) {
        self.backbone
            .record_achf_candidate_calibration_checkpoint(step);
    }

    pub fn finalize_achf_candidate_calibration(
        &mut self,
        steps: usize,
        masked_moment_max_abs: f64,
    ) {
        self.backbone
            .finalize_achf_candidate_calibration(steps, masked_moment_max_abs);
    }

    pub fn achf_orthogonal_penalty(&self) -> Option<Tensor> {
        self.backbone.achf_orthogonal_penalty()
    }

    pub fn achf_config(&self) -> Option<AchfConfig> {
        self.backbone.blocks.iter().find_map(|block| {
            block
                .achf_ffn
                .as_ref()
                .or(block.mla_layer.achf_wo.as_ref())
                .map(|achf| achf.config.clone())
        })
    }

    // Returns (action_idx, log_prob, value)
    pub fn step(&self, state: &Tensor, pity: &[usize], top_k: usize) -> (usize, f64, f64) {
        let (logits, value) = self.forward_actor_critic(state, pity);
        let logits_data = logits.data_as_f64_vec();
        let logits_f32: Vec<f32> = logits_data.iter().map(|&v| v as f32).collect();
        let (action_idx, log_prob) = softmax_sample(&logits_f32, top_k);
        let val = value.data_as_f64_vec()[0];
        (action_idx, log_prob as f64, val)
    }

    // Fast inference without Autograd graph
    pub fn step_inference(&self, state: &[f32], top_k: usize) -> usize {
        let seq = self.backbone.forward_inference(state);
        let last = self.backbone.last_token_inference(&seq);
        let logits = self.actor_head.forward_inference(&last);
        softmax_sample(&logits, top_k).0
    }

    pub fn step_inference_cached_with_value(
        &self,
        state: &[f32],
        kv_cache: &mut [KVCache],
        start_pos: usize,
        top_k: usize,
    ) -> (usize, f32, f32) {
        CACHED_STEP_SCRATCH.with(|scratch_cell| {
            let mut scratch = scratch_cell.borrow_mut();
            let CachedStepScratch {
                last,
                logits,
                value,
            } = &mut *scratch;
            self.backbone
                .forward_inference_step_into(state, kv_cache, start_pos, last);
            self.actor_head.forward_inference_into(last, logits);
            self.critic_head.forward_inference_into(last, value);
            let (action_idx, log_prob) = softmax_sample(logits, top_k);
            (action_idx, log_prob, value[0])
        })
    }

    fn step_sequence_with_value(
        &self,
        state: &[f64],
        seq_len: usize,
        top_k: usize,
    ) -> (usize, f32, f32) {
        if state.len() != seq_len.saturating_mul(DIM) || seq_len == 0 {
            let fallback_prob = 1.0 / ACTION_SPACE as f32;
            return (0, fallback_prob.ln(), 0.0);
        }
        let seq_f32: Vec<f32> = state.iter().map(|&v| v as f32).collect();
        let backbone_out = self.backbone.forward_inference(&seq_f32);
        let last = self.backbone.last_token_inference(&backbone_out);
        let logits = self.actor_head.forward_inference(&last);
        let value = self.critic_head.forward_inference(&last);
        let (action_idx, log_prob) = softmax_sample(&logits, top_k);
        (action_idx, log_prob, value.first().copied().unwrap_or(0.0))
    }

    pub fn step_inference_cached(
        &self,
        state: &[f32],
        kv_cache: &mut [KVCache],
        start_pos: usize,
        top_k: usize,
    ) -> usize {
        CACHED_STEP_SCRATCH.with(|scratch_cell| {
            let mut scratch = scratch_cell.borrow_mut();
            let CachedStepScratch {
                last,
                logits,
                value: _,
            } = &mut *scratch;
            self.backbone
                .forward_inference_step_into(state, kv_cache, start_pos, last);
            self.actor_head.forward_inference_into(last, logits);
            softmax_sample(logits, top_k).0
        })
    }

    pub(crate) fn step_inference_cached_greedy(
        &self,
        state: &[f32],
        kv_cache: &mut [KVCache],
        start_pos: usize,
    ) -> usize {
        CACHED_STEP_SCRATCH.with(|scratch_cell| {
            let mut scratch = scratch_cell.borrow_mut();
            let CachedStepScratch {
                last,
                logits,
                value: _,
            } = &mut *scratch;
            self.backbone
                .forward_inference_step_into(state, kv_cache, start_pos, last);
            self.actor_head.forward_inference_into(last, logits);
            argmax_action(logits)
        })
    }

    pub fn prune_cache(&self, kv_cache: &mut [KVCache], max_len: usize) {
        self.backbone.prune_kv_cache(kv_cache, max_len);
    }

    pub fn achf_cache_stats_iter(&self) -> impl Iterator<Item = crate::achf::AchfCacheStats> + '_ {
        self.backbone.achf_cache_stats_iter()
    }

    pub fn achf_cache_stats_aggregate(&self) -> crate::achf::AchfCacheStats {
        self.backbone.achf_cache_stats_aggregate()
    }

    pub fn achf_memory_stats_aggregate(&self) -> crate::achf::AchfMemoryStats {
        self.backbone.achf_memory_stats_aggregate()
    }

    pub fn fork_inference_runtime(&self) -> Self {
        let mut out = self.clone();
        out.backbone = self.backbone.fork_inference_runtime();
        out
    }

    pub fn set_achf_inference_mode(&mut self, mode: &str, sample_every: u64) {
        self.backbone.set_achf_inference_mode(mode, sample_every);
    }

    pub fn rebuild_achf_inference_candidates_target(&mut self, target_sparsity: f64) {
        self.backbone
            .rebuild_achf_inference_candidates_target(target_sparsity);
    }

    pub fn disable_achf_runtime(&mut self) {
        self.backbone.disable_achf_runtime();
    }

    pub fn snapshot_achf(&self) -> Option<crate::achf::AchfStateSnapshot> {
        self.backbone.snapshot_achf()
    }

    /// First ACHF layer + its input width, for isolating operator latency.
    pub fn first_achf_layer(&self) -> Option<(&crate::achf::AchfLayer, usize)> {
        self.backbone.first_achf_layer()
    }

    pub fn param_count(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.shape.iter().product::<usize>())
            .sum()
    }
}

// --- Optimizer (Adam) ---

// --- Reward Normalization ---

struct RunningMeanStd {
    count: f64,
    mean: f64,
    var: f64,
}

impl RunningMeanStd {
    fn new() -> Self {
        RunningMeanStd {
            count: 1e-4, // Avoid division by zero
            mean: 0.0,
            var: 1.0,
        }
    }

    fn update(&mut self, x: f64) {
        let batch_mean = x;
        let batch_var = 0.0; // Single sample update for simplicity in this loop
        let batch_count = 1.0;

        let delta = batch_mean - self.mean;
        let tot_count = self.count + batch_count;

        let new_mean = self.mean + delta * batch_count / tot_count;
        let m_a = self.var * self.count;
        let m_b = batch_var * batch_count;
        let m_2 = m_a + m_b + delta.powi(2) * self.count * batch_count / tot_count;

        self.mean = new_mean;
        self.var = m_2 / tot_count;
        self.count = tot_count;
    }

    #[allow(dead_code)]
    fn normalize(&self, x: f64) -> f64 {
        (x - self.mean) / (self.var.sqrt() + 1e-8)
    }
}

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

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self) {
        self.t += 1;

        // Global gradient clipping via SIMD dot_product for L2 norm (f32)
        let mut total_norm_sq = 0.0f32;
        for param in &self.params {
            let grad = param.grad_to_f32_vec();
            total_norm_sq += crate::simd::dot_product_f32(&grad, &grad);
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
        let b1 = self.beta1 as f32;
        let b2 = self.beta2 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let wd = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = param.grad_to_f32_vec();
            if param.dtype == crate::dtype::Dtype::F64 {
                let mut data = param.data_write_f64();
                let m = &mut self.m[i];
                let v = &mut self.v[i];
                let len = data.len();
                let mut j = 0;
                while j + 4 <= len {
                    for k in j..j + 4 {
                        let g = grad[k] * clip;
                        m[k] = b1 * m[k] + (1.0 - b1) * g;
                        v[k] = b2 * v[k] + (1.0 - b2) * g * g;
                        let m_hat = m[k] / bc1;
                        let v_hat = v[k] / bc2;
                        data[k] -=
                            lr as f64 * (m_hat / (v_hat.sqrt() + eps) + wd * data[k] as f32) as f64;
                    }
                    j += 4;
                }
                while j < len {
                    let g = grad[j] * clip;
                    m[j] = b1 * m[j] + (1.0 - b1) * g;
                    v[j] = b2 * v[j] + (1.0 - b2) * g * g;
                    let m_hat = m[j] / bc1;
                    let v_hat = v[j] / bc2;
                    data[j] -=
                        lr as f64 * (m_hat / (v_hat.sqrt() + eps) + wd * data[j] as f32) as f64;
                    j += 1;
                }
            } else {
                let mut data = param.data_write_f32();
                let m = &mut self.m[i];
                let v = &mut self.v[i];
                let len = data.len();
                let mut j = 0;
                while j + 4 <= len {
                    for k in j..j + 4 {
                        let g = grad[k] * clip;
                        m[k] = b1 * m[k] + (1.0 - b1) * g;
                        v[k] = b2 * v[k] + (1.0 - b2) * g * g;
                        let m_hat = m[k] / bc1;
                        let v_hat = v[k] / bc2;
                        data[k] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * data[k]);
                    }
                    j += 4;
                }
                while j < len {
                    let g = grad[j] * clip;
                    m[j] = b1 * m[j] + (1.0 - b1) * g;
                    v[j] = b2 * v[j] + (1.0 - b2) * g * g;
                    let m_hat = m[j] / bc1;
                    let v_hat = v[j] / bc2;
                    data[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * data[j]);
                    j += 1;
                }
            }
        }
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn masked_moment_max_abs(&self, masks: &[Option<Vec<u8>>]) -> f64 {
        let mut maximum = 0.0f64;
        for ((first_moment, second_moment), mask) in
            self.m.iter().zip(self.v.iter()).zip(masks.iter())
        {
            let Some(mask) = mask else {
                continue;
            };
            for ((first, second), keep) in first_moment
                .iter()
                .zip(second_moment.iter())
                .zip(mask.iter())
            {
                if *keep == 0 {
                    if !first.is_finite() || !second.is_finite() {
                        return f64::INFINITY;
                    }
                    maximum = maximum.max(first.abs() as f64).max(second.abs() as f64);
                }
            }
        }
        maximum
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
    poisoned: bool,
}

#[cfg(cuda)]
impl GpuAdam {
    fn new(params: Vec<Tensor>, lr: f64) -> crate::cuda::error::CudaResult<Self> {
        use crate::cuda::error::CudaError;
        use crate::cuda::memory::CudaBuffer;
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            let len = p.cuda_storage_len();
            if p.device != crate::autograd::Device::Cuda {
                return Err(CudaError::InvalidInput {
                    op: "GpuAdam::new",
                    message: "all parameters must reside on CUDA",
                });
            }
            let (d_m, d_v) = match p.dtype {
                crate::dtype::Dtype::F32 => {
                    let dm = crate::cuda::memory::alloc::<f32>(len)?;
                    let dv = crate::cuda::memory::alloc::<f32>(len)?;
                    let zeros = vec![0.0f32; len];
                    crate::cuda::memory::copy_h2d(&dm, &zeros)?;
                    crate::cuda::memory::copy_h2d(&dv, &zeros)?;
                    (CudaBuffer::F32(dm), CudaBuffer::F32(dv))
                }
                crate::dtype::Dtype::F64 => {
                    let dm = crate::cuda::memory::alloc::<f64>(len)?;
                    let dv = crate::cuda::memory::alloc::<f64>(len)?;
                    let zeros = vec![0.0f64; len];
                    crate::cuda::memory::copy_h2d(&dm, &zeros)?;
                    crate::cuda::memory::copy_h2d(&dv, &zeros)?;
                    (CudaBuffer::F64(dm), CudaBuffer::F64(dv))
                }
                _ => {
                    return Err(CudaError::InvalidInput {
                        op: "GpuAdam::new",
                        message: "only f32/f64 parameters are supported",
                    });
                }
            };
            m.push(std::sync::Arc::new(d_m));
            v.push(std::sync::Arc::new(d_v));
        }
        Ok(GpuAdam {
            params,
            m,
            v,
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
            poisoned: false,
        })
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self) -> crate::cuda::error::CudaResult<()> {
        use crate::cuda::error::CudaError;
        use crate::cuda::memory::CudaBuffer;
        if self.poisoned {
            return Err(CudaError::InvalidInput {
                op: "GpuAdam::step",
                message: "optimizer is poisoned after a previous CUDA failure",
            });
        }
        self.t += 1;
        crate::cuda::record_optimizer_attempt();
        if let Err(error) = crate::autograd::cuda_clip_gradients_in_place(&self.params, 1.0, 1e-6) {
            self.poisoned = true;
            crate::cuda::record_optimizer_fallback();
            return Err(error);
        }

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in self.params.iter().enumerate() {
            let len = param.cuda_storage_len();
            if len == 0 {
                continue;
            }
            let d_params = match param.cuda_get_or_upload_buffer() {
                Ok(buffer) => buffer,
                Err((_, error)) => {
                    self.poisoned = true;
                    crate::cuda::record_optimizer_fallback();
                    return Err(error);
                }
            };
            let d_grads = match param.cuda_grad_get_or_upload_buffer() {
                Ok(buffer) => buffer,
                Err((_, error)) => {
                    self.poisoned = true;
                    crate::cuda::record_optimizer_fallback();
                    return Err(error);
                }
            };
            let d_m = self.m[i].clone();
            let d_v = self.v[i].clone();

            let step_result = match (param.dtype, &*d_params, &*d_grads, &*d_m, &*d_v) {
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
                ),
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
                ),
                _ => Err(CudaError::InvalidInput {
                    op: "GpuAdam::step",
                    message: "parameter/gradient/moment dtype mismatch",
                }),
            };
            if let Err(error) = step_result {
                self.poisoned = true;
                crate::cuda::record_optimizer_fallback();
                return Err(error);
            }
        }
        crate::cuda::record_optimizer_success();
        Ok(())
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn masked_moment_max_abs(&self, masks: &[Option<Vec<u8>>]) -> f64 {
        use crate::cuda::memory::{copy_d2h, CudaBuffer};
        let mut maximum = 0.0f64;
        for ((first_moment, second_moment), mask) in
            self.m.iter().zip(self.v.iter()).zip(masks.iter())
        {
            let Some(mask) = mask else {
                continue;
            };
            let (first, second): (Vec<f64>, Vec<f64>) =
                match (first_moment.as_ref(), second_moment.as_ref()) {
                    (CudaBuffer::F32(first), CudaBuffer::F32(second)) => {
                        let mut first_host = vec![0.0f32; first.len()];
                        let mut second_host = vec![0.0f32; second.len()];
                        if copy_d2h(&mut first_host, first).is_err()
                            || copy_d2h(&mut second_host, second).is_err()
                        {
                            return f64::INFINITY;
                        }
                        (
                            first_host.into_iter().map(f64::from).collect(),
                            second_host.into_iter().map(f64::from).collect(),
                        )
                    }
                    (CudaBuffer::F64(first), CudaBuffer::F64(second)) => {
                        let mut first_host = vec![0.0f64; first.len()];
                        let mut second_host = vec![0.0f64; second.len()];
                        if copy_d2h(&mut first_host, first).is_err()
                            || copy_d2h(&mut second_host, second).is_err()
                        {
                            return f64::INFINITY;
                        }
                        (first_host, second_host)
                    }
                    _ => return f64::INFINITY,
                };
            for ((first, second), keep) in first.iter().zip(second.iter()).zip(mask.iter()) {
                if *keep == 0 {
                    if !first.is_finite() || !second.is_finite() {
                        return f64::INFINITY;
                    }
                    maximum = maximum.max(first.abs()).max(second.abs());
                }
            }
        }
        maximum
    }
}

enum Optimizer {
    Cpu(Adam),
    #[cfg(cuda)]
    Gpu(GpuAdam),
}

impl Optimizer {
    fn new(params: Vec<Tensor>, lr: f64) -> Result<Self, String> {
        #[cfg(cuda)]
        {
            if params
                .iter()
                .any(|parameter| parameter.device == crate::autograd::Device::Cuda)
            {
                GpuAdam::new(params, lr)
                    .map(Optimizer::Gpu)
                    .map_err(|error| error.to_string())
            } else {
                Ok(Optimizer::Cpu(Adam::new(params, lr)))
            }
        }
        #[cfg(not(cuda))]
        {
            Ok(Optimizer::Cpu(Adam::new(params, lr)))
        }
    }

    fn set_lr(&mut self, lr: f64) {
        match self {
            Optimizer::Cpu(o) => o.set_lr(lr),
            #[cfg(cuda)]
            Optimizer::Gpu(o) => o.set_lr(lr),
        }
    }

    fn step(&mut self) -> Result<(), String> {
        match self {
            Optimizer::Cpu(o) => {
                o.step();
                Ok(())
            }
            #[cfg(cuda)]
            Optimizer::Gpu(o) => o.step().map_err(|error| error.to_string()),
        }
    }

    fn zero_grad(&self) {
        match self {
            Optimizer::Cpu(o) => o.zero_grad(),
            #[cfg(cuda)]
            Optimizer::Gpu(o) => o.zero_grad(),
        }
    }

    fn masked_moment_max_abs(&self, masks: &[Option<Vec<u8>>]) -> f64 {
        match self {
            Optimizer::Cpu(optimizer) => optimizer.masked_moment_max_abs(masks),
            #[cfg(cuda)]
            Optimizer::Gpu(optimizer) => optimizer.masked_moment_max_abs(masks),
        }
    }
}

// --- PPO Trainer ---

struct Memory {
    states_raw: Vec<Vec<f64>>,
    state_lens: Vec<usize>,
    pities: Vec<Vec<usize>>,
    actions: Vec<usize>,
    log_probs: Vec<f64>,
    rewards: Vec<f64>,
    is_terminals: Vec<bool>,
    values: Vec<f64>,
}

pub(crate) struct PpoStoreRawInput {
    state: Vec<f64>,
    seq_len: usize,
    pity: Vec<usize>,
    action: usize,
    log_prob: f64,
    reward: f64,
    done: bool,
    value: f64,
}

/// Cached CUDA tensors reused across PPO update minibatches.
#[cfg(cuda)]
struct PpoCudaUpdateCache {
    ones_action_1: Option<Tensor>,
}

#[cfg(cuda)]
impl PpoCudaUpdateCache {
    fn ones_action_1(&mut self, step: usize) -> Result<Tensor, TrainingError> {
        if self.ones_action_1.is_none() {
            let ones = Tensor::new_f32(vec![1.0; ACTION_SPACE], vec![ACTION_SPACE, 1]);
            let ones = ones.to_cuda().map_err(|error| {
                TrainingError::new("PPO", "constant upload", step, error.to_string())
            })?;
            self.ones_action_1 = Some(ones);
        }
        Ok(self
            .ones_action_1
            .as_ref()
            .expect("ones_action_1 cache")
            .clone())
    }
}

fn ppo_tensor_on_device(
    tensor: Tensor,
    use_cuda: bool,
    stage: &'static str,
    step: usize,
) -> Result<Tensor, TrainingError> {
    #[cfg(cuda)]
    {
        if use_cuda {
            return tensor
                .to_cuda()
                .map_err(|error| TrainingError::new("PPO", stage, step, error.to_string()));
        }
    }
    #[cfg(not(cuda))]
    let _ = (use_cuda, stage, step);
    Ok(tensor)
}

/// PPO trainer with clipped surrogate objective and GAE.
pub struct Ppo {
    pub policy: ActorCritic,
    ema_policy: Option<ActorCritic>, // EMA teacher for self-distillation
    optimizer: Optimizer,
    memory: Memory,
    k_epochs: usize,
    batch_size: usize,
    reward_normalizer: RunningMeanStd,
    distill_ema_decay: f64,
    distill_kl_coef: f64,
    distill_warmup_steps: usize,
    distill_update_counter: usize,
    optimizer_step_counter: usize,
    achf_orthogonal_penalty_interval: usize,
    achf_calibration_states: Vec<(Vec<f64>, usize)>,
    achf_calibration_seen: usize,
    use_cuda: bool,
    #[cfg(cuda)]
    cuda_update_cache: PpoCudaUpdateCache,
}

impl Ppo {
    fn achf_orthogonal_penalty_interval_from_config(config: &AchfConfig) -> usize {
        if config.enabled && config.apply_ffn && config.lambda_ortho > 0.0 {
            config.ortho_penalty_freq.max(1).saturating_mul(4)
        } else {
            usize::MAX
        }
    }

    fn achf_orthogonal_penalty_interval_for_policy(policy: &ActorCritic) -> usize {
        policy
            .achf_config()
            .as_ref()
            .map(Self::achf_orthogonal_penalty_interval_from_config)
            .unwrap_or(usize::MAX)
    }

    pub fn new(seed: u64, k_epochs: usize, batch_size: usize, config: &Config) -> Self {
        let policy = ActorCritic::new_with_config(config, seed);
        let mut ppo = Self::from_policy_on_device(policy, k_epochs, batch_size, config.device);
        ppo.achf_orthogonal_penalty_interval =
            Self::achf_orthogonal_penalty_interval_from_config(&config.achf);
        ppo
    }

    pub fn from_policy(policy: ActorCritic, k_epochs: usize, batch_size: usize) -> Self {
        Self::from_policy_on_device(policy, k_epochs, batch_size, ComputeDevice::Cpu)
    }

    pub fn from_policy_on_device(
        policy: ActorCritic,
        k_epochs: usize,
        batch_size: usize,
        device: ComputeDevice,
    ) -> Self {
        #[cfg(cuda)]
        let (policy, optimizer, use_cuda) = if device == ComputeDevice::Cuda {
            let mut cuda_policy = policy.clone();
            let cuda_backend = cuda_policy
                .try_to_cuda()
                .and_then(|()| GpuAdam::new(cuda_policy.parameters(), 0.0003));
            match cuda_backend {
                Ok(optimizer) => (cuda_policy, Optimizer::Gpu(optimizer), true),
                Err(error) => {
                    eprintln!(
                        "[CUDA] PPO training backend initialization failed ({error}); using CPU before training starts"
                    );
                    let optimizer = Optimizer::Cpu(Adam::new(policy.parameters(), 0.0003));
                    (policy, optimizer, false)
                }
            }
        } else {
            let optimizer = Optimizer::Cpu(Adam::new(policy.parameters(), 0.0003));
            (policy, optimizer, false)
        };
        #[cfg(not(cuda))]
        let (policy, optimizer, use_cuda) = {
            let _ = device;
            let optimizer = Optimizer::Cpu(Adam::new(policy.parameters(), 0.0003));
            (policy, optimizer, false)
        };
        let safe_batch_size = batch_size.max(1);
        let achf_orthogonal_penalty_interval =
            Self::achf_orthogonal_penalty_interval_for_policy(&policy);
        Ppo {
            policy,
            ema_policy: None,
            optimizer,
            memory: Memory {
                states_raw: vec![],
                state_lens: vec![],
                pities: vec![],
                actions: vec![],
                log_probs: vec![],
                rewards: vec![],
                is_terminals: vec![],
                values: vec![],
            },
            k_epochs,
            batch_size: safe_batch_size,
            reward_normalizer: RunningMeanStd::new(),
            distill_ema_decay: 0.995,
            distill_kl_coef: 0.0,
            distill_warmup_steps: 500,
            distill_update_counter: 0,
            optimizer_step_counter: 0,
            achf_orthogonal_penalty_interval,
            achf_calibration_states: Vec::new(),
            achf_calibration_seen: 0,
            use_cuda,
            #[cfg(cuda)]
            cuda_update_cache: PpoCudaUpdateCache {
                ones_action_1: None,
            },
        }
    }

    /// Initialize self-distillation: create EMA teacher and set distillation params
    pub fn init_distillation(&mut self, config: &Config) {
        if config.distill_enabled {
            self.ema_policy = Some(self.policy.clone());
            self.distill_ema_decay = config.distill_ema_decay;
            self.distill_kl_coef = config.distill_kl_coef;
            self.distill_warmup_steps = config.distill_warmup_steps;
            self.distill_update_counter = 0;
            println!(
                "[PPO] Distillation enabled: EMA decay={}, KL coef={}, warmup={}",
                self.distill_ema_decay, self.distill_kl_coef, self.distill_warmup_steps
            );
        }
    }

    /// Update EMA teacher weights: teacher = decay * teacher + (1 - decay) * student
    /// The per-update decay is derived from distill_ema_decay per batch:
    ///   effective_decay_per_step = distill_ema_decay ^ (1/batch_size)
    /// With batch_size=64 and distill_ema_decay=0.995:
    ///   effective_decay = 0.995^(1/64) ≈ 0.99922 per step
    fn update_ema_teacher(&mut self) {
        let Some(ref mut ema) = self.ema_policy else {
            return;
        };
        self.distill_update_counter += 1;

        // During warmup, teacher evolves without distillation penalty (freeema phase).
        // Distillation loss is only applied after warmup_steps updates.
        let decay = self.distill_ema_decay;
        let inv = 1.0 - decay;

        let student_params = self.policy.parameters();
        let ema_params = ema.parameters();

        assert_eq!(
            ema_params.len(),
            student_params.len(),
            "EMA teacher/student parameter topology mismatch"
        );
        for (index, (ema_p, stud_p)) in ema_params.iter().zip(student_params.iter()).enumerate() {
            assert_eq!(
                ema_p.shape, stud_p.shape,
                "EMA teacher/student parameter shape mismatch at index {index}"
            );
            assert_eq!(
                ema_p.dtype, stud_p.dtype,
                "EMA teacher/student parameter dtype mismatch at index {index}"
            );
            #[cfg(cuda)]
            if ema_p.cuda_lerp_in_place_from(stud_p, inv) {
                continue;
            }

            if ema_p.dtype == crate::dtype::Dtype::F64 {
                let mut ema_data = ema_p.data_write_f64();
                let stud_data = stud_p.data_f64();
                for (e, s) in ema_data.iter_mut().zip(stud_data.iter()) {
                    *e = decay * (*e) + inv * (*s);
                }
            } else {
                let mut ema_data = ema_p.data_write_f32();
                let stud_data = stud_p.data_f32();
                let decay_f32 = decay as f32;
                let inv_f32 = inv as f32;
                for (e, s) in ema_data.iter_mut().zip(stud_data.iter()) {
                    *e = decay_f32 * (*e) + inv_f32 * (*s);
                }
            }
        }
        ema.refresh_achf_after_optimizer_step();
    }

    fn achf_calibration_batch(
        corpus: &[(Vec<f64>, usize)],
        indices: &[usize],
        use_cuda: bool,
    ) -> Result<Option<Tensor>, TrainingError> {
        if indices.is_empty() {
            return Ok(None);
        }
        let Some(max_sequence_length) = indices
            .iter()
            .filter_map(|&index| corpus.get(index).map(|entry| entry.1))
            .max()
        else {
            return Ok(None);
        };
        let max_sequence_length = max_sequence_length.max(1);
        let mut batch_data = Vec::with_capacity(indices.len() * max_sequence_length * DIM);
        for &index in indices {
            let Some((state, sequence_length)) = corpus.get(index) else {
                return Ok(None);
            };
            if *sequence_length == 0 || state.len() != sequence_length.saturating_mul(DIM) {
                return Ok(None);
            }
            batch_data.extend_from_slice(state);
            if *sequence_length < max_sequence_length {
                batch_data.resize(
                    batch_data.len() + (max_sequence_length - sequence_length) * DIM,
                    0.0,
                );
            }
        }
        let batch = Tensor::new_f32(batch_data, vec![indices.len(), max_sequence_length, DIM]);
        #[cfg(cuda)]
        {
            if use_cuda {
                return batch.to_cuda().map(Some).map_err(|error| {
                    TrainingError::new("PPO", "ACHF calibration upload", 0, error.to_string())
                });
            }
            Ok(Some(batch))
        }
        #[cfg(not(cuda))]
        {
            let _ = use_cuda;
            Ok(Some(batch))
        }
    }

    fn validate_achf_candidates(
        policy: &mut ActorCritic,
        corpus: &[(Vec<f64>, usize)],
        validation_indices: &[usize],
        batch_size: usize,
        step: usize,
        use_cuda: bool,
    ) -> Result<(), TrainingError> {
        policy.set_achf_candidate_calibration_validation();
        for chunk in validation_indices.chunks(batch_size.max(1)) {
            if let Some(batch) = Self::achf_calibration_batch(corpus, chunk, use_cuda)? {
                let _ = policy.forward_actor_critic_batch(&batch);
            }
        }
        policy.record_achf_candidate_calibration_checkpoint(step);
        Ok(())
    }

    fn achf_candidate_calibration_target_met(
        policy: &ActorCritic,
        minimum_samples: usize,
        maximum_relative_error: f64,
    ) -> bool {
        let records = policy.achf_candidate_calibration_records();
        !records.is_empty()
            && records.iter().all(|record| {
                record.trace.last().is_some_and(|point| {
                    point.samples >= minimum_samples
                        && point.output_relative_error.is_finite()
                        && point.output_relative_error <= maximum_relative_error
                })
            })
    }

    fn calibrate_achf_candidates(&mut self) -> Result<(), TrainingError> {
        let Some(config) = self.policy.achf_config() else {
            return Ok(());
        };
        if !config.enabled
            || config.candidate_mode != "sparse"
            || config.candidate_train_from_scratch
            || config.candidate_calibration_steps == 0
            || config.candidate_calibration_max_samples == 0
        {
            return Ok(());
        }
        let corpus: Vec<(Vec<f64>, usize)> = self
            .achf_calibration_states
            .iter()
            .filter(|(state, sequence_length)| {
                *sequence_length > 0 && state.len() == sequence_length.saturating_mul(DIM)
            })
            .cloned()
            .collect();
        if corpus.len() < 2 {
            eprintln!("[ACHF] Candidate calibration skipped: fewer than two valid rollout states");
            return Ok(());
        }
        let active_layers = self.policy.begin_achf_candidate_calibration();
        if active_layers == 0 {
            return Ok(());
        }
        let parameters = self.policy.achf_candidate_calibration_parameters();
        let parameter_masks = self.policy.achf_candidate_calibration_parameter_masks();
        assert_eq!(
            parameters.len(),
            parameter_masks.len(),
            "ACHF candidate calibration parameter/mask topology mismatch"
        );
        let mut optimizer = Optimizer::new(parameters, config.candidate_calibration_lr)
            .map_err(|error| TrainingError::new("PPO", "ACHF calibration setup", 0, error))?;

        let mut training_indices = Vec::new();
        let mut validation_indices = Vec::new();
        for index in 0..corpus.len() {
            if index % 4 == 0 {
                validation_indices.push(index);
            } else {
                training_indices.push(index);
            }
        }
        if training_indices.is_empty() {
            training_indices.push(validation_indices[0]);
        }
        if validation_indices.is_empty() {
            validation_indices.push(training_indices[0]);
        }

        let total_steps = config.candidate_calibration_steps;
        let checkpoint_interval = (total_steps / 4).max(1);
        Self::validate_achf_candidates(
            &mut self.policy,
            &corpus,
            &validation_indices,
            self.batch_size,
            0,
            self.use_cuda,
        )?;
        let mut completed_steps = 0usize;
        for step in 1..=total_steps {
            let start = ((step - 1) * self.batch_size) % training_indices.len();
            let count = self.batch_size.min(training_indices.len()).max(1);
            let batch_indices: Vec<usize> = (0..count)
                .map(|offset| training_indices[(start + offset) % training_indices.len()])
                .collect();
            self.policy.set_achf_candidate_calibration_training();
            optimizer.zero_grad();
            let Some(batch) = Self::achf_calibration_batch(&corpus, &batch_indices, self.use_cuda)?
            else {
                continue;
            };
            let _ = self.policy.forward_actor_critic_batch(&batch);
            let Some(loss) = self.policy.take_achf_candidate_calibration_loss() else {
                continue;
            };
            loss.backward();
            optimizer
                .step()
                .map_err(|error| TrainingError::optimizer("PPO ACHF calibration", step, error))?;
            self.policy.enforce_achf_candidate_masks();
            completed_steps = step;
            if step.is_multiple_of(checkpoint_interval) || step == total_steps {
                Self::validate_achf_candidates(
                    &mut self.policy,
                    &corpus,
                    &validation_indices,
                    self.batch_size,
                    step,
                    self.use_cuda,
                )?;
                if Self::achf_candidate_calibration_target_met(
                    &self.policy,
                    config.candidate_min_calibration_samples,
                    config.candidate_max_output_relative_error,
                ) {
                    break;
                }
            }
        }
        if completed_steps != total_steps {
            Self::validate_achf_candidates(
                &mut self.policy,
                &corpus,
                &validation_indices,
                self.batch_size,
                completed_steps,
                self.use_cuda,
            )?;
        }
        let masked_moment_max_abs = optimizer.masked_moment_max_abs(&parameter_masks);
        self.policy
            .finalize_achf_candidate_calibration(completed_steps, masked_moment_max_abs);
        let memory = self.policy.achf_memory_stats_aggregate();
        println!(
            "[ACHF] Candidate calibration: layers={active_layers}, steps={completed_steps}, validation_samples={}, output_error={:.6}, eligible={}/{}",
            memory.candidate_output_samples,
            memory.candidate_output_relative_error().unwrap_or(f64::INFINITY),
            memory.eligible_candidate_layers,
            memory.candidate_layers,
        );
        Ok(())
    }

    pub(crate) fn store_raw(&mut self, input: PpoStoreRawInput) {
        let PpoStoreRawInput {
            state,
            seq_len,
            pity,
            action,
            log_prob,
            reward,
            done,
            value,
        } = input;
        // Update reward normalizer
        self.reward_normalizer.update(reward);
        self.memory.states_raw.push(state);
        self.memory.state_lens.push(seq_len);
        self.memory.pities.push(pity);
        self.memory.actions.push(action);
        self.memory.log_probs.push(log_prob);
        self.memory.rewards.push(reward);
        self.memory.is_terminals.push(done);
        self.memory.values.push(value);
    }

    pub fn update(&mut self, current_lr: f64) -> Result<f64, TrainingError> {
        self.update_with_progress(current_lr, |_, _, _, _| {})
    }

    fn capture_achf_calibration_states(&mut self) {
        let maximum = self
            .policy
            .achf_config()
            .map_or(0, |config| config.candidate_calibration_max_samples);
        if maximum == 0 {
            return;
        }
        for (state, &sequence_length) in self
            .memory
            .states_raw
            .iter()
            .zip(self.memory.state_lens.iter())
        {
            let slot = self.achf_calibration_seen % maximum;
            if self.achf_calibration_states.len() < maximum {
                self.achf_calibration_states
                    .push((state.clone(), sequence_length));
            } else {
                self.achf_calibration_states[slot] = (state.clone(), sequence_length);
            }
            self.achf_calibration_seen = self.achf_calibration_seen.saturating_add(1);
        }
    }

    fn update_with_progress<F>(
        &mut self,
        current_lr: f64,
        mut on_batch: F,
    ) -> Result<f64, TrainingError>
    where
        F: FnMut(usize, usize, usize, usize),
    {
        if self.memory.states_raw.is_empty() {
            return Ok(0.0);
        }

        // Update Learning Rate
        self.optimizer.set_lr(current_lr);

        let len = self.memory.states_raw.len();
        self.capture_achf_calibration_states();
        let states_raw = std::mem::take(&mut self.memory.states_raw);
        let state_lens = std::mem::take(&mut self.memory.state_lens);
        let _pities = std::mem::take(&mut self.memory.pities);
        let actions = std::mem::take(&mut self.memory.actions);
        let log_probs = std::mem::take(&mut self.memory.log_probs);
        let rewards = std::mem::take(&mut self.memory.rewards);
        let is_terminals = std::mem::take(&mut self.memory.is_terminals);
        let values = std::mem::take(&mut self.memory.values);
        let states: Vec<(Vec<f64>, usize)> = states_raw.into_iter().zip(state_lens).collect();
        let mut advantages = vec![0.0; len];
        let mut returns = vec![0.0; len];

        let mut last_gae_lam = 0.0;

        // Batch-normalize rewards for stable GAE (use uniform stats across the batch)
        let r_mean = sum_f64(&rewards) / len as f64;
        let r_std = (sum_sq_diff(&rewards, r_mean) / len as f64)
            .sqrt()
            .max(1e-8);
        let norm_rewards: Vec<f64> = rewards
            .iter()
            .map(|&r| ((r - r_mean) / r_std).clamp(-10.0, 10.0))
            .collect();

        for t in (0..len).rev() {
            let non_terminal = if is_terminals[t] { 0.0 } else { 1.0 };
            let val_t = values[t];
            let val_next = if t < len - 1 {
                values[t + 1]
            } else if is_terminals[t] {
                0.0
            } else {
                values[t]
            };

            // Use normalized rewards for training signal
            let delta = norm_rewards[t] + GAMMA * val_next * non_terminal - val_t;
            let gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae_lam;

            advantages[t] = gae;
            returns[t] = gae + val_t;

            last_gae_lam = if is_terminals[t] { 0.0 } else { gae };
        }

        let adv_mean: f64 = sum_f64(&advantages) / len as f64;
        let adv_std: f64 = (sum_sq_diff(&advantages, adv_mean) / len as f64).sqrt() + 1e-8;
        let norm_advantages: Vec<f64> = normalize_slice(&advantages, adv_mean, adv_std);

        // Target KL Divergence for Early Stopping
        let target_kl = 0.015;
        let mut indices: Vec<usize> = (0..len).collect();
        let loss_sum_tensor = Tensor::zeros_f32(vec![1]);
        let mut loss_sum_tensor = ppo_tensor_on_device(
            loss_sum_tensor,
            self.use_cuda,
            "loss accumulator upload",
            self.optimizer_step_counter,
        )?;
        let mut loss_count = 0usize;
        let planned_batches_per_epoch = len.div_ceil(self.batch_size);
        let planned_batches = self.k_epochs * planned_batches_per_epoch;
        let mut completed_batches = 0usize;
        for epoch_idx in 0..self.k_epochs {
            indices.shuffle(&mut rand::rng());
            let mut early_stop = false;

            for chunk in indices.chunks(self.batch_size) {
                self.optimizer.zero_grad();

                let batch_len = chunk.len();
                let inv_batch = 1.0 / batch_len as f64;

                // Batched forward: pad states to max seq_len and stack
                let max_seq_len = chunk.iter().map(|&i| states[i].1).max().unwrap_or(1);
                let mut batch_data = Vec::with_capacity(batch_len * max_seq_len * DIM);
                for &i in chunk {
                    let (state_data, seq_len) = (&states[i].0, states[i].1);
                    batch_data.extend_from_slice(state_data);
                    if seq_len < max_seq_len {
                        batch_data.resize(batch_data.len() + (max_seq_len - seq_len) * DIM, 0.0);
                    }
                }
                let batch_states = Tensor::new_f32(batch_data, vec![batch_len, max_seq_len, DIM]);
                let batch_states = ppo_tensor_on_device(
                    batch_states,
                    self.use_cuda,
                    "training batch upload",
                    self.optimizer_step_counter,
                )?;
                let (batch_logits, batch_values) =
                    self.policy.forward_actor_critic_batch(&batch_states);
                let batch_log_probs = batch_logits.log_softmax();
                let batch_log_probs_copy = batch_log_probs.clone();
                let batch_probs = batch_log_probs_copy.exp();
                #[cfg(cuda)]
                let ones_action_1 = if self.use_cuda {
                    self.cuda_update_cache
                        .ones_action_1(self.optimizer_step_counter)?
                } else {
                    Tensor::new_f32(vec![1.0; ACTION_SPACE], vec![ACTION_SPACE, 1])
                };
                #[cfg(not(cuda))]
                let ones_action_1 = Tensor::new_f32(vec![1.0; ACTION_SPACE], vec![ACTION_SPACE, 1]);
                let batch_entropy =
                    -(batch_probs.clone() * batch_log_probs_copy).matmul(&ones_action_1);

                // Batched teacher forward only after warmup; detach teacher logits so KL trains
                // the student policy without backpropagating into the EMA teacher.
                let distillation_active = self.distill_kl_coef > 0.0
                    && self.distill_update_counter >= self.distill_warmup_steps;
                let teacher_batch_logits = if distillation_active {
                    if let Some(ema) = self.ema_policy.as_ref() {
                        let (t_logits, _) = ema.forward_actor_critic_batch(&batch_states);
                        Some(t_logits.detach())
                    } else {
                        None
                    }
                } else {
                    None
                };

                let mut distill_accum = Tensor::zeros_f32(vec![1]);
                distill_accum = ppo_tensor_on_device(
                    distill_accum,
                    self.use_cuda,
                    "distillation accumulator upload",
                    self.optimizer_step_counter,
                )?;

                let mut action_mask_data = vec![0.0; batch_len * ACTION_SPACE];
                let mut old_log_prob_data = Vec::with_capacity(batch_len);
                let mut advantage_data = Vec::with_capacity(batch_len);
                let mut return_data = Vec::with_capacity(batch_len);
                for (row, &i) in chunk.iter().enumerate() {
                    let action_idx = actions[i].min(ACTION_SPACE - 1);
                    action_mask_data[row * ACTION_SPACE + action_idx] = 1.0;
                    old_log_prob_data.push(log_probs[i]);
                    advantage_data.push(norm_advantages[i]);
                    return_data.push(returns[i]);
                }

                let action_mask = Tensor::new_f32(action_mask_data, vec![batch_len, ACTION_SPACE]);
                let old_log_prob_tensor = Tensor::new_f32(old_log_prob_data, vec![batch_len, 1]);
                let advantage_tensor = Tensor::new_f32(advantage_data, vec![batch_len, 1]);
                let return_tensor = Tensor::new_f32(return_data, vec![batch_len, 1]);
                let one_batch = Tensor::new_f32(vec![1.0; batch_len], vec![batch_len, 1]);
                let clip_low =
                    Tensor::new_f32(vec![1.0 - CLIP_EPSILON; batch_len], vec![batch_len, 1]);
                let clip_high =
                    Tensor::new_f32(vec![1.0 + CLIP_EPSILON; batch_len], vec![batch_len, 1]);
                let value_coef = Tensor::new_f32(vec![VALUE_COEF; batch_len], vec![batch_len, 1]);
                let entropy_coef =
                    Tensor::new_f32(vec![ENTROPY_COEF; batch_len], vec![batch_len, 1]);

                let action_mask = ppo_tensor_on_device(
                    action_mask,
                    self.use_cuda,
                    "action mask upload",
                    self.optimizer_step_counter,
                )?;
                let old_log_prob_tensor = ppo_tensor_on_device(
                    old_log_prob_tensor,
                    self.use_cuda,
                    "old log-probability upload",
                    self.optimizer_step_counter,
                )?;
                let advantage_tensor = ppo_tensor_on_device(
                    advantage_tensor,
                    self.use_cuda,
                    "advantage upload",
                    self.optimizer_step_counter,
                )?;
                let return_tensor = ppo_tensor_on_device(
                    return_tensor,
                    self.use_cuda,
                    "return upload",
                    self.optimizer_step_counter,
                )?;
                let one_batch = ppo_tensor_on_device(
                    one_batch,
                    self.use_cuda,
                    "unit tensor upload",
                    self.optimizer_step_counter,
                )?;
                let clip_low = ppo_tensor_on_device(
                    clip_low,
                    self.use_cuda,
                    "clip lower-bound upload",
                    self.optimizer_step_counter,
                )?;
                let clip_high = ppo_tensor_on_device(
                    clip_high,
                    self.use_cuda,
                    "clip upper-bound upload",
                    self.optimizer_step_counter,
                )?;
                let value_coef = ppo_tensor_on_device(
                    value_coef,
                    self.use_cuda,
                    "value coefficient upload",
                    self.optimizer_step_counter,
                )?;
                let entropy_coef = ppo_tensor_on_device(
                    entropy_coef,
                    self.use_cuda,
                    "entropy coefficient upload",
                    self.optimizer_step_counter,
                )?;

                let selected_log_probs =
                    (batch_log_probs.clone() * action_mask).matmul(&ones_action_1);
                let log_ratio = selected_log_probs - old_log_prob_tensor;
                let ratio = log_ratio.exp();
                let approx_kl_tensor =
                    ((ratio.clone() - one_batch.clone()) - log_ratio.clone()).sum();

                let surr1 = ratio.clone() * advantage_tensor.clone();
                let ratio_clipped = {
                    let clipped_low = clip_low.clone() + (ratio.clone() - clip_low.clone()).relu();
                    clipped_low.clone() - (clipped_low - clip_high.clone()).relu()
                };
                let surr2 = ratio_clipped * advantage_tensor;
                let policy_loss = surr1.clone() - (surr1 - surr2).relu();

                let value_err = batch_values.clone() - return_tensor;
                let value_loss = value_err.clone() * value_err;
                let loss_elements =
                    -policy_loss + value_loss * value_coef - batch_entropy * entropy_coef;

                // Batched distillation KL after the per-sample loop
                if let Some(ref teacher_logits) = teacher_batch_logits {
                    let teacher_log_probs = teacher_logits.log_softmax();
                    let kl_elements = batch_probs * (batch_log_probs.clone() - teacher_log_probs);
                    let total_kl = kl_elements.matmul(&ones_action_1).sum();
                    distill_accum = total_kl;
                }

                let batch_size_tensor = Tensor::new_f32(vec![inv_batch], vec![1]);
                let batch_size_tensor = ppo_tensor_on_device(
                    batch_size_tensor,
                    self.use_cuda,
                    "batch scale upload",
                    self.optimizer_step_counter,
                )?;
                let mut final_loss = loss_elements.sum() * batch_size_tensor.clone();
                // Add distillation loss after warmup: distill_coef * mean(kl_divs)
                if teacher_batch_logits.is_some() {
                    let distill_coef_tensor = Tensor::new_f32(vec![self.distill_kl_coef], vec![1]);
                    let distill_coef_tensor = ppo_tensor_on_device(
                        distill_coef_tensor,
                        self.use_cuda,
                        "distillation coefficient upload",
                        self.optimizer_step_counter,
                    )?;
                    let distill_term = distill_accum * distill_coef_tensor * batch_size_tensor;
                    final_loss = final_loss + distill_term;
                }
                if self.achf_orthogonal_penalty_interval != usize::MAX
                    && (self.optimizer_step_counter + 1)
                        .is_multiple_of(self.achf_orthogonal_penalty_interval)
                {
                    if let Some(reg) = self.policy.achf_orthogonal_penalty() {
                        final_loss = final_loss + reg;
                    }
                }
                loss_sum_tensor = loss_sum_tensor.detach() + final_loss.detach();
                loss_count += 1;
                final_loss.backward();
                self.policy.update_achf_after_backward();
                self.optimizer.step().map_err(|error| {
                    TrainingError::optimizer("PPO", self.optimizer_step_counter + 1, error)
                })?;
                self.policy.refresh_achf_after_optimizer_step();
                self.optimizer_step_counter += 1;
                completed_batches += 1;
                on_batch(
                    completed_batches,
                    planned_batches,
                    epoch_idx + 1,
                    self.k_epochs,
                );
                // Update EMA teacher after each batch for self-distillation
                self.update_ema_teacher();

                if completed_batches.is_multiple_of(KL_CHECK_INTERVAL) {
                    let approx_kl = approx_kl_tensor.detach().item() / chunk.len() as f64;
                    if approx_kl > target_kl * 1.5 {
                        early_stop = true;
                    }
                }

                if early_stop {
                    break;
                }
            }

            if early_stop {
                break;
            }

            // Early Stopping check
            if early_stop {
                break;
            }
        }
        if loss_count > 0 {
            Ok(loss_sum_tensor.item() / loss_count as f64)
        } else {
            Ok(0.0)
        }
    }
}

struct PpoEnvState {
    state_struct: PullState,
    env_noise: f64,
    env_bias: f64,
    pulls_done: usize,
    history_buffer: VecDeque<Vec<f64>>,
    pity_buffer: VecDeque<usize>,
    flat_data: Vec<f64>,
    pity_vec: Vec<usize>,
    kv_cache: Vec<KVCache>,
    kv_cache_valid: bool,
    episode_reward: f64,
    episode_breakdown: RewardBreakdown,
    rng: Rng,
}

impl PpoEnvState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        seed: u64,
        env_net: &EnvNet,
        context_len: usize,
        num_heads: usize,
        num_layers: usize,
        kv_lora_rank: usize,
        v_head_dim: usize,
        qk_rope_dim: usize,
        max_seq_len: usize,
        config: &Config,
    ) -> Self {
        let mut rng = Rng::from_seed(seed);
        let (env_noise, env_bias) = env_net_env(env_net, &mut rng, 0, 0, 0, 0);
        let mut caches: Vec<_> = (0..num_layers).map(|_| KVCache::new(num_heads)).collect();
        for cache in &mut caches {
            cache.preallocate(
                num_heads,
                kv_lora_rank,
                v_head_dim,
                qk_rope_dim,
                max_seq_len,
            );
        }
        Self {
            state_struct: PullState::new(config),
            env_noise,
            env_bias,
            pulls_done: 0,
            history_buffer: VecDeque::with_capacity(context_len),
            pity_buffer: VecDeque::with_capacity(context_len),
            flat_data: Vec::with_capacity(context_len * DIM),
            pity_vec: Vec::with_capacity(context_len),
            kv_cache: caches,
            kv_cache_valid: true,
            episode_reward: 0.0,
            episode_breakdown: RewardBreakdown::default(),
            rng,
        }
    }

    fn reset(&mut self, env_net: &EnvNet, config: &Config) {
        self.history_buffer.clear();
        self.pity_buffer.clear();
        for cache in self.kv_cache.iter_mut() {
            cache.clear();
        }
        self.kv_cache_valid = true;
        self.state_struct = PullState::new(config);
        let (env_noise, env_bias) = env_net_env(env_net, &mut self.rng, 0, 0, 0, 0);
        self.env_noise = env_noise;
        self.env_bias = env_bias;
        self.pulls_done = 0;
        self.episode_reward = 0.0;
        self.episode_breakdown = RewardBreakdown::default();
    }

    fn prepare_policy_step(
        &mut self,
        policy: &ActorCritic,
        config: &Config,
        context_len: usize,
    ) -> usize {
        let current_state_raw = build_features_with_luck_budget(
            self.state_struct.pity_6,
            self.pulls_done,
            self.env_noise,
            self.state_struct.streak_4_star,
            self.env_bias,
            self.state_struct.loss_streak,
            self.state_struct.luck_budget,
            config,
        )
        .to_vec();

        let current_pity = self.state_struct.pity_6;

        self.history_buffer.push_back(current_state_raw);
        self.pity_buffer.push_back(current_pity);
        if self.history_buffer.len() > context_len {
            self.history_buffer.pop_front();
            self.pity_buffer.pop_front();
            if self.kv_cache_valid {
                policy.prune_cache(&mut self.kv_cache, context_len.saturating_sub(1));
            }
        }

        let seq_len = self.history_buffer.len();
        self.flat_data.clear();
        for s in self.history_buffer.iter() {
            self.flat_data.extend_from_slice(s);
        }
        self.pity_vec.clear();
        self.pity_vec.extend(self.pity_buffer.iter().copied());
        seq_len
    }

    fn current_token_f32(&self, out: &mut Vec<f32>) {
        let token = self
            .history_buffer
            .back()
            .expect("history_buffer should not be empty after push")
            .as_slice();
        out.clear();
        out.extend(token.iter().map(|&v| v as f32));
    }

    fn apply_policy_step(
        &mut self,
        action_idx: usize,
        log_prob: f32,
        val: f32,
        env_net: &EnvNet,
        config: &Config,
    ) -> PpoStepResult {
        let outcome = step_pull(
            &mut self.state_struct,
            &mut self.rng,
            config,
            config.big_pity_requires_not_up,
            GachaAction::ppo(action_idx, ACTIONS[action_idx], log_prob as f64, val as f64),
        );
        self.pulls_done += 1;

        let mut reward_breakdown = compute_reward_ppo_breakdown(
            outcome.rarity == 6,
            outcome.is_up,
            self.state_struct.loss_streak,
            outcome.luck_modifier,
            config.luck_action_cost,
        );
        if outcome.rarity == 6 && outcome.is_up {
            if self.pulls_done < EARLY_UP_BONUS_THRESHOLD_1 {
                reward_breakdown.add_early_bonus(5.0);
            }
            if self.pulls_done < EARLY_UP_BONUS_THRESHOLD_2 {
                reward_breakdown.add_early_bonus(5.0);
            }
        }
        let reward = reward_breakdown.total();
        self.episode_reward += reward;
        self.episode_breakdown.add_assign(reward_breakdown);

        let done = outcome.is_up || self.pulls_done >= EPISODE_MAX_PULLS;

        let experience = PpoStoreRawInput {
            state: self.flat_data.clone(),
            seq_len: self.history_buffer.len(),
            pity: self.pity_vec.clone(),
            action: outcome.action.unwrap_or(action_idx),
            log_prob: outcome.ppo_log_prob.unwrap_or(log_prob as f64),
            reward,
            done,
            value: outcome.ppo_value.unwrap_or(val as f64),
        };

        let finished_reward = if done {
            Some(self.episode_reward)
        } else {
            None
        };
        let finished_breakdown = if done {
            Some(self.episode_breakdown)
        } else {
            None
        };
        if done {
            self.reset(env_net, config);
        }

        PpoStepResult {
            experience,
            finished_reward,
            finished_breakdown,
        }
    }

    fn step(
        &mut self,
        policy: &ActorCritic,
        env_net: &EnvNet,
        config: &Config,
        context_len: usize,
    ) -> PpoStepResult {
        let seq_len = self.prepare_policy_step(policy, config, context_len);
        let (action_idx, log_prob, val) = if self.kv_cache_valid {
            let mut token_f32 = Vec::with_capacity(DIM);
            self.current_token_f32(&mut token_f32);
            policy.step_inference_cached_with_value(
                &token_f32,
                &mut self.kv_cache,
                seq_len - 1,
                config.ppo_top_k,
            )
        } else {
            policy.step_sequence_with_value(&self.flat_data, seq_len, config.ppo_top_k)
        };
        self.apply_policy_step(action_idx, log_prob, val, env_net, config)
    }
}

struct PpoStepResult {
    experience: PpoStoreRawInput,
    finished_reward: Option<f64>,
    finished_breakdown: Option<RewardBreakdown>,
}

fn rollout_envs_sequential(
    envs: &mut [PpoEnvState],
    policy: &ActorCritic,
    env_net: &EnvNet,
    config: &Config,
    context_len: usize,
) -> Vec<PpoStepResult> {
    envs.iter_mut()
        .map(|env| env.step(policy, env_net, config, context_len))
        .collect()
}

fn rollout_cpu_round(
    worker: &GoodJobWorker,
    envs: &mut [PpoEnvState],
    policy: &ActorCritic,
    env_net: &EnvNet,
    config: &Config,
    context_len: usize,
) -> Vec<PpoStepResult> {
    worker
        .execute(|| {
            envs.par_iter_mut()
                .map(|env| env.step(policy, env_net, config, context_len))
                .collect()
        })
        .unwrap_or_else(|msg| {
            log::error!(
                "[PPO] Worker execution failed: {}. Falling back to sequential rollout.",
                msg
            );
            rollout_envs_sequential(envs, policy, env_net, config, context_len)
        })
}

#[cfg(cuda)]
fn rollout_cuda_round(
    envs: &mut [PpoEnvState],
    policy: &ActorCritic,
    env_net: &EnvNet,
    config: &Config,
    context_len: usize,
) -> Option<Vec<PpoStepResult>> {
    if envs.is_empty() || !crate::cuda::is_available() {
        return None;
    }
    Some(rollout_envs_sequential(
        envs,
        policy,
        env_net,
        config,
        context_len,
    ))
}

/// Train a PPO agent with multi-environment rollouts.
pub fn train_ppo(
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
) -> Result<ActorCritic, TrainingError> {
    train_ppo_impl(rng, env_net, config, TrainingMetricsSink::noop())
}

fn train_ppo_impl(
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    mut metrics: impl TrainingMetrics,
) -> Result<ActorCritic, TrainingError> {
    println!("\n[PPO] Initializing PPO Training (Actor-Critic)...");
    let fast_mode = config.fast_init || config.ppo_mode == "fast";
    let total_steps = if config.ppo_total_steps > 0 {
        config.ppo_total_steps
    } else if fast_mode {
        4_000
    } else {
        20_000
    };
    let steps_per_update = if config.ppo_steps_per_update > 0 {
        config.ppo_steps_per_update
    } else if fast_mode {
        256
    } else {
        1_024
    };
    let k_epochs = if config.ppo_k_epochs > 0 {
        config.ppo_k_epochs
    } else if fast_mode {
        2
    } else {
        3
    };
    let batch_size = if config.ppo_batch_size > 0 {
        config.ppo_batch_size
    } else {
        128
    };
    let context_len = if config.ppo_context_len > 0 {
        config.ppo_context_len
    } else if fast_mode {
        6
    } else {
        8
    };
    let configured_num_envs = if config.ppo_num_envs > 0 {
        config.ppo_num_envs
    } else {
        1
    };
    let num_envs = configured_num_envs.min(steps_per_update.max(1)).max(1);
    let worker = GoodJobWorker::new_with_config(config).expect("Failed to build PPO worker pool");
    let mut ppo = Ppo::new(rng.next_u64(), k_epochs, batch_size, config);
    ppo.init_distillation(config);
    let mut steps_done = 0;

    let env_seeds: Vec<u64> = (0..num_envs).map(|_| rng.next_u64()).collect();
    let mla_cfg = &ppo.policy.backbone.blocks[0].mla_layer.config;
    let mut envs: Vec<PpoEnvState> = env_seeds
        .into_iter()
        .map(|seed| {
            PpoEnvState::new(
                seed,
                env_net,
                context_len,
                mla_cfg.num_heads,
                ppo.policy.backbone.blocks.len(),
                mla_cfg.kv_lora_rank,
                mla_cfg.v_head_dim,
                mla_cfg.qk_rope_dim,
                mla_cfg.max_seq_len,
                config,
            )
        })
        .collect();

    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(50);
    let mut recent_reward_breakdowns: VecDeque<RewardBreakdown> = VecDeque::with_capacity(50);
    let mut _episode_count = 0;

    // Linear LR decay
    let initial_lr = 0.0003;

    let heartbeat_every = if fast_mode { 128 } else { 512 };
    let mut last_heartbeat = Instant::now();
    let mut remainder_offset = 0usize;
    let snapshot_every = (total_steps / 200).max(1);
    let pb = create_bar(total_steps as u64, "PPO Training");
    while steps_done < total_steps {
        // Calculate LR
        let progress = steps_done as f64 / total_steps as f64;
        let current_lr = initial_lr * (1.0 - progress).max(0.1); // Decay to 10%

        let remaining_steps = total_steps - steps_done;
        let update_steps = steps_per_update.min(remaining_steps);
        let rounds = update_steps / num_envs;
        let remainder = update_steps % num_envs;
        let mut collected = 0usize;
        let mut next_heartbeat = heartbeat_every.min(update_steps).max(1);
        for _ in 0..rounds {
            #[cfg(cuda)]
            let step_results =
                rollout_cuda_round(&mut envs, &ppo.policy, env_net, config, context_len)
                    .unwrap_or_else(|| {
                        rollout_cpu_round(
                            &worker,
                            &mut envs,
                            &ppo.policy,
                            env_net,
                            config,
                            context_len,
                        )
                    });
            #[cfg(not(cuda))]
            let step_results = rollout_cpu_round(
                &worker,
                &mut envs,
                &ppo.policy,
                env_net,
                config,
                context_len,
            );
            if step_results.is_empty() {
                break;
            }
            for result in step_results {
                ppo.store_raw(result.experience);
                if let Some(done_reward) = result.finished_reward {
                    _episode_count += 1;
                    recent_rewards.push_back(done_reward);
                    if recent_rewards.len() > 50 {
                        recent_rewards.pop_front();
                    }
                }
                if let Some(done_breakdown) = result.finished_breakdown {
                    recent_reward_breakdowns.push_back(done_breakdown);
                    if recent_reward_breakdowns.len() > 50 {
                        recent_reward_breakdowns.pop_front();
                    }
                }
            }
            collected += num_envs;
            if collected >= next_heartbeat && last_heartbeat.elapsed() >= Duration::from_millis(300)
            {
                let global_step = (steps_done + collected).min(total_steps);
                let avg_env_reward =
                    envs.iter().map(|e| e.episode_reward).sum::<f64>() / num_envs as f64;
                let cur_breakdown =
                    RewardBreakdown::average(envs.iter().map(|env| env.episode_breakdown));
                pb.set_position(global_step as u64);
                pb.set_message(format!(
                    "CurRet: {:.2} | {} | LR: {:.6}",
                    avg_env_reward,
                    cur_breakdown.format_compact(),
                    current_lr
                ));
                last_heartbeat = Instant::now();
                while next_heartbeat <= collected {
                    next_heartbeat += heartbeat_every;
                }
            }
        }
        if remainder > 0 {
            let start = remainder_offset % num_envs;
            for i in 0..remainder {
                let idx = (start + i) % num_envs;
                let result = envs[idx].step(&ppo.policy, env_net, config, context_len);
                ppo.store_raw(result.experience);
                if let Some(done_reward) = result.finished_reward {
                    _episode_count += 1;
                    recent_rewards.push_back(done_reward);
                    if recent_rewards.len() > 50 {
                        recent_rewards.pop_front();
                    }
                }
                if let Some(done_breakdown) = result.finished_breakdown {
                    recent_reward_breakdowns.push_back(done_breakdown);
                    if recent_reward_breakdowns.len() > 50 {
                        recent_reward_breakdowns.pop_front();
                    }
                }
                collected += 1;
                if collected >= next_heartbeat
                    && last_heartbeat.elapsed() >= Duration::from_millis(300)
                {
                    let global_step = (steps_done + collected).min(total_steps);
                    let cur_breakdown = envs[idx].episode_breakdown;
                    pb.set_position(global_step as u64);
                    pb.set_message(format!(
                        "CurRet: {:.2} | {} | LR: {:.6}",
                        envs[idx].episode_reward,
                        cur_breakdown.format_compact(),
                        current_lr
                    ));
                    last_heartbeat = Instant::now();
                    while next_heartbeat <= collected {
                        next_heartbeat += heartbeat_every;
                    }
                }
            }
            remainder_offset = (remainder_offset + remainder) % num_envs;
        }

        let rollout_position = (steps_done + collected).min(total_steps);
        if collected == 0 {
            log::error!("[PPO] No rollout samples collected; stopping PPO training loop.");
            break;
        }
        pb.set_position(rollout_position as u64);
        pb.set_message(format!(
            "Updating ({} samples, {} epochs)...",
            collected, k_epochs
        ));
        let update_loss = ppo.update_with_progress(
            current_lr,
            |done_batches, total_batches, epoch_idx, total_epochs| {
                if last_heartbeat.elapsed() >= Duration::from_millis(300)
                    || done_batches == total_batches
                {
                    pb.set_position(rollout_position as u64);
                    pb.set_message(format!(
                        "Updating: batch {}/{} | epoch {}/{} | LR: {:.6}",
                        done_batches, total_batches, epoch_idx, total_epochs, current_lr
                    ));
                    last_heartbeat = Instant::now();
                }
            },
        )?;
        steps_done += collected;

        if config.achf.cache_log_interval_steps > 0
            && steps_done % config.achf.cache_log_interval_steps == 0
        {
            if config.achf.cache_log_per_layer {
                for (idx, stats) in ppo.policy.achf_cache_stats_iter().enumerate() {
                    if stats.calls > 0 {
                        println!(
                            "\n[ACHF-L{}] {}",
                            idx,
                            crate::utils::format_achf_stats(&stats)
                        );
                    }
                }
            } else {
                let stats = ppo.policy.achf_cache_stats_aggregate();
                if stats.calls > 0 {
                    println!("\n{}", crate::utils::format_achf_stats(&stats));
                }
            }
        }

        let avg_r = if recent_rewards.is_empty() {
            0.0
        } else {
            recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64
        };
        let avg_breakdown = RewardBreakdown::average(recent_reward_breakdowns.iter().copied());
        pb.set_position(steps_done as u64);
        pb.set_message(format!(
            "AvgRet: {:.2} | {} | LR: {:.6}",
            avg_r,
            avg_breakdown.format_compact(),
            current_lr
        ));

        if config.policy_eval_interval > 0
            && steps_done % config.policy_eval_interval < update_steps
        {
            let eval = evaluate_ppo_policy(
                &ppo.policy,
                env_net,
                config,
                context_len,
                config.policy_eval_episodes,
                config.policy_eval_seed,
            );
            println!("\n{}", format_policy_eval("PPO", steps_done, &eval));
        }

        if metrics.is_enabled() && steps_done % snapshot_every < collected {
            metrics.emit_achf_snapshot(steps_done, update_loss, avg_r, ppo.policy.snapshot_achf());
        }
    }
    pb.finish_with_message("PPO Training Complete.");
    ppo.calibrate_achf_candidates()?;
    ppo.policy.freeze_achf_for_inference();
    Ok(ppo.policy)
}

/// Train PPO with optional metrics collection for benchmarking.
pub fn train_ppo_with_metrics(
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    metrics_tx: Option<std::sync::mpsc::Sender<StepSnapshot>>,
) -> Result<ActorCritic, TrainingError> {
    train_ppo_impl(rng, env_net, config, TrainingMetricsSink::from(metrics_tx))
}

/// Incremental PPO trainer for online learning during interactive mode.
pub struct OnlinePpoTrainer {
    ppo: Ppo,
    steps_done: usize,
}

impl OnlinePpoTrainer {
    #[allow(dead_code)]
    pub fn new(seed: u64, k_epochs: usize, batch_size: usize, config: &Config) -> Self {
        Self::from_policy_on_device(
            ActorCritic::new_with_config(config, seed),
            k_epochs,
            batch_size,
            config.device,
        )
    }

    pub fn from_policy(policy: ActorCritic, k_epochs: usize, batch_size: usize) -> Self {
        Self::from_policy_on_device(policy, k_epochs, batch_size, ComputeDevice::Cpu)
    }

    pub fn from_policy_on_device(
        policy: ActorCritic,
        k_epochs: usize,
        batch_size: usize,
        device: ComputeDevice,
    ) -> Self {
        Self {
            ppo: Ppo::from_policy_on_device(policy, k_epochs, batch_size, device),
            steps_done: 0,
        }
    }

    pub fn push(&mut self, exp: PpoExperience) {
        self.ppo.store_raw(PpoStoreRawInput {
            state: exp.state,
            seq_len: exp.seq_len,
            pity: exp.pity,
            action: exp.action,
            log_prob: exp.log_prob,
            reward: exp.reward,
            done: exp.done,
            value: exp.value,
        });
    }

    pub fn train_step(&mut self, current_lr: f64) -> Result<bool, TrainingError> {
        if self.ppo.memory.states_raw.len() < self.ppo.batch_size {
            return Ok(false);
        }
        self.ppo.update(current_lr)?;
        self.steps_done += 1;
        Ok(true)
    }

    pub fn sync_to(&self, shared: &std::sync::RwLock<ActorCritic>) {
        for attempt in 0..3u64 {
            if let Ok(mut guard) = shared.try_write() {
                guard.load_state_dict(&self.ppo.policy);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1 + attempt));
        }
        if let Ok(mut guard) = shared.write() {
            guard.load_state_dict(&self.ppo.policy);
        }
    }

    pub fn policy(&self) -> &ActorCritic {
        &self.ppo.policy
    }

    pub fn steps_done(&self) -> usize {
        self.steps_done
    }

    pub fn buffer_len(&self) -> usize {
        self.ppo.memory.states_raw.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute KL divergence KL(student || teacher) for two categorical distributions.
    /// KL(P||Q) = sum(P_i * (log(P_i) - log(Q_i)))
    fn kl_categorical(student: &[f64], teacher: &[f64]) -> f64 {
        assert_eq!(student.len(), teacher.len());
        student
            .iter()
            .zip(teacher.iter())
            .map(|(p, q)| {
                if *p > 0.0 && *q > 0.0 {
                    p * (p.ln() - q.ln())
                } else {
                    0.0
                }
            })
            .sum()
    }

    #[test]
    fn kl_divergence_identical_distributions_is_zero() {
        // When student and teacher are identical, KL should be exactly 0
        let p = vec![0.5, 0.5];
        let kl = kl_categorical(&p, &p);
        assert!(
            (kl - 0.0).abs() < 1e-9,
            "KL divergence of identical distributions should be 0, got {}",
            kl
        );

        let p2 = vec![0.9, 0.1];
        let kl2 = kl_categorical(&p2, &p2);
        assert!(
            (kl2 - 0.0).abs() < 1e-9,
            "KL divergence of identical distributions should be 0, got {}",
            kl2
        );

        let p3 = vec![0.25, 0.25, 0.25, 0.25];
        let kl3 = kl_categorical(&p3, &p3);
        assert!(
            (kl3 - 0.0).abs() < 1e-9,
            "KL divergence of identical distributions should be 0, got {}",
            kl3
        );
    }

    #[test]
    fn kl_divergence_is_not_symmetric() {
        // KL is NOT symmetric: KL(student||teacher) != KL(teacher||student)
        let student = vec![0.5, 0.5];
        let teacher = vec![0.9, 0.1];

        let kl_forward = kl_categorical(&student, &teacher);
        let kl_reverse = kl_categorical(&teacher, &student);

        // They should be different (KL is not symmetric)
        assert!(
            (kl_forward - kl_reverse).abs() > 1e-6,
            "KL should not be symmetric: forward={}, reverse={}",
            kl_forward,
            kl_reverse
        );

        // Verify specific values: KL([0.5,0.5]||[0.9,0.1]) should be positive
        assert!(
            kl_forward > 0.0,
            "KL divergence should be positive, got {}",
            kl_forward
        );

        // Check manual computation: 0.5*ln(0.5/0.9) + 0.5*ln(0.5/0.1) ≈ 0.5108
        let expected = 0.5 * (0.5_f64.ln() - 0.9_f64.ln()) + 0.5 * (0.5_f64.ln() - 0.1_f64.ln());
        assert!(
            (kl_forward - expected).abs() < 1e-6,
            "KL forward = {}, expected = {}",
            kl_forward,
            expected
        );
    }

    #[test]
    fn sum_f64_matches_scalar() {
        let values = vec![1.0, -2.5, 3.25, 4.0, -5.0, 6.5, 7.75, -8.0, 9.0];
        let mut expected = 0.0;
        for v in &values {
            expected += v;
        }
        let got = sum_f64(&values);
        assert!((got - expected).abs() < 1e-9);
    }

    #[test]
    fn normalize_slice_zero_mean_unit_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mean = sum_f64(&values) / values.len() as f64;
        let std = (sum_sq_diff(&values, mean) / values.len() as f64).sqrt() + 1e-8;
        let norm = normalize_slice(&values, mean, std);
        let norm_mean = sum_f64(&norm) / norm.len() as f64;
        let norm_std = (sum_sq_diff(&norm, 0.0) / norm.len() as f64).sqrt();
        assert!(norm_mean.abs() < 1e-9);
        assert!((norm_std - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_actor_critic_shapes() {
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default(), 64, 2);

        // Case 1: 1D input [DIM] (e.g. [8])
        let state_1d = Tensor::new_f32(vec![0.5; DIM], vec![DIM]);
        let pity = vec![0];
        let _ = policy.forward_actor(&state_1d, &pity);
        let _ = policy.forward_critic(&state_1d, &pity);

        // Case 2: 2D input [Seq, DIM] (e.g. [5, 8])
        let seq_len = 5;
        let state_2d = Tensor::new_f32(vec![0.5; seq_len * DIM], vec![seq_len, DIM]);
        let _ = policy.forward_actor(&state_2d, &pity);
        let _ = policy.forward_critic(&state_2d, &pity);

        // Case 3: 3D input [1, Seq, DIM]
        let state_3d = Tensor::new_f32(vec![0.5; seq_len * DIM], vec![1, seq_len, DIM]);
        let _ = policy.forward_actor(&state_3d, &pity);
        let _ = policy.forward_critic(&state_3d, &pity);
    }

    #[test]
    fn online_trainer_from_policy_preserves_initial_weights() {
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default(), 64, 2);
        let expected = policy.parameters()[0].data_f32().clone();

        let trainer = OnlinePpoTrainer::from_policy(policy, 2, 128);
        let got = trainer.ppo.policy.parameters()[0].data_f32().clone();

        assert_eq!(got, expected);
    }

    #[test]
    fn cpu_device_keeps_online_ppo_entirely_on_cpu() {
        let policy = ActorCritic::new(42, &AchfConfig::default(), 16, 1);
        let trainer = OnlinePpoTrainer::from_policy_on_device(policy, 1, 4, ComputeDevice::Cpu);

        assert!(!trainer.ppo.use_cuda);
        assert!(!trainer.ppo.policy.uses_cuda());
    }

    #[cfg(cuda)]
    #[test]
    fn ppo_gpu_adam_partial_failure_is_hard_and_poisons_optimizer() {
        if !crate::cuda::is_available() {
            return;
        }
        let first = Tensor::new_f32(vec![1.0], vec![1]);
        let second = Tensor::new_f32(vec![2.0], vec![1]);
        first.grad_write_f32()[0] = 0.25;
        second.grad_write_f32()[0] = 0.25;
        let first = first.to_cuda().expect("first CUDA upload should succeed");
        let second = second.to_cuda().expect("second CUDA upload should succeed");
        let mut optimizer =
            GpuAdam::new(vec![first, second], 0.001).expect("GPU Adam should initialize");
        let before = crate::cuda::runtime_stats();

        crate::cuda::kernels::inject_adam_failure_after(1);
        let failure = optimizer.step();
        let after_failure = crate::cuda::runtime_stats();

        assert!(failure.is_err(), "injected CUDA failure must be returned");
        assert!(optimizer.poisoned);
        assert!(
            optimizer.params[0].data_to_f32_vec()[0] < 1.0,
            "first parameter demonstrates that the failed step may be partial"
        );
        assert_eq!(
            optimizer.params[1].data_to_f32_vec()[0],
            2.0,
            "second parameter must remain untouched at the injected failure"
        );
        assert_eq!(
            after_failure.optimizer_attempts,
            before.optimizer_attempts + 1
        );
        assert_eq!(
            after_failure.optimizer_successes,
            before.optimizer_successes
        );
        assert_eq!(
            after_failure.optimizer_fallback_param,
            before.optimizer_fallback_param + 1
        );

        let retry = optimizer.step();
        assert!(matches!(
            retry,
            Err(crate::cuda::error::CudaError::InvalidInput {
                message: "optimizer is poisoned after a previous CUDA failure",
                ..
            })
        ));
    }

    #[test]
    fn online_trainer_from_policy_preserves_achf_penalty_interval() {
        let achf = crate::config::AchfConfig {
            enabled: true,
            apply_ffn: true,
            lambda_ortho: 0.001,
            ortho_penalty_freq: 64,
            ..crate::config::AchfConfig::default()
        };
        let policy = ActorCritic::new(42, &achf, 16, 1);

        let trainer = OnlinePpoTrainer::from_policy(policy, 2, 128);

        assert_eq!(trainer.ppo.achf_orthogonal_penalty_interval, 256);
    }

    #[test]
    fn ema_update_blends_teacher_student_with_decay_0_5() {
        // Use same seed for both so they start with identical weights
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default(), 64, 2);
        let mut ppo = Ppo::from_policy(policy, 2, 128);

        // Create EMA as separate instance with same seed (not clone, to avoid Arc sharing)
        ppo.ema_policy = Some(ActorCritic::new(
            42,
            &crate::config::AchfConfig::default(),
            64,
            2,
        ));
        ppo.distill_ema_decay = 0.5;

        // Capture original teacher values (initial = policy values since same seed)
        let teacher_before = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f32()
            .clone();

        // Modify student first parameter to 1.0
        let params = ppo.policy.parameters();
        let mut data = params[0].data_write_f32();
        for val in data.iter_mut() {
            *val = 1.0f32;
        }
        drop(data); // Release lock before EMA update to avoid deadlock
        drop(params); // Explicitly drop to release all locks

        // Perform EMA update
        ppo.update_ema_teacher();

        // With decay=0.5, teacher_new = 0.5 * teacher_old + 0.5 * student
        let teacher_after = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f32()
            .clone();

        for (before, after) in teacher_before.iter().zip(teacher_after.iter()) {
            let expected = 0.5 * before + 0.5 * 1.0f32;
            assert!((after - expected).abs() < 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "EMA teacher/student parameter shape mismatch at index")]
    fn ema_update_rejects_parameter_shape_mismatch() {
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default(), 64, 2);
        let mut ppo = Ppo::from_policy(policy, 2, 128);
        ppo.ema_policy = Some(ActorCritic::new(
            42,
            &crate::config::AchfConfig::default(),
            32,
            2,
        ));

        ppo.update_ema_teacher();
    }

    #[test]
    fn ema_update_refreshes_achf_candidate_from_updated_reference() {
        let achf = AchfConfig {
            enabled: true,
            candidate_mode: "sparse".to_string(),
            candidate_refresh_freq: 1,
            prune_threshold: 0.5,
            candidate_min_sparsity: 0.0,
            candidate_max_relative_error: 1.0,
            ..Default::default()
        };
        let policy = ActorCritic::new(42, &achf, 16, 1);
        let mut ppo = Ppo::from_policy(policy, 2, 128);
        ppo.ema_policy = Some(ActorCritic::new(42, &achf, 16, 1));
        ppo.distill_ema_decay = 0.5;

        let candidate_before = ppo.ema_policy.as_ref().unwrap().backbone.blocks[0]
            .achf_ffn
            .as_ref()
            .unwrap()
            .sparse_weight
            .as_ref()
            .unwrap()
            .weight
            .data_to_f32_vec();
        {
            let student_achf = ppo.policy.backbone.blocks[0].achf_ffn.as_ref().unwrap();
            student_achf.weight.weight.data_write_f32().fill(2.0);
        }

        ppo.update_ema_teacher();

        let teacher_achf = ppo.ema_policy.as_ref().unwrap().backbone.blocks[0]
            .achf_ffn
            .as_ref()
            .unwrap();
        let reference = teacher_achf.weight.weight.data_to_f32_vec();
        let candidate = teacher_achf
            .sparse_weight
            .as_ref()
            .unwrap()
            .weight
            .data_to_f32_vec();
        assert_ne!(candidate, candidate_before);
        assert!(candidate
            .iter()
            .zip(reference.iter())
            .all(|(candidate, reference)| *candidate == 0.0 || candidate == reference));
        assert_eq!(teacher_achf.state.read().unwrap().step, 1);
    }

    #[test]
    fn ema_update_applies_to_all_parameter_tensors() {
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default(), 64, 2);
        let mut ppo = Ppo::from_policy(policy, 2, 128);

        // Create separate EMA with same seed (not clone)
        ppo.ema_policy = Some(ActorCritic::new(
            42,
            &crate::config::AchfConfig::default(),
            64,
            2,
        ));
        ppo.distill_ema_decay = 0.5;

        let num_params = ppo.policy.parameters().len();
        assert!(num_params > 0, "Should have parameters");

        // Record first and last tensor values before update
        let before_first = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f32()
            .clone();
        let before_last = ppo.ema_policy.as_ref().unwrap().parameters()[num_params - 1]
            .data_f32()
            .clone();

        // Set all student parameters to 2.0
        for param in ppo.policy.parameters() {
            let mut data = param.data_write_f32();
            for val in data.iter_mut() {
                *val = 2.0f32;
            }
        }

        ppo.update_ema_teacher();

        // Check first and last tensor values after update
        let after_first = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f32()
            .clone();
        let after_last = ppo.ema_policy.as_ref().unwrap().parameters()[num_params - 1]
            .data_f32()
            .clone();

        // First tensor should have changed
        let first_changed = before_first
            .iter()
            .zip(after_first.iter())
            .any(|(b, a)| (*b - *a).abs() > 1e-9);
        assert!(first_changed, "First parameter tensor was not updated");

        // Last tensor should have changed
        let last_changed = before_last
            .iter()
            .zip(after_last.iter())
            .any(|(b, a)| (*b - *a).abs() > 1e-9);
        assert!(last_changed, "Last parameter tensor was not updated");
    }

    #[test]
    fn ema_approaches_student_after_multiple_updates() {
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default(), 64, 2);
        let mut ppo = Ppo::from_policy(policy, 2, 128);

        // Create separate EMA with same seed
        ppo.ema_policy = Some(ActorCritic::new(
            42,
            &crate::config::AchfConfig::default(),
            64,
            2,
        ));
        ppo.distill_ema_decay = 0.5;

        // Set all student params to 10.0
        for param in ppo.policy.parameters() {
            let mut data = param.data_write_f32();
            for val in data.iter_mut() {
                *val = 10.0f32;
            }
        }

        // Perform many EMA updates
        for _ in 0..100 {
            ppo.update_ema_teacher();
        }

        // After 100 updates with decay=0.5, EMA should be very close to student (10.0)
        let teacher_final = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f32()
            .clone();

        for val in teacher_final.iter() {
            assert!(
                (*val - 10.0f32).abs() < 1e-6,
                "EMA did not converge: got {}",
                val
            );
        }
    }

    #[test]
    fn ppo_distillation_does_not_backpropagate_into_ema_teacher() {
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default(), 16, 1);
        let mut ppo = Ppo::from_policy(policy, 1, 2);
        ppo.ema_policy = Some(ActorCritic::new(
            43,
            &crate::config::AchfConfig::default(),
            16,
            1,
        ));
        ppo.distill_kl_coef = 0.1;
        ppo.distill_warmup_steps = 0;

        for i in 0..2 {
            ppo.store_raw(PpoStoreRawInput {
                state: vec![0.05 + i as f64 * 0.01; DIM],
                seq_len: 1,
                pity: vec![0],
                action: i % ACTION_SPACE,
                log_prob: 0.0,
                reward: if i == 0 { 1.0 } else { 0.5 },
                done: i == 1,
                value: 0.0,
            });
        }

        let _ = ppo.update(0.0003);

        let teacher = ppo.ema_policy.as_ref().expect("EMA teacher exists");
        for (idx, param) in teacher.parameters().iter().enumerate() {
            let grad = param.grad_to_f32_vec();
            assert!(
                grad.iter().all(|v| v.abs() <= 1e-8),
                "EMA teacher parameter {idx} received gradient"
            );
        }
    }

    #[test]
    fn softmax_sample_top_k_zero_like_full_softmax() {
        // top_k=0 should behave identically to full softmax (no truncation)
        let logits: Vec<f32> = (0..ACTION_SPACE).map(|i| (i + 1) as f32).collect();
        let probs_0 = softmax_probs(&logits, 0);
        let probs_full = softmax_probs(&logits, ACTION_SPACE);
        assert_eq!(probs_0, probs_full);
    }

    #[test]
    fn softmax_sample_top_k_gte_action_space_like_full_softmax() {
        // top_k >= ACTION_SPACE should behave identically to full softmax
        let logits: Vec<f32> = (0..ACTION_SPACE).map(|i| (i + 1) as f32).collect();
        let probs_large = softmax_probs(&logits, ACTION_SPACE + 10);
        let probs_full = softmax_probs(&logits, ACTION_SPACE);
        assert_eq!(probs_large, probs_full);
    }

    #[test]
    fn softmax_sample_top_k_boundary_identical_values() {
        // When multiple logits have identical values at the threshold boundary,
        // all logits with that value should be treated consistently.
        // Test case: logits where 3rd and 4th highest values differ
        // [5.0, 4.0, 4.0, 3.0, 2.0] with top_k=3
        // Sorted: [5.0, 4.0, 4.0, 3.0, 2.0], threshold = 4.0
        // Correct behavior with <: indices 0,1,2 survive and 3,4 are masked.
        let logits = vec![5.0, 4.0, 4.0, 3.0, 2.0];
        let top_k = 3;

        let probs = softmax_probs(&logits, top_k);

        assert!(probs[0] > 0.0, "Index 0 (5.0) should survive");
        assert!(probs[1] > 0.0, "Index 1 (4.0) should survive");
        assert!(probs[2] > 0.0, "Index 2 (4.0) should survive");
        assert_eq!(probs[3], 0.0, "Index 3 (3.0) should be masked");
        assert_eq!(probs[4], 0.0, "Index 4 (2.0) should be masked");
    }

    #[test]
    fn ppo_from_policy_clamps_zero_batch_size() {
        let policy = ActorCritic::new(7, &crate::config::AchfConfig::default(), 64, 2);
        let ppo = Ppo::from_policy(policy, 1, 0);
        assert_eq!(ppo.batch_size, 1);
    }

    #[test]
    fn ppo_update_handles_minibatch_with_vectorized_loss() {
        let policy = ActorCritic::new(7, &crate::config::AchfConfig::default(), 16, 1);
        let mut ppo = Ppo::from_policy(policy, 1, 4);

        for i in 0..8 {
            let seq_len = 1 + (i % 3);
            ppo.store_raw(PpoStoreRawInput {
                state: vec![0.01 * (i as f64 + 1.0); seq_len * DIM],
                seq_len,
                pity: vec![0; seq_len],
                action: i % ACTION_SPACE,
                log_prob: -(ACTION_SPACE as f64).ln(),
                reward: if i % 2 == 0 { 1.0 } else { -0.1 },
                done: i % 4 == 3,
                value: 0.0,
            });
        }

        let loss = ppo.update(0.0003).unwrap();

        assert!(loss.is_finite(), "PPO update loss should stay finite");
        assert!(
            ppo.memory.states_raw.is_empty(),
            "PPO update should consume rollout memory"
        );
    }

    #[test]
    fn ppo_batch_forward_backpropagates_into_backbone_on_cpu() {
        let policy = ActorCritic::new(9, &crate::config::AchfConfig::default(), 64, 1);
        let states = Tensor::new_f32(vec![0.05; 2 * 2 * DIM], vec![2, 2, DIM]);

        let (logits, values) = policy.forward_actor_critic_batch(&states);
        (logits.sum() + values.sum()).backward();

        let embed_grad = policy.backbone.embed.weight.grad_to_f32_vec();
        assert!(
            embed_grad.iter().any(|g| g.abs() > 1e-9),
            "PPO backbone embed gradient should not be cut off by last_token"
        );
    }

    #[test]
    fn actor_critic_freeze_prunes_backbone_achf_for_cache_hits() {
        let achf = crate::config::AchfConfig {
            mode: "fixed_cached".to_string(),
            cache_min_reuse: 0,
            enabled: true,
            cache_cost_bias: 0.0,
            infer_gate: "one".to_string(),
            prune_threshold: 0.01,
            ortho_penalty_freq: 0,
            ..Default::default()
        };
        let mut policy = ActorCritic::new(7, &achf, 16, 1);
        assert!(policy.backbone.blocks[0]
            .achf_ffn
            .as_ref()
            .is_some_and(|achf| achf.sparse_weight.is_some()));

        assert!(policy.backbone.blocks[0]
            .achf_ffn
            .as_ref()
            .is_some_and(|achf| { achf.cache.read().unwrap().dense.is_none() }));

        policy.freeze_achf_for_inference();
        assert!(policy.backbone.blocks[0]
            .achf_ffn
            .as_ref()
            .is_some_and(|achf| achf.sparse_weight.is_some()));

        let state = vec![0.1; DIM];
        let _ = policy.step_inference(&state, 0);
        let stats = policy.achf_cache_stats_aggregate();
        // Default config applies ACHF to both the FFN and the MLA w_o
        // projection, so a single block contributes two ACHF calls.
        assert_eq!(stats.calls, 2);
        assert_eq!(stats.candidate_paths, 2);
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.dense_paths, 0);
        assert_eq!(stats.sparse_paths, 0);
    }

    #[test]
    fn ppo_candidate_calibration_uses_rollouts_and_preserves_reference() {
        let achf = crate::config::AchfConfig {
            enabled: true,
            candidate_mode: "sparse".to_string(),
            candidate_target_sparsity: 0.75,
            prune_threshold: 0.0,
            candidate_min_sparsity: 0.5,
            candidate_max_output_relative_error: 1.0,
            candidate_min_calibration_samples: 1,
            candidate_calibration_steps: 2,
            candidate_calibration_max_samples: 16,
            apply_attn: false,
            apply_ffn: true,
            ..Default::default()
        };
        let policy = ActorCritic::new(21, &achf, 16, 1);
        let reference_before = policy.backbone.blocks[0]
            .achf_ffn
            .as_ref()
            .unwrap()
            .weight
            .weight
            .data_to_f32_vec();
        let mut ppo = Ppo::from_policy(policy, 1, 4);
        ppo.achf_calibration_states = (0..12)
            .map(|sample| {
                let sequence_length = 1 + sample % 3;
                (
                    vec![0.01 * (sample as f64 + 1.0); sequence_length * DIM],
                    sequence_length,
                )
            })
            .collect();

        ppo.calibrate_achf_candidates().unwrap();

        let layer = ppo.policy.backbone.blocks[0].achf_ffn.as_ref().unwrap();
        assert_eq!(layer.weight.weight.data_to_f32_vec(), reference_before);
        assert!(layer.candidate_calibration.calibrated);
        assert_eq!(layer.candidate_calibration.steps, 1);
        assert!(
            layer.candidate_calibration.output_relative_error
                <= achf.candidate_max_output_relative_error
        );
        assert!(layer.candidate_calibration.output_samples > 0);
        assert!(layer.candidate_calibration.trace.len() >= 2);
        assert_eq!(layer.candidate_calibration.masked_weight_max_abs, 0.0);
        assert_eq!(layer.candidate_calibration.masked_gradient_max_abs, 0.0);
        assert_eq!(layer.candidate_calibration.masked_moment_max_abs, 0.0);
    }
}
