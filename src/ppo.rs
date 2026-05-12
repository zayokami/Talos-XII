use crate::autograd::Tensor;
use crate::config::{AchfConfig, Config};
use crate::env_net::EnvNet;
use crate::neural::DIM;
use crate::nn::{Linear, Module};
use crate::rng::Rng;
use crate::sim::{build_features, env_net_env, prob_6, PpoExperience, PullState};
use crate::transformer::{KVCache, LuckTransformer};
use crate::utils::{
    create_bar, normalize_slice, sum_f64, sum_sq_diff, ACTIONS, ACTION_SPACE, EPISODE_MAX_PULLS,
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
const VALUE_COEF: f64 = 0.5;
const ENTROPY_COEF: f64 = 0.01;
const EARLY_UP_BONUS_THRESHOLD_1: usize = 80;
const EARLY_UP_BONUS_THRESHOLD_2: usize = 50;

#[derive(Default)]
struct CachedStepScratch {
    last: Vec<f64>,
    logits: Vec<f64>,
    value: Vec<f64>,
}

thread_local! {
    static CACHED_STEP_SCRATCH: RefCell<CachedStepScratch> =
        RefCell::new(CachedStepScratch::default());
}

/// Pre-allocated scalar tensors for training loop to avoid per-sample heap allocations.
struct TrainingScratch {
    old_log_prob: Tensor,
    advantage: Tensor,
    return_val: Tensor,
}

impl Default for TrainingScratch {
    fn default() -> Self {
        TrainingScratch {
            old_log_prob: Tensor::zeros(vec![1]),
            advantage: Tensor::zeros(vec![1]),
            return_val: Tensor::zeros(vec![1]),
        }
    }
}

thread_local! {
    static TRAINING_SCRATCH: RefCell<TrainingScratch> = RefCell::new(TrainingScratch::default());
}

// Softmax + categorical sample over fixed ACTION_SPACE logits.
// Returns (action_index, log_prob_of_action).
// If top_k > 0, only the top_k logits are kept (others set to -inf).
#[inline]
fn softmax_sample(logits: &[f64], top_k: usize) -> (usize, f64) {
    if logits.len() != ACTION_SPACE {
        let fallback_prob = 1.0 / ACTION_SPACE as f64;
        return (0, fallback_prob.ln());
    }

    // Find max for numerical stability
    let mut max_l = f64::NEG_INFINITY;
    for &v in logits {
        if v > max_l {
            max_l = v;
        }
    }

    // Top-k truncation: set non-top-k logits to -inf
    let mut probs = [0.0; ACTION_SPACE];
    if top_k > 0 && top_k < ACTION_SPACE {
        // Find top_k values
        let mut sorted_logits = logits.to_vec();
        sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted_logits[top_k.saturating_sub(1)];

        // Compute softmax with top-k masking
        let mut sum_exp = 0.0;
        for (i, prob) in probs.iter_mut().enumerate() {
            if logits[i] < threshold {
                *prob = 0.0; // masked out
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
            let uniform = 1.0 / ACTION_SPACE as f64;
            probs.fill(uniform);
        }
    } else {
        // Full softmax (top_k disabled or >= ACTION_SPACE)
        let mut sum_exp = 0.0;
        for (i, prob) in probs.iter_mut().enumerate() {
            *prob = (logits[i] - max_l).exp();
            sum_exp += *prob;
        }
        for prob in probs.iter_mut() {
            *prob /= sum_exp;
        }
    }

    // Categorical sampling
    let mut r = rand::random::<f64>();
    let mut idx = ACTION_SPACE - 1;
    for (i, &p) in probs.iter().enumerate() {
        if r < p {
            idx = i;
            break;
        }
        r -= p;
    }
    let log_prob = probs[idx].max(f64::MIN_POSITIVE).ln();
    (idx, log_prob)
}

/// Actor-Critic model combining a LuckTransformer backbone with action/value heads.
#[derive(Clone, Serialize, Deserialize)]
pub struct ActorCritic {
    pub backbone: LuckTransformer,
    pub actor_head: Linear,
    pub critic_head: Linear,
}

impl ActorCritic {
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
    /// Defaults (2048, 4) match production; tests should pass smaller values.
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

    pub fn update_achf_after_backward(&self) {
        self.backbone.update_achf_after_backward();
    }

    pub fn freeze_achf_for_inference(&self) {
        self.backbone.freeze_achf_for_inference();
    }

    pub fn achf_orthogonal_penalty(&self) -> Option<Tensor> {
        self.backbone.achf_orthogonal_penalty()
    }

    // Returns (action_idx, log_prob, value)
    pub fn step(&self, state: &Tensor, pity: &[usize], top_k: usize) -> (usize, f64, f64) {
        let (logits, value) = self.forward_actor_critic(state, pity);
        let logits_data = logits.data_f64();
        let (action_idx, log_prob) = softmax_sample(&logits_data, top_k);
        let val = value.data_f64()[0];
        (action_idx, log_prob, val)
    }

    // Fast inference without Autograd graph
    pub fn step_inference(&self, state: &[f64], top_k: usize) -> usize {
        let seq = self.backbone.forward_inference(state);
        let last = self.backbone.last_token_inference(&seq);
        let logits = self.actor_head.forward_inference(&last);
        softmax_sample(&logits, top_k).0
    }

    pub fn step_inference_cached_with_value(
        &self,
        state: &[f64],
        kv_cache: &mut [KVCache],
        start_pos: usize,
        top_k: usize,
    ) -> (usize, f64, f64) {
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

    pub fn step_inference_cached(
        &self,
        state: &[f64],
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

    pub fn prune_cache(&self, kv_cache: &mut [KVCache], max_len: usize) {
        self.backbone.prune_kv_cache(kv_cache, max_len);
    }

    pub fn achf_cache_stats_iter(&self) -> impl Iterator<Item = crate::achf::AchfCacheStats> + '_ {
        self.backbone.achf_cache_stats_iter()
    }

    pub fn achf_cache_stats_aggregate(&self) -> crate::achf::AchfCacheStats {
        self.backbone.achf_cache_stats_aggregate()
    }

    pub fn snapshot_achf(&self) -> Option<crate::achf::AchfStateSnapshot> {
        self.backbone.snapshot_achf()
    }

    pub fn forward_inference_forced_path(&self, x: &[f64], forced_path: u8) -> Vec<f64> {
        self.backbone.forward_inference_forced_path(x, forced_path)
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
}

#[cfg(cuda)]
struct GpuAdam {
    params: Vec<Tensor>,
    m: Vec<std::sync::Arc<crate::cuda::memory::DevicePtr<f64>>>,
    v: Vec<std::sync::Arc<crate::cuda::memory::DevicePtr<f64>>>,
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
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in &params {
            let len = p.data.len();
            if p.device != crate::autograd::Device::Cuda {
                return None;
            }
            let d_m = crate::cuda::memory::alloc::<f64>(len).ok()?;
            let d_v = crate::cuda::memory::alloc::<f64>(len).ok()?;
            let zeros = vec![0.0_f64; len];
            crate::cuda::memory::copy_h2d(&d_m, &zeros).ok()?;
            crate::cuda::memory::copy_h2d(&d_v, &zeros).ok()?;
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

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self) {
        self.t += 1;
        crate::cuda::record_optimizer_attempt();

        // Compute global gradient norm on CPU (copy grads from GPU)
        let mut total_norm = 0.0;
        for param in &self.params {
            let grad = param.grad_to_f64_vec();
            total_norm += crate::simd::dot_product(&grad, &grad);
        }
        total_norm = total_norm.sqrt();
        let clip_coef = if total_norm > 1.0 {
            1.0 / total_norm
        } else {
            1.0
        };

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        let mut all_ok = true;
        for (i, param) in self.params.iter().enumerate() {
            let len = param.data.len();
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
            let d_m = &self.m[i];
            let d_v = &self.v[i];

            if crate::cuda::kernels::adam_step(
                &d_params,
                &d_grads,
                d_m,
                d_v,
                len,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bias_correction1,
                bias_correction2,
                clip_coef,
            )
            .is_err()
            {
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
    fn set_lr(&mut self, lr: f64) {
        match self {
            Optimizer::Cpu(o) => o.set_lr(lr),
            #[cfg(cuda)]
            Optimizer::Gpu(o) => o.set_lr(lr),
        }
    }

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
}

impl Ppo {
    pub fn new(seed: u64, k_epochs: usize, batch_size: usize, config: &Config) -> Self {
        let policy = ActorCritic::new_with_config(config, seed);
        Self::from_policy(policy, k_epochs, batch_size)
    }

    pub fn from_policy(policy: ActorCritic, k_epochs: usize, batch_size: usize) -> Self {
        #[cfg(cuda)]
        let policy = {
            let mut p = policy;
            p.to_cuda();
            p
        };
        let params = policy.parameters();
        #[cfg(cuda)]
        let optimizer = GpuAdam::new(params, 0.0003)
            .map(Optimizer::Gpu)
            .unwrap_or_else(|| Optimizer::Cpu(Adam::new(policy.parameters(), 0.0003)));
        #[cfg(not(cuda))]
        let optimizer = Optimizer::Cpu(Adam::new(params, 0.0003));
        let safe_batch_size = batch_size.max(1);
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

        for (ema_p, stud_p) in ema_params.iter().zip(student_params.iter()) {
            let mut ema_data = ema_p.data_write_f64();
            let stud_data = stud_p.data_f64();
            for (e, s) in ema_data.iter_mut().zip(stud_data.iter()) {
                *e = decay * (*e) + inv * (*s);
            }
        }
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

    pub fn update(&mut self, current_lr: f64) -> f64 {
        if self.memory.states_raw.is_empty() {
            return 0.0;
        }

        // Update Learning Rate
        self.optimizer.set_lr(current_lr);

        let len = self.memory.states_raw.len();
        let states_raw = std::mem::take(&mut self.memory.states_raw);
        let state_lens = std::mem::take(&mut self.memory.state_lens);
        let _pities = std::mem::take(&mut self.memory.pities);
        let actions = std::mem::take(&mut self.memory.actions);
        let log_probs = std::mem::take(&mut self.memory.log_probs);
        let rewards = std::mem::take(&mut self.memory.rewards);
        let is_terminals = std::mem::take(&mut self.memory.is_terminals);
        let values = std::mem::take(&mut self.memory.values);
        let states: Vec<Tensor> = states_raw
            .into_iter()
            .zip(state_lens)
            .map(|(data, seq_len)| Tensor::new(data, vec![seq_len, DIM]))
            .collect();
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
        let value_coef_tensor = Tensor::new(vec![VALUE_COEF], vec![1]);
        let entropy_coef_tensor = Tensor::new(vec![ENTROPY_COEF], vec![1]);
        #[cfg(cuda)]
        let value_coef_tensor = match value_coef_tensor.to_cuda() {
            Ok(t) => t,
            Err(_) => value_coef_tensor,
        };
        #[cfg(cuda)]
        let entropy_coef_tensor = match entropy_coef_tensor.to_cuda() {
            Ok(t) => t,
            Err(_) => entropy_coef_tensor,
        };

        #[cfg(cuda)]
        {
            TRAINING_SCRATCH.with(|s| {
                let mut scratch = s.borrow_mut();
                if let Ok(t) = scratch.old_log_prob.to_cuda() {
                    scratch.old_log_prob = t;
                }
                if let Ok(t) = scratch.advantage.to_cuda() {
                    scratch.advantage = t;
                }
                if let Ok(t) = scratch.return_val.to_cuda() {
                    scratch.return_val = t;
                }
            });
        }

        let mut loss_sum = 0.0_f64;
        let mut loss_count = 0usize;
        for _ in 0..self.k_epochs {
            indices.shuffle(&mut rand::rng());
            let mut approx_kl = 0.0;
            let mut batch_count = 0.0;
            let mut early_stop = false;

            for chunk in indices.chunks(self.batch_size) {
                self.optimizer.zero_grad();

                let batch_len = chunk.len();
                let inv_batch = 1.0 / batch_len as f64;

                // Batched forward: pad states to max seq_len and stack
                let max_seq_len = chunk.iter().map(|&i| states[i].shape[0]).max().unwrap_or(1);
                let mut batch_data = Vec::with_capacity(batch_len * max_seq_len * DIM);
                for &i in chunk {
                    let state_data = states[i].data_f64();
                    let seq_len = states[i].shape[0];
                    batch_data.extend_from_slice(&state_data);
                    if seq_len < max_seq_len {
                        batch_data.resize(batch_data.len() + (max_seq_len - seq_len) * DIM, 0.0);
                    }
                }
                let batch_states = Tensor::new(batch_data, vec![batch_len, max_seq_len, DIM]);
                #[cfg(cuda)]
                let batch_states = match batch_states.to_cuda() {
                    Ok(t) => t,
                    Err(_) => batch_states,
                };
                let (batch_logits, batch_values) =
                    self.policy.forward_actor_critic_batch(&batch_states);
                let batch_log_probs = batch_logits.log_softmax();
                let batch_log_probs_copy = batch_log_probs.clone();
                let batch_probs = batch_log_probs_copy.exp();
                let ones_action_1 = Tensor::new(vec![1.0; ACTION_SPACE], vec![ACTION_SPACE, 1]);
                #[cfg(cuda)]
                let ones_action_1 = match ones_action_1.to_cuda() {
                    Ok(t) => t,
                    Err(_) => ones_action_1,
                };
                let batch_entropy =
                    -(batch_probs.clone() * batch_log_probs_copy).matmul(&ones_action_1);

                // Batched teacher forward (if distilling)
                let teacher_batch_logits = if self.distill_kl_coef > 0.0 {
                    if let Some(ema) = self.ema_policy.as_ref() {
                        let (t_logits, _) = ema.forward_actor_critic_batch(&batch_states);
                        Some(t_logits)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let mut loss_accum = Tensor::zeros(vec![1]);
                let mut distill_accum = Tensor::zeros(vec![1]);
                #[cfg(cuda)]
                {
                    loss_accum = match loss_accum.to_cuda() {
                        Ok(t) => t,
                        Err(_) => loss_accum,
                    };
                    distill_accum = match distill_accum.to_cuda() {
                        Ok(t) => t,
                        Err(_) => distill_accum,
                    };
                }

                for (chunk_idx, &i) in chunk.iter().enumerate() {
                    let action_idx = actions[i];
                    let old_log_prob = log_probs[i];
                    let advantage = norm_advantages[i];
                    let return_val = returns[i];

                    let log_prob =
                        batch_log_probs.index_select(chunk_idx * ACTION_SPACE + action_idx);
                    let value = batch_values.index_select(chunk_idx);
                    let entropy = batch_entropy.index_select(chunk_idx);

                    let log_prob_val = log_prob.data_f64()[0];
                    let log_ratio_val = log_prob_val - old_log_prob;
                    let ratio_val = log_ratio_val.exp();
                    approx_kl += (ratio_val - 1.0) - log_ratio_val;
                    batch_count += 1.0;

                    // Use scratch tensors to avoid per-sample heap allocations
                    let old_log_prob_tensor = TRAINING_SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.old_log_prob.fill_(old_log_prob);
                        scratch.old_log_prob.clone()
                    });
                    let log_ratio = log_prob - old_log_prob_tensor;
                    let ratio = log_ratio.exp();

                    let adv_tensor = TRAINING_SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.advantage.fill_(advantage);
                        scratch.advantage.clone()
                    });
                    let surr1 = ratio.clone() * adv_tensor.clone();
                    let ratio_clipped = ratio.clip(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON);
                    let surr2 = ratio_clipped * adv_tensor;

                    let s1_val = surr1.data_f64()[0];
                    let s2_val = surr2.data_f64()[0];
                    let policy_loss = if s1_val < s2_val { surr1 } else { surr2 };

                    let ret_tensor = TRAINING_SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.return_val.fill_(return_val);
                        scratch.return_val.clone()
                    });
                    let value_err = value - ret_tensor;
                    let v_loss = (value_err.clone() * value_err).sum();

                    let loss = -policy_loss + v_loss * value_coef_tensor.clone()
                        - entropy * entropy_coef_tensor.clone();

                    loss_accum = loss_accum + loss;
                }

                // Batched distillation KL after the per-sample loop
                if let Some(ref teacher_logits) = teacher_batch_logits {
                    let teacher_log_probs = teacher_logits.log_softmax();
                    let kl_elements = batch_probs * (batch_log_probs.clone() - teacher_log_probs);
                    let total_kl = kl_elements.matmul(&ones_action_1).sum();
                    distill_accum = total_kl;
                }

                let batch_size_tensor = Tensor::new(vec![inv_batch], vec![1]);
                #[cfg(cuda)]
                let batch_size_tensor = match batch_size_tensor.to_cuda() {
                    Ok(t) => t,
                    Err(_) => batch_size_tensor,
                };
                let mut final_loss = loss_accum * batch_size_tensor.clone();
                // Add distillation loss after warmup: distill_coef * mean(kl_divs)
                if teacher_batch_logits.is_some()
                    && self.distill_update_counter >= self.distill_warmup_steps
                {
                    let distill_coef_tensor = Tensor::new(vec![self.distill_kl_coef], vec![1]);
                    #[cfg(cuda)]
                    let distill_coef_tensor = match distill_coef_tensor.to_cuda() {
                        Ok(t) => t,
                        Err(_) => distill_coef_tensor,
                    };
                    let distill_term = distill_accum * distill_coef_tensor * batch_size_tensor;
                    final_loss = final_loss + distill_term;
                }
                if let Some(reg) = self.policy.achf_orthogonal_penalty() {
                    final_loss = final_loss + reg;
                }
                loss_sum += final_loss.data_f64()[0];
                loss_count += 1;
                final_loss.backward();
                self.policy.update_achf_after_backward();
                self.optimizer.step();
                // Update EMA teacher after each batch for self-distillation
                self.update_ema_teacher();

                if batch_count > 0.0 && (approx_kl / batch_count) > target_kl * 1.5 {
                    early_stop = true;
                }

                if early_stop {
                    break;
                }
            }

            if early_stop {
                break;
            }

            // Early Stopping check
            if batch_count > 0.0 {
                approx_kl /= batch_count;
                if approx_kl > target_kl * 1.5 {
                    break;
                }
            }
        }
        if loss_count > 0 {
            loss_sum / loss_count as f64
        } else {
            0.0
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
    episode_reward: f64,
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
            state_struct: PullState {
                pity_6: 0,
                total_pulls_in_pool: 0,
                has_obtained_up: false,
                streak_4_star: 0,
                loss_streak: 0,
            },
            env_noise,
            env_bias,
            pulls_done: 0,
            history_buffer: VecDeque::with_capacity(context_len),
            pity_buffer: VecDeque::with_capacity(context_len),
            flat_data: Vec::with_capacity(context_len * DIM),
            pity_vec: Vec::with_capacity(context_len),
            kv_cache: caches,
            episode_reward: 0.0,
            rng,
        }
    }

    fn reset(&mut self, env_net: &EnvNet) {
        self.history_buffer.clear();
        self.pity_buffer.clear();
        for cache in self.kv_cache.iter_mut() {
            cache.clear();
        }
        self.state_struct = PullState {
            pity_6: 0,
            total_pulls_in_pool: 0,
            has_obtained_up: false,
            streak_4_star: 0,
            loss_streak: 0,
        };
        let (env_noise, env_bias) = env_net_env(env_net, &mut self.rng, 0, 0, 0, 0);
        self.env_noise = env_noise;
        self.env_bias = env_bias;
        self.pulls_done = 0;
        self.episode_reward = 0.0;
    }

    fn step(
        &mut self,
        policy: &ActorCritic,
        env_net: &EnvNet,
        config: &Config,
        context_len: usize,
    ) -> PpoStepResult {
        let current_state_raw = build_features(
            self.state_struct.pity_6,
            self.pulls_done,
            self.env_noise,
            self.state_struct.streak_4_star,
            self.env_bias,
            self.state_struct.loss_streak,
            config,
        )
        .to_vec();

        let current_pity = self.state_struct.pity_6;

        self.history_buffer.push_back(current_state_raw);
        self.pity_buffer.push_back(current_pity);
        if self.history_buffer.len() > context_len {
            self.history_buffer.pop_front();
            self.pity_buffer.pop_front();
            policy.prune_cache(&mut self.kv_cache, context_len);
        }

        let seq_len = self.history_buffer.len();
        self.flat_data.clear();
        for s in self.history_buffer.iter() {
            self.flat_data.extend_from_slice(s);
        }
        self.pity_vec.clear();
        self.pity_vec.extend(self.pity_buffer.iter().copied());
        let token = self
            .history_buffer
            .back()
            .expect("history_buffer should not be empty after push")
            .as_slice();
        let (action_idx, log_prob, val) = policy.step_inference_cached_with_value(
            token,
            &mut self.kv_cache,
            seq_len - 1,
            config.ppo_top_k,
        );

        let luck_modifier = ACTIONS[action_idx];
        let base_prob_6 = prob_6(self.state_struct.pity_6, config);
        let final_prob_6 = (base_prob_6 + luck_modifier).clamp(0.0, 1.0);

        let r = self.rng.next_f64();
        let mut is_six = false;
        let mut is_up = false;

        self.state_struct.pity_6 += 1;
        self.state_struct.total_pulls_in_pool += 1;

        let big_pity_gate = if config.big_pity_requires_not_up {
            !self.state_struct.has_obtained_up
        } else {
            true
        };
        // Intentionally duplicated blocks: up_pity_soft and big_pity_cumulative are
        // semantically distinct pity thresholds that happen to share the same outcome.
        // Merging them would obscure game-mechanical intent.
        #[allow(clippy::if_same_then_else)]
        if config.up_pity_soft > 0
            && self.state_struct.total_pulls_in_pool == config.up_pity_soft
            && big_pity_gate
        {
            is_six = true;
            is_up = true;
            self.state_struct.pity_6 = 0;
            self.state_struct.streak_4_star = 0;
            self.state_struct.loss_streak = 0;
            self.state_struct.has_obtained_up = true;
        } else if config.big_pity_cumulative > 0
            && self.state_struct.total_pulls_in_pool == config.big_pity_cumulative
            && big_pity_gate
        {
            is_six = true;
            is_up = true;
            self.state_struct.pity_6 = 0;
            self.state_struct.streak_4_star = 0;
            self.state_struct.loss_streak = 0;
            self.state_struct.has_obtained_up = true;
        } else if r < final_prob_6 {
            is_six = true;
            self.state_struct.pity_6 = 0;
            self.state_struct.streak_4_star = 0;
            if config.up_rate > 0.0 && !config.up_six.is_empty() {
                if self.rng.next_f64() < config.up_rate {
                    is_up = true;
                    self.state_struct.loss_streak = 0;
                    self.state_struct.has_obtained_up = true;
                } else {
                    self.state_struct.loss_streak += 1;
                }
            }
        } else if config.always_5_star
            || (config.five_star_pity > 0
                && self.state_struct.streak_4_star >= config.five_star_pity - 1)
            || r < (final_prob_6 + config.prob_5_base).min(1.0)
        {
            self.state_struct.streak_4_star = 0;
        } else {
            self.state_struct.streak_4_star += 1;
        }
        self.pulls_done += 1;

        let mut reward =
            crate::utils::compute_reward_ppo(is_six, is_up, self.state_struct.loss_streak);
        if is_six && is_up {
            if self.pulls_done < EARLY_UP_BONUS_THRESHOLD_1 {
                reward += 5.0;
            }
            if self.pulls_done < EARLY_UP_BONUS_THRESHOLD_2 {
                reward += 5.0;
            }
        }
        self.episode_reward += reward;

        let done = is_up || self.pulls_done >= EPISODE_MAX_PULLS;

        let experience = PpoStoreRawInput {
            state: self.flat_data.clone(),
            seq_len,
            pity: self.pity_vec.clone(),
            action: action_idx,
            log_prob,
            reward,
            done,
            value: val,
        };

        let finished_reward = if done {
            let r = self.episode_reward;
            self.reset(env_net);
            Some(r)
        } else {
            None
        };

        PpoStepResult {
            experience,
            finished_reward,
        }
    }
}

struct PpoStepResult {
    experience: PpoStoreRawInput,
    finished_reward: Option<f64>,
}

/// Train a PPO agent with multi-environment rollouts.
pub fn train_ppo(rng: &mut Rng, env_net: &EnvNet, config: &Config) -> ActorCritic {
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
    let num_envs = if config.ppo_num_envs > 0 {
        config.ppo_num_envs
    } else {
        1
    };
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
            )
        })
        .collect();

    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(50);
    let mut _episode_count = 0;

    // Linear LR decay
    let initial_lr = 0.0003;

    let heartbeat_every = if fast_mode { 128 } else { 512 };
    let mut last_heartbeat = Instant::now();
    let mut remainder_offset = 0usize;
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
        for _ in 0..rounds {
            let step_results: Vec<PpoStepResult> = worker
                .execute(|| {
                    envs.par_iter_mut()
                        .map(|env| env.step(&ppo.policy, env_net, config, context_len))
                        .collect()
                })
                .unwrap_or_else(|msg| {
                    log::error!("[PPO] Worker execution failed: {}", msg);
                    vec![]
                });
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
            }
            collected += num_envs;
            if collected.is_multiple_of(heartbeat_every)
                && last_heartbeat.elapsed() >= Duration::from_millis(300)
            {
                let global_step = (steps_done + collected).min(total_steps);
                let avg_env_reward =
                    envs.iter().map(|e| e.episode_reward).sum::<f64>() / num_envs as f64;
                pb.set_position(global_step as u64);
                pb.set_message(format!(
                    "Avg R: {:.2} | LR: {:.6}",
                    avg_env_reward, current_lr
                ));
                last_heartbeat = Instant::now();
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
                collected += 1;
                if collected.is_multiple_of(heartbeat_every)
                    && last_heartbeat.elapsed() >= Duration::from_millis(300)
                {
                    let global_step = (steps_done + collected).min(total_steps);
                    pb.set_position(global_step as u64);
                    pb.set_message(format!(
                        "Avg R: {:.2} | LR: {:.6}",
                        envs[idx].episode_reward, current_lr
                    ));
                    last_heartbeat = Instant::now();
                }
            }
            remainder_offset = (remainder_offset + remainder) % num_envs;
        }

        pb.set_message(format!(
            "Updating ({} samples, {} epochs)...",
            collected, k_epochs
        ));
        ppo.update(current_lr);
        steps_done += update_steps;

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
        pb.set_position(steps_done as u64);
        pb.set_message(format!("Avg R: {:.2} | LR: {:.6}", avg_r, current_lr));
    }
    pb.finish_with_message("PPO Training Complete.");
    ppo.policy.freeze_achf_for_inference();
    ppo.policy
}

/// Train PPO with optional metrics collection for benchmarking.
pub fn train_ppo_with_metrics(
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    metrics_tx: Option<std::sync::mpsc::Sender<crate::bench::StepSnapshot>>,
) -> ActorCritic {
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
    let num_envs = if config.ppo_num_envs > 0 {
        config.ppo_num_envs
    } else {
        1
    };
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
            )
        })
        .collect();

    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(50);
    let mut _episode_count = 0;
    let initial_lr = 0.0003;
    let heartbeat_every = if fast_mode { 128 } else { 512 };
    let mut last_heartbeat = Instant::now();
    let snapshot_every = (total_steps / 200).max(1);
    let mut remainder_offset = 0usize;
    let pb = create_bar(total_steps as u64, "PPO Training");

    while steps_done < total_steps {
        let progress = steps_done as f64 / total_steps as f64;
        let current_lr = initial_lr * (1.0 - progress).max(0.1);

        let remaining_steps = total_steps - steps_done;
        let update_steps = steps_per_update.min(remaining_steps);
        let rounds = update_steps / num_envs;
        let remainder = update_steps % num_envs;
        let mut collected = 0usize;
        for _ in 0..rounds {
            let step_results: Vec<PpoStepResult> = worker
                .execute(|| {
                    envs.par_iter_mut()
                        .map(|env| env.step(&ppo.policy, env_net, config, context_len))
                        .collect()
                })
                .unwrap_or_else(|msg| {
                    log::error!("[PPO] Worker execution failed: {}", msg);
                    vec![]
                });
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
            }
            collected += num_envs;
            if collected.is_multiple_of(heartbeat_every)
                && last_heartbeat.elapsed() >= Duration::from_millis(300)
            {
                let global_step = (steps_done + collected).min(total_steps);
                let avg_env_reward =
                    envs.iter().map(|e| e.episode_reward).sum::<f64>() / num_envs as f64;
                pb.set_position(global_step as u64);
                pb.set_message(format!(
                    "Avg R: {:.2} | LR: {:.6}",
                    avg_env_reward, current_lr
                ));
                last_heartbeat = Instant::now();
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
            }
            remainder_offset = (remainder_offset + remainder) % num_envs;
        }

        pb.set_message(format!(
            "Updating ({} samples, {} epochs)...",
            collected, k_epochs
        ));
        let update_loss = ppo.update(current_lr);
        steps_done += update_steps;

        let avg_r = if recent_rewards.is_empty() {
            0.0
        } else {
            recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64
        };

        if let Some(ref tx) = metrics_tx {
            if steps_done % snapshot_every < update_steps {
                let achf_snap = ppo.policy.snapshot_achf();
                let snapshot = crate::bench::StepSnapshot {
                    step: steps_done,
                    gate_value: achf_snap.map_or(1.0, |s| s.gate),
                    g_min: achf_snap.map_or(0.0, |s| s.g_min),
                    grad_ema: achf_snap.map_or(0.0, |s| s.grad_ema),
                    loss: update_loss,
                    reward: avg_r,
                    cache_hit_rate: achf_snap.map_or(0.0, |s| s.cache_hit_rate),
                    sparse_ratio: achf_snap.map_or(0.0, |s| s.low_rank_ratio),
                    ema_cached_ns: achf_snap.map_or(0.0, |s| s.ema_cached_ns),
                    ema_sparse_ns: achf_snap.map_or(0.0, |s| s.ema_sparse_ns),
                    adaptive_bias: achf_snap.map_or(1.0, |s| s.adaptive_bias),
                };
                let _ = tx.send(snapshot);
            }
        }

        pb.set_position(steps_done as u64);
        pb.set_message(format!("Avg R: {:.2} | LR: {:.6}", avg_r, current_lr));
    }
    pb.finish_with_message("PPO Training Complete.");
    ppo.policy.freeze_achf_for_inference();
    ppo.policy
}

/// Incremental PPO trainer for online learning during interactive mode.
pub struct OnlinePpoTrainer {
    ppo: Ppo,
    steps_done: usize,
}

impl OnlinePpoTrainer {
    #[allow(dead_code)]
    pub fn new(seed: u64, k_epochs: usize, batch_size: usize, config: &Config) -> Self {
        Self::from_policy(
            ActorCritic::new_with_config(config, seed),
            k_epochs,
            batch_size,
        )
    }

    pub fn from_policy(policy: ActorCritic, k_epochs: usize, batch_size: usize) -> Self {
        Self {
            ppo: Ppo::from_policy(policy, k_epochs, batch_size),
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

    pub fn train_step(&mut self, current_lr: f64) -> bool {
        if self.ppo.memory.states_raw.len() < self.ppo.batch_size {
            return false;
        }
        self.ppo.update(current_lr);
        self.steps_done += 1;
        true
    }

    pub fn sync_to(&self, shared: &std::sync::RwLock<ActorCritic>) {
        for attempt in 0..3u64 {
            if let Ok(mut guard) = shared.try_write() {
                *guard = self.ppo.policy.clone();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1 + attempt));
        }
        if let Ok(mut guard) = shared.write() {
            *guard = self.ppo.policy.clone();
        }
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
        let expected = policy.parameters()[0].data_f64().clone();

        let trainer = OnlinePpoTrainer::from_policy(policy, 2, 128);
        let got = trainer.ppo.policy.parameters()[0].data_f64().clone();

        assert_eq!(got, expected);
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
            .data_f64()
            .clone();

        // Modify student first parameter to 1.0
        let params = ppo.policy.parameters();
        let mut data = params[0].data_write_f64();
        for val in data.iter_mut() {
            *val = 1.0;
        }
        drop(data); // Release lock before EMA update to avoid deadlock
        drop(params); // Explicitly drop to release all locks

        // Perform EMA update
        ppo.update_ema_teacher();

        // With decay=0.5, teacher_new = 0.5 * teacher_old + 0.5 * student
        let teacher_after = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f64()
            .clone();

        for (before, after) in teacher_before.iter().zip(teacher_after.iter()) {
            let expected = 0.5 * before + 0.5 * 1.0;
            assert!((after - expected).abs() < 1e-9);
        }
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
            .data_f64()
            .clone();
        let before_last = ppo.ema_policy.as_ref().unwrap().parameters()[num_params - 1]
            .data_f64()
            .clone();

        // Set all student parameters to 2.0
        for param in ppo.policy.parameters() {
            let mut data = param.data_write_f64();
            for val in data.iter_mut() {
                *val = 2.0;
            }
        }

        ppo.update_ema_teacher();

        // Check first and last tensor values after update
        let after_first = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f64()
            .clone();
        let after_last = ppo.ema_policy.as_ref().unwrap().parameters()[num_params - 1]
            .data_f64()
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
            let mut data = param.data_write_f64();
            for val in data.iter_mut() {
                *val = 10.0;
            }
        }

        // Perform many EMA updates
        for _ in 0..100 {
            ppo.update_ema_teacher();
        }

        // After 100 updates with decay=0.5, EMA should be very close to student (10.0)
        let teacher_final = ppo.ema_policy.as_ref().unwrap().parameters()[0]
            .data_f64()
            .clone();

        for val in teacher_final.iter() {
            assert!(
                (*val - 10.0).abs() < 1e-6,
                "EMA did not converge: got {}",
                val
            );
        }
    }

    #[test]
    fn softmax_sample_top_k_zero_like_full_softmax() {
        // top_k=0 should behave identically to full softmax (no truncation)
        // Compare empirical distributions over many samples
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let trials = 1000;

        let mut counts_0 = [0usize; ACTION_SPACE];
        let mut counts_full = [0usize; ACTION_SPACE];

        for _ in 0..trials {
            let (a0, _) = softmax_sample(&logits, 0);
            let (af, _) = softmax_sample(&logits, ACTION_SPACE);
            counts_0[a0] += 1;
            counts_full[af] += 1;
        }

        // Distributions should be identical (normalized counts should match)
        for i in 0..ACTION_SPACE {
            let p0 = counts_0[i] as f64 / trials as f64;
            let pf = counts_full[i] as f64 / trials as f64;
            assert!(
                (p0 - pf).abs() < 0.05,
                "Distribution mismatch at index {}: top_k=0 has {:.3}, full softmax has {:.3}",
                i,
                p0,
                pf
            );
        }
    }

    #[test]
    fn softmax_sample_top_k_gte_action_space_like_full_softmax() {
        // top_k >= ACTION_SPACE should behave identically to full softmax
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let trials = 1000;

        let mut counts_large = [0usize; ACTION_SPACE];
        let mut counts_full = [0usize; ACTION_SPACE];

        for _ in 0..trials {
            let (al, _) = softmax_sample(&logits, ACTION_SPACE + 10);
            let (af, _) = softmax_sample(&logits, ACTION_SPACE);
            counts_large[al] += 1;
            counts_full[af] += 1;
        }

        // Distributions should be identical
        for i in 0..ACTION_SPACE {
            let pl = counts_large[i] as f64 / trials as f64;
            let pf = counts_full[i] as f64 / trials as f64;
            assert!(
                (pl - pf).abs() < 0.05,
                "Distribution mismatch at index {}: top_k large has {:.3}, full softmax has {:.3}",
                i,
                pl,
                pf
            );
        }
    }

    #[test]
    fn softmax_sample_top_k_boundary_identical_values() {
        // When multiple logits have identical values at the threshold boundary,
        // all logits with that value should be treated consistently.
        // Test case: logits where 3rd and 4th highest values differ
        // [5.0, 4.0, 4.0, 3.0, 2.0] with top_k=3
        // Sorted: [5.0, 4.0, 4.0, 3.0, 2.0], threshold = 4.0
        // Correct behavior with <: indices 0,1,2 survive and 3,4 are masked.
        let logits = [5.0, 4.0, 4.0, 3.0, 2.0];
        let top_k = 3;

        let mut counts = [0usize; ACTION_SPACE];
        for _ in 0..1000 {
            let (action, _) = softmax_sample(&logits, top_k);
            counts[action] += 1;
        }

        assert!(counts[0] > 0, "Index 0 (5.0) should be sampled");
        assert!(counts[1] > 0, "Index 1 (4.0) should be sampled");
        assert!(counts[2] > 0, "Index 2 (4.0) should be sampled");
        assert_eq!(counts[3], 0, "Index 3 (3.0) should be masked");
        assert_eq!(counts[4], 0, "Index 4 (2.0) should be masked");
    }

    #[test]
    fn ppo_from_policy_clamps_zero_batch_size() {
        let policy = ActorCritic::new(7, &crate::config::AchfConfig::default(), 64, 2);
        let ppo = Ppo::from_policy(policy, 1, 0);
        assert_eq!(ppo.batch_size, 1);
    }
}
