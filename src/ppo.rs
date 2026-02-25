use crate::autograd::Tensor;
use crate::config::Config;
use crate::dbn::Dbn;
use crate::neural::DIM;
use crate::nn::{Linear, Module};
use crate::rng::Rng;
use crate::sim::{build_features, dbn_env, prob_6, PpoExperience, PullState};
use crate::transformer::{KVCache, LuckTransformer};
use crate::worker::GoodJobWorker;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use rayon::prelude::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

// --- PPO Components ---

const ACTION_SPACE: usize = 5;
pub const ACTIONS: [f64; ACTION_SPACE] = [0.0, 0.005, 0.015, -0.005, -0.015];
const CLIP_EPSILON: f64 = 0.2;
const GAMMA: f64 = 0.99;
const GAE_LAMBDA: f64 = 0.95;
const VALUE_COEF: f64 = 0.5;
const ENTROPY_COEF: f64 = 0.01;

#[inline(always)]
fn sum_f64(values: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                return sum_f64_avx2(values);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return sum_f64_neon(values);
        }
    }
    let mut sum = 0.0;
    for &v in values {
        sum += v;
    }
    sum
}

#[inline(always)]
fn sum_sq_diff(values: &[f64], mean: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                return sum_sq_diff_avx2(values, mean);
            }
        }
    }
    let mut sum = 0.0;
    for &v in values {
        let d = v - mean;
        sum += d * d;
    }
    sum
}

#[inline(always)]
fn normalize_slice(values: &[f64], mean: f64, std: f64) -> Vec<f64> {
    let len = values.len();
    let mut out = vec![0.0; len];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                normalize_slice_avx2(values, &mut out, mean, std);
            }
            return out;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            normalize_slice_neon(values, &mut out, mean, std);
        }
        return out;
    }
    for i in 0..len {
        out[i] = (values[i] - mean) / std;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f64_avx2(values: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = _mm256_setzero_pd();
    while i + 4 <= len {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        acc = _mm256_add_pd(acc, v);
        i += 4;
    }
    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < len {
        sum += *values.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_sq_diff_avx2(values: &[f64], mean: f64) -> f64 {
    use core::arch::x86_64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = _mm256_setzero_pd();
    let mean_vec = _mm256_set1_pd(mean);
    while i + 4 <= len {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let d = _mm256_sub_pd(v, mean_vec);
        let prod = _mm256_mul_pd(d, d);
        acc = _mm256_add_pd(acc, prod);
        i += 4;
    }
    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < len {
        let d = *values.get_unchecked(i) - mean;
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_slice_avx2(values: &[f64], out: &mut [f64], mean: f64, std: f64) {
    use core::arch::x86_64::*;
    let mut i = 0;
    let len = values.len();
    let mean_vec = _mm256_set1_pd(mean);
    let std_vec = _mm256_set1_pd(std);
    while i + 4 <= len {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let d = _mm256_sub_pd(v, mean_vec);
        let n = _mm256_div_pd(d, std_vec);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), n);
        i += 4;
    }
    while i < len {
        *out.get_unchecked_mut(i) = (*values.get_unchecked(i) - mean) / std;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_f64_neon(values: &[f64]) -> f64 {
    use core::arch::aarch64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = vdupq_n_f64(0.0);
    while i + 2 <= len {
        let v = vld1q_f64(values.as_ptr().add(i));
        acc = vaddq_f64(acc, v);
        i += 2;
    }
    let mut tmp = [0.0; 2];
    vst1q_f64(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1];
    while i < len {
        sum += *values.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_sq_diff_neon(values: &[f64], mean: f64) -> f64 {
    use core::arch::aarch64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = vdupq_n_f64(0.0);
    let mean_vec = vdupq_n_f64(mean);
    while i + 2 <= len {
        let v = vld1q_f64(values.as_ptr().add(i));
        let d = vsubq_f64(v, mean_vec);
        let prod = vmulq_f64(d, d);
        acc = vaddq_f64(acc, prod);
        i += 2;
    }
    let mut tmp = [0.0; 2];
    vst1q_f64(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1];
    while i < len {
        let d = *values.get_unchecked(i) - mean;
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_slice_neon(values: &[f64], out: &mut [f64], mean: f64, std: f64) {
    use core::arch::aarch64::*;
    let mut i = 0;
    let len = values.len();
    let mean_vec = vdupq_n_f64(mean);
    let std_vec = vdupq_n_f64(std);
    while i + 2 <= len {
        let v = vld1q_f64(values.as_ptr().add(i));
        let d = vsubq_f64(v, mean_vec);
        let n = vdivq_f64(d, std_vec);
        vst1q_f64(out.as_mut_ptr().add(i), n);
        i += 2;
    }
    while i < len {
        *out.get_unchecked_mut(i) = (*values.get_unchecked(i) - mean) / std;
        i += 1;
    }
}

// Softmax + categorical sample over fixed ACTION_SPACE logits.
// Returns (action_index, log_prob_of_action).
#[inline]
fn softmax_sample(logits: &[f64]) -> (usize, f64) {
    let mut max_l = f64::NEG_INFINITY;
    for &v in logits {
        if v > max_l {
            max_l = v;
        }
    }
    let mut sum_exp = 0.0;
    let mut probs = [0.0; ACTION_SPACE];
    for (i, prob) in probs.iter_mut().enumerate() {
        *prob = (logits[i] - max_l).exp();
        sum_exp += *prob;
    }
    for prob in probs.iter_mut() {
        *prob /= sum_exp;
    }
    let mut r = rand::random::<f64>();
    let mut idx = ACTION_SPACE - 1;
    for (i, &p) in probs.iter().enumerate() {
        if r < p {
            idx = i;
            break;
        }
        r -= p;
    }
    (idx, probs[idx].ln())
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ActorCritic {
    pub backbone: LuckTransformer,
    pub actor_head: Linear,
    pub critic_head: Linear,
}

impl ActorCritic {
    pub fn new(seed: u64, achf: &crate::config::AchfConfig) -> Self {
        let backbone = LuckTransformer::new(DIM, 64, true, seed, achf);
        let actor_head = Linear::new(64, ACTION_SPACE, true, seed.wrapping_add(100));
        let critic_head = Linear::new(64, 1, true, seed.wrapping_add(200));

        ActorCritic {
            backbone,
            actor_head,
            critic_head,
        }
    }

    #[allow(dead_code)]
    pub fn forward_actor(&self, state: &Tensor, pity: &[usize]) -> Tensor {
        let x = if state.shape.len() == 1 {
            state.reshape(vec![1, 1, state.shape[0]])
        } else if state.shape.len() == 2 {
            state.reshape(vec![1, state.shape[0], state.shape[1]])
        } else {
            state.clone()
        };
        let seq = self.backbone.forward(&x, pity);
        let last = self.backbone.last_token(&seq);
        let logits = self.actor_head.forward(&last);
        if logits.shape.len() == 2 && logits.shape[0] == 1 {
            logits.reshape(vec![logits.shape[1]])
        } else {
            logits
        }
    }

    #[allow(dead_code)]
    pub fn forward_critic(&self, state: &Tensor, pity: &[usize]) -> Tensor {
        let x = if state.shape.len() == 1 {
            state.reshape(vec![1, 1, state.shape[0]])
        } else if state.shape.len() == 2 {
            state.reshape(vec![1, state.shape[0], state.shape[1]])
        } else {
            state.clone()
        };
        let seq = self.backbone.forward(&x, pity);
        let last = self.backbone.last_token(&seq);
        let value = self.critic_head.forward(&last);
        if value.shape.len() == 2 && value.shape[0] == 1 {
            value.reshape(vec![value.shape[1]])
        } else {
            value
        }
    }

    pub fn forward_actor_critic(&self, state: &Tensor, pity: &[usize]) -> (Tensor, Tensor) {
        let x = if state.shape.len() == 1 {
            state.reshape(vec![1, 1, state.shape[0]])
        } else if state.shape.len() == 2 {
            state.reshape(vec![1, state.shape[0], state.shape[1]])
        } else {
            state.clone()
        };
        let seq = self.backbone.forward(&x, pity);
        let last = self.backbone.last_token(&seq);
        let logits = self.actor_head.forward(&last);
        let value = self.critic_head.forward(&last);
        let logits = if logits.shape.len() == 2 && logits.shape[0] == 1 {
            logits.reshape(vec![logits.shape[1]])
        } else {
            logits
        };
        let value = if value.shape.len() == 2 && value.shape[0] == 1 {
            value.reshape(vec![value.shape[1]])
        } else {
            value
        };
        (logits, value)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.backbone.parameters();
        p.extend(self.actor_head.parameters());
        p.extend(self.critic_head.parameters());
        p
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
    pub fn step(&self, state: &Tensor, pity: &[usize]) -> (usize, f64, f64) {
        let (logits, value) = self.forward_actor_critic(state, pity);
        let logits_data = logits.data.read().unwrap();
        let (action_idx, log_prob) = softmax_sample(&logits_data);
        let val = value.data.read().unwrap()[0];
        (action_idx, log_prob, val)
    }

    // Fast inference without Autograd graph
    pub fn step_inference(&self, state: &[f64]) -> usize {
        let seq = self.backbone.forward_inference(state);
        let last = self.backbone.last_token_inference(&seq);
        let logits = self.actor_head.forward_inference(&last);
        softmax_sample(&logits).0
    }

    pub fn step_inference_cached_with_value(
        &self,
        state: &[f64],
        kv_cache: &mut KVCache,
        start_pos: usize,
    ) -> (usize, f64, f64) {
        let last = self
            .backbone
            .forward_inference_step(state, kv_cache, start_pos);
        let logits = self.actor_head.forward_inference(&last);
        let value = self.critic_head.forward_inference(&last);
        let (action_idx, log_prob) = softmax_sample(&logits);
        (action_idx, log_prob, value[0])
    }

    pub fn step_inference_cached(
        &self,
        state: &[f64],
        kv_cache: &mut KVCache,
        start_pos: usize,
    ) -> usize {
        let last = self
            .backbone
            .forward_inference_step(state, kv_cache, start_pos);
        let logits = self.actor_head.forward_inference(&last);
        softmax_sample(&logits).0
    }

    pub fn prune_cache(&self, kv_cache: &mut KVCache, max_len: usize) {
        self.backbone.prune_kv_cache(kv_cache, max_len);
    }

    pub fn achf_cache_stats_iter(&self) -> impl Iterator<Item = crate::achf::AchfCacheStats> + '_ {
        self.backbone.achf_cache_stats_iter()
    }

    pub fn achf_cache_stats_aggregate(&self) -> crate::achf::AchfCacheStats {
        self.backbone.achf_cache_stats_aggregate()
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

    fn normalize(&self, x: f64) -> f64 {
        (x - self.mean) / (self.var.sqrt() + 1e-8)
    }
}

struct Adam {
    params: Vec<Tensor>,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl Adam {
    fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let m = params
            .iter()
            .map(|p| vec![0.0; p.data.read().unwrap().len()])
            .collect();
        let v = params
            .iter()
            .map(|p| vec![0.0; p.data.read().unwrap().len()])
            .collect();
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

        // Global gradient clipping (max_norm = 1.0)
        let mut total_norm = 0.0;
        for param in &self.params {
            let grad = param.grad.read().unwrap();
            for &g in grad.iter() {
                total_norm += g * g;
            }
        }
        total_norm = total_norm.sqrt();
        let clip_coef = if total_norm > 1.0 {
            1.0 / total_norm
        } else {
            1.0
        };

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = param.grad.read().unwrap();
            let mut data = param.data.write().unwrap();
            for j in 0..data.len() {
                let g = grad[j] * clip_coef;
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;
                // AdamW: decoupled weight decay applied directly to parameters
                data[j] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * data[j]);
            }
        }
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
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

pub struct Ppo {
    pub policy: ActorCritic,
    optimizer: Adam,
    memory: Memory,
    k_epochs: usize,
    batch_size: usize,
    reward_normalizer: RunningMeanStd,
}

impl Ppo {
    pub fn new(
        seed: u64,
        k_epochs: usize,
        batch_size: usize,
        achf: &crate::config::AchfConfig,
    ) -> Self {
        let policy = ActorCritic::new(seed, achf);
        let optimizer = Adam::new(policy.parameters(), 0.0003);
        Ppo {
            policy,
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
            batch_size,
            reward_normalizer: RunningMeanStd::new(),
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

    pub fn update(&mut self, current_lr: f64) {
        if self.memory.states_raw.is_empty() {
            return;
        }

        // Update Learning Rate
        self.optimizer.set_lr(current_lr);

        let len = self.memory.states_raw.len();
        let states_raw = std::mem::take(&mut self.memory.states_raw);
        let state_lens = std::mem::take(&mut self.memory.state_lens);
        let pities = std::mem::take(&mut self.memory.pities);
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

        // Normalize rewards for GAE calculation
        let norm_rewards: Vec<f64> = rewards
            .iter()
            .map(|&r| self.reward_normalizer.normalize(r).clamp(-10.0, 10.0)) // Clip for stability
            .collect();

        for t in (0..len).rev() {
            let non_terminal = if is_terminals[t] { 0.0 } else { 1.0 };
            let val_t = values[t];
            let val_next = if t < len - 1 { values[t + 1] } else { 0.0 };

            // Use normalized rewards for training signal
            let delta = norm_rewards[t] + GAMMA * val_next * non_terminal - val_t;
            let gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae_lam;

            advantages[t] = gae;
            returns[t] = gae + val_t;

            last_gae_lam = gae;
        }

        let adv_mean: f64 = sum_f64(&advantages) / len as f64;
        let adv_std: f64 = (sum_sq_diff(&advantages, adv_mean) / len as f64).sqrt() + 1e-8;
        let norm_advantages: Vec<f64> = normalize_slice(&advantages, adv_mean, adv_std);

        // Target KL Divergence for Early Stopping
        let target_kl = 0.015;
        let mut indices: Vec<usize> = (0..len).collect();
        let action_masks: Vec<Tensor> = (0..ACTION_SPACE)
            .map(|idx| {
                let mut mask_vec = vec![0.0; ACTION_SPACE];
                mask_vec[idx] = 1.0;
                Tensor::new(mask_vec, vec![ACTION_SPACE])
            })
            .collect();

        let total_batches = len.div_ceil(self.batch_size);
        let mut last_update = Instant::now();
        let update_every = Duration::from_millis(500);
        let mut update_batches_done = 0usize;
        for epoch_idx in 0..self.k_epochs {
            indices.shuffle(&mut rand::rng());
            let mut approx_kl = 0.0;
            let mut batch_count = 0.0;
            let mut early_stop = false;

            for (batch_idx, chunk) in indices.chunks(self.batch_size).enumerate() {
                self.optimizer.zero_grad();

                let mut loss_accum = Tensor::zeros(vec![1]);
                let batch_len = chunk.len();

                for &i in chunk {
                    let state = &states[i];
                    let pity = &pities[i];
                    let action_idx = actions[i];
                    let old_log_prob = log_probs[i];
                    let advantage = norm_advantages[i];
                    let return_val = returns[i];

                    let (logits, value) = self.policy.forward_actor_critic(state, pity);

                    let max_logit = logits
                        .data
                        .read()
                        .unwrap()
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_logits = (logits.clone()
                        + Tensor::new(vec![-max_logit; ACTION_SPACE], vec![ACTION_SPACE]))
                    .exp();
                    let sum_exp = exp_logits.sum();
                    let log_sum_exp = sum_exp.log() + Tensor::new(vec![max_logit], vec![1]);

                    let mask = action_masks[action_idx].clone();

                    let log_probs = logits.clone() - log_sum_exp.broadcast(vec![ACTION_SPACE]);

                    let log_prob = (log_probs.clone() * mask).sum();

                    let old_log_prob_tensor = Tensor::new(vec![old_log_prob], vec![1]);
                    // Use references to avoid moving, cleaner than explicit clones
                    let log_ratio = &log_prob - &old_log_prob_tensor;
                    let ratio = log_ratio.clone().exp();

                    // Calculate KL Divergence
                    // kl = (ratio - 1) - log_ratio
                    let kl = (ratio.clone() - Tensor::new(vec![1.0], vec![1])) - log_ratio.clone();
                    approx_kl += kl.data.read().unwrap()[0];
                    batch_count += 1.0;

                    let adv_tensor = Tensor::new(vec![advantage], vec![1]);
                    let surr1 = ratio.clone() * adv_tensor.clone();
                    let ratio_clipped = ratio.clip(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON);
                    let surr2 = ratio_clipped * adv_tensor;

                    let s1_val = surr1.data.read().unwrap()[0];
                    let s2_val = surr2.data.read().unwrap()[0];
                    let policy_loss = if s1_val < s2_val { surr1 } else { surr2 };

                    let ret_tensor = Tensor::new(vec![return_val], vec![1]);
                    let v_loss = (value - ret_tensor).mse_loss(&Tensor::zeros(vec![1]));

                    let p = log_probs.exp();
                    let entropy = -(p * log_probs).sum();

                    let loss = -policy_loss + v_loss * Tensor::new(vec![VALUE_COEF], vec![1])
                        - entropy * Tensor::new(vec![ENTROPY_COEF], vec![1]);

                    loss_accum = loss_accum + loss;
                }

                let batch_size_tensor = Tensor::new(vec![batch_len as f64], vec![1]);
                let mut final_loss = loss_accum / batch_size_tensor;
                if let Some(reg) = self.policy.achf_orthogonal_penalty() {
                    final_loss = final_loss + reg;
                }
                final_loss.backward();
                self.policy.update_achf_after_backward();
                self.optimizer.step();

                if batch_count > 0.0 && (approx_kl / batch_count) > target_kl * 1.5 {
                    early_stop = true;
                }

                update_batches_done += 1;
                if last_update.elapsed() >= update_every {
                    print!(
                        "\r[PPO] Updating: {}/{} | Epoch {}/{} | Batch {}/{}",
                        update_batches_done,
                        total_batches * self.k_epochs,
                        epoch_idx + 1,
                        self.k_epochs,
                        batch_idx + 1,
                        total_batches
                    );
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                    last_update = Instant::now();
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
                    // println!("  [PPO] Early stopping at epoch {} due to KL {:.4}", _, approx_kl);
                    break;
                }
            }
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
    kv_cache: KVCache,
    episode_reward: f64,
    rng: Rng,
}

impl PpoEnvState {
    fn new(seed: u64, dbn: &Dbn, context_len: usize, num_heads: usize) -> Self {
        let mut rng = Rng::from_seed(seed);
        let (env_noise, env_bias) = dbn_env(dbn, &mut rng);
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
            kv_cache: KVCache::new(num_heads),
            episode_reward: 0.0,
            rng,
        }
    }

    fn reset(&mut self, dbn: &Dbn) {
        self.history_buffer.clear();
        self.pity_buffer.clear();
        self.kv_cache.clear();
        self.state_struct = PullState {
            pity_6: 0,
            total_pulls_in_pool: 0,
            has_obtained_up: false,
            streak_4_star: 0,
            loss_streak: 0,
        };
        let (env_noise, env_bias) = dbn_env(dbn, &mut self.rng);
        self.env_noise = env_noise;
        self.env_bias = env_bias;
        self.pulls_done = 0;
        self.episode_reward = 0.0;
    }

    fn step(
        &mut self,
        policy: &ActorCritic,
        dbn: &Dbn,
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
        let token = self.history_buffer.back().unwrap().as_slice();
        let (action_idx, log_prob, val) =
            policy.step_inference_cached_with_value(token, &mut self.kv_cache, seq_len - 1);

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
            || r < final_prob_6 + config.prob_5_base
        {
            self.state_struct.streak_4_star = 0;
        } else {
            self.state_struct.streak_4_star += 1;
        }
        self.pulls_done += 1;

        let mut reward = -0.1;
        if is_six {
            if is_up {
                reward += 10.0;
                if self.pulls_done < 80 {
                    reward += 5.0;
                }
                if self.pulls_done < 50 {
                    reward += 5.0;
                }
            } else {
                reward += 2.0;
            }
        }
        if self.state_struct.loss_streak >= 2 {
            reward -= (self.state_struct.loss_streak as f64) * 2.0;
        }
        self.episode_reward += reward;

        let done = is_up || self.pulls_done >= 300;

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
            self.reset(dbn);
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

pub fn train_ppo(rng: &mut Rng, dbn: &Dbn, config: &Config) -> ActorCritic {
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
    let worker = GoodJobWorker::new_with_config(config);
    let mut ppo = Ppo::new(rng.next_u64(), k_epochs, batch_size, &config.achf);
    let mut steps_done = 0;

    let env_seeds: Vec<u64> = (0..num_envs).map(|_| rng.next_u64()).collect();
    let mut envs: Vec<PpoEnvState> = env_seeds
        .into_iter()
        .map(|seed| {
            PpoEnvState::new(
                seed,
                dbn,
                context_len,
                ppo.policy.backbone.mla_layer.config.num_heads,
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
    while steps_done < total_steps {
        // Calculate LR
        let progress = steps_done as f64 / total_steps as f64;
        let current_lr = initial_lr * (1.0 - progress).max(0.1); // Decay to 10%

        let rounds = steps_per_update / num_envs;
        let remainder = steps_per_update % num_envs;
        let mut collected = 0usize;
        for _ in 0..rounds {
            let step_results: Vec<PpoStepResult> = worker
                .execute(|| {
                    envs.par_iter_mut()
                        .map(|env| env.step(&ppo.policy, dbn, config, context_len))
                        .collect()
                })
                .unwrap_or_else(|msg| panic!("{}", msg));
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
                print!(
                    "\r[PPO] Collecting: {}/{} | Avg Reward: {:.2} | LR: {:.6}",
                    global_step, total_steps, avg_env_reward, current_lr
                );
                use std::io::Write;
                std::io::stdout().flush().unwrap();
                last_heartbeat = Instant::now();
            }
        }
        if remainder > 0 {
            let start = remainder_offset % num_envs;
            for i in 0..remainder {
                let idx = (start + i) % num_envs;
                let result = envs[idx].step(&ppo.policy, dbn, config, context_len);
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
                print!(
                    "\r[PPO] Collecting: {}/{} | Avg Reward: {:.2} | LR: {:.6}",
                    global_step, total_steps, envs[idx].episode_reward, current_lr
                );
                use std::io::Write;
                std::io::stdout().flush().unwrap();
                last_heartbeat = Instant::now();
            }
            }
            remainder_offset = (remainder_offset + remainder) % num_envs;
        }

        ppo.update(current_lr);
        steps_done += steps_per_update;

        if config.achf.cache_log_interval_steps > 0
            && steps_done % config.achf.cache_log_interval_steps == 0
        {
            if config.achf.cache_log_per_layer {
                for (idx, stats) in ppo.policy.achf_cache_stats_iter().enumerate() {
                    if stats.calls > 0 {
                        println!("\n[ACHF-L{}] {}", idx, format_achf_stats(&stats));
                    }
                }
            } else {
                let stats = ppo.policy.achf_cache_stats_aggregate();
                if stats.calls > 0 {
                    println!("\n{}", format_achf_stats(&stats));
                }
            }
        }

        let avg_r = if recent_rewards.is_empty() {
            0.0
        } else {
            recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64
        };
        print!(
            "\r[PPO] Steps: {}/{} | Avg Reward: {:.2} | LR: {:.6}",
            steps_done, total_steps, avg_r, current_lr
        );
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
    println!("\n[PPO] Training Complete.");
    ppo.policy.freeze_achf_for_inference();
    ppo.policy
}

fn format_achf_stats(stats: &crate::achf::AchfCacheStats) -> String {
    let calls = stats.calls as f64;
    let hit_rate = if calls > 0.0 {
        stats.cache_hits as f64 / calls
    } else {
        0.0
    };
    format!(
        "[ACHF] Calls: {} | Hit: {:.2}% | Miss: {} | Skip: {} | LowRank: {} | Dense: {} | CachedEMA(ns): {:.1}/{:.1} | LowRankEMA(ns): {:.1}/{:.1} | DecisionEMA(ns): {:.1}/{:.1} | Bias: {:.3} | Samples: {}/{}",
        stats.calls,
        hit_rate * 100.0,
        stats.cache_misses,
        stats.cache_skips,
        stats.low_rank_paths,
        stats.dense_paths,
        stats.ema_cached_ns,
        stats.ema_cached_long_ns,
        stats.ema_low_rank_ns,
        stats.ema_low_rank_long_ns,
        stats.decision_ema_ns,
        stats.decision_ema_long_ns,
        stats.adaptive_bias,
        stats.latency_samples,
        stats.decision_samples
    )
}

pub struct OnlinePpoTrainer {
    ppo: Ppo,
    steps_done: usize,
}

impl OnlinePpoTrainer {
    pub fn new(
        seed: u64,
        k_epochs: usize,
        batch_size: usize,
        achf: &crate::config::AchfConfig,
    ) -> Self {
        Self {
            ppo: Ppo::new(seed, k_epochs, batch_size, achf),
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
        let policy = ActorCritic::new(42, &crate::config::AchfConfig::default());

        // Case 1: 1D input [DIM] (e.g. [8])
        let state_1d = Tensor::new(vec![0.5; DIM], vec![DIM]);
        let pity = vec![0];
        let _ = policy.forward_actor(&state_1d, &pity);
        let _ = policy.forward_critic(&state_1d, &pity);

        // Case 2: 2D input [Seq, DIM] (e.g. [5, 8])
        let seq_len = 5;
        let state_2d = Tensor::new(vec![0.5; seq_len * DIM], vec![seq_len, DIM]);
        let _ = policy.forward_actor(&state_2d, &pity);
        let _ = policy.forward_critic(&state_2d, &pity);

        // Case 3: 3D input [1, Seq, DIM]
        let state_3d = Tensor::new(vec![0.5; seq_len * DIM], vec![1, seq_len, DIM]);
        let _ = policy.forward_actor(&state_3d, &pity);
        let _ = policy.forward_critic(&state_3d, &pity);
    }
}
