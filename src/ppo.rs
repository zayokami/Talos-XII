use crate::autograd::Tensor;
use crate::config::Config;
use crate::dbn::Dbn;
use crate::neural::DIM;
use crate::nn::{Linear, Module};
use crate::rng::Rng;
use crate::sim::{build_features, dbn_env, prob_6, PpoExperience, PullState};
use crate::transformer::{KVCache, LuckTransformer};
use std::collections::VecDeque;

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
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return sum_sq_diff_neon(values, mean);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return sum_sq_diff_neon(values, mean);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return sum_sq_diff_neon(values, mean);
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

#[derive(Clone, Serialize, Deserialize)]
pub struct ActorCritic {
    pub backbone: LuckTransformer,
    pub actor_head: Linear,
    pub critic_head: Linear,
}

impl ActorCritic {
    pub fn new(seed: u64) -> Self {
        let backbone = LuckTransformer::new(DIM, 64, true, seed);
        let actor_head = Linear::new(64, ACTION_SPACE, true, seed.wrapping_add(100));
        let critic_head = Linear::new(64, 1, true, seed.wrapping_add(200));

        ActorCritic {
            backbone,
            actor_head,
            critic_head,
        }
    }

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

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.backbone.parameters();
        p.extend(self.actor_head.parameters());
        p.extend(self.critic_head.parameters());
        p
    }

    // Returns (action_idx, log_prob, value)
    pub fn step(&self, state: &Tensor, pity: &[usize]) -> (usize, f64, f64) {
        let logits = self.forward_actor(state, pity);
        let value = self.forward_critic(state, pity);

        // Softmax
        let logits_data = logits.data.read().unwrap();
        let mut max_l = -1e9;
        for &v in logits_data.iter() {
            if v > max_l {
                max_l = v;
            }
        }
        let mut sum_exp = 0.0;
        let mut probs = vec![0.0; ACTION_SPACE];
        for i in 0..ACTION_SPACE {
            probs[i] = (logits_data[i] - max_l).exp();
            sum_exp += probs[i];
        }
        for i in 0..ACTION_SPACE {
            probs[i] /= sum_exp;
        }

        // Sample
        let mut r = rand::random::<f64>();
        let mut action_idx = 0;
        for i in 0..ACTION_SPACE {
            if r < probs[i] {
                action_idx = i;
                break;
            }
            r -= probs[i];
        }
        if action_idx == ACTION_SPACE {
            action_idx = ACTION_SPACE - 1;
        }

        let log_prob = probs[action_idx].ln();
        let val = value.data.read().unwrap()[0];

        (action_idx, log_prob, val)
    }

    // Fast inference without Autograd graph
    pub fn step_inference(&self, state: &[f64]) -> usize {
        let seq = self.backbone.forward_inference(state);
        let last = self.backbone.last_token_inference(&seq);
        let logits = self.actor_head.forward_inference(&last);

        // Softmax
        let mut max_l = -1e9;
        for &v in logits.iter() {
            if v > max_l {
                max_l = v;
            }
        }
        let mut sum_exp = 0.0;
        let mut probs = vec![0.0; ACTION_SPACE];
        for i in 0..ACTION_SPACE {
            probs[i] = (logits[i] - max_l).exp();
            sum_exp += probs[i];
        }
        for i in 0..ACTION_SPACE {
            probs[i] /= sum_exp;
        }

        // Sample
        let mut r = rand::random::<f64>();
        let mut action_idx = 0;
        for i in 0..ACTION_SPACE {
            if r < probs[i] {
                action_idx = i;
                break;
            }
            r -= probs[i];
        }
        if action_idx == ACTION_SPACE {
            action_idx = ACTION_SPACE - 1;
        }
        action_idx
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

        // Softmax
        let mut max_l = -1e9;
        for &v in logits.iter() {
            if v > max_l {
                max_l = v;
            }
        }
        let mut sum_exp = 0.0;
        let mut probs = vec![0.0; ACTION_SPACE];
        for i in 0..ACTION_SPACE {
            probs[i] = (logits[i] - max_l).exp();
            sum_exp += probs[i];
        }
        for i in 0..ACTION_SPACE {
            probs[i] /= sum_exp;
        }

        // Sample
        let mut r = rand::random::<f64>();
        let mut action_idx = 0;
        for i in 0..ACTION_SPACE {
            if r < probs[i] {
                action_idx = i;
                break;
            }
            r -= probs[i];
        }
        if action_idx == ACTION_SPACE {
            action_idx = ACTION_SPACE - 1;
        }
        action_idx
    }

    pub fn prune_cache(&self, kv_cache: &mut KVCache, max_len: usize) {
        self.backbone.prune_kv_cache(kv_cache, max_len);
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
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self) {
        self.t += 1;
        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = param.grad.read().unwrap();
            let mut data = param.data.write().unwrap();
            for j in 0..data.len() {
                let g = grad[j];
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[i][j] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powi(self.t as i32));
                data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
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

pub struct PPO {
    pub policy: ActorCritic,
    optimizer: Adam,
    memory: Memory,
    k_epochs: usize,
    batch_size: usize,
    reward_normalizer: RunningMeanStd,
}

impl PPO {
    pub fn new(seed: u64, k_epochs: usize, batch_size: usize) -> Self {
        let policy = ActorCritic::new(seed);
        let optimizer = Adam::new(policy.parameters(), 0.0003);
        PPO {
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

    pub fn store(
        &mut self,
        state: Tensor,
        pity: Vec<usize>,
        action: usize,
        log_prob: f64,
        reward: f64,
        done: bool,
        value: f64,
    ) {
        let seq_len = state.shape.get(0).copied().unwrap_or(1);
        let data = state.data.read().unwrap().clone();
        self.store_raw(data, seq_len, pity, action, log_prob, reward, done, value);
    }

    pub fn store_raw(
        &mut self,
        state: Vec<f64>,
        seq_len: usize,
        pity: Vec<usize>,
        action: usize,
        log_prob: f64,
        reward: f64,
        done: bool,
        value: f64,
    ) {
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
            .zip(state_lens.into_iter())
            .map(|(data, seq_len)| Tensor::new(data, vec![seq_len, DIM]))
            .collect();
        let mut advantages = vec![0.0; len];
        let mut returns = vec![0.0; len];

        let mut last_gae_lam = 0.0;

        // Normalize rewards for GAE calculation
        let norm_rewards: Vec<f64> = rewards
            .iter()
            .map(|&r| self.reward_normalizer.normalize(r).max(-10.0).min(10.0)) // Clip for stability
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

        for _ in 0..self.k_epochs {
            let indices: Vec<usize> = (0..len).collect();
            let mut approx_kl = 0.0;
            let mut batch_count = 0.0;

            for chunk in indices.chunks(self.batch_size) {
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

                    let logits = self.policy.forward_actor(state, pity);
                    let value = self.policy.forward_critic(state, pity);

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

                    let mut mask_vec = vec![0.0; ACTION_SPACE];
                    mask_vec[action_idx] = 1.0;
                    let mask = Tensor::new(mask_vec, vec![ACTION_SPACE]);

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
                let final_loss = loss_accum / batch_size_tensor;

                final_loss.backward();
                self.optimizer.step();
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
    let mut ppo = PPO::new(rng.next_u64(), k_epochs, batch_size);
    let mut steps_done = 0;

    let mut state_struct = PullState {
        pity_6: 0,
        total_pulls_in_pool: 0,
        has_obtained_up: false,
        streak_4_star: 0,
        loss_streak: 0,
    };
    let (mut env_noise, mut env_bias) = dbn_env(dbn, rng);
    let mut pulls_done = 0;

    let mut history_buffer: VecDeque<Vec<f64>> = VecDeque::with_capacity(context_len);
    let mut pity_buffer: VecDeque<usize> = VecDeque::with_capacity(context_len);
    let mut flat_data: Vec<f64> = Vec::with_capacity(context_len * DIM);
    let mut pity_vec: Vec<usize> = Vec::with_capacity(context_len);

    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(50);
    let mut _episode_count = 0;

    // Linear LR decay
    let initial_lr = 0.0003;

    while steps_done < total_steps {
        let mut episode_reward = 0.0;

        // Calculate LR
        let progress = steps_done as f64 / total_steps as f64;
        let current_lr = initial_lr * (1.0 - progress).max(0.1); // Decay to 10%

        for _ in 0..steps_per_update {
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

            let current_pity = state_struct.pity_6 as usize;

            history_buffer.push_back(current_state_raw);
            pity_buffer.push_back(current_pity);
            if history_buffer.len() > context_len {
                history_buffer.pop_front();
                pity_buffer.pop_front();
            }

            let seq_len = history_buffer.len();
            flat_data.clear();
            for s in history_buffer.iter() {
                flat_data.extend_from_slice(s);
            }
            let current_state_tensor = Tensor::new(flat_data.clone(), vec![seq_len, DIM]);

            pity_vec.clear();
            pity_vec.extend(pity_buffer.iter().copied());
            let (action_idx, log_prob, val) = ppo.policy.step(&current_state_tensor, &pity_vec);

            let luck_modifier = ACTIONS[action_idx];
            let base_prob_6 = prob_6(state_struct.pity_6, config);
            let final_prob_6 = (base_prob_6 + luck_modifier).clamp(0.0, 1.0);

            let r = rng.next_f64();
            let mut is_six = false;
            let mut is_up = false;

            state_struct.pity_6 += 1;
            state_struct.total_pulls_in_pool += 1;

            if state_struct.total_pulls_in_pool == config.big_pity_cumulative
                && !state_struct.has_obtained_up
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
                if rng.next_f64() < 0.5 {
                    is_up = true;
                    state_struct.loss_streak = 0;
                    state_struct.has_obtained_up = true;
                } else {
                    state_struct.loss_streak += 1;
                }
            } else {
                if state_struct.streak_4_star >= 9 || r < final_prob_6 + config.prob_5_base {
                    state_struct.streak_4_star = 0;
                } else {
                    state_struct.streak_4_star += 1;
                }
            }
            pulls_done += 1;

            let mut reward = -0.1;
            if is_six {
                // Massive reward for getting UP, especially early
                if is_up {
                    reward += 10.0;
                    // Bonus for efficiency: early pulls get more reward
                    if pulls_done < 80 {
                        reward += 5.0;
                    }
                    if pulls_done < 50 {
                        reward += 5.0;
                    }
                } else {
                    // Small reward for non-up 6* to encourage hitting 6* at least
                    reward += 2.0;
                }
            }
            if state_struct.loss_streak >= 2 {
                reward -= (state_struct.loss_streak as f64) * 2.0; // Heavily penalize consecutive losses
            }
            episode_reward += reward;

            let done = is_up || pulls_done >= 300;

            ppo.store(
                current_state_tensor,
                pity_vec.clone(),
                action_idx,
                log_prob,
                reward,
                done,
                val,
            );

            if done {
                history_buffer.clear();
                pity_buffer.clear();
                _episode_count += 1;
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
                let new_env = dbn_env(dbn, rng);
                env_noise = new_env.0;
                env_bias = new_env.1;
                pulls_done = 0;
                episode_reward = 0.0;
            }
        }

        ppo.update(current_lr);
        steps_done += steps_per_update;

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
    ppo.policy
}

pub struct OnlinePpoTrainer {
    ppo: PPO,
    steps_done: usize,
}

impl OnlinePpoTrainer {
    pub fn new(seed: u64, k_epochs: usize, batch_size: usize) -> Self {
        Self {
            ppo: PPO::new(seed, k_epochs, batch_size),
            steps_done: 0,
        }
    }

    pub fn push(&mut self, exp: PpoExperience) {
        self.ppo.store_raw(
            exp.state,
            exp.seq_len,
            exp.pity,
            exp.action,
            exp.log_prob,
            exp.reward,
            exp.done,
            exp.value,
        );
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
        let policy = ActorCritic::new(42);

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
