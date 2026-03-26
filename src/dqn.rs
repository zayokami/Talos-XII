use crate::achf::AchfLayer;
use crate::autograd::{Tensor, TensorReadGuard};
use crate::config::{AchfConfig, Config};
use crate::dbn::Dbn;
use crate::neural::{NeuralLuckOptimizer, DIM};
use crate::nn::{Linear, Module};
use crate::rng::Rng;
use crate::sim::{build_features, dbn_env, prob_6, PullState};
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
use crate::utils::{create_bar, ACTIONS, ACTION_SPACE, EPISODE_MAX_PULLS};

// PER Hyperparameters (Schaul et al. 2016)
const PER_ALPHA: f64 = 0.6;
const PER_BETA_START: f64 = 0.4;
const PER_BETA_END: f64 = 1.0;
const PER_EPSILON: f64 = 1e-6;

// --- Layers ---
// Linear layer is now imported from crate::nn

// --- Dueling Q-Network ---
// Feature Extractor (from NeuralLuckOptimizer) -> Hidden -> Value + Advantage

/// Dueling Q-Network for discrete luck action selection.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DuelingQNetwork {
    l1: Linear,
    l2: Linear,
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
        p.extend(self.val_head.parameters());
        p.extend(self.adv_head.parameters());
        if let Some(achf) = &self.achf {
            p.extend(achf.parameters());
        }
        p
    }
}

impl DuelingQNetwork {
    pub fn new(seed: u64, achf: &AchfConfig) -> Self {
        let l1 = Linear::new(DIM, 1024, true, seed);
        let l2 = Linear::new(1024, 1024, true, seed.wrapping_add(1));
        let val_head = Linear::new(1024, 1, true, seed.wrapping_add(2));
        let adv_head = Linear::new(1024, ACTION_SPACE, true, seed.wrapping_add(3));
        let achf_layer = if achf.enabled && achf.apply_dqn {
            Some(AchfLayer::new(1024, achf.clone(), seed.wrapping_add(500)))
        } else {
            None
        };

        DuelingQNetwork {
            l1,
            l2,
            val_head,
            adv_head,
            achf: achf_layer,
        }
    }

    pub fn forward_impl(&self, state: &Tensor) -> Tensor {
        // state: (Batch, 8) or (8)
        let x = self.l1.forward(state).relu();
        let mut x = self.l2.forward(&x).relu();
        if let Some(achf) = &self.achf {
            let residual = achf.forward_residual(&x);
            x = &x + &residual;
        }

        let val = self.val_head.forward(&x); // (Batch, 1) or (1)
        let adv = self.adv_head.forward(&x); // (Batch, 5) or (5)

        // Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))

        if state.shape.len() == 2 && state.shape[0] > 1 {
            // Batch Mode

            // val is (B, 1). Expand to (B, 5).
            // Multiply by ones(1, 5) -> (B, 5)
            // MatMul: (B, 1) x (1, 5) -> (B, 5)
            let ones_1_5 = Tensor::new(vec![1.0; 5], vec![1, 5]);
            let val_expanded = val.matmul(&ones_1_5);

            // Mean Adv: (B, 5) -> (B, 1)
            // Multiply by ones(5, 1) / 5.0
            let ones_5_1 = Tensor::new(vec![0.2; 5], vec![5, 1]); // 1/5 = 0.2
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

    pub fn update_achf_after_backward(&self) {
        if let Some(achf) = &self.achf {
            achf.update_after_backward();
        }
    }

    pub fn freeze_achf_for_inference(&self) {
        if let Some(achf) = &self.achf {
            achf.freeze_for_inference();
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

    // Copy weights
    pub fn load_state_dict(&mut self, other: &Self) {
        fn copy_tensor(dst: &mut Tensor, src: &Tensor) {
            let src_data = src.data.read().unwrap().clone();
            let mut dst_data = dst.data.write().unwrap();
            *dst_data = src_data;
        }

        let copy_linear = |dst: &mut Linear, src: &Linear| {
            copy_tensor(&mut dst.weight, &src.weight);
            if let (Some(db), Some(sb)) = (&mut dst.bias, &src.bias) {
                copy_tensor(db, sb);
            }
        };

        copy_linear(&mut self.l1, &other.l1);
        copy_linear(&mut self.l2, &other.l2);
        copy_linear(&mut self.val_head, &other.val_head);
        copy_linear(&mut self.adv_head, &other.adv_head);
        if let (Some(dst), Some(src)) = (&mut self.achf, &other.achf) {
            dst.load_state_dict(src);
        }
    }

    pub fn soft_update(&mut self, source: &Self, tau: f64) {
        fn interpolate(target: &mut Tensor, source: &Tensor, tau: f64) {
            let mut t_data = target.data.write().unwrap();
            let s_data = source.data.read().unwrap();
            for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                *t = *t * (1.0 - tau) + *s * tau;
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
        update_linear(&mut self.val_head, &source.val_head);
        update_linear(&mut self.adv_head, &source.adv_head);
        if let (Some(dst), Some(src)) = (&mut self.achf, &source.achf) {
            dst.soft_update(src, tau);
        }
    }

    pub fn predict_action(&self, state: &Tensor) -> (usize, f64) {
        let q_values = self.forward(state);
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;
        let q_data = q_values.data.read().unwrap();
        for (i, &val) in q_data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        (max_idx, ACTIONS[max_idx])
    }

    /// Zero-allocation inference: compute Q-values from a raw feature slice
    /// using `Linear::forward_inference_into`, bypassing the autograd `Tensor` graph.
    ///
    /// This function uses thread-local scratch buffers to avoid allocations in hot paths.
    pub fn predict_action_fast(&self, state: &[f64]) -> (usize, f64) {
        struct Scratch {
            h1: Vec<f64>,
            h2: Vec<f64>,
            val: Vec<f64>,
            adv: Vec<f64>,
        }

        thread_local! {
            static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch {
                h1: Vec::new(),
                h2: Vec::new(),
                val: Vec::new(),
                adv: Vec::new(),
            }) };
        }

        SCRATCH.with(|scratch| {
            let mut s = scratch.borrow_mut();
            let Scratch { h1, h2, val, adv } = &mut *s;

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
                let residual = achf.forward_inference_residual(h2);
                for (dst, &r) in h2.iter_mut().zip(residual.iter()) {
                    *dst += r;
                }
            }

            self.val_head.forward_inference_into(h2, val);
            self.adv_head.forward_inference_into(h2, adv);

            let mean_adv: f64 = adv.iter().sum::<f64>() / ACTION_SPACE as f64;
            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0;
            let base = val.first().copied().unwrap_or(0.0);
            for (i, &a) in adv.iter().enumerate() {
                let q = base + a - mean_adv;
                if q > max_val {
                    max_val = q;
                    max_idx = i;
                }
            }
            (max_idx, ACTIONS[max_idx])
        })
    }
}

// --- Optimizer ---

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
                data[j] -=
                    self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * data[j]);
            }
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
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
}

struct ReplayBuffer {
    tree: SumTree,
    alpha: f64,
    max_priority: f64,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        ReplayBuffer {
            tree: SumTree::new(capacity),
            alpha: PER_ALPHA,
            max_priority: 1.0,
        }
    }

    fn push(&mut self, exp: Experience) {
        let priority = self.max_priority.powf(self.alpha);
        self.tree.add(priority, exp);
    }

    /// Proportional PER sampling with importance-sampling weights.
    /// `beta` controls IS correction strength (annealed from PER_BETA_START to PER_BETA_END).
    fn sample(&self, rng: &mut Rng, batch_size: usize, beta: f64) -> PERSample {
        assert!(batch_size > 0, "batch_size must be > 0");
        assert!(self.tree.size > 0, "cannot sample from empty buffer");

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
}

// --- Training Loop ---

/// Train a DQN agent using Double DQN with Prioritized Experience Replay.
pub fn train_dqn(
    _initial_model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    dbn: &Dbn,
    config: &Config,
) -> DuelingQNetwork {
    train_dqn_impl(_initial_model, rng, dbn, config, None)
}

fn train_dqn_impl(
    _initial_model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    dbn: &Dbn,
    config: &Config,
    metrics_tx: Option<std::sync::mpsc::Sender<crate::bench::StepSnapshot>>,
) -> DuelingQNetwork {
    println!("\n[DQN] Initializing Double Dueling DQN Training...");

    let policy_net = DuelingQNetwork::new(rng.next_u64(), &config.achf);
    let mut target_net = DuelingQNetwork::new(rng.next_u64(), &config.achf);
    target_net.load_state_dict(&policy_net); // Sync weights

    let mut optimizer = Adam::new(policy_net.parameters(), LEARNING_RATE);
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
    let (mut env_noise, mut env_bias) = dbn_env(dbn, rng);
    let mut pulls_done = 0;

    let mut episode_reward = 0.0;
    let mut episode_count = 0;
    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(51);

    let beta_anneal_steps = total_steps as f64;
    let snapshot_every = (total_steps / 200).max(1);
    let mut last_train_loss = 0.0_f64;

    let pb = create_bar(total_steps as u64, "DQN Training");

    for step in 0..total_steps {
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

        let current_state_tensor = Tensor::new(current_state_raw.clone(), vec![DIM]);

        // 2. Select Action
        let action = if rng.next_f64() < epsilon {
            rng.next_u64_bounded(ACTION_SPACE as u64) as usize
        } else {
            let q_values = policy_net.forward(&current_state_tensor);
            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0;
            let q_data = q_values.data.read().unwrap();
            for (i, &val) in q_data.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            max_idx
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

            let mut states_vec = Vec::with_capacity(BATCH_SIZE * DIM);
            let mut next_states_vec = Vec::with_capacity(BATCH_SIZE * DIM);
            let mut actions_vec = Vec::with_capacity(BATCH_SIZE * ACTION_SPACE);
            let mut rewards_vec = Vec::with_capacity(BATCH_SIZE);
            let mut dones_vec = Vec::with_capacity(BATCH_SIZE);

            for exp in &per_sample.experiences {
                states_vec.extend_from_slice(&exp.state);
                next_states_vec.extend_from_slice(&exp.next_state);

                let mut mask = vec![0.0; ACTION_SPACE];
                mask[exp.action] = 1.0;
                actions_vec.extend_from_slice(&mask);

                rewards_vec.push(exp.reward);
                dones_vec.push(if exp.done { 1.0 } else { 0.0 });
            }

            let batch_state = Tensor::new(states_vec, vec![BATCH_SIZE, DIM]);
            let batch_next_state = Tensor::new(next_states_vec, vec![BATCH_SIZE, DIM]);
            let batch_mask = Tensor::new(actions_vec, vec![BATCH_SIZE, ACTION_SPACE]);

            // 2. Policy Forward
            let q_values = policy_net.forward(&batch_state); // (B, 5)

            // Select Action Q-Values: (B, 5) * (B, 5) -> (B, 5) [one non-zero per row]
            // Sum across dim 1 to get (B, 1)
            // MatMul by ones(5, 1) -> (B, 1)
            let ones_5_1 = Tensor::new(vec![1.0; 5], vec![5, 1]);
            let q_actions = (q_values * batch_mask).matmul(&ones_5_1); // (B, 1)

            // 3. Compute Targets (Double DQN)
            // Select action using Policy Net
            let q_next_eval = policy_net.forward(&batch_next_state); // (B, 5)

            // Evaluate value using Target Net
            let q_next_target = target_net.forward(&batch_next_state); // (B, 5)

            // Use batch lock for better performance
            let guards = TensorReadGuard::new(&[&q_next_eval, &q_next_target]);
            let q_next_eval_data = guards.get(0);
            let q_next_target_data = guards.get(1);

            let mut target_vals = Vec::with_capacity(BATCH_SIZE);

            for i in 0..BATCH_SIZE {
                let start = i * ACTION_SPACE;
                let end = start + ACTION_SPACE;

                // Argmax from Policy Net
                let row_eval = &q_next_eval_data[start..end];
                let mut max_idx = 0;
                let mut max_val = f64::NEG_INFINITY;
                for (k, &v) in row_eval.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = k;
                    }
                }

                // Value from Target Net
                let next_q_val = q_next_target_data[start + max_idx];

                let r = rewards_vec[i];
                let d = dones_vec[i];
                // if done (d=1.0), target = r. else r + gamma * next_q_val
                let target = r + GAMMA * next_q_val * (1.0 - d);
                target_vals.push(target);
            }

            let target_tensor = Tensor::new(target_vals, vec![BATCH_SIZE, 1]);

            // IS-weighted loss: w_i * (q - target)^2, normalized
            let is_weights_tensor = Tensor::new(per_sample.is_weights.clone(), vec![BATCH_SIZE, 1]);
            let mut loss = q_actions.weighted_mse_loss(&target_tensor, &is_weights_tensor);
            if let Some(reg) = policy_net.achf_orthogonal_penalty() {
                loss = loss + reg;
            }

            last_train_loss = loss.data.read().unwrap()[0];
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
                let q_data = q_actions.data.read().unwrap();
                let t_data = target_tensor.data.read().unwrap();
                let td_errors: Vec<f64> = q_data
                    .iter()
                    .zip(t_data.iter())
                    .map(|(&q, &t)| (q - t).abs())
                    .collect();
                replay_buffer.update_priorities(&per_sample.indices, &td_errors);
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
            let new_env = dbn_env(dbn, rng);
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

        if let Some(ref tx) = metrics_tx {
            if step % snapshot_every == 0 {
                let avg_r = if recent_rewards.is_empty() {
                    0.0
                } else {
                    recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64
                };
                let achf_snap = policy_net.snapshot_achf();
                let snapshot = crate::bench::StepSnapshot {
                    step,
                    gate_value: achf_snap.map_or(1.0, |s| s.gate),
                    g_min: achf_snap.map_or(0.0, |s| s.g_min),
                    grad_ema: achf_snap.map_or(0.0, |s| s.grad_ema),
                    loss: last_train_loss,
                    reward: avg_r,
                    cache_hit_rate: achf_snap.map_or(0.0, |s| s.cache_hit_rate),
                    low_rank_ratio: achf_snap.map_or(0.0, |s| s.low_rank_ratio),
                    ema_cached_ns: achf_snap.map_or(0.0, |s| s.ema_cached_ns),
                    ema_low_rank_ns: achf_snap.map_or(0.0, |s| s.ema_low_rank_ns),
                    adaptive_bias: achf_snap.map_or(1.0, |s| s.adaptive_bias),
                };
                let _ = tx.send(snapshot);
            }
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
    policy_net.freeze_achf_for_inference();
    policy_net
}

/// Train a DQN agent with optional metrics collection for benchmarking.
pub fn train_dqn_with_metrics(
    initial_model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    dbn: &Dbn,
    config: &Config,
    metrics_tx: Option<std::sync::mpsc::Sender<crate::bench::StepSnapshot>>,
) -> DuelingQNetwork {
    train_dqn_impl(initial_model, rng, dbn, config, metrics_tx)
}

/// Incremental DQN trainer for online learning during interactive mode.
pub struct OnlineDqnTrainer {
    policy: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: Adam,
    replay_buffer: ReplayBuffer,
    steps_done: usize,
}

impl OnlineDqnTrainer {
    pub fn from_policy(policy: DuelingQNetwork, seed: u64) -> Self {
        let achf = policy.achf_config();
        let mut target = DuelingQNetwork::new(seed, &achf);
        target.load_state_dict(&policy);
        let optimizer = Adam::new(policy.parameters(), LEARNING_RATE);
        Self {
            policy,
            target,
            optimizer,
            replay_buffer: ReplayBuffer::new(BUFFER_CAPACITY),
            steps_done: 0,
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

        let mut states_vec = Vec::with_capacity(BATCH_SIZE * DIM);
        let mut next_states_vec = Vec::with_capacity(BATCH_SIZE * DIM);
        let mut actions_vec = Vec::with_capacity(BATCH_SIZE * ACTION_SPACE);
        let mut rewards_vec = Vec::with_capacity(BATCH_SIZE);
        let mut dones_vec = Vec::with_capacity(BATCH_SIZE);

        for exp in &per_sample.experiences {
            states_vec.extend_from_slice(&exp.state);
            next_states_vec.extend_from_slice(&exp.next_state);
            let mut mask = vec![0.0; ACTION_SPACE];
            mask[exp.action] = 1.0;
            actions_vec.extend_from_slice(&mask);
            rewards_vec.push(exp.reward);
            dones_vec.push(if exp.done { 1.0 } else { 0.0 });
        }

        let batch_state = Tensor::new(states_vec, vec![BATCH_SIZE, DIM]);
        let batch_next_state = Tensor::new(next_states_vec, vec![BATCH_SIZE, DIM]);
        let batch_mask = Tensor::new(actions_vec, vec![BATCH_SIZE, ACTION_SPACE]);

        let q_values = self.policy.forward(&batch_state);
        let ones_5_1 = Tensor::new(vec![1.0; 5], vec![5, 1]);
        let q_actions = (q_values * batch_mask).matmul(&ones_5_1);

        let q_next_eval = self.policy.forward(&batch_next_state);
        let q_next_target = self.target.forward(&batch_next_state);

        // Use batch lock for better performance
        let guards = TensorReadGuard::new(&[&q_next_eval, &q_next_target]);
        let q_next_eval_data = guards.get(0);
        let q_next_target_data = guards.get(1);

        let mut target_vals = Vec::with_capacity(BATCH_SIZE);
        for i in 0..BATCH_SIZE {
            let start = i * ACTION_SPACE;
            let end = start + ACTION_SPACE;
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
            let r = rewards_vec[i];
            let d = dones_vec[i];
            let target = r + GAMMA * next_q_val * (1.0 - d);
            target_vals.push(target);
        }
        let target_tensor = Tensor::new(target_vals, vec![BATCH_SIZE, 1]);
        let is_weights_tensor = Tensor::new(per_sample.is_weights.clone(), vec![BATCH_SIZE, 1]);
        let mut loss = q_actions.weighted_mse_loss(&target_tensor, &is_weights_tensor);
        if let Some(reg) = self.policy.achf_orthogonal_penalty() {
            loss = loss + reg;
        }
        loss.backward();
        self.policy.update_achf_after_backward();
        self.optimizer.step();

        // Write back per-sample TD errors for priority update
        {
            let q_data = q_actions.data.read().unwrap();
            let t_data = target_tensor.data.read().unwrap();
            let td_errors: Vec<f64> = q_data
                .iter()
                .zip(t_data.iter())
                .map(|(&q, &t)| (q - t).abs())
                .collect();
            self.replay_buffer
                .update_priorities(&per_sample.indices, &td_errors);
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

    pub fn steps_done(&self) -> usize {
        self.steps_done
    }

    pub fn buffer_len(&self) -> usize {
        self.replay_buffer.len()
    }
}
