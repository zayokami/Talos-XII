use crate::neural::{NeuralLuckOptimizer, DIM};
use crate::rng::Rng;
use crate::config::Config;
use crate::dbn::Dbn;
use crate::{build_features, PullState, dbn_env, prob_6};
use crate::autograd::Tensor;

// DQN Hyperparameters
const GAMMA: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const BUFFER_CAPACITY: usize = 10000;
const EPSILON_START: f64 = 1.0;
const EPSILON_END: f64 = 0.1;
const EPSILON_DECAY: usize = 50000;
const LEARNING_RATE: f64 = 0.001;
const TRAIN_FREQ: usize = 10; // Train every 10 steps to improve performance
const LOG_FREQ: usize = 100;  // Log every 100 steps

// Actions
const ACTION_SPACE: usize = 5;
const ACTIONS: [f64; ACTION_SPACE] = [0.0, 0.005, 0.015, -0.005, -0.015];

// --- Layers ---

#[derive(Clone)]
struct Linear {
    weights: Tensor, // (In, Out)
    bias: Tensor,    // (Out)
}

impl Linear {
    fn new(in_features: usize, out_features: usize, rng_seed: u64) -> Self {
        // Xavier initialization
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let weights = Tensor::rand(vec![in_features, out_features], -limit, limit, rng_seed);
        let bias = Tensor::zeros(vec![out_features]);
        Linear { weights, bias }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // input: (Batch, In)
        let out = input.matmul(&self.weights);
        out + self.bias.clone()
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

// --- Dueling Q-Network ---
// Feature Extractor (from NeuralLuckOptimizer) -> Hidden -> Value + Advantage

#[derive(Clone)]
pub struct DuelingQNetwork {
    l1: Linear,
    l2: Linear,
    val_head: Linear,
    adv_head: Linear,
}

impl DuelingQNetwork {
    pub fn new(seed: u64) -> Self {
        let l1 = Linear::new(DIM, 64, seed);
        let l2 = Linear::new(64, 64, seed.wrapping_add(1));
        let val_head = Linear::new(64, 1, seed.wrapping_add(2));
        let adv_head = Linear::new(64, ACTION_SPACE, seed.wrapping_add(3));
        
        DuelingQNetwork { l1, l2, val_head, adv_head }
    }

    pub fn forward(&self, state: &Tensor) -> Tensor {
        // state: (8)
        let x = self.l1.forward(state).relu();
        let x = self.l2.forward(&x).relu();
        
        let val = self.val_head.forward(&x); // (1)
        let adv = self.adv_head.forward(&x); // (5)
        
        // Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        // Calculate mean of Advantage
        let mean_adv_scalar = adv.mean(); // (1)
        let mean_adv = mean_adv_scalar.broadcast(vec![ACTION_SPACE]); // (5)
        
        // Expand Value to (5)
        let val_expanded = val.broadcast(vec![ACTION_SPACE]); // (5)
        
        // Q = V + A - Mean(A)
        // Note: We need negation. Tensor doesn't have neg() yet, use 0 - mean_adv
        let zero = Tensor::zeros(vec![ACTION_SPACE]);
        let neg_mean_adv = zero - mean_adv;
        
        val_expanded + adv + neg_mean_adv
    }
    
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.l1.parameters());
        p.extend(self.l2.parameters());
        p.extend(self.val_head.parameters());
        p.extend(self.adv_head.parameters());
        p
    }
    
    // Copy weights
    pub fn load_state_dict(&mut self, other: &Self) {
        // We need a helper to copy data specifically.
        fn copy_tensor(dst: &mut Tensor, src: &Tensor) {
            let src_data = src.data.read().unwrap().clone();
            let mut dst_data = dst.data.write().unwrap();
            *dst_data = src_data;
        }
        
        copy_tensor(&mut self.l1.weights, &other.l1.weights);
        copy_tensor(&mut self.l1.bias, &other.l1.bias);
        copy_tensor(&mut self.l2.weights, &other.l2.weights);
        copy_tensor(&mut self.l2.bias, &other.l2.bias);
        copy_tensor(&mut self.val_head.weights, &other.val_head.weights);
        copy_tensor(&mut self.val_head.bias, &other.val_head.bias);
        copy_tensor(&mut self.adv_head.weights, &other.adv_head.weights);
        copy_tensor(&mut self.adv_head.bias, &other.adv_head.bias);
    }

    pub fn soft_update(&mut self, source: &Self, tau: f64) {
        fn interpolate(target: &mut Tensor, source: &Tensor, tau: f64) {
            let mut t_data = target.data.write().unwrap();
            let s_data = source.data.read().unwrap();
            for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                *t = *t * (1.0 - tau) + *s * tau;
            }
        }
        
        interpolate(&mut self.l1.weights, &source.l1.weights, tau);
        interpolate(&mut self.l1.bias, &source.l1.bias, tau);
        interpolate(&mut self.l2.weights, &source.l2.weights, tau);
        interpolate(&mut self.l2.bias, &source.l2.bias, tau);
        interpolate(&mut self.val_head.weights, &source.val_head.weights, tau);
        interpolate(&mut self.val_head.bias, &source.val_head.bias, tau);
        interpolate(&mut self.adv_head.weights, &source.adv_head.weights, tau);
        interpolate(&mut self.adv_head.bias, &source.adv_head.bias, tau);
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
}

impl Adam {
    fn new(params: Vec<Tensor>, lr: f64) -> Self {
        // We need to read the data length, so we lock
        let m = params.iter().map(|p| vec![0.0; p.data.read().unwrap().len()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.data.read().unwrap().len()]).collect();
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
        for param in &self.params {
            param.zero_grad();
        }
    }
}

// --- Replay Buffer ---

#[derive(Clone)]
pub struct Experience {
    state: Vec<f64>,
    action: usize,
    reward: f64,
    next_state: Vec<f64>,
    done: bool,
}

struct ReplayBuffer {
    buffer: Vec<Experience>,
    position: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: Vec::with_capacity(capacity),
            position: 0,
        }
    }

    fn push(&mut self, exp: Experience) {
        if self.buffer.len() < self.buffer.capacity() {
            self.buffer.push(exp);
        } else {
            self.buffer[self.position] = exp;
        }
        self.position = (self.position + 1) % self.buffer.capacity();
    }

    fn sample(&self, rng: &mut Rng, batch_size: usize) -> Vec<Experience> {
        let len = self.buffer.len();
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = rng.next_u64_bounded(len as u64) as usize;
            batch.push(self.buffer[idx].clone());
        }
        batch
    }
    
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

// --- Training Loop ---

pub fn train_dqn(
    _initial_model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    dbn: &Dbn,
    config: &Config,
) -> DuelingQNetwork {
    println!("\n[DQN] Initializing Double Dueling DQN Training...");
    
    let policy_net = DuelingQNetwork::new(rng.next_u64());
    let mut target_net = DuelingQNetwork::new(rng.next_u64());
    target_net.load_state_dict(&policy_net); // Sync weights
    
    let mut optimizer = Adam::new(policy_net.parameters(), LEARNING_RATE);
    let mut replay_buffer = ReplayBuffer::new(BUFFER_CAPACITY);
    
    let total_steps = if config.fast_init { 10_000 } else { 50_000 };
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
    let mut recent_rewards = Vec::new();

    for step in 0..total_steps {
        // 1. Build State
        let current_state_raw = build_features(
            state_struct.pity_6,
            pulls_done,
            env_noise,
            state_struct.streak_4_star,
            env_bias,
            state_struct.loss_streak,
            config
        ).to_vec();
        
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
        
        if state_struct.total_pulls_in_pool == config.big_pity_cumulative && !state_struct.has_obtained_up {
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
            if is_up { reward += 10.0; } else { reward += 2.0; }
        }
        if state_struct.loss_streak >= 2 {
            reward -= (state_struct.loss_streak as f64) * 0.5;
        }
        
        episode_reward += reward;
        
        let next_state_raw = build_features(
            state_struct.pity_6,
            pulls_done,
            env_noise,
            state_struct.streak_4_star,
            env_bias,
            state_struct.loss_streak,
            config
        ).to_vec();
        
        let done = is_up || pulls_done >= 300;
        
        replay_buffer.push(Experience {
            state: current_state_raw,
            action,
            reward,
            next_state: next_state_raw,
            done,
        });
        
        // 4. Train
        if replay_buffer.len() > BATCH_SIZE && step % TRAIN_FREQ == 0 {
            let batch = replay_buffer.sample(rng, BATCH_SIZE);
            optimizer.zero_grad();
            
            let mut total_loss = Tensor::new(vec![0.0], vec![1]);
            
            // Process batch
            for exp in batch {
                let s = Tensor::new(exp.state, vec![DIM]);
                let ns = Tensor::new(exp.next_state, vec![DIM]);
                
                let q_s = policy_net.forward(&s);
                let mut mask_vec = vec![0.0; ACTION_SPACE];
                mask_vec[exp.action] = 1.0;
                let mask = Tensor::new(mask_vec, vec![ACTION_SPACE]);
                let q_a = (q_s * mask).sum();
                
                let q_ns_policy = policy_net.forward(&ns); 
                let mut max_val = f64::NEG_INFINITY;
                let mut best_a = 0;
                {
                    let q_data = q_ns_policy.data.read().unwrap();
                    for (i, &val) in q_data.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            best_a = i;
                        }
                    }
                }
                
                let q_ns_target = target_net.forward(&ns);
                let target_val = if exp.done {
                    exp.reward
                } else {
                    exp.reward + GAMMA * q_ns_target.data.read().unwrap()[best_a]
                };
                
                let target_tensor = Tensor::new(vec![target_val], vec![1]);
                
                let loss = q_a.mse_loss(&target_tensor);
                total_loss = total_loss + loss;
            }
            
            total_loss.backward();
            optimizer.step();
            
            // Soft Update Target Network
            target_net.soft_update(&policy_net, 0.005);
        }
        
        // Removed hard update logic (step % TARGET_UPDATE_FREQ == 0)
        // if step % TARGET_UPDATE_FREQ == 0 { ... }
        
        if epsilon > EPSILON_END {
            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY as f64;
        }
        
        if done {
            episode_count += 1;
            recent_rewards.push(episode_reward);
            if recent_rewards.len() > 50 { recent_rewards.remove(0); }
            
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
            let avg_r = if recent_rewards.is_empty() { 0.0 } else { recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64 };
            print!("\r[DQN] Step {:>6}/{} | Ep {:>4} | Avg R: {:>6.2} | Eps: {:.3}", step, total_steps, episode_count, avg_r, epsilon);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    println!("\n[DQN] Training Complete.");
    policy_net
}
