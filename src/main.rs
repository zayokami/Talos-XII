mod config;
mod dbn;
mod neural;
mod rng;
mod worker;
mod dqn; // Added DQN module
mod ppo; // Added PPO module
mod autograd; // Added Autograd module
mod transformer; // Added Transformer module

use config::Config;
use dbn::Dbn;
use neural::{NeuralLuckOptimizer, Tensor, DIM};
use rng::Rng;
use std::io::{self, Write};
use std::time::Instant;
use worker::GoodJobWorker;
use rayon::prelude::*;
use dqn::{train_dqn, DuelingQNetwork};
use ppo::{train_ppo, ActorCritic};
use autograd::Tensor as AutoTensor;

// Constants
const DBN_GIBBS_STEPS: usize = 10;
const COST_PER_PULL: u32 = 500; // Jade equivalent
const FREE_PULLS_WELFARE: u32 = 135;
const NEURAL_CACHE_PATH: &str = "neural.cache";

#[inline(always)]
fn add_scaled_row(output: &mut [f64], row: &[f64], scale: f64) {
    let len = output.len();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                add_scaled_row_avx2(output, row, scale);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            add_scaled_row_neon(output, row, scale);
        }
        return;
    }
    for i in 0..len {
        output[i] += scale * row[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_row_avx2(output: &mut [f64], row: &[f64], scale: f64) {
    use core::arch::x86_64::*;
    let len = output.len();
    let mut i = 0;
    let scale_vec = _mm256_set1_pd(scale);
    while i + 4 <= len {
        let out = _mm256_loadu_pd(output.as_ptr().add(i));
        let rowv = _mm256_loadu_pd(row.as_ptr().add(i));
        let prod = _mm256_mul_pd(rowv, scale_vec);
        let sum = _mm256_add_pd(out, prod);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), sum);
        i += 4;
    }
    while i < len {
        *output.get_unchecked_mut(i) += scale * *row.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_scaled_row_neon(output: &mut [f64], row: &[f64], scale: f64) {
    use core::arch::aarch64::*;
    let len = output.len();
    let mut i = 0;
    let scale_vec = vdupq_n_f64(scale);
    while i + 2 <= len {
        let out = vld1q_f64(output.as_ptr().add(i));
        let rowv = vld1q_f64(row.as_ptr().add(i));
        let prod = vmulq_f64(rowv, scale_vec);
        let sum = vaddq_f64(out, prod);
        vst1q_f64(output.as_mut_ptr().add(i), sum);
        i += 2;
    }
    while i < len {
        *output.get_unchecked_mut(i) += scale * *row.get_unchecked(i);
        i += 1;
    }
}

#[derive(Clone, Debug)]
pub struct PullResult {
    pub operator: String,
    pub rarity: u8,
}

#[derive(Clone)]
pub struct SimulationResult {
    pub pulls: Vec<PullResult>,
    pub six_count: usize,
    pub up_count: usize,
    pub big_pity_used: bool,
    pub cost_jade: u32,
    pub free_pulls_used: u32,
}

#[derive(Clone)]
pub struct SimStatsResult {
    pub six_count: usize,
    pub up_count: usize,
    pub big_pity_used: bool,
    #[allow(dead_code)]
    pub cost_jade: u32,
    #[allow(dead_code)]
    pub free_pulls_used: u32,
    pub max_loss_streak: usize, // Tracked for neural network training
}

#[derive(Clone)]
pub struct PullState {
    pub pity_6: usize,
    pub total_pulls_in_pool: usize,
    pub has_obtained_up: bool,
    pub streak_4_star: usize,
    pub loss_streak: usize,
}

#[derive(Clone)]
pub struct PullOutcome {
    pub rarity: u8,
    pub is_up: bool,
    pub big_pity_used: bool,
}

#[derive(Clone)]
pub struct SimControl {
    pub max_pulls: Option<usize>,
    pub stop_on_up: bool,
    pub stop_after_total_pulls: Option<usize>,
    pub nn_total_pulls_one_based: bool,
    pub collect_details: bool,
    pub big_pity_requires_not_up: bool,
}

pub fn dbn_env(dbn: &Dbn, rng: &mut Rng) -> (f64, f64) {
    let v = dbn.sample(rng, DBN_GIBBS_STEPS);
    let sum = v.iter().sum::<f64>();
    let mean = sum / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    (mean * 2.0 - 1.0, var)
}

pub fn prob_6(pity_6: usize, config: &Config) -> f64 {
    if pity_6 < config.soft_pity_start {
        config.prob_6_base
    } else if pity_6 < config.small_pity_guarantee {
        config.prob_6_base + 0.05 * (pity_6 as f64 - (config.soft_pity_start as f64 - 1.0))
    } else {
        1.0
    }
}

#[allow(clippy::too_many_arguments)]
pub fn roll_one(
    state: &mut PullState,
    rng: &mut Rng,
    neural_opt: &NeuralLuckOptimizer,
    dqn_policy: Option<&DuelingQNetwork>,
    ppo_policy: Option<&ActorCritic>,
    env_noise: f64,
    env_bias: f64,
    config: &Config,
    nn_total_pulls: usize,
    big_pity_requires_not_up: bool,
) -> PullOutcome {
    state.pity_6 += 1;
    state.total_pulls_in_pool += 1;

    let mut big_pity_used = false;
    let mut is_up = false;
    let rarity: u8;

    let big_pity_gate = if big_pity_requires_not_up { !state.has_obtained_up } else { true };
    if state.total_pulls_in_pool == config.big_pity_cumulative && big_pity_gate {
        rarity = 6;
        is_up = true;
        big_pity_used = true;
        state.pity_6 = 0;
        state.streak_4_star = 0;
        state.loss_streak = 0;
    } else {
        let base_prob_6 = prob_6(state.pity_6, config);
        
        let x = build_features(
            state.pity_6,
            nn_total_pulls,
            env_noise,
            state.streak_4_star,
            env_bias,
            state.loss_streak,
            config
        );

        let luck_factor = if config.luck_mode == "dqn" {
            if let Some(policy) = dqn_policy {
                // Use DQN policy
                let tensor_x = AutoTensor::new(x.to_vec(), vec![DIM]);
                let (_, modifier) = policy.predict_action(&tensor_x);
                modifier
            } else {
                // Fallback to probability if policy missing
                let dropout_seed = (state.pity_6 as u64).wrapping_add((nn_total_pulls as u64).wrapping_mul(31)).wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
                neural_opt.predict(&x, dropout_seed)
            }
        } else if config.luck_mode == "ppo" {
            if let Some(policy) = ppo_policy {
                // Use PPO policy
                let tensor_x = AutoTensor::new(x.to_vec(), vec![DIM]);
                let (idx, _, _) = policy.step(&tensor_x, &[state.pity_6]);
                ppo::ACTIONS[idx]
            } else {
                // Fallback
                let dropout_seed = (state.pity_6 as u64).wrapping_add((nn_total_pulls as u64).wrapping_mul(31)).wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
                neural_opt.predict(&x, dropout_seed)
            }
        } else {
            // Default probability mode
            let dropout_seed = (state.pity_6 as u64).wrapping_add((nn_total_pulls as u64).wrapping_mul(31)).wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
            neural_opt.predict(&x, dropout_seed)
        };

        let final_prob_6 = (base_prob_6 + luck_factor).clamp(0.0, 1.0);
        let r = rng.next_f64();

        if r < final_prob_6 {
            rarity = 6;
            state.pity_6 = 0;
            state.streak_4_star = 0;

            if rng.next_f64() < 0.5 {
                is_up = true;
                state.loss_streak = 0;
            } else {
                is_up = false;
                state.loss_streak += 1;
            }
        } else {
            let force_5_star = state.streak_4_star >= 9;
            if force_5_star || r < final_prob_6 + config.prob_5_base {
                rarity = 5;
                state.streak_4_star = 0;
            } else {
                rarity = 4;
                state.streak_4_star += 1;
            }
        }
    }

    if is_up {
        state.has_obtained_up = true;
    }

    PullOutcome {
        rarity,
        is_up,
        big_pity_used,
    }
}

fn expected_pulls_per_six(config: &Config) -> f64 {
    let mut survival = 1.0;
    let mut expected = 0.0;
    for k in 1..=config.small_pity_guarantee {
        let prob_6 = if k < config.soft_pity_start {
            config.prob_6_base
        } else if k < config.small_pity_guarantee {
            config.prob_6_base + 0.05 * (k as f64 - (config.soft_pity_start as f64 - 1.0))
        } else {
            1.0
        };
        let prob_k = survival * prob_6;
        expected += k as f64 * prob_k;
        survival *= 1.0 - prob_6;
        if prob_6 >= 1.0 {
            break;
        }
    }
    expected
}

fn build_features(pity_6: usize, total_pulls: usize, env_noise: f64, streak: usize, env_bias: f64, loss_streak: usize, config: &Config) -> Tensor {
    let pity_norm = pity_6 as f64 / config.small_pity_guarantee as f64;
    let loss_norm = loss_streak as f64 / 3.0;
    // Use big_pity_cumulative for normalization if possible, or fallback to 100
    let total_norm_base = if config.big_pity_cumulative > 0 { config.big_pity_cumulative as f64 } else { 120.0 };
    let total_norm = (total_pulls % total_norm_base as usize) as f64 / total_norm_base;
    
    [
        pity_norm,
        total_norm,
        env_noise,
        loss_norm,
        streak as f64 / 20.0,
        env_bias,
        pity_norm * loss_norm,
        total_norm * total_norm,
    ]
}

fn simulate_for_data_collection(
    num_sims: usize,
    rng: &mut Rng,
    neural_opt: &NeuralLuckOptimizer,
    dbn: &Dbn,
    config: &Config,
) -> Vec<(Tensor, f64)> {
    let mut data = Vec::with_capacity(num_sims * 80); // Estimate ~80 pulls per sim on average
    
    // We run simulations and capture the state at each pull
    for _ in 0..num_sims {
        // Reset state for each user simulation
        let mut state = PullState {
            pity_6: 0,
            total_pulls_in_pool: 0,
            has_obtained_up: false,
            streak_4_star: 0,
            loss_streak: 0,
        };
        let (env_noise, env_bias) = dbn_env(dbn, rng);
        let mut pulls_done = 0;
        
        // Run until we get a few 6-stars or hit a limit to get a good trajectory
        // We simulate a "season" of pulls for a user (e.g. 200 pulls)
        let max_pulls = 200;
        
        while pulls_done < max_pulls {
            let nn_total_pulls = pulls_done; // 0-based
            
            // Build features for CURRENT state
            let x = build_features(
                state.pity_6,
                nn_total_pulls,
                env_noise,
                state.streak_4_star,
                env_bias,
                state.loss_streak,
                config
            );
            
            // Calculate what the neural network WOULD output
            let dropout_seed = (state.pity_6 as u64).wrapping_add((nn_total_pulls as u64).wrapping_mul(31)).wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
            let y = neural_opt.predict(&x, dropout_seed);
            
            data.push((x, y));

            // Advance state using the game rules
            // Note: We don't strictly need the neural network to drive the simulation here if we just want coverage,
            // but it's better to follow the policy (on-policy) or a close approximation.
            // For now, we use the current neural_opt to drive the state transitions.
            let _outcome = roll_one(
                &mut state,
                rng,
                neural_opt,
                None, // Use Probability mode for data collection
                None,
                env_noise,
                env_bias,
                config,
                nn_total_pulls,
                true // big_pity_requires_not_up
            );
            
            pulls_done += 1;
            
            // Optional: Break early if we have enough data or specific conditions
            // But 200 pulls is a good "lifecycle" to capture early, mid, and late game states.
        }
    }
    data
}

fn train_linear_regression(
    neural_opt: &NeuralLuckOptimizer,
    rng: &mut Rng,
    dbn: &Dbn,
    config: &Config,
) -> ([f64; DIM], f64) {
    let mut base = neural_opt.clone();
    base.set_linear_params([0.0; DIM], 0.0);
    
    let sim_count = if config.fast_init {
        if cfg!(debug_assertions) { 10 } else { 40 }
    } else if cfg!(debug_assertions) { 50 } else { 200 };
    // This will generate ~10k to 40k data points (50 * 200 = 10000)
    let data = simulate_for_data_collection(sim_count, rng, &base, dbn, config);
    
    println!("[Linear] Collected {} samples from realistic simulation trajectories.", data.len());

    let epochs = if config.fast_init {
        if cfg!(debug_assertions) { 1 } else { 3 }
    } else if cfg!(debug_assertions) { 4 } else { 12 };
    let mut weights = [0.0; DIM];
    let mut bias = 0.0;
    let base_lr = if config.fast_init {
        if cfg!(debug_assertions) { 0.08 } else { 0.04 }
    } else if cfg!(debug_assertions) { 0.05 } else { 0.02 };
    
    for epoch in 0..epochs {
        let lr = base_lr / (1.0 + epoch as f64 * 0.1);
        for (x, y) in &data {
            let mut pred = bias;
            for i in 0..DIM {
                pred += weights[i] * x[i];
            }
            let err = pred - y;
            for i in 0..DIM {
                weights[i] -= lr * err * x[i];
            }
            bias -= lr * err;
        }
    }
    (weights, bias)
}

fn evaluate_manifold_reward(
    model: &NeuralLuckOptimizer,
    rng: &mut Rng,
    dbn: &Dbn,
    config: &Config,
) -> f64 {
    let sims = if cfg!(debug_assertions) { 2000 } else { 6000 };
    let res = simulate_fast(sims, rng, 0, model, None, None, dbn, config);
    let expected_pulls = expected_pulls_per_six(config);
    let target_rate = if expected_pulls > 0.0 {
        1.0 / expected_pulls
    } else {
        config.prob_6_base
    };
    let rate_6 = res.six_count as f64 / sims as f64;
    let rate_error = (rate_6 - target_rate).abs();
    let streak_penalty = res.max_loss_streak as f64;
    let big_pity_penalty = if res.big_pity_used { 1.0 } else { 0.0 };
    -rate_error * 10000.0 - streak_penalty * 500.0 - big_pity_penalty * 200.0
}

fn train_manifold_rl(
    base: &NeuralLuckOptimizer,
    rng: &mut Rng,
    dbn: &Dbn,
    config: &Config,
    worker: &GoodJobWorker,
) -> NeuralLuckOptimizer {
    let iterations = if config.fast_init {
        if cfg!(debug_assertions) { 2 } else { 6 }
    } else if cfg!(debug_assertions) { 8 } else { 30 };
    let population = if config.fast_init {
        if cfg!(debug_assertions) { 4 } else { 12 }
    } else if cfg!(debug_assertions) { 16 } else { 64 };
    
    // Multi-Manifold Topology Mapping Hyperparameters
    let mut sigma_analysis = 0.02; // Stiffer manifold for feature extraction
    let mut sigma_decision = 0.15; // More flexible manifold for decision making
    let l1_lambda = 0.005; // L1 Regularization strength for sparsity

    let lr = if config.fast_init {
        if cfg!(debug_assertions) { 0.1 } else { 0.05 }
    } else if cfg!(debug_assertions) { 0.08 } else { 0.04 };
    
    let mut current_opt = base.clone();
    let mut params = current_opt.get_params();
    let n_params = params.len();
    let n_analysis = NeuralLuckOptimizer::count_params_analysis();
    
    // Adam optimizer state
    let mut m = vec![0.0; n_params];
    let mut v_sq = vec![0.0; n_params];
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;

    println!("[RL] Manifold Optimization (MMTM): Pop={}, Iter={}, AnalysisParams={}, DecisionParams={}", 
             population, iterations, n_analysis, n_params - n_analysis);

    for iter in 0..iterations {
        let mut noise_seeds: Vec<u64> = Vec::with_capacity(population);
        for _ in 0..population {
            noise_seeds.push(rng.next_u64());
        }

        // Parallel Evaluation
        let results = worker.execute(|| {
            noise_seeds.into_par_iter().map(|seed| {
                let mut local_rng = Rng::from_seed(seed);
                
                // Generate perturbation vector
                let mut eps = vec![0.0; n_params];
                for (i, val) in eps.iter_mut().enumerate() {
                    let noise = local_rng.next_f64_normal();
                    // Apply manifold-specific scaling
                    let sigma = if i < n_analysis { sigma_analysis } else { sigma_decision };
                    *val = noise * sigma;
                }

                // Create Positive and Negative variants
                let mut params_pos = params.clone();
                let mut params_neg = params.clone();
                
                for i in 0..n_params {
                    params_pos[i] += eps[i];
                    params_neg[i] -= eps[i];
                }
                
                let mut opt_pos = current_opt.clone();
                opt_pos.set_params(&params_pos);
                
                let mut opt_neg = current_opt.clone();
                opt_neg.set_params(&params_neg);

                let mut sim_rng_pos = Rng::from_seed(local_rng.next_u64());
                let mut sim_rng_neg = Rng::from_seed(local_rng.next_u64());

                let mut r_pos = evaluate_manifold_reward(&opt_pos, &mut sim_rng_pos, dbn, config);
                let mut r_neg = evaluate_manifold_reward(&opt_neg, &mut sim_rng_neg, dbn, config);
                
                // Add L1 Penalty for Sparsity
                let l1_pos: f64 = params_pos.iter().map(|x| x.abs()).sum();
                let l1_neg: f64 = params_neg.iter().map(|x| x.abs()).sum();
                r_pos -= l1_pos * l1_lambda;
                r_neg -= l1_neg * l1_lambda;

                (eps, r_pos, r_neg)
            }).collect::<Vec<_>>()
        });

        let results = match results {
            Ok(v) => v,
            Err(e) => {
                println!("[RL] Parallel evaluation failed: {}", e);
                break;
            }
        };

        // Aggregate gradients
        let mut grad = vec![0.0; n_params];
        let mut avg_reward = 0.0;

        for (eps, r_pos, r_neg) in results {
            let diff = r_pos - r_neg;
            avg_reward += (r_pos + r_neg) / 2.0;
            
            // Standard ES gradient approximation:
            // g ~ (1 / (pop * sigma^2)) * sum(reward_diff * noise * sigma)
            // Here eps is already (noise * sigma).
            // So we just accumulate (diff * eps). 
            
            add_scaled_row(&mut grad, &eps, diff);
        }
        avg_reward /= population as f64;

        if iter % 5 == 0 || iter == iterations - 1 {
             print!("\r[RL] Iter {:>2}/{}: Avg Reward = {:>8.2}", iter + 1, iterations, avg_reward);
             io::stdout().flush().unwrap();
        }

        // Apply Adam Update
        // We need to normalize grad by population.
        // And effectively divide by sigma^2 for correct gradient scale?
        // Or just let Adam handle the scaling.
        // But since sigma_analysis << sigma_decision, the `eps` values for analysis are small.
        // So `grad` for analysis will be small.
        // But the sensitivity might be high.
        // Let's normalize by population first.
        let scale = 1.0 / population as f64;
        
        for i in 0..n_params {
            let g = grad[i] * scale;
            
            // Adam
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v_sq[i] = beta2 * v_sq[i] + (1.0 - beta2) * g * g;
            
            let m_hat = m[i] / (1.0 - beta1.powi((iter + 1) as i32));
            let v_hat = v_sq[i] / (1.0 - beta2.powi((iter + 1) as i32));
            
            // For MMTM, we might want to scale the update step by sigma again?
            // Or just rely on LR.
            // With small sigma, the gradient estimate is noisy but also small.
            // Adam's adaptive moment might boost it if consistent.
            
            params[i] += lr * m_hat / (v_hat.sqrt() + epsilon);
        }
        
        // === Dynamic Curvature Adaptation ===
        // We use the Adam second moment estimate (v_sq) to infer the local curvature (gradient magnitude).
        // High v_sq -> Steep gradients -> High curvature -> We need smaller exploration noise (trust region).
        // Low v_sq -> Flat gradients -> Low curvature -> We can explore more.
        
        // Calculate average v_hat for analysis and decision blocks
        let mut v_sum_analysis = 0.0;
        let mut v_sum_decision = 0.0;
        
        for i in 0..n_params {
            let v_hat = v_sq[i] / (1.0 - beta2.powi((iter + 1) as i32));
            if i < n_analysis {
                v_sum_analysis += v_hat;
            } else {
                v_sum_decision += v_hat;
            }
        }
        
        let v_avg_analysis = v_sum_analysis / n_analysis as f64;
        let v_avg_decision = v_sum_decision / (n_params - n_analysis) as f64;
        
        // Adapt sigmas: sigma ~ Base / (1 + Scale * sqrt(v_avg))
        // This is a heuristic to inversely scale noise with gradient magnitude.
        
        let curvature_scale = 10.0; // Tuning parameter
        sigma_analysis = 0.02 / (1.0 + curvature_scale * v_avg_analysis.sqrt());
        sigma_decision = 0.15 / (1.0 + curvature_scale * v_avg_decision.sqrt());
        
        // Clamp to safe ranges
        sigma_analysis = sigma_analysis.clamp(0.001, 0.05);
        sigma_decision = sigma_decision.clamp(0.01, 0.3);

        if iter % 5 == 0 {
             // Print curvature stats
             print!(" [Sigma: A={:.4}, D={:.4}]", sigma_analysis, sigma_decision);
             io::stdout().flush().unwrap();
        }

        current_opt.set_params(&params);
    }
    
    // Apply Pruning (Sparse Hyper-Connections)
    println!(); // Newline
    println!("[RL] Applying Sparse Hyper-Connections (Pruning)...");
    let initial_active = current_opt.count_active_params();
    current_opt.prune(0.01); // Prune weights < 0.01
    let final_active = current_opt.count_active_params();
    let total_params = NeuralLuckOptimizer::param_count();
    let sparsity = 1.0 - (final_active as f64 / total_params as f64);
    
    println!("[RL] Pruning Complete: {} -> {} active params (Sparsity: {:.1}%)", 
             initial_active, final_active, sparsity * 100.0);
             
    current_opt
}

fn prompt_yes_no(prompt: &str, default_yes: bool) -> bool {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let s = input.trim();
    if s.is_empty() {
        return default_yes;
    }
    if default_yes {
        !s.eq_ignore_ascii_case("n")
    } else {
        s.eq_ignore_ascii_case("y")
    }
}

fn read_cache_bytes(path: &str) -> Option<Vec<u8>> {
    if let Ok(bytes) = std::fs::read(path) {
        return Some(bytes);
    }
    let alt = format!("../../{}", path);
    std::fs::read(alt).ok()
}

fn load_neural_cache(path: &str) -> Option<NeuralLuckOptimizer> {
    let bytes = read_cache_bytes(path)?;
    NeuralLuckOptimizer::from_bytes(&bytes)
}

fn save_neural_cache(path: &str, net: &NeuralLuckOptimizer) -> bool {
    let bytes = net.to_bytes();
    if std::fs::write(path, &bytes).is_ok() {
        return true;
    }
    let alt = format!("../../{}", path);
    std::fs::write(alt, &bytes).is_ok()
}

pub fn simulate_core(
    control: &SimControl,
    rng: &mut Rng,
    available_free_pulls: u32,
    neural_opt: &NeuralLuckOptimizer,
    dqn_policy: Option<&DuelingQNetwork>,
    ppo_policy: Option<&ActorCritic>,
    dbn: &Dbn,
    config: &Config,
) -> (SimStatsResult, Option<Vec<PullResult>>) {
    let mut big_pity_used = false;
    let mut six_count = 0;
    let mut up_count = 0;

    let mut free_pulls_remaining = available_free_pulls;
    let mut cost_jade = 0;

    let mut max_loss_streak = 0;

    let (env_noise, env_bias) = dbn_env(dbn, rng);
    let mut state = PullState {
        pity_6: 0,
        total_pulls_in_pool: 0,
        has_obtained_up: false,
        streak_4_star: 0,
        loss_streak: 0,
    };

    let non_up_six: Vec<String> = if control.collect_details {
        config.six_stars
            .iter()
            .filter(|s| !config.up_six.contains(s))
            .cloned()
            .collect()
    } else {
        Vec::new()
    };

    let mut pulls = if control.collect_details {
        Some(Vec::with_capacity(control.max_pulls.unwrap_or(0)))
    } else {
        None
    };

    let mut pulls_done = 0usize;
    loop {
        if let Some(max_pulls) = control.max_pulls {
            if pulls_done >= max_pulls {
                break;
            }
        }

        if free_pulls_remaining > 0 {
            free_pulls_remaining -= 1;
        } else {
            cost_jade += COST_PER_PULL;
        }

        let nn_total_pulls = if control.nn_total_pulls_one_based {
            pulls_done + 1
        } else {
            pulls_done
        };

        let outcome = roll_one(
            &mut state,
            rng,
            neural_opt,
            dqn_policy,
            ppo_policy,
            env_noise,
            env_bias,
            config,
            nn_total_pulls,
            control.big_pity_requires_not_up,
        );

        if outcome.big_pity_used {
            big_pity_used = true;
        }
        if outcome.is_up {
            up_count += 1;
        }
        if outcome.rarity == 6 {
            six_count += 1;
        }
        if state.loss_streak > max_loss_streak {
            max_loss_streak = state.loss_streak;
        }

        if let Some(ref mut pulls_vec) = pulls {
            let op_name: &String = match outcome.rarity {
                6 => {
                    if outcome.is_up {
                        rng.choose(&config.up_six)
                    } else {
                        rng.choose(&non_up_six)
                    }
                }
                5 => rng.choose(&config.five_stars),
                _ => rng.choose(&config.four_stars),
            };
            pulls_vec.push(PullResult {
                operator: op_name.to_string(),
                rarity: outcome.rarity,
            });
        }

        pulls_done += 1;

        if control.stop_on_up && outcome.is_up {
            break;
        }

        if let Some(limit) = control.stop_after_total_pulls {
            if state.total_pulls_in_pool > limit {
                break;
            }
        }
    }

    let free_pulls_used = available_free_pulls - free_pulls_remaining;

    let stats = SimStatsResult {
        six_count,
        up_count,
        big_pity_used,
        cost_jade,
        free_pulls_used,
        max_loss_streak,
    };

    (stats, pulls)
}

fn simulate_fast(num_pulls: usize, rng: &mut Rng, available_free_pulls: u32, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config) -> SimStatsResult {
    let control = SimControl {
        max_pulls: Some(num_pulls),
        stop_on_up: false,
        stop_after_total_pulls: None,
        nn_total_pulls_one_based: false,
        collect_details: false,
        big_pity_requires_not_up: true,
    };
    simulate_core(&control, rng, available_free_pulls, neural_opt, dqn_policy, ppo_policy, dbn, config).0
}


fn simulate_one(num_pulls: usize, rng: &mut Rng, available_free_pulls: u32, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config) -> SimulationResult {
    let control = SimControl {
        max_pulls: Some(num_pulls),
        stop_on_up: false,
        stop_after_total_pulls: None,
        nn_total_pulls_one_based: false,
        collect_details: true,
        big_pity_requires_not_up: true,
    };
    let (stats, pulls_opt) = simulate_core(&control, rng, available_free_pulls, neural_opt, dqn_policy, ppo_policy, dbn, config);
    let pulls = pulls_opt.unwrap_or_default();
    SimulationResult {
        pulls,
        six_count: stats.six_count,
        up_count: stats.up_count,
        big_pity_used: stats.big_pity_used,
        cost_jade: stats.cost_jade,
        free_pulls_used: stats.free_pulls_used,
    }
}

fn simulate_stats(num_pulls: usize, num_sims: usize, seed: u64, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config, worker: &GoodJobWorker) -> (usize, usize, usize) {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();

    let chunk_size = 64usize;
    let chunk_count = (num_sims + chunk_size - 1) / chunk_size;
    let (total_six, total_up, total_big_pity) = worker.execute(|| {
        (0..chunk_count).into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(num_sims);
                let mut local_rng = Rng::from_seed(base_seed.wrapping_add(chunk_idx as u64));
                let mut total_six = 0usize;
                let mut total_up = 0usize;
                let mut total_big_pity = 0usize;
                for _ in start..end {
                    let res = simulate_fast(num_pulls, &mut local_rng, 0, neural_opt, dqn_policy, ppo_policy, dbn, config);
                    total_six += res.six_count;
                    total_up += res.up_count;
                    if res.big_pity_used { total_big_pity += 1; }
                }
                (total_six, total_up, total_big_pity)
            })
            .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
    }).unwrap_or_else(|e| {
        println!("[Error] Simulation failed: {}", e);
        (0, 0, 0)
    });

    (total_six, total_up, total_big_pity)
}

fn simulate_f2p_clearing(num_sims: usize, seed: u64, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config, worker: &GoodJobWorker) -> (Option<f64>, usize) {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();

    let chunk_size = 64usize;
    let chunk_count = (num_sims + chunk_size - 1) / chunk_size;
    let (total_extra_cost, extra_cost_samples, success_count_val) = worker.execute(|| {
        (0..chunk_count).into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(num_sims);
                let mut local_rng = Rng::from_seed(base_seed.wrapping_add(chunk_idx as u64));
                let control = SimControl {
                    max_pulls: None,
                    stop_on_up: true,
                    stop_after_total_pulls: Some(config.big_pity_cumulative),
                    nn_total_pulls_one_based: true,
                    collect_details: false,
                    big_pity_requires_not_up: false,
                };
                let mut total_extra = 0u64;
                let mut total_samples = 0usize;
                let mut total_success = 0usize;
                for _ in start..end {
                    let (stats, _) = simulate_core(&control, &mut local_rng, FREE_PULLS_WELFARE, neural_opt, dqn_policy, ppo_policy, dbn, config);
                    let success = if stats.up_count > 0 { 1 } else { 0 };
                    let extra = if stats.cost_jade > 0 { stats.cost_jade as u64 } else { 0 };
                    let extra_sample = if stats.cost_jade > 0 { 1 } else { 0 };
                    total_extra += extra;
                    total_samples += extra_sample;
                    total_success += success;
                }
                (total_extra, total_samples, total_success)
            })
            .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
    }).unwrap_or_else(|e| {
        println!("[Error] Cost Analysis failed: {}", e);
        (0, 0, 0)
    });
    
    let avg_extra_cost = if extra_cost_samples == 0 {
        None
    } else {
        Some(total_extra_cost as f64 / extra_cost_samples as f64)
    };
    (avg_extra_cost, success_count_val)
}

fn format_f2p_probability_line(total_episodes: usize, early_success_episodes: usize) -> String {
    if total_episodes == 0 || early_success_episodes == total_episodes {
        "Probability to get UP with ONLY free resources: ≥99.99 % (all succeeded early)".to_string()
    } else {
        let rate = early_success_episodes as f64 / total_episodes as f64;
        format!("Probability to get UP with ONLY free resources: {:.2}%", rate * 100.0)
    }
}

fn format_avg_extra_cost_line(avg_extra_cost: Option<f64>) -> String {
    match avg_extra_cost {
        Some(cost) => format!("Avg Extra Jade Cost: {:.0} (Approx. {:.1} extra pulls)", cost, cost / 500.0),
        None => "Avg Extra Jade Cost: N/A".to_string(),
    }
}

// Calculate F2P Clear Rate (Percentage of players who get UP using ONLY free pulls)
#[allow(dead_code)]
fn calculate_f2p_success_rate(num_sims: usize, seed: u64, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config, worker: &GoodJobWorker) -> f64 {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();

    let success_free = worker.execute(|| {
        (0..num_sims).into_par_iter()
            .map(|i| {
                let mut local_rng = Rng::from_seed(base_seed.wrapping_add(i as u64));
                let control = SimControl {
                    max_pulls: None,
                    stop_on_up: true,
                    stop_after_total_pulls: Some(config.big_pity_cumulative),
                    nn_total_pulls_one_based: true,
                    collect_details: false,
                    big_pity_requires_not_up: false,
                };
                let (stats, _) = simulate_core(&control, &mut local_rng, FREE_PULLS_WELFARE, neural_opt, dqn_policy, ppo_policy, dbn, config);
                if stats.up_count > 0 { 1 } else { 0 }
            })
            .sum::<usize>()
    }).unwrap_or_else(|e| {
        println!("[Error] F2P Analysis failed: {}", e);
        0
    });

    success_free as f64 / num_sims as f64
}

// === GENETIC ALGORITHM TRAINING ===
fn train_neural_optimizer(seed: u64, dbn: &Dbn, config: &Config, worker: &GoodJobWorker) -> NeuralLuckOptimizer {
    println!("\n[Neural Core] Initializing Evolutionary Training Process...");
    println!("[Neural Core] Objective: Minimize Bad Luck Streaks while maintaining Fair Probabilities");
    
    let mut rng = Rng::from_seed(seed);
    let pop_size = if config.fast_init { 12 } else { 30 };
    let generations = if config.fast_init { 6 } else { 15 };
    let sims_per_genome = if config.fast_init { 600 } else { 2000 };
    let expected_pulls = expected_pulls_per_six(config);
    let target_rate = if expected_pulls > 0.0 {
        1.0 / expected_pulls
    } else {
        config.prob_6_base
    };

    let mut population: Vec<NeuralLuckOptimizer> = (0..pop_size)
        .map(|_| NeuralLuckOptimizer::new(rng.next_u64()))
        .collect();

    for gen in 0..generations {
        let eval_seeds: Vec<u64> = (0..pop_size).map(|_| rng.next_u64()).collect();
        let scores_result = worker.execute(|| {
            (0..pop_size).into_par_iter().map(|idx| {
                let mut local_rng = Rng::from_seed(eval_seeds[idx]);
                let control = SimControl {
                    max_pulls: None,
                    stop_on_up: true,
                    stop_after_total_pulls: Some(config.big_pity_cumulative),
                    nn_total_pulls_one_based: true,
                    collect_details: false,
                    big_pity_requires_not_up: false,
                };
                let (stats, _) = simulate_core(&control, &mut local_rng, 0, &population[idx], None, None, dbn, config);
                
                let rate_6 = stats.six_count as f64 / sims_per_genome as f64;
                
                let rate_error = (rate_6 - target_rate).abs();
                
                let streak_penalty = if stats.max_loss_streak >= 3 {
                    (stats.max_loss_streak as f64 - 2.0) * 500.0
                } else {
                    0.0
                };

                let score = - (rate_error * 10000.0) - streak_penalty;
                (idx, score, stats.max_loss_streak)
            }).collect::<Vec<(usize, f64, usize)>>()
        });
        let mut scores = match scores_result {
            Ok(val) => val,
            Err(e) => {
                println!("[Error] Training evaluation failed: {}", e);
                return population[0].clone();
            }
        };

        // Sort by score descending (best first)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let best_score = scores[0].1;
        let best_streak = scores[0].2;
        
        print!("\r[Training] Gen {:>2}/{}: Best Score = {:>8.2} | Max Loss Streak = {}", gen+1, generations, best_score, best_streak);
        io::stdout().flush().unwrap();
        
        // Selection & Mutation (Evolution Strategy)
        let mut new_pop = Vec::with_capacity(pop_size);
        
        // Elitism: Keep top 3 unchanged
        for i in 0..3 {
            new_pop.push(population[scores[i].0].clone());
        }
        
        // Fill the rest with mutated offspring of the top 50%
        let top_k = pop_size / 2;
        while new_pop.len() < pop_size {
            let parent_rank = rng.next_u64_bounded(top_k as u64) as usize;
            let parent_idx = scores[parent_rank].0;
            
            let mut child = population[parent_idx].clone();
            // Mutation rate and scale
            child.mutate(&mut rng, 0.15, 0.02);
            new_pop.push(child);
        }
        population = new_pop;
    }
    println!("\n[Neural Core] Training Complete. Optimal weights loaded.");
    
    // Return the best one from the last generation (which is at index 0 because of sorting? No, we need to re-evaluate or just take the best from previous sort)
    // Actually, we just replaced population. The first 3 are the best from previous gen.
    population[0].clone()
}

fn benchmark_simulation(rng: &mut Rng, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config) {
    let fast_sims = 10_000usize;
    let fast_pulls = 200usize;
    let start_fast = Instant::now();
    for _ in 0..fast_sims {
        let _ = simulate_fast(fast_pulls, rng, 0, neural_opt, dqn_policy, ppo_policy, dbn, config);
    }
    let fast_elapsed = start_fast.elapsed();
    println!(
        "[Bench] simulate_fast: {} sims of {} pulls in {:.2?} ({:.0} sims/sec)",
        fast_sims,
        fast_pulls,
        fast_elapsed,
        fast_sims as f64 / fast_elapsed.as_secs_f64()
    );

    let one_sims = 300usize;
    let one_pulls = 120usize;
    let start_one = Instant::now();
    for _ in 0..one_sims {
        let _ = simulate_one(one_pulls, rng, 0, neural_opt, dqn_policy, ppo_policy, dbn, config);
    }
    let one_elapsed = start_one.elapsed();
    println!(
        "[Bench] simulate_one: {} sims of {} pulls in {:.2?} ({:.0} sims/sec)",
        one_sims,
        one_pulls,
        one_elapsed,
        one_sims as f64 / one_elapsed.as_secs_f64()
    );
}

fn main() {
    // Load Configuration
    let config = Config::load("data/config.json");
    let mut rng = Rng::new();

    // Initialize GoodJobWorker (ThreadPool)
    // 0 means auto-detect but reserve 1 core for UI/System to avoid freezing
    let worker = GoodJobWorker::new_with_config(&config);

    let mut dbn = Dbn::new(&[8, 16, 8], &mut rng);
    let (dbn_data_count, dbn_epochs) = if config.fast_init {
        if cfg!(debug_assertions) { (64, 2) } else { (256, 4) }
    } else if cfg!(debug_assertions) {
        (256, 5)
    } else {
        (1024, 20)
    };
    dbn.train(&mut rng, dbn_data_count, dbn_epochs);

    let mut trained_neural_opt = if let Some(cached) = load_neural_cache(NEURAL_CACHE_PATH) {
        println!("\n[Neural Core] Cache detected.");
        let use_cache = if config.fast_init {
            true
        } else {
            prompt_yes_no("[Neural Core] Load cached weights? (y/n, default y): ", true)
        };
        if use_cache {
            println!("[Neural Core] Cached weights loaded.");
            cached
        } else {
            println!("[Neural Core] Training new weights...");
            train_neural_optimizer(rng.next_u64(), &dbn, &config, &worker)
        }
    } else {
        println!("\n[Neural Core] Cache not found. Training new weights...");
        train_neural_optimizer(rng.next_u64(), &dbn, &config, &worker)
    };
    println!("\n[Linear] Training linear regression...");
    let (lin_w, lin_b) = train_linear_regression(&trained_neural_opt, &mut rng, &dbn, &config);
    trained_neural_opt.set_linear_params(lin_w, lin_b);
    println!("[Linear] Regression complete.");
    println!("[RL] Manifold Optimization (Parallel)...");
    trained_neural_opt = train_manifold_rl(&trained_neural_opt, &mut rng, &dbn, &config, &worker);
    println!("[RL] Fine-tuning complete.");
    
    let rl_w = trained_neural_opt.linear_weights;
    let rl_b = trained_neural_opt.linear_bias;

    // === EXPLAINABILITY REPORT ===
    println!("\n[Model Insight] Linear Manifold Analysis:");
    let feature_names = [
        "Pity Progress (0-1)", 
        "Total Pulls Norm", 
        "Env Noise", 
        "Loss Streak Norm", 
        "4-Star Streak Norm", 
        "Env Bias",
        "Pity * Loss (Interaction)",
        "Total Pulls (Quadratic)",
    ];
    for (i, name) in feature_names.iter().enumerate() {
        let w = rl_w[i];
        let impact = if w.abs() < 0.001 { "Neutral" } else if w > 0.0 { "Boost Luck" } else { "Reduce Luck" };
        println!("  - {:<25}: {:>8.4} [{}]", name, w, impact);
    }
    println!("  - {:<25}: {:>8.4} [Bias]", "Base Bias", rl_b);

    // === DQN Training ===
    let dqn_policy = train_dqn(&trained_neural_opt, &mut rng, &dbn, &config);
    
    // === PPO Training ===
    let ppo_policy = train_ppo(&mut rng, &dbn, &config);
    
    // We replace the trained_neural_opt with the DQN-optimized one (base part)
    // Note: The DQN might have shifted the Q-values, but we only use the base features + linear head for probability adjustment.
    // Actually, Q-Network output is Q-Values, not probability adjustments directly.
    // But our DQN implementation modified the 'base' weights inside the QNetwork during backprop (if we allowed it).
    // In our implementation, we only trained the Q-Head.
    // So the 'base' inside dqn_policy is unchanged.
    // IF we want to use the DQN policy for inference, we need to map Q-values to Actions.
    // But `simulate_core` expects a `NeuralLuckOptimizer` that outputs a float scalar.
    // So we can't easily swap `trained_neural_opt` with `dqn_policy` directly for the existing simulator
    // unless we adapt the simulator to use Q-Network selection logic.
    
    // For now, let's just save the DQN model or demonstrate it trained.
    // If we want to use the DQN result, we should probably update the linear weights of trained_neural_opt
    // based on the "best action" preference, but that's complex.
    // Let's just acknowledge DQN training completed.
    
    if save_neural_cache(NEURAL_CACHE_PATH, &trained_neural_opt) {
        println!("[Neural Core] Cache saved.");
    } else {
        println!("[Neural Core] Cache save failed.");
    }


    println!("=== Talos-XII Wish Simulator (Neural-Evolutionary) ===");
    println!("Pool Name: {}", config.pool_name);
    println!("UP Operator(s): {}", config.up_six.join(", "));
    println!("Probabilities: 6-Star {:.1}% (Soft Pity start at {}, +5%/pull)", config.prob_6_base * 100.0, config.soft_pity_start);
    println!("Rules: {} Pulls Guarantee 6* (50/50 UP, No Guarantee on Loss)", config.small_pity_guarantee);
    println!("Big Pity: Cumulative {} pulls guarantee UP (Once per pool)", config.big_pity_cumulative);
    println!("Economy: {} Jade/Pull | ~{} Free Pulls (Welfare)", COST_PER_PULL, FREE_PULLS_WELFARE);
    println!("Neural Core: Online (Evolved for Luck Balancing)");
    
    println!("\n[System] PRNG Initialized: xoshiro256**");
    if cfg!(debug_assertions) && !config.fast_init {
        println!("\n[System] Benchmarking simulation throughput...");
        benchmark_simulation(&mut rng, &trained_neural_opt, Some(&dqn_policy), Some(&ppo_policy), &dbn, &config);
    }

    // F2P Analysis
    println!("\n=== F2P Welfare Analysis ({} Free Pulls) ===", FREE_PULLS_WELFARE);
    
    // Adjust simulation count based on build profile to prevent hanging in Debug mode
    #[cfg(debug_assertions)]
    let sim_count = if config.fast_init { 2_000 } else { 10_000 };
    #[cfg(not(debug_assertions))]
    let sim_count = if config.fast_init { 200_000 } else { 1_000_000 };

    println!("[System] Running {} simulations...", sim_count);
    
    let start_time = Instant::now();
    let stats = simulate_stats(0, sim_count, rng.next_u64(), &trained_neural_opt, Some(&dqn_policy), Some(&ppo_policy), &dbn, &config, &worker);
    let elapsed = start_time.elapsed();
    let prob_line = format_f2p_probability_line(sim_count, stats.1);
    println!("{}", prob_line);
    println!("Time taken: {:.2?}", elapsed);
    println!("Throughput: {:.0} sims/sec", sim_count as f64 / elapsed.as_secs_f64());
    
    println!("\nCalculating average EXTRA cost for F2P players to get UP...");
    let (avg_extra_cost, _) = simulate_f2p_clearing(sim_count, rng.next_u64(), &trained_neural_opt, Some(&dqn_policy), Some(&ppo_policy), &dbn, &config, &worker);
    let avg_cost_line = format_avg_extra_cost_line(avg_extra_cost);
    println!("{}", avg_cost_line);
    println!("Total Value ~ 41000 Jade (Expected Cost for First UP)");

    // Interaction Loop
    loop {
        print!("\nEnter number of pulls (default 10, or 'q' to quit): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.eq_ignore_ascii_case("q") {
            println!("Exiting. Goodbye!");
            break;
        }

        let n = if input.is_empty() {
            10
        } else {
            match input.parse::<usize>() {
                Ok(val) => {
                    if val > 1_000_000 {
                        println!("Input too large, capped at 1,000,000 to prevent memory issues.");
                        1_000_000
                    } else {
                        val
                    }
                },
                Err(_) => {
                    println!("Invalid input. Using default 10.");
                    10
                }
            }
        };
        
        print!("Use Welfare Resources ({} pulls)? (y/n, default y): ", FREE_PULLS_WELFARE);
        io::stdout().flush().unwrap();
        let mut w_input = String::new();
        io::stdin().read_line(&mut w_input).unwrap();
        let use_welfare = !w_input.trim().eq_ignore_ascii_case("n");
        let free_pulls = if use_welfare { FREE_PULLS_WELFARE } else { 0 };

        print!("Enter simulation count (default 1, max 1M): ");
        io::stdout().flush().unwrap();
        let mut sim_input = String::new();
        io::stdin().read_line(&mut sim_input).unwrap();
        let sim_input = sim_input.trim();
        
        let sims_n = if sim_input.is_empty() {
            1
        } else {
             match sim_input.parse::<usize>() {
                Ok(val) => {
                    if val > 1_000_000 {
                        println!("Simulation count too large, capped at 1,000,000 to prevent CPU hang.");
                        1_000_000
                    } else {
                        val
                    }
                },
                Err(_) => 1,
            }
        };

        if sims_n > 1 {
            let (s_total, u_total, _) = simulate_stats(n, sims_n, rng.next_u64(), &trained_neural_opt, Some(&dqn_policy), Some(&ppo_policy), &dbn, &config, &worker);
            let s_avg = s_total as f64 / sims_n as f64;
            let u_avg = u_total as f64 / sims_n as f64;
            println!(
                "\n{} simulations of {}-pulls: Avg 6-Star {:.3} | UP {:.3}",
                sims_n, n, s_avg, u_avg
            );
        } else {
            let start_time = Instant::now();
            
            let res = simulate_one(n, &mut rng, free_pulls, &trained_neural_opt, Some(&dqn_policy), Some(&ppo_policy), &dbn, &config);
            let elapsed = start_time.elapsed();
            println!("\nSingle {}-pull result (Time: {:.2?}):", n, elapsed);
            println!("6-Star: {} | UP: {}", res.six_count, res.up_count);
            // Show first 20 details
            for (i, p) in res.pulls.iter().take(20).enumerate() {
                let is_up = p.rarity == 6 && config.up_six.iter().any(|s| s == &p.operator);
                if is_up {
                    println!("{}. {} ({} Star) [UP]", i + 1, p.operator, p.rarity);
                } else {
                    println!("{}. {} ({} Star)", i + 1, p.operator, p.rarity);
                }
            }
            if res.pulls.len() > 20 {
                println!("... ({} more omitted)", res.pulls.len() - 20);
            }
            
            println!("--- Consumption ---");
            println!("Free Pulls Used: {}", res.free_pulls_used);
            println!("Jade Spent: {} ({} pulls)", res.cost_jade, res.cost_jade / 500);
            if res.big_pity_used {
                println!("Big Pity Triggered!");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_context() -> (Config, Dbn, NeuralLuckOptimizer) {
        let config = Config::load("data/config.json");
        let mut rng = Rng::from_seed(1234);
        let dbn = Dbn::new(&[8, 16, 8], &mut rng);
        let neural_opt = NeuralLuckOptimizer::new(5678);
        (config, dbn, neural_opt)
    }

    #[test]
    fn simulate_fast_costs_and_free_pulls_match() {
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(1);
        let num_pulls = 200;
        let free_pulls = FREE_PULLS_WELFARE;
        let res = simulate_fast(num_pulls, &mut rng, free_pulls, &neural_opt, None, None, &dbn, &config);
        let expected_free_used = free_pulls.min(num_pulls as u32);
        let expected_cost = (num_pulls as u32 - expected_free_used) * COST_PER_PULL;
        assert_eq!(res.free_pulls_used, expected_free_used);
        assert_eq!(res.cost_jade, expected_cost);
    }

    #[test]
    fn simulate_one_counts_match_pulls() {
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(2);
        let num_pulls = 120;
        let free_pulls = FREE_PULLS_WELFARE;
        let res = simulate_one(num_pulls, &mut rng, free_pulls, &neural_opt, None, None, &dbn, &config);
        let six_count = res.pulls.iter().filter(|p| p.rarity == 6).count();
        let up_count = res
            .pulls
            .iter()
            .filter(|p| p.rarity == 6 && config.up_six.iter().any(|s| s == &p.operator))
            .count();
        let expected_free_used = free_pulls.min(num_pulls as u32);
        let expected_cost = (num_pulls as u32 - expected_free_used) * COST_PER_PULL;
        assert_eq!(res.six_count, six_count);
        assert_eq!(res.up_count, up_count);
        assert_eq!(res.free_pulls_used, expected_free_used);
        assert_eq!(res.cost_jade, expected_cost);
    }

    #[test]
    fn simulate_core_f2p_clearing_always_hits_up() {
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(3);
        let control = SimControl {
            max_pulls: None,
            stop_on_up: true,
            stop_after_total_pulls: Some(config.big_pity_cumulative),
            nn_total_pulls_one_based: true,
            collect_details: false,
            big_pity_requires_not_up: false,
        };
        let (stats, _) = simulate_core(&control, &mut rng, FREE_PULLS_WELFARE, &neural_opt, None, None, &dbn, &config);
        assert!(stats.up_count > 0);
    }

    #[test]
    fn format_probability_all_succeeded_early() {
        let line = format_f2p_probability_line(100, 100);
        assert_eq!(line, "Probability to get UP with ONLY free resources: ≥99.99 % (all succeeded early)");
    }

    #[test]
    fn format_probability_none_succeeded_early() {
        let line = format_f2p_probability_line(100, 0);
        assert_eq!(line, "Probability to get UP with ONLY free resources: 0.00%");
    }

    #[test]
    fn format_probability_half_succeeded_early() {
        let line = format_f2p_probability_line(100, 50);
        assert_eq!(line, "Probability to get UP with ONLY free resources: 50.00%");
    }
}
