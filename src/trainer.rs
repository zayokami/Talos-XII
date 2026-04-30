use crate::config::Config;
use crate::env_net::EnvNet;
use crate::neural::{NeuralLuckOptimizer, DIM};
use crate::rng::Rng;
use crate::sim::{
    expected_pulls_per_six, simulate_fast, simulate_for_data_collection, NeuralSample,
    SimModelContext, FREE_PULLS_WELFARE,
};
use crate::simd::add_scaled_row;
use crate::utils::create_bar;
use crate::worker::GoodJobWorker;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::RwLock;

const MANIFOLD_SIGMA_ANALYSIS_INIT: f64 = 0.02;
const MANIFOLD_SIGMA_DECISION_INIT: f64 = 0.15;
const MANIFOLD_L1_LAMBDA: f64 = 0.005;
const MANIFOLD_CURVATURE_SCALE: f64 = 10.0;
const MANIFOLD_SIGMA_ANALYSIS_RANGE: (f64, f64) = (0.001, 0.05);
const MANIFOLD_SIGMA_DECISION_RANGE: (f64, f64) = (0.01, 0.3);
const MANIFOLD_PRUNE_THRESHOLD: f64 = 0.01;

const EVAL_RATE_ERROR_WEIGHT: f64 = 10000.0;
const EVAL_STREAK_PENALTY_WEIGHT: f64 = 500.0;
const EVAL_BIG_PITY_PENALTY: f64 = 200.0;

/// Train linear regression on neural optimizer outputs for initialization.
pub fn train_linear_regression(
    neural_opt: &NeuralLuckOptimizer,
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
) -> ([f64; DIM], f64) {
    let mut base = neural_opt.clone();
    base.set_linear_params([0.0; DIM], 0.0);

    let sim_count = if config.fast_init {
        if cfg!(debug_assertions) {
            10
        } else {
            40
        }
    } else if cfg!(debug_assertions) {
        50
    } else {
        200
    };
    // This will generate ~10k to 40k data points (50 * 200 = 10000)
    let data = simulate_for_data_collection(sim_count, rng, &base, env_net, config);

    println!(
        "[Linear] Collected {} samples from realistic simulation trajectories.",
        data.len()
    );

    let epochs = if config.fast_init {
        if cfg!(debug_assertions) {
            1
        } else {
            3
        }
    } else if cfg!(debug_assertions) {
        4
    } else {
        12
    };
    let mut weights = [0.0; DIM];
    let mut bias = 0.0;
    let base_lr = if config.fast_init {
        if cfg!(debug_assertions) {
            0.08
        } else {
            0.04
        }
    } else if cfg!(debug_assertions) {
        0.05
    } else {
        0.02
    };

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
    env_net: &EnvNet,
    config: &Config,
) -> f64 {
    let sims = if cfg!(debug_assertions) { 2000 } else { 6000 };
    let ctx = SimModelContext {
        neural_opt: model,
        dqn_policy: None,
        ppo_policy: None,
        env_net,
        config,
        exp_sender: None,
        neural_sender: None,
        ppo_sender: None,
    };
    let res = simulate_fast(sims, rng, 0, &ctx);
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
    -rate_error * EVAL_RATE_ERROR_WEIGHT
        - streak_penalty * EVAL_STREAK_PENALTY_WEIGHT
        - big_pity_penalty * EVAL_BIG_PITY_PENALTY
}

/// Optimize neural weights via manifold-based evolution strategy with Adam.
pub fn train_manifold_rl(
    base: &NeuralLuckOptimizer,
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    worker: &GoodJobWorker,
) -> NeuralLuckOptimizer {
    let iterations = if config.fast_init {
        if cfg!(debug_assertions) {
            2
        } else {
            6
        }
    } else if cfg!(debug_assertions) {
        8
    } else {
        30
    };
    let population = if config.fast_init {
        if cfg!(debug_assertions) {
            4
        } else {
            12
        }
    } else if cfg!(debug_assertions) {
        16
    } else {
        64
    };

    // Multi-Manifold Topology Mapping Hyperparameters
    let mut sigma_analysis = MANIFOLD_SIGMA_ANALYSIS_INIT; // Stiffer manifold for feature extraction
    let mut sigma_decision = MANIFOLD_SIGMA_DECISION_INIT; // More flexible manifold for decision making
    let l1_lambda = MANIFOLD_L1_LAMBDA; // L1 Regularization strength for sparsity

    let lr = if config.fast_init {
        if cfg!(debug_assertions) {
            0.1
        } else {
            0.05
        }
    } else if cfg!(debug_assertions) {
        0.08
    } else {
        0.04
    };

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

    println!(
        "[RL] Manifold Optimization (MMTM): Pop={}, Iter={}, AnalysisParams={}, DecisionParams={}",
        population,
        iterations,
        n_analysis,
        n_params - n_analysis
    );

    let pb = create_bar(iterations as u64, "Manifold RL");

    for iter in 0..iterations {
        let mut noise_seeds: Vec<u64> = Vec::with_capacity(population);
        for _ in 0..population {
            noise_seeds.push(rng.next_u64());
        }

        // Parallel Evaluation
        let results = worker.execute(|| {
            noise_seeds
                .into_par_iter()
                .map(|seed| {
                    let mut local_rng = Rng::from_seed(seed);

                    // Generate perturbation vector
                    let mut eps = vec![0.0; n_params];
                    for (i, val) in eps.iter_mut().enumerate() {
                        let noise = local_rng.next_f64_normal();
                        // Apply manifold-specific scaling
                        let sigma = if i < n_analysis {
                            sigma_analysis
                        } else {
                            sigma_decision
                        };
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

                    let mut r_pos =
                        evaluate_manifold_reward(&opt_pos, &mut sim_rng_pos, env_net, config);
                    let mut r_neg =
                        evaluate_manifold_reward(&opt_neg, &mut sim_rng_neg, env_net, config);

                    // Add L1 Penalty for Sparsity
                    let l1_pos: f64 = params_pos.iter().map(|x| x.abs()).sum();
                    let l1_neg: f64 = params_neg.iter().map(|x| x.abs()).sum();
                    r_pos -= l1_pos * l1_lambda;
                    r_neg -= l1_neg * l1_lambda;

                    (eps, r_pos, r_neg)
                })
                .collect::<Vec<_>>()
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

        pb.set_position((iter + 1) as u64);
        pb.set_message(format!("Avg Reward: {:.2}", avg_reward));

        let scale = 1.0 / population as f64;

        for i in 0..n_params {
            let g = grad[i] * scale;

            // Adam
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v_sq[i] = beta2 * v_sq[i] + (1.0 - beta2) * g * g;

            let m_hat = m[i] / (1.0 - beta1.powi(iter + 1));
            let v_hat = v_sq[i] / (1.0 - beta2.powi(iter + 1));

            params[i] += lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        // === Dynamic Curvature Adaptation ===
        // We use the Adam second moment estimate (v_sq) to infer the local curvature (gradient magnitude).
        // High v_sq -> Steep gradients -> High curvature -> We need smaller exploration noise (trust region).
        // Low v_sq -> Flat gradients -> Low curvature -> We can explore more.

        // Calculate average v_hat for analysis and decision blocks
        let mut v_sum_analysis = 0.0;
        let mut v_sum_decision = 0.0;

        for (i, v_val) in v_sq.iter().enumerate() {
            let v_hat = v_val / (1.0 - beta2.powi(iter + 1));
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

        sigma_analysis =
            MANIFOLD_SIGMA_ANALYSIS_INIT / (1.0 + MANIFOLD_CURVATURE_SCALE * v_avg_analysis.sqrt());
        sigma_decision =
            MANIFOLD_SIGMA_DECISION_INIT / (1.0 + MANIFOLD_CURVATURE_SCALE * v_avg_decision.sqrt());

        // Clamp to safe ranges
        sigma_analysis = sigma_analysis.clamp(
            MANIFOLD_SIGMA_ANALYSIS_RANGE.0,
            MANIFOLD_SIGMA_ANALYSIS_RANGE.1,
        );
        sigma_decision = sigma_decision.clamp(
            MANIFOLD_SIGMA_DECISION_RANGE.0,
            MANIFOLD_SIGMA_DECISION_RANGE.1,
        );

        current_opt.set_params(&params);
    }

    pb.finish_with_message("Manifold RL Complete.");
    println!("[RL] Applying Sparse Hyper-Connections (Pruning)...");
    let initial_active = current_opt.count_active_params();
    current_opt.prune(MANIFOLD_PRUNE_THRESHOLD); // Prune weights < threshold
    let final_active = current_opt.count_active_params();
    let total_params = NeuralLuckOptimizer::param_count();
    let sparsity = 1.0 - (final_active as f64 / total_params as f64);

    println!(
        "[RL] Pruning Complete: {} -> {} active params (Sparsity: {:.1}%)",
        initial_active,
        final_active,
        sparsity * 100.0
    );

    current_opt
}

// === GENETIC ALGORITHM TRAINING ===

/// Train neural luck optimizer using genetic algorithm.
pub fn train_neural_optimizer(
    seed: u64,
    env_net: &EnvNet,
    config: &Config,
    worker: &GoodJobWorker,
) -> NeuralLuckOptimizer {
    println!("\n[Neural Core] Initializing Evolutionary Training Process...");
    println!(
        "[Neural Core] Objective: Minimize Bad Luck Streaks while maintaining Fair Probabilities"
    );

    let mut rng = Rng::from_seed(seed);
    let pop_size = if config.fast_init { 12 } else { 30 };
    let generations = if config.fast_init { 6 } else { 15 };
    let _sims_per_genome = if config.fast_init { 600 } else { 2000 };
    let expected_pulls = expected_pulls_per_six(config);
    let target_rate = if expected_pulls > 0.0 {
        1.0 / expected_pulls
    } else {
        config.prob_6_base
    };

    let mut population: Vec<NeuralLuckOptimizer> = (0..pop_size)
        .map(|_| NeuralLuckOptimizer::new(rng.next_u64()))
        .collect();

    let mut best_ever = population[0].clone();
    let mut best_ever_score = f64::NEG_INFINITY;

    let pb = create_bar(generations as u64, "Neural Training");

    for gen in 0..generations {
        let eval_seeds: Vec<u64> = (0..pop_size).map(|_| rng.next_u64()).collect();
        let scores_result = worker.execute(|| {
            (0..pop_size)
                .into_par_iter()
                .map(|idx| {
                    // Use crate::sim::SimControl instead of local struct
                    use crate::sim::{simulate_core, SimControl, SimModelContext};

                    let mut local_rng = Rng::from_seed(eval_seeds[idx]);
                    let control = SimControl {
                        max_pulls: None,
                        stop_on_up: true,
                        stop_after_total_pulls: Some(
                            (FREE_PULLS_WELFARE as usize).max(config.big_pity_cumulative),
                        ),
                        nn_total_pulls_one_based: true,
                        collect_details: false,
                        big_pity_requires_not_up: false,
                        fast_inference: true,
                    };
                    let model_ctx = SimModelContext {
                        neural_opt: &population[idx],
                        dqn_policy: None,
                        ppo_policy: None,
                        env_net,
                        config,
                        exp_sender: None,
                        neural_sender: None,
                        ppo_sender: None,
                    };
                    let (stats, _) = simulate_core(&control, &mut local_rng, 0, &model_ctx);

                    let actual_pulls = (stats.cost_jade / crate::sim::COST_PER_PULL) as usize;
                    let rate_6 = if actual_pulls > 0 {
                        stats.six_count as f64 / actual_pulls as f64
                    } else {
                        0.0
                    };

                    let rate_error = (rate_6 - target_rate).abs();

                    let streak_penalty = if stats.max_loss_streak >= 3 {
                        (stats.max_loss_streak as f64 - 2.0) * 500.0
                    } else {
                        0.0
                    };

                    let score = -(rate_error * 10000.0) - streak_penalty;
                    (idx, score, stats.max_loss_streak)
                })
                .collect::<Vec<(usize, f64, usize)>>()
        });
        let mut scores = match scores_result {
            Ok(val) => val,
            Err(e) => {
                println!("[Error] Training evaluation failed: {}", e);
                return best_ever;
            }
        };

        // Sort by score descending (best first)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let best_score = scores[0].1;
        let best_streak = scores[0].2;

        if best_score > best_ever_score {
            best_ever_score = best_score;
            best_ever = population[scores[0].0].clone();
        }

        pb.set_position((gen + 1) as u64);
        pb.set_message(format!("Best: {:.2} | Streak: {}", best_score, best_streak));

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
    pb.finish_with_message("Neural Training Complete.");

    best_ever
}

/// Incremental neural optimizer trainer for online learning.
pub struct OnlineNeuralTrainer {
    model: NeuralLuckOptimizer,
    ema_linear: [f64; DIM],
    ema_bias: f64,
    lr: f64,
    ema_decay: f64,
    steps_done: usize,
}

impl OnlineNeuralTrainer {
    pub fn from_model(model: NeuralLuckOptimizer) -> Self {
        Self {
            ema_linear: model.linear_weights,
            ema_bias: model.linear_bias,
            model,
            lr: 0.0005,
            ema_decay: 0.98,
            steps_done: 0,
        }
    }

    pub fn train_step(&mut self, sample: &NeuralSample) -> bool {
        let mut linear_sum = self.model.linear_bias;
        for i in 0..DIM {
            linear_sum += self.model.linear_weights[i] * sample.state[i];
        }
        if linear_sum.abs() >= 1.0 {
            return false;
        }
        let res_out = self.model.res_block.forward(&sample.state);
        let activation = res_out[0].clamp(-1.0, 1.0) * 0.015;
        let pred = activation + linear_sum * 0.01;
        let target = (sample.reward / 10.0).clamp(-1.0, 1.0) * 0.02;
        let error = pred - target;
        let grad = 2.0 * error * 0.01;
        for i in 0..DIM {
            self.model.linear_weights[i] -= self.lr * grad * sample.state[i];
        }
        self.model.linear_bias -= self.lr * grad;
        for i in 0..DIM {
            self.ema_linear[i] = self.ema_linear[i] * self.ema_decay
                + self.model.linear_weights[i] * (1.0 - self.ema_decay);
        }
        self.ema_bias =
            self.ema_bias * self.ema_decay + self.model.linear_bias * (1.0 - self.ema_decay);
        self.steps_done += 1;
        true
    }

    pub fn sync_to(&self, shared: &RwLock<NeuralLuckOptimizer>) {
        for attempt in 0..3u64 {
            if let Ok(mut guard) = shared.try_write() {
                guard.res_block = self.model.res_block.clone();
                guard.linear_weights = self.ema_linear;
                guard.linear_bias = self.ema_bias;
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1 + attempt));
        }
        if let Ok(mut guard) = shared.write() {
            guard.res_block = self.model.res_block.clone();
            guard.linear_weights = self.ema_linear;
            guard.linear_bias = self.ema_bias;
        }
    }

    pub fn steps_done(&self) -> usize {
        self.steps_done
    }
}
