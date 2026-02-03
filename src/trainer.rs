use crate::neural::{NeuralLuckOptimizer, DIM};
use crate::rng::Rng;
use crate::dbn::Dbn;
use crate::config::Config;
use crate::sim::{simulate_for_data_collection, simulate_fast, expected_pulls_per_six};
use crate::worker::GoodJobWorker;
use rayon::prelude::*;
use crate::simd::add_scaled_row;
use std::io::{self, Write};

pub fn train_linear_regression(
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

pub fn train_manifold_rl(
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

        
        let scale = 1.0 / population as f64;
        
        for i in 0..n_params {
            let g = grad[i] * scale;
            
            // Adam
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v_sq[i] = beta2 * v_sq[i] + (1.0 - beta2) * g * g;
            
            let m_hat = m[i] / (1.0 - beta1.powi((iter + 1) as i32));
            let v_hat = v_sq[i] / (1.0 - beta2.powi((iter + 1) as i32));
            
            
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

// === GENETIC ALGORITHM TRAINING ===
pub fn train_neural_optimizer(seed: u64, dbn: &Dbn, config: &Config, worker: &GoodJobWorker) -> NeuralLuckOptimizer {
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
                // Use crate::sim::SimControl instead of local struct
                use crate::sim::{SimControl, simulate_core};
                
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
    
    population[0].clone()
}
