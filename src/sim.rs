use crate::config::Config;
use crate::dbn::Dbn;
use crate::neural::{NeuralLuckOptimizer, Tensor, DIM};
use crate::rng::Rng;
use crate::dqn::DuelingQNetwork;
use crate::ppo::{self, ActorCritic};
use crate::autograd::Tensor as AutoTensor;
use crate::worker::GoodJobWorker;
use rayon::prelude::*;

// Constants
pub const DBN_GIBBS_STEPS: usize = 10;
pub const COST_PER_PULL: u32 = 500; // Jade equivalent
pub const FREE_PULLS_WELFARE: u32 = 135;

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

pub fn expected_pulls_per_six(config: &Config) -> f64 {
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

pub fn build_features(pity_6: usize, total_pulls: usize, env_noise: f64, streak: usize, env_bias: f64, loss_streak: usize, config: &Config) -> Tensor {
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

pub fn simulate_fast(num_pulls: usize, rng: &mut Rng, available_free_pulls: u32, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config) -> SimStatsResult {
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


pub fn simulate_one(num_pulls: usize, rng: &mut Rng, available_free_pulls: u32, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config) -> SimulationResult {
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

pub fn simulate_stats(num_pulls: usize, num_sims: usize, seed: u64, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config, worker: &GoodJobWorker) -> (usize, usize, usize, usize) {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();

    let chunk_size = 64usize;
    let chunk_count = (num_sims + chunk_size - 1) / chunk_size;
    let (total_six, total_up, total_big_pity, total_with_up) = worker.execute(|| {
        (0..chunk_count).into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(num_sims);
                let mut local_rng = Rng::from_seed(base_seed.wrapping_add(chunk_idx as u64));
                let mut total_six = 0usize;
                let mut total_up = 0usize;
                let mut total_big_pity = 0usize;
                let mut total_with_up = 0usize;
                for _ in start..end {
                    let res = simulate_fast(num_pulls, &mut local_rng, 0, neural_opt, dqn_policy, ppo_policy, dbn, config);
                    total_six += res.six_count;
                    total_up += res.up_count;
                    if res.big_pity_used { total_big_pity += 1; }
                    if res.up_count > 0 { total_with_up += 1; }
                }
                (total_six, total_up, total_big_pity, total_with_up)
            })
            .reduce(|| (0, 0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3))
    }).unwrap_or_else(|e| {
        println!("[Error] Simulation failed: {}", e);
        (0, 0, 0, 0)
    });

    (total_six, total_up, total_big_pity, total_with_up)
}

pub fn simulate_f2p_clearing(num_sims: usize, seed: u64, neural_opt: &NeuralLuckOptimizer, dqn_policy: Option<&DuelingQNetwork>, ppo_policy: Option<&ActorCritic>, dbn: &Dbn, config: &Config, worker: &GoodJobWorker) -> (Option<f64>, usize) {
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
                    // Fix: Ensure we use all available free pulls, covering the big pity if enough
                    stop_after_total_pulls: Some(FREE_PULLS_WELFARE.max(config.big_pity_cumulative as u32) as usize),
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

pub fn simulate_for_data_collection(
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

pub fn format_f2p_probability_line(total_episodes: usize, early_success_episodes: usize) -> String {
    if total_episodes == 0 || early_success_episodes == total_episodes {
        "Probability to get UP with ONLY free resources: ≥99.99 % (all succeeded early)".to_string()
    } else {
        let rate = early_success_episodes as f64 / total_episodes as f64;
        format!("Probability to get UP with ONLY free resources: {:.2}%", rate * 100.0)
    }
}

pub fn format_avg_extra_cost_line(avg_extra_cost: Option<f64>) -> String {
    match avg_extra_cost {
        Some(cost) => format!("Avg Extra Jade Cost: {:.0} (Approx. {:.1} extra pulls)", cost, cost / 500.0),
        None => "Avg Extra Jade Cost: N/A".to_string(),
    }
}
