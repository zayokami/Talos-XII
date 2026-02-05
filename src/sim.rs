use crate::autograd::Tensor as AutoTensor;
use crate::config::Config;
use crate::dbn::Dbn;
use crate::dqn::{DuelingQNetwork, Experience};
use crate::neural::{NeuralLuckOptimizer, Tensor, DIM};
use crate::ppo::{self, ActorCritic};
use crate::rng::Rng;
use crate::transformer::KVCache;
use crate::worker::GoodJobWorker;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::mpsc::Sender;

// Constants
pub const DBN_GIBBS_STEPS: usize = 10;
pub const COST_PER_PULL: u32 = 500; // Jade equivalent
pub const FREE_PULLS_WELFARE: u32 = 135;

#[derive(Clone, Debug)]
pub struct PullResult {
    pub rarity: u8,
    pub operator_idx: usize,
    pub is_up: bool,
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
    pub action: Option<usize>,
    pub ppo_log_prob: Option<f64>,
    pub ppo_value: Option<f64>,
}

#[derive(Clone)]
pub struct NeuralSample {
    pub state: Tensor,
    pub reward: f64,
}

#[derive(Clone)]
pub struct PpoExperience {
    pub state: Vec<f64>,
    pub seq_len: usize,
    pub pity: Vec<usize>,
    pub action: usize,
    pub log_prob: f64,
    pub reward: f64,
    pub done: bool,
    pub value: f64,
}

#[derive(Clone)]
pub struct SimControl {
    pub max_pulls: Option<usize>,
    pub stop_on_up: bool,
    pub stop_after_total_pulls: Option<usize>,
    pub nn_total_pulls_one_based: bool,
    pub collect_details: bool,
    pub big_pity_requires_not_up: bool,
    pub fast_inference: bool,
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

pub fn build_features(
    pity_6: usize,
    total_pulls: usize,
    env_noise: f64,
    streak: usize,
    env_bias: f64,
    loss_streak: usize,
    config: &Config,
) -> Tensor {
    let pity_norm = pity_6 as f64 / config.small_pity_guarantee as f64;
    let loss_norm = loss_streak as f64 / 3.0;
    // Use big_pity_cumulative for normalization if possible, or fallback to 100
    let total_norm_base = if config.big_pity_cumulative > 0 {
        config.big_pity_cumulative as f64
    } else {
        120.0
    };
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
    ppo_state_seq: Option<&AutoTensor>,
    ppo_pity_seq: Option<&[usize]>,
    fast_inference: bool,
    ppo_seq_data: Option<&[f64]>,
    kv_cache: &mut Option<KVCache>,
    start_pos: usize,
) -> PullOutcome {
    state.pity_6 += 1;
    state.total_pulls_in_pool += 1;

    let mut big_pity_used = false;
    let mut is_up = false;
    let mut action_used = None;
    let mut ppo_log_prob = None;
    let mut ppo_value = None;
    let rarity: u8;

    let big_pity_gate = if big_pity_requires_not_up {
        !state.has_obtained_up
    } else {
        true
    };
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
            config,
        );

        let luck_factor = if config.luck_mode == "dqn" {
            if let Some(policy) = dqn_policy {
                // Use DQN policy
                let tensor_x = AutoTensor::new(x.to_vec(), vec![DIM]);
                let (idx, modifier) = policy.predict_action(&tensor_x);
                action_used = Some(idx);
                modifier
            } else {
                // Fallback to probability if policy missing
                let dropout_seed = (state.pity_6 as u64)
                    .wrapping_add((nn_total_pulls as u64).wrapping_mul(31))
                    .wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
                neural_opt.predict(&x, dropout_seed)
            }
        } else if config.luck_mode == "ppo" {
            if let Some(policy) = ppo_policy {
                // Use PPO policy
                if fast_inference {
                    if let Some(cache) = kv_cache {
                        let idx = policy.step_inference_cached(&x, cache, start_pos);
                        action_used = Some(idx);
                        ppo::ACTIONS[idx]
                    } else if let Some(seq_data) = ppo_seq_data {
                        let idx = policy.step_inference(seq_data);
                        action_used = Some(idx);
                        ppo::ACTIONS[idx]
                    } else {
                        // Fallback: should usually have seq data if PPO is active
                        0.0 
                    }
                } else {
                    let (idx, log_prob, value) =
                        if let (Some(seq), Some(pities)) = (ppo_state_seq, ppo_pity_seq) {
                            policy.step(seq, pities)
                        } else {
                            let tensor_x = AutoTensor::new(x.to_vec(), vec![DIM]);
                            policy.step(&tensor_x, &[state.pity_6])
                        };
                    action_used = Some(idx);
                    ppo_log_prob = Some(log_prob);
                    ppo_value = Some(value);
                    ppo::ACTIONS[idx]
                }
            } else {
                // Fallback
                let dropout_seed = (state.pity_6 as u64)
                    .wrapping_add((nn_total_pulls as u64).wrapping_mul(31))
                    .wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
                neural_opt.predict(&x, dropout_seed)
            }
        } else {
            // Default probability mode
            let dropout_seed = (state.pity_6 as u64)
                .wrapping_add((nn_total_pulls as u64).wrapping_mul(31))
                .wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
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
        action: action_used,
        ppo_log_prob,
        ppo_value,
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
    exp_sender: Option<&Sender<Experience>>,
    neural_sender: Option<&Sender<NeuralSample>>,
    ppo_sender: Option<&Sender<PpoExperience>>,
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

    let non_up_six = if control.collect_details {
        build_non_up_six(config)
    } else {
        Vec::new()
    };

    let mut pulls = if control.collect_details {
        Some(Vec::with_capacity(control.max_pulls.unwrap_or(0)))
    } else {
        None
    };

    let mut pulls_done = 0usize;
    let ppo_active = ppo_policy.is_some() && config.luck_mode == "ppo";
    let context_len = if config.ppo_context_len > 0 {
        config.ppo_context_len
    } else {
        8
    };
    let mut history_buffer: VecDeque<Tensor> = VecDeque::with_capacity(context_len);
    let mut pity_buffer: VecDeque<usize> = VecDeque::with_capacity(context_len);
    let mut seq_data: Vec<f64> = Vec::with_capacity(context_len * DIM);
    let mut pity_vec: Vec<usize> = Vec::with_capacity(context_len);

    let mut kv_cache = if ppo_active && control.fast_inference {
        if let Some(policy) = ppo_policy {
            Some(KVCache::new(policy.backbone.mla_layer.config.num_heads))
        } else {
            None
        }
    } else {
        None
    };

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

        let current_pity = state.pity_6;
        let nn_total_pulls = if control.nn_total_pulls_one_based {
            pulls_done + 1
        } else {
            pulls_done
        };

        let current_state = build_features(
            state.pity_6,
            nn_total_pulls,
            env_noise,
            state.streak_4_star,
            env_bias,
            state.loss_streak,
            config,
        );
        let mut ppo_state_tensor: Option<AutoTensor> = None;
        let mut ppo_seq_data: Option<Vec<f64>> = None;
        let mut ppo_seq_slice: Option<&[f64]> = None;
        let mut ppo_seq_len = 0usize;
        let mut ppo_pity_vec: Option<Vec<usize>> = None;
        let mut ppo_pity_slice: Option<&[usize]> = None;
        if ppo_active {
            history_buffer.push_back(current_state);
            pity_buffer.push_back(current_pity);
            if history_buffer.len() > context_len {
                history_buffer.pop_front();
            }
            if pity_buffer.len() > context_len {
                pity_buffer.pop_front();
            }
            let seq_len = history_buffer.len();
            seq_data.clear();
            for s in history_buffer.iter() {
                seq_data.extend_from_slice(s);
            }
            
            ppo_seq_slice = Some(&seq_data);
            
            if !control.fast_inference || ppo_sender.is_some() {
                ppo_state_tensor = Some(AutoTensor::new(seq_data.clone(), vec![seq_len, DIM]));
            }
            if ppo_sender.is_some() {
                ppo_seq_data = Some(seq_data.clone());
            }
            ppo_seq_len = seq_len;
            pity_vec.clear();
            pity_vec.extend(pity_buffer.iter().copied());
            ppo_pity_slice = Some(&pity_vec);
            if ppo_sender.is_some() {
                ppo_pity_vec = Some(pity_vec.clone());
            }
        }

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
            ppo_state_tensor.as_ref(),
            ppo_pity_slice,
            control.fast_inference,
            ppo_seq_slice,
            &mut kv_cache,
            pulls_done,
        );

        if let (Some(policy), Some(cache)) = (ppo_policy, &mut kv_cache) {
             policy.prune_cache(cache, context_len);
        }

        let next_state = build_features(
            state.pity_6,
            nn_total_pulls + 1,
            env_noise,
            state.streak_4_star,
            env_bias,
            state.loss_streak,
            config,
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

        if let (Some(action), Some(sender)) = (outcome.action, exp_sender) {
            let mut reward = -0.1;
            if outcome.rarity == 6 {
                if outcome.is_up {
                    reward += 10.0;
                } else {
                    reward += 2.0;
                }
            }
            if state.loss_streak >= 2 {
                reward -= (state.loss_streak as f64) * 0.5;
            }
            let done = outcome.is_up || (pulls_done + 1) >= 300;
            let _ = sender.send(Experience {
                state: current_state.to_vec(),
                action,
                reward,
                next_state: next_state.to_vec(),
                done,
                td_error: 1.0,
            });
        }
        if let Some(sender) = neural_sender {
            let mut reward = -0.1;
            if outcome.rarity == 6 {
                if outcome.is_up {
                    reward += 1.0;
                } else {
                    reward += 0.2;
                }
            }
            if state.loss_streak >= 2 {
                reward -= (state.loss_streak as f64) * 0.2;
            }
            let _ = sender.send(NeuralSample {
                state: current_state,
                reward,
            });
        }
        if let (Some(log_prob), Some(value), Some(sender), Some(state_data), Some(pity_vec)) = (
            outcome.ppo_log_prob,
            outcome.ppo_value,
            ppo_sender,
            ppo_seq_data,
            ppo_pity_vec,
        ) {
            if let Some(action) = outcome.action {
                let mut reward = -0.1;
                if outcome.rarity == 6 {
                    if outcome.is_up {
                        reward += 10.0;
                    } else {
                        reward += 2.0;
                    }
                }
                if state.loss_streak >= 2 {
                    reward -= (state.loss_streak as f64) * 2.0;
                }
                let done = outcome.is_up || (pulls_done + 1) >= 300;
                let _ = sender.send(PpoExperience {
                    state: state_data,
                    seq_len: ppo_seq_len,
                    pity: pity_vec,
                    action,
                    log_prob,
                    reward,
                    done,
                    value,
                });
            }
        }

        if let Some(ref mut pulls_vec) = pulls {
            let op_idx = match outcome.rarity {
                6 => {
                    if outcome.is_up {
                        rng.next_u64_bounded(config.up_six.len() as u64) as usize
                    } else {
                        rng.next_u64_bounded(non_up_six.len() as u64) as usize
                    }
                }
                5 => rng.next_u64_bounded(config.five_stars.len() as u64) as usize,
                _ => rng.next_u64_bounded(config.four_stars.len() as u64) as usize,
            };
            pulls_vec.push(PullResult {
                rarity: outcome.rarity,
                operator_idx: op_idx,
                is_up: outcome.is_up,
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

pub fn build_non_up_six(config: &Config) -> Vec<String> {
    config
        .six_stars
        .iter()
        .filter(|s| !config.up_six.contains(s))
        .cloned()
        .collect()
}

pub fn resolve_operator_name<'a>(
    pull: &PullResult,
    config: &'a Config,
    non_up_six: &'a [String],
) -> &'a str {
    match pull.rarity {
        6 => {
            if pull.is_up {
                &config.up_six[pull.operator_idx]
            } else {
                &non_up_six[pull.operator_idx]
            }
        }
        5 => &config.five_stars[pull.operator_idx],
        _ => &config.four_stars[pull.operator_idx],
    }
}

pub fn simulate_fast(
    num_pulls: usize,
    rng: &mut Rng,
    available_free_pulls: u32,
    neural_opt: &NeuralLuckOptimizer,
    dqn_policy: Option<&DuelingQNetwork>,
    ppo_policy: Option<&ActorCritic>,
    dbn: &Dbn,
    config: &Config,
    exp_sender: Option<&Sender<Experience>>,
    neural_sender: Option<&Sender<NeuralSample>>,
    ppo_sender: Option<&Sender<PpoExperience>>,
) -> SimStatsResult {
    let control = SimControl {
        max_pulls: Some(num_pulls),
        stop_on_up: false,
        stop_after_total_pulls: None,
        nn_total_pulls_one_based: false,
        collect_details: false,
        big_pity_requires_not_up: true,
        fast_inference: true,
    };
    simulate_core(
        &control,
        rng,
        available_free_pulls,
        neural_opt,
        dqn_policy,
        ppo_policy,
        dbn,
        config,
        exp_sender,
        neural_sender,
        ppo_sender,
    )
    .0
}

pub fn simulate_one(
    num_pulls: usize,
    rng: &mut Rng,
    available_free_pulls: u32,
    neural_opt: &NeuralLuckOptimizer,
    dqn_policy: Option<&DuelingQNetwork>,
    ppo_policy: Option<&ActorCritic>,
    dbn: &Dbn,
    config: &Config,
    exp_sender: Option<&Sender<Experience>>,
    neural_sender: Option<&Sender<NeuralSample>>,
    ppo_sender: Option<&Sender<PpoExperience>>,
) -> SimulationResult {
    let control = SimControl {
        max_pulls: Some(num_pulls),
        stop_on_up: false,
        stop_after_total_pulls: None,
        nn_total_pulls_one_based: false,
        collect_details: true,
        big_pity_requires_not_up: true,
        fast_inference: false,
    };
    let (stats, pulls_opt) = simulate_core(
        &control,
        rng,
        available_free_pulls,
        neural_opt,
        dqn_policy,
        ppo_policy,
        dbn,
        config,
        exp_sender,
        neural_sender,
        ppo_sender,
    );
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

fn compute_chunk_size(num_sims: usize, worker: &GoodJobWorker) -> usize {
    if num_sims == 0 {
        return 1;
    }
    let threads = worker.thread_count().max(1);
    let target_chunks = threads.saturating_mul(8).max(1);
    let mut size = num_sims.div_ceil(target_chunks);
    if size < 64 {
        size = 64;
    }
    if size > num_sims {
        size = num_sims;
    }
    size
}

pub fn simulate_stats(
    num_pulls: usize,
    num_sims: usize,
    seed: u64,
    neural_opt: &NeuralLuckOptimizer,
    dqn_policy: Option<&DuelingQNetwork>,
    ppo_policy: Option<&ActorCritic>,
    dbn: &Dbn,
    config: &Config,
    worker: &GoodJobWorker,
    exp_sender: Option<&Sender<Experience>>,
    neural_sender: Option<&Sender<NeuralSample>>,
    ppo_sender: Option<&Sender<PpoExperience>>,
) -> (usize, usize, usize, usize) {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();

    let chunk_size = compute_chunk_size(num_sims, worker);
    let chunk_count = num_sims.div_ceil(chunk_size);
    let (total_six, total_up, total_big_pity, total_with_up) = worker
        .execute(|| {
            (0..chunk_count)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(num_sims);
                    let mut local_rng = Rng::from_seed(base_seed.wrapping_add(chunk_idx as u64));
                    let mut total_six = 0usize;
                    let mut total_up = 0usize;
                    let mut total_big_pity = 0usize;
                    let mut total_with_up = 0usize;
                    for _ in start..end {
                        let res = simulate_fast(
                            num_pulls,
                            &mut local_rng,
                            0,
                            neural_opt,
                            dqn_policy,
                            ppo_policy,
                            dbn,
                            config,
                            exp_sender,
                            neural_sender,
                            ppo_sender,
                        );
                        total_six += res.six_count;
                        total_up += res.up_count;
                        if res.big_pity_used {
                            total_big_pity += 1;
                        }
                        if res.up_count > 0 {
                            total_with_up += 1;
                        }
                    }
                    (total_six, total_up, total_big_pity, total_with_up)
                })
                .reduce(
                    || (0, 0, 0, 0),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
                )
        })
        .unwrap_or_else(|e| {
            println!("[Error] Simulation failed: {}", e);
            (0, 0, 0, 0)
        });

    (total_six, total_up, total_big_pity, total_with_up)
}

pub fn simulate_f2p_clearing(
    num_sims: usize,
    seed: u64,
    neural_opt: &NeuralLuckOptimizer,
    dqn_policy: Option<&DuelingQNetwork>,
    ppo_policy: Option<&ActorCritic>,
    dbn: &Dbn,
    config: &Config,
    worker: &GoodJobWorker,
    exp_sender: Option<&Sender<Experience>>,
    neural_sender: Option<&Sender<NeuralSample>>,
    ppo_sender: Option<&Sender<PpoExperience>>,
) -> (u64, usize, usize) {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();

    let chunk_size = compute_chunk_size(num_sims, worker);
    let chunk_count = num_sims.div_ceil(chunk_size);
    let (total_extra_cost, extra_cost_samples, success_count_val) = worker
        .execute(|| {
            (0..chunk_count)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(num_sims);
                    let mut local_rng = Rng::from_seed(base_seed.wrapping_add(chunk_idx as u64));
                    let control = SimControl {
                        max_pulls: None,
                        stop_on_up: true,
                        // Fix: Ensure we use all available free pulls, covering the big pity if enough
                        stop_after_total_pulls: Some(
                            FREE_PULLS_WELFARE.max(config.big_pity_cumulative as u32) as usize,
                        ),
                        nn_total_pulls_one_based: true,
                        collect_details: false,
                        big_pity_requires_not_up: false,
                        fast_inference: true,
                    };
                    let mut total_extra = 0u64;
                    let mut total_samples = 0usize;
                    let mut total_success = 0usize;
                    for _ in start..end {
                        let (stats, _) = simulate_core(
                            &control,
                            &mut local_rng,
                            FREE_PULLS_WELFARE,
                            neural_opt,
                            dqn_policy,
                            ppo_policy,
                            dbn,
                            config,
                            exp_sender,
                            neural_sender,
                            ppo_sender,
                        );
                        let success = if stats.up_count > 0 { 1 } else { 0 };
                        let extra = if stats.cost_jade > 0 {
                            stats.cost_jade as u64
                        } else {
                            0
                        };
                        let extra_sample = if stats.cost_jade > 0 { 1 } else { 0 };
                        total_extra += extra;
                        total_samples += extra_sample;
                        total_success += success;
                    }
                    (total_extra, total_samples, total_success)
                })
                .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
        })
        .unwrap_or_else(|e| {
            println!("[Error] Cost Analysis failed: {}", e);
            (0, 0, 0)
        });

    (total_extra_cost, extra_cost_samples, success_count_val)
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
                config,
            );

            // Calculate what the neural network WOULD output
            let dropout_seed = (state.pity_6 as u64)
                .wrapping_add((nn_total_pulls as u64).wrapping_mul(31))
                .wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
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
                true, // big_pity_requires_not_up
                None,
                None,
                false,
                None,
                &mut None,
                0,
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
        format!(
            "Probability to get UP with ONLY free resources: {:.2}%",
            rate * 100.0
        )
    }
}

pub fn format_avg_extra_cost_line(avg_extra_cost: Option<f64>) -> String {
    match avg_extra_cost {
        Some(cost) => format!(
            "Avg Extra Jade Cost: {:.0} (Approx. {:.1} extra pulls)",
            cost,
            cost / 500.0
        ),
        None => "Avg Extra Jade Cost: N/A".to_string(),
    }
}
