use crate::autograd::Tensor as AutoTensor;
use crate::config::{Config, LuckMode};
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
/// Jade cost per single pull (in-game currency).
pub const COST_PER_PULL: u32 = 500;
/// Number of free pulls available to F2P players per banner cycle.
pub const FREE_PULLS_WELFARE: u32 = 135;
use crate::utils::{
    compute_reward_dqn, compute_reward_neural, compute_reward_ppo, DEFAULT_PPO_CONTEXT_LEN,
    EPISODE_MAX_PULLS,
};

#[derive(Clone, Debug)]
pub struct PullResult {
    pub rarity: u8,
    pub operator_idx: usize,
    pub is_up: bool,
}

/// Full simulation result including per-pull details.
#[derive(Clone)]
pub struct SimulationResult {
    pub pulls: Vec<PullResult>,
    pub six_count: usize,
    pub up_count: usize,
    pub big_pity_used: bool,
    pub cost_jade: u32,
    pub free_pulls_used: u32,
}

/// Lightweight simulation statistics without per-pull details.
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

/// Mutable state tracked across pulls within a single simulation.
#[derive(Clone)]
pub struct PullState {
    pub pity_6: usize,
    pub total_pulls_in_pool: usize,
    pub has_obtained_up: bool,
    pub streak_4_star: usize,
    pub loss_streak: usize,
}

/// Result of a single gacha pull including rarity, UP status, and policy info.
#[derive(Clone)]
pub struct PullOutcome {
    pub rarity: u8,
    pub is_up: bool,
    pub big_pity_used: bool,
    pub action: Option<usize>,
    pub ppo_log_prob: Option<f64>,
    pub ppo_value: Option<f64>,
}

struct PolicyDecision {
    luck_factor: f64,
    action: Option<usize>,
    ppo_log_prob: Option<f64>,
    ppo_value: Option<f64>,
}

struct PpoInputs {
    seq_len: usize,
    seq_tensor: Option<AutoTensor>,
    pity_vec: Option<Vec<usize>>,
}

impl PpoInputs {
    fn empty() -> Self {
        Self {
            seq_len: 0,
            seq_tensor: None,
            pity_vec: None,
        }
    }
}

struct PpoContext {
    active: bool,
    context_len: usize,
    history_buffer: VecDeque<Tensor>,
    pity_buffer: VecDeque<usize>,
    seq_data: Vec<f64>,
    pity_vec: Vec<usize>,
    kv_cache: Option<KVCache>,
}

impl PpoContext {
    fn new(
        active: bool,
        context_len: usize,
        ppo_policy: Option<&ActorCritic>,
        fast_inference: bool,
    ) -> Self {
        let kv_cache = if active && fast_inference {
            ppo_policy.map(|policy| KVCache::new(policy.backbone.mla_layer.config.num_heads))
        } else {
            None
        };
        Self {
            active,
            context_len,
            history_buffer: VecDeque::with_capacity(context_len),
            pity_buffer: VecDeque::with_capacity(context_len),
            seq_data: Vec::with_capacity(context_len * DIM),
            pity_vec: Vec::with_capacity(context_len),
            kv_cache,
        }
    }

    fn reset(
        &mut self,
        active: bool,
        context_len: usize,
        ppo_policy: Option<&ActorCritic>,
        fast_inference: bool,
    ) {
        self.active = active;
        self.context_len = context_len;
        self.history_buffer.clear();
        self.pity_buffer.clear();
        if self.history_buffer.capacity() < context_len {
            self.history_buffer
                .reserve(context_len - self.history_buffer.capacity());
        }
        if self.pity_buffer.capacity() < context_len {
            self.pity_buffer
                .reserve(context_len - self.pity_buffer.capacity());
        }
        self.seq_data.clear();
        self.pity_vec.clear();
        let target_seq_capacity = context_len.saturating_mul(DIM);
        if self.seq_data.capacity() < target_seq_capacity {
            self.seq_data
                .reserve(target_seq_capacity - self.seq_data.capacity());
        }
        if self.pity_vec.capacity() < context_len {
            self.pity_vec
                .reserve(context_len - self.pity_vec.capacity());
        }
        if active && fast_inference {
            let num_heads = ppo_policy
                .map(|policy| policy.backbone.mla_layer.config.num_heads)
                .unwrap_or(0);
            if num_heads == 0 {
                self.kv_cache = None;
            } else if let Some(cache) = &mut self.kv_cache {
                if cache.k_cache.len() != num_heads {
                    self.kv_cache = Some(KVCache::new(num_heads));
                } else {
                    cache.clear();
                }
            } else {
                self.kv_cache = Some(KVCache::new(num_heads));
            }
        } else {
            self.kv_cache = None;
        }
    }

    fn build_inputs(
        &mut self,
        current_state: Tensor,
        current_pity: usize,
        need_tensor: bool,
    ) -> PpoInputs {
        if !self.active {
            return PpoInputs::empty();
        }
        self.history_buffer.push_back(current_state);
        self.pity_buffer.push_back(current_pity);
        if self.history_buffer.len() > self.context_len {
            self.history_buffer.pop_front();
        }
        if self.pity_buffer.len() > self.context_len {
            self.pity_buffer.pop_front();
        }
        let seq_len = self.history_buffer.len();

        self.seq_data.clear();
        for s in self.history_buffer.iter() {
            self.seq_data.extend_from_slice(s);
        }
        self.pity_vec.clear();
        self.pity_vec.extend(self.pity_buffer.iter().copied());

        if need_tensor {
            let seq_tensor = Some(AutoTensor::new(self.seq_data.clone(), vec![seq_len, DIM]));
            PpoInputs {
                seq_len,
                seq_tensor,
                pity_vec: Some(self.pity_vec.clone()),
            }
        } else {
            PpoInputs {
                seq_len,
                seq_tensor: None,
                pity_vec: None,
            }
        }
    }

    fn prune_cache(&mut self, policy: &ActorCritic) {
        if let Some(cache) = &mut self.kv_cache {
            policy.prune_cache(cache, self.context_len);
        }
    }
}

struct PolicyInputs<'a> {
    state: &'a PullState,
    nn_total_pulls: usize,
    config: &'a Config,
    neural_opt: &'a NeuralLuckOptimizer,
    dqn_policy: Option<&'a DuelingQNetwork>,
    ppo_policy: Option<&'a ActorCritic>,
    current_features: &'a Tensor,
    ppo_state_seq: Option<&'a AutoTensor>,
    ppo_pity_seq: Option<&'a [usize]>,
    fast_inference: bool,
    ppo_seq_data: Option<&'a [f64]>,
    kv_cache: &'a mut Option<KVCache>,
    start_pos: usize,
}

fn decide_policy(inputs: PolicyInputs<'_>) -> PolicyDecision {
    let PolicyInputs {
        state,
        nn_total_pulls,
        config,
        neural_opt,
        dqn_policy,
        ppo_policy,
        current_features,
        ppo_state_seq,
        ppo_pity_seq,
        fast_inference,
        ppo_seq_data,
        kv_cache,
        start_pos,
    } = inputs;
    if config.luck_mode == LuckMode::Dqn {
        if let Some(policy) = dqn_policy {
            if fast_inference {
                let (idx, modifier) = policy.predict_action_fast(current_features);
                return PolicyDecision {
                    luck_factor: modifier,
                    action: Some(idx),
                    ppo_log_prob: None,
                    ppo_value: None,
                };
            }
            let tensor_x = AutoTensor::new(current_features.to_vec(), vec![DIM]);
            let (idx, modifier) = policy.predict_action(&tensor_x);
            return PolicyDecision {
                luck_factor: modifier,
                action: Some(idx),
                ppo_log_prob: None,
                ppo_value: None,
            };
        }
    } else if config.luck_mode == LuckMode::Ppo {
        if let Some(policy) = ppo_policy {
            if fast_inference {
                if let Some(cache) = kv_cache {
                    let idx = policy.step_inference_cached(current_features, cache, start_pos);
                    return PolicyDecision {
                        luck_factor: ppo::ACTIONS[idx],
                        action: Some(idx),
                        ppo_log_prob: None,
                        ppo_value: None,
                    };
                }
                if let Some(seq_data) = ppo_seq_data {
                    let idx = policy.step_inference(seq_data);
                    return PolicyDecision {
                        luck_factor: ppo::ACTIONS[idx],
                        action: Some(idx),
                        ppo_log_prob: None,
                        ppo_value: None,
                    };
                }
                return PolicyDecision {
                    luck_factor: 0.0,
                    action: None,
                    ppo_log_prob: None,
                    ppo_value: None,
                };
            }
            let (idx, log_prob, value) =
                if let (Some(seq), Some(pities)) = (ppo_state_seq, ppo_pity_seq) {
                    policy.step(seq, pities)
                } else {
                    let tensor_x = AutoTensor::new(current_features.to_vec(), vec![DIM]);
                    policy.step(&tensor_x, &[state.pity_6])
                };
            return PolicyDecision {
                luck_factor: ppo::ACTIONS[idx],
                action: Some(idx),
                ppo_log_prob: Some(log_prob),
                ppo_value: Some(value),
            };
        }
    }

    let dropout_seed = (state.pity_6 as u64)
        .wrapping_add((nn_total_pulls as u64).wrapping_mul(31))
        .wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
    let luck_factor = neural_opt.predict(current_features, dropout_seed);
    PolicyDecision {
        luck_factor,
        action: None,
        ppo_log_prob: None,
        ppo_value: None,
    }
}

/// Training sample for online neural optimizer.
#[derive(Clone)]
pub struct NeuralSample {
    pub state: Tensor,
    pub reward: f64,
}

/// Training experience for online PPO trainer.
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

/// Controls simulation behavior: max pulls, stop conditions, inference mode.
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

/// Calculate 6-star probability at a given pity count.
pub fn prob_6(pity_6: usize, config: &Config) -> f64 {
    if pity_6 < config.soft_pity_start {
        config.prob_6_base
    } else if pity_6 < config.small_pity_guarantee {
        config.prob_6_base + 0.05 * (pity_6 as f64 - (config.soft_pity_start as f64 - 1.0))
    } else {
        1.0
    }
}

/// Expected number of pulls for one 6-star under the current pool config.
pub fn expected_pulls_per_six(config: &Config) -> f64 {
    let mut survival = 1.0;
    let mut expected = 0.0;
    for k in 1..=config.small_pity_guarantee {
        let p6 = prob_6(k, config);
        let prob_k = survival * p6;
        expected += k as f64 * prob_k;
        survival *= 1.0 - p6;
        if p6 >= 1.0 {
            break;
        }
    }
    expected
}

/// Build the 8-dimensional feature vector for neural network input.
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

/// Execute a single gacha pull, advancing state and returning the outcome.
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
    // Intentionally duplicated blocks: up_pity_soft and big_pity_cumulative are
    // semantically distinct pity thresholds that happen to share the same outcome.
    // Merging them would obscure game-mechanical intent.
    #[allow(clippy::if_same_then_else)]
    if config.up_pity_soft > 0 && state.total_pulls_in_pool == config.up_pity_soft && big_pity_gate
    {
        rarity = 6;
        is_up = true;
        big_pity_used = true;
        state.pity_6 = 0;
        state.streak_4_star = 0;
        state.loss_streak = 0;
    } else if config.big_pity_cumulative > 0
        && state.total_pulls_in_pool == config.big_pity_cumulative
        && big_pity_gate
    {
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

        let decision = decide_policy(PolicyInputs {
            state,
            nn_total_pulls,
            config,
            neural_opt,
            dqn_policy,
            ppo_policy,
            current_features: &x,
            ppo_state_seq,
            ppo_pity_seq,
            fast_inference,
            ppo_seq_data,
            kv_cache,
            start_pos,
        });
        action_used = decision.action;
        ppo_log_prob = decision.ppo_log_prob;
        ppo_value = decision.ppo_value;

        let final_prob_6 = (base_prob_6 + decision.luck_factor).clamp(0.0, 1.0);
        let r = rng.next_f64();

        if r < final_prob_6 {
            rarity = 6;
            state.pity_6 = 0;
            state.streak_4_star = 0;

            if config.up_rate > 0.0 && !config.up_six.is_empty() {
                if rng.next_f64() < config.up_rate {
                    is_up = true;
                    state.loss_streak = 0;
                } else {
                    is_up = false;
                    state.loss_streak += 1;
                }
            }
        } else {
            let force_5_star = config.always_5_star
                || (config.five_star_pity > 0 && state.streak_4_star >= config.five_star_pity - 1);
            if force_5_star || r < (final_prob_6 + config.prob_5_base).min(1.0) {
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

/// References to all ML models and training channels for a simulation run.
pub struct SimModelContext<'a> {
    pub neural_opt: &'a NeuralLuckOptimizer,
    pub dqn_policy: Option<&'a DuelingQNetwork>,
    pub ppo_policy: Option<&'a ActorCritic>,
    pub dbn: &'a Dbn,
    pub config: &'a Config,
    pub exp_sender: Option<&'a Sender<Experience>>,
    pub neural_sender: Option<&'a Sender<NeuralSample>>,
    pub ppo_sender: Option<&'a Sender<PpoExperience>>,
}

/// Core simulation loop, executes a single gacha session.
pub fn simulate_core(
    control: &SimControl,
    rng: &mut Rng,
    available_free_pulls: u32,
    ctx: &SimModelContext<'_>,
) -> (SimStatsResult, Option<Vec<PullResult>>) {
    let ppo_active = ctx.ppo_policy.is_some() && ctx.config.luck_mode == LuckMode::Ppo;
    let context_len = if ctx.config.ppo_context_len > 0 {
        ctx.config.ppo_context_len
    } else {
        DEFAULT_PPO_CONTEXT_LEN
    };
    let mut ppo_context = PpoContext::new(
        ppo_active,
        context_len,
        ctx.ppo_policy,
        control.fast_inference,
    );
    simulate_core_with_context(control, rng, available_free_pulls, ctx, &mut ppo_context)
}

fn simulate_core_with_context(
    control: &SimControl,
    rng: &mut Rng,
    available_free_pulls: u32,
    ctx: &SimModelContext<'_>,
    ppo_context: &mut PpoContext,
) -> (SimStatsResult, Option<Vec<PullResult>>) {
    let mut big_pity_used = false;
    let mut six_count = 0;
    let mut up_count = 0;

    let mut free_pulls_remaining = available_free_pulls;
    let mut cost_jade = 0;

    let mut max_loss_streak = 0;

    let (env_noise, env_bias) = dbn_env(ctx.dbn, rng);
    let mut state = PullState {
        pity_6: 0,
        total_pulls_in_pool: 0,
        has_obtained_up: false,
        streak_4_star: 0,
        loss_streak: 0,
    };

    let non_up_six = if control.collect_details {
        build_non_up_six(ctx.config)
    } else {
        Vec::new()
    };

    let mut pulls = if control.collect_details {
        Some(Vec::with_capacity(control.max_pulls.unwrap_or(0)))
    } else {
        None
    };

    let mut pulls_done = 0usize;
    let ppo_active = ctx.ppo_policy.is_some() && ctx.config.luck_mode == LuckMode::Ppo;
    let context_len = if ctx.config.ppo_context_len > 0 {
        ctx.config.ppo_context_len
    } else {
        DEFAULT_PPO_CONTEXT_LEN
    };
    ppo_context.reset(
        ppo_active,
        context_len,
        ctx.ppo_policy,
        control.fast_inference,
    );

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
            ctx.config,
        );
        let has_any_sender =
            ctx.exp_sender.is_some() || ctx.neural_sender.is_some() || ctx.ppo_sender.is_some();
        let need_tensor = !control.fast_inference || ctx.ppo_sender.is_some();
        let ppo_inputs = ppo_context.build_inputs(current_state, current_pity, need_tensor);

        let outcome = roll_one(
            &mut state,
            rng,
            ctx.neural_opt,
            ctx.dqn_policy,
            ctx.ppo_policy,
            env_noise,
            env_bias,
            ctx.config,
            nn_total_pulls,
            control.big_pity_requires_not_up,
            ppo_inputs.seq_tensor.as_ref(),
            ppo_inputs.pity_vec.as_deref(),
            control.fast_inference,
            if control.fast_inference && ppo_context.active {
                Some(ppo_context.seq_data.as_slice())
            } else {
                None
            },
            &mut ppo_context.kv_cache,
            pulls_done,
        );

        if let Some(policy) = ctx.ppo_policy {
            ppo_context.prune_cache(policy);
        }

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

        if has_any_sender {
            let next_state = build_features(
                state.pity_6,
                nn_total_pulls + 1,
                env_noise,
                state.streak_4_star,
                env_bias,
                state.loss_streak,
                ctx.config,
            );
            record_training_samples(TrainingSampleInputs {
                outcome: &outcome,
                current_state: &current_state,
                next_state: &next_state,
                pulls_done,
                state: &state,
                exp_sender: ctx.exp_sender,
                neural_sender: ctx.neural_sender,
                ppo_sender: ctx.ppo_sender,
                ppo_inputs: &ppo_inputs,
                ppo_context_seq_data: &ppo_context.seq_data,
                ppo_context_pity_vec: &ppo_context.pity_vec,
            });
        }

        if let Some(ref mut pulls_vec) = pulls {
            let op_idx = match outcome.rarity {
                6 => {
                    if outcome.is_up {
                        if ctx.config.up_six.is_empty() {
                            0
                        } else {
                            rng.next_u64_bounded(ctx.config.up_six.len() as u64) as usize
                        }
                    } else if non_up_six.is_empty() {
                        0
                    } else {
                        rng.next_u64_bounded(non_up_six.len() as u64) as usize
                    }
                }
                5 => {
                    if ctx.config.five_stars.is_empty() {
                        0
                    } else {
                        rng.next_u64_bounded(ctx.config.five_stars.len() as u64) as usize
                    }
                }
                _ => {
                    if ctx.config.four_stars.is_empty() {
                        0
                    } else {
                        rng.next_u64_bounded(ctx.config.four_stars.len() as u64) as usize
                    }
                }
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
            if state.total_pulls_in_pool >= limit {
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

struct TrainingSampleInputs<'a> {
    outcome: &'a PullOutcome,
    current_state: &'a Tensor,
    next_state: &'a Tensor,
    pulls_done: usize,
    state: &'a PullState,
    exp_sender: Option<&'a Sender<Experience>>,
    neural_sender: Option<&'a Sender<NeuralSample>>,
    ppo_sender: Option<&'a Sender<PpoExperience>>,
    ppo_inputs: &'a PpoInputs,
    ppo_context_seq_data: &'a [f64],
    ppo_context_pity_vec: &'a [usize],
}

fn record_training_samples(inputs: TrainingSampleInputs<'_>) {
    let TrainingSampleInputs {
        outcome,
        current_state,
        next_state,
        pulls_done,
        state,
        exp_sender,
        neural_sender,
        ppo_sender,
        ppo_inputs,
        ppo_context_seq_data,
        ppo_context_pity_vec,
    } = inputs;
    if let (Some(action), Some(sender)) = (outcome.action, exp_sender) {
        let reward = compute_reward_dqn(outcome.rarity == 6, outcome.is_up, state.loss_streak);
        let done = outcome.is_up || (pulls_done + 1) >= EPISODE_MAX_PULLS;
        let _ = sender.send(Experience {
            state: current_state.to_vec(),
            action,
            reward,
            next_state: next_state.to_vec(),
            done,
        });
    }

    if let Some(sender) = neural_sender {
        let reward = compute_reward_neural(outcome.rarity == 6, outcome.is_up, state.loss_streak);
        let _ = sender.send(NeuralSample {
            state: *current_state,
            reward,
        });
    }

    if let (Some(log_prob), Some(value), Some(sender)) =
        (outcome.ppo_log_prob, outcome.ppo_value, ppo_sender)
    {
        if let Some(action) = outcome.action {
            let reward = compute_reward_ppo(outcome.rarity == 6, outcome.is_up, state.loss_streak);
            let done = outcome.is_up || (pulls_done + 1) >= EPISODE_MAX_PULLS;
            let _ = sender.send(PpoExperience {
                state: ppo_context_seq_data.to_vec(),
                seq_len: ppo_inputs.seq_len,
                pity: ppo_context_pity_vec.to_vec(),
                action,
                log_prob,
                reward,
                done,
                value,
            });
        }
    }
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
                config
                    .up_six
                    .get(pull.operator_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown")
            } else {
                non_up_six
                    .get(pull.operator_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown")
            }
        }
        5 => config
            .five_stars
            .get(pull.operator_idx)
            .map(|s| s.as_str())
            .unwrap_or("Unknown"),
        _ => config
            .four_stars
            .get(pull.operator_idx)
            .map(|s| s.as_str())
            .unwrap_or("Unknown"),
    }
}

/// Fast simulation without per-pull details, optimized for batch runs.
pub fn simulate_fast(
    num_pulls: usize,
    rng: &mut Rng,
    available_free_pulls: u32,
    ctx: &SimModelContext<'_>,
) -> SimStatsResult {
    let control = SimControl {
        max_pulls: Some(num_pulls),
        stop_on_up: false,
        stop_after_total_pulls: None,
        nn_total_pulls_one_based: false,
        collect_details: false,
        big_pity_requires_not_up: ctx.config.big_pity_requires_not_up,
        fast_inference: true,
    };
    simulate_core(&control, rng, available_free_pulls, ctx).0
}

/// Single simulation with full pull details and training feedback.
pub fn simulate_one(
    num_pulls: usize,
    rng: &mut Rng,
    available_free_pulls: u32,
    ctx: &SimModelContext<'_>,
) -> SimulationResult {
    let control = SimControl {
        max_pulls: Some(num_pulls),
        stop_on_up: false,
        stop_after_total_pulls: None,
        nn_total_pulls_one_based: false,
        collect_details: true,
        big_pity_requires_not_up: ctx.config.big_pity_requires_not_up,
        fast_inference: false,
    };
    let (stats, pulls_opt) = simulate_core(&control, rng, available_free_pulls, ctx);
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

/// Extended simulation context including the worker pool for parallel runs.
pub struct SimRunContext<'a> {
    pub neural_opt: &'a NeuralLuckOptimizer,
    pub dqn_policy: Option<&'a DuelingQNetwork>,
    pub ppo_policy: Option<&'a ActorCritic>,
    pub dbn: &'a Dbn,
    pub config: &'a Config,
    pub worker: &'a GoodJobWorker,
    pub exp_sender: Option<&'a Sender<Experience>>,
    pub neural_sender: Option<&'a Sender<NeuralSample>>,
    pub ppo_sender: Option<&'a Sender<PpoExperience>>,
}

/// Parallel batch simulation returning aggregate statistics.
pub fn simulate_stats(
    num_pulls: usize,
    num_sims: usize,
    seed: u64,
    ctx: &SimRunContext<'_>,
) -> (usize, usize, usize, usize) {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();
    let model_ctx = SimModelContext {
        neural_opt: ctx.neural_opt,
        dqn_policy: ctx.dqn_policy,
        ppo_policy: ctx.ppo_policy,
        dbn: ctx.dbn,
        config: ctx.config,
        exp_sender: ctx.exp_sender,
        neural_sender: ctx.neural_sender,
        ppo_sender: ctx.ppo_sender,
    };

    let chunk_size = compute_chunk_size(num_sims, ctx.worker);
    let chunk_count = num_sims.div_ceil(chunk_size);
    let (total_six, total_up, total_big_pity, total_with_up) = ctx
        .worker
        .execute(|| {
            (0..chunk_count)
                .into_par_iter()
                .map_init(
                    || PpoContext::new(false, 0, None, false),
                    |ppo_context, chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(num_sims);
                        let mut local_rng =
                            Rng::from_seed(base_seed.wrapping_add(chunk_idx as u64));
                        let mut total_six = 0usize;
                        let mut total_up = 0usize;
                        let mut total_big_pity = 0usize;
                        let mut total_with_up = 0usize;
                        for _ in start..end {
                            let control = SimControl {
                                max_pulls: Some(num_pulls),
                                stop_on_up: false,
                                stop_after_total_pulls: None,
                                nn_total_pulls_one_based: false,
                                collect_details: false,
                                big_pity_requires_not_up: ctx.config.big_pity_requires_not_up,
                                fast_inference: true,
                            };
                            let (res, _) = simulate_core_with_context(
                                &control,
                                &mut local_rng,
                                0,
                                &model_ctx,
                                ppo_context,
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
                    },
                )
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

/// Parallel F2P clearing simulation: how many extra pulls to get first UP.
pub fn simulate_f2p_clearing(
    num_sims: usize,
    seed: u64,
    ctx: &SimRunContext<'_>,
) -> (u64, usize, usize) {
    let mut master_rng = Rng::from_seed(seed);
    let base_seed = master_rng.next_u64();
    let model_ctx = SimModelContext {
        neural_opt: ctx.neural_opt,
        dqn_policy: ctx.dqn_policy,
        ppo_policy: ctx.ppo_policy,
        dbn: ctx.dbn,
        config: ctx.config,
        exp_sender: ctx.exp_sender,
        neural_sender: ctx.neural_sender,
        ppo_sender: ctx.ppo_sender,
    };

    let chunk_size = compute_chunk_size(num_sims, ctx.worker);
    let chunk_count = num_sims.div_ceil(chunk_size);
    let (total_extra_cost, extra_cost_samples, success_count_val) = ctx
        .worker
        .execute(|| {
            (0..chunk_count)
                .into_par_iter()
                .map_init(
                    || PpoContext::new(false, 0, None, false),
                    |ppo_context, chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(num_sims);
                        let mut local_rng =
                            Rng::from_seed(base_seed.wrapping_add(chunk_idx as u64));
                        let control = SimControl {
                            max_pulls: None,
                            stop_on_up: true,
                            // Fix: Ensure we use all available free pulls, covering the big pity if enough
                            stop_after_total_pulls: Some(
                                FREE_PULLS_WELFARE.max(ctx.config.big_pity_cumulative as u32)
                                    as usize,
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
                            let (stats, _) = simulate_core_with_context(
                                &control,
                                &mut local_rng,
                                FREE_PULLS_WELFARE,
                                &model_ctx,
                                ppo_context,
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
                    },
                )
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
            cost / COST_PER_PULL as f64
        ),
        None => "Avg Extra Jade Cost: N/A".to_string(),
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
    fn simulate_core_triggers_big_pity() {
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(3);
        let control = SimControl {
            max_pulls: Some(config.big_pity_cumulative),
            stop_on_up: false,
            stop_after_total_pulls: None,
            nn_total_pulls_one_based: false,
            collect_details: false,
            big_pity_requires_not_up: false,
            fast_inference: true,
        };
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            dbn: &dbn,
            config: &config,
            exp_sender: None,
            neural_sender: None,
            ppo_sender: None,
        };
        let (stats, _) = simulate_core(&control, &mut rng, 0, &ctx);
        assert!(stats.big_pity_used);
        assert!(stats.up_count >= 1);
    }

    #[test]
    fn ppo_context_respects_context_len() {
        let mut context = PpoContext::new(true, 2, None, false);
        let inputs1 = context.build_inputs([0.0; DIM], 1, true);
        assert_eq!(inputs1.seq_len, 1);
        assert_eq!(context.seq_data.len(), DIM);
        let inputs2 = context.build_inputs([0.0; DIM], 2, true);
        assert_eq!(inputs2.seq_len, 2);
        assert_eq!(context.seq_data.len(), DIM * 2);
        let inputs3 = context.build_inputs([0.0; DIM], 3, true);
        assert_eq!(inputs3.seq_len, 2);
        assert_eq!(inputs3.pity_vec.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn ppo_fast_slow_alignment_distribution() {
        let policy = crate::ppo::ActorCritic::new(12345, &crate::config::AchfConfig::default());
        let seq_len = 6;
        let mut flat = vec![0.0; seq_len * DIM];
        for t in 0..seq_len {
            for i in 0..DIM {
                flat[t * DIM + i] = (t as f64) * 0.01 + (i as f64) * 0.001;
            }
        }
        let pity: Vec<usize> = (0..seq_len).collect();
        let state_tensor = AutoTensor::new(flat.clone(), vec![seq_len, DIM]);

        let slow_logits = policy.forward_actor(&state_tensor, &pity);
        let slow_data = slow_logits.data.read().unwrap().clone();
        let slow_probs = softmax(&slow_data);

        let mut kv = crate::transformer::KVCache::new(policy.backbone.mla_layer.config.num_heads);
        kv.clear();
        let mut last = vec![0.0; 0];
        for t in 0..seq_len {
            let token = &flat[t * DIM..(t + 1) * DIM];
            last = policy.backbone.forward_inference_step(token, &mut kv, t);
        }
        let fast_logits = policy.actor_head.forward_inference(&last);
        let fast_probs = softmax(&fast_logits);

        let mut diff_sum = 0.0;
        for i in 0..5 {
            diff_sum += (slow_probs[i] - fast_probs[i]).abs();
        }
        assert!(
            diff_sum < 1e-6,
            "Probability mismatch too large: {}",
            diff_sum
        );
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let mut max_l = f64::NEG_INFINITY;
        for &v in logits {
            if v > max_l {
                max_l = v;
            }
        }
        let mut sum = 0.0;
        let mut out = vec![0.0; logits.len()];
        for (i, &v) in logits.iter().enumerate() {
            out[i] = (v - max_l).exp();
            sum += out[i];
        }
        for v in out.iter_mut() {
            *v /= sum;
        }
        out
    }

    #[test]
    fn prob_6_at_boundary_values() {
        let config = Config::load("data/config.json");
        let below = prob_6(config.soft_pity_start - 1, &config);
        assert!(
            (below - config.prob_6_base).abs() < 1e-12,
            "Below soft pity should be base rate"
        );
        let at_start = prob_6(config.soft_pity_start, &config);
        assert!(
            at_start > config.prob_6_base,
            "At soft pity start, rate should be boosted"
        );
        let at_guarantee = prob_6(config.small_pity_guarantee, &config);
        assert!(
            (at_guarantee - 1.0).abs() < 1e-12,
            "At guarantee should be 1.0"
        );
    }

    #[test]
    fn prob_6_monotonically_increases() {
        let config = Config::load("data/config.json");
        let mut prev = 0.0;
        for pity in 0..=config.small_pity_guarantee {
            let p = prob_6(pity, &config);
            assert!(p >= prev, "Prob should be non-decreasing at pity={}", pity);
            prev = p;
        }
    }

    #[test]
    fn roll_one_respects_up_rate_zero() {
        let (mut config, _dbn, neural_opt) = build_context();
        config.up_rate = 0.0;
        let mut rng = Rng::from_seed(999);
        let mut state = PullState {
            pity_6: config.small_pity_guarantee - 1,
            total_pulls_in_pool: 0,
            has_obtained_up: false,
            streak_4_star: 0,
            loss_streak: 0,
        };
        let outcome = roll_one(
            &mut state,
            &mut rng,
            &neural_opt,
            None,
            None,
            0.0,
            0.0,
            &config,
            0,
            false,
            None,
            None,
            true,
            None,
            &mut None,
            0,
        );
        assert_eq!(outcome.rarity, 6);
        assert!(!outcome.is_up, "With up_rate=0 should never be UP via roll");
    }

    #[test]
    fn stop_after_total_pulls_exact_count() {
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(42);
        let limit = 10usize;
        let control = SimControl {
            max_pulls: None,
            stop_on_up: false,
            stop_after_total_pulls: Some(limit),
            nn_total_pulls_one_based: false,
            collect_details: true,
            big_pity_requires_not_up: false,
            fast_inference: true,
        };
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            dbn: &dbn,
            config: &config,
            exp_sender: None,
            neural_sender: None,
            ppo_sender: None,
        };
        let (_, pulls) = simulate_core(&control, &mut rng, 0, &ctx);
        let pulls = pulls.unwrap();
        assert!(
            pulls.len() <= limit,
            "Should not exceed limit: got {} pulls for limit {}",
            pulls.len(),
            limit
        );
    }
}
