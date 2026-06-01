use crate::autograd::Tensor;
use crate::config::Config;
use crate::dqn::DuelingQNetwork;
use crate::env_net::EnvNet;
use crate::gacha_env::{step_pull, GachaAction, GachaStep};
use crate::neural::DIM;
use crate::ppo::{ActorCritic, EARLY_UP_BONUS_THRESHOLD_1, EARLY_UP_BONUS_THRESHOLD_2};
use crate::rng::Rng;
use crate::sim::{build_features_with_luck_budget, env_net_env, PullState};
use crate::transformer::{KVCache, MLAConfig};
use crate::utils::{
    compute_reward_dqn, compute_reward_ppo, ACTIONS, ACTION_SPACE, EPISODE_MAX_PULLS,
};
use std::collections::VecDeque;

const EVAL_SEED_STRIDE: u64 = 0x9E37_79B9;
#[derive(Clone, Debug, Default)]
pub struct PolicyEvalStats {
    pub episodes: usize,
    pub avg_reward: f64,
    pub up_rate: f64,
    pub forced_up_rate: f64,
    pub avg_pulls: f64,
    pub avg_effective_luck: f64,
    pub avg_abs_effective_luck: f64,
    pub avg_terminal_luck_budget: f64,
    pub action_counts: [usize; ACTION_SPACE],
}

impl PolicyEvalStats {
    pub fn action_distribution(&self) -> [f64; ACTION_SPACE] {
        let total = self.total_actions() as f64;
        if total <= 0.0 {
            return [0.0; ACTION_SPACE];
        }
        let mut out = [0.0; ACTION_SPACE];
        for (dst, &count) in out.iter_mut().zip(self.action_counts.iter()) {
            *dst = count as f64 / total;
        }
        out
    }

    pub fn total_actions(&self) -> usize {
        self.action_counts.iter().sum()
    }

    pub fn positive_action_rate(&self) -> f64 {
        let total = self.total_actions();
        if total == 0 {
            return 0.0;
        }
        (self.action_counts[1] + self.action_counts[2]) as f64 / total as f64
    }

    pub fn negative_action_rate(&self) -> f64 {
        let total = self.total_actions();
        if total == 0 {
            return 0.0;
        }
        (self.action_counts[3] + self.action_counts[4]) as f64 / total as f64
    }
}

pub fn format_policy_eval(label: &str, step: usize, stats: &PolicyEvalStats) -> String {
    let dist = stats.action_distribution();
    format!(
        "[Eval:{label}] Step {step}: episodes={} AvgR={:.3} UP={:.1}% ForcedUP={:.1}% AvgPulls={:.1} EffLuck={:+.5}/{:.5} EndBudget={:.4} Pos/Neg={:.1}%/{:.1}% Actions=[0:{:.1}%, +.005:{:.1}%, +.015:{:.1}%, -.005:{:.1}%, -.015:{:.1}%]",
        stats.episodes,
        stats.avg_reward,
        stats.up_rate * 100.0,
        stats.forced_up_rate * 100.0,
        stats.avg_pulls,
        stats.avg_effective_luck,
        stats.avg_abs_effective_luck,
        stats.avg_terminal_luck_budget,
        stats.positive_action_rate() * 100.0,
        stats.negative_action_rate() * 100.0,
        dist[0] * 100.0,
        dist[1] * 100.0,
        dist[2] * 100.0,
        dist[3] * 100.0,
        dist[4] * 100.0
    )
}

pub fn evaluate_dqn_policy(
    policy: &DuelingQNetwork,
    env_net: &EnvNet,
    config: &Config,
    episodes: usize,
    seed: u64,
) -> PolicyEvalStats {
    let mut selector = DqnEvalSelector { policy };
    evaluate_policy(
        env_net,
        config,
        episodes,
        seed,
        &mut selector,
        |outcome, state, _, config| {
            compute_reward_dqn(
                outcome.rarity == 6,
                outcome.is_up,
                state.loss_streak,
                outcome.luck_modifier,
                config.luck_action_cost,
            )
        },
    )
}

pub fn evaluate_ppo_policy(
    policy: &ActorCritic,
    env_net: &EnvNet,
    config: &Config,
    context_len: usize,
    episodes: usize,
    seed: u64,
) -> PolicyEvalStats {
    let mut selector = PpoEvalSelector::new(policy, context_len);
    evaluate_policy(
        env_net,
        config,
        episodes,
        seed,
        &mut selector,
        |outcome, state, pulls_done, config| {
            let mut reward = compute_reward_ppo(
                outcome.rarity == 6,
                outcome.is_up,
                state.loss_streak,
                outcome.luck_modifier,
                config.luck_action_cost,
            );
            if outcome.rarity == 6 && outcome.is_up {
                if pulls_done < EARLY_UP_BONUS_THRESHOLD_1 {
                    reward += 5.0;
                }
                if pulls_done < EARLY_UP_BONUS_THRESHOLD_2 {
                    reward += 5.0;
                }
            }
            reward
        },
    )
}

trait EvalActionSelector {
    fn reset_episode(&mut self);
    fn select_action(&mut self, features: &[f64; DIM], pulls_done: usize) -> (usize, f64);
}

struct DqnEvalSelector<'a> {
    policy: &'a DuelingQNetwork,
}

impl EvalActionSelector for DqnEvalSelector<'_> {
    fn reset_episode(&mut self) {}

    fn select_action(&mut self, features: &[f64; DIM], _pulls_done: usize) -> (usize, f64) {
        let current_state_tensor = Tensor::new_f32(features.to_vec(), vec![1, DIM]);
        #[cfg(cuda)]
        let current_state_tensor = match current_state_tensor.to_cuda() {
            Ok(t) => t,
            Err(_) => current_state_tensor,
        };
        let (action, requested_luck_modifier) = self.policy.predict_action(&current_state_tensor);
        checked_policy_action(action, requested_luck_modifier as f64)
    }
}

struct PpoEvalSelector<'a> {
    policy: &'a ActorCritic,
    context_len: usize,
    mla_cfg: MLAConfig,
    num_layers: usize,
    history_buffer: VecDeque<Vec<f64>>,
    kv_cache: Vec<KVCache>,
}

impl<'a> PpoEvalSelector<'a> {
    fn new(policy: &'a ActorCritic, context_len: usize) -> Self {
        let context_len = context_len.max(1);
        let mla_cfg = policy
            .backbone
            .blocks
            .first()
            .expect("ActorCritic backbone should have at least one block")
            .mla_layer
            .config
            .clone();
        let num_layers = policy.backbone.blocks.len();
        let mut selector = Self {
            policy,
            context_len,
            mla_cfg,
            num_layers,
            history_buffer: VecDeque::with_capacity(context_len),
            kv_cache: Vec::new(),
        };
        selector.reset_episode();
        selector
    }

    fn new_kv_cache(&self) -> Vec<KVCache> {
        let mut kv_cache: Vec<_> = (0..self.num_layers)
            .map(|_| KVCache::new(self.mla_cfg.num_heads))
            .collect();
        for cache in &mut kv_cache {
            cache.preallocate(
                self.mla_cfg.num_heads,
                self.mla_cfg.kv_lora_rank,
                self.mla_cfg.v_head_dim,
                self.mla_cfg.qk_rope_dim,
                self.mla_cfg.max_seq_len,
            );
        }
        kv_cache
    }
}

impl EvalActionSelector for PpoEvalSelector<'_> {
    fn reset_episode(&mut self) {
        self.history_buffer.clear();
        self.kv_cache = self.new_kv_cache();
    }

    fn select_action(&mut self, features: &[f64; DIM], _pulls_done: usize) -> (usize, f64) {
        self.history_buffer.push_back(features.to_vec());
        if self.history_buffer.len() > self.context_len {
            self.history_buffer.pop_front();
            self.policy
                .prune_cache(&mut self.kv_cache, self.context_len);
        }

        let seq_len = self.history_buffer.len();
        let token = self
            .history_buffer
            .back()
            .expect("history_buffer should not be empty after push");
        let token_f32: Vec<f32> = token.iter().map(|&v| v as f32).collect();
        let action =
            self.policy
                .step_inference_cached_greedy(&token_f32, &mut self.kv_cache, seq_len - 1);
        checked_policy_action(action, ACTIONS.get(action).copied().unwrap_or(0.0))
    }
}

fn evaluate_policy<S, R>(
    env_net: &EnvNet,
    config: &Config,
    episodes: usize,
    seed: u64,
    selector: &mut S,
    mut reward_fn: R,
) -> PolicyEvalStats
where
    S: EvalActionSelector,
    R: FnMut(&GachaStep, &PullState, usize, &Config) -> f64,
{
    let episodes = episodes.max(1);
    let mut accumulator = PolicyEvalAccumulator::default();

    for episode in 0..episodes {
        let mut rng =
            Rng::from_seed(seed.wrapping_add((episode as u64).wrapping_mul(EVAL_SEED_STRIDE)));
        let mut state = PullState::new(config);
        let (env_noise, env_bias) = env_net_env(env_net, &mut rng, 0, 0, 0, 0);
        let mut pulls_done = 0usize;
        let mut episode_reward = 0.0;
        selector.reset_episode();

        loop {
            let features = build_features_with_luck_budget(
                state.pity_6,
                pulls_done,
                env_noise,
                state.streak_4_star,
                env_bias,
                state.loss_streak,
                state.luck_budget,
                config,
            );
            let (action, requested_luck) = selector.select_action(&features, pulls_done);
            let outcome = step_pull(
                &mut state,
                &mut rng,
                config,
                config.big_pity_requires_not_up,
                GachaAction::policy(action, requested_luck),
            );
            pulls_done += 1;

            let reward = reward_fn(&outcome, &state, pulls_done, config);
            episode_reward += reward;
            accumulator.record_step(action, &outcome);

            if outcome.is_up || pulls_done >= EPISODE_MAX_PULLS {
                accumulator.finish_episode(&state, pulls_done, episode_reward, &outcome);
                break;
            }
        }
    }

    accumulator.finish(episodes)
}

#[derive(Default)]
struct PolicyEvalAccumulator {
    total_reward: f64,
    total_pulls: usize,
    up_hits: usize,
    forced_up_hits: usize,
    total_effective_luck: f64,
    total_abs_effective_luck: f64,
    total_terminal_luck_budget: f64,
    action_counts: [usize; ACTION_SPACE],
}

impl PolicyEvalAccumulator {
    fn record_step(&mut self, action: usize, outcome: &GachaStep) {
        if let Some(count) = self.action_counts.get_mut(action) {
            *count += 1;
        }
        self.total_effective_luck += outcome.luck_modifier;
        self.total_abs_effective_luck += outcome.luck_modifier.abs();
    }

    fn finish_episode(
        &mut self,
        state: &PullState,
        pulls_done: usize,
        episode_reward: f64,
        outcome: &GachaStep,
    ) {
        if outcome.is_up {
            self.up_hits += 1;
            if outcome.big_pity_used {
                self.forced_up_hits += 1;
            }
        }
        self.total_pulls += pulls_done;
        self.total_reward += episode_reward;
        self.total_terminal_luck_budget += state.luck_budget;
    }

    fn finish(self, episodes: usize) -> PolicyEvalStats {
        let total_actions = self.action_counts.iter().sum::<usize>().max(1) as f64;
        PolicyEvalStats {
            episodes,
            avg_reward: self.total_reward / episodes as f64,
            up_rate: self.up_hits as f64 / episodes as f64,
            forced_up_rate: self.forced_up_hits as f64 / episodes as f64,
            avg_pulls: self.total_pulls as f64 / episodes as f64,
            avg_effective_luck: self.total_effective_luck / total_actions,
            avg_abs_effective_luck: self.total_abs_effective_luck / total_actions,
            avg_terminal_luck_budget: self.total_terminal_luck_budget / episodes as f64,
            action_counts: self.action_counts,
        }
    }
}

fn checked_policy_action(action: usize, requested_luck: f64) -> (usize, f64) {
    if action < ACTION_SPACE && requested_luck.is_finite() {
        return (action, requested_luck);
    }
    (0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_eval_distribution_is_zero_without_actions() {
        let stats = PolicyEvalStats::default();

        assert_eq!(stats.action_distribution(), [0.0; ACTION_SPACE]);
        assert_eq!(stats.positive_action_rate(), 0.0);
        assert_eq!(stats.negative_action_rate(), 0.0);
    }

    #[test]
    fn policy_eval_distribution_and_rates_are_normalized() {
        let stats = PolicyEvalStats {
            action_counts: [2, 1, 1, 3, 3],
            ..PolicyEvalStats::default()
        };

        let dist = stats.action_distribution();

        assert!((dist[0] - 0.2).abs() < f64::EPSILON);
        assert!((stats.positive_action_rate() - 0.2).abs() < f64::EPSILON);
        assert!((stats.negative_action_rate() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn policy_eval_format_includes_quality_signals() {
        let stats = PolicyEvalStats {
            episodes: 4,
            avg_reward: 1.25,
            up_rate: 0.5,
            forced_up_rate: 0.25,
            avg_pulls: 72.0,
            avg_effective_luck: 0.001,
            avg_abs_effective_luck: 0.002,
            avg_terminal_luck_budget: 0.01,
            action_counts: [1, 1, 1, 1, 0],
        };

        let line = format_policy_eval("DQN", 100, &stats);

        assert!(line.contains("[Eval:DQN] Step 100"));
        assert!(line.contains("ForcedUP=25.0%"));
        assert!(line.contains("EffLuck=+0.00100/0.00200"));
        assert!(line.contains("Pos/Neg=50.0%/25.0%"));
    }

    #[test]
    fn invalid_policy_action_falls_back_to_neutral() {
        assert_eq!(checked_policy_action(ACTION_SPACE, 0.01), (0, 0.0));
        assert_eq!(checked_policy_action(1, f64::NAN), (0, 0.0));
    }
}
