use crate::autograd::Tensor as AutoTensor;
use crate::config::{Config, LuckMode};
use crate::dqn::DuelingQNetwork;
use crate::neural::{NeuralLuckOptimizer, Tensor, DIM};
use crate::ppo::ActorCritic;
use crate::sim::PullState;
use crate::transformer::KVCache;
use crate::utils::ACTIONS;

/// Policy output consumed by the gacha simulation loop.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StrategyDecision {
    pub(crate) luck_factor: f64,
    pub(crate) action: Option<usize>,
    pub(crate) ppo_log_prob: Option<f64>,
    pub(crate) ppo_value: Option<f64>,
}

/// Inputs needed for a single strategy decision.
pub(crate) struct StrategyInputs<'a> {
    pub(crate) state: &'a PullState,
    pub(crate) nn_total_pulls: usize,
    pub(crate) config: &'a Config,
    pub(crate) neural_opt: &'a NeuralLuckOptimizer,
    pub(crate) dqn_policy: Option<&'a DuelingQNetwork>,
    pub(crate) ppo_policy: Option<&'a ActorCritic>,
    pub(crate) current_features: &'a Tensor,
    pub(crate) ppo_state_seq: Option<&'a AutoTensor>,
    pub(crate) ppo_pity_seq: Option<&'a [usize]>,
    pub(crate) fast_inference: bool,
    pub(crate) ppo_seq_data: Option<&'a [f64]>,
    pub(crate) kv_cache: &'a mut Option<Vec<KVCache>>,
    pub(crate) start_pos: usize,
}

fn features_to_f32(features: &Tensor) -> [f32; DIM] {
    let mut out = [0.0; DIM];
    for (dst, src) in out.iter_mut().zip(features.iter()) {
        *dst = *src as f32;
    }
    out
}

fn ppo_sequence_to_f32(seq_data: &[f64]) -> Vec<f32> {
    seq_data.iter().map(|&v| v as f32).collect()
}

fn decide_ppo_slow(
    policy: &ActorCritic,
    state: &PullState,
    config: &Config,
    current_features: &Tensor,
    ppo_state_seq: Option<&AutoTensor>,
    ppo_pity_seq: Option<&[usize]>,
) -> StrategyDecision {
    let (idx, log_prob, value) = if let (Some(seq), Some(pities)) = (ppo_state_seq, ppo_pity_seq) {
        policy.step(seq, pities, config.ppo_top_k)
    } else {
        let tensor_x = AutoTensor::new(current_features.to_vec(), vec![DIM]);
        policy.step(&tensor_x, &[state.pity_6], config.ppo_top_k)
    };
    StrategyDecision {
        luck_factor: ACTIONS[idx],
        action: Some(idx),
        ppo_log_prob: Some(log_prob),
        ppo_value: Some(value),
    }
}

pub(crate) fn decide(inputs: StrategyInputs<'_>) -> StrategyDecision {
    let StrategyInputs {
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
                let features_f32 = features_to_f32(current_features);
                let (idx, modifier) = policy.predict_action_fast(&features_f32);
                return StrategyDecision {
                    luck_factor: modifier as f64,
                    action: Some(idx),
                    ppo_log_prob: None,
                    ppo_value: None,
                };
            }
            let tensor_x = AutoTensor::new(current_features.to_vec(), vec![DIM]);
            let (idx, modifier) = policy.predict_action(&tensor_x);
            return StrategyDecision {
                luck_factor: modifier as f64,
                action: Some(idx),
                ppo_log_prob: None,
                ppo_value: None,
            };
        }
    } else if config.luck_mode == LuckMode::Ppo {
        if let Some(policy) = ppo_policy {
            if fast_inference {
                if let Some(cache) = kv_cache {
                    let features_f32 = features_to_f32(current_features);
                    let idx = policy.step_inference_cached(
                        &features_f32,
                        cache,
                        start_pos,
                        config.ppo_top_k,
                    );
                    return StrategyDecision {
                        luck_factor: ACTIONS[idx],
                        action: Some(idx),
                        ppo_log_prob: None,
                        ppo_value: None,
                    };
                }
                if let Some(seq_data) = ppo_seq_data {
                    let seq_f32 = ppo_sequence_to_f32(seq_data);
                    let idx = policy.step_inference(&seq_f32, config.ppo_top_k);
                    return StrategyDecision {
                        luck_factor: ACTIONS[idx],
                        action: Some(idx),
                        ppo_log_prob: None,
                        ppo_value: None,
                    };
                }
                return decide_ppo_slow(
                    policy,
                    state,
                    config,
                    current_features,
                    ppo_state_seq,
                    ppo_pity_seq,
                );
            }
            return decide_ppo_slow(
                policy,
                state,
                config,
                current_features,
                ppo_state_seq,
                ppo_pity_seq,
            );
        }
    }

    let dropout_seed = (state.pity_6 as u64)
        .wrapping_add((nn_total_pulls as u64).wrapping_mul(31))
        .wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
    let luck_factor = neural_opt.predict(current_features, dropout_seed);
    StrategyDecision {
        luck_factor,
        action: None,
        ppo_log_prob: None,
        ppo_value: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::ACTION_SPACE;

    fn test_state() -> PullState {
        PullState {
            pity_6: 4,
            total_pulls_in_pool: 9,
            has_obtained_up: false,
            streak_4_star: 2,
            loss_streak: 1,
            ..PullState::new(&Config::default())
        }
    }

    #[test]
    fn neural_fallback_uses_existing_dropout_seed_formula() {
        let config = Config {
            luck_mode: LuckMode::Probability,
            ..Default::default()
        };
        let neural_opt = NeuralLuckOptimizer::new(7);
        let state = test_state();
        let nn_total_pulls = 11;
        let mut features = [0.0; DIM];
        for (i, feature) in features.iter_mut().enumerate() {
            *feature = i as f64 * 0.001;
        }
        let mut kv_cache = None;

        let decision = decide(StrategyInputs {
            state: &state,
            nn_total_pulls,
            config: &config,
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            current_features: &features,
            ppo_state_seq: None,
            ppo_pity_seq: None,
            fast_inference: true,
            ppo_seq_data: None,
            kv_cache: &mut kv_cache,
            start_pos: 0,
        });

        let dropout_seed = (state.pity_6 as u64)
            .wrapping_add((nn_total_pulls as u64).wrapping_mul(31))
            .wrapping_add((state.streak_4_star as u64).wrapping_mul(17));
        let expected = neural_opt.predict(&features, dropout_seed);

        assert!((decision.luck_factor - expected).abs() < 1e-12);
        assert_eq!(decision.action, None);
        assert_eq!(decision.ppo_log_prob, None);
        assert_eq!(decision.ppo_value, None);
    }

    #[test]
    fn dqn_slow_path_returns_discrete_luck_action() {
        let config = Config {
            luck_mode: LuckMode::Dqn,
            model_hidden_dim: 8,
            achf: crate::config::AchfConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let neural_opt = NeuralLuckOptimizer::new(7);
        let dqn = DuelingQNetwork::new_with_config(&config, 13);
        let state = test_state();
        let features = [0.01; DIM];
        let mut kv_cache = None;

        let decision = decide(StrategyInputs {
            state: &state,
            nn_total_pulls: 5,
            config: &config,
            neural_opt: &neural_opt,
            dqn_policy: Some(&dqn),
            ppo_policy: None,
            current_features: &features,
            ppo_state_seq: None,
            ppo_pity_seq: None,
            fast_inference: false,
            ppo_seq_data: None,
            kv_cache: &mut kv_cache,
            start_pos: 0,
        });

        let action = decision.action.expect("DQN should choose an action");
        assert!(action < ACTION_SPACE);
        assert!((decision.luck_factor - ACTIONS[action]).abs() < 1e-9);
        assert_eq!(decision.ppo_log_prob, None);
        assert_eq!(decision.ppo_value, None);
    }

    #[test]
    fn ppo_slow_path_returns_action_log_prob_and_value() {
        let config = Config {
            luck_mode: LuckMode::Ppo,
            ppo_top_k: 0,
            ..Default::default()
        };
        let neural_opt = NeuralLuckOptimizer::new(7);
        let policy = ActorCritic::new(17, &crate::config::AchfConfig::default(), 64, 2);
        let state = test_state();
        let features = [0.02; DIM];
        let seq = AutoTensor::new(features.to_vec(), vec![DIM]);
        let pity = [state.pity_6];
        let mut kv_cache = None;

        let decision = decide(StrategyInputs {
            state: &state,
            nn_total_pulls: 5,
            config: &config,
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: Some(&policy),
            current_features: &features,
            ppo_state_seq: Some(&seq),
            ppo_pity_seq: Some(&pity),
            fast_inference: false,
            ppo_seq_data: None,
            kv_cache: &mut kv_cache,
            start_pos: 0,
        });

        let action = decision.action.expect("PPO should choose an action");
        assert!(action < ACTION_SPACE);
        assert_eq!(decision.luck_factor, ACTIONS[action]);
        assert!(decision.ppo_log_prob.expect("log_prob").is_finite());
        assert!(decision.ppo_value.expect("value").is_finite());
    }

    #[test]
    fn ppo_fast_missing_sequence_falls_back_to_slow_path() {
        let config = Config {
            luck_mode: LuckMode::Ppo,
            ppo_top_k: 0,
            ..Default::default()
        };
        let neural_opt = NeuralLuckOptimizer::new(7);
        let policy = ActorCritic::new(19, &crate::config::AchfConfig::default(), 64, 2);
        let state = test_state();
        let features = [0.03; DIM];
        let mut kv_cache = None;

        let decision = decide(StrategyInputs {
            state: &state,
            nn_total_pulls: 5,
            config: &config,
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: Some(&policy),
            current_features: &features,
            ppo_state_seq: None,
            ppo_pity_seq: None,
            fast_inference: true,
            ppo_seq_data: None,
            kv_cache: &mut kv_cache,
            start_pos: 0,
        });

        assert!(decision.action.expect("PPO should choose an action") < ACTION_SPACE);
        assert!(decision.ppo_log_prob.expect("log_prob").is_finite());
        assert!(decision.ppo_value.expect("value").is_finite());
    }
}
