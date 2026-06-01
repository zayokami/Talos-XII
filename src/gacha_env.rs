use crate::config::Config;
use crate::rng::Rng;
use crate::sim::{prob_6, PullState};
use crate::utils::apply_luck_budget;

#[derive(Clone, Copy, Debug)]
pub struct GachaAction {
    pub action: Option<usize>,
    pub requested_luck: f64,
    pub ppo_log_prob: Option<f64>,
    pub ppo_value: Option<f64>,
}

impl GachaAction {
    pub fn none() -> Self {
        Self {
            action: None,
            requested_luck: 0.0,
            ppo_log_prob: None,
            ppo_value: None,
        }
    }

    pub fn policy(action: usize, requested_luck: f64) -> Self {
        Self {
            action: Some(action),
            requested_luck,
            ppo_log_prob: None,
            ppo_value: None,
        }
    }

    pub fn ppo(action: usize, requested_luck: f64, log_prob: f64, value: f64) -> Self {
        Self {
            action: Some(action),
            requested_luck,
            ppo_log_prob: Some(log_prob),
            ppo_value: Some(value),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GachaStep {
    pub rarity: u8,
    pub is_up: bool,
    pub big_pity_used: bool,
    pub action: Option<usize>,
    pub luck_modifier: f64,
    pub ppo_log_prob: Option<f64>,
    pub ppo_value: Option<f64>,
}

pub fn force_up_pity_triggered(
    state: &PullState,
    config: &Config,
    big_pity_requires_not_up: bool,
) -> bool {
    let next_total_pulls = state.total_pulls_in_pool + 1;
    let big_pity_gate = if big_pity_requires_not_up {
        !state.has_obtained_up
    } else {
        true
    };
    big_pity_gate
        && ((config.up_pity_soft > 0 && next_total_pulls == config.up_pity_soft)
            || (config.big_pity_cumulative > 0 && next_total_pulls == config.big_pity_cumulative))
}

pub fn step_pull(
    state: &mut PullState,
    rng: &mut Rng,
    config: &Config,
    big_pity_requires_not_up: bool,
    action: GachaAction,
) -> GachaStep {
    let current_pity = state.pity_6;
    let current_total_pulls = state.total_pulls_in_pool + 1;
    let mut rarity = 4;
    let mut is_up = false;
    let mut big_pity_used = false;
    let mut luck_modifier = 0.0;

    if force_up_pity_triggered(state, config, big_pity_requires_not_up) {
        rarity = 6;
        is_up = true;
        big_pity_used = true;
        state.pity_6 = 0;
        state.streak_4_star = 0;
        state.loss_streak = 0;
        state.has_obtained_up = true;
        let _ = apply_luck_budget(0.0, &mut state.luck_budget, config);
    } else {
        luck_modifier = apply_luck_budget(action.requested_luck, &mut state.luck_budget, config);
        let final_prob_6 = (prob_6(current_pity, config) + luck_modifier).clamp(0.0, 1.0);
        let r = rng.next_f64();

        if r < final_prob_6 {
            rarity = 6;
            state.pity_6 = 0;
            state.streak_4_star = 0;

            if config.up_rate > 0.0 && !config.up_six.is_empty() {
                if rng.next_f64() < config.up_rate {
                    is_up = true;
                    state.loss_streak = 0;
                    state.has_obtained_up = true;
                } else {
                    state.loss_streak += 1;
                }
            }
        } else {
            state.pity_6 = current_pity + 1;
            let force_5_star = config.always_5_star
                || (config.five_star_pity > 0 && state.streak_4_star >= config.five_star_pity - 1);
            if force_5_star || r < (final_prob_6 + config.prob_5_base).min(1.0) {
                rarity = 5;
                state.streak_4_star = 0;
            } else {
                state.streak_4_star += 1;
            }
        }
    }

    state.total_pulls_in_pool = current_total_pulls;

    GachaStep {
        rarity,
        is_up,
        big_pity_used,
        action: action.action,
        luck_modifier,
        ppo_log_prob: action.ppo_log_prob,
        ppo_value: action.ppo_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_up_pity_advances_total_pulls_without_spending_luck() {
        let config = Config {
            up_pity_soft: 2,
            big_pity_cumulative: 0,
            luck_budget_enabled: true,
            luck_budget_initial: 0.01,
            luck_budget_max: 0.03,
            luck_budget_recovery_per_pull: 0.0,
            ..Config::default()
        };
        let mut state = PullState {
            total_pulls_in_pool: 1,
            luck_budget: 0.01,
            ..PullState::new(&config)
        };
        let mut rng = Rng::from_seed(1);

        let outcome = step_pull(
            &mut state,
            &mut rng,
            &config,
            true,
            GachaAction::policy(4, 0.02),
        );

        assert_eq!(outcome.rarity, 6);
        assert!(outcome.is_up);
        assert!(outcome.big_pity_used);
        assert_eq!(outcome.luck_modifier, 0.0);
        assert_eq!(state.total_pulls_in_pool, 2);
        assert_eq!(state.luck_budget, 0.01);
    }

    #[test]
    fn non_six_pull_increments_pity_and_charges_budgeted_luck() {
        let config = Config {
            prob_6_base: 0.0,
            soft_pity_start: 100,
            small_pity_guarantee: 120,
            prob_5_base: 0.0,
            luck_budget_enabled: true,
            luck_budget_initial: 0.01,
            luck_budget_max: 0.03,
            luck_budget_recovery_per_pull: 0.0,
            ..Config::default()
        };
        let mut state = PullState::new(&config);
        let mut rng = Rng::from_seed(2);

        let outcome = step_pull(
            &mut state,
            &mut rng,
            &config,
            true,
            GachaAction::policy(4, 0.02),
        );

        assert_eq!(outcome.rarity, 4);
        assert_eq!(outcome.luck_modifier, 0.01);
        assert_eq!(state.pity_6, 1);
        assert_eq!(state.total_pulls_in_pool, 1);
        assert_eq!(state.luck_budget, 0.0);
    }
}
