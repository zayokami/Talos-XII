//! Shared utility functions, constants, and SIMD-accelerated math helpers.

use crate::achf::AchfCacheStats;
use crate::config::Config;
use indicatif::{ProgressBar, ProgressStyle};

/// Reward constants for training signal computation.
pub const REWARD_BASE: f64 = -0.1;
pub const REWARD_SIX_UP_DQN: f64 = 10.0;
pub const REWARD_SIX_NON_UP_DQN: f64 = 2.0;
pub const REWARD_SIX_UP_NEURAL: f64 = 1.0;
pub const REWARD_SIX_NON_UP_NEURAL: f64 = 0.2;
pub const REWARD_SIX_UP_PPO: f64 = 10.0;
pub const REWARD_SIX_NON_UP_PPO: f64 = 2.0;
pub const STREAK_PENALTY_DQN: f64 = 0.5;
pub const STREAK_PENALTY_NEURAL: f64 = 0.2;
pub const STREAK_PENALTY_PPO: f64 = 2.0;
pub const STREAK_PENALTY_THRESHOLD: usize = 2;

/// Episode termination limit for DQN/PPO training environments.
pub const EPISODE_MAX_PULLS: usize = 300;

/// Maximum number of online training experience items to drain per tick.
pub const MAX_DRAIN_PER_TICK: usize = 2048;

/// Report interval for online trainers (seconds).
pub const ONLINE_REPORT_INTERVAL_SECS: f64 = 2.0;

/// Default PPO online learning rate.
pub const PPO_ONLINE_LR: f64 = 0.0003;

/// Batch count for F2P simulation progress display.
#[allow(dead_code)]
pub const F2P_BATCH_COUNT: usize = 100;

/// History capacity for interactive simulation history.
pub const SIM_HISTORY_CAPACITY: usize = 20;

/// Maximum display items for pull detail list.
pub const PULL_DISPLAY_LIMIT: usize = 20;

/// Maximum user-input pulls or sims before capping.
pub const INPUT_CAP: usize = 1_000_000;

/// Default PPO context length when not specified.
pub const DEFAULT_PPO_CONTEXT_LEN: usize = 8;

/// Discrete luck-factor actions shared by DQN and PPO.
pub const ACTION_SPACE: usize = 5;
pub const ACTIONS: [f64; ACTION_SPACE] = [0.0, 0.005, 0.015, -0.005, -0.015];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewardBreakdown {
    pub base: f64,
    pub hit: f64,
    pub streak: f64,
    pub luck: f64,
    pub early: f64,
}

impl RewardBreakdown {
    pub fn total(self) -> f64 {
        self.base + self.hit + self.streak + self.luck + self.early
    }

    pub fn add_assign(&mut self, other: Self) {
        self.base += other.base;
        self.hit += other.hit;
        self.streak += other.streak;
        self.luck += other.luck;
        self.early += other.early;
    }

    pub fn add_early_bonus(&mut self, bonus: f64) {
        self.early += bonus;
    }

    pub fn averaged(self, count: usize) -> Self {
        if count == 0 {
            return Self::default();
        }
        let denom = count as f64;
        Self {
            base: self.base / denom,
            hit: self.hit / denom,
            streak: self.streak / denom,
            luck: self.luck / denom,
            early: self.early / denom,
        }
    }

    pub fn average<I>(items: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        let mut total = Self::default();
        let mut count = 0usize;
        for item in items {
            total.add_assign(item);
            count += 1;
        }
        total.averaged(count)
    }

    pub fn format_compact(self) -> String {
        format!(
            "R[base={:+.2} hit={:+.2} streak={:+.2} luck={:+.2} early={:+.2}]",
            self.base, self.hit, self.streak, self.luck, self.early
        )
    }
}

/// Cost applied to policy-controlled luck interventions.
pub fn luck_action_penalty(luck_modifier: f64, action_cost: f64) -> f64 {
    if !luck_modifier.is_finite()
        || !action_cost.is_finite()
        || action_cost <= 0.0
        || luck_modifier <= 0.0
    {
        return 0.0;
    }
    luck_modifier * action_cost
}

pub fn initial_luck_budget(config: &Config) -> f64 {
    if !config.luck_budget_enabled || config.luck_budget_max <= 0.0 {
        return 0.0;
    }
    config
        .luck_budget_initial
        .clamp(0.0, config.luck_budget_max)
}

pub fn luck_budget_ratio(config: &Config, budget: f64) -> f64 {
    if !config.luck_budget_enabled || config.luck_budget_max <= 0.0 || !budget.is_finite() {
        return 1.0;
    }
    (budget / config.luck_budget_max).clamp(0.0, 1.0)
}

pub fn apply_luck_budget(requested_modifier: f64, budget: &mut f64, config: &Config) -> f64 {
    if !config.luck_budget_enabled || config.luck_budget_max <= 0.0 {
        return requested_modifier;
    }

    if !budget.is_finite() {
        *budget = initial_luck_budget(config);
    }
    *budget = budget.clamp(0.0, config.luck_budget_max);

    let actual = if requested_modifier > 0.0 {
        let granted = requested_modifier.min(*budget);
        *budget -= granted;
        granted
    } else if requested_modifier < 0.0 {
        let refund = (-requested_modifier) * config.luck_budget_negative_refund;
        *budget = (*budget + refund).min(config.luck_budget_max);
        requested_modifier
    } else {
        0.0
    };

    if config.luck_budget_recovery_per_pull > 0.0 {
        *budget = (*budget + config.luck_budget_recovery_per_pull).min(config.luck_budget_max);
    }
    actual
}

#[derive(Clone, Copy)]
struct RewardCoefficients {
    up_reward: f64,
    non_up_reward: f64,
    streak_penalty: f64,
}

fn compute_reward_breakdown(
    is_six: bool,
    is_up: bool,
    loss_streak: usize,
    luck_modifier: f64,
    luck_action_cost: f64,
    coeffs: RewardCoefficients,
) -> RewardBreakdown {
    let hit = if is_six {
        if is_up {
            coeffs.up_reward
        } else {
            coeffs.non_up_reward
        }
    } else {
        0.0
    };
    let streak = if is_six && !is_up && loss_streak >= STREAK_PENALTY_THRESHOLD {
        -(loss_streak as f64) * coeffs.streak_penalty
    } else {
        0.0
    };
    let luck = -luck_action_penalty(luck_modifier, luck_action_cost);
    RewardBreakdown {
        base: REWARD_BASE,
        hit,
        streak,
        luck,
        early: 0.0,
    }
}

pub fn compute_reward_dqn_breakdown(
    is_six: bool,
    is_up: bool,
    loss_streak: usize,
    luck_modifier: f64,
    luck_action_cost: f64,
) -> RewardBreakdown {
    compute_reward_breakdown(
        is_six,
        is_up,
        loss_streak,
        luck_modifier,
        luck_action_cost,
        RewardCoefficients {
            up_reward: REWARD_SIX_UP_DQN,
            non_up_reward: REWARD_SIX_NON_UP_DQN,
            streak_penalty: STREAK_PENALTY_DQN,
        },
    )
}

/// Compute DQN-style reward for experience replay.
pub fn compute_reward_dqn(
    is_six: bool,
    is_up: bool,
    loss_streak: usize,
    luck_modifier: f64,
    luck_action_cost: f64,
) -> f64 {
    compute_reward_dqn_breakdown(is_six, is_up, loss_streak, luck_modifier, luck_action_cost)
        .total()
}

pub fn compute_reward_neural_breakdown(
    is_six: bool,
    is_up: bool,
    loss_streak: usize,
) -> RewardBreakdown {
    compute_reward_breakdown(
        is_six,
        is_up,
        loss_streak,
        0.0,
        0.0,
        RewardCoefficients {
            up_reward: REWARD_SIX_UP_NEURAL,
            non_up_reward: REWARD_SIX_NON_UP_NEURAL,
            streak_penalty: STREAK_PENALTY_NEURAL,
        },
    )
}

/// Compute Neural-style reward for online neural training.
pub fn compute_reward_neural(is_six: bool, is_up: bool, loss_streak: usize) -> f64 {
    compute_reward_neural_breakdown(is_six, is_up, loss_streak).total()
}

pub fn compute_reward_ppo_breakdown(
    is_six: bool,
    is_up: bool,
    loss_streak: usize,
    luck_modifier: f64,
    luck_action_cost: f64,
) -> RewardBreakdown {
    compute_reward_breakdown(
        is_six,
        is_up,
        loss_streak,
        luck_modifier,
        luck_action_cost,
        RewardCoefficients {
            up_reward: REWARD_SIX_UP_PPO,
            non_up_reward: REWARD_SIX_NON_UP_PPO,
            streak_penalty: STREAK_PENALTY_PPO,
        },
    )
}

/// Compute PPO-style reward for PPO experience replay.
pub fn compute_reward_ppo(
    is_six: bool,
    is_up: bool,
    loss_streak: usize,
    luck_modifier: f64,
    luck_action_cost: f64,
) -> f64 {
    compute_reward_ppo_breakdown(is_six, is_up, loss_streak, luck_modifier, luck_action_cost)
        .total()
}

/// Format ACHF cache statistics into a human-readable summary string.
pub fn format_achf_stats(stats: &AchfCacheStats) -> String {
    let calls = stats.calls as f64;
    let hit_rate = if calls > 0.0 {
        stats.cache_hits as f64 / calls
    } else {
        0.0
    };
    format!(
        "[ACHF] Calls: {} | Hit: {:.2}% | Miss: {} | Skip: {} | Sparse: {} | Dense: {} | CachedEMA(ns): {:.1}/{:.1} | SparseEMA(ns): {:.1}/{:.1} | DecisionEMA(ns): {:.1}/{:.1} | Bias: {:.3} | Samples: {}/{}",
        stats.calls,
        hit_rate * 100.0,
        stats.cache_misses,
        stats.cache_skips,
        stats.sparse_paths,
        stats.dense_paths,
        stats.ema_cached_ns,
        stats.ema_cached_long_ns,
        stats.ema_sparse_ns,
        stats.ema_sparse_long_ns,
        stats.decision_ema_ns,
        stats.decision_ema_long_ns,
        stats.adaptive_bias,
        stats.latency_samples,
        stats.decision_samples
    )
}

// ── SIMD-accelerated math helpers ────────────────────────────────────────

/// Sum all elements in a slice, using SIMD when available.
#[inline(always)]
pub fn sum_f64(values: &[f64]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        sum_f64_neon(values)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    return sum_f64_avx2(values);
                }
            }
        }
        let mut sum = 0.0;
        for &v in values {
            sum += v;
        }
        sum
    }
}

/// Sum of squared differences from a mean, using SIMD when available.
#[inline(always)]
pub fn sum_sq_diff(values: &[f64], mean: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        sum_sq_diff_neon(values, mean)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    return sum_sq_diff_avx2(values, mean);
                }
            }
        }
        let mut sum = 0.0;
        for &v in values {
            let d = v - mean;
            sum += d * d;
        }
        sum
    }
}

/// Normalize a slice to zero mean / unit std, using SIMD when available.
#[inline(always)]
pub fn normalize_slice(values: &[f64], mean: f64, std: f64) -> Vec<f64> {
    let len = values.len();
    let mut out = vec![0.0; len];
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            normalize_slice_neon(values, &mut out, mean, std);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    normalize_slice_avx2(values, &mut out, mean, std);
                }
                return out;
            }
        }
        for i in 0..len {
            out[i] = (values[i] - mean) / std;
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f64_avx2(values: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = _mm256_setzero_pd();
    while i + 4 <= len {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        acc = _mm256_add_pd(acc, v);
        i += 4;
    }
    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < len {
        sum += *values.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_sq_diff_avx2(values: &[f64], mean: f64) -> f64 {
    use core::arch::x86_64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = _mm256_setzero_pd();
    let mean_vec = _mm256_set1_pd(mean);
    while i + 4 <= len {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let d = _mm256_sub_pd(v, mean_vec);
        let prod = _mm256_mul_pd(d, d);
        acc = _mm256_add_pd(acc, prod);
        i += 4;
    }
    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < len {
        let d = *values.get_unchecked(i) - mean;
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_slice_avx2(values: &[f64], out: &mut [f64], mean: f64, std: f64) {
    use core::arch::x86_64::*;
    let mut i = 0;
    let len = values.len();
    let mean_vec = _mm256_set1_pd(mean);
    let std_vec = _mm256_set1_pd(std);
    while i + 4 <= len {
        let v = _mm256_loadu_pd(values.as_ptr().add(i));
        let d = _mm256_sub_pd(v, mean_vec);
        let n = _mm256_div_pd(d, std_vec);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), n);
        i += 4;
    }
    while i < len {
        *out.get_unchecked_mut(i) = (*values.get_unchecked(i) - mean) / std;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luck_action_penalty_charges_nonzero_policy_actions() {
        assert_eq!(luck_action_penalty(0.0, 8.0), 0.0);
        assert!((luck_action_penalty(0.015, 8.0) - 0.12).abs() < 1e-12);
        assert_eq!(luck_action_penalty(-0.005, 8.0), 0.0);
    }

    #[test]
    fn policy_reward_accounts_for_luck_action_cost() {
        let free = compute_reward_dqn(false, false, 0, 0.0, 8.0);
        let boosted = compute_reward_dqn(false, false, 0, 0.015, 8.0);
        assert!((free - boosted - 0.12).abs() < 1e-12);
    }

    #[test]
    fn reward_breakdown_totals_match_scalar_helpers() {
        let dqn = compute_reward_dqn_breakdown(true, false, 3, 0.015, 8.0);
        let neural = compute_reward_neural_breakdown(true, false, 3);
        let ppo = compute_reward_ppo_breakdown(true, true, 0, 0.0, 8.0);

        assert!((dqn.total() - compute_reward_dqn(true, false, 3, 0.015, 8.0)).abs() < 1e-12);
        assert!((neural.total() - compute_reward_neural(true, false, 3)).abs() < 1e-12);
        assert!((ppo.total() - compute_reward_ppo(true, true, 0, 0.0, 8.0)).abs() < 1e-12);
    }

    #[test]
    fn policy_reward_does_not_repeat_loss_streak_penalty_on_non_six_pulls() {
        assert!((compute_reward_dqn(false, false, 3, 0.0, 8.0) - REWARD_BASE).abs() < 1e-12);
        assert!((compute_reward_neural(false, false, 3) - REWARD_BASE).abs() < 1e-12);
        assert!((compute_reward_ppo(false, false, 3, 0.0, 8.0) - REWARD_BASE).abs() < 1e-12);
    }

    #[test]
    fn policy_reward_applies_loss_streak_penalty_to_non_up_six() {
        let loss_streak = 3;
        assert!(
            (compute_reward_dqn(true, false, loss_streak, 0.0, 8.0)
                - (REWARD_BASE + REWARD_SIX_NON_UP_DQN - loss_streak as f64 * STREAK_PENALTY_DQN))
                .abs()
                < 1e-12
        );
        assert!(
            (compute_reward_neural(true, false, loss_streak)
                - (REWARD_BASE + REWARD_SIX_NON_UP_NEURAL
                    - loss_streak as f64 * STREAK_PENALTY_NEURAL))
                .abs()
                < 1e-12
        );
        assert!(
            (compute_reward_ppo(true, false, loss_streak, 0.0, 8.0)
                - (REWARD_BASE + REWARD_SIX_NON_UP_PPO - loss_streak as f64 * STREAK_PENALTY_PPO))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn luck_budget_caps_positive_actions_and_refunds_negative_actions() {
        let config = Config {
            luck_budget_enabled: true,
            luck_budget_max: 0.03,
            luck_budget_initial: 0.01,
            luck_budget_recovery_per_pull: 0.0,
            luck_budget_negative_refund: 1.0,
            ..Config::default()
        };
        let mut budget = initial_luck_budget(&config);
        assert!((apply_luck_budget(0.015, &mut budget, &config) - 0.01).abs() < 1e-12);
        assert_eq!(budget, 0.0);
        assert!((apply_luck_budget(-0.005, &mut budget, &config) + 0.005).abs() < 1e-12);
        assert!((budget - 0.005).abs() < 1e-12);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_f64_neon(values: &[f64]) -> f64 {
    use core::arch::aarch64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = vdupq_n_f64(0.0);
    while i + 2 <= len {
        let v = vld1q_f64(values.as_ptr().add(i));
        acc = vaddq_f64(acc, v);
        i += 2;
    }
    let mut tmp = [0.0; 2];
    vst1q_f64(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1];
    while i < len {
        sum += *values.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_sq_diff_neon(values: &[f64], mean: f64) -> f64 {
    use core::arch::aarch64::*;
    let mut i = 0;
    let len = values.len();
    let mut acc = vdupq_n_f64(0.0);
    let mean_vec = vdupq_n_f64(mean);
    while i + 2 <= len {
        let v = vld1q_f64(values.as_ptr().add(i));
        let d = vsubq_f64(v, mean_vec);
        let prod = vmulq_f64(d, d);
        acc = vaddq_f64(acc, prod);
        i += 2;
    }
    let mut tmp = [0.0; 2];
    vst1q_f64(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1];
    while i < len {
        let d = *values.get_unchecked(i) - mean;
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_slice_neon(values: &[f64], out: &mut [f64], mean: f64, std: f64) {
    use core::arch::aarch64::*;
    let mut i = 0;
    let len = values.len();
    let mean_vec = vdupq_n_f64(mean);
    let std_vec = vdupq_n_f64(std);
    while i + 2 <= len {
        let v = vld1q_f64(values.as_ptr().add(i));
        let d = vsubq_f64(v, mean_vec);
        let n = vdivq_f64(d, std_vec);
        vst1q_f64(out.as_mut_ptr().add(i), n);
        i += 2;
    }
    while i < len {
        *out.get_unchecked_mut(i) = (*values.get_unchecked(i) - mean) / std;
        i += 1;
    }
}

// ── Progress bar helper ─────────────────────────────────────────────────

/// Create a styled progress bar with the given total and prefix message.
/// Enables a background steady tick so the spinner and elapsed time keep updating
/// even when the main thread is blocked by heavy computation.
pub fn create_bar(total: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg} [{elapsed_precise}]")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(200));
    pb
}

/// Create a spinner (unknown total) with a message.
#[allow(dead_code)]
pub fn create_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb
}

// ── Unicode table renderer ──────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum Align {
    Left,
    Right,
}

pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    alignments: Vec<Align>,
}

impl Table {
    pub fn new(headers: &[&str]) -> Self {
        let n = headers.len();
        Self {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
            alignments: vec![Align::Left; n],
        }
    }

    #[allow(dead_code)]
    pub fn align(mut self, col: usize, a: Align) -> Self {
        if col < self.alignments.len() {
            self.alignments[col] = a;
        }
        self
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    pub fn render(&self) -> String {
        let cols = self.headers.len();
        let mut widths = vec![0usize; cols];
        for (i, h) in self.headers.iter().enumerate() {
            widths[i] = display_width(h);
        }
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < cols {
                    widths[i] = widths[i].max(display_width(cell));
                }
            }
        }

        let mut out = String::new();
        render_border(&mut out, &widths, '┌', '┬', '┐');
        render_row(&mut out, &self.headers, &widths, &self.alignments);
        render_border(&mut out, &widths, '├', '┼', '┤');
        for row in &self.rows {
            render_data_row(&mut out, row, &widths, &self.alignments, cols);
        }
        render_border_no_newline(&mut out, &widths, '└', '┴', '┘');
        out
    }
}

fn render_border(out: &mut String, widths: &[usize], left: char, mid: char, right: char) {
    out.push(left);
    for (i, &w) in widths.iter().enumerate() {
        for _ in 0..w + 2 {
            out.push('─');
        }
        if i + 1 < widths.len() {
            out.push(mid);
        }
    }
    out.push(right);
    out.push('\n');
}

fn render_border_no_newline(
    out: &mut String,
    widths: &[usize],
    left: char,
    mid: char,
    right: char,
) {
    out.push(left);
    for (i, &w) in widths.iter().enumerate() {
        for _ in 0..w + 2 {
            out.push('─');
        }
        if i + 1 < widths.len() {
            out.push(mid);
        }
    }
    out.push(right);
}

fn render_row(out: &mut String, cells: &[String], widths: &[usize], aligns: &[Align]) {
    out.push('│');
    for (i, cell) in cells.iter().enumerate() {
        out.push(' ');
        out.push_str(&pad_cell(
            cell,
            widths[i],
            aligns.get(i).copied().unwrap_or(Align::Left),
        ));
        out.push(' ');
        out.push('│');
    }
    out.push('\n');
}

fn render_data_row(
    out: &mut String,
    row: &[String],
    widths: &[usize],
    aligns: &[Align],
    cols: usize,
) {
    out.push('│');
    for (i, w) in widths.iter().enumerate().take(cols) {
        let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
        out.push(' ');
        out.push_str(&pad_cell(
            cell,
            *w,
            aligns.get(i).copied().unwrap_or(Align::Left),
        ));
        out.push(' ');
        out.push('│');
    }
    out.push('\n');
}

/// Approximate display width accounting for CJK characters (width 2).
fn display_width(s: &str) -> usize {
    s.chars().map(|c| if is_wide_char(c) { 2 } else { 1 }).sum()
}

fn is_wide_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x3000..=0x303F).contains(&cp)
        || (0xFF01..=0xFF60).contains(&cp)
        || (0xFE30..=0xFE4F).contains(&cp)
        || (0x2E80..=0x2EFF).contains(&cp)
        || (0x3100..=0x312F).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
}

fn pad_cell(s: &str, width: usize, align: Align) -> String {
    let w = display_width(s);
    if w >= width {
        return s.to_string();
    }
    let diff = width - w;
    match align {
        Align::Left => format!("{}{}", s, " ".repeat(diff)),
        Align::Right => format!("{}{}", " ".repeat(diff), s),
    }
}

// ── Levenshtein distance ────────────────────────────────────────────────

pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Find the closest command from a list, returning it if within distance threshold.
pub fn suggest_command<'a>(input: &str, commands: &[&'a str], max_dist: usize) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for &cmd in commands {
        let d = levenshtein(input, cmd);
        if d <= max_dist && (best.is_none() || d < best.unwrap().1) {
            best = Some((cmd, d));
        }
    }
    best.map(|(cmd, _)| cmd)
}
