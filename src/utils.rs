//! Shared utility functions, constants, and SIMD-accelerated math helpers.

use crate::achf::AchfCacheStats;

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
pub const F2P_BATCH_COUNT: usize = 100;

/// History capacity for interactive simulation history.
pub const SIM_HISTORY_CAPACITY: usize = 20;

/// Maximum display items for pull detail list.
pub const PULL_DISPLAY_LIMIT: usize = 20;

/// Maximum user-input pulls or sims before capping.
pub const INPUT_CAP: usize = 1_000_000;

/// Default PPO context length when not specified.
pub const DEFAULT_PPO_CONTEXT_LEN: usize = 8;

/// Compute DQN-style reward for experience replay.
pub fn compute_reward_dqn(is_six: bool, is_up: bool, loss_streak: usize) -> f64 {
    let mut reward = REWARD_BASE;
    if is_six {
        if is_up {
            reward += REWARD_SIX_UP_DQN;
        } else {
            reward += REWARD_SIX_NON_UP_DQN;
        }
    }
    if loss_streak >= STREAK_PENALTY_THRESHOLD {
        reward -= (loss_streak as f64) * STREAK_PENALTY_DQN;
    }
    reward
}

/// Compute Neural-style reward for online neural training.
pub fn compute_reward_neural(is_six: bool, is_up: bool, loss_streak: usize) -> f64 {
    let mut reward = REWARD_BASE;
    if is_six {
        if is_up {
            reward += REWARD_SIX_UP_NEURAL;
        } else {
            reward += REWARD_SIX_NON_UP_NEURAL;
        }
    }
    if loss_streak >= STREAK_PENALTY_THRESHOLD {
        reward -= (loss_streak as f64) * STREAK_PENALTY_NEURAL;
    }
    reward
}

/// Compute PPO-style reward for PPO experience replay.
pub fn compute_reward_ppo(is_six: bool, is_up: bool, loss_streak: usize) -> f64 {
    let mut reward = REWARD_BASE;
    if is_six {
        if is_up {
            reward += REWARD_SIX_UP_PPO;
        } else {
            reward += REWARD_SIX_NON_UP_PPO;
        }
    }
    if loss_streak >= STREAK_PENALTY_THRESHOLD {
        reward -= (loss_streak as f64) * STREAK_PENALTY_PPO;
    }
    reward
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
        "[ACHF] Calls: {} | Hit: {:.2}% | Miss: {} | Skip: {} | LowRank: {} | Dense: {} | CachedEMA(ns): {:.1}/{:.1} | LowRankEMA(ns): {:.1}/{:.1} | DecisionEMA(ns): {:.1}/{:.1} | Bias: {:.3} | Samples: {}/{}",
        stats.calls,
        hit_rate * 100.0,
        stats.cache_misses,
        stats.cache_skips,
        stats.low_rank_paths,
        stats.dense_paths,
        stats.ema_cached_ns,
        stats.ema_cached_long_ns,
        stats.ema_low_rank_ns,
        stats.ema_low_rank_long_ns,
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
