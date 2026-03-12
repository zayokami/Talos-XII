//! Bayesian calibration: use collected player data to refine gacha probability estimates.

#![allow(dead_code)] // Module API for future CLI integration (collect/calibrate subcommands)

use crate::collect::{PlayerDatabase, PoolEmpiricalStats};
use crate::config::Config;
use crate::i18n::{I18n, Language};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const CALIBRATION_MIN_PULLS: usize = 500;

/// Result of Bayesian calibration for one parameter.
#[derive(Debug, Clone)]
pub struct BayesianEstimate {
    pub prior_mean: f64,
    pub posterior_mean: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub significant: bool,
}

/// Calibrated parameters for one pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolCalibration {
    pub pool_id: String,
    pub prob_6_base: Option<f64>,
    pub soft_pity_slope: Option<f64>,
    pub up_rate: Option<f64>,
    pub sample_pulls: usize,
    pub sample_six_stars: usize,
}

/// All calibrated parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibrationData {
    pub pools: HashMap<String, PoolCalibration>,
}

impl CalibrationData {
    pub fn load(path: &str) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            match serde_json::from_str(&data) {
                Ok(cal) => return cal,
                Err(e) => log::warn!("[Calibrate] Failed to parse {}: {}", path, e),
            }
        }
        CalibrationData::default()
    }

    pub fn save(&self, path: &str) -> bool {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => std::fs::write(path, json).is_ok(),
            Err(_) => false,
        }
    }
}

// ── Beta distribution helpers ────────────────────────────────────────────

/// Beta distribution posterior mean: (α + k) / (α + β + n)
fn beta_posterior_mean(alpha_prior: f64, beta_prior: f64, successes: f64, trials: f64) -> f64 {
    (alpha_prior + successes) / (alpha_prior + beta_prior + trials)
}

/// Regularized incomplete beta function I_x(a, b) using continued fraction (Lentz).
fn reg_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if a <= 0.0 || b <= 0.0 {
        return 0.5;
    }

    let symmetry = x > (a + 1.0) / (a + b + 2.0);
    let (x_eff, a_eff, b_eff) = if symmetry { (1.0 - x, b, a) } else { (x, a, b) };

    let ln_prefix = a_eff * x_eff.ln() + b_eff * (1.0 - x_eff).ln() - ln_beta(a_eff, b_eff);
    let prefix = ln_prefix.exp();

    let cf = beta_cf(x_eff, a_eff, b_eff);
    let result = prefix * cf / a_eff;

    if symmetry {
        1.0 - result
    } else {
        result
    }
}

/// Continued fraction for the incomplete beta function.
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = (1.0 - (a + b) * x / (a + 1.0)).recip().max(tiny);
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = (1.0 + num_even * d).recip().max(tiny);
        c = (1.0 + num_even / c).max(tiny);
        h *= d * c;

        let num_odd = -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = (1.0 + num_odd * d).recip().max(tiny);
        c = (1.0 + num_odd / c).max(tiny);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// Log of the Beta function: ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation for ln(Γ(x))
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (j, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + j as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Inverse of the regularized incomplete beta function (quantile).
/// Uses bisection on I_x(a, b) = p.
fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let tol = 1e-10;

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let val = reg_incomplete_beta(mid, a, b);
        if (val - p).abs() < tol {
            return mid;
        }
        if val < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Compute 95% Bayesian credible interval using Beta posterior.
fn beta_credible_interval(
    alpha_prior: f64,
    beta_prior: f64,
    successes: f64,
    trials: f64,
) -> (f64, f64) {
    let a_post = alpha_prior + successes;
    let b_post = beta_prior + (trials - successes);
    let lower = beta_quantile(0.025, a_post, b_post);
    let upper = beta_quantile(0.975, a_post, b_post);
    (lower, upper)
}

// ── Calibration logic ────────────────────────────────────────────────────

fn estimate_base_rate(
    stats: &PoolEmpiricalStats,
    config_base: f64,
    soft_start: usize,
) -> BayesianEstimate {
    let pre_soft_pulls: usize = {
        let total_episodes: usize = stats.pity_hits.iter().sum();
        if total_episodes == 0 {
            return BayesianEstimate {
                prior_mean: config_base,
                posterior_mean: config_base,
                ci_lower: config_base * 0.5,
                ci_upper: config_base * 2.0,
                significant: false,
            };
        }
        let mut surviving = total_episodes;
        let mut total_pre_soft = 0usize;
        for k in 1..soft_start.min(stats.pity_hits.len()) {
            total_pre_soft += surviving;
            surviving = surviving.saturating_sub(stats.pity_hits[k]);
        }
        total_pre_soft.max(1)
    };

    let pre_soft_six: usize = stats.pity_hits.iter().take(soft_start).sum();

    // Prior: centered on config_base with moderate confidence (equivalent to ~1000 prior pulls)
    let n_prior = 1000.0;
    let alpha_prior = config_base * n_prior;
    let beta_prior = (1.0 - config_base) * n_prior;

    let posterior_mean = beta_posterior_mean(
        alpha_prior,
        beta_prior,
        pre_soft_six as f64,
        pre_soft_pulls as f64,
    );
    let (ci_lower, ci_upper) = beta_credible_interval(
        alpha_prior,
        beta_prior,
        pre_soft_six as f64,
        pre_soft_pulls as f64,
    );

    let significant = ci_lower > config_base || ci_upper < config_base;

    BayesianEstimate {
        prior_mean: config_base,
        posterior_mean,
        ci_lower,
        ci_upper,
        significant,
    }
}

fn estimate_up_rate(stats: &PoolEmpiricalStats, config_up_rate: f64) -> BayesianEstimate {
    // Prior: centered on config_up_rate with moderate confidence
    let n_prior = 100.0;
    let alpha_prior = config_up_rate * n_prior;
    let beta_prior = (1.0 - config_up_rate) * n_prior;

    let posterior_mean = beta_posterior_mean(
        alpha_prior,
        beta_prior,
        stats.total_up as f64,
        stats.total_six_star as f64,
    );
    let (ci_lower, ci_upper) = beta_credible_interval(
        alpha_prior,
        beta_prior,
        stats.total_up as f64,
        stats.total_six_star as f64,
    );

    let significant = ci_lower > config_up_rate || ci_upper < config_up_rate;

    BayesianEstimate {
        prior_mean: config_up_rate,
        posterior_mean,
        ci_lower,
        ci_upper,
        significant,
    }
}

fn estimate_soft_pity_slope(
    stats: &PoolEmpiricalStats,
    config_slope: f64,
    config_base: f64,
    soft_start: usize,
    guarantee: usize,
) -> BayesianEstimate {
    // Estimate slope from observed rates at each pity level in the soft range.
    // We count successes/trials at each pity level and fit a linear model.
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut n_data = 0.0;

    // We need both the number of 6-stars at each pity AND the number of
    // pulls that reached that pity. The latter requires knowing the survival
    // function, which we approximate from the pity_hits distribution.
    let total_episodes: usize = stats.pity_hits.iter().sum();
    if total_episodes == 0 {
        return BayesianEstimate {
            prior_mean: config_slope,
            posterior_mean: config_slope,
            ci_lower: config_slope * 0.8,
            ci_upper: config_slope * 1.2,
            significant: false,
        };
    }

    // Survival at pity k = episodes that haven't gotten 6-star by pity k
    let mut surviving = total_episodes;
    for k in 1..guarantee.min(stats.pity_hits.len()) {
        if surviving == 0 {
            break;
        }
        let hits = stats.pity_hits[k];

        if k >= soft_start && k < guarantee {
            let rate = hits as f64 / surviving as f64;
            let x = (k - soft_start + 1) as f64;
            let y = (rate - config_base).max(0.0);

            sum_xx += x * x;
            sum_xy += x * y;
            n_data += 1.0;
        }
        surviving = surviving.saturating_sub(hits);
    }

    if n_data < 3.0 {
        return BayesianEstimate {
            prior_mean: config_slope,
            posterior_mean: config_slope,
            ci_lower: config_slope * 0.8,
            ci_upper: config_slope * 1.2,
            significant: false,
        };
    }

    // Simple linear regression: y = slope * x (no intercept, since base is subtracted)
    let observed_slope = sum_xy / sum_xx;

    // Bayesian shrinkage toward prior
    let prior_precision = 100.0;
    let data_precision = n_data / (config_slope * 0.2).powi(2).max(1e-8);
    let posterior_precision = prior_precision + data_precision;
    let posterior_mean =
        (prior_precision * config_slope + data_precision * observed_slope) / posterior_precision;
    let posterior_std = (1.0 / posterior_precision).sqrt();

    let ci_lower = (posterior_mean - 1.96 * posterior_std).max(0.0);
    let ci_upper = posterior_mean + 1.96 * posterior_std;
    let significant = ci_lower > config_slope || ci_upper < config_slope;

    BayesianEstimate {
        prior_mean: config_slope,
        posterior_mean,
        ci_lower,
        ci_upper,
        significant,
    }
}

/// Run Bayesian calibration on all pools with sufficient data.
pub fn run_calibration(db: &PlayerDatabase, config: &Config, lang: Language) -> CalibrationData {
    println!("{}", I18n::get(lang, "cal_header"));

    let pool_stats = db.compute_pool_stats(config);

    println!(
        "  {}: {}    {}: {}",
        I18n::get(lang, "cal_total_samples"),
        db.total_pulls(),
        I18n::get(lang, "cal_sessions"),
        db.sessions.len(),
    );

    let mut calibration = CalibrationData::default();

    for ps in &pool_stats {
        let pool_cfg = config.pools.iter().find(|p| p.id == ps.pool_id);

        let (cfg_base, cfg_slope, cfg_up, soft_start, guarantee) = match pool_cfg {
            Some(p) => (
                p.prob_6_base,
                p.soft_pity_slope,
                p.up_rate,
                p.soft_pity_start,
                p.small_pity_guarantee,
            ),
            None => (
                config.prob_6_base,
                config.soft_pity_slope,
                config.up_rate,
                config.soft_pity_start,
                config.small_pity_guarantee,
            ),
        };

        println!(
            "\n  ── {} ({} {}) ──",
            ps.pool_name,
            ps.total_pulls,
            I18n::get(lang, "cal_unit_pulls"),
        );

        if ps.total_pulls < CALIBRATION_MIN_PULLS {
            println!(
                "  ⚠ {} (< {} {})",
                I18n::get(lang, "cal_insufficient"),
                CALIBRATION_MIN_PULLS,
                I18n::get(lang, "cal_unit_pulls"),
            );

            print_estimate_row(
                &I18n::get(lang, "cal_base_rate"),
                &BayesianEstimate {
                    prior_mean: cfg_base,
                    posterior_mean: ps.observed_base_rate,
                    ci_lower: 0.0,
                    ci_upper: 1.0,
                    significant: false,
                },
                true,
                lang,
            );
            continue;
        }

        let base_est = estimate_base_rate(ps, cfg_base, soft_start);
        let up_est = estimate_up_rate(ps, cfg_up);
        let slope_est = estimate_soft_pity_slope(ps, cfg_slope, cfg_base, soft_start, guarantee);

        println!(
            "\n  {:<20} {:>10} {:>10} {:>18} {:>6}",
            I18n::get(lang, "cal_col_param"),
            I18n::get(lang, "cal_col_official"),
            I18n::get(lang, "cal_col_calibrated"),
            I18n::get(lang, "cal_col_ci"),
            I18n::get(lang, "cal_col_sig"),
        );
        println!("  {}", "-".repeat(68));

        print_estimate_row(&I18n::get(lang, "cal_base_rate"), &base_est, true, lang);
        print_estimate_row(&I18n::get(lang, "cal_slope"), &slope_est, false, lang);
        print_estimate_row(&I18n::get(lang, "cal_up_rate"), &up_est, true, lang);

        let mut pool_cal = PoolCalibration {
            pool_id: ps.pool_id.clone(),
            prob_6_base: None,
            soft_pity_slope: None,
            up_rate: None,
            sample_pulls: ps.total_pulls,
            sample_six_stars: ps.total_six_star,
        };

        if ps.total_pulls >= CALIBRATION_MIN_PULLS {
            pool_cal.prob_6_base = Some(base_est.posterior_mean);
            pool_cal.soft_pity_slope = Some(slope_est.posterior_mean);
            pool_cal.up_rate = Some(up_est.posterior_mean);
        }

        calibration.pools.insert(ps.pool_id.clone(), pool_cal);
    }

    if pool_stats.is_empty() {
        println!("\n  {}", I18n::get(lang, "cal_no_data"));
    }

    // Sample size guidance
    let total_six: usize = pool_stats.iter().map(|p| p.total_six_star).sum();
    if total_six > 0 && total_six < 400 {
        let needed = 400 - total_six;
        println!(
            "\n  {} ≈ {} {} 6★ {} ±3% {}",
            I18n::get(lang, "cal_sample_hint"),
            needed,
            I18n::get(lang, "cal_sample_hint_more"),
            I18n::get(lang, "cal_sample_hint_narrow"),
            I18n::get(lang, "cal_sample_hint_acc"),
        );
    }

    println!();
    calibration
}

fn print_estimate_row(name: &str, est: &BayesianEstimate, as_percent: bool, lang: Language) {
    let (prior_s, post_s, ci_s) = if as_percent {
        (
            format!("{:.3}%", est.prior_mean * 100.0),
            format!("{:.3}%", est.posterior_mean * 100.0),
            format!(
                "[{:.2}%, {:.2}%]",
                est.ci_lower * 100.0,
                est.ci_upper * 100.0
            ),
        )
    } else {
        (
            format!("{:.4}", est.prior_mean),
            format!("{:.4}", est.posterior_mean),
            format!("[{:.4}, {:.4}]", est.ci_lower, est.ci_upper),
        )
    };
    let sig = if est.significant {
        I18n::get(lang, "cal_sig_yes")
    } else {
        I18n::get(lang, "cal_sig_no")
    };
    println!(
        "  {:<20} {:>10} {:>10} {:>18} {:>6}",
        name, prior_s, post_s, ci_s, sig
    );
}

/// Apply calibration data to a Config, overriding pool parameters where calibrated.
pub fn apply_calibration(config: &mut Config, cal: &CalibrationData) {
    if let Some(active) = &config.active_pool {
        if let Some(pool_cal) = cal.pools.get(active) {
            if let Some(base) = pool_cal.prob_6_base {
                config.prob_6_base = base;
            }
            if let Some(slope) = pool_cal.soft_pity_slope {
                config.soft_pity_slope = slope;
            }
            if let Some(up) = pool_cal.up_rate {
                config.up_rate = up;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_posterior_mean_basic() {
        let mean = beta_posterior_mean(1.0, 1.0, 5.0, 10.0);
        assert!((mean - 0.5).abs() < 1e-9);
    }

    #[test]
    fn beta_posterior_with_prior() {
        // Prior: Beta(75, 25) → mean 0.75
        // Data: 8 UP / 10 six-stars
        let mean = beta_posterior_mean(75.0, 25.0, 8.0, 10.0);
        // Should be close to 0.75 (prior-dominated due to strong prior)
        assert!(mean > 0.74 && mean < 0.77);
    }

    #[test]
    fn beta_quantile_symmetry() {
        let q50 = beta_quantile(0.5, 2.0, 2.0);
        assert!(
            (q50 - 0.5).abs() < 1e-6,
            "Symmetric Beta median should be 0.5, got {}",
            q50
        );
    }

    #[test]
    fn beta_credible_interval_covers_mean() {
        let (lo, hi) = beta_credible_interval(10.0, 10.0, 5.0, 10.0);
        let mean = beta_posterior_mean(10.0, 10.0, 5.0, 10.0);
        assert!(lo < mean && mean < hi);
    }

    #[test]
    fn reg_incomplete_beta_boundaries() {
        assert!((reg_incomplete_beta(0.0, 2.0, 3.0) - 0.0).abs() < 1e-12);
        assert!((reg_incomplete_beta(1.0, 2.0, 3.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reg_incomplete_beta_known_value() {
        // I_0.5(1, 1) = 0.5 (Uniform distribution)
        let val = reg_incomplete_beta(0.5, 1.0, 1.0);
        assert!((val - 0.5).abs() < 1e-10);
    }
}
