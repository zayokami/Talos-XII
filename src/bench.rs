use crate::achf::{
    aggregate_cache_stats_iter, AchfCacheStats, AchfLayer, AchfMemoryStats, AchfSparsityStats,
};
use crate::chart::{self, ChartFormat};
use crate::config::{AchfConfig, Config, LuckMode};
use crate::dqn::{train_dqn_with_metrics, DuelingQNetwork};
use crate::env_net::EnvNet;
use crate::neural::NeuralLuckOptimizer;
use crate::policy_eval::{evaluate_dqn_policy, evaluate_ppo_policy};
use crate::ppo::{train_ppo_with_metrics, ActorCritic};
use crate::rng::Rng;
use crate::sim::{simulate_fast, SimModelContext};
use crate::trainer::{train_linear_regression, train_manifold_rl, train_neural_optimizer};
use crate::training_metrics::StepSnapshot;
use crate::worker::GoodJobWorker;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const BENCH_EXPERIMENTS: &[&str] = &[
    "ablation",
    "mode",
    "path",
    "gate",
    "scale",
    "apply",
    "convergence",
    "crossover",
    "regime",
];
const THROUGHPUT_SIMS: usize = 1000;
const THROUGHPUT_PULLS: usize = 100;
const THROUGHPUT_WARMUP_SIMS: usize = 100;
const BENCH_EVAL_EPISODES: usize = 128;
const PATH_WARMUP_ROUNDS: usize = 100;
const PATH_SAMPLES: usize = 1000;
const PATH_CALLS_PER_SAMPLE: usize = 64;
const PATH_PRUNE_THRESHOLD: f64 = 0.005;
const CROSSOVER_DIMS: [usize; 3] = [256, 1024, 2048];
const CROSSOVER_SPARSITIES: [f64; 5] = [0.5, 0.8, 0.9, 0.95, 0.99];
const CROSSOVER_BATCH: usize = 32;
const CROSSOVER_WARMUP_ROUNDS: usize = 20;
const CROSSOVER_SAMPLES: usize = 200;
const REGIME_DIM: usize = 1024;
const REGIME_SPARSITIES: [f64; 4] = [0.8, 0.9, 0.95, 0.98];
const REGIME_SMALL_BATCH: usize = 1;
const REGIME_LARGE_BATCH: usize = 128;
const REGIME_WARMUP_CALLS: usize = 300;
const REGIME_MEASURE_CALLS: usize = 1200;
const REGIME_FORCED_WARMUP_ROUNDS: usize = 40;
const REGIME_FORCED_CALLS_PER_SAMPLE: usize = 20;
const SEED_TRIAL: u64 = 0xA076_1D64_78BD_642F;
const SEED_BASE_MODELS: u64 = 0xE703_7ED1_A0B4_28DB;
const SEED_ENV_NET: u64 = 0xD1B5_4A32_D192_ED03;
const SEED_NEURAL_OPT: u64 = 0xABC9_8388_FB8F_AC03;
const SEED_LINEAR_REGRESSION: u64 = 0x8CB9_2BA7_2F3D_8DD7;
const SEED_MANIFOLD_RL: u64 = 0xDB4F_0B91_75AE_2165;
const SEED_DQN_TRAIN: u64 = 0x8EBC_6AF0_9C88_C6E3;
const SEED_PPO_TRAIN: u64 = 0x5899_65CC_7537_4CC3;
const SEED_POLICY_EVAL: u64 = 0x1D8E_4E27_C47D_124F;
const SEED_THROUGHPUT: u64 = 0xEB44_ACC9_AB45_54A3;

type OwnedCiSeries = (String, Vec<chart::CiPoint>);

// ── Data structures ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BenchRunResult {
    pub label: String,
    pub policy: String,
    pub config_fingerprint: String,
    pub condition_config: serde_json::Value,
    pub train_time_ms: f64,
    pub throughput_sims_per_sec: f64,
    pub eval_reward: f64,
    pub train_loss: f64,
    pub param_count: usize,
    pub applied_rank: Option<usize>,
    pub candidate_relative_error: Option<f64>,
    pub memory_stats: Option<AchfMemoryStats>,
    pub snapshots: Vec<StepSnapshot>,
    pub cache_stats: Option<AchfCacheStats>,
}

pub struct BenchConfig {
    pub output_dir: String,
    pub format: ChartFormat,
    pub only: Option<Vec<String>>,
    pub num_trials: usize,
}

#[derive(Clone, Debug)]
pub struct TrialStats {
    pub mean: f64,
    pub std_dev: f64,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    pub values: Vec<f64>,
}

impl TrialStats {
    fn from_values(vals: &[f64]) -> Self {
        assert!(
            vals.iter().all(|value| value.is_finite()),
            "benchmark statistics contain a non-finite value: {vals:?}"
        );
        if vals.is_empty() {
            return TrialStats {
                mean: 0.0,
                std_dev: 0.0,
                ci_low: None,
                ci_high: None,
                values: Vec::new(),
            };
        }
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std_dev = variance.sqrt();
        let (ci_low, ci_high) = if vals.len() > 1 {
            let t_val = student_t_critical_95(vals.len() - 1);
            let margin = t_val * std_dev / n.sqrt();
            (Some(mean - margin), Some(mean + margin))
        } else {
            (None, None)
        };
        TrialStats {
            mean,
            std_dev,
            ci_low,
            ci_high,
            values: vals.to_vec(),
        }
    }
}

fn student_t_critical_95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if df == 0 {
        return f64::NAN;
    }
    if df <= T.len() {
        return T[df - 1];
    }
    let df = df as f64;
    let z: f64 = 1.959_963_984_540_054;
    z + (z.powi(3) + z) / (4.0 * df)
        + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df.powi(2))
}

#[derive(Clone, Debug)]
pub struct PairedComparison {
    pub baseline: String,
    pub throughput_delta: TrialStats,
    pub throughput_relative_delta_pct: TrialStats,
    pub eval_reward_delta: TrialStats,
    pub train_loss_delta: TrialStats,
    pub train_time_ms_delta: TrialStats,
}

#[derive(Clone, Debug)]
pub struct CurvePointStats {
    pub step: usize,
    pub samples: usize,
    pub train_loss: TrialStats,
    pub train_reward: TrialStats,
    pub gate_value: TrialStats,
    pub gate_velocity: TrialStats,
    pub g_min: TrialStats,
    pub candidate_eligible_rate: TrialStats,
    pub candidate_sparsity: TrialStats,
    pub candidate_relative_error: TrialStats,
    pub candidate_weight_error_ema: TrialStats,
    pub connection_candidate_weight: TrialStats,
    pub grad_ema: TrialStats,
    pub gradient_cosine: TrialStats,
    pub cached_path_rate: TrialStats,
    pub sparse_ratio: TrialStats,
    pub ema_cached_ns: TrialStats,
    pub ema_sparse_ns: TrialStats,
    pub adaptive_bias: TrialStats,
    pub connection_projection_iterations: TrialStats,
    pub connection_row_max_deviation: TrialStats,
    pub connection_col_max_deviation: TrialStats,
    pub connection_min_value: TrialStats,
    pub connection_negative_ratio: TrialStats,
    pub low_rank_applied_rank: TrialStats,
}

#[derive(Clone, Debug)]
pub struct AggregatedResult {
    pub label: String,
    pub policy: String,
    pub config_fingerprint: String,
    pub condition_config: serde_json::Value,
    pub throughput: TrialStats,
    pub eval_reward: TrialStats,
    pub train_loss: TrialStats,
    pub train_time_ms: TrialStats,
    pub param_count: usize,
    pub applied_rank: Option<usize>,
    pub candidate_relative_error: Option<TrialStats>,
    pub memory_stats: Vec<AchfMemoryStats>,
    pub curve: Vec<CurvePointStats>,
    pub cache_stats: Option<AchfCacheStats>,
    pub cache_trial_count: usize,
    pub paired: Option<PairedComparison>,
}

#[derive(Clone, Debug)]
struct PathLatencyResult {
    label: String,
    trial_samples: Vec<Vec<f64>>,
    trial_input_dims: Vec<usize>,
    trial_sparsity: Vec<AchfSparsityStats>,
}

/// One cell of the path-crossover grid: trial-level per-path latency at a
/// requested (dim, weight_sparsity) operating point, including the actual
/// frozen nnz so the benchmark cannot silently measure a different sparsity.
#[derive(Clone, Debug)]
struct CrossoverCell {
    dim: usize,
    requested_sparsity: f64,
    actual_sparsity: f64,
    total_weights: usize,
    nonzero_weights: usize,
    cached_ns: TrialStats,
    sparse_ns: TrialStats,
    dense_ns: TrialStats,
    winner: String,
    significant_winner_95: Option<String>,
    cached_minus_sparse_ns: TrialStats,
    cached_minus_dense_ns: TrialStats,
    sparse_minus_dense_ns: TrialStats,
}

/// Per-regime latency for the adaptation experiment. For one batch size on a
/// fixed layer we record the live adaptive selector's achieved latency, the two
/// fixed-path latencies, the oracle (best fixed path), and the sparse-selection
/// fraction. The oracle-gap (adaptive / oracle) is the robust adaptation
/// metric: near a crossover both fixed paths cost ~the same, so the selector's
/// achieved latency tracks the oracle even where its path *choice* is noisy.
#[derive(Clone, Debug)]
struct RegimeLatency {
    batch: usize,
    adaptive_ns: TrialStats,
    plain_ema_ns: TrialStats,
    cached_ns: TrialStats,
    sparse_ns: TrialStats,
    dense_ns: TrialStats,
    oracle_ns: TrialStats,
    oracle_gap: TrialStats,
    plain_ema_oracle_gap: TrialStats,
    cached_oracle_gap: TrialStats,
    sparse_oracle_gap: TrialStats,
    dense_oracle_gap: TrialStats,
    oracle_path: String,
    oracle_path_counts: BTreeMap<String, usize>,
    oracle_paths: Vec<String>,
    sparse_frac: TrialStats,
    plain_ema_sparse_frac: TrialStats,
}

/// One row of the regime-adaptation experiment: the small-batch (decode-like)
/// and large-batch (prefill-like) measurements for one fixed weight sparsity.
/// When the two regimes have DIFFERENT oracle paths yet the adaptive selector
/// stays near-oracle in both, that is the "true adaptive" result: a single
/// fixed path cannot win both regimes, but the batch-aware selector does.
#[derive(Clone, Debug)]
struct RegimeRow {
    requested_sparsity: f64,
    actual_sparsity: f64,
    total_weights: usize,
    nonzero_weights: usize,
    small: RegimeLatency,
    large: RegimeLatency,
}

#[derive(Clone, Debug)]
struct RegimeTrialLatency {
    batch: usize,
    adaptive_ns: f64,
    plain_ema_ns: f64,
    cached_ns: f64,
    sparse_ns: f64,
    dense_ns: f64,
    oracle_ns: f64,
    oracle_path: String,
    sparse_frac: f64,
    plain_ema_sparse_frac: f64,
}

#[derive(Clone, Debug)]
struct PathLatencyStats {
    label: String,
    trials: usize,
    samples: usize,
    mean_ns: f64,
    std_dev_ns: f64,
    ci_low_ns: Option<f64>,
    ci_high_ns: Option<f64>,
    min_ns: f64,
    p50_ns: f64,
    p90_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
    max_ns: f64,
}

fn aggregate_trials(runs: &[BenchRunResult]) -> AggregatedResult {
    assert!(!runs.is_empty(), "cannot aggregate zero benchmark trials");
    let label = runs[0].label.clone();
    let tputs: Vec<f64> = runs.iter().map(|r| r.throughput_sims_per_sec).collect();
    let rewards: Vec<f64> = runs.iter().map(|r| r.eval_reward).collect();
    let losses: Vec<f64> = runs.iter().map(|r| r.train_loss).collect();
    let times: Vec<f64> = runs.iter().map(|r| r.train_time_ms).collect();
    assert!(
        runs.iter().all(|run| run.policy == runs[0].policy),
        "mixed active policies under benchmark label {label}"
    );
    assert!(
        runs.iter()
            .all(|run| run.config_fingerprint == runs[0].config_fingerprint),
        "condition configuration changed across trials for {label}"
    );
    assert!(
        runs.iter()
            .all(|run| run.param_count == runs[0].param_count),
        "parameter count changed across trials for {label}"
    );
    assert!(
        runs.iter()
            .all(|run| run.applied_rank == runs[0].applied_rank),
        "applied rank changed across trials for {label}"
    );
    let cache_values: Vec<AchfCacheStats> = runs.iter().filter_map(|run| run.cache_stats).collect();
    let candidate_relative_error_values: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.candidate_relative_error)
        .collect();
    let memory_stats: Vec<AchfMemoryStats> =
        runs.iter().filter_map(|run| run.memory_stats).collect();
    assert!(
        candidate_relative_error_values.is_empty()
            || candidate_relative_error_values.len() == runs.len(),
        "pruning diagnostics missing from some trials for {label}"
    );
    assert!(
        memory_stats.is_empty() || memory_stats.len() == runs.len(),
        "memory diagnostics missing from some trials for {label}"
    );
    let cache_stats = (!cache_values.is_empty())
        .then(|| aggregate_cache_stats_iter(cache_values.iter().copied()));
    AggregatedResult {
        label,
        policy: runs[0].policy.clone(),
        config_fingerprint: runs[0].config_fingerprint.clone(),
        condition_config: runs[0].condition_config.clone(),
        throughput: TrialStats::from_values(&tputs),
        eval_reward: TrialStats::from_values(&rewards),
        train_loss: TrialStats::from_values(&losses),
        train_time_ms: TrialStats::from_values(&times),
        param_count: runs[0].param_count,
        applied_rank: runs[0].applied_rank,
        candidate_relative_error: (!candidate_relative_error_values.is_empty())
            .then(|| TrialStats::from_values(&candidate_relative_error_values)),
        memory_stats,
        curve: aggregate_snapshots(runs),
        cache_stats,
        cache_trial_count: cache_values.len(),
        paired: None,
    }
}

fn aggregate_snapshots(runs: &[BenchRunResult]) -> Vec<CurvePointStats> {
    let mut by_step: BTreeMap<usize, Vec<&StepSnapshot>> = BTreeMap::new();
    for snapshot in runs.iter().flat_map(|run| run.snapshots.iter()) {
        by_step.entry(snapshot.step).or_default().push(snapshot);
    }
    by_step
        .into_iter()
        .map(|(step, snapshots)| {
            let stats = |value: fn(&StepSnapshot) -> f64| {
                TrialStats::from_values(
                    &snapshots
                        .iter()
                        .map(|snapshot| value(snapshot))
                        .collect::<Vec<_>>(),
                )
            };
            CurvePointStats {
                step,
                samples: snapshots.len(),
                train_loss: stats(|snapshot| snapshot.loss),
                train_reward: stats(|snapshot| snapshot.reward),
                gate_value: stats(|snapshot| snapshot.gate_value),
                gate_velocity: stats(|snapshot| snapshot.gate_velocity),
                g_min: stats(|snapshot| snapshot.g_min),
                candidate_eligible_rate: stats(|snapshot| {
                    if snapshot.candidate_eligible {
                        1.0
                    } else {
                        0.0
                    }
                }),
                candidate_sparsity: stats(|snapshot| snapshot.candidate_sparsity),
                candidate_relative_error: stats(|snapshot| snapshot.candidate_relative_error),
                candidate_weight_error_ema: stats(|snapshot| snapshot.candidate_weight_error_ema),
                connection_candidate_weight: stats(|snapshot| snapshot.connection_candidate_weight),
                grad_ema: stats(|snapshot| snapshot.grad_ema),
                gradient_cosine: stats(|snapshot| snapshot.gradient_cosine),
                cached_path_rate: stats(|snapshot| snapshot.cached_path_rate),
                sparse_ratio: stats(|snapshot| snapshot.sparse_ratio),
                ema_cached_ns: stats(|snapshot| snapshot.ema_cached_ns),
                ema_sparse_ns: stats(|snapshot| snapshot.ema_sparse_ns),
                adaptive_bias: stats(|snapshot| snapshot.adaptive_bias),
                connection_projection_iterations: stats(|snapshot| {
                    snapshot.connection_projection_iterations as f64
                }),
                connection_row_max_deviation: stats(|snapshot| {
                    snapshot.connection_row_max_deviation
                }),
                connection_col_max_deviation: stats(|snapshot| {
                    snapshot.connection_col_max_deviation
                }),
                connection_min_value: stats(|snapshot| snapshot.connection_min_value),
                connection_negative_ratio: stats(|snapshot| snapshot.connection_negative_ratio),
                low_rank_applied_rank: stats(|snapshot| snapshot.low_rank_applied_rank as f64),
            }
        })
        .collect()
}

// ── Helper: build neural + worker from config ───────────────────────────

fn derive_seed(root: u64, domain: u64) -> u64 {
    let mut value = root ^ domain;
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn benchmark_trial_seed(seed: u64, trial: usize) -> u64 {
    derive_seed(seed, SEED_TRIAL.wrapping_mul(trial as u64 + 1))
}

fn build_base_models_with_worker(
    config: &Config,
    seed: u64,
    worker: &GoodJobWorker,
) -> (EnvNet, NeuralLuckOptimizer) {
    let mut env_rng = Rng::from_seed(derive_seed(seed, SEED_ENV_NET));
    let mut env_net = EnvNet::new(&mut env_rng);
    let (count, epochs) = if config.fast_init {
        (256, 10)
    } else {
        (1024, 50)
    };
    env_net.pretrain(&mut env_rng, config, count, epochs);
    env_net.set_train(false);

    let mut neural_opt =
        train_neural_optimizer(derive_seed(seed, SEED_NEURAL_OPT), &env_net, config, worker);
    let mut linear_rng = Rng::from_seed(derive_seed(seed, SEED_LINEAR_REGRESSION));
    let (w, b) = train_linear_regression(&neural_opt, &mut linear_rng, &env_net, config);
    neural_opt.set_linear_params(w, b);
    let mut manifold_rng = Rng::from_seed(derive_seed(seed, SEED_MANIFOLD_RL));
    neural_opt = train_manifold_rl(&neural_opt, &mut manifold_rng, &env_net, config, worker);

    (env_net, neural_opt)
}

fn build_base_models(config: &Config, seed: u64) -> (EnvNet, NeuralLuckOptimizer, GoodJobWorker) {
    let worker =
        GoodJobWorker::new_with_config(config).expect("Failed to build benchmark worker pool");
    let (env_net, neural_opt) = build_base_models_with_worker(config, seed, &worker);
    (env_net, neural_opt, worker)
}

fn bench_sized_config(base_config: &Config) -> Config {
    let mut cfg = base_config.clone();
    cfg.fast_init = true;
    cfg.luck_mode = LuckMode::Ppo;
    cfg.policy_eval_interval = 0;
    cfg.model_dim = crate::neural::DIM;
    cfg.model_hidden_dim = cfg.model_hidden_dim.clamp(32, 64);
    cfg.model_num_layers = cfg.model_num_layers.clamp(1, 2);
    cfg.model_num_heads = cfg.model_num_heads.clamp(1, 4).min(cfg.model_hidden_dim);
    while !cfg.model_hidden_dim.is_multiple_of(cfg.model_num_heads) {
        cfg.model_num_heads -= 1;
    }
    cfg.model_kv_lora_rank = cfg
        .model_kv_lora_rank
        .clamp(1, cfg.model_hidden_dim.min(16));
    cfg.model_qk_rope_dim = cfg
        .model_qk_rope_dim
        .clamp(2, (cfg.model_hidden_dim / cfg.model_num_heads).clamp(2, 4));
    if !cfg.model_qk_rope_dim.is_multiple_of(2) {
        cfg.model_qk_rope_dim -= 1;
    }
    cfg.multi_stream_factor = cfg.multi_stream_factor.clamp(1, 2);
    cfg.achf.prune_threshold = cfg.achf.prune_threshold.min(0.005);
    cfg.achf.cache_min_reuse = 0;
    cfg
}

fn validate_candidate_memory(label: &str, memory: Option<AchfMemoryStats>) {
    if let Some(memory) = memory {
        assert!(
            memory.layers > 0
                && memory.candidate_total_weights > 0
                && memory.candidate_nonzero_weights > 0,
            "benchmark condition '{label}' produced an absent or all-zero ACHF candidate; lower prune_threshold or fix projection/pruning before using the data"
        );
        let relative_error = memory.candidate_relative_error().unwrap_or_else(|| {
            panic!("benchmark condition '{label}' has no aggregate candidate-error diagnostic")
        });
        assert!(
            relative_error.is_finite()
                && relative_error <= 1.0 + f64::EPSILON
                && memory.max_layer_candidate_relative_error.is_finite()
                && memory.max_layer_candidate_relative_error <= 1.0 + f64::EPSILON,
            "benchmark condition '{label}' has invalid candidate error: aggregate={relative_error}, max_layer={}",
            memory.max_layer_candidate_relative_error
        );
        if memory.eligible_candidate_layers < memory.candidate_layers {
            eprintln!(
                "[Bench Warning] condition '{label}' materialized {} candidate layers but only {} satisfy production entry criteria; fixed modes are diagnostic overrides and normal inference falls back to reference",
                memory.candidate_layers, memory.eligible_candidate_layers
            );
        }
    }
}

struct ThroughputParams<'a> {
    neural_opt: &'a NeuralLuckOptimizer,
    dqn: Option<&'a DuelingQNetwork>,
    ppo: Option<&'a ActorCritic>,
    env_net: &'a EnvNet,
    config: &'a Config,
    sims: usize,
    pulls: usize,
}

fn measure_inference_throughput(seed: u64, params: &ThroughputParams<'_>) -> f64 {
    let ctx = SimModelContext {
        neural_opt: params.neural_opt,
        dqn_policy: params.dqn,
        ppo_policy: params.ppo,
        env_net: params.env_net,
        config: params.config,
        exp_sender: None,
        neural_sender: None,
        ppo_sender: None,
    };
    let warmup = THROUGHPUT_WARMUP_SIMS;
    let pb = crate::utils::create_bar(warmup as u64, "Warming throughput benchmark");
    for i in 0..warmup {
        let mut rng = Rng::from_seed(derive_seed(seed, i as u64));
        std::hint::black_box(simulate_fast(params.pulls, &mut rng, 0, &ctx));
        pb.inc(1);
        if i == 0 {
            pb.set_message(format!("warmup {}/{}", i + 1, warmup));
        }
    }
    pb.finish_and_clear();
    let start = Instant::now();
    for i in 0..params.sims {
        let mut rng = Rng::from_seed(derive_seed(seed, (warmup + i) as u64));
        std::hint::black_box(simulate_fast(params.pulls, &mut rng, 0, &ctx));
    }
    let elapsed = start.elapsed();
    params.sims as f64 / elapsed.as_secs_f64()
}

fn cache_stats_delta(before: AchfCacheStats, after: AchfCacheStats) -> AchfCacheStats {
    let delta = |name: &str, before: u64, after: u64| {
        after.checked_sub(before).unwrap_or_else(|| {
            panic!("ACHF counter {name} decreased during frozen evaluation: {before} -> {after}")
        })
    };
    AchfCacheStats {
        calls: delta("calls", before.calls, after.calls),
        cache_hits: delta("cache_hits", before.cache_hits, after.cache_hits),
        cache_misses: delta("cache_misses", before.cache_misses, after.cache_misses),
        cache_skips: delta("cache_skips", before.cache_skips, after.cache_skips),
        memo_hits: delta("memo_hits", before.memo_hits, after.memo_hits),
        reference_paths: delta(
            "reference_paths",
            before.reference_paths,
            after.reference_paths,
        ),
        candidate_paths: delta(
            "candidate_paths",
            before.candidate_paths,
            after.candidate_paths,
        ),
        candidate_rejections: delta(
            "candidate_rejections",
            before.candidate_rejections,
            after.candidate_rejections,
        ),
        sparse_paths: delta("sparse_paths", before.sparse_paths, after.sparse_paths),
        dense_paths: delta("dense_paths", before.dense_paths, after.dense_paths),
        ema_cached_ns: after.ema_cached_ns,
        ema_cached_long_ns: after.ema_cached_long_ns,
        ema_sparse_ns: after.ema_sparse_ns,
        ema_sparse_long_ns: after.ema_sparse_long_ns,
        ema_dense_ns: after.ema_dense_ns,
        ema_dense_long_ns: after.ema_dense_long_ns,
        decision_ema_ns: after.decision_ema_ns,
        decision_ema_long_ns: after.decision_ema_long_ns,
        cached_cold_ema_ns: after.cached_cold_ema_ns,
        cached_warm_ema_ns: after.cached_warm_ema_ns,
        sparse_cold_ema_ns: after.sparse_cold_ema_ns,
        sparse_warm_ema_ns: after.sparse_warm_ema_ns,
        dense_cold_ema_ns: after.dense_cold_ema_ns,
        dense_warm_ema_ns: after.dense_warm_ema_ns,
        cached_warmness: after.cached_warmness,
        sparse_warmness: after.sparse_warmness,
        dense_warmness: after.dense_warmness,
        cached_stale_age: after.cached_stale_age,
        sparse_stale_age: after.sparse_stale_age,
        dense_stale_age: after.dense_stale_age,
        path_switches: delta("path_switches", before.path_switches, after.path_switches),
        path_probes: delta("path_probes", before.path_probes, after.path_probes),
        adaptive_bias: after.adaptive_bias,
        latency_samples: delta(
            "latency_samples",
            before.latency_samples,
            after.latency_samples,
        ),
        dense_latency_samples: delta(
            "dense_latency_samples",
            before.dense_latency_samples,
            after.dense_latency_samples,
        ),
        decision_samples: delta(
            "decision_samples",
            before.decision_samples,
            after.decision_samples,
        ),
    }
}

fn aggregate_model_cache_stats(
    config: &Config,
    dqn: Option<&DuelingQNetwork>,
    ppo: Option<&ActorCritic>,
) -> Option<AchfCacheStats> {
    if !config.achf.enabled {
        return None;
    }
    let dqn_stats = dqn.and_then(DuelingQNetwork::achf_cache_stats);
    let ppo_stats = ppo.map(ActorCritic::achf_cache_stats_aggregate);
    Some(aggregate_cache_stats_iter(
        dqn_stats.into_iter().chain(ppo_stats),
    ))
}

fn should_run(bench_cfg: &BenchConfig, name: &str) -> bool {
    match &bench_cfg.only {
        None => true,
        Some(list) => list.iter().any(|s| s.eq_ignore_ascii_case(name)),
    }
}

pub fn parse_chart_format(value: &str) -> Result<ChartFormat, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "svg" => Ok(ChartFormat::Svg),
        "png" => Ok(ChartFormat::Png),
        other => Err(format!(
            "unsupported benchmark chart format '{other}', expected svg or png"
        )),
    }
}

pub fn parse_only_filter(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn validate_bench_config(bench_cfg: &BenchConfig) -> Result<(), String> {
    if bench_cfg.num_trials == 0 {
        return Err("benchmark trials must be at least 1".to_string());
    }
    validate_only_filter(bench_cfg.only.as_deref())
}

fn validate_only_filter(only: Option<&[String]>) -> Result<(), String> {
    let Some(only) = only else {
        return Ok(());
    };
    let unknown: Vec<&str> = only
        .iter()
        .map(String::as_str)
        .filter(|name| {
            !BENCH_EXPERIMENTS
                .iter()
                .any(|known| name.eq_ignore_ascii_case(known))
        })
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "unknown benchmark experiment(s): {}. Expected one or more of: {}",
            unknown.join(", "),
            BENCH_EXPERIMENTS.join(", ")
        ));
    }
    Ok(())
}

fn ext(fmt: &ChartFormat) -> &'static str {
    match fmt {
        ChartFormat::Svg => "svg",
        ChartFormat::Png => "png",
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn sha256_hex(data: &[u8]) -> String {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    state
        .iter()
        .map(|word| format!("{word:08x}"))
        .collect::<String>()
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    fs::read(path).map(|bytes| sha256_hex(&bytes))
}

fn command_output(program: &str, args: &[&str], cwd: Option<&Path>) -> Option<String> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn find_repo_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    for candidate in candidates {
        for ancestor in candidate.ancestors() {
            if ancestor.join(".git").exists() {
                return Some(ancestor.to_path_buf());
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn total_physical_memory_bytes() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    let success = unsafe { GlobalMemoryStatusEx(&mut status) };
    (success != 0 && status.total_phys > 0).then_some(status.total_phys)
}

fn hardware_snapshot() -> serde_json::Value {
    #[cfg(target_os = "windows")]
    {
        let memory_bytes = total_physical_memory_bytes();
        let script = "$cpuKey=[Microsoft.Win32.Registry]::LocalMachine.OpenSubKey('HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0');$cvKey=[Microsoft.Win32.Registry]::LocalMachine.OpenSubKey('SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion');[ordered]@{cpu_name=([string]$cpuKey.GetValue('ProcessorNameString')).Trim();logical_processors=[Environment]::ProcessorCount;memory_bytes=[GC]::GetGCMemoryInfo().TotalAvailableMemoryBytes;os_product_name=[string]$cvKey.GetValue('ProductName');os_display_version=[string]$cvKey.GetValue('DisplayVersion');os_build=('{0}.{1}' -f $cvKey.GetValue('CurrentBuildNumber'),$cvKey.GetValue('UBR'))}|ConvertTo-Json -Compress";
        let script = script.replace(
            "memory_bytes=[GC]::GetGCMemoryInfo().TotalAvailableMemoryBytes;",
            "",
        );
        if let Some(output) = command_output(
            "powershell.exe",
            &["-NoProfile", "-NonInteractive", "-Command", &script],
            None,
        ) {
            if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&output) {
                value["memory_bytes"] = serde_json::json!(memory_bytes);
                value["memory_source"] = serde_json::json!("GlobalMemoryStatusEx");
                return value;
            }
        }
        serde_json::json!({
            "cpu_name": std::env::var("PROCESSOR_IDENTIFIER").ok(),
            "logical_processors": std::thread::available_parallelism().map_or(1, usize::from),
            "memory_bytes": memory_bytes,
            "memory_source": "GlobalMemoryStatusEx",
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        })
    }
    #[cfg(not(target_os = "windows"))]
    serde_json::json!({
        "cpu_name": std::env::var("PROCESSOR_IDENTIFIER").ok(),
        "logical_processors": std::thread::available_parallelism().map_or(1, usize::from),
        "memory_bytes": serde_json::Value::Null,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
    })
}

fn cache_artifact_snapshot() -> Vec<serde_json::Value> {
    const NAMES: &[&str] = &[
        "env_net.cache",
        "neural.cache",
        "dqn.cache.bin",
        "ppo.cache.bin",
        "dqn.cache.bf16.bin",
        "ppo.cache.bf16.bin",
    ];
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    roots.sort();
    roots.dedup();
    roots
        .iter()
        .flat_map(|root| {
            NAMES.iter().map(move |name| {
                let path = root.join(name);
                let metadata = fs::metadata(&path).ok();
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "exists": metadata.is_some(),
                    "size_bytes": metadata.map(|value| value.len()),
                })
            })
        })
        .collect()
}

fn build_run_manifest(
    base_config: &Config,
    seed: u64,
    bench_cfg: &BenchConfig,
) -> serde_json::Value {
    let repo_root = find_repo_root();
    let git = repo_root.as_deref().map(|root| {
        let commit = command_output("git", &["rev-parse", "HEAD"], Some(root));
        let status = command_output(
            "git",
            &["status", "--porcelain=v1", "--untracked-files=normal"],
            Some(root),
        )
        .unwrap_or_default();
        let tracked_status = command_output(
            "git",
            &["status", "--porcelain=v1", "--untracked-files=no"],
            Some(root),
        )
        .unwrap_or_default();
        serde_json::json!({
            "repo_root": root.to_string_lossy(),
            "commit": commit,
            "worktree_dirty": !status.is_empty(),
            "tracked_worktree_dirty": !tracked_status.is_empty(),
            "source_reproducible_from_commit": tracked_status.is_empty(),
            "worktree_status": status.lines().collect::<Vec<_>>(),
        })
    });
    let executable = std::env::current_exe().ok();
    let executable_metadata = executable
        .as_deref()
        .and_then(|path| fs::metadata(path).ok());
    let executable_hash = executable
        .as_deref()
        .and_then(|path| sha256_file(path).ok());
    let config_snapshot =
        serde_json::to_value(base_config).expect("benchmark Config must be serializable");
    let config_bytes =
        serde_json::to_vec(&config_snapshot).expect("benchmark Config JSON must serialize");
    let cargo_config = repo_root
        .as_deref()
        .map(|root| root.join(".cargo").join("config.toml"))
        .filter(|path| path.exists());
    let chart_format = match bench_cfg.format {
        ChartFormat::Svg => "svg",
        ChartFormat::Png => "png",
    };
    serde_json::json!({
        "schema_version": 1,
        "status": "running",
        "started_unix_ms": unix_time_ms(),
        "run": {
            "command": std::env::args().collect::<Vec<_>>(),
            "output_dir": bench_cfg.output_dir,
            "seed": seed,
            "num_trials": bench_cfg.num_trials,
            "selected_experiments": bench_cfg.only.clone().unwrap_or_else(|| {
                BENCH_EXPERIMENTS.iter().map(|name| (*name).to_string()).collect()
            }),
            "chart_format": chart_format,
            "trial_unit": "independent deterministic seed within one operating-system process",
            "independent_process_repetitions": 1,
            "condition_order": "rotating interleaved within each trial",
        },
        "source": git,
        "build": {
            "package_name": env!("CARGO_PKG_NAME"),
            "package_version": env!("CARGO_PKG_VERSION"),
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "debug_assertions": cfg!(debug_assertions),
            "target_os": std::env::consts::OS,
            "target_arch": std::env::consts::ARCH,
            "features": {
                "cuda": cfg!(feature = "cuda"),
                "python": cfg!(feature = "python"),
            },
            "rustc": command_output("rustc", &["--version", "--verbose"], None),
            "cargo": command_output("cargo", &["--version"], None),
            "rustflags_env": std::env::var("RUSTFLAGS").ok(),
            "cargo_config_path": cargo_config.as_ref().map(|path| path.to_string_lossy()),
            "cargo_config_sha256": cargo_config.as_deref().and_then(|path| sha256_file(path).ok()),
            "executable_path": executable.as_ref().map(|path| path.to_string_lossy()),
            "executable_size_bytes": executable_metadata.map(|value| value.len()),
            "executable_sha256": executable_hash,
        },
        "hardware": hardware_snapshot(),
        "environment": {
            "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(),
            "number_of_processors": std::env::var("NUMBER_OF_PROCESSORS").ok(),
        },
        "configuration": {
            "sha256": sha256_hex(&config_bytes),
            "snapshot": config_snapshot,
        },
        "cache_policy": {
            "base_model_cache_used": false,
            "note": "paper benchmark rebuilds EnvNet and policy models from domain-separated seeds",
            "cache_files_present_but_ignored": cache_artifact_snapshot(),
        },
        "statistical_contract": {
            "confidence_level": 0.95,
            "confidence_interval": "two-sided Student-t over trial-level values",
            "paired_conditions": true,
            "microbenchmark_ci_unit": "per-trial warmed mean, never pooled inner-loop samples",
        },
    })
}

fn is_generated_benchmark_file(name: &str) -> bool {
    matches!(
        name,
        "summary.txt"
            | "summary.json"
            | "run_manifest.json"
            | "ablation.csv"
            | "mode.csv"
            | "scale.csv"
            | "apply.csv"
            | "convergence.csv"
            | "path_latency.csv"
            | "path_latency_summary.json"
            | "gate_curve.csv"
            | "gate_curve_summary.json"
            | "path_crossover.csv"
            | "path_crossover_summary.json"
            | "regime_adaptation.csv"
            | "regime_adaptation_summary.json"
    ) || [
        "ablation_throughput.",
        "ablation_reward.",
        "mode_comparison.",
        "path_latency_boxplot.",
        "gate_curve.",
        "scale_test.",
        "apply_combination.",
        "convergence_loss.",
        "convergence_reward.",
        "regime_adaptation.",
        "path_crossover_dim",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn prepare_output_directory(dir: &str) {
    fs::create_dir_all(dir).expect("failed to create benchmark output directory");
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to inspect benchmark output directory {dir}: {err}"))
        .flatten()
    {
        if entry.file_type().is_ok_and(|file_type| file_type.is_file()) {
            let name = entry.file_name();
            if is_generated_benchmark_file(&name.to_string_lossy()) {
                fs::remove_file(entry.path()).unwrap_or_else(|err| {
                    panic!(
                        "failed to remove stale benchmark artifact {}: {err}",
                        entry.path().display()
                    )
                });
            }
        }
    }
}

fn write_run_manifest(dir: &str, manifest: &serde_json::Value) {
    let json = serde_json::to_string_pretty(manifest)
        .expect("run manifest should always be JSON serializable");
    write_text_file(&format!("{dir}/run_manifest.json"), &json);
}

fn finalize_run_manifest(
    dir: &str,
    manifest: &mut serde_json::Value,
    elapsed: std::time::Duration,
) {
    let mut artifacts = Vec::new();
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to enumerate benchmark artifacts: {err}"))
        .flatten()
    {
        if !entry.file_type().is_ok_and(|file_type| file_type.is_file())
            || entry.file_name() == "run_manifest.json"
        {
            continue;
        }
        let path = entry.path();
        let metadata = fs::metadata(&path).ok();
        artifacts.push(serde_json::json!({
            "file": entry.file_name().to_string_lossy(),
            "size_bytes": metadata.map(|value| value.len()),
            "sha256": sha256_file(&path).ok(),
        }));
    }
    artifacts.sort_by(|left, right| left["file"].as_str().cmp(&right["file"].as_str()));
    manifest["status"] = serde_json::json!("complete");
    manifest["completed_unix_ms"] = serde_json::json!(unix_time_ms());
    manifest["duration_seconds"] = serde_json::json!(elapsed.as_secs_f64());
    manifest["artifacts"] = serde_json::Value::Array(artifacts);
    write_run_manifest(dir, manifest);
}

// ── Main entry point ────────────────────────────────────────────────────

pub fn run_achf_benchmarks(base_config: &Config, seed: u64, bench_cfg: &BenchConfig) {
    validate_bench_config(bench_cfg).unwrap_or_else(|err| panic!("{err}"));

    let dir = &bench_cfg.output_dir;
    let nt = bench_cfg.num_trials;
    prepare_output_directory(dir);
    let run_started = Instant::now();
    let mut run_manifest = build_run_manifest(base_config, seed, bench_cfg);
    write_run_manifest(dir, &run_manifest);

    println!("\n========================================");
    println!("  ACHF Benchmark Suite");
    println!("  Trials per experiment: {}", nt);
    println!("========================================\n");

    let mut all_agg: Vec<(&str, Vec<AggregatedResult>)> = Vec::new();
    let mut path_latencies: Option<Vec<PathLatencyResult>> = None;
    let mut gate_curve: Option<AggregatedResult> = None;
    let mut crossover: Option<Vec<CrossoverCell>> = None;
    let mut regime: Option<Vec<RegimeRow>> = None;

    if should_run(bench_cfg, "ablation") {
        let agg = run_ablation(base_config, seed, nt);
        print_agg_summary("Ablation (dense/static pruning/full ACHF)", &agg);
        let e = ext(&bench_cfg.format);
        chart_ablation(&agg, dir, e);
        all_agg.push(("ablation", agg));
    }

    if should_run(bench_cfg, "mode") {
        let agg = run_mode_comparison(base_config, seed, nt);
        print_agg_summary("Mode Comparison (fixed/plain EMA/guarded AMA)", &agg);
        let e = ext(&bench_cfg.format);
        chart_mode(&agg, dir, e);
        all_agg.push(("mode", agg));
    }

    if should_run(bench_cfg, "path") {
        let latencies = run_path_comparison(base_config, seed, nt);
        println!("[Bench] Path Comparison complete.");
        for stats in path_latency_stats(&latencies) {
            println!(
                "  {}: avg {:.1} ns across {} trials ({} samples)",
                stats.label, stats.mean_ns, stats.trials, stats.samples
            );
        }
        let e = ext(&bench_cfg.format);
        chart_path_latency(&latencies, dir, e);
        write_path_latency_outputs(&latencies, dir);
        path_latencies = Some(latencies);
    }

    if should_run(bench_cfg, "gate") {
        let result = run_gate_curve(base_config, seed, nt);
        println!(
            "[Bench] Gate Curve: {} aggregated points over {} steps",
            result.curve.len(),
            result.curve.last().map_or(0, |point| point.step)
        );
        let e = ext(&bench_cfg.format);
        chart_gate_curve(&result, dir, e);
        write_gate_curve_outputs(&result, dir);
        gate_curve = Some(result);
    }

    if should_run(bench_cfg, "scale") {
        let agg = run_scale_test(base_config, seed, nt);
        print_agg_summary("Scale Test (varying rank)", &agg);
        let e = ext(&bench_cfg.format);
        chart_scale(&agg, dir, e);
        all_agg.push(("scale", agg));
    }

    if should_run(bench_cfg, "apply") {
        let agg = run_apply_combination(base_config, seed, nt);
        print_agg_summary("Apply Combination", &agg);
        let e = ext(&bench_cfg.format);
        chart_apply(&agg, dir, e);
        all_agg.push(("apply", agg));
    }

    if should_run(bench_cfg, "convergence") {
        let agg = run_convergence(base_config, seed, nt);
        print_agg_summary("Convergence (loss curve)", &agg);
        let e = ext(&bench_cfg.format);
        chart_convergence(&agg, dir, e);
        all_agg.push(("convergence", agg));
    }

    if should_run(bench_cfg, "crossover") {
        let cells = run_path_crossover(seed, nt);
        println!("[Bench] Path Crossover complete ({} cells).", cells.len());
        let e = ext(&bench_cfg.format);
        chart_crossover(&cells, dir, e);
        write_crossover_outputs(&cells, dir);
        crossover = Some(cells);
    }

    if should_run(bench_cfg, "regime") {
        let rows = run_regime_adaptation(seed, nt);
        println!("[Bench] Regime Adaptation complete ({} rows).", rows.len());
        let e = ext(&bench_cfg.format);
        chart_regime(&rows, dir, e);
        write_regime_outputs(&rows, dir);
        regime = Some(rows);
    }

    write_summary_txt(
        &all_agg,
        path_latencies.as_deref(),
        gate_curve.as_ref(),
        crossover.as_deref(),
        regime.as_deref(),
        nt,
        dir,
    );
    write_summary_json(
        &all_agg,
        path_latencies.as_deref(),
        gate_curve.as_ref(),
        crossover.as_deref(),
        regime.as_deref(),
        (seed, nt),
        dir,
    );
    write_csvs(&all_agg, dir);
    finalize_run_manifest(dir, &mut run_manifest, run_started.elapsed());

    println!("\n========================================");
    println!("  All benchmarks complete.");
    println!("  Output: {}/", dir);
    println!("========================================");
}

// ── Experiment implementations ──────────────────────────────────────────

fn run_interleaved_conditions(
    conditions: &[(String, Config)],
    seed: u64,
    num_trials: usize,
) -> Vec<Vec<BenchRunResult>> {
    assert!(
        !conditions.is_empty(),
        "benchmark needs at least one condition"
    );
    let mut runs: Vec<Vec<BenchRunResult>> = (0..conditions.len())
        .map(|_| Vec::with_capacity(num_trials))
        .collect();
    for trial in 0..num_trials {
        let trial_seed = benchmark_trial_seed(seed, trial);
        let start = trial % conditions.len();
        for offset in 0..conditions.len() {
            let condition_index = (start + offset) % conditions.len();
            let (label, config) = &conditions[condition_index];
            println!("  [{label}] trial {}/{}", trial + 1, num_trials);
            let result = train_and_measure(label, config, trial_seed);
            println!(
                "    {:.1}s | {:.0} sims/sec | eval reward: {:.3} | train loss: {:.4}",
                result.train_time_ms / 1000.0,
                result.throughput_sims_per_sec,
                result.eval_reward,
                result.train_loss
            );
            runs[condition_index].push(result);
        }
    }
    runs
}

fn aggregate_conditions(
    runs: &[Vec<BenchRunResult>],
    baseline_for_condition: &[usize],
) -> Vec<AggregatedResult> {
    assert_eq!(runs.len(), baseline_for_condition.len());
    let mut aggregated: Vec<AggregatedResult> = runs
        .iter()
        .map(|condition| aggregate_trials(condition))
        .collect();
    for condition_index in 0..aggregated.len() {
        let baseline_index = baseline_for_condition[condition_index];
        assert!(baseline_index < runs.len());
        if condition_index == baseline_index {
            continue;
        }
        let candidate = &runs[condition_index];
        let baseline = &runs[baseline_index];
        assert_eq!(
            candidate.len(),
            baseline.len(),
            "paired benchmark conditions have different trial counts"
        );
        let paired = |value: fn(&BenchRunResult) -> f64| {
            TrialStats::from_values(
                &candidate
                    .iter()
                    .zip(baseline.iter())
                    .map(|(candidate, baseline)| value(candidate) - value(baseline))
                    .collect::<Vec<_>>(),
            )
        };
        let throughput_relative_delta_pct = TrialStats::from_values(
            &candidate
                .iter()
                .zip(baseline.iter())
                .map(|(candidate, baseline)| {
                    (candidate.throughput_sims_per_sec / baseline.throughput_sims_per_sec - 1.0)
                        * 100.0
                })
                .collect::<Vec<_>>(),
        );
        aggregated[condition_index].paired = Some(PairedComparison {
            baseline: aggregated[baseline_index].label.clone(),
            throughput_delta: paired(|run| run.throughput_sims_per_sec),
            throughput_relative_delta_pct,
            eval_reward_delta: paired(|run| run.eval_reward),
            train_loss_delta: paired(|run| run.train_loss),
            train_time_ms_delta: paired(|run| run.train_time_ms),
        });
    }
    aggregated
}

fn cap_ppo_training(config: &mut Config, max_steps: usize) {
    config.ppo_total_steps = config.ppo_total_steps.min(max_steps);
    config.ppo_steps_per_update = config.ppo_steps_per_update.min(256);
    config.ppo_k_epochs = config.ppo_k_epochs.min(2);
}

fn run_static_pruning_pair(
    training_config: &Config,
    dense_config: &Config,
    static_config: &Config,
    trial_seed: u64,
    trial: usize,
    num_trials: usize,
) -> [BenchRunResult; 2] {
    let (env_net, neural_opt, _worker) =
        build_base_models(training_config, derive_seed(trial_seed, SEED_BASE_MODELS));
    let (tx, rx) = std::sync::mpsc::channel();
    let mut train_rng = Rng::from_seed(derive_seed(trial_seed, SEED_PPO_TRAIN));
    let train_start = Instant::now();
    let trained_policy =
        train_ppo_with_metrics(&mut train_rng, &env_net, training_config, Some(tx));
    let train_time_ms = train_start.elapsed().as_secs_f64() * 1000.0;
    let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();
    assert!(
        !snapshots.is_empty(),
        "shared dense/static PPO benchmark emitted no training snapshots"
    );

    let mut results: [Option<BenchRunResult>; 2] = [None, None];
    let start = trial % results.len();
    for offset in 0..results.len() {
        let condition_index = (start + offset) % results.len();
        let (label, config, policy) = if condition_index == 0 {
            let mut policy = trained_policy.fork_inference_runtime();
            policy.disable_achf_runtime();
            ("Dense reference", dense_config, policy)
        } else {
            let mut policy = trained_policy.fork_inference_runtime();
            policy.rebuild_achf_inference_candidates(static_config.achf.prune_threshold);
            policy.set_achf_inference_mode("fixed_sparse", u64::MAX);
            ("Static magnitude pruning", static_config, policy)
        };
        println!(
            "  [{label}] trial {}/{} (shared trained policy)",
            trial + 1,
            num_trials
        );
        let result = measure_trained_ppo(TrainedPpoParams {
            label,
            policy: &policy,
            env_net: &env_net,
            neural_opt: &neural_opt,
            training_config,
            config,
            trial_seed,
            train_time_ms,
            snapshots: &snapshots,
        });
        println!(
            "    shared_train={:.1}s | {:.0} sims/sec | eval reward: {:.3} | train loss: {:.4}",
            train_time_ms / 1000.0,
            result.throughput_sims_per_sec,
            result.eval_reward,
            result.train_loss
        );
        results[condition_index] = Some(result);
    }
    [
        results[0].take().expect("dense result missing"),
        results[1].take().expect("static pruning result missing"),
    ]
}

fn run_ablation(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Ablation Experiment (dense/static pruning/full ACHF)...");
    let mut dense = bench_sized_config(base_config);
    dense.achf.enabled = false;
    cap_ppo_training(&mut dense, 2000);

    let mut static_pruned = bench_sized_config(base_config);
    static_pruned.achf.enabled = true;
    static_pruned.achf.proj_mode = "none".to_string();
    static_pruned.achf.ortho_penalty_freq = 0;
    static_pruned.achf.lambda_ortho = 0.0;
    static_pruned.achf.rank = 0;
    static_pruned.achf.gate_warmup_steps = 0;
    static_pruned.achf.gate_transition_steps = 0;
    static_pruned.achf.g_min = 1.0;
    static_pruned.achf.gate_alpha = 50.0;
    static_pruned.achf.gate_beta = 0.0;
    static_pruned.achf.infer_gate = "one".to_string();
    static_pruned.achf.mode = "fixed_sparse".to_string();
    static_pruned.achf.cache_latency_sample_every = u64::MAX;
    cap_ppo_training(&mut static_pruned, 2000);

    let mut shared_training = static_pruned.clone();
    shared_training.achf.prune_threshold = 0.0;
    shared_training.achf.mode = "fixed_dense".to_string();

    let mut guarded_ama = bench_sized_config(base_config);
    guarded_ama.achf.enabled = true;
    guarded_ama.achf.mode = "full".to_string();
    guarded_ama.achf.adaptive_inference = false;
    cap_ppo_training(&mut guarded_ama, 2000);

    let mut runs: Vec<Vec<BenchRunResult>> = (0..3).map(|_| Vec::with_capacity(nt)).collect();
    for trial in 0..nt {
        let trial_seed = benchmark_trial_seed(seed, trial);
        let (pair, full) = if trial.is_multiple_of(2) {
            let pair = run_static_pruning_pair(
                &shared_training,
                &dense,
                &static_pruned,
                trial_seed,
                trial,
                nt,
            );
            println!("  [Full ACHF (guarded AMA)] trial {}/{}", trial + 1, nt);
            let full = train_and_measure("Full ACHF (guarded AMA)", &guarded_ama, trial_seed);
            (pair, full)
        } else {
            println!("  [Full ACHF (guarded AMA)] trial {}/{}", trial + 1, nt);
            let full = train_and_measure("Full ACHF (guarded AMA)", &guarded_ama, trial_seed);
            let pair = run_static_pruning_pair(
                &shared_training,
                &dense,
                &static_pruned,
                trial_seed,
                trial,
                nt,
            );
            (pair, full)
        };
        println!(
            "    {:.1}s | {:.0} sims/sec | eval reward: {:.3} | train loss: {:.4}",
            full.train_time_ms / 1000.0,
            full.throughput_sims_per_sec,
            full.eval_reward,
            full.train_loss
        );
        runs[0].push(pair[0].clone());
        runs[1].push(pair[1].clone());
        runs[2].push(full);
    }
    aggregate_conditions(&runs, &[0, 0, 0])
}

fn run_mode_comparison(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Mode Comparison (fixed paths/plain EMA/guarded AMA)...");
    let modes = [
        ("Fixed Cached", "fixed_cached"),
        ("Fixed Sparse", "fixed_sparse"),
        ("Fixed Dense", "fixed_dense"),
        ("Plain EMA", "plain_ema"),
        ("Guarded AMA", "full"),
    ];
    let mut runs: Vec<Vec<BenchRunResult>> = modes.iter().map(|_| Vec::with_capacity(nt)).collect();

    for trial in 0..nt {
        let trial_seed = benchmark_trial_seed(seed, trial);
        let mut training_cfg = bench_sized_config(base_config);
        training_cfg.achf.enabled = true;
        training_cfg.achf.mode = "full".to_string();
        training_cfg.achf.adaptive_inference = false;
        cap_ppo_training(&mut training_cfg, 2000);
        let (env_net, neural_opt, _worker) =
            build_base_models(&training_cfg, derive_seed(trial_seed, SEED_BASE_MODELS));
        let (tx, rx) = std::sync::mpsc::channel();
        let mut train_rng = Rng::from_seed(derive_seed(trial_seed, SEED_PPO_TRAIN));
        let train_start = Instant::now();
        let trained_policy =
            train_ppo_with_metrics(&mut train_rng, &env_net, &training_cfg, Some(tx));
        let train_time_ms = train_start.elapsed().as_secs_f64() * 1000.0;
        let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();
        assert!(
            !snapshots.is_empty(),
            "shared PPO mode benchmark emitted no training snapshots"
        );
        validate_candidate_memory(
            "shared mode policy",
            Some(trained_policy.achf_memory_stats_aggregate()),
        );

        let start = trial % modes.len();
        for offset in 0..modes.len() {
            let mode_index = (start + offset) % modes.len();
            let (label, mode) = modes[mode_index];
            println!(
                "  [{label}] trial {}/{} (shared trained policy)",
                trial + 1,
                nt
            );
            let sample_every = if mode.starts_with("fixed_") {
                u64::MAX
            } else {
                training_cfg.achf.cache_latency_sample_every
            };
            let mut cfg = training_cfg.clone();
            cfg.achf.mode = mode.to_string();
            cfg.achf.cache_latency_sample_every = sample_every;
            let mut policy = trained_policy.fork_inference_runtime();
            policy.set_achf_inference_mode(mode, sample_every);
            let result = measure_trained_ppo(TrainedPpoParams {
                label,
                policy: &policy,
                env_net: &env_net,
                neural_opt: &neural_opt,
                training_config: &training_cfg,
                config: &cfg,
                trial_seed,
                train_time_ms,
                snapshots: &snapshots,
            });
            println!(
                "    shared_train={:.1}s | {:.0} sims/sec | eval reward: {:.3}",
                train_time_ms / 1000.0,
                result.throughput_sims_per_sec,
                result.eval_reward
            );
            runs[mode_index].push(result);
        }
    }
    aggregate_conditions(&runs, &[0, 0, 0, 0, 0])
}

fn run_path_comparison(
    base_config: &Config,
    seed: u64,
    num_trials: usize,
) -> Vec<PathLatencyResult> {
    println!("[Bench] Running Path Comparison (Cached/Sparse/Dense)...");
    let labels = ["Cached", "Sparse", "Dense"];
    let mut all_latencies: Vec<PathLatencyResult> = labels
        .iter()
        .map(|label| PathLatencyResult {
            label: (*label).to_string(),
            trial_samples: Vec::with_capacity(num_trials),
            trial_input_dims: Vec::with_capacity(num_trials),
            trial_sparsity: Vec::with_capacity(num_trials),
        })
        .collect();

    for trial in 0..num_trials {
        let trial_seed = benchmark_trial_seed(seed, trial);
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = true;
        cfg.achf.candidate_mode = "sparse".to_string();
        cfg.achf.rank = 0;
        cfg.achf.prune_threshold = PATH_PRUNE_THRESHOLD;
        cap_ppo_training(&mut cfg, 2000);
        let (env_net, _neural_opt, _worker) =
            build_base_models(&cfg, derive_seed(trial_seed, SEED_BASE_MODELS));
        let mut ppo_rng = Rng::from_seed(derive_seed(trial_seed, SEED_PPO_TRAIN));
        let ppo = train_ppo_with_metrics(&mut ppo_rng, &env_net, &cfg, None);
        let (achf, input_dim) = ppo
            .first_achf_layer()
            .expect("path comparison requires an ACHF layer; achf.enabled=true");
        let sparsity = achf
            .inference_sparsity_stats()
            .expect("path comparison ACHF layer must expose frozen sparsity");
        assert!(
            sparsity.nonzero_weights > 0,
            "path comparison requires a nonzero frozen operator, got nnz={}/{}",
            sparsity.nonzero_weights,
            sparsity.total_weights,
        );
        let sample_input: Vec<f32> = (0..input_dim)
            .map(|index| index as f32 * 0.1 + 0.05)
            .collect();
        assert_forced_paths_agree(achf, &sample_input);
        let trial_values = measure_forced_path_samples(
            achf,
            &sample_input,
            PATH_WARMUP_ROUNDS,
            PATH_SAMPLES,
            PATH_CALLS_PER_SAMPLE,
            trial,
        );
        for path_index in 0..labels.len() {
            let stats = TrialStats::from_values(&trial_values[path_index]);
            println!(
                "  trial {}/{} [{}] avg={:.1}ns, p50={:.1}ns, p99={:.1}ns",
                trial + 1,
                num_trials,
                labels[path_index],
                stats.mean,
                percentile(&trial_values[path_index], 50),
                percentile(&trial_values[path_index], 99),
            );
            all_latencies[path_index]
                .trial_samples
                .push(trial_values[path_index].clone());
            all_latencies[path_index].trial_input_dims.push(input_dim);
            all_latencies[path_index].trial_sparsity.push(sparsity);
        }
    }
    all_latencies
}

fn assert_forced_paths_agree(layer: &AchfLayer, input: &[f32]) {
    let cached = layer.forward_inference_forced_path(input, 0);
    let sparse = layer.forward_inference_forced_path(input, 1);
    let dense = layer.forward_inference_forced_path(input, 2);
    assert_eq!(cached.len(), sparse.len());
    assert_eq!(cached.len(), dense.len());
    for (index, ((cached, sparse), dense)) in cached
        .iter()
        .zip(sparse.iter())
        .zip(dense.iter())
        .enumerate()
    {
        assert!(
            (cached - sparse).abs() <= 1e-4 && (cached - dense).abs() <= 1e-4,
            "forced inference paths compute different outputs at index {index}: cached={cached}, sparse={sparse}, dense={dense}"
        );
    }
}

/// Build a frozen square ACHF layer with an EXACT target weight sparsity by
/// zeroing the first `sparsity` fraction of every row (deterministic). The CSR
/// sparse view keys on `w != 0.0`, so this controls the sparse path's FLOP
/// count precisely — unlike magnitude pruning, whose sparsity depends on the
/// random weight distribution. `adaptive` toggles the live AMA selector.
fn build_synthetic_achf_layer(
    dim: usize,
    weight_sparsity: f64,
    adaptive: bool,
    seed: u64,
) -> AchfLayer {
    assert!((0.0..=1.0).contains(&weight_sparsity));
    let cfg = AchfConfig {
        enabled: true,
        mode: if adaptive { "full" } else { "lite" }.to_string(),
        adaptive_inference: false,
        cache_latency_sample_every: 1,
        gate_warmup_steps: 0,
        gate_transition_steps: 0,
        g_min: 0.0,
        infer_gate: "one".to_string(),
        rank: 0,
        proj_mode: "none".to_string(),
        prune_threshold: 0.0,
        // Disable memoization: with a repeated benchmark input it would return a
        // memo hit and bypass the selector/path entirely, invalidating timing.
        cache_min_reuse: 0,
        ..Default::default()
    };
    let mut layer = AchfLayer::new(dim, dim, false, cfg, derive_seed(seed, dim as u64));
    {
        let mut w = layer.weight.weight.data_write_f32();
        let zero_per_row = (dim as f64 * weight_sparsity).floor() as usize;
        for r in 0..dim {
            for c in 0..dim {
                let v = &mut w[r * dim + c];
                if c < zero_per_row {
                    *v = 0.0;
                } else if *v == 0.0 {
                    *v = 0.01; // keep survivors nonzero so CSR nnz is exact
                }
            }
        }
    }
    layer.freeze_for_inference();
    layer
}

fn measure_forced_path_samples(
    layer: &AchfLayer,
    input: &[f32],
    warmup_rounds: usize,
    samples: usize,
    calls_per_sample: usize,
    rotation: usize,
) -> [Vec<f64>; 3] {
    assert!(calls_per_sample > 0);
    for round in 0..warmup_rounds {
        for offset in 0..3 {
            let path = (rotation + round + offset) % 3;
            std::hint::black_box(
                layer.forward_inference_forced_path(std::hint::black_box(input), path as u8),
            );
        }
    }
    let mut values: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(samples));
    for sample in 0..samples {
        for offset in 0..3 {
            let path = (rotation + sample + offset) % 3;
            let start = Instant::now();
            for _ in 0..calls_per_sample {
                std::hint::black_box(
                    layer.forward_inference_forced_path(std::hint::black_box(input), path as u8),
                );
            }
            values[path].push(start.elapsed().as_nanos() as f64 / calls_per_sample as f64);
        }
    }
    values
}

/// Path-crossover experiment: sweep (dim x weight_sparsity), force each path,
/// and record which is fastest. Demonstrates that no single fixed path wins
/// everywhere — the premise that makes adaptive path selection worthwhile.
fn run_path_crossover(seed: u64, num_trials: usize) -> Vec<CrossoverCell> {
    println!("[Bench] Running Path Crossover (dim x weight-sparsity)...");
    let specs: Vec<(usize, f64)> = CROSSOVER_DIMS
        .iter()
        .flat_map(|&dim| {
            CROSSOVER_SPARSITIES
                .iter()
                .map(move |&sparsity| (dim, sparsity))
        })
        .collect();
    let mut measurements: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = specs
        .iter()
        .map(|_| {
            (
                Vec::with_capacity(num_trials),
                Vec::with_capacity(num_trials),
                Vec::with_capacity(num_trials),
            )
        })
        .collect();
    let mut shapes: Vec<Option<(usize, usize, f64)>> = vec![None; specs.len()];

    for trial in 0..num_trials {
        let trial_seed = benchmark_trial_seed(seed, trial);
        let start = trial % specs.len();
        for offset in 0..specs.len() {
            let cell_index = (start + offset) % specs.len();
            let (dim, requested_sparsity) = specs[cell_index];
            let layer = build_synthetic_achf_layer(
                dim,
                requested_sparsity,
                false,
                derive_seed(trial_seed, cell_index as u64),
            );
            let sparsity = layer
                .inference_sparsity_stats()
                .expect("synthetic ACHF layer must expose frozen sparsity");
            let shape = (
                sparsity.total_weights,
                sparsity.nonzero_weights,
                sparsity.sparsity,
            );
            if let Some(expected) = shapes[cell_index] {
                assert_eq!(shape, expected, "synthetic sparsity changed across trials");
            } else {
                shapes[cell_index] = Some(shape);
            }
            let input: Vec<f32> = (0..dim * CROSSOVER_BATCH)
                .map(|index| ((index % 7) as f32) * 0.1 + 0.05)
                .collect();
            assert_forced_paths_agree(&layer, &input);
            let values = measure_forced_path_samples(
                &layer,
                &input,
                CROSSOVER_WARMUP_ROUNDS,
                CROSSOVER_SAMPLES,
                1,
                trial + cell_index,
            );
            let means = values.map(|path| path.iter().sum::<f64>() / path.len() as f64);
            measurements[cell_index].0.push(means[0]);
            measurements[cell_index].1.push(means[1]);
            measurements[cell_index].2.push(means[2]);
            println!(
                "  trial={:<2} dim={dim:<5} requested={requested_sparsity:<5.2} actual={:.4} \
                 nnz={}/{} cached={:>9.0}ns sparse={:>9.0}ns dense={:>9.0}ns",
                trial + 1,
                sparsity.sparsity,
                sparsity.nonzero_weights,
                sparsity.total_weights,
                means[0],
                means[1],
                means[2],
            );
        }
    }

    specs
        .into_iter()
        .enumerate()
        .map(|(cell_index, (dim, requested_sparsity))| {
            let (total_weights, nonzero_weights, actual_sparsity) =
                shapes[cell_index].expect("crossover cell was not measured");
            let cached_ns = TrialStats::from_values(&measurements[cell_index].0);
            let sparse_ns = TrialStats::from_values(&measurements[cell_index].1);
            let dense_ns = TrialStats::from_values(&measurements[cell_index].2);
            let paired_delta = |left: &[f64], right: &[f64]| {
                assert_eq!(left.len(), right.len());
                TrialStats::from_values(
                    &left
                        .iter()
                        .zip(right.iter())
                        .map(|(left, right)| left - right)
                        .collect::<Vec<_>>(),
                )
            };
            let cached_minus_sparse_ns =
                paired_delta(&measurements[cell_index].0, &measurements[cell_index].1);
            let cached_minus_dense_ns =
                paired_delta(&measurements[cell_index].0, &measurements[cell_index].2);
            let sparse_minus_dense_ns =
                paired_delta(&measurements[cell_index].1, &measurements[cell_index].2);
            let winner = if sparse_ns.mean <= cached_ns.mean && sparse_ns.mean <= dense_ns.mean {
                "Sparse"
            } else if cached_ns.mean <= dense_ns.mean {
                "Cached"
            } else {
                "Dense"
            };
            let significant_winner_95 = match (
                cached_minus_sparse_ns.ci_low,
                cached_minus_sparse_ns.ci_high,
                cached_minus_dense_ns.ci_low,
                cached_minus_dense_ns.ci_high,
                sparse_minus_dense_ns.ci_low,
                sparse_minus_dense_ns.ci_high,
            ) {
                (_, Some(cached_sparse_high), _, Some(cached_dense_high), _, _)
                    if cached_sparse_high < 0.0 && cached_dense_high < 0.0 =>
                {
                    Some("Cached".to_string())
                }
                (Some(cached_sparse_low), _, _, _, _, Some(sparse_dense_high))
                    if cached_sparse_low > 0.0 && sparse_dense_high < 0.0 =>
                {
                    Some("Sparse".to_string())
                }
                (_, _, Some(cached_dense_low), _, Some(sparse_dense_low), _)
                    if cached_dense_low > 0.0 && sparse_dense_low > 0.0 =>
                {
                    Some("Dense".to_string())
                }
                _ => None,
            };
            CrossoverCell {
                dim,
                requested_sparsity,
                actual_sparsity,
                total_weights,
                nonzero_weights,
                cached_ns,
                sparse_ns,
                dense_ns,
                winner: winner.to_string(),
                significant_winner_95,
                cached_minus_sparse_ns,
                cached_minus_dense_ns,
                sparse_minus_dense_ns,
            }
        })
        .collect()
}

/// Regime-adaptation experiment: on ONE fixed frozen layer, run the LIVE
/// adaptive selector at a small batch (decode-like) then a large batch
/// (prefill-like) and record how often it chose the sparse path in each. When
/// the small-batch sparse fraction exceeds the large-batch one, the selector is
/// adapting its path choice to the operating point — the core "true adaptive"
/// claim. Batch-bucketed latency EMAs are what make this possible.
fn run_regime_adaptation(seed: u64, num_trials: usize) -> Vec<RegimeRow> {
    println!("[Bench] Running Regime Adaptation (batch-driven path switching)...");
    let mut trials: Vec<(Vec<RegimeTrialLatency>, Vec<RegimeTrialLatency>)> = REGIME_SPARSITIES
        .iter()
        .map(|_| {
            (
                Vec::with_capacity(num_trials),
                Vec::with_capacity(num_trials),
            )
        })
        .collect();
    let mut shapes: Vec<Option<(usize, usize, f64)>> = vec![None; REGIME_SPARSITIES.len()];

    for trial in 0..num_trials {
        let trial_seed = benchmark_trial_seed(seed, trial);
        let start = trial % REGIME_SPARSITIES.len();
        for offset in 0..REGIME_SPARSITIES.len() {
            let sparsity_index = (start + offset) % REGIME_SPARSITIES.len();
            let requested_sparsity = REGIME_SPARSITIES[sparsity_index];
            let layer = build_synthetic_achf_layer(
                REGIME_DIM,
                requested_sparsity,
                true,
                derive_seed(trial_seed, sparsity_index as u64),
            );
            let mut plain_ema_layer = build_synthetic_achf_layer(
                REGIME_DIM,
                requested_sparsity,
                true,
                derive_seed(trial_seed, sparsity_index as u64),
            );
            plain_ema_layer.config.mode = "plain_ema".to_string();
            let sparsity = layer
                .inference_sparsity_stats()
                .expect("synthetic ACHF layer must expose frozen sparsity");
            let shape = (
                sparsity.total_weights,
                sparsity.nonzero_weights,
                sparsity.sparsity,
            );
            if let Some(expected) = shapes[sparsity_index] {
                assert_eq!(shape, expected, "synthetic sparsity changed across trials");
            } else {
                shapes[sparsity_index] = Some(shape);
            }
            let validation_input: Vec<f32> = (0..REGIME_DIM)
                .map(|index| ((index % 7) as f32) * 0.1 + 0.05)
                .collect();
            assert_forced_paths_agree(&layer, &validation_input);
            let (small, large) = if (trial + sparsity_index).is_multiple_of(2) {
                (
                    measure_regime(
                        &layer,
                        &plain_ema_layer,
                        REGIME_DIM,
                        REGIME_SMALL_BATCH,
                        trial + sparsity_index,
                    ),
                    measure_regime(
                        &layer,
                        &plain_ema_layer,
                        REGIME_DIM,
                        REGIME_LARGE_BATCH,
                        trial + sparsity_index + 1,
                    ),
                )
            } else {
                let large = measure_regime(
                    &layer,
                    &plain_ema_layer,
                    REGIME_DIM,
                    REGIME_LARGE_BATCH,
                    trial + sparsity_index,
                );
                let small = measure_regime(
                    &layer,
                    &plain_ema_layer,
                    REGIME_DIM,
                    REGIME_SMALL_BATCH,
                    trial + sparsity_index + 1,
                );
                (small, large)
            };
            println!(
                "  trial={:<2} requested={requested_sparsity:<5.2} actual={:.4} \
                 b{:<3}: AMA={:.2}x EMA={:.2}x oracle={} | b{:<3}: AMA={:.2}x EMA={:.2}x oracle={}",
                trial + 1,
                sparsity.sparsity,
                REGIME_SMALL_BATCH,
                small.adaptive_ns / small.oracle_ns.max(1.0),
                small.plain_ema_ns / small.oracle_ns.max(1.0),
                small.oracle_path,
                REGIME_LARGE_BATCH,
                large.adaptive_ns / large.oracle_ns.max(1.0),
                large.plain_ema_ns / large.oracle_ns.max(1.0),
                large.oracle_path,
            );
            trials[sparsity_index].0.push(small);
            trials[sparsity_index].1.push(large);
        }
    }

    REGIME_SPARSITIES
        .into_iter()
        .enumerate()
        .map(|(index, requested_sparsity)| {
            let (total_weights, nonzero_weights, actual_sparsity) =
                shapes[index].expect("regime row was not measured");
            RegimeRow {
                requested_sparsity,
                actual_sparsity,
                total_weights,
                nonzero_weights,
                small: aggregate_regime_trials(&trials[index].0),
                large: aggregate_regime_trials(&trials[index].1),
            }
        })
        .collect()
}

/// Measure one (layer, batch) operating point: warm the batch bucket, time the
/// live adaptive selector, time each forced fixed path, and record the sparse
/// selection fraction over a fresh measurement window.
fn measure_regime(
    layer: &AchfLayer,
    plain_ema_layer: &AchfLayer,
    dim: usize,
    batch: usize,
    rotation: usize,
) -> RegimeTrialLatency {
    let x: Vec<f32> = (0..dim * batch)
        .map(|i| ((i % 7) as f32) * 0.1 + 0.05)
        .collect();
    let measure_selector = |selector: &AchfLayer| {
        for _ in 0..REGIME_WARMUP_CALLS {
            let _ =
                std::hint::black_box(selector.forward_inference_residual(std::hint::black_box(&x)));
        }
        let before = selector.cache_stats();
        let start = Instant::now();
        for _ in 0..REGIME_MEASURE_CALLS {
            let _ =
                std::hint::black_box(selector.forward_inference_residual(std::hint::black_box(&x)));
        }
        let latency_ns = start.elapsed().as_nanos() as f64 / REGIME_MEASURE_CALLS as f64;
        let after = selector.cache_stats();
        let sparse = (after.sparse_paths - before.sparse_paths) as f64;
        let total = ((after.cache_hits - before.cache_hits)
            + (after.sparse_paths - before.sparse_paths)
            + (after.dense_paths - before.dense_paths)) as f64;
        (latency_ns, sparse / total.max(1.0))
    };
    let ((adaptive_ns, sparse_frac), (plain_ema_ns, plain_ema_sparse_frac)) =
        if rotation.is_multiple_of(2) {
            (measure_selector(layer), measure_selector(plain_ema_layer))
        } else {
            let plain = measure_selector(plain_ema_layer);
            let guarded = measure_selector(layer);
            (guarded, plain)
        };

    // Forced fixed-path costs at the same operating point (reproducible).
    let forced = measure_forced_path_samples(
        layer,
        &x,
        REGIME_FORCED_WARMUP_ROUNDS,
        REGIME_MEASURE_CALLS / REGIME_FORCED_CALLS_PER_SAMPLE,
        REGIME_FORCED_CALLS_PER_SAMPLE,
        rotation,
    );
    let [cached_values, sparse_values, dense_values] = forced;
    let cached_ns = cached_values.iter().sum::<f64>() / cached_values.len() as f64;
    let sparse_ns = sparse_values.iter().sum::<f64>() / sparse_values.len() as f64;
    let dense_ns = dense_values.iter().sum::<f64>() / dense_values.len() as f64;
    let (oracle_ns, oracle_path) = [
        (cached_ns, "Cached"),
        (sparse_ns, "Sparse"),
        (dense_ns, "Dense"),
    ]
    .into_iter()
    .fold((f64::INFINITY, "Cached"), |acc, (ns, name)| {
        if ns < acc.0 {
            (ns, name)
        } else {
            acc
        }
    });
    RegimeTrialLatency {
        batch,
        adaptive_ns,
        plain_ema_ns,
        cached_ns,
        sparse_ns,
        dense_ns,
        oracle_ns,
        oracle_path: oracle_path.to_string(),
        sparse_frac,
        plain_ema_sparse_frac,
    }
}

fn aggregate_regime_trials(trials: &[RegimeTrialLatency]) -> RegimeLatency {
    assert!(!trials.is_empty(), "cannot aggregate zero regime trials");
    assert!(
        trials.iter().all(|trial| trial.batch == trials[0].batch),
        "mixed batch sizes in regime aggregation"
    );
    let values = |value: fn(&RegimeTrialLatency) -> f64| {
        TrialStats::from_values(&trials.iter().map(value).collect::<Vec<_>>())
    };
    let mut oracle_path_counts = BTreeMap::new();
    for trial in trials {
        *oracle_path_counts
            .entry(trial.oracle_path.clone())
            .or_insert(0usize) += 1;
    }
    let oracle_path = oracle_path_counts
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(left.0)))
        .map(|(path, _)| path.clone())
        .expect("regime oracle path counts should not be empty");
    RegimeLatency {
        batch: trials[0].batch,
        adaptive_ns: values(|trial| trial.adaptive_ns),
        plain_ema_ns: values(|trial| trial.plain_ema_ns),
        cached_ns: values(|trial| trial.cached_ns),
        sparse_ns: values(|trial| trial.sparse_ns),
        dense_ns: values(|trial| trial.dense_ns),
        oracle_ns: values(|trial| trial.oracle_ns),
        oracle_gap: values(|trial| trial.adaptive_ns / trial.oracle_ns.max(1.0)),
        plain_ema_oracle_gap: values(|trial| trial.plain_ema_ns / trial.oracle_ns.max(1.0)),
        cached_oracle_gap: values(|trial| trial.cached_ns / trial.oracle_ns.max(1.0)),
        sparse_oracle_gap: values(|trial| trial.sparse_ns / trial.oracle_ns.max(1.0)),
        dense_oracle_gap: values(|trial| trial.dense_ns / trial.oracle_ns.max(1.0)),
        oracle_path,
        oracle_path_counts,
        oracle_paths: trials
            .iter()
            .map(|trial| trial.oracle_path.clone())
            .collect(),
        sparse_frac: values(|trial| trial.sparse_frac),
        plain_ema_sparse_frac: values(|trial| trial.plain_ema_sparse_frac),
    }
}

fn run_gate_curve(base_config: &Config, seed: u64, num_trials: usize) -> AggregatedResult {
    println!("[Bench] Running Gate Curve Experiment...");
    let mut cfg = bench_sized_config(base_config);
    cfg.achf.enabled = true;
    cfg.achf.mode = "full".to_string();
    cfg.achf.adaptive_inference = false;
    cfg.achf.diagnostics_enabled = true;
    cap_ppo_training(&mut cfg, 4000);
    let conditions = vec![("Gate Curve".to_string(), cfg)];
    let runs = run_interleaved_conditions(&conditions, seed, num_trials);
    aggregate_conditions(&runs, &[0])
        .into_iter()
        .next()
        .expect("gate curve aggregation should produce one result")
}

fn run_scale_test(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Scale Test (varying rank + ACHF off baseline)...");
    let mut conditions = Vec::new();
    let mut baseline = bench_sized_config(base_config);
    baseline.achf.enabled = false;
    cap_ppo_training(&mut baseline, 2000);
    conditions.push(("No ACHF".to_string(), baseline));
    // The bench-sized FFN candidate has smaller dimension 64. This sweep uses
    // the mutually exclusive low-rank candidate mode; rank 64 is retained as
    // an explicit invalid/no-op boundary. Candidate entry remains subject to
    // the configured approximation-error ceiling, so rejected ranks execute
    // the reference path and are reported as ineligible rather than forced.
    for rank in [8, 16, 32, 48, 64] {
        let label = if rank == 64 {
            "rank=64 (no-op control)".to_string()
        } else {
            format!("rank={rank}")
        };
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = true;
        cfg.achf.rank = rank;
        cfg.achf.candidate_mode = "low_rank".to_string();
        cfg.achf.prune_threshold = 0.0;
        cap_ppo_training(&mut cfg, 2000);
        conditions.push((label, cfg));
    }
    let runs = run_interleaved_conditions(&conditions, seed, nt);
    aggregate_conditions(&runs, &vec![0; conditions.len()])
}

fn run_apply_combination(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Apply Placement Experiment (active policy only)...");
    let combos: Vec<(&str, LuckMode, bool, bool, bool)> = vec![
        ("PPO: no ACHF", LuckMode::Ppo, false, false, false),
        ("PPO: FFN", LuckMode::Ppo, false, true, false),
        ("PPO: Attn", LuckMode::Ppo, true, false, false),
        ("PPO: FFN+Attn", LuckMode::Ppo, true, true, false),
        ("DQN: no ACHF", LuckMode::Dqn, false, false, false),
        ("DQN: ACHF", LuckMode::Dqn, false, false, true),
    ];
    let conditions: Vec<(String, Config)> = combos
        .into_iter()
        .map(|(label, policy, attn, ffn, dqn)| {
            let mut cfg = bench_sized_config(base_config);
            cfg.luck_mode = policy;
            cfg.achf.enabled = attn || ffn || dqn;
            cfg.achf.apply_attn = attn;
            cfg.achf.apply_ffn = ffn;
            cfg.achf.apply_dqn = dqn;
            cap_ppo_training(&mut cfg, 2000);
            (label.to_string(), cfg)
        })
        .collect();
    let runs = run_interleaved_conditions(&conditions, seed, nt);
    aggregate_conditions(&runs, &[0, 0, 0, 0, 4, 4])
}

fn run_convergence(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Convergence Experiment (ACHF on/off loss curves)...");
    let conditions: Vec<(String, Config)> = [("ACHF Disabled", false), ("ACHF Enabled", true)]
        .into_iter()
        .map(|(label, enabled)| {
            let mut cfg = bench_sized_config(base_config);
            cfg.achf.enabled = enabled;
            cap_ppo_training(&mut cfg, 4000);
            (label.to_string(), cfg)
        })
        .collect();
    let runs = run_interleaved_conditions(&conditions, seed, nt);
    aggregate_conditions(&runs, &[0, 0])
}

// ── Shared training + measurement helper ────────────────────────────────

fn condition_config_snapshot(config: &Config) -> serde_json::Value {
    serde_json::json!({
        "luck_mode": config.luck_mode,
        "fast_init": config.fast_init,
        "model": {
            "dim": config.model_dim,
            "hidden_dim": config.model_hidden_dim,
            "num_layers": config.model_num_layers,
            "num_heads": config.model_num_heads,
            "kv_lora_rank": config.model_kv_lora_rank,
            "qk_rope_dim": config.model_qk_rope_dim,
            "use_multi_stream": config.use_multi_stream,
            "multi_stream_factor": config.multi_stream_factor,
        },
        "ppo": {
            "mode": config.ppo_mode,
            "total_steps": config.ppo_total_steps,
            "steps_per_update": config.ppo_steps_per_update,
            "k_epochs": config.ppo_k_epochs,
            "batch_size": config.ppo_batch_size,
            "context_len": config.ppo_context_len,
            "num_envs": config.ppo_num_envs,
            "top_k": config.ppo_top_k,
        },
        "achf": config.achf,
    })
}

struct TrainedPpoParams<'a> {
    label: &'a str,
    policy: &'a ActorCritic,
    env_net: &'a EnvNet,
    neural_opt: &'a NeuralLuckOptimizer,
    training_config: &'a Config,
    config: &'a Config,
    trial_seed: u64,
    train_time_ms: f64,
    snapshots: &'a [StepSnapshot],
}

fn measure_trained_ppo(params: TrainedPpoParams<'_>) -> BenchRunResult {
    let TrainedPpoParams {
        label,
        policy,
        env_net,
        neural_opt,
        training_config,
        config,
        trial_seed,
        train_time_ms,
        snapshots,
    } = params;
    let condition_config = serde_json::json!({
        "shared_trained_policy": true,
        "training": condition_config_snapshot(training_config),
        "inference": condition_config_snapshot(config),
    });
    let config_fingerprint = sha256_hex(
        &serde_json::to_vec(&condition_config).expect("PPO condition configuration must serialize"),
    );
    let train_loss = snapshots
        .last()
        .expect("shared PPO benchmark emitted no training snapshots")
        .loss;
    let before = aggregate_model_cache_stats(config, None, Some(policy));
    let context_len = if config.ppo_context_len > 0 {
        config.ppo_context_len
    } else if config.fast_init {
        6
    } else {
        8
    };
    let eval = evaluate_ppo_policy(
        policy,
        env_net,
        config,
        context_len,
        BENCH_EVAL_EPISODES,
        derive_seed(trial_seed, SEED_POLICY_EVAL),
    );
    let throughput = measure_inference_throughput(
        derive_seed(trial_seed, SEED_THROUGHPUT),
        &ThroughputParams {
            neural_opt,
            dqn: None,
            ppo: Some(policy),
            env_net,
            config,
            sims: THROUGHPUT_SIMS,
            pulls: THROUGHPUT_PULLS,
        },
    );
    let after = aggregate_model_cache_stats(config, None, Some(policy));
    let cache_stats = before
        .zip(after)
        .map(|(before, after)| cache_stats_delta(before, after));
    let achf = policy.snapshot_achf();
    let memory_stats = config
        .achf
        .enabled
        .then(|| policy.achf_memory_stats_aggregate());
    validate_candidate_memory(label, memory_stats);
    BenchRunResult {
        label: label.to_string(),
        policy: "PPO".to_string(),
        config_fingerprint,
        condition_config,
        train_time_ms,
        throughput_sims_per_sec: throughput,
        eval_reward: eval.avg_reward,
        train_loss,
        param_count: policy.param_count(),
        applied_rank: achf.map(|snapshot| snapshot.low_rank_applied_rank),
        candidate_relative_error: memory_stats.and_then(|stats| stats.candidate_relative_error()),
        memory_stats,
        snapshots: snapshots.to_vec(),
        cache_stats,
    }
}

fn train_and_measure(label: &str, config: &Config, seed: u64) -> BenchRunResult {
    let cfg = config.clone();
    let condition_config = condition_config_snapshot(&cfg);
    let config_fingerprint = sha256_hex(
        &serde_json::to_vec(&condition_config)
            .expect("condition configuration must serialize for provenance"),
    );
    let (env_net, neural_opt, _worker) =
        build_base_models(&cfg, derive_seed(seed, SEED_BASE_MODELS));
    let eval_seed = derive_seed(seed, SEED_POLICY_EVAL);
    let throughput_seed = derive_seed(seed, SEED_THROUGHPUT);

    match cfg.luck_mode {
        LuckMode::Dqn => {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut train_rng = Rng::from_seed(derive_seed(seed, SEED_DQN_TRAIN));
            let train_start = Instant::now();
            let dqn = train_dqn_with_metrics(&neural_opt, &mut train_rng, &env_net, &cfg, Some(tx));
            let train_elapsed = train_start.elapsed();
            let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();
            let train_loss = snapshots
                .last()
                .expect("DQN benchmark emitted no training snapshots")
                .loss;
            let before = aggregate_model_cache_stats(&cfg, Some(&dqn), None);
            let eval = evaluate_dqn_policy(&dqn, &env_net, &cfg, BENCH_EVAL_EPISODES, eval_seed);
            let throughput = measure_inference_throughput(
                throughput_seed,
                &ThroughputParams {
                    neural_opt: &neural_opt,
                    dqn: Some(&dqn),
                    ppo: None,
                    env_net: &env_net,
                    config: &cfg,
                    sims: THROUGHPUT_SIMS,
                    pulls: THROUGHPUT_PULLS,
                },
            );
            let after = aggregate_model_cache_stats(&cfg, Some(&dqn), None);
            let cache_stats = before.zip(after).map(|(before, after)| {
                let stats = cache_stats_delta(before, after);
                AchfCacheStats::debug_print(&[stats]);
                stats
            });
            let achf = dqn.snapshot_achf();
            let memory_stats = dqn.achf_memory_stats();
            validate_candidate_memory(label, memory_stats);
            let candidate_relative_error =
                memory_stats.and_then(|stats| stats.candidate_relative_error());
            log_applied_rank(achf);
            BenchRunResult {
                label: label.to_string(),
                policy: "DQN".to_string(),
                config_fingerprint: config_fingerprint.clone(),
                condition_config: condition_config.clone(),
                train_time_ms: train_elapsed.as_secs_f64() * 1000.0,
                throughput_sims_per_sec: throughput,
                eval_reward: eval.avg_reward,
                train_loss,
                param_count: dqn.param_count(),
                applied_rank: achf.map(|snapshot| snapshot.low_rank_applied_rank),
                candidate_relative_error,
                memory_stats,
                snapshots,
                cache_stats,
            }
        }
        LuckMode::Ppo => {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut train_rng = Rng::from_seed(derive_seed(seed, SEED_PPO_TRAIN));
            let train_start = Instant::now();
            let ppo = train_ppo_with_metrics(&mut train_rng, &env_net, &cfg, Some(tx));
            let train_elapsed = train_start.elapsed();
            let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();
            let train_loss = snapshots
                .last()
                .expect("PPO benchmark emitted no training snapshots")
                .loss;
            let before = aggregate_model_cache_stats(&cfg, None, Some(&ppo));
            let context_len = if cfg.ppo_context_len > 0 {
                cfg.ppo_context_len
            } else if cfg.fast_init {
                6
            } else {
                8
            };
            let eval = evaluate_ppo_policy(
                &ppo,
                &env_net,
                &cfg,
                context_len,
                BENCH_EVAL_EPISODES,
                eval_seed,
            );
            let throughput = measure_inference_throughput(
                throughput_seed,
                &ThroughputParams {
                    neural_opt: &neural_opt,
                    dqn: None,
                    ppo: Some(&ppo),
                    env_net: &env_net,
                    config: &cfg,
                    sims: THROUGHPUT_SIMS,
                    pulls: THROUGHPUT_PULLS,
                },
            );
            let after = aggregate_model_cache_stats(&cfg, None, Some(&ppo));
            let cache_stats = before.zip(after).map(|(before, after)| {
                let stats = cache_stats_delta(before, after);
                AchfCacheStats::debug_print(&[stats]);
                stats
            });
            let achf = ppo.snapshot_achf();
            let memory_stats = cfg.achf.enabled.then(|| ppo.achf_memory_stats_aggregate());
            validate_candidate_memory(label, memory_stats);
            let candidate_relative_error =
                memory_stats.and_then(|stats| stats.candidate_relative_error());
            log_applied_rank(achf);
            BenchRunResult {
                label: label.to_string(),
                policy: "PPO".to_string(),
                config_fingerprint,
                condition_config,
                train_time_ms: train_elapsed.as_secs_f64() * 1000.0,
                throughput_sims_per_sec: throughput,
                eval_reward: eval.avg_reward,
                train_loss,
                param_count: ppo.param_count(),
                applied_rank: achf.map(|snapshot| snapshot.low_rank_applied_rank),
                candidate_relative_error,
                memory_stats,
                snapshots,
                cache_stats,
            }
        }
        LuckMode::Probability => {
            panic!("ACHF policy benchmark requires luck_mode=dqn or luck_mode=ppo")
        }
    }
}

fn log_applied_rank(snapshot: Option<crate::achf::AchfStateSnapshot>) {
    if let Some(snapshot) = snapshot {
        println!(
            "    [ACHF] applied_rank={} low_rank_rel_err={:.4} candidate_weight_relative_frobenius_error={:.4}",
            snapshot.low_rank_applied_rank,
            snapshot.low_rank_rel_err,
            snapshot.candidate_relative_error
        );
    }
}

// ── Chart wrappers (aggregated) ──────────────────────────────────────────

fn ci_bounds(stats: &TrialStats) -> (f64, f64) {
    (
        stats.ci_low.unwrap_or(stats.mean),
        stats.ci_high.unwrap_or(stats.mean),
    )
}

fn agg_bars_with_ci(agg: &[AggregatedResult]) -> Vec<(&str, f64, f64, f64)> {
    agg.iter()
        .map(|result| {
            let (low, high) = ci_bounds(&result.throughput);
            (result.label.as_str(), result.throughput.mean, low, high)
        })
        .collect()
}

fn chart_ablation(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_ci(agg);
    let path = format!("{}/ablation_throughput.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_ci(
            &path,
            "Ablation: Throughput (mean and 95% CI)",
            "Configuration",
            "Sims/sec",
            &bars,
            800,
            500,
        ),
    );
    chart_agg_reward_curve(
        agg,
        dir,
        ext,
        "ablation_reward",
        "Ablation: Training Reward Curve",
    );
}

fn chart_mode(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_ci(agg);
    let path = format!("{}/mode_comparison.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_ci(
            &path,
            "Inference Strategy: Throughput (mean and 95% CI)",
            "Mode",
            "Sims/sec",
            &bars,
            800,
            500,
        ),
    );
}

fn chart_path_latency(latencies: &[PathLatencyResult], dir: &str, ext: &str) {
    let stats: Vec<(&str, [f64; 5])> = latencies
        .iter()
        .filter(|result| result.trial_samples.iter().any(|values| !values.is_empty()))
        .map(|result| {
            let mut sorted: Vec<f64> = result.trial_samples.iter().flatten().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            let q1 = percentile_sorted(&sorted, 25);
            let median = percentile_sorted(&sorted, 50);
            let q3 = percentile_sorted(&sorted, 75);
            // Whiskers use Tukey fences (1.5*IQR) clamped to the observed data,
            // NOT the absolute min/max. A single scheduling-spike sample (e.g. a
            // 12us OS hiccup on a ~500ns operator) would otherwise set the axis
            // range and compress every box — including a genuinely tight
            // distribution like Cached (IQR ~10ns) — down to sub-pixel height,
            // which reads as a broken/empty chart. Capping the whiskers keeps the
            // box and IQR legible; extreme outliers beyond the fence are simply
            // not drawn, which is standard box-plot practice.
            let iqr = q3 - q1;
            let lower_fence = (q1 - 1.5 * iqr).max(sorted[0]);
            let upper_fence = (q3 + 1.5 * iqr).min(sorted[n - 1]);
            let q = [lower_fence, q1, median, q3, upper_fence];
            (result.label.as_str(), q)
        })
        .collect();
    let path = format!("{}/path_latency_boxplot.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_box_plot(
            &path,
            "Inference Path Latency",
            "Path",
            "Latency (ns)",
            &stats,
            800,
            500,
        ),
    );
}

fn chart_gate_curve(result: &AggregatedResult, dir: &str, ext: &str) {
    if result.curve.is_empty() {
        return;
    }
    let gate: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.gate_value);
            (point.step as f64, point.gate_value.mean, low, high)
        })
        .collect();
    let gmin: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.g_min);
            (point.step as f64, point.g_min.mean, low, high)
        })
        .collect();
    let gate_velocity: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.gate_velocity);
            (point.step as f64, point.gate_velocity.mean, low, high)
        })
        .collect();
    let grad: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.grad_ema);
            (point.step as f64, point.grad_ema.mean, low, high)
        })
        .collect();
    let gradient_cosine: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.gradient_cosine);
            (point.step as f64, point.gradient_cosine.mean, low, high)
        })
        .collect();
    let hit: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.cached_path_rate);
            (point.step as f64, point.cached_path_rate.mean, low, high)
        })
        .collect();
    let lr_ratio: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.sparse_ratio);
            (point.step as f64, point.sparse_ratio.mean, low, high)
        })
        .collect();
    let abias: Vec<(f64, f64, f64, f64)> = result
        .curve
        .iter()
        .map(|point| {
            let (low, high) = ci_bounds(&point.adaptive_bias);
            (point.step as f64, point.adaptive_bias.mean, low, high)
        })
        .collect();

    let series: Vec<chart::CiSeries<'_>> = vec![
        ("Reference Gate", &gate),
        ("Reference Gate Floor", &gmin),
        ("Reference Gate Velocity", &gate_velocity),
        ("Grad EMA", &grad),
        ("Gradient Cosine", &gradient_cosine),
        ("Cached Within Candidate", &hit),
        ("Sparse Within Candidate", &lr_ratio),
        ("Adaptive Bias", &abias),
    ];
    let path = format!("{}/gate_curve.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_line_chart_with_ci(
            &path,
            "Gate Dynamics During Training",
            "Training Step",
            "Value",
            &series,
            1000,
            600,
        ),
    );
}

/// One line chart per dim: per-path latency vs weight sparsity (log y). The
/// crossings between the Cached and Sparse lines are the operating points where
/// the fastest path flips — the visual core of the crossover argument.
fn chart_crossover(cells: &[CrossoverCell], dir: &str, ext: &str) {
    let mut dims: Vec<usize> = cells.iter().map(|c| c.dim).collect();
    dims.sort_unstable();
    dims.dedup();
    for dim in dims {
        let rows: Vec<&CrossoverCell> = cells.iter().filter(|c| c.dim == dim).collect();
        if rows.is_empty() {
            continue;
        }
        let cached: Vec<(f64, f64, f64, f64)> = rows
            .iter()
            .map(|cell| {
                let (low, high) = ci_bounds(&cell.cached_ns);
                (cell.actual_sparsity, cell.cached_ns.mean, low, high)
            })
            .collect();
        let sparse: Vec<(f64, f64, f64, f64)> = rows
            .iter()
            .map(|cell| {
                let (low, high) = ci_bounds(&cell.sparse_ns);
                (cell.actual_sparsity, cell.sparse_ns.mean, low, high)
            })
            .collect();
        let dense: Vec<(f64, f64, f64, f64)> = rows
            .iter()
            .map(|cell| {
                let (low, high) = ci_bounds(&cell.dense_ns);
                (cell.actual_sparsity, cell.dense_ns.mean, low, high)
            })
            .collect();
        let series: Vec<chart::CiSeries<'_>> =
            vec![("Cached", &cached), ("Sparse", &sparse), ("Dense", &dense)];
        let path = format!("{}/path_crossover_dim{}.{}", dir, dim, ext);
        write_chart(
            &path,
            chart::draw_line_chart_with_ci(
                &path,
                &format!("Path Latency vs Weight Sparsity (dim={dim}, 95% CI)"),
                "Weight Sparsity",
                "Latency (ns per forward)",
                &series,
                900,
                600,
            ),
        );
    }
}

/// Grouped bars: oracle-gap (adaptive latency / best-fixed-path latency) at
/// small vs large batch for each weight sparsity. Bars near 1.0 mean the
/// adaptive selector matched the best fixed path for that regime. The point of
/// the chart is that the ORACLE PATH differs across regimes (shown in labels),
/// yet the adaptive selector stays near 1.0 in both — a single fixed path
/// cannot. A ratio <1.0 can occur when interleaved probing warms caches the
/// isolated forced-path timing doesn't see; it still means "matched oracle".
fn chart_regime(rows: &[RegimeRow], dir: &str, ext: &str) {
    if rows.is_empty() {
        return;
    }
    let labels: Vec<String> = rows
        .iter()
        .flat_map(|r| {
            [
                format!(
                    "wsp{:.2} b{} AMA(oracle={})",
                    r.actual_sparsity, r.small.batch, r.small.oracle_path
                ),
                format!(
                    "wsp{:.2} b{} EMA(oracle={})",
                    r.actual_sparsity, r.small.batch, r.small.oracle_path
                ),
                format!(
                    "wsp{:.2} b{} AMA(oracle={})",
                    r.actual_sparsity, r.large.batch, r.large.oracle_path
                ),
                format!(
                    "wsp{:.2} b{} EMA(oracle={})",
                    r.actual_sparsity, r.large.batch, r.large.oracle_path
                ),
            ]
        })
        .collect();
    let mut bars: Vec<(&str, f64, f64, f64)> = Vec::with_capacity(labels.len());
    for (i, r) in rows.iter().enumerate() {
        let (small_low, small_high) = ci_bounds(&r.small.oracle_gap);
        let (small_ema_low, small_ema_high) = ci_bounds(&r.small.plain_ema_oracle_gap);
        let (large_low, large_high) = ci_bounds(&r.large.oracle_gap);
        let (large_ema_low, large_ema_high) = ci_bounds(&r.large.plain_ema_oracle_gap);
        bars.push((
            labels[4 * i].as_str(),
            r.small.oracle_gap.mean,
            small_low,
            small_high,
        ));
        bars.push((
            labels[4 * i + 1].as_str(),
            r.small.plain_ema_oracle_gap.mean,
            small_ema_low,
            small_ema_high,
        ));
        bars.push((
            labels[4 * i + 2].as_str(),
            r.large.oracle_gap.mean,
            large_low,
            large_high,
        ));
        bars.push((
            labels[4 * i + 3].as_str(),
            r.large.plain_ema_oracle_gap.mean,
            large_ema_low,
            large_ema_high,
        ));
    }
    let path = format!("{}/regime_adaptation.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_ci(
            &path,
            "Guarded AMA vs Plain EMA Oracle-Gap (mean and 95% CI)",
            "Weight Sparsity x Batch x Selector",
            "Selector / Oracle Latency",
            &bars,
            1500,
            600,
        ),
    );
}

fn chart_scale(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_ci(agg);
    let path = format!("{}/scale_test.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_ci(
            &path,
            "Rank-Constrained Sparsity: Throughput (mean and 95% CI)",
            "Configuration",
            "Sims/sec",
            &bars,
            900,
            500,
        ),
    );
}

fn chart_apply(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_ci(agg);
    let path = format!("{}/apply_combination.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_ci(
            &path,
            "Apply Combination: Throughput (mean and 95% CI)",
            "Configuration",
            "Sims/sec",
            &bars,
            1000,
            500,
        ),
    );
}

fn chart_convergence(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let loss_series: Vec<OwnedCiSeries> = agg
        .iter()
        .filter(|result| !result.curve.is_empty())
        .map(|a| {
            let pts: Vec<(f64, f64, f64, f64)> = a
                .curve
                .iter()
                .map(|point| {
                    let (low, high) = ci_bounds(&point.train_loss);
                    (point.step as f64, point.train_loss.mean, low, high)
                })
                .collect();
            (a.label.clone(), pts)
        })
        .collect();
    let loss_ref: Vec<chart::CiSeries<'_>> = loss_series
        .iter()
        .map(|(l, d)| (l.as_str(), d.as_slice()))
        .collect();
    if loss_ref.is_empty() {
        eprintln!("[Bench] Skipping convergence_loss chart: no loss snapshots collected");
        return;
    }
    let path = format!("{}/convergence_loss.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_line_chart_with_ci(
            &path,
            "Convergence: Loss Curve (mean and 95% CI)",
            "Training Step",
            "Loss",
            &loss_ref,
            900,
            500,
        ),
    );

    chart_agg_reward_curve(
        agg,
        dir,
        ext,
        "convergence_reward",
        "Convergence: Training Reward Curve",
    );
}

fn chart_agg_reward_curve(
    agg: &[AggregatedResult],
    dir: &str,
    ext: &str,
    filename: &str,
    title: &str,
) {
    let series: Vec<OwnedCiSeries> = agg
        .iter()
        .filter(|result| !result.curve.is_empty())
        .map(|a| {
            let pts: Vec<(f64, f64, f64, f64)> = a
                .curve
                .iter()
                .map(|point| {
                    let (low, high) = ci_bounds(&point.train_reward);
                    (point.step as f64, point.train_reward.mean, low, high)
                })
                .collect();
            (a.label.clone(), pts)
        })
        .collect();
    let series_ref: Vec<chart::CiSeries<'_>> = series
        .iter()
        .map(|(l, d)| (l.as_str(), d.as_slice()))
        .collect();
    if series_ref.is_empty() {
        eprintln!("[Bench] Skipping {filename} chart: no reward snapshots collected");
        return;
    }
    let path = format!("{}/{}.{}", dir, filename, ext);
    write_chart(
        &path,
        chart::draw_line_chart_with_ci(
            &path,
            title,
            "Training Step",
            "Training Avg Reward",
            &series_ref,
            900,
            500,
        ),
    );
}

// ── Output: summary, JSON, CSV ──────────────────────────────────────────

/// Effective low-rank truncation actually applied by the active frozen policy.
/// A value of 0 means the requested rank was a no-op (>= the layer's smaller
/// dimension), which is exactly the degenerate case worth surfacing.
fn effective_applied_rank(a: &AggregatedResult) -> Option<usize> {
    a.applied_rank
}

fn format_ci(stats: &TrialStats) -> String {
    match (stats.ci_low, stats.ci_high) {
        (Some(low), Some(high)) => format!("95% CI [{low:.4}, {high:.4}]"),
        _ => "95% CI unavailable (n=1)".to_string(),
    }
}

fn print_agg_summary(name: &str, agg: &[AggregatedResult]) {
    println!("[Bench] {} complete:", name);
    for a in agg {
        let rank_note = match effective_applied_rank(a) {
            Some(0) => " | applied_rank=0 (no-op)".to_string(),
            Some(r) => format!(" | applied_rank={}", r),
            None => String::new(),
        };
        println!(
            "  {:20} | policy={} | tput: {:.0} +/- {:.0} | eval_reward: {:.4} +/- {:.4} | train_loss: {:.4} +/- {:.4} | params: {}{}",
            a.label,
            a.policy,
            a.throughput.mean, a.throughput.std_dev,
            a.eval_reward.mean, a.eval_reward.std_dev,
            a.train_loss.mean, a.train_loss.std_dev,
            a.param_count,
            rank_note
        );
        if let Some(paired) = &a.paired {
            println!(
                "    paired vs {}: delta_tput={:.1} ({}) | relative={:.2}% ({}) | delta_eval_reward={:.4} ({})",
                paired.baseline,
                paired.throughput_delta.mean,
                format_ci(&paired.throughput_delta),
                paired.throughput_relative_delta_pct.mean,
                format_ci(&paired.throughput_relative_delta_pct),
                paired.eval_reward_delta.mean,
                format_ci(&paired.eval_reward_delta),
            );
        }
        if let Some(candidate_relative_error) = &a.candidate_relative_error {
            println!(
                "    candidate_weight_relative_frobenius_error={:.4} ({})",
                candidate_relative_error.mean,
                format_ci(candidate_relative_error)
            );
        }
        if !a.memory_stats.is_empty() {
            let total = TrialStats::from_values(
                &a.memory_stats
                    .iter()
                    .map(|stats| stats.total_materialized_bytes as f64)
                    .collect::<Vec<_>>(),
            );
            let sparsity = TrialStats::from_values(
                &a.memory_stats
                    .iter()
                    .map(|stats| {
                        1.0 - stats.candidate_nonzero_weights as f64
                            / stats.candidate_total_weights.max(1) as f64
                    })
                    .collect::<Vec<_>>(),
            );
            println!(
                "    materialized_bytes={:.0} ({}) | candidate_sparsity={:.4} ({})",
                total.mean,
                format_ci(&total),
                sparsity.mean,
                format_ci(&sparsity)
            );
        }
    }
}

fn has_unpruned_candidate(all: &[(&str, Vec<AggregatedResult>)]) -> bool {
    all.iter().any(|(_, results)| {
        results.iter().any(|result| {
            result.memory_stats.iter().any(|stats| {
                stats.candidate_total_weights > 0
                    && stats.candidate_nonzero_weights == stats.candidate_total_weights
            })
        })
    })
}

fn path_has_unpruned_candidate(path_latencies: Option<&[PathLatencyResult]>) -> bool {
    path_latencies.is_some_and(|results| {
        results.iter().any(|result| {
            result.trial_sparsity.iter().any(|stats| {
                stats.total_weights > 0 && stats.nonzero_weights == stats.total_weights
            })
        })
    })
}
fn has_ineligible_candidate(all: &[(&str, Vec<AggregatedResult>)]) -> bool {
    all.iter().any(|(_, results)| {
        results.iter().any(|result| {
            result
                .memory_stats
                .iter()
                .any(|stats| stats.eligible_candidate_layers < stats.candidate_layers)
        })
    })
}

fn missing_benchmark_experiments(
    all: &[(&str, Vec<AggregatedResult>)],
    path_latencies: Option<&[PathLatencyResult]>,
    gate_curve: Option<&AggregatedResult>,
    crossover: Option<&[CrossoverCell]>,
    regime: Option<&[RegimeRow]>,
) -> Vec<&'static str> {
    BENCH_EXPERIMENTS
        .iter()
        .copied()
        .filter(|name| {
            let completed = all.iter().any(|(completed, _)| completed == name)
                || (*name == "path" && path_latencies.is_some())
                || (*name == "gate" && gate_curve.is_some())
                || (*name == "crossover" && crossover.is_some())
                || (*name == "regime" && regime.is_some());
            !completed
        })
        .collect()
}

fn write_summary_txt(
    all: &[(&str, Vec<AggregatedResult>)],
    path_latencies: Option<&[PathLatencyResult]>,
    gate_curve: Option<&AggregatedResult>,
    crossover: Option<&[CrossoverCell]>,
    regime: Option<&[RegimeRow]>,
    num_trials: usize,
    dir: &str,
) {
    let missing_experiments =
        missing_benchmark_experiments(all, path_latencies, gate_curve, crossover, regime);
    let mut lines = Vec::new();
    for (name, agg) in all {
        lines.push(format!("=== {} ===", name));
        for a in agg {
            let rank_note = match effective_applied_rank(a) {
                Some(0) => " | applied_rank=0 (no-op)".to_string(),
                Some(r) => format!(" | applied_rank={}", r),
                None => String::new(),
            };
            lines.push(format!(
                "  {:20} | policy={} | trials={} | tput={:.0}+/-{:.0} ({}) | eval_reward={:.4}+/-{:.4} ({}) | train_loss={:.4}+/-{:.4} ({}) | policy_train_ms={:.1}+/-{:.1} ({}) | params={}{}",
                a.label,
                a.policy,
                a.throughput.values.len(),
                a.throughput.mean, a.throughput.std_dev,
                format_ci(&a.throughput),
                a.eval_reward.mean, a.eval_reward.std_dev,
                format_ci(&a.eval_reward),
                a.train_loss.mean, a.train_loss.std_dev,
                format_ci(&a.train_loss),
                a.train_time_ms.mean, a.train_time_ms.std_dev,
                format_ci(&a.train_time_ms),
                a.param_count,
                rank_note
            ));
            if let Some(paired) = &a.paired {
                lines.push(format!(
                    "    paired vs {}: delta_tput={:.2} ({}) | relative_tput={:.2}% ({}) | delta_eval_reward={:.4} ({}) | delta_train_loss={:.4} ({}) | delta_train_ms={:.1} ({})",
                    paired.baseline,
                    paired.throughput_delta.mean,
                    format_ci(&paired.throughput_delta),
                    paired.throughput_relative_delta_pct.mean,
                    format_ci(&paired.throughput_relative_delta_pct),
                    paired.eval_reward_delta.mean,
                    format_ci(&paired.eval_reward_delta),
                    paired.train_loss_delta.mean,
                    format_ci(&paired.train_loss_delta),
                    paired.train_time_ms_delta.mean,
                    format_ci(&paired.train_time_ms_delta),
                ));
            }
            if let Some(ref stats) = a.cache_stats {
                let calls = stats.calls as f64;
                let hit_pct = if calls > 0.0 {
                    stats.cache_hits as f64 / calls * 100.0
                } else {
                    0.0
                };
                lines.push(format!(
                    "    frozen ACHF ({} trials): calls={} hit={:.1}% sparse={} dense={} latency_samples={} switches={} probes={} warmness=[{:.2},{:.2},{:.2}] stale=[{},{},{}] bias={:.3}",
                    a.cache_trial_count,
                    stats.calls,
                    hit_pct,
                    stats.sparse_paths,
                    stats.dense_paths,
                    stats.latency_samples,
                    stats.path_switches,
                    stats.path_probes,
                    stats.cached_warmness,
                    stats.sparse_warmness,
                    stats.dense_warmness,
                    stats.cached_stale_age,
                    stats.sparse_stale_age,
                    stats.dense_stale_age,
                    stats.adaptive_bias
                ));
            }
            if let Some(candidate_relative_error) = &a.candidate_relative_error {
                lines.push(format!(
                    "    candidate_weight_relative_frobenius_error={:.6} ({})",
                    candidate_relative_error.mean,
                    format_ci(candidate_relative_error)
                ));
            }
            if !a.memory_stats.is_empty() {
                let total = TrialStats::from_values(
                    &a.memory_stats
                        .iter()
                        .map(|stats| stats.total_materialized_bytes as f64)
                        .collect::<Vec<_>>(),
                );
                let sparsity = TrialStats::from_values(
                    &a.memory_stats
                        .iter()
                        .map(|stats| {
                            1.0 - stats.candidate_nonzero_weights as f64
                                / stats.candidate_total_weights.max(1) as f64
                        })
                        .collect::<Vec<_>>(),
                );
                lines.push(format!(
                    "    materialized_bytes={:.0} ({}) | candidate_sparsity={:.6} ({})",
                    total.mean,
                    format_ci(&total),
                    sparsity.mean,
                    format_ci(&sparsity)
                ));
            }
        }
        lines.push(String::new());
    }
    if let Some(latencies) = path_latencies {
        lines.push("=== path_latency ===".to_string());
        for stats in path_latency_stats(latencies) {
            lines.push(format!(
                "  {:20} | trials={} | samples={} | mean={:.1}+/-{:.1}ns | {} | p50={:.1}ns | p95={:.1}ns | p99={:.1}ns",
                stats.label,
                stats.trials,
                stats.samples,
                stats.mean_ns,
                stats.std_dev_ns,
                match (stats.ci_low_ns, stats.ci_high_ns) {
                    (Some(low), Some(high)) => format!("95% CI [{low:.1}, {high:.1}]"),
                    _ => "95% CI unavailable (n=1)".to_string(),
                },
                stats.p50_ns,
                stats.p95_ns,
                stats.p99_ns
            ));
        }
        lines.push(String::new());
    }
    if let Some(result) = gate_curve {
        lines.push("=== gate_curve ===".to_string());
        lines.push(format!(
            "  {:20} | trials={} | curve_points={} | tput={:.0} | eval_reward={:.4} | train_loss={:.4} | params={} | policy_train_time_ms={:.1}",
            result.label,
            result.throughput.values.len(),
            result.curve.len(),
            result.throughput.mean,
            result.eval_reward.mean,
            result.train_loss.mean,
            result.param_count,
            result.train_time_ms.mean
        ));
        if let Some(last) = result.curve.last() {
            lines.push(format!(
                "    final aggregated training point: step={} n={} reference_gate={:.4} reference_floor={:.4} cached_within_candidate={:.1}% sparse_within_candidate={:.1}% bias={:.3}",
                last.step,
                last.samples,
                last.gate_value.mean,
                last.g_min.mean,
                last.cached_path_rate.mean * 100.0,
                last.sparse_ratio.mean * 100.0,
                last.adaptive_bias.mean
            ));
        }
        if let Some(stats) = result.cache_stats {
            let calls = stats.calls as f64;
            let candidate_calls = stats.candidate_paths as f64;
            let call_pct = |n: u64| {
                if calls > 0.0 {
                    n as f64 / calls * 100.0
                } else {
                    0.0
                }
            };
            let candidate_pct = |n: u64| {
                if candidate_calls > 0.0 {
                    n as f64 / candidate_calls * 100.0
                } else {
                    0.0
                }
            };
            lines.push(format!(
                "    ACHF inference: calls={} memo={:.1}% reference={:.1}% candidate={:.1}% rejected={} | candidate routes cached={:.1}% sparse={:.1}% dense={:.1}% | latency_samples={} decision_ns={:.1}",
                stats.calls,
                call_pct(stats.memo_hits),
                call_pct(stats.reference_paths),
                call_pct(stats.candidate_paths),
                stats.candidate_rejections,
                candidate_pct(stats.cache_hits),
                candidate_pct(stats.sparse_paths),
                candidate_pct(stats.dense_paths),
                stats.latency_samples,
                stats.decision_ema_ns
            ));
        }
        lines.push(String::new());
    }
    if let Some(cells) = crossover {
        lines.push("=== crossover ===".to_string());
        for cell in cells {
            lines.push(format!(
                "  dim={} requested={:.4} actual={:.4} nnz={}/{} | cached={:.1}ns ({}) sparse={:.1}ns ({}) dense={:.1}ns ({}) | winner_by_mean={} | winner_significant_95={}",
                cell.dim,
                cell.requested_sparsity,
                cell.actual_sparsity,
                cell.nonzero_weights,
                cell.total_weights,
                cell.cached_ns.mean,
                format_ci(&cell.cached_ns),
                cell.sparse_ns.mean,
                format_ci(&cell.sparse_ns),
                cell.dense_ns.mean,
                format_ci(&cell.dense_ns),
                cell.winner,
                cell.significant_winner_95.as_deref().unwrap_or("indeterminate"),
            ));
        }
        lines.push(String::new());
    }
    if let Some(rows) = regime {
        lines.push("=== regime ===".to_string());
        for row in rows {
            for latency in [&row.small, &row.large] {
                lines.push(format!(
                    "  requested={:.4} actual={:.4} nnz={}/{} batch={} | guarded_ama={:.1}ns gap={:.3} ({}) sparse_frac={:.3} | plain_ema={:.1}ns gap={:.3} ({}) sparse_frac={:.3} | oracle={:.1}ns path={} counts={:?}",
                    row.requested_sparsity,
                    row.actual_sparsity,
                    row.nonzero_weights,
                    row.total_weights,
                    latency.batch,
                    latency.adaptive_ns.mean,
                    latency.oracle_gap.mean,
                    format_ci(&latency.oracle_gap),
                    latency.sparse_frac.mean,
                    latency.plain_ema_ns.mean,
                    latency.plain_ema_oracle_gap.mean,
                    format_ci(&latency.plain_ema_oracle_gap),
                    latency.plain_ema_sparse_frac.mean,
                    latency.oracle_ns.mean,
                    latency.oracle_path,
                    latency.oracle_path_counts,
                ));
            }
        }
        lines.push(String::new());
    }
    lines.push("=== publication_readiness ===".to_string());
    lines.push("  final_submission_ready=false".to_string());
    if cfg!(debug_assertions) {
        lines.push("  blocker: debug build; rerun the release executable".to_string());
    }
    if num_trials < 5 {
        lines.push("  blocker: fewer than 5 seeded trials".to_string());
    }
    if !missing_experiments.is_empty() {
        lines.push(format!(
            "  blocker: partial benchmark suite; missing={}",
            missing_experiments.join(",")
        ));
    }
    if has_unpruned_candidate(all) || path_has_unpruned_candidate(path_latencies) {
        lines.push(
            "  blocker: at least one real ACHF candidate has zero realized sparsity; do not claim sparse-candidate speedup"
                .to_string(),
        );
    }
    lines.push("  blocker: only one operating-system process repetition".to_string());
    lines.push("  blocker: only one hardware/software environment".to_string());
    lines.push(
        "  blocker: prune-and-fine-tune and sparse-training quality baselines are absent"
            .to_string(),
    );
    lines.push(
        "  blocker: time-resolved candidate-output discrepancy and ACHF-local Adam moment drift are absent"
            .to_string(),
    );
    if has_ineligible_candidate(all) {
        lines.push(
            "  blocker: at least one materialized candidate failed production entry criteria; fixed-mode results are diagnostic only"
                .to_string(),
        );
    }
    lines.push(
        "  note: use these results as a transparent single-machine pilot, not final submission evidence"
            .to_string(),
    );
    let path = format!("{}/summary.txt", dir);
    write_text_file(&path, &lines.join("\n"));
    println!("[Bench] Summary -> {}", path);
}

fn write_summary_json(
    all: &[(&str, Vec<AggregatedResult>)],
    path_latencies: Option<&[PathLatencyResult]>,
    gate_curve: Option<&AggregatedResult>,
    crossover: Option<&[CrossoverCell]>,
    regime: Option<&[RegimeRow]>,
    metadata: (u64, usize),
    dir: &str,
) {
    let (seed, num_trials) = metadata;
    let logical_cpus = std::thread::available_parallelism().map_or(1, usize::from);
    let mut root = serde_json::Map::new();
    root.insert(
        "metadata".to_string(),
        serde_json::json!({
            "schema_version": 4,
            "run_manifest": "run_manifest.json",
            "package_version": env!("CARGO_PKG_VERSION"),
            "target_os": std::env::consts::OS,
            "target_arch": std::env::consts::ARCH,
            "debug_assertions": cfg!(debug_assertions),
            "logical_cpus": logical_cpus,
            "seed": seed,
            "num_trials": num_trials,
            "confidence_level": 0.95,
            "confidence_interval": "two-sided Student-t; null when n < 2",
            "trial_order": "rotating interleaved conditions",
            "trial_unit": "independent deterministic seed within one operating-system process",
            "independent_process_repetitions": 1,
            "base_model_cache": "disabled; EnvNet is rebuilt from its domain seed for every condition",
            "policy_eval": {
                "episodes": BENCH_EVAL_EPISODES,
                "timing": "after training freeze",
                "seed_domain": format!("0x{SEED_POLICY_EVAL:016X}"),
            },
            "throughput": {
                "simulations": THROUGHPUT_SIMS,
                "pulls_per_simulation": THROUGHPUT_PULLS,
                "warmup_simulations": THROUGHPUT_WARMUP_SIMS,
                "timing": "after training freeze and held-out evaluation",
                "seed_domain": format!("0x{SEED_THROUGHPUT:016X}"),
            },
            "microbenchmarks": {
                "path": {
                    "candidate_mode": "sparse",
                    "candidate_selection": "forced diagnostic; bypasses production entry only for kernel equivalence/timing",
                    "prune_threshold": PATH_PRUNE_THRESHOLD,
                    "warmup_rounds": PATH_WARMUP_ROUNDS,
                    "samples_per_trial": PATH_SAMPLES,
                    "calls_per_sample": PATH_CALLS_PER_SAMPLE,
                    "path_order": "rotating interleaved",
                },
                "crossover": {
                    "dims": CROSSOVER_DIMS,
                    "requested_sparsities": CROSSOVER_SPARSITIES,
                    "batch": CROSSOVER_BATCH,
                    "warmup_rounds": CROSSOVER_WARMUP_ROUNDS,
                    "samples_per_trial": CROSSOVER_SAMPLES,
                    "path_order": "rotating interleaved",
                },
                "regime": {
                    "dim": REGIME_DIM,
                    "requested_sparsities": REGIME_SPARSITIES,
                    "small_batch": REGIME_SMALL_BATCH,
                    "large_batch": REGIME_LARGE_BATCH,
                    "adaptive_warmup_calls": REGIME_WARMUP_CALLS,
                    "adaptive_measure_calls": REGIME_MEASURE_CALLS,
                    "forced_warmup_rounds": REGIME_FORCED_WARMUP_ROUNDS,
                    "forced_calls_per_sample": REGIME_FORCED_CALLS_PER_SAMPLE,
                    "batch_order": "rotating across trials",
                    "path_order": "rotating interleaved",
                },
            },
            "achf_modes": {
                "lite": "production quality gate, then deterministic candidate cached/sparse fallback routing",
                "fixed_cached": "diagnostic override: force candidate selection, then request Cached execution",
                "fixed_sparse": "diagnostic override: force candidate selection, then request Sparse execution",
                "fixed_dense": "diagnostic override: force candidate selection, then request Dense execution",
                "plain_ema": "single short-window latency EMA with fixed stale probing; no cold/warm separation, long EMA blend, or hysteresis",
                "full": "guarded AMA with cold/warm and short/long EMAs, hysteresis, stale probing, and per-batch buckets",
            },
            "metric_semantics": {
                "eval_reward": "held-out reward from the active frozen policy using that policy's native reward; compare only within the same active_policy",
                "train_loss": "last training snapshot from the active policy using that policy's native loss; compare only within the same active_policy",
                "policy_train_time_ms": "active policy training only; base-model preparation excluded",
                "param_count": "active policy trainable parameters; derived sparse inference copies excluded",
                "rank": "rank applies only when candidate_mode=low_rank; sparse pruning is mutually exclusive and controlled by prune_threshold; the low-rank candidate remains materialized densely",
                "training_curve_scope": "ACHF snapshots come from the first FFN layer, or first attention layer when no FFN layer is active; aggregate candidate weight error is recomputed over every active ACHF layer",
                "gate": "quality selection blends reference and candidate; only after candidate selection do Cached/Sparse/Dense choose numerically equivalent physical execution of that candidate",
            },
        }),
    );
    for (name, agg) in all {
        let entries: Vec<serde_json::Value> = agg.iter().map(aggregated_result_json).collect();
        root.insert((*name).to_string(), serde_json::Value::Array(entries));
    }
    if let Some(latencies) = path_latencies {
        root.insert(
            "path_latency".to_string(),
            serde_json::Value::Array(path_latency_stats_json(latencies)),
        );
    }
    if let Some(result) = gate_curve {
        root.insert("gate_curve".to_string(), gate_curve_summary_json(result));
    }
    if let Some(cells) = crossover {
        root.insert(
            "crossover".to_string(),
            serde_json::Value::Array(cells.iter().map(crossover_cell_json).collect()),
        );
    }
    if let Some(rows) = regime {
        root.insert(
            "regime".to_string(),
            serde_json::Value::Array(regime_rows_json(rows)),
        );
    }
    let completed: Vec<&str> = all
        .iter()
        .map(|(name, _)| *name)
        .chain(path_latencies.is_some().then_some("path"))
        .chain(gate_curve.is_some().then_some("gate"))
        .chain(crossover.is_some().then_some("crossover"))
        .chain(regime.is_some().then_some("regime"))
        .collect();
    let missing_experiments =
        missing_benchmark_experiments(all, path_latencies, gate_curve, crossover, regime);
    let significant_crossover_cells = crossover.map_or(0, |cells| {
        cells
            .iter()
            .filter(|cell| cell.significant_winner_95.is_some())
            .count()
    });
    let mut blockers = Vec::new();
    if cfg!(debug_assertions) {
        blockers.push("debug build; rerun the release executable");
    }
    if num_trials < 5 {
        blockers.push("fewer than 5 seeded trials");
    }
    if !missing_experiments.is_empty() {
        blockers.push("partial benchmark suite");
    }
    if has_unpruned_candidate(all) || path_has_unpruned_candidate(path_latencies) {
        blockers.push(
            "at least one real ACHF candidate has zero realized sparsity; sparse-candidate speedup claims are ineligible",
        );
    }
    blockers.push("only one operating-system process repetition");
    blockers.push("only one hardware/software environment");
    blockers.push("prune-and-fine-tune and sparse-training quality baselines are not implemented");
    blockers.push(
        "time-resolved candidate-output discrepancy and ACHF-local optimizer-moment drift are not implemented",
    );
    if has_ineligible_candidate(all) {
        blockers.push("at least one materialized candidate failed production entry criteria");
    }
    root.insert(
        "publication_readiness".to_string(),
        serde_json::json!({
            "final_submission_ready": false,
            "data_tier": if cfg!(debug_assertions) || num_trials < 2 {
                "diagnostic"
            } else {
                "single-machine pilot"
            },
            "completed_experiments": completed,
            "missing_experiments": missing_experiments,
            "significant_crossover_cells_95": significant_crossover_cells,
            "claim_eligibility": {
                "runtime_crossover": crossover.is_some() && num_trials >= 2,
                "single_machine_selector_regret": regime.is_some() && num_trials >= 2,
                "task_quality_superiority": false,
                "convergence_superiority": false,
                "cross_hardware_generalization": false,
                "rank_as_storage_compression": false,
                "production_candidate_execution": !has_ineligible_candidate(all),
            },
            "blocking_reasons": blockers,
            "note": "false is intentional: software cannot manufacture independent machines, process repetitions, or missing scientific baselines",
        }),
    );
    let json = serde_json::to_string_pretty(&serde_json::Value::Object(root))
        .expect("benchmark summary JSON should be serializable");
    let path = format!("{}/summary.json", dir);
    write_text_file(&path, &json);
    println!("[Bench] JSON  -> {}", path);
}

fn trial_stats_json(stats: &TrialStats) -> serde_json::Value {
    serde_json::json!({
        "n": stats.values.len(),
        "mean": stats.mean,
        "std_dev": stats.std_dev,
        "ci_95": [stats.ci_low, stats.ci_high],
        "values": stats.values,
    })
}

fn paired_comparison_json(paired: &PairedComparison) -> serde_json::Value {
    serde_json::json!({
        "baseline": paired.baseline,
        "throughput_delta": trial_stats_json(&paired.throughput_delta),
        "throughput_relative_delta_pct": trial_stats_json(&paired.throughput_relative_delta_pct),
        "throughput_significant_95": paired.throughput_delta.ci_low.zip(paired.throughput_delta.ci_high)
            .is_some_and(|(low, high)| low > 0.0 || high < 0.0),
        "eval_reward_delta": trial_stats_json(&paired.eval_reward_delta),
        "eval_reward_significant_95": paired.eval_reward_delta.ci_low.zip(paired.eval_reward_delta.ci_high)
            .is_some_and(|(low, high)| low > 0.0 || high < 0.0),
        "train_loss_delta": trial_stats_json(&paired.train_loss_delta),
        "policy_train_time_ms_delta": trial_stats_json(&paired.train_time_ms_delta),
    })
}

fn aggregated_result_json(result: &AggregatedResult) -> serde_json::Value {
    serde_json::json!({
        "label": result.label,
        "active_policy": result.policy,
        "condition_config_fingerprint_sha256": result.config_fingerprint,
        "condition_config": result.condition_config,
        "throughput_sims_per_sec": trial_stats_json(&result.throughput),
        "eval_reward": trial_stats_json(&result.eval_reward),
        "train_loss": trial_stats_json(&result.train_loss),
        "policy_train_time_ms": trial_stats_json(&result.train_time_ms),
        "param_count": result.param_count,
        "applied_rank": result.applied_rank,
        "candidate_relative_frobenius_error": result.candidate_relative_error.as_ref().map(trial_stats_json),
        "inference_memory": memory_stats_json(&result.memory_stats),
        "paired_vs_baseline": result.paired.as_ref().map(paired_comparison_json),
        "curve": result.curve.iter().map(curve_point_json).collect::<Vec<_>>(),
        "cache_trial_count": result.cache_trial_count,
        "frozen_cache_stats": result.cache_stats.map(cache_stats_json),
    })
}

fn memory_stats_json(values: &[AchfMemoryStats]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    let stats = |value: fn(&AchfMemoryStats) -> usize| {
        TrialStats::from_values(
            &values
                .iter()
                .map(|entry| value(entry) as f64)
                .collect::<Vec<_>>(),
        )
    };
    let ratios = |value: fn(&AchfMemoryStats) -> f64| {
        TrialStats::from_values(&values.iter().map(value).collect::<Vec<_>>())
    };
    serde_json::json!({
        "semantics": "materialized runtime bytes for reference, dedicated connection logits, derived candidate, execution layouts, and exact memo input/output",
        "layers": trial_stats_json(&stats(|entry| entry.layers)),
        "candidate_total_weights": trial_stats_json(&stats(|entry| entry.candidate_total_weights)),
        "candidate_nonzero_weights": trial_stats_json(&stats(|entry| entry.candidate_nonzero_weights)),
        "candidate_layers": trial_stats_json(&stats(|entry| entry.candidate_layers)),
        "eligible_candidate_layers": trial_stats_json(&stats(|entry| entry.eligible_candidate_layers)),
        "candidate_sparsity": trial_stats_json(&ratios(|entry| {
            if entry.candidate_total_weights == 0 {
                0.0
            } else {
                1.0 - entry.candidate_nonzero_weights as f64
                    / entry.candidate_total_weights as f64
            }
        })),
        "candidate_relative_frobenius_error": trial_stats_json(&ratios(|entry| {
            entry.candidate_relative_error().unwrap_or(0.0)
        })),
        "max_layer_candidate_relative_frobenius_error": trial_stats_json(&ratios(|entry| {
            entry.max_layer_candidate_relative_error
        })),
        "reference_parameter_bytes": trial_stats_json(&stats(|entry| entry.reference_parameter_bytes)),
        "candidate_dense_bytes": trial_stats_json(&stats(|entry| entry.candidate_dense_bytes)),
        "sparse_mask_bytes": trial_stats_json(&stats(|entry| entry.sparse_mask_bytes)),
        "cached_dense_bytes": trial_stats_json(&stats(|entry| entry.cached_dense_bytes)),
        "cached_bias_bytes": trial_stats_json(&stats(|entry| entry.cached_bias_bytes)),
        "csr_row_ptr_bytes": trial_stats_json(&stats(|entry| entry.csr_row_ptr_bytes)),
        "csr_column_bytes": trial_stats_json(&stats(|entry| entry.csr_column_bytes)),
        "csr_value_bytes": trial_stats_json(&stats(|entry| entry.csr_value_bytes)),
        "connection_parameter_bytes": trial_stats_json(&stats(|entry| entry.connection_parameter_bytes)),
        "memoized_input_bytes": trial_stats_json(&stats(|entry| entry.memoized_input_bytes)),
        "memoized_output_bytes": trial_stats_json(&stats(|entry| entry.memoized_output_bytes)),
        "total_materialized_bytes": trial_stats_json(&stats(|entry| entry.total_materialized_bytes)),
        "materialization_ratio_vs_reference": trial_stats_json(&ratios(|entry| {
            entry.total_materialized_bytes as f64
                / entry.reference_parameter_bytes.max(1) as f64
        })),
        "per_trial": values,
    })
}

fn write_csvs(all: &[(&str, Vec<AggregatedResult>)], dir: &str) {
    for (name, agg) in all {
        let mut csv = String::from(
            "label,active_policy,trial,throughput_sims_per_sec,eval_reward,train_loss,policy_train_time_ms,param_count,applied_rank,candidate_relative_frobenius_error,candidate_total_weights,candidate_nonzero_weights,total_materialized_bytes\n",
        );
        for a in agg {
            let label = csv_escape(&a.label);
            let policy = csv_escape(&a.policy);
            for (trial, (((&throughput, &eval_reward), &train_loss), &train_time_ms)) in a
                .throughput
                .values
                .iter()
                .zip(a.eval_reward.values.iter())
                .zip(a.train_loss.values.iter())
                .zip(a.train_time_ms.values.iter())
                .enumerate()
            {
                csv.push_str(&format!(
                    "{},{},{},{:.2},{:.6},{:.6},{:.3},{},{},{},{},{},{}\n",
                    label,
                    policy,
                    trial + 1,
                    throughput,
                    eval_reward,
                    train_loss,
                    train_time_ms,
                    a.param_count,
                    a.applied_rank
                        .map_or_else(String::new, |rank| rank.to_string()),
                    a.candidate_relative_error
                        .as_ref()
                        .and_then(|stats| stats.values.get(trial))
                        .map_or_else(String::new, |value| format!("{value:.10}")),
                    a.memory_stats
                        .get(trial)
                        .map_or_else(String::new, |stats| stats
                            .candidate_total_weights
                            .to_string()),
                    a.memory_stats
                        .get(trial)
                        .map_or_else(String::new, |stats| stats
                            .candidate_nonzero_weights
                            .to_string()),
                    a.memory_stats
                        .get(trial)
                        .map_or_else(String::new, |stats| stats
                            .total_materialized_bytes
                            .to_string()),
                ));
            }
        }
        let path = format!("{}/{}.csv", dir, name);
        write_text_file(&path, &csv);
        println!("[Bench] CSV   -> {}", path);
    }
}

fn write_path_latency_outputs(latencies: &[PathLatencyResult], dir: &str) {
    let mut csv = String::from(
        "label,trial,input_dim,actual_sparsity,total_weights,nonzero_weights,sample,latency_ns\n",
    );
    for result in latencies {
        let label = csv_escape(&result.label);
        for (trial, values) in result.trial_samples.iter().enumerate() {
            if values.is_empty() {
                continue;
            }
            let input_dim = result.trial_input_dims[trial];
            let sparsity = result.trial_sparsity[trial];
            for (sample, latency_ns) in values.iter().enumerate() {
                csv.push_str(&format!(
                    "{},{},{},{:.8},{},{},{},{:.3}\n",
                    label,
                    trial + 1,
                    input_dim,
                    sparsity.sparsity,
                    sparsity.total_weights,
                    sparsity.nonzero_weights,
                    sample + 1,
                    latency_ns
                ));
            }
        }
    }
    let csv_path = format!("{}/path_latency.csv", dir);
    write_text_file(&csv_path, &csv);
    println!("[Bench] CSV   -> {}", csv_path);

    let json = serde_json::to_string_pretty(&serde_json::Value::Array(path_latency_stats_json(
        latencies,
    )))
    .expect("path latency summary JSON should be serializable");
    let json_path = format!("{}/path_latency_summary.json", dir);
    write_text_file(&json_path, &json);
    println!("[Bench] JSON  -> {}", json_path);
}

fn write_crossover_outputs(cells: &[CrossoverCell], dir: &str) {
    let mut csv = String::from(
        "dim,requested_sparsity,actual_sparsity,total_weights,nonzero_weights,trial,cached_ns,sparse_ns,dense_ns,winner\n",
    );
    for cell in cells {
        for trial in 0..cell.cached_ns.values.len() {
            let cached = cell.cached_ns.values[trial];
            let sparse = cell.sparse_ns.values[trial];
            let dense = cell.dense_ns.values[trial];
            let winner = if sparse <= cached && sparse <= dense {
                "Sparse"
            } else if cached <= dense {
                "Cached"
            } else {
                "Dense"
            };
            csv.push_str(&format!(
                "{},{:.6},{:.6},{},{},{},{:.3},{:.3},{:.3},{}\n",
                cell.dim,
                cell.requested_sparsity,
                cell.actual_sparsity,
                cell.total_weights,
                cell.nonzero_weights,
                trial + 1,
                cached,
                sparse,
                dense,
                winner,
            ));
        }
    }
    let csv_path = format!("{}/path_crossover.csv", dir);
    write_text_file(&csv_path, &csv);
    println!("[Bench] CSV   -> {}", csv_path);

    let arr: Vec<serde_json::Value> = cells.iter().map(crossover_cell_json).collect();
    let json = serde_json::to_string_pretty(&serde_json::Value::Array(arr))
        .expect("crossover summary JSON should be serializable");
    let json_path = format!("{}/path_crossover_summary.json", dir);
    write_text_file(&json_path, &json);
    println!("[Bench] JSON  -> {}", json_path);
}

fn crossover_cell_json(cell: &CrossoverCell) -> serde_json::Value {
    serde_json::json!({
        "dim": cell.dim,
        "requested_sparsity": cell.requested_sparsity,
        "actual_sparsity": cell.actual_sparsity,
        "total_weights": cell.total_weights,
        "nonzero_weights": cell.nonzero_weights,
        "cached_ns": trial_stats_json(&cell.cached_ns),
        "sparse_ns": trial_stats_json(&cell.sparse_ns),
        "dense_ns": trial_stats_json(&cell.dense_ns),
        "winner_by_mean": cell.winner,
        "winner_significant_95": cell.significant_winner_95,
        "paired_path_differences_ns": {
            "cached_minus_sparse": trial_stats_json(&cell.cached_minus_sparse_ns),
            "cached_minus_dense": trial_stats_json(&cell.cached_minus_dense_ns),
            "sparse_minus_dense": trial_stats_json(&cell.sparse_minus_dense_ns),
        },
    })
}

fn write_regime_outputs(rows: &[RegimeRow], dir: &str) {
    let mut csv = String::from(
        "requested_sparsity,actual_sparsity,total_weights,nonzero_weights,batch,trial,guarded_ama_ns,plain_ema_ns,cached_ns,sparse_ns,dense_ns,oracle_ns,oracle_path,guarded_ama_oracle_gap,plain_ema_oracle_gap,guarded_ama_sparse_frac,plain_ema_sparse_frac\n",
    );
    for row in rows {
        for latency in [&row.small, &row.large] {
            for trial in 0..latency.adaptive_ns.values.len() {
                csv.push_str(&format!(
                    "{:.6},{:.6},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.6},{:.6},{:.6},{:.6}\n",
                    row.requested_sparsity,
                    row.actual_sparsity,
                    row.total_weights,
                    row.nonzero_weights,
                    latency.batch,
                    trial + 1,
                    latency.adaptive_ns.values[trial],
                    latency.plain_ema_ns.values[trial],
                    latency.cached_ns.values[trial],
                    latency.sparse_ns.values[trial],
                    latency.dense_ns.values[trial],
                    latency.oracle_ns.values[trial],
                    latency.oracle_paths[trial],
                    latency.oracle_gap.values[trial],
                    latency.plain_ema_oracle_gap.values[trial],
                    latency.sparse_frac.values[trial],
                    latency.plain_ema_sparse_frac.values[trial],
                ));
            }
        }
    }
    let csv_path = format!("{}/regime_adaptation.csv", dir);
    write_text_file(&csv_path, &csv);
    println!("[Bench] CSV   -> {}", csv_path);

    let arr = regime_rows_json(rows);
    let json = serde_json::to_string_pretty(&serde_json::Value::Array(arr))
        .expect("regime summary JSON should be serializable");
    let json_path = format!("{}/regime_adaptation_summary.json", dir);
    write_text_file(&json_path, &json);
    println!("[Bench] JSON  -> {}", json_path);
}

fn regime_rows_json(rows: &[RegimeRow]) -> Vec<serde_json::Value> {
    rows.iter()
        .flat_map(|row| {
            [&row.small, &row.large].map(|latency| {
                serde_json::json!({
                    "requested_sparsity": row.requested_sparsity,
                    "actual_sparsity": row.actual_sparsity,
                    "total_weights": row.total_weights,
                    "nonzero_weights": row.nonzero_weights,
                    "batch": latency.batch,
                    "adaptive_ns": trial_stats_json(&latency.adaptive_ns),
                    "guarded_ama_ns": trial_stats_json(&latency.adaptive_ns),
                    "plain_ema_ns": trial_stats_json(&latency.plain_ema_ns),
                    "cached_ns": trial_stats_json(&latency.cached_ns),
                    "sparse_ns": trial_stats_json(&latency.sparse_ns),
                    "dense_ns": trial_stats_json(&latency.dense_ns),
                    "oracle_ns": trial_stats_json(&latency.oracle_ns),
                    "oracle_gap": trial_stats_json(&latency.oracle_gap),
                    "guarded_ama_oracle_gap": trial_stats_json(&latency.oracle_gap),
                    "plain_ema_oracle_gap": trial_stats_json(&latency.plain_ema_oracle_gap),
                    "fixed_path_oracle_gaps": {
                        "cached": trial_stats_json(&latency.cached_oracle_gap),
                        "sparse": trial_stats_json(&latency.sparse_oracle_gap),
                        "dense": trial_stats_json(&latency.dense_oracle_gap),
                    },
                    "oracle_path_by_majority": latency.oracle_path,
                    "oracle_path_counts": latency.oracle_path_counts,
                    "oracle_paths": latency.oracle_paths,
                    "sparse_fraction": trial_stats_json(&latency.sparse_frac),
                    "guarded_ama_sparse_fraction": trial_stats_json(&latency.sparse_frac),
                    "plain_ema_sparse_fraction": trial_stats_json(&latency.plain_ema_sparse_frac),
                })
            })
        })
        .collect()
}

fn write_gate_curve_outputs(result: &AggregatedResult, dir: &str) {
    let mut csv = String::from("step,samples,metric,mean,std_dev,ci_95_low,ci_95_high\n");
    for point in &result.curve {
        let metrics: [(&str, &TrialStats); 23] = [
            ("train_loss", &point.train_loss),
            ("train_reward", &point.train_reward),
            ("reference_gate", &point.gate_value),
            ("reference_gate_velocity", &point.gate_velocity),
            ("reference_gate_floor", &point.g_min),
            ("candidate_eligible_rate", &point.candidate_eligible_rate),
            ("candidate_sparsity", &point.candidate_sparsity),
            (
                "candidate_weight_relative_frobenius_error",
                &point.candidate_relative_error,
            ),
            (
                "candidate_weight_error_ema",
                &point.candidate_weight_error_ema,
            ),
            (
                "connection_candidate_weight",
                &point.connection_candidate_weight,
            ),
            ("grad_ema", &point.grad_ema),
            ("gradient_cosine", &point.gradient_cosine),
            ("cached_path_rate_within_candidate", &point.cached_path_rate),
            ("sparse_path_rate_within_candidate", &point.sparse_ratio),
            ("ema_cached_ns", &point.ema_cached_ns),
            ("ema_sparse_ns", &point.ema_sparse_ns),
            ("adaptive_bias", &point.adaptive_bias),
            (
                "connection_normalization_iterations",
                &point.connection_projection_iterations,
            ),
            (
                "connection_row_max_deviation",
                &point.connection_row_max_deviation,
            ),
            (
                "connection_col_max_deviation",
                &point.connection_col_max_deviation,
            ),
            ("connection_min_value", &point.connection_min_value),
            (
                "connection_negative_ratio",
                &point.connection_negative_ratio,
            ),
            ("low_rank_applied_rank", &point.low_rank_applied_rank),
        ];
        for (metric, stats) in metrics {
            csv.push_str(&format!(
                "{},{},{},{:.10},{:.10},{},{}\n",
                point.step,
                point.samples,
                metric,
                stats.mean,
                stats.std_dev,
                stats
                    .ci_low
                    .map_or_else(String::new, |value| format!("{value:.10}")),
                stats
                    .ci_high
                    .map_or_else(String::new, |value| format!("{value:.10}")),
            ));
        }
    }
    let csv_path = format!("{}/gate_curve.csv", dir);
    write_text_file(&csv_path, &csv);
    println!("[Bench] CSV   -> {}", csv_path);

    let json = serde_json::to_string_pretty(&gate_curve_summary_json(result))
        .expect("gate curve summary JSON should be serializable");
    let json_path = format!("{}/gate_curve_summary.json", dir);
    write_text_file(&json_path, &json);
    println!("[Bench] JSON  -> {}", json_path);
}

fn path_latency_stats(latencies: &[PathLatencyResult]) -> Vec<PathLatencyStats> {
    latencies
        .iter()
        .filter(|result| result.trial_samples.iter().any(|values| !values.is_empty()))
        .map(|result| {
            let trial_means: Vec<f64> = result
                .trial_samples
                .iter()
                .filter(|values| !values.is_empty())
                .map(|values| values.iter().sum::<f64>() / values.len() as f64)
                .collect();
            let trial_stats = TrialStats::from_values(&trial_means);
            let mut sorted: Vec<f64> = result.trial_samples.iter().flatten().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let samples = sorted.len();
            PathLatencyStats {
                label: result.label.clone(),
                trials: trial_stats.values.len(),
                samples,
                mean_ns: trial_stats.mean,
                std_dev_ns: trial_stats.std_dev,
                ci_low_ns: trial_stats.ci_low,
                ci_high_ns: trial_stats.ci_high,
                min_ns: sorted[0],
                p50_ns: percentile_sorted(&sorted, 50),
                p90_ns: percentile_sorted(&sorted, 90),
                p95_ns: percentile_sorted(&sorted, 95),
                p99_ns: percentile_sorted(&sorted, 99),
                max_ns: sorted[samples - 1],
            }
        })
        .collect()
}

fn path_latency_stats_json(latencies: &[PathLatencyResult]) -> Vec<serde_json::Value> {
    path_latency_stats(latencies)
        .into_iter()
        .zip(
            latencies
                .iter()
                .filter(|result| result.trial_samples.iter().any(|values| !values.is_empty())),
        )
        .map(|(stats, result)| {
            serde_json::json!({
                "label": stats.label,
                "trials": stats.trials,
                "samples": stats.samples,
                "mean_ns": stats.mean_ns,
                "std_dev_ns": stats.std_dev_ns,
                "ci_95_ns": [stats.ci_low_ns, stats.ci_high_ns],
                "min_ns": stats.min_ns,
                "p50_ns": stats.p50_ns,
                "p90_ns": stats.p90_ns,
                "p95_ns": stats.p95_ns,
                "p99_ns": stats.p99_ns,
                "max_ns": stats.max_ns,
                "trial_operating_points": result.trial_input_dims.iter()
                    .zip(result.trial_sparsity.iter())
                    .enumerate()
                    .map(|(trial, (&input_dim, sparsity))| serde_json::json!({
                        "trial": trial + 1,
                        "input_dim": input_dim,
                        "actual_sparsity": sparsity.sparsity,
                        "total_weights": sparsity.total_weights,
                        "nonzero_weights": sparsity.nonzero_weights,
                    }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect()
}

fn gate_curve_summary_json(result: &AggregatedResult) -> serde_json::Value {
    let mut value = aggregated_result_json(result);
    value["curve_point_count"] = serde_json::json!(result.curve.len());
    value["final_step"] = serde_json::json!(result.curve.last().map_or(0, |point| point.step));
    value["final_curve_point"] = result
        .curve
        .last()
        .map(curve_point_json)
        .unwrap_or(serde_json::Value::Null);
    value
}

fn curve_point_json(point: &CurvePointStats) -> serde_json::Value {
    serde_json::json!({
        "step": point.step,
        "samples": point.samples,
        "train_loss": trial_stats_json(&point.train_loss),
        "train_reward": trial_stats_json(&point.train_reward),
        "reference_gate": trial_stats_json(&point.gate_value),
        "reference_gate_velocity": trial_stats_json(&point.gate_velocity),
        "reference_gate_floor": trial_stats_json(&point.g_min),
        "candidate_eligible_rate": trial_stats_json(&point.candidate_eligible_rate),
        "candidate_sparsity": trial_stats_json(&point.candidate_sparsity),
        "candidate_weight_relative_frobenius_error": trial_stats_json(
            &point.candidate_relative_error
        ),
        "candidate_weight_error_ema": trial_stats_json(&point.candidate_weight_error_ema),
        "connection_candidate_weight": trial_stats_json(
            &point.connection_candidate_weight
        ),
        "grad_ema": trial_stats_json(&point.grad_ema),
        "gradient_cosine": trial_stats_json(&point.gradient_cosine),
        "cached_path_rate_within_candidate": trial_stats_json(&point.cached_path_rate),
        "sparse_path_rate_within_candidate": trial_stats_json(&point.sparse_ratio),
        "ema_cached_ns": trial_stats_json(&point.ema_cached_ns),
        "ema_sparse_ns": trial_stats_json(&point.ema_sparse_ns),
        "adaptive_bias": trial_stats_json(&point.adaptive_bias),
        "connection_normalization_iterations": trial_stats_json(&point.connection_projection_iterations),
        "connection_row_max_deviation": trial_stats_json(&point.connection_row_max_deviation),
        "connection_col_max_deviation": trial_stats_json(&point.connection_col_max_deviation),
        "connection_min_value": trial_stats_json(&point.connection_min_value),
        "connection_negative_ratio": trial_stats_json(&point.connection_negative_ratio),
        "low_rank_applied_rank": trial_stats_json(&point.low_rank_applied_rank),
    })
}

fn cache_stats_json(stats: AchfCacheStats) -> serde_json::Value {
    let calls = stats.calls as f64;
    let rate = |n: u64| if calls > 0.0 { n as f64 / calls } else { 0.0 };
    let candidate_calls = stats.candidate_paths as f64;
    let candidate_rate = |n: u64| {
        if candidate_calls > 0.0 {
            n as f64 / candidate_calls
        } else {
            0.0
        }
    };
    let fastest_path_ns = [stats.ema_cached_ns, stats.ema_sparse_ns, stats.ema_dense_ns]
        .into_iter()
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min);
    let selector_overhead_ratio = fastest_path_ns
        .map(|path_ns| stats.decision_ema_ns / (stats.decision_ema_ns + path_ns).max(f64::EPSILON));
    serde_json::json!({
        "calls": stats.calls,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "cache_skips": stats.cache_skips,
        "memo_hits": stats.memo_hits,
        "reference_paths": stats.reference_paths,
        "candidate_paths": stats.candidate_paths,
        "candidate_rejections": stats.candidate_rejections,
        "sparse_paths": stats.sparse_paths,
        "dense_paths": stats.dense_paths,
        "memo_hit_rate": rate(stats.memo_hits),
        "reference_execution_rate": rate(stats.reference_paths),
        "candidate_execution_rate": rate(stats.candidate_paths),
        "candidate_rejection_rate": rate(stats.candidate_rejections),
        "cached_path_rate_within_candidate": candidate_rate(stats.cache_hits),
        "sparse_path_rate_within_candidate": candidate_rate(stats.sparse_paths),
        "dense_path_rate_within_candidate": candidate_rate(stats.dense_paths),
        "ema_cached_ns": stats.ema_cached_ns,
        "ema_cached_long_ns": stats.ema_cached_long_ns,
        "ema_sparse_ns": stats.ema_sparse_ns,
        "ema_sparse_long_ns": stats.ema_sparse_long_ns,
        "ema_dense_ns": stats.ema_dense_ns,
        "ema_dense_long_ns": stats.ema_dense_long_ns,
        "decision_ema_ns": stats.decision_ema_ns,
        "decision_ema_long_ns": stats.decision_ema_long_ns,
        "selector_overhead_ratio_vs_fastest_ema": selector_overhead_ratio,
        "cold_warm_latency_ema_ns": {
            "cached": {
                "cold": stats.cached_cold_ema_ns,
                "warm": stats.cached_warm_ema_ns,
            },
            "sparse": {
                "cold": stats.sparse_cold_ema_ns,
                "warm": stats.sparse_warm_ema_ns,
            },
            "dense": {
                "cold": stats.dense_cold_ema_ns,
                "warm": stats.dense_warm_ema_ns,
            },
        },
        "path_warmness": {
            "cached": stats.cached_warmness,
            "sparse": stats.sparse_warmness,
            "dense": stats.dense_warmness,
        },
        "stale_age_calls": {
            "cached": stats.cached_stale_age,
            "sparse": stats.sparse_stale_age,
            "dense": stats.dense_stale_age,
        },
        "path_switches": stats.path_switches,
        "path_probes": stats.path_probes,
        "switch_rate_per_call": rate(stats.path_switches),
        "probe_rate_per_call": rate(stats.path_probes),
        "adaptive_bias": stats.adaptive_bias,
        "latency_samples": stats.latency_samples,
        "dense_latency_samples": stats.dense_latency_samples,
        "decision_samples": stats.decision_samples,
    })
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_chart(path: &str, result: Result<(), Box<dyn std::error::Error>>) {
    match result {
        Ok(()) => println!("  -> {}", path),
        Err(err) => panic!("failed to write benchmark chart {path}: {err}"),
    }
}

fn write_text_file(path: &str, content: &str) {
    try_write_text_file(path, content)
        .unwrap_or_else(|err| panic!("failed to write benchmark output {path}: {err}"));
}

fn try_write_text_file(path: &str, content: &str) -> std::io::Result<()> {
    fs::write(path, content)
}

fn percentile(data: &[f64], pct: usize) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, pct)
}

fn percentile_sorted(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = (pct.min(100) as f64 / 100.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "talos_xii_bench_{}_{}_{}",
            prefix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_hardware_snapshot_reports_physical_memory() {
        let snapshot = hardware_snapshot();
        assert!(
            snapshot["memory_bytes"]
                .as_u64()
                .is_some_and(|bytes| bytes > 0),
            "hardware snapshot must record physical memory: {snapshot}"
        );
        assert_eq!(snapshot["memory_source"], "GlobalMemoryStatusEx");
    }

    #[test]
    fn output_preparation_removes_only_known_stale_artifacts() {
        let dir = unique_temp_dir("prepare_output");
        std::fs::write(dir.join("summary.json"), "{}").unwrap();
        std::fs::write(dir.join("path_crossover_dim256.svg"), "<svg/>").unwrap();
        std::fs::write(dir.join("keep-me.txt"), "user data").unwrap();

        prepare_output_directory(&dir.to_string_lossy());

        assert!(!dir.join("summary.json").exists());
        assert!(!dir.join("path_crossover_dim256.svg").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join("keep-me.txt")).unwrap(),
            "user data"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn finalized_manifest_hashes_every_output_artifact() {
        let dir = unique_temp_dir("manifest");
        std::fs::write(dir.join("summary.txt"), "result").unwrap();
        let mut manifest = serde_json::json!({"status": "running"});

        finalize_run_manifest(
            &dir.to_string_lossy(),
            &mut manifest,
            std::time::Duration::from_millis(25),
        );

        let persisted: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("run_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(persisted["status"], "complete");
        assert_eq!(persisted["artifacts"][0]["file"], "summary.txt");
        assert_eq!(persisted["artifacts"][0]["sha256"], sha256_hex(b"result"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn bench_config_with_only(only: Option<Vec<String>>) -> BenchConfig {
        BenchConfig {
            output_dir: "unused".to_string(),
            format: ChartFormat::Svg,
            only,
            num_trials: 1,
        }
    }

    fn test_snapshot(step: usize, loss: f64, reward: f64) -> StepSnapshot {
        StepSnapshot {
            step,
            gate_value: 0.8,
            gate_velocity: -0.01,
            g_min: 0.2,
            grad_ema: 0.3,
            candidate_eligible: true,
            candidate_sparsity: 0.75,
            candidate_relative_error: 0.01,
            candidate_weight_error_ema: 0.01,
            connection_candidate_weight: 0.6,
            gradient_cosine: 0.25,
            loss,
            reward,
            cached_path_rate: 0.5,
            sparse_ratio: 0.75,
            ema_cached_ns: 11.0,
            ema_sparse_ns: 22.0,
            adaptive_bias: 1.1,
            connection_projection_iterations: 20,
            connection_row_max_deviation: 0.0001,
            connection_col_max_deviation: 0.0002,
            connection_min_value: 0.01,
            connection_negative_ratio: 0.0,
            low_rank_applied_rank: 0,
        }
    }

    fn test_cache_stats(calls: u64, hits: u64) -> AchfCacheStats {
        AchfCacheStats {
            calls,
            cache_hits: hits,
            cache_misses: 1,
            candidate_paths: calls,
            cache_skips: 1,
            sparse_paths: calls.saturating_sub(hits + 1),
            dense_paths: 1,
            ema_cached_ns: 11.0,
            ema_cached_long_ns: 12.0,
            ema_sparse_ns: 22.0,
            ema_sparse_long_ns: 23.0,
            ema_dense_ns: 33.0,
            ema_dense_long_ns: 34.0,
            decision_ema_ns: 4.0,
            decision_ema_long_ns: 5.0,
            adaptive_bias: 1.1,
            latency_samples: calls,
            dense_latency_samples: 2,
            decision_samples: calls,
            ..Default::default()
        }
    }

    fn test_run(
        label: &str,
        throughput: f64,
        eval_reward: f64,
        train_loss: f64,
        train_time_ms: f64,
        snapshots: Vec<StepSnapshot>,
    ) -> BenchRunResult {
        BenchRunResult {
            label: label.to_string(),
            policy: "PPO".to_string(),
            config_fingerprint: "test-config".to_string(),
            condition_config: serde_json::json!({"test": true}),
            train_time_ms,
            throughput_sims_per_sec: throughput,
            eval_reward,
            train_loss,
            param_count: 42,
            applied_rank: Some(0),
            candidate_relative_error: Some(0.125),
            memory_stats: Some(AchfMemoryStats {
                layers: 1,
                candidate_total_weights: 16,
                candidate_nonzero_weights: 8,
                reference_parameter_bytes: 64,
                total_materialized_bytes: 160,
                ..Default::default()
            }),
            snapshots,
            cache_stats: Some(test_cache_stats(8, 2)),
        }
    }

    #[test]
    fn parse_only_filter_normalizes_and_drops_empty_entries() {
        assert_eq!(
            parse_only_filter(" path, Gate ,,CONVERGENCE "),
            vec![
                "path".to_string(),
                "gate".to_string(),
                "convergence".to_string()
            ]
        );
    }

    #[test]
    fn validate_only_filter_accepts_known_experiments_case_insensitively() {
        let cfg = bench_config_with_only(Some(vec!["Path".to_string(), "GATE".to_string()]));
        validate_bench_config(&cfg).unwrap();
        assert!(should_run(&cfg, "path"));
        assert!(should_run(&cfg, "gate"));
        assert!(!should_run(&cfg, "scale"));
    }

    #[test]
    fn validate_only_filter_rejects_unknown_experiments() {
        let cfg = bench_config_with_only(Some(vec!["path".to_string(), "missing".to_string()]));
        let message = validate_bench_config(&cfg).unwrap_err();
        assert!(message.contains("unknown benchmark experiment(s): missing"));
    }

    #[test]
    fn validate_bench_config_rejects_zero_trials() {
        let cfg = BenchConfig {
            output_dir: "unused".to_string(),
            format: ChartFormat::Svg,
            only: None,
            num_trials: 0,
        };
        assert_eq!(
            validate_bench_config(&cfg).unwrap_err(),
            "benchmark trials must be at least 1"
        );
    }

    #[test]
    fn parse_chart_format_rejects_unknown_formats() {
        assert!(matches!(
            parse_chart_format("svg").unwrap(),
            ChartFormat::Svg
        ));
        assert!(matches!(
            parse_chart_format("PNG").unwrap(),
            ChartFormat::Png
        ));
        assert!(parse_chart_format("jpg").is_err());
    }

    #[test]
    fn bench_sized_config_clamps_large_model_for_smoke_benchmarks() {
        let cfg = Config {
            model_dim: 2048,
            model_hidden_dim: 8192,
            model_num_layers: 24,
            model_num_heads: 32,
            model_kv_lora_rank: 1024,
            model_qk_rope_dim: 256,
            multi_stream_factor: 4,
            ..Config::default()
        };

        let bench_cfg = bench_sized_config(&cfg);

        assert!(bench_cfg.fast_init);
        assert_eq!(bench_cfg.model_dim, crate::neural::DIM);
        assert_eq!(bench_cfg.model_hidden_dim, 64);
        assert_eq!(bench_cfg.model_num_layers, 2);
        assert!(bench_cfg.model_num_heads <= 4);
        assert_eq!(bench_cfg.model_hidden_dim % bench_cfg.model_num_heads, 0);
        assert!(bench_cfg.model_kv_lora_rank <= 16);
        assert!(bench_cfg.model_qk_rope_dim <= 4);
        assert_eq!(bench_cfg.model_qk_rope_dim % 2, 0);
        assert!(bench_cfg.multi_stream_factor <= 2);
        assert_eq!(bench_cfg.luck_mode, LuckMode::Ppo);
    }

    #[test]
    fn write_text_file_reports_io_errors() {
        let dir = unique_temp_dir("write_error");
        let path = dir.to_string_lossy().to_string();

        let result = try_write_text_file(&path, "content");
        std::fs::remove_dir(&dir).unwrap();

        assert!(result.is_err());
    }

    #[test]
    fn summary_text_marks_partial_suite() {
        let dir = unique_temp_dir("partial_summary");
        let output_dir = dir.to_string_lossy().to_string();

        write_summary_txt(&[], None, None, None, None, 5, &output_dir);

        let summary_path = dir.join("summary.txt");
        let summary = std::fs::read_to_string(&summary_path).unwrap();
        assert!(summary.contains("blocker: partial benchmark suite"));
        assert!(summary
            .contains("missing=ablation,mode,path,gate,scale,apply,convergence,crossover,regime"));

        std::fs::remove_file(summary_path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn write_summary_json_escapes_labels() {
        let dir = unique_temp_dir("json");
        let output_dir = dir.to_string_lossy().to_string();
        let result = aggregate_trials(&[test_run(
            "label,with\"quote",
            10.0,
            0.5,
            0.25,
            12.0,
            vec![test_snapshot(10, 0.25, 1.0)],
        )]);

        write_summary_json(
            &[("exp", vec![result])],
            None,
            None,
            None,
            None,
            (7, 1),
            &output_dir,
        );

        let json_path = dir.join("summary.json");
        let json = std::fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["exp"][0]["label"], "label,with\"quote");
        assert_eq!(parsed["metadata"]["schema_version"], 4);
        assert_eq!(parsed["exp"][0]["active_policy"], "PPO");
        assert_eq!(parsed["exp"][0]["eval_reward"]["mean"], 0.5);
        assert_eq!(
            parsed["exp"][0]["candidate_relative_frobenius_error"]["mean"],
            0.125
        );
        assert!(parsed["exp"][0]["eval_reward"]["ci_95"][0].is_null());
        assert_eq!(parsed["exp"][0]["frozen_cache_stats"]["calls"], 8);
        assert_eq!(
            parsed["exp"][0]["frozen_cache_stats"]["cached_path_rate_within_candidate"],
            0.25
        );

        std::fs::remove_file(json_path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn csv_escape_quotes_commas_quotes_and_newlines() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("a,b\"c"), "\"a,b\"\"c\"");
        assert_eq!(csv_escape("line\nbreak"), "\"line\nbreak\"");
    }

    #[test]
    fn trial_stats_use_sample_size_specific_student_t() {
        let one = TrialStats::from_values(&[3.0]);
        assert_eq!(one.std_dev, 0.0);
        assert_eq!(one.ci_low, None);
        assert_eq!(one.ci_high, None);

        let five = TrialStats::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((five.mean - 3.0).abs() < 1e-12);
        assert!((student_t_critical_95(4) - 2.776).abs() < 1e-12);
        assert!((five.ci_low.unwrap() - 1.037_071_575).abs() < 1e-8);
        assert!((five.ci_high.unwrap() - 4.962_928_425).abs() < 1e-8);
        assert!((student_t_critical_95(9) - 2.262).abs() < 1e-12);
    }

    #[test]
    fn domain_seeds_are_deterministic_and_separated() {
        let trial = benchmark_trial_seed(1234, 2);
        assert_eq!(trial, benchmark_trial_seed(1234, 2));
        assert_ne!(trial, benchmark_trial_seed(1234, 3));
        let domains = [
            derive_seed(trial, SEED_BASE_MODELS),
            derive_seed(trial, SEED_DQN_TRAIN),
            derive_seed(trial, SEED_PPO_TRAIN),
            derive_seed(trial, SEED_POLICY_EVAL),
            derive_seed(trial, SEED_THROUGHPUT),
        ];
        for left in 0..domains.len() {
            for right in left + 1..domains.len() {
                assert_ne!(domains[left], domains[right]);
            }
        }
    }

    #[test]
    fn aggregation_uses_all_trials_and_builds_paired_deltas() {
        let baseline = vec![
            test_run(
                "baseline",
                100.0,
                1.0,
                0.5,
                20.0,
                vec![test_snapshot(10, 0.5, 1.0), test_snapshot(20, 0.4, 1.2)],
            ),
            test_run(
                "baseline",
                120.0,
                2.0,
                0.3,
                24.0,
                vec![test_snapshot(10, 0.3, 2.0)],
            ),
        ];
        let candidate = vec![
            test_run(
                "candidate",
                110.0,
                1.5,
                0.4,
                18.0,
                vec![test_snapshot(10, 0.4, 1.5)],
            ),
            test_run(
                "candidate",
                150.0,
                2.5,
                0.2,
                20.0,
                vec![test_snapshot(10, 0.2, 2.5), test_snapshot(20, 0.1, 2.7)],
            ),
        ];
        let aggregated = aggregate_conditions(&[baseline, candidate], &[0, 0]);

        assert_eq!(aggregated[0].curve.len(), 2);
        assert_eq!(aggregated[0].curve[0].samples, 2);
        assert_eq!(aggregated[0].curve[1].samples, 1);
        assert_eq!(aggregated[0].curve[0].train_loss.mean, 0.4);
        assert_eq!(aggregated[0].cache_trial_count, 2);
        assert_eq!(aggregated[0].cache_stats.unwrap().calls, 16);
        let paired = aggregated[1].paired.as_ref().unwrap();
        assert_eq!(paired.baseline, "baseline");
        assert_eq!(paired.throughput_delta.values, vec![10.0, 30.0]);
        assert_eq!(paired.eval_reward_delta.values, vec![0.5, 0.5]);
        assert!(paired
            .train_loss_delta
            .values
            .iter()
            .all(|value| (*value + 0.1).abs() < 1e-12));
        assert_eq!(paired.train_time_ms_delta.values, vec![-2.0, -4.0]);
    }

    #[test]
    fn path_latency_outputs_include_stats_in_csv_json_and_summary() {
        let dir = unique_temp_dir("path_latency");
        let output_dir = dir.to_string_lossy().to_string();
        let latencies = vec![
            PathLatencyResult {
                label: "Dense,path".to_string(),
                trial_samples: vec![vec![30.0, 10.0], vec![20.0, 40.0]],
                trial_input_dims: vec![64, 64],
                trial_sparsity: vec![
                    AchfSparsityStats {
                        total_weights: 4096,
                        nonzero_weights: 1024,
                        zero_weights: 3072,
                        sparsity: 0.75,
                    },
                    AchfSparsityStats {
                        total_weights: 4096,
                        nonzero_weights: 1024,
                        zero_weights: 3072,
                        sparsity: 0.75,
                    },
                ],
            },
            PathLatencyResult {
                label: "Empty".to_string(),
                trial_samples: vec![Vec::new()],
                trial_input_dims: Vec::new(),
                trial_sparsity: Vec::new(),
            },
        ];

        let stats = path_latency_stats(&latencies);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].label, "Dense,path");
        assert_eq!(stats[0].trials, 2);
        assert_eq!(stats[0].samples, 4);
        assert_eq!(stats[0].mean_ns, 25.0);
        assert_eq!(stats[0].min_ns, 10.0);
        assert_eq!(stats[0].p50_ns, 25.0);
        assert_eq!(stats[0].p95_ns, 38.5);

        write_path_latency_outputs(&latencies, &output_dir);
        write_summary_json(&[], Some(&latencies), None, None, None, (1, 2), &output_dir);

        let csv = std::fs::read_to_string(dir.join("path_latency.csv")).unwrap();
        assert!(csv.contains("\"Dense,path\",1,64,0.75000000,4096,1024,1,30.000"));
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(json["path_latency"][0]["label"], "Dense,path");
        assert_eq!(json["path_latency"][0]["trials"], 2);
        assert_eq!(
            json["path_latency"][0]["trial_operating_points"][0]["nonzero_weights"],
            1024
        );
        assert_eq!(json["path_latency"][0]["p99_ns"], 39.7);
        let summary: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("path_latency_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary[0]["samples"], 4);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn synthetic_achf_layer_hits_target_sparsity_and_paths_agree() {
        // The synthetic builder must produce the requested weight sparsity
        // exactly (CSR nnz), and all three forced paths must agree numerically
        // on a frozen layer — otherwise the crossover/regime timings would be
        // comparing paths that compute different things.
        let dim = 64usize;
        let layer = build_synthetic_achf_layer(dim, 0.75, false, 123);
        let sparsity = layer.inference_sparsity_stats().unwrap();
        assert_eq!(sparsity.total_weights, 4096);
        assert_eq!(sparsity.nonzero_weights, 1024);
        assert_eq!(sparsity.zero_weights, 3072);
        assert!((sparsity.sparsity - 0.75).abs() < f64::EPSILON);
        assert_eq!(layer.snapshot_state().low_rank_applied_rank, 0);
        let x: Vec<f32> = (0..dim).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        let cached = layer.forward_inference_forced_path(&x, 0);
        let sparse = layer.forward_inference_forced_path(&x, 1);
        let dense = layer.forward_inference_forced_path(&x, 2);
        assert_eq!(cached.len(), dense.len());
        assert_eq!(sparse.len(), dense.len());
        for i in 0..dense.len() {
            assert!(
                (cached[i] - dense[i]).abs() < 1e-4,
                "cached vs dense at {i}"
            );
            assert!(
                (sparse[i] - dense[i]).abs() < 1e-4,
                "sparse vs dense at {i}"
            );
        }
    }

    #[test]
    fn crossover_outputs_write_parseable_csv_and_json() {
        let dir = unique_temp_dir("crossover");
        let output_dir = dir.to_string_lossy().to_string();
        let cells = vec![
            CrossoverCell {
                dim: 256,
                requested_sparsity: 0.9,
                actual_sparsity: 0.898_437_5,
                total_weights: 65_536,
                nonzero_weights: 6_656,
                cached_ns: TrialStats::from_values(&[100.0, 110.0]),
                sparse_ns: TrialStats::from_values(&[80.0, 90.0]),
                dense_ns: TrialStats::from_values(&[120.0, 130.0]),
                winner: "Sparse".to_string(),
                significant_winner_95: Some("Sparse".to_string()),
                cached_minus_sparse_ns: TrialStats::from_values(&[20.0, 20.0]),
                cached_minus_dense_ns: TrialStats::from_values(&[-20.0, -20.0]),
                sparse_minus_dense_ns: TrialStats::from_values(&[-40.0, -40.0]),
            },
            CrossoverCell {
                dim: 256,
                requested_sparsity: 0.5,
                actual_sparsity: 0.5,
                total_weights: 65_536,
                nonzero_weights: 32_768,
                cached_ns: TrialStats::from_values(&[70.0, 75.0]),
                sparse_ns: TrialStats::from_values(&[300.0, 310.0]),
                dense_ns: TrialStats::from_values(&[90.0, 95.0]),
                winner: "Cached".to_string(),
                significant_winner_95: Some("Cached".to_string()),
                cached_minus_sparse_ns: TrialStats::from_values(&[-230.0, -235.0]),
                cached_minus_dense_ns: TrialStats::from_values(&[-20.0, -20.0]),
                sparse_minus_dense_ns: TrialStats::from_values(&[210.0, 215.0]),
            },
        ];
        write_crossover_outputs(&cells, &output_dir);
        write_summary_json(&[], None, None, Some(&cells), None, (1, 2), &output_dir);
        let csv = std::fs::read_to_string(dir.join("path_crossover.csv")).unwrap();
        assert!(csv.contains("requested_sparsity,actual_sparsity,total_weights,nonzero_weights"));
        assert!(csv.contains("256,0.900000,0.898438,65536,6656,1,100.000,80.000,120.000,Sparse"));
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("path_crossover_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(json[0]["winner_by_mean"], "Sparse");
        assert_eq!(json[0]["cached_ns"]["n"], 2);
        assert_eq!(json[1]["dim"], 256);
        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["crossover"][0]["winner_by_mean"], "Sparse");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn regime_outputs_write_parseable_csv_and_json_with_oracle_gap() {
        let dir = unique_temp_dir("regime");
        let output_dir = dir.to_string_lossy().to_string();
        let rows = vec![RegimeRow {
            requested_sparsity: 0.9,
            actual_sparsity: 0.899_414_062_5,
            total_weights: 1_048_576,
            nonzero_weights: 105_472,
            small: aggregate_regime_trials(&[RegimeTrialLatency {
                batch: 1,
                adaptive_ns: 90.0,
                plain_ema_ns: 100.0,
                cached_ns: 120.0,
                sparse_ns: 70.0,
                dense_ns: 140.0,
                oracle_ns: 70.0,
                oracle_path: "Sparse".to_string(),
                sparse_frac: 0.6,
                plain_ema_sparse_frac: 0.5,
            }]),
            large: aggregate_regime_trials(&[RegimeTrialLatency {
                batch: 128,
                adaptive_ns: 210.0,
                plain_ema_ns: 220.0,
                cached_ns: 200.0,
                sparse_ns: 280.0,
                dense_ns: 250.0,
                oracle_ns: 200.0,
                oracle_path: "Cached".to_string(),
                sparse_frac: 0.1,
                plain_ema_sparse_frac: 0.2,
            }]),
        }];
        write_regime_outputs(&rows, &output_dir);
        write_summary_json(&[], None, None, None, Some(&rows), (1, 1), &output_dir);
        let csv = std::fs::read_to_string(dir.join("regime_adaptation.csv")).unwrap();
        // Small-batch oracle is Sparse, large-batch oracle is Cached: the two
        // regimes have different best fixed paths — the core adaptation result.
        assert!(csv.contains(
            "0.900000,0.899414,1048576,105472,1,1,90.000,100.000,120.000,70.000,140.000,70.000,Sparse,1.285714,1.428571,0.600000,0.500000"
        ));
        assert!(csv.contains(
            "0.900000,0.899414,1048576,105472,128,1,210.000,220.000,200.000,280.000,250.000,200.000,Cached,1.050000,1.100000,0.100000,0.200000"
        ));
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("regime_adaptation_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(json[0]["oracle_path_by_majority"], "Sparse");
        assert_eq!(json[1]["oracle_path_by_majority"], "Cached");
        assert_eq!(json[0]["oracle_gap"]["n"], 1);
        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["regime"][1]["oracle_path_by_majority"], "Cached");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn gate_curve_outputs_include_snapshot_and_cache_summary() {
        let dir = unique_temp_dir("gate_curve");
        let output_dir = dir.to_string_lossy().to_string();
        let mut second = test_snapshot(10, 0.375, 1.75);
        second.gate_value = 1.0;
        let result = aggregate_trials(&[
            test_run(
                "Gate Curve",
                100.0,
                1.25,
                0.125,
                12.5,
                vec![test_snapshot(10, 0.125, 1.25)],
            ),
            test_run("Gate Curve", 120.0, 1.75, 0.375, 15.5, vec![second]),
        ]);

        write_gate_curve_outputs(&result, &output_dir);
        write_summary_json(&[], None, Some(&result), None, None, (1, 2), &output_dir);

        let csv = std::fs::read_to_string(dir.join("gate_curve.csv")).unwrap();
        assert!(csv.contains("10,2,reference_gate,0.9000000000"));
        assert!(csv.contains("10,2,candidate_eligible_rate,1.0000000000"));
        assert!(csv.contains("10,2,candidate_weight_relative_frobenius_error,0.0100000000"));
        assert!(csv.contains("10,2,cached_path_rate_within_candidate,0.5000000000"));
        assert!(csv.contains("10,2,train_loss,0.2500000000"));
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(json["gate_curve"]["curve_point_count"], 1);
        assert_eq!(json["gate_curve"]["final_curve_point"]["step"], 10);
        assert_eq!(
            json["gate_curve"]["final_curve_point"]["reference_gate"]["mean"],
            0.9
        );
        assert_eq!(
            json["gate_curve"]["final_curve_point"]["candidate_weight_relative_frobenius_error"]
                ["mean"],
            0.01
        );
        assert_eq!(
            json["gate_curve"]["final_curve_point"]["candidate_eligible_rate"]["mean"],
            1.0
        );
        assert_eq!(
            json["gate_curve"]["frozen_cache_stats"]["cached_path_rate_within_candidate"],
            0.25
        );
        // All candidate-internal path rates share `candidate_paths` (=8) and are
        // mutually consistent: cached 2/8, sparse 5/8, dense 1/8. This guards
        // against mixing total calls with candidate execution denominators.
        assert_eq!(
            json["gate_curve"]["frozen_cache_stats"]["cached_path_rate_within_candidate"],
            0.25
        );
        assert_eq!(
            json["gate_curve"]["frozen_cache_stats"]["sparse_path_rate_within_candidate"],
            0.625
        );
        assert_eq!(
            json["gate_curve"]["frozen_cache_stats"]["dense_path_rate_within_candidate"],
            0.125
        );
        let summary: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("gate_curve_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["final_step"], 10);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
