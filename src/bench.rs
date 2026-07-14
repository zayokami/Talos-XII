use crate::achf::{aggregate_cache_stats_iter, AchfCacheStats, AchfLayer, AchfSparsityStats};
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
use std::time::Instant;

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
const PATH_RANK: usize = 16;
const PATH_PRUNE_THRESHOLD: f64 = 0.0;
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

// ── Data structures ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BenchRunResult {
    pub label: String,
    pub policy: String,
    pub train_time_ms: f64,
    pub throughput_sims_per_sec: f64,
    pub eval_reward: f64,
    pub train_loss: f64,
    pub param_count: usize,
    pub applied_rank: Option<usize>,
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
    pub g_min: TrialStats,
    pub grad_ema: TrialStats,
    pub cache_hit_rate: TrialStats,
    pub sparse_ratio: TrialStats,
    pub ema_cached_ns: TrialStats,
    pub ema_sparse_ns: TrialStats,
    pub adaptive_bias: TrialStats,
    pub sinkhorn_iterations: TrialStats,
    pub sinkhorn_row_max_dev: TrialStats,
    pub sinkhorn_col_max_dev: TrialStats,
    pub sinkhorn_min_value: TrialStats,
    pub sinkhorn_negative_ratio: TrialStats,
    pub sinkhorn_warm_started_rate: f64,
    pub low_rank_applied_rank: TrialStats,
}

#[derive(Clone, Debug)]
pub struct AggregatedResult {
    pub label: String,
    pub policy: String,
    pub throughput: TrialStats,
    pub eval_reward: TrialStats,
    pub train_loss: TrialStats,
    pub train_time_ms: TrialStats,
    pub param_count: usize,
    pub applied_rank: Option<usize>,
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
    cached_ns: TrialStats,
    sparse_ns: TrialStats,
    dense_ns: TrialStats,
    oracle_ns: TrialStats,
    oracle_gap: TrialStats,
    oracle_path: String,
    oracle_path_counts: BTreeMap<String, usize>,
    oracle_paths: Vec<String>,
    sparse_frac: TrialStats,
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
    cached_ns: f64,
    sparse_ns: f64,
    dense_ns: f64,
    oracle_ns: f64,
    oracle_path: String,
    sparse_frac: f64,
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
            .all(|run| run.param_count == runs[0].param_count),
        "parameter count changed across trials for {label}"
    );
    assert!(
        runs.iter()
            .all(|run| run.applied_rank == runs[0].applied_rank),
        "applied rank changed across trials for {label}"
    );
    let cache_values: Vec<AchfCacheStats> = runs.iter().filter_map(|run| run.cache_stats).collect();
    let cache_stats = (!cache_values.is_empty())
        .then(|| aggregate_cache_stats_iter(cache_values.iter().copied()));
    AggregatedResult {
        label,
        policy: runs[0].policy.clone(),
        throughput: TrialStats::from_values(&tputs),
        eval_reward: TrialStats::from_values(&rewards),
        train_loss: TrialStats::from_values(&losses),
        train_time_ms: TrialStats::from_values(&times),
        param_count: runs[0].param_count,
        applied_rank: runs[0].applied_rank,
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
                g_min: stats(|snapshot| snapshot.g_min),
                grad_ema: stats(|snapshot| snapshot.grad_ema),
                cache_hit_rate: stats(|snapshot| snapshot.cache_hit_rate),
                sparse_ratio: stats(|snapshot| snapshot.sparse_ratio),
                ema_cached_ns: stats(|snapshot| snapshot.ema_cached_ns),
                ema_sparse_ns: stats(|snapshot| snapshot.ema_sparse_ns),
                adaptive_bias: stats(|snapshot| snapshot.adaptive_bias),
                sinkhorn_iterations: stats(|snapshot| snapshot.sinkhorn_iterations as f64),
                sinkhorn_row_max_dev: stats(|snapshot| snapshot.sinkhorn_row_max_dev),
                sinkhorn_col_max_dev: stats(|snapshot| snapshot.sinkhorn_col_max_dev),
                sinkhorn_min_value: stats(|snapshot| snapshot.sinkhorn_min_value),
                sinkhorn_negative_ratio: stats(|snapshot| snapshot.sinkhorn_negative_ratio),
                sinkhorn_warm_started_rate: snapshots
                    .iter()
                    .filter(|snapshot| snapshot.sinkhorn_warm_started)
                    .count() as f64
                    / snapshots.len() as f64,
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
    cfg
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
    let total = (warmup + params.sims) as u64;
    let pb = crate::utils::create_bar(total, "Measuring throughput");
    for i in 0..warmup {
        let mut rng = Rng::from_seed(derive_seed(seed, i as u64));
        let _ = simulate_fast(params.pulls, &mut rng, 0, &ctx);
        pb.inc(1);
        if i == 0 {
            pb.set_message(format!("warmup {}/{}", i + 1, warmup));
        }
    }
    pb.set_message("measuring".to_string());
    let start = Instant::now();
    for i in 0..params.sims {
        let mut rng = Rng::from_seed(derive_seed(seed, (warmup + i) as u64));
        let _ = simulate_fast(params.pulls, &mut rng, 0, &ctx);
        pb.inc(1);
        if i > 0 && i % 10 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            pb.set_message(format!("{:.0} sims/s", rate));
        }
    }
    let elapsed = start.elapsed();
    pb.finish_and_clear();
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

// ── Main entry point ────────────────────────────────────────────────────

pub fn run_achf_benchmarks(base_config: &Config, seed: u64, bench_cfg: &BenchConfig) {
    validate_bench_config(bench_cfg).unwrap_or_else(|err| panic!("{err}"));

    let dir = &bench_cfg.output_dir;
    let nt = bench_cfg.num_trials;
    fs::create_dir_all(dir).expect("Failed to create output directory");

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
        print_agg_summary("Ablation (ACHF on/off)", &agg);
        let e = ext(&bench_cfg.format);
        chart_ablation(&agg, dir, e);
        all_agg.push(("ablation", agg));
    }

    if should_run(bench_cfg, "mode") {
        let agg = run_mode_comparison(base_config, seed, nt);
        print_agg_summary("Mode Comparison (lite vs full)", &agg);
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
        aggregated[condition_index].paired = Some(PairedComparison {
            baseline: aggregated[baseline_index].label.clone(),
            throughput_delta: paired(|run| run.throughput_sims_per_sec),
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

fn run_ablation(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Ablation Experiment (ACHF on/off)...");
    let conditions: Vec<(String, Config)> = [("ACHF Disabled", false), ("ACHF Enabled", true)]
        .into_iter()
        .map(|(label, enabled)| {
            let mut cfg = bench_sized_config(base_config);
            cfg.achf.enabled = enabled;
            cap_ppo_training(&mut cfg, 2000);
            (label.to_string(), cfg)
        })
        .collect();
    let runs = run_interleaved_conditions(&conditions, seed, nt);
    aggregate_conditions(&runs, &[0, 0])
}

fn run_mode_comparison(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Mode Comparison (lite vs full)...");
    let conditions: Vec<(String, Config)> = [("Lite", "lite"), ("Full", "full")]
        .into_iter()
        .map(|(label, mode)| {
            let mut cfg = bench_sized_config(base_config);
            cfg.achf.enabled = true;
            cfg.achf.mode = mode.to_string();
            cfg.achf.adaptive_inference = false;
            cap_ppo_training(&mut cfg, 2000);
            (label.to_string(), cfg)
        })
        .collect();
    let runs = run_interleaved_conditions(&conditions, seed, nt);
    aggregate_conditions(&runs, &[0, 0])
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
        cfg.achf.rank = PATH_RANK;
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
            sparsity.nonzero_weights > 0 && sparsity.nonzero_weights < sparsity.total_weights,
            "path comparison requires a non-degenerate frozen operator, got nnz={}/{}",
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
            let winner = if sparse_ns.mean <= cached_ns.mean && sparse_ns.mean <= dense_ns.mean {
                "Sparse"
            } else if cached_ns.mean <= dense_ns.mean {
                "Cached"
            } else {
                "Dense"
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
                        REGIME_DIM,
                        REGIME_SMALL_BATCH,
                        trial + sparsity_index,
                    ),
                    measure_regime(
                        &layer,
                        REGIME_DIM,
                        REGIME_LARGE_BATCH,
                        trial + sparsity_index + 1,
                    ),
                )
            } else {
                let large = measure_regime(
                    &layer,
                    REGIME_DIM,
                    REGIME_LARGE_BATCH,
                    trial + sparsity_index,
                );
                let small = measure_regime(
                    &layer,
                    REGIME_DIM,
                    REGIME_SMALL_BATCH,
                    trial + sparsity_index + 1,
                );
                (small, large)
            };
            println!(
                "  trial={:<2} requested={requested_sparsity:<5.2} actual={:.4} \
                 b{:<3}: gap={:.2}x oracle={} | b{:<3}: gap={:.2}x oracle={}",
                trial + 1,
                sparsity.sparsity,
                REGIME_SMALL_BATCH,
                small.adaptive_ns / small.oracle_ns.max(1.0),
                small.oracle_path,
                REGIME_LARGE_BATCH,
                large.adaptive_ns / large.oracle_ns.max(1.0),
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
    dim: usize,
    batch: usize,
    rotation: usize,
) -> RegimeTrialLatency {
    let x: Vec<f32> = (0..dim * batch)
        .map(|i| ((i % 7) as f32) * 0.1 + 0.05)
        .collect();
    // Warm the adaptive selector's bucket for this batch, then time it live.
    for _ in 0..REGIME_WARMUP_CALLS {
        let _ = std::hint::black_box(layer.forward_inference_residual(std::hint::black_box(&x)));
    }
    let before = layer.cache_stats();
    let start = Instant::now();
    for _ in 0..REGIME_MEASURE_CALLS {
        let _ = std::hint::black_box(layer.forward_inference_residual(std::hint::black_box(&x)));
    }
    let adaptive_ns = start.elapsed().as_nanos() as f64 / REGIME_MEASURE_CALLS as f64;
    let after = layer.cache_stats();
    let sparse = (after.sparse_paths - before.sparse_paths) as f64;
    let total = ((after.cache_hits - before.cache_hits)
        + (after.sparse_paths - before.sparse_paths)
        + (after.dense_paths - before.dense_paths)) as f64;
    let sparse_frac = sparse / total.max(1.0);

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
        cached_ns,
        sparse_ns,
        dense_ns,
        oracle_ns,
        oracle_path: oracle_path.to_string(),
        sparse_frac,
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
        cached_ns: values(|trial| trial.cached_ns),
        sparse_ns: values(|trial| trial.sparse_ns),
        dense_ns: values(|trial| trial.dense_ns),
        oracle_ns: values(|trial| trial.oracle_ns),
        oracle_gap: values(|trial| trial.adaptive_ns / trial.oracle_ns.max(1.0)),
        oracle_path,
        oracle_path_counts,
        oracle_paths: trials
            .iter()
            .map(|trial| trial.oracle_path.clone())
            .collect(),
        sparse_frac: values(|trial| trial.sparse_frac),
    }
}

fn run_gate_curve(base_config: &Config, seed: u64, num_trials: usize) -> AggregatedResult {
    println!("[Bench] Running Gate Curve Experiment...");
    let mut cfg = bench_sized_config(base_config);
    cfg.achf.enabled = true;
    cfg.achf.mode = "full".to_string();
    cfg.achf.adaptive_inference = false;
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
    // The ACHF FFN in the bench-sized model is hidden_dim*2 -> hidden_dim, and
    // bench_sized_config clamps hidden_dim to 64, so the layer's smaller
    // dimension is 64. `effective_rank` only truncates when rank < 64, and the
    // prune step only introduces row sparsity when rank*1.5 < 64 (rank < ~43).
    // The previous sweep [16,32,64,128,256] therefore had THREE degenerate
    // entries: rank 64/128/256 all resolve to "no truncation" and are identical
    // to the dense baseline, so any reward/throughput difference among them was
    // pure trial noise. This sweep spans the meaningful regime instead:
    //   8, 16, 32  -> truncate AND sparsify
    //   48         -> truncate only (rank < 64 but rank*1.5 > 64)
    //   64         -> no-op boundary (kept intentionally to show the ceiling)
    // The effective applied rank is reported per config (see print_agg_summary),
    // so the no-op at 64 is visible rather than masquerading as a real setting.
    for rank in [8, 16, 32, 48, 64] {
        let label = format!("rank={}", rank);
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = true;
        cfg.achf.rank = rank;
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

fn train_and_measure(label: &str, config: &Config, seed: u64) -> BenchRunResult {
    let cfg = config.clone();
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
            log_applied_rank(achf);
            BenchRunResult {
                label: label.to_string(),
                policy: "DQN".to_string(),
                train_time_ms: train_elapsed.as_secs_f64() * 1000.0,
                throughput_sims_per_sec: throughput,
                eval_reward: eval.avg_reward,
                train_loss,
                param_count: dqn.param_count(),
                applied_rank: achf.map(|snapshot| snapshot.low_rank_applied_rank),
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
            log_applied_rank(achf);
            BenchRunResult {
                label: label.to_string(),
                policy: "PPO".to_string(),
                train_time_ms: train_elapsed.as_secs_f64() * 1000.0,
                throughput_sims_per_sec: throughput,
                eval_reward: eval.avg_reward,
                train_loss,
                param_count: ppo.param_count(),
                applied_rank: achf.map(|snapshot| snapshot.low_rank_applied_rank),
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
    if let Some(snapshot) = snapshot.filter(|snapshot| snapshot.low_rank_applied_rank > 0) {
        println!(
            "    [ACHF] low-rank: rank={} rel_err={:.4}",
            snapshot.low_rank_applied_rank, snapshot.low_rank_rel_err
        );
    }
}

// ── Chart wrappers (aggregated) ──────────────────────────────────────────

fn agg_bars_with_error(agg: &[AggregatedResult]) -> Vec<(&str, f64, f64)> {
    agg.iter()
        .map(|a| (a.label.as_str(), a.throughput.mean, a.throughput.std_dev))
        .collect()
}

fn chart_ablation(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_error(agg);
    let path = format!("{}/ablation_throughput.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_error(
            &path,
            "Ablation: Throughput (mean +/- std)",
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
    let bars = agg_bars_with_error(agg);
    let path = format!("{}/mode_comparison.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_error(
            &path,
            "Mode: Throughput (lite vs full, mean +/- std)",
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
            let q1 = sorted[n / 4];
            let median = sorted[n / 2];
            let q3 = sorted[3 * n / 4];
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
    let gate: Vec<(f64, f64)> = result
        .curve
        .iter()
        .map(|point| (point.step as f64, point.gate_value.mean))
        .collect();
    let gmin: Vec<(f64, f64)> = result
        .curve
        .iter()
        .map(|point| (point.step as f64, point.g_min.mean))
        .collect();
    let grad: Vec<(f64, f64)> = result
        .curve
        .iter()
        .map(|point| (point.step as f64, point.grad_ema.mean))
        .collect();
    let hit: Vec<(f64, f64)> = result
        .curve
        .iter()
        .map(|point| (point.step as f64, point.cache_hit_rate.mean))
        .collect();
    let lr_ratio: Vec<(f64, f64)> = result
        .curve
        .iter()
        .map(|point| (point.step as f64, point.sparse_ratio.mean))
        .collect();
    let abias: Vec<(f64, f64)> = result
        .curve
        .iter()
        .map(|point| (point.step as f64, point.adaptive_bias.mean))
        .collect();

    let series: Vec<(&str, &[(f64, f64)])> = vec![
        ("Gate Value", &gate),
        ("g_min", &gmin),
        ("Grad EMA", &grad),
        ("Cache Hit Rate", &hit),
        ("Sparse Ratio", &lr_ratio),
        ("Adaptive Bias", &abias),
    ];
    let path = format!("{}/gate_curve.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_line_chart(
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
        let cached: Vec<(f64, f64)> = rows
            .iter()
            .map(|cell| (cell.actual_sparsity, cell.cached_ns.mean))
            .collect();
        let sparse: Vec<(f64, f64)> = rows
            .iter()
            .map(|cell| (cell.actual_sparsity, cell.sparse_ns.mean))
            .collect();
        let dense: Vec<(f64, f64)> = rows
            .iter()
            .map(|cell| (cell.actual_sparsity, cell.dense_ns.mean))
            .collect();
        let series: Vec<(&str, &[(f64, f64)])> =
            vec![("Cached", &cached), ("Sparse", &sparse), ("Dense", &dense)];
        let path = format!("{}/path_crossover_dim{}.{}", dir, dim, ext);
        write_chart(
            &path,
            chart::draw_line_chart(
                &path,
                &format!("Path Latency vs Weight Sparsity (dim={dim})"),
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
                    "wsp{:.2} b{}={}",
                    r.actual_sparsity, r.small.batch, r.small.oracle_path
                ),
                format!(
                    "wsp{:.2} b{}={}",
                    r.actual_sparsity, r.large.batch, r.large.oracle_path
                ),
            ]
        })
        .collect();
    let mut bars: Vec<(&str, f64)> = Vec::with_capacity(labels.len());
    for (i, r) in rows.iter().enumerate() {
        bars.push((labels[2 * i].as_str(), r.small.oracle_gap.mean));
        bars.push((labels[2 * i + 1].as_str(), r.large.oracle_gap.mean));
    }
    let path = format!("{}/regime_adaptation.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart(
            &path,
            "Adaptive Selector Oracle-Gap by Batch Regime (1.0 = matched best fixed path)",
            "Weight Sparsity x Batch (=oracle path)",
            "Adaptive / Oracle Latency",
            &bars,
            1100,
            600,
        ),
    );
}

fn chart_scale(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_error(agg);
    let path = format!("{}/scale_test.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_error(
            &path,
            "Scalability: Throughput by Rank (mean +/- std)",
            "Configuration",
            "Sims/sec",
            &bars,
            900,
            500,
        ),
    );
}

fn chart_apply(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_error(agg);
    let path = format!("{}/apply_combination.{}", dir, ext);
    write_chart(
        &path,
        chart::draw_bar_chart_with_error(
            &path,
            "Apply Combination: Throughput (mean +/- std)",
            "Configuration",
            "Sims/sec",
            &bars,
            1000,
            500,
        ),
    );
}

fn chart_convergence(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let loss_series: Vec<(String, Vec<(f64, f64)>)> = agg
        .iter()
        .filter(|result| !result.curve.is_empty())
        .map(|a| {
            let pts: Vec<(f64, f64)> = a
                .curve
                .iter()
                .map(|point| (point.step as f64, point.train_loss.mean))
                .collect();
            (a.label.clone(), pts)
        })
        .collect();
    let loss_ref: Vec<(&str, &[(f64, f64)])> = loss_series
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
        chart::draw_line_chart(
            &path,
            "Convergence: Loss Curve",
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
    let series: Vec<(String, Vec<(f64, f64)>)> = agg
        .iter()
        .filter(|result| !result.curve.is_empty())
        .map(|a| {
            let pts: Vec<(f64, f64)> = a
                .curve
                .iter()
                .map(|point| (point.step as f64, point.train_reward.mean))
                .collect();
            (a.label.clone(), pts)
        })
        .collect();
    let series_ref: Vec<(&str, &[(f64, f64)])> = series
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
        chart::draw_line_chart(
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
                "    paired vs {}: delta_tput={:.1} ({}) | delta_eval_reward={:.4} ({})",
                paired.baseline,
                paired.throughput_delta.mean,
                format_ci(&paired.throughput_delta),
                paired.eval_reward_delta.mean,
                format_ci(&paired.eval_reward_delta),
            );
        }
    }
}

fn write_summary_txt(
    all: &[(&str, Vec<AggregatedResult>)],
    path_latencies: Option<&[PathLatencyResult]>,
    gate_curve: Option<&AggregatedResult>,
    crossover: Option<&[CrossoverCell]>,
    regime: Option<&[RegimeRow]>,
    dir: &str,
) {
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
                    "    paired vs {}: delta_tput={:.2} ({}) | delta_eval_reward={:.4} ({}) | delta_train_loss={:.4} ({}) | delta_train_ms={:.1} ({})",
                    paired.baseline,
                    paired.throughput_delta.mean,
                    format_ci(&paired.throughput_delta),
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
                    "    frozen ACHF ({} trials): calls={} hit={:.1}% sparse={} dense={} latency_samples={} bias={:.3}",
                    a.cache_trial_count,
                    stats.calls,
                    hit_pct,
                    stats.sparse_paths,
                    stats.dense_paths,
                    stats.latency_samples,
                    stats.adaptive_bias
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
                "    final aggregated training point: step={} n={} gate={:.4} g_min={:.4} hit={:.1}% sparse={:.1}% bias={:.3}",
                last.step,
                last.samples,
                last.gate_value.mean,
                last.g_min.mean,
                last.cache_hit_rate.mean * 100.0,
                last.sparse_ratio.mean * 100.0,
                last.adaptive_bias.mean
            ));
        }
        if let Some(stats) = result.cache_stats {
            let calls = stats.calls as f64;
            // Share the denominator `calls` across hit/sparse/dense so the three
            // percentages are comparable and sum to ~100%. Dividing the path
            // rates by only (sparse+dense) previously excluded cache hits,
            // making dense read 100% even when 45% of calls were cache hits.
            let pct = |n: u64| {
                if calls > 0.0 {
                    n as f64 / calls * 100.0
                } else {
                    0.0
                }
            };
            let hit_pct = pct(stats.cache_hits);
            let sparse_pct = pct(stats.sparse_paths);
            let dense_pct = pct(stats.dense_paths);
            lines.push(format!(
                "    ACHF inference: calls={} hit={:.1}% sparse={:.1}% dense={:.1}% latency_samples={} decision_ns={:.1}",
                stats.calls,
                hit_pct,
                sparse_pct,
                dense_pct,
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
                "  dim={} requested={:.4} actual={:.4} nnz={}/{} | cached={:.1}ns ({}) sparse={:.1}ns ({}) dense={:.1}ns ({}) | winner={}",
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
            ));
        }
        lines.push(String::new());
    }
    if let Some(rows) = regime {
        lines.push("=== regime ===".to_string());
        for row in rows {
            for latency in [&row.small, &row.large] {
                lines.push(format!(
                    "  requested={:.4} actual={:.4} nnz={}/{} batch={} | adaptive={:.1}ns oracle={:.1}ns path={} counts={:?} gap={:.3} ({}) sparse_frac={:.3}",
                    row.requested_sparsity,
                    row.actual_sparsity,
                    row.nonzero_weights,
                    row.total_weights,
                    latency.batch,
                    latency.adaptive_ns.mean,
                    latency.oracle_ns.mean,
                    latency.oracle_path,
                    latency.oracle_path_counts,
                    latency.oracle_gap.mean,
                    format_ci(&latency.oracle_gap),
                    latency.sparse_frac.mean,
                ));
            }
        }
        lines.push(String::new());
    }
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
            "schema_version": 2,
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
                    "rank": PATH_RANK,
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
                "lite": "deterministic frozen cached/sparse/dense selection",
                "full": "frozen weights with online latency-adaptive AMA selection",
            },
            "metric_semantics": {
                "eval_reward": "held-out reward from the active frozen policy using that policy's native reward; compare only within the same active_policy",
                "train_loss": "last training snapshot from the active policy using that policy's native loss; compare only within the same active_policy",
                "policy_train_time_ms": "active policy training only; base-model preparation excluded",
                "param_count": "active policy trainable parameters; derived sparse inference copies excluded",
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
        "eval_reward_delta": trial_stats_json(&paired.eval_reward_delta),
        "train_loss_delta": trial_stats_json(&paired.train_loss_delta),
        "policy_train_time_ms_delta": trial_stats_json(&paired.train_time_ms_delta),
    })
}

fn aggregated_result_json(result: &AggregatedResult) -> serde_json::Value {
    serde_json::json!({
        "label": result.label,
        "active_policy": result.policy,
        "throughput_sims_per_sec": trial_stats_json(&result.throughput),
        "eval_reward": trial_stats_json(&result.eval_reward),
        "train_loss": trial_stats_json(&result.train_loss),
        "policy_train_time_ms": trial_stats_json(&result.train_time_ms),
        "param_count": result.param_count,
        "applied_rank": result.applied_rank,
        "paired_vs_baseline": result.paired.as_ref().map(paired_comparison_json),
        "curve": result.curve.iter().map(curve_point_json).collect::<Vec<_>>(),
        "cache_trial_count": result.cache_trial_count,
        "frozen_cache_stats": result.cache_stats.map(cache_stats_json),
    })
}

fn write_csvs(all: &[(&str, Vec<AggregatedResult>)], dir: &str) {
    for (name, agg) in all {
        let mut csv = String::from(
            "label,active_policy,trial,throughput_sims_per_sec,eval_reward,train_loss,policy_train_time_ms,param_count,applied_rank\n",
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
                    "{},{},{},{:.2},{:.6},{:.6},{:.3},{},{}\n",
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
    })
}

fn write_regime_outputs(rows: &[RegimeRow], dir: &str) {
    let mut csv = String::from(
        "requested_sparsity,actual_sparsity,total_weights,nonzero_weights,batch,trial,adaptive_ns,cached_ns,sparse_ns,dense_ns,oracle_ns,oracle_path,oracle_gap,sparse_frac\n",
    );
    for row in rows {
        for latency in [&row.small, &row.large] {
            for trial in 0..latency.adaptive_ns.values.len() {
                csv.push_str(&format!(
                    "{:.6},{:.6},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.6},{:.6}\n",
                    row.requested_sparsity,
                    row.actual_sparsity,
                    row.total_weights,
                    row.nonzero_weights,
                    latency.batch,
                    trial + 1,
                    latency.adaptive_ns.values[trial],
                    latency.cached_ns.values[trial],
                    latency.sparse_ns.values[trial],
                    latency.dense_ns.values[trial],
                    latency.oracle_ns.values[trial],
                    latency.oracle_paths[trial],
                    latency.oracle_gap.values[trial],
                    latency.sparse_frac.values[trial],
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
                    "cached_ns": trial_stats_json(&latency.cached_ns),
                    "sparse_ns": trial_stats_json(&latency.sparse_ns),
                    "dense_ns": trial_stats_json(&latency.dense_ns),
                    "oracle_ns": trial_stats_json(&latency.oracle_ns),
                    "oracle_gap": trial_stats_json(&latency.oracle_gap),
                    "oracle_path_by_majority": latency.oracle_path,
                    "oracle_path_counts": latency.oracle_path_counts,
                    "oracle_paths": latency.oracle_paths,
                    "sparse_fraction": trial_stats_json(&latency.sparse_frac),
                })
            })
        })
        .collect()
}

fn write_gate_curve_outputs(result: &AggregatedResult, dir: &str) {
    let mut csv = String::from("step,samples,metric,mean,std_dev,ci_95_low,ci_95_high\n");
    for point in &result.curve {
        let warm_started_count =
            (point.sinkhorn_warm_started_rate * point.samples as f64).round() as usize;
        let warm_started_values: Vec<f64> = (0..point.samples)
            .map(|index| if index < warm_started_count { 1.0 } else { 0.0 })
            .collect();
        let warm_started_stats = TrialStats::from_values(&warm_started_values);
        let metrics: [(&str, &TrialStats); 17] = [
            ("train_loss", &point.train_loss),
            ("train_reward", &point.train_reward),
            ("gate_value", &point.gate_value),
            ("g_min", &point.g_min),
            ("grad_ema", &point.grad_ema),
            ("cache_hit_rate", &point.cache_hit_rate),
            ("sparse_ratio", &point.sparse_ratio),
            ("ema_cached_ns", &point.ema_cached_ns),
            ("ema_sparse_ns", &point.ema_sparse_ns),
            ("adaptive_bias", &point.adaptive_bias),
            ("sinkhorn_iterations", &point.sinkhorn_iterations),
            ("sinkhorn_row_max_dev", &point.sinkhorn_row_max_dev),
            ("sinkhorn_col_max_dev", &point.sinkhorn_col_max_dev),
            ("sinkhorn_min_value", &point.sinkhorn_min_value),
            ("sinkhorn_negative_ratio", &point.sinkhorn_negative_ratio),
            ("sinkhorn_warm_started_rate", &warm_started_stats),
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
        "gate_value": trial_stats_json(&point.gate_value),
        "g_min": trial_stats_json(&point.g_min),
        "grad_ema": trial_stats_json(&point.grad_ema),
        "cache_hit_rate": trial_stats_json(&point.cache_hit_rate),
        "sparse_ratio": trial_stats_json(&point.sparse_ratio),
        "ema_cached_ns": trial_stats_json(&point.ema_cached_ns),
        "ema_sparse_ns": trial_stats_json(&point.ema_sparse_ns),
        "adaptive_bias": trial_stats_json(&point.adaptive_bias),
        "sinkhorn_iterations": trial_stats_json(&point.sinkhorn_iterations),
        "sinkhorn_row_max_dev": trial_stats_json(&point.sinkhorn_row_max_dev),
        "sinkhorn_col_max_dev": trial_stats_json(&point.sinkhorn_col_max_dev),
        "sinkhorn_min_value": trial_stats_json(&point.sinkhorn_min_value),
        "sinkhorn_negative_ratio": trial_stats_json(&point.sinkhorn_negative_ratio),
        "sinkhorn_warm_started_rate": point.sinkhorn_warm_started_rate,
        "low_rank_applied_rank": trial_stats_json(&point.low_rank_applied_rank),
    })
}

fn cache_stats_json(stats: AchfCacheStats) -> serde_json::Value {
    let calls = stats.calls as f64;
    // Every call resolves to exactly one path: Cached (cache_hits), Sparse, or
    // Dense. All three rates therefore share the SAME denominator (total calls)
    // so they are directly comparable and sum to ~1.0. Previously hit_rate was
    // divided by `calls` while the path rates were divided by only
    // (sparse+dense), which excluded cache hits from the denominator — that is
    // why a run could report hit_rate=0.45 alongside dense_path_rate=1.0 for the
    // same data, an apparent contradiction that was purely a denominator
    // mismatch. `cached_path_rate` is added so the three paths are reported on
    // equal footing.
    let rate = |n: u64| if calls > 0.0 { n as f64 / calls } else { 0.0 };
    serde_json::json!({
        "calls": stats.calls,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "cache_skips": stats.cache_skips,
        "sparse_paths": stats.sparse_paths,
        "dense_paths": stats.dense_paths,
        "hit_rate": rate(stats.cache_hits),
        "cached_path_rate": rate(stats.cache_hits),
        "sparse_path_rate": rate(stats.sparse_paths),
        "dense_path_rate": rate(stats.dense_paths),
        "ema_cached_ns": stats.ema_cached_ns,
        "ema_cached_long_ns": stats.ema_cached_long_ns,
        "ema_sparse_ns": stats.ema_sparse_ns,
        "ema_sparse_long_ns": stats.ema_sparse_long_ns,
        "ema_dense_ns": stats.ema_dense_ns,
        "ema_dense_long_ns": stats.ema_dense_long_ns,
        "decision_ema_ns": stats.decision_ema_ns,
        "decision_ema_long_ns": stats.decision_ema_long_ns,
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
    let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
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
            g_min: 0.2,
            grad_ema: 0.3,
            loss,
            reward,
            cache_hit_rate: 0.5,
            sparse_ratio: 0.75,
            ema_cached_ns: 11.0,
            ema_sparse_ns: 22.0,
            adaptive_bias: 1.1,
            sinkhorn_iterations: 20,
            sinkhorn_row_max_dev: 0.0001,
            sinkhorn_col_max_dev: 0.0002,
            sinkhorn_min_value: 0.01,
            sinkhorn_negative_ratio: 0.0,
            sinkhorn_warm_started: true,
            low_rank_applied_rank: 0,
        }
    }

    fn test_cache_stats(calls: u64, hits: u64) -> AchfCacheStats {
        AchfCacheStats {
            calls,
            cache_hits: hits,
            cache_misses: 1,
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
            train_time_ms,
            throughput_sims_per_sec: throughput,
            eval_reward,
            train_loss,
            param_count: 42,
            applied_rank: Some(0),
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
        assert_eq!(parsed["metadata"]["schema_version"], 2);
        assert_eq!(parsed["exp"][0]["active_policy"], "PPO");
        assert_eq!(parsed["exp"][0]["eval_reward"]["mean"], 0.5);
        assert!(parsed["exp"][0]["eval_reward"]["ci_95"][0].is_null());
        assert_eq!(parsed["exp"][0]["frozen_cache_stats"]["calls"], 8);
        assert_eq!(parsed["exp"][0]["frozen_cache_stats"]["hit_rate"], 0.25);

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
        assert_eq!(stats[0].p50_ns, 30.0);
        assert_eq!(stats[0].p95_ns, 40.0);

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
        assert_eq!(json["path_latency"][0]["p99_ns"], 40.0);
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
                cached_ns: 120.0,
                sparse_ns: 70.0,
                dense_ns: 140.0,
                oracle_ns: 70.0,
                oracle_path: "Sparse".to_string(),
                sparse_frac: 0.6,
            }]),
            large: aggregate_regime_trials(&[RegimeTrialLatency {
                batch: 128,
                adaptive_ns: 210.0,
                cached_ns: 200.0,
                sparse_ns: 280.0,
                dense_ns: 250.0,
                oracle_ns: 200.0,
                oracle_path: "Cached".to_string(),
                sparse_frac: 0.1,
            }]),
        }];
        write_regime_outputs(&rows, &output_dir);
        write_summary_json(&[], None, None, None, Some(&rows), (1, 1), &output_dir);
        let csv = std::fs::read_to_string(dir.join("regime_adaptation.csv")).unwrap();
        // Small-batch oracle is Sparse, large-batch oracle is Cached: the two
        // regimes have different best fixed paths — the core adaptation result.
        assert!(csv.contains(
            "0.900000,0.899414,1048576,105472,1,1,90.000,120.000,70.000,140.000,70.000,Sparse,1.285714,0.600000"
        ));
        assert!(csv.contains(
            "0.900000,0.899414,1048576,105472,128,1,210.000,200.000,280.000,250.000,200.000,Cached,1.050000,0.100000"
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
        assert!(csv.contains("10,2,gate_value,0.9000000000"));
        assert!(csv.contains("10,2,train_loss,0.2500000000"));
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(json["gate_curve"]["curve_point_count"], 1);
        assert_eq!(json["gate_curve"]["final_curve_point"]["step"], 10);
        assert_eq!(
            json["gate_curve"]["final_curve_point"]["gate_value"]["mean"],
            0.9
        );
        assert_eq!(json["gate_curve"]["frozen_cache_stats"]["hit_rate"], 0.25);
        // All path rates share the denominator `calls` (=8) and are mutually
        // consistent: cached 2/8, sparse 3/8, dense 1/8. This guards against the
        // former denominator mismatch where path rates excluded cache hits.
        assert_eq!(
            json["gate_curve"]["frozen_cache_stats"]["cached_path_rate"],
            0.25
        );
        assert_eq!(
            json["gate_curve"]["frozen_cache_stats"]["sparse_path_rate"],
            0.625
        );
        assert_eq!(
            json["gate_curve"]["frozen_cache_stats"]["dense_path_rate"],
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
