use crate::achf::{aggregate_cache_stats_iter, AchfCacheStats, AchfLayer};
use crate::chart::{self, ChartFormat};
use crate::config::{AchfConfig, Config, LuckMode};
use crate::dqn::{train_dqn_with_metrics, DuelingQNetwork};
use crate::env_net::EnvNet;
use crate::model_io::{
    env_net_cache_manifest, load_env_net_cache_with_manifest, save_env_net_cache_with_manifest,
    CacheQualitySummary,
};
use crate::neural::NeuralLuckOptimizer;
use crate::ppo::{train_ppo_with_metrics, ActorCritic};
use crate::rng::Rng;
use crate::sim::{simulate_fast, SimModelContext};
use crate::trainer::{train_linear_regression, train_manifold_rl, train_neural_optimizer};
use crate::training_metrics::StepSnapshot;
use crate::worker::GoodJobWorker;
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
const THROUGHPUT_SIMS: usize = 200;
const THROUGHPUT_PULLS: usize = 100;
const CURVE_THROUGHPUT_SIMS: usize = 100;

// ── Data structures ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BenchRunResult {
    pub label: String,
    pub total_time_ms: f64,
    pub throughput_sims_per_sec: f64,
    pub final_avg_reward: f64,
    pub final_loss: f64,
    pub param_count: usize,
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
    pub ci_low: f64,
    pub ci_high: f64,
    pub values: Vec<f64>,
}

impl TrialStats {
    fn from_values(vals: &[f64]) -> Self {
        if vals.is_empty() {
            return TrialStats {
                mean: 0.0,
                std_dev: 0.0,
                ci_low: 0.0,
                ci_high: 0.0,
                values: Vec::new(),
            };
        }
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std_dev = variance.sqrt();
        let t_val = 2.776;
        let se = std_dev / n.sqrt();
        TrialStats {
            mean,
            std_dev,
            ci_low: mean - t_val * se,
            ci_high: mean + t_val * se,
            values: vals.to_vec(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AggregatedResult {
    pub label: String,
    pub throughput: TrialStats,
    pub reward: TrialStats,
    pub loss: TrialStats,
    pub time_ms: TrialStats,
    pub param_count: usize,
    pub best_snapshots: Vec<StepSnapshot>,
    pub cache_stats: Option<AchfCacheStats>,
}

/// One cell of the path-crossover grid: measured per-path latency at a given
/// (dim, weight_sparsity) operating point, plus which path was fastest.
#[derive(Clone, Debug)]
struct CrossoverCell {
    dim: usize,
    weight_sparsity: f32,
    cached_ns: f64,
    sparse_ns: f64,
    dense_ns: f64,
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
    adaptive_ns: f64,
    cached_ns: f64,
    sparse_ns: f64,
    oracle_ns: f64,
    oracle_path: String,
    sparse_frac: f64,
}

/// One row of the regime-adaptation experiment: the small-batch (decode-like)
/// and large-batch (prefill-like) measurements for one fixed weight sparsity.
/// When the two regimes have DIFFERENT oracle paths yet the adaptive selector
/// stays near-oracle in both, that is the "true adaptive" result: a single
/// fixed path cannot win both regimes, but the batch-aware selector does.
#[derive(Clone, Debug)]
struct RegimeRow {
    weight_sparsity: f32,
    small: RegimeLatency,
    large: RegimeLatency,
}

#[derive(Clone, Debug)]
struct PathLatencyStats {
    label: String,
    samples: usize,
    mean_ns: f64,
    min_ns: f64,
    p50_ns: f64,
    p90_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
    max_ns: f64,
}

fn aggregate_trials(runs: &[BenchRunResult]) -> AggregatedResult {
    let label = runs[0].label.clone();
    let tputs: Vec<f64> = runs.iter().map(|r| r.throughput_sims_per_sec).collect();
    let rewards: Vec<f64> = runs.iter().map(|r| r.final_avg_reward).collect();
    let losses: Vec<f64> = runs.iter().map(|r| r.final_loss).collect();
    let times: Vec<f64> = runs.iter().map(|r| r.total_time_ms).collect();
    let best = runs
        .iter()
        .max_by(|a, b| a.snapshots.len().cmp(&b.snapshots.len()))
        .unwrap();
    AggregatedResult {
        label,
        throughput: TrialStats::from_values(&tputs),
        reward: TrialStats::from_values(&rewards),
        loss: TrialStats::from_values(&losses),
        time_ms: TrialStats::from_values(&times),
        param_count: runs[0].param_count,
        best_snapshots: best.snapshots.clone(),
        cache_stats: best.cache_stats,
    }
}

// ── Helper: build neural + worker from config ───────────────────────────

fn build_base_models_with_worker(
    config: &Config,
    rng: &mut Rng,
    worker: &GoodJobWorker,
) -> (EnvNet, NeuralLuckOptimizer) {
    let manifest = env_net_cache_manifest(config);
    let mut env_net = if let Some(cached) =
        load_env_net_cache_with_manifest("env_net.cache", config, &manifest)
    {
        cached
    } else {
        let mut net = EnvNet::new(rng);
        let (count, epochs) = if config.fast_init {
            (256, 10)
        } else {
            (1024, 50)
        };
        net.pretrain(rng, config, count, epochs);
        let _ = save_env_net_cache_with_manifest(
            "env_net.cache",
            &net,
            manifest.with_quality(CacheQualitySummary::note(format!(
                "{count}x{epochs} pretrain"
            ))),
        );
        net
    };
    env_net.set_train(false);

    let mut neural_opt = train_neural_optimizer(rng.next_u64(), &env_net, config, worker);
    let (w, b) = train_linear_regression(&neural_opt, rng, &env_net, config);
    neural_opt.set_linear_params(w, b);
    neural_opt = train_manifold_rl(&neural_opt, rng, &env_net, config, worker);

    (env_net, neural_opt)
}

fn build_base_models(
    config: &Config,
    rng: &mut Rng,
) -> (EnvNet, NeuralLuckOptimizer, GoodJobWorker) {
    let worker =
        GoodJobWorker::new_with_config(config).expect("Failed to build benchmark worker pool");
    let (env_net, neural_opt) = build_base_models_with_worker(config, rng, &worker);
    (env_net, neural_opt, worker)
}

fn bench_sized_config(base_config: &Config) -> Config {
    let mut cfg = base_config.clone();
    cfg.fast_init = true;
    cfg.luck_mode = LuckMode::Ppo;
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

fn measure_inference_throughput(rng: &mut Rng, params: &ThroughputParams<'_>) -> f64 {
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
    let warmup = 20u64;
    let total = warmup + params.sims as u64;
    let pb = crate::utils::create_bar(total, "Measuring throughput");
    for i in 0..warmup {
        let _ = simulate_fast(params.pulls, rng, 0, &ctx);
        pb.inc(1);
        if i == 0 {
            pb.set_message(format!("warmup {}/{}", i + 1, warmup));
        }
    }
    pb.set_message("measuring".to_string());
    let start = Instant::now();
    for i in 0..params.sims {
        let _ = simulate_fast(params.pulls, rng, 0, &ctx);
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
    let mut path_latencies: Option<Vec<(String, Vec<f64>)>> = None;
    let mut gate_curve: Option<BenchRunResult> = None;

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
        let latencies = run_path_comparison(base_config, seed);
        println!("[Bench] Path Comparison complete.");
        for stats in path_latency_stats(&latencies) {
            println!(
                "  {}: avg {:.1} ns ({} samples)",
                stats.label, stats.mean_ns, stats.samples
            );
        }
        let e = ext(&bench_cfg.format);
        chart_path_latency(&latencies, dir, e);
        write_path_latency_outputs(&latencies, dir);
        path_latencies = Some(latencies);
    }

    if should_run(bench_cfg, "gate") {
        let result = run_gate_curve(base_config, seed);
        println!(
            "[Bench] Gate Curve: {} snapshots collected over {} steps",
            result.snapshots.len(),
            result.snapshots.last().map_or(0, |s| s.step)
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
        let cells = run_path_crossover();
        println!("[Bench] Path Crossover complete ({} cells).", cells.len());
        let e = ext(&bench_cfg.format);
        chart_crossover(&cells, dir, e);
        write_crossover_outputs(&cells, dir);
    }

    if should_run(bench_cfg, "regime") {
        let rows = run_regime_adaptation();
        println!("[Bench] Regime Adaptation complete ({} rows).", rows.len());
        let e = ext(&bench_cfg.format);
        chart_regime(&rows, dir, e);
        write_regime_outputs(&rows, dir);
    }

    write_summary_txt(
        &all_agg,
        path_latencies.as_deref(),
        gate_curve.as_ref(),
        dir,
    );
    write_summary_json(
        &all_agg,
        path_latencies.as_deref(),
        gate_curve.as_ref(),
        dir,
    );
    write_csvs(&all_agg, dir);

    println!("\n========================================");
    println!("  All benchmarks complete.");
    println!("  Output: {}/", dir);
    println!("========================================");
}

// ── Experiment implementations ──────────────────────────────────────────

fn run_multi_trial(
    label: &str,
    config: &Config,
    seed: u64,
    num_trials: usize,
) -> Vec<BenchRunResult> {
    (0..num_trials)
        .map(|t| {
            let trial_seed = seed.wrapping_add(t as u64 * 1337);
            let result = train_and_measure(label, config, trial_seed);
            println!(
                "    trial {}/{}: {:.1}s | {:.0} sims/sec | reward: {:.3} | loss: {:.4}",
                t + 1,
                num_trials,
                result.total_time_ms / 1000.0,
                result.throughput_sims_per_sec,
                result.final_avg_reward,
                result.final_loss
            );
            result
        })
        .collect()
}

fn run_ablation(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Ablation Experiment (ACHF on/off)...");
    let mut agg = Vec::new();
    for (label, enabled) in [("ACHF Enabled", true), ("ACHF Disabled", false)] {
        println!("  [{}]", label);
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = enabled;
        let runs = run_multi_trial(label, &cfg, seed, nt);
        agg.push(aggregate_trials(&runs));
    }
    agg
}

fn run_mode_comparison(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Mode Comparison (lite vs full)...");
    let mut agg = Vec::new();
    for (label, mode) in [("Lite", "lite"), ("Full", "full")] {
        println!("  [{}]", label);
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = true;
        cfg.achf.mode = mode.to_string();
        let runs = run_multi_trial(label, &cfg, seed, nt);
        agg.push(aggregate_trials(&runs));
    }
    agg
}

fn run_path_comparison(base_config: &Config, seed: u64) -> Vec<(String, Vec<f64>)> {
    println!("[Bench] Running Path Comparison (Cached/Sparse/Dense)...");
    let mut rng = Rng::from_seed(seed);
    let mut cfg = bench_sized_config(base_config);
    cfg.achf.enabled = true;
    cfg.ppo_total_steps = cfg.ppo_total_steps.min(2000);
    cfg.ppo_steps_per_update = cfg.ppo_steps_per_update.min(256);
    cfg.ppo_k_epochs = cfg.ppo_k_epochs.min(2);

    let (env_net, _neural_opt, _worker) = build_base_models(&cfg, &mut rng);
    let ppo = train_ppo_with_metrics(&mut rng, &env_net, &cfg, None);

    // Measure the ACHF operator in ISOLATION, not the whole transformer forward.
    // A full forward is dominated by embed/norm/attention/FFN work that is
    // identical across paths, which dilutes the Cached/Sparse/Dense difference
    // into single-digit percent. Timing the operator alone is what the path
    // comparison is supposed to show.
    let (achf, input_dim) = ppo
        .first_achf_layer()
        .expect("path comparison requires an ACHF layer; achf.enabled=true");
    let sample_input: Vec<f32> = (0..input_dim).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let warmup_iterations = 100;
    let iterations = 2000;
    // The per-call operator cost (~hundreds of ns) is below the platform clock
    // granularity (Instant resolves to ~100ns on Windows), so timing a single
    // call quantizes every sample to a multiple of 100ns. That collapses the
    // latency distribution into ~5 discrete buckets, which in turn makes the
    // box plot's IQR degenerate (q1==q2==q3, zero-height boxes) and hides the
    // real spread between paths. Time a BATCH of calls per sample and divide by
    // the batch size: the timed window becomes tens of microseconds (far above
    // clock granularity) and the reported per-call latency regains sub-ns
    // resolution. `black_box` prevents the optimizer from hoisting/eliding the
    // repeated identical calls.
    let batch = 64usize;

    let mut all_latencies: Vec<(String, Vec<f64>)> = Vec::new();

    // 0 = Cached, 1 = Sparse, 2 = Dense
    for (path_name, path_id) in [("Cached", 0u8), ("Sparse", 1), ("Dense", 2)] {
        for _ in 0..warmup_iterations {
            let _ = std::hint::black_box(
                achf.forward_inference_forced_path(std::hint::black_box(&sample_input), path_id),
            );
        }

        let mut latencies = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            for _ in 0..batch {
                let out = achf
                    .forward_inference_forced_path(std::hint::black_box(&sample_input), path_id);
                std::hint::black_box(out);
            }
            let ns = start.elapsed().as_nanos() as f64 / batch as f64;
            latencies.push(ns);
        }
        println!(
            "  [{}] avg={:.1}ns, p50={:.1}ns, p99={:.1}ns",
            path_name,
            latencies.iter().sum::<f64>() / latencies.len() as f64,
            percentile(&latencies, 50),
            percentile(&latencies, 99),
        );
        all_latencies.push((path_name.to_string(), latencies));
    }
    all_latencies
}

/// Build a frozen square ACHF layer with an EXACT target weight sparsity by
/// zeroing the first `sparsity` fraction of every row (deterministic). The CSR
/// sparse view keys on `w != 0.0`, so this controls the sparse path's FLOP
/// count precisely — unlike magnitude pruning, whose sparsity depends on the
/// random weight distribution. `adaptive` toggles the live AMA selector.
fn build_synthetic_achf_layer(dim: usize, weight_sparsity: f32, adaptive: bool) -> AchfLayer {
    let cfg = AchfConfig {
        enabled: true,
        adaptive_inference: adaptive,
        cache_latency_sample_every: 1,
        gate_warmup_steps: 0,
        gate_transition_steps: 0,
        g_min: 0.0,
        infer_gate: "one".to_string(),
        prune_threshold: 0.0,
        // Disable memoization: with a repeated benchmark input it would return a
        // memo hit and bypass the selector/path entirely, invalidating timing.
        cache_min_reuse: 0,
        ..Default::default()
    };
    let mut layer = AchfLayer::new(dim, dim, false, cfg, 0x5EED ^ dim as u64);
    {
        let mut w = layer.weight.weight.data_write_f32();
        let zero_per_row = (dim as f32 * weight_sparsity) as usize;
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

/// Time one forced path (mean ns per single forward) over `iters` batches of
/// `batch` rows, after `warmup` untimed calls. Batching keeps the timed window
/// well above clock granularity (see run_path_comparison for the rationale).
fn time_forced_path(layer: &AchfLayer, x: &[f32], path_id: u8, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        std::hint::black_box(layer.forward_inference_forced_path(std::hint::black_box(x), path_id));
    }
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(layer.forward_inference_forced_path(std::hint::black_box(x), path_id));
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

/// Path-crossover experiment: sweep (dim x weight_sparsity), force each path,
/// and record which is fastest. Demonstrates that no single fixed path wins
/// everywhere — the premise that makes adaptive path selection worthwhile.
fn run_path_crossover() -> Vec<CrossoverCell> {
    println!("[Bench] Running Path Crossover (dim x weight-sparsity)...");
    let dims = [256usize, 1024, 2048];
    let sparsities = [0.5f32, 0.8, 0.9, 0.95, 0.99];
    let batch = 32usize;
    let warmup = 20usize;
    let iters = 200usize;
    let mut cells = Vec::new();
    for &dim in &dims {
        for &weight_sparsity in &sparsities {
            let layer = build_synthetic_achf_layer(dim, weight_sparsity, false);
            let x: Vec<f32> = (0..dim * batch).map(|i| ((i % 7) as f32) * 0.1).collect();
            let cached_ns = time_forced_path(&layer, &x, 0, warmup, iters);
            let sparse_ns = time_forced_path(&layer, &x, 1, warmup, iters);
            let dense_ns = time_forced_path(&layer, &x, 2, warmup, iters);
            let winner = if sparse_ns <= cached_ns && sparse_ns <= dense_ns {
                "Sparse"
            } else if cached_ns <= dense_ns {
                "Cached"
            } else {
                "Dense"
            }
            .to_string();
            println!(
                "  dim={dim:<5} wsp={weight_sparsity:<5} cached={cached_ns:>9.0}ns \
                 sparse={sparse_ns:>9.0}ns dense={dense_ns:>9.0}ns -> {winner}"
            );
            cells.push(CrossoverCell {
                dim,
                weight_sparsity,
                cached_ns,
                sparse_ns,
                dense_ns,
                winner,
            });
        }
    }
    cells
}

/// Regime-adaptation experiment: on ONE fixed frozen layer, run the LIVE
/// adaptive selector at a small batch (decode-like) then a large batch
/// (prefill-like) and record how often it chose the sparse path in each. When
/// the small-batch sparse fraction exceeds the large-batch one, the selector is
/// adapting its path choice to the operating point — the core "true adaptive"
/// claim. Batch-bucketed latency EMAs are what make this possible.
fn run_regime_adaptation() -> Vec<RegimeRow> {
    println!("[Bench] Running Regime Adaptation (batch-driven path switching)...");
    let dim = 1024usize;
    let sparsities = [0.8f32, 0.9, 0.95, 0.98];
    let small_batch = 1usize;
    let large_batch = 128usize;
    let mut rows = Vec::new();

    for &weight_sparsity in &sparsities {
        let layer = build_synthetic_achf_layer(dim, weight_sparsity, true);
        let small = measure_regime(&layer, dim, small_batch);
        let large = measure_regime(&layer, dim, large_batch);
        println!(
            "  wsp={weight_sparsity:<5} b{small_batch:<3}: adaptive={:.0}ns oracle={:.0}ns \
             ({}) gap={:.2}x sparse_frac={:.2} | b{large_batch:<3}: adaptive={:.0}ns \
             oracle={:.0}ns ({}) gap={:.2}x sparse_frac={:.2}",
            small.adaptive_ns,
            small.oracle_ns,
            small.oracle_path,
            small.adaptive_ns / small.oracle_ns.max(1.0),
            small.sparse_frac,
            large.adaptive_ns,
            large.oracle_ns,
            large.oracle_path,
            large.adaptive_ns / large.oracle_ns.max(1.0),
            large.sparse_frac,
        );
        rows.push(RegimeRow {
            weight_sparsity,
            small,
            large,
        });
    }
    rows
}

/// Measure one (layer, batch) operating point: warm the batch bucket, time the
/// live adaptive selector, time each forced fixed path, and record the sparse
/// selection fraction over a fresh measurement window.
fn measure_regime(layer: &AchfLayer, dim: usize, batch: usize) -> RegimeLatency {
    let x: Vec<f32> = (0..dim * batch)
        .map(|i| ((i % 7) as f32) * 0.1 + 0.05)
        .collect();
    let warm = 300usize;
    let iters = 600usize;
    // Warm the adaptive selector's bucket for this batch, then time it live.
    for _ in 0..warm {
        let _ = std::hint::black_box(layer.forward_inference_residual(std::hint::black_box(&x)));
    }
    let before = layer.cache_stats();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(layer.forward_inference_residual(std::hint::black_box(&x)));
    }
    let adaptive_ns = start.elapsed().as_nanos() as f64 / iters as f64;
    let after = layer.cache_stats();
    let sparse = (after.sparse_paths - before.sparse_paths) as f64;
    let total = ((after.cache_hits - before.cache_hits)
        + (after.sparse_paths - before.sparse_paths)
        + (after.dense_paths - before.dense_paths)) as f64;
    let sparse_frac = sparse / total.max(1.0);

    // Forced fixed-path costs at the same operating point (reproducible).
    let cached_ns = time_forced_path(layer, &x, 0, 40, iters);
    let sparse_ns = time_forced_path(layer, &x, 1, 40, iters);
    let dense_ns = time_forced_path(layer, &x, 2, 40, iters);
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
    RegimeLatency {
        batch,
        adaptive_ns,
        cached_ns,
        sparse_ns,
        oracle_ns,
        oracle_path: oracle_path.to_string(),
        sparse_frac,
    }
}

fn run_gate_curve(base_config: &Config, seed: u64) -> BenchRunResult {
    println!("[Bench] Running Gate Curve Experiment...");
    let mut rng = Rng::from_seed(seed);
    let mut cfg = bench_sized_config(base_config);
    cfg.achf.enabled = true;
    // The gate-curve experiment exists to exercise the adaptive path-selection
    // (AMA) machinery: latency probing, EMA scoring, and re-selection across
    // Cached/Sparse/Dense. A frozen layer's deterministic fast path only ever
    // resolves to Cached-or-Dense (the fused cached operator is permanently
    // cheapest, so Sparse is structurally unreachable and no latency samples are
    // taken). Without this flag the summary reports latency_samples=0 and
    // sparse_paths=0 — the very mechanism the experiment is meant to measure
    // never runs. Enabling adaptive inference keeps the selector live after
    // freeze (weights fixed, path adaptive).
    cfg.achf.adaptive_inference = true;
    cfg.ppo_total_steps = cfg.ppo_total_steps.min(4000);
    cfg.ppo_steps_per_update = cfg.ppo_steps_per_update.min(256);
    cfg.ppo_k_epochs = cfg.ppo_k_epochs.min(2);

    let (env_net, neural_opt, _worker) = build_base_models(&cfg, &mut rng);

    let (snapshots_tx, snapshots_rx) = std::sync::mpsc::channel();
    let start = Instant::now();
    let ppo = train_ppo_with_metrics(&mut rng, &env_net, &cfg, Some(snapshots_tx));
    let elapsed = start.elapsed();

    let snapshots: Vec<StepSnapshot> = snapshots_rx.try_iter().collect();
    let final_reward = snapshots.last().map_or(0.0, |s| s.reward);
    let throughput = measure_inference_throughput(
        &mut rng,
        &ThroughputParams {
            neural_opt: &neural_opt,
            dqn: None,
            ppo: Some(&ppo),
            env_net: &env_net,
            config: &cfg,
            sims: CURVE_THROUGHPUT_SIMS,
            pulls: THROUGHPUT_PULLS,
        },
    );

    BenchRunResult {
        label: "Gate Curve".to_string(),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_sims_per_sec: throughput,
        final_avg_reward: final_reward,
        final_loss: snapshots.last().map_or(0.0, |s| s.loss),
        param_count: ppo.param_count(),
        snapshots,
        cache_stats: aggregate_model_cache_stats(&cfg, None, Some(&ppo)),
    }
}

fn run_scale_test(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Scale Test (varying rank + ACHF off baseline)...");
    let mut agg = Vec::new();
    // Baseline: ACHF disabled
    {
        let label = "No ACHF";
        println!("  [{}]", label);
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = false;
        let runs = run_multi_trial(label, &cfg, seed, nt);
        agg.push(aggregate_trials(&runs));
    }
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
        println!("  [{}]", label);
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = true;
        cfg.achf.rank = rank;
        let runs = run_multi_trial(&label, &cfg, seed, nt);
        agg.push(aggregate_trials(&runs));
    }
    agg
}

fn run_apply_combination(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Apply Combination Experiment...");
    let combos: Vec<(&str, bool, bool, bool)> = vec![
        ("None", false, false, false),
        ("FFN only", false, true, false),
        ("Attn only", true, false, false),
        ("DQN only", false, false, true),
        ("FFN+Attn", true, true, false),
        ("FFN+Attn+DQN", true, true, true),
    ];
    let mut agg = Vec::new();
    for (label, attn, ffn, dqn_flag) in combos {
        println!("  [{}]", label);
        let mut cfg = bench_sized_config(base_config);
        cfg.achf.enabled = attn || ffn || dqn_flag;
        cfg.achf.apply_attn = attn;
        cfg.achf.apply_ffn = ffn;
        cfg.achf.apply_dqn = dqn_flag;
        cfg.luck_mode = if dqn_flag && !attn && !ffn {
            LuckMode::Dqn
        } else {
            LuckMode::Ppo
        };
        let runs = run_multi_trial(label, &cfg, seed, nt);
        agg.push(aggregate_trials(&runs));
    }
    agg
}

fn run_convergence(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Convergence Experiment (ACHF on/off loss curves)...");
    let mut agg = Vec::new();
    for (label, enabled) in [("ACHF Enabled", true), ("ACHF Disabled", false)] {
        println!("  [{}]", label);
        let mut runs = Vec::new();
        for t in 0..nt {
            let trial_seed = seed.wrapping_add(t as u64 * 1337);
            let mut rng = Rng::from_seed(trial_seed);
            let mut cfg = bench_sized_config(base_config);
            cfg.achf.enabled = enabled;
            cfg.ppo_total_steps = cfg.ppo_total_steps.min(4000);
            cfg.ppo_steps_per_update = cfg.ppo_steps_per_update.min(256);
            cfg.ppo_k_epochs = cfg.ppo_k_epochs.min(2);
            let (env_net, neural_opt, _worker) = build_base_models(&cfg, &mut rng);
            let (tx, rx) = std::sync::mpsc::channel();
            let start = Instant::now();
            let ppo = train_ppo_with_metrics(&mut rng, &env_net, &cfg, Some(tx));
            let elapsed = start.elapsed();
            let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();
            let final_reward = snapshots.last().map_or(0.0, |s| s.reward);
            let final_loss = snapshots.last().map_or(0.0, |s| s.loss);
            let throughput = measure_inference_throughput(
                &mut rng,
                &ThroughputParams {
                    neural_opt: &neural_opt,
                    dqn: None,
                    ppo: Some(&ppo),
                    env_net: &env_net,
                    config: &cfg,
                    sims: CURVE_THROUGHPUT_SIMS,
                    pulls: THROUGHPUT_PULLS,
                },
            );
            let cache_stats = if enabled {
                let stats = aggregate_model_cache_stats(&cfg, None, Some(&ppo))
                    .unwrap_or_else(|| ppo.achf_cache_stats_aggregate());
                AchfCacheStats::debug_print(&[stats]);
                Some(stats)
            } else {
                None
            };
            println!(
                "    trial {}/{}: {:.1}s | {:.0} sims/sec | reward: {:.3} | loss: {:.4} | {} snapshots",
                t + 1,
                nt,
                elapsed.as_secs_f64(),
                throughput,
                final_reward,
                final_loss,
                snapshots.len()
            );
            runs.push(BenchRunResult {
                label: label.to_string(),
                total_time_ms: elapsed.as_secs_f64() * 1000.0,
                throughput_sims_per_sec: throughput,
                final_avg_reward: final_reward,
                final_loss,
                param_count: ppo.param_count(),
                snapshots,
                cache_stats,
            });
        }
        agg.push(aggregate_trials(&runs));
    }
    agg
}

// ── Shared training + measurement helper ────────────────────────────────

fn train_and_measure(label: &str, config: &Config, seed: u64) -> BenchRunResult {
    let mut rng = Rng::from_seed(seed);
    let mut cfg = config.clone();
    if cfg.fast_init {
        cfg.ppo_total_steps = cfg.ppo_total_steps.min(2000);
        cfg.ppo_steps_per_update = cfg.ppo_steps_per_update.min(256);
        cfg.ppo_k_epochs = cfg.ppo_k_epochs.min(2);
    }
    let (env_net, neural_opt, _worker) = build_base_models(&cfg, &mut rng);

    let (tx, rx) = std::sync::mpsc::channel();
    let train_start = Instant::now();
    let dqn = train_dqn_with_metrics(&neural_opt, &mut rng, &env_net, &cfg, None);
    let ppo = train_ppo_with_metrics(&mut rng, &env_net, &cfg, Some(tx));
    let train_elapsed = train_start.elapsed();

    let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();

    let throughput = measure_inference_throughput(
        &mut rng,
        &ThroughputParams {
            neural_opt: &neural_opt,
            dqn: Some(&dqn),
            ppo: Some(&ppo),
            env_net: &env_net,
            config: &cfg,
            sims: THROUGHPUT_SIMS,
            pulls: THROUGHPUT_PULLS,
        },
    );

    let cache_stats = if cfg.achf.enabled {
        let stats = aggregate_model_cache_stats(&cfg, Some(&dqn), Some(&ppo))
            .expect("enabled ACHF benchmark should produce aggregate stats");
        AchfCacheStats::debug_print(&[stats]);
        if let Some(snapshot) = ppo.snapshot_achf().or_else(|| dqn.snapshot_achf()) {
            if snapshot.low_rank_applied_rank > 0 {
                println!(
                    "    [ACHF] low-rank: rank={} rel_err={:.4}",
                    snapshot.low_rank_applied_rank, snapshot.low_rank_rel_err
                );
            }
        }
        Some(stats)
    } else {
        None
    };

    let pc = ppo.param_count() + dqn.param_count();
    BenchRunResult {
        label: label.to_string(),
        total_time_ms: train_elapsed.as_secs_f64() * 1000.0,
        throughput_sims_per_sec: throughput,
        final_avg_reward: snapshots.last().map_or(0.0, |s| s.reward),
        final_loss: snapshots.last().map_or(0.0, |s| s.loss),
        param_count: pc,
        snapshots,
        cache_stats,
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
    chart_agg_reward_curve(agg, dir, ext, "ablation_reward", "Ablation: Reward Curve");
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

fn chart_path_latency(latencies: &[(String, Vec<f64>)], dir: &str, ext: &str) {
    let stats: Vec<(&str, [f64; 5])> = latencies
        .iter()
        .filter(|(_, vals)| !vals.is_empty())
        .map(|(name, vals)| {
            let mut sorted = vals.clone();
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
            (name.as_str(), q)
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

fn chart_gate_curve(result: &BenchRunResult, dir: &str, ext: &str) {
    if result.snapshots.is_empty() {
        return;
    }
    let gate: Vec<(f64, f64)> = result
        .snapshots
        .iter()
        .map(|s| (s.step as f64, s.gate_value))
        .collect();
    let gmin: Vec<(f64, f64)> = result
        .snapshots
        .iter()
        .map(|s| (s.step as f64, s.g_min))
        .collect();
    let grad: Vec<(f64, f64)> = result
        .snapshots
        .iter()
        .map(|s| (s.step as f64, s.grad_ema))
        .collect();
    let hit: Vec<(f64, f64)> = result
        .snapshots
        .iter()
        .map(|s| (s.step as f64, s.cache_hit_rate))
        .collect();
    let lr_ratio: Vec<(f64, f64)> = result
        .snapshots
        .iter()
        .map(|s| (s.step as f64, s.sparse_ratio))
        .collect();
    let abias: Vec<(f64, f64)> = result
        .snapshots
        .iter()
        .map(|s| (s.step as f64, s.adaptive_bias))
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
            .map(|c| (c.weight_sparsity as f64, c.cached_ns))
            .collect();
        let sparse: Vec<(f64, f64)> = rows
            .iter()
            .map(|c| (c.weight_sparsity as f64, c.sparse_ns))
            .collect();
        let dense: Vec<(f64, f64)> = rows
            .iter()
            .map(|c| (c.weight_sparsity as f64, c.dense_ns))
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
                    r.weight_sparsity, r.small.batch, r.small.oracle_path
                ),
                format!(
                    "wsp{:.2} b{}={}",
                    r.weight_sparsity, r.large.batch, r.large.oracle_path
                ),
            ]
        })
        .collect();
    let mut bars: Vec<(&str, f64)> = Vec::with_capacity(labels.len());
    for (i, r) in rows.iter().enumerate() {
        bars.push((
            labels[2 * i].as_str(),
            r.small.adaptive_ns / r.small.oracle_ns.max(1.0),
        ));
        bars.push((
            labels[2 * i + 1].as_str(),
            r.large.adaptive_ns / r.large.oracle_ns.max(1.0),
        ));
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
        .filter(|a| !a.best_snapshots.is_empty())
        .map(|a| {
            let pts: Vec<(f64, f64)> = a
                .best_snapshots
                .iter()
                .map(|s| (s.step as f64, s.loss))
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
        "Convergence: Reward Curve",
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
        .filter(|a| !a.best_snapshots.is_empty())
        .map(|a| {
            let pts: Vec<(f64, f64)> = a
                .best_snapshots
                .iter()
                .map(|s| (s.step as f64, s.reward))
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
            "Avg Reward",
            &series_ref,
            900,
            500,
        ),
    );
}

// ── Output: summary, JSON, CSV ──────────────────────────────────────────

/// Effective low-rank truncation actually applied by a config's ACHF layer,
/// taken from its final snapshot. Returns None when the config produced no
/// snapshots. A value of 0 means the requested rank was a no-op (>= the layer's
/// smaller dimension), which is exactly the degenerate case worth surfacing.
fn effective_applied_rank(a: &AggregatedResult) -> Option<usize> {
    a.best_snapshots.last().map(|s| s.low_rank_applied_rank)
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
            "  {:20} | tput: {:.0} +/- {:.0} | reward: {:.4} +/- {:.4} | loss: {:.4} +/- {:.4} | params: {}{}",
            a.label,
            a.throughput.mean, a.throughput.std_dev,
            a.reward.mean, a.reward.std_dev,
            a.loss.mean, a.loss.std_dev,
            a.param_count,
            rank_note
        );
    }
}

fn write_summary_txt(
    all: &[(&str, Vec<AggregatedResult>)],
    path_latencies: Option<&[(String, Vec<f64>)]>,
    gate_curve: Option<&BenchRunResult>,
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
                "  {:20} | tput={:.0}+/-{:.0} | reward={:.4}+/-{:.4} | loss={:.4}+/-{:.4} | params={}{}",
                a.label,
                a.throughput.mean, a.throughput.std_dev,
                a.reward.mean, a.reward.std_dev,
                a.loss.mean, a.loss.std_dev,
                a.param_count,
                rank_note
            ));
            if let Some(ref stats) = a.cache_stats {
                let calls = stats.calls as f64;
                let hit_pct = if calls > 0.0 {
                    stats.cache_hits as f64 / calls * 100.0
                } else {
                    0.0
                };
                lines.push(format!(
                    "    ACHF: calls={} hit={:.1}% lr={} dense={} latency_samples={} bias={:.3}",
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
                "  {:20} | samples={} | mean={:.1}ns | p50={:.1}ns | p95={:.1}ns | p99={:.1}ns",
                stats.label, stats.samples, stats.mean_ns, stats.p50_ns, stats.p95_ns, stats.p99_ns
            ));
        }
        lines.push(String::new());
    }
    if let Some(result) = gate_curve {
        lines.push("=== gate_curve ===".to_string());
        lines.push(format!(
            "  {:20} | snapshots={} | tput={:.0} | reward={:.4} | loss={:.4} | params={} | train_time_ms={:.1}",
            result.label,
            result.snapshots.len(),
            result.throughput_sims_per_sec,
            result.final_avg_reward,
            result.final_loss,
            result.param_count,
            result.total_time_ms
        ));
        if let Some(last) = result.snapshots.last() {
            lines.push(format!(
                "    final training: step={} gate={:.4} g_min={:.4} hit={:.1}% sparse={:.1}% bias={:.3}",
                last.step,
                last.gate_value,
                last.g_min,
                last.cache_hit_rate * 100.0,
                last.sparse_ratio * 100.0,
                last.adaptive_bias
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
    let path = format!("{}/summary.txt", dir);
    write_text_file(&path, &lines.join("\n"));
    println!("[Bench] Summary -> {}", path);
}

fn write_summary_json(
    all: &[(&str, Vec<AggregatedResult>)],
    path_latencies: Option<&[(String, Vec<f64>)]>,
    gate_curve: Option<&BenchRunResult>,
    dir: &str,
) {
    let mut root = serde_json::Map::new();
    for (name, agg) in all {
        let entries: Vec<serde_json::Value> = agg
            .iter()
            .map(|a| {
                let mut entry = serde_json::json!({
                    "label": a.label.as_str(),
                    "throughput_mean": a.throughput.mean,
                    "throughput_std": a.throughput.std_dev,
                    "reward_mean": a.reward.mean,
                    "reward_std": a.reward.std_dev,
                    "loss_mean": a.loss.mean,
                    "loss_std": a.loss.std_dev,
                    "param_count": a.param_count,
                    "throughput_ci": [a.throughput.ci_low, a.throughput.ci_high],
                    "reward_ci": [a.reward.ci_low, a.reward.ci_high],
                });
                if let Some(stats) = a.cache_stats {
                    entry["cache_stats"] = cache_stats_json(stats);
                }
                entry
            })
            .collect();
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
    let json = serde_json::to_string_pretty(&serde_json::Value::Object(root))
        .expect("benchmark summary JSON should be serializable");
    let path = format!("{}/summary.json", dir);
    write_text_file(&path, &json);
    println!("[Bench] JSON  -> {}", path);
}

fn write_csvs(all: &[(&str, Vec<AggregatedResult>)], dir: &str) {
    for (name, agg) in all {
        let mut csv = String::from("label,trial,throughput,reward,loss,time_ms,param_count\n");
        for a in agg {
            let label = csv_escape(&a.label);
            for (t, ((&tput, &rew), &loss)) in a
                .throughput
                .values
                .iter()
                .zip(a.reward.values.iter())
                .zip(a.loss.values.iter())
                .enumerate()
            {
                let time = if t < a.time_ms.values.len() {
                    a.time_ms.values[t]
                } else {
                    0.0
                };
                csv.push_str(&format!(
                    "{},{},{:.2},{:.4},{:.4},{:.1},{}\n",
                    label,
                    t + 1,
                    tput,
                    rew,
                    loss,
                    time,
                    a.param_count
                ));
            }
        }
        let path = format!("{}/{}.csv", dir, name);
        write_text_file(&path, &csv);
        println!("[Bench] CSV   -> {}", path);
    }
}

fn write_path_latency_outputs(latencies: &[(String, Vec<f64>)], dir: &str) {
    let mut csv = String::from("label,sample,latency_ns\n");
    for (label, values) in latencies {
        let label = csv_escape(label);
        for (idx, latency_ns) in values.iter().enumerate() {
            csv.push_str(&format!("{},{},{:.3}\n", label, idx + 1, latency_ns));
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
    let mut csv = String::from("dim,weight_sparsity,cached_ns,sparse_ns,dense_ns,winner\n");
    for c in cells {
        csv.push_str(&format!(
            "{},{:.4},{:.3},{:.3},{:.3},{}\n",
            c.dim, c.weight_sparsity, c.cached_ns, c.sparse_ns, c.dense_ns, c.winner
        ));
    }
    let csv_path = format!("{}/path_crossover.csv", dir);
    write_text_file(&csv_path, &csv);
    println!("[Bench] CSV   -> {}", csv_path);

    let arr: Vec<serde_json::Value> = cells
        .iter()
        .map(|c| {
            serde_json::json!({
                "dim": c.dim,
                "weight_sparsity": c.weight_sparsity,
                "cached_ns": c.cached_ns,
                "sparse_ns": c.sparse_ns,
                "dense_ns": c.dense_ns,
                "winner": c.winner,
            })
        })
        .collect();
    let json = serde_json::to_string_pretty(&serde_json::Value::Array(arr))
        .expect("crossover summary JSON should be serializable");
    let json_path = format!("{}/path_crossover_summary.json", dir);
    write_text_file(&json_path, &json);
    println!("[Bench] JSON  -> {}", json_path);
}

fn write_regime_outputs(rows: &[RegimeRow], dir: &str) {
    let mut csv = String::from(
        "weight_sparsity,batch,adaptive_ns,cached_ns,sparse_ns,dense_oracle_ns,oracle_path,oracle_gap,sparse_frac\n",
    );
    let push = |wsp: f32, r: &RegimeLatency, csv: &mut String| {
        csv.push_str(&format!(
            "{:.4},{},{:.3},{:.3},{:.3},{:.3},{},{:.4},{:.4}\n",
            wsp,
            r.batch,
            r.adaptive_ns,
            r.cached_ns,
            r.sparse_ns,
            r.oracle_ns,
            r.oracle_path,
            r.adaptive_ns / r.oracle_ns.max(1.0),
            r.sparse_frac,
        ));
    };
    for r in rows {
        push(r.weight_sparsity, &r.small, &mut csv);
        push(r.weight_sparsity, &r.large, &mut csv);
    }
    let csv_path = format!("{}/regime_adaptation.csv", dir);
    write_text_file(&csv_path, &csv);
    println!("[Bench] CSV   -> {}", csv_path);

    let cell = |wsp: f32, r: &RegimeLatency| {
        serde_json::json!({
            "weight_sparsity": wsp,
            "batch": r.batch,
            "adaptive_ns": r.adaptive_ns,
            "cached_ns": r.cached_ns,
            "sparse_ns": r.sparse_ns,
            "oracle_ns": r.oracle_ns,
            "oracle_path": r.oracle_path,
            "oracle_gap": r.adaptive_ns / r.oracle_ns.max(1.0),
            "sparse_frac": r.sparse_frac,
        })
    };
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .flat_map(|r| {
            [
                cell(r.weight_sparsity, &r.small),
                cell(r.weight_sparsity, &r.large),
            ]
        })
        .collect();
    let json = serde_json::to_string_pretty(&serde_json::Value::Array(arr))
        .expect("regime summary JSON should be serializable");
    let json_path = format!("{}/regime_adaptation_summary.json", dir);
    write_text_file(&json_path, &json);
    println!("[Bench] JSON  -> {}", json_path);
}

fn write_gate_curve_outputs(result: &BenchRunResult, dir: &str) {
    let mut csv = String::from(
        "step,gate_value,g_min,grad_ema,loss,reward,cache_hit_rate,sparse_ratio,ema_cached_ns,ema_sparse_ns,adaptive_bias,sinkhorn_iterations,sinkhorn_row_max_dev,sinkhorn_col_max_dev,sinkhorn_min_value,sinkhorn_negative_ratio,sinkhorn_warm_started\n",
    );
    for snapshot in &result.snapshots {
        csv.push_str(&format!(
            "{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3},{:.3},{:.8},{},{:.8},{:.8},{:.8},{:.8},{}\n",
            snapshot.step,
            snapshot.gate_value,
            snapshot.g_min,
            snapshot.grad_ema,
            snapshot.loss,
            snapshot.reward,
            snapshot.cache_hit_rate,
            snapshot.sparse_ratio,
            snapshot.ema_cached_ns,
            snapshot.ema_sparse_ns,
            snapshot.adaptive_bias,
            snapshot.sinkhorn_iterations,
            snapshot.sinkhorn_row_max_dev,
            snapshot.sinkhorn_col_max_dev,
            snapshot.sinkhorn_min_value,
            snapshot.sinkhorn_negative_ratio,
            snapshot.sinkhorn_warm_started
        ));
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

fn path_latency_stats(latencies: &[(String, Vec<f64>)]) -> Vec<PathLatencyStats> {
    latencies
        .iter()
        .filter(|(_, values)| !values.is_empty())
        .map(|(label, values)| {
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let samples = sorted.len();
            let mean_ns = sorted.iter().sum::<f64>() / samples as f64;
            PathLatencyStats {
                label: label.clone(),
                samples,
                mean_ns,
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

fn path_latency_stats_json(latencies: &[(String, Vec<f64>)]) -> Vec<serde_json::Value> {
    path_latency_stats(latencies)
        .into_iter()
        .map(|stats| {
            serde_json::json!({
                "label": stats.label,
                "samples": stats.samples,
                "mean_ns": stats.mean_ns,
                "min_ns": stats.min_ns,
                "p50_ns": stats.p50_ns,
                "p90_ns": stats.p90_ns,
                "p95_ns": stats.p95_ns,
                "p99_ns": stats.p99_ns,
                "max_ns": stats.max_ns,
            })
        })
        .collect()
}

fn gate_curve_summary_json(result: &BenchRunResult) -> serde_json::Value {
    let final_step = result.snapshots.last().map_or(0, |s| s.step);
    serde_json::json!({
        "label": result.label.as_str(),
        "total_time_ms": result.total_time_ms,
        "throughput_sims_per_sec": result.throughput_sims_per_sec,
        "final_avg_reward": result.final_avg_reward,
        "final_loss": result.final_loss,
        "param_count": result.param_count,
        "snapshot_count": result.snapshots.len(),
        "final_step": final_step,
        "final_snapshot": result.snapshots.last().map(step_snapshot_json),
        "cache_stats": result.cache_stats.map(cache_stats_json),
    })
}

fn step_snapshot_json(snapshot: &StepSnapshot) -> serde_json::Value {
    serde_json::json!({
        "step": snapshot.step,
        "gate_value": snapshot.gate_value,
        "g_min": snapshot.g_min,
        "grad_ema": snapshot.grad_ema,
        "loss": snapshot.loss,
        "reward": snapshot.reward,
        "cache_hit_rate": snapshot.cache_hit_rate,
        "sparse_ratio": snapshot.sparse_ratio,
        "ema_cached_ns": snapshot.ema_cached_ns,
        "ema_sparse_ns": snapshot.ema_sparse_ns,
        "adaptive_bias": snapshot.adaptive_bias,
        "sinkhorn_iterations": snapshot.sinkhorn_iterations,
        "sinkhorn_row_max_dev": snapshot.sinkhorn_row_max_dev,
        "sinkhorn_col_max_dev": snapshot.sinkhorn_col_max_dev,
        "sinkhorn_min_value": snapshot.sinkhorn_min_value,
        "sinkhorn_negative_ratio": snapshot.sinkhorn_negative_ratio,
        "sinkhorn_warm_started": snapshot.sinkhorn_warm_started,
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
        "decision_ema_ns": stats.decision_ema_ns,
        "decision_ema_long_ns": stats.decision_ema_long_ns,
        "adaptive_bias": stats.adaptive_bias,
        "latency_samples": stats.latency_samples,
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
        let result = AggregatedResult {
            label: "label,with\"quote".to_string(),
            throughput: TrialStats::from_values(&[10.0]),
            reward: TrialStats::from_values(&[0.5]),
            loss: TrialStats::from_values(&[0.25]),
            time_ms: TrialStats::from_values(&[12.0]),
            param_count: 42,
            best_snapshots: Vec::new(),
            cache_stats: Some(AchfCacheStats {
                calls: 4,
                cache_hits: 1,
                cache_misses: 1,
                cache_skips: 0,
                sparse_paths: 2,
                dense_paths: 1,
                ema_cached_ns: 10.0,
                ema_cached_long_ns: 11.0,
                ema_sparse_ns: 20.0,
                ema_sparse_long_ns: 21.0,
                ema_dense_ns: 30.0,
                ema_dense_long_ns: 31.0,
                decision_ema_ns: 3.0,
                decision_ema_long_ns: 4.0,
                adaptive_bias: 1.0,
                latency_samples: 3,
                dense_latency_samples: 1,
                decision_samples: 2,
            }),
        };

        write_summary_json(&[("exp", vec![result])], None, None, &output_dir);

        let json_path = dir.join("summary.json");
        let json = std::fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["exp"][0]["label"], "label,with\"quote");
        assert_eq!(parsed["exp"][0]["cache_stats"]["calls"], 4);
        assert_eq!(parsed["exp"][0]["cache_stats"]["hit_rate"], 0.25);

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
    fn path_latency_outputs_include_stats_in_csv_json_and_summary() {
        let dir = unique_temp_dir("path_latency");
        let output_dir = dir.to_string_lossy().to_string();
        let latencies = vec![
            ("Dense,path".to_string(), vec![30.0, 10.0, 20.0, 40.0]),
            ("Empty".to_string(), Vec::new()),
        ];

        let stats = path_latency_stats(&latencies);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].label, "Dense,path");
        assert_eq!(stats[0].samples, 4);
        assert_eq!(stats[0].mean_ns, 25.0);
        assert_eq!(stats[0].min_ns, 10.0);
        assert_eq!(stats[0].p50_ns, 30.0);
        assert_eq!(stats[0].p95_ns, 40.0);

        write_path_latency_outputs(&latencies, &output_dir);
        write_summary_json(&[], Some(&latencies), None, &output_dir);

        let csv = std::fs::read_to_string(dir.join("path_latency.csv")).unwrap();
        assert!(csv.contains("\"Dense,path\",1,30.000"));
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(json["path_latency"][0]["label"], "Dense,path");
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
        let layer = build_synthetic_achf_layer(dim, 0.75, false);
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
                weight_sparsity: 0.9,
                cached_ns: 100.0,
                sparse_ns: 80.0,
                dense_ns: 120.0,
                winner: "Sparse".to_string(),
            },
            CrossoverCell {
                dim: 256,
                weight_sparsity: 0.5,
                cached_ns: 70.0,
                sparse_ns: 300.0,
                dense_ns: 90.0,
                winner: "Cached".to_string(),
            },
        ];
        write_crossover_outputs(&cells, &output_dir);
        let csv = std::fs::read_to_string(dir.join("path_crossover.csv")).unwrap();
        assert!(csv.contains("dim,weight_sparsity,cached_ns,sparse_ns,dense_ns,winner"));
        assert!(csv.contains("256,0.9000,100.000,80.000,120.000,Sparse"));
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("path_crossover_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(json[0]["winner"], "Sparse");
        assert_eq!(json[1]["dim"], 256);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn regime_outputs_write_parseable_csv_and_json_with_oracle_gap() {
        let dir = unique_temp_dir("regime");
        let output_dir = dir.to_string_lossy().to_string();
        let rows = vec![RegimeRow {
            weight_sparsity: 0.9,
            small: RegimeLatency {
                batch: 1,
                adaptive_ns: 90.0,
                cached_ns: 120.0,
                sparse_ns: 70.0,
                oracle_ns: 70.0,
                oracle_path: "Sparse".to_string(),
                sparse_frac: 0.6,
            },
            large: RegimeLatency {
                batch: 128,
                adaptive_ns: 210.0,
                cached_ns: 200.0,
                sparse_ns: 280.0,
                oracle_ns: 200.0,
                oracle_path: "Cached".to_string(),
                sparse_frac: 0.1,
            },
        }];
        write_regime_outputs(&rows, &output_dir);
        let csv = std::fs::read_to_string(dir.join("regime_adaptation.csv")).unwrap();
        // Small-batch oracle is Sparse, large-batch oracle is Cached: the two
        // regimes have different best fixed paths — the core adaptation result.
        assert!(csv.contains("0.9000,1,90.000,120.000,70.000,70.000,Sparse,1.2857,0.6000"));
        assert!(csv.contains("0.9000,128,210.000,200.000,280.000,200.000,Cached,1.0500,0.1000"));
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("regime_adaptation_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(json[0]["oracle_path"], "Sparse");
        assert_eq!(json[1]["oracle_path"], "Cached");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn gate_curve_outputs_include_snapshot_and_cache_summary() {
        let dir = unique_temp_dir("gate_curve");
        let output_dir = dir.to_string_lossy().to_string();
        let result = BenchRunResult {
            label: "Gate Curve".to_string(),
            total_time_ms: 12.5,
            throughput_sims_per_sec: 0.0,
            final_avg_reward: 1.25,
            final_loss: 0.125,
            param_count: 7,
            snapshots: vec![StepSnapshot {
                step: 10,
                gate_value: 0.8,
                g_min: 0.2,
                grad_ema: 0.3,
                loss: 0.125,
                reward: 1.25,
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
            }],
            cache_stats: Some(AchfCacheStats {
                calls: 8,
                cache_hits: 2,
                cache_misses: 1,
                cache_skips: 1,
                sparse_paths: 3,
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
                latency_samples: 6,
                dense_latency_samples: 2,
                decision_samples: 4,
            }),
        };

        write_gate_curve_outputs(&result, &output_dir);
        write_summary_json(&[], None, Some(&result), &output_dir);

        let csv = std::fs::read_to_string(dir.join("gate_curve.csv")).unwrap();
        assert!(csv.contains("10,0.80000000,0.20000000,0.30000000"));
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(json["gate_curve"]["snapshot_count"], 1);
        assert_eq!(json["gate_curve"]["final_snapshot"]["step"], 10);
        assert_eq!(
            json["gate_curve"]["final_snapshot"]["sinkhorn_iterations"],
            20
        );
        assert_eq!(json["gate_curve"]["cache_stats"]["hit_rate"], 0.25);
        // All path rates share the denominator `calls` (=8) and are mutually
        // consistent: cached 2/8, sparse 3/8, dense 1/8. This guards against the
        // former denominator mismatch where path rates excluded cache hits.
        assert_eq!(json["gate_curve"]["cache_stats"]["cached_path_rate"], 0.25);
        assert_eq!(json["gate_curve"]["cache_stats"]["sparse_path_rate"], 0.375);
        assert_eq!(json["gate_curve"]["cache_stats"]["dense_path_rate"], 0.125);
        let summary: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("gate_curve_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["final_step"], 10);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
