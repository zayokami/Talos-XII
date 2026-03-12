use crate::achf::AchfCacheStats;
use crate::chart::{self, ChartFormat};
use crate::config::Config;
use crate::dbn::Dbn;
use crate::dqn::{train_dqn_with_metrics, DuelingQNetwork};
use crate::neural::NeuralLuckOptimizer;
use crate::ppo::{train_ppo_with_metrics, ActorCritic};
use crate::rng::Rng;
use crate::sim::{simulate_fast, SimModelContext};
use crate::trainer::{train_linear_regression, train_manifold_rl, train_neural_optimizer};
use crate::worker::GoodJobWorker;
use std::fs;
use std::time::Instant;

// ── Data structures ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct StepSnapshot {
    pub step: usize,
    pub gate_value: f64,
    pub g_min: f64,
    pub grad_ema: f64,
    pub loss: f64,
    pub reward: f64,
    pub cache_hit_rate: f64,
    pub low_rank_ratio: f64,
    pub ema_cached_ns: f64,
    pub ema_low_rank_ns: f64,
    pub adaptive_bias: f64,
}

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
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std_dev = variance.sqrt();
        let t_val = 2.776; // t-distribution 95% CI, df=4 (conservative for small n)
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

fn build_base_models(config: &Config, rng: &mut Rng) -> (Dbn, NeuralLuckOptimizer, GoodJobWorker) {
    let worker = GoodJobWorker::new_with_config(config);
    let mut dbn = Dbn::new(&[8, 16, 8], rng);
    let (count, epochs) = if config.fast_init {
        (256, 4)
    } else {
        (1024, 20)
    };
    dbn.train(rng, count, epochs);

    let mut neural_opt = train_neural_optimizer(rng.next_u64(), &dbn, config, &worker);
    let (w, b) = train_linear_regression(&neural_opt, rng, &dbn, config);
    neural_opt.set_linear_params(w, b);
    neural_opt = train_manifold_rl(&neural_opt, rng, &dbn, config, &worker);

    (dbn, neural_opt, worker)
}

struct ThroughputParams<'a> {
    neural_opt: &'a NeuralLuckOptimizer,
    dqn: Option<&'a DuelingQNetwork>,
    ppo: Option<&'a ActorCritic>,
    dbn: &'a Dbn,
    config: &'a Config,
    sims: usize,
    pulls: usize,
}

fn measure_inference_throughput(rng: &mut Rng, params: &ThroughputParams<'_>) -> f64 {
    let ctx = SimModelContext {
        neural_opt: params.neural_opt,
        dqn_policy: params.dqn,
        ppo_policy: params.ppo,
        dbn: params.dbn,
        config: params.config,
        exp_sender: None,
        neural_sender: None,
        ppo_sender: None,
    };
    // Warmup: stabilize CPU cache and branch predictors
    for _ in 0..500 {
        let _ = simulate_fast(params.pulls, rng, 0, &ctx);
    }
    let start = Instant::now();
    for _ in 0..params.sims {
        let _ = simulate_fast(params.pulls, rng, 0, &ctx);
    }
    let elapsed = start.elapsed();
    params.sims as f64 / elapsed.as_secs_f64()
}

fn should_run(bench_cfg: &BenchConfig, name: &str) -> bool {
    match &bench_cfg.only {
        None => true,
        Some(list) => list.iter().any(|s| s == name),
    }
}

fn ext(fmt: &ChartFormat) -> &'static str {
    match fmt {
        ChartFormat::Svg => "svg",
        ChartFormat::Png => "png",
    }
}

// ── Main entry point ────────────────────────────────────────────────────

pub fn run_paper_benchmarks(base_config: &Config, seed: u64, bench_cfg: &BenchConfig) {
    let dir = &bench_cfg.output_dir;
    let nt = bench_cfg.num_trials;
    fs::create_dir_all(dir).expect("Failed to create output directory");

    println!("\n========================================");
    println!("  ACHF Paper Benchmark Suite");
    println!("  Trials per experiment: {}", nt);
    println!("========================================\n");

    let mut all_agg: Vec<(&str, Vec<AggregatedResult>)> = Vec::new();

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
        for (name, vals) in &latencies {
            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
            println!("  {}: avg {:.1} ns ({} samples)", name, avg, vals.len());
        }
        let e = ext(&bench_cfg.format);
        chart_path_latency(&latencies, dir, e);
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

    write_summary_txt(&all_agg, dir);
    write_summary_json(&all_agg, dir);
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
        let mut cfg = base_config.clone();
        cfg.achf.enabled = enabled;
        cfg.fast_init = true;
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
        let mut cfg = base_config.clone();
        cfg.achf.enabled = true;
        cfg.achf.mode = mode.to_string();
        cfg.fast_init = true;
        let runs = run_multi_trial(label, &cfg, seed, nt);
        agg.push(aggregate_trials(&runs));
    }
    agg
}

fn run_path_comparison(base_config: &Config, seed: u64) -> Vec<(String, Vec<f64>)> {
    println!("[Bench] Running Path Comparison (Cached/LowRank/Dense)...");
    let mut rng = Rng::from_seed(seed);
    let mut cfg = base_config.clone();
    cfg.achf.enabled = true;
    cfg.achf.rank = 16;
    cfg.fast_init = true;

    let (dbn, _neural_opt, _worker) = build_base_models(&cfg, &mut rng);
    let ppo = train_ppo_with_metrics(&mut rng, &dbn, &cfg, None);

    let input_dim = 8;
    let sample_input: Vec<f64> = (0..input_dim).map(|i| (i as f64) * 0.1 + 0.05).collect();
    let iterations = 5000;

    let mut all_latencies: Vec<(String, Vec<f64>)> = Vec::new();

    // 0 = Cached, 1 = LowRank, 2 = Dense
    for (path_name, path_id) in [("Cached", 0u8), ("LowRank", 1), ("Dense", 2)] {
        let mut latencies = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let _out = ppo.forward_inference_forced_path(&sample_input, path_id);
            let ns = start.elapsed().as_nanos() as f64;
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

fn run_gate_curve(base_config: &Config, seed: u64) -> BenchRunResult {
    println!("[Bench] Running Gate Curve Experiment...");
    let mut rng = Rng::from_seed(seed);
    let mut cfg = base_config.clone();
    cfg.achf.enabled = true;
    cfg.fast_init = true;

    let (dbn, _neural_opt, _worker) = build_base_models(&cfg, &mut rng);

    let (snapshots_tx, snapshots_rx) = std::sync::mpsc::channel();
    let start = Instant::now();
    let ppo = train_ppo_with_metrics(&mut rng, &dbn, &cfg, Some(snapshots_tx));
    let elapsed = start.elapsed();

    let snapshots: Vec<StepSnapshot> = snapshots_rx.try_iter().collect();
    let final_reward = snapshots.last().map_or(0.0, |s| s.reward);

    BenchRunResult {
        label: "Gate Curve".to_string(),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_sims_per_sec: 0.0,
        final_avg_reward: final_reward,
        final_loss: snapshots.last().map_or(0.0, |s| s.loss),
        param_count: ppo.param_count(),
        snapshots,
        cache_stats: Some(ppo.achf_cache_stats_aggregate()),
    }
}

fn run_scale_test(base_config: &Config, seed: u64, nt: usize) -> Vec<AggregatedResult> {
    println!("[Bench] Running Scale Test (varying rank + ACHF off baseline)...");
    let mut agg = Vec::new();
    // Baseline: ACHF disabled
    {
        let label = "No ACHF";
        println!("  [{}]", label);
        let mut cfg = base_config.clone();
        cfg.achf.enabled = false;
        cfg.fast_init = true;
        let runs = run_multi_trial(label, &cfg, seed, nt);
        agg.push(aggregate_trials(&runs));
    }
    for rank in [4, 8, 16, 32, 48] {
        let label = format!("rank={}", rank);
        println!("  [{}]", label);
        let mut cfg = base_config.clone();
        cfg.achf.enabled = true;
        cfg.achf.rank = rank;
        cfg.fast_init = true;
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
        let mut cfg = base_config.clone();
        cfg.achf.enabled = attn || ffn || dqn_flag;
        cfg.achf.apply_attn = attn;
        cfg.achf.apply_ffn = ffn;
        cfg.achf.apply_dqn = dqn_flag;
        cfg.fast_init = true;
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
            let mut cfg = base_config.clone();
            cfg.achf.enabled = enabled;
            cfg.fast_init = true;
            let (dbn, _neural_opt, _worker) = build_base_models(&cfg, &mut rng);
            let (tx, rx) = std::sync::mpsc::channel();
            let start = Instant::now();
            let ppo = train_ppo_with_metrics(&mut rng, &dbn, &cfg, Some(tx));
            let elapsed = start.elapsed();
            let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();
            let final_reward = snapshots.last().map_or(0.0, |s| s.reward);
            let final_loss = snapshots.last().map_or(0.0, |s| s.loss);
            let cache_stats = if enabled {
                Some(ppo.achf_cache_stats_aggregate())
            } else {
                None
            };
            println!(
                "    trial {}/{}: {:.1}s | reward: {:.3} | loss: {:.4} | {} snapshots",
                t + 1,
                nt,
                elapsed.as_secs_f64(),
                final_reward,
                final_loss,
                snapshots.len()
            );
            runs.push(BenchRunResult {
                label: label.to_string(),
                total_time_ms: elapsed.as_secs_f64() * 1000.0,
                throughput_sims_per_sec: 0.0,
                final_avg_reward: final_reward,
                final_loss,
                param_count: 0,
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
    let (dbn, neural_opt, _worker) = build_base_models(config, &mut rng);

    let (tx, rx) = std::sync::mpsc::channel();
    let train_start = Instant::now();
    let dqn = train_dqn_with_metrics(&neural_opt, &mut rng, &dbn, config, None);
    let ppo = train_ppo_with_metrics(&mut rng, &dbn, config, Some(tx));
    let train_elapsed = train_start.elapsed();

    let snapshots: Vec<StepSnapshot> = rx.try_iter().collect();

    let throughput = measure_inference_throughput(
        &mut rng,
        &ThroughputParams {
            neural_opt: &neural_opt,
            dqn: Some(&dqn),
            ppo: Some(&ppo),
            dbn: &dbn,
            config,
            sims: 5000,
            pulls: 200,
        },
    );

    let cache_stats = if config.achf.enabled {
        Some(ppo.achf_cache_stats_aggregate())
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
    if chart::draw_bar_chart_with_error(
        &path,
        "Ablation: Throughput (mean +/- std)",
        "Configuration",
        "Sims/sec",
        &bars,
        800,
        500,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }
    chart_agg_reward_curve(agg, dir, ext, "ablation_reward", "Ablation: Reward Curve");
}

fn chart_mode(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_error(agg);
    let path = format!("{}/mode_comparison.{}", dir, ext);
    if chart::draw_bar_chart_with_error(
        &path,
        "Mode: Throughput (lite vs full, mean +/- std)",
        "Mode",
        "Sims/sec",
        &bars,
        800,
        500,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }
}

fn chart_path_latency(latencies: &[(String, Vec<f64>)], dir: &str, ext: &str) {
    let stats: Vec<(&str, [f64; 5])> = latencies
        .iter()
        .map(|(name, vals)| {
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            let q = [
                sorted[0],
                sorted[n / 4],
                sorted[n / 2],
                sorted[3 * n / 4],
                sorted[n - 1],
            ];
            (name.as_str(), q)
        })
        .collect();
    let path = format!("{}/path_latency_boxplot.{}", dir, ext);
    if chart::draw_box_plot(
        &path,
        "Inference Path Latency",
        "Path",
        "Latency (ns)",
        &stats,
        800,
        500,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }
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
        .map(|s| (s.step as f64, s.low_rank_ratio))
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
        ("LowRank Ratio", &lr_ratio),
        ("Adaptive Bias", &abias),
    ];
    let path = format!("{}/gate_curve.{}", dir, ext);
    if chart::draw_line_chart(
        &path,
        "Gate Dynamics During Training",
        "Training Step",
        "Value",
        &series,
        1000,
        600,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }
}

fn chart_scale(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_error(agg);
    let path = format!("{}/scale_test.{}", dir, ext);
    if chart::draw_bar_chart_with_error(
        &path,
        "Scalability: Throughput by Rank (mean +/- std)",
        "Configuration",
        "Sims/sec",
        &bars,
        900,
        500,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }
}

fn chart_apply(agg: &[AggregatedResult], dir: &str, ext: &str) {
    let bars = agg_bars_with_error(agg);
    let path = format!("{}/apply_combination.{}", dir, ext);
    if chart::draw_bar_chart_with_error(
        &path,
        "Apply Combination: Throughput (mean +/- std)",
        "Configuration",
        "Sims/sec",
        &bars,
        1000,
        500,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }
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
    let path = format!("{}/convergence_loss.{}", dir, ext);
    if chart::draw_line_chart(
        &path,
        "Convergence: Loss Curve",
        "Training Step",
        "Loss",
        &loss_ref,
        900,
        500,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }

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
    let path = format!("{}/{}.{}", dir, filename, ext);
    if chart::draw_line_chart(
        &path,
        title,
        "Training Step",
        "Avg Reward",
        &series_ref,
        900,
        500,
    )
    .is_ok()
    {
        println!("  -> {}", path);
    }
}

// ── Output: summary, JSON, CSV ──────────────────────────────────────────

fn print_agg_summary(name: &str, agg: &[AggregatedResult]) {
    println!("[Bench] {} complete:", name);
    for a in agg {
        println!(
            "  {:20} | tput: {:.0} +/- {:.0} | reward: {:.4} +/- {:.4} | loss: {:.4} +/- {:.4} | params: {}",
            a.label,
            a.throughput.mean, a.throughput.std_dev,
            a.reward.mean, a.reward.std_dev,
            a.loss.mean, a.loss.std_dev,
            a.param_count
        );
    }
}

fn write_summary_txt(all: &[(&str, Vec<AggregatedResult>)], dir: &str) {
    let mut lines = Vec::new();
    for (name, agg) in all {
        lines.push(format!("=== {} ===", name));
        for a in agg {
            lines.push(format!(
                "  {:20} | tput={:.0}+/-{:.0} | reward={:.4}+/-{:.4} | loss={:.4}+/-{:.4} | params={}",
                a.label,
                a.throughput.mean, a.throughput.std_dev,
                a.reward.mean, a.reward.std_dev,
                a.loss.mean, a.loss.std_dev,
                a.param_count
            ));
            if let Some(ref stats) = a.cache_stats {
                let calls = stats.calls as f64;
                let hit_pct = if calls > 0.0 {
                    stats.cache_hits as f64 / calls * 100.0
                } else {
                    0.0
                };
                lines.push(format!(
                    "    ACHF: calls={} hit={:.1}% lr={} dense={} bias={:.3}",
                    stats.calls,
                    hit_pct,
                    stats.low_rank_paths,
                    stats.dense_paths,
                    stats.adaptive_bias
                ));
            }
        }
        lines.push(String::new());
    }
    let path = format!("{}/summary.txt", dir);
    fs::write(&path, lines.join("\n")).ok();
    println!("[Bench] Summary -> {}", path);
}

fn write_summary_json(all: &[(&str, Vec<AggregatedResult>)], dir: &str) {
    let mut json = String::from("{\n");
    for (i, (name, agg)) in all.iter().enumerate() {
        json.push_str(&format!("  \"{}\": [\n", name));
        for (j, a) in agg.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"label\":\"{}\",\"throughput_mean\":{:.2},\"throughput_std\":{:.2},\"reward_mean\":{:.4},\"reward_std\":{:.4},\"loss_mean\":{:.4},\"loss_std\":{:.4},\"param_count\":{},\"throughput_ci\":[{:.2},{:.2}],\"reward_ci\":[{:.4},{:.4}]}}",
                a.label,
                a.throughput.mean, a.throughput.std_dev,
                a.reward.mean, a.reward.std_dev,
                a.loss.mean, a.loss.std_dev,
                a.param_count,
                a.throughput.ci_low, a.throughput.ci_high,
                a.reward.ci_low, a.reward.ci_high
            ));
            if j + 1 < agg.len() {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  ]");
        if i + 1 < all.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push('}');
    let path = format!("{}/summary.json", dir);
    fs::write(&path, &json).ok();
    println!("[Bench] JSON  -> {}", path);
}

fn write_csvs(all: &[(&str, Vec<AggregatedResult>)], dir: &str) {
    for (name, agg) in all {
        let mut csv = String::from("label,trial,throughput,reward,loss,time_ms,param_count\n");
        for a in agg {
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
                    a.label,
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
        fs::write(&path, &csv).ok();
        println!("[Bench] CSV   -> {}", path);
    }
}

fn percentile(data: &[f64], pct: usize) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}
