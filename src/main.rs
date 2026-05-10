#![allow(clippy::wrong_self_convention)]

mod achf;
mod autograd;
mod bench;
mod binary_codec;
mod calibrate;
mod chart;
mod collect;
mod config;
#[cfg(cuda)]
mod cuda;
mod dbn;
mod dqn;
mod env_net;
#[cfg(test)]
mod grad_check;
mod i18n;
mod model_io;
mod neural;
mod nn;
mod panic_guard;
mod ppo;
mod rng;
mod sim;
mod simd;
mod trainer;
mod transformer;
mod utils;
mod worker;

use autograd::Tensor as AutoTensor;
use calibrate::{apply_calibration, run_calibration, CalibrationData};
use clap::{Parser, Subcommand};
use collect::{add_session_interactive, import_from_json, print_stats, PlayerDatabase};
use colored::Colorize;
use config::{ComputeDevice, Config, LuckMode};
use dqn::{train_dqn, DuelingQNetwork, Experience, OnlineDqnTrainer};
use env_net::EnvNet;
use i18n::{I18n, Language};
use log::{info, warn};
use neural::{NeuralLuckOptimizer, DIM};
use ppo::{train_ppo, ActorCritic, OnlinePpoTrainer};
use rng::Rng;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use worker::GoodJobWorker;

use sim::{
    build_non_up_six, format_avg_extra_cost_line, format_f2p_probability_line,
    resolve_operator_name, simulate_f2p_clearing_with_progress, simulate_fast, simulate_one,
    simulate_stats, simulate_stats_with_progress, NeuralSample, PpoExperience, SimModelContext,
    SimRunContext, COST_PER_PULL, FREE_PULLS_WELFARE,
};
use trainer::{
    train_linear_regression, train_manifold_rl, train_neural_optimizer, OnlineNeuralTrainer,
};

const NEURAL_CACHE_PATH: &str = "neural.cache";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "data/config.json")]
    config: String,

    /// Random seed (optional)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Force retraining models (ignore cache)
    #[arg(short, long)]
    force: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// Run the interactive simulator (default)
    Interactive,
    /// Run a batch of simulations
    Simulate {
        /// Number of simulations
        #[arg(short = 'n', long, default_value_t = 1000)]
        count: usize,
        /// Number of pulls per simulation
        #[arg(short = 'p', long, default_value_t = 100)]
        pulls: usize,
    },
    /// Benchmark performance
    Benchmark {
        #[command(subcommand)]
        action: Option<BenchAction>,
    },
    /// Analyze F2P welfare
    F2p,
    /// Collect player pull data
    Collect {
        #[command(subcommand)]
        action: CollectAction,
    },
    /// Train/calibrate model using collected player data
    Train,
}

#[derive(Subcommand, Clone)]
enum BenchAction {
    /// Run full ACHF benchmark suite
    Achf {
        /// Run only specific experiments (comma-separated: ablation,mode,path,gate,scale,apply,convergence)
        #[arg(long)]
        only: Option<String>,
        /// Output directory for charts and data
        #[arg(long, default_value = "bench_output")]
        output_dir: String,
        /// Chart format: svg or png
        #[arg(long, default_value = "svg")]
        format: String,
        /// Number of independent trials per experiment for statistical significance
        #[arg(long, default_value_t = 3)]
        trials: usize,
    },
}

#[derive(Subcommand, Clone)]
enum CollectAction {
    /// Interactively add a player session
    Add,
    /// Import player data from a JSON file
    Import {
        /// Path to JSON file
        file: String,
    },
    /// Show statistics of collected data
    Stats,
}

struct SimHistoryEntry {
    pool_name: String,
    pulls: usize,
    sims: usize,
    avg_six: f64,
    avg_up: f64,
    elapsed_ms: u64,
}

fn prompt_yes_no(prompt: &str, default_yes: bool) -> bool {
    for attempt in 0..3 {
        print!("{}", prompt);
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return default_yes;
        }
        let s = input.trim();
        if s.is_empty() {
            return default_yes;
        }
        if s.eq_ignore_ascii_case("y") || s.eq_ignore_ascii_case("yes") {
            return true;
        }
        if s.eq_ignore_ascii_case("n") || s.eq_ignore_ascii_case("no") {
            return false;
        }
        if attempt < 2 {
            print!("{}", "  Please enter y or n: ".yellow());
            let _ = io::stdout().flush();
        }
    }
    default_yes
}

use model_io::{
    load_env_net_cache, load_model, load_neural_cache, save_env_net_cache, save_model,
    save_neural_cache,
};
use utils::{
    INPUT_CAP, MAX_DRAIN_PER_TICK, ONLINE_REPORT_INTERVAL_SECS, PPO_ONLINE_LR, PULL_DISPLAY_LIMIT,
    SIM_HISTORY_CAPACITY,
};

/// Resolve the effective simulation count for F2P probability estimation.
fn resolve_f2p_sim_count_prob(config: &Config) -> usize {
    if config.f2p_sim_count_prob > 0 {
        return config.f2p_sim_count_prob;
    }
    if config.f2p_sim_count > 0 {
        return config.f2p_sim_count;
    }
    #[cfg(debug_assertions)]
    {
        if config.fast_init {
            2_000
        } else {
            10_000
        }
    }
    #[cfg(not(debug_assertions))]
    {
        if config.fast_init {
            50_000
        } else {
            200_000
        }
    }
}

/// Resolve the effective simulation count for F2P cost estimation.
fn resolve_f2p_sim_count_cost(config: &Config, sim_count_prob: usize) -> usize {
    if config.f2p_sim_count_cost > 0 {
        return config.f2p_sim_count_cost;
    }
    if config.f2p_sim_count > 0 {
        return config.f2p_sim_count;
    }
    #[cfg(debug_assertions)]
    let threshold = 50_000;
    #[cfg(not(debug_assertions))]
    let threshold = 200_000;

    if sim_count_prob >= threshold {
        sim_count_prob / 2
    } else {
        sim_count_prob
    }
}

fn resolve_ppo_online_train_params(config: &Config) -> (usize, usize) {
    let fast_mode = config.fast_init || config.ppo_mode == "fast";
    let k_epochs = if config.ppo_k_epochs > 0 {
        config.ppo_k_epochs
    } else if fast_mode {
        2
    } else {
        3
    };
    let batch_size = if config.ppo_batch_size > 0 {
        config.ppo_batch_size
    } else {
        128
    };
    (k_epochs, batch_size)
}

fn default_pool_index(config: &Config) -> usize {
    config
        .active_pool
        .as_ref()
        .and_then(|active| config.pools.iter().position(|pool| &pool.id == active))
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn resolve_f2p_luck_mode(config: &Config) -> LuckMode {
    config.f2p_luck_mode.unwrap_or({
        if config.luck_mode == LuckMode::Ppo {
            LuckMode::Probability
        } else {
            config.luck_mode
        }
    })
}

struct F2pAnalysisCtx<'a> {
    config: &'a Config,
    neural_opt: &'a NeuralLuckOptimizer,
    dqn_policy: Option<&'a DuelingQNetwork>,
    ppo_policy: Option<&'a ActorCritic>,
    env_net: &'a EnvNet,
    worker: &'a GoodJobWorker,
    lang: Language,
}

fn run_f2p_analysis(ctx: &F2pAnalysisCtx<'_>, rng: &mut Rng) {
    let lang = ctx.lang;
    let mut f2p_config = ctx.config.clone();
    f2p_config.luck_mode = resolve_f2p_luck_mode(ctx.config);
    let f2p_dqn_policy = if f2p_config.luck_mode == LuckMode::Dqn {
        ctx.dqn_policy
    } else {
        None
    };
    let f2p_ppo_policy = if f2p_config.luck_mode == LuckMode::Ppo {
        ctx.ppo_policy
    } else {
        None
    };
    println!(
        "{}",
        I18n::get(lang, "f2p_header").replace("{}", &FREE_PULLS_WELFARE.to_string())
    );

    let sim_count_prob = resolve_f2p_sim_count_prob(ctx.config);
    let sim_count_cost = resolve_f2p_sim_count_cost(ctx.config, sim_count_prob);

    println!(
        "{}",
        I18n::get(lang, "sys_run_prob").replace("{}", &sim_count_prob.to_string())
    );

    let start_time = Instant::now();
    let sim_ctx = SimRunContext {
        neural_opt: ctx.neural_opt,
        dqn_policy: f2p_dqn_policy,
        ppo_policy: f2p_ppo_policy,
        env_net: ctx.env_net,
        config: &f2p_config,
        worker: ctx.worker,
        exp_sender: None,
        neural_sender: None,
        ppo_sender: None,
    };

    let pb_prob = utils::create_bar(
        sim_count_prob as u64,
        &I18n::get(lang, "sys_run_prob").replace("{}", &sim_count_prob.to_string()),
    );
    let (_, total_up_agg, _, total_with_up_agg) = simulate_stats_with_progress(
        FREE_PULLS_WELFARE as usize,
        sim_count_prob,
        rng.next_u64(),
        FREE_PULLS_WELFARE,
        &sim_ctx,
        Some(&pb_prob),
    );
    pb_prob.finish_and_clear();

    let elapsed = start_time.elapsed();
    let total_sims_run = sim_count_prob;

    let prob_line = format_f2p_probability_line(total_sims_run, total_with_up_agg, lang);
    println!("{}", prob_line);
    println!(
        "{}",
        I18n::get(lang, "expected_up").replace(
            "{:.2}",
            &format!("{:.2}", total_up_agg as f64 / total_sims_run as f64)
        )
    );
    println!(
        "{}",
        I18n::get(lang, "time_taken").replace("{:.2?}", &format!("{:.2?}", elapsed))
    );
    println!(
        "{}",
        I18n::get(lang, "throughput").replace(
            "{:.0}",
            &format!("{:.0}", total_sims_run as f64 / elapsed.as_secs_f64())
        )
    );

    println!("{}", I18n::get(lang, "calc_cost"));
    println!(
        "{}",
        I18n::get(lang, "sys_run_cost").replace("{}", &sim_count_cost.to_string())
    );

    let pb_cost = utils::create_bar(
        sim_count_cost as u64,
        &I18n::get(lang, "sys_run_cost").replace("{}", &sim_count_cost.to_string()),
    );
    let (total_extra_cost_agg, extra_cost_samples_agg, _) = simulate_f2p_clearing_with_progress(
        sim_count_cost,
        rng.next_u64(),
        &sim_ctx,
        Some(&pb_cost),
    );
    pb_cost.finish_and_clear();

    let avg_extra_cost = if extra_cost_samples_agg == 0 {
        None
    } else {
        Some(total_extra_cost_agg as f64 / extra_cost_samples_agg as f64)
    };

    let avg_cost_line = format_avg_extra_cost_line(avg_extra_cost, lang);
    println!("{}", avg_cost_line);
}

fn print_explainability_report(neural_opt: &NeuralLuckOptimizer, lang: Language) {
    println!("{}", I18n::get(lang, "insight_header"));
    let rl_w = neural_opt.linear_weights;
    let rl_b = neural_opt.linear_bias;

    let mut feature_names = vec![
        I18n::get(lang, "feat_pity"),
        I18n::get(lang, "feat_total_norm"),
        I18n::get(lang, "feat_env_noise"),
        I18n::get(lang, "feat_loss_norm"),
        I18n::get(lang, "feat_streak_4"),
        I18n::get(lang, "feat_env_bias"),
        I18n::get(lang, "feat_pity_loss"),
        I18n::get(lang, "feat_total_sq"),
    ];
    while feature_names.len() < DIM {
        feature_names.push(format!("Feature {:02}", feature_names.len() + 1));
    }

    for i in 0..DIM {
        let w = rl_w[i];
        let impact = if w.abs() < 0.001 {
            I18n::get(lang, "impact_neutral")
        } else if w > 0.0 {
            I18n::get(lang, "impact_boost")
        } else {
            I18n::get(lang, "impact_reduce")
        };
        println!("  - {:<25}: {:>8.4} [{}]", feature_names[i], w, impact);
    }
    println!(
        "  - {:<25}: {:>8.4} {}",
        I18n::get(lang, "lbl_base_bias"),
        rl_b,
        I18n::get(lang, "impact_base")
    );
}

fn demo_mmap_tensor() {
    println!("\n[System] Demonstrating High-Performance Tensor I/O (Mmap)...");
    let shape = vec![1000, 1000]; // 1M elements, ~8MB
    let t = AutoTensor::rand(shape.clone(), 0.0, 1.0, 12345);
    let path = "temp_tensor.bin";

    let start = Instant::now();
    if t.save_binary(path).is_ok() {
        println!("Saved tensor (1M floats) in {:.2?}", start.elapsed());

        let start_load = Instant::now();
        match AutoTensor::from_mmap(path, shape) {
            Ok(t_loaded) => {
                println!("Loaded tensor via Mmap in {:.2?}", start_load.elapsed());
                println!("Verification: Shape={:?}", t_loaded.shape);
            }
            Err(e) => println!("Mmap failed: {}", e),
        }
    }
    // Cleanup
    let _ = std::fs::remove_file(path);
}

fn benchmark_simulation(
    rng: &mut Rng,
    neural_opt: &NeuralLuckOptimizer,
    dqn_policy: Option<&DuelingQNetwork>,
    ppo_policy: Option<&ActorCritic>,
    env_net: &EnvNet,
    config: &Config,
    lang: Language,
) {
    let ctx = SimModelContext {
        neural_opt,
        dqn_policy,
        ppo_policy,
        env_net,
        config,
        exp_sender: None,
        neural_sender: None,
        ppo_sender: None,
    };
    let fast_sims = 500usize;
    let fast_pulls = 100usize;
    let pb_fast = utils::create_bar(fast_sims as u64, "Benchmark (fast)");
    let start_fast = Instant::now();
    for _ in 0..fast_sims {
        let _ = simulate_fast(fast_pulls, rng, 0, &ctx);
        pb_fast.inc(1);
    }
    let fast_elapsed = start_fast.elapsed();
    pb_fast.finish_and_clear();
    println!(
        "{}",
        I18n::get(lang, "bench_fast")
            .replacen("{}", &fast_sims.to_string(), 1)
            .replacen("{}", &fast_pulls.to_string(), 1)
            .replace("{:.2?}", &format!("{:.2?}", fast_elapsed))
            .replace(
                "{:.0}",
                &format!("{:.0}", fast_sims as f64 / fast_elapsed.as_secs_f64())
            )
    );

    let one_sims = 100usize;
    let one_pulls = 100usize;
    let pb_one = utils::create_bar(one_sims as u64, "Benchmark (detailed)");
    let start_one = Instant::now();
    for _ in 0..one_sims {
        let _ = simulate_one(one_pulls, rng, 0, &ctx);
        pb_one.inc(1);
    }
    let one_elapsed = start_one.elapsed();
    pb_one.finish_and_clear();
    println!(
        "{}",
        I18n::get(lang, "bench_one")
            .replacen("{}", &one_sims.to_string(), 1)
            .replacen("{}", &one_pulls.to_string(), 1)
            .replace("{:.2?}", &format!("{:.2?}", one_elapsed))
            .replace(
                "{:.0}",
                &format!("{:.0}", one_sims as f64 / one_elapsed.as_secs_f64())
            )
    );
}

fn initialize_system(
    args: &Args,
) -> (
    Config,
    EnvNet,
    NeuralLuckOptimizer,
    DuelingQNetwork,
    ActorCritic,
    GoodJobWorker,
    Rng,
) {
    let mut config = Config::load(&args.config);
    apply_compute_device_policy(&mut config);
    if config.model_hidden_dim >= 8192 {
        warn!(
            "Large model detected ({} dim x {} layers). Training will take significantly longer and may require substantial memory.",
            config.model_hidden_dim, config.model_num_layers
        );
    }
    let mut rng = if let Some(seed) = args.seed {
        Rng::from_seed(seed)
    } else {
        Rng::new()
    };

    let worker = match GoodJobWorker::new_with_config(&config) {
        Ok(w) => w,
        Err(e) => {
            log::error!(
                "Worker initialization failed: {}. Running without worker pool.",
                e
            );
            // Create a minimal fallback worker with a single thread.
            // This preserves all functionality but may be slower.
            match GoodJobWorker::new(1) {
                Ok(w) => w,
                Err(_) => {
                    log::error!("Fallback worker also failed. Exiting.");
                    std::process::exit(1);
                }
            }
        }
    };

    let env_net = if !args.force {
        if let Some(cached) = load_env_net_cache("env_net.cache") {
            info!("[EnvNet] Cache loaded.");
            cached
        } else {
            info!("[EnvNet] Pre-training environment noise model...");
            let mut env_net = EnvNet::new(&mut rng);
            let (count, epochs) = if config.fast_init {
                (256, 10)
            } else {
                (1024, 50)
            };
            env_net.pretrain(&mut rng, &config, count, epochs);
            if save_env_net_cache("env_net.cache", &env_net) {
                info!("[EnvNet] Cache saved.");
            }
            env_net
        }
    } else {
        info!("[EnvNet] Force pre-training...");
        let mut env_net = EnvNet::new(&mut rng);
        env_net.pretrain(&mut rng, &config, 1024, 50);
        env_net
    };

    let mut trained_neural_opt = if !args.force {
        if let Some(cached) = load_neural_cache(NEURAL_CACHE_PATH) {
            info!("[Neural Core] Cache detected. Cached weights loaded.");
            cached
        } else {
            info!("[Neural Core] Cache not found. Training new weights...");
            train_neural_optimizer(rng.next_u64(), &env_net, &config, &worker)
        }
    } else {
        info!("[Neural Core] Force training new weights...");
        train_neural_optimizer(rng.next_u64(), &env_net, &config, &worker)
    };

    info!("[Linear] Training linear regression...");
    let (lin_w, lin_b) = train_linear_regression(&trained_neural_opt, &mut rng, &env_net, &config);
    trained_neural_opt.set_linear_params(lin_w, lin_b);

    info!("[RL] Manifold Optimization (Parallel)...");
    trained_neural_opt =
        train_manifold_rl(&trained_neural_opt, &mut rng, &env_net, &config, &worker);

    // Save Neural Cache
    if save_neural_cache(NEURAL_CACHE_PATH, &trained_neural_opt) {
        info!("[Neural Core] Cache saved.");
    }

    // DQN
    let dqn_policy = if config.online_train && config.online_train_dqn {
        DuelingQNetwork::new_with_config(&config, rng.next_u64())
    } else if !args.force {
        if let Some(cached) = load_model::<DuelingQNetwork>("dqn.cache", "DQN") {
            cached.freeze_achf_for_inference();
            info!("[DQN] Cached model loaded.");
            cached
        } else {
            info!("[DQN] Training new model...");
            let d = train_dqn(&trained_neural_opt, &mut rng, &env_net, &config);
            save_model(&d, "dqn.cache", "DQN");
            d
        }
    } else {
        info!("[DQN] Force training new model...");
        let d = train_dqn(&trained_neural_opt, &mut rng, &env_net, &config);
        save_model(&d, "dqn.cache", "DQN");
        d
    };

    // PPO
    let ppo_policy = if !args.force {
        if let Some(cached) = load_model::<ActorCritic>("ppo.cache", "PPO") {
            cached.freeze_achf_for_inference();
            info!("[PPO] Cached model loaded.");
            cached
        } else {
            info!("[PPO] Training new model...");
            let p = train_ppo(&mut rng, &env_net, &config);
            println!("[PPO] Saving model...");
            save_model(&p, "ppo.cache", "PPO");
            p
        }
    } else {
        info!("[PPO] Force training new model...");
        let p = train_ppo(&mut rng, &env_net, &config);
        println!("[PPO] Saving model...");
        save_model(&p, "ppo.cache", "PPO");
        p
    };

    (
        config,
        env_net,
        trained_neural_opt,
        dqn_policy,
        ppo_policy,
        worker,
        rng,
    )
}

fn apply_compute_device_policy(config: &mut Config) {
    match config.device {
        ComputeDevice::Cpu => {
            info!("[Device] Using CPU backend.");
        }
        ComputeDevice::Auto => {
            #[cfg(cuda)]
            {
                match cuda::device_count() {
                    Ok(count) if count > 0 => {
                        if let Ok(dev) = cuda::get_device_info(0) {
                            info!(
                                "[Device] Auto-selected CUDA: {} (CC {}.{})",
                                dev.name, dev.compute_capability.0, dev.compute_capability.1
                            );
                        } else {
                            info!("[Device] Auto-selected CUDA.");
                        }
                        config.device = ComputeDevice::Cuda;
                    }
                    Ok(_) => {
                        info!("[Device] Auto requested, but no CUDA devices found. Falling back to CPU.");
                        config.device = ComputeDevice::Cpu;
                    }
                    Err(err) => {
                        info!(
                            "[Device] Auto requested, CUDA unavailable ({}). Falling back to CPU.",
                            err
                        );
                        config.device = ComputeDevice::Cpu;
                    }
                }
            }
            #[cfg(not(cuda))]
            {
                info!("[Device] Auto requested, but binary was built without CUDA. Using CPU.");
                config.device = ComputeDevice::Cpu;
            }
        }
        ComputeDevice::Cuda => {
            #[cfg(cuda)]
            {
                match cuda::device_count() {
                    Ok(count) if count > 0 => {
                        if let Ok(dev) = cuda::get_device_info(0) {
                            info!(
                                "[Device] CUDA requested: {} (CC {}.{})",
                                dev.name, dev.compute_capability.0, dev.compute_capability.1
                            );
                        } else {
                            info!("[Device] CUDA requested and initialized.");
                        }
                    }
                    Ok(_) => {
                        info!("[Device] CUDA requested, but no CUDA devices found. Falling back to CPU.");
                        config.device = ComputeDevice::Cpu;
                    }
                    Err(err) => {
                        info!(
                            "[Device] CUDA requested, but unavailable ({}). Falling back to CPU.",
                            err
                        );
                        config.device = ComputeDevice::Cpu;
                    }
                }
            }
            #[cfg(not(cuda))]
            {
                info!("[Device] CUDA requested, but binary was built without CUDA. Using CPU.");
                config.device = ComputeDevice::Cpu;
            }
        }
    }
}

fn main() {
    panic_guard::install();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    if matches!(
        args.command,
        Some(Commands::Collect { .. }) | Some(Commands::Train)
    ) {
        let config = Config::load(&args.config);
        let lang = Language::from_config(&config);
        match args.command.clone().unwrap() {
            Commands::Collect { action } => {
                let mut db = PlayerDatabase::load(&config.player_data_path);
                match action {
                    CollectAction::Add => {
                        if let Some(session) = add_session_interactive(&config, lang) {
                            db.add_session(session);
                            if db.save(&config.player_data_path) {
                                println!("{}", I18n::get(lang, "data_saved"));
                            }
                        }
                    }
                    CollectAction::Import { file } => match import_from_json(&file) {
                        Ok(sessions) => {
                            let count = sessions.len();
                            for s in sessions {
                                db.add_session(s);
                            }
                            if db.save(&config.player_data_path) {
                                println!(
                                    "{} {} {}",
                                    I18n::get(lang, "imported"),
                                    count,
                                    I18n::get(lang, "unit_sessions")
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("\x1b[1;31m[Error]\x1b[0m {}", e);
                        }
                    },
                    CollectAction::Stats => {
                        print_stats(&db, &config, lang);
                    }
                }
            }
            Commands::Train => {
                let db = PlayerDatabase::load(&config.player_data_path);
                if db.sessions.is_empty() {
                    println!("{}", I18n::get(lang, "cal_no_player_data"));
                } else {
                    let cal = run_calibration(&db, &config, lang);
                    if cal.save(&config.calibrated_path) {
                        println!("{}", I18n::get(lang, "cal_params_saved"));
                    }
                }
            }
            _ => unreachable!(),
        }
        return;
    }

    let (mut config, env_net, trained_neural_opt, dqn_policy, ppo_policy, worker, mut rng) =
        initialize_system(&args);
    let lang = Language::from_config(&config);

    // Auto-load calibrated parameters if available
    let calibration = if config.use_calibrated {
        let cal = CalibrationData::load(&config.calibrated_path);
        if !cal.pools.is_empty() {
            apply_calibration(&mut config, &cal);
            info!("[Calibration] Loaded calibrated parameters.");
            Some(cal)
        } else {
            None
        }
    } else {
        None
    };

    match args.command.clone().unwrap_or(Commands::Interactive) {
        Commands::Interactive => {
            run_interactive(RunInteractiveArgs {
                config,
                env_net,
                trained_neural_opt,
                dqn_policy,
                ppo_policy,
                worker,
                rng,
                lang,
                calibration,
            });
        }
        Commands::Simulate { count, pulls } => {
            // Run simulation
            let ctx = SimRunContext {
                neural_opt: &trained_neural_opt,
                dqn_policy: Some(&dqn_policy),
                ppo_policy: Some(&ppo_policy),
                env_net: &env_net,
                config: &config,
                worker: &worker,
                exp_sender: None,
                neural_sender: None,
                ppo_sender: None,
            };
            let (six_count, up_count, _, _) = simulate_stats(pulls, count, rng.next_u64(), 0, &ctx);
            println!(
                "{}",
                I18n::get(lang, "batch_sim_header")
                    .replacen("{}", &count.to_string(), 1)
                    .replacen("{}", &pulls.to_string(), 1)
            );
            println!(
                "{}",
                I18n::get(lang, "avg_6_star").replace(
                    "{:.4}",
                    &format!(
                        "{:.4}",
                        if count > 0 {
                            six_count as f64 / count as f64
                        } else {
                            0.0
                        }
                    )
                )
            );
            println!(
                "{}",
                I18n::get(lang, "avg_up").replace(
                    "{:.4}",
                    &format!(
                        "{:.4}",
                        if count > 0 {
                            up_count as f64 / count as f64
                        } else {
                            0.0
                        }
                    )
                )
            );
        }
        Commands::Benchmark { action } => match action {
            Some(BenchAction::Achf {
                only,
                output_dir,
                format,
                trials,
            }) => {
                let chart_fmt = match format.as_str() {
                    "png" => chart::ChartFormat::Png,
                    _ => chart::ChartFormat::Svg,
                };
                let bench_cfg = bench::BenchConfig {
                    output_dir,
                    format: chart_fmt,
                    only: only.map(|s| s.split(',').map(|x| x.trim().to_string()).collect()),
                    num_trials: trials.max(1),
                };
                let seed = args.seed.unwrap_or(42);
                bench::run_achf_benchmarks(&config, seed, &bench_cfg);
            }
            None => {
                benchmark_simulation(
                    &mut rng,
                    &trained_neural_opt,
                    Some(&dqn_policy),
                    Some(&ppo_policy),
                    &env_net,
                    &config,
                    lang,
                );
                demo_mmap_tensor();
            }
        },
        Commands::F2p => {
            let f2p_ctx = F2pAnalysisCtx {
                config: &config,
                neural_opt: &trained_neural_opt,
                dqn_policy: Some(&dqn_policy),
                ppo_policy: Some(&ppo_policy),
                env_net: &env_net,
                worker: &worker,
                lang,
            };
            run_f2p_analysis(&f2p_ctx, &mut rng);
        }
        Commands::Collect { .. } | Commands::Train => unreachable!(),
    }
}

struct RunInteractiveArgs {
    config: Config,
    env_net: EnvNet,
    trained_neural_opt: NeuralLuckOptimizer,
    dqn_policy: DuelingQNetwork,
    ppo_policy: ActorCritic,
    worker: GoodJobWorker,
    rng: Rng,
    lang: Language,
    calibration: Option<CalibrationData>,
}

fn run_interactive(args: RunInteractiveArgs) {
    let RunInteractiveArgs {
        mut config,
        env_net,
        trained_neural_opt,
        dqn_policy,
        ppo_policy,
        worker,
        mut rng,
        lang,
        calibration,
    } = args;
    let dqn_shared = Arc::new(RwLock::new(dqn_policy.clone()));
    let neural_shared = Arc::new(RwLock::new(trained_neural_opt.clone()));
    let ppo_shared = Arc::new(RwLock::new(ppo_policy.clone()));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut online_handles: Vec<thread::JoinHandle<()>> = Vec::new();
    let mut dqn_sender: Option<mpsc::Sender<Experience>> = None;
    let mut neural_sender: Option<mpsc::Sender<NeuralSample>> = None;
    let mut ppo_sender: Option<mpsc::Sender<PpoExperience>> = None;

    if config.online_train && config.online_train_dqn && config.luck_mode == LuckMode::Dqn {
        let (tx, rx) = mpsc::channel::<Experience>();
        dqn_sender = Some(tx);
        let shared = Arc::clone(&dqn_shared);
        let stop = Arc::clone(&stop_flag);
        let interval_ms = config.train_interval_ms.max(1) as u64;
        let max_steps = config.max_train_steps_per_tick;
        let trainer_seed = rng.next_u64();
        let mut trainer = OnlineDqnTrainer::from_policy(dqn_policy, trainer_seed);
        online_handles.push(thread::spawn(move || {
            let mut local_rng = Rng::from_seed(trainer_seed.wrapping_add(1));
            let mut last_report = Instant::now();
            let mut last_sync = Instant::now();
            let max_drain = MAX_DRAIN_PER_TICK;
            loop {
                if stop.load(Ordering::Relaxed) || max_steps == 0 {
                    break;
                }
                let mut drained = 0usize;
                while let Ok(exp) = rx.try_recv() {
                    trainer.push(exp);
                    drained += 1;
                    if drained >= max_drain {
                        break;
                    }
                }
                let mut steps = 0usize;
                while steps < max_steps {
                    if trainer.train_step(&mut local_rng) {
                        steps += 1;
                    } else {
                        break;
                    }
                }
                if steps > 0 && last_sync.elapsed() >= Duration::from_millis(interval_ms) {
                    trainer.sync_to(&shared);
                    last_sync = Instant::now();
                    if last_report.elapsed().as_secs_f64() >= ONLINE_REPORT_INTERVAL_SECS {
                        info!(
                            "[Online DQN] steps={} buffer={}",
                            trainer.steps_done(),
                            trainer.buffer_len()
                        );
                        last_report = Instant::now();
                    }
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }
        }));
    }

    if config.online_train && config.online_train_neural {
        let (tx, rx) = mpsc::channel::<NeuralSample>();
        neural_sender = Some(tx);
        let shared = Arc::clone(&neural_shared);
        let stop = Arc::clone(&stop_flag);
        let interval_ms = (config.train_interval_ms.max(1) as u64).max(5);
        let max_steps = config.max_train_steps_per_tick;
        let mut trainer = OnlineNeuralTrainer::from_model(trained_neural_opt.clone());
        online_handles.push(thread::spawn(move || {
            let mut last_report = Instant::now();
            let mut last_sync = Instant::now();
            let max_drain = max_steps.max(1).clamp(1, MAX_DRAIN_PER_TICK);
            loop {
                if stop.load(Ordering::Relaxed) || max_steps == 0 {
                    break;
                }
                let mut drained = 0usize;
                while let Ok(sample) = rx.try_recv() {
                    if drained >= max_drain {
                        break;
                    }
                    let _ = trainer.train_step(&sample);
                    drained += 1;
                }
                if trainer.steps_done() > 0
                    && last_sync.elapsed() >= Duration::from_millis(interval_ms)
                {
                    trainer.sync_to(&shared);
                    last_sync = Instant::now();
                    if last_report.elapsed().as_secs_f64() >= ONLINE_REPORT_INTERVAL_SECS {
                        info!("[Online Neural] steps={}", trainer.steps_done());
                        last_report = Instant::now();
                    }
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }
        }));
    }

    if config.online_train && config.online_train_ppo && config.luck_mode == LuckMode::Ppo {
        let (tx, rx) = mpsc::channel::<PpoExperience>();
        ppo_sender = Some(tx);
        let shared = Arc::clone(&ppo_shared);
        let stop = Arc::clone(&stop_flag);
        let interval_ms = (config.train_interval_ms.max(1) as u64).max(5);
        let max_steps = config.max_train_steps_per_tick;
        let (k_epochs, batch_size) = resolve_ppo_online_train_params(&config);
        let mut trainer = OnlinePpoTrainer::from_policy(ppo_policy, k_epochs, batch_size);
        online_handles.push(thread::spawn(move || {
            let mut last_report = Instant::now();
            let mut last_sync = Instant::now();
            let max_drain = MAX_DRAIN_PER_TICK;
            loop {
                if stop.load(Ordering::Relaxed) || max_steps == 0 {
                    break;
                }
                let mut drained = 0usize;
                while let Ok(exp) = rx.try_recv() {
                    trainer.push(exp);
                    drained += 1;
                    if drained >= max_drain {
                        break;
                    }
                }
                let mut steps = 0usize;
                while steps < max_steps {
                    if trainer.train_step(PPO_ONLINE_LR) {
                        steps += 1;
                    } else {
                        break;
                    }
                }
                if steps > 0 && last_sync.elapsed() >= Duration::from_millis(interval_ms) {
                    trainer.sync_to(&shared);
                    last_sync = Instant::now();
                    if last_report.elapsed().as_secs_f64() >= ONLINE_REPORT_INTERVAL_SECS {
                        info!(
                            "[Online PPO] steps={} buffer={}",
                            trainer.steps_done(),
                            trainer.buffer_len()
                        );
                        last_report = Instant::now();
                    }
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }
        }));
    }

    // === EXPLAINABILITY REPORT ===
    print_explainability_report(&trained_neural_opt, lang);

    let pool_type_label = |pool_type: &str, lang: Language| -> String {
        match pool_type {
            "character_up" => I18n::get(lang, "pool_type_character_up"),
            "weapon_up" => I18n::get(lang, "pool_type_weapon_up"),
            "standard" => I18n::get(lang, "pool_type_standard"),
            "beginner" => I18n::get(lang, "pool_type_beginner"),
            "permanent" => I18n::get(lang, "pool_type_permanent"),
            _ => I18n::get(lang, "pool_type_unknown"),
        }
    };
    let print_pool_header = |config: &Config, lang: Language| {
        let up_label = if config.up_six.is_empty() {
            I18n::get(lang, "label_none")
        } else {
            config.up_six.join(", ")
        };
        let pool_type = config
            .active_pool
            .as_ref()
            .and_then(|id| config.pools.iter().find(|p| &p.id == id))
            .map(|p| pool_type_label(&p.pool_type, lang))
            .unwrap_or_else(|| pool_type_label("unknown", lang));
        let five_star_rule = if config.always_5_star {
            I18n::get(lang, "rule_5star_every")
        } else if config.five_star_pity > 0 {
            I18n::get(lang, "rule_5star_pity").replace("{}", &config.five_star_pity.to_string())
        } else {
            I18n::get(lang, "rule_5star_off")
        };
        let big_pity_requires_not_up = if config.big_pity_requires_not_up {
            I18n::get(lang, "label_yes")
        } else {
            I18n::get(lang, "label_no")
        };
        println!("{}", I18n::get(lang, "header_title"));
        println!(
            "{}",
            I18n::get(lang, "header_pool").replace("{}", &config.pool_name)
        );
        println!(
            "{}",
            I18n::get(lang, "header_pool_type").replace("{}", &pool_type)
        );
        println!("{}", I18n::get(lang, "header_up").replace("{}", &up_label));
        println!(
            "{}",
            I18n::get(lang, "header_prob")
                .replace("{:.1}", &format!("{:.1}", config.prob_6_base * 100.0))
                .replace("{}", &config.soft_pity_start.to_string())
        );
        println!(
            "{}",
            I18n::get(lang, "header_rules")
                .replace("{}", &config.small_pity_guarantee.to_string())
                .replace("{}", &format!("{:.0}", config.up_rate * 100.0))
        );
        println!(
            "{}",
            if config.up_pity_soft > 0 {
                I18n::get(lang, "header_up_pity").replace("{}", &config.up_pity_soft.to_string())
            } else {
                I18n::get(lang, "header_up_pity_off")
            }
        );
        println!(
            "{}",
            I18n::get(lang, "header_five_star_rule").replace("{}", &five_star_rule)
        );
        println!(
            "{}",
            if config.big_pity_cumulative > 0 {
                I18n::get(lang, "header_big_pity_on")
                    .replace("{}", &config.big_pity_cumulative.to_string())
                    .replace("{}", &big_pity_requires_not_up)
            } else {
                I18n::get(lang, "header_big_pity_off")
            }
        );
        println!(
            "{}",
            I18n::get(lang, "header_economy")
                .replacen("{}", &COST_PER_PULL.to_string(), 1)
                .replacen("{}", &FREE_PULLS_WELFARE.to_string(), 1)
        );
        println!("{}", I18n::get(lang, "header_neural"));
    };

    // === Ask User for Interaction Mode ===
    let mut use_ppo = prompt_yes_no(&I18n::get(lang, "prompt_ppo"), true);
    let mut default_pulls = 10usize;
    let mut default_sims = 1usize;
    let mut use_welfare_default = true;
    let mut selected_pool_ids: Vec<String> = if let Some(active) = config.active_pool.clone() {
        vec![active]
    } else if !config.pools.is_empty() {
        vec![config.pools[0].id.clone()]
    } else {
        vec![]
    };

    if !config.pools.is_empty() {
        println!("{}", I18n::get(lang, "init_pool_list"));
        for (idx, pool) in config.pools.iter().enumerate() {
            let archived_tag = if pool.is_archived {
                I18n::get(lang, "pool_archived_tag")
            } else {
                String::new()
            };
            let line = I18n::get(lang, "init_pool_item")
                .replacen("{}", &(idx + 1).to_string(), 1)
                .replacen("{}", &format!("{}{}", pool.name, archived_tag), 1)
                .replacen("{}", &pool_type_label(&pool.pool_type, lang), 1);
            println!("{}", line);
        }
        let default_index = default_pool_index(&config);
        print!(
            "{}",
            I18n::get(lang, "prompt_pool_select").replace("{}", &default_index.to_string())
        );
        let _ = io::stdout().flush();
        let mut pool_input = String::new();
        let _ = io::stdin().read_line(&mut pool_input);
        let pool_input = pool_input.trim();
        if pool_input.eq_ignore_ascii_case("all") {
            let all_ids: Vec<String> = config.pools.iter().map(|p| p.id.clone()).collect();
            if !all_ids.is_empty() && config.apply_pool(&all_ids[0]) {
                if let Some(cal) = &calibration {
                    apply_calibration(&mut config, cal);
                }
                selected_pool_ids = all_ids;
            }
        } else {
            let mut indices = Vec::new();
            if pool_input.is_empty() {
                indices.push(default_index);
            } else {
                for token in pool_input
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                {
                    if let Ok(idx) = token.parse::<usize>() {
                        if idx >= 1 && idx <= config.pools.len() {
                            indices.push(idx);
                        }
                    }
                }
            }
            indices.sort_unstable();
            indices.dedup();
            if !indices.is_empty() {
                let mut ids = Vec::new();
                for idx in indices {
                    if let Some(pool) = config.pools.get(idx - 1) {
                        ids.push(pool.id.clone());
                    }
                }
                if !ids.is_empty() && config.apply_pool(&ids[0]) {
                    if let Some(cal) = &calibration {
                        apply_calibration(&mut config, cal);
                    }
                    selected_pool_ids = ids;
                }
            }
        }
    }

    print_pool_header(&config, lang);

    println!("\n{}", I18n::get(lang, "sys_prng"));
    if cfg!(debug_assertions) && !config.fast_init {
        println!("\n{}", I18n::get(lang, "sys_bench"));
        let dqn_guard = panic_guard::read_shared(&dqn_shared);
        let neural_guard = panic_guard::read_shared(&neural_shared);
        let ppo_guard = panic_guard::read_shared(&ppo_shared);
        benchmark_simulation(
            &mut rng,
            &neural_guard,
            Some(&dqn_guard),
            Some(&ppo_guard),
            &env_net,
            &config,
            lang,
        );
        demo_mmap_tensor();
    }

    // F2P Analysis
    {
        let dqn_guard = panic_guard::read_shared(&dqn_shared);
        let neural_guard = panic_guard::read_shared(&neural_shared);
        let ppo_guard = panic_guard::read_shared(&ppo_shared);
        let f2p_ctx = F2pAnalysisCtx {
            config: &config,
            neural_opt: &neural_guard,
            dqn_policy: Some(&dqn_guard),
            ppo_policy: Some(&ppo_guard),
            env_net: &env_net,
            worker: &worker,
            lang,
        };
        run_f2p_analysis(&f2p_ctx, &mut rng);
    }
    println!("{}", I18n::get(lang, "total_value"));
    println!("{}", I18n::get(lang, "tui_quick_run_tip"));

    let mut history: std::collections::VecDeque<SimHistoryEntry> =
        std::collections::VecDeque::with_capacity(SIM_HISTORY_CAPACITY);
    let parse_quick_run = |s: &str| -> Option<(usize, usize)> {
        for sep in ['x', 'X', '*'] {
            if let Some((left, right)) = s.split_once(sep) {
                let pulls = left.trim().parse::<usize>().ok()?;
                let sims = right.trim().parse::<usize>().ok()?;
                if pulls > 0 && sims > 0 {
                    return Some((pulls, sims));
                }
            }
        }
        None
    };
    let resolve_pool_token = |token: &str, cfg: &Config| -> Option<String> {
        let t = token.trim();
        if t.is_empty() {
            return None;
        }
        if let Ok(idx) = t.parse::<usize>() {
            if idx >= 1 && idx <= cfg.pools.len() {
                return cfg.pools.get(idx - 1).map(|p| p.id.clone());
            }
        }
        let t_lower = t.to_lowercase();
        cfg.pools
            .iter()
            .find(|p| p.id.eq_ignore_ascii_case(t) || p.name.to_lowercase() == t_lower)
            .map(|p| p.id.clone())
    };

    loop {
        let welfare_label = if use_welfare_default {
            I18n::get(lang, "label_on")
        } else {
            I18n::get(lang, "label_off")
        };
        print!(
            "{}",
            I18n::get(lang, "prompt_pulls_status")
                .replacen("{}", &default_pulls.to_string(), 1)
                .replacen("{}", &default_sims.to_string(), 1)
                .replacen("{}", &welfare_label, 1)
        );
        let _ = io::stdout().flush();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) | Err(_) => break,
            _ => {}
        }
        let input = input.trim();

        if input.eq_ignore_ascii_case("q") {
            println!("{}", I18n::get(lang, "exit_msg"));
            break;
        }

        let mut parts = input.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        let cmd_lower = cmd.to_lowercase();
        if cmd_lower == "h" || cmd_lower == "help" || cmd_lower == "?" {
            println!("{}", I18n::get(lang, "cmd_help"));
            continue;
        }
        if cmd_lower == "ppo" {
            use_ppo = !use_ppo;
            let key = if use_ppo { "cmd_ppo_on" } else { "cmd_ppo_off" };
            println!("{}", I18n::get(lang, key));
            continue;
        }
        if cmd_lower == "w" || cmd_lower == "welfare" {
            use_welfare_default = !use_welfare_default;
            let key = if use_welfare_default {
                "cmd_welfare_on"
            } else {
                "cmd_welfare_off"
            };
            println!("{}", I18n::get(lang, key));
            continue;
        }
        if cmd_lower == "status" || cmd_lower == "st" {
            println!("{}", I18n::get(lang, "cmd_status_header"));
            println!(
                "{}",
                I18n::get(lang, "cmd_status_pool").replace("{}", &config.pool_name)
            );
            println!(
                "{}",
                I18n::get(lang, "cmd_status_pulls").replace("{}", &default_pulls.to_string())
            );
            println!(
                "{}",
                I18n::get(lang, "cmd_status_sims").replace("{}", &default_sims.to_string())
            );
            let w_label = if use_welfare_default {
                I18n::get(lang, "label_on")
            } else {
                I18n::get(lang, "label_off")
            };
            println!(
                "{}",
                I18n::get(lang, "cmd_status_welfare").replace("{}", &w_label)
            );
            let ppo_label = if use_ppo {
                I18n::get(lang, "label_on")
            } else {
                I18n::get(lang, "label_off")
            };
            println!(
                "{}",
                I18n::get(lang, "cmd_status_ppo").replace("{}", &ppo_label)
            );
            println!("{}", I18n::get(lang, "cmd_status_footer"));
            continue;
        }
        if cmd_lower == "info" {
            print_pool_header(&config, lang);
            continue;
        }
        if cmd_lower == "history" || cmd_lower == "hi" {
            println!("{}", I18n::get(lang, "cmd_history_header"));
            if history.is_empty() {
                println!("{}", I18n::get(lang, "cmd_history_empty"));
            } else {
                let mut table = utils::Table::new(&[
                    &I18n::get(lang, "tbl_hist_id"),
                    &I18n::get(lang, "tbl_hist_pool"),
                    &I18n::get(lang, "tbl_hist_pulls"),
                    &I18n::get(lang, "tbl_hist_sims"),
                    &I18n::get(lang, "tbl_hist_avg6"),
                    &I18n::get(lang, "tbl_hist_avgup"),
                    &I18n::get(lang, "tbl_hist_time"),
                ]);
                for (i, entry) in history.iter().enumerate() {
                    table.add_row(vec![
                        (i + 1).to_string(),
                        entry.pool_name.clone(),
                        entry.pulls.to_string(),
                        entry.sims.to_string(),
                        format!("{:.3}", entry.avg_six),
                        format!("{:.3}", entry.avg_up),
                        format!("{}ms", entry.elapsed_ms),
                    ]);
                }
                println!("{}", table.render());
            }
            continue;
        }
        if cmd_lower == "bench" || cmd_lower == "benchmark" {
            let sub = parts.next().unwrap_or("quick");
            let sub_lower = sub.to_lowercase();
            match sub_lower.as_str() {
                "quick" | "q" => {
                    println!("{}", I18n::get(lang, "bench_quick_start"));
                    let dqn_guard = panic_guard::read_shared(&dqn_shared);
                    let neural_guard = panic_guard::read_shared(&neural_shared);
                    let ppo_guard = panic_guard::read_shared(&ppo_shared);
                    benchmark_simulation(
                        &mut rng,
                        &neural_guard,
                        Some(&dqn_guard),
                        Some(&ppo_guard),
                        &env_net,
                        &config,
                        lang,
                    );
                }
                "paper" | "p" => {
                    let mut only_filter: Option<Vec<String>> = None;
                    let mut trials = 3usize;
                    let mut format_str = "svg".to_string();
                    let mut output_dir = "bench_output".to_string();

                    // Parse remaining args: --only X --trials N --format F --output-dir D
                    while let Some(arg) = parts.next() {
                        match arg {
                            "--only" | "-o" => {
                                if let Some(val) = parts.next() {
                                    only_filter = Some(
                                        val.split(',').map(|s| s.trim().to_string()).collect(),
                                    );
                                }
                            }
                            "--trials" | "-t" => {
                                if let Some(val) = parts.next() {
                                    trials = val.parse().unwrap_or(3).max(1);
                                }
                            }
                            "--format" | "-f" => {
                                if let Some(val) = parts.next() {
                                    format_str = val.to_string();
                                }
                            }
                            "--output-dir" | "-d" => {
                                if let Some(val) = parts.next() {
                                    output_dir = val.to_string();
                                }
                            }
                            _ => {}
                        }
                    }
                    let chart_fmt = match format_str.as_str() {
                        "png" => chart::ChartFormat::Png,
                        _ => chart::ChartFormat::Svg,
                    };
                    let bench_cfg = bench::BenchConfig {
                        output_dir,
                        format: chart_fmt,
                        only: only_filter,
                        num_trials: trials,
                    };
                    let seed = rng.next_u64();
                    bench::run_achf_benchmarks(&config, seed, &bench_cfg);
                }
                "list" | "l" => {
                    println!("{}", I18n::get(lang, "bench_list_header"));
                    println!("  {}", I18n::get(lang, "bench_list_ablation"));
                    println!("  {}", I18n::get(lang, "bench_list_mode"));
                    println!("  {}", I18n::get(lang, "bench_list_path"));
                    println!("  {}", I18n::get(lang, "bench_list_gate"));
                    println!("  {}", I18n::get(lang, "bench_list_scale"));
                    println!("  {}", I18n::get(lang, "bench_list_apply"));
                    println!("  {}", I18n::get(lang, "bench_list_convergence"));
                    println!("{}", I18n::get(lang, "bench_usage_header"));
                    println!("{}", I18n::get(lang, "bench_usage_quick"));
                    println!("{}", I18n::get(lang, "bench_usage_paper"));
                    println!("{}", I18n::get(lang, "bench_usage_only"));
                    println!("{}", I18n::get(lang, "bench_usage_trials"));
                    println!("{}", I18n::get(lang, "bench_usage_output"));
                }
                _ => {
                    // Treat as a shortcut: bench <experiment_name>
                    let chart_fmt = chart::ChartFormat::Svg;
                    let bench_cfg = bench::BenchConfig {
                        output_dir: "bench_output".to_string(),
                        format: chart_fmt,
                        only: Some(vec![sub_lower]),
                        num_trials: 3,
                    };
                    let seed = rng.next_u64();
                    bench::run_achf_benchmarks(&config, seed, &bench_cfg);
                }
            }
            continue;
        }
        if cmd_lower == "pool" {
            let sub = parts.next().unwrap_or("list");
            if sub.eq_ignore_ascii_case("list") {
                if config.pools.is_empty() {
                    println!(
                        "{}",
                        I18n::get(lang, "cmd_pool_list")
                            .replace("{}", &I18n::get(lang, "label_none"))
                    );
                } else {
                    let mut table = utils::Table::new(&[
                        &I18n::get(lang, "tbl_pool_id"),
                        &I18n::get(lang, "tbl_pool_name"),
                        &I18n::get(lang, "tbl_pool_type"),
                        &I18n::get(lang, "tbl_pool_up"),
                        &I18n::get(lang, "tbl_pool_status"),
                    ]);
                    for (idx, pool) in config.pools.iter().enumerate() {
                        let up_label = if pool.up_six.is_empty() {
                            I18n::get(lang, "label_none")
                        } else {
                            pool.up_six.join(", ")
                        };
                        let status = if pool.is_archived {
                            I18n::get(lang, "pool_status_archived")
                        } else {
                            I18n::get(lang, "pool_status_active")
                        };
                        table.add_row(vec![
                            (idx + 1).to_string(),
                            pool.name.clone(),
                            pool_type_label(&pool.pool_type, lang),
                            up_label,
                            status,
                        ]);
                    }
                    println!("{}", table.render());
                }
            } else if sub.eq_ignore_ascii_case("multi") {
                let mut valid_ids = Vec::new();
                let mut list_tokens = Vec::new();
                if let Some(first) = parts.next() {
                    list_tokens.push(first.to_string());
                    list_tokens.extend(parts.map(|s| s.to_string()));
                }
                let joined = list_tokens.join(" ");
                for id in joined
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                {
                    if let Some(resolved) = resolve_pool_token(id, &config) {
                        valid_ids.push(resolved);
                    }
                }
                valid_ids.sort_unstable();
                valid_ids.dedup();
                if valid_ids.is_empty() {
                    println!("{}", I18n::get(lang, "cmd_pool_multi_empty"));
                } else {
                    let first = valid_ids[0].clone();
                    if config.apply_pool(&first) {
                        if let Some(cal) = &calibration {
                            apply_calibration(&mut config, cal);
                        }
                        selected_pool_ids = valid_ids;
                        println!(
                            "{}",
                            I18n::get(lang, "cmd_pool_multi_set")
                                .replace("{}", &selected_pool_ids.join(", "))
                        );
                        print_pool_header(&config, lang);
                    }
                }
            } else if sub.eq_ignore_ascii_case("all") {
                let all_ids: Vec<String> = config.pools.iter().map(|p| p.id.clone()).collect();
                if !all_ids.is_empty() {
                    let first = all_ids[0].clone();
                    if config.apply_pool(&first) {
                        if let Some(cal) = &calibration {
                            apply_calibration(&mut config, cal);
                        }
                        selected_pool_ids = all_ids;
                        println!("{}", I18n::get(lang, "cmd_pool_all_set"));
                        print_pool_header(&config, lang);
                    }
                } else {
                    println!("{}", I18n::get(lang, "cmd_pool_multi_empty"));
                }
            } else {
                let mut tokens = vec![sub.to_string()];
                tokens.extend(parts.map(|s| s.to_string()));
                let is_multi_like = tokens.len() > 1 || sub.contains(',');
                let mut resolved_ids = Vec::new();
                for token in tokens
                    .join(" ")
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                {
                    if let Some(id) = resolve_pool_token(token, &config) {
                        resolved_ids.push(id);
                    }
                }
                resolved_ids.sort_unstable();
                resolved_ids.dedup();
                if resolved_ids.is_empty() {
                    println!(
                        "{}",
                        I18n::get(lang, "cmd_pool_not_found").replace("{}", sub)
                    );
                } else if is_multi_like || resolved_ids.len() > 1 {
                    let first = resolved_ids[0].clone();
                    if config.apply_pool(&first) {
                        if let Some(cal) = &calibration {
                            apply_calibration(&mut config, cal);
                        }
                        selected_pool_ids = resolved_ids;
                        println!(
                            "{}",
                            I18n::get(lang, "cmd_pool_multi_set")
                                .replace("{}", &selected_pool_ids.join(", "))
                        );
                        print_pool_header(&config, lang);
                    }
                } else {
                    let selected_id = resolved_ids[0].clone();
                    if config.apply_pool(&selected_id) {
                        if let Some(cal) = &calibration {
                            apply_calibration(&mut config, cal);
                        }
                        selected_pool_ids = vec![selected_id];
                        println!(
                            "{}",
                            I18n::get(lang, "cmd_pool_switched").replace("{}", &config.pool_name)
                        );
                        print_pool_header(&config, lang);
                    }
                }
            }
            continue;
        }
        if cmd_lower == "p" || cmd_lower == "pulls" {
            if let Some(value) = parts.next() {
                match value.parse::<usize>() {
                    Ok(val) if val > 0 => {
                        default_pulls = val.min(INPUT_CAP);
                        if val > INPUT_CAP {
                            println!(
                                "{}",
                                I18n::get(lang, "input_capped")
                                    .replacen("{}", &val.to_string(), 1)
                                    .replacen("{}", &INPUT_CAP.to_string(), 1)
                            );
                        }
                        println!(
                            "{}",
                            I18n::get(lang, "cmd_set_default_pulls")
                                .replace("{}", &default_pulls.to_string())
                        );
                    }
                    _ => println!("{}", I18n::get(lang, "invalid_input")),
                }
            } else {
                println!("{}", I18n::get(lang, "cmd_invalid_command"));
            }
            continue;
        }
        if cmd_lower == "s" || cmd_lower == "sims" {
            if let Some(value) = parts.next() {
                match value.parse::<usize>() {
                    Ok(val) if val > 0 => {
                        default_sims = val.min(INPUT_CAP);
                        if val > INPUT_CAP {
                            println!(
                                "{}",
                                I18n::get(lang, "input_capped")
                                    .replacen("{}", &val.to_string(), 1)
                                    .replacen("{}", &INPUT_CAP.to_string(), 1)
                            );
                        }
                        println!(
                            "{}",
                            I18n::get(lang, "cmd_set_default_sims")
                                .replace("{}", &default_sims.to_string())
                        );
                    }
                    _ => println!("{}", I18n::get(lang, "invalid_input")),
                }
            } else {
                println!("{}", I18n::get(lang, "cmd_invalid_command"));
            }
            continue;
        }

        let mut sims_n = default_sims;
        let n = if input.is_empty() {
            default_pulls
        } else if let Some((pulls, sims)) = parse_quick_run(input) {
            let capped_pulls = pulls.min(INPUT_CAP);
            let capped_sims = sims.min(INPUT_CAP);
            if pulls > INPUT_CAP {
                println!(
                    "{}",
                    I18n::get(lang, "input_capped")
                        .replacen("{}", &pulls.to_string(), 1)
                        .replacen("{}", &INPUT_CAP.to_string(), 1)
                );
            }
            if sims > INPUT_CAP {
                println!(
                    "{}",
                    I18n::get(lang, "input_capped")
                        .replacen("{}", &sims.to_string(), 1)
                        .replacen("{}", &INPUT_CAP.to_string(), 1)
                );
            }
            sims_n = capped_sims;
            capped_pulls
        } else {
            match input.parse::<usize>() {
                Ok(val) => {
                    if val > INPUT_CAP {
                        println!(
                            "{}",
                            I18n::get(lang, "input_capped")
                                .replacen("{}", &val.to_string(), 1)
                                .replacen("{}", &INPUT_CAP.to_string(), 1)
                        );
                        INPUT_CAP
                    } else {
                        val
                    }
                }
                Err(_) => {
                    let known_commands: &[&str] = &[
                        "h",
                        "help",
                        "?",
                        "q",
                        "ppo",
                        "w",
                        "welfare",
                        "status",
                        "st",
                        "info",
                        "history",
                        "hi",
                        "pool",
                        "bench",
                        "benchmark",
                        "p",
                        "pulls",
                        "s",
                        "sims",
                    ];
                    if let Some(suggestion) = utils::suggest_command(&cmd_lower, known_commands, 2)
                    {
                        println!(
                            "{}",
                            I18n::get(lang, "cmd_suggest")
                                .replacen("{}", &cmd_lower, 1)
                                .replacen("{}", suggestion, 1)
                        );
                    } else {
                        println!("{}", I18n::get(lang, "cmd_invalid_command"));
                    }
                    continue;
                }
            }
        };

        let free_pulls = if use_welfare_default {
            FREE_PULLS_WELFARE
        } else {
            0
        };
        if sims_n > 1 {
            let dqn_guard = panic_guard::read_shared(&dqn_shared);
            let neural_guard = panic_guard::read_shared(&neural_shared);
            let ppo_guard = panic_guard::read_shared(&ppo_shared);
            let active_ppo = if use_ppo { Some(&*ppo_guard) } else { None };
            if selected_pool_ids.len() > 1 {
                for pool_id in selected_pool_ids.iter() {
                    let mut pool_config = config.clone();
                    if !pool_config.apply_pool(pool_id) {
                        continue;
                    }
                    if let Some(cal) = &calibration {
                        apply_calibration(&mut pool_config, cal);
                    }
                    println!(
                        "{}",
                        I18n::get(lang, "sim_pool_header").replace("{}", &pool_config.pool_name)
                    );
                    let pool_start = Instant::now();
                    let ctx = SimRunContext {
                        neural_opt: &neural_guard,
                        dqn_policy: Some(&dqn_guard),
                        ppo_policy: active_ppo,
                        env_net: &env_net,
                        config: &pool_config,
                        worker: &worker,
                        exp_sender: None,
                        neural_sender: None,
                        ppo_sender: None,
                    };
                    let (s_total, u_total, _, _) =
                        simulate_stats(n, sims_n, rng.next_u64(), free_pulls, &ctx);
                    let s_avg = s_total as f64 / sims_n as f64;
                    let u_avg = u_total as f64 / sims_n as f64;
                    let elapsed_ms = pool_start.elapsed().as_millis() as u64;
                    println!(
                        "{}",
                        I18n::get(lang, "sim_result_stats")
                            .replacen("{}", &sims_n.to_string(), 1)
                            .replacen("{}", &n.to_string(), 1)
                            .replacen("{:.3}", &format!("{:.3}", s_avg), 1)
                            .replacen("{:.3}", &format!("{:.3}", u_avg), 1)
                    );
                    if history.len() >= SIM_HISTORY_CAPACITY {
                        history.pop_front();
                    }
                    history.push_back(SimHistoryEntry {
                        pool_name: pool_config.pool_name.clone(),
                        pulls: n,
                        sims: sims_n,
                        avg_six: s_avg,
                        avg_up: u_avg,
                        elapsed_ms,
                    });
                }
            } else {
                let sim_start = Instant::now();
                let ctx = SimRunContext {
                    neural_opt: &neural_guard,
                    dqn_policy: Some(&dqn_guard),
                    ppo_policy: active_ppo,
                    env_net: &env_net,
                    config: &config,
                    worker: &worker,
                    exp_sender: None,
                    neural_sender: None,
                    ppo_sender: None,
                };
                let (s_total, u_total, _, _) =
                    simulate_stats(n, sims_n, rng.next_u64(), free_pulls, &ctx);
                let s_avg = s_total as f64 / sims_n as f64;
                let u_avg = u_total as f64 / sims_n as f64;
                let elapsed_ms = sim_start.elapsed().as_millis() as u64;
                println!(
                    "{}",
                    I18n::get(lang, "sim_result_stats")
                        .replacen("{}", &sims_n.to_string(), 1)
                        .replacen("{}", &n.to_string(), 1)
                        .replacen("{:.3}", &format!("{:.3}", s_avg), 1)
                        .replacen("{:.3}", &format!("{:.3}", u_avg), 1)
                );
                if history.len() >= SIM_HISTORY_CAPACITY {
                    history.pop_front();
                }
                history.push_back(SimHistoryEntry {
                    pool_name: config.pool_name.clone(),
                    pulls: n,
                    sims: sims_n,
                    avg_six: s_avg,
                    avg_up: u_avg,
                    elapsed_ms,
                });
            }
        } else {
            let dqn_guard = panic_guard::read_shared(&dqn_shared);
            let neural_guard = panic_guard::read_shared(&neural_shared);
            let ppo_guard = panic_guard::read_shared(&ppo_shared);
            let active_ppo = if use_ppo { Some(&*ppo_guard) } else { None };
            if selected_pool_ids.len() > 1 {
                for pool_id in selected_pool_ids.iter() {
                    let mut pool_config = config.clone();
                    if !pool_config.apply_pool(pool_id) {
                        continue;
                    }
                    if let Some(cal) = &calibration {
                        apply_calibration(&mut pool_config, cal);
                    }
                    println!(
                        "{}",
                        I18n::get(lang, "sim_pool_header").replace("{}", &pool_config.pool_name)
                    );
                    let ctx = SimRunContext {
                        neural_opt: &neural_guard,
                        dqn_policy: Some(&dqn_guard),
                        ppo_policy: active_ppo,
                        env_net: &env_net,
                        config: &pool_config,
                        worker: &worker,
                        exp_sender: None,
                        neural_sender: None,
                        ppo_sender: None,
                    };
                    let (s_total, u_total, _, _) =
                        simulate_stats(n, 1, rng.next_u64(), free_pulls, &ctx);
                    let s_avg = s_total as f64;
                    let u_avg = u_total as f64;
                    println!(
                        "{}",
                        I18n::get(lang, "sim_result_stats")
                            .replacen("{}", "1", 1)
                            .replacen("{}", &n.to_string(), 1)
                            .replacen("{:.3}", &format!("{:.3}", s_avg), 1)
                            .replacen("{:.3}", &format!("{:.3}", u_avg), 1)
                    );
                }
                continue;
            }
            let start_time = Instant::now();
            let ctx = SimModelContext {
                neural_opt: &neural_guard,
                dqn_policy: Some(&dqn_guard),
                ppo_policy: active_ppo,
                env_net: &env_net,
                config: &config,
                exp_sender: dqn_sender.as_ref(),
                neural_sender: neural_sender.as_ref(),
                ppo_sender: ppo_sender.as_ref(),
            };
            let res = simulate_one(n, &mut rng, free_pulls, &ctx);
            let elapsed = start_time.elapsed();
            println!(
                "{}",
                I18n::get(lang, "single_sim_result")
                    .replace("{}", &n.to_string())
                    .replace("{:.2?}", &format!("{:.2?}", elapsed))
            );
            println!(
                "{}",
                I18n::get(lang, "single_stats")
                    .replacen("{}", &res.six_count.to_string(), 1)
                    .replacen("{}", &res.up_count.to_string(), 1)
            );
            let non_up_six = build_non_up_six(&config);
            for (i, p) in res.pulls.iter().take(PULL_DISPLAY_LIMIT).enumerate() {
                let op_name = resolve_operator_name(p, &config, &non_up_six);
                let star_label = I18n::get(lang, "unit_star");
                let line = if p.is_up {
                    format!(
                        "{}. {} ({} {}) {}",
                        i + 1,
                        op_name.yellow().bold(),
                        p.rarity,
                        star_label,
                        "[UP]".red().bold()
                    )
                } else {
                    match p.rarity {
                        6 => format!(
                            "{}. {} ({} {})",
                            i + 1,
                            op_name.yellow().bold(),
                            p.rarity,
                            star_label
                        ),
                        5 => format!(
                            "{}. {} ({} {})",
                            i + 1,
                            op_name.purple(),
                            p.rarity,
                            star_label
                        ),
                        _ => format!(
                            "{}. {} ({} {})",
                            i + 1,
                            op_name.dimmed(),
                            p.rarity,
                            star_label
                        ),
                    }
                };
                println!("{}", line);
            }
            if res.pulls.len() > PULL_DISPLAY_LIMIT {
                println!(
                    "{}",
                    I18n::get(lang, "pull_list_omitted")
                        .replace("{}", &(res.pulls.len() - PULL_DISPLAY_LIMIT).to_string())
                );
            }

            println!("{}", I18n::get(lang, "consumption_header"));
            println!(
                "{}",
                I18n::get(lang, "consumption_free").replace("{}", &res.free_pulls_used.to_string())
            );
            println!(
                "{}",
                I18n::get(lang, "consumption_jade")
                    .replacen("{}", &res.cost_jade.to_string(), 1)
                    .replacen("{}", &(res.cost_jade / COST_PER_PULL).to_string(), 1)
            );
            if res.big_pity_used {
                println!("{}", I18n::get(lang, "big_pity_triggered"));
            }
            println!("{}", I18n::get(lang, "consumption_footer"));
            if history.len() >= SIM_HISTORY_CAPACITY {
                history.pop_front();
            }
            history.push_back(SimHistoryEntry {
                pool_name: config.pool_name.clone(),
                pulls: n,
                sims: 1,
                avg_six: res.six_count as f64,
                avg_up: res.up_count as f64,
                elapsed_ms: elapsed.as_millis() as u64,
            });
        }
    }

    stop_flag.store(true, Ordering::Relaxed);
    for handle in online_handles {
        let _ = handle.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PoolConfig;
    use sim::{simulate_core, simulate_fast, simulate_one, SimControl, SimModelContext};

    fn build_context() -> (Config, EnvNet, NeuralLuckOptimizer) {
        let mut config = Config::load("data/config.json");
        // Force small model dims in tests to avoid OOM with large defaults
        config.model_dim = 32;
        config.model_hidden_dim = 2048;
        config.model_num_layers = 4;
        config.model_num_heads = 8;
        config.model_kv_lora_rank = 128;
        config.model_qk_rope_dim = 64;
        let mut rng = Rng::from_seed(1234);
        let env_net = EnvNet::new(&mut rng);
        let neural_opt = NeuralLuckOptimizer::new(5678);
        (config, env_net, neural_opt)
    }

    #[test]
    fn default_pool_index_uses_active_pool_position() {
        let config = Config {
            active_pool: Some("pool_b".to_string()),
            pools: vec![
                PoolConfig {
                    id: "pool_a".to_string(),
                    name: "Pool A".to_string(),
                    pool_type: "character_up".to_string(),
                    up_six: vec![],
                    up_rate: 0.0,
                    prob_6_base: 0.008,
                    prob_5_base: 0.08,
                    prob_4_base: 0.912,
                    soft_pity_start: 65,
                    soft_pity_slope: 0.05,
                    small_pity_guarantee: 80,
                    big_pity_cumulative: 120,
                    up_pity_soft: 0,
                    five_star_pity: 10,
                    always_5_star: false,
                    big_pity_requires_not_up: true,
                    six_stars: vec![],
                    five_stars: vec![],
                    four_stars: vec![],
                    is_archived: false,
                },
                PoolConfig {
                    id: "pool_b".to_string(),
                    name: "Pool B".to_string(),
                    pool_type: "character_up".to_string(),
                    up_six: vec![],
                    up_rate: 0.0,
                    prob_6_base: 0.008,
                    prob_5_base: 0.08,
                    prob_4_base: 0.912,
                    soft_pity_start: 65,
                    soft_pity_slope: 0.05,
                    small_pity_guarantee: 80,
                    big_pity_cumulative: 120,
                    up_pity_soft: 0,
                    five_star_pity: 10,
                    always_5_star: false,
                    big_pity_requires_not_up: true,
                    six_stars: vec![],
                    five_stars: vec![],
                    four_stars: vec![],
                    is_archived: false,
                },
            ],
            ..Default::default()
        };

        assert_eq!(default_pool_index(&config), 2);
    }

    #[test]
    fn resolve_f2p_luck_mode_defaults_to_probability_when_global_mode_is_ppo() {
        let config = Config {
            luck_mode: LuckMode::Ppo,
            ..Default::default()
        };

        assert_eq!(resolve_f2p_luck_mode(&config), LuckMode::Probability);
    }

    #[test]
    fn resolve_f2p_luck_mode_respects_explicit_override() {
        let config = Config {
            luck_mode: LuckMode::Ppo,
            f2p_luck_mode: Some(LuckMode::Dqn),
            ..Default::default()
        };

        assert_eq!(resolve_f2p_luck_mode(&config), LuckMode::Dqn);
    }

    #[test]
    fn simulate_fast_costs_and_free_pulls_match() {
        let (config, env_net, neural_opt) = build_context();
        let mut rng = Rng::from_seed(1);
        let num_pulls = 200;
        let free_pulls = FREE_PULLS_WELFARE;
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            env_net: &env_net,
            config: &config,
            exp_sender: None,
            neural_sender: None,
            ppo_sender: None,
        };
        let res = simulate_fast(num_pulls, &mut rng, free_pulls, &ctx);
        let expected_free_used = free_pulls.min(num_pulls as u32);
        let expected_cost = (num_pulls as u32 - expected_free_used) * COST_PER_PULL;
        assert_eq!(res.free_pulls_used, expected_free_used);
        assert_eq!(res.cost_jade, expected_cost);
    }

    #[test]
    fn simulate_one_counts_match_pulls() {
        let (config, env_net, neural_opt) = build_context();
        let mut rng = Rng::from_seed(2);
        let num_pulls = 120;
        let free_pulls = FREE_PULLS_WELFARE;
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            env_net: &env_net,
            config: &config,
            exp_sender: None,
            neural_sender: None,
            ppo_sender: None,
        };
        let res = simulate_one(num_pulls, &mut rng, free_pulls, &ctx);
        let six_count = res.pulls.iter().filter(|p| p.rarity == 6).count();
        let up_count = res.pulls.iter().filter(|p| p.is_up).count();
        let expected_free_used = free_pulls.min(num_pulls as u32);
        let expected_cost = (num_pulls as u32 - expected_free_used) * COST_PER_PULL;
        assert_eq!(res.six_count, six_count);
        assert_eq!(res.up_count, up_count);
        assert_eq!(res.free_pulls_used, expected_free_used);
        assert_eq!(res.cost_jade, expected_cost);
    }

    #[test]
    fn simulate_core_f2p_clearing_always_hits_up() {
        let (config, env_net, neural_opt) = build_context();
        let mut rng = Rng::from_seed(3);
        let control = SimControl {
            max_pulls: None,
            stop_on_up: true,
            // Ensure test uses max range to guarantee hit
            stop_after_total_pulls: Some(
                FREE_PULLS_WELFARE.max(config.big_pity_cumulative as u32) as usize
            ),
            nn_total_pulls_one_based: true,
            collect_details: false,
            big_pity_requires_not_up: false,
            fast_inference: true,
        };
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            env_net: &env_net,
            config: &config,
            exp_sender: None,
            neural_sender: None,
            ppo_sender: None,
        };
        let (stats, _) = simulate_core(&control, &mut rng, FREE_PULLS_WELFARE, &ctx);
        assert!(stats.up_count > 0);
    }

    #[test]
    fn dqn_training_produces_valid_q_values() {
        let (mut config, env_net, neural_opt) = build_context();
        config.fast_init = true;
        // Use tiny hidden dim so the test finishes quickly and doesn't OOM
        config.model_hidden_dim = 64;
        config.model_num_layers = 2;
        let mut rng = Rng::from_seed(7777);
        let dqn = train_dqn(&neural_opt, &mut rng, &env_net, &config);
        let state = AutoTensor::new(vec![0.5; DIM], vec![DIM]);
        let q_values = dqn.forward(&state);
        let q_data = q_values.data.read().unwrap();
        assert_eq!(
            q_data.len(),
            5,
            "Q-values should have ACTION_SPACE=5 outputs"
        );
        for &v in q_data.iter() {
            assert!(v.is_finite(), "Q-values must be finite, got {}", v);
        }
    }

    #[test]
    fn benchmark_simulate_fast_throughput() {
        let (config, env_net, neural_opt) = build_context();
        let mut rng = Rng::from_seed(42);
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            env_net: &env_net,
            config: &config,
            exp_sender: None,
            neural_sender: None,
            ppo_sender: None,
        };
        let iterations = 5000;
        let pulls = 200;

        // Warmup
        for _ in 0..100 {
            let _ = simulate_fast(pulls, &mut rng, 0, &ctx);
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = simulate_fast(pulls, &mut rng, 0, &ctx);
        }
        let elapsed = start.elapsed();
        let throughput = iterations as f64 / elapsed.as_secs_f64();
        println!(
            "\n[PERF] simulate_fast (no senders, fast_inference=true): {} iters x {} pulls in {:.2?} ({:.0} sims/sec)",
            iterations, pulls, elapsed, throughput
        );
        assert!(throughput > 100.0, "Throughput too low: {:.0}", throughput);
    }

    #[test]
    fn benchmark_dqn_predict_action_fast() {
        let dqn = DuelingQNetwork::new(42, &crate::config::AchfConfig::default());
        let features = [0.5_f64; DIM];
        let iterations = 10_000;

        // Warmup
        for _ in 0..100 {
            let _ = dqn.predict_action_fast(&features);
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = dqn.predict_action_fast(&features);
        }
        let fast_elapsed = start.elapsed();

        let start2 = std::time::Instant::now();
        for _ in 0..iterations {
            let tensor_x = AutoTensor::new(features.to_vec(), vec![DIM]);
            let _ = dqn.predict_action(&tensor_x);
        }
        let tensor_elapsed = start2.elapsed();

        let speedup = tensor_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64();
        println!(
            "\n[PERF] DQN predict: fast={:.2?} vs tensor={:.2?} (speedup: {:.2}x)",
            fast_elapsed, tensor_elapsed, speedup
        );
        assert!(
            speedup > 0.95,
            "predict_action_fast should be faster than tensor path, got {:.2}x",
            speedup
        );
    }
}
