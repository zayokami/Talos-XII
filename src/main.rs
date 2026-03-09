mod achf;
mod autograd;
mod binary_codec;
mod calibrate;
mod collect;
mod config;
mod dbn;
mod dqn;
#[cfg(test)]
mod grad_check;
mod i18n;
mod model_io;
mod neural;
mod nn;
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
use config::{Config, LuckMode};
use dbn::Dbn;
use dqn::{train_dqn, DuelingQNetwork, Experience, OnlineDqnTrainer};
use i18n::{I18n, Language};
use log::info;
use neural::NeuralLuckOptimizer;
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
    resolve_operator_name, simulate_f2p_clearing, simulate_fast, simulate_one, simulate_stats,
    NeuralSample, PpoExperience, SimModelContext, SimRunContext, COST_PER_PULL, FREE_PULLS_WELFARE,
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
    Benchmark,
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
    if default_yes {
        !s.eq_ignore_ascii_case("n")
    } else {
        s.eq_ignore_ascii_case("y")
    }
}

use model_io::{load_model, load_neural_cache, save_model, save_neural_cache};
use utils::{
    F2P_BATCH_COUNT, INPUT_CAP, MAX_DRAIN_PER_TICK, ONLINE_REPORT_INTERVAL_SECS, PPO_ONLINE_LR,
    PULL_DISPLAY_LIMIT, SIM_HISTORY_CAPACITY,
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
            200_000
        } else {
            1_000_000
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

struct F2pAnalysisCtx<'a> {
    config: &'a Config,
    neural_opt: &'a NeuralLuckOptimizer,
    dqn_policy: Option<&'a DuelingQNetwork>,
    ppo_policy: Option<&'a ActorCritic>,
    dbn: &'a Dbn,
    worker: &'a GoodJobWorker,
    lang: Language,
}

fn run_f2p_analysis(ctx: &F2pAnalysisCtx<'_>, rng: &mut Rng) {
    let lang = ctx.lang;
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

    let batches: usize = F2P_BATCH_COUNT;
    let batch_base = sim_count_prob / batches;
    let batch_remainder = sim_count_prob % batches;
    let mut total_up_agg = 0;
    let mut total_with_up_agg = 0;

    let start_time = Instant::now();
    let sim_ctx = SimRunContext {
        neural_opt: ctx.neural_opt,
        dqn_policy: ctx.dqn_policy,
        ppo_policy: ctx.ppo_policy,
        dbn: ctx.dbn,
        config: ctx.config,
        worker: ctx.worker,
        exp_sender: None,
        neural_sender: None,
        ppo_sender: None,
    };

    for i in 0..batches {
        let this_batch = batch_base + if i < batch_remainder { 1 } else { 0 };
        if this_batch == 0 {
            continue;
        }
        let (_, total_up, _, total_with_up) = simulate_stats(
            FREE_PULLS_WELFARE as usize,
            this_batch,
            rng.next_u64(),
            &sim_ctx,
        );
        total_up_agg += total_up;
        total_with_up_agg += total_with_up;

        print!(
            "\r{}",
            I18n::get(lang, "progress").replace("{:>3}", &format!("{:>3}", i + 1))
        );
        let _ = io::stdout().flush();
    }
    println!();

    let elapsed = start_time.elapsed();
    let total_sims_run = sim_count_prob;

    let prob_line = format_f2p_probability_line(total_sims_run, total_with_up_agg);
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

    let cost_batch_base = sim_count_cost / batches;
    let cost_batch_remainder = sim_count_cost % batches;
    let mut total_extra_cost_agg = 0u64;
    let mut extra_cost_samples_agg = 0usize;

    for i in 0..batches {
        let this_batch = cost_batch_base + if i < cost_batch_remainder { 1 } else { 0 };
        if this_batch == 0 {
            continue;
        }
        let (cost_sum, samples, _) = simulate_f2p_clearing(this_batch, rng.next_u64(), &sim_ctx);
        total_extra_cost_agg += cost_sum;
        extra_cost_samples_agg += samples;

        print!(
            "\r{}",
            I18n::get(lang, "progress").replace("{:>3}", &format!("{:>3}", i + 1))
        );
        let _ = io::stdout().flush();
    }
    println!();

    let avg_extra_cost = if extra_cost_samples_agg == 0 {
        None
    } else {
        Some(total_extra_cost_agg as f64 / extra_cost_samples_agg as f64)
    };

    let avg_cost_line = format_avg_extra_cost_line(avg_extra_cost);
    println!("{}", avg_cost_line);
}

fn print_explainability_report(neural_opt: &NeuralLuckOptimizer, lang: Language) {
    println!("{}", I18n::get(lang, "insight_header"));
    let rl_w = neural_opt.linear_weights;
    let rl_b = neural_opt.linear_bias;

    let feature_names = [
        I18n::get(lang, "feat_pity"),
        I18n::get(lang, "feat_total_norm"),
        I18n::get(lang, "feat_env_noise"),
        I18n::get(lang, "feat_loss_norm"),
        I18n::get(lang, "feat_streak_4"),
        I18n::get(lang, "feat_env_bias"),
        I18n::get(lang, "feat_pity_loss"),
        I18n::get(lang, "feat_total_sq"),
    ];
    for (i, name) in feature_names.iter().enumerate() {
        let w = rl_w[i];
        let impact = if w.abs() < 0.001 {
            I18n::get(lang, "impact_neutral")
        } else if w > 0.0 {
            I18n::get(lang, "impact_boost")
        } else {
            I18n::get(lang, "impact_reduce")
        };
        println!("  - {:<25}: {:>8.4} [{}]", name, w, impact);
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
    dbn: &Dbn,
    config: &Config,
    lang: Language,
) {
    let ctx = SimModelContext {
        neural_opt,
        dqn_policy,
        ppo_policy,
        dbn,
        config,
        exp_sender: None,
        neural_sender: None,
        ppo_sender: None,
    };
    let fast_sims = 10_000usize;
    let fast_pulls = 200usize;
    let start_fast = Instant::now();
    for _ in 0..fast_sims {
        let _ = simulate_fast(fast_pulls, rng, 0, &ctx);
    }
    let fast_elapsed = start_fast.elapsed();
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

    let one_sims = 300usize;
    let one_pulls = 120usize;
    let start_one = Instant::now();
    for _ in 0..one_sims {
        let _ = simulate_one(one_pulls, rng, 0, &ctx);
    }
    let one_elapsed = start_one.elapsed();
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
    Dbn,
    NeuralLuckOptimizer,
    DuelingQNetwork,
    ActorCritic,
    GoodJobWorker,
    Rng,
) {
    let config = Config::load(&args.config);
    let mut rng = if let Some(seed) = args.seed {
        Rng::from_seed(seed)
    } else {
        Rng::new()
    };

    let worker = GoodJobWorker::new_with_config(&config);

    let mut dbn = Dbn::new(&[8, 16, 8], &mut rng);
    let (dbn_data_count, dbn_epochs) = if config.fast_init {
        if cfg!(debug_assertions) {
            (64, 2)
        } else {
            (256, 4)
        }
    } else if cfg!(debug_assertions) {
        (256, 5)
    } else {
        (1024, 20)
    };
    dbn.train(&mut rng, dbn_data_count, dbn_epochs);

    let mut trained_neural_opt = if !args.force {
        if let Some(cached) = load_neural_cache(NEURAL_CACHE_PATH) {
            info!("[Neural Core] Cache detected. Cached weights loaded.");
            cached
        } else {
            info!("[Neural Core] Cache not found. Training new weights...");
            train_neural_optimizer(rng.next_u64(), &dbn, &config, &worker)
        }
    } else {
        info!("[Neural Core] Force training new weights...");
        train_neural_optimizer(rng.next_u64(), &dbn, &config, &worker)
    };

    info!("[Linear] Training linear regression...");
    let (lin_w, lin_b) = train_linear_regression(&trained_neural_opt, &mut rng, &dbn, &config);
    trained_neural_opt.set_linear_params(lin_w, lin_b);

    info!("[RL] Manifold Optimization (Parallel)...");
    trained_neural_opt = train_manifold_rl(&trained_neural_opt, &mut rng, &dbn, &config, &worker);

    // Save Neural Cache
    if save_neural_cache(NEURAL_CACHE_PATH, &trained_neural_opt) {
        info!("[Neural Core] Cache saved.");
    }

    // DQN
    let dqn_policy = if config.online_train && config.online_train_dqn {
        DuelingQNetwork::new(rng.next_u64(), &config.achf)
    } else if !args.force {
        if let Some(cached) = load_model::<DuelingQNetwork>("dqn.cache", "DQN") {
            info!("[DQN] Cached model loaded.");
            cached
        } else {
            info!("[DQN] Training new model...");
            let d = train_dqn(&trained_neural_opt, &mut rng, &dbn, &config);
            save_model(&d, "dqn.cache", "DQN");
            d
        }
    } else {
        info!("[DQN] Force training new model...");
        let d = train_dqn(&trained_neural_opt, &mut rng, &dbn, &config);
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
            let p = train_ppo(&mut rng, &dbn, &config);
            save_model(&p, "ppo.cache", "PPO");
            p
        }
    } else {
        info!("[PPO] Force training new model...");
        let p = train_ppo(&mut rng, &dbn, &config);
        save_model(&p, "ppo.cache", "PPO");
        p
    };

    (
        config,
        dbn,
        trained_neural_opt,
        dqn_policy,
        ppo_policy,
        worker,
        rng,
    )
}

fn main() {
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
                                println!(
                                    "{}",
                                    if lang == Language::Cn {
                                        "✓ 数据已保存。"
                                    } else {
                                        "✓ Data saved."
                                    }
                                );
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
                                    if lang == Language::Cn {
                                        "✓ 已导入"
                                    } else {
                                        "✓ Imported"
                                    },
                                    count,
                                    if lang == Language::Cn {
                                        "个会话。"
                                    } else {
                                        "sessions."
                                    },
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("{}", e);
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
                    println!(
                        "{}",
                        if lang == Language::Cn {
                            "[校准] 没有玩家数据。请先使用 collect add 录入数据。"
                        } else {
                            "[Calibrate] No player data. Use 'collect add' to record data first."
                        }
                    );
                } else {
                    let cal = run_calibration(&db, &config, lang);
                    if cal.save(&config.calibrated_path) {
                        println!(
                            "{}",
                            if lang == Language::Cn {
                                "✓ 校准参数已保存。下次模拟将自动加载。"
                            } else {
                                "✓ Calibrated parameters saved. Next simulation will auto-load."
                            }
                        );
                    }
                }
            }
            _ => unreachable!(),
        }
        return;
    }

    let (mut config, dbn, trained_neural_opt, dqn_policy, ppo_policy, worker, mut rng) =
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
                dbn,
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
                dbn: &dbn,
                config: &config,
                worker: &worker,
                exp_sender: None,
                neural_sender: None,
                ppo_sender: None,
            };
            let (six_count, up_count, _, _) = simulate_stats(pulls, count, rng.next_u64(), &ctx);
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
        Commands::Benchmark => {
            benchmark_simulation(
                &mut rng,
                &trained_neural_opt,
                Some(&dqn_policy),
                Some(&ppo_policy),
                &dbn,
                &config,
                lang,
            );
            demo_mmap_tensor();
        }
        Commands::F2p => {
            let f2p_ctx = F2pAnalysisCtx {
                config: &config,
                neural_opt: &trained_neural_opt,
                dqn_policy: Some(&dqn_policy),
                ppo_policy: Some(&ppo_policy),
                dbn: &dbn,
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
    dbn: Dbn,
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
        dbn,
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
        let trainer_seed = rng.next_u64();
        let achf = config.achf.clone();
        let mut trainer = OnlinePpoTrainer::new(trainer_seed, 2, 128, &achf);
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
        match (pool_type, lang) {
            ("character_up", Language::Cn) => "角色 UP".to_string(),
            ("character_up", _) => "Character UP".to_string(),
            ("weapon_up", Language::Cn) => "武器 UP".to_string(),
            ("weapon_up", _) => "Weapon UP".to_string(),
            ("standard", Language::Cn) => "基础寻访".to_string(),
            ("standard", _) => "Standard".to_string(),
            ("beginner", Language::Cn) => "启程寻访".to_string(),
            ("beginner", _) => "Beginner".to_string(),
            ("permanent", Language::Cn) => "常驻".to_string(),
            ("permanent", _) => "Permanent".to_string(),
            (_, Language::Cn) => "未知".to_string(),
            _ => "Unknown".to_string(),
        }
    };
    let print_pool_header = |config: &Config, lang: Language| {
        let up_label = if config.up_six.is_empty() {
            match lang {
                Language::Cn => "无".to_string(),
                _ => "None".to_string(),
            }
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
        let default_index = 1usize;
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
        let dqn_guard = dqn_shared.read().unwrap();
        let neural_guard = neural_shared.read().unwrap();
        let ppo_guard = ppo_shared.read().unwrap();
        benchmark_simulation(
            &mut rng,
            &neural_guard,
            Some(&dqn_guard),
            Some(&ppo_guard),
            &dbn,
            &config,
            lang,
        );
        demo_mmap_tensor();
    }

    // F2P Analysis
    {
        let dqn_guard = dqn_shared.read().unwrap();
        let neural_guard = neural_shared.read().unwrap();
        let ppo_guard = ppo_shared.read().unwrap();
        let f2p_ctx = F2pAnalysisCtx {
            config: &config,
            neural_opt: &neural_guard,
            dqn_policy: Some(&*dqn_guard),
            ppo_policy: Some(&*ppo_guard),
            dbn: &dbn,
            worker: &worker,
            lang,
        };
        run_f2p_analysis(&f2p_ctx, &mut rng);
    }
    println!("{}", I18n::get(lang, "total_value"));

    let mut history: std::collections::VecDeque<SimHistoryEntry> =
        std::collections::VecDeque::with_capacity(SIM_HISTORY_CAPACITY);

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
        let cmd_lower = cmd.to_ascii_lowercase();
        if cmd_lower == "h" || cmd_lower == "help" {
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
                for (i, entry) in history.iter().enumerate() {
                    println!(
                        "{}",
                        I18n::get(lang, "cmd_history_item")
                            .replacen("{}", &(i + 1).to_string(), 1)
                            .replacen("{}", &entry.pool_name, 1)
                            .replacen("{}", &entry.pulls.to_string(), 1)
                            .replacen("{}", &entry.sims.to_string(), 1)
                            .replacen("{:.3}", &format!("{:.3}", entry.avg_six), 1)
                            .replacen("{:.3}", &format!("{:.3}", entry.avg_up), 1)
                            .replacen("{}", &entry.elapsed_ms.to_string(), 1)
                    );
                }
            }
            println!("{}", I18n::get(lang, "cmd_history_footer"));
            continue;
        }
        if cmd_lower == "pool" {
            let sub = parts.next().unwrap_or("list");
            if sub.eq_ignore_ascii_case("list") {
                let list_label = if config.pools.is_empty() {
                    if lang == Language::Cn {
                        "无".to_string()
                    } else {
                        "None".to_string()
                    }
                } else {
                    config
                        .pools
                        .iter()
                        .map(|p| {
                            let tag = if p.is_archived {
                                I18n::get(lang, "pool_archived_tag")
                            } else {
                                String::new()
                            };
                            format!(
                                "{}={}{}/{}",
                                p.id,
                                p.name,
                                tag,
                                pool_type_label(&p.pool_type, lang)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                println!(
                    "{}",
                    I18n::get(lang, "cmd_pool_list").replace("{}", &list_label)
                );
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
                    if config.pools.iter().any(|p| p.id == id) {
                        valid_ids.push(id.to_string());
                    }
                }
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
            } else if config.apply_pool(sub) {
                if let Some(cal) = &calibration {
                    apply_calibration(&mut config, cal);
                }
                selected_pool_ids = vec![sub.to_string()];
                println!(
                    "{}",
                    I18n::get(lang, "cmd_pool_switched").replace("{}", &config.pool_name)
                );
                print_pool_header(&config, lang);
            } else {
                println!(
                    "{}",
                    I18n::get(lang, "cmd_pool_not_found").replace("{}", sub)
                );
            }
            continue;
        }
        if cmd_lower == "p" || cmd_lower == "pulls" {
            if let Some(value) = parts.next() {
                match value.parse::<usize>() {
                    Ok(val) if val > 0 => {
                        default_pulls = val.min(INPUT_CAP);
                        if val > INPUT_CAP {
                            println!("{}", I18n::get(lang, "input_too_large"));
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
                            println!("{}", I18n::get(lang, "sim_count_too_large"));
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

        let n = if input.is_empty() {
            default_pulls
        } else {
            match input.parse::<usize>() {
                Ok(val) => {
                    if val > INPUT_CAP {
                        println!("{}", I18n::get(lang, "input_too_large"));
                        INPUT_CAP
                    } else {
                        val
                    }
                }
                Err(_) => {
                    println!("{}", I18n::get(lang, "invalid_input"));
                    continue;
                }
            }
        };

        let free_pulls = if use_welfare_default {
            FREE_PULLS_WELFARE
        } else {
            0
        };
        let sims_n = default_sims;

        if sims_n > 1 {
            let dqn_guard = dqn_shared.read().unwrap();
            let neural_guard = neural_shared.read().unwrap();
            let ppo_guard = ppo_shared.read().unwrap();
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
                        dbn: &dbn,
                        config: &pool_config,
                        worker: &worker,
                        exp_sender: None,
                        neural_sender: None,
                        ppo_sender: None,
                    };
                    let (s_total, u_total, _, _) = simulate_stats(n, sims_n, rng.next_u64(), &ctx);
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
                    dbn: &dbn,
                    config: &config,
                    worker: &worker,
                    exp_sender: None,
                    neural_sender: None,
                    ppo_sender: None,
                };
                let (s_total, u_total, _, _) = simulate_stats(n, sims_n, rng.next_u64(), &ctx);
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
            let dqn_guard = dqn_shared.read().unwrap();
            let neural_guard = neural_shared.read().unwrap();
            let ppo_guard = ppo_shared.read().unwrap();
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
                        dbn: &dbn,
                        config: &pool_config,
                        worker: &worker,
                        exp_sender: None,
                        neural_sender: None,
                        ppo_sender: None,
                    };
                    let (s_total, u_total, _, _) = simulate_stats(n, 1, rng.next_u64(), &ctx);
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
                dbn: &dbn,
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
    use sim::{simulate_core, simulate_fast, simulate_one, SimControl, SimModelContext};

    fn build_context() -> (Config, Dbn, NeuralLuckOptimizer) {
        let config = Config::load("data/config.json");
        let mut rng = Rng::from_seed(1234);
        let dbn = Dbn::new(&[8, 16, 8], &mut rng);
        let neural_opt = NeuralLuckOptimizer::new(5678);
        (config, dbn, neural_opt)
    }

    #[test]
    fn simulate_fast_costs_and_free_pulls_match() {
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(1);
        let num_pulls = 200;
        let free_pulls = FREE_PULLS_WELFARE;
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            dbn: &dbn,
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
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(2);
        let num_pulls = 120;
        let free_pulls = FREE_PULLS_WELFARE;
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            dbn: &dbn,
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
        let (config, dbn, neural_opt) = build_context();
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
            dbn: &dbn,
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
        let (mut config, dbn, neural_opt) = build_context();
        config.fast_init = true;
        let mut rng = Rng::from_seed(7777);
        let dqn = train_dqn(&neural_opt, &mut rng, &dbn, &config);
        let state = AutoTensor::new(vec![0.5; 8], vec![8]);
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
        let (config, dbn, neural_opt) = build_context();
        let mut rng = Rng::from_seed(42);
        let ctx = SimModelContext {
            neural_opt: &neural_opt,
            dqn_policy: None,
            ppo_policy: None,
            dbn: &dbn,
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
        let features = [0.5_f64; 8];
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
            let tensor_x = AutoTensor::new(features.to_vec(), vec![8]);
            let _ = dqn.predict_action(&tensor_x);
        }
        let tensor_elapsed = start2.elapsed();

        let speedup = tensor_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64();
        println!(
            "\n[PERF] DQN predict: fast={:.2?} vs tensor={:.2?} (speedup: {:.2}x)",
            fast_elapsed, tensor_elapsed, speedup
        );
        assert!(
            speedup > 1.0,
            "predict_action_fast should be faster, got {:.2}x",
            speedup
        );
    }
}
