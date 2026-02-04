mod autograd;
mod config;
mod dbn;
mod dqn;
#[cfg(test)]
mod grad_check;
mod neural;
mod nn;
mod ppo;
mod rng;
mod sim; // Simulation logic
mod simd;
mod trainer; // Training logic
mod transformer;
mod worker;
mod i18n;

use autograd::Tensor as AutoTensor;
use clap::{Parser, Subcommand};
use config::Config;
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
    NeuralSample, PpoExperience, COST_PER_PULL, FREE_PULLS_WELFARE,
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
}

fn prompt_yes_no(prompt: &str, default_yes: bool) -> bool {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
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

fn read_cache_bytes(path: &str) -> Option<Vec<u8>> {
    if let Ok(bytes) = std::fs::read(path) {
        return Some(bytes);
    }
    let alt = format!("../../{}", path);
    std::fs::read(alt).ok()
}

fn load_neural_cache(path: &str) -> Option<NeuralLuckOptimizer> {
    let bytes = read_cache_bytes(path)?;
    NeuralLuckOptimizer::from_bytes(&bytes)
}

fn save_neural_cache(path: &str, net: &NeuralLuckOptimizer) -> bool {
    let bytes = net.to_bytes();
    if std::fs::write(path, &bytes).is_ok() {
        return true;
    }
    let alt = format!("../../{}", path);
    std::fs::write(alt, &bytes).is_ok()
}

fn save_ppo_model(model: &ActorCritic, path: &str) {
    // Try saving as binary (bincode) first for speed
    let bin_path = format!("{}.bin", path);
    if let Ok(file) = std::fs::File::create(&bin_path) {
        let writer = std::io::BufWriter::new(file);
        if bincode::serialize_into(writer, model).is_ok() {
            info!("[PPO] Model saved to {} (Binary)", bin_path);
        }
    }

    // Also save as JSON for compatibility/debugging
    if let Ok(file) = std::fs::File::create(path) {
        let writer = std::io::BufWriter::new(file);
        if serde_json::to_writer(writer, model).is_ok() {
            info!("[PPO] Model saved to {} (JSON)", path);
        } else {
            info!("[PPO] Failed to serialize model (JSON)");
        }
    } else {
        info!("[PPO] Failed to create file {}", path);
    }
}

fn load_ppo_model(path: &str) -> Option<ActorCritic> {
    // Try binary first
    let bin_path = format!("{}.bin", path);
    if let Ok(file) = std::fs::File::open(&bin_path) {
        let reader = std::io::BufReader::new(file);
        if let Ok(model) = bincode::deserialize_from(reader) {
            info!("[PPO] Loaded model from {} (Binary)", bin_path);
            return Some(model);
        }
    }

    // Fallback to JSON
    if let Ok(file) = std::fs::File::open(path) {
        let reader = std::io::BufReader::new(file);
        if let Ok(model) = serde_json::from_reader(reader) {
            info!("[PPO] Loaded model from {} (JSON)", path);
            return Some(model);
        }
    }
    None
}

// Demo of "Crazy" Mmap loading
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
    let fast_sims = 10_000usize;
    let fast_pulls = 200usize;
    let start_fast = Instant::now();
    for _ in 0..fast_sims {
        let _ = simulate_fast(
            fast_pulls, rng, 0, neural_opt, dqn_policy, ppo_policy, dbn, config, None, None, None,
        );
    }
    let fast_elapsed = start_fast.elapsed();
    println!(
        "{}",
        I18n::get(lang, "bench_fast")
            .replacen("{}", &fast_sims.to_string(), 1)
            .replacen("{}", &fast_pulls.to_string(), 1)
            .replace("{:.2?}", &format!("{:.2?}", fast_elapsed))
            .replace("{:.0}", &format!("{:.0}", fast_sims as f64 / fast_elapsed.as_secs_f64()))
    );

    let one_sims = 300usize;
    let one_pulls = 120usize;
    let start_one = Instant::now();
    for _ in 0..one_sims {
        let _ = simulate_one(
            one_pulls, rng, 0, neural_opt, dqn_policy, ppo_policy, dbn, config, None, None, None,
        );
    }
    let one_elapsed = start_one.elapsed();
    println!(
        "{}",
        I18n::get(lang, "bench_one")
            .replacen("{}", &one_sims.to_string(), 1)
            .replacen("{}", &one_pulls.to_string(), 1)
            .replace("{:.2?}", &format!("{:.2?}", one_elapsed))
            .replace("{:.0}", &format!("{:.0}", one_sims as f64 / one_elapsed.as_secs_f64()))
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
        DuelingQNetwork::new(rng.next_u64())
    } else {
        train_dqn(&trained_neural_opt, &mut rng, &dbn, &config)
    };

    // PPO
    let ppo_policy = if !args.force {
        if let Some(cached) = load_ppo_model("ppo.cache") {
            info!("[PPO] Cached model loaded.");
            cached
        } else {
            info!("[PPO] Training new model...");
            let p = train_ppo(&mut rng, &dbn, &config);
            save_ppo_model(&p, "ppo.cache");
            p
        }
    } else {
        info!("[PPO] Force training new model...");
        let p = train_ppo(&mut rng, &dbn, &config);
        save_ppo_model(&p, "ppo.cache");
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

    let (config, dbn, trained_neural_opt, dqn_policy, ppo_policy, worker, mut rng) =
        initialize_system(&args);
    let lang = Language::from_config(&config);

    match args.command.clone().unwrap_or(Commands::Interactive) {
        Commands::Interactive => {
            run_interactive(
                config,
                dbn,
                trained_neural_opt,
                dqn_policy,
                ppo_policy,
                worker,
                rng,
                lang,
            );
        }
        Commands::Simulate { count, pulls } => {
            // Run simulation
            let (six_count, up_count, _, _) = simulate_stats(
                pulls,
                count,
                rng.next_u64(),
                &trained_neural_opt,
                Some(&dqn_policy),
                Some(&ppo_policy),
                &dbn,
                &config,
                &worker,
                None,
                None,
                None,
            );
            println!(
                "{}",
                I18n::get(lang, "batch_sim_header")
                    .replacen("{}", &count.to_string(), 1)
                    .replacen("{}", &pulls.to_string(), 1)
            );
            println!(
                "{}",
                I18n::get(lang, "avg_6_star").replace("{:.4}", &format!("{:.4}", six_count as f64 / count as f64))
            );
            println!(
                "{}",
                I18n::get(lang, "avg_up").replace("{:.4}", &format!("{:.4}", up_count as f64 / count as f64))
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
            println!(
                "{}",
                I18n::get(lang, "f2p_header").replace("{}", &FREE_PULLS_WELFARE.to_string())
            );
            #[cfg(debug_assertions)]
            let sim_count_prob = if config.f2p_sim_count_prob > 0 {
                config.f2p_sim_count_prob
            } else if config.f2p_sim_count > 0 {
                config.f2p_sim_count
            } else if config.fast_init {
                2_000
            } else {
                10_000
            };
            #[cfg(not(debug_assertions))]
            let sim_count_prob = if config.f2p_sim_count_prob > 0 {
                config.f2p_sim_count_prob
            } else if config.f2p_sim_count > 0 {
                config.f2p_sim_count
            } else if config.fast_init {
                200_000
            } else {
                1_000_000
            };
            #[cfg(debug_assertions)]
            let sim_count_cost = if config.f2p_sim_count_cost > 0 {
                config.f2p_sim_count_cost
            } else if config.f2p_sim_count > 0 {
                config.f2p_sim_count
            } else if sim_count_prob >= 50_000 {
                sim_count_prob / 2
            } else {
                sim_count_prob
            };
            #[cfg(not(debug_assertions))]
            let sim_count_cost = if config.f2p_sim_count_cost > 0 {
                config.f2p_sim_count_cost
            } else if config.f2p_sim_count > 0 {
                config.f2p_sim_count
            } else if sim_count_prob >= 200_000 {
                sim_count_prob / 2
            } else {
                sim_count_prob
            };

            println!(
                "{}",
                I18n::get(lang, "sys_run_prob").replace("{}", &sim_count_prob.to_string())
            );

            let batches = 100;
            let batch_size_prob = sim_count_prob / batches;
            let mut total_up_agg = 0;
            let mut total_with_up_agg = 0;

            let start_time = Instant::now();
            
            for i in 0..batches {
                let (_, total_up, _, total_with_up) = simulate_stats(
                    FREE_PULLS_WELFARE as usize,
                    batch_size_prob,
                    rng.next_u64(),
                    &trained_neural_opt,
                    Some(&dqn_policy),
                    Some(&ppo_policy),
                    &dbn,
                    &config,
                    &worker,
                    None,
                    None,
                    None,
                );
                total_up_agg += total_up;
                total_with_up_agg += total_with_up;
                
                print!("\r{}", I18n::get(lang, "progress").replace("{:>3}", &format!("{:>3}", i + 1)));
                io::stdout().flush().unwrap();
            }
            println!(); // Newline after progress

            let elapsed = start_time.elapsed();
            // Recalculate total sims actually run (integer division might lose a few, but negligible)
            let total_sims_run = batch_size_prob * batches;
            
            let prob_line = format_f2p_probability_line(total_sims_run, total_with_up_agg);
            println!("{}", prob_line);
            println!(
                "{}",
                I18n::get(lang, "expected_up").replace("{:.2}", &format!("{:.2}", total_up_agg as f64 / total_sims_run as f64))
            );
            println!("{}", I18n::get(lang, "time_taken").replace("{:.2?}", &format!("{:.2?}", elapsed)));
            println!(
                "{}",
                I18n::get(lang, "throughput").replace("{:.0}", &format!("{:.0}", total_sims_run as f64 / elapsed.as_secs_f64()))
            );

            println!("{}", I18n::get(lang, "calc_cost"));
            println!(
                "{}",
                I18n::get(lang, "sys_run_cost").replace("{}", &sim_count_cost.to_string())
            );
            
            let batch_size_cost = sim_count_cost / batches;
            let mut total_extra_cost_agg = 0u64;
            let mut extra_cost_samples_agg = 0usize;

            for i in 0..batches {
                let (cost_sum, samples, _) = simulate_f2p_clearing(
                    batch_size_cost,
                    rng.next_u64(),
                    &trained_neural_opt,
                    Some(&dqn_policy),
                    Some(&ppo_policy),
                    &dbn,
                    &config,
                    &worker,
                    None,
                    None,
                    None,
                );
                total_extra_cost_agg += cost_sum;
                extra_cost_samples_agg += samples;
                
                print!("\r{}", I18n::get(lang, "progress").replace("{:>3}", &format!("{:>3}", i + 1)));
                io::stdout().flush().unwrap();
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
    }
}

fn run_interactive(
    config: Config,
    dbn: Dbn,
    trained_neural_opt: NeuralLuckOptimizer,
    dqn_policy: DuelingQNetwork,
    ppo_policy: ActorCritic,
    worker: GoodJobWorker,
    mut rng: Rng,
    lang: Language,
) {
    let dqn_shared = Arc::new(RwLock::new(dqn_policy.clone()));
    let neural_shared = Arc::new(RwLock::new(trained_neural_opt.clone()));
    let ppo_shared = Arc::new(RwLock::new(ppo_policy.clone()));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut online_handles: Vec<thread::JoinHandle<()>> = Vec::new();
    let mut dqn_sender: Option<mpsc::Sender<Experience>> = None;
    let mut neural_sender: Option<mpsc::Sender<NeuralSample>> = None;
    let mut ppo_sender: Option<mpsc::Sender<PpoExperience>> = None;

    if config.online_train && config.online_train_dqn && config.luck_mode == "dqn" {
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
            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let mut drained = 0usize;
                while let Ok(exp) = rx.try_recv() {
                    trainer.push(exp);
                    drained += 1;
                    if drained > 4096 {
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
                if steps > 0 {
                    trainer.sync_to(&shared);
                    if last_report.elapsed().as_secs_f64() >= 2.0 {
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
        let max_steps = config.max_train_steps_per_tick.max(1);
        let mut trainer = OnlineNeuralTrainer::from_model(trained_neural_opt.clone());
        online_handles.push(thread::spawn(move || {
            let mut last_report = Instant::now();
            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let mut drained = 0usize;
                while let Ok(sample) = rx.try_recv() {
                    let _ = trainer.train_step(&sample);
                    drained += 1;
                    if drained > 4096 {
                        break;
                    }
                }
                if trainer.steps_done() > 0 {
                    trainer.sync_to(&shared);
                    if last_report.elapsed().as_secs_f64() >= 2.0 {
                        info!("[Online Neural] steps={}", trainer.steps_done());
                        last_report = Instant::now();
                    }
                }
                thread::sleep(Duration::from_millis(interval_ms));
                if max_steps == 0 {
                    break;
                }
            }
        }));
    }

    if config.online_train && config.online_train_ppo && config.luck_mode == "ppo" {
        let (tx, rx) = mpsc::channel::<PpoExperience>();
        ppo_sender = Some(tx);
        let shared = Arc::clone(&ppo_shared);
        let stop = Arc::clone(&stop_flag);
        let interval_ms = (config.train_interval_ms.max(1) as u64).max(5);
        let max_steps = config.max_train_steps_per_tick.max(1);
        let trainer_seed = rng.next_u64();
        let mut trainer = OnlinePpoTrainer::new(trainer_seed, 2, 128);
        online_handles.push(thread::spawn(move || {
            let mut last_report = Instant::now();
            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let mut drained = 0usize;
                while let Ok(exp) = rx.try_recv() {
                    trainer.push(exp);
                    drained += 1;
                    if drained > 4096 {
                        break;
                    }
                }
                let mut steps = 0usize;
                while steps < max_steps {
                    if trainer.train_step(0.0003) {
                        steps += 1;
                    } else {
                        break;
                    }
                }
                if steps > 0 {
                    trainer.sync_to(&shared);
                    if last_report.elapsed().as_secs_f64() >= 2.0 {
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
    println!("{}", I18n::get(lang, "insight_header"));
    let rl_w = trained_neural_opt.linear_weights;
    let rl_b = trained_neural_opt.linear_bias;

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
    println!("  - {:<25}: {:>8.4} {}", I18n::get(lang, "lbl_base_bias"), rl_b, I18n::get(lang, "impact_base"));

    // === Ask User for Interaction Mode ===
    let use_ppo = prompt_yes_no(
        &I18n::get(lang, "prompt_ppo"),
        true,
    );

    println!("{}", I18n::get(lang, "header_title"));
    println!("{}", I18n::get(lang, "header_pool").replace("{}", &config.pool_name));
    println!("{}", I18n::get(lang, "header_up").replace("{}", &config.up_six.join(", ")));
    println!(
        "{}",
        I18n::get(lang, "header_prob")
            .replace("{:.1}", &format!("{:.1}", config.prob_6_base * 100.0))
            .replace("{}", &config.soft_pity_start.to_string())
    );
    println!(
        "{}",
        I18n::get(lang, "header_rules").replace("{}", &config.small_pity_guarantee.to_string())
    );
    println!(
        "{}",
        I18n::get(lang, "header_big_pity").replace("{}", &config.big_pity_cumulative.to_string())
    );
    println!(
        "{}",
        I18n::get(lang, "header_economy")
            .replacen("{}", &COST_PER_PULL.to_string(), 1)
            .replacen("{}", &FREE_PULLS_WELFARE.to_string(), 1)
    );
    println!("{}", I18n::get(lang, "header_neural"));

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
    println!(
        "{}",
        I18n::get(lang, "f2p_header").replace("{}", &FREE_PULLS_WELFARE.to_string())
    );

    // Adjust simulation count based on build profile to prevent hanging in Debug mode
    #[cfg(debug_assertions)]
    let sim_count_prob = if config.f2p_sim_count_prob > 0 {
        config.f2p_sim_count_prob
    } else if config.f2p_sim_count > 0 {
        config.f2p_sim_count
    } else if config.fast_init {
        2_000
    } else {
        10_000
    };
    #[cfg(not(debug_assertions))]
    let sim_count_prob = if config.f2p_sim_count_prob > 0 {
        config.f2p_sim_count_prob
    } else if config.f2p_sim_count > 0 {
        config.f2p_sim_count
    } else if config.fast_init {
        200_000
    } else {
        1_000_000
    };
    #[cfg(debug_assertions)]
    let sim_count_cost = if config.f2p_sim_count_cost > 0 {
        config.f2p_sim_count_cost
    } else if config.f2p_sim_count > 0 {
        config.f2p_sim_count
    } else if sim_count_prob >= 50_000 {
        sim_count_prob / 2
    } else {
        sim_count_prob
    };
    #[cfg(not(debug_assertions))]
    let sim_count_cost = if config.f2p_sim_count_cost > 0 {
        config.f2p_sim_count_cost
    } else if config.f2p_sim_count > 0 {
        config.f2p_sim_count
    } else if sim_count_prob >= 200_000 {
        sim_count_prob / 2
    } else {
        sim_count_prob
    };

    println!(
        "{}",
        I18n::get(lang, "sys_run_prob").replace("{}", &sim_count_prob.to_string())
    );

    let batches = 100;
            let batch_size_prob = sim_count_prob / batches;
            let mut total_up_agg = 0;
            let mut total_with_up_agg = 0;

            let start_time = Instant::now();
            // Fix: Pass FREE_PULLS_WELFARE as num_pulls instead of 0, otherwise simulation exits immediately!
            let dqn_guard = dqn_shared.read().unwrap();
            let neural_guard = neural_shared.read().unwrap();
            let ppo_guard = ppo_shared.read().unwrap();
            
            for i in 0..batches {
                let (_, total_up, _, total_with_up) = simulate_stats(
                    FREE_PULLS_WELFARE as usize,
                    batch_size_prob,
                    rng.next_u64(),
                    &neural_guard,
                    Some(&dqn_guard),
                    Some(&ppo_guard),
                    &dbn,
                    &config,
                    &worker,
                    None,
                    None,
                    None,
                );
                total_up_agg += total_up;
                total_with_up_agg += total_with_up;
                
                print!("\r{}", I18n::get(lang, "progress").replace("{:>3}", &format!("{:>3}", i + 1)));
                io::stdout().flush().unwrap();
            }
    println!();

    let elapsed = start_time.elapsed();
    let total_sims_run = batch_size_prob * batches;
    let prob_line = format_f2p_probability_line(total_sims_run, total_with_up_agg);
    println!("{}", prob_line);
    println!(
        "{}",
        I18n::get(lang, "expected_up").replace("{:.2}", &format!("{:.2}", total_up_agg as f64 / total_sims_run as f64))
    );
    println!("{}", I18n::get(lang, "time_taken").replace("{:.2?}", &format!("{:.2?}", elapsed)));
    println!(
        "{}",
        I18n::get(lang, "throughput").replace("{:.0}", &format!("{:.0}", total_sims_run as f64 / elapsed.as_secs_f64()))
    );

    println!("{}", I18n::get(lang, "calc_cost"));
    println!(
        "{}",
        I18n::get(lang, "sys_run_cost").replace("{}", &sim_count_cost.to_string())
    );
    
    let batch_size_cost = sim_count_cost / batches;
    let mut total_extra_cost_agg = 0u64;
    let mut extra_cost_samples_agg = 0usize;

    for i in 0..batches {
        let (cost_sum, samples, _) = simulate_f2p_clearing(
            batch_size_cost,
            rng.next_u64(),
            &neural_guard,
            Some(&dqn_guard),
            Some(&ppo_guard),
            &dbn,
            &config,
            &worker,
            None,
            None,
            None,
        );
        total_extra_cost_agg += cost_sum;
        extra_cost_samples_agg += samples;
        
        print!("\r{}", I18n::get(lang, "progress").replace("{:>3}", &format!("{:>3}", i + 1)));
        io::stdout().flush().unwrap();
    }
    println!();

    let avg_extra_cost = if extra_cost_samples_agg == 0 {
        None
    } else {
        Some(total_extra_cost_agg as f64 / extra_cost_samples_agg as f64)
    };
    let avg_cost_line = format_avg_extra_cost_line(avg_extra_cost);
    println!("{}", avg_cost_line);
    println!("{}", I18n::get(lang, "total_value"));
    drop(dqn_guard);
    drop(neural_guard);
    drop(ppo_guard);

    loop {
        print!("{}", I18n::get(lang, "prompt_pulls"));
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.eq_ignore_ascii_case("q") {
            println!("{}", I18n::get(lang, "exit_msg"));
            break;
        }

        let n = if input.is_empty() {
            10
        } else {
            match input.parse::<usize>() {
                Ok(val) => {
                    if val > 1_000_000 {
                        println!("{}", I18n::get(lang, "input_too_large"));
                        1_000_000
                    } else {
                        val
                    }
                }
                Err(_) => {
                    println!("{}", I18n::get(lang, "invalid_input"));
                    10
                }
            }
        };

        print!(
            "{}",
            I18n::get(lang, "prompt_welfare").replace("{}", &FREE_PULLS_WELFARE.to_string())
        );
        io::stdout().flush().unwrap();
        let mut w_input = String::new();
        io::stdin().read_line(&mut w_input).unwrap();
        let use_welfare = !w_input.trim().eq_ignore_ascii_case("n");
        let free_pulls = if use_welfare { FREE_PULLS_WELFARE } else { 0 };

        print!("{}", I18n::get(lang, "prompt_sim_count"));
        io::stdout().flush().unwrap();
        let mut sim_input = String::new();
        io::stdin().read_line(&mut sim_input).unwrap();
        let sim_input = sim_input.trim();

        let sims_n = if sim_input.is_empty() {
            1
        } else {
            match sim_input.parse::<usize>() {
                Ok(val) => {
                    if val > 1_000_000 {
                        println!("{}", I18n::get(lang, "sim_count_too_large"));
                        1_000_000
                    } else {
                        val
                    }
                }
                Err(_) => 1,
            }
        };

        if sims_n > 1 {
            let dqn_guard = dqn_shared.read().unwrap();
            let neural_guard = neural_shared.read().unwrap();
            let ppo_guard = ppo_shared.read().unwrap();
            let active_ppo = if use_ppo { Some(&*ppo_guard) } else { None };
            let (s_total, u_total, _, _) = simulate_stats(
                n,
                sims_n,
                rng.next_u64(),
                &neural_guard,
                Some(&dqn_guard),
                active_ppo,
                &dbn,
                &config,
                &worker,
                None,
                None,
                None,
            );
            let s_avg = s_total as f64 / sims_n as f64;
            let u_avg = u_total as f64 / sims_n as f64;
            println!(
                "{}",
                I18n::get(lang, "sim_result_stats")
                    .replacen("{}", &sims_n.to_string(), 1)
                    .replacen("{}", &n.to_string(), 1)
                    .replacen("{:.3}", &format!("{:.3}", s_avg), 1)
                    .replacen("{:.3}", &format!("{:.3}", u_avg), 1)
            );
        } else {
            let start_time = Instant::now();
            let dqn_guard = dqn_shared.read().unwrap();
            let neural_guard = neural_shared.read().unwrap();
            let ppo_guard = ppo_shared.read().unwrap();
            let active_ppo = if use_ppo { Some(&*ppo_guard) } else { None };
            let res = simulate_one(
                n,
                &mut rng,
                free_pulls,
                &neural_guard,
                Some(&dqn_guard),
                active_ppo,
                &dbn,
                &config,
                dqn_sender.as_ref(),
                neural_sender.as_ref(),
                ppo_sender.as_ref(),
            );
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
            // Show first 20 details
            for (i, p) in res.pulls.iter().take(20).enumerate() {
                let op_name = resolve_operator_name(p, &config, &non_up_six);
                if p.is_up {
                    println!(
                        "{}. {} ({} {}) [UP]",
                        i + 1,
                        op_name,
                        p.rarity,
                        I18n::get(lang, "unit_star")
                    );
                } else {
                    println!(
                        "{}. {} ({} {})",
                        i + 1,
                        op_name,
                        p.rarity,
                        I18n::get(lang, "unit_star")
                    );
                }
            }
            if res.pulls.len() > 20 {
                println!("... ({} more omitted)", res.pulls.len() - 20);
            }

            println!("--- Consumption ---");
            println!("Free Pulls Used: {}", res.free_pulls_used);
            println!(
                "Jade Spent: {} ({} pulls)",
                res.cost_jade,
                res.cost_jade / 500
            );
            if res.big_pity_used {
                println!("Big Pity Triggered!");
            }
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
    use sim::{simulate_core, simulate_fast, simulate_one, SimControl}; // Updated import

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
        let res = simulate_fast(
            num_pulls,
            &mut rng,
            free_pulls,
            &neural_opt,
            None,
            None,
            &dbn,
            &config,
            None,
            None,
            None,
        );
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
        let res = simulate_one(
            num_pulls,
            &mut rng,
            free_pulls,
            &neural_opt,
            None,
            None,
            &dbn,
            &config,
            None,
            None,
            None,
        );
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
        let (stats, _) = simulate_core(
            &control,
            &mut rng,
            FREE_PULLS_WELFARE,
            &neural_opt,
            None,
            None,
            &dbn,
            &config,
            None,
            None,
            None,
        );
        assert!(stats.up_count > 0);
    }
}
