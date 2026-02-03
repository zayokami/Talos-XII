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

use autograd::Tensor as AutoTensor;
use clap::{Parser, Subcommand};
use config::Config;
use dbn::Dbn;
use dqn::{train_dqn, DuelingQNetwork, Experience, OnlineDqnTrainer};
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
        "[Bench] simulate_fast: {} sims of {} pulls in {:.2?} ({:.0} sims/sec)",
        fast_sims,
        fast_pulls,
        fast_elapsed,
        fast_sims as f64 / fast_elapsed.as_secs_f64()
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
        "[Bench] simulate_one: {} sims of {} pulls in {:.2?} ({:.0} sims/sec)",
        one_sims,
        one_pulls,
        one_elapsed,
        one_sims as f64 / one_elapsed.as_secs_f64()
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
            println!("Simulations: {}, Pulls per sim: {}", count, pulls);
            println!("Avg 6-Star: {:.4}", six_count as f64 / count as f64);
            println!("Avg UP: {:.4}", up_count as f64 / count as f64);
        }
        Commands::Benchmark => {
            benchmark_simulation(
                &mut rng,
                &trained_neural_opt,
                Some(&dqn_policy),
                Some(&ppo_policy),
                &dbn,
                &config,
            );
            demo_mmap_tensor();
        }
        Commands::F2p => {
            println!(
                "\n=== F2P Welfare Analysis ({} Free Pulls) ===",
                FREE_PULLS_WELFARE
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
                "[System] Running {} simulations for probability...",
                sim_count_prob
            );
            let start_time = Instant::now();
            let (_, total_up, _, total_with_up) = simulate_stats(
                FREE_PULLS_WELFARE as usize,
                sim_count_prob,
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
            let elapsed = start_time.elapsed();
            let prob_line = format_f2p_probability_line(sim_count_prob, total_with_up);
            println!("{}", prob_line);
            println!(
                "Expected UP Count: {:.2}",
                total_up as f64 / sim_count_prob as f64
            );
            println!("Time taken: {:.2?}", elapsed);
            println!(
                "Throughput: {:.0} sims/sec",
                sim_count_prob as f64 / elapsed.as_secs_f64()
            );

            println!("\nCalculating average EXTRA cost for F2P players to get UP...");
            let (avg_extra_cost, _) = simulate_f2p_clearing(
                sim_count_cost,
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
    println!("\n[Model Insight] Linear Manifold Analysis:");
    let rl_w = trained_neural_opt.linear_weights;
    let rl_b = trained_neural_opt.linear_bias;

    let feature_names = [
        "Pity Progress (0-1)",
        "Total Pulls Norm",
        "Env Noise",
        "Loss Streak Norm",
        "4-Star Streak Norm",
        "Env Bias",
        "Pity * Loss (Interaction)",
        "Total Pulls (Quadratic)",
    ];
    for (i, name) in feature_names.iter().enumerate() {
        let w = rl_w[i];
        let impact = if w.abs() < 0.001 {
            "Neutral"
        } else if w > 0.0 {
            "Boost Luck"
        } else {
            "Reduce Luck"
        };
        println!("  - {:<25}: {:>8.4} [{}]", name, w, impact);
    }
    println!("  - {:<25}: {:>8.4} [Bias]", "Base Bias", rl_b);

    // === Ask User for Interaction Mode ===
    let use_ppo = prompt_yes_no(
        "[System] Use PPO (Transformer) Brain for simulation? (y/n): ",
        true,
    );

    println!("=== Talos-XII Wish Simulator (Neural-Evolutionary) ===");
    println!("Pool Name: {}", config.pool_name);
    println!("UP Operator(s): {}", config.up_six.join(", "));
    println!(
        "Probabilities: 6-Star {:.1}% (Soft Pity start at {}, +5%/pull)",
        config.prob_6_base * 100.0,
        config.soft_pity_start
    );
    println!(
        "Rules: {} Pulls Guarantee 6* (50/50 UP, No Guarantee on Loss)",
        config.small_pity_guarantee
    );
    println!(
        "Big Pity: Cumulative {} pulls guarantee UP (Once per pool)",
        config.big_pity_cumulative
    );
    println!(
        "Economy: {} Jade/Pull | ~{} Free Pulls (Welfare)",
        COST_PER_PULL, FREE_PULLS_WELFARE
    );
    println!("Neural Core: Online (Evolved for Luck Balancing)");

    println!("\n[System] PRNG Initialized: xoshiro256**");
    if cfg!(debug_assertions) && !config.fast_init {
        println!("\n[System] Benchmarking simulation throughput...");
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
        );
        demo_mmap_tensor();
    }

    // F2P Analysis
    println!(
        "\n=== F2P Welfare Analysis ({} Free Pulls) ===",
        FREE_PULLS_WELFARE
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
        "[System] Running {} simulations for probability...",
        sim_count_prob
    );

    let start_time = Instant::now();
    // Fix: Pass FREE_PULLS_WELFARE as num_pulls instead of 0, otherwise simulation exits immediately!
    let dqn_guard = dqn_shared.read().unwrap();
    let neural_guard = neural_shared.read().unwrap();
    let ppo_guard = ppo_shared.read().unwrap();
    let (_, total_up, _, total_with_up) = simulate_stats(
        FREE_PULLS_WELFARE as usize,
        sim_count_prob,
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
    let elapsed = start_time.elapsed();
    let prob_line = format_f2p_probability_line(sim_count_prob, total_with_up);
    println!("{}", prob_line);
    println!(
        "Expected UP Count: {:.2}",
        total_up as f64 / sim_count_prob as f64
    );
    println!("Time taken: {:.2?}", elapsed);
    println!(
        "Throughput: {:.0} sims/sec",
        sim_count_prob as f64 / elapsed.as_secs_f64()
    );

    println!("\nCalculating average EXTRA cost for F2P players to get UP...");
    let (avg_extra_cost, _) = simulate_f2p_clearing(
        sim_count_cost,
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
    let avg_cost_line = format_avg_extra_cost_line(avg_extra_cost);
    println!("{}", avg_cost_line);
    println!("Total Value ~ 41000 Jade (Expected Cost for First UP)");
    drop(dqn_guard);
    drop(neural_guard);
    drop(ppo_guard);

    loop {
        print!("\nEnter number of pulls (default 10, or 'q' to quit): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.eq_ignore_ascii_case("q") {
            println!("Exiting. Goodbye!");
            break;
        }

        let n = if input.is_empty() {
            10
        } else {
            match input.parse::<usize>() {
                Ok(val) => {
                    if val > 1_000_000 {
                        println!("Input too large, capped at 1,000,000 to prevent memory issues.");
                        1_000_000
                    } else {
                        val
                    }
                }
                Err(_) => {
                    println!("Invalid input. Using default 10.");
                    10
                }
            }
        };

        print!(
            "Use Welfare Resources ({} pulls)? (y/n, default y): ",
            FREE_PULLS_WELFARE
        );
        io::stdout().flush().unwrap();
        let mut w_input = String::new();
        io::stdin().read_line(&mut w_input).unwrap();
        let use_welfare = !w_input.trim().eq_ignore_ascii_case("n");
        let free_pulls = if use_welfare { FREE_PULLS_WELFARE } else { 0 };

        print!("Enter simulation count (default 1, max 1M): ");
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
                        println!(
                            "Simulation count too large, capped at 1,000,000 to prevent CPU hang."
                        );
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
                "\n{} simulations of {}-pulls: Avg 6-Star {:.3} | UP {:.3}",
                sims_n, n, s_avg, u_avg
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
            println!("\nSingle {}-pull result (Time: {:.2?}):", n, elapsed);
            println!("6-Star: {} | UP: {}", res.six_count, res.up_count);
            let non_up_six = build_non_up_six(&config);
            // Show first 20 details
            for (i, p) in res.pulls.iter().take(20).enumerate() {
                let op_name = resolve_operator_name(p, &config, &non_up_six);
                if p.is_up {
                    println!("{}. {} ({} Star) [UP]", i + 1, op_name, p.rarity);
                } else {
                    println!("{}. {} ({} Star)", i + 1, op_name, p.rarity);
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
