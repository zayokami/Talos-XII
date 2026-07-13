use crate::calibrate::{apply_calibration, CalibrationData};
use crate::config::{ComputeDevice, Config};
use crate::dqn::{train_dqn, DuelingQNetwork};
use crate::env_net::EnvNet;
use crate::model_io::{
    cache_artifact_hash, dqn_inference_cache_manifest, dqn_master_cache_manifest,
    env_net_cache_manifest, load_env_net_cache_with_manifest, load_model_with_manifest,
    load_model_with_manifest_allow_source_mismatch, load_neural_cache_with_manifest,
    model_artifact_hash, neural_cache_manifest, ppo_inference_cache_manifest,
    ppo_master_cache_manifest, save_env_net_cache_with_manifest, save_model_with_manifest,
    save_neural_cache_with_manifest, CacheQualitySummary,
};
use crate::neural::NeuralLuckOptimizer;
use crate::ppo::{train_ppo, ActorCritic};
use crate::rng::Rng;
use crate::trainer::{train_linear_regression, train_manifold_rl, train_neural_optimizer};
use crate::worker::GoodJobWorker;
use log::{error, info, warn};

pub const DQN_MASTER_CACHE_PATH: &str = "dqn.cache";
pub const DQN_INFERENCE_CACHE_PATH: &str = "dqn.cache.bf16";
pub const PPO_MASTER_CACHE_PATH: &str = "ppo.cache";
pub const PPO_INFERENCE_CACHE_PATH: &str = "ppo.cache.bf16";

pub const ENV_NET_CACHE_PATH: &str = "env_net.cache";
pub const NEURAL_CACHE_PATH: &str = "neural.cache";

const UNAVAILABLE_SOURCE_HASH: &str = "unavailable";

#[derive(Clone, Copy, Debug, Default)]
pub struct ModelInitOptions {
    pub force: bool,
    pub allow_online_bootstrap: bool,
}

pub fn initialize_system(
    config_path: &str,
    seed: Option<u64>,
    options: ModelInitOptions,
) -> (
    Config,
    EnvNet,
    NeuralLuckOptimizer,
    DuelingQNetwork,
    ActorCritic,
    GoodJobWorker,
    Rng,
) {
    let mut config = resolve_runtime_config(config_path);
    apply_compute_device_policy(&mut config);
    if config.model_hidden_dim >= 8192 {
        warn!(
            "Large model detected ({} dim x {} layers). Training will take significantly longer and may require substantial memory.",
            config.model_hidden_dim, config.model_num_layers
        );
    }

    let mut rng = build_rng(seed);
    let worker = build_worker(&config);
    let env_net = build_env_net(&mut rng, &config, options.force);
    let env_net_hash = required_source_hash(cache_artifact_hash(ENV_NET_CACHE_PATH));
    let trained_neural_opt = build_trained_neural_opt(
        &mut rng,
        &env_net,
        &config,
        &worker,
        options.force,
        env_net_hash.clone(),
    );
    let neural_hash = required_source_hash(cache_artifact_hash(NEURAL_CACHE_PATH));
    let (dqn_master, dqn_master_rebuilt) = build_dqn_master(
        &mut rng,
        &env_net,
        &trained_neural_opt,
        &config,
        options,
        neural_hash,
    );
    let dqn_policy = prepare_dqn_gpu_policy(
        &dqn_master,
        options.force || dqn_master_rebuilt,
        config.device,
        &config,
    );
    let (ppo_master, ppo_master_rebuilt) =
        build_ppo_master(&mut rng, &env_net, &config, options.force, env_net_hash);
    let ppo_policy = prepare_ppo_gpu_policy(
        &ppo_master,
        options.force || ppo_master_rebuilt,
        config.device,
        &config,
    );

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

fn resolve_runtime_config(config_path: &str) -> Config {
    let config = Config::load(config_path);
    if !config.use_calibrated {
        return config;
    }

    let calibration = CalibrationData::load(&config.calibrated_path);
    apply_runtime_calibration(config, calibration)
}

fn apply_runtime_calibration(mut config: Config, calibration: CalibrationData) -> Config {
    if calibration.pools.is_empty() {
        return config;
    }

    let applied_pools = apply_calibration(&mut config, &calibration);
    if applied_pools > 0 {
        info!(
            "[Calibration] Applied calibrated parameters to {} pool(s) before model initialization.",
            applied_pools
        );
    } else {
        warn!("[Calibration] No valid overrides matched the configured pools.");
    }
    config
}

fn build_rng(seed: Option<u64>) -> Rng {
    if let Some(seed) = seed {
        Rng::from_seed(seed)
    } else {
        Rng::new()
    }
}

pub(crate) fn required_source_hash(hash: Option<String>) -> Option<String> {
    Some(hash.unwrap_or_else(|| UNAVAILABLE_SOURCE_HASH.to_string()))
}

fn build_worker(config: &Config) -> GoodJobWorker {
    match GoodJobWorker::new_with_config(config) {
        Ok(worker) => worker,
        Err(err) => {
            error!(
                "Worker initialization failed: {}. Running without worker pool.",
                err
            );
            // Preserve functionality with a single-thread fallback if the configured pool fails.
            match GoodJobWorker::new(1) {
                Ok(worker) => worker,
                Err(_) => {
                    error!("Fallback worker also failed. Exiting.");
                    std::process::exit(1);
                }
            }
        }
    }
}

fn apply_compute_device_policy(config: &mut Config) {
    match config.device {
        ComputeDevice::Cpu => {
            info!("[Device] Using CPU backend.");
        }
        ComputeDevice::Auto => {
            #[cfg(cuda)]
            {
                match crate::cuda::device_count() {
                    Ok(count) if count > 0 => {
                        if let Ok(dev) = crate::cuda::get_device_info(0) {
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
                        info!(
                            "[Device] Auto requested, but no CUDA devices found. Falling back to CPU."
                        );
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
                match crate::cuda::device_count() {
                    Ok(count) if count > 0 => {
                        if let Ok(dev) = crate::cuda::get_device_info(0) {
                            info!(
                                "[Device] CUDA requested: {} (CC {}.{})",
                                dev.name, dev.compute_capability.0, dev.compute_capability.1
                            );
                        } else {
                            info!("[Device] CUDA requested and initialized.");
                        }
                    }
                    Ok(_) => {
                        info!(
                            "[Device] CUDA requested, but no CUDA devices found. Falling back to CPU."
                        );
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

fn build_env_net(rng: &mut Rng, config: &Config, force: bool) -> EnvNet {
    if !force {
        if let Some(cached) = load_env_net_cache_with_manifest(
            ENV_NET_CACHE_PATH,
            config,
            &env_net_cache_manifest(config),
        ) {
            info!("[EnvNet] Cache loaded.");
            return cached;
        }

        info!("[EnvNet] Pre-training environment noise model...");
        let mut env_net = EnvNet::new(rng);
        let (count, epochs) = resolve_env_net_pretrain_counts(config.fast_init);
        env_net.pretrain(rng, config, count, epochs);
        if save_env_net_cache_with_manifest(
            ENV_NET_CACHE_PATH,
            &env_net,
            env_net_cache_manifest(config).with_quality(CacheQualitySummary::note(format!(
                "{count}x{epochs} pretrain"
            ))),
        ) {
            info!("[EnvNet] Cache saved.");
        }
        return env_net;
    }

    info!("[EnvNet] Force pre-training...");
    let mut env_net = EnvNet::new(rng);
    env_net.pretrain(rng, config, 1024, 50);
    if save_env_net_cache_with_manifest(
        ENV_NET_CACHE_PATH,
        &env_net,
        env_net_cache_manifest(config)
            .with_quality(CacheQualitySummary::note("1024x50 forced pretrain")),
    ) {
        info!("[EnvNet] Cache saved.");
    }
    env_net
}

fn resolve_env_net_pretrain_counts(fast_init: bool) -> (usize, usize) {
    if fast_init {
        (256, 10)
    } else {
        (1024, 50)
    }
}

fn build_trained_neural_opt(
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    worker: &GoodJobWorker,
    force: bool,
    env_net_hash: Option<String>,
) -> NeuralLuckOptimizer {
    let neural_manifest = neural_cache_manifest(config).with_source_hash(env_net_hash);
    let mut trained_neural_opt = if !force {
        if let Some(cached) =
            load_neural_cache_with_manifest(NEURAL_CACHE_PATH, config, &neural_manifest)
        {
            info!("[Neural Core] Cache detected. Cached weights loaded.");
            return cached;
        }
        info!("[Neural Core] Cache not found. Training new weights...");
        train_neural_optimizer(rng.next_u64(), env_net, config, worker)
    } else {
        info!("[Neural Core] Force training new weights...");
        train_neural_optimizer(rng.next_u64(), env_net, config, worker)
    };

    info!("[Linear] Training linear regression...");
    let (lin_w, lin_b) = train_linear_regression(&trained_neural_opt, rng, env_net, config);
    trained_neural_opt.set_linear_params(lin_w, lin_b);

    info!("[RL] Manifold Optimization (Parallel)...");
    trained_neural_opt = train_manifold_rl(&trained_neural_opt, rng, env_net, config, worker);

    if save_neural_cache_with_manifest(
        NEURAL_CACHE_PATH,
        &trained_neural_opt,
        neural_manifest.with_quality(
            CacheQualitySummary::training_steps(1)
                .with_note("evolutionary + linear regression + manifold RL"),
        ),
    ) {
        info!("[Neural Core] Cache saved.");
    }

    trained_neural_opt
}

fn build_dqn_master(
    rng: &mut Rng,
    env_net: &EnvNet,
    trained_neural_opt: &NeuralLuckOptimizer,
    config: &Config,
    options: ModelInitOptions,
    neural_hash: Option<String>,
) -> (DuelingQNetwork, bool) {
    let online_dqn_allowed =
        options.allow_online_bootstrap && config.online_train && config.online_train_dqn;
    let load_quality = if online_dqn_allowed {
        CacheQualitySummary::online_updated("online training compatible master weights")
    } else {
        dqn_training_quality(config)
    };
    let dqn_master_manifest =
        dqn_master_cache_manifest(config, load_quality).with_source_hash(neural_hash.clone());
    let trained_dqn_manifest = dqn_master_cache_manifest(config, dqn_training_quality(config))
        .with_source_hash(neural_hash);
    if !options.force {
        if let Some(mut cached) = load_model_with_manifest_allow_source_mismatch::<DuelingQNetwork>(
            DQN_MASTER_CACHE_PATH,
            "DQN",
            config,
            &dqn_master_manifest,
        ) {
            cached.prune_achf(config.achf.prune_threshold);
            cached.freeze_achf_for_inference();
            info!("[DQN] Cached model loaded.");
            return (cached, false);
        }

        if online_dqn_allowed {
            info!("[DQN] Initializing online training model...");
            let d = DuelingQNetwork::new_with_config(config, rng.next_u64());
            let _ = save_model_with_manifest(
                &d,
                DQN_MASTER_CACHE_PATH,
                "DQN",
                trained_dqn_manifest.with_quality(CacheQualitySummary::online_bootstrap(
                    "online training initialized from random weights",
                )),
            );
            return (d, true);
        }

        info!("[DQN] Training new model...");
        let d = train_dqn(trained_neural_opt, rng, env_net, config);
        let _ = save_model_with_manifest(
            &d,
            DQN_MASTER_CACHE_PATH,
            "DQN",
            trained_dqn_manifest.clone(),
        );
        return (d, true);
    }

    if online_dqn_allowed {
        info!("[DQN] Force initializing online training model...");
        let d = DuelingQNetwork::new_with_config(config, rng.next_u64());
        let _ = save_model_with_manifest(
            &d,
            DQN_MASTER_CACHE_PATH,
            "DQN",
            trained_dqn_manifest.with_quality(CacheQualitySummary::online_bootstrap(
                "online training initialized from random weights",
            )),
        );
        return (d, true);
    }

    info!("[DQN] Force training new model...");
    let d = train_dqn(trained_neural_opt, rng, env_net, config);
    let _ = save_model_with_manifest(&d, DQN_MASTER_CACHE_PATH, "DQN", trained_dqn_manifest);
    (d, true)
}

#[cfg(cuda)]
fn prepare_dqn_gpu_policy(
    master: &DuelingQNetwork,
    force_refresh: bool,
    device: ComputeDevice,
    config: &Config,
) -> DuelingQNetwork {
    let mut policy = prepare_dqn_inference_cache(master, force_refresh, config);
    if device == ComputeDevice::Cuda {
        policy.to_cuda();
        info!("[DQN] BF16 inference cache moved to CUDA for Tensor Core matmul.");
    }
    policy
}

#[cfg(not(cuda))]
fn prepare_dqn_gpu_policy(
    master: &DuelingQNetwork,
    force_refresh: bool,
    _device: ComputeDevice,
    config: &Config,
) -> DuelingQNetwork {
    prepare_dqn_inference_cache(master, force_refresh, config)
}

fn prepare_dqn_inference_cache(
    master: &DuelingQNetwork,
    force_refresh: bool,
    config: &Config,
) -> DuelingQNetwork {
    let expected_manifest = dqn_inference_cache_manifest(
        config,
        required_source_hash(model_artifact_hash(DQN_MASTER_CACHE_PATH)),
    );
    if !force_refresh {
        if let Some(mut cached) = load_model_with_manifest::<DuelingQNetwork>(
            DQN_INFERENCE_CACHE_PATH,
            "DQN BF16",
            config,
            &expected_manifest,
        ) {
            cached.prune_achf(config.achf.prune_threshold);
            cached.freeze_achf_for_inference();
            info!("[DQN] BF16 inference cache loaded.");
            return cached;
        }
    }

    let mut bf16 = master.to_inference_bf16();
    bf16.prune_achf(config.achf.prune_threshold);
    bf16.freeze_achf_for_inference();
    let _ = save_model_with_manifest(
        &bf16,
        DQN_INFERENCE_CACHE_PATH,
        "DQN BF16",
        expected_manifest,
    );
    bf16
}

fn build_ppo_master(
    rng: &mut Rng,
    env_net: &EnvNet,
    config: &Config,
    force: bool,
    env_net_hash: Option<String>,
) -> (ActorCritic, bool) {
    let ppo_master_manifest = ppo_master_cache_manifest(config, ppo_training_quality(config))
        .with_source_hash(env_net_hash);
    if !force {
        if let Some(mut cached) = load_model_with_manifest_allow_source_mismatch::<ActorCritic>(
            PPO_MASTER_CACHE_PATH,
            "PPO",
            config,
            &ppo_master_manifest,
        ) {
            cached.prune_achf(config.achf.prune_threshold);
            cached.freeze_achf_for_inference();
            info!("[PPO] Cached model loaded.");
            return (cached, false);
        }

        info!("[PPO] Training new model...");
        let p = train_ppo(rng, env_net, config);
        println!("[PPO] Saving model...");
        let _ = save_model_with_manifest(
            &p,
            PPO_MASTER_CACHE_PATH,
            "PPO",
            ppo_master_manifest.clone(),
        );
        return (p, true);
    }

    info!("[PPO] Force training new model...");
    let p = train_ppo(rng, env_net, config);
    println!("[PPO] Saving model...");
    let _ = save_model_with_manifest(
        &p,
        PPO_MASTER_CACHE_PATH,
        "PPO",
        ppo_master_manifest.clone(),
    );
    (p, true)
}

#[cfg(cuda)]
fn prepare_ppo_gpu_policy(
    master: &ActorCritic,
    force_refresh: bool,
    device: ComputeDevice,
    config: &Config,
) -> ActorCritic {
    let mut policy = prepare_ppo_inference_cache(master, force_refresh, config);
    if device == ComputeDevice::Cuda {
        policy.to_cuda();
        info!("[PPO] BF16 inference cache moved to CUDA for Tensor Core matmul.");
    }
    policy
}

#[cfg(not(cuda))]
fn prepare_ppo_gpu_policy(
    master: &ActorCritic,
    force_refresh: bool,
    _device: ComputeDevice,
    config: &Config,
) -> ActorCritic {
    prepare_ppo_inference_cache(master, force_refresh, config)
}

fn prepare_ppo_inference_cache(
    master: &ActorCritic,
    force_refresh: bool,
    config: &Config,
) -> ActorCritic {
    let expected_manifest = ppo_inference_cache_manifest(
        config,
        required_source_hash(model_artifact_hash(PPO_MASTER_CACHE_PATH)),
    );
    if !force_refresh {
        if let Some(mut cached) = load_model_with_manifest::<ActorCritic>(
            PPO_INFERENCE_CACHE_PATH,
            "PPO BF16",
            config,
            &expected_manifest,
        ) {
            cached.prune_achf(config.achf.prune_threshold);
            cached.freeze_achf_for_inference();
            info!("[PPO] BF16 inference cache loaded.");
            return cached;
        }
    }

    let mut bf16 = master.to_inference_bf16();
    bf16.prune_achf(config.achf.prune_threshold);
    bf16.freeze_achf_for_inference();
    let _ = save_model_with_manifest(
        &bf16,
        PPO_INFERENCE_CACHE_PATH,
        "PPO BF16",
        expected_manifest,
    );
    bf16
}

pub(crate) fn dqn_training_quality(config: &Config) -> CacheQualitySummary {
    CacheQualitySummary::training_steps(if config.fast_init { 5_000 } else { 50_000 })
}

pub(crate) fn ppo_training_quality(config: &Config) -> CacheQualitySummary {
    let fast_mode = config.fast_init || config.ppo_mode == "fast";
    let steps = if config.ppo_total_steps > 0 {
        config.ppo_total_steps
    } else if fast_mode {
        4_000
    } else {
        20_000
    };
    CacheQualitySummary::training_steps(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::PoolCalibration;
    use crate::model_io::env_net_cache_manifest;

    #[test]
    fn resolve_env_net_pretrain_counts_uses_fast_init_schedule() {
        assert_eq!(resolve_env_net_pretrain_counts(true), (256, 10));
        assert_eq!(resolve_env_net_pretrain_counts(false), (1024, 50));
    }

    #[test]
    fn runtime_calibration_precedes_model_cache_fingerprinting() {
        let config = Config {
            active_pool: Some("pool-a".to_string()),
            use_calibrated: true,
            prob_6_base: 0.008,
            soft_pity_slope: 0.05,
            up_rate: 0.5,
            ..Config::default()
        };
        let uncalibrated_fingerprint = env_net_cache_manifest(&config).config_fingerprint;
        let mut calibration = CalibrationData::default();
        calibration.pools.insert(
            "pool-a".to_string(),
            PoolCalibration {
                pool_id: "pool-a".to_string(),
                prob_6_base: Some(0.009),
                soft_pity_slope: Some(0.06),
                up_rate: Some(0.6),
                sample_pulls: 10_000,
                sample_six_stars: 100,
            },
        );

        let resolved = apply_runtime_calibration(config, calibration);
        let calibrated_fingerprint = env_net_cache_manifest(&resolved).config_fingerprint;

        assert_eq!(resolved.prob_6_base, 0.009);
        assert_eq!(resolved.soft_pity_slope, 0.06);
        assert_eq!(resolved.up_rate, 0.6);
        assert_ne!(uncalibrated_fingerprint, calibrated_fingerprint);
    }
}
