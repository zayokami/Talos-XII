use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

const EMBEDDED_CONFIG: &str = include_str!("../data/config.json");
const EMBEDDED_POOLS: &str = include_str!("../data/pools.json");
const DEFAULT_CONFIG_PATH: &str = "data/config.json";
const DEFAULT_SOFT_PITY_START: usize = 65;
const DEFAULT_SMALL_PITY_GUARANTEE: usize = 80;
const DEFAULT_BIG_PITY_CUMULATIVE: usize = 120;
const MAX_SMALL_PITY_GUARANTEE: usize = 10_000;
const MAX_BIG_PITY_CUMULATIVE: usize = 100_000;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Json {
        document: String,
        source: serde_json::Error,
    },
    Validation {
        document: String,
        errors: Vec<String>,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "failed to read '{}': {source}", path.display())
            }
            Self::Json { document, source } => {
                write!(formatter, "invalid JSON/schema in {document}: {source}")
            }
            Self::Validation { document, errors } => {
                writeln!(formatter, "configuration validation failed for {document}:")?;
                for error in errors {
                    writeln!(formatter, "  - {error}")?;
                }
                Ok(())
            }
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Validation { .. } => None,
        }
    }
}

fn strip_json_comments(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let characters: Vec<char> = input.chars().collect();
    let mut index = 0;
    let mut in_string = false;

    while index < characters.len() {
        if in_string {
            output.push(characters[index]);
            if characters[index] == '\\' && index + 1 < characters.len() {
                index += 1;
                output.push(characters[index]);
            } else if characters[index] == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }

        if characters[index] == '"' {
            in_string = true;
            output.push(characters[index]);
            index += 1;
        } else if characters[index] == '/'
            && index + 1 < characters.len()
            && characters[index + 1] == '/'
        {
            while index < characters.len() && characters[index] != '\n' {
                index += 1;
            }
        } else if characters[index] == '/'
            && index + 1 < characters.len()
            && characters[index + 1] == '*'
        {
            index += 2;
            while index + 1 < characters.len()
                && !(characters[index] == '*' && characters[index + 1] == '/')
            {
                index += 1;
            }
            index = (index + 2).min(characters.len());
        } else {
            output.push(characters[index]);
            index += 1;
        }
    }

    output
}

/// Determines which policy drives the luck factor during simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LuckMode {
    Probability,
    Dqn,
    Ppo,
}

impl LuckMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Probability => "probability",
            Self::Dqn => "dqn",
            Self::Ppo => "ppo",
        }
    }
}

/// Requested compute device for neural-network operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Auto,
}

impl ComputeDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Auto => "auto",
        }
    }
}

/// Configuration for Adaptive Cache-aware Hyper-Connections (ACHF).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AchfConfig {
    pub enabled: bool,
    pub mode: String,
    pub candidate_mode: String,
    pub candidate_refresh_freq: usize,
    pub proj_mode: String,
    #[serde(alias = "proj_freq")]
    pub ortho_penalty_freq: usize,
    pub proj_steps: usize,
    pub lambda_ortho: f64,
    pub gate_mode: String,
    pub gate_momentum: f64,
    pub gate_beta: f64,
    pub gate_alpha: f64,
    pub g_min: f64,
    pub gate_warmup_steps: usize,
    pub gate_transition_steps: usize,
    pub gate_k_clip: f64,
    pub g_target_min: f64,
    pub g_target_max: f64,
    pub g_min_adapt_rate: f64,
    pub g_min_momentum: f64,
    pub cache_min_rows: usize,
    pub cache_min_nonzero_ratio: f64,
    pub cache_min_reuse: usize,
    pub path_warmup_samples: usize,
    pub path_min_dwell: usize,
    pub cache_sparsity_sample_rows: usize,
    pub cache_cost_bias: f64,
    pub cache_adapt_rate: f64,
    pub cache_bias_min: f64,
    pub cache_bias_max: f64,
    pub cache_latency_ema: f64,
    pub cache_latency_long_ema: f64,
    pub cache_adapt_blend: f64,
    pub cache_latency_sample_every: u64,
    pub cache_log_interval_steps: usize,
    pub cache_log_per_layer: bool,
    pub diagnostics_enabled: bool,
    pub rank: usize,
    pub prune_threshold: f64,
    pub candidate_target_sparsity: f64,
    pub candidate_min_sparsity: f64,
    pub candidate_max_relative_error: f64,
    pub candidate_max_output_relative_error: f64,
    pub candidate_min_calibration_samples: usize,
    pub candidate_calibration_steps: usize,
    pub candidate_calibration_lr: f64,
    pub candidate_calibration_max_samples: usize,
    pub candidate_train_from_scratch: bool,
    #[serde(alias = "candidate_discrepancy_momentum")]
    pub candidate_weight_error_momentum: f64,
    pub apply_attn: bool,
    pub apply_ffn: bool,
    pub apply_dqn: bool,
    pub infer_gate: String,
    pub adaptive_inference: bool,
}

impl Default for AchfConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "lite".to_string(),
            candidate_mode: "sparse".to_string(),
            candidate_refresh_freq: 1,
            proj_mode: "sinkhorn".to_string(),
            ortho_penalty_freq: 8,
            proj_steps: 0,
            lambda_ortho: 0.0,
            gate_mode: "grad_ema".to_string(),
            gate_momentum: 0.95,
            gate_beta: 0.7,
            gate_alpha: 0.0,
            g_min: 0.2,
            gate_warmup_steps: 100,
            gate_transition_steps: 50,
            gate_k_clip: 0.0,
            g_target_min: 0.3,
            g_target_max: 0.8,
            g_min_adapt_rate: 0.0,
            g_min_momentum: 0.9,
            cache_min_rows: 1,
            cache_min_nonzero_ratio: 0.0,
            cache_min_reuse: 2,
            path_warmup_samples: 2,
            path_min_dwell: 2,
            cache_sparsity_sample_rows: 0,
            cache_cost_bias: 1.0,
            cache_adapt_rate: 0.0,
            cache_bias_min: 0.2,
            cache_bias_max: 5.0,
            cache_latency_ema: 0.9,
            cache_latency_long_ema: 0.99,
            cache_adapt_blend: 0.5,
            cache_latency_sample_every: 1,
            cache_log_interval_steps: 0,
            cache_log_per_layer: false,
            diagnostics_enabled: false,
            rank: 0,
            prune_threshold: 0.01,
            candidate_target_sparsity: 0.0,
            candidate_min_sparsity: 0.5,
            candidate_max_relative_error: 0.05,
            candidate_max_output_relative_error: 0.05,
            candidate_min_calibration_samples: 0,
            candidate_calibration_steps: 256,
            candidate_calibration_lr: 1e-3,
            candidate_calibration_max_samples: 256,
            candidate_train_from_scratch: false,
            candidate_weight_error_momentum: 0.9,
            apply_attn: true,
            apply_ffn: true,
            apply_dqn: false,
            infer_gate: "candidate".to_string(),
            adaptive_inference: false,
        }
    }
}

impl AchfConfig {
    pub fn uses_adaptive_inference(&self) -> bool {
        matches!(self.mode.as_str(), "full" | "plain_ema") || self.adaptive_inference
    }

    pub fn uses_frozen_cached_fast_path(&self) -> bool {
        matches!(self.mode.as_str(), "lite" | "fixed_cached") && !self.adaptive_inference
    }

    fn validate(&self, errors: &mut Vec<String>) {
        validate_choice(
            errors,
            "achf.mode",
            &self.mode,
            &[
                "lite",
                "full",
                "fixed_cached",
                "fixed_sparse",
                "fixed_dense",
                "plain_ema",
            ],
        );
        validate_choice(
            errors,
            "achf.candidate_mode",
            &self.candidate_mode,
            &["none", "sparse", "low_rank"],
        );
        validate_choice(
            errors,
            "achf.proj_mode",
            &self.proj_mode,
            &["none", "rowcol", "sinkhorn"],
        );
        validate_choice(
            errors,
            "achf.gate_mode",
            &self.gate_mode,
            &["grad_ema", "fim_trace"],
        );
        validate_choice(
            errors,
            "achf.infer_gate",
            &self.infer_gate,
            &["candidate", "reference", "last", "g_min"],
        );

        validate_non_negative(errors, "achf.lambda_ortho", self.lambda_ortho);
        validate_finite(errors, "achf.gate_alpha", self.gate_alpha);
        validate_finite(errors, "achf.gate_beta", self.gate_beta);
        validate_unit(errors, "achf.gate_momentum", self.gate_momentum);
        validate_unit(errors, "achf.g_min", self.g_min);
        validate_non_negative(errors, "achf.gate_k_clip", self.gate_k_clip);
        validate_unit(errors, "achf.g_target_min", self.g_target_min);
        validate_unit(errors, "achf.g_target_max", self.g_target_max);
        validate_non_negative(errors, "achf.g_min_adapt_rate", self.g_min_adapt_rate);
        validate_unit(errors, "achf.g_min_momentum", self.g_min_momentum);
        validate_unit(
            errors,
            "achf.cache_min_nonzero_ratio",
            self.cache_min_nonzero_ratio,
        );
        validate_positive(errors, "achf.cache_cost_bias", self.cache_cost_bias);
        validate_non_negative(errors, "achf.cache_adapt_rate", self.cache_adapt_rate);
        validate_positive(errors, "achf.cache_bias_min", self.cache_bias_min);
        validate_positive(errors, "achf.cache_bias_max", self.cache_bias_max);
        validate_unit(errors, "achf.cache_latency_ema", self.cache_latency_ema);
        validate_unit(
            errors,
            "achf.cache_latency_long_ema",
            self.cache_latency_long_ema,
        );
        validate_unit(errors, "achf.cache_adapt_blend", self.cache_adapt_blend);
        validate_non_negative(errors, "achf.prune_threshold", self.prune_threshold);
        validate_unit(
            errors,
            "achf.candidate_target_sparsity",
            self.candidate_target_sparsity,
        );
        validate_unit(
            errors,
            "achf.candidate_min_sparsity",
            self.candidate_min_sparsity,
        );
        validate_unit(
            errors,
            "achf.candidate_max_relative_error",
            self.candidate_max_relative_error,
        );
        validate_unit(
            errors,
            "achf.candidate_max_output_relative_error",
            self.candidate_max_output_relative_error,
        );
        validate_positive(
            errors,
            "achf.candidate_calibration_lr",
            self.candidate_calibration_lr,
        );
        validate_unit(
            errors,
            "achf.candidate_weight_error_momentum",
            self.candidate_weight_error_momentum,
        );

        if self.g_target_max < self.g_target_min {
            errors.push("achf.g_target_max must be >= achf.g_target_min".to_string());
        }
        if self.cache_bias_max < self.cache_bias_min {
            errors.push("achf.cache_bias_max must be >= achf.cache_bias_min".to_string());
        }
        if self.candidate_min_calibration_samples > self.candidate_calibration_max_samples {
            errors.push(
                "achf.candidate_min_calibration_samples must be <= achf.candidate_calibration_max_samples"
                    .to_string(),
            );
        }
        if self.proj_mode != "none" && self.lambda_ortho > 0.0 {
            errors.push(
                "achf.lambda_ortho cannot be combined with rowcol/sinkhorn projection".to_string(),
            );
        }
        if self.candidate_mode == "sparse" && self.rank > 0 {
            errors.push("achf.rank is only valid for candidate_mode=low_rank".to_string());
        }
        if self.candidate_mode == "low_rank"
            && (self.prune_threshold > 0.0 || self.candidate_target_sparsity > 0.0)
        {
            errors.push(
                "low-rank candidates cannot use sparse pruning or target sparsity".to_string(),
            );
        }
        if self.candidate_mode != "sparse" && self.candidate_train_from_scratch {
            errors.push(
                "achf.candidate_train_from_scratch requires candidate_mode=sparse".to_string(),
            );
        }
        if self.candidate_train_from_scratch
            && (self.candidate_min_calibration_samples > 0 || self.candidate_calibration_steps > 0)
        {
            errors.push(
                "sparse training from scratch cannot be combined with post-training calibration"
                    .to_string(),
            );
        }
        if self.candidate_mode != "sparse" && self.candidate_min_calibration_samples > 0 {
            errors.push(
                "ACHF candidate calibration currently requires candidate_mode=sparse".to_string(),
            );
        }
        if self.candidate_target_sparsity > 0.0 && self.prune_threshold > 0.0 {
            errors.push(
                "achf.candidate_target_sparsity and achf.prune_threshold are mutually exclusive"
                    .to_string(),
            );
        }
        if self.adaptive_inference {
            errors.push(
                "achf.adaptive_inference is a legacy alias; use achf.mode=full explicitly"
                    .to_string(),
            );
        }
    }
}

fn default_soft_pity_slope() -> f64 {
    0.05
}

/// Per-pool configuration defining gacha rules and operator rosters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    pub id: String,
    pub name: String,
    pub pool_type: String,
    pub up_six: Vec<String>,
    pub up_rate: f64,
    pub prob_6_base: f64,
    pub prob_5_base: f64,
    pub prob_4_base: f64,
    pub soft_pity_start: usize,
    #[serde(default = "default_soft_pity_slope")]
    pub soft_pity_slope: f64,
    pub small_pity_guarantee: usize,
    pub big_pity_cumulative: usize,
    pub up_pity_soft: usize,
    pub five_star_pity: usize,
    pub always_5_star: bool,
    pub big_pity_requires_not_up: bool,
    pub six_stars: Vec<String>,
    pub five_stars: Vec<String>,
    pub four_stars: Vec<String>,
    #[serde(default)]
    pub is_archived: bool,
}

/// Complete runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub pool_name: String,
    pub up_six: Vec<String>,
    pub up_rate: f64,
    pub prob_6_base: f64,
    pub prob_5_base: f64,
    pub prob_4_base: f64,
    pub soft_pity_start: usize,
    pub soft_pity_slope: f64,
    pub small_pity_guarantee: usize,
    pub big_pity_cumulative: usize,
    pub up_pity_soft: usize,
    pub five_star_pity: usize,
    pub always_5_star: bool,
    pub big_pity_requires_not_up: bool,
    pub six_stars: Vec<String>,
    pub five_stars: Vec<String>,
    pub four_stars: Vec<String>,
    pub pools: Vec<PoolConfig>,
    pub active_pool: Option<String>,
    pub pools_path: String,
    pub luck_mode: LuckMode,
    pub use_calibrated: bool,
    pub calibrated_path: String,
    pub player_data_path: String,
    pub fast_init: bool,
    pub device: ComputeDevice,
    pub ppo_mode: String,
    pub ppo_total_steps: usize,
    pub ppo_steps_per_update: usize,
    pub ppo_k_epochs: usize,
    pub ppo_batch_size: usize,
    pub ppo_context_len: usize,
    pub ppo_num_envs: usize,
    pub ppo_top_k: usize,
    pub luck_action_cost: f64,
    pub luck_budget_enabled: bool,
    pub luck_budget_max: f64,
    pub luck_budget_initial: f64,
    pub luck_budget_recovery_per_pull: f64,
    pub luck_budget_negative_refund: f64,
    pub policy_eval_interval: usize,
    pub policy_eval_episodes: usize,
    pub policy_eval_seed: u64,
    pub distill_enabled: bool,
    pub distill_ema_decay: f64,
    pub distill_kl_coef: f64,
    pub distill_warmup_steps: usize,
    pub worker_max_threads: usize,
    pub worker_reserve_cores: usize,
    pub worker_priority: String,
    pub worker_stack_size_mb: usize,
    pub f2p_sim_count: usize,
    pub f2p_sim_count_prob: usize,
    pub f2p_sim_count_cost: usize,
    pub f2p_luck_mode: Option<LuckMode>,
    pub online_train: bool,
    pub online_train_dqn: bool,
    pub online_train_neural: bool,
    pub online_train_ppo: bool,
    pub train_interval_ms: usize,
    pub max_train_steps_per_tick: usize,
    pub language: Option<String>,
    pub model_dim: usize,
    pub model_hidden_dim: usize,
    pub model_num_layers: usize,
    pub model_num_heads: usize,
    pub model_kv_lora_rank: usize,
    pub model_qk_rope_dim: usize,
    pub use_multi_stream: bool,
    pub multi_stream_factor: usize,
    pub achf: AchfConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pool_name: "Unknown".to_string(),
            up_six: Vec::new(),
            up_rate: 0.5,
            prob_6_base: 0.008,
            prob_5_base: 0.08,
            prob_4_base: 0.912,
            soft_pity_start: DEFAULT_SOFT_PITY_START,
            soft_pity_slope: default_soft_pity_slope(),
            small_pity_guarantee: DEFAULT_SMALL_PITY_GUARANTEE,
            big_pity_cumulative: DEFAULT_BIG_PITY_CUMULATIVE,
            up_pity_soft: 0,
            five_star_pity: 10,
            always_5_star: false,
            big_pity_requires_not_up: true,
            six_stars: Vec::new(),
            five_stars: Vec::new(),
            four_stars: Vec::new(),
            pools: Vec::new(),
            active_pool: None,
            pools_path: "pools.json".to_string(),
            luck_mode: LuckMode::Probability,
            use_calibrated: true,
            calibrated_path: "calibrated.json".to_string(),
            player_data_path: "player_data.json".to_string(),
            fast_init: false,
            device: ComputeDevice::Auto,
            ppo_mode: "balanced".to_string(),
            ppo_total_steps: 0,
            ppo_steps_per_update: 0,
            ppo_k_epochs: 0,
            ppo_batch_size: 0,
            ppo_context_len: 0,
            ppo_num_envs: 1,
            ppo_top_k: 0,
            luck_action_cost: 8.0,
            luck_budget_enabled: true,
            luck_budget_max: 0.045,
            luck_budget_initial: 0.03,
            luck_budget_recovery_per_pull: 0.001,
            luck_budget_negative_refund: 1.0,
            policy_eval_interval: 0,
            policy_eval_episodes: 128,
            policy_eval_seed: 0x5EED_1234,
            distill_enabled: false,
            distill_ema_decay: 0.995,
            distill_kl_coef: 0.1,
            distill_warmup_steps: 500,
            worker_max_threads: 0,
            worker_reserve_cores: 1,
            worker_priority: "above_normal".to_string(),
            worker_stack_size_mb: 4,
            f2p_sim_count: 0,
            f2p_sim_count_prob: 0,
            f2p_sim_count_cost: 0,
            f2p_luck_mode: None,
            online_train: false,
            online_train_dqn: false,
            online_train_neural: false,
            online_train_ppo: false,
            train_interval_ms: 50,
            max_train_steps_per_tick: 1,
            language: None,
            model_dim: 32,
            model_hidden_dim: 1024,
            model_num_layers: 4,
            model_num_heads: 8,
            model_kv_lora_rank: 128,
            model_qk_rope_dim: 64,
            use_multi_stream: true,
            multi_stream_factor: 2,
            achf: AchfConfig::default(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PoolsDocument {
    active_pool: Option<String>,
    pools: Vec<PoolConfig>,
}

impl Config {
    /// Load and validate a configuration. Only the standard default path may
    /// fall back to the resources embedded in the executable.
    pub fn try_load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let requested = path.as_ref();
        let use_embedded = requested == Path::new("default");
        let is_standard_path = requested == Path::new(DEFAULT_CONFIG_PATH);

        if use_embedded {
            return Self::load_documents(
                EMBEDDED_CONFIG,
                "<embedded config>",
                Path::new("data"),
                true,
            );
        }

        match fs::read_to_string(requested) {
            Ok(contents) => {
                let base_dir = requested.parent().unwrap_or_else(|| Path::new("."));
                Self::load_documents(
                    &contents,
                    &requested.display().to_string(),
                    base_dir,
                    is_standard_path,
                )
            }
            Err(source) if source.kind() == io::ErrorKind::NotFound && is_standard_path => {
                Self::load_documents(
                    EMBEDDED_CONFIG,
                    "<embedded config: data/config.json missing>",
                    Path::new("data"),
                    true,
                )
            }
            Err(source) => Err(ConfigError::Io {
                path: requested.to_path_buf(),
                source,
            }),
        }
    }

    /// Parse a validated configuration from memory. Relative paths are
    /// resolved against `base_dir`; omitted pool data uses embedded defaults.
    pub fn try_from_json(contents: &str, base_dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::load_documents(contents, "<in-memory config>", base_dir.as_ref(), true)
    }

    /// Compatibility convenience for internal callers and tests. Unlike the
    /// previous loader, this never returns a silently repaired configuration.
    pub fn load(path: &str) -> Self {
        Self::try_load(path).unwrap_or_else(|error| panic!("{error}"))
    }

    fn load_documents(
        contents: &str,
        document: &str,
        base_dir: &Path,
        allow_embedded_pools: bool,
    ) -> Result<Self, ConfigError> {
        let mut root = parse_json(contents, document)?;
        let root_object = root_object_mut(&mut root, document)?;
        remove_documentation_fields(root_object);
        let pools_path_was_explicit = root_object.contains_key("pools_path");
        let has_embedded_pools = root_object.contains_key("pools");
        let language = root_object
            .get("language")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        prepare_localized_fields(root_object, language.as_deref());

        let mut config: Config =
            serde_json::from_value(root).map_err(|source| ConfigError::Json {
                document: document.to_string(),
                source,
            })?;

        if config.pools_path.trim().is_empty() {
            return Err(ConfigError::Validation {
                document: document.to_string(),
                errors: vec!["pools_path must not be empty".to_string()],
            });
        }

        let resolved_pools_path = resolve_relative_path(base_dir, &config.pools_path);
        if !has_embedded_pools {
            let (pool_contents, pool_document) = match fs::read_to_string(&resolved_pools_path) {
                Ok(contents) => (contents, resolved_pools_path.display().to_string()),
                Err(source)
                    if source.kind() == io::ErrorKind::NotFound
                        && (allow_embedded_pools || !pools_path_was_explicit) =>
                {
                    (EMBEDDED_POOLS.to_string(), "<embedded pools>".to_string())
                }
                Err(source) => {
                    return Err(ConfigError::Io {
                        path: resolved_pools_path,
                        source,
                    });
                }
            };
            let mut pools_root = parse_json(&pool_contents, &pool_document)?;
            let pools_object = root_object_mut(&mut pools_root, &pool_document)?;
            remove_documentation_fields(pools_object);
            prepare_localized_pool_entries(pools_object, language.as_deref());
            let pools: PoolsDocument =
                serde_json::from_value(pools_root).map_err(|source| ConfigError::Json {
                    document: pool_document,
                    source,
                })?;
            config.active_pool = pools.active_pool;
            config.pools = pools.pools;
        }

        config.pools_path = path_to_string(&resolved_pools_path);
        config.calibrated_path =
            path_to_string(&resolve_relative_path(base_dir, &config.calibrated_path));
        config.player_data_path =
            path_to_string(&resolve_relative_path(base_dir, &config.player_data_path));

        let mut selection_errors = Vec::new();
        if config.pools.is_empty() {
            selection_errors.push("at least one pool must be configured".to_string());
        } else if let Some(active_pool) = config.active_pool.clone() {
            if !config.apply_pool(&active_pool) {
                selection_errors.push(format!(
                    "active_pool '{}' does not match any configured pool ID",
                    active_pool
                ));
            }
        } else {
            selection_errors.push("active_pool is required when pools are configured".to_string());
        }
        if !selection_errors.is_empty() {
            return Err(ConfigError::Validation {
                document: document.to_string(),
                errors: selection_errors,
            });
        }

        config.validate_named(document)?;
        Ok(config)
    }

    /// Validate a programmatically constructed configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_named("<runtime configuration>")
    }

    /// Switch the active pool by ID, updating all pool-specific settings.
    pub fn apply_pool(&mut self, pool_id: &str) -> bool {
        let Some(pool) = self.pools.iter().find(|pool| pool.id == pool_id).cloned() else {
            return false;
        };
        self.pool_name = pool.name;
        self.up_six = pool.up_six;
        self.up_rate = pool.up_rate;
        self.prob_6_base = pool.prob_6_base;
        self.prob_5_base = pool.prob_5_base;
        self.prob_4_base = pool.prob_4_base;
        self.soft_pity_start = pool.soft_pity_start;
        self.soft_pity_slope = pool.soft_pity_slope;
        self.small_pity_guarantee = pool.small_pity_guarantee;
        self.big_pity_cumulative = pool.big_pity_cumulative;
        self.up_pity_soft = pool.up_pity_soft;
        self.five_star_pity = pool.five_star_pity;
        self.always_5_star = pool.always_5_star;
        self.big_pity_requires_not_up = pool.big_pity_requires_not_up;
        self.six_stars = pool.six_stars;
        self.five_stars = pool.five_stars;
        self.four_stars = pool.four_stars;
        self.active_pool = Some(pool_id.to_string());
        true
    }
}

fn parse_json(contents: &str, document: &str) -> Result<Value, ConfigError> {
    serde_json::from_str(&strip_json_comments(contents)).map_err(|source| ConfigError::Json {
        document: document.to_string(),
        source,
    })
}

fn root_object_mut<'a>(
    root: &'a mut Value,
    document: &str,
) -> Result<&'a mut serde_json::Map<String, Value>, ConfigError> {
    root.as_object_mut().ok_or_else(|| ConfigError::Validation {
        document: document.to_string(),
        errors: vec!["document root must be a JSON object".to_string()],
    })
}

fn remove_documentation_fields(object: &mut serde_json::Map<String, Value>) {
    object.retain(|key, _| !key.starts_with("_comment"));
}

fn prepare_localized_fields(object: &mut serde_json::Map<String, Value>, language: Option<&str>) {
    localize_field(object, "pool_name", "pool_name_en", language);
    localize_field(object, "up_six", "up_six_en", language);
    localize_field(object, "six_stars", "six_stars_en", language);
    localize_field(object, "five_stars", "five_stars_en", language);
    localize_field(object, "four_stars", "four_stars_en", language);
    prepare_localized_pool_entries(object, language);
}

fn prepare_localized_pool_entries(
    object: &mut serde_json::Map<String, Value>,
    language: Option<&str>,
) {
    let Some(Value::Array(pools)) = object.get_mut("pools") else {
        return;
    };
    for pool in pools {
        let Some(pool_object) = pool.as_object_mut() else {
            continue;
        };
        localize_field(pool_object, "name", "name_en", language);
        localize_field(pool_object, "up_six", "up_six_en", language);
        localize_field(pool_object, "six_stars", "six_stars_en", language);
        localize_field(pool_object, "five_stars", "five_stars_en", language);
        localize_field(pool_object, "four_stars", "four_stars_en", language);
    }
}

fn localize_field(
    object: &mut serde_json::Map<String, Value>,
    primary: &str,
    english: &str,
    language: Option<&str>,
) {
    if let Some(english_value) = object.remove(english) {
        if language == Some("en") && localization_value_is_usable(&english_value) {
            object.insert(primary.to_string(), english_value);
        }
    }
}

fn localization_value_is_usable(value: &Value) -> bool {
    match value {
        Value::String(value) => !value.is_empty(),
        Value::Array(values) => !values.is_empty(),
        _ => false,
    }
}

fn resolve_relative_path(base_dir: &Path, configured: &str) -> PathBuf {
    let path = Path::new(configured);
    if path.is_absolute() {
        normalize_path(path)
    } else {
        normalize_path(&base_dir.join(path))
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

impl Config {
    fn validate_named(&self, document: &str) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        validate_choice(
            &mut errors,
            "ppo_mode",
            &self.ppo_mode,
            &["auto", "fast", "balanced"],
        );
        validate_choice(
            &mut errors,
            "worker_priority",
            &self.worker_priority,
            &[
                "highest",
                "above_normal",
                "normal",
                "below_normal",
                "lowest",
                "idle",
            ],
        );
        if let Some(language) = &self.language {
            validate_choice(&mut errors, "language", language, &["zh", "en"]);
        }

        if self.model_dim == 0 {
            errors.push("model_dim must be greater than zero".to_string());
        }
        if self.model_hidden_dim == 0 {
            errors.push("model_hidden_dim must be greater than zero".to_string());
        }
        if self.model_num_layers == 0 {
            errors.push("model_num_layers must be greater than zero".to_string());
        }
        if self.model_num_heads == 0 {
            errors.push("model_num_heads must be greater than zero".to_string());
        } else if !self.model_hidden_dim.is_multiple_of(self.model_num_heads) {
            errors.push("model_hidden_dim must be divisible by model_num_heads".to_string());
        }
        if self.model_kv_lora_rank == 0 {
            errors.push("model_kv_lora_rank must be greater than zero".to_string());
        }
        if self.model_qk_rope_dim == 0 || !self.model_qk_rope_dim.is_multiple_of(2) {
            errors.push("model_qk_rope_dim must be a positive even number".to_string());
        }
        if self.use_multi_stream && self.multi_stream_factor < 2 {
            errors.push(
                "multi_stream_factor must be at least 2 when use_multi_stream=true".to_string(),
            );
        }
        if self.ppo_num_envs == 0 {
            errors.push("ppo_num_envs must be greater than zero".to_string());
        }
        if self.policy_eval_episodes == 0 {
            errors.push("policy_eval_episodes must be greater than zero".to_string());
        }
        if self.worker_stack_size_mb == 0 || self.worker_stack_size_mb > 512 {
            errors.push("worker_stack_size_mb must be in 1..=512".to_string());
        }
        if self.online_train && self.train_interval_ms == 0 {
            errors.push(
                "train_interval_ms must be greater than zero for online training".to_string(),
            );
        }
        if self.online_train && self.max_train_steps_per_tick == 0 {
            errors.push(
                "max_train_steps_per_tick must be greater than zero for online training"
                    .to_string(),
            );
        }

        validate_non_negative(&mut errors, "luck_action_cost", self.luck_action_cost);
        validate_non_negative(&mut errors, "luck_budget_max", self.luck_budget_max);
        validate_non_negative(&mut errors, "luck_budget_initial", self.luck_budget_initial);
        validate_non_negative(
            &mut errors,
            "luck_budget_recovery_per_pull",
            self.luck_budget_recovery_per_pull,
        );
        validate_non_negative(
            &mut errors,
            "luck_budget_negative_refund",
            self.luck_budget_negative_refund,
        );
        if self.luck_budget_initial > self.luck_budget_max {
            errors.push("luck_budget_initial must be <= luck_budget_max".to_string());
        }
        if self.luck_budget_enabled && self.luck_budget_max <= 0.0 {
            errors.push("luck_budget_max must be > 0 when luck budgeting is enabled".to_string());
        }
        validate_unit(&mut errors, "distill_ema_decay", self.distill_ema_decay);
        if self.distill_ema_decay == 1.0 {
            errors.push("distill_ema_decay must be less than 1".to_string());
        }
        validate_non_negative(&mut errors, "distill_kl_coef", self.distill_kl_coef);

        if self.calibrated_path.trim().is_empty() {
            errors.push("calibrated_path must not be empty".to_string());
        }
        if self.player_data_path.trim().is_empty() {
            errors.push("player_data_path must not be empty".to_string());
        }

        validate_probability_model(
            &mut errors,
            "active pool",
            self.up_rate,
            self.prob_6_base,
            self.prob_5_base,
            self.prob_4_base,
            self.soft_pity_start,
            self.soft_pity_slope,
            self.small_pity_guarantee,
            self.big_pity_cumulative,
            self.five_star_pity,
            self.up_six.is_empty(),
        );

        let mut pool_ids = HashSet::new();
        for (index, pool) in self.pools.iter().enumerate() {
            let scope = if pool.id.is_empty() {
                format!("pools[{index}]")
            } else {
                format!("pool '{}'", pool.id)
            };
            if pool.id.trim().is_empty() {
                errors.push(format!("{scope}.id must not be empty"));
            } else if !pool_ids.insert(pool.id.as_str()) {
                errors.push(format!("duplicate pool ID '{}'", pool.id));
            }
            if pool.name.trim().is_empty() {
                errors.push(format!("{scope}.name must not be empty"));
            }
            validate_choice(
                &mut errors,
                &format!("{scope}.pool_type"),
                &pool.pool_type,
                &["character_up", "weapon_up", "standard", "beginner"],
            );
            validate_probability_model(
                &mut errors,
                &scope,
                pool.up_rate,
                pool.prob_6_base,
                pool.prob_5_base,
                pool.prob_4_base,
                pool.soft_pity_start,
                pool.soft_pity_slope,
                pool.small_pity_guarantee,
                pool.big_pity_cumulative,
                pool.five_star_pity,
                pool.up_six.is_empty(),
            );
            let six_stars: HashSet<&str> = pool.six_stars.iter().map(String::as_str).collect();
            for operator in &pool.up_six {
                if !six_stars.contains(operator.as_str()) {
                    errors.push(format!(
                        "{scope}.up_six entry '{operator}' is missing from six_stars"
                    ));
                }
            }
        }

        if let Some(active_pool) = &self.active_pool {
            if !pool_ids.contains(active_pool.as_str()) {
                errors.push(format!(
                    "active_pool '{}' does not match any configured pool ID",
                    active_pool
                ));
            }
        } else {
            errors.push("active_pool is required".to_string());
        }

        self.achf.validate(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation {
                document: document.to_string(),
                errors,
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_probability_model(
    errors: &mut Vec<String>,
    scope: &str,
    up_rate: f64,
    probability_six: f64,
    probability_five: f64,
    probability_four: f64,
    soft_pity_start: usize,
    soft_pity_slope: f64,
    small_pity_guarantee: usize,
    big_pity_cumulative: usize,
    five_star_pity: usize,
    up_list_is_empty: bool,
) {
    validate_unit(errors, &format!("{scope}.up_rate"), up_rate);
    validate_unit(errors, &format!("{scope}.prob_6_base"), probability_six);
    validate_unit(errors, &format!("{scope}.prob_5_base"), probability_five);
    validate_unit(errors, &format!("{scope}.prob_4_base"), probability_four);
    let probability_sum = probability_six + probability_five + probability_four;
    if probability_sum.is_finite() && (probability_sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        errors.push(format!(
            "{scope} base probabilities must sum to 1 (got {probability_sum:.17})"
        ));
    }
    if up_list_is_empty && up_rate != 0.0 {
        errors.push(format!("{scope}.up_rate must be 0 when up_six is empty"));
    }
    if !up_list_is_empty && up_rate <= 0.0 {
        errors.push(format!(
            "{scope}.up_rate must be > 0 when up_six is non-empty"
        ));
    }
    if soft_pity_start == 0 {
        errors.push(format!("{scope}.soft_pity_start must be greater than zero"));
    }
    if small_pity_guarantee == 0 || small_pity_guarantee > MAX_SMALL_PITY_GUARANTEE {
        errors.push(format!(
            "{scope}.small_pity_guarantee must be in 1..={MAX_SMALL_PITY_GUARANTEE}"
        ));
    }
    if soft_pity_start > small_pity_guarantee {
        errors.push(format!(
            "{scope}.soft_pity_start must be <= small_pity_guarantee"
        ));
    }
    if big_pity_cumulative > MAX_BIG_PITY_CUMULATIVE {
        errors.push(format!(
            "{scope}.big_pity_cumulative must be <= {MAX_BIG_PITY_CUMULATIVE}"
        ));
    }
    if big_pity_cumulative > 0 && big_pity_cumulative < small_pity_guarantee {
        errors.push(format!(
            "{scope}.big_pity_cumulative must be 0 or >= small_pity_guarantee"
        ));
    }
    if five_star_pity == 0 {
        errors.push(format!("{scope}.five_star_pity must be greater than zero"));
    }
    validate_non_negative(errors, &format!("{scope}.soft_pity_slope"), soft_pity_slope);
}

fn validate_choice(errors: &mut Vec<String>, key: &str, value: &str, allowed: &[&str]) {
    if !allowed.contains(&value) {
        errors.push(format!(
            "{key} has unsupported value '{value}'; expected one of: {}",
            allowed.join(", ")
        ));
    }
}

fn validate_finite(errors: &mut Vec<String>, key: &str, value: f64) {
    if !value.is_finite() {
        errors.push(format!("{key} must be finite"));
    }
}

fn validate_non_negative(errors: &mut Vec<String>, key: &str, value: f64) {
    if !value.is_finite() || value < 0.0 {
        errors.push(format!("{key} must be finite and >= 0"));
    }
}

fn validate_positive(errors: &mut Vec<String>, key: &str, value: f64) {
    if !value.is_finite() || value <= 0.0 {
        errors.push(format!("{key} must be finite and > 0"));
    }
}

fn validate_unit(errors: &mut Vec<String>, key: &str, value: f64) {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        errors.push(format!("{key} must be finite and in [0, 1]"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_configuration_is_strictly_valid() {
        let config = Config::try_load(DEFAULT_CONFIG_PATH).unwrap();
        assert!(!config.pools.is_empty());
        assert!(config.active_pool.is_some());
    }

    #[test]
    fn default_keyword_uses_embedded_resources() {
        let config = Config::try_load("default").unwrap();
        assert!(!config.pools.is_empty());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let error = Config::try_from_json(r#"{"devcie":"cpu"}"#, ".").unwrap_err();
        assert!(error.to_string().contains("unknown field `devcie`"));
    }

    #[test]
    fn invalid_enum_values_are_rejected() {
        let error = Config::try_from_json(r#"{"device":"gpu"}"#, ".").unwrap_err();
        assert!(error.to_string().contains("unknown variant `gpu`"));
    }

    #[test]
    fn wrong_types_are_rejected() {
        let error = Config::try_from_json(r#"{"model_dim":"32"}"#, ".").unwrap_err();
        assert!(error.to_string().contains("invalid type"));
    }

    #[test]
    fn documentation_fields_are_explicitly_ignored() {
        let config = Config::try_from_json(
            r#"{"_comment":{"device":"cpu/cuda/auto"},"_comment_extra":"docs"}"#,
            ".",
        )
        .unwrap();
        assert!(!config.pools.is_empty());
    }

    #[test]
    fn relative_paths_are_based_on_config_directory() {
        let base = Path::new("custom/config");
        let config = Config::try_from_json(
            r#"{"calibrated_path":"models/calibrated.json","player_data_path":"players.json"}"#,
            base,
        )
        .unwrap();
        assert_eq!(
            Path::new(&config.calibrated_path),
            Path::new("custom/config/models/calibrated.json")
        );
        assert_eq!(
            Path::new(&config.player_data_path),
            Path::new("custom/config/players.json")
        );
    }

    #[test]
    fn invalid_achf_combinations_are_rejected() {
        let error = Config::try_from_json(
            r#"{"achf":{"proj_mode":"sinkhorn","lambda_ortho":0.1}}"#,
            ".",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("lambda_ortho cannot be combined"));
    }

    #[test]
    fn legacy_achf_field_aliases_are_parsed_without_silent_value_changes() {
        let config = Config::try_from_json(
            r#"{"achf":{"proj_freq":3,"candidate_discrepancy_momentum":0.7}}"#,
            ".",
        )
        .unwrap();
        assert_eq!(config.achf.ortho_penalty_freq, 3);
        assert_eq!(config.achf.candidate_weight_error_momentum, 0.7);
    }

    #[test]
    fn comments_inside_json_are_supported() {
        let config = Config::try_from_json("{/* comment */\"device\":\"cpu\"}", ".").unwrap();
        assert_eq!(config.device, ComputeDevice::Cpu);
    }
}
