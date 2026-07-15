use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub use serde_json::Value as JsonValue;

fn json_to_string_vec(v: &JsonValue) -> Vec<String> {
    match v {
        JsonValue::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

fn strip_json_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut in_string = false;

    while i < len {
        if in_string {
            out.push(chars[i]);
            if chars[i] == '\\' && i + 1 < len {
                i += 1;
                out.push(chars[i]);
            } else if chars[i] == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if chars[i] == '"' {
            in_string = true;
            out.push(chars[i]);
            i += 1;
        } else if chars[i] == '/' && i + 1 < len && chars[i + 1] == '/' {
            while i < len && chars[i] != '\n' {
                i += 1;
            }
        } else if chars[i] == '/' && i + 1 < len && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            if i + 1 < len {
                i += 2;
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

const DEFAULT_SOFT_PITY_START: usize = 65;
const DEFAULT_SMALL_PITY_GUARANTEE: usize = 80;
const DEFAULT_BIG_PITY_CUMULATIVE: usize = 120;
const MAX_SMALL_PITY_GUARANTEE: usize = 10_000;
const MAX_BIG_PITY_CUMULATIVE: usize = 100_000;

fn fallback_parent_path(path: &str, levels: usize) -> Option<String> {
    let requested = Path::new(path);
    if requested.is_absolute() || requested.components().any(|c| c == Component::ParentDir) {
        return None;
    }
    let mut p = PathBuf::new();
    for _ in 0..levels {
        p.push("..");
    }
    p.push(requested);
    Some(p.to_string_lossy().into_owned())
}

fn sanitize_pity_settings(
    soft_pity_start: &mut usize,
    small_pity_guarantee: &mut usize,
    big_pity_cumulative: &mut usize,
    scope: &str,
) {
    if *small_pity_guarantee == 0 {
        eprintln!(
            "[Config Warning] {scope} small_pity_guarantee is 0, fallback to {}",
            DEFAULT_SMALL_PITY_GUARANTEE
        );
        *small_pity_guarantee = DEFAULT_SMALL_PITY_GUARANTEE;
    }
    if *small_pity_guarantee > MAX_SMALL_PITY_GUARANTEE {
        eprintln!(
            "[Config Warning] {scope} small_pity_guarantee ({}) exceeds max {}, clamping",
            *small_pity_guarantee, MAX_SMALL_PITY_GUARANTEE
        );
        *small_pity_guarantee = MAX_SMALL_PITY_GUARANTEE;
    }

    if *soft_pity_start == 0 {
        eprintln!("[Config Warning] {scope} soft_pity_start is 0, fallback to 1");
        *soft_pity_start = 1;
    }

    if *soft_pity_start > *small_pity_guarantee {
        eprintln!(
            "[Config Warning] {scope} soft_pity_start ({}) exceeds small_pity_guarantee ({}), clamping",
            *soft_pity_start, *small_pity_guarantee
        );
        *soft_pity_start = *small_pity_guarantee;
    }

    if *big_pity_cumulative > 0 && *big_pity_cumulative < *small_pity_guarantee {
        eprintln!(
            "[Config Warning] {scope} big_pity_cumulative ({}) is below small_pity_guarantee ({}), clamping",
            *big_pity_cumulative, *small_pity_guarantee
        );
        *big_pity_cumulative = *small_pity_guarantee;
    }
    if *big_pity_cumulative > MAX_BIG_PITY_CUMULATIVE {
        eprintln!(
            "[Config Warning] {scope} big_pity_cumulative ({}) exceeds max {}, clamping",
            *big_pity_cumulative, MAX_BIG_PITY_CUMULATIVE
        );
        *big_pity_cumulative = MAX_BIG_PITY_CUMULATIVE;
    }
}

fn sanitize_unit_interval(value: &mut f64, key: &str, default: f64) {
    if !value.is_finite() {
        eprintln!(
            "[Config Warning] {key} is non-finite, fallback to {}",
            default
        );
        *value = default;
        return;
    }
    if *value < 0.0 {
        eprintln!("[Config Warning] {key} ({}) below 0, clamping", *value);
        *value = 0.0;
    } else if *value > 1.0 {
        eprintln!("[Config Warning] ACHF {key} ({}) above 1, clamping", *value);
        *value = 1.0;
    }
}

fn sanitize_non_negative(value: &mut f64, key: &str, default: f64) {
    if !value.is_finite() {
        eprintln!(
            "[Config Warning] ACHF {key} is non-finite, fallback to {}",
            default
        );
        *value = default;
        return;
    }
    if *value < 0.0 {
        eprintln!("[Config Warning] ACHF {key} ({}) below 0, clamping", *value);
        *value = 0.0;
    }
}

fn sanitize_positive(value: &mut f64, key: &str, default: f64) {
    if !value.is_finite() || *value <= 0.0 {
        eprintln!(
            "[Config Warning] ACHF {key} ({}) must be > 0, fallback to {}",
            *value, default
        );
        *value = default;
    }
}

fn sanitize_finite(value: &mut f64, key: &str, default: f64) {
    if !value.is_finite() {
        eprintln!(
            "[Config Warning] ACHF {key} is non-finite, fallback to {}",
            default
        );
        *value = default;
    }
}

fn sanitize_achf_config(achf: &mut AchfConfig) {
    let defaults = AchfConfig::default();
    let normalized_mode = achf.mode.trim().to_ascii_lowercase();
    match normalized_mode.as_str() {
        "lite" | "full" | "fixed_cached" | "fixed_sparse" | "fixed_dense" | "plain_ema" => {
            achf.mode = normalized_mode
        }
        _ => {
            eprintln!(
                "[Config Warning] ACHF mode '{}' is unsupported, fallback to '{}'",
                achf.mode, defaults.mode
            );
            achf.mode = defaults.mode;
        }
    }
    let normalized_candidate_mode = achf.candidate_mode.trim().to_ascii_lowercase();
    match normalized_candidate_mode.as_str() {
        "none" | "sparse" | "low_rank" => achf.candidate_mode = normalized_candidate_mode,
        _ => {
            eprintln!(
                "[Config Warning] ACHF candidate_mode '{}' is unsupported, fallback to '{}'",
                achf.candidate_mode, defaults.candidate_mode
            );
            achf.candidate_mode = defaults.candidate_mode.clone();
        }
    }
    let normalized_projection = achf.proj_mode.trim().to_ascii_lowercase();
    match normalized_projection.as_str() {
        "none" | "rowcol" | "sinkhorn" => achf.proj_mode = normalized_projection,
        _ => {
            eprintln!(
                "[Config Warning] ACHF proj_mode '{}' is unsupported, fallback to '{}'",
                achf.proj_mode, defaults.proj_mode
            );
            achf.proj_mode = defaults.proj_mode.clone();
        }
    }
    sanitize_non_negative(
        &mut achf.lambda_ortho,
        "lambda_ortho",
        defaults.lambda_ortho,
    );
    sanitize_finite(&mut achf.gate_alpha, "gate_alpha", defaults.gate_alpha);
    sanitize_finite(&mut achf.gate_beta, "gate_beta", defaults.gate_beta);
    if achf.proj_mode != "none" && achf.lambda_ortho > 0.0 {
        eprintln!(
            "[Config Warning] ACHF lambda_ortho cannot be stacked with '{}' on the dedicated connection map; disabling lambda_ortho",
            achf.proj_mode
        );
        achf.lambda_ortho = 0.0;
    }
    if achf.candidate_mode == "sparse" && achf.rank > 0 {
        eprintln!(
            "[Config Warning] ACHF rank is a low-rank candidate setting and cannot be stacked with sparse pruning; disabling rank"
        );
        achf.rank = 0;
    }
    if achf.candidate_mode == "low_rank" && achf.prune_threshold > 0.0 {
        eprintln!(
            "[Config Warning] ACHF prune_threshold cannot be stacked with a low-rank candidate; disabling pruning"
        );
        achf.prune_threshold = 0.0;
    }
    if achf.adaptive_inference && achf.mode != "full" {
        eprintln!(
            "[Config Warning] ACHF adaptive_inference=true is a legacy alias for mode='full'"
        );
        achf.mode = "full".to_string();
    }
    // The mode is canonical after sanitization. Clear the legacy alias so a
    // later runtime switch from full back to lite cannot remain stuck in AMA.
    achf.adaptive_inference = false;
    sanitize_unit_interval(
        &mut achf.gate_momentum,
        "gate_momentum",
        defaults.gate_momentum,
    );
    sanitize_unit_interval(&mut achf.g_min, "g_min", defaults.g_min);
    sanitize_unit_interval(
        &mut achf.g_target_min,
        "g_target_min",
        defaults.g_target_min,
    );
    sanitize_unit_interval(
        &mut achf.g_target_max,
        "g_target_max",
        defaults.g_target_max,
    );
    if achf.g_target_max < achf.g_target_min {
        eprintln!(
            "[Config Warning] ACHF g_target_max ({}) below g_target_min ({}), clamping",
            achf.g_target_max, achf.g_target_min
        );
        achf.g_target_max = achf.g_target_min;
    }
    sanitize_non_negative(
        &mut achf.g_min_adapt_rate,
        "g_min_adapt_rate",
        defaults.g_min_adapt_rate,
    );
    sanitize_unit_interval(
        &mut achf.g_min_momentum,
        "g_min_momentum",
        defaults.g_min_momentum,
    );
    sanitize_non_negative(&mut achf.gate_k_clip, "gate_k_clip", defaults.gate_k_clip);
    sanitize_unit_interval(
        &mut achf.cache_min_nonzero_ratio,
        "cache_min_nonzero_ratio",
        defaults.cache_min_nonzero_ratio,
    );
    sanitize_positive(
        &mut achf.cache_cost_bias,
        "cache_cost_bias",
        defaults.cache_cost_bias,
    );
    sanitize_non_negative(
        &mut achf.cache_adapt_rate,
        "cache_adapt_rate",
        defaults.cache_adapt_rate,
    );
    sanitize_positive(
        &mut achf.cache_bias_min,
        "cache_bias_min",
        defaults.cache_bias_min,
    );
    sanitize_positive(
        &mut achf.cache_bias_max,
        "cache_bias_max",
        defaults.cache_bias_max,
    );
    if achf.cache_bias_max < achf.cache_bias_min {
        eprintln!(
            "[Config Warning] ACHF cache_bias_max ({}) below cache_bias_min ({}), clamping",
            achf.cache_bias_max, achf.cache_bias_min
        );
        achf.cache_bias_max = achf.cache_bias_min;
    }
    sanitize_unit_interval(
        &mut achf.cache_latency_ema,
        "cache_latency_ema",
        defaults.cache_latency_ema,
    );
    sanitize_unit_interval(
        &mut achf.cache_latency_long_ema,
        "cache_latency_long_ema",
        defaults.cache_latency_long_ema,
    );
    sanitize_unit_interval(
        &mut achf.cache_adapt_blend,
        "cache_adapt_blend",
        defaults.cache_adapt_blend,
    );
    sanitize_non_negative(
        &mut achf.prune_threshold,
        "prune_threshold",
        defaults.prune_threshold,
    );
    sanitize_unit_interval(
        &mut achf.candidate_min_sparsity,
        "candidate_min_sparsity",
        defaults.candidate_min_sparsity,
    );
    sanitize_unit_interval(
        &mut achf.candidate_max_relative_error,
        "candidate_max_relative_error",
        defaults.candidate_max_relative_error,
    );
    sanitize_unit_interval(
        &mut achf.candidate_weight_error_momentum,
        "candidate_weight_error_momentum",
        defaults.candidate_weight_error_momentum,
    );
    let normalized_infer_gate = achf.infer_gate.trim().to_ascii_lowercase();
    achf.infer_gate = match normalized_infer_gate.as_str() {
        "candidate" | "reference" | "last" | "g_min" => normalized_infer_gate,
        "one" => "candidate".to_string(),
        _ => defaults.infer_gate,
    };
}
/// Determines which policy drives the luck factor during simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LuckMode {
    /// Default mode: neural network probability adjustment.
    Probability,
    /// Deep Q-Network selects discrete luck actions.
    Dqn,
    /// Proximal Policy Optimization (Actor-Critic) selects actions.
    Ppo,
}

impl LuckMode {
    /// Parse from a config string. Unrecognized values map to `Probability`.
    pub fn from_str(s: &str) -> Self {
        match s {
            "dqn" => Self::Dqn,
            "ppo" => Self::Ppo,
            _ => Self::Probability,
        }
    }

    /// Return the canonical string representation.
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Probability => "probability",
            Self::Dqn => "dqn",
            Self::Ppo => "ppo",
        }
    }
}

/// Compute device for neural network operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ComputeDevice {
    /// CPU (default).
    Cpu,
    /// NVIDIA CUDA GPU (when available).
    Cuda,
    /// Auto-select: CUDA if available, otherwise CPU.
    Auto,
}

impl ComputeDevice {
    /// Parse from a config string. Unrecognized values map to `Cpu`.
    pub fn from_str(s: &str) -> Self {
        match s {
            "cuda" => Self::Cuda,
            "auto" => Self::Auto,
            _ => Self::Cpu,
        }
    }

    /// Return the canonical string representation.
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Auto => "auto",
        }
    }
}

// --- Configuration (Data-Driven) ---

/// Configuration for Adaptive Cache-aware Hyper-Connections (ACHF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchfConfig {
    pub enabled: bool,
    pub mode: String,
    #[serde(default = "default_achf_candidate_mode")]
    pub candidate_mode: String,
    #[serde(default = "default_achf_candidate_refresh_freq")]
    pub candidate_refresh_freq: usize,
    pub proj_mode: String,
    #[serde(default = "default_achf_ortho_penalty_freq", alias = "proj_freq")]
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
    #[serde(default = "default_achf_path_warmup_samples")]
    pub path_warmup_samples: usize,
    #[serde(default = "default_achf_path_min_dwell")]
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
    #[serde(default = "default_achf_candidate_min_sparsity")]
    pub candidate_min_sparsity: f64,
    #[serde(default = "default_achf_candidate_max_relative_error")]
    pub candidate_max_relative_error: f64,
    #[serde(
        default = "default_achf_candidate_weight_error_momentum",
        alias = "candidate_discrepancy_momentum"
    )]
    pub candidate_weight_error_momentum: f64,
    pub apply_attn: bool,
    pub apply_ffn: bool,
    pub apply_dqn: bool,
    pub infer_gate: String,
    /// Legacy compatibility alias for `mode = "full"`. Config sanitization
    /// migrates this flag into `mode` and clears it; new callers should set
    /// `mode` directly.
    pub adaptive_inference: bool,
}

impl Default for AchfConfig {
    fn default() -> Self {
        AchfConfig {
            enabled: false,
            mode: "lite".to_string(),
            candidate_mode: default_achf_candidate_mode(),
            candidate_refresh_freq: default_achf_candidate_refresh_freq(),
            proj_mode: "sinkhorn".to_string(),
            ortho_penalty_freq: default_achf_ortho_penalty_freq(),
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
            path_warmup_samples: default_achf_path_warmup_samples(),
            path_min_dwell: default_achf_path_min_dwell(),
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
            candidate_min_sparsity: default_achf_candidate_min_sparsity(),
            candidate_max_relative_error: default_achf_candidate_max_relative_error(),
            candidate_weight_error_momentum: default_achf_candidate_weight_error_momentum(),
            apply_attn: true,
            apply_ffn: true,
            apply_dqn: false,
            infer_gate: "candidate".to_string(),
            adaptive_inference: false,
        }
    }
}

fn default_achf_candidate_mode() -> String {
    "sparse".to_string()
}

fn default_achf_candidate_refresh_freq() -> usize {
    1
}

fn default_achf_ortho_penalty_freq() -> usize {
    8
}

fn default_achf_path_warmup_samples() -> usize {
    2
}

fn default_achf_path_min_dwell() -> usize {
    2
}

fn default_achf_candidate_min_sparsity() -> f64 {
    0.5
}

fn default_achf_candidate_max_relative_error() -> f64 {
    0.05
}

fn default_achf_candidate_weight_error_momentum() -> f64 {
    0.9
}

impl AchfConfig {
    /// Whether frozen inference should keep the latency-adaptive AMA selector
    /// active. `adaptive_inference` remains a compatibility alias for configs
    /// constructed by older callers; sanitized configs canonicalize it to
    /// `mode = "full"`.
    pub fn uses_adaptive_inference(&self) -> bool {
        matches!(
            self.mode.trim().to_ascii_lowercase().as_str(),
            "full" | "plain_ema"
        ) || self.adaptive_inference
    }

    pub fn uses_frozen_cached_fast_path(&self) -> bool {
        matches!(
            self.mode.trim().to_ascii_lowercase().as_str(),
            "lite" | "fixed_cached"
        ) && !self.adaptive_inference
    }
}

/// Per-pool configuration defining gacha rules and operator rosters.
#[derive(Debug, Clone, Serialize)]
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
    pub is_archived: bool,
}

/// Configuration for the gacha simulation, loaded from JSON.
#[derive(Debug, Clone, Serialize)]
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
        Config {
            pool_name: "Unknown".to_string(),
            up_six: vec![],
            up_rate: 0.5,
            prob_6_base: 0.008,
            prob_5_base: 0.08,
            prob_4_base: 0.912,
            soft_pity_start: DEFAULT_SOFT_PITY_START,
            soft_pity_slope: 0.05,
            small_pity_guarantee: DEFAULT_SMALL_PITY_GUARANTEE,
            big_pity_cumulative: DEFAULT_BIG_PITY_CUMULATIVE,
            up_pity_soft: 0,
            five_star_pity: 10,
            always_5_star: false,
            big_pity_requires_not_up: true,
            six_stars: vec![],
            five_stars: vec![],
            four_stars: vec![],
            pools: vec![],
            active_pool: None,
            pools_path: "data/pools.json".to_string(),
            luck_mode: LuckMode::Probability,
            use_calibrated: true,
            calibrated_path: "data/calibrated.json".to_string(),
            player_data_path: "data/player_data.json".to_string(),
            fast_init: false,
            device: ComputeDevice::Auto,
            ppo_mode: "balanced".to_string(),
            ppo_total_steps: 0,
            ppo_steps_per_update: 0,
            ppo_k_epochs: 0,
            ppo_batch_size: 0,
            ppo_context_len: 0,
            ppo_num_envs: 1,
            ppo_top_k: 0, // 0 = disabled (full softmax), >0 = top-k truncation
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

impl Config {
    /// Load configuration from a JSON file path, falling back to defaults.
    pub fn load(path: &str) -> Self {
        if path == "default" {
            eprintln!("[System] Using built-in default configuration.");
            return Config::default();
        }

        let file_result = File::open(path);

        // Robustness: If file not found, try to look in parent directories
        // (useful for IDE/target builds: target/release/exe, target/cuda/release/exe, target/python/release/exe)
        let mut file = match file_result {
            Ok(f) => f,
            Err(_) => {
                // Try 2 and 3 levels up (standard cargo layout with custom target-dir)
                let mut found = None;
                for levels in [2, 3] {
                    if let Some(fallback) = fallback_parent_path(path, levels) {
                        if let Ok(f) = File::open(&fallback) {
                            println!(
                                "[System] Config found in parent directory ({} levels up).",
                                levels
                            );
                            found = Some(f);
                            break;
                        }
                    }
                }
                match found {
                    Some(f) => f,
                    None => {
                        eprintln!("\x1b[1;31m[Error]\x1b[0m Configuration file not found.");
                        eprintln!("  Looked at: './{path}' and parent directories (2-3 levels up)");
                        eprintln!(
                            "  Tip: Use --config <path> or --config default for built-in defaults."
                        );
                        if path == "data/config.json" {
                            eprintln!(
                                "\x1b[33m[Warning]\x1b[0m Missing data/config.json. Falling back to built-in defaults."
                            );
                            return Config::default();
                        }
                        std::process::exit(1);
                    }
                }
            }
        };

        let mut contents = String::new();
        if let Err(err) = file.read_to_string(&mut contents) {
            log::error!("Failed to read config file '{}': {}", path, err);
            log::warn!("Falling back to default configuration.");
            return Config::default();
        }

        let stripped = strip_json_comments(&contents);
        let root: JsonValue = match serde_json::from_str(&stripped) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("\x1b[1;31m[Error]\x1b[0m JSON parse error: {}", err);
                eprintln!("  Tip: Check for trailing commas, missing quotes, or invalid syntax.");
                std::process::exit(1);
            }
        };

        let mut config = Config::default();

        if let JsonValue::Object(ref map) = root {
            warn_unknown_fields(map);
            if let Some(v) = map.get("pools_path") {
                config.pools_path = v.as_str().unwrap_or("data/pools.json").to_string();
            }
            if let Some(v) = map.get("luck_mode") {
                config.luck_mode = LuckMode::from_str(v.as_str().unwrap_or("probability"));
            }
            if let Some(v) = map.get("fast_init") {
                config.fast_init = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("device") {
                config.device = ComputeDevice::from_str(v.as_str().unwrap_or("cpu"));
            }
            if let Some(v) = map.get("ppo_mode") {
                config.ppo_mode = v.as_str().unwrap_or("balanced").to_string();
            }
            if let Some(v) = map.get("ppo_total_steps") {
                config.ppo_total_steps = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("ppo_steps_per_update") {
                config.ppo_steps_per_update = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("ppo_k_epochs") {
                config.ppo_k_epochs = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("ppo_batch_size") {
                config.ppo_batch_size = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("ppo_context_len") {
                config.ppo_context_len = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("ppo_num_envs") {
                config.ppo_num_envs = v.as_f64().unwrap_or(1.0).round() as usize;
            }
            if let Some(v) = map.get("ppo_top_k") {
                config.ppo_top_k = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("luck_action_cost") {
                config.luck_action_cost = v.as_f64().unwrap_or(8.0);
            }
            if let Some(v) = map.get("luck_budget_enabled") {
                config.luck_budget_enabled = v.as_bool().unwrap_or(true);
            }
            if let Some(v) = map.get("luck_budget_max") {
                config.luck_budget_max = v.as_f64().unwrap_or(0.045);
            }
            if let Some(v) = map.get("luck_budget_initial") {
                config.luck_budget_initial = v.as_f64().unwrap_or(0.03);
            }
            if let Some(v) = map.get("luck_budget_recovery_per_pull") {
                config.luck_budget_recovery_per_pull = v.as_f64().unwrap_or(0.001);
            }
            if let Some(v) = map.get("luck_budget_negative_refund") {
                config.luck_budget_negative_refund = v.as_f64().unwrap_or(1.0);
            }
            if let Some(v) = map.get("policy_eval_interval") {
                config.policy_eval_interval = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("policy_eval_episodes") {
                config.policy_eval_episodes = v.as_f64().unwrap_or(128.0).round() as usize;
            }
            if let Some(v) = map.get("policy_eval_seed") {
                config.policy_eval_seed = v.as_f64().unwrap_or(0x5EED_1234 as f64).round() as u64;
            }
            if let Some(v) = map.get("distill_enabled") {
                config.distill_enabled = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("distill_ema_decay") {
                config.distill_ema_decay = v.as_f64().unwrap_or(0.995);
            }
            if let Some(v) = map.get("distill_kl_coef") {
                config.distill_kl_coef = v.as_f64().unwrap_or(0.1);
            }
            if let Some(v) = map.get("distill_warmup_steps") {
                config.distill_warmup_steps = v.as_f64().unwrap_or(500.0).round() as usize;
            }
            if let Some(v) = map.get("worker_max_threads") {
                config.worker_max_threads = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("worker_reserve_cores") {
                config.worker_reserve_cores = v.as_f64().unwrap_or(1.0).round() as usize;
            }
            if let Some(v) = map.get("worker_priority") {
                config.worker_priority = v.as_str().unwrap_or("time_critical").to_string();
            }
            if let Some(v) = map.get("worker_stack_size_mb") {
                config.worker_stack_size_mb = v.as_f64().unwrap_or(4.0).round() as usize;
            }
            if let Some(v) = map.get("f2p_sim_count") {
                config.f2p_sim_count = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("f2p_sim_count_prob") {
                config.f2p_sim_count_prob = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("f2p_sim_count_cost") {
                config.f2p_sim_count_cost = v.as_f64().unwrap_or(0.0).round() as usize;
            }
            if let Some(v) = map.get("f2p_luck_mode") {
                config.f2p_luck_mode = v.as_str().map(LuckMode::from_str).or(config.f2p_luck_mode);
            }
            if let Some(v) = map.get("online_train") {
                config.online_train = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("online_train_dqn") {
                config.online_train_dqn = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("online_train_neural") {
                config.online_train_neural = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("online_train_ppo") {
                config.online_train_ppo = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("train_interval_ms") {
                config.train_interval_ms = v.as_f64().unwrap_or(50.0).round() as usize;
            }
            if let Some(v) = map.get("max_train_steps_per_tick") {
                config.max_train_steps_per_tick = v.as_f64().unwrap_or(1.0).round() as usize;
            }
            if let Some(v) = map.get("language") {
                config.language = v.as_str().map(|s| s.to_string());
            }
            if let Some(v) = map.get("model_dim") {
                config.model_dim = v.as_f64().unwrap_or(32.0).round() as usize;
            }
            if let Some(v) = map.get("model_hidden_dim") {
                config.model_hidden_dim = v.as_f64().unwrap_or(1024.0).round() as usize;
            }
            if let Some(v) = map.get("model_num_layers") {
                config.model_num_layers = v.as_f64().unwrap_or(4.0).round() as usize;
            }
            if let Some(v) = map.get("model_num_heads") {
                config.model_num_heads = v.as_f64().unwrap_or(8.0).round() as usize;
            }
            if let Some(v) = map.get("model_kv_lora_rank") {
                config.model_kv_lora_rank = v.as_f64().unwrap_or(128.0).round() as usize;
            }
            if let Some(v) = map.get("model_qk_rope_dim") {
                config.model_qk_rope_dim = v.as_f64().unwrap_or(64.0).round() as usize;
            }
            if let Some(v) = map.get("use_multi_stream") {
                config.use_multi_stream = v.as_bool().unwrap_or(true);
            }
            if let Some(v) = map.get("multi_stream_factor") {
                config.multi_stream_factor = v.as_f64().unwrap_or(2.0).round() as usize;
            }
            if let Some(v) = map.get("use_calibrated") {
                config.use_calibrated = v.as_bool().unwrap_or(true);
            }
            if let Some(v) = map.get("calibrated_path") {
                config.calibrated_path = v.as_str().unwrap_or("data/calibrated.json").to_string();
            }
            if let Some(v) = map.get("player_data_path") {
                config.player_data_path = v.as_str().unwrap_or("data/player_data.json").to_string();
            }
            if let Some(JsonValue::Object(achf_map)) = map.get("achf") {
                if let Some(v) = achf_map.get("enabled") {
                    config.achf.enabled = v.as_bool().unwrap_or(false);
                }
                if let Some(v) = achf_map.get("mode") {
                    config.achf.mode = v.as_str().unwrap_or("lite").to_string();
                }
                if let Some(v) = achf_map.get("candidate_mode") {
                    config.achf.candidate_mode = v.as_str().unwrap_or("sparse").to_string();
                }
                if let Some(v) = achf_map.get("proj_mode") {
                    config.achf.proj_mode = v.as_str().unwrap_or("sinkhorn").to_string();
                }
                if let Some(v) = achf_map.get("candidate_refresh_freq") {
                    config.achf.candidate_refresh_freq = v.as_f64().unwrap_or(1.0).round() as usize;
                }
                if let Some(v) = achf_map
                    .get("ortho_penalty_freq")
                    .or_else(|| achf_map.get("proj_freq"))
                {
                    config.achf.ortho_penalty_freq = v.as_f64().unwrap_or(8.0).round() as usize;
                }
                if let Some(v) = achf_map.get("proj_steps") {
                    config.achf.proj_steps = v.as_f64().unwrap_or(0.0).round() as usize;
                }
                if let Some(v) = achf_map.get("lambda_ortho") {
                    config.achf.lambda_ortho = v.as_f64().unwrap_or(0.0);
                }
                if let Some(v) = achf_map.get("gate_mode") {
                    config.achf.gate_mode = v.as_str().unwrap_or("grad_ema").to_string();
                }
                if let Some(v) = achf_map.get("gate_momentum") {
                    config.achf.gate_momentum = v.as_f64().unwrap_or(0.95);
                }
                if let Some(v) = achf_map.get("gate_beta") {
                    config.achf.gate_beta = v.as_f64().unwrap_or(0.7);
                }
                if let Some(v) = achf_map.get("gate_alpha") {
                    config.achf.gate_alpha = v.as_f64().unwrap_or(0.0);
                }
                if let Some(v) = achf_map.get("g_min") {
                    config.achf.g_min = v.as_f64().unwrap_or(0.2);
                }
                if let Some(v) = achf_map.get("gate_warmup_steps") {
                    config.achf.gate_warmup_steps = v.as_f64().unwrap_or(0.0).round() as usize;
                }
                if let Some(v) = achf_map.get("gate_transition_steps") {
                    config.achf.gate_transition_steps = v.as_f64().unwrap_or(50.0).round() as usize;
                }
                if let Some(v) = achf_map.get("gate_k_clip") {
                    config.achf.gate_k_clip = v.as_f64().unwrap_or(0.0);
                }
                if let Some(v) = achf_map.get("g_target_min") {
                    config.achf.g_target_min = v.as_f64().unwrap_or(0.3);
                }
                if let Some(v) = achf_map.get("g_target_max") {
                    config.achf.g_target_max = v.as_f64().unwrap_or(0.8);
                }
                if let Some(v) = achf_map.get("g_min_adapt_rate") {
                    config.achf.g_min_adapt_rate = v.as_f64().unwrap_or(0.0);
                }
                if let Some(v) = achf_map.get("g_min_momentum") {
                    config.achf.g_min_momentum = v.as_f64().unwrap_or(0.9);
                }
                if let Some(v) = achf_map.get("cache_min_rows") {
                    config.achf.cache_min_rows = v.as_f64().unwrap_or(1.0).round() as usize;
                }
                if let Some(v) = achf_map.get("cache_min_nonzero_ratio") {
                    config.achf.cache_min_nonzero_ratio = v.as_f64().unwrap_or(0.0);
                }
                if let Some(v) = achf_map.get("cache_min_reuse") {
                    config.achf.cache_min_reuse = v.as_f64().unwrap_or(2.0).round() as usize;
                }
                if let Some(v) = achf_map.get("path_warmup_samples") {
                    config.achf.path_warmup_samples = v.as_f64().unwrap_or(2.0).round() as usize;
                }
                if let Some(v) = achf_map.get("path_min_dwell") {
                    config.achf.path_min_dwell = v.as_f64().unwrap_or(2.0).round() as usize;
                }
                if let Some(v) = achf_map.get("cache_sparsity_sample_rows") {
                    config.achf.cache_sparsity_sample_rows =
                        v.as_f64().unwrap_or(0.0).round() as usize;
                }
                if let Some(v) = achf_map.get("cache_cost_bias") {
                    config.achf.cache_cost_bias = v.as_f64().unwrap_or(1.0);
                }
                if let Some(v) = achf_map.get("cache_adapt_rate") {
                    config.achf.cache_adapt_rate = v.as_f64().unwrap_or(0.0);
                }
                if let Some(v) = achf_map.get("cache_bias_min") {
                    config.achf.cache_bias_min = v.as_f64().unwrap_or(0.2);
                }
                if let Some(v) = achf_map.get("cache_bias_max") {
                    config.achf.cache_bias_max = v.as_f64().unwrap_or(5.0);
                }
                if let Some(v) = achf_map.get("cache_latency_ema") {
                    config.achf.cache_latency_ema = v.as_f64().unwrap_or(0.9);
                }
                if let Some(v) = achf_map.get("cache_latency_long_ema") {
                    config.achf.cache_latency_long_ema = v.as_f64().unwrap_or(0.99);
                }
                if let Some(v) = achf_map.get("cache_adapt_blend") {
                    config.achf.cache_adapt_blend = v.as_f64().unwrap_or(0.5);
                }
                if let Some(v) = achf_map.get("cache_latency_sample_every") {
                    config.achf.cache_latency_sample_every =
                        v.as_f64().unwrap_or(1.0).round() as u64;
                }
                if let Some(v) = achf_map.get("cache_log_interval_steps") {
                    config.achf.cache_log_interval_steps =
                        v.as_f64().unwrap_or(0.0).round() as usize;
                }
                if let Some(v) = achf_map.get("cache_log_per_layer") {
                    config.achf.cache_log_per_layer = v.as_bool().unwrap_or(false);
                }
                if let Some(v) = achf_map.get("diagnostics_enabled") {
                    config.achf.diagnostics_enabled = v.as_bool().unwrap_or(false);
                }
                if let Some(v) = achf_map.get("rank") {
                    config.achf.rank = v.as_f64().unwrap_or(0.0).round() as usize;
                }
                if let Some(v) = achf_map.get("prune_threshold") {
                    config.achf.prune_threshold = v.as_f64().unwrap_or(0.01);
                }
                if let Some(v) = achf_map.get("candidate_min_sparsity") {
                    config.achf.candidate_min_sparsity = v.as_f64().unwrap_or(0.5);
                }
                if let Some(v) = achf_map.get("candidate_max_relative_error") {
                    config.achf.candidate_max_relative_error = v.as_f64().unwrap_or(0.05);
                }
                if let Some(v) = achf_map
                    .get("candidate_weight_error_momentum")
                    .or_else(|| achf_map.get("candidate_discrepancy_momentum"))
                {
                    config.achf.candidate_weight_error_momentum = v.as_f64().unwrap_or(0.9);
                }

                if let Some(v) = achf_map.get("apply_attn") {
                    config.achf.apply_attn = v.as_bool().unwrap_or(false);
                }
                if let Some(v) = achf_map.get("apply_ffn") {
                    config.achf.apply_ffn = v.as_bool().unwrap_or(true);
                }
                if let Some(v) = achf_map.get("apply_dqn") {
                    config.achf.apply_dqn = v.as_bool().unwrap_or(false);
                }
                if let Some(v) = achf_map.get("infer_gate") {
                    config.achf.infer_gate = v.as_str().unwrap_or("candidate").to_string();
                }
                if let Some(v) = achf_map.get("adaptive_inference") {
                    config.achf.adaptive_inference = v.as_bool().unwrap_or(false);
                }
            }
        }

        // Pools live in a separate file (config.pools_path). For backward
        // compatibility, if the main config still embeds a `pools` array we
        // parse pools from there instead of loading the external file.
        let pools_root: JsonValue = match &root {
            JsonValue::Object(map) if map.contains_key("pools") => root.clone(),
            _ => load_pools_root(&config.pools_path),
        };
        if let JsonValue::Object(ref pmap) = pools_root {
            extract_pool_fields(&mut config, pmap);
        }

        sanitize_pity_settings(
            &mut config.soft_pity_start,
            &mut config.small_pity_guarantee,
            &mut config.big_pity_cumulative,
            "root config",
        );
        sanitize_non_negative(&mut config.luck_action_cost, "luck_action_cost", 8.0);
        sanitize_non_negative(&mut config.luck_budget_max, "luck_budget_max", 0.045);
        sanitize_non_negative(&mut config.luck_budget_initial, "luck_budget_initial", 0.03);
        sanitize_non_negative(
            &mut config.luck_budget_recovery_per_pull,
            "luck_budget_recovery_per_pull",
            0.001,
        );
        sanitize_non_negative(
            &mut config.luck_budget_negative_refund,
            "luck_budget_negative_refund",
            1.0,
        );
        if config.luck_budget_max > 0.0 && config.luck_budget_initial > config.luck_budget_max {
            eprintln!(
                "[Config Warning] luck_budget_initial ({}) exceeds luck_budget_max ({}), clamping",
                config.luck_budget_initial, config.luck_budget_max
            );
            config.luck_budget_initial = config.luck_budget_max;
        }
        if config.policy_eval_episodes == 0 {
            eprintln!("[Config Warning] policy_eval_episodes is 0, fallback to 1");
            config.policy_eval_episodes = 1;
        }
        sanitize_achf_config(&mut config.achf);

        // Sanitize model dimension settings
        if config.model_dim == 0 {
            eprintln!("[Config Warning] model_dim is 0, fallback to 32");
            config.model_dim = 32;
        }
        if config.model_hidden_dim == 0 {
            eprintln!("[Config Warning] model_hidden_dim is 0, fallback to 1024");
            config.model_hidden_dim = 1024;
        }
        if config.model_num_layers == 0 {
            eprintln!("[Config Warning] model_num_layers is 0, fallback to 4");
            config.model_num_layers = 4;
        }
        if config.model_num_heads == 0 {
            eprintln!("[Config Warning] model_num_heads is 0, fallback to 8");
            config.model_num_heads = 8;
        }
        if config.model_dim % config.model_num_heads != 0 {
            eprintln!(
                "[Config Warning] model_dim ({}) not divisible by model_num_heads ({}), adjusting model_dim",
                config.model_dim, config.model_num_heads
            );
            config.model_dim = (config.model_dim / config.model_num_heads) * config.model_num_heads;
            if config.model_dim == 0 {
                config.model_dim = config.model_num_heads;
            }
        }
        if config.multi_stream_factor == 0 {
            config.multi_stream_factor = 2;
        }

        if !config.pools.is_empty() {
            if let Some(active) = config.active_pool.clone() {
                if !config.apply_pool(&active) {
                    let first = config.pools[0].id.clone();
                    config.apply_pool(&first);
                    config.active_pool = Some(first);
                }
            } else {
                let first = config.pools[0].id.clone();
                config.apply_pool(&first);
                config.active_pool = Some(first);
            }
        }

        if config.language.as_deref() == Some("en") {
            if let JsonValue::Object(ref map) = pools_root {
                if let Some(JsonValue::Array(pools_arr)) = map.get("pools") {
                    for (i, pool_val) in pools_arr.iter().enumerate() {
                        if let (JsonValue::Object(ref pm), Some(pool)) =
                            (pool_val, config.pools.get_mut(i))
                        {
                            if let Some(s) = pm.get("name_en").and_then(|v| v.as_str()) {
                                if !s.is_empty() {
                                    pool.name = s.to_string();
                                }
                            }
                            let swap = |field: &mut Vec<String>, key: &str| {
                                if let Some(v) = pm.get(key) {
                                    let en = json_to_string_vec(v);
                                    if !en.is_empty() {
                                        *field = en;
                                    }
                                }
                            };
                            swap(&mut pool.up_six, "up_six_en");
                            swap(&mut pool.six_stars, "six_stars_en");
                            swap(&mut pool.five_stars, "five_stars_en");
                            swap(&mut pool.four_stars, "four_stars_en");
                        }
                    }
                }
                if let Some(active) = config.active_pool.clone() {
                    config.apply_pool(&active);
                }
                if let Some(s) = map.get("pool_name_en").and_then(|v| v.as_str()) {
                    if !s.is_empty() && config.active_pool.is_none() {
                        config.pool_name = s.to_string();
                    }
                }
            }
        }

        config
    }

    /// Switch the active pool by ID, updating all pool-specific settings.
    pub fn apply_pool(&mut self, pool_id: &str) -> bool {
        let pool = match self.pools.iter().find(|p| p.id == pool_id) {
            Some(p) => p.clone(),
            None => return false,
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

/// Read and parse the external pools file (config.pools_path). Applies the
/// same parent-directory fallback as the main config loader so it works from
/// `target/**/exe` layouts. Returns `JsonValue::Null` if the file is missing
/// or invalid (the caller then simply has no pools defined).
fn load_pools_root(path: &str) -> JsonValue {
    let mut file = match File::open(path) {
        Ok(f) => Some(f),
        Err(_) => {
            let mut found = None;
            for levels in [2, 3] {
                if let Some(fallback) = fallback_parent_path(path, levels) {
                    if let Ok(f) = File::open(&fallback) {
                        println!(
                            "[System] Pools file found in parent directory ({} levels up).",
                            levels
                        );
                        found = Some(f);
                        break;
                    }
                }
            }
            found
        }
    };

    let Some(ref mut f) = file else {
        eprintln!(
            "\x1b[33m[Warning]\x1b[0m Pools file '{}' not found; no pools loaded.",
            path
        );
        return JsonValue::Null;
    };

    let mut contents = String::new();
    if f.read_to_string(&mut contents).is_err() {
        log::error!("Failed to read pools file '{}'", path);
        return JsonValue::Null;
    }
    let stripped = strip_json_comments(&contents);
    match serde_json::from_str(&stripped) {
        Ok(value) => value,
        Err(err) => {
            eprintln!(
                "\x1b[1;31m[Error]\x1b[0m Pools JSON parse error in '{}': {}",
                path, err
            );
            JsonValue::Null
        }
    }
}

/// Populate the pool-related fields of `config` from a JSON object (either the
/// external pools file root or, for backward compatibility, the main config
/// root when it still embeds a `pools` array).
fn extract_pool_fields(config: &mut Config, map: &serde_json::Map<String, JsonValue>) {
    if let Some(v) = map.get("pool_name") {
        config.pool_name = v.as_str().unwrap_or("").to_string();
    }
    if let Some(v) = map.get("up_six") {
        config.up_six = json_to_string_vec(v);
    }
    if let Some(v) = map.get("up_rate") {
        config.up_rate = v.as_f64().unwrap_or(0.5);
    }
    if let Some(v) = map.get("prob_6_base") {
        config.prob_6_base = v.as_f64().unwrap_or(0.008);
    }
    if let Some(v) = map.get("prob_5_base") {
        config.prob_5_base = v.as_f64().unwrap_or(0.08);
    }
    if let Some(v) = map.get("prob_4_base") {
        config.prob_4_base = v.as_f64().unwrap_or(0.912);
    }
    if let Some(v) = map.get("soft_pity_start") {
        config.soft_pity_start = v.as_f64().unwrap_or(65.0).round() as usize;
    }
    if let Some(v) = map.get("soft_pity_slope") {
        config.soft_pity_slope = v.as_f64().unwrap_or(0.05);
    }
    if let Some(v) = map.get("small_pity_guarantee") {
        config.small_pity_guarantee = v.as_f64().unwrap_or(80.0).round() as usize;
    }
    if let Some(v) = map.get("big_pity_cumulative") {
        config.big_pity_cumulative = v.as_f64().unwrap_or(120.0).round() as usize;
    }
    if let Some(v) = map.get("up_pity_soft") {
        config.up_pity_soft = v.as_f64().unwrap_or(0.0).round() as usize;
    }
    if let Some(v) = map.get("five_star_pity") {
        config.five_star_pity = v.as_f64().unwrap_or(10.0).round() as usize;
    }
    if let Some(v) = map.get("always_5_star") {
        config.always_5_star = v.as_bool().unwrap_or(false);
    }
    if let Some(v) = map.get("big_pity_requires_not_up") {
        config.big_pity_requires_not_up = v.as_bool().unwrap_or(true);
    }
    if let Some(v) = map.get("six_stars") {
        config.six_stars = json_to_string_vec(v);
    }
    if let Some(v) = map.get("five_stars") {
        config.five_stars = json_to_string_vec(v);
    }
    if let Some(v) = map.get("four_stars") {
        config.four_stars = json_to_string_vec(v);
    }
    if let Some(v) = map.get("active_pool") {
        config.active_pool = v.as_str().map(|s| s.to_string());
    }
    if let Some(JsonValue::Array(pools)) = map.get("pools") {
        config.pools = pools
            .iter()
            .filter_map(|v| match v {
                JsonValue::Object(pool_map) => Some(parse_pool_config(pool_map)),
                _ => None,
            })
            .collect();
    }
}

fn parse_pool_config(pool_map: &serde_json::Map<String, JsonValue>) -> PoolConfig {
    let mut pool = PoolConfig {
        id: pool_map
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        name: pool_map
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string(),
        pool_type: pool_map
            .get("pool_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        up_six: pool_map
            .get("up_six")
            .map(json_to_string_vec)
            .unwrap_or_default(),
        up_rate: pool_map
            .get("up_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5),
        prob_6_base: pool_map
            .get("prob_6_base")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.008),
        prob_5_base: pool_map
            .get("prob_5_base")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.08),
        prob_4_base: pool_map
            .get("prob_4_base")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.912),
        soft_pity_start: pool_map
            .get("soft_pity_start")
            .and_then(|v| v.as_f64())
            .unwrap_or(65.0)
            .round() as usize,
        soft_pity_slope: pool_map
            .get("soft_pity_slope")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.05),
        small_pity_guarantee: pool_map
            .get("small_pity_guarantee")
            .and_then(|v| v.as_f64())
            .unwrap_or(80.0)
            .round() as usize,
        big_pity_cumulative: pool_map
            .get("big_pity_cumulative")
            .and_then(|v| v.as_f64())
            .unwrap_or(120.0)
            .round() as usize,
        up_pity_soft: pool_map
            .get("up_pity_soft")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .round() as usize,
        five_star_pity: pool_map
            .get("five_star_pity")
            .and_then(|v| v.as_f64())
            .unwrap_or(10.0)
            .round() as usize,
        always_5_star: pool_map
            .get("always_5_star")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        big_pity_requires_not_up: pool_map
            .get("big_pity_requires_not_up")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        six_stars: pool_map
            .get("six_stars")
            .map(json_to_string_vec)
            .unwrap_or_default(),
        five_stars: pool_map
            .get("five_stars")
            .map(json_to_string_vec)
            .unwrap_or_default(),
        four_stars: pool_map
            .get("four_stars")
            .map(json_to_string_vec)
            .unwrap_or_default(),
        is_archived: pool_map
            .get("is_archived")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };
    if pool.up_rate <= 0.0 || pool.up_six.is_empty() {
        pool.up_rate = 0.0;
    }
    let scope = if pool.id.is_empty() {
        "pool <unknown>".to_string()
    } else {
        format!("pool '{}'", pool.id)
    };
    sanitize_pity_settings(
        &mut pool.soft_pity_start,
        &mut pool.small_pity_guarantee,
        &mut pool.big_pity_cumulative,
        &scope,
    );
    pool
}

fn warn_unknown_fields(map: &serde_json::Map<String, JsonValue>) {
    let known: HashSet<&'static str> = [
        "pool_name",
        "up_six",
        "up_rate",
        "prob_6_base",
        "prob_5_base",
        "prob_4_base",
        "soft_pity_start",
        "small_pity_guarantee",
        "big_pity_cumulative",
        "up_pity_soft",
        "five_star_pity",
        "always_5_star",
        "big_pity_requires_not_up",
        "six_stars",
        "five_stars",
        "four_stars",
        "pools",
        "active_pool",
        "pools_path",
        "luck_mode",
        "fast_init",
        "ppo_mode",
        "ppo_total_steps",
        "ppo_steps_per_update",
        "ppo_k_epochs",
        "ppo_batch_size",
        "ppo_context_len",
        "ppo_num_envs",
        "ppo_top_k",
        "luck_action_cost",
        "luck_budget_enabled",
        "luck_budget_max",
        "luck_budget_initial",
        "luck_budget_recovery_per_pull",
        "luck_budget_negative_refund",
        "policy_eval_interval",
        "policy_eval_episodes",
        "policy_eval_seed",
        "distill_enabled",
        "distill_ema_decay",
        "distill_kl_coef",
        "distill_warmup_steps",
        "device",
        "worker_max_threads",
        "worker_reserve_cores",
        "worker_priority",
        "worker_stack_size_mb",
        "f2p_sim_count",
        "f2p_sim_count_prob",
        "f2p_sim_count_cost",
        "f2p_luck_mode",
        "online_train",
        "online_train_dqn",
        "online_train_neural",
        "online_train_ppo",
        "train_interval_ms",
        "max_train_steps_per_tick",
        "language",
        "achf",
        "soft_pity_slope",
        "use_calibrated",
        "calibrated_path",
        "player_data_path",
        "pool_name_en",
        "up_six_en",
        "six_stars_en",
        "five_stars_en",
        "four_stars_en",
        "model_dim",
        "model_hidden_dim",
        "model_num_layers",
        "model_num_heads",
        "model_kv_lora_rank",
        "model_qk_rope_dim",
        "use_multi_stream",
        "multi_stream_factor",
    ]
    .into_iter()
    .collect();

    for key in map.keys() {
        if !known.contains(key.as_str()) {
            eprintln!("[Config Warning] Unknown field: {}", key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(input: &str) -> JsonValue {
        let stripped = strip_json_comments(input);
        serde_json::from_str(&stripped).unwrap()
    }

    #[test]
    fn parse_empty_object() {
        let value = parse_ok("{}");
        if let JsonValue::Object(map) = value {
            assert!(map.is_empty());
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn parse_nested_array() {
        let value = parse_ok("[1, [2, 3], 4]");
        if let JsonValue::Array(arr) = value {
            assert_eq!(arr.len(), 3);
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn parse_unicode_escape() {
        let value = parse_ok(r#""\u4e2d\u6587""#);
        if let JsonValue::String(s) = value {
            assert_eq!(s, "中文");
        } else {
            panic!("Expected string");
        }
    }

    #[test]
    fn parse_scientific_number() {
        let value = parse_ok(r#"[1e-3, -2.5E+2]"#);
        if let JsonValue::Array(arr) = value {
            assert!((arr[0].as_f64().unwrap() - 0.001).abs() < 1e-12);
            assert!((arr[1].as_f64().unwrap() + 250.0).abs() < 1e-9);
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn parse_escape_sequences() {
        let value = parse_ok(r#""line1\nline2\t\"""#);
        if let JsonValue::String(s) = value {
            assert_eq!(s, "line1\nline2\t\"");
        } else {
            panic!("Expected string");
        }
    }

    #[test]
    fn default_model_uses_compact_dimensions() {
        let config = Config::default();

        assert_eq!(config.model_dim, 32);
        assert_eq!(config.model_hidden_dim, 1024);
        assert_eq!(config.model_num_layers, 4);
        assert_eq!(config.model_num_heads, 8);
        assert_eq!(config.model_kv_lora_rank, 128);
        assert_eq!(config.model_qk_rope_dim, 64);
        assert_eq!(config.multi_stream_factor, 2);
        assert_eq!(config.achf.rank, 0);
    }

    #[test]
    fn config_load_preserves_zero_achf_rank() {
        let path =
            std::env::temp_dir().join(format!("talos_xii_rank_zero_{}.json", std::process::id()));
        std::fs::write(&path, r#"{"achf":{"rank":0}}"#).unwrap();
        let config = Config::load(path.to_str().unwrap());
        let _ = std::fs::remove_file(path);

        assert_eq!(config.achf.rank, 0);
    }

    #[test]
    fn sanitize_pity_clamps_large_small_guarantee() {
        let mut soft = 50usize;
        let mut small = usize::MAX;
        let mut big = usize::MAX;
        sanitize_pity_settings(&mut soft, &mut small, &mut big, "test");
        assert_eq!(small, MAX_SMALL_PITY_GUARANTEE);
        assert_eq!(big, MAX_BIG_PITY_CUMULATIVE);
    }

    #[test]
    fn sanitize_pity_clamps_soft_to_small() {
        let mut soft = 200usize;
        let mut small = 80usize;
        let mut big = 100usize;
        sanitize_pity_settings(&mut soft, &mut small, &mut big, "test");
        assert_eq!(soft, 80);
        assert_eq!(small, 80);
        assert_eq!(big, 100);
    }

    #[test]
    fn sanitize_achf_clamps_numeric_bounds() {
        let defaults = AchfConfig::default();
        let mut achf = AchfConfig {
            gate_momentum: 2.0,
            gate_alpha: f64::NAN,
            gate_beta: f64::INFINITY,
            lambda_ortho: f64::NAN,
            g_min: -0.1,
            g_target_min: 1.5,
            g_target_max: -0.4,
            g_min_adapt_rate: -0.2,
            g_min_momentum: f64::NAN,
            gate_k_clip: -1.0,
            cache_min_nonzero_ratio: 1.2,
            cache_cost_bias: 0.0,
            cache_adapt_rate: -0.3,
            cache_bias_min: -1.0,
            cache_bias_max: 0.1,
            cache_latency_ema: -0.5,
            cache_latency_long_ema: f64::INFINITY,
            cache_adapt_blend: -0.8,
            ..Default::default()
        };

        sanitize_achf_config(&mut achf);

        assert_eq!(achf.gate_momentum, 1.0);
        assert_eq!(achf.gate_alpha, defaults.gate_alpha);
        assert_eq!(achf.gate_beta, defaults.gate_beta);
        assert_eq!(achf.lambda_ortho, defaults.lambda_ortho);
        assert_eq!(achf.g_min, 0.0);
        assert_eq!(achf.g_target_min, 1.0);
        assert_eq!(achf.g_target_max, 1.0);
        assert_eq!(achf.g_min_adapt_rate, 0.0);
        assert_eq!(achf.g_min_momentum, defaults.g_min_momentum);
        assert_eq!(achf.gate_k_clip, 0.0);
        assert_eq!(achf.cache_min_nonzero_ratio, 1.0);
        assert_eq!(achf.cache_cost_bias, defaults.cache_cost_bias);
        assert_eq!(achf.cache_adapt_rate, 0.0);
        assert_eq!(achf.cache_bias_min, defaults.cache_bias_min);
        assert!(achf.cache_bias_max >= achf.cache_bias_min);
        assert_eq!(achf.cache_latency_ema, 0.0);
        assert_eq!(achf.cache_latency_long_ema, defaults.cache_latency_long_ema);
        assert_eq!(achf.cache_adapt_blend, 0.0);
    }

    #[test]
    fn sanitize_achf_canonicalizes_inference_mode() {
        let mut full = AchfConfig {
            mode: " FULL ".to_string(),
            ..Default::default()
        };
        sanitize_achf_config(&mut full);
        assert_eq!(full.mode, "full");
        assert!(!full.adaptive_inference);
        assert!(full.uses_adaptive_inference());

        for mode in ["fixed_cached", "fixed_sparse", "fixed_dense", "plain_ema"] {
            let mut config = AchfConfig {
                mode: mode.to_string(),
                ..Default::default()
            };
            sanitize_achf_config(&mut config);
            assert_eq!(config.mode, mode);
        }

        let mut invalid = AchfConfig {
            mode: "turbo".to_string(),
            ..Default::default()
        };
        sanitize_achf_config(&mut invalid);
        assert_eq!(invalid.mode, "lite");
        assert!(!invalid.adaptive_inference);
        assert!(!invalid.uses_adaptive_inference());

        let mut legacy = AchfConfig {
            mode: "lite".to_string(),
            adaptive_inference: true,
            ..Default::default()
        };
        sanitize_achf_config(&mut legacy);
        assert_eq!(legacy.mode, "full");
        assert!(!legacy.adaptive_inference);
        assert!(legacy.uses_adaptive_inference());
    }

    #[test]
    fn sanitize_achf_separates_candidate_and_connection_constraints() {
        let mut sparse = AchfConfig {
            candidate_mode: "sparse".to_string(),
            rank: 8,
            ..Default::default()
        };
        sanitize_achf_config(&mut sparse);
        assert_eq!(sparse.rank, 0);

        let mut low_rank = AchfConfig {
            candidate_mode: "low_rank".to_string(),
            rank: 8,
            prune_threshold: 0.5,
            ..Default::default()
        };
        sanitize_achf_config(&mut low_rank);
        assert_eq!(low_rank.prune_threshold, 0.0);

        let mut constrained_connection = AchfConfig {
            proj_mode: "sinkhorn".to_string(),
            lambda_ortho: 0.1,
            ..Default::default()
        };
        sanitize_achf_config(&mut constrained_connection);
        assert_eq!(constrained_connection.lambda_ortho, 0.0);
    }

    #[test]
    fn config_load_migrates_legacy_achf_frequency_and_error_keys() {
        let path = std::env::temp_dir().join(format!(
            "talos_xii_achf_legacy_keys_{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{"achf":{"proj_freq":3,"candidate_discrepancy_momentum":0.7,"path_warmup_samples":5,"path_min_dwell":7,"gate_transition_steps":23}}"#,
        )
        .unwrap();
        let config = Config::load(path.to_str().unwrap());
        let _ = std::fs::remove_file(path);

        assert_eq!(config.achf.ortho_penalty_freq, 3);
        assert_eq!(config.achf.candidate_weight_error_momentum, 0.7);
        assert_eq!(config.achf.candidate_refresh_freq, 1);
        assert_eq!(config.achf.path_warmup_samples, 5);
        assert_eq!(config.achf.path_min_dwell, 7);
        assert_eq!(config.achf.gate_transition_steps, 23);
    }
}
