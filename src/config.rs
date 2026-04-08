use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;

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
/// Determines which policy drives the luck factor during simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

// --- Configuration (Data-Driven) ---

/// Configuration for Adaptive Cache-aware Hyper-Connections (ACHF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchfConfig {
    pub enabled: bool,
    pub mode: String,
    pub proj_mode: String,
    pub proj_freq: usize,
    pub proj_steps: usize,
    pub lambda_ortho: f64,
    pub gate_mode: String,
    pub gate_momentum: f64,
    pub gate_beta: f64,
    pub gate_alpha: f64,
    pub g_min: f64,
    pub gate_warmup_steps: usize,
    pub gate_k_clip: f64,
    pub g_target_min: f64,
    pub g_target_max: f64,
    pub g_min_adapt_rate: f64,
    pub g_min_momentum: f64,
    pub cache_min_rows: usize,
    pub cache_min_nonzero_ratio: f64,
    pub cache_min_reuse: usize,
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
    pub rank: usize,
    pub apply_attn: bool,
    pub apply_ffn: bool,
    pub apply_dqn: bool,
    pub infer_gate: String,
}

impl Default for AchfConfig {
    fn default() -> Self {
        AchfConfig {
            enabled: false,
            mode: "lite".to_string(),
            proj_mode: "rowcol".to_string(),
            proj_freq: 8,
            proj_steps: 0,
            lambda_ortho: 1e-3,
            gate_mode: "grad_ema".to_string(),
            gate_momentum: 0.95,
            gate_beta: 0.7,
            gate_alpha: 0.0,
            g_min: 0.2,
            gate_warmup_steps: 0,
            gate_k_clip: 0.0,
            g_target_min: 0.3,
            g_target_max: 0.8,
            g_min_adapt_rate: 0.0,
            g_min_momentum: 0.9,
            cache_min_rows: 1,
            cache_min_nonzero_ratio: 0.0,
            cache_min_reuse: 2,
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
            rank: 32,
            apply_attn: true,
            apply_ffn: true,
            apply_dqn: false,
            infer_gate: "g_min".to_string(),
        }
    }
}

/// Per-pool configuration defining gacha rules and operator rosters.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    pub luck_mode: LuckMode,
    pub use_calibrated: bool,
    pub calibrated_path: String,
    pub player_data_path: String,
    pub fast_init: bool,
    pub ppo_mode: String,
    pub ppo_total_steps: usize,
    pub ppo_steps_per_update: usize,
    pub ppo_k_epochs: usize,
    pub ppo_batch_size: usize,
    pub ppo_context_len: usize,
    pub ppo_num_envs: usize,
    pub ppo_top_k: usize,
    pub distill_enabled: bool,
    pub distill_ema_decay: f64,
    pub distill_kl_coef: f64,
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
            pools: vec![],
            active_pool: None,
            luck_mode: LuckMode::Probability,
            use_calibrated: true,
            calibrated_path: "data/calibrated.json".to_string(),
            player_data_path: "data/player_data.json".to_string(),
            fast_init: false,
            ppo_mode: "balanced".to_string(),
            ppo_total_steps: 0,
            ppo_steps_per_update: 0,
            ppo_k_epochs: 0,
            ppo_batch_size: 0,
            ppo_context_len: 0,
            ppo_num_envs: 1,
            ppo_top_k: 0, // 0 = disabled (full softmax), >0 = top-k truncation
            distill_enabled: false,
            distill_ema_decay: 0.995,
            distill_kl_coef: 0.1,
            worker_max_threads: 0,
            worker_reserve_cores: 1,
            worker_priority: "time_critical".to_string(),
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

        // Robustness: If file not found, try to look in parent directories (useful for IDE/target builds)
        let mut file = match file_result {
            Ok(f) => f,
            Err(_) => {
                // Try ../../data/config.json (standard cargo layout: target/release/exe vs project/data)
                match File::open(format!("../../{}", path)) {
                    Ok(f) => {
                        println!("[System] Config found in parent directory.");
                        f
                    }
                    Err(_) => {
                        eprintln!("\x1b[1;31m[Error]\x1b[0m Configuration file not found.");
                        eprintln!("  Looked at: './{path}' and '../../{path}'");
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
        file.read_to_string(&mut contents)
            .expect("Failed to read config file");

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
            if let Some(v) = map.get("luck_mode") {
                config.luck_mode = LuckMode::from_str(v.as_str().unwrap_or("probability"));
            }
            if let Some(v) = map.get("fast_init") {
                config.fast_init = v.as_bool().unwrap_or(false);
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
            if let Some(v) = map.get("distill_enabled") {
                config.distill_enabled = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("distill_ema_decay") {
                config.distill_ema_decay = v.as_f64().unwrap_or(0.995);
            }
            if let Some(v) = map.get("distill_kl_coef") {
                config.distill_kl_coef = v.as_f64().unwrap_or(0.1);
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
                if let Some(v) = achf_map.get("proj_mode") {
                    config.achf.proj_mode = v.as_str().unwrap_or("rowcol").to_string();
                }
                if let Some(v) = achf_map.get("proj_freq") {
                    config.achf.proj_freq = v.as_f64().unwrap_or(8.0).round() as usize;
                }
                if let Some(v) = achf_map.get("proj_steps") {
                    config.achf.proj_steps = v.as_f64().unwrap_or(0.0).round() as usize;
                }
                if let Some(v) = achf_map.get("lambda_ortho") {
                    config.achf.lambda_ortho = v.as_f64().unwrap_or(1e-3);
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
                if let Some(v) = achf_map.get("rank") {
                    let r = v.as_f64().unwrap_or(32.0).round() as usize;
                    config.achf.rank = if r == 0 { 32 } else { r };
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
                    config.achf.infer_gate = v.as_str().unwrap_or("g_min").to_string();
                }
            }
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
            if let JsonValue::Object(ref map) = root {
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
        "distill_enabled",
        "distill_ema_decay",
        "distill_kl_coef",
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
}
