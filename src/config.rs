use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

// --- Simple JSON Parser (std only) ---

#[derive(Debug, Clone)]
pub enum JsonValue {
    Null,
    #[allow(dead_code)]
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

impl JsonValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    // Helper to extract string array
    fn to_string_vec(&self) -> Vec<String> {
        match self {
            JsonValue::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => Vec::new(),
        }
    }
}

struct JsonParser {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl JsonParser {
    fn new(input: &str) -> Self {
        JsonParser {
            chars: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn parse(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        if self.pos >= self.chars.len() {
            return Ok(JsonValue::Null);
        }

        let value = self.parse_value()?;
        self.skip_whitespace();
        if self.pos < self.chars.len() {
            return Err(self.error("Unexpected trailing content"));
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        if self.peek().is_none() {
            return Err(self.error("Unexpected EOF"));
        }
        match self.peek().unwrap() {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' => self.parse_string().map(JsonValue::String),
            't' => {
                self.consume("true")?;
                Ok(JsonValue::Bool(true))
            }
            'f' => {
                self.consume("false")?;
                Ok(JsonValue::Bool(false))
            }
            'n' => {
                self.consume("null")?;
                Ok(JsonValue::Null)
            }
            c if c == '-' || c.is_ascii_digit() => self.parse_number(),
            c => Err(self.error(&format!("Unexpected character: {}", c))),
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn consume(&mut self, s: &str) -> Result<(), String> {
        for c in s.chars() {
            if self.peek() != Some(c) {
                return Err(self.error(&format!("Expected '{}'", s)));
            }
            self.advance();
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some('"') {
            return Err(self.error("Expected '\"' to start string"));
        }
        self.advance();
        let mut s = String::new();
        while let Some(c) = self.advance() {
            if c == '"' {
                return Ok(s);
            }
            if c == '\\' {
                let escaped = match self.advance() {
                    Some(ch) => ch,
                    None => return Err(self.error("Unexpected EOF in string escape")),
                };
                match escaped {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'b' => s.push('\x08'),
                    'f' => s.push('\x0c'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    'u' => {
                        let code_point = self.parse_unicode_escape()?;
                        if let Some(ch) = std::char::from_u32(code_point) {
                            s.push(ch);
                        } else {
                            return Err(self.error("Invalid Unicode code point"));
                        }
                    }
                    _ => return Err(self.error(&format!("Invalid escape: \\{}", escaped))),
                }
            } else {
                s.push(c);
            }
        }
        Err(self.error("Unexpected EOF in string"))
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let mut s = String::new();
        if let Some(c) = self.peek() {
            if c == '-' {
                s.push(c);
                self.advance();
            }
        }

        let mut int_digits = 0usize;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
                int_digits += 1;
            } else {
                break;
            }
        }
        if int_digits == 0 {
            return Err(self.error("Invalid number"));
        }

        if self.peek() == Some('.') {
            s.push('.');
            self.advance();
            let mut frac_digits = 0usize;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    s.push(c);
                    self.advance();
                    frac_digits += 1;
                } else {
                    break;
                }
            }
            if frac_digits == 0 {
                return Err(self.error("Invalid fraction"));
            }
        }

        if let Some(c) = self.peek() {
            if c == 'e' || c == 'E' {
                s.push(c);
                self.advance();
                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        s.push(sign);
                        self.advance();
                    }
                }
                let mut exp_digits = 0usize;
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        s.push(c);
                        self.advance();
                        exp_digits += 1;
                    } else {
                        break;
                    }
                }
                if exp_digits == 0 {
                    return Err(self.error("Invalid exponent"));
                }
            }
        }

        s.parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| self.error(&format!("Invalid number: {}", s)))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        if self.peek() != Some('[') {
            return Err(self.error("Expected '['"));
        }
        self.advance();
        let mut arr = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(']') {
            self.advance();
            return Ok(JsonValue::Array(arr));
        }
        loop {
            self.skip_whitespace();
            arr.push(self.parse_value()?);
            self.skip_whitespace();
            if self.peek().is_none() {
                return Err(self.error("Unexpected EOF in array"));
            }
            match self.peek().unwrap() {
                ',' => {
                    self.advance();
                }
                ']' => {
                    self.advance();
                    return Ok(JsonValue::Array(arr));
                }
                c => return Err(self.error(&format!("Expected ',' or ']' in array, found {}", c))),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        if self.peek() != Some('{') {
            return Err(self.error("Expected '{'"));
        }
        self.advance();
        let mut map = HashMap::new();
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.advance();
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some('"') {
                return Err(self.error("Expected string key in object"));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.peek() != Some(':') {
                return Err(self.error("Expected ':' after key"));
            }
            self.advance();
            self.skip_whitespace();
            let value = self.parse_value()?;
            map.insert(key, value);

            self.skip_whitespace();
            if self.peek().is_none() {
                return Err(self.error("Unexpected EOF in object"));
            }
            match self.peek().unwrap() {
                ',' => {
                    self.advance();
                }
                '}' => {
                    self.advance();
                    return Ok(JsonValue::Object(map));
                }
                c => {
                    return Err(self.error(&format!("Expected ',' or '}}' in object, found {}", c)))
                }
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        if self.pos >= self.chars.len() {
            return None;
        }
        let c = self.chars[self.pos];
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn parse_unicode_escape(&mut self) -> Result<u32, String> {
        let mut code: u32 = 0;
        for _ in 0..4 {
            let c = match self.advance() {
                Some(ch) => ch,
                None => return Err(self.error("Unexpected EOF in unicode escape")),
            };
            let digit = c
                .to_digit(16)
                .ok_or_else(|| self.error("Invalid unicode escape"))?;
            code = (code << 4) | digit;
        }
        if (0xD800..=0xDBFF).contains(&code) {
            let saved_pos = self.pos;
            let saved_line = self.line;
            let saved_col = self.col;
            if self.peek() == Some('\\') {
                self.advance();
                if self.peek() == Some('u') {
                    self.advance();
                    let mut low: u32 = 0;
                    for _ in 0..4 {
                        let c = match self.advance() {
                            Some(ch) => ch,
                            None => return Err(self.error("Unexpected EOF in unicode escape")),
                        };
                        let digit = c
                            .to_digit(16)
                            .ok_or_else(|| self.error("Invalid unicode escape"))?;
                        low = (low << 4) | digit;
                    }
                    if (0xDC00..=0xDFFF).contains(&low) {
                        let high_ten = code - 0xD800;
                        let low_ten = low - 0xDC00;
                        return Ok(0x10000 + ((high_ten << 10) | low_ten));
                    } else {
                        return Err(self.error("Invalid unicode surrogate pair"));
                    }
                }
            }
            self.pos = saved_pos;
            self.line = saved_line;
            self.col = saved_col;
            return Err(self.error("Invalid unicode surrogate pair"));
        }
        Ok(code)
    }

    fn error(&self, msg: &str) -> String {
        let pos = self.pos.min(self.chars.len());
        let mut line_start = pos;
        while line_start > 0 && self.chars[line_start - 1] != '\n' {
            line_start -= 1;
        }
        let mut line_end = pos;
        while line_end < self.chars.len() && self.chars[line_end] != '\n' {
            line_end += 1;
        }
        let line_text: String = self.chars[line_start..line_end].iter().collect();
        let caret_pos = if self.col == 0 { 1 } else { self.col };
        let caret = " ".repeat(caret_pos.saturating_sub(1)) + "^";
        format!(
            "JSON parse error at line {}, col {}: {}\n{}\n{}",
            self.line, self.col, msg, line_text, caret
        )
    }
}

// --- Configuration (Data-Driven) ---

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

impl AchfConfig {
    pub fn default() -> Self {
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
            cache_min_rows: 0,
            cache_min_nonzero_ratio: 0.0,
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
            rank: 0,
            apply_attn: false,
            apply_ffn: true,
            apply_dqn: false,
            infer_gate: "g_min".to_string(),
        }
    }
}

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

#[derive(Debug, Clone)]
pub struct Config {
    pub pool_name: String,
    pub up_six: Vec<String>,
    pub up_rate: f64,
    pub prob_6_base: f64,
    pub prob_5_base: f64,
    pub prob_4_base: f64,
    pub soft_pity_start: usize,
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
    pub luck_mode: String, // "probability" (default) or "dqn"
    pub fast_init: bool,
    pub ppo_mode: String,
    pub ppo_total_steps: usize,
    pub ppo_steps_per_update: usize,
    pub ppo_k_epochs: usize,
    pub ppo_batch_size: usize,
    pub ppo_context_len: usize,
    pub ppo_num_envs: usize,
    pub worker_max_threads: usize,
    pub worker_reserve_cores: usize,
    pub worker_priority: String,
    pub worker_stack_size_mb: usize,
    pub f2p_sim_count: usize,
    pub f2p_sim_count_prob: usize,
    pub f2p_sim_count_cost: usize,
    pub online_train: bool,
    pub online_train_dqn: bool,
    pub online_train_neural: bool,
    pub online_train_ppo: bool,
    pub train_interval_ms: usize,
    pub max_train_steps_per_tick: usize,
    pub language: Option<String>,
    pub achf: AchfConfig,
}

impl Config {
    pub fn default() -> Self {
        Config {
            pool_name: "Unknown".to_string(),
            up_six: vec![],
            up_rate: 0.5,
            prob_6_base: 0.008,
            prob_5_base: 0.08,
            prob_4_base: 0.912,
            soft_pity_start: 65,
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
            luck_mode: "probability".to_string(),
            fast_init: false,
            ppo_mode: "balanced".to_string(),
            ppo_total_steps: 0,
            ppo_steps_per_update: 0,
            ppo_k_epochs: 0,
            ppo_batch_size: 0,
            ppo_context_len: 0,
            ppo_num_envs: 1,
            worker_max_threads: 0,
            worker_reserve_cores: 1,
            worker_priority: "time_critical".to_string(),
            worker_stack_size_mb: 4,
            f2p_sim_count: 0,
            f2p_sim_count_prob: 0,
            f2p_sim_count_cost: 0,
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
                        eprintln!("[FATAL ERROR] Configuration file not found.");
                        eprintln!("Looked at: './{}' and '../../{}'", path, path);
                        eprintln!("Use --config <path> or explicitly pass --config default to use built-in defaults.");
                        if path == "data/config.json" {
                            eprintln!("[WARNING] Missing data/config.json. Falling back to built-in defaults for development.");
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

        let mut parser = JsonParser::new(&contents);
        let root = match parser.parse() {
            Ok(value) => value,
            Err(err) => {
                eprintln!("[FATAL ERROR] {}", err);
                std::process::exit(1);
            }
        };

        let mut config = Config::default();

        if let JsonValue::Object(map) = root {
            warn_unknown_fields(&map);
            if let Some(v) = map.get("pool_name") {
                config.pool_name = v.as_str().unwrap_or("").to_string();
            }
            if let Some(v) = map.get("up_six") {
                config.up_six = v.to_string_vec();
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
                config.soft_pity_start = v.as_f64().unwrap_or(65.0) as usize;
            }
            if let Some(v) = map.get("small_pity_guarantee") {
                config.small_pity_guarantee = v.as_f64().unwrap_or(80.0) as usize;
            }
            if let Some(v) = map.get("big_pity_cumulative") {
                config.big_pity_cumulative = v.as_f64().unwrap_or(120.0) as usize;
            }
            if let Some(v) = map.get("up_pity_soft") {
                config.up_pity_soft = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("five_star_pity") {
                config.five_star_pity = v.as_f64().unwrap_or(10.0) as usize;
            }
            if let Some(v) = map.get("always_5_star") {
                config.always_5_star = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("big_pity_requires_not_up") {
                config.big_pity_requires_not_up = v.as_bool().unwrap_or(true);
            }
            if let Some(v) = map.get("six_stars") {
                config.six_stars = v.to_string_vec();
            }
            if let Some(v) = map.get("five_stars") {
                config.five_stars = v.to_string_vec();
            }
            if let Some(v) = map.get("four_stars") {
                config.four_stars = v.to_string_vec();
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
                config.luck_mode = v.as_str().unwrap_or("probability").to_string();
            }
            if let Some(v) = map.get("fast_init") {
                config.fast_init = v.as_bool().unwrap_or(false);
            }
            if let Some(v) = map.get("ppo_mode") {
                config.ppo_mode = v.as_str().unwrap_or("balanced").to_string();
            }
            if let Some(v) = map.get("ppo_total_steps") {
                config.ppo_total_steps = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("ppo_steps_per_update") {
                config.ppo_steps_per_update = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("ppo_k_epochs") {
                config.ppo_k_epochs = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("ppo_batch_size") {
                config.ppo_batch_size = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("ppo_context_len") {
                config.ppo_context_len = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("ppo_num_envs") {
                config.ppo_num_envs = v.as_f64().unwrap_or(1.0) as usize;
            }
            if let Some(v) = map.get("worker_max_threads") {
                config.worker_max_threads = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("worker_reserve_cores") {
                config.worker_reserve_cores = v.as_f64().unwrap_or(1.0) as usize;
            }
            if let Some(v) = map.get("worker_priority") {
                config.worker_priority = v.as_str().unwrap_or("above_normal").to_string();
            }
            if let Some(v) = map.get("worker_stack_size_mb") {
                config.worker_stack_size_mb = v.as_f64().unwrap_or(4.0) as usize;
            }
            if let Some(v) = map.get("f2p_sim_count") {
                config.f2p_sim_count = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("f2p_sim_count_prob") {
                config.f2p_sim_count_prob = v.as_f64().unwrap_or(0.0) as usize;
            }
            if let Some(v) = map.get("f2p_sim_count_cost") {
                config.f2p_sim_count_cost = v.as_f64().unwrap_or(0.0) as usize;
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
                config.train_interval_ms = v.as_f64().unwrap_or(50.0) as usize;
            }
            if let Some(v) = map.get("max_train_steps_per_tick") {
                config.max_train_steps_per_tick = v.as_f64().unwrap_or(1.0) as usize;
            }
            if let Some(v) = map.get("language") {
                config.language = v.as_str().map(|s| s.to_string());
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
                    config.achf.proj_freq = v.as_f64().unwrap_or(8.0) as usize;
                }
                if let Some(v) = achf_map.get("proj_steps") {
                    config.achf.proj_steps = v.as_f64().unwrap_or(0.0) as usize;
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
                    config.achf.gate_warmup_steps = v.as_f64().unwrap_or(0.0) as usize;
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
                    config.achf.cache_min_rows = v.as_f64().unwrap_or(0.0) as usize;
                }
                if let Some(v) = achf_map.get("cache_min_nonzero_ratio") {
                    config.achf.cache_min_nonzero_ratio = v.as_f64().unwrap_or(0.0);
                }
                if let Some(v) = achf_map.get("cache_sparsity_sample_rows") {
                    config.achf.cache_sparsity_sample_rows = v.as_f64().unwrap_or(0.0) as usize;
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
                    config.achf.cache_latency_sample_every = v.as_f64().unwrap_or(1.0) as u64;
                }
                if let Some(v) = achf_map.get("cache_log_interval_steps") {
                    config.achf.cache_log_interval_steps = v.as_f64().unwrap_or(0.0) as usize;
                }
                if let Some(v) = achf_map.get("cache_log_per_layer") {
                    config.achf.cache_log_per_layer = v.as_bool().unwrap_or(false);
                }
                if let Some(v) = achf_map.get("rank") {
                    config.achf.rank = v.as_f64().unwrap_or(0.0) as usize;
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

        config
    }
}

impl Config {
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

fn parse_pool_config(pool_map: &HashMap<String, JsonValue>) -> PoolConfig {
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
            .map(|v| v.to_string_vec())
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
            .unwrap_or(65.0) as usize,
        small_pity_guarantee: pool_map
            .get("small_pity_guarantee")
            .and_then(|v| v.as_f64())
            .unwrap_or(80.0) as usize,
        big_pity_cumulative: pool_map
            .get("big_pity_cumulative")
            .and_then(|v| v.as_f64())
            .unwrap_or(120.0) as usize,
        up_pity_soft: pool_map
            .get("up_pity_soft")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as usize,
        five_star_pity: pool_map
            .get("five_star_pity")
            .and_then(|v| v.as_f64())
            .unwrap_or(10.0) as usize,
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
            .map(|v| v.to_string_vec())
            .unwrap_or_default(),
        five_stars: pool_map
            .get("five_stars")
            .map(|v| v.to_string_vec())
            .unwrap_or_default(),
        four_stars: pool_map
            .get("four_stars")
            .map(|v| v.to_string_vec())
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

fn warn_unknown_fields(map: &HashMap<String, JsonValue>) {
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
        "worker_max_threads",
        "worker_reserve_cores",
        "worker_priority",
        "worker_stack_size_mb",
        "f2p_sim_count",
        "f2p_sim_count_prob",
        "f2p_sim_count_cost",
        "online_train",
        "online_train_dqn",
        "online_train_neural",
        "online_train_ppo",
        "train_interval_ms",
        "max_train_steps_per_tick",
        "language",
        "achf",
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
        let mut parser = JsonParser::new(input);
        parser.parse().unwrap()
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
