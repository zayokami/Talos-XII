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

#[derive(Debug, Clone)]
pub struct Config {
    pub pool_name: String,
    pub up_six: Vec<String>,
    pub prob_6_base: f64,
    pub prob_5_base: f64,
    pub prob_4_base: f64,
    pub soft_pity_start: usize,
    pub small_pity_guarantee: usize,
    pub big_pity_cumulative: usize,
    pub six_stars: Vec<String>,
    pub five_stars: Vec<String>,
    pub four_stars: Vec<String>,
    pub luck_mode: String, // "probability" (default) or "dqn"
    pub fast_init: bool,
    pub ppo_mode: String,
    pub ppo_total_steps: usize,
    pub ppo_steps_per_update: usize,
    pub ppo_k_epochs: usize,
    pub ppo_batch_size: usize,
    pub ppo_context_len: usize,
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
}

impl Config {
    pub fn default() -> Self {
        Config {
            pool_name: "Unknown".to_string(),
            up_six: vec![],
            prob_6_base: 0.008,
            prob_5_base: 0.08,
            prob_4_base: 0.912,
            soft_pity_start: 65,
            small_pity_guarantee: 80,
            big_pity_cumulative: 120,
            six_stars: vec![],
            five_stars: vec![],
            four_stars: vec![],
            luck_mode: "probability".to_string(),
            fast_init: false,
            ppo_mode: "balanced".to_string(),
            ppo_total_steps: 0,
            ppo_steps_per_update: 0,
            ppo_k_epochs: 0,
            ppo_batch_size: 0,
            ppo_context_len: 0,
            worker_max_threads: 0,
            worker_reserve_cores: 1,
            worker_priority: "above_normal".to_string(),
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
            if let Some(v) = map.get("six_stars") {
                config.six_stars = v.to_string_vec();
            }
            if let Some(v) = map.get("five_stars") {
                config.five_stars = v.to_string_vec();
            }
            if let Some(v) = map.get("four_stars") {
                config.four_stars = v.to_string_vec();
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
        }

        config
    }
}

fn warn_unknown_fields(map: &HashMap<String, JsonValue>) {
    let known: HashSet<&'static str> = [
        "pool_name",
        "up_six",
        "prob_6_base",
        "prob_5_base",
        "prob_4_base",
        "soft_pity_start",
        "small_pity_guarantee",
        "big_pity_cumulative",
        "six_stars",
        "five_stars",
        "four_stars",
        "luck_mode",
        "fast_init",
        "ppo_mode",
        "ppo_total_steps",
        "ppo_steps_per_update",
        "ppo_k_epochs",
        "ppo_batch_size",
        "ppo_context_len",
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
