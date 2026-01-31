use std::fs::File;
use std::collections::HashMap;
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
            JsonValue::Array(arr) => arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => Vec::new(),
        }
    }
}

struct JsonParser {
    chars: Vec<char>,
    pos: usize,
}

impl JsonParser {
    fn new(input: &str) -> Self {
        JsonParser {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn parse(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        if self.pos >= self.chars.len() {
            return Ok(JsonValue::Null);
        }

        match self.chars[self.pos] {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' => self.parse_string().map(JsonValue::String),
            't' => { self.consume("true")?; Ok(JsonValue::Bool(true)) }
            'f' => { self.consume("false")?; Ok(JsonValue::Bool(false)) }
            'n' => { self.consume("null")?; Ok(JsonValue::Null) }
            c if c == '-' || c.is_ascii_digit() => self.parse_number(),
            c => Err(format!("Unexpected character: {}", c)),
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn consume(&mut self, s: &str) -> Result<(), String> {
        for c in s.chars() {
            if self.pos >= self.chars.len() || self.chars[self.pos] != c {
                return Err(format!("Expected '{}'", s));
            }
            self.pos += 1;
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.pos += 1; // skip "
        let mut s = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            self.pos += 1;
            if c == '"' {
                return Ok(s);
            }
            if c == '\\' {
                if self.pos >= self.chars.len() { return Err("Unexpected EOF in string".to_string()); }
                let escaped = self.chars[self.pos];
                self.pos += 1;
                match escaped {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'b' => s.push('\x08'),
                    'f' => s.push('\x0c'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    // Basic unicode support could be added here if needed
                    _ => s.push(escaped),
                }
            } else {
                s.push(c);
            }
        }
        Err("Unexpected EOF in string".to_string())
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.pos < self.chars.len() && self.chars[self.pos] == '-' {
            self.pos += 1;
        }
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.chars.len() && self.chars[self.pos] == '.' {
            self.pos += 1;
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| format!("Invalid number: {}", s))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.pos += 1; // skip [
        let mut arr = Vec::new();
        self.skip_whitespace();
        if self.pos < self.chars.len() && self.chars[self.pos] == ']' {
            self.pos += 1;
            return Ok(JsonValue::Array(arr));
        }
        loop {
            arr.push(self.parse()?);
            self.skip_whitespace();
            if self.pos >= self.chars.len() { return Err("Unexpected EOF in array".to_string()); }
            match self.chars[self.pos] {
                ',' => self.pos += 1,
                ']' => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(arr));
                }
                c => return Err(format!("Expected ',' or ']' in array, found {}", c)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.pos += 1; // skip {
        let mut map = HashMap::new();
        self.skip_whitespace();
        if self.pos < self.chars.len() && self.chars[self.pos] == '}' {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_whitespace();
            if self.chars[self.pos] != '"' {
                return Err("Expected string key in object".to_string());
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.pos >= self.chars.len() || self.chars[self.pos] != ':' {
                return Err("Expected ':' after key".to_string());
            }
            self.pos += 1; // skip :
            let value = self.parse()?;
            map.insert(key, value);
            
            self.skip_whitespace();
            if self.pos >= self.chars.len() { return Err("Unexpected EOF in object".to_string()); }
            match self.chars[self.pos] {
                ',' => self.pos += 1,
                '}' => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                c => return Err(format!("Expected ',' or '}}' in object, found {}", c)),
            }
        }
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
        }
    }

    pub fn load(path: &str) -> Self {
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
                    },
                    Err(_) => {
                        // If still not found, we CANNOT proceed safely without config in a data-driven app.
                        // We must pause to let user see the error before "crashing" (closing window).
                        println!("\n[FATAL ERROR] Configuration file not found!");
                        println!("Looked at: './{}' and '../../{}'", path, path);
                        println!("Please ensure the 'data' folder is in the same directory as the executable.");
                        println!("\nPress Enter to exit...");
                        let mut s = String::new();
                        std::io::stdin().read_line(&mut s).unwrap();
                        std::process::exit(1);
                    }
                }
            }
        };

        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("Failed to read config file");
        
        let mut parser = JsonParser::new(&contents);
        let root = parser.parse().expect("Failed to parse JSON config");
        
        let mut config = Config::default();

        if let JsonValue::Object(map) = root {
             if let Some(v) = map.get("pool_name") { config.pool_name = v.as_str().unwrap_or("").to_string(); }
             if let Some(v) = map.get("up_six") { config.up_six = v.to_string_vec(); }
             if let Some(v) = map.get("prob_6_base") { config.prob_6_base = v.as_f64().unwrap_or(0.008); }
             if let Some(v) = map.get("prob_5_base") { config.prob_5_base = v.as_f64().unwrap_or(0.08); }
             if let Some(v) = map.get("prob_4_base") { config.prob_4_base = v.as_f64().unwrap_or(0.912); }
             if let Some(v) = map.get("soft_pity_start") { config.soft_pity_start = v.as_f64().unwrap_or(65.0) as usize; }
             if let Some(v) = map.get("small_pity_guarantee") { config.small_pity_guarantee = v.as_f64().unwrap_or(80.0) as usize; }
             if let Some(v) = map.get("big_pity_cumulative") { config.big_pity_cumulative = v.as_f64().unwrap_or(120.0) as usize; }
             if let Some(v) = map.get("six_stars") { config.six_stars = v.to_string_vec(); }
             if let Some(v) = map.get("five_stars") { config.five_stars = v.to_string_vec(); }
             if let Some(v) = map.get("four_stars") { config.four_stars = v.to_string_vec(); }
             if let Some(v) = map.get("luck_mode") { config.luck_mode = v.as_str().unwrap_or("probability").to_string(); }
             if let Some(v) = map.get("fast_init") { config.fast_init = v.as_bool().unwrap_or(false); }
             if let Some(v) = map.get("ppo_mode") { config.ppo_mode = v.as_str().unwrap_or("balanced").to_string(); }
             if let Some(v) = map.get("ppo_total_steps") { config.ppo_total_steps = v.as_f64().unwrap_or(0.0) as usize; }
             if let Some(v) = map.get("ppo_steps_per_update") { config.ppo_steps_per_update = v.as_f64().unwrap_or(0.0) as usize; }
             if let Some(v) = map.get("ppo_k_epochs") { config.ppo_k_epochs = v.as_f64().unwrap_or(0.0) as usize; }
             if let Some(v) = map.get("ppo_batch_size") { config.ppo_batch_size = v.as_f64().unwrap_or(0.0) as usize; }
             if let Some(v) = map.get("ppo_context_len") { config.ppo_context_len = v.as_f64().unwrap_or(0.0) as usize; }
             if let Some(v) = map.get("worker_max_threads") { config.worker_max_threads = v.as_f64().unwrap_or(0.0) as usize; }
             if let Some(v) = map.get("worker_reserve_cores") { config.worker_reserve_cores = v.as_f64().unwrap_or(1.0) as usize; }
             if let Some(v) = map.get("worker_priority") { config.worker_priority = v.as_str().unwrap_or("above_normal").to_string(); }
             if let Some(v) = map.get("worker_stack_size_mb") { config.worker_stack_size_mb = v.as_f64().unwrap_or(4.0) as usize; }
        }

        config
    }
}
