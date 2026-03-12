use crate::autograd::{Tensor, TensorReadGuard};
use crate::config::AchfConfig;
use crate::nn::{Linear, Module};
use crate::simd::add_scaled_row;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum GateMode {
    GradEma,
    FimTrace,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProjMode {
    None,
    RowCol,
    Sinkhorn,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AchfLayer {
    pub weight: Linear,
    pub down: Option<Linear>,
    pub up: Option<Linear>,
    pub config: AchfConfig,
    #[serde(skip, default = "default_state")]
    pub state: Arc<RwLock<AchfState>>,
    #[serde(skip, default = "default_cache")]
    pub cache: Arc<RwLock<AchfCache>>,
}

#[derive(Clone)]
pub struct AchfState {
    pub step: usize,
    pub gate_step: usize,
    pub grad_ema: f64,
    pub fim_ema: f64,
    pub last_gate: f64,
    pub g_min_ema: f64,
    pub freeze_projection: bool,
}

#[derive(Clone)]
pub struct AchfCache {
    pub dense: Option<Vec<f64>>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub calls: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_skips: u64,
    pub low_rank_paths: u64,
    pub dense_paths: u64,
    pub ema_cached_ns: f64,
    pub ema_cached_long_ns: f64,
    pub ema_low_rank_ns: f64,
    pub ema_low_rank_long_ns: f64,
    pub decision_ema_ns: f64,
    pub decision_ema_long_ns: f64,
    pub adaptive_bias: f64,
    pub latency_samples: u64,
    pub decision_samples: u64,
}

fn default_state() -> Arc<RwLock<AchfState>> {
    Arc::new(RwLock::new(AchfState {
        step: 0,
        gate_step: 0,
        grad_ema: 0.0,
        fim_ema: 0.0,
        last_gate: 1.0,
        g_min_ema: 0.0,
        freeze_projection: false,
    }))
}

fn default_cache() -> Arc<RwLock<AchfCache>> {
    Arc::new(RwLock::new(AchfCache {
        dense: None,
        in_dim: 0,
        out_dim: 0,
        calls: 0,
        cache_hits: 0,
        cache_misses: 0,
        cache_skips: 0,
        low_rank_paths: 0,
        dense_paths: 0,
        ema_cached_ns: 0.0,
        ema_cached_long_ns: 0.0,
        ema_low_rank_ns: 0.0,
        ema_low_rank_long_ns: 0.0,
        decision_ema_ns: 0.0,
        decision_ema_long_ns: 0.0,
        adaptive_bias: 0.0,
        latency_samples: 0,
        decision_samples: 0,
    }))
}

#[derive(Clone, Copy)]
pub struct AchfCacheStats {
    pub calls: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_skips: u64,
    pub low_rank_paths: u64,
    pub dense_paths: u64,
    pub ema_cached_ns: f64,
    pub ema_cached_long_ns: f64,
    pub ema_low_rank_ns: f64,
    pub ema_low_rank_long_ns: f64,
    pub decision_ema_ns: f64,
    pub decision_ema_long_ns: f64,
    pub adaptive_bias: f64,
    pub latency_samples: u64,
    pub decision_samples: u64,
}

pub fn aggregate_cache_stats_iter<I>(iter: I) -> AchfCacheStats
where
    I: IntoIterator<Item = AchfCacheStats>,
{
    let mut out = AchfCacheStats {
        calls: 0,
        cache_hits: 0,
        cache_misses: 0,
        cache_skips: 0,
        low_rank_paths: 0,
        dense_paths: 0,
        ema_cached_ns: 0.0,
        ema_cached_long_ns: 0.0,
        ema_low_rank_ns: 0.0,
        ema_low_rank_long_ns: 0.0,
        decision_ema_ns: 0.0,
        decision_ema_long_ns: 0.0,
        adaptive_bias: 1.0,
        latency_samples: 0,
        decision_samples: 0,
    };
    let mut count_cached = 0usize;
    let mut count_low_rank = 0usize;
    let mut count_cached_long = 0usize;
    let mut count_low_rank_long = 0usize;
    let mut count_decision = 0usize;
    let mut count_decision_long = 0usize;
    let mut count_bias = 0usize;
    for s in iter {
        out.calls += s.calls;
        out.cache_hits += s.cache_hits;
        out.cache_misses += s.cache_misses;
        out.cache_skips += s.cache_skips;
        out.low_rank_paths += s.low_rank_paths;
        out.dense_paths += s.dense_paths;
        out.latency_samples += s.latency_samples;
        out.decision_samples += s.decision_samples;
        if s.ema_cached_ns > 0.0 {
            out.ema_cached_ns += s.ema_cached_ns;
            count_cached += 1;
        }
        if s.ema_cached_long_ns > 0.0 {
            out.ema_cached_long_ns += s.ema_cached_long_ns;
            count_cached_long += 1;
        }
        if s.ema_low_rank_ns > 0.0 {
            out.ema_low_rank_ns += s.ema_low_rank_ns;
            count_low_rank += 1;
        }
        if s.ema_low_rank_long_ns > 0.0 {
            out.ema_low_rank_long_ns += s.ema_low_rank_long_ns;
            count_low_rank_long += 1;
        }
        if s.decision_ema_ns > 0.0 {
            out.decision_ema_ns += s.decision_ema_ns;
            count_decision += 1;
        }
        if s.decision_ema_long_ns > 0.0 {
            out.decision_ema_long_ns += s.decision_ema_long_ns;
            count_decision_long += 1;
        }
        if s.adaptive_bias > 0.0 {
            out.adaptive_bias += s.adaptive_bias;
            count_bias += 1;
        }
    }
    if count_cached > 0 {
        out.ema_cached_ns /= count_cached as f64;
    }
    if count_low_rank > 0 {
        out.ema_low_rank_ns /= count_low_rank as f64;
    }
    if count_cached_long > 0 {
        out.ema_cached_long_ns /= count_cached_long as f64;
    }
    if count_low_rank_long > 0 {
        out.ema_low_rank_long_ns /= count_low_rank_long as f64;
    }
    if count_decision > 0 {
        out.decision_ema_ns /= count_decision as f64;
    }
    if count_decision_long > 0 {
        out.decision_ema_long_ns /= count_decision_long as f64;
    }
    if count_bias > 0 {
        out.adaptive_bias /= count_bias as f64;
    } else {
        out.adaptive_bias = 1.0;
    }
    out
}

impl AchfLayer {
    pub fn new(dim: usize, config: AchfConfig, seed: u64) -> Self {
        let weight = Linear::new(dim, dim, false, seed);
        let rank = if config.rank > 0 && config.rank < dim {
            config.rank
        } else {
            0
        };
        let down = if rank > 0 {
            Some(Linear::new(dim, rank, false, seed.wrapping_add(1)))
        } else {
            None
        };
        let up = if rank > 0 {
            Some(Linear::new(rank, dim, false, seed.wrapping_add(2)))
        } else {
            None
        };
        let layer = Self {
            weight,
            down,
            up,
            config,
            state: default_state(),
            cache: default_cache(),
        };
        {
            let mut cache = layer.cache.write().unwrap();
            cache.adaptive_bias = layer.config.cache_cost_bias;
        }
        layer
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        if self.is_low_rank() {
            let mut p = Vec::new();
            if let Some(down) = &self.down {
                p.extend(down.parameters());
            }
            if let Some(up) = &self.up {
                p.extend(up.parameters());
            }
            p
        } else {
            self.weight.parameters()
        }
    }

    pub fn forward_residual(&self, x: &Tensor) -> Tensor {
        if !self.config.enabled {
            return Tensor::zeros(x.shape.clone());
        }
        self.maybe_project();
        let g = self.compute_gate();
        let g_tensor = Tensor::new(vec![g], vec![1]).broadcast(x.shape.clone());
        let hc = if self.is_low_rank() {
            let down = self.down.as_ref().unwrap();
            let up = self.up.as_ref().unwrap();
            let low = down.forward(x);
            up.forward(&low)
        } else {
            self.weight.forward(x)
        };
        hc * g_tensor
    }

    pub fn forward_inference_residual(&self, x: &[f64]) -> Vec<f64> {
        if !self.config.enabled {
            return vec![0.0; x.len()];
        }
        self.maybe_project();
        let g = self.infer_gate_value();
        let sample_latency = self.should_sample_latency();
        let (path, decision_ns) = if sample_latency {
            let start = Instant::now();
            let path = self.choose_inference_path(x);
            (path, start.elapsed().as_nanos() as f64)
        } else {
            (self.choose_inference_path(x), 0.0)
        };
        if decision_ns > 0.0 {
            self.record_decision_latency(decision_ns);
        }
        let (mut out, elapsed_ns) = if sample_latency {
            let start = Instant::now();
            let out = match path {
                InferencePath::Cached => self.forward_inference_cached(x).unwrap(),
                InferencePath::LowRank => {
                    let down = self.down.as_ref().unwrap();
                    let up = self.up.as_ref().unwrap();
                    let low = down.forward_inference(x);
                    up.forward_inference(&low)
                }
                InferencePath::Dense => self.weight.forward_inference(x),
            };
            (out, start.elapsed().as_nanos() as f64)
        } else {
            let out = match path {
                InferencePath::Cached => self.forward_inference_cached(x).unwrap(),
                InferencePath::LowRank => {
                    let down = self.down.as_ref().unwrap();
                    let up = self.up.as_ref().unwrap();
                    let low = down.forward_inference(x);
                    up.forward_inference(&low)
                }
                InferencePath::Dense => self.weight.forward_inference(x),
            };
            (out, 0.0)
        };
        if elapsed_ns > 0.0 {
            self.record_path_latency(path, elapsed_ns);
        }
        for v in out.iter_mut() {
            *v *= g;
        }
        out
    }

    pub fn update_after_backward(&self) {
        if !self.config.enabled {
            return;
        }
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        if self.is_low_rank() {
            if let Some(down) = &self.down {
                let grad = down.weight.grad.read().unwrap();
                for &v in grad.iter() {
                    sum_sq += v * v;
                }
                count += grad.len();
            }
            if let Some(up) = &self.up {
                let grad = up.weight.grad.read().unwrap();
                for &v in grad.iter() {
                    sum_sq += v * v;
                }
                count += grad.len();
            }
        } else {
            let grad = self.weight.weight.grad.read().unwrap();
            for &v in grad.iter() {
                sum_sq += v * v;
            }
            count += grad.len();
        }
        if count == 0 {
            return;
        }
        let mean_sq = sum_sq / count as f64;
        let grad_rms = mean_sq.sqrt();
        let mut state = self.state.write().unwrap();
        match self.gate_mode() {
            GateMode::GradEma => {
                state.grad_ema = self.config.gate_momentum * state.grad_ema
                    + (1.0 - self.config.gate_momentum) * grad_rms;
            }
            GateMode::FimTrace => {
                state.fim_ema = self.config.gate_momentum * state.fim_ema
                    + (1.0 - self.config.gate_momentum) * mean_sq;
            }
        }
    }

    pub fn orthogonal_penalty(&self) -> Option<Tensor> {
        if !self.config.enabled {
            return None;
        }
        if self.config.lambda_ortho <= 0.0 {
            return None;
        }
        let w = if self.is_low_rank() {
            self.up.as_ref().unwrap().weight.clone()
        } else {
            self.weight.weight.clone()
        };
        let wt = w.transpose2d();
        let wtw = wt.matmul(&w);
        let dim = w.shape[0];
        let mut id_data = vec![0.0; dim * dim];
        for i in 0..dim {
            id_data[i * dim + i] = 1.0;
        }
        let id = Tensor::new(id_data, vec![dim, dim]);
        let diff = wtw - id;
        let sq = &diff * &diff;
        let mean = sq.mean();
        let scale = Tensor::new(vec![self.config.lambda_ortho], vec![1]);
        Some(mean * scale)
    }

    #[cfg(test)]
    pub fn last_gate(&self) -> f64 {
        self.state.read().unwrap().last_gate
    }

    #[cfg(test)]
    pub fn last_g_min(&self) -> f64 {
        self.state.read().unwrap().g_min_ema
    }

    pub fn cache_stats(&self) -> AchfCacheStats {
        let cache = self.cache.read().unwrap();
        AchfCacheStats {
            calls: cache.calls,
            cache_hits: cache.cache_hits,
            cache_misses: cache.cache_misses,
            cache_skips: cache.cache_skips,
            low_rank_paths: cache.low_rank_paths,
            dense_paths: cache.dense_paths,
            ema_cached_ns: cache.ema_cached_ns,
            ema_cached_long_ns: cache.ema_cached_long_ns,
            ema_low_rank_ns: cache.ema_low_rank_ns,
            ema_low_rank_long_ns: cache.ema_low_rank_long_ns,
            decision_ema_ns: cache.decision_ema_ns,
            decision_ema_long_ns: cache.decision_ema_long_ns,
            adaptive_bias: cache.adaptive_bias,
            latency_samples: cache.latency_samples,
            decision_samples: cache.decision_samples,
        }
    }

    pub fn freeze_for_inference(&self) {
        if !self.config.enabled {
            return;
        }
        self.project_weight();
        self.prepare_inference_cache();
        let mut state = self.state.write().unwrap();
        state.freeze_projection = true;
        state.g_min_ema = self.config.g_min;
    }

    fn gate_mode(&self) -> GateMode {
        match self.config.gate_mode.as_str() {
            "fim_trace" | "fim" => GateMode::FimTrace,
            _ => GateMode::GradEma,
        }
    }

    fn proj_mode(&self) -> ProjMode {
        match self.config.proj_mode.as_str() {
            "rowcol" => ProjMode::RowCol,
            "sinkhorn" => ProjMode::Sinkhorn,
            _ => ProjMode::None,
        }
    }

    fn compute_gate(&self) -> f64 {
        let mut state = self.state.write().unwrap();
        state.gate_step += 1;
        if self.config.gate_warmup_steps > 0 && state.gate_step <= self.config.gate_warmup_steps {
            state.last_gate = 1.0;
            state.g_min_ema = self.config.g_min;
            return 1.0;
        }
        let mut k = match self.gate_mode() {
            GateMode::GradEma => state.grad_ema,
            GateMode::FimTrace => state.fim_ema.sqrt(),
        };
        if self.config.gate_k_clip > 0.0 && k > self.config.gate_k_clip {
            k = self.config.gate_k_clip;
        }
        let x = self.config.gate_alpha - self.config.gate_beta * k;
        let mut g = 1.0 / (1.0 + (-x).exp());
        let target_mid = 0.5 * (self.config.g_target_min + self.config.g_target_max);
        let mut g_min = self.config.g_min;
        if self.config.g_min_adapt_rate > 0.0 && target_mid > 0.0 {
            let diff = (target_mid - g) / target_mid;
            g_min *= (1.0 + self.config.g_min_adapt_rate * diff).max(0.1);
            if g_min < self.config.g_min {
                g_min = self.config.g_min;
            }
            if g_min > 0.95 {
                g_min = 0.95;
            }
        }
        if g < g_min {
            g = g_min;
        }
        if g > 1.0 {
            g = 1.0;
        }
        state.last_gate = g;
        state.g_min_ema = self.config.g_min_momentum * state.g_min_ema
            + (1.0 - self.config.g_min_momentum) * g_min;
        g
    }

    fn infer_gate_value(&self) -> f64 {
        match self.config.infer_gate.as_str() {
            "one" => 1.0,
            "last" => self.state.read().unwrap().last_gate.max(self.config.g_min),
            _ => self.config.g_min,
        }
    }

    fn maybe_project(&self) {
        if !self.config.enabled {
            return;
        }
        if self.config.proj_freq == 0 {
            return;
        }
        let mut state = self.state.write().unwrap();
        if state.freeze_projection {
            return;
        }
        state.step += 1;
        if !state.step.is_multiple_of(self.config.proj_freq) {
            return;
        }
        drop(state);
        self.project_weight();
    }

    fn project_weight(&self) {
        match self.proj_mode() {
            ProjMode::None => {}
            ProjMode::RowCol => self.project_rowcol(),
            ProjMode::Sinkhorn => self.project_sinkhorn(),
        }
        self.clear_cache();
    }

    fn project_rowcol(&self) {
        if self.is_low_rank() {
            if let Some(down) = &self.down {
                let mut w = down.weight.data.write().unwrap();
                let rows = down.in_features;
                let cols = down.out_features;
                rowcol_project(&mut w, rows, cols);
            }
            if let Some(up) = &self.up {
                let mut w = up.weight.data.write().unwrap();
                let rows = up.in_features;
                let cols = up.out_features;
                rowcol_project(&mut w, rows, cols);
            }
        } else {
            let mut w = self.weight.weight.data.write().unwrap();
            let rows = self.weight.in_features;
            let cols = self.weight.out_features;
            rowcol_project(&mut w, rows, cols);
        }
    }

    fn project_sinkhorn(&self) {
        let steps = if self.config.proj_steps == 0 {
            1
        } else {
            self.config.proj_steps
        };
        if self.is_low_rank() {
            if let Some(down) = &self.down {
                let mut w = down.weight.data.write().unwrap();
                sinkhorn_project(&mut w, down.in_features, down.out_features, steps);
            }
            if let Some(up) = &self.up {
                let mut w = up.weight.data.write().unwrap();
                sinkhorn_project(&mut w, up.in_features, up.out_features, steps);
            }
        } else {
            let mut w = self.weight.weight.data.write().unwrap();
            sinkhorn_project(
                &mut w,
                self.weight.in_features,
                self.weight.out_features,
                steps,
            );
        }
    }

    pub fn load_state_dict(&mut self, other: &AchfLayer) {
        if self.is_low_rank() {
            if let (Some(dst), Some(src)) = (&mut self.down, &other.down) {
                copy_linear(dst, src);
            }
            if let (Some(dst), Some(src)) = (&mut self.up, &other.up) {
                copy_linear(dst, src);
            }
        } else {
            copy_linear(&mut self.weight, &other.weight);
        }
        self.clear_cache();
    }

    pub fn soft_update(&mut self, source: &AchfLayer, tau: f64) {
        if self.is_low_rank() {
            if let (Some(dst), Some(src)) = (&mut self.down, &source.down) {
                soft_update_linear(dst, src, tau);
            }
            if let (Some(dst), Some(src)) = (&mut self.up, &source.up) {
                soft_update_linear(dst, src, tau);
            }
        } else {
            soft_update_linear(&mut self.weight, &source.weight, tau);
        }
        self.clear_cache();
    }

    fn is_low_rank(&self) -> bool {
        self.down.is_some() && self.up.is_some()
    }

    fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.dense = None;
        cache.in_dim = 0;
        cache.out_dim = 0;
        cache.ema_cached_ns = 0.0;
        cache.ema_cached_long_ns = 0.0;
        cache.ema_low_rank_ns = 0.0;
        cache.ema_low_rank_long_ns = 0.0;
        cache.decision_ema_ns = 0.0;
        cache.decision_ema_long_ns = 0.0;
        cache.adaptive_bias = self.config.cache_cost_bias;
        cache.latency_samples = 0;
        cache.decision_samples = 0;
    }

    fn prepare_inference_cache(&self) {
        if !self.is_low_rank() {
            return;
        }
        let down = self.down.as_ref().unwrap();
        let up = self.up.as_ref().unwrap();
        let in_dim = down.in_features;
        let rank = down.out_features;
        let out_dim = up.out_features;
        // Use batch lock for better performance
        let guards = TensorReadGuard::new(&[&down.weight, &up.weight]);
        let down_data = guards.get(0);
        let up_data = guards.get(1);
        let mut dense = vec![0.0; in_dim * out_dim];
        for i in 0..in_dim {
            let down_row = &down_data[i * rank..(i + 1) * rank];
            let out_row = &mut dense[i * out_dim..(i + 1) * out_dim];
            for r in 0..rank {
                let scale = down_row[r];
                if scale == 0.0 {
                    continue;
                }
                let up_row = &up_data[r * out_dim..(r + 1) * out_dim];
                add_scaled_row(out_row, up_row, scale);
            }
        }
        let mut cache = self.cache.write().unwrap();
        cache.dense = Some(dense);
        cache.in_dim = in_dim;
        cache.out_dim = out_dim;
    }

    fn forward_inference_cached(&self, x: &[f64]) -> Option<Vec<f64>> {
        let cache = self.cache.read().unwrap();
        let dense = cache.dense.as_ref()?;
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        if in_dim == 0 || out_dim == 0 {
            return None;
        }
        let num_rows = x.len() / in_dim;
        if self.config.cache_min_rows > 0 && num_rows < self.config.cache_min_rows {
            return None;
        }
        let mut out = vec![0.0; num_rows * out_dim];
        for r in 0..num_rows {
            let row_offset_in = r * in_dim;
            let row_offset_out = r * out_dim;
            for i in 0..in_dim {
                let scale = x[row_offset_in + i];
                if scale == 0.0 {
                    continue;
                }
                let w_row = &dense[i * out_dim..(i + 1) * out_dim];
                let out_row = &mut out[row_offset_out..row_offset_out + out_dim];
                add_scaled_row(out_row, w_row, scale);
            }
        }
        Some(out)
    }

    fn choose_inference_path(&self, x: &[f64]) -> InferencePath {
        if !self.is_low_rank() {
            let mut cache = self.cache.write().unwrap();
            cache.calls += 1;
            cache.dense_paths += 1;
            return InferencePath::Dense;
        }
        let (use_cache, skip_cache, has_cache) = self.should_use_cache(x);
        let mut cache = self.cache.write().unwrap();
        cache.calls += 1;
        if use_cache {
            cache.cache_hits += 1;
            return InferencePath::Cached;
        }
        if has_cache {
            if skip_cache {
                cache.cache_skips += 1;
            } else {
                cache.cache_misses += 1;
            }
        }
        cache.low_rank_paths += 1;
        InferencePath::LowRank
    }

    fn should_use_cache(&self, x: &[f64]) -> (bool, bool, bool) {
        let cache = self.cache.read().unwrap();
        if cache.dense.is_none() {
            return (false, false, false);
        }
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        if in_dim == 0 || out_dim == 0 {
            return (false, false, false);
        }
        let num_rows = x.len() / in_dim;
        if self.config.cache_min_rows > 0 && num_rows < self.config.cache_min_rows {
            return (false, true, true);
        }
        let nonzero_ratio = self.estimate_nonzero_ratio(x, in_dim, num_rows);
        if self.config.cache_min_nonzero_ratio > 0.0
            && nonzero_ratio < self.config.cache_min_nonzero_ratio
        {
            return (false, true, true);
        }
        let rank = self.down.as_ref().unwrap().out_features;
        let dense_cost = nonzero_ratio * (in_dim * out_dim) as f64;
        let low_rank_cost = nonzero_ratio * (in_dim * rank) as f64 + (rank * out_dim) as f64;
        let bias = if self.config.cache_adapt_rate > 0.0 {
            cache.adaptive_bias
        } else {
            self.config.cache_cost_bias
        };
        let use_cache = dense_cost * bias <= low_rank_cost;
        (use_cache, false, true)
    }

    fn estimate_nonzero_ratio(&self, x: &[f64], in_dim: usize, num_rows: usize) -> f64 {
        if x.is_empty() || in_dim == 0 || num_rows == 0 {
            return 1.0;
        }
        let sample_rows = if self.config.cache_sparsity_sample_rows == 0 {
            num_rows
        } else {
            self.config.cache_sparsity_sample_rows.min(num_rows)
        };
        let mut nonzero = 0usize;
        let mut total = 0usize;
        for r in 0..sample_rows {
            let start = r * in_dim;
            let end = start + in_dim;
            for &v in &x[start..end] {
                if v != 0.0 {
                    nonzero += 1;
                }
            }
            total += in_dim;
        }
        if total == 0 {
            1.0
        } else {
            nonzero as f64 / total as f64
        }
    }

    fn record_path_latency(&self, path: InferencePath, elapsed_ns: f64) {
        if elapsed_ns <= 0.0 {
            return;
        }
        let mut cache = self.cache.write().unwrap();
        let ema = self.config.cache_latency_ema;
        let ema_long = self.config.cache_latency_long_ema;
        match path {
            InferencePath::Cached => {
                if cache.ema_cached_ns == 0.0 || ema <= 0.0 {
                    cache.ema_cached_ns = elapsed_ns;
                } else {
                    cache.ema_cached_ns = ema * cache.ema_cached_ns + (1.0 - ema) * elapsed_ns;
                }
                if cache.ema_cached_long_ns == 0.0 || ema_long <= 0.0 {
                    cache.ema_cached_long_ns = elapsed_ns;
                } else {
                    cache.ema_cached_long_ns =
                        ema_long * cache.ema_cached_long_ns + (1.0 - ema_long) * elapsed_ns;
                }
            }
            InferencePath::LowRank => {
                if cache.ema_low_rank_ns == 0.0 || ema <= 0.0 {
                    cache.ema_low_rank_ns = elapsed_ns;
                } else {
                    cache.ema_low_rank_ns = ema * cache.ema_low_rank_ns + (1.0 - ema) * elapsed_ns;
                }
                if cache.ema_low_rank_long_ns == 0.0 || ema_long <= 0.0 {
                    cache.ema_low_rank_long_ns = elapsed_ns;
                } else {
                    cache.ema_low_rank_long_ns =
                        ema_long * cache.ema_low_rank_long_ns + (1.0 - ema_long) * elapsed_ns;
                }
            }
            InferencePath::Dense => {}
        }
        cache.latency_samples += 1;
        if self.config.cache_adapt_rate > 0.0
            && cache.ema_cached_ns > 0.0
            && cache.ema_low_rank_ns > 0.0
        {
            let short_ratio = (cache.ema_cached_ns - cache.ema_low_rank_ns) / cache.ema_low_rank_ns;
            let mut ratio = short_ratio;
            if cache.ema_cached_long_ns > 0.0 && cache.ema_low_rank_long_ns > 0.0 {
                let long_ratio = (cache.ema_cached_long_ns - cache.ema_low_rank_long_ns)
                    / cache.ema_low_rank_long_ns;
                let alpha = self.config.cache_adapt_blend;
                ratio = alpha * long_ratio + (1.0 - alpha) * short_ratio;
            }
            let mut bias = cache.adaptive_bias;
            bias *= 1.0 + self.config.cache_adapt_rate * ratio;
            if bias < self.config.cache_bias_min {
                bias = self.config.cache_bias_min;
            }
            if bias > self.config.cache_bias_max {
                bias = self.config.cache_bias_max;
            }
            cache.adaptive_bias = bias;
        }
    }

    fn record_decision_latency(&self, elapsed_ns: f64) {
        if elapsed_ns <= 0.0 {
            return;
        }
        let mut cache = self.cache.write().unwrap();
        let ema = self.config.cache_latency_ema;
        let ema_long = self.config.cache_latency_long_ema;
        if cache.decision_ema_ns == 0.0 || ema <= 0.0 {
            cache.decision_ema_ns = elapsed_ns;
        } else {
            cache.decision_ema_ns = ema * cache.decision_ema_ns + (1.0 - ema) * elapsed_ns;
        }
        if cache.decision_ema_long_ns == 0.0 || ema_long <= 0.0 {
            cache.decision_ema_long_ns = elapsed_ns;
        } else {
            cache.decision_ema_long_ns =
                ema_long * cache.decision_ema_long_ns + (1.0 - ema_long) * elapsed_ns;
        }
        cache.decision_samples += 1;
    }

    fn should_sample_latency(&self) -> bool {
        if self.config.cache_latency_sample_every == 0 {
            return true;
        }
        let cache = self.cache.read().unwrap();
        cache
            .calls
            .is_multiple_of(self.config.cache_latency_sample_every)
    }
}

#[derive(Clone, Copy)]
enum InferencePath {
    Cached,
    LowRank,
    Dense,
}

fn rowcol_project(w: &mut [f64], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum_sq = 0.0;
        for c in 0..cols {
            let v = w[r * cols + c];
            sum_sq += v * v;
        }
        if sum_sq > 0.0 {
            let norm = sum_sq.sqrt();
            for c in 0..cols {
                w[r * cols + c] /= norm;
            }
        }
    }
    for c in 0..cols {
        let mut sum_sq = 0.0;
        for r in 0..rows {
            let v = w[r * cols + c];
            sum_sq += v * v;
        }
        if sum_sq > 0.0 {
            let norm = sum_sq.sqrt();
            for r in 0..rows {
                w[r * cols + c] /= norm;
            }
        }
    }
}

fn sinkhorn_project(w: &mut [f64], rows: usize, cols: usize, steps: usize) {
    let eps = 1e-12;
    let convergence_tol = 1e-6;

    for _ in 0..steps {
        // Row normalization
        for r in 0..rows {
            let mut sum = 0.0;
            for c in 0..cols {
                sum += w[r * cols + c].abs();
            }
            let denom = if sum < eps { 1.0 } else { sum };
            for c in 0..cols {
                w[r * cols + c] /= denom;
            }
        }
        // Column normalization
        for c in 0..cols {
            let mut sum = 0.0;
            for r in 0..rows {
                sum += w[r * cols + c].abs();
            }
            let denom = if sum < eps { 1.0 } else { sum };
            for r in 0..rows {
                w[r * cols + c] /= denom;
            }
        }

        // Early termination: check max deviation of column sums from 1.0
        let mut max_dev = 0.0_f64;
        for c in 0..cols {
            let mut sum = 0.0;
            for r in 0..rows {
                sum += w[r * cols + c].abs();
            }
            max_dev = max_dev.max((sum - 1.0).abs());
        }
        if max_dev < convergence_tol {
            break;
        }
    }
}

fn copy_linear(dst: &mut Linear, src: &Linear) {
    let src_data = src.weight.data.read().unwrap().clone();
    let mut dst_data = dst.weight.data.write().unwrap();
    *dst_data = src_data;
}

fn soft_update_linear(dst: &mut Linear, src: &Linear, tau: f64) {
    let mut t_data = dst.weight.data.write().unwrap();
    let s_data = src.weight.data.read().unwrap();
    for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
        *t = *t * (1.0 - tau) + *s * tau;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn achf_forward_and_gate_update() {
        let cfg = AchfConfig {
            enabled: true,
            proj_freq: 1,
            proj_mode: "rowcol".to_string(),
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 42);
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 123);
        let out = layer.forward_residual(&x);
        let loss = out.mean();
        loss.backward();
        layer.update_after_backward();
        let g = layer.last_gate();
        assert!(g >= layer.config.g_min && g <= 1.0);
    }

    #[test]
    fn achf_rowcol_projection_normalizes_rows() {
        let cfg = AchfConfig {
            enabled: true,
            proj_freq: 1,
            proj_mode: "rowcol".to_string(),
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 7);
        let x = Tensor::rand(vec![1, 4], -0.1, 0.1, 9);
        let _ = layer.forward_residual(&x);
        let w = layer.weight.weight.data.read().unwrap();
        for c in 0..4 {
            let mut sum_sq = 0.0;
            for r in 0..4 {
                let v = w[r * 4 + c];
                sum_sq += v * v;
            }
            let norm = sum_sq.sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn achf_freeze_stops_projection() {
        let cfg = AchfConfig {
            enabled: true,
            proj_freq: 1,
            proj_mode: "rowcol".to_string(),
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 11);
        let x = Tensor::rand(vec![1, 4], -0.1, 0.1, 12);
        let _ = layer.forward_residual(&x);
        layer.freeze_for_inference();
        let w_before = layer.weight.weight.data.read().unwrap().clone();
        let x_data = x.data.read().unwrap().clone();
        let _ = layer.forward_inference_residual(&x_data);
        let w_after = layer.weight.weight.data.read().unwrap().clone();
        assert_eq!(w_before, w_after);
    }

    #[test]
    fn achf_adaptive_g_min_tracks_gate_floor() {
        let cfg = AchfConfig {
            enabled: true,
            g_min_adapt_rate: 0.5,
            g_target_min: 0.5,
            g_target_max: 0.9,
            gate_alpha: -10.0,
            gate_beta: 0.0,
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 21);
        let x = Tensor::rand(vec![1, 4], -0.1, 0.1, 22);
        let _ = layer.forward_residual(&x);
        let g = layer.last_gate();
        let g_min = layer.last_g_min();
        assert!(g >= g_min);
    }

    #[test]
    fn achf_low_rank_cache_matches_unfused() {
        let cfg = AchfConfig {
            enabled: true,
            rank: 2,
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 31);
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 32);
        let x_data = x.data.read().unwrap().clone();
        layer.freeze_for_inference();
        let out_cached = layer.forward_inference_residual(&x_data);
        layer.clear_cache();
        let out_unfused = layer.forward_inference_residual(&x_data);
        assert_eq!(out_cached.len(), out_unfused.len());
        for (a, b) in out_cached.iter().zip(out_unfused.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn achf_cache_threshold_skips_small_batches() {
        let cfg = AchfConfig {
            enabled: true,
            rank: 2,
            cache_min_rows: 4,
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 41);
        layer.freeze_for_inference();
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 42);
        let x_data = x.data.read().unwrap().clone();
        let out_cached = layer.forward_inference_residual(&x_data);
        layer.clear_cache();
        let out_unfused = layer.forward_inference_residual(&x_data);
        assert_eq!(out_cached.len(), out_unfused.len());
        for (a, b) in out_cached.iter().zip(out_unfused.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn achf_cache_stats_tracks_hits_and_paths() {
        let cfg = AchfConfig {
            enabled: true,
            rank: 2,
            cache_cost_bias: 0.0,
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 51);
        layer.freeze_for_inference();
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 52);
        let x_data = x.data.read().unwrap().clone();
        let _ = layer.forward_inference_residual(&x_data);
        let stats = layer.cache_stats();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.dense_paths, 0);
        assert_eq!(stats.low_rank_paths, 0);
        layer.clear_cache();
        let _ = layer.forward_inference_residual(&x_data);
        let stats = layer.cache_stats();
        assert_eq!(stats.calls, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.dense_paths, 0);
        assert_eq!(stats.low_rank_paths, 1);
    }

    #[test]
    fn achf_cache_stats_track_sparsity_skip() {
        let cfg = AchfConfig {
            enabled: true,
            rank: 2,
            cache_min_nonzero_ratio: 0.9,
            cache_sparsity_sample_rows: 1,
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 61);
        layer.freeze_for_inference();
        let x_data = vec![0.0; 8];
        let _ = layer.forward_inference_residual(&x_data);
        let stats = layer.cache_stats();
        assert_eq!(stats.cache_skips, 1);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.dense_paths, 0);
    }

    #[test]
    fn achf_cache_adapts_bias_with_latency() {
        let cfg = AchfConfig {
            enabled: true,
            rank: 2,
            cache_adapt_rate: 0.5,
            cache_latency_ema: 0.0,
            cache_bias_min: 0.5,
            cache_bias_max: 2.0,
            cache_cost_bias: 1.0,
            ..Default::default()
        };
        let layer = AchfLayer::new(4, cfg, 71);
        layer.record_path_latency(InferencePath::LowRank, 100.0);
        layer.record_path_latency(InferencePath::Cached, 50.0);
        let cache = layer.cache.read().unwrap();
        assert!(cache.adaptive_bias < 1.0);
    }
}
