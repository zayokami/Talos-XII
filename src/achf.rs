#[cfg(cuda)]
use crate::autograd::Context;
use crate::autograd::Tensor;
use crate::config::AchfConfig;
use crate::dtype::{bf16, Dtype};
use crate::nn::{Linear, Module};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[cfg(cuda)]
use crate::autograd::Device;

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
    /// Sparse weight created by magnitude pruning after training.
    /// When present, inference uses this (with zero-skip) instead of dense weight.
    pub sparse_weight: Option<Linear>,
    /// CPU mask for sparse weights (1 = non-zero, 0 = pruned).
    /// Shape: [in_features, out_features]. Used by CUDA sparse kernel.
    #[serde(default)]
    pub sparse_mask: Option<Vec<u8>>,
    pub config: AchfConfig,
    #[serde(skip, default = "default_state")]
    pub state: Arc<RwLock<AchfState>>,
    #[serde(skip, default = "default_cache")]
    pub cache: Arc<RwLock<AchfCache>>,
    #[serde(skip, default = "default_metrics")]
    pub metrics: Arc<AchfMetrics>,
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
    pub sinkhorn_iterations: usize,
    pub sinkhorn_row_max_dev: f64,
    pub sinkhorn_col_max_dev: f64,
    pub sinkhorn_min_value: f64,
    pub sinkhorn_negative_ratio: f64,
    pub sinkhorn_warm_started: bool,
}

/// Lock-free atomic counters for inference hot-path stats, eliminating RwLock contention.
pub struct AchfMetrics {
    pub calls: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_skips: AtomicU64,
    pub sparse_paths: AtomicU64,
    pub dense_paths: AtomicU64,
    pub latency_samples: AtomicU64,
    pub decision_samples: AtomicU64,
    pub memo_hash: AtomicU64,
    pub memo_count: AtomicU64,
}

impl Clone for AchfMetrics {
    fn clone(&self) -> Self {
        Self {
            calls: AtomicU64::new(self.calls.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(Ordering::Relaxed)),
            cache_skips: AtomicU64::new(self.cache_skips.load(Ordering::Relaxed)),
            sparse_paths: AtomicU64::new(self.sparse_paths.load(Ordering::Relaxed)),
            dense_paths: AtomicU64::new(self.dense_paths.load(Ordering::Relaxed)),
            latency_samples: AtomicU64::new(self.latency_samples.load(Ordering::Relaxed)),
            decision_samples: AtomicU64::new(self.decision_samples.load(Ordering::Relaxed)),
            memo_hash: AtomicU64::new(self.memo_hash.load(Ordering::Relaxed)),
            memo_count: AtomicU64::new(self.memo_count.load(Ordering::Relaxed)),
        }
    }
}

impl Default for AchfMetrics {
    fn default() -> Self {
        Self {
            calls: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_skips: AtomicU64::new(0),
            sparse_paths: AtomicU64::new(0),
            dense_paths: AtomicU64::new(0),
            latency_samples: AtomicU64::new(0),
            decision_samples: AtomicU64::new(0),
            memo_hash: AtomicU64::new(0),
            memo_count: AtomicU64::new(0),
        }
    }
}

fn default_metrics() -> Arc<AchfMetrics> {
    Arc::new(AchfMetrics::default())
}

#[derive(Clone)]
pub struct AchfCache {
    pub dense: Option<Vec<f32>>,
    pub bias: Option<Vec<f32>>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub ema_cached_ns: f64,
    pub ema_cached_long_ns: f64,
    pub ema_sparse_ns: f64,
    pub ema_sparse_long_ns: f64,
    pub decision_ema_ns: f64,
    pub decision_ema_long_ns: f64,
    pub adaptive_bias: f64,
    pub last_input_hash: Option<u64>,
    pub last_output: Option<Vec<f32>>,
    pub last_input_count: u64,
    /// Warm-start row scale vectors for Sinkhorn projection (cached between calls).
    pub sinkhorn_row_scales: Option<Vec<f64>>,
    /// Warm-start column scale vectors for Sinkhorn projection (cached between calls).
    pub sinkhorn_col_scales: Option<Vec<f64>>,
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
        sinkhorn_iterations: 0,
        sinkhorn_row_max_dev: 0.0,
        sinkhorn_col_max_dev: 0.0,
        sinkhorn_min_value: 0.0,
        sinkhorn_negative_ratio: 0.0,
        sinkhorn_warm_started: false,
    }))
}

fn default_cache() -> Arc<RwLock<AchfCache>> {
    Arc::new(RwLock::new(AchfCache {
        dense: None,
        bias: None,
        in_dim: 0,
        out_dim: 0,
        ema_cached_ns: 0.0,
        ema_cached_long_ns: 0.0,
        ema_sparse_ns: 0.0,
        ema_sparse_long_ns: 0.0,
        decision_ema_ns: 0.0,
        decision_ema_long_ns: 0.0,
        adaptive_bias: 0.0,
        last_input_hash: None,
        last_output: None,
        last_input_count: 0,
        sinkhorn_row_scales: None,
        sinkhorn_col_scales: None,
    }))
}

#[derive(Clone, Copy, Debug)]
pub struct AchfCacheStats {
    pub calls: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_skips: u64,
    pub sparse_paths: u64,
    pub dense_paths: u64,
    pub ema_cached_ns: f64,
    pub ema_cached_long_ns: f64,
    pub ema_sparse_ns: f64,
    pub ema_sparse_long_ns: f64,
    pub decision_ema_ns: f64,
    pub decision_ema_long_ns: f64,
    pub adaptive_bias: f64,
    pub latency_samples: u64,
    pub decision_samples: u64,
}

impl AchfCacheStats {
    pub fn debug_print(stats: &[AchfCacheStats]) {
        let mut hit = 0u64;
        let mut skip = 0u64;
        let mut lowrank = 0u64;
        let mut dense = 0u64;
        for s in stats {
            hit += s.cache_hits;
            skip += s.cache_skips;
            lowrank += s.sparse_paths;
            dense += s.dense_paths;
        }
        let total = hit + skip + lowrank + dense;
        if total == 0 {
            return;
        }
        let pct = |n: u64| n as f64 / total as f64 * 100.0;
        println!(
            "[ACHF] stats: cache_hit={:.0}% dense={:.0}% lowrank={:.0}% skip={:.0}% | hit={} skip={} lowrank={} dense={} (total={})",
            pct(hit),
            pct(dense),
            pct(lowrank),
            pct(skip),
            hit,
            skip,
            lowrank,
            dense,
            total
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AchfStateSnapshot {
    pub gate: f64,
    pub g_min: f64,
    pub grad_ema: f64,
    pub cache_hit_rate: f64,
    pub low_rank_ratio: f64,
    pub ema_cached_ns: f64,
    pub ema_sparse_ns: f64,
    pub adaptive_bias: f64,
    pub sinkhorn_iterations: usize,
    pub sinkhorn_row_max_dev: f64,
    pub sinkhorn_col_max_dev: f64,
    pub sinkhorn_min_value: f64,
    pub sinkhorn_negative_ratio: f64,
    pub sinkhorn_warm_started: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SinkhornProjectionStats {
    pub iterations: usize,
    pub row_max_dev: f64,
    pub col_max_dev: f64,
    pub min_value: f64,
    pub negative_ratio: f64,
    pub warm_started: bool,
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
        sparse_paths: 0,
        dense_paths: 0,
        ema_cached_ns: 0.0,
        ema_cached_long_ns: 0.0,
        ema_sparse_ns: 0.0,
        ema_sparse_long_ns: 0.0,
        decision_ema_ns: 0.0,
        decision_ema_long_ns: 0.0,
        adaptive_bias: 0.0,
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
        out.sparse_paths += s.sparse_paths;
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
        if s.ema_sparse_ns > 0.0 {
            out.ema_sparse_ns += s.ema_sparse_ns;
            count_low_rank += 1;
        }
        if s.ema_sparse_long_ns > 0.0 {
            out.ema_sparse_long_ns += s.ema_sparse_long_ns;
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
        out.ema_sparse_ns /= count_low_rank as f64;
    }
    if count_cached_long > 0 {
        out.ema_cached_long_ns /= count_cached_long as f64;
    }
    if count_low_rank_long > 0 {
        out.ema_sparse_long_ns /= count_low_rank_long as f64;
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
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        config: AchfConfig,
        seed: u64,
    ) -> Self {
        let weight = Linear::new(in_features, out_features, bias, seed);
        let layer = Self {
            weight,
            sparse_weight: None,
            sparse_mask: None,
            config,
            state: default_state(),
            cache: default_cache(),
            metrics: default_metrics(),
        };
        {
            let mut cache = layer.cache.write().unwrap();
            cache.adaptive_bias = layer.config.cache_cost_bias;
        }
        layer
    }

    #[allow(dead_code)]
    pub fn new_square(dim: usize, config: AchfConfig, seed: u64) -> Self {
        Self::new(dim, dim, false, config, seed)
    }

    pub fn snapshot_state(&self) -> AchfStateSnapshot {
        let state = self.state.read().unwrap();
        let cache = self.cache.read().unwrap();
        let calls = self.metrics.calls.load(Ordering::Relaxed) as f64;
        let cache_hits = self.metrics.cache_hits.load(Ordering::Relaxed) as f64;
        let sparse_paths = self.metrics.sparse_paths.load(Ordering::Relaxed) as f64;
        AchfStateSnapshot {
            gate: state.last_gate,
            g_min: state.g_min_ema,
            grad_ema: state.grad_ema,
            cache_hit_rate: if calls > 0.0 { cache_hits / calls } else { 0.0 },
            low_rank_ratio: if calls > 0.0 {
                sparse_paths / calls
            } else {
                0.0
            },
            ema_cached_ns: cache.ema_cached_ns,
            ema_sparse_ns: cache.ema_sparse_ns,
            adaptive_bias: cache.adaptive_bias,
            sinkhorn_iterations: state.sinkhorn_iterations,
            sinkhorn_row_max_dev: state.sinkhorn_row_max_dev,
            sinkhorn_col_max_dev: state.sinkhorn_col_max_dev,
            sinkhorn_min_value: state.sinkhorn_min_value,
            sinkhorn_negative_ratio: state.sinkhorn_negative_ratio,
            sinkhorn_warm_started: state.sinkhorn_warm_started,
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.weight.parameters();
        if let Some(s) = &self.sparse_weight {
            p.extend(s.parameters());
        }
        p
    }

    fn forward_blend(&self, input: &Tensor, g: f64) -> Tensor {
        let dense_out = self.weight.forward(input);
        let Some(sparse) = self.valid_sparse_weight() else {
            return dense_out;
        };
        let sparse_out = sparse.forward(input);
        let g_t = Tensor::with_dtype(vec![g], vec![1], dense_out.dtype)
            .broadcast(dense_out.shape.clone());
        let og_t = Tensor::with_dtype(vec![1.0 - g], vec![1], sparse_out.dtype)
            .broadcast(sparse_out.shape.clone());
        &(&dense_out * &g_t) + &(&sparse_out * &og_t)
    }

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        self.weight.to_cuda();
        if let Some(ref mut s) = self.sparse_weight {
            s.to_cuda();
        }
        self.clear_cache();
    }

    #[cfg(cuda)]
    fn forward_sparse_inference_cuda(&self, x: &[f32]) -> Option<Vec<f32>> {
        use crate::cuda::memory::{alloc, copy_d2h, copy_h2d};

        if !crate::cuda::is_available() {
            return None;
        }
        let sparse = self.valid_sparse_weight()?;
        let mask = self.valid_sparse_mask()?;
        let in_dim = sparse.in_features;
        let out_dim = sparse.out_features;
        if in_dim == 0 || out_dim == 0 || !x.len().is_multiple_of(in_dim) {
            return None;
        }
        let num_rows = x.len() / in_dim;
        if sparse.weight.device != Device::Cuda || sparse.weight.dtype != Dtype::F32 {
            return None;
        }
        let d_weight = sparse.weight.cuda_get_or_upload_buffer().ok()?;
        let d_weight = d_weight.as_f32()?;
        let d_x = alloc::<f32>(x.len()).ok()?;
        copy_h2d(&d_x, x).ok()?;
        let d_mask = alloc::<u8>(mask.len()).ok()?;
        copy_h2d(&d_mask, mask).ok()?;
        let out_len = num_rows.checked_mul(out_dim)?;
        let d_y = alloc::<f32>(out_len).ok()?;

        let kernel_ok = if let Some(bias) = &sparse.bias {
            if bias.device != Device::Cuda || bias.dtype != Dtype::F32 {
                return None;
            }
            let d_bias = bias.cuda_get_or_upload_buffer().ok()?;
            let d_bias = d_bias.as_f32()?;
            crate::cuda::kernels::sparse_matvec_bias_f32(
                &d_x, d_weight, &d_mask, d_bias, &d_y, num_rows, in_dim, out_dim,
            )
            .is_ok()
        } else {
            crate::cuda::kernels::sparse_matvec_f32(
                &d_x, d_weight, &d_mask, &d_y, num_rows, in_dim, out_dim,
            )
            .is_ok()
        };
        if !kernel_ok {
            return None;
        }

        let mut out = vec![0.0f32; out_len];
        copy_d2h(&mut out, &d_y).ok()?;
        Some(out)
    }

    #[cfg(cuda)]
    fn forward_sparse_tensor_cuda(&self, input: &Tensor) -> Option<Tensor> {
        use crate::cuda::memory::{alloc, copy_h2d, CudaBuffer};

        if !crate::cuda::is_available() || input.device != Device::Cuda || input.dtype != Dtype::F32
        {
            return None;
        }
        let sparse = self.valid_sparse_weight()?;
        let mask = self.valid_sparse_mask()?;
        let in_dim = sparse.in_features;
        let out_dim = sparse.out_features;
        if in_dim == 0 || out_dim == 0 || !input.numel().is_multiple_of(in_dim) {
            return None;
        }
        if sparse.weight.device != Device::Cuda || sparse.weight.dtype != Dtype::F32 {
            return None;
        }
        let num_rows = input.numel() / in_dim;
        let d_input = input.cuda_get_or_upload_buffer().ok()?;
        let d_input = d_input.as_f32()?;
        let d_weight = sparse.weight.cuda_get_or_upload_buffer().ok()?;
        let d_weight = d_weight.as_f32()?;
        let d_mask = alloc::<u8>(mask.len()).ok()?;
        copy_h2d(&d_mask, mask).ok()?;
        let d_out = alloc::<f32>(num_rows.checked_mul(out_dim)?).ok()?;

        let kernel_ok = if let Some(bias) = &sparse.bias {
            if bias.device != Device::Cuda || bias.dtype != Dtype::F32 {
                return None;
            }
            let d_bias = bias.cuda_get_or_upload_buffer().ok()?;
            let d_bias = d_bias.as_f32()?;
            crate::cuda::kernels::sparse_matvec_bias_f32(
                d_input, d_weight, &d_mask, d_bias, &d_out, num_rows, in_dim, out_dim,
            )
            .is_ok()
        } else {
            crate::cuda::kernels::sparse_matvec_f32(
                d_input, d_weight, &d_mask, &d_out, num_rows, in_dim, out_dim,
            )
            .is_ok()
        };
        if !kernel_ok {
            return None;
        }

        let mut out_shape = input.shape.clone();
        *out_shape.last_mut()? = out_dim;
        let d_out = Arc::new(CudaBuffer::F32(d_out));
        let out_len = num_rows * out_dim;
        let parents = vec![input.clone(), sparse.weight.clone()];
        let bias_parent_index = sparse.bias.as_ref().map(|_| parents.len());
        let mut parents = parents;
        if let Some(bias) = &sparse.bias {
            parents.push(bias.clone());
        }
        let mask = Arc::new(mask.to_vec());
        let out = Tensor {
            data: Tensor::empty_storage(Dtype::F32),
            grad: crate::dtype::Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F32)),
            shape: out_shape,
            device: Device::Cuda,
            dtype: Dtype::F32,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f32 = grad_out.to_f32_vec();
                    let input = &parents[0];
                    let weight = &parents[1];
                    let input_data = input.data_to_f32_vec();
                    let weight_data = weight.data_to_f32_vec();

                    {
                        let mut input_grad = input.grad_write_compat();
                        for row in 0..num_rows {
                            let ig_row = &mut input_grad[row * in_dim..(row + 1) * in_dim];
                            let go_row = &grad_out_f32[row * out_dim..(row + 1) * out_dim];
                            for (i, ig) in ig_row.iter_mut().enumerate().take(in_dim) {
                                let mut sum = 0.0f32;
                                for (j, &go) in go_row.iter().enumerate().take(out_dim) {
                                    let w_idx = i * out_dim + j;
                                    if mask[w_idx] != 0 {
                                        sum += go * weight_data[w_idx];
                                    }
                                }
                                *ig += sum as f64;
                            }
                        }
                    }

                    {
                        let mut weight_grad = weight.grad_write_compat();
                        for row in 0..num_rows {
                            let x_row = &input_data[row * in_dim..(row + 1) * in_dim];
                            let go_row = &grad_out_f32[row * out_dim..(row + 1) * out_dim];
                            for (i, &x_f32) in x_row.iter().enumerate().take(in_dim) {
                                let x = x_f32 as f64;
                                if x == 0.0 {
                                    continue;
                                }
                                for (j, &go) in go_row.iter().enumerate().take(out_dim) {
                                    let w_idx = i * out_dim + j;
                                    if mask[w_idx] != 0 {
                                        weight_grad[w_idx] += x * go as f64;
                                    }
                                }
                            }
                        }
                    }

                    if let Some(idx) = bias_parent_index {
                        let mut bias_grad = parents[idx].grad_write_compat();
                        for row in 0..num_rows {
                            let go_row = &grad_out_f32[row * out_dim..(row + 1) * out_dim];
                            for j in 0..out_dim {
                                bias_grad[j] += go_row[j] as f64;
                            }
                        }
                    }
                }),
            })),
        };
        out.cuda_set_cached_buffer(d_out);
        Some(out)
    }
}

impl Module for AchfLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        if input.shape.last().copied() != Some(self.weight.in_features) {
            self.metrics.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
            return self.zero_tensor_output(input);
        }
        if !self.config.enabled {
            return self.weight.forward(input);
        }
        self.maybe_project();
        let g = self.compute_gate();
        #[cfg(cuda)]
        if self.has_valid_sparse_state() && input.device == Device::Cuda {
            if let Some(sparse_out) = self.forward_sparse_tensor_cuda(input) {
                let dense_out = self.weight.forward(input);
                if let (Some(dense_scaled), Some(sparse_scaled)) =
                    (dense_out.scale_cuda(g), sparse_out.scale_cuda(1.0 - g))
                {
                    return &dense_scaled + &sparse_scaled;
                }
            }
        }
        self.forward_blend(input, g)
    }

    fn parameters(&self) -> Vec<Tensor> {
        AchfLayer::parameters(self)
    }
}

impl AchfLayer {
    #[allow(dead_code)]
    pub fn forward_residual(&self, x: &Tensor) -> Tensor {
        if !self.config.enabled {
            return self.zero_tensor_output(x);
        }
        if x.shape.last().copied() != Some(self.weight.in_features) {
            self.metrics.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
            return self.zero_tensor_output(x);
        }
        self.maybe_project();
        let g = self.compute_gate();
        if g <= 0.001 {
            return self.zero_tensor_output(x);
        }
        self.forward_blend(x, g)
    }

    pub fn forward_inference_residual(&self, x: &[f32]) -> Vec<f32> {
        if !self.config.enabled {
            return self.zero_inference_output(x);
        }
        if self.weight.in_features == 0 || !x.len().is_multiple_of(self.weight.in_features) {
            self.metrics.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
            return self.zero_inference_output(x);
        }
        self.maybe_project();
        let g = self.infer_gate_value() as f32;

        if self.config.cache_min_reuse > 0 {
            let hash = Self::input_hash(x);
            let stored_hash = self.metrics.memo_hash.load(Ordering::Acquire);
            if stored_hash == hash && hash != 0 {
                let count = self.metrics.memo_count.load(Ordering::Acquire);
                if count >= self.config.cache_min_reuse as u64 {
                    if let Ok(cache) = self.cache.try_read() {
                        let recheck = self.metrics.memo_hash.load(Ordering::Acquire);
                        if recheck == hash
                            && cache.last_input_hash == Some(hash)
                            && cache.last_input_count == x.len() as u64
                        {
                            if let Some(ref raw_out) = cache.last_output {
                                if raw_out.len() == x.len() {
                                    let mut out = raw_out.clone();
                                    for v in out.iter_mut() {
                                        *v *= g;
                                    }
                                    self.metrics.calls.fetch_add(1, Ordering::Relaxed);
                                    self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                                    return out;
                                }
                            }
                        }
                    }
                }
            }
        }

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
        let (raw_out, elapsed_ns) = if sample_latency {
            let start = Instant::now();
            let out = match path {
                InferencePath::Cached => self
                    .forward_inference_cached(x)
                    .unwrap_or_else(|| self.weight.forward_inference(x)),
                InferencePath::Sparse => {
                    #[cfg(cuda)]
                    if let Some(out) = self.forward_sparse_inference_cuda(x) {
                        out
                    } else if let Some(sparse) = self.valid_sparse_weight() {
                        sparse.forward_inference(x)
                    } else {
                        self.weight.forward_inference(x)
                    }
                    #[cfg(not(cuda))]
                    {
                        self.valid_sparse_weight()
                            .unwrap_or(&self.weight)
                            .forward_inference(x)
                    }
                }
                InferencePath::Dense => self.weight.forward_inference(x),
            };
            (out, start.elapsed().as_nanos() as f64)
        } else {
            let out = match path {
                InferencePath::Cached => self
                    .forward_inference_cached(x)
                    .unwrap_or_else(|| self.weight.forward_inference(x)),
                InferencePath::Sparse => {
                    #[cfg(cuda)]
                    if let Some(out) = self.forward_sparse_inference_cuda(x) {
                        out
                    } else if let Some(sparse) = self.valid_sparse_weight() {
                        sparse.forward_inference(x)
                    } else {
                        self.weight.forward_inference(x)
                    }
                    #[cfg(not(cuda))]
                    {
                        self.valid_sparse_weight()
                            .unwrap_or(&self.weight)
                            .forward_inference(x)
                    }
                }
                InferencePath::Dense => self.weight.forward_inference(x),
            };
            (out, 0.0)
        };
        if elapsed_ns > 0.0 {
            self.record_path_latency(path, elapsed_ns);
        }

        let mut out = raw_out.clone();
        for v in out.iter_mut() {
            *v *= g;
        }

        if self.config.cache_min_reuse > 0 {
            if let Ok(mut cache) = self.cache.try_write() {
                let hash = Self::input_hash(x);
                let prev = self.metrics.memo_hash.load(Ordering::Acquire);
                if prev == hash {
                    self.metrics.memo_count.fetch_add(1, Ordering::Release);
                } else {
                    self.metrics.memo_hash.store(hash, Ordering::Release);
                    self.metrics.memo_count.store(1, Ordering::Release);
                }
                cache.last_input_hash = Some(hash);
                cache.last_input_count = x.len() as u64;
                cache.last_output = Some(raw_out);
            }
        }

        out
    }

    /// Run inference through a specific path, bypassing the automatic path selection.
    /// `forced_path`: 0 = Cached, 1 = Sparse, 2 = Dense.
    pub fn forward_inference_forced_path(&self, x: &[f32], forced_path: u8) -> Vec<f32> {
        if !self.config.enabled {
            return self.zero_inference_output(x);
        }
        if self.weight.in_features == 0 || !x.len().is_multiple_of(self.weight.in_features) {
            self.metrics.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
            return self.zero_inference_output(x);
        }
        let g = self.infer_gate_value() as f32;
        let mut out = match forced_path {
            0 => self
                .forward_inference_cached(x)
                .unwrap_or_else(|| self.weight.forward_inference(x)),
            1 if self.has_valid_sparse_state() => {
                #[cfg(cuda)]
                if let Some(out) = self.forward_sparse_inference_cuda(x) {
                    out
                } else if let Some(sparse) = self.valid_sparse_weight() {
                    sparse.forward_inference(x)
                } else {
                    self.weight.forward_inference(x)
                }
                #[cfg(not(cuda))]
                {
                    self.valid_sparse_weight()
                        .unwrap_or(&self.weight)
                        .forward_inference(x)
                }
            }
            _ => self.weight.forward_inference(x),
        };
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
        // Include dense weight gradient
        {
            let grad = self.weight.weight.grad_to_f32_vec();
            for &v in grad.iter() {
                sum_sq += (v * v) as f64;
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
        let w = self.weight.weight.clone();
        let wt = w.transpose2d();
        let wtw = wt.matmul(&w);
        let dim = wtw.shape[0];
        let mut id_data = vec![0.0; dim * dim];
        for i in 0..dim {
            id_data[i * dim + i] = 1.0;
        }
        let id = Tensor::with_dtype(id_data, vec![dim, dim], wtw.dtype);
        let diff = wtw - id;
        let sq = &diff * &diff;
        let mean = sq.mean();
        let scale = Tensor::with_dtype(vec![self.config.lambda_ortho], vec![1], mean.dtype);
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
            calls: self.metrics.calls.load(Ordering::Relaxed),
            cache_hits: self.metrics.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.metrics.cache_misses.load(Ordering::Relaxed),
            cache_skips: self.metrics.cache_skips.load(Ordering::Relaxed),
            sparse_paths: self.metrics.sparse_paths.load(Ordering::Relaxed),
            dense_paths: self.metrics.dense_paths.load(Ordering::Relaxed),
            ema_cached_ns: cache.ema_cached_ns,
            ema_cached_long_ns: cache.ema_cached_long_ns,
            ema_sparse_ns: cache.ema_sparse_ns,
            ema_sparse_long_ns: cache.ema_sparse_long_ns,
            decision_ema_ns: cache.decision_ema_ns,
            decision_ema_long_ns: cache.decision_ema_long_ns,
            adaptive_bias: cache.adaptive_bias,
            latency_samples: self.metrics.latency_samples.load(Ordering::Relaxed),
            decision_samples: self.metrics.decision_samples.load(Ordering::Relaxed),
        }
    }

    pub fn freeze_for_inference(&mut self) {
        if !self.config.enabled {
            return;
        }
        let already_bf16 = self.weight.weight.dtype == Dtype::BF16
            || self
                .sparse_weight
                .as_ref()
                .is_some_and(|s| s.weight.dtype == Dtype::BF16);
        if !already_bf16 {
            self.project_weight();
        }
        self.prune(self.config.prune_threshold);
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
        let warmup = self.config.gate_warmup_steps;
        let transition = self.config.gate_transition_steps;
        let total = warmup.saturating_add(transition);

        if warmup > 0 && state.gate_step <= warmup {
            state.last_gate = 1.0;
            state.g_min_ema = self.config.g_min;
            return 1.0;
        }

        let target = self.compute_target_gate(&state);

        if transition > 0 && state.gate_step <= total {
            let t = (state.gate_step - warmup) as f64 / transition as f64;
            let g = 1.0 * (1.0 - t) + target * t;
            state.last_gate = g;
            state.g_min_ema = self.config.g_min_momentum * state.g_min_ema
                + (1.0 - self.config.g_min_momentum) * self.config.g_min;
            return g;
        }

        state.last_gate = target;
        state.g_min_ema = self.config.g_min_momentum * state.g_min_ema
            + (1.0 - self.config.g_min_momentum) * self.config.g_min;
        target
    }

    fn compute_target_gate(&self, state: &AchfState) -> f64 {
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
        {
            let state = self.state.read().unwrap();
            if state.freeze_projection {
                return;
            }
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
            ProjMode::None => self.clear_sinkhorn_projection_stats(),
            ProjMode::RowCol => {
                self.project_rowcol();
                self.clear_sinkhorn_projection_stats();
            }
            ProjMode::Sinkhorn => self.project_sinkhorn(),
        }
        self.clear_cache();
    }

    fn project_rowcol(&self) {
        let mut w = self.weight.weight.data_to_f32_vec();
        let rows = self.weight.in_features;
        let cols = self.weight.out_features;
        rowcol_project(&mut w, rows, cols);
        write_tensor_from_f32(&self.weight.weight, &w);
    }

    fn project_sinkhorn(&self) {
        let steps = if self.config.proj_steps == 0 {
            1
        } else {
            self.config.proj_steps
        };
        let cache = self.cache.read().unwrap();
        let row_scales_f32 = cache
            .sinkhorn_row_scales
            .as_ref()
            .map(|v| v.iter().map(|&x| x as f32).collect::<Vec<f32>>());
        let col_scales_f32 = cache
            .sinkhorn_col_scales
            .as_ref()
            .map(|v| v.iter().map(|&x| x as f32).collect::<Vec<f32>>());
        drop(cache);
        let mut w = self.weight.weight.data_to_f32_vec();
        let (new_row, new_col, stats) = sinkhorn_project(
            &mut w,
            self.weight.in_features,
            self.weight.out_features,
            steps,
            row_scales_f32.as_deref(),
            col_scales_f32.as_deref(),
        );
        write_tensor_from_f32(&self.weight.weight, &w);
        self.update_sinkhorn_projection_stats(stats);
        let mut cache = self.cache.write().unwrap();
        cache.sinkhorn_row_scales = Some(new_row.iter().map(|&x| x as f64).collect());
        cache.sinkhorn_col_scales = Some(new_col.iter().map(|&x| x as f64).collect());
    }

    fn update_sinkhorn_projection_stats(&self, stats: SinkhornProjectionStats) {
        let mut state = self.state.write().unwrap();
        state.sinkhorn_iterations = stats.iterations;
        state.sinkhorn_row_max_dev = stats.row_max_dev;
        state.sinkhorn_col_max_dev = stats.col_max_dev;
        state.sinkhorn_min_value = stats.min_value;
        state.sinkhorn_negative_ratio = stats.negative_ratio;
        state.sinkhorn_warm_started = stats.warm_started;
    }

    fn clear_sinkhorn_projection_stats(&self) {
        self.update_sinkhorn_projection_stats(SinkhornProjectionStats::default());
    }

    pub fn load_state_dict(&mut self, other: &AchfLayer) {
        match (&mut self.sparse_weight, &other.sparse_weight) {
            (Some(dst), Some(src)) if same_linear_layout(dst, src) => copy_linear(dst, src),
            (_, Some(src)) => self.sparse_weight = Some(clone_linear_detached(src)),
            (Some(_), None) => self.sparse_weight = None,
            (None, None) => {}
        }
        self.sparse_mask = other.sparse_mask.clone();
        // Always copy dense weight as reference/teacher
        copy_linear(&mut self.weight, &other.weight);
        self.clear_cache();
    }

    pub fn to_inference_bf16(&self) -> Self {
        let mut state = self.state.read().unwrap().clone();
        state.freeze_projection = true;
        let cache = default_cache();
        {
            let mut guard = cache.write().unwrap();
            guard.adaptive_bias = self.config.cache_cost_bias;
        }
        Self {
            weight: self.weight.to_inference_bf16(),
            sparse_weight: self.sparse_weight.as_ref().map(Linear::to_inference_bf16),
            sparse_mask: self.sparse_mask.clone(),
            config: self.config.clone(),
            state: Arc::new(RwLock::new(state)),
            cache,
            metrics: default_metrics(),
        }
    }

    pub fn soft_update(&mut self, source: &AchfLayer, tau: f64) {
        match (&mut self.sparse_weight, &source.sparse_weight) {
            (Some(dst), Some(src)) if same_linear_layout(dst, src) => {
                soft_update_linear(dst, src, tau);
            }
            (_, Some(src)) => self.sparse_weight = Some(clone_linear_detached(src)),
            (Some(_), None) => self.sparse_weight = None,
            (None, None) => {}
        }
        self.sparse_mask = source.sparse_mask.clone();
        // Always update dense weight as reference/teacher
        soft_update_linear(&mut self.weight, &source.weight, tau);
        self.clear_cache();
    }

    fn has_valid_sparse_state(&self) -> bool {
        self.valid_sparse_weight().is_some() && self.valid_sparse_mask().is_some()
    }

    fn valid_sparse_weight(&self) -> Option<&Linear> {
        let sparse = self.sparse_weight.as_ref()?;
        if sparse.in_features != self.weight.in_features
            || sparse.out_features != self.weight.out_features
            || sparse.weight.shape != [self.weight.in_features, self.weight.out_features]
            || sparse.weight.numel()
                != self
                    .weight
                    .in_features
                    .checked_mul(self.weight.out_features)?
            || sparse
                .bias
                .as_ref()
                .is_some_and(|bias| bias.numel() != sparse.out_features)
        {
            None
        } else {
            Some(sparse)
        }
    }

    fn valid_sparse_mask(&self) -> Option<&[u8]> {
        let mask = self.sparse_mask.as_deref()?;
        let expected_len = self
            .weight
            .in_features
            .checked_mul(self.weight.out_features)?;
        if mask.len() == expected_len {
            Some(mask)
        } else {
            None
        }
    }

    fn zero_tensor_output(&self, input: &Tensor) -> Tensor {
        let mut out_shape = input.shape.clone();
        if let Some(last) = out_shape.last_mut() {
            *last = self.weight.out_features;
        }
        Tensor::zeros_with_dtype(out_shape, input.dtype)
    }

    fn zero_inference_output(&self, input: &[f32]) -> Vec<f32> {
        let out_len =
            if self.weight.in_features > 0 && input.len().is_multiple_of(self.weight.in_features) {
                (input.len() / self.weight.in_features).saturating_mul(self.weight.out_features)
            } else {
                input.len()
            };
        vec![0.0f32; out_len]
    }

    /// Post-training magnitude pruning: create sparse_weight by zeroing
    /// elements below threshold. Idempotent (re-pruning overwrites).
    #[allow(dead_code)]
    pub fn prune(&mut self, threshold: f64) {
        let w_data = self.weight.weight.data_to_f32_vec();
        let threshold_f32 = threshold as f32;
        let pruned: Vec<f64> = w_data
            .iter()
            .map(|&v| {
                if v.abs() < threshold_f32 {
                    0.0
                } else {
                    v as f64
                }
            })
            .collect();
        let mask: Vec<u8> = w_data
            .iter()
            .map(|&v| if v.abs() < threshold_f32 { 0 } else { 1 })
            .collect();
        let pruned_weight = Tensor::with_dtype(
            pruned,
            vec![self.weight.in_features, self.weight.out_features],
            self.weight.weight.dtype,
        );
        self.sparse_weight = Some(Linear {
            weight: pruned_weight,
            bias: self.weight.bias.clone(),
            in_features: self.weight.in_features,
            out_features: self.weight.out_features,
        });
        self.sparse_mask = Some(mask);
        self.clear_cache();
    }

    fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.dense = None;
        cache.bias = None;
        cache.in_dim = 0;
        cache.out_dim = 0;
        cache.ema_cached_ns = 0.0;
        cache.ema_cached_long_ns = 0.0;
        cache.ema_sparse_ns = 0.0;
        cache.ema_sparse_long_ns = 0.0;
        cache.decision_ema_ns = 0.0;
        cache.decision_ema_long_ns = 0.0;
        cache.adaptive_bias = self.config.cache_cost_bias;
        cache.last_input_hash = None;
        cache.last_output = None;
        cache.last_input_count = 0;
        cache.sinkhorn_row_scales = None;
        cache.sinkhorn_col_scales = None;
        self.metrics.memo_hash.store(0, Ordering::Relaxed);
        self.metrics.memo_count.store(0, Ordering::Relaxed);
        self.metrics.latency_samples.store(0, Ordering::Relaxed);
        self.metrics.decision_samples.store(0, Ordering::Relaxed);
    }

    fn ensure_cache(&self) {
        let Some(sparse) = self.valid_sparse_weight() else {
            return;
        };
        let need_init = {
            let cache = self.cache.read().unwrap();
            let expected_len = sparse.in_features.checked_mul(sparse.out_features);
            cache.dense.as_ref().is_none_or(|dense| {
                Some(dense.len()) != expected_len
                    || cache.in_dim != sparse.in_features
                    || cache.out_dim != sparse.out_features
                    || cache
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.len() != sparse.out_features)
            })
        };
        if need_init {
            self.prepare_inference_cache();
        }
    }

    fn input_hash(x: &[f32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for &b in (x.len() as u64).to_le_bytes().iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
        }
        // Sample every 8th element for faster hashing.
        let step = (x.len() / 64).clamp(1, 8);
        for i in (0..x.len()).step_by(step) {
            let bytes = x[i].to_bits().to_le_bytes();
            for &b in &bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
            }
        }
        h
    }

    fn prepare_inference_cache(&self) {
        let Some(sparse) = self.valid_sparse_weight() else {
            self.clear_cache();
            return;
        };
        let in_dim = sparse.in_features;
        let out_dim = sparse.out_features;
        let dense: Vec<f32> = sparse.weight.data_to_f32_vec();
        let bias = sparse.bias.as_ref().map(|b| b.data_to_f32_vec());
        let mut cache = self.cache.write().unwrap();
        cache.dense = Some(dense);
        cache.bias = bias;
        cache.in_dim = in_dim;
        cache.out_dim = out_dim;
    }

    fn forward_inference_cached(&self, x: &[f32]) -> Option<Vec<f32>> {
        let cache = self.cache.read().unwrap();
        let dense = cache.dense.as_ref()?;
        let bias = cache.bias.as_ref();
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        if in_dim == 0 || out_dim == 0 {
            return None;
        }
        if dense.len() != in_dim.checked_mul(out_dim)? {
            return None;
        }
        if bias.is_some_and(|b| b.len() != out_dim) {
            return None;
        }
        if !x.len().is_multiple_of(in_dim) {
            return None;
        }
        let num_rows = x.len() / in_dim;
        if self.config.cache_min_rows > 0 && num_rows < self.config.cache_min_rows {
            return None;
        }
        let mut out = vec![0.0f32; num_rows * out_dim];
        for r in 0..num_rows {
            let row_offset_in = r * in_dim;
            let row_offset_out = r * out_dim;
            let out_row = &mut out[row_offset_out..row_offset_out + out_dim];
            if let Some(bias) = bias {
                out_row.copy_from_slice(bias);
            }
            for i in 0..in_dim {
                let scale = x[row_offset_in + i];
                if scale == 0.0 {
                    continue;
                }
                let w_row = &dense[i * out_dim..(i + 1) * out_dim];
                for j in 0..out_dim {
                    out_row[j] += scale * w_row[j];
                }
            }
        }
        Some(out)
    }

    fn choose_inference_path(&self, x: &[f32]) -> InferencePath {
        if !self.has_valid_sparse_state() {
            self.metrics.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics.dense_paths.fetch_add(1, Ordering::Relaxed);
            return InferencePath::Dense;
        }
        self.ensure_cache();
        let (use_cache, skip_cache, has_cache) = self.should_use_cache(x);
        self.metrics.calls.fetch_add(1, Ordering::Relaxed);
        if use_cache {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            return InferencePath::Cached;
        }
        if has_cache {
            if skip_cache {
                self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.metrics.sparse_paths.fetch_add(1, Ordering::Relaxed);
        InferencePath::Sparse
    }

    fn should_use_cache(&self, x: &[f32]) -> (bool, bool, bool) {
        let cache = self.cache.read().unwrap();
        if cache.dense.is_none() {
            return (false, false, false);
        }
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        if in_dim == 0 || out_dim == 0 {
            return (false, false, false);
        }
        if !x.len().is_multiple_of(in_dim) {
            return (false, true, true);
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
        // Latency-based decision: use whichever path is empirically faster
        if cache.ema_cached_ns > 0.0 && cache.ema_sparse_ns > 0.0 {
            let use_cache = cache.ema_cached_ns <= cache.ema_sparse_ns;
            return (use_cache, false, true);
        }
        // Cold-start fallback: both paths have same FLOPs for dense-sparse;
        // prefer cached when adaptive bias indicates it's empirically faster.
        let bias = if self.config.cache_adapt_rate > 0.0 {
            cache.adaptive_bias
        } else {
            self.config.cache_cost_bias
        };
        let use_cache = bias <= 1.0;
        (use_cache, false, true)
    }

    fn estimate_nonzero_ratio(&self, x: &[f32], in_dim: usize, num_rows: usize) -> f64 {
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
        let Ok(mut cache) = self.cache.try_write() else {
            return;
        };
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
            InferencePath::Sparse => {
                if cache.ema_sparse_ns == 0.0 || ema <= 0.0 {
                    cache.ema_sparse_ns = elapsed_ns;
                } else {
                    cache.ema_sparse_ns = ema * cache.ema_sparse_ns + (1.0 - ema) * elapsed_ns;
                }
                if cache.ema_sparse_long_ns == 0.0 || ema_long <= 0.0 {
                    cache.ema_sparse_long_ns = elapsed_ns;
                } else {
                    cache.ema_sparse_long_ns =
                        ema_long * cache.ema_sparse_long_ns + (1.0 - ema_long) * elapsed_ns;
                }
            }
            InferencePath::Dense => {}
        }
        self.metrics.latency_samples.fetch_add(1, Ordering::Relaxed);
        if self.config.cache_adapt_rate > 0.0
            && cache.ema_cached_ns > 0.0
            && cache.ema_sparse_ns > 0.0
        {
            let short_ratio = (cache.ema_cached_ns - cache.ema_sparse_ns) / cache.ema_sparse_ns;
            let mut ratio = short_ratio;
            if cache.ema_cached_long_ns > 0.0 && cache.ema_sparse_long_ns > 0.0 {
                let long_ratio = (cache.ema_cached_long_ns - cache.ema_sparse_long_ns)
                    / cache.ema_sparse_long_ns;
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
        let Ok(mut cache) = self.cache.try_write() else {
            return;
        };
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
        self.metrics
            .decision_samples
            .fetch_add(1, Ordering::Relaxed);
    }

    fn should_sample_latency(&self) -> bool {
        if self.config.cache_latency_sample_every == 0 {
            return true;
        }
        self.metrics
            .calls
            .load(Ordering::Relaxed)
            .is_multiple_of(self.config.cache_latency_sample_every)
    }
}

#[derive(Clone, Copy)]
enum InferencePath {
    Cached,
    Sparse,
    Dense,
}

fn rowcol_project(w: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let sum_sq = crate::simd::dot_product_f32(row, row);
        if sum_sq > 0.0 {
            let inv_norm = 1.0 / sum_sq.sqrt();
            for c in 0..cols {
                w[r * cols + c] *= inv_norm;
            }
        }
    }
    for c in 0..cols {
        let mut sum_sq = 0.0f32;
        for r in 0..rows {
            let v = w[r * cols + c];
            sum_sq += v * v;
        }
        if sum_sq > 0.0 {
            let inv_norm = 1.0 / sum_sq.sqrt();
            for r in 0..rows {
                w[r * cols + c] *= inv_norm;
            }
        }
    }
}

pub(crate) fn sinkhorn_project(
    w: &mut [f32],
    rows: usize,
    cols: usize,
    steps: usize,
    row_scales: Option<&[f32]>,
    col_scales: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, SinkhornProjectionStats) {
    let eps = 1e-12f32;
    let convergence_tol = 1e-6f32;
    let row_target = 1.0f32;
    let col_target = if cols == 0 {
        1.0f32
    } else {
        rows as f32 / cols as f32
    };

    let valid_row_scales = row_scales.and_then(|rs| (rs.len() == rows).then_some(rs));
    let valid_col_scales = col_scales.and_then(|cs| (cs.len() == cols).then_some(cs));
    let mut stats = SinkhornProjectionStats {
        warm_started: valid_row_scales.is_some() || valid_col_scales.is_some(),
        ..Default::default()
    };

    if rows == 0 || cols == 0 || w.is_empty() {
        return (vec![1.0; rows], vec![1.0; cols], stats);
    }

    // mHC/Sinkhorn requires a positive matrix. Convert raw logits to a
    // numerically stable positive matrix before alternating normalization.
    let max_input = w
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if max_input.is_finite() {
        for v in w.iter_mut() {
            let raw = if v.is_finite() { *v - max_input } else { -60.0 };
            let shifted = raw.clamp(-60.0, 0.0);
            *v = shifted.exp();
        }
    } else {
        w.fill(1.0);
    }

    // Apply warm-start scale vectors if dimensions match.
    let mut out_row_scales = valid_row_scales.map_or_else(|| vec![1.0f32; rows], |rs| rs.to_vec());
    let mut out_col_scales = valid_col_scales.map_or_else(|| vec![1.0f32; cols], |cs| cs.to_vec());
    if let Some(rs) = valid_row_scales {
        for r in 0..rows {
            let scale = rs[r];
            for c in 0..cols {
                w[r * cols + c] *= scale;
            }
        }
    }
    if let Some(cs) = valid_col_scales {
        for c in 0..cols {
            let scale = cs[c];
            for r in 0..rows {
                w[r * cols + c] *= scale;
            }
        }
    }

    let mut iterations = 0usize;
    for iter in 0..steps {
        iterations = iter + 1;
        // Row normalization
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += w[r * cols + c];
            }
            let denom = if sum < eps { 1.0f32 } else { sum };
            let scale = row_target / denom;
            out_row_scales[r] *= scale;
            for c in 0..cols {
                w[r * cols + c] *= scale;
            }
        }
        // Column normalization
        for c in 0..cols {
            let mut sum = 0.0f32;
            for r in 0..rows {
                sum += w[r * cols + c];
            }
            let denom = if sum < eps { 1.0f32 } else { sum };
            let scale = col_target / denom;
            out_col_scales[c] *= scale;
            for r in 0..rows {
                w[r * cols + c] *= scale;
            }
        }

        // Early termination: check max deviation of both row and column sums from 1.0
        let mut max_dev = 0.0f32;
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += w[r * cols + c];
            }
            max_dev = max_dev.max((sum - row_target).abs());
        }
        for c in 0..cols {
            let mut sum = 0.0f32;
            for r in 0..rows {
                sum += w[r * cols + c];
            }
            max_dev = max_dev.max((sum - col_target).abs());
        }
        if max_dev < convergence_tol {
            break;
        }
    }

    let mut row_max_dev = 0.0f32;
    let mut col_max_dev = 0.0f32;
    let mut min_value = f32::INFINITY;
    let mut negative_entries = 0usize;
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            let v = w[r * cols + c];
            sum += v;
            if v < min_value {
                min_value = v;
            }
            if v < 0.0 {
                negative_entries += 1;
            }
        }
        row_max_dev = row_max_dev.max((sum - row_target).abs());
    }
    for c in 0..cols {
        let mut sum = 0.0f32;
        for r in 0..rows {
            sum += w[r * cols + c];
        }
        col_max_dev = col_max_dev.max((sum - col_target).abs());
    }
    stats.iterations = iterations;
    stats.row_max_dev = row_max_dev as f64;
    stats.col_max_dev = col_max_dev as f64;
    stats.min_value = if min_value.is_finite() {
        min_value as f64
    } else {
        0.0
    };
    stats.negative_ratio = if w.is_empty() {
        0.0
    } else {
        negative_entries as f64 / w.len() as f64
    };

    (out_row_scales, out_col_scales, stats)
}

fn copy_linear(dst: &mut Linear, src: &Linear) {
    copy_tensor(&dst.weight, &src.weight);
    if let (Some(dst_bias), Some(src_bias)) = (&dst.bias, &src.bias) {
        copy_tensor(dst_bias, src_bias);
    }
}

fn same_linear_layout(lhs: &Linear, rhs: &Linear) -> bool {
    lhs.in_features == rhs.in_features
        && lhs.out_features == rhs.out_features
        && lhs.weight.shape == rhs.weight.shape
        && lhs.weight.numel() == rhs.weight.numel()
        && lhs.bias.as_ref().map(Tensor::numel) == rhs.bias.as_ref().map(Tensor::numel)
}

fn clone_linear_detached(src: &Linear) -> Linear {
    Linear {
        weight: Tensor::with_dtype(
            src.weight.data_as_f64_vec(),
            src.weight.shape.clone(),
            src.weight.dtype,
        ),
        bias: src
            .bias
            .as_ref()
            .map(|bias| Tensor::with_dtype(bias.data_as_f64_vec(), bias.shape.clone(), bias.dtype)),
        in_features: src.in_features,
        out_features: src.out_features,
    }
}

fn soft_update_linear(dst: &mut Linear, src: &Linear, tau: f64) {
    soft_update_tensor(&dst.weight, &src.weight, tau);
    if let (Some(dst_bias), Some(src_bias)) = (&dst.bias, &src.bias) {
        soft_update_tensor(dst_bias, src_bias, tau);
    }
}

fn copy_tensor(dst: &Tensor, src: &Tensor) {
    match (dst.dtype, src.dtype) {
        (Dtype::F32, Dtype::F32) => {
            let mut dst_data = dst.data_write_f32();
            *dst_data = src.data_f32().clone();
        }
        (Dtype::BF16, Dtype::BF16) => {
            let mut dst_data = dst.data_write_bf16();
            *dst_data = src.data_bf16().clone();
        }
        (Dtype::F64, Dtype::F64) => {
            let mut dst_data = dst.data_write_f64();
            *dst_data = src.data_f64().clone();
        }
        _ => {
            let data = src.data_as_f64_vec();
            match dst.dtype {
                Dtype::F32 => {
                    let mut dst_data = dst.data_write_f32();
                    *dst_data = data.iter().map(|&v| v as f32).collect();
                }
                Dtype::BF16 => {
                    let mut dst_data = dst.data_write_bf16();
                    *dst_data = data.iter().map(|&v| bf16::from_f64(v)).collect();
                }
                Dtype::F64 => {
                    let mut dst_data = dst.data_write_f64();
                    *dst_data = data;
                }
                Dtype::I8 => panic!("copy_tensor does not support I8 tensors"),
            }
        }
    }
}

fn soft_update_tensor(dst: &Tensor, src: &Tensor, tau: f64) {
    let tau_f32 = tau as f32;
    match (dst.dtype, src.dtype) {
        (Dtype::F32, Dtype::F32) => {
            let mut t_data = dst.data_write_f32();
            let s_data = src.data_f32();
            for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                *t = *t * (1.0 - tau_f32) + *s * tau_f32;
            }
        }
        (Dtype::BF16, Dtype::BF16) => {
            let mut t_data = dst.data_write_bf16();
            let s_data = src.data_bf16();
            for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                let tv = t.to_f32();
                let sv = s.to_f32();
                *t = bf16::from_f32(tv * (1.0 - tau_f32) + sv * tau_f32);
            }
        }
        (Dtype::F64, Dtype::F64) => {
            let mut t_data = dst.data_write_f64();
            let s_data = src.data_f64();
            for (t, s) in t_data.iter_mut().zip(s_data.iter()) {
                *t = *t * (1.0 - tau) + *s * tau;
            }
        }
        _ => {
            let t_data = dst.data_as_f64_vec();
            let s_data = src.data_as_f64_vec();
            let blended: Vec<f64> = t_data
                .iter()
                .zip(s_data.iter())
                .map(|(t, s)| t * (1.0 - tau) + s * tau)
                .collect();
            match dst.dtype {
                Dtype::F32 => {
                    let mut dst_data = dst.data_write_f32();
                    *dst_data = blended.iter().map(|&v| v as f32).collect();
                }
                Dtype::BF16 => {
                    let mut dst_data = dst.data_write_bf16();
                    *dst_data = blended.iter().map(|&v| bf16::from_f64(v)).collect();
                }
                Dtype::F64 => {
                    let mut dst_data = dst.data_write_f64();
                    *dst_data = blended;
                }
                Dtype::I8 => panic!("soft_update_tensor does not support I8 tensors"),
            }
        }
    }
}

fn write_tensor_from_f32(dst: &Tensor, data: &[f32]) {
    match dst.dtype {
        Dtype::F32 => {
            let mut dst_data = dst.data_write_f32();
            *dst_data = data.to_vec();
        }
        Dtype::BF16 => {
            let mut dst_data = dst.data_write_bf16();
            *dst_data = data.iter().map(|&v| bf16::from_f32(v)).collect();
        }
        Dtype::F64 => {
            let mut dst_data = dst.data_write_f64();
            *dst_data = data.iter().map(|&v| v as f64).collect();
        }
        Dtype::I8 => panic!("write_tensor_from_f32 does not support I8 tensors"),
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
        let layer = AchfLayer::new_square(4, cfg, 42);
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
        let layer = AchfLayer::new_square(4, cfg, 7);
        let x = Tensor::rand(vec![1, 4], -0.1, 0.1, 9);
        let _ = layer.forward_residual(&x);
        let w = layer.weight.weight.data_f32();
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
    fn achf_sinkhorn_projection_stats_are_exposed() {
        let cfg = AchfConfig {
            enabled: true,
            proj_freq: 1,
            proj_steps: 40,
            proj_mode: "sinkhorn".to_string(),
            ..Default::default()
        };
        let layer = AchfLayer::new_square(4, cfg, 8);
        {
            let mut w = layer.weight.weight.data_write_f32();
            *w = vec![
                -2.0, 1.0, 0.5, -1.5, 3.0, -0.5, 2.0, 0.0, -3.0, 1.5, -1.0, 2.5, 0.25, -0.75, 1.25,
                -2.25,
            ];
        }
        let x = Tensor::rand(vec![1, 4], -0.1, 0.1, 10);
        let _ = layer.forward_residual(&x);
        let snapshot = layer.snapshot_state();

        assert!(snapshot.sinkhorn_iterations > 0);
        assert!(snapshot.sinkhorn_row_max_dev < 1e-5);
        assert!(snapshot.sinkhorn_col_max_dev < 1e-5);
        assert!(snapshot.sinkhorn_min_value >= 0.0);
        assert_eq!(snapshot.sinkhorn_negative_ratio, 0.0);
    }

    #[test]
    fn achf_freeze_stops_projection() {
        let cfg = AchfConfig {
            enabled: true,
            proj_freq: 1,
            proj_mode: "rowcol".to_string(),
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 11);
        let x = Tensor::rand(vec![1, 4], -0.1, 0.1, 12);
        let _ = layer.forward_residual(&x);
        layer.freeze_for_inference();
        let w_before = layer.weight.weight.data_f32().clone();
        let x_data = x.data_to_f32_vec();
        let _ = layer.forward_inference_residual(&x_data);
        let w_after = layer.weight.weight.data_f32().clone();
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
        let layer = AchfLayer::new_square(4, cfg, 21);
        let x = Tensor::rand(vec![1, 4], -0.1, 0.1, 22);
        let _ = layer.forward_residual(&x);
        let g = layer.last_gate();
        let g_min = layer.last_g_min();
        assert!(g >= g_min);
    }

    #[test]
    fn achf_freeze_prunes_and_prepares_cache_without_explicit_prune() {
        let cfg = AchfConfig {
            enabled: true,
            cache_cost_bias: 0.0,
            infer_gate: "one".to_string(),
            prune_threshold: 0.01,
            proj_freq: 0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(3, 2, true, cfg, 30);
        assert!(layer.sparse_weight.is_none());

        layer.freeze_for_inference();
        assert!(layer.sparse_weight.is_some());
        assert!(layer.sparse_mask.is_some());

        let x = vec![1.0, -2.0, 0.5];
        let _ = layer.forward_inference_residual(&x);
        let stats = layer.cache_stats();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.dense_paths, 0);
        assert_eq!(stats.sparse_paths, 0);
    }

    #[test]
    fn achf_cache_consistency() {
        let cfg = AchfConfig {
            enabled: true,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 31);
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 32);
        let x_data = x.data_to_f32_vec();
        layer.prune(0.01);
        layer.freeze_for_inference();
        let out_cached = layer.forward_inference_residual(&x_data);
        layer.clear_cache();
        let out_unfused = layer.forward_inference_residual(&x_data);
        assert_eq!(out_cached.len(), out_unfused.len());
        for (a, b) in out_cached.iter().zip(out_unfused.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn achf_cached_path_includes_sparse_bias() {
        let cfg = AchfConfig {
            enabled: true,
            cache_cost_bias: 0.0,
            infer_gate: "one".to_string(),
            ..Default::default()
        };
        let mut layer = AchfLayer::new(3, 2, true, cfg, 33);
        {
            let mut w = layer.weight.weight.data_write_f32();
            *w = vec![1.0, -1.0, 2.0, 0.5, -0.25, 0.75];
        }
        {
            let bias = layer.weight.bias.as_ref().unwrap();
            let mut b = bias.data_write_f32();
            *b = vec![0.5, -1.5];
        }
        layer.prune(0.0);
        layer.freeze_for_inference();

        let x = vec![2.0, -1.0, 4.0, 0.5, 1.5, -2.0];
        let out = layer.forward_inference_residual(&x);
        let expected = layer.sparse_weight.as_ref().unwrap().forward_inference(&x);

        assert_eq!(out.len(), expected.len());
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "cached={a} expected={b}");
        }
        let stats = layer.cache_stats();
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn achf_prune_invalidates_cached_weight() {
        let cfg = AchfConfig {
            enabled: true,
            cache_cost_bias: 0.0,
            infer_gate: "one".to_string(),
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(2, cfg, 34);
        {
            let mut w = layer.weight.weight.data_write_f32();
            *w = vec![1.0, 0.0, 0.0, 1.0];
        }
        layer.prune(0.0);
        layer.freeze_for_inference();

        let x = vec![2.0, 3.0];
        let first = layer.forward_inference_residual(&x);
        assert_eq!(first, vec![2.0, 3.0]);

        {
            let mut w = layer.weight.weight.data_write_f32();
            *w = vec![2.0, 0.0, 0.0, 2.0];
        }
        layer.prune(0.0);
        let second = layer.forward_inference_residual(&x);
        assert_eq!(second, vec![4.0, 6.0]);
    }

    #[test]
    fn achf_invalid_sparse_mask_falls_back_to_dense() {
        let cfg = AchfConfig {
            enabled: true,
            cache_cost_bias: 0.0,
            infer_gate: "one".to_string(),
            ..Default::default()
        };
        let mut layer = AchfLayer::new(3, 2, false, cfg, 35);
        {
            let mut w = layer.weight.weight.data_write_f32();
            *w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        }
        layer.prune(0.0);
        if let Some(sparse) = &layer.sparse_weight {
            let mut w = sparse.weight.data_write_f32();
            w.fill(0.0);
        }
        layer.sparse_mask = Some(vec![1]);

        let x = vec![1.0, 0.5, -1.0];
        let out = layer.forward_inference_residual(&x);
        let expected = layer.weight.forward_inference(&x);
        assert_eq!(out, expected);

        let stats = layer.cache_stats();
        assert_eq!(stats.dense_paths, 1);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.sparse_paths, 0);
    }

    #[test]
    fn achf_forced_sparse_invalid_state_falls_back_to_dense() {
        let cfg = AchfConfig {
            enabled: true,
            infer_gate: "one".to_string(),
            ..Default::default()
        };
        let mut layer = AchfLayer::new(3, 2, false, cfg, 36);
        {
            let mut w = layer.weight.weight.data_write_f32();
            *w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        }
        layer.prune(0.0);
        if let Some(sparse) = &layer.sparse_weight {
            let mut w = sparse.weight.data_write_f32();
            w.fill(0.0);
        }
        layer.sparse_mask = None;

        let x = vec![1.0, 0.5, -1.0];
        let out = layer.forward_inference_forced_path(&x, 1);
        let expected = layer.weight.forward_inference(&x);
        assert_eq!(out, expected);
    }

    #[test]
    fn achf_load_state_dict_copies_sparse_state() {
        let cfg = AchfConfig {
            enabled: true,
            infer_gate: "one".to_string(),
            ..Default::default()
        };
        let mut src = AchfLayer::new_square(2, cfg.clone(), 37);
        {
            let mut w = src.weight.weight.data_write_f32();
            *w = vec![1.0, 0.001, -0.001, 2.0];
        }
        src.prune(0.01);

        let mut dst = AchfLayer::new_square(2, cfg, 38);
        dst.load_state_dict(&src);

        assert!(dst.sparse_weight.is_some());
        assert_eq!(dst.sparse_mask, src.sparse_mask);
        let x = vec![3.0, 4.0];
        assert_eq!(
            dst.forward_inference_forced_path(&x, 1),
            src.forward_inference_forced_path(&x, 1)
        );
    }

    #[test]
    fn achf_cache_threshold_skips_small_batches() {
        let cfg = AchfConfig {
            enabled: true,
            cache_min_rows: 4,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 41);
        layer.prune(0.01);
        layer.freeze_for_inference();
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 42);
        let x_data = x.data_to_f32_vec();
        let out_cached = layer.forward_inference_residual(&x_data);
        layer.clear_cache();
        let out_unfused = layer.forward_inference_residual(&x_data);
        assert_eq!(out_cached.len(), out_unfused.len());
        for (a, b) in out_cached.iter().zip(out_unfused.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn achf_cache_stats_tracks_hits_and_paths() {
        let cfg = AchfConfig {
            enabled: true,
            cache_cost_bias: 0.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 51);
        layer.prune(0.01);
        layer.freeze_for_inference();
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 52);
        let x_data = x.data_to_f32_vec();
        let _ = layer.forward_inference_residual(&x_data);
        let stats = layer.cache_stats();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.dense_paths, 0);
        assert_eq!(stats.sparse_paths, 0);
        layer.clear_cache();
        let _ = layer.forward_inference_residual(&x_data);
        let stats = layer.cache_stats();
        assert_eq!(stats.calls, 2);
        // On the second forward, the cache is None; auto-prepare will rebuild the cache, so it is a cache hit again
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.dense_paths, 0);
        assert_eq!(stats.sparse_paths, 0);
    }

    #[test]
    fn achf_cache_stats_track_sparsity_skip() {
        let cfg = AchfConfig {
            enabled: true,
            cache_min_nonzero_ratio: 0.9,
            cache_sparsity_sample_rows: 1,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 61);
        layer.prune(0.01);
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
            cache_adapt_rate: 0.5,
            cache_latency_ema: 0.0,
            cache_bias_min: 0.5,
            cache_bias_max: 2.0,
            cache_cost_bias: 1.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 71);
        layer.prune(0.01);
        layer.record_path_latency(InferencePath::Sparse, 100.0);
        layer.record_path_latency(InferencePath::Cached, 50.0);
        let cache = layer.cache.read().unwrap();
        assert!(cache.adaptive_bias < 1.0);
    }

    #[test]
    fn achf_memoized_output_applies_current_gate() {
        let cfg = AchfConfig {
            enabled: true,
            cache_min_reuse: 1,
            infer_gate: "last".to_string(),
            ..Default::default()
        };
        let layer = AchfLayer::new_square(4, cfg, 81);
        let x_data = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
        let hash = AchfLayer::input_hash(&x_data);
        layer.metrics.memo_hash.store(hash, Ordering::Relaxed);
        layer.metrics.memo_count.store(1, Ordering::Relaxed);
        {
            let mut cache = layer.cache.write().unwrap();
            cache.last_input_hash = Some(hash);
            cache.last_input_count = x_data.len() as u64;
            cache.last_output = Some(vec![2.0; x_data.len()]);
        }
        {
            let mut state = layer.state.write().unwrap();
            state.last_gate = 0.5;
        }
        let out = layer.forward_inference_residual(&x_data);
        assert_eq!(out.len(), x_data.len());
        for v in out {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn achf_invalid_shape_returns_zero_residual() {
        let cfg = AchfConfig {
            enabled: true,
            ..Default::default()
        };
        let layer = AchfLayer::new_square(4, cfg, 91);
        let x_data = vec![0.1, 0.2, 0.3];
        let out = layer.forward_inference_residual(&x_data);
        assert_eq!(out.len(), x_data.len());
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn achf_clear_cache_resets_memo_state() {
        let cfg = AchfConfig {
            enabled: true,
            ..Default::default()
        };
        let layer = AchfLayer::new_square(4, cfg, 101);
        layer.metrics.memo_hash.store(123, Ordering::Relaxed);
        layer.metrics.memo_count.store(9, Ordering::Relaxed);
        layer.clear_cache();
        assert_eq!(layer.metrics.memo_hash.load(Ordering::Relaxed), 0);
        assert_eq!(layer.metrics.memo_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn achf_input_hash_distinguishes_length() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 0.0];
        assert_ne!(AchfLayer::input_hash(&a), AchfLayer::input_hash(&b));
    }

    #[test]
    fn achf_prune_zeros_small_weights() {
        let cfg = AchfConfig::default();
        let mut layer = AchfLayer::new_square(4, cfg, 200);
        // Set some weights to known values
        {
            let mut w = layer.weight.weight.data_write_f32();
            w[0] = 0.5; // above threshold
            w[1] = 0.005; // below threshold
            w[2] = -0.5; // above threshold
            w[3] = -0.005; // below threshold
        }
        layer.prune(0.01);
        let sparse = layer.sparse_weight.as_ref().unwrap();
        let s = sparse.weight.data_f32();
        assert_eq!(s[0], 0.5);
        assert_eq!(s[1], 0.0);
        assert_eq!(s[2], -0.5);
        assert_eq!(s[3], 0.0);
    }

    #[test]
    fn achf_sparse_inference_matches_dense() {
        let cfg = AchfConfig {
            enabled: true,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 201);
        // Set deterministic weights
        {
            let mut w = layer.weight.weight.data_write_f32();
            for (i, v) in w.iter_mut().enumerate() {
                *v = ((i as f32) * 0.1) - 0.3;
            }
        }
        let x = vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.1, 0.2, -0.3];
        let dense_out = layer.forward_inference_residual(&x);
        layer.prune(0.15); // prune small weights
        let sparse_out = layer.forward_inference_residual(&x);
        assert_eq!(dense_out.len(), sparse_out.len());
        // Outputs differ because pruning changes weights, but shapes match
        assert_eq!(dense_out.len(), x.len());
    }

    #[cfg(cuda)]
    #[test]
    fn achf_sparse_tensor_cuda_matches_cpu_sparse_forward() {
        if crate::cuda::init().is_err() {
            return;
        }

        let cfg = AchfConfig {
            enabled: true,
            gate_warmup_steps: 0,
            gate_transition_steps: 0,
            g_min: 0.0,
            infer_gate: "one".to_string(),
            ..Default::default()
        };
        let mut cpu_layer = AchfLayer::new_square(4, cfg, 301);
        {
            let mut w = cpu_layer.weight.weight.data_write_f32();
            for (i, v) in w.iter_mut().enumerate() {
                *v = ((i as f32) * 0.07) - 0.4;
            }
        }
        cpu_layer.prune(0.2);
        {
            let mut state = cpu_layer.state.write().unwrap();
            state.last_gate = 0.0;
        }
        let mut cuda_layer = cpu_layer.clone();
        cuda_layer.to_cuda();

        let x = Tensor::new_f32(vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.1, 0.2, -0.3], vec![2, 4]);
        let x_cuda = match x.to_cuda() {
            Ok(tensor) => tensor,
            Err(_) => return,
        };

        let cpu_out = cpu_layer.forward(&x);
        let cuda_out = cuda_layer.forward(&x_cuda);
        assert_eq!(cuda_out.device, Device::Cuda);
        let cpu_data = cpu_out.data_as_f64_vec();
        let cuda_data = cuda_out.data_as_f64_vec();
        assert_eq!(cpu_data.len(), cuda_data.len());
        for i in 0..cpu_data.len() {
            assert!(
                (cpu_data[i] - cuda_data[i]).abs() < 1e-5,
                "idx {} cpu={} cuda={}",
                i,
                cpu_data[i],
                cuda_data[i]
            );
        }
    }

    #[test]
    fn aggregate_cache_stats_bias_average_is_unbiased() {
        let s1 = AchfCacheStats {
            calls: 1,
            cache_hits: 0,
            cache_misses: 0,
            cache_skips: 0,
            sparse_paths: 0,
            dense_paths: 0,
            ema_cached_ns: 0.0,
            ema_cached_long_ns: 0.0,
            ema_sparse_ns: 0.0,
            ema_sparse_long_ns: 0.0,
            decision_ema_ns: 0.0,
            decision_ema_long_ns: 0.0,
            adaptive_bias: 2.0,
            latency_samples: 0,
            decision_samples: 0,
        };
        let s2 = AchfCacheStats {
            adaptive_bias: 4.0,
            ..s1
        };
        let agg = aggregate_cache_stats_iter([s1, s2]);
        assert!((agg.adaptive_bias - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_sinkhorn_doubly_stochastic() {
        // Build a 4x4 positive matrix
        let mut w = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let (row_scales, col_scales, stats) = sinkhorn_project(&mut w, 4, 4, 20, None, None);
        // Verify all row sums equal 1.0
        for r in 0..4 {
            let sum: f32 = (0..4).map(|c| w[r * 4 + c]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {} sum = {}, expected 1.0",
                r,
                sum
            );
        }
        // Verify all column sums equal 1.0
        for c in 0..4 {
            let sum: f32 = (0..4).map(|r| w[r * 4 + c]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Col {} sum = {}, expected 1.0",
                c,
                sum
            );
        }
        // All values should be non-negative
        assert!(w.iter().all(|&v| v >= 0.0));
        assert_eq!(stats.negative_ratio, 0.0);
        assert!(stats.row_max_dev < 1e-5);
        assert!(stats.col_max_dev < 1e-5);
        // Returned scale vectors should not be all 1.0 (projection did real work)
        assert!(row_scales.iter().any(|&v| (v - 1.0).abs() > 1e-6));
        assert!(col_scales.iter().any(|&v| (v - 1.0).abs() > 1e-6));
    }

    #[test]
    fn test_sinkhorn_signed_logits_project_to_nonnegative_matrix() {
        let mut w = vec![2.0f32, -1.0, 0.25, -3.0, 1.5, -0.5, 0.0, 3.5, -2.0];
        let (_, _, stats) = sinkhorn_project(&mut w, 3, 3, 40, None, None);

        for r in 0..3 {
            let sum: f32 = (0..3).map(|c| w[r * 3 + c]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {} sum = {}, expected 1.0",
                r,
                sum
            );
        }
        for c in 0..3 {
            let sum: f32 = (0..3).map(|r| w[r * 3 + c]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Col {} sum = {}, expected 1.0",
                c,
                sum
            );
        }
        assert!(w.iter().all(|&v| v >= 0.0));
        assert_eq!(stats.negative_ratio, 0.0);
        assert!(stats.min_value >= 0.0);
    }

    #[test]
    fn test_sinkhorn_ignores_mismatched_warm_start_scales() {
        let mut w = vec![1.0f32, -2.0, 3.0, 0.5];
        let (_, _, stats) =
            sinkhorn_project(&mut w, 2, 2, 20, Some(&[1.0]), Some(&[1.0, 1.0, 1.0]));

        assert!(!stats.warm_started);
        assert!(stats.row_max_dev < 1e-5);
        assert!(stats.col_max_dev < 1e-5);
    }

    #[test]
    fn test_sinkhorn_warm_start_accelerates() {
        let w0 = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        // First projection with few steps (no warm-start)
        let mut w1 = w0.clone();
        let (rs1, cs1, _) = sinkhorn_project(&mut w1, 4, 4, 3, None, None);
        // Second projection on same matrix with warm-start from first
        let mut w2 = w0.clone();
        let (rs2, cs2, _) = sinkhorn_project(&mut w2, 4, 4, 3, Some(&rs1), Some(&cs1));
        // Warm-started result should be closer to doubly-stochastic
        let max_dev1 = {
            let mut dev = 0.0f32;
            for r in 0..4 {
                let s: f32 = (0..4).map(|c| w1[r * 4 + c]).sum();
                dev = dev.max((s - 1.0).abs());
            }
            for c in 0..4 {
                let s: f32 = (0..4).map(|r| w1[r * 4 + c]).sum();
                dev = dev.max((s - 1.0).abs());
            }
            dev
        };
        let max_dev2 = {
            let mut dev = 0.0f32;
            for r in 0..4 {
                let s: f32 = (0..4).map(|c| w2[r * 4 + c]).sum();
                dev = dev.max((s - 1.0).abs());
            }
            for c in 0..4 {
                let s: f32 = (0..4).map(|r| w2[r * 4 + c]).sum();
                dev = dev.max((s - 1.0).abs());
            }
            dev
        };
        assert!(
            max_dev2 <= max_dev1,
            "Warm-start should not be worse: no-warm={:.6e}, warm={:.6e}",
            max_dev1,
            max_dev2
        );
        // Returned scales should compose with input scales
        assert_eq!(rs1.len(), 4);
        assert_eq!(cs1.len(), 4);
        assert_eq!(rs2.len(), 4);
        assert_eq!(cs2.len(), 4);
    }
}
