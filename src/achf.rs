#[cfg(cuda)]
use crate::autograd::Context;
use crate::autograd::Tensor;
use crate::config::AchfConfig;
use crate::dtype::{bf16, Dtype};
use crate::nn::{Linear, Module};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

const AMA_SPARSE_DENSE_ENABLE_MARGIN: f64 = 0.98;

/// Upper bound on how far a consistently-losing path's re-probe interval can be
/// stretched, as a multiple of the base stale limit. A path that is ~Nx slower
/// than the current winner is re-probed every `min(N, this) * base` calls
/// instead of on the fixed base cadence. This is the fix for the steady-state
/// exploration tax: the old fixed cadence re-probed every loser every `base`
/// calls forever, capping even a 10x winner near 60% selection. With backoff a
/// clear winner is selected ~>95% of the time, while a loser whose score
/// approaches the winner's (a regime shift) snaps its interval back to `base`
/// and is tracked tightly again. 16 keeps worst-case re-probe latency bounded
/// (<=128 calls at base 8) so genuine regime changes are still caught quickly.
const AMA_MAX_REPROBE_MULT: u64 = 16;

thread_local! {
    // Reused per-thread accumulator row for the fused frozen-inference residual
    // add-into path, avoiding a heap allocation per call.
    static ACHF_ROW_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

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
    pub gradient_cosine: f64,
    pub previous_gradient: Vec<f32>,
    pub fim_ema: f64,
    pub last_gate: f64,
    pub gate_velocity: f64,
    pub g_min_ema: f64,
    pub freeze_projection: bool,
    pub sinkhorn_iterations: usize,
    pub sinkhorn_row_max_dev: f64,
    pub sinkhorn_col_max_dev: f64,
    pub sinkhorn_min_value: f64,
    pub sinkhorn_negative_ratio: f64,
    pub sinkhorn_warm_started: bool,
    /// Relative Frobenius error of the last rank-r truncation:
    /// ||W - W_r||_F / ||W||_F. Zero when no truncation was applied.
    pub low_rank_rel_err: f64,
    /// Relative Frobenius discrepancy introduced by the final pruning mask,
    /// measured against the projected dense reference weight.
    pub prune_rel_err: f64,
    /// Rank actually used by the last truncation (0 = truncation inactive).
    pub low_rank_applied_rank: usize,
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
    pub dense_latency_samples: AtomicU64,
    pub decision_samples: AtomicU64,
    pub memo_hash: AtomicU64,
    pub memo_count: AtomicU64,
    /// Lock-free mirror of `AchfState::freeze_projection`, set once by
    /// `freeze_for_inference`. Lets the inference hot path detect frozen mode
    /// without taking the `state` RwLock on every call.
    pub frozen: AtomicBool,
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
            dense_latency_samples: AtomicU64::new(
                self.dense_latency_samples.load(Ordering::Relaxed),
            ),
            decision_samples: AtomicU64::new(self.decision_samples.load(Ordering::Relaxed)),
            memo_hash: AtomicU64::new(self.memo_hash.load(Ordering::Relaxed)),
            memo_count: AtomicU64::new(self.memo_count.load(Ordering::Relaxed)),
            frozen: AtomicBool::new(self.frozen.load(Ordering::Relaxed)),
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
            dense_latency_samples: AtomicU64::new(0),
            decision_samples: AtomicU64::new(0),
            memo_hash: AtomicU64::new(0),
            memo_count: AtomicU64::new(0),
            frozen: AtomicBool::new(false),
        }
    }
}

fn default_metrics() -> Arc<AchfMetrics> {
    Arc::new(AchfMetrics::default())
}

/// Number of latency buckets keyed by input row count (batch size). Bucket
/// index is floor(log2(num_rows)) clamped to [0, AMA_NUM_BUCKETS). This groups
/// operating points of similar cost (batch 1, 2-3, 4-7, 8-15, ...) so the
/// selector never compares a path measured at batch=1 (~tens of µs) against one
/// measured at batch=64 (~ms). Global (unbucketed) EMAs blend those regimes and
/// make the selector unable to tell which regime it is in — the root cause of
/// the cross-regime thrash A is meant to fix. 16 buckets covers 1..=32768 rows.
const AMA_NUM_BUCKETS: usize = 16;

/// Per-(batch-bucket) snapshot of the AMA selector's latency/selection state.
/// The live selector reads/writes the flat `ama_*` / `ema_*` fields on
/// AchfCache; `switch_ama_bucket` swaps this snapshot in and out of those flat
/// fields when the input's batch bucket changes. Keeping the flat fields lets
/// all existing selector code stay untouched.
#[derive(Clone, Default)]
struct PathBucket {
    ema_cached_ns: f64,
    ema_cached_long_ns: f64,
    ema_sparse_ns: f64,
    ema_sparse_long_ns: f64,
    ema_dense_ns: f64,
    ema_dense_long_ns: f64,
    ama_cached_cold_ns: f64,
    ama_cached_warm_ns: f64,
    ama_sparse_cold_ns: f64,
    ama_sparse_warm_ns: f64,
    ama_dense_cold_ns: f64,
    ama_dense_warm_ns: f64,
    ama_cached_warm_count: u64,
    ama_sparse_warm_count: u64,
    ama_dense_warm_count: u64,
    ama_cached_stale: u64,
    ama_sparse_stale: u64,
    ama_dense_stale: u64,
    ama_prev_path: Option<InferencePath>,
    ama_dwell: u64,
    adaptive_bias: f64,
    initialized: bool,
}

#[derive(Clone)]
pub struct AchfCache {
    pub dense: Option<Vec<f32>>,
    pub bias: Option<Vec<f32>>,
    pub in_dim: usize,
    pub out_dim: usize,
    /// CSR-style sparse operator for the pruned weight, indexed by INPUT
    /// dimension (the layout is input-stationary: for input dim `i`, the
    /// nonzero output columns and their weights live in
    /// `csr_cols[csr_row_ptr[i]..csr_row_ptr[i+1]]` /
    /// `csr_vals[...]`). This is what makes the Sparse path genuinely skip
    /// pruned weights instead of multiplying by stored zeros — it trades fewer
    /// FMAs (win at high sparsity) for scattered output writes and no
    /// contiguous SIMD (loss at low sparsity), which is the exact
    /// locality-vs-FLOPs crossover ACHF's path selector exists to navigate.
    pub csr_row_ptr: Option<Vec<u32>>,
    pub csr_cols: Option<Vec<u32>>,
    pub csr_vals: Option<Vec<f32>>,
    pub ema_cached_ns: f64,
    pub ema_cached_long_ns: f64,
    pub ema_sparse_ns: f64,
    pub ema_sparse_long_ns: f64,
    pub ema_dense_ns: f64,
    pub ema_dense_long_ns: f64,
    pub decision_ema_ns: f64,
    pub decision_ema_long_ns: f64,
    ama_cached_cold_ns: f64,
    ama_cached_warm_ns: f64,
    ama_sparse_cold_ns: f64,
    ama_sparse_warm_ns: f64,
    ama_dense_cold_ns: f64,
    ama_dense_warm_ns: f64,
    ama_cached_warm_count: u64,
    ama_sparse_warm_count: u64,
    ama_dense_warm_count: u64,
    ama_cached_stale: u64,
    ama_sparse_stale: u64,
    ama_dense_stale: u64,
    ama_prev_path: Option<InferencePath>,
    ama_dwell: u64,
    ama_switches: u64,
    ama_probes: u64,
    ama_force_latency_sample: bool,
    pub adaptive_bias: f64,
    /// Per-batch-bucket selector state. Empty until the first adaptive call
    /// lazily initializes it. The flat `ama_*`/`ema_*` fields above mirror
    /// `ama_buckets[ama_active_bucket]`; `switch_ama_bucket` keeps them in sync.
    ama_buckets: Vec<PathBucket>,
    ama_active_bucket: usize,
    pub last_input_hash: Option<u64>,
    pub last_output: Option<Vec<f32>>,
    pub last_input_count: u64,
    /// Warm-start row scale vectors for Sinkhorn projection (cached between calls).
    pub sinkhorn_row_scales: Option<Vec<f64>>,
    /// Warm-start column scale vectors for Sinkhorn projection (cached between calls).
    pub sinkhorn_col_scales: Option<Vec<f64>>,
    #[cfg(cuda)]
    pub sparse_mask_cuda: Option<Arc<crate::cuda::memory::DevicePtr<u8>>>,
}

fn default_state() -> Arc<RwLock<AchfState>> {
    Arc::new(RwLock::new(AchfState {
        step: 0,
        gate_step: 0,
        grad_ema: 0.0,
        gradient_cosine: 0.0,
        previous_gradient: Vec::new(),
        fim_ema: 0.0,
        last_gate: 1.0,
        gate_velocity: 0.0,
        g_min_ema: 0.0,
        freeze_projection: false,
        sinkhorn_iterations: 0,
        sinkhorn_row_max_dev: 0.0,
        sinkhorn_col_max_dev: 0.0,
        sinkhorn_min_value: 0.0,
        sinkhorn_negative_ratio: 0.0,
        sinkhorn_warm_started: false,
        low_rank_rel_err: 0.0,
        prune_rel_err: 0.0,
        low_rank_applied_rank: 0,
    }))
}

fn default_cache() -> Arc<RwLock<AchfCache>> {
    Arc::new(RwLock::new(AchfCache {
        dense: None,
        bias: None,
        in_dim: 0,
        out_dim: 0,
        csr_row_ptr: None,
        csr_cols: None,
        csr_vals: None,
        ema_cached_ns: 0.0,
        ema_cached_long_ns: 0.0,
        ema_sparse_ns: 0.0,
        ema_sparse_long_ns: 0.0,
        ema_dense_ns: 0.0,
        ema_dense_long_ns: 0.0,
        decision_ema_ns: 0.0,
        decision_ema_long_ns: 0.0,
        ama_cached_cold_ns: 0.0,
        ama_cached_warm_ns: 0.0,
        ama_sparse_cold_ns: 0.0,
        ama_sparse_warm_ns: 0.0,
        ama_dense_cold_ns: 0.0,
        ama_dense_warm_ns: 0.0,
        ama_cached_warm_count: 0,
        ama_sparse_warm_count: 0,
        ama_dense_warm_count: 0,
        ama_cached_stale: 0,
        ama_sparse_stale: 0,
        ama_dense_stale: 0,
        ama_prev_path: None,
        ama_dwell: 0,
        ama_switches: 0,
        ama_probes: 0,
        ama_force_latency_sample: false,
        adaptive_bias: 0.0,
        ama_buckets: Vec::new(),
        ama_active_bucket: 0,
        last_input_hash: None,
        last_output: None,
        last_input_count: 0,
        sinkhorn_row_scales: None,
        sinkhorn_col_scales: None,
        #[cfg(cuda)]
        sparse_mask_cuda: None,
    }))
}

/// Map an input row count (batch size) to a latency bucket index:
/// floor(log2(num_rows)) clamped to [0, AMA_NUM_BUCKETS).
fn ama_bucket_index(num_rows: usize) -> usize {
    if num_rows <= 1 {
        return 0;
    }
    // ilog2(num_rows) is the index of the highest set bit; e.g. 1->0, 2->1,
    // 3->1, 4->2, ... which is exactly the "batch 2-3, 4-7, ..." grouping.
    (num_rows.ilog2() as usize).min(AMA_NUM_BUCKETS - 1)
}

impl AchfCache {
    /// Copy the flat `ama_*`/`ema_*` fields into the given bucket snapshot.
    fn store_active_bucket(&self, b: &mut PathBucket) {
        b.ema_cached_ns = self.ema_cached_ns;
        b.ema_cached_long_ns = self.ema_cached_long_ns;
        b.ema_sparse_ns = self.ema_sparse_ns;
        b.ema_sparse_long_ns = self.ema_sparse_long_ns;
        b.ema_dense_ns = self.ema_dense_ns;
        b.ema_dense_long_ns = self.ema_dense_long_ns;
        b.ama_cached_cold_ns = self.ama_cached_cold_ns;
        b.ama_cached_warm_ns = self.ama_cached_warm_ns;
        b.ama_sparse_cold_ns = self.ama_sparse_cold_ns;
        b.ama_sparse_warm_ns = self.ama_sparse_warm_ns;
        b.ama_dense_cold_ns = self.ama_dense_cold_ns;
        b.ama_dense_warm_ns = self.ama_dense_warm_ns;
        b.ama_cached_warm_count = self.ama_cached_warm_count;
        b.ama_sparse_warm_count = self.ama_sparse_warm_count;
        b.ama_dense_warm_count = self.ama_dense_warm_count;
        b.ama_cached_stale = self.ama_cached_stale;
        b.ama_sparse_stale = self.ama_sparse_stale;
        b.ama_dense_stale = self.ama_dense_stale;
        b.ama_prev_path = self.ama_prev_path;
        b.ama_dwell = self.ama_dwell;
        b.adaptive_bias = self.adaptive_bias;
        b.initialized = true;
    }

    /// Load a bucket snapshot into the flat `ama_*`/`ema_*` fields.
    fn load_active_bucket(&mut self, b: &PathBucket) {
        self.ema_cached_ns = b.ema_cached_ns;
        self.ema_cached_long_ns = b.ema_cached_long_ns;
        self.ema_sparse_ns = b.ema_sparse_ns;
        self.ema_sparse_long_ns = b.ema_sparse_long_ns;
        self.ema_dense_ns = b.ema_dense_ns;
        self.ema_dense_long_ns = b.ema_dense_long_ns;
        self.ama_cached_cold_ns = b.ama_cached_cold_ns;
        self.ama_cached_warm_ns = b.ama_cached_warm_ns;
        self.ama_sparse_cold_ns = b.ama_sparse_cold_ns;
        self.ama_sparse_warm_ns = b.ama_sparse_warm_ns;
        self.ama_dense_cold_ns = b.ama_dense_cold_ns;
        self.ama_dense_warm_ns = b.ama_dense_warm_ns;
        self.ama_cached_warm_count = b.ama_cached_warm_count;
        self.ama_sparse_warm_count = b.ama_sparse_warm_count;
        self.ama_dense_warm_count = b.ama_dense_warm_count;
        self.ama_cached_stale = b.ama_cached_stale;
        self.ama_sparse_stale = b.ama_sparse_stale;
        self.ama_dense_stale = b.ama_dense_stale;
        self.ama_prev_path = b.ama_prev_path;
        self.ama_dwell = b.ama_dwell;
        self.adaptive_bias = b.adaptive_bias;
    }

    /// Ensure the flat fields reflect the bucket for `num_rows`. If the target
    /// bucket differs from the active one, save the active bucket and load the
    /// target (initializing it with the configured cost bias on first use).
    /// Called at the start of both selection and latency recording with the
    /// same `num_rows`, so the two always agree on the active bucket.
    fn switch_ama_bucket(&mut self, num_rows: usize, cost_bias: f64) {
        if self.ama_buckets.is_empty() {
            let mut init = PathBucket {
                adaptive_bias: cost_bias,
                ..Default::default()
            };
            // Seed bucket 0 from whatever the flat fields already hold.
            self.store_active_bucket(&mut init);
            self.ama_buckets = vec![init; AMA_NUM_BUCKETS];
            self.ama_active_bucket = ama_bucket_index(num_rows);
            // Loading the (possibly different) target bucket; all buckets start
            // identical here, so this just sets the active index consistently.
            let target = self.ama_buckets[self.ama_active_bucket].clone();
            self.load_active_bucket(&target);
            return;
        }
        let target = ama_bucket_index(num_rows);
        if target == self.ama_active_bucket {
            return;
        }
        let mut prev = std::mem::take(&mut self.ama_buckets[self.ama_active_bucket]);
        self.store_active_bucket(&mut prev);
        self.ama_buckets[self.ama_active_bucket] = prev;
        self.ama_active_bucket = target;
        let mut next = self.ama_buckets[target].clone();
        if !next.initialized {
            next.adaptive_bias = cost_bias;
            next.initialized = true;
        }
        self.load_active_bucket(&next);
    }
}

#[derive(Clone, Copy, Debug, Default)]
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
    pub ema_dense_ns: f64,
    pub ema_dense_long_ns: f64,
    pub decision_ema_ns: f64,
    pub decision_ema_long_ns: f64,
    pub cached_cold_ema_ns: f64,
    pub cached_warm_ema_ns: f64,
    pub sparse_cold_ema_ns: f64,
    pub sparse_warm_ema_ns: f64,
    pub dense_cold_ema_ns: f64,
    pub dense_warm_ema_ns: f64,
    pub cached_warmness: f64,
    pub sparse_warmness: f64,
    pub dense_warmness: f64,
    pub cached_stale_age: u64,
    pub sparse_stale_age: u64,
    pub dense_stale_age: u64,
    pub path_switches: u64,
    pub path_probes: u64,
    pub adaptive_bias: f64,
    pub latency_samples: u64,
    pub dense_latency_samples: u64,
    pub decision_samples: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AchfSparsityStats {
    pub total_weights: usize,
    pub nonzero_weights: usize,
    pub zero_weights: usize,
    pub sparsity: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct AchfMemoryStats {
    pub layers: usize,
    pub candidate_total_weights: usize,
    pub candidate_nonzero_weights: usize,
    pub reference_parameter_bytes: usize,
    pub candidate_dense_bytes: usize,
    pub sparse_mask_bytes: usize,
    pub cached_dense_bytes: usize,
    pub cached_bias_bytes: usize,
    pub csr_row_ptr_bytes: usize,
    pub csr_column_bytes: usize,
    pub csr_value_bytes: usize,
    pub memoized_output_bytes: usize,
    pub sinkhorn_state_bytes: usize,
    pub total_materialized_bytes: usize,
}

pub fn aggregate_memory_stats_iter<I>(iter: I) -> AchfMemoryStats
where
    I: IntoIterator<Item = AchfMemoryStats>,
{
    let mut out = AchfMemoryStats::default();
    for stats in iter {
        out.layers += stats.layers;
        out.candidate_total_weights += stats.candidate_total_weights;
        out.candidate_nonzero_weights += stats.candidate_nonzero_weights;
        out.reference_parameter_bytes += stats.reference_parameter_bytes;
        out.candidate_dense_bytes += stats.candidate_dense_bytes;
        out.sparse_mask_bytes += stats.sparse_mask_bytes;
        out.cached_dense_bytes += stats.cached_dense_bytes;
        out.cached_bias_bytes += stats.cached_bias_bytes;
        out.csr_row_ptr_bytes += stats.csr_row_ptr_bytes;
        out.csr_column_bytes += stats.csr_column_bytes;
        out.csr_value_bytes += stats.csr_value_bytes;
        out.memoized_output_bytes += stats.memoized_output_bytes;
        out.sinkhorn_state_bytes += stats.sinkhorn_state_bytes;
        out.total_materialized_bytes += stats.total_materialized_bytes;
    }
    out
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
    pub gate_velocity: f64,
    pub g_min: f64,
    pub grad_ema: f64,
    pub gradient_cosine: f64,
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
    /// Relative Frobenius error of the last rank-r truncation (0 when inactive).
    pub low_rank_rel_err: f64,
    /// Relative Frobenius discrepancy introduced by final magnitude pruning.
    pub prune_rel_err: f64,
    /// Rank actually applied by the last truncation (0 when inactive).
    pub low_rank_applied_rank: usize,
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
    let mut out = AchfCacheStats::default();
    let mut count_cached = 0usize;
    let mut count_low_rank = 0usize;
    let mut count_dense = 0usize;
    let mut count_cached_long = 0usize;
    let mut count_low_rank_long = 0usize;
    let mut count_dense_long = 0usize;
    let mut count_decision = 0usize;
    let mut count_decision_long = 0usize;
    let mut count_bias = 0usize;
    let mut count_cached_cold = 0usize;
    let mut count_cached_warm = 0usize;
    let mut count_sparse_cold = 0usize;
    let mut count_sparse_warm = 0usize;
    let mut count_dense_cold = 0usize;
    let mut count_dense_warm = 0usize;
    let mut count_warmness = 0usize;
    for s in iter {
        out.calls += s.calls;
        out.cache_hits += s.cache_hits;
        out.cache_misses += s.cache_misses;
        out.cache_skips += s.cache_skips;
        out.sparse_paths += s.sparse_paths;
        out.dense_paths += s.dense_paths;
        out.latency_samples += s.latency_samples;
        out.dense_latency_samples += s.dense_latency_samples;
        out.decision_samples += s.decision_samples;
        out.path_switches += s.path_switches;
        out.path_probes += s.path_probes;
        out.cached_stale_age = out.cached_stale_age.max(s.cached_stale_age);
        out.sparse_stale_age = out.sparse_stale_age.max(s.sparse_stale_age);
        out.dense_stale_age = out.dense_stale_age.max(s.dense_stale_age);
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
        if s.ema_dense_ns > 0.0 {
            out.ema_dense_ns += s.ema_dense_ns;
            count_dense += 1;
        }
        if s.ema_dense_long_ns > 0.0 {
            out.ema_dense_long_ns += s.ema_dense_long_ns;
            count_dense_long += 1;
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
        if s.cached_cold_ema_ns > 0.0 {
            out.cached_cold_ema_ns += s.cached_cold_ema_ns;
            count_cached_cold += 1;
        }
        if s.cached_warm_ema_ns > 0.0 {
            out.cached_warm_ema_ns += s.cached_warm_ema_ns;
            count_cached_warm += 1;
        }
        if s.sparse_cold_ema_ns > 0.0 {
            out.sparse_cold_ema_ns += s.sparse_cold_ema_ns;
            count_sparse_cold += 1;
        }
        if s.sparse_warm_ema_ns > 0.0 {
            out.sparse_warm_ema_ns += s.sparse_warm_ema_ns;
            count_sparse_warm += 1;
        }
        if s.dense_cold_ema_ns > 0.0 {
            out.dense_cold_ema_ns += s.dense_cold_ema_ns;
            count_dense_cold += 1;
        }
        if s.dense_warm_ema_ns > 0.0 {
            out.dense_warm_ema_ns += s.dense_warm_ema_ns;
            count_dense_warm += 1;
        }
        out.cached_warmness += s.cached_warmness;
        out.sparse_warmness += s.sparse_warmness;
        out.dense_warmness += s.dense_warmness;
        count_warmness += 1;
    }
    if count_cached > 0 {
        out.ema_cached_ns /= count_cached as f64;
    }
    if count_low_rank > 0 {
        out.ema_sparse_ns /= count_low_rank as f64;
    }
    if count_dense > 0 {
        out.ema_dense_ns /= count_dense as f64;
    }
    if count_cached_long > 0 {
        out.ema_cached_long_ns /= count_cached_long as f64;
    }
    if count_low_rank_long > 0 {
        out.ema_sparse_long_ns /= count_low_rank_long as f64;
    }
    if count_dense_long > 0 {
        out.ema_dense_long_ns /= count_dense_long as f64;
    }
    if count_decision > 0 {
        out.decision_ema_ns /= count_decision as f64;
    }
    if count_decision_long > 0 {
        out.decision_ema_long_ns /= count_decision_long as f64;
    }
    if count_cached_cold > 0 {
        out.cached_cold_ema_ns /= count_cached_cold as f64;
    }
    if count_cached_warm > 0 {
        out.cached_warm_ema_ns /= count_cached_warm as f64;
    }
    if count_sparse_cold > 0 {
        out.sparse_cold_ema_ns /= count_sparse_cold as f64;
    }
    if count_sparse_warm > 0 {
        out.sparse_warm_ema_ns /= count_sparse_warm as f64;
    }
    if count_dense_cold > 0 {
        out.dense_cold_ema_ns /= count_dense_cold as f64;
    }
    if count_dense_warm > 0 {
        out.dense_warm_ema_ns /= count_dense_warm as f64;
    }
    if count_warmness > 0 {
        out.cached_warmness /= count_warmness as f64;
        out.sparse_warmness /= count_warmness as f64;
        out.dense_warmness /= count_warmness as f64;
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
        Self::from_linear(Linear::new(in_features, out_features, bias, seed), config)
    }

    pub fn from_linear(weight: Linear, config: AchfConfig) -> Self {
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
            gate_velocity: state.gate_velocity,
            g_min: state.g_min_ema,
            grad_ema: state.grad_ema,
            gradient_cosine: state.gradient_cosine,
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
            low_rank_rel_err: state.low_rank_rel_err,
            prune_rel_err: state.prune_rel_err,
            low_rank_applied_rank: state.low_rank_applied_rank,
        }
    }

    pub fn memory_stats(&self) -> AchfMemoryStats {
        let dtype_bytes = |dtype: Dtype| match dtype {
            Dtype::F64 => std::mem::size_of::<f64>(),
            Dtype::F32 => std::mem::size_of::<f32>(),
            Dtype::BF16 => std::mem::size_of::<bf16>(),
            Dtype::I8 => std::mem::size_of::<i8>(),
        };
        let reference_parameter_bytes = self.weight.weight.numel()
            * dtype_bytes(self.weight.weight.dtype)
            + self
                .weight
                .bias
                .as_ref()
                .map_or(0, |bias| bias.numel() * dtype_bytes(bias.dtype));
        let candidate_dense_bytes = self.sparse_weight.as_ref().map_or(0, |candidate| {
            candidate.weight.numel() * dtype_bytes(candidate.weight.dtype)
        });
        let sparsity = self.inference_sparsity_stats();
        let sparse_mask_bytes = self.sparse_mask.as_ref().map_or(0, Vec::len);
        let cache = self.cache.read().unwrap();
        let cached_dense_bytes = cache
            .dense
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<f32>());
        let cached_bias_bytes = cache
            .bias
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<f32>());
        let csr_row_ptr_bytes = cache
            .csr_row_ptr
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<u32>());
        let csr_column_bytes = cache
            .csr_cols
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<u32>());
        let csr_value_bytes = cache
            .csr_vals
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<f32>());
        let memoized_output_bytes = cache
            .last_output
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<f32>());
        let sinkhorn_state_bytes = cache
            .sinkhorn_row_scales
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<f64>())
            + cache
                .sinkhorn_col_scales
                .as_ref()
                .map_or(0, |values| values.len() * std::mem::size_of::<f64>());
        let total_materialized_bytes = reference_parameter_bytes
            + candidate_dense_bytes
            + sparse_mask_bytes
            + cached_dense_bytes
            + cached_bias_bytes
            + csr_row_ptr_bytes
            + csr_column_bytes
            + csr_value_bytes
            + memoized_output_bytes
            + sinkhorn_state_bytes;
        AchfMemoryStats {
            layers: 1,
            candidate_total_weights: sparsity.map_or(0, |stats| stats.total_weights),
            candidate_nonzero_weights: sparsity.map_or(0, |stats| stats.nonzero_weights),
            reference_parameter_bytes,
            candidate_dense_bytes,
            sparse_mask_bytes,
            cached_dense_bytes,
            cached_bias_bytes,
            csr_row_ptr_bytes,
            csr_column_bytes,
            csr_value_bytes,
            memoized_output_bytes,
            sinkhorn_state_bytes,
            total_materialized_bytes,
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        // `sparse_weight` is a detached, derived inference representation of
        // `weight`, not an independently trainable parameter set.
        self.weight.parameters()
    }

    pub fn inference_sparsity_stats(&self) -> Option<AchfSparsityStats> {
        let sparse = self.valid_sparse_weight()?;
        let total_weights = sparse.in_features.checked_mul(sparse.out_features)?;
        if total_weights == 0 {
            return Some(AchfSparsityStats {
                total_weights: 0,
                nonzero_weights: 0,
                zero_weights: 0,
                sparsity: 0.0,
            });
        }
        // Count materialized values, not `sparse_mask`: threshold=0 may mark an
        // exact zero as "kept", while the CSR path intentionally omits it.
        // This statistic describes the work inference actually performs.
        let nonzero_weights = sparse
            .weight
            .data_to_f32_vec()
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let zero_weights = total_weights.saturating_sub(nonzero_weights);
        Some(AchfSparsityStats {
            total_weights,
            nonzero_weights,
            zero_weights,
            sparsity: zero_weights as f64 / total_weights as f64,
        })
    }

    fn forward_gated_candidate(&self, input: &Tensor, gate: f64) -> Tensor {
        let output = self.valid_sparse_weight().map_or_else(
            || self.weight.forward(input),
            |sparse| sparse.forward(input),
        );
        self.scale_tensor_output(output, gate)
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
    fn sparse_mask_cuda(&self, mask: &[u8]) -> Option<Arc<crate::cuda::memory::DevicePtr<u8>>> {
        use crate::cuda::memory::{alloc, copy_h2d};

        if mask.is_empty() {
            return None;
        }
        {
            let cache = self.cache.read().unwrap();
            if let Some(mask_cuda) = cache.sparse_mask_cuda.as_ref() {
                if mask_cuda.len() == mask.len() {
                    return Some(mask_cuda.clone());
                }
            }
        }

        let d_mask = alloc::<u8>(mask.len()).ok()?;
        copy_h2d(&d_mask, mask).ok()?;
        let d_mask = Arc::new(d_mask);
        let mut cache = self.cache.write().unwrap();
        cache.sparse_mask_cuda = Some(d_mask.clone());
        Some(d_mask)
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
        let d_mask = self.sparse_mask_cuda(mask)?;
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
        use crate::cuda::memory::{alloc, CudaBuffer};

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
        let d_mask = self.sparse_mask_cuda(mask)?;
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
        if self.is_training_mode() {
            let gate = self.compute_gate();
            return self.scale_tensor_output(self.weight.forward(input), gate);
        }
        let g = self.infer_gate_value();
        #[cfg(cuda)]
        if self.has_valid_sparse_state() && input.device == Device::Cuda {
            if let Some(sparse_out) = self.forward_sparse_tensor_cuda(input) {
                if let Some(sparse_scaled) = sparse_out.scale_cuda(g) {
                    return sparse_scaled;
                }
            }
        }
        self.forward_gated_candidate(input, g)
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
        if self.is_training_mode() {
            let g = self.compute_gate();
            if g <= 0.001 {
                return self.zero_tensor_output(x);
            }
            return self.scale_tensor_output(self.weight.forward(x), g);
        }
        let g = self.infer_gate_value();
        if g <= 0.001 {
            return self.zero_tensor_output(x);
        }
        self.forward_gated_candidate(x, g)
    }

    pub fn forward_inference_residual(&self, x: &[f32]) -> Vec<f32> {
        if !self.config.enabled {
            return self.zero_inference_output(x);
        }
        // Frozen fast path: skip input hashing, memoization write-back and
        // latency probing (all training-only aids). Produces the same values as
        // the general path below when the frozen selector would pick Cached.
        // Disabled in full mode, which deliberately routes through the full
        // path so the AMA selector and latency sampling stay live.
        if self.metrics.frozen.load(Ordering::Relaxed) && self.config.uses_frozen_cached_fast_path()
        {
            if let Some(out) = self.try_frozen_cached_scaled(x) {
                return out;
            }
        }
        if self.weight.in_features == 0 || !x.len().is_multiple_of(self.weight.in_features) {
            self.metrics.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
            return self.zero_inference_output(x);
        }
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
        let sample_latency = sample_latency || self.consume_forced_latency_sample();
        if decision_ns > 0.0 {
            self.record_decision_latency(decision_ns);
        }
        let (raw_out, elapsed_ns) = if sample_latency {
            let start = Instant::now();
            let out = match path {
                InferencePath::Cached => self
                    .forward_inference_cached(x)
                    .unwrap_or_else(|| self.forward_inference_dense_path(x)),
                InferencePath::Sparse => self.forward_inference_sparse(x),
                InferencePath::Dense => self.forward_inference_dense_path(x),
            };
            (out, start.elapsed().as_nanos() as f64)
        } else {
            let out = match path {
                InferencePath::Cached => self
                    .forward_inference_cached(x)
                    .unwrap_or_else(|| self.forward_inference_dense_path(x)),
                InferencePath::Sparse => self.forward_inference_sparse(x),
                InferencePath::Dense => self.forward_inference_dense_path(x),
            };
            (out, 0.0)
        };
        if elapsed_ns > 0.0 {
            let num_rows = x.len().checked_div(self.weight.in_features).unwrap_or(1);
            self.record_path_latency(path, elapsed_ns, num_rows);
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

    /// Fused frozen-inference residual: computes the cached dense operator,
    /// scales by the inference gate, and ADDS it into `out` in place — with no
    /// heap allocation and none of the adaptive machinery (input hashing,
    /// memoization write-back, latency probing) that `forward_inference_residual`
    /// performs. That machinery only helps while weights change during training;
    /// on a frozen layer with per-step-varying inputs it is pure overhead.
    ///
    /// Falls back to the general (allocating) path and a manual add whenever the
    /// layer is not frozen or the cached operator is not the selected path, so
    /// results are identical to `forward_inference_residual` in every case.
    pub fn forward_inference_residual_add_into(&self, x: &[f32], out: &mut [f32]) {
        if self.metrics.frozen.load(Ordering::Relaxed)
            && self.config.uses_frozen_cached_fast_path()
            && self.config.enabled
            && self.try_frozen_cached_add_into(x, out)
        {
            return;
        }
        let residual = self.forward_inference_residual(x);
        let n = residual.len().min(out.len());
        for (o, &r) in out[..n].iter_mut().zip(&residual[..n]) {
            *o += r;
        }
    }

    /// Frozen fast path that allocates and returns the gated residual (used by
    /// `forward_inference_residual`). Returns `None` — leaving the caller to run
    /// the general path — whenever the cached operator is not the selected path,
    /// mirroring `select_frozen_path`'s validity checks so numerics stay identical.
    fn try_frozen_cached_scaled(&self, x: &[f32]) -> Option<Vec<f32>> {
        if !self.frozen_cache_selectable(x) {
            return None;
        }
        let mut out = self.forward_inference_cached(x)?;
        let g = self.infer_gate_value() as f32;
        for v in out.iter_mut() {
            *v *= g;
        }
        self.metrics.calls.fetch_add(1, Ordering::Relaxed);
        self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
        Some(out)
    }

    /// Shared validity gate mirroring `select_frozen_path`: true when the fused
    /// cached operator is shape/rows/sparsity-valid for `x` and would be picked.
    fn frozen_cache_selectable(&self, x: &[f32]) -> bool {
        let Ok(cache) = self.cache.try_read() else {
            return false;
        };
        let Some(dense) = cache.dense.as_ref() else {
            return false;
        };
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        if in_dim == 0 || out_dim == 0 {
            return false;
        }
        if dense.len() != in_dim.saturating_mul(out_dim) {
            return false;
        }
        if cache.bias.as_ref().is_some_and(|b| b.len() != out_dim) {
            return false;
        }
        if !x.len().is_multiple_of(in_dim) {
            return false;
        }
        let num_rows = x.len() / in_dim;
        if self.config.cache_min_rows > 0 && num_rows < self.config.cache_min_rows {
            return false;
        }
        if self.config.cache_min_nonzero_ratio > 0.0 {
            let ratio = self.estimate_nonzero_ratio(x, in_dim, num_rows);
            if ratio < self.config.cache_min_nonzero_ratio {
                return false;
            }
        }
        true
    }

    /// Attempt the fused cached matmul + gate scale + add-into. Returns `false`
    /// (touching nothing) when the cache is unavailable or the frozen path would
    /// not select `Cached`, mirroring `select_frozen_path`'s validity checks so
    /// numerics stay identical.
    fn try_frozen_cached_add_into(&self, x: &[f32], out: &mut [f32]) -> bool {
        use crate::simd::add_scaled_row_f32;
        let Ok(cache) = self.cache.try_read() else {
            return false;
        };
        let Some(dense) = cache.dense.as_ref() else {
            return false;
        };
        let bias = cache.bias.as_ref();
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        if in_dim == 0 || out_dim == 0 {
            return false;
        }
        if dense.len() != in_dim.saturating_mul(out_dim) {
            return false;
        }
        if bias.is_some_and(|b| b.len() != out_dim) {
            return false;
        }
        if !x.len().is_multiple_of(in_dim) {
            return false;
        }
        let num_rows = x.len() / in_dim;
        if self.config.cache_min_rows > 0 && num_rows < self.config.cache_min_rows {
            return false;
        }
        if out.len() != num_rows * out_dim {
            return false;
        }
        // Sparsity gate: matches select_frozen_path so we only take the Cached
        // path when it would have been chosen.
        if self.config.cache_min_nonzero_ratio > 0.0 {
            let ratio = self.estimate_nonzero_ratio(x, in_dim, num_rows);
            if ratio < self.config.cache_min_nonzero_ratio {
                return false;
            }
        }
        let g = self.infer_gate_value() as f32;
        ACHF_ROW_SCRATCH.with(|cell| {
            let mut acc = cell.borrow_mut();
            acc.resize(out_dim, 0.0);
            for r in 0..num_rows {
                let row_in = r * in_dim;
                let row_out = r * out_dim;
                if let Some(b) = bias {
                    acc.copy_from_slice(b);
                } else {
                    acc.iter_mut().for_each(|v| *v = 0.0);
                }
                for i in 0..in_dim {
                    let scale = x[row_in + i];
                    if scale == 0.0 {
                        continue;
                    }
                    let w_row = &dense[i * out_dim..(i + 1) * out_dim];
                    add_scaled_row_f32(&mut acc, w_row, scale);
                }
                add_scaled_row_f32(&mut out[row_out..row_out + out_dim], &acc, g);
            }
        });
        self.metrics.calls.fetch_add(1, Ordering::Relaxed);
        self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
        true
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
                .unwrap_or_else(|| self.forward_inference_dense_path(x)),
            1 if self.has_valid_sparse_state() => self.forward_inference_sparse(x),
            _ => self.forward_inference_dense_path(x),
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
        #[cfg(cuda)]
        {
            if self.weight.weight.device == Device::Cuda
                && crate::cuda::is_available()
                && !self.weight.weight.grad.is_empty()
            {
                if let Some(mean_sq) = crate::cuda::achf::grad_mean_sq(&self.weight.weight) {
                    let grad_rms = mean_sq.sqrt();
                    let diagnostic_gradient = self
                        .config
                        .diagnostics_enabled
                        .then(|| self.weight.weight.grad_to_f32_vec());
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
                    if let Some(gradient) = diagnostic_gradient.as_deref() {
                        Self::update_gradient_cosine(&mut state, gradient);
                    }
                    return;
                }
            }
        }
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        let gradient = self.weight.weight.grad_to_f32_vec();
        for &value in &gradient {
            sum_sq += (value * value) as f64;
        }
        count += gradient.len();
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
        if self.config.diagnostics_enabled {
            Self::update_gradient_cosine(&mut state, &gradient);
        }
    }

    fn update_gradient_cosine(state: &mut AchfState, gradient: &[f32]) {
        if state.previous_gradient.len() == gradient.len() && !gradient.is_empty() {
            let mut dot = 0.0;
            let mut previous_norm_sq = 0.0;
            let mut current_norm_sq = 0.0;
            for (&previous, &current) in state.previous_gradient.iter().zip(gradient.iter()) {
                let previous = previous as f64;
                let current = current as f64;
                dot += previous * current;
                previous_norm_sq += previous * previous;
                current_norm_sq += current * current;
            }
            let denominator = (previous_norm_sq * current_norm_sq).sqrt();
            state.gradient_cosine = if denominator > 0.0 {
                (dot / denominator).clamp(-1.0, 1.0)
            } else {
                0.0
            };
        } else {
            state.gradient_cosine = 0.0;
        }
        state.previous_gradient.clear();
        state.previous_gradient.extend_from_slice(gradient);
    }

    /// Run periodic manifold projection after an optimizer step (not during forward).
    pub fn maybe_project_after_optimizer_step(&self) {
        self.maybe_project();
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
        let warmup_samples = self.ama_warmup_samples();
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
            ema_dense_ns: cache.ema_dense_ns,
            ema_dense_long_ns: cache.ema_dense_long_ns,
            decision_ema_ns: cache.decision_ema_ns,
            decision_ema_long_ns: cache.decision_ema_long_ns,
            cached_cold_ema_ns: cache.ama_cached_cold_ns,
            cached_warm_ema_ns: cache.ama_cached_warm_ns,
            sparse_cold_ema_ns: cache.ama_sparse_cold_ns,
            sparse_warm_ema_ns: cache.ama_sparse_warm_ns,
            dense_cold_ema_ns: cache.ama_dense_cold_ns,
            dense_warm_ema_ns: cache.ama_dense_warm_ns,
            cached_warmness: Self::ama_warmness(cache.ama_cached_warm_count, warmup_samples),
            sparse_warmness: Self::ama_warmness(cache.ama_sparse_warm_count, warmup_samples),
            dense_warmness: Self::ama_warmness(cache.ama_dense_warm_count, warmup_samples),
            cached_stale_age: cache.ama_cached_stale,
            sparse_stale_age: cache.ama_sparse_stale,
            dense_stale_age: cache.ama_dense_stale,
            path_switches: cache.ama_switches,
            path_probes: cache.ama_probes,
            adaptive_bias: cache.adaptive_bias,
            latency_samples: self.metrics.latency_samples.load(Ordering::Relaxed),
            dense_latency_samples: self.metrics.dense_latency_samples.load(Ordering::Relaxed),
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
        state.previous_gradient.clear();
        drop(state);
        self.metrics.frozen.store(true, Ordering::Release);
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
        let previous_gate = state.last_gate;
        state.gate_step += 1;
        let warmup = self.config.gate_warmup_steps;
        let transition = self.config.gate_transition_steps;
        let total = warmup.saturating_add(transition);

        if warmup > 0 && state.gate_step <= warmup {
            state.last_gate = 1.0;
            state.gate_velocity = state.last_gate - previous_gate;
            state.g_min_ema = self.config.g_min;
            return 1.0;
        }

        let target = self.compute_target_gate(&state);

        if transition > 0 && state.gate_step <= total {
            let t = (state.gate_step - warmup) as f64 / transition as f64;
            let g = 1.0 * (1.0 - t) + target * t;
            state.last_gate = g;
            state.gate_velocity = state.last_gate - previous_gate;
            state.g_min_ema = self.config.g_min_momentum * state.g_min_ema
                + (1.0 - self.config.g_min_momentum) * self.config.g_min;
            return g;
        }

        state.last_gate = target;
        state.gate_velocity = state.last_gate - previous_gate;
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

    fn is_training_mode(&self) -> bool {
        !self.state.read().unwrap().freeze_projection
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
        // Project every `proj_freq` optimizer steps. Do NOT call a helper that
        // re-locks `self.state` here: we are holding the write guard above, and
        // `std::sync::RwLock` is not reentrant, so re-locking would deadlock the
        // thread. `proj_freq` is honored as-is; to reduce projection cost during
        // training, raise `proj_freq` in the config rather than scaling it here.
        let freq = self.config.proj_freq;
        if freq == 0 || !state.step.is_multiple_of(freq) {
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
        // Rank-r truncation runs AFTER the manifold projection: the manifold
        // step (rowcol/sinkhorn) fixes the scale/marginals of the operator, and
        // truncating last guarantees the operator that actually reaches the
        // pruning/caching stage satisfies the low-rank constraint. Running the
        // truncation first would let the subsequent normalization re-inflate
        // the rank. The truncation slightly perturbs the manifold property
        // (e.g. doubly-stochastic marginals); this is the standard
        // project-then-truncate compromise and is surfaced via
        // `low_rank_rel_err` diagnostics.
        self.apply_low_rank_truncation();
        self.clear_cache();
    }

    /// Effective truncation rank: `Some(r)` only when 0 < r < min(rows, cols).
    /// `rank == 0` or `rank >= min(dims)` means "no low-rank constraint"
    /// (backward compatible with configs predating real rank semantics).
    fn effective_rank(&self) -> Option<usize> {
        let rows = self.weight.in_features;
        let cols = self.weight.out_features;
        let rank = self.config.rank;
        if rank == 0 || rows == 0 || cols == 0 || rank >= rows.min(cols) {
            None
        } else {
            Some(rank)
        }
    }

    fn apply_low_rank_truncation(&self) {
        let Some(rank) = self.effective_rank() else {
            let mut state = self.state.write().unwrap();
            state.low_rank_rel_err = 0.0;
            state.low_rank_applied_rank = 0;
            return;
        };
        let rows = self.weight.in_features;
        let cols = self.weight.out_features;
        let seed = 0xAC4F_5EEDu64 ^ (rows as u64) ^ ((cols as u64) << 20) ^ ((rank as u64) << 40);
        #[cfg(cuda)]
        {
            if self.weight.weight.device == Device::Cuda && crate::cuda::is_available() {
                if let Some(rel_err) = crate::cuda::achf::apply_low_rank_truncation(
                    &self.weight.weight,
                    rows,
                    cols,
                    rank,
                    seed,
                ) {
                    let mut state = self.state.write().unwrap();
                    state.low_rank_rel_err = rel_err;
                    state.low_rank_applied_rank = rank;
                    return;
                }
            }
        }
        let mut w = self.weight.weight.data_to_f32_vec();
        let rel_err = low_rank_truncate(&mut w, rows, cols, rank, seed);
        sync_weight_from_host_f32(&self.weight.weight, &w);
        let mut state = self.state.write().unwrap();
        state.low_rank_rel_err = rel_err;
        state.low_rank_applied_rank = rank;
    }

    fn project_rowcol(&self) {
        let rows = self.weight.in_features;
        let cols = self.weight.out_features;
        #[cfg(cuda)]
        {
            if self.weight.weight.device == Device::Cuda
                && crate::cuda::is_available()
                && crate::cuda::achf::project_rowcol(&self.weight.weight, rows, cols)
            {
                return;
            }
        }
        let mut w = self.weight.weight.data_to_f32_vec();
        rowcol_project(&mut w, rows, cols);
        sync_weight_from_host_f32(&self.weight.weight, &w);
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
        #[cfg(cuda)]
        {
            if self.weight.weight.device == Device::Cuda && crate::cuda::is_available() {
                if let Some((new_row, new_col, stats)) = crate::cuda::achf::project_sinkhorn(
                    &self.weight.weight,
                    self.weight.in_features,
                    self.weight.out_features,
                    steps,
                    row_scales_f32.as_deref(),
                    col_scales_f32.as_deref(),
                ) {
                    self.update_sinkhorn_projection_stats(stats);
                    let mut cache = self.cache.write().unwrap();
                    cache.sinkhorn_row_scales = Some(new_row.iter().map(|&x| x as f64).collect());
                    cache.sinkhorn_col_scales = Some(new_col.iter().map(|&x| x as f64).collect());
                    return;
                }
            }
        }
        let mut w = self.weight.weight.data_to_f32_vec();
        let (new_row, new_col, stats) = sinkhorn_project(
            &mut w,
            self.weight.in_features,
            self.weight.out_features,
            steps,
            row_scales_f32.as_deref(),
            col_scales_f32.as_deref(),
        );
        sync_weight_from_host_f32(&self.weight.weight, &w);
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

    fn scale_tensor_output(&self, output: Tensor, gate: f64) -> Tensor {
        if (gate - 1.0).abs() <= f64::EPSILON {
            return output;
        }
        let gate_tensor =
            Tensor::with_dtype(vec![gate], vec![1], output.dtype).broadcast(output.shape.clone());
        &output * &gate_tensor
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
    ///
    /// When `config.rank > 0`, the parameter count of a rank-r factorization,
    /// `r * (rows + cols)`, acts as an nnz budget for the sparse path. The
    /// budget is distributed per row (each row keeps at most
    /// `floor(budget / rows)` of its largest-magnitude entries) and intersects
    /// with the magnitude threshold — whichever is sparser wins. With
    /// `rank == 0` (or a rank large enough that the budget exceeds the dense
    /// size) this degenerates to pure threshold pruning, the historic behavior.
    #[allow(dead_code)]
    pub fn prune(&mut self, threshold: f64) {
        let rows = self.weight.in_features;
        let cols = self.weight.out_features;
        let w_data = self.weight.weight.data_to_f32_vec();
        let threshold_f32 = threshold as f32;
        let mut keep: Vec<bool> = w_data.iter().map(|&v| v.abs() >= threshold_f32).collect();

        if self.config.rank > 0 && rows > 0 && cols > 0 && w_data.len() == rows * cols {
            let budget = self.config.rank.saturating_mul(rows + cols);
            // rank >= 1 implies budget >= rows + cols > rows, so per_row >= 1.
            let per_row = (budget / rows).min(cols);
            if per_row < cols {
                let mut row_indices: Vec<usize> = Vec::with_capacity(cols);
                for r in 0..rows {
                    let row = &w_data[r * cols..(r + 1) * cols];
                    row_indices.clear();
                    row_indices.extend((0..cols).filter(|&c| keep[r * cols + c]));
                    if row_indices.len() > per_row {
                        row_indices.sort_unstable_by(|&a, &b| {
                            row[b]
                                .abs()
                                .partial_cmp(&row[a].abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for &c in &row_indices[per_row..] {
                            keep[r * cols + c] = false;
                        }
                    }
                }
            }
        }

        let reference_norm_sq = w_data
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let prune_diff_sq = w_data
            .iter()
            .zip(keep.iter())
            .filter_map(|(&value, &is_kept)| (!is_kept).then_some((value as f64).powi(2)))
            .sum::<f64>();
        let prune_rel_err = if reference_norm_sq > 0.0 {
            (prune_diff_sq / reference_norm_sq).sqrt()
        } else {
            0.0
        };
        let pruned: Vec<f64> = w_data
            .iter()
            .zip(keep.iter())
            .map(|(&v, &k)| if k { v as f64 } else { 0.0 })
            .collect();
        let mask: Vec<u8> = keep.iter().map(|&k| if k { 1 } else { 0 }).collect();
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
        self.state.write().unwrap().prune_rel_err = prune_rel_err;
        self.clear_cache();
    }

    fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.dense = None;
        cache.bias = None;
        cache.in_dim = 0;
        cache.out_dim = 0;
        cache.csr_row_ptr = None;
        cache.csr_cols = None;
        cache.csr_vals = None;
        cache.ema_cached_ns = 0.0;
        cache.ema_cached_long_ns = 0.0;
        cache.ema_sparse_ns = 0.0;
        cache.ema_sparse_long_ns = 0.0;
        cache.ema_dense_ns = 0.0;
        cache.ema_dense_long_ns = 0.0;
        cache.decision_ema_ns = 0.0;
        cache.decision_ema_long_ns = 0.0;
        cache.ama_cached_cold_ns = 0.0;
        cache.ama_cached_warm_ns = 0.0;
        cache.ama_sparse_cold_ns = 0.0;
        cache.ama_sparse_warm_ns = 0.0;
        cache.ama_dense_cold_ns = 0.0;
        cache.ama_dense_warm_ns = 0.0;
        cache.ama_cached_warm_count = 0;
        cache.ama_sparse_warm_count = 0;
        cache.ama_dense_warm_count = 0;
        cache.ama_cached_stale = 0;
        cache.ama_sparse_stale = 0;
        cache.ama_dense_stale = 0;
        cache.ama_prev_path = None;
        cache.ama_dwell = 0;
        cache.ama_switches = 0;
        cache.ama_probes = 0;
        cache.ama_force_latency_sample = false;
        cache.adaptive_bias = self.config.cache_cost_bias;
        cache.ama_buckets = Vec::new();
        cache.ama_active_bucket = 0;
        cache.last_input_hash = None;
        cache.last_output = None;
        cache.last_input_count = 0;
        cache.sinkhorn_row_scales = None;
        cache.sinkhorn_col_scales = None;
        #[cfg(cuda)]
        {
            cache.sparse_mask_cuda = None;
        }
        self.metrics.memo_hash.store(0, Ordering::Relaxed);
        self.metrics.memo_count.store(0, Ordering::Relaxed);
        self.metrics.latency_samples.store(0, Ordering::Relaxed);
        self.metrics
            .dense_latency_samples
            .store(0, Ordering::Relaxed);
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

        // Build the input-stationary CSR view of the pruned weight so the Sparse
        // path can skip zero weights. Row `i` (input dim) holds the (col, val)
        // pairs for every output column whose weight survived pruning. We build
        // it from the materialized dense buffer (not the mask) so a weight that
        // is exactly 0.0 but "kept" by the mask still contributes no work —
        // multiplying by it is wasted, and CSR is precisely about not doing
        // that. row_ptr has in_dim+1 entries (standard CSR prefix-sum layout).
        let mut csr_row_ptr: Vec<u32> = Vec::with_capacity(in_dim + 1);
        let mut csr_cols: Vec<u32> = Vec::new();
        let mut csr_vals: Vec<f32> = Vec::new();
        csr_row_ptr.push(0);
        for i in 0..in_dim {
            let w_row = &dense[i * out_dim..(i + 1) * out_dim];
            for (j, &w) in w_row.iter().enumerate() {
                if w != 0.0 {
                    csr_cols.push(j as u32);
                    csr_vals.push(w);
                }
            }
            csr_row_ptr.push(csr_cols.len() as u32);
        }

        let mut cache = self.cache.write().unwrap();
        cache.dense = Some(dense);
        cache.bias = bias;
        cache.in_dim = in_dim;
        cache.out_dim = out_dim;
        cache.csr_row_ptr = Some(csr_row_ptr);
        cache.csr_cols = Some(csr_cols);
        cache.csr_vals = Some(csr_vals);
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
        use crate::simd::add_scaled_row_f32;
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
                add_scaled_row_f32(out_row, w_row, scale);
            }
        }
        Some(out)
    }

    /// True sparse (CSR SpMV) inference path: for each nonzero input, scatter
    /// only the surviving (pruned) weights into the output. Unlike the Cached
    /// and Dense paths this does NOT touch every output column per input — it
    /// does one FMA-equivalent per stored nonzero. That is the whole point: at
    /// high weight sparsity it does far less arithmetic than Dense/Cached; at
    /// low sparsity the scattered writes and lack of contiguous SIMD make it
    /// lose. Returns None (caller falls back to a dense path) when the CSR view
    /// is absent or the input shape is incompatible, so correctness never
    /// depends on the sparse view being present.
    fn forward_inference_sparse_csr(&self, x: &[f32]) -> Option<Vec<f32>> {
        let cache = self.cache.read().unwrap();
        let row_ptr = cache.csr_row_ptr.as_ref()?;
        let cols = cache.csr_cols.as_ref()?;
        let vals = cache.csr_vals.as_ref()?;
        let bias = cache.bias.as_ref();
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        if in_dim == 0 || out_dim == 0 || row_ptr.len() != in_dim + 1 {
            return None;
        }
        if bias.is_some_and(|b| b.len() != out_dim) {
            return None;
        }
        if !x.len().is_multiple_of(in_dim) {
            return None;
        }
        let num_rows = x.len() / in_dim;
        let mut out = vec![0.0f32; num_rows * out_dim];
        for r in 0..num_rows {
            let row_offset_in = r * in_dim;
            let out_row = &mut out[r * out_dim..(r + 1) * out_dim];
            if let Some(bias) = bias {
                out_row.copy_from_slice(bias);
            }
            for i in 0..in_dim {
                let scale = x[row_offset_in + i];
                if scale == 0.0 {
                    continue;
                }
                let start = row_ptr[i] as usize;
                let end = row_ptr[i + 1] as usize;
                // Scatter: only the surviving weights of input dim `i`.
                for k in start..end {
                    let col = cols[k] as usize;
                    out_row[col] += scale * vals[k];
                }
            }
        }
        Some(out)
    }

    /// Dispatch for the Sparse inference path. Prefers the CUDA masked kernel
    /// when built with CUDA, then the CPU CSR SpMV, and finally falls back to a
    /// dense forward on the pruned weight if no sparse view is available. This
    /// centralizes what used to be three copy-pasted match arms.
    fn forward_inference_sparse(&self, x: &[f32]) -> Vec<f32> {
        #[cfg(cuda)]
        {
            if let Some(out) = self.forward_sparse_inference_cuda(x) {
                return out;
            }
        }
        if let Some(out) = self.forward_inference_sparse_csr(x) {
            return out;
        }
        self.valid_sparse_weight()
            .unwrap_or(&self.weight)
            .forward_inference(x)
    }

    /// Ordinary dense execution of the same frozen inference operator used by
    /// Cached and Sparse. If the sparse state is invalid, fall back to the
    /// original trainable weight as the safe reference implementation.
    fn forward_inference_dense_path(&self, x: &[f32]) -> Vec<f32> {
        if self.has_valid_sparse_state() {
            return self
                .valid_sparse_weight()
                .expect("valid sparse state must include a shape-valid weight")
                .forward_inference(x);
        }
        self.weight.forward_inference(x)
    }

    fn choose_inference_path(&self, x: &[f32]) -> InferencePath {
        if !self.has_valid_sparse_state() {
            self.metrics.calls.fetch_add(1, Ordering::Relaxed);
            self.metrics.dense_paths.fetch_add(1, Ordering::Relaxed);
            return InferencePath::Dense;
        }
        self.ensure_cache();
        // Path selection has two regimes:
        //   * Adaptive (AMA): latency-driven probing + EMA scoring. Used while
        //     training (weights change, so the best path can shift), and also at
        //     inference in full mode — the weights are frozen
        //     but the runtime keeps measuring path latency and re-selecting. This
        //     is what actually exercises the latency-feedback machinery online.
        //   * Frozen deterministic: a frozen layer's fused cached operator is
        //     permanently valid and cheapest, so skip probing and use it. This is
        //     the peak-throughput default once weights stop changing.
        let mode = self.config.mode.trim().to_ascii_lowercase();
        let decision = if self.is_training_mode() {
            self.select_ama_path(x)
        } else {
            match mode.as_str() {
                "fixed_cached" => self.select_fixed_path(x, InferencePath::Cached),
                "fixed_sparse" => self.select_fixed_path(x, InferencePath::Sparse),
                "fixed_dense" => self.select_fixed_path(x, InferencePath::Dense),
                "plain_ema" => self.select_plain_ema_path(x),
                "full" if self.config.uses_adaptive_inference() => self.select_ama_path(x),
                _ if self.config.adaptive_inference => self.select_ama_path(x),
                _ => self.select_frozen_path(x),
            }
        };
        self.metrics.calls.fetch_add(1, Ordering::Relaxed);
        match decision.path {
            InferencePath::Cached => {
                self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            InferencePath::Sparse => {
                if decision.has_cache && decision.cache_skipped {
                    self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
                } else if decision.has_cache {
                    self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
                }
                self.metrics.sparse_paths.fetch_add(1, Ordering::Relaxed);
            }
            InferencePath::Dense => {
                if decision.has_cache && decision.cache_skipped {
                    self.metrics.cache_skips.fetch_add(1, Ordering::Relaxed);
                }
                self.metrics.dense_paths.fetch_add(1, Ordering::Relaxed);
            }
        }
        decision.path
    }

    /// Deterministic path selection for a frozen (inference-only) layer.
    /// Weights no longer change, so the fused cached operator is always valid
    /// and fastest; use it whenever the cache is shape/rows/sparsity-valid,
    /// otherwise fall back to the sparse (or dense) operator. This mirrors the
    /// cache-validity checks in `select_ama_path` but skips latency probing.
    fn select_frozen_path(&self, x: &[f32]) -> AmaPathDecision {
        let Ok(cache) = self.cache.try_read() else {
            return AmaPathDecision {
                path: InferencePath::Sparse,
                has_cache: false,
                cache_skipped: false,
            };
        };
        let has_cache = cache.dense.is_some();
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        let cache_shape_ok =
            has_cache && in_dim > 0 && out_dim > 0 && x.len().is_multiple_of(in_dim);
        let num_rows = if cache_shape_ok { x.len() / in_dim } else { 0 };
        let min_rows_ok = self.config.cache_min_rows == 0 || num_rows >= self.config.cache_min_rows;
        let nonzero_ratio = if cache_shape_ok && min_rows_ok {
            self.estimate_nonzero_ratio(x, in_dim, num_rows)
        } else {
            1.0
        };
        let sparsity_ok = self.config.cache_min_nonzero_ratio <= 0.0
            || nonzero_ratio >= self.config.cache_min_nonzero_ratio;
        let cache_valid = cache_shape_ok && min_rows_ok && sparsity_ok;
        let cache_skipped = has_cache && !cache_valid;
        let path = if cache_valid {
            InferencePath::Cached
        } else if self.has_valid_sparse_state() {
            InferencePath::Sparse
        } else {
            InferencePath::Dense
        };
        AmaPathDecision {
            path,
            has_cache,
            cache_skipped,
        }
    }

    fn select_fixed_path(&self, x: &[f32], requested: InferencePath) -> AmaPathDecision {
        let Ok(cache) = self.cache.try_read() else {
            return AmaPathDecision {
                path: if requested == InferencePath::Dense {
                    InferencePath::Dense
                } else {
                    InferencePath::Sparse
                },
                has_cache: false,
                cache_skipped: requested == InferencePath::Cached,
            };
        };
        let has_cache = cache.dense.is_some();
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        let cache_shape_ok =
            has_cache && in_dim > 0 && out_dim > 0 && x.len().is_multiple_of(in_dim);
        let num_rows = if cache_shape_ok { x.len() / in_dim } else { 0 };
        let min_rows_ok = self.config.cache_min_rows == 0 || num_rows >= self.config.cache_min_rows;
        let nonzero_ratio = if cache_shape_ok && min_rows_ok {
            self.estimate_nonzero_ratio(x, in_dim, num_rows)
        } else {
            1.0
        };
        let sparsity_ok = self.config.cache_min_nonzero_ratio <= 0.0
            || nonzero_ratio >= self.config.cache_min_nonzero_ratio;
        let cache_valid = cache_shape_ok && min_rows_ok && sparsity_ok;
        let cache_skipped = requested == InferencePath::Cached && has_cache && !cache_valid;
        let path = match requested {
            InferencePath::Cached if cache_valid => InferencePath::Cached,
            InferencePath::Cached | InferencePath::Sparse if self.has_valid_sparse_state() => {
                InferencePath::Sparse
            }
            _ => InferencePath::Dense,
        };
        AmaPathDecision {
            path,
            has_cache,
            cache_skipped,
        }
    }

    fn select_ama_path(&self, x: &[f32]) -> AmaPathDecision {
        self.select_dynamic_path(x, true)
    }

    fn select_plain_ema_path(&self, x: &[f32]) -> AmaPathDecision {
        self.select_dynamic_path(x, false)
    }

    fn select_dynamic_path(&self, x: &[f32], guarded: bool) -> AmaPathDecision {
        let Ok(mut cache) = self.cache.try_write() else {
            return AmaPathDecision {
                path: InferencePath::Sparse,
                has_cache: false,
                cache_skipped: false,
            };
        };

        let has_cache = cache.dense.is_some();
        let in_dim = cache.in_dim;
        let out_dim = cache.out_dim;
        let cache_shape_ok =
            has_cache && in_dim > 0 && out_dim > 0 && x.len().is_multiple_of(in_dim);
        let num_rows = if cache_shape_ok { x.len() / in_dim } else { 0 };
        let min_rows_ok = self.config.cache_min_rows == 0 || num_rows >= self.config.cache_min_rows;
        let nonzero_ratio = if cache_shape_ok && min_rows_ok {
            self.estimate_nonzero_ratio(x, in_dim, num_rows)
        } else {
            1.0
        };
        let sparsity_ok = self.config.cache_min_nonzero_ratio <= 0.0
            || nonzero_ratio >= self.config.cache_min_nonzero_ratio;
        let cache_valid = cache_shape_ok && min_rows_ok && sparsity_ok;
        let cache_skipped = has_cache && !cache_valid;

        if !cache_valid {
            let path = if self.has_valid_sparse_state() {
                InferencePath::Sparse
            } else {
                InferencePath::Dense
            };
            Self::ama_commit_path(&mut cache, path, false, self.ama_warmup_samples());
            return AmaPathDecision {
                path,
                has_cache,
                cache_skipped,
            };
        }

        // Select within the latency bucket for this batch size so we never
        // compare a path measured at one batch against another. record_path_
        // latency switches to the same bucket (same num_rows) before recording.
        cache.switch_ama_bucket(num_rows, self.config.cache_cost_bias);
        let path = if guarded {
            self.ama_select_guarded_path(&mut cache)
        } else {
            self.ama_select_plain_ema_path(&mut cache)
        };
        AmaPathDecision {
            path,
            has_cache,
            cache_skipped,
        }
    }

    fn ama_select_plain_ema_path(&self, cache: &mut AchfCache) -> InferencePath {
        let stale_limit = self.ama_stale_limit();
        let cached_needs_probe =
            cache.ema_cached_ns <= 0.0 || cache.ama_cached_stale >= stale_limit;
        let sparse_needs_probe =
            cache.ema_sparse_ns <= 0.0 || cache.ama_sparse_stale >= stale_limit;
        let dense_needs_probe = cache.ema_dense_ns <= 0.0 || cache.ama_dense_stale >= stale_limit;
        if cached_needs_probe || sparse_needs_probe || dense_needs_probe {
            let path = self.ama_probe_path(
                cache,
                cached_needs_probe,
                sparse_needs_probe,
                dense_needs_probe,
            );
            Self::ama_commit_path(cache, path, true, 0);
            return path;
        }

        let path = [
            (InferencePath::Cached, cache.ema_cached_ns),
            (InferencePath::Sparse, cache.ema_sparse_ns),
            (InferencePath::Dense, cache.ema_dense_ns),
        ]
        .into_iter()
        .min_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(InferencePath::Dense, |(path, _)| path);
        Self::ama_commit_path(cache, path, false, 0);
        path
    }

    /// Re-probe interval for one path, given its current effective score and the
    /// best (lowest) score among the warm paths. A path that scores ~Nx worse
    /// than the winner is re-probed every `min(N, AMA_MAX_REPROBE_MULT) * base`
    /// calls instead of every `base`; a path scoring at/near the winner keeps
    /// the tight base cadence. This is a pure function of the live scores, so a
    /// regime shift that improves a loser's score immediately shortens its
    /// interval and restores tight tracking — no persisted backoff state.
    fn ama_reprobe_interval(base: u64, score: f64, best: f64) -> u64 {
        if !score.is_finite() || !best.is_finite() || best <= 0.0 || score <= best {
            return base;
        }
        let mult = (score / best)
            .round()
            .clamp(1.0, AMA_MAX_REPROBE_MULT as f64) as u64;
        base.saturating_mul(mult)
    }

    fn ama_select_guarded_path(&self, cache: &mut AchfCache) -> InferencePath {
        let warmup_samples = self.ama_warmup_samples();
        let cached_warmness = Self::ama_warmness(cache.ama_cached_warm_count, warmup_samples);
        let sparse_warmness = Self::ama_warmness(cache.ama_sparse_warm_count, warmup_samples);
        let dense_warmness = Self::ama_warmness(cache.ama_dense_warm_count, warmup_samples);
        let base_stale = self.ama_stale_limit();

        // Live scores drive the per-path re-probe cadence. A path that loses by
        // a wide margin is re-probed exponentially less often (bounded), which
        // removes the fixed-cadence exploration tax while still catching regime
        // shifts (a newly-competitive loser's interval collapses back to base).
        let cached_score = self.ama_effective_latency(cache, InferencePath::Cached);
        let sparse_score = self.ama_effective_latency(cache, InferencePath::Sparse);
        let dense_score = self.ama_effective_latency(cache, InferencePath::Dense);
        let sparse_allowed = self.ama_sparse_beats_dense(cache);
        // Best score among paths currently eligible to win (sparse only if it
        // beats dense). Used only to scale re-probe intervals.
        let mut best_score = f64::INFINITY;
        for (allowed, score) in [
            (true, cached_score),
            (sparse_allowed, sparse_score),
            (true, dense_score),
        ] {
            if allowed && score.is_finite() && score < best_score {
                best_score = score;
            }
        }
        let cached_interval = Self::ama_reprobe_interval(base_stale, cached_score, best_score);
        let sparse_interval = Self::ama_reprobe_interval(base_stale, sparse_score, best_score);
        let dense_interval = Self::ama_reprobe_interval(base_stale, dense_score, best_score);

        let cached_needs_probe = cache.ema_cached_ns <= 0.0
            || cache.ama_cached_stale >= cached_interval
            || cached_warmness < 1.0;
        let sparse_needs_probe = cache.ema_sparse_ns <= 0.0
            || cache.ama_sparse_stale >= sparse_interval
            || sparse_warmness < 1.0;
        let dense_needs_probe = cache.ema_dense_ns <= 0.0
            || cache.ama_dense_stale >= dense_interval
            || dense_warmness < 1.0;

        if cached_needs_probe || sparse_needs_probe || dense_needs_probe {
            let path = self.ama_probe_path(
                cache,
                cached_needs_probe,
                sparse_needs_probe,
                dense_needs_probe,
            );
            Self::ama_commit_path(cache, path, true, warmup_samples);
            return path;
        }

        let mut selected = InferencePath::Dense;
        let mut selected_score = dense_score;
        if cached_score < selected_score {
            selected = InferencePath::Cached;
            selected_score = cached_score;
        }
        if sparse_allowed && sparse_score < selected_score {
            selected = InferencePath::Sparse;
        }

        if let Some(prev) = cache.ama_prev_path {
            let prev_score = self.ama_effective_latency(cache, prev);
            let selected_score = self.ama_effective_latency(cache, selected);
            let margin = self.ama_switch_margin(cache, prev_score, selected_score);
            let prev_still_allowed = prev != InferencePath::Sparse || sparse_allowed;
            if selected != prev
                && prev_still_allowed
                && (cache.ama_dwell < self.ama_min_dwell() || selected_score + margin >= prev_score)
            {
                selected = prev;
            }
        }

        Self::ama_commit_path(cache, selected, false, warmup_samples);
        selected
    }

    fn ama_probe_path(
        &self,
        cache: &AchfCache,
        cached_needs_probe: bool,
        sparse_needs_probe: bool,
        dense_needs_probe: bool,
    ) -> InferencePath {
        let mut selected = None;
        let mut selected_stale = 0u64;
        for (path, needs_probe, stale) in [
            (
                InferencePath::Cached,
                cached_needs_probe,
                cache.ama_cached_stale,
            ),
            (
                InferencePath::Sparse,
                sparse_needs_probe,
                cache.ama_sparse_stale,
            ),
            (
                InferencePath::Dense,
                dense_needs_probe,
                cache.ama_dense_stale,
            ),
        ] {
            if needs_probe && selected.is_none_or(|_| stale >= selected_stale) {
                selected = Some(path);
                selected_stale = stale;
            }
        }
        selected.unwrap_or(InferencePath::Dense)
    }

    fn ama_sparse_beats_dense(&self, cache: &AchfCache) -> bool {
        if cache.ema_dense_ns <= 0.0 || cache.ema_sparse_ns <= 0.0 {
            return true;
        }
        let sparse_score = self.ama_effective_latency(cache, InferencePath::Sparse);
        let dense_score = self.ama_effective_latency(cache, InferencePath::Dense);
        sparse_score < dense_score * AMA_SPARSE_DENSE_ENABLE_MARGIN
    }

    fn ama_commit_path(
        cache: &mut AchfCache,
        path: InferencePath,
        is_probe: bool,
        warmup_samples: u64,
    ) {
        match path {
            InferencePath::Cached => {
                cache.ama_cached_stale = 0;
                cache.ama_sparse_stale = cache.ama_sparse_stale.saturating_add(1);
                cache.ama_dense_stale = cache.ama_dense_stale.saturating_add(1);
            }
            InferencePath::Sparse => {
                cache.ama_sparse_stale = 0;
                cache.ama_cached_stale = cache.ama_cached_stale.saturating_add(1);
                cache.ama_dense_stale = cache.ama_dense_stale.saturating_add(1);
            }
            InferencePath::Dense => {
                cache.ama_cached_stale = cache.ama_cached_stale.saturating_add(1);
                cache.ama_sparse_stale = cache.ama_sparse_stale.saturating_add(1);
                cache.ama_dense_stale = 0;
            }
        }

        // A probe is a one-shot measurement, NOT an exploitation choice, so it
        // must not disturb the hysteresis state (prev_path / dwell). Previously
        // a probe reset prev_path to the (losing) probed arm and zeroed dwell,
        // which made the very next exploit call trip the `dwell < min_dwell`
        // gate and stay on the loser for an extra call — doubling the probe tax
        // and dragging a clear winner's selection share down to ~60%. Now only
        // genuine exploit selections advance dwell / count switches; a probe
        // leaves the committed winner intact so exploitation resumes next call.
        if !is_probe {
            if cache.ama_prev_path == Some(path) {
                cache.ama_dwell = cache.ama_dwell.saturating_add(1);
            } else {
                if cache.ama_prev_path.is_some() {
                    cache.ama_switches = cache.ama_switches.saturating_add(1);
                }
                cache.ama_prev_path = Some(path);
                cache.ama_dwell = 0;
            }
        } else {
            cache.ama_probes = cache.ama_probes.saturating_add(1);
            cache.ama_force_latency_sample = true;
        }

        let warm_count = match path {
            InferencePath::Cached => cache.ama_cached_warm_count,
            InferencePath::Sparse => cache.ama_sparse_warm_count,
            InferencePath::Dense => cache.ama_dense_warm_count,
        };
        if warm_count < warmup_samples {
            cache.ama_force_latency_sample = true;
        }
    }

    fn ama_effective_latency(&self, cache: &AchfCache, path: InferencePath) -> f64 {
        let blend = self.config.cache_adapt_blend.clamp(0.0, 1.0);
        let (short, long, cold, warm, warm_count) = match path {
            InferencePath::Cached => (
                cache.ema_cached_ns,
                cache.ema_cached_long_ns,
                cache.ama_cached_cold_ns,
                cache.ama_cached_warm_ns,
                cache.ama_cached_warm_count,
            ),
            InferencePath::Sparse => (
                cache.ema_sparse_ns,
                cache.ema_sparse_long_ns,
                cache.ama_sparse_cold_ns,
                cache.ama_sparse_warm_ns,
                cache.ama_sparse_warm_count,
            ),
            // Dense is a genuine, measured competitor (its EMA is recorded in
            // record_path_latency). It used to be hardcoded to INFINITY here,
            // which contradicted the probe machinery (which kept probing dense
            // because its EMA was never set) and pinned selection to dense ~90%
            // of the time even when Cached was empirically fastest. Now dense is
            // scored on its measurements like the other paths; INFINITY survives
            // only as the cold-start prior below, before any measurement exists.
            InferencePath::Dense => (
                cache.ema_dense_ns,
                cache.ema_dense_long_ns,
                cache.ama_dense_cold_ns,
                cache.ama_dense_warm_ns,
                cache.ama_dense_warm_count,
            ),
        };

        let base = match (short > 0.0, long > 0.0) {
            (true, true) => blend * long + (1.0 - blend) * short,
            (true, false) => short,
            (false, true) => long,
            (false, false) => {
                // No measurement yet: use a cold-start prior. Cached is biased
                // by its configured cost, Sparse gets a neutral unit prior, and
                // Dense is treated as worst (INFINITY) so the selector prefers to
                // probe the cheaper candidates first rather than commit to the
                // full dense matmul before it has any data.
                return match path {
                    InferencePath::Cached => self.config.cache_cost_bias.max(0.0),
                    InferencePath::Sparse => 1.0,
                    InferencePath::Dense => f64::INFINITY,
                };
            }
        };

        let warmness = Self::ama_warmness(warm_count, self.ama_warmup_samples());
        let cold_penalty = if cold > 0.0 && warm > 0.0 {
            (1.0 - warmness) * (cold - warm).max(0.0)
        } else {
            (1.0 - warmness) * self.ama_selector_overhead(cache)
        };
        let mut score = base + cold_penalty + self.ama_selector_overhead(cache);
        if path == InferencePath::Cached {
            let bias = if self.config.cache_adapt_rate > 0.0 {
                cache.adaptive_bias
            } else {
                self.config.cache_cost_bias
            };
            score *= bias.max(0.0);
        }
        score
    }

    fn ama_switch_margin(&self, cache: &AchfCache, prev_score: f64, next_score: f64) -> f64 {
        let overhead = self.ama_selector_overhead(cache);
        let scale = prev_score.max(next_score).max(1.0);
        overhead.max(scale * 0.05)
    }

    fn ama_selector_overhead(&self, cache: &AchfCache) -> f64 {
        if cache.decision_ema_ns > 0.0 {
            cache.decision_ema_ns
        } else {
            0.0
        }
    }

    fn ama_warmup_samples(&self) -> u64 {
        self.config.cache_min_reuse.max(1) as u64
    }

    fn ama_min_dwell(&self) -> u64 {
        self.config.cache_min_reuse.max(1) as u64
    }

    fn ama_stale_limit(&self) -> u64 {
        let sample_every = self.config.cache_latency_sample_every.max(1);
        (sample_every * 4).max(self.ama_warmup_samples() * 2).max(8)
    }

    fn ama_warmness(count: u64, warmup_samples: u64) -> f64 {
        if warmup_samples == 0 {
            1.0
        } else {
            (count as f64 / warmup_samples as f64).clamp(0.0, 1.0)
        }
    }

    fn consume_forced_latency_sample(&self) -> bool {
        let Ok(mut cache) = self.cache.try_write() else {
            return false;
        };
        let forced = cache.ama_force_latency_sample;
        cache.ama_force_latency_sample = false;
        forced
    }

    #[allow(dead_code)]
    fn should_use_cache(&self, x: &[f32]) -> (bool, bool, bool) {
        let decision = self.select_ama_path(x);
        (
            decision.path == InferencePath::Cached,
            decision.cache_skipped,
            decision.has_cache,
        )
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

    fn update_ema(current: &mut f64, sample: f64, momentum: f64) {
        if *current == 0.0 || momentum <= 0.0 {
            *current = sample;
        } else {
            *current = momentum * *current + (1.0 - momentum) * sample;
        }
    }

    fn record_ama_latency(
        cache: &mut AchfCache,
        path: InferencePath,
        elapsed_ns: f64,
        warmup_samples: u64,
        ema: f64,
    ) {
        match path {
            InferencePath::Cached => {
                if cache.ama_cached_warm_count < warmup_samples {
                    Self::update_ema(&mut cache.ama_cached_cold_ns, elapsed_ns, ema);
                } else {
                    Self::update_ema(&mut cache.ama_cached_warm_ns, elapsed_ns, ema);
                }
                cache.ama_cached_warm_count = cache.ama_cached_warm_count.saturating_add(1);
            }
            InferencePath::Sparse => {
                if cache.ama_sparse_warm_count < warmup_samples {
                    Self::update_ema(&mut cache.ama_sparse_cold_ns, elapsed_ns, ema);
                } else {
                    Self::update_ema(&mut cache.ama_sparse_warm_ns, elapsed_ns, ema);
                }
                cache.ama_sparse_warm_count = cache.ama_sparse_warm_count.saturating_add(1);
            }
            InferencePath::Dense => {
                if cache.ama_dense_warm_count < warmup_samples {
                    Self::update_ema(&mut cache.ama_dense_cold_ns, elapsed_ns, ema);
                } else {
                    Self::update_ema(&mut cache.ama_dense_warm_ns, elapsed_ns, ema);
                }
                cache.ama_dense_warm_count = cache.ama_dense_warm_count.saturating_add(1);
            }
        }
    }

    fn record_path_latency(&self, path: InferencePath, elapsed_ns: f64, num_rows: usize) {
        if elapsed_ns <= 0.0 {
            return;
        }
        let Ok(mut cache) = self.cache.try_write() else {
            return;
        };
        // Record into the same batch bucket the selector chose for this call.
        cache.switch_ama_bucket(num_rows, self.config.cache_cost_bias);
        let ema = self.config.cache_latency_ema;
        let ema_long = self.config.cache_latency_long_ema;
        Self::record_ama_latency(&mut cache, path, elapsed_ns, self.ama_warmup_samples(), ema);
        match path {
            InferencePath::Cached => {
                Self::update_ema(&mut cache.ema_cached_ns, elapsed_ns, ema);
                Self::update_ema(&mut cache.ema_cached_long_ns, elapsed_ns, ema_long);
            }
            InferencePath::Sparse => {
                Self::update_ema(&mut cache.ema_sparse_ns, elapsed_ns, ema);
                Self::update_ema(&mut cache.ema_sparse_long_ns, elapsed_ns, ema_long);
            }
            InferencePath::Dense => {
                Self::update_ema(&mut cache.ema_dense_ns, elapsed_ns, ema);
                Self::update_ema(&mut cache.ema_dense_long_ns, elapsed_ns, ema_long);
            }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InferencePath {
    Cached,
    Sparse,
    Dense,
}

#[derive(Clone, Copy, Debug)]
struct AmaPathDecision {
    path: InferencePath,
    has_cache: bool,
    cache_skipped: bool,
}

/// Truncated rank-r approximation of a row-major `rows x cols` matrix via
/// randomized subspace (power) iteration — a dependency-free truncated SVD.
///
/// Writes the rank-r approximation back into `w` and returns the relative
/// Frobenius error `||W - W_r||_F / ||W||_F`. When `rank == 0` or
/// `rank >= min(rows, cols)` the matrix is left untouched and 0.0 is returned.
///
/// Algorithm: sample a Gaussian test matrix `Omega (cols x r)`, form
/// `Y = W * Omega`, orthonormalize, then run a few power iterations
/// (`Z = W^T Y`, `Y = W Z`, re-orthonormalizing each time) so the columns of
/// `Y` converge to the dominant r-dimensional left-singular subspace.
/// The approximation is `W_r = Y (Y^T W)`. Three iterations are ample for the
/// small operators used here (Halko, Martinsson & Tropp, 2011).
pub(crate) fn low_rank_truncate(
    w: &mut [f32],
    rows: usize,
    cols: usize,
    rank: usize,
    seed: u64,
) -> f64 {
    if rank == 0 || rows == 0 || cols == 0 || rank >= rows.min(cols) || w.len() != rows * cols {
        return 0.0;
    }
    const POWER_ITERATIONS: usize = 3;
    let r = rank;
    let wf: Vec<f64> = w.iter().map(|&v| v as f64).collect();

    // Gaussian test matrix Omega: cols x r (row-major).
    let mut rng = crate::rng::Rng::from_seed(seed);
    let mut omega = vec![0.0f64; cols * r];
    for v in omega.iter_mut() {
        *v = rng.next_f64_normal();
    }

    // Y = W * Omega -> rows x r.
    let mut y = vec![0.0f64; rows * r];
    matmul_rowmajor(&wf, &omega, &mut y, rows, cols, r);
    orthonormalize_columns(&mut y, rows, r);

    // Subspace (power) iteration: Z = W^T Y, Y = W Z.
    let mut z = vec![0.0f64; cols * r];
    for _ in 0..POWER_ITERATIONS {
        matmul_transposed_lhs(&wf, &y, &mut z, rows, cols, r);
        orthonormalize_columns(&mut z, cols, r);
        matmul_rowmajor(&wf, &z, &mut y, rows, cols, r);
        orthonormalize_columns(&mut y, rows, r);
    }

    // B = Y^T W -> r x cols, then W_r = Y * B.
    let mut b = vec![0.0f64; r * cols];
    for j in 0..r {
        for c in 0..cols {
            let mut sum = 0.0;
            for i in 0..rows {
                sum += y[i * r + j] * wf[i * cols + c];
            }
            b[j * cols + c] = sum;
        }
    }
    let mut w_r = vec![0.0f64; rows * cols];
    matmul_rowmajor(&y, &b, &mut w_r, rows, r, cols);

    let mut err_sq = 0.0f64;
    let mut norm_sq = 0.0f64;
    for (orig, approx) in wf.iter().zip(w_r.iter()) {
        let d = orig - approx;
        err_sq += d * d;
        norm_sq += orig * orig;
    }
    for (dst, &src) in w.iter_mut().zip(w_r.iter()) {
        *dst = src as f32;
    }
    if norm_sq > 0.0 {
        (err_sq / norm_sq).sqrt()
    } else {
        0.0
    }
}

/// C = A * B with row-major A (m x k), B (k x n), C (m x n).
fn matmul_rowmajor(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 {
                continue;
            }
            let b_row = &b[p * n..(p + 1) * n];
            let c_row = &mut c[i * n..(i + 1) * n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *cv += av * bv;
            }
        }
    }
}

/// C = A^T * B with row-major A (m x k), B (m x n), C (k x n).
fn matmul_transposed_lhs(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let b_row = &b[i * n..(i + 1) * n];
        for (p, &av) in a_row.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            let c_row = &mut c[p * n..(p + 1) * n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *cv += av * bv;
            }
        }
    }
}

/// Modified Gram-Schmidt over the columns of a row-major `rows x cols` matrix.
fn orthonormalize_columns(m: &mut [f64], rows: usize, cols: usize) {
    for j in 0..cols {
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..rows {
                dot += m[i * cols + j] * m[i * cols + k];
            }
            for i in 0..rows {
                m[i * cols + j] -= dot * m[i * cols + k];
            }
        }
        let mut norm_sq = 0.0;
        for i in 0..rows {
            let v = m[i * cols + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for i in 0..rows {
                m[i * cols + j] /= norm;
            }
        } else {
            for i in 0..rows {
                m[i * cols + j] = 0.0;
            }
        }
    }
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

pub(crate) fn sync_weight_from_host_f32(dst: &Tensor, data: &[f32]) {
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
        Dtype::I8 => panic!("sync_weight_from_host_f32 does not support I8 tensors"),
    }
    #[cfg(cuda)]
    if dst.device == Device::Cuda {
        dst.cuda_remove_cached_buffer();
        let _ = dst.cuda_get_or_upload_buffer();
        dst.cuda_clear_host_data_preserve_cache();
    }
}

#[allow(dead_code)]
fn write_tensor_from_f32(dst: &Tensor, data: &[f32]) {
    sync_weight_from_host_f32(dst, data);
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
    fn achf_training_gate_scales_the_forward_output() {
        let cfg = AchfConfig {
            enabled: true,
            proj_mode: "none".to_string(),
            proj_freq: 0,
            gate_warmup_steps: 0,
            gate_transition_steps: 0,
            gate_alpha: 0.0,
            gate_beta: 0.0,
            g_min: 0.0,
            ..Default::default()
        };
        let layer = AchfLayer::new_square(4, cfg, 99);
        let input = Tensor::rand(vec![2, 4], -0.1, 0.1, 100);
        let expected = layer.weight.forward(&input).data_to_f32_vec();
        let actual = layer.forward(&input).data_to_f32_vec();
        assert!((layer.last_gate() - 0.5).abs() < 1e-12);
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected * 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn achf_diagnostics_report_gradient_cosine() {
        let cfg = AchfConfig {
            enabled: true,
            diagnostics_enabled: true,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(2, cfg, 101);
        layer.weight.weight.grad_write_compat().fill(1.0);
        layer.update_after_backward();
        assert_eq!(layer.snapshot_state().gradient_cosine, 0.0);

        layer.weight.weight.zero_grad();
        layer.weight.weight.grad_write_compat().fill(-1.0);
        layer.update_after_backward();
        let cosine = layer.snapshot_state().gradient_cosine;
        assert!((cosine + 1.0).abs() < 1e-6, "cosine={cosine}");

        layer.freeze_for_inference();
        assert!(layer.state.read().unwrap().previous_gradient.is_empty());
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
        layer.maybe_project_after_optimizer_step();
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
        layer.maybe_project_after_optimizer_step();
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
        layer.maybe_project_after_optimizer_step();
        layer.freeze_for_inference();
        let w_before = layer.weight.weight.data_f32().clone();
        let x_data = x.data_to_f32_vec();
        let _ = layer.forward_inference_residual(&x_data);
        let w_after = layer.weight.weight.data_f32().clone();
        assert_eq!(w_before, w_after);
    }

    #[test]
    fn achf_inference_does_not_advance_projection() {
        let cfg = AchfConfig {
            enabled: true,
            proj_freq: 1,
            proj_mode: "rowcol".to_string(),
            ..Default::default()
        };
        let layer = AchfLayer::new_square(4, cfg, 13);
        let x_data = vec![0.1, -0.2, 0.3, -0.4];
        let step_before = layer.state.read().unwrap().step;
        let w_before = layer.weight.weight.data_f32().clone();

        let _ = layer.forward_inference_residual(&x_data);

        let step_after = layer.state.read().unwrap().step;
        let w_after = layer.weight.weight.data_f32().clone();
        assert_eq!(step_before, step_after);
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
    fn achf_freeze_reports_pruning_discrepancy_and_materialized_memory() {
        let cfg = AchfConfig {
            enabled: true,
            proj_mode: "none".to_string(),
            proj_freq: 0,
            rank: 0,
            prune_threshold: 0.5,
            infer_gate: "one".to_string(),
            ..Default::default()
        };
        let mut layer = AchfLayer::new(2, 2, false, cfg, 31);
        {
            let mut weights = layer.weight.weight.data_write_f32();
            *weights = vec![0.1, 1.0, -0.2, -2.0];
        }
        layer.freeze_for_inference();

        let snapshot = layer.snapshot_state();
        assert!(snapshot.prune_rel_err > 0.0);
        let memory = layer.memory_stats();
        assert_eq!(memory.layers, 1);
        assert_eq!(memory.candidate_total_weights, 4);
        assert_eq!(memory.candidate_nonzero_weights, 2);
        assert_eq!(
            memory.total_materialized_bytes,
            memory.reference_parameter_bytes
                + memory.candidate_dense_bytes
                + memory.sparse_mask_bytes
                + memory.cached_dense_bytes
                + memory.cached_bias_bytes
                + memory.csr_row_ptr_bytes
                + memory.csr_column_bytes
                + memory.csr_value_bytes
                + memory.memoized_output_bytes
                + memory.sinkhorn_state_bytes
        );
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
    fn achf_full_mode_runs_ama_on_frozen_layer() {
        // In full mode, a frozen layer must keep exercising the
        // latency-driven AMA selector: latency samples must accumulate (proving
        // the machinery is live), whereas the default frozen fast path records
        // none. This is the core fix — otherwise the AMA code is unreachable.
        let cfg = AchfConfig {
            enabled: true,
            mode: "full".to_string(),
            adaptive_inference: false,
            cache_latency_sample_every: 1,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(4, 3, true, cfg, 55);
        layer.prune(0.01);
        layer.freeze_for_inference();

        let x = vec![0.3, -0.7, 0.9, -0.1];
        for _ in 0..64 {
            let _ = layer.forward_inference_residual(&x);
        }
        let stats = layer.cache_stats();
        assert!(
            stats.latency_samples > 0,
            "full mode must produce latency samples, got {}",
            stats.latency_samples
        );

        // Control: the same frozen layer in lite mode records no AMA samples.
        let cfg2 = AchfConfig {
            enabled: true,
            mode: "lite".to_string(),
            adaptive_inference: false,
            ..Default::default()
        };
        let mut frozen = AchfLayer::new(4, 3, true, cfg2, 55);
        frozen.prune(0.01);
        frozen.freeze_for_inference();
        for _ in 0..64 {
            let _ = frozen.forward_inference_residual(&x);
        }
        assert_eq!(
            frozen.cache_stats().latency_samples,
            0,
            "lite mode's deterministic frozen path must not sample latency"
        );
    }

    #[test]
    fn achf_adaptive_selector_converges_to_fastest_path() {
        // Regression for the selector bug where dense latency was never recorded
        // (ema_dense_ns stuck at 0), which forced perpetual probing and pinned
        // selection to dense ~90% of the time even when Cached was fastest.
        // Here we build a low-sparsity layer so the CSR sparse path is the WORST
        // (scatter writes, no SIMD) and the fused Cached path is fastest. After
        // warmup the adaptive selector must pick Cached for the clear majority
        // of calls — NOT dense. This asserts the fix end to end.
        let cfg = AchfConfig {
            enabled: true,
            adaptive_inference: true,
            cache_latency_sample_every: 1,
            gate_warmup_steps: 0,
            gate_transition_steps: 0,
            g_min: 0.0,
            infer_gate: "one".to_string(),
            prune_threshold: 0.0,
            cache_min_reuse: 8,
            ..Default::default()
        };
        // 256-wide, dense weights (no zeros) => sparse path is strictly worse.
        let dim = 256usize;
        let mut layer = AchfLayer::new(dim, dim, false, cfg, 909);
        {
            let mut w = layer.weight.weight.data_write_f32();
            for (i, v) in w.iter_mut().enumerate() {
                // All nonzero, moderate magnitude: nothing gets pruned at
                // threshold 0, so CSR is fully dense => sparse path is slowest.
                *v = (((i % 13) as f32) - 6.0) * 0.05 + 0.01;
            }
        }
        layer.freeze_for_inference();

        // Dense (nonzero) input so cached's zero-input skip doesn't distort.
        let x: Vec<f32> = (0..dim).map(|i| ((i % 5) as f32) * 0.2 + 0.1).collect();

        // Warm up all three paths, then measure the selected distribution.
        for _ in 0..200 {
            let _ = layer.forward_inference_residual(&x);
        }
        let before = layer.cache_stats();
        for _ in 0..2000 {
            let _ = layer.forward_inference_residual(&x);
        }
        let after = layer.cache_stats();

        let cached = (after.cache_hits - before.cache_hits) as f64;
        let dense = (after.dense_paths - before.dense_paths) as f64;
        let sparse = (after.sparse_paths - before.sparse_paths) as f64;
        let total = cached + dense + sparse;
        assert!(total > 0.0, "no path selections recorded");
        let cached_frac = cached / total;
        // The pre-fix code sat around cached ~5% / dense ~90%. Post-fix, Cached
        // must dominate. We assert a conservative majority to stay robust to
        // periodic re-probing of the other paths.
        assert!(
            cached_frac > 0.6,
            "adaptive selector should converge to Cached (fastest); got \
             cached={cached_frac:.2} dense={:.2} sparse={:.2}",
            dense / total,
            sparse / total
        );
    }

    #[test]
    // The batch-dependent latency crossover this asserts only manifests in
    // optimized builds (debug leaves the cached SIMD and CSR skip un-vectorized,
    // collapsing the margin). Runs under `cargo test --release`; shown as
    // ignored in debug rather than silently compiled out.
    #[cfg_attr(debug_assertions, ignore)]
    fn achf_adaptive_selector_adapts_across_batch_regimes() {
        // True cross-regime adaptation: on ONE fixed frozen layer, a small batch
        // (batch=1) should favor the CSR sparse path (fewer FLOPs win when the
        // per-row scatter cost isn't amortized), while a large batch (batch=64)
        // should favor the fused Cached path (contiguous SIMD amortizes). This
        // is only possible because latency EMAs are keyed by batch bucket; with
        // global EMAs the two regimes blended and the selector thrashed. We
        // assert the DIRECTION (batch=1 picks sparse more than batch=64 does),
        // which is robust to absolute timing noise across machines.
        let dim = 1024usize;
        // weight_sparsity ~0.9 sits between the batch=1 and batch=64 crossover,
        // so the two buckets settle on opposite paths.
        let weight_sparsity = 0.9f32;
        let cfg = AchfConfig {
            enabled: true,
            adaptive_inference: true,
            cache_latency_sample_every: 1,
            gate_warmup_steps: 0,
            gate_transition_steps: 0,
            g_min: 0.0,
            infer_gate: "one".to_string(),
            prune_threshold: 0.0,
            cache_min_reuse: 0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(dim, dim, false, cfg, 77);
        {
            let mut w = layer.weight.weight.data_write_f32();
            let zero_per_row = (dim as f32 * weight_sparsity) as usize;
            for r in 0..dim {
                for c in 0..dim {
                    let v = &mut w[r * dim + c];
                    if c < zero_per_row {
                        *v = 0.0;
                    } else if *v == 0.0 {
                        *v = 0.01;
                    }
                }
            }
        }
        layer.freeze_for_inference();

        let x1: Vec<f32> = (0..dim).map(|i| ((i % 7) as f32) * 0.1 + 0.05).collect();
        let x64: Vec<f32> = (0..dim * 64)
            .map(|i| ((i % 7) as f32) * 0.1 + 0.05)
            .collect();
        let run = |x: &[f32], n: usize| {
            for _ in 0..n {
                let _ = layer.forward_inference_residual(x);
            }
        };
        // Warm both buckets first, then measure settled selection per regime.
        run(&x1, 300);
        run(&x64, 300);

        let b1 = layer.cache_stats();
        run(&x1, 600);
        let a1 = layer.cache_stats();
        let sparse_frac_b1 = (a1.sparse_paths - b1.sparse_paths) as f64 / 600.0;

        let b64 = layer.cache_stats();
        run(&x64, 600);
        let a64 = layer.cache_stats();
        let sparse_frac_b64 = (a64.sparse_paths - b64.sparse_paths) as f64 / 600.0;

        assert!(
            sparse_frac_b1 > sparse_frac_b64 + 0.2,
            "expected batch=1 to select sparse markedly more than batch=64 \
             (adaptation across regimes); got sparse_frac batch1={sparse_frac_b1:.2} \
             batch64={sparse_frac_b64:.2}"
        );
    }

    #[test]
    fn achf_residual_add_into_matches_residual() {
        // The fused frozen add-into path must equal `forward_inference_residual`
        // added onto an existing accumulator, element for element.
        let cfg = AchfConfig {
            enabled: true,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(3, 2, true, cfg, 77);
        layer.prune(0.01);
        layer.freeze_for_inference();

        let x = vec![0.7, -1.3, 0.2];
        let base = vec![10.0f32, -4.0];

        let residual = layer.forward_inference_residual(&x);
        let mut expected = base.clone();
        for (e, r) in expected.iter_mut().zip(&residual) {
            *e += *r;
        }

        let mut got = base.clone();
        layer.forward_inference_residual_add_into(&x, &mut got);

        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "add_into {g} vs expected {e}");
        }
    }

    #[test]
    fn achf_frozen_fast_path_matches_cached_path() {
        // On a frozen layer, the fast path (`forward_inference_residual` taking
        // the frozen branch) must equal the explicitly forced Cached path — same
        // weights, same operator, so the optimization changes nothing numerically.
        let cfg = AchfConfig {
            enabled: true,
            ..Default::default()
        };
        let mut frozen = AchfLayer::new(4, 3, true, cfg, 91);
        frozen.prune(0.01);
        frozen.freeze_for_inference();
        assert!(frozen.metrics.frozen.load(Ordering::Relaxed));

        let x = vec![0.4, -0.9, 1.1, -0.2];
        let fast_out = frozen.forward_inference_residual(&x);
        let forced_cached = frozen.forward_inference_forced_path(&x, 0);

        assert_eq!(fast_out.len(), forced_cached.len());
        for (f, c) in fast_out.iter().zip(&forced_cached) {
            assert!((f - c).abs() < 1e-5, "fast {f} vs forced-cached {c}");
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
        layer.record_path_latency(InferencePath::Sparse, 100.0, 1);
        layer.record_path_latency(InferencePath::Cached, 50.0, 1);
        let cache = layer.cache.read().unwrap();
        assert!(cache.adaptive_bias < 1.0);
    }

    #[test]
    fn achf_ama_steady_state_exploration_tax_is_bounded() {
        // ROOT-CAUSE PROBE: with a clear winner already warmed, how large is
        // the forced re-probe tax in steady state? Seed sparse as the obvious
        // best (10ns vs cached 100ns, dense 200ns), fully warmed, then drive the
        // selector many times WITHOUT recording new latencies (fixed input =
        // one bucket, EMAs frozen at the seeded values). The winner's selection
        // fraction is then determined purely by the probe cadence.
        let cfg = AchfConfig {
            enabled: true,
            cache_min_reuse: 1,
            cache_latency_sample_every: 1,
            cache_cost_bias: 1.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 91);
        layer.prune(0.01);
        layer.freeze_for_inference();
        {
            let mut cache = layer.cache.write().unwrap();
            cache.ema_sparse_ns = 10.0;
            cache.ema_sparse_long_ns = 10.0;
            cache.ema_cached_ns = 100.0;
            cache.ema_cached_long_ns = 100.0;
            cache.ema_dense_ns = 200.0;
            cache.ema_dense_long_ns = 200.0;
            cache.ama_sparse_warm_count = 100;
            cache.ama_cached_warm_count = 100;
            cache.ama_dense_warm_count = 100;
            cache.ama_prev_path = Some(InferencePath::Sparse);
            cache.ama_dwell = 100;
        }
        let x = vec![0.1f32, -0.2, 0.3, 0.4];
        let n = 900usize;
        let mut sparse = 0usize;
        for _ in 0..n {
            if layer.select_ama_path(&x).path == InferencePath::Sparse {
                sparse += 1;
            }
        }
        let frac = sparse as f64 / n as f64;
        println!("steady-state sparse fraction (clear 10x winner) = {frac:.3}");
        // With exploration backoff, a clear 10x winner is now exploited the
        // overwhelming majority of the time: the two losers (10x and 20x worse)
        // are re-probed on stretched intervals, not the fixed base cadence. The
        // old fixed-cadence code pinned this at ~0.60; anything below ~0.9 means
        // the backoff or the probe/dwell decoupling regressed.
        assert!(
            frac >= 0.9,
            "expected clear winner exploited >=90% with backoff, got {frac:.3}"
        );
    }

    #[test]
    fn achf_ama_backoff_still_detects_regime_shift() {
        // SAFETY OF THE FIX: exploration backoff must not blind the selector to
        // a loser that becomes the winner. Seed sparse as the settled winner,
        // let it back off cached/dense, then flip the regime — make cached the
        // clear winner by rewriting its EMA (as a real latency sample would) on
        // every call — and assert the selector re-probes cached, discovers the
        // shift, and commits to it within a bounded number of calls.
        let cfg = AchfConfig {
            enabled: true,
            cache_min_reuse: 1,
            cache_latency_sample_every: 1,
            cache_cost_bias: 1.0,
            cache_adapt_rate: 0.0,
            cache_adapt_blend: 0.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 92);
        layer.prune(0.01);
        layer.freeze_for_inference();
        {
            let mut cache = layer.cache.write().unwrap();
            cache.ema_sparse_ns = 10.0;
            cache.ema_sparse_long_ns = 10.0;
            cache.ema_cached_ns = 100.0;
            cache.ema_cached_long_ns = 100.0;
            cache.ema_dense_ns = 200.0;
            cache.ema_dense_long_ns = 200.0;
            cache.ama_sparse_warm_count = 100;
            cache.ama_cached_warm_count = 100;
            cache.ama_dense_warm_count = 100;
            cache.ama_prev_path = Some(InferencePath::Sparse);
            cache.ama_dwell = 100;
        }
        let x = vec![0.1f32, -0.2, 0.3, 0.4];
        // True per-path latency the hardware would report RIGHT NOW. Before the
        // flip sparse is fastest; after it cached is the clear 10x winner. The
        // selector only learns a path's latency when it actually runs (probes)
        // that path, exactly like production.
        let true_ns = |path: InferencePath, flipped: bool| -> f64 {
            match (path, flipped) {
                (InferencePath::Sparse, false) => 10.0,
                (InferencePath::Cached, false) => 100.0,
                (InferencePath::Cached, true) => 10.0,
                (InferencePath::Sparse, true) => 100.0,
                (InferencePath::Dense, _) => 200.0,
            }
        };
        // Phase 1: drive real selections + measurements so sparse settles and
        // the losers' re-probe intervals stretch out.
        for _ in 0..200 {
            let path = layer.select_ama_path(&x).path;
            layer.record_path_latency(path, true_ns(path, false), 1);
        }
        // Phase 2: regime flips. Crucially we do NOT touch any EMA directly —
        // the selector must DISCOVER the shift by re-probing cached on its
        // (stretched) interval and measuring the new latency itself. This is the
        // real test of whether backoff blinds the selector to a shift.
        let mut calls_to_commit = None;
        let mut cached_streak = 0u32;
        for i in 0..600 {
            let path = layer.select_ama_path(&x).path;
            layer.record_path_latency(path, true_ns(path, true), 1);
            // "Committed" = the selector is now consistently exploiting cached,
            // not just probing it once. Require a short run to rule out a lone
            // probe being mistaken for detection.
            if path == InferencePath::Cached {
                cached_streak += 1;
                if cached_streak >= 5 && calls_to_commit.is_none() {
                    calls_to_commit = Some(i + 1);
                    break;
                }
            } else {
                cached_streak = 0;
            }
        }
        let calls = calls_to_commit.expect("selector never committed to cached after regime shift");
        println!("regime shift discovered + committed to cached in {calls} calls");
        // Worst-case re-probe interval is AMA_MAX_REPROBE_MULT * base = 16*8 =
        // 128; add the EMA settle + dwell/margin. Must be far below "never".
        assert!(
            calls <= 200,
            "backoff delayed regime-shift detection too long: {calls} calls"
        );
    }

    #[test]
    fn achf_ama_records_cold_then_warm_latency() {
        let cfg = AchfConfig {
            enabled: true,
            cache_min_reuse: 1,
            cache_latency_ema: 0.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 72);
        layer.prune(0.01);
        layer.freeze_for_inference();

        layer.record_path_latency(InferencePath::Cached, 120.0, 1);
        layer.record_path_latency(InferencePath::Cached, 40.0, 1);

        let cache = layer.cache.read().unwrap();
        assert_eq!(cache.ama_cached_warm_count, 2);
        assert_eq!(cache.ama_cached_cold_ns, 120.0);
        assert_eq!(cache.ama_cached_warm_ns, 40.0);
    }

    #[test]
    fn achf_ama_stale_path_probe_forces_latency_sample() {
        let cfg = AchfConfig {
            enabled: true,
            cache_min_reuse: 1,
            cache_latency_sample_every: 16,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 73);
        layer.prune(0.01);
        layer.freeze_for_inference();
        let x_data = vec![0.1, -0.2, 0.3, 0.4];
        {
            let mut cache = layer.cache.write().unwrap();
            cache.ema_cached_ns = 100.0;
            cache.ema_sparse_ns = 100.0;
            cache.ama_cached_warm_count = 1;
            cache.ama_sparse_warm_count = 1;
            cache.ama_prev_path = Some(InferencePath::Cached);
            cache.ama_dwell = 16;
            cache.ama_sparse_stale = layer.ama_stale_limit();
        }

        let decision = layer.select_ama_path(&x_data);

        assert_eq!(decision.path, InferencePath::Sparse);
        let cache = layer.cache.read().unwrap();
        assert_eq!(cache.ama_sparse_stale, 0);
        assert_eq!(cache.ama_probes, 1);
        assert!(cache.ama_force_latency_sample);
    }

    #[test]
    fn achf_ama_hysteresis_keeps_previous_path_inside_margin() {
        let cfg = AchfConfig {
            enabled: true,
            cache_min_reuse: 1,
            cache_cost_bias: 1.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 74);
        layer.prune(0.01);
        layer.freeze_for_inference();
        let x_data = vec![0.1, -0.2, 0.3, 0.4];
        {
            let mut cache = layer.cache.write().unwrap();
            cache.ema_cached_ns = 100.0;
            cache.ema_sparse_ns = 98.0;
            // Warm all three arms: hysteresis only applies once every arm has a
            // latency estimate, otherwise the un-probed arm forces a probe and
            // short-circuits the score/margin comparison this test exercises.
            cache.ema_dense_ns = 200.0;
            cache.ama_cached_warm_count = 1;
            cache.ama_sparse_warm_count = 1;
            cache.ama_dense_warm_count = 1;
            cache.ama_prev_path = Some(InferencePath::Cached);
            cache.ama_dwell = 16;
        }

        let decision = layer.select_ama_path(&x_data);

        assert_eq!(decision.path, InferencePath::Cached);
        let cache = layer.cache.read().unwrap();
        assert_eq!(cache.ama_switches, 0);
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

    #[test]
    fn achf_inference_paths_match_on_pruned_operator() {
        // The CSR SpMV path (forced_path=1) scatters only surviving weights,
        // Cached (forced_path=0) and Dense (forced_path=2) both execute the
        // materialized pruned weight densely. After freeze all three paths
        // operate on the SAME pruned weight with the SAME gate, so their outputs
        // MUST be bit-close.
        // This is the numerical anchor for the scatter-write kernel — the old
        // test only checked shapes and would not have caught an indexing bug.
        let cfg = AchfConfig {
            enabled: true,
            gate_warmup_steps: 0,
            gate_transition_steps: 0,
            g_min: 0.0,
            infer_gate: "one".to_string(),
            prune_threshold: 0.15,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(6, cfg, 4242);
        {
            let mut w = layer.weight.weight.data_write_f32();
            for (i, v) in w.iter_mut().enumerate() {
                // Mix of large and small magnitudes so pruning actually zeros
                // some entries and the CSR view is genuinely sparse.
                *v = (((i * 7) % 11) as f32 - 5.0) * 0.08;
            }
        }
        layer.freeze_for_inference();

        // Two input rows, including zeros to exercise the scale==0 skip.
        let x = vec![
            0.3, -0.2, 0.0, 0.5, -0.4, 0.1, // row 0
            0.0, 0.25, -0.15, 0.0, 0.35, -0.05, // row 1
        ];
        let cached = layer.forward_inference_forced_path(&x, 0);
        let sparse = layer.forward_inference_forced_path(&x, 1);
        let dense = layer.forward_inference_forced_path(&x, 2);
        assert_eq!(cached.len(), sparse.len());
        assert_eq!(cached.len(), dense.len());
        for ((c, s), d) in cached.iter().zip(sparse.iter()).zip(dense.iter()) {
            assert!(
                (c - s).abs() < 1e-5,
                "CSR sparse output {s} diverged from cached dense-pruned {c}"
            );
            assert!(
                (c - d).abs() < 1e-5,
                "ordinary dense output {d} diverged from cached dense-pruned {c}"
            );
        }
        // Sanity: the CSR view must actually be sparser than dense (pruning
        // removed entries), otherwise the test proves nothing about skipping.
        let cache = layer.cache.read().unwrap();
        let nnz = cache.csr_vals.as_ref().expect("csr built on freeze").len();
        let dense_entries = cache.in_dim * cache.out_dim;
        assert!(
            nnz < dense_entries,
            "expected CSR nnz {nnz} < dense entries {dense_entries}"
        );
    }

    // Confirms the adaptive selector actually SWITCHES paths when a phased
    // workload changes regime (batch=1 decode-like vs batch=64 prefill-like) on
    // ONE fixed frozen layer. This is the crux of the "true adaptive" claim.
    // Run with:
    //   cargo test --release --bin talos_xii adaptive_switch_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn adaptive_switch_probe() {
        let dim = 1024usize;
        let make_x = |batch: usize| -> Vec<f32> {
            (0..dim * batch)
                .map(|i| ((i % 7) as f32) * 0.1 + 0.05)
                .collect()
        };
        let phase = |layer: &AchfLayer, batch: usize, n: usize| -> (u64, u64, u64) {
            let x = make_x(batch);
            let before = layer.cache_stats();
            for _ in 0..n {
                let _ = layer.forward_inference_residual(&x);
            }
            let after = layer.cache_stats();
            (
                after.cache_hits - before.cache_hits,
                after.sparse_paths - before.sparse_paths,
                after.dense_paths - before.dense_paths,
            )
        };
        let winner = |c: u64, s: u64, d: u64| -> &'static str {
            if s >= c && s >= d {
                "SPARSE"
            } else if c >= d {
                "cached"
            } else {
                "dense"
            }
        };

        println!(
            "\n[bucketed] dim={dim}, adaptive on; do batch=1 and batch=64 pick DIFFERENT paths?"
        );
        for &weight_sparsity in &[0.8f32, 0.9, 0.95, 0.98] {
            let cfg = AchfConfig {
                enabled: true,
                adaptive_inference: true,
                cache_latency_sample_every: 1,
                gate_warmup_steps: 0,
                gate_transition_steps: 0,
                g_min: 0.0,
                infer_gate: "one".to_string(),
                prune_threshold: 0.0,
                cache_min_reuse: 0, // force every call through the selector
                ..Default::default()
            };
            let mut layer = AchfLayer::new(dim, dim, false, cfg, 77);
            {
                let mut w = layer.weight.weight.data_write_f32();
                let zero_per_row = (dim as f32 * weight_sparsity) as usize;
                for r in 0..dim {
                    for c in 0..dim {
                        let v = &mut w[r * dim + c];
                        if c < zero_per_row {
                            *v = 0.0;
                        } else if *v == 0.0 {
                            *v = 0.01;
                        }
                    }
                }
            }
            layer.freeze_for_inference();

            // Warm each bucket, then measure the settled selection.
            phase(&layer, 1, 300);
            phase(&layer, 64, 300);
            let (c1, s1, d1) = phase(&layer, 1, 600);
            let (c64, s64, d64) = phase(&layer, 64, 600);
            println!(
                "  wsp={weight_sparsity}: batch=1 -> {} (c={c1} s={s1} d={d1}) | \
                 batch=64 -> {} (c={c64} s={s64} d={d64})",
                winner(c1, s1, d1),
                winner(c64, s64, d64)
            );
        }
    }

    // Premise probe for "true adaptive": at a FIXED frozen layer (fixed weight
    // sparsity), does varying INPUT sparsity or BATCH size flip which path is
    // fastest? If it does, per-call adaptation is justified. If not, the honest
    // adaptive story is per-layer/per-hardware convergence, not mid-run flips.
    // Run with:
    //   cargo test --release --bin talos_xii adaptive_premise_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn adaptive_premise_probe() {
        use std::time::Instant;
        let dim = 1024usize; // near the cached/sparse crossover band
        let weight_sparsity = 0.92f32;
        let input_sparsities = [0.0f32, 0.5, 0.9];
        let batches = [1usize, 8, 64];
        let warmup = 20usize;
        let iters = 200usize;

        // One fixed frozen layer for the whole sweep.
        let cfg = AchfConfig {
            enabled: true,
            gate_warmup_steps: 0,
            gate_transition_steps: 0,
            g_min: 0.0,
            infer_gate: "one".to_string(),
            prune_threshold: 0.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(dim, dim, false, cfg, 1234);
        {
            let mut w = layer.weight.weight.data_write_f32();
            let zero_per_row = (dim as f32 * weight_sparsity) as usize;
            for r in 0..dim {
                for c in 0..dim {
                    let v = &mut w[r * dim + c];
                    if c < zero_per_row {
                        *v = 0.0;
                    } else if *v == 0.0 {
                        *v = 0.01;
                    }
                }
            }
        }
        layer.freeze_for_inference();

        println!("\n(fixed layer: dim={dim} weight_sparsity={weight_sparsity})");
        println!("in_sparsity  batch  cached_ns  sparse_ns  dense_ns   oracle");
        for &in_sp in &input_sparsities {
            for &batch in &batches {
                let mut x: Vec<f32> = (0..dim * batch)
                    .map(|i| ((i % 7) as f32) * 0.1 + 0.05)
                    .collect();
                // Zero out an `in_sp` fraction of each input row.
                let zeros = (dim as f32 * in_sp) as usize;
                for row in 0..batch {
                    for c in 0..zeros {
                        x[row * dim + c] = 0.0;
                    }
                }
                let time_path = |pid: u8| -> f64 {
                    for _ in 0..warmup {
                        std::hint::black_box(
                            layer.forward_inference_forced_path(std::hint::black_box(&x), pid),
                        );
                    }
                    let start = Instant::now();
                    for _ in 0..iters {
                        std::hint::black_box(
                            layer.forward_inference_forced_path(std::hint::black_box(&x), pid),
                        );
                    }
                    start.elapsed().as_nanos() as f64 / iters as f64
                };
                let cached = time_path(0);
                let sparse = time_path(1);
                let dense = time_path(2);
                let oracle = if sparse <= cached && sparse <= dense {
                    "SPARSE"
                } else if cached <= dense {
                    "cached"
                } else {
                    "dense"
                };
                println!(
                    "{in_sp:<12} {batch:<6} {cached:>9.0} {sparse:>10.0} {dense:>9.0}   {oracle}"
                );
            }
        }
    }

    // Exploratory probe (not a pass/fail assertion): prints per-path latency
    // across a dims x sparsity grid so we can see WHERE the CSR sparse path
    // actually beats the dense/cached paths. Run with:
    //   cargo test --release --bin talos_xii csr_crossover_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn csr_crossover_probe() {
        use std::time::Instant;
        let dims = [64usize, 256, 1024, 2048];
        let sparsities = [0.5f32, 0.8, 0.9, 0.95, 0.99];
        let batch = 32usize;
        let warmup = 20usize;
        let iters = 200usize;

        println!("\ndim   sparsity  cached_ns  sparse_ns  dense_ns   winner");
        for &dim in &dims {
            for &sparsity in &sparsities {
                let cfg = AchfConfig {
                    enabled: true,
                    gate_warmup_steps: 0,
                    gate_transition_steps: 0,
                    g_min: 0.0,
                    infer_gate: "one".to_string(),
                    prune_threshold: 0.0,
                    ..Default::default()
                };
                let mut layer = AchfLayer::new(dim, dim, false, cfg, 7 + dim as u64);
                // Force an EXACT target sparsity by zeroing the first `sparsity`
                // fraction of each row (deterministic, no RNG needed for a probe).
                {
                    let mut w = layer.weight.weight.data_write_f32();
                    let zero_per_row = (dim as f32 * sparsity) as usize;
                    for r in 0..dim {
                        for c in 0..dim {
                            let v = &mut w[r * dim + c];
                            if c < zero_per_row {
                                *v = 0.0;
                            } else if *v == 0.0 {
                                *v = 0.01; // keep survivors nonzero
                            }
                        }
                    }
                }
                layer.freeze_for_inference();

                let x: Vec<f32> = (0..dim * batch).map(|i| ((i % 7) as f32) * 0.1).collect();
                let time_path = |pid: u8| -> f64 {
                    for _ in 0..warmup {
                        std::hint::black_box(
                            layer.forward_inference_forced_path(std::hint::black_box(&x), pid),
                        );
                    }
                    let start = Instant::now();
                    for _ in 0..iters {
                        std::hint::black_box(
                            layer.forward_inference_forced_path(std::hint::black_box(&x), pid),
                        );
                    }
                    start.elapsed().as_nanos() as f64 / iters as f64
                };
                let cached = time_path(0);
                let sparse = time_path(1);
                let dense = time_path(2);
                let winner = if sparse < cached && sparse < dense {
                    "SPARSE"
                } else if cached < dense {
                    "cached"
                } else {
                    "dense"
                };
                println!(
                    "{dim:<5} {sparsity:<8} {cached:>9.0} {sparse:>10.0} {dense:>9.0}   {winner}"
                );
            }
        }
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
            ema_dense_ns: 0.0,
            ema_dense_long_ns: 0.0,
            decision_ema_ns: 0.0,
            decision_ema_long_ns: 0.0,
            adaptive_bias: 2.0,
            latency_samples: 0,
            dense_latency_samples: 0,
            decision_samples: 0,
            ..Default::default()
        };
        let s2 = AchfCacheStats {
            adaptive_bias: 4.0,
            ..s1
        };
        let agg = aggregate_cache_stats_iter([s1, s2]);
        assert!((agg.adaptive_bias - 3.0).abs() < 1e-12);
    }

    fn rel_err_for_rank(w0: &[f32], rows: usize, cols: usize, rank: usize) -> f64 {
        let mut w = w0.to_vec();
        low_rank_truncate(&mut w, rows, cols, rank, 42)
    }

    #[test]
    fn low_rank_truncate_error_decreases_with_rank() {
        let rows = 16;
        let cols = 12;
        let w0: Vec<f32> = Tensor::rand(vec![rows * cols], -1.0, 1.0, 99).data_to_f32_vec();
        let err2 = rel_err_for_rank(&w0, rows, cols, 2);
        let err4 = rel_err_for_rank(&w0, rows, cols, 4);
        let err8 = rel_err_for_rank(&w0, rows, cols, 8);
        assert!(err2 > 0.0 && err4 > 0.0 && err8 > 0.0);
        assert!(
            err2 >= err4 && err4 >= err8,
            "error must be monotonically non-increasing in rank: {err2} {err4} {err8}"
        );
    }

    #[test]
    fn low_rank_truncate_full_rank_is_identity() {
        let rows = 6;
        let cols = 4;
        let w0: Vec<f32> = Tensor::rand(vec![rows * cols], -1.0, 1.0, 7).data_to_f32_vec();
        for rank in [0, cols, cols + 3] {
            let mut w = w0.clone();
            let err = low_rank_truncate(&mut w, rows, cols, rank, 42);
            assert_eq!(err, 0.0);
            assert_eq!(w, w0, "rank={rank} must leave the weight unchanged");
        }
    }

    #[test]
    fn low_rank_truncate_recovers_exact_low_rank_matrix() {
        // Build a true rank-2 matrix from two outer products.
        let rows = 10;
        let cols = 8;
        let u1: Vec<f64> = (0..rows).map(|i| (i as f64 * 0.37).sin() + 0.5).collect();
        let v1: Vec<f64> = (0..cols).map(|j| (j as f64 * 0.71).cos() - 0.2).collect();
        let u2: Vec<f64> = (0..rows).map(|i| (i as f64 * 0.13).cos() - 0.4).collect();
        let v2: Vec<f64> = (0..cols).map(|j| (j as f64 * 0.29).sin() + 0.3).collect();
        let mut w = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                w[i * cols + j] = (u1[i] * v1[j] + u2[i] * v2[j]) as f32;
            }
        }
        let err = low_rank_truncate(&mut w, rows, cols, 2, 42);
        assert!(
            err < 1e-4,
            "rank-2 truncation of a rank-2 matrix: err={err}"
        );
    }

    #[test]
    fn achf_project_weight_applies_rank_truncation_and_snapshot() {
        let cfg = AchfConfig {
            enabled: true,
            proj_mode: "none".to_string(),
            proj_freq: 0,
            rank: 2,
            prune_threshold: 0.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(8, 8, false, cfg, 77);
        let w_before = layer.weight.weight.data_f32().clone();
        layer.freeze_for_inference();
        let w_after = layer.weight.weight.data_f32().clone();
        assert_ne!(
            w_before, w_after,
            "rank-2 truncation must change an 8x8 random weight"
        );

        let snapshot = layer.snapshot_state();
        assert_eq!(snapshot.low_rank_applied_rank, 2);
        assert!(snapshot.low_rank_rel_err > 0.0);
    }

    #[test]
    fn achf_rank_zero_skips_truncation() {
        let cfg = AchfConfig {
            enabled: true,
            proj_mode: "none".to_string(),
            proj_freq: 0,
            rank: 0,
            prune_threshold: 0.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(8, 8, false, cfg, 78);
        let w_before = layer.weight.weight.data_f32().clone();
        layer.freeze_for_inference();
        let w_after = layer.weight.weight.data_f32().clone();
        assert_eq!(w_before, w_after);
        let snapshot = layer.snapshot_state();
        assert_eq!(snapshot.low_rank_applied_rank, 0);
        assert_eq!(snapshot.low_rank_rel_err, 0.0);
    }

    #[test]
    fn achf_prune_respects_rank_budget() {
        let rows = 8;
        let cols = 8;
        let cfg = AchfConfig {
            enabled: true,
            rank: 1,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(rows, cols, false, cfg, 200);
        // Threshold 0.0 alone would keep everything; the rank budget must cap nnz.
        layer.prune(0.0);
        let mask = layer.sparse_mask.as_ref().unwrap();
        let nnz: usize = mask.iter().map(|&m| m as usize).sum();
        let budget = rows + cols; // rank = 1
        assert!(nnz <= budget, "nnz={nnz} exceeds budget={budget}");
        // Sparse weight must agree with the mask.
        let sparse = layer.sparse_weight.as_ref().unwrap().weight.data_f32();
        for (v, &m) in sparse.iter().zip(mask.iter()) {
            if m == 0 {
                assert_eq!(*v, 0.0);
            }
        }
        // Each row keeps its largest-magnitude entries.
        let dense = layer.weight.weight.data_f32();
        for r in 0..rows {
            let kept_min = (0..cols)
                .filter(|&c| mask[r * cols + c] == 1)
                .map(|c| dense[r * cols + c].abs())
                .fold(f32::INFINITY, f32::min);
            let dropped_max = (0..cols)
                .filter(|&c| mask[r * cols + c] == 0)
                .map(|c| dense[r * cols + c].abs())
                .fold(0.0f32, f32::max);
            assert!(
                kept_min >= dropped_max,
                "row {r}: kept_min={kept_min} < dropped_max={dropped_max}"
            );
        }
    }

    #[test]
    fn achf_prune_rank_zero_keeps_pure_threshold_behavior() {
        let cfg = AchfConfig {
            enabled: true,
            rank: 0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new_square(4, cfg, 201);
        layer.prune(0.0);
        let mask = layer.sparse_mask.as_ref().unwrap();
        assert!(mask.iter().all(|&m| m == 1));
        let stats = layer.inference_sparsity_stats().unwrap();
        assert_eq!(stats.total_weights, 16);
        assert_eq!(stats.nonzero_weights, 16);
        assert_eq!(stats.zero_weights, 0);
        assert_eq!(stats.sparsity, 0.0);
    }

    #[test]
    fn achf_sparse_inference_copy_does_not_inflate_parameter_count() {
        let cfg = AchfConfig {
            enabled: true,
            rank: 0,
            proj_mode: "none".to_string(),
            prune_threshold: 0.0,
            ..Default::default()
        };
        let mut layer = AchfLayer::new(4, 3, true, cfg, 202);
        let before: usize = layer
            .parameters()
            .iter()
            .map(|parameter| parameter.shape.iter().product::<usize>())
            .sum();
        layer.freeze_for_inference();
        let after: usize = layer
            .parameters()
            .iter()
            .map(|parameter| parameter.shape.iter().product::<usize>())
            .sum();

        assert_eq!(before, 15);
        assert_eq!(after, before);
        assert!(layer.sparse_weight.is_some());
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
