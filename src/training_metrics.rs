use crate::achf::AchfStateSnapshot;
use std::sync::mpsc::Sender;

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub struct StepSnapshot {
    pub step: usize,
    pub gate_value: f64,
    pub g_min: f64,
    pub grad_ema: f64,
    pub loss: f64,
    pub reward: f64,
    pub cache_hit_rate: f64,
    pub sparse_ratio: f64,
    pub ema_cached_ns: f64,
    pub ema_sparse_ns: f64,
    pub adaptive_bias: f64,
    pub sinkhorn_iterations: usize,
    pub sinkhorn_row_max_dev: f64,
    pub sinkhorn_col_max_dev: f64,
    pub sinkhorn_min_value: f64,
    pub sinkhorn_negative_ratio: f64,
    pub sinkhorn_warm_started: bool,
    /// Rank actually applied by the low-rank truncation (0 = no truncation
    /// happened, e.g. because the requested rank was >= the layer's smaller
    /// dimension). Surfacing this makes degenerate rank sweeps visible instead
    /// of hiding them behind an unchanged stored parameter count.
    pub low_rank_applied_rank: usize,
}

impl StepSnapshot {
    pub fn from_achf(step: usize, loss: f64, reward: f64, achf: Option<AchfStateSnapshot>) -> Self {
        Self {
            step,
            gate_value: achf.map_or(1.0, |s| s.gate),
            g_min: achf.map_or(0.0, |s| s.g_min),
            grad_ema: achf.map_or(0.0, |s| s.grad_ema),
            loss,
            reward,
            cache_hit_rate: achf.map_or(0.0, |s| s.cache_hit_rate),
            sparse_ratio: achf.map_or(0.0, |s| s.low_rank_ratio),
            ema_cached_ns: achf.map_or(0.0, |s| s.ema_cached_ns),
            ema_sparse_ns: achf.map_or(0.0, |s| s.ema_sparse_ns),
            adaptive_bias: achf.map_or(1.0, |s| s.adaptive_bias),
            sinkhorn_iterations: achf.map_or(0, |s| s.sinkhorn_iterations),
            sinkhorn_row_max_dev: achf.map_or(0.0, |s| s.sinkhorn_row_max_dev),
            sinkhorn_col_max_dev: achf.map_or(0.0, |s| s.sinkhorn_col_max_dev),
            sinkhorn_min_value: achf.map_or(0.0, |s| s.sinkhorn_min_value),
            sinkhorn_negative_ratio: achf.map_or(0.0, |s| s.sinkhorn_negative_ratio),
            sinkhorn_warm_started: achf.is_some_and(|s| s.sinkhorn_warm_started),
            low_rank_applied_rank: achf.map_or(0, |s| s.low_rank_applied_rank),
        }
    }
}

pub trait TrainingMetrics {
    fn is_enabled(&self) -> bool;

    fn emit(&mut self, snapshot: StepSnapshot);

    fn emit_achf_snapshot(
        &mut self,
        step: usize,
        loss: f64,
        reward: f64,
        achf: Option<AchfStateSnapshot>,
    ) {
        if self.is_enabled() {
            self.emit(StepSnapshot::from_achf(step, loss, reward, achf));
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopTrainingMetrics;

impl TrainingMetrics for NoopTrainingMetrics {
    fn is_enabled(&self) -> bool {
        false
    }

    fn emit(&mut self, _snapshot: StepSnapshot) {}
}

pub struct ChannelTrainingMetrics {
    sender: Sender<StepSnapshot>,
}

impl ChannelTrainingMetrics {
    pub fn new(sender: Sender<StepSnapshot>) -> Self {
        Self { sender }
    }
}

impl TrainingMetrics for ChannelTrainingMetrics {
    fn is_enabled(&self) -> bool {
        true
    }

    fn emit(&mut self, snapshot: StepSnapshot) {
        let _ = self.sender.send(snapshot);
    }
}

pub enum TrainingMetricsSink {
    Noop(NoopTrainingMetrics),
    Channel(ChannelTrainingMetrics),
}

impl TrainingMetricsSink {
    pub fn noop() -> Self {
        Self::Noop(NoopTrainingMetrics)
    }

    pub fn channel(sender: Sender<StepSnapshot>) -> Self {
        Self::Channel(ChannelTrainingMetrics::new(sender))
    }
}

impl Default for TrainingMetricsSink {
    fn default() -> Self {
        Self::noop()
    }
}

impl TrainingMetrics for TrainingMetricsSink {
    fn is_enabled(&self) -> bool {
        match self {
            Self::Noop(metrics) => metrics.is_enabled(),
            Self::Channel(metrics) => metrics.is_enabled(),
        }
    }

    fn emit(&mut self, snapshot: StepSnapshot) {
        match self {
            Self::Noop(metrics) => metrics.emit(snapshot),
            Self::Channel(metrics) => metrics.emit(snapshot),
        }
    }
}

impl From<NoopTrainingMetrics> for TrainingMetricsSink {
    fn from(metrics: NoopTrainingMetrics) -> Self {
        Self::Noop(metrics)
    }
}

impl From<ChannelTrainingMetrics> for TrainingMetricsSink {
    fn from(metrics: ChannelTrainingMetrics) -> Self {
        Self::Channel(metrics)
    }
}

impl From<Sender<StepSnapshot>> for TrainingMetricsSink {
    fn from(sender: Sender<StepSnapshot>) -> Self {
        Self::channel(sender)
    }
}

impl From<Option<Sender<StepSnapshot>>> for TrainingMetricsSink {
    fn from(sender: Option<Sender<StepSnapshot>>) -> Self {
        match sender {
            Some(sender) => Self::channel(sender),
            None => Self::noop(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_metrics_are_disabled() {
        let mut metrics = NoopTrainingMetrics;

        assert!(!metrics.is_enabled());
        metrics.emit(StepSnapshot::from_achf(1, 2.0, 3.0, None));
    }

    #[test]
    fn channel_metrics_send_snapshots() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut metrics = ChannelTrainingMetrics::new(tx);
        let snapshot = StepSnapshot::from_achf(12, 0.25, 1.5, None);

        assert!(metrics.is_enabled());
        metrics.emit(snapshot.clone());

        assert_eq!(rx.try_recv().unwrap(), snapshot);
    }

    #[test]
    fn snapshot_defaults_without_achf_state() {
        let snapshot = StepSnapshot::from_achf(7, 0.5, 3.0, None);

        assert_eq!(snapshot.step, 7);
        assert_eq!(snapshot.gate_value, 1.0);
        assert_eq!(snapshot.g_min, 0.0);
        assert_eq!(snapshot.grad_ema, 0.0);
        assert_eq!(snapshot.loss, 0.5);
        assert_eq!(snapshot.reward, 3.0);
        assert_eq!(snapshot.cache_hit_rate, 0.0);
        assert_eq!(snapshot.sparse_ratio, 0.0);
        assert_eq!(snapshot.ema_cached_ns, 0.0);
        assert_eq!(snapshot.ema_sparse_ns, 0.0);
        assert_eq!(snapshot.adaptive_bias, 1.0);
        assert_eq!(snapshot.sinkhorn_iterations, 0);
        assert_eq!(snapshot.sinkhorn_row_max_dev, 0.0);
        assert_eq!(snapshot.sinkhorn_col_max_dev, 0.0);
        assert_eq!(snapshot.sinkhorn_min_value, 0.0);
        assert_eq!(snapshot.sinkhorn_negative_ratio, 0.0);
        assert!(!snapshot.sinkhorn_warm_started);
    }
}
