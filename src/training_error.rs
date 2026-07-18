use std::error::Error;
use std::fmt;

/// A hard training failure. Once returned, the in-memory optimizer/model state
/// must be treated as unusable and must not be persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingError {
    pub algorithm: &'static str,
    pub stage: &'static str,
    pub step: usize,
    pub detail: String,
}

impl TrainingError {
    pub fn new(
        algorithm: &'static str,
        stage: &'static str,
        step: usize,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            algorithm,
            stage,
            step,
            detail: detail.into(),
        }
    }

    pub fn optimizer(algorithm: &'static str, step: usize, detail: impl Into<String>) -> Self {
        Self::new(algorithm, "optimizer", step, detail)
    }
}

impl fmt::Display for TrainingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} training failed during {} at step {}: {}",
            self.algorithm, self.stage, self.step, self.detail
        )
    }
}

impl Error for TrainingError {}
