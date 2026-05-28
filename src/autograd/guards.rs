use crate::autograd::Tensor;
use crate::dtype::Storage;

enum TensorReadData<'a> {
    Borrowed(std::sync::RwLockReadGuard<'a, Vec<f64>>),
    Owned(Vec<f64>),
}

impl<'a> std::ops::Deref for TensorReadData<'a> {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        match self {
            TensorReadData::Borrowed(guard) => guard,
            TensorReadData::Owned(data) => data,
        }
    }
}

/// Batch read helper that exposes tensor data as f64 while preserving native
/// storage for F32/BF16 tensors. F64 tensors are borrowed; lower-precision
/// tensors are widened into an owned scratch buffer for compatibility.
pub struct TensorReadGuard<'a> {
    guards: Vec<TensorReadData<'a>>,
}

impl<'a> TensorReadGuard<'a> {
    /// Acquire read locks for multiple tensors at once.
    /// If any lock is poisoned, logs a warning and skips that tensor's data
    /// (the guard will contain `None` for that index and `get` will panic).
    pub fn new(tensors: &[&'a Tensor]) -> Self {
        let guards: Vec<_> = tensors
            .iter()
            .map(|t| match &t.data {
                Storage::F64(v) => match v.read() {
                    Ok(guard) => TensorReadData::Borrowed(guard),
                    Err(poison) => {
                        log::warn!(
                            target: "resilience",
                            "TensorReadGuard: data lock poisoned, recovering F64 data"
                        );
                        TensorReadData::Borrowed(poison.into_inner())
                    }
                },
                _ => TensorReadData::Owned(t.data_as_f64_vec()),
            })
            .collect();
        TensorReadGuard { guards }
    }

    /// Get data by index
    pub fn get(&self, idx: usize) -> &[f64] {
        &self.guards[idx]
    }
}
