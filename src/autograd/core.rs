use crate::dtype::{Dtype, Storage};
use std::sync::Arc;

/// Device where tensor data resides
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Device {
    /// CPU (default)
    #[default]
    Cpu,
    /// CUDA GPU (when cuda feature is enabled)
    #[cfg(cuda)]
    Cuda,
}

pub type BackwardOp = Box<dyn Fn(&Storage, &Vec<Tensor>) + Send + Sync>;

pub struct Context {
    pub parents: Vec<Tensor>,
    pub backward_op: BackwardOp,
}

pub enum GradWriteCompat<'a> {
    F64(std::sync::RwLockWriteGuard<'a, Vec<f64>>),
    F32Buffer(Vec<f64>, &'a Storage),
}

impl<'a> std::ops::Deref for GradWriteCompat<'a> {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        match self {
            GradWriteCompat::F64(g) => g,
            GradWriteCompat::F32Buffer(buf, _) => buf,
        }
    }
}

impl<'a> std::ops::DerefMut for GradWriteCompat<'a> {
    fn deref_mut(&mut self) -> &mut [f64] {
        match self {
            GradWriteCompat::F64(g) => g,
            GradWriteCompat::F32Buffer(buf, _) => buf,
        }
    }
}

impl<'a> Drop for GradWriteCompat<'a> {
    fn drop(&mut self) {
        if let GradWriteCompat::F32Buffer(buf, storage) = self {
            storage.accumulate_f64_slice(buf);
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Storage,
    pub grad: Storage,
    pub shape: Vec<usize>,
    pub device: Device,
    pub dtype: Dtype,
    pub _ctx: Option<Arc<Context>>,
}
