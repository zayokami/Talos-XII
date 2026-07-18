// --- Autograd Engine ---

mod activation;
mod binary_ops;
mod conv_pooling;
mod core;
#[cfg(cuda)]
mod cuda_bridge;
#[cfg(cuda)]
mod cuda_ops;
mod extra_ops;
mod guards;
mod lifecycle;
mod loss;
mod matmul;
mod operators;
mod reductions;
mod serde_impl;
mod shape_ops;
mod softmax;
mod storage;
mod unary_ops;

#[cfg(cuda)]
pub(crate) use core::BackwardOp;
pub use core::{Context, Device, GradWriteCompat, Tensor};
#[cfg(cuda)]
use cuda_bridge::cuda_sync_grad_to_host;
#[cfg(cuda)]
pub(crate) use cuda_bridge::{cuda_cached_grad_to_f64_vec, cuda_grad_out_buffer};
#[cfg(cuda)]
pub(crate) use cuda_ops::cuda_clip_gradients_in_place;
pub use guards::TensorReadGuard;

// Minimum element count to justify Rayon parallel dispatch.
// Below this, serial iteration is faster due to scheduling overhead.
pub(crate) const PAR_THRESHOLD: usize = 4096;

#[cfg(test)]
mod tests;
