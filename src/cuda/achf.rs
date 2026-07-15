//! GPU-side ACHF diagnostics.

use crate::autograd::{Device, Tensor};
pub fn grad_mean_sq(weight_grad: &Tensor) -> Option<f64> {
    if weight_grad.device != Device::Cuda || !crate::cuda::is_available() {
        return None;
    }
    let grad = weight_grad.cuda_grad_get_or_upload_buffer().ok()?;
    let grad = grad.as_f32()?;
    let len = weight_grad.grad.len();
    if len == 0 {
        return Some(0.0);
    }
    crate::cuda::kernels::grad_mean_sq_f32(grad, len).ok()
}
