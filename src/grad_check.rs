use crate::autograd::Tensor;
use crate::dtype::{Dtype, Storage};
use std::sync::{Arc, RwLock};

/// Numerical gradient check using finite differences.
/// Compares analytical gradients (via backward()) against numerical approximation.
///
/// Returns (max_abs_diff, all_passed) where:
/// - max_abs_diff: Maximum absolute difference between analytical and numerical gradients
/// - all_passed: true if all gradient differences are within tolerance
pub fn numerical_grad_check<F>(
    tensor: &Tensor,
    loss_fn: F,
    epsilon: f64,
    tolerance: f64,
) -> (f64, bool)
where
    F: Fn(&Tensor) -> Tensor,
{
    let original_data = tensor.data_f64().clone();
    let shape = tensor.shape.clone();
    let n = original_data.len();

    // Compute analytical gradient via backward
    let loss = loss_fn(tensor);
    loss.backward();
    let analytical_grad = tensor.grad_read_f64().clone();

    // Compute numerical gradients via finite differences
    let mut max_diff: f64 = 0.0;
    for i in 0..n {
        // loss at x + eps
        let mut data_plus = original_data.clone();
        data_plus[i] += epsilon;
        let t_plus = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data_plus))),
            grad: Storage::zeros(n, Tensor::grad_dtype_for(Dtype::F64)),
            shape: shape.clone(),
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: None,
        };
        let loss_plus = loss_fn(&t_plus);
        let loss_plus_val = loss_plus.data_f64()[0];

        // loss at x - eps
        let mut data_minus = original_data.clone();
        data_minus[i] -= epsilon;
        let t_minus = Tensor {
            data: Storage::F64(Arc::new(RwLock::new(data_minus))),
            grad: Storage::zeros(n, Tensor::grad_dtype_for(Dtype::F64)),
            shape: shape.clone(),
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: None,
        };
        let loss_minus = loss_fn(&t_minus);
        let loss_minus_val = loss_minus.data_f64()[0];

        // Numerical gradient via centered difference
        let numerical_grad = (loss_plus_val - loss_minus_val) / (2.0 * epsilon);

        let diff = (analytical_grad[i] - numerical_grad).abs();
        max_diff = max_diff.max(diff);
    }

    let all_passed = max_diff <= tolerance;
    (max_diff, all_passed)
}

#[cfg(test)]
mod tests {
    use crate::autograd::Tensor;

    #[test]
    fn test_autograd_memory_leak_fix() {
        // Create a chain
        let a = Tensor::new(vec![2.0], vec![1]);
        let mut b = a.clone();

        // Loop to create a deep graph
        for _ in 0..100 {
            b = b * Tensor::new(vec![1.1], vec![1]);
        }

        // b holds the graph.
        assert!(b._ctx.is_some());

        // Detach
        let c = b.detach();
        assert!(c._ctx.is_none());

        // Clear graph
        let mut d = b.clone();
        d.clear_graph();
        assert!(d._ctx.is_none());
    }

    #[test]
    fn test_grad_check_matmul() {
        // y = A * x
        // A = [[2, 3], [4, 5]]
        // x = [1, 2]
        // y = [2*1 + 3*2, 4*1 + 5*2] = [8, 14]
        // Loss L = sum(y) = 8 + 14 = 22
        // dL/dy = [1, 1]
        // dL/dx = A^T * dL/dy = [[2, 4], [3, 5]] * [1, 1] = [6, 8]
        // dL/dA = dL/dy * x^T = [1, 1] * [1, 2] = [[1, 2], [1, 2]]

        let a_data = vec![2.0, 3.0, 4.0, 5.0];
        let x_data = vec![1.0, 2.0];

        let a = Tensor::new(a_data.clone(), vec![2, 2]);
        let x = Tensor::new(x_data.clone(), vec![2, 1]); // Column vector

        let y = a.matmul(&x);
        let loss = y.sum();

        loss.backward();

        let a_grad = a.grad_read_f64();
        let x_grad = x.grad_read_f64();

        // Expected x_grad: [6, 8]
        assert!((x_grad[0] - 6.0).abs() < 1e-6);
        assert!((x_grad[1] - 8.0).abs() < 1e-6);

        // Expected a_grad: [1, 2, 1, 2]
        assert!((a_grad[0] - 1.0).abs() < 1e-6);
        assert!((a_grad[1] - 2.0).abs() < 1e-6);
        assert!((a_grad[2] - 1.0).abs() < 1e-6);
        assert!((a_grad[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_grad_check_broadcast() {
        // z = x + y (broadcast)
        // x = [1, 2] (2)
        // y = [10] (1) -> broadcast to [10, 10]
        // z = [11, 12]
        // L = sum(z) = 23
        // dL/dx = [1, 1]
        // dL/dy = sum([1, 1]) = 2

        let x = Tensor::new(vec![1.0, 2.0], vec![2]);
        let y = Tensor::new(vec![10.0], vec![1]);

        let y_b = y.broadcast(vec![2]);
        let z = x.clone() + y_b;
        let loss = z.sum();

        loss.backward();

        let x_grad = x.grad_read_f64();
        let y_grad = y.grad_read_f64();

        assert!((x_grad[0] - 1.0).abs() < 1e-6);
        assert!((x_grad[1] - 1.0).abs() < 1e-6);
        assert!((y_grad[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_gradient_check() {
        // Verify autograd against finite difference
        // f(x) = x^2 * sin(x)
        // f'(x) = 2x*sin(x) + x^2*cos(x)
        // Check at x = 2.0

        let x = Tensor::new(vec![2.0], vec![1]);
        let loss_fn = |t: &Tensor| t.clone() * t.clone() * t.sin();
        let (_, all_passed) = super::numerical_grad_check(&x, loss_fn, 1e-6, 1e-4);
        assert!(all_passed);
    }
}
