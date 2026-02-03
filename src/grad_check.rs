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
        
        let a_grad = a.grad.read().unwrap();
        let x_grad = x.grad.read().unwrap();
        
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
        
        let x_grad = x.grad.read().unwrap();
        let y_grad = y.grad.read().unwrap();
        
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
        
        let x_val = 2.0;
        let epsilon = 1e-6;
        
        // Analytical (Autograd)
        let x = Tensor::new(vec![x_val], vec![1]);
        let y = (&x * &x) * x.sin();
        y.backward();
        let grad_auto = x.grad.read().unwrap()[0];
        
        // Numerical
        let f = |v: f64| v * v * v.sin();
        let grad_num = (f(x_val + epsilon) - f(x_val - epsilon)) / (2.0 * epsilon);
        
        println!("Auto: {}, Num: {}", grad_auto, grad_num);
        assert!((grad_auto - grad_num).abs() < 1e-4);
    }
}
