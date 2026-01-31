use std::sync::{Arc, RwLock};
use std::ops::{Add, Sub, Mul, Div, Neg};

// --- Autograd Engine ---

#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<RwLock<Vec<f64>>>,
    pub grad: Arc<RwLock<Vec<f64>>>,
    pub shape: Vec<usize>,
    pub _ctx: Option<Arc<Context>>, // Keeps the graph alive
}

pub struct Context {
    pub parents: Vec<Tensor>,
    pub backward_op: Box<dyn Fn(&Vec<f64>, &Vec<Tensor>) + Send + Sync>, // receives grad_output, parents
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let len = data.len();
        assert_eq!(len, shape.iter().product(), "Data length must match shape");
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape,
            _ctx: None,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Tensor::new(vec![0.0; len], shape)
    }
    
    pub fn rand(shape: Vec<usize>, min: f64, max: f64, seed: u64) -> Self {
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        let mut x = seed;
        for _ in 0..len {
             x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
             let r = (x >> 33) as f64 / u32::MAX as f64;
             data.push(min + r * (max - min));
        }
        Tensor::new(data, shape)
    }

    pub fn backward(&self) {
        // Topological sort
        let mut visited = std::collections::HashSet::new();
        let mut topo = Vec::new();
        fn build_topo(t: &Tensor, visited: &mut std::collections::HashSet<usize>, topo: &mut Vec<Tensor>) {
            // Use pointer address of grad RwLock as ID
            let id = Arc::as_ptr(&t.grad) as usize;
            if !visited.contains(&id) {
                visited.insert(id);
                if let Some(ctx) = &t._ctx {
                    for parent in &ctx.parents {
                        build_topo(parent, visited, topo);
                    }
                }
                topo.push(t.clone());
            }
        }
        
        build_topo(self, &mut visited, &mut topo);
        
        // Seed gradient of this tensor to 1.0
        {
            let mut g = self.grad.write().unwrap();
            for v in g.iter_mut() { *v = 1.0; }
        }
        
        // Backprop
        for t in topo.iter().rev() {
            if let Some(ctx) = &t._ctx {
                let grad = t.grad.read().unwrap();
                (ctx.backward_op)(&grad, &ctx.parents);
            }
        }
    }
    
    pub fn zero_grad(&self) {
        let mut g = self.grad.write().unwrap();
        for v in g.iter_mut() { *v = 0.0; }
    }
    
    // Operations
    
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() <= 2 && other.shape.len() == 2);
        
        let (m, k) = if self.shape.len() == 1 { (1, self.shape[0]) } else { (self.shape[0], self.shape[1]) };
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "MatMul dimension mismatch");
        
        let mut out_data = vec![0.0; m * n];
        
        {
            let lhs_data = self.data.read().unwrap();
            let rhs_data = other.data.read().unwrap();
            
            for r in 0..m {
                for c in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += lhs_data[r * k + i] * rhs_data[i * n + c];
                    }
                    out_data[r * n + c] = sum;
                }
            }
        }
        
        let out_shape = if self.shape.len() == 1 { vec![n] } else { vec![m, n] };
        
        let parents = vec![self.clone(), other.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; m * n])),
            shape: out_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    
                    let lhs_data = lhs.data.read().unwrap();
                    let rhs_data = rhs.data.read().unwrap();
                    
                    // dL/dLHS = grad_out * RHS^T
                    {
                        let mut lhs_grad = lhs.grad.write().unwrap();
                        for r in 0..m {
                            for i in 0..k {
                                let mut sum = 0.0;
                                for c in 0..n {
                                    sum += grad_out[r * n + c] * rhs_data[i * n + c];
                                }
                                lhs_grad[r * k + i] += sum;
                            }
                        }
                    }
                    
                    // dL/dRHS = LHS^T * grad_out
                    {
                        let mut rhs_grad = rhs.grad.write().unwrap();
                        for i in 0..k {
                            for c in 0..n {
                                let mut sum = 0.0;
                                for r in 0..m {
                                    sum += lhs_data[r * k + i] * grad_out[r * n + c];
                                }
                                rhs_grad[i * n + c] += sum;
                            }
                        }
                    }
                }),
            })),
        }
    }
    
    pub fn relu(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        if input_data[i] > 0.0 {
                            inp_grad[i] += g;
                        }
                    }
                }),
            })),
        }
    }

    pub fn log(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().map(|&x| x.ln()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[i] += g / input_data[i];
                    }
                }),
            })),
        }
    }

    pub fn exp(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().map(|&x| x.exp()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    // d/dx(e^x) = e^x
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[i] += g * input_data[i].exp();
                    }
                }),
            })),
        }
    }

    pub fn sum(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let sum_val: f64 = self_data.iter().sum();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(vec![sum_val])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let g = grad_out[0];
                    for v in inp_grad.iter_mut() {
                        *v += g;
                    }
                }),
            })),
        }
    }

    pub fn mean(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len = self_data.len();
        let sum_val: f64 = self_data.iter().sum();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(vec![sum_val / len as f64])),
            grad: Arc::new(RwLock::new(vec![0.0])),
            shape: vec![1],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let g = grad_out[0] / len as f64;
                    for v in inp_grad.iter_mut() {
                        *v += g;
                    }
                }),
            })),
        }
    }

    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().map(|&x| x.max(min).min(max)).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        let x = input_data[i];
                        if x >= min && x <= max {
                            inp_grad[i] += g;
                        }
                    }
                }),
            })),
        }
    }

    pub fn broadcast(&self, new_shape: Vec<usize>) -> Tensor {
        assert_eq!(self.shape, vec![1], "Broadcast only supported for scalar (1) to N");
        let len = new_shape.iter().product();
        let val = self.data.read().unwrap()[0];
        let data = vec![val; len];
        
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: new_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let sum_grad: f64 = grad_out.iter().sum();
                    inp_grad[0] += sum_grad;
                }),
            })),
        }
    }

    pub fn sin(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().map(|&x| x.sin()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[i] += g * input_data[i].cos();
                    }
                }),
            })),
        }
    }

    pub fn cos(&self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().map(|&x| x.cos()).collect();
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let input_data = input.data.read().unwrap();
                    let mut inp_grad = input.grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[i] -= g * input_data[i].sin();
                    }
                }),
            })),
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        assert!(self.shape.len() == 2, "Transpose only supported for 2D tensors");
        assert!(dim0 < 2 && dim1 < 2 && dim0 != dim1, "Invalid dims for transpose");
        
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let self_data = self.data.read().unwrap();
        let mut data = vec![0.0; rows * cols];
        
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self_data[r * cols + c];
            }
        }
        
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; rows * cols])),
            shape: vec![cols, rows],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    for r in 0..rows {
                        for c in 0..cols {
                            inp_grad[r * cols + c] += grad_out[c * rows + r];
                        }
                    }
                }),
            })),
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let self_data = self.data.read().unwrap();
        let len: usize = new_shape.iter().product();
        assert_eq!(len, self_data.len(), "Reshape dimension mismatch");
        
        let parents = vec![self.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(self_data.clone())),
            grad: Arc::new(RwLock::new(vec![0.0; len])),
            shape: new_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[i] += g;
                    }
                }),
            })),
        }
    }
    
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        let diff = self - target;
        let sq = &diff * &diff;
        sq.mean()
    }
}

// Operator overloads

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Add shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().zip(rhs_data.iter()).map(|(a, b)| a + b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    // Scope 1: LHS
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        for (i, &g) in grad_out.iter().enumerate() {
                            lhs_grad[i] += g;
                        }
                    }
                    // Scope 2: RHS
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        for (i, &g) in grad_out.iter().enumerate() {
                            rhs_grad[i] += g;
                        }
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor { self.clone() + rhs.clone() }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Sub shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().zip(rhs_data.iter()).map(|(a, b)| a - b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    // Scope 1: LHS
                    {
                        let mut lhs_grad = parents[0].grad.write().unwrap();
                        for (i, &g) in grad_out.iter().enumerate() {
                            lhs_grad[i] += g;
                        }
                    }
                    // Scope 2: RHS
                    {
                        let mut rhs_grad = parents[1].grad.write().unwrap();
                        for (i, &g) in grad_out.iter().enumerate() {
                            rhs_grad[i] -= g;
                        }
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor { self.clone() - rhs.clone() }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        // Broadcast support? For now assuming same shape or scalar broadcast
        // Simplifying to same shape for now to avoid complexity
        assert_eq!(self.shape, rhs.shape, "Mul shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().zip(rhs_data.iter()).map(|(a, b)| a * b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let lhs_data = lhs.data.read().unwrap().clone();
                    let rhs_data = rhs.data.read().unwrap().clone();
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        for (i, &g) in grad_out.iter().enumerate() {
                            grad[i] += g * (lhs_data[i] + rhs_data[i]);
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            for (i, &g) in grad_out.iter().enumerate() {
                                lhs_grad[i] += g * rhs_data[i];
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            for (i, &g) in grad_out.iter().enumerate() {
                                rhs_grad[i] += g * lhs_data[i];
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor { self.clone() * rhs.clone() }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Div shape mismatch");
        let self_data = self.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().zip(rhs_data.iter()).map(|(a, b)| a / b).collect();
        let parents = vec![self.clone(), rhs.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let lhs = &parents[0];
                    let rhs = &parents[1];
                    let lhs_data = lhs.data.read().unwrap().clone();
                    let rhs_data = rhs.data.read().unwrap().clone();
                    if Arc::ptr_eq(&lhs.grad, &rhs.grad) {
                        let mut grad = lhs.grad.write().unwrap();
                        for (i, &g) in grad_out.iter().enumerate() {
                            let l = lhs_data[i];
                            let r = rhs_data[i];
                            let v = g / r - g * l / (r * r);
                            grad[i] += v;
                        }
                    } else {
                        {
                            let mut lhs_grad = lhs.grad.write().unwrap();
                            for (i, &g) in grad_out.iter().enumerate() {
                                let r = rhs_data[i];
                                lhs_grad[i] += g / r;
                            }
                        }
                        {
                            let mut rhs_grad = rhs.grad.write().unwrap();
                            for (i, &g) in grad_out.iter().enumerate() {
                                let l = lhs_data[i];
                                let r = rhs_data[i];
                                rhs_grad[i] -= g * l / (r * r);
                            }
                        }
                    }
                }),
            })),
        }
    }
}

impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn div(self, rhs: &'b Tensor) -> Tensor { self.clone() / rhs.clone() }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        let self_data = self.data.read().unwrap();
        let data: Vec<f64> = self_data.iter().map(|&x| -x).collect();
        let parents = vec![self.clone()];
        Tensor {
            data: Arc::new(RwLock::new(data)),
            grad: Arc::new(RwLock::new(vec![0.0; self_data.len()])),
            shape: self.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(|grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[i] -= g;
                    }
                }),
            })),
        }
    }
}

impl<'a> Neg for &'a Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor { -self.clone() }
}
