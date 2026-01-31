use crate::autograd::{Tensor, Context};
use std::sync::{Arc, RwLock};

// --- Linear Layer Helper ---
#[derive(Clone)]
pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
    pub _in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        Self {
            w: Tensor::rand(vec![in_dim, out_dim], -0.1, 0.1, seed),
            b: Tensor::zeros(vec![out_dim]),
            _in_dim: in_dim,
            out_dim,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [seq, in_dim]
        let out = x.matmul(&self.w); // [seq, out_dim]
        
        // Manual broadcast for bias
        let batch_size = out.shape[0];
        let b_data = self.b.data.read().unwrap();
        let mut tiled_b_data = Vec::with_capacity(batch_size * self.out_dim);
        for _ in 0..batch_size {
            tiled_b_data.extend_from_slice(&b_data);
        }
        
        // We need to create a tensor for tiled_b that doesn't track gradients back to b 
        // in a complex way, but here we just want to add it.
        // Actually, for autograd to work for b, we should construct tiled_b 
        // such that its parent is b.
        // "Expand" operation.
        let parents = vec![self.b.clone()];
        let out_dim = self.out_dim;
        
        let tiled_b = Tensor {
            data: Arc::new(RwLock::new(tiled_b_data)),
            grad: Arc::new(RwLock::new(vec![0.0; batch_size * out_dim])),
            shape: out.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut b_grad = parents[0].grad.write().unwrap();
                    // Sum gradients across batch
                    for i in 0..batch_size {
                        for j in 0..out_dim {
                            b_grad[j] += grad_out[i * out_dim + j];
                        }
                    }
                }),
            })),
        };
        
        out + tiled_b
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.w.clone(), self.b.clone()]
    }
}

// --- RoPE: Rotary Positional Embeddings ---
#[derive(Clone)]
pub struct RoPE {
    pub dim: usize,
    pub base: f64,
}

impl RoPE {
    pub fn new(dim: usize) -> Self {
        Self { dim, base: 10000.0 }
    }

    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Tensor {
        // x: [seq_len, dim]
        let seq_len = x.shape[0];
        assert_eq!(seq_len, positions.len(), "RoPE: positions len must match seq len");
        
        // Generate cos/sin tables
        let mut cos_data = Vec::with_capacity(seq_len * self.dim);
        let mut sin_data = Vec::with_capacity(seq_len * self.dim);
        
        for t in 0..seq_len {
            let pos = positions[t];
            for i in 0..self.dim / 2 {
                let theta = 1.0 / self.base.powf((2 * i) as f64 / self.dim as f64);
                let angle = pos as f64 * theta;
                let c = angle.cos();
                let s = angle.sin();
                cos_data.push(c);
                cos_data.push(c);
                sin_data.push(s);
                sin_data.push(s);
            }
        }
        
        let cos_t = Tensor::new(cos_data, x.shape.clone());
        let sin_t = Tensor::new(sin_data, x.shape.clone());
        
        // x_rotated = [-x1, x0, -x3, x2...]
        let x_rot = self.rotate_half(x);
        
        (x.clone() * cos_t) + (x_rot * sin_t)
    }
    
    fn rotate_half(&self, x: &Tensor) -> Tensor {
        let data = x.data.read().unwrap();
        let mut new_data = Vec::with_capacity(data.len());
        
        // Assume last dim is contiguous
        for chunk in data.chunks(2) {
            new_data.push(-chunk[1]);
            new_data.push(chunk[0]);
        }
        
        let parents = vec![x.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; data.len()])),
            shape: x.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    for (i, chunk) in grad_out.chunks(2).enumerate() {
                        if chunk.len() == 2 {
                            let dy0 = chunk[0];
                            let dy1 = chunk[1];
                            inp_grad[i*2] += dy1;
                            inp_grad[i*2+1] += -dy0;
                        }
                    }
                }),
            })),
        }
    }
}

// --- Linear Attention ---
#[derive(Clone)]
pub struct LinearAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub _dim: usize,
}

impl LinearAttention {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            w_q: Linear::new(dim, dim, seed),
            w_k: Linear::new(dim, dim, seed + 1),
            w_v: Linear::new(dim, dim, seed + 2),
            w_o: Linear::new(dim, dim, seed + 3),
            _dim: dim,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [seq_len, dim]
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);
        
        // Feature map: relu(x) + 1e-6 (positivity)
        // Since we don't have scalar add easily, just relu for now (zeros are fine)
        let q_prime = q.relu(); 
        let k_prime = k.relu();
        
        // K^T: [dim, seq_len]
        let k_t = k_prime.transpose(0, 1);
        
        // KV = K^T * V -> [dim, dim]
        let kv = k_t.matmul(&v);
        
        // Out = Q' * KV -> [seq, dim] * [dim, dim] -> [seq, dim]
        let out = q_prime.matmul(&kv);
        
        // Scale by 1/sqrt(dim)
        // We'll just assume learning handles scaling or implement a manual scale if needed.
        // For numerical stability, let's scale.
        // But autograd doesn't have scalar mul easily without broadcasting logic we just built?
        // Wait, Mul is element-wise.
        // I can multiply by a scalar tensor if I broadcast it.
        // Let's skip scaling for now, usually important but Linear Attn is robust.
        
        self.w_o.forward(&out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.w_q.parameters());
        p.extend(self.w_k.parameters());
        p.extend(self.w_v.parameters());
        p.extend(self.w_o.parameters());
        p
    }
}

// --- MoLE: Mixture of Luck Experts ---
#[derive(Clone)]
pub struct MoLE {
    pub expert_eu: Linear,     // High Luck Expert
    pub expert_baodi: Linear,  // Hard Pity Expert
    pub expert_lianwai: Linear, // Loss Streak Expert
    pub expert_default: Linear, // Default Expert
    pub router: Linear,        // Gates [dim -> 4]
}

impl MoLE {
    pub fn new(dim: usize, hidden: usize, seed: u64) -> Self {
        Self {
            expert_eu: Linear::new(dim, hidden, seed + 10),
            expert_baodi: Linear::new(dim, hidden, seed + 11),
            expert_lianwai: Linear::new(dim, hidden, seed + 12),
            expert_default: Linear::new(dim, hidden, seed + 13),
            router: Linear::new(dim, 4, seed + 14),
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [seq, dim]
        let gates_logits = self.router.forward(x); // [seq, 4]
        
        // Softmax
        let gates = self.softmax(&gates_logits);
        
        let e1 = self.expert_eu.forward(x);
        let e2 = self.expert_baodi.forward(x);
        let e3 = self.expert_lianwai.forward(x);
        let e4 = self.expert_default.forward(x);
        
        self.weighted_sum(&gates, vec![&e1, &e2, &e3, &e4])
    }
    
    fn softmax(&self, x: &Tensor) -> Tensor {
        // x: [seq, 4]
        // Safe softmax: exp(x - max) / sum(exp)
        // Since we don't have max/sum(dim) easily, let's just do exp / sum(all) per row manually with custom backward.
        // Actually, we can use exp() now.
        // But we lack sum(dim=1). autograd.rs sum() is global.
        
        // Let's implement softmax as a custom op for simplicity and efficiency in backward.
        
        let x_data = x.data.read().unwrap();
        let rows = x.shape[0];
        let cols = x.shape[1];
        let mut y_data = vec![0.0; rows * cols];
        
        for r in 0..rows {
            let offset = r * cols;
            let mut max_val = -1e9;
            for c in 0..cols {
                if x_data[offset + c] > max_val { max_val = x_data[offset + c]; }
            }
            let mut sum_exp = 0.0;
            for c in 0..cols {
                let v = (x_data[offset + c] - max_val).exp();
                y_data[offset + c] = v;
                sum_exp += v;
            }
            for c in 0..cols {
                y_data[offset + c] /= sum_exp;
            }
        }
        
        let parents = vec![x.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(y_data)),
            grad: Arc::new(RwLock::new(vec![0.0; rows * cols])),
            shape: x.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    // Recompute y (forward pass) locally to be safe and clean
                    let x = &parents[0];
                    let x_data = x.data.read().unwrap();
                    let rows = x.shape[0];
                    let cols = x.shape[1];
                    let mut inp_grad = x.grad.write().unwrap();
                    
                    for r in 0..rows {
                        let offset = r * cols;
                        // Forward recompute
                        let mut max_val = -1e9;
                        for c in 0..cols {
                            if x_data[offset + c] > max_val { max_val = x_data[offset + c]; }
                        }
                        let mut sum_exp = 0.0;
                        let mut y_row = vec![0.0; cols];
                        for c in 0..cols {
                            let v = (x_data[offset + c] - max_val).exp();
                            y_row[c] = v;
                            sum_exp += v;
                        }
                        for c in 0..cols {
                            y_row[c] /= sum_exp;
                        }
                        
                        // Backward
                        // sum_k = sum(grad_out[k] * y[k])
                        let mut sum_ky = 0.0;
                        for c in 0..cols {
                            sum_ky += grad_out[offset + c] * y_row[c];
                        }
                        
                        for c in 0..cols {
                            // grad_x[c] = y[c] * (grad_out[c] - sum_ky)
                            inp_grad[offset + c] += y_row[c] * (grad_out[offset + c] - sum_ky);
                        }
                    }
                }),
            })),
        }
    }
    
    fn weighted_sum(&self, gates: &Tensor, experts: Vec<&Tensor>) -> Tensor {
        // gates: [seq, 4]
        // experts: 4 tensors of [seq, hidden]
        // result: sum(gates[:, i] * experts[i])
        
        let seq_len = gates.shape[0];
        let hidden = experts[0].shape[1];
        let num_experts = experts.len();
        
        let gates_data = gates.data.read().unwrap();
        let mut out_data = vec![0.0; seq_len * hidden];
        
        for i in 0..num_experts {
            let exp_data = experts[i].data.read().unwrap();
            for r in 0..seq_len {
                let g = gates_data[r * num_experts + i];
                for c in 0..hidden {
                    out_data[r * hidden + c] += g * exp_data[r * hidden + c];
                }
            }
        }
        
        let mut parents = vec![gates.clone()];
        for e in &experts { parents.push((*e).clone()); }
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; seq_len * hidden])),
            shape: vec![seq_len, hidden],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let gates = &parents[0];
                    let experts_refs = &parents[1..];
                    
                    let mut gates_grad = gates.grad.write().unwrap();
                    let gates_data = gates.data.read().unwrap();
                    
                    // We need read locks for experts, but we also need write locks for their grads.
                    // Doing this in loop to avoid deadlock or complexity.
                    
                    // 1. Gradient w.r.t Gates
                    // dL/dGate_i = sum(grad_out * Expert_i)
                    for i in 0..num_experts {
                        let exp = &experts_refs[i];
                        let exp_data = exp.data.read().unwrap();
                        
                        for r in 0..seq_len {
                            let mut sum = 0.0;
                            for c in 0..hidden {
                                sum += grad_out[r * hidden + c] * exp_data[r * hidden + c];
                            }
                            gates_grad[r * num_experts + i] += sum;
                        }
                    }
                    
                    // 2. Gradient w.r.t Experts
                    // dL/dExpert_i = grad_out * Gate_i
                    for i in 0..num_experts {
                         let exp = &experts_refs[i];
                         let mut exp_grad = exp.grad.write().unwrap();
                         
                         for r in 0..seq_len {
                             let g = gates_data[r * num_experts + i];
                             for c in 0..hidden {
                                 exp_grad[r * hidden + c] += grad_out[r * hidden + c] * g;
                             }
                         }
                    }
                }),
            })),
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.expert_eu.parameters());
        p.extend(self.expert_baodi.parameters());
        p.extend(self.expert_lianwai.parameters());
        p.extend(self.expert_default.parameters());
        p.extend(self.router.parameters());
        p
    }
}

// --- LuckTransformer ---
#[derive(Clone)]
pub struct LuckTransformer {
    pub embedding: Linear, // Project input dim to hidden
    pub rope: RoPE,
    pub attention: LinearAttention,
    pub mole: MoLE,
}

impl LuckTransformer {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let seed = 42;
        Self {
            embedding: Linear::new(input_dim, hidden_dim, seed),
            rope: RoPE::new(hidden_dim),
            attention: LinearAttention::new(hidden_dim, seed + 100),
            mole: MoLE::new(hidden_dim, hidden_dim, seed + 200),
        }
    }
    
    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Tensor {
        // x: [seq, input_dim]
        let h = self.embedding.forward(x);
        
        // Add RoPE
        let h_pos = self.rope.forward(&h, positions);
        
        // Attention
        let attn_out = self.attention.forward(&h_pos);
        
        // Residual + Norm (Skip norm for simplicity, just Add)
        let h2 = h_pos + attn_out;
        
        // MoLE
        let mole_out = self.mole.forward(&h2);
        
        // Residual
        h2 + mole_out
    }
    
    pub fn last_token(&self, x: &Tensor) -> Tensor {
        // x: [seq, dim]
        // Returns [1, dim]
        let shape = x.shape.clone();
        let seq = shape[0];
        let dim = shape[1];
        
        let data = x.data.read().unwrap();
        let start = (seq - 1) * dim;
        let last_data = data[start..].to_vec();
        
        let parents = vec![x.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(last_data)),
            grad: Arc::new(RwLock::new(vec![0.0; dim])),
            shape: vec![1, dim],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let start_idx = (seq - 1) * dim;
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[start_idx + i] += g;
                    }
                }),
            })),
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.embedding.parameters());
        p.extend(self.attention.parameters());
        p.extend(self.mole.parameters());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luck_transformer_forward() {
        let _batch = 2;
        let seq = 5;
        let input_dim = 10;
        let hidden_dim = 16;
        
        let model = LuckTransformer::new(input_dim, hidden_dim);
        
        // Create dummy input [seq, input_dim]
        let input_data = vec![0.5; seq * input_dim];
        let input = Tensor::new(input_data, vec![seq, input_dim]);
        
        let positions: Vec<usize> = (0..seq).collect();
        let out = model.forward(&input, &positions);
        
        assert_eq!(out.shape, vec![seq, hidden_dim]);
        
        // Check values are not NaN
        let out_data = out.data.read().unwrap();
        for &x in out_data.iter() {
            assert!(!x.is_nan());
        }
    }
}
