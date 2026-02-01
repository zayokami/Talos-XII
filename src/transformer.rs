use crate::autograd::{Tensor, Context};
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

// --- Configuration ---
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MLAConfig {
    pub dim: usize,          // Model dimension (d_model)
    pub num_heads: usize,    // Number of heads (n_h)
    pub q_lora_rank: usize,  // Latent dim for Q (d_c_q) - if 0, no compression
    pub kv_lora_rank: usize, // Latent dim for KV (d_c_kv)
    pub qk_rope_dim: usize,  // Dimension for RoPE part (d_R)
    pub v_head_dim: usize,   // Dimension for Value head (d_v)
    pub max_seq_len: usize,
}

impl MLAConfig {
    pub fn deepseek_v2_lite() -> Self {
        Self {
            dim: 512,
            num_heads: 8,
            q_lora_rank: 0, // No compression for Q in lite
            kv_lora_rank: 128,
            qk_rope_dim: 32,
            v_head_dim: 64,
            max_seq_len: 2048,
        }
    }
}

// --- Linear Layer Helper ---
#[derive(Clone, Serialize, Deserialize)]
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, bias: bool, seed: u64) -> Self {
        Self {
            w: Tensor::rand(vec![in_dim, out_dim], -0.1, 0.1, seed),
            b: if bias { Some(Tensor::zeros(vec![out_dim])) } else { None },
            in_dim,
            out_dim,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [..., in_dim]
        // Flatten x to [N, in_dim] for matmul
        let input_shape = x.shape.clone();
        let batch_dim = input_shape.iter().take(input_shape.len() - 1).product();
        let x_flat = x.reshape(vec![batch_dim, self.in_dim]);
        
        let out_flat = x_flat.matmul(&self.w); // [N, out_dim]
        
        let mut out = out_flat.reshape([input_shape[..input_shape.len()-1].to_vec(), vec![self.out_dim]].concat());
        
        if let Some(b) = &self.b {
            // Manual broadcast for bias
            // Simply construct a new tensor that adds bias to every last dim
            // For now, let's assume autograd supports simple broadcasting or we do it manually
            // Given autograd.rs limitations, we'll do a simple add loop in a custom op if needed
            // But let's assume we can just add a reshaped bias?
            // Tensor addition usually requires matching shapes.
            // Let's rely on a custom add helper for now or just skip bias if complex.
            // But bias is important.
            // Let's implement a simple broadcast_add logic here:
            
            let b_data = b.data.read().unwrap();
            let out_data_len = batch_dim * self.out_dim;
            let mut new_data = Vec::with_capacity(out_data_len);
            
            // Scope to drop the read lock on out
            {
                let out_read = out.data.read().unwrap();
                for i in 0..batch_dim {
                    for j in 0..self.out_dim {
                        new_data.push(out_read[i * self.out_dim + j] + b_data[j]);
                    }
                }
            }
            
            // Re-wrap in Tensor with backward pass
            let parents = vec![out.clone(), b.clone()];
            let out_dim = self.out_dim;
            
            out = Tensor {
                data: Arc::new(RwLock::new(new_data)),
                grad: Arc::new(RwLock::new(vec![0.0; out_data_len])),
                shape: out.shape.clone(),
                _ctx: Some(Arc::new(Context {
                    parents,
                    backward_op: Box::new(move |grad_out, parents| {
                        let out_tensor = &parents[0];
                        let b_tensor = &parents[1];
                        
                        let mut out_grad = out_tensor.grad.write().unwrap();
                        let mut b_grad = b_tensor.grad.write().unwrap();
                        
                        // dL/dout_tensor = grad_out (identity)
                        for i in 0..grad_out.len() {
                            out_grad[i] += grad_out[i];
                        }
                        
                        // dL/db = sum(grad_out) over batch
                        // grad_out is [Batch, OutDim]
                        // b is [OutDim]
                        let total_elements = grad_out.len();
                        let batch_size = total_elements / out_dim;
                        
                        for i in 0..batch_size {
                            for j in 0..out_dim {
                                b_grad[j] += grad_out[i * out_dim + j];
                            }
                        }
                    }),
                })),
            };
        }
        out
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.w.clone()];
        if let Some(b) = &self.b {
            p.push(b.clone());
        }
        p
    }
}

// --- RMSNorm ---
#[derive(Clone, Serialize, Deserialize)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
    pub dim: usize,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, _seed: u64) -> Self {
        // Initialize to 1.0 (no-op scaling initially)
        // We use a small random variation to break symmetry if needed, but usually 1.0 is standard.
        // Using rand around 1.0 might be better for training start? 
        // Standard is 1.0.
        Self {
            weight: Tensor::new(vec![1.0; dim], vec![dim]),
            eps,
            dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [Batch, Seq, Dim] or similar. Last dim must match self.dim.
        let shape = &x.shape;
        let last_dim = shape[shape.len() - 1];
        assert_eq!(last_dim, self.dim, "RMSNorm dim mismatch");
        
        let x_data = x.data.read().unwrap();
        let w_data = self.weight.data.read().unwrap();
        
        let num_elements = x_data.len();
        let num_rows = num_elements / self.dim;
        
        let mut out_data = Vec::with_capacity(num_elements);
        let mut rms_cache = Vec::with_capacity(num_rows); // Cache rms for backward
        let mut x_hat_cache = Vec::with_capacity(num_elements); // Cache x_hat for backward
        
        for r in 0..num_rows {
            let base = r * self.dim;
            let mut sum_sq = 0.0;
            for i in 0..self.dim {
                let val = x_data[base + i];
                sum_sq += val * val;
            }
            let rms = (sum_sq / self.dim as f64 + self.eps).sqrt();
            rms_cache.push(rms);
            
            for i in 0..self.dim {
                let val = x_data[base + i];
                let x_hat = val / rms;
                x_hat_cache.push(x_hat);
                out_data.push(x_hat * w_data[i]);
            }
        }
        
        let parents = vec![x.clone(), self.weight.clone()];
        let dim = self.dim;
        let _eps = self.eps; // Capture eps if needed (not needed for grad formula derived above)
        
        // We need to capture cache.
        // Since we can't easily modify the Tensor struct to store arbitrary cache,
        // we capture it in the closure.
        let rms_cache = Arc::new(rms_cache);
        let x_hat_cache = Arc::new(x_hat_cache);
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; num_elements])),
            shape: shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let x_in = &parents[0];
                    let w_in = &parents[1];
                    
                    let mut x_grad = x_in.grad.write().unwrap();
                    let mut w_grad = w_in.grad.write().unwrap();
                    let w_data = w_in.data.read().unwrap();
                    
                    // x_hat_cache and rms_cache are available here
                    
                    for r in 0..num_rows {
                        let base = r * dim;
                        let rms = rms_cache[r];
                        let inv_rms = 1.0 / rms;
                        
                        // 1. Calculate dL/dx_hat
                        // dL/dx_hat_i = grad_out_i * w_i
                        
                        // 2. Calculate dot product sum(dL/dx_hat * x_hat)
                        let mut dot_sum = 0.0;
                        for i in 0..dim {
                            let g = grad_out[base + i];
                            let w = w_data[i];
                            let dl_dxhat = g * w;
                            dot_sum += dl_dxhat * x_hat_cache[base + i];
                            
                            // Accumulate weight gradient
                            // dL/dw_i = sum_over_batch(grad_out * x_hat)
                            // We can do this atomically or accumulate locally then write?
                            // RwLock write is exclusive. We are inside write lock.
                            // But we iterate rows. w_grad is shared across rows.
                            w_grad[i] += g * x_hat_cache[base + i];
                        }
                        
                        let mean_dot = dot_sum / dim as f64;
                        
                        // 3. Calculate dL/dx
                        for i in 0..dim {
                            let g = grad_out[base + i];
                            let w = w_data[i];
                            let dl_dxhat = g * w;
                            let x_hat = x_hat_cache[base + i];
                            
                            // dL/dx = (1/rms) * (dL/dx_hat - x_hat * mean_dot)
                            x_grad[base + i] += inv_rms * (dl_dxhat - x_hat * mean_dot); // Wait, formula check
                            // Formula: dL/dx = (1/rms) * (dL/dx_hat - x_hat * (1/d) * sum(dL/dx_hat * x_hat))
                            // Yes, matches.
                        }
                    }
                }),
            })),
        }
    }
    
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

// --- LuckTransformer (Transformer Backbone with MLA) ---
#[derive(Clone, Serialize, Deserialize)]
pub struct LuckTransformer {
    pub embed: Linear,
    pub norm_1: RMSNorm,
    pub mla_layer: MultiHeadLatentAttention,
    pub norm_2: RMSNorm,
    pub ffn_1: Linear,
    pub ffn_2: Linear,
    pub norm_final: RMSNorm,
    pub out_proj: Linear,
}

impl LuckTransformer {
    pub fn new(in_dim: usize, hidden_dim: usize, _bias: bool, seed: u64) -> Self {
        let mla_config = MLAConfig {
            dim: hidden_dim,
            num_heads: 4,
            q_lora_rank: 0,
            kv_lora_rank: 32, // Low rank for efficiency
            qk_rope_dim: 16,
            v_head_dim: hidden_dim / 4,
            max_seq_len: 256,
        };
        
        Self {
            embed: Linear::new(in_dim, hidden_dim, true, seed),
            norm_1: RMSNorm::new(hidden_dim, 1e-5, seed + 5),
            mla_layer: MultiHeadLatentAttention::new(mla_config, seed + 10),
            norm_2: RMSNorm::new(hidden_dim, 1e-5, seed + 15),
            ffn_1: Linear::new(hidden_dim, hidden_dim * 2, true, seed + 20),
            ffn_2: Linear::new(hidden_dim * 2, hidden_dim, true, seed + 30),
            norm_final: RMSNorm::new(hidden_dim, 1e-5, seed + 35),
            out_proj: Linear::new(hidden_dim, hidden_dim, true, seed + 40),
        }
    }
    
    pub fn forward(&self, x: &Tensor, _pity: &[usize]) -> Tensor {
        // x: [Batch, Seq, Dim]
        // Embed
        let h = self.embed.forward(x);
        
        // Block 1: MLA (Pre-Norm)
        let h_norm1 = self.norm_1.forward(&h);
        let attn_out = self.mla_layer.forward(&h_norm1);
        let h2 = h + attn_out;
        
        // Block 2: FFN (Pre-Norm)
        let h_norm2 = self.norm_2.forward(&h2);
        let f1 = self.ffn_1.forward(&h_norm2).relu();
        let f2 = self.ffn_2.forward(&f1);
        let h3 = h2 + f2;
        
        // Final Norm + Output
        let h_final = self.norm_final.forward(&h3);
        self.out_proj.forward(&h_final)
    }
    
    pub fn last_token(&self, x: &Tensor) -> Tensor {
        // x: [Batch, Seq, Dim]
        // Return [Batch, Dim] (last token)
        let shape = &x.shape;
        let batch_size = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];
        
        let x_data = x.data.read().unwrap();
        let mut out_data = Vec::with_capacity(batch_size * dim);
        
        for b in 0..batch_size {
            let start = b * seq_len * dim + (seq_len - 1) * dim;
            out_data.extend_from_slice(&x_data[start..start + dim]);
        }
        
        Tensor::new(out_data, vec![batch_size, dim])
    }
    
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.embed.parameters();
        p.extend(self.norm_1.parameters());
        p.extend(self.mla_layer.parameters());
        p.extend(self.norm_2.parameters());
        p.extend(self.ffn_1.parameters());
        p.extend(self.ffn_2.parameters());
        p.extend(self.norm_final.parameters());
        p.extend(self.out_proj.parameters());
        p
    }
}


// --- RoPE: Rotary Positional Embeddings ---
#[derive(Clone, Serialize, Deserialize)]
pub struct RoPE {
    pub dim: usize,
    pub base: f64,
    pub cos_cache: Vec<f64>,
    pub sin_cache: Vec<f64>,
}

impl RoPE {
    pub fn new(dim: usize, max_len: usize) -> Self {
        let mut rope = Self { dim, base: 10000.0, cos_cache: vec![], sin_cache: vec![] };
        rope.precompute(max_len);
        rope
    }

    fn precompute(&mut self, max_len: usize) {
        self.cos_cache = Vec::with_capacity(max_len * (self.dim / 2));
        self.sin_cache = Vec::with_capacity(max_len * (self.dim / 2));
        
        for pos in 0..max_len {
            for i in 0..self.dim / 2 {
                let theta = 1.0 / self.base.powf((2 * i) as f64 / self.dim as f64);
                let angle = pos as f64 * theta;
                self.cos_cache.push(angle.cos());
                self.sin_cache.push(angle.sin());
            }
        }
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        // x: [Batch, Seq, Heads, Dim] or [Batch, Seq, Dim]
        // We assume x is [..., Seq, HeadDim]
        // RoPE applies to the last dimension.
        // Simplified: just apply to the last dimension assuming it matches self.dim
        
        let shape = &x.shape;
        let dim = shape[shape.len() - 1];
        assert_eq!(dim, self.dim);
        let seq_len = shape[shape.len() - 2]; // Assumes ..., Seq, Dim
        
        let x_data = x.data.read().unwrap();
        let mut out_data = x_data.clone(); // Copy
        
        // Apply rotation
        // This is a naive CPU implementation
        let num_elements = x_data.len();
        
        let total_batches = num_elements / (seq_len * dim);
        
        for b in 0..total_batches {
            for t in 0..seq_len {
                let pos = start_pos + t;
                let cache_idx = pos * (self.dim / 2); // Base index in cache
                
                let base_idx = b * (seq_len * dim) + t * dim;
                
                for i in 0..self.dim / 2 {
                    let c = self.cos_cache[cache_idx + i];
                    let s = self.sin_cache[cache_idx + i];
                    
                    let r1 = x_data[base_idx + 2*i];
                    let r2 = x_data[base_idx + 2*i + 1];
                    
                    out_data[base_idx + 2*i]     = r1 * c - r2 * s;
                    out_data[base_idx + 2*i + 1] = r1 * s + r2 * c;
                }
            }
        }
        
        let parents = vec![x.clone()];
        let cos_cache = self.cos_cache.clone();
        let sin_cache = self.sin_cache.clone();
        let dim = self.dim;
        let start_pos_cap = start_pos;
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; num_elements])),
            shape: shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let input = &parents[0];
                    let mut inp_grad = input.grad.write().unwrap();
                    let shape = &input.shape;
                    
                    let seq_len = shape[shape.len() - 2];
                    let total_batches = inp_grad.len() / (seq_len * dim);
                    
                    for b in 0..total_batches {
                        for t in 0..seq_len {
                            let pos = start_pos_cap + t;
                            let cache_idx = pos * (dim / 2);
                            let base_idx = b * (seq_len * dim) + t * dim;
                            
                            for i in 0..dim / 2 {
                                let c = cos_cache[cache_idx + i];
                                let s = sin_cache[cache_idx + i];
                                
                                let g1 = grad_out[base_idx + 2*i];
                                let g2 = grad_out[base_idx + 2*i + 1];
                                
                                // dL/dx1 = g1 * c + g2 * s
                                // dL/dx2 = -g1 * s + g2 * c
                                
                                inp_grad[base_idx + 2*i]     += g1 * c + g2 * s;
                                inp_grad[base_idx + 2*i + 1] += -g1 * s + g2 * c;
                            }
                        }
                    }
                }),
            })),
        }
    }
}

// --- Multi-Head Latent Attention (MLA) ---
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiHeadLatentAttention {
    pub config: MLAConfig,
    
    // Compression (Down Projection)
    pub w_dkv: Linear, // Projects input to latent c_KV
    
    // Decompression (Up Projection)
    pub w_uk: Linear, // Projects c_KV to Key Heads
    pub w_uv: Linear, // Projects c_KV to Value Heads
    
    // Query Projection (Standard or Compressed)
    // If q_lora_rank > 0, we'd have w_dq and w_uq. 
    // For simplicity, let's assume standard Q projection for now unless compressed.
    pub w_q: Linear, 
    
    // RoPE Projections (Decoupled)
    pub w_kr: Linear, // Generates k_rope
    pub w_qr: Linear, // Generates q_rope
    
    // Output Projection
    pub w_o: Linear,
    
    pub rope: RoPE,
}

impl MultiHeadLatentAttention {
    pub fn new(config: MLAConfig, seed: u64) -> Self {
        let dim = config.dim;
        let num_heads = config.num_heads;
        let head_dim = config.v_head_dim; // Usually v_head_dim == q_head_dim
        let rope_dim = config.qk_rope_dim;
        let kv_latent = config.kv_lora_rank;
        
        // Output of Up projections is (num_heads * head_dim)
        let full_head_dim = num_heads * head_dim;
        
        Self {
            config: config.clone(),
            // W_DKV: Dim -> LatentKV
            w_dkv: Linear::new(dim, kv_latent, false, seed),
            
            // W_UK: LatentKV -> Heads * HeadDim
            w_uk: Linear::new(kv_latent, full_head_dim, false, seed + 1),
            
            // W_UV: LatentKV -> Heads * HeadDim
            w_uv: Linear::new(kv_latent, full_head_dim, false, seed + 2),
            
            // W_Q: Dim -> Heads * HeadDim (Simplified: No Q compression for now)
            w_q: Linear::new(dim, full_head_dim, false, seed + 3),
            
            // W_KR: Dim -> Heads * RoPE_Dim (Usually RoPE is shared or per head?)
            // DeepSeek: RoPE part is per head.
            w_kr: Linear::new(dim, num_heads * rope_dim, false, seed + 4),
            
            // W_QR: Dim -> Heads * RoPE_Dim
            w_qr: Linear::new(dim, num_heads * rope_dim, false, seed + 5),
            
            // W_O: Heads * HeadDim -> Dim
            w_o: Linear::new(full_head_dim, dim, false, seed + 6),
            
            rope: RoPE::new(rope_dim, config.max_seq_len),
        }
    }
    
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.w_dkv.parameters());
        p.extend(self.w_uk.parameters());
        p.extend(self.w_uv.parameters());
        p.extend(self.w_q.parameters());
        p.extend(self.w_kr.parameters());
        p.extend(self.w_qr.parameters());
        p.extend(self.w_o.parameters());
        p
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: [Batch, Seq, Dim]
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.v_head_dim;
        let rope_dim = self.config.qk_rope_dim;
        
        // 1. Compress KV
        let c_kv = self.w_dkv.forward(x); // [Batch, Seq, KV_Latent]
        
        // 2. Decompress to Heads (Content Part)
        let k_c = self.w_uk.forward(&c_kv); // [Batch, Seq, Heads * HeadDim]
        let v_c = self.w_uv.forward(&c_kv); // [Batch, Seq, Heads * HeadDim]
        
        // 3. Generate RoPE Parts
        let k_r_flat = self.w_kr.forward(x); // [Batch, Seq, Heads * RoPE_Dim]
        
        // 4. Generate Query
        let q_c = self.w_q.forward(x); // [Batch, Seq, Heads * HeadDim]
        let q_r_flat = self.w_qr.forward(x); // [Batch, Seq, Heads * RoPE_Dim]
        
        // 5. Reshape and Apply RoPE
        // We need to reshape to [Batch, Seq, Heads, RoPE_Dim] to apply RoPE correctly
        let k_r = k_r_flat.reshape(vec![batch_size, seq_len, num_heads, rope_dim]);
        let q_r = q_r_flat.reshape(vec![batch_size, seq_len, num_heads, rope_dim]);
        
        let k_r_rot = self.rope.forward(&k_r, 0);
        let q_r_rot = self.rope.forward(&q_r, 0);
        
        // 6. Concatenate Content + RoPE
        // Q = [q_c, q_r_rot], K = [k_c, k_r_rot]
        // But dimensions must align. 
        // q_c is [Batch, Seq, Heads * HeadDim] -> Reshape to [Batch, Seq, Heads, HeadDim]
        let q_c_reshaped = q_c.reshape(vec![batch_size, seq_len, num_heads, head_dim]);
        let k_c_reshaped = k_c.reshape(vec![batch_size, seq_len, num_heads, head_dim]);
        
        // We need to concatenate along the last dimension.
        // autograd.rs doesn't have concat yet. 
        // Implementing simple concatenation:
        let q = self.concat_last_dim(&q_c_reshaped, &q_r_rot);
        let k = self.concat_last_dim(&k_c_reshaped, &k_r_rot);
        let v = v_c.reshape(vec![batch_size, seq_len, num_heads, head_dim]);
        
        // 7. Attention: Softmax(Q * K^T / sqrt(d)) * V
        // Reshape to [Batch, Heads, Seq, Dim] for matmul
        let q_t = q.transpose(1, 2); // [Batch, Heads, Seq, TotalDim]
        let k_t = k.transpose(1, 2); // [Batch, Heads, Seq, TotalDim]
        let v_t = v.transpose(1, 2); // [Batch, Heads, Seq, HeadDim]
        
        // Scaled Dot Product
        // We need [Batch, Heads, Seq, Seq]
        let bh = batch_size * num_heads;
        let total_dim = head_dim + rope_dim;
        let head_dim_v = head_dim;
        
        // Flatten batch*heads for manual matrix mul: [BH, Seq, Dim]
        let q_flat = q_t.reshape(vec![bh, seq_len, total_dim]);
        let k_flat = k_t.reshape(vec![bh, seq_len, total_dim]);
        let v_flat = v_t.reshape(vec![bh, seq_len, head_dim_v]);
        
        // Manual Attention Loop
        // Since autograd matmul is 2D, we iterate over BH dimension
        // Output will be [BH, Seq, HeadDim]
        
        // To do this efficiently with current autograd structure (which doesn't support slicing easily),
        // we might be stuck. 
        // BUT, our matmul supports [M, K] * [K, N].
        // Here we have BH independent multiplications.
        // If we can't slice, we can't do it.
        // Let's implement a 'batched_matmul' in autograd or simulate it here by splitting.
        // For now, let's assume we can add a 'split' method or similar.
        // Or, simpler: Just use a custom loop with data access.
        // But we need to preserve the graph.
        // The only way to preserve graph with current simple autograd is to implement a new BatchedMatMul op.
        // OR, since this is a "pair programming" task and we are simulating,
        // we can implement a simplified "chunked" attention.
        
        // Let's implement a helper in autograd for batched matmul if possible.
        // Or, use the fact that Tensor stores a flat Vec.
        // We can implement `batched_matmul` inside `Tensor`?
        // Let's check `autograd.rs` again. It has `matmul`.
        // Let's add `batched_matmul` to `autograd.rs`!
        
        // For now, to unblock, I will add `batched_matmul` to Tensor in `autograd.rs` 
        // and then use it here.
        
        // Placeholder until batched_matmul is available:
        // Assume `batched_matmul` exists on Tensor.
        // q: [BH, Seq, D], k: [BH, Seq, D] -> [BH, Seq, Seq]
        // k_t: [BH, D, Seq] (we need to transpose the last two dims of k_flat)
        // Actually k_flat is [BH, Seq, D]. We need per-batch transpose.
        // Tensor `transpose` works on global dims. 
        // If we reshape k_flat to [BH * Seq, D] and transpose? No.
        
        // OK, critical path: We need a `batched_matmul` that takes [B, M, K] and [B, K, N] -> [B, M, N].
        // And `batched_transpose`?
        
        // Let's implement a `batched_matmul` in transformer.rs as a local helper that 
        // manually constructs the result data and backward pass.
        // This is complex but correct.
        
        let att_scores = self.batched_matmul_qt_k(&q_flat, &k_flat, bh, seq_len, total_dim);
        
        // Scale
        let scale = 1.0 / (total_dim as f64).sqrt();
        let att_scores_scaled = self.scale_tensor(&att_scores, scale);
        
        // Softmax (Manual along last dim)
        let att_probs = self.softmax(&att_scores_scaled, seq_len);
        
        // Output: probs [BH, Seq, Seq] * v [BH, Seq, DimV] -> [BH, Seq, DimV]
        let att_out_flat = self.batched_matmul_probs_v(&att_probs, &v_flat, bh, seq_len, head_dim_v);
        
        // Reshape back to [Batch, Seq, Heads * HeadDim]
        // [BH, Seq, DimV] -> [Batch, Heads, Seq, DimV] -> [Batch, Seq, Heads, DimV] -> [Batch, Seq, Heads*DimV]
        let att_out_reshaped = att_out_flat.reshape(vec![batch_size, num_heads, seq_len, head_dim_v]);
        let att_out_transposed = att_out_reshaped.transpose(1, 2); // [Batch, Seq, Heads, DimV]
        let final_out = att_out_transposed.reshape(vec![batch_size, seq_len, num_heads * head_dim]);
        
        self.w_o.forward(&final_out)
    }
    
    // Helper: Batched MatMul Q * K^T -> Scores [B, Seq, Seq]
    fn batched_matmul_qt_k(&self, q: &Tensor, k: &Tensor, b: usize, seq: usize, dim: usize) -> Tensor {
        // q: [B, Seq, Dim], k: [B, Seq, Dim]
        // out: [B, Seq, Seq]
        // out[b, i, j] = sum_d (q[b, i, d] * k[b, j, d])
        
        let q_data = q.data.read().unwrap();
        let k_data = k.data.read().unwrap();
        let mut out_data = Vec::with_capacity(b * seq * seq);
        
        for batch_idx in 0..b {
            let base_q = batch_idx * seq * dim;
            let base_k = batch_idx * seq * dim;
            
            for i in 0..seq {
                for j in 0..seq {
                    let mut sum = 0.0;
                    for d in 0..dim {
                        sum += q_data[base_q + i * dim + d] * k_data[base_k + j * dim + d];
                    }
                    out_data.push(sum);
                }
            }
        }
        
        // Backward pass implementation
        let parents = vec![q.clone(), k.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; b * seq * seq])),
            shape: vec![b, seq, seq],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let q_in = &parents[0];
                    let k_in = &parents[1];
                    let q_data = q_in.data.read().unwrap();
                    let k_data = k_in.data.read().unwrap();
                    
                    let mut q_grad = q_in.grad.write().unwrap();
                    let mut k_grad = k_in.grad.write().unwrap();
                    
                    for batch_idx in 0..b {
                        let base_q = batch_idx * seq * dim;
                        let base_k = batch_idx * seq * dim;
                        let base_grad = batch_idx * seq * seq;
                        
                        // dL/dQ = grad_out * K
                        // dL/dK = grad_out^T * Q (careful with indices)
                        
                        for i in 0..seq {
                            for j in 0..seq {
                                let g = grad_out[base_grad + i * seq + j];
                                for d in 0..dim {
                                    // dL/dQ[b, i, d] += g * K[b, j, d]
                                    q_grad[base_q + i * dim + d] += g * k_data[base_k + j * dim + d];
                                    
                                    // dL/dK[b, j, d] += g * Q[b, i, d]
                                    k_grad[base_k + j * dim + d] += g * q_data[base_q + i * dim + d];
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }

    // Helper: Batched MatMul Probs * V -> Out [B, Seq, DimV]
    fn batched_matmul_probs_v(&self, probs: &Tensor, v: &Tensor, b: usize, seq: usize, dim_v: usize) -> Tensor {
        // probs: [B, Seq, Seq], v: [B, Seq, DimV]
        // out[b, i, d] = sum_j (probs[b, i, j] * v[b, j, d])
        
        let p_data = probs.data.read().unwrap();
        let v_data = v.data.read().unwrap();
        let mut out_data = Vec::with_capacity(b * seq * dim_v);
        
        for batch_idx in 0..b {
            let base_p = batch_idx * seq * seq;
            let base_v = batch_idx * seq * dim_v;
            
            for i in 0..seq {
                for d in 0..dim_v {
                    let mut sum = 0.0;
                    for j in 0..seq {
                        sum += p_data[base_p + i * seq + j] * v_data[base_v + j * dim_v + d];
                    }
                    out_data.push(sum);
                }
            }
        }
        
        // Backward pass implementation
        let parents = vec![probs.clone(), v.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(out_data)),
            grad: Arc::new(RwLock::new(vec![0.0; b * seq * dim_v])),
            shape: vec![b, seq, dim_v],
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let p_in = &parents[0];
                    let v_in = &parents[1];
                    let p_data = p_in.data.read().unwrap();
                    let v_data = v_in.data.read().unwrap();
                    
                    let mut p_grad = p_in.grad.write().unwrap();
                    let mut v_grad = v_in.grad.write().unwrap();
                    
                    for batch_idx in 0..b {
                        let base_p = batch_idx * seq * seq;
                        let base_v = batch_idx * seq * dim_v;
                        let base_grad = batch_idx * seq * dim_v;
                        
                        for i in 0..seq {
                            for d in 0..dim_v {
                                let g = grad_out[base_grad + i * dim_v + d];
                                for j in 0..seq {
                                    // dL/dP[b, i, j] += g * V[b, j, d]
                                    p_grad[base_p + i * seq + j] += g * v_data[base_v + j * dim_v + d];
                                    
                                    // dL/dV[b, j, d] += g * P[b, i, j]
                                    v_grad[base_v + j * dim_v + d] += g * p_data[base_p + i * seq + j];
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }

    
    fn scale_tensor(&self, t: &Tensor, scale: f64) -> Tensor {
        let data = t.data.read().unwrap();
        let new_data: Vec<f64> = data.iter().map(|&x| x * scale).collect();
        
        let parents = vec![t.clone()];
        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; data.len()])),
            shape: t.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    for (i, &g) in grad_out.iter().enumerate() {
                        inp_grad[i] += g * scale;
                    }
                }),
            })),
        }
    }
    
    fn softmax(&self, t: &Tensor, seq_len: usize) -> Tensor {
        // t: [B, Seq, Seq]
        // Softmax along last dimension
        let data = t.data.read().unwrap();
        let mut new_data = Vec::with_capacity(data.len());
        let total_rows = data.len() / seq_len;
        
        for r in 0..total_rows {
            let start = r * seq_len;
            let end = start + seq_len;
            let row = &data[start..end];
            
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            let mut exps = Vec::with_capacity(seq_len);
            
            for &x in row {
                let e = (x - max_val).exp();
                exps.push(e);
                sum_exp += e;
            }
            
            for e in exps {
                new_data.push(e / sum_exp);
            }
        }
        
        let parents = vec![t.clone()];
        let out_data_clone = new_data.clone();
        
        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; data.len()])),
            shape: t.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    let out_data = &out_data_clone;
                    
                    for r in 0..total_rows {
                        let start = r * seq_len;
                        let end = start + seq_len;
                        
                        let mut sum_gy = 0.0;
                        for i in start..end {
                            sum_gy += grad_out[i] * out_data[i];
                        }
                        
                        for i in start..end {
                            let yi = out_data[i];
                            let gi = grad_out[i];
                            inp_grad[i] += yi * (gi - sum_gy);
                        }
                    }
                }),
            })),
        }
    }
    
    fn concat_last_dim(&self, a: &Tensor, b: &Tensor) -> Tensor {
        // Naive concat along last dim
        let shape_a = &a.shape;
        let shape_b = &b.shape;
        let last_dim_a = shape_a[shape_a.len()-1];
        let last_dim_b = shape_b[shape_b.len()-1];
        let batch_dims = &shape_a[..shape_a.len()-1];
        
        let a_data = a.data.read().unwrap();
        let b_data = b.data.read().unwrap();
        
        let total_elements = batch_dims.iter().product::<usize>();
        let mut new_data = Vec::with_capacity(total_elements * (last_dim_a + last_dim_b));
        
        for i in 0..total_elements {
            let start_a = i * last_dim_a;
            let start_b = i * last_dim_b;
            new_data.extend_from_slice(&a_data[start_a..start_a + last_dim_a]);
            new_data.extend_from_slice(&b_data[start_b..start_b + last_dim_b]);
        }
        
        let mut new_shape = batch_dims.to_vec();
        new_shape.push(last_dim_a + last_dim_b);
        
        let parents = vec![a.clone(), b.clone()];
        
        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; total_elements * (last_dim_a + last_dim_b)])),
            shape: new_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut a_grad = parents[0].grad.write().unwrap();
                    let mut b_grad = parents[1].grad.write().unwrap();
                    
                    let stride = last_dim_a + last_dim_b;
                    
                    for i in 0..total_elements {
                        let base_out = i * stride;
                        let base_a = i * last_dim_a;
                        let base_b = i * last_dim_b;
                        
                        for k in 0..last_dim_a {
                            a_grad[base_a + k] += grad_out[base_out + k];
                        }
                        for k in 0..last_dim_b {
                            b_grad[base_b + k] += grad_out[base_out + last_dim_a + k];
                        }
                    }
                }),
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tensor;

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_reshaped = t.reshape(vec![4]);
        assert_eq!(t_reshaped.shape, vec![4]);
        
        // In this simple autograd implementation, reshape creates a copy.
        // So we verify that the copy has the correct data initially.
        let data_reshaped = t_reshaped.data.read().unwrap();
        assert_eq!(data_reshaped[0], 1.0);
        
        // And verify independence if we modify original
        {
            let mut data = t.data.write().unwrap();
            data[0] = 10.0;
        }
        let data_reshaped_after = t_reshaped.data.read().unwrap();
        assert_eq!(data_reshaped_after[0], 1.0); // Should still be 1.0
    }

    #[test]
    fn test_tensor_transpose() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // [[1, 2, 3], [4, 5, 6]]
        let t_t = t.transpose(0, 1);
        // [[1, 4], [2, 5], [3, 6]]
        assert_eq!(t_t.shape, vec![3, 2]);
        let data = t_t.data.read().unwrap();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 4.0);
        assert_eq!(data[2], 2.0);
        assert_eq!(data[3], 5.0);
        assert_eq!(data[4], 3.0);
        assert_eq!(data[5], 6.0);
    }

    #[test]
    fn test_mla_forward_backward() {
        let config = MLAConfig {
            dim: 16,
            num_heads: 2,
            q_lora_rank: 0,
            kv_lora_rank: 8,
            qk_rope_dim: 4,
            v_head_dim: 8,
            max_seq_len: 10,
        };
        
        let mla = MultiHeadLatentAttention::new(config, 42);
        
        // Input: [Batch=1, Seq=3, Dim=16]
        let input = Tensor::rand(vec![1, 3, 16], -0.1, 0.1, 123);
        
        // Forward
        let output = mla.forward(&input);
        assert_eq!(output.shape, vec![1, 3, 16]); // Should match input dim
        
        // Backward
        // We need a scalar loss to backprop
        let loss = output.mean();
        loss.backward();
        
        // Check if gradients propagated to weights
        // W_DKV is the first projection
        let w_dkv_grad = mla.w_dkv.w.grad.read().unwrap();
        let grad_sum: f64 = w_dkv_grad.iter().sum();
        assert!(grad_sum.abs() > 0.0, "Gradient should not be zero");
        
        // Check W_UK
        let w_uk_grad = mla.w_uk.w.grad.read().unwrap();
        let grad_sum_uk: f64 = w_uk_grad.iter().sum();
        assert!(grad_sum_uk.abs() > 0.0, "Gradient for W_UK should not be zero");
    }

    #[test]
    fn test_rope_backward() {
        let dim = 4;
        let rope = RoPE::new(dim, 10);
        let x = Tensor::rand(vec![1, 2, 4], -1.0, 1.0, 123);
        
        let out = rope.forward(&x, 0);
        let loss = out.sum();
        loss.backward();
        
        // Verify x.grad is not zero
        let x_grad = x.grad.read().unwrap();
        assert!(x_grad.iter().any(|&g| g.abs() > 1e-6));
    }

    #[test]
    fn test_rmsnorm_backward() {
        let dim = 4;
        let norm = RMSNorm::new(dim, 1e-5, 123);
        let x = Tensor::rand(vec![2, 4], 0.1, 1.0, 123); // Positive inputs to avoid 0 div just in case
        
        let out = norm.forward(&x);
        let loss = out.sum();
        loss.backward();
        
        let x_grad = x.grad.read().unwrap();
        assert!(x_grad.iter().any(|&g| g.abs() > 1e-6));
        
        let w_grad = norm.weight.grad.read().unwrap();
        // Weight init is 1.0. Gradient should flow.
        assert!(w_grad.iter().any(|&g| g.abs() > 1e-6));
    }
    
    #[test]
    fn test_luck_transformer_integration() {
        // Use small dims for test
        let t = LuckTransformer::new(8, 8, true, 42);
        let x = Tensor::rand(vec![1, 5, 8], -0.1, 0.1, 123);
        
        let out = t.forward(&x, &[]);
        let loss = out.mean();
        loss.backward();
        
        // Check params count
        let params = t.parameters();
        assert!(params.len() > 10, "Should have many parameters");
        
        // Check if Embed gradients exist
        let embed_grad = t.embed.w.grad.read().unwrap();
        assert!(embed_grad.iter().any(|&g| g.abs() > 0.0), "Embed grad missing");
        
        // Check Norm grad
        let norm_grad = t.norm_1.weight.grad.read().unwrap();
        assert!(norm_grad.iter().any(|&g| g.abs() > 0.0), "Norm grad missing");
    }
}
