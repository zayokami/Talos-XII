use crate::autograd::{Context, Tensor};
use crate::nn::{Linear, Module, RMSNorm};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

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
    #[allow(dead_code)]
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
}

impl Module for LuckTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, &[]) // Default usage without pity for Module trait
    }

    fn parameters(&self) -> Vec<Tensor> {
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
        let mut rope = Self {
            dim,
            base: 10000.0,
            cos_cache: vec![],
            sin_cache: vec![],
        };
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

                    let r1 = x_data[base_idx + 2 * i];
                    let r2 = x_data[base_idx + 2 * i + 1];

                    out_data[base_idx + 2 * i] = r1 * c - r2 * s;
                    out_data[base_idx + 2 * i + 1] = r1 * s + r2 * c;
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

                                let g1 = grad_out[base_idx + 2 * i];
                                let g2 = grad_out[base_idx + 2 * i + 1];

                                // dL/dx1 = g1 * c + g2 * s
                                // dL/dx2 = -g1 * s + g2 * c

                                inp_grad[base_idx + 2 * i] += g1 * c + g2 * s;
                                inp_grad[base_idx + 2 * i + 1] += -g1 * s + g2 * c;
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

        // Compress KV into latent space
        let c_kv = self.w_dkv.forward(x); // [Batch, Seq, KV_Latent]

        // Decompress to Heads (Content Part)
        let k_c = self.w_uk.forward(&c_kv); // [Batch, Seq, Heads * HeadDim]
        let v_c = self.w_uv.forward(&c_kv); // [Batch, Seq, Heads * HeadDim]

        // Generate RoPE Parts
        let k_r_flat = self.w_kr.forward(x); // [Batch, Seq, Heads * RoPE_Dim]

        // Generate Query
        let q_c = self.w_q.forward(x); // [Batch, Seq, Heads * HeadDim]
        let q_r_flat = self.w_qr.forward(x); // [Batch, Seq, Heads * RoPE_Dim]

        // Reshape and Apply RoPE
        // Need to reshape to [Batch, Seq, Heads, RoPE_Dim] for rotation
        let k_r = k_r_flat.reshape(vec![batch_size, seq_len, num_heads, rope_dim]);
        let q_r = q_r_flat.reshape(vec![batch_size, seq_len, num_heads, rope_dim]);

        let k_r_rot = self.rope.forward(&k_r, 0);
        let q_r_rot = self.rope.forward(&q_r, 0);

        // Concatenate Content + RoPE
        // Q = [q_c, q_r_rot], K = [k_c, k_r_rot]
        // Reshape first to align dimensions
        let q_c_reshaped = q_c.reshape(vec![batch_size, seq_len, num_heads, head_dim]);
        let k_c_reshaped = k_c.reshape(vec![batch_size, seq_len, num_heads, head_dim]);

        // Hack: autograd lacks concat, so we use a custom helper here
        let q = self.concat_last_dim(&q_c_reshaped, &q_r_rot);
        let k = self.concat_last_dim(&k_c_reshaped, &k_r_rot);
        let v = v_c.reshape(vec![batch_size, seq_len, num_heads, head_dim]);

        // Attention: Softmax(Q * K^T / sqrt(d)) * V
        // Reshape to [Batch, Heads, Seq, Dim] for matmul
        let q_t = q.transpose(1, 2); // [Batch, Heads, Seq, TotalDim]
        let k_t = k.transpose(1, 2); // [Batch, Heads, Seq, TotalDim]
        let v_t = v.transpose(1, 2); // [Batch, Heads, Seq, HeadDim]

        // Scaled Dot Product
        let bh = batch_size * num_heads;
        let total_dim = head_dim + rope_dim;
        let head_dim_v = head_dim;

        // Flatten batch*heads for manual matrix mul: [BH, Seq, Dim]
        let q_flat = q_t.reshape(vec![bh, seq_len, total_dim]);
        let k_flat = k_t.reshape(vec![bh, seq_len, total_dim]);
        let v_flat = v_t.reshape(vec![bh, seq_len, head_dim_v]);

        // We need batched matmul (BH independent multiplications).
        // Since our core Autograd only supports 2D matmul, we use a custom implementation here.
        // It's a bit ugly but gets the job done for now.

        let att_scores = self.batched_matmul_qt_k(&q_flat, &k_flat, bh, seq_len, total_dim);

        // Scale
        let scale = 1.0 / (total_dim as f64).sqrt();
        let att_scores_scaled = self.scale_tensor(&att_scores, scale);

        // Softmax (along last dim)
        let att_probs = self.softmax(&att_scores_scaled, seq_len);

        // Output: probs [BH, Seq, Seq] * v [BH, Seq, DimV] -> [BH, Seq, DimV]
        let att_out_flat =
            self.batched_matmul_probs_v(&att_probs, &v_flat, bh, seq_len, head_dim_v);

        // Reshape back to [Batch, Seq, Heads * HeadDim]
        // [BH, Seq, DimV] -> [Batch, Heads, Seq, DimV] -> [Batch, Seq, Heads, DimV] -> [Batch, Seq, Heads*DimV]
        let att_out_reshaped =
            att_out_flat.reshape(vec![batch_size, num_heads, seq_len, head_dim_v]);
        let att_out_transposed = att_out_reshaped.transpose(1, 2); // [Batch, Seq, Heads, DimV]
        let final_out = att_out_transposed.reshape(vec![batch_size, seq_len, num_heads * head_dim]);

        self.w_o.forward(&final_out)
    }

    // Helper: Batched MatMul Q * K^T -> Scores [B, Seq, Seq]
    fn batched_matmul_qt_k(
        &self,
        q: &Tensor,
        k: &Tensor,
        b: usize,
        seq: usize,
        dim: usize,
    ) -> Tensor {
        // q: [B, Seq, Dim], k: [B, Seq, Dim]
        // out: [B, Seq, Seq]
        // out[b, i, j] = sum_d (q[b, i, d] * k[b, j, d])

        let q_data = q.data.read().unwrap();
        let k_data = k.data.read().unwrap();

        let out_data: Vec<f64> = (0..b)
            .into_par_iter()
            .flat_map_iter(|batch_idx| {
                let base_q = batch_idx * seq * dim;
                let base_k = batch_idx * seq * dim;
                let mut batch_out = Vec::with_capacity(seq * seq);

                for i in 0..seq {
                    for j in 0..seq {
                        let mut sum = 0.0;
                        for d in 0..dim {
                            sum += q_data[base_q + i * dim + d] * k_data[base_k + j * dim + d];
                        }
                        batch_out.push(sum);
                    }
                }
                batch_out
            })
            .collect();

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

                    let chunk_size_grad = seq * dim;
                    let chunk_size_out = seq * seq;

                    q_grad
                        .par_chunks_mut(chunk_size_grad)
                        .zip(k_grad.par_chunks_mut(chunk_size_grad))
                        .zip(grad_out.par_chunks(chunk_size_out))
                        .enumerate()
                        .for_each(|(batch_idx, ((q_g_chunk, k_g_chunk), g_out_chunk))| {
                            let base_data = batch_idx * seq * dim;
                            let q_slice = &q_data[base_data..base_data + chunk_size_grad];
                            let k_slice = &k_data[base_data..base_data + chunk_size_grad];

                            for i in 0..seq {
                                for j in 0..seq {
                                    let g = g_out_chunk[i * seq + j];
                                    for d in 0..dim {
                                        // dL/dQ[b, i, d] += g * K[b, j, d]
                                        q_g_chunk[i * dim + d] += g * k_slice[j * dim + d];

                                        // dL/dK[b, j, d] += g * Q[b, i, d]
                                        k_g_chunk[j * dim + d] += g * q_slice[i * dim + d];
                                    }
                                }
                            }
                        });
                }),
            })),
        }
    }

    // Batched MatMul: probs * v -> out
    fn batched_matmul_probs_v(
        &self,
        probs: &Tensor,
        v: &Tensor,
        b: usize,
        seq: usize,
        dim_v: usize,
    ) -> Tensor {
        // probs: [B, Seq, Seq], v: [B, Seq, DimV]
        // out[b, i, d] = sum_j (probs[b, i, j] * v[b, j, d])

        let p_data = probs.data.read().unwrap();
        let v_data = v.data.read().unwrap();

        let out_data: Vec<f64> = (0..b)
            .into_par_iter()
            .flat_map_iter(|batch_idx| {
                let base_p = batch_idx * seq * seq;
                let base_v = batch_idx * seq * dim_v;
                let mut batch_out = Vec::with_capacity(seq * dim_v);

                for i in 0..seq {
                    for d in 0..dim_v {
                        let mut sum = 0.0;
                        for j in 0..seq {
                            sum += p_data[base_p + i * seq + j] * v_data[base_v + j * dim_v + d];
                        }
                        batch_out.push(sum);
                    }
                }
                batch_out
            })
            .collect();

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

                    let chunk_size_p = seq * seq;
                    let chunk_size_v = seq * dim_v;
                    let chunk_size_grad = seq * dim_v;

                    p_grad
                        .par_chunks_mut(chunk_size_p)
                        .zip(v_grad.par_chunks_mut(chunk_size_v))
                        .zip(grad_out.par_chunks(chunk_size_grad))
                        .enumerate()
                        .for_each(|(batch_idx, ((p_g_chunk, v_g_chunk), g_out_chunk))| {
                            let base_p = batch_idx * seq * seq;
                            let base_v = batch_idx * seq * dim_v;
                            let p_slice = &p_data[base_p..base_p + chunk_size_p];
                            let v_slice = &v_data[base_v..base_v + chunk_size_v];

                            for i in 0..seq {
                                for d in 0..dim_v {
                                    let g = g_out_chunk[i * dim_v + d];
                                    for j in 0..seq {
                                        // dL/dP[b, i, j] += g * V[b, j, d]
                                        p_g_chunk[i * seq + j] += g * v_slice[j * dim_v + d];

                                        // dL/dV[b, j, d] += g * P[b, i, j]
                                        v_g_chunk[j * dim_v + d] += g * p_slice[i * seq + j];
                                    }
                                }
                            }
                        });
                }),
            })),
        }
    }

    fn scale_tensor(&self, t: &Tensor, scale: f64) -> Tensor {
        let data = t.data.read().unwrap();
        let new_data: Vec<f64> = data.par_iter().map(|&x| x * scale).collect();

        let parents = vec![t.clone()];
        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![0.0; data.len()])),
            shape: t.shape.clone(),
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut inp_grad = parents[0].grad.write().unwrap();
                    inp_grad
                        .par_iter_mut()
                        .zip(grad_out.par_iter())
                        .for_each(|(ig, &g)| *ig += g * scale);
                }),
            })),
        }
    }

    fn softmax(&self, t: &Tensor, seq_len: usize) -> Tensor {
        // t: [B, Seq, Seq]
        // Softmax along last dimension
        let data = t.data.read().unwrap();

        let new_data: Vec<f64> = data
            .par_chunks(seq_len)
            .flat_map_iter(|row| {
                let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exps: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f64 = exps.iter().sum();
                exps.into_iter().map(move |e| e / sum_exp)
            })
            .collect();

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

                    inp_grad
                        .par_chunks_mut(seq_len)
                        .zip(grad_out.par_chunks(seq_len))
                        .zip(out_data.par_chunks(seq_len))
                        .for_each(|((ig_row, g_row), out_row)| {
                            let mut sum_gy = 0.0;
                            for i in 0..seq_len {
                                sum_gy += g_row[i] * out_row[i];
                            }

                            for i in 0..seq_len {
                                let yi = out_row[i];
                                let gi = g_row[i];
                                ig_row[i] += yi * (gi - sum_gy);
                            }
                        });
                }),
            })),
        }
    }

    fn concat_last_dim(&self, a: &Tensor, b: &Tensor) -> Tensor {
        // Concatenate tensors. This is slow because it allocates.
        let shape_a = &a.shape;
        let shape_b = &b.shape;
        let last_dim_a = shape_a[shape_a.len() - 1];
        let last_dim_b = shape_b[shape_b.len() - 1];
        let batch_dims = &shape_a[..shape_a.len() - 1];

        let a_data = a.data.read().unwrap();
        let b_data = b.data.read().unwrap();

        let total_elements = batch_dims.iter().product::<usize>();
        let mut new_data = vec![0.0; total_elements * (last_dim_a + last_dim_b)];

        new_data
            .par_chunks_mut(last_dim_a + last_dim_b)
            .enumerate()
            .for_each(|(i, chunk)| {
                let start_a = i * last_dim_a;
                let start_b = i * last_dim_b;

                chunk[0..last_dim_a].copy_from_slice(&a_data[start_a..start_a + last_dim_a]);
                chunk[last_dim_a..].copy_from_slice(&b_data[start_b..start_b + last_dim_b]);
            });

        let mut new_shape = batch_dims.to_vec();
        new_shape.push(last_dim_a + last_dim_b);

        let parents = vec![a.clone(), b.clone()];

        Tensor {
            data: Arc::new(RwLock::new(new_data)),
            grad: Arc::new(RwLock::new(vec![
                0.0;
                total_elements * (last_dim_a + last_dim_b)
            ])),
            shape: new_shape,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let mut a_grad = parents[0].grad.write().unwrap();
                    let mut b_grad = parents[1].grad.write().unwrap();

                    let stride = last_dim_a + last_dim_b;

                    a_grad
                        .par_chunks_mut(last_dim_a)
                        .zip(b_grad.par_chunks_mut(last_dim_b))
                        .zip(grad_out.par_chunks(stride))
                        .for_each(|((ag_row, bg_row), g_row)| {
                            for k in 0..last_dim_a {
                                ag_row[k] += g_row[k];
                            }
                            for k in 0..last_dim_b {
                                bg_row[k] += g_row[last_dim_a + k];
                            }
                        });
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
        let w_dkv_grad = mla.w_dkv.weight.grad.read().unwrap();
        let grad_sum: f64 = w_dkv_grad.iter().sum();
        assert!(grad_sum.abs() > 0.0, "Gradient should not be zero");

        // Check W_UK
        let w_uk_grad = mla.w_uk.weight.grad.read().unwrap();
        let grad_sum_uk: f64 = w_uk_grad.iter().sum();
        assert!(
            grad_sum_uk.abs() > 0.0,
            "Gradient for W_UK should not be zero"
        );
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
        let embed_grad = t.embed.weight.grad.read().unwrap();
        assert!(
            embed_grad.iter().any(|&g: &f64| g.abs() > 0.0),
            "Embed grad missing"
        );

        // Check Norm grad
        let norm_grad = t.norm_1.weight.grad.read().unwrap();
        assert!(
            norm_grad.iter().any(|&g| g.abs() > 0.0),
            "Norm grad missing"
        );
    }
}
