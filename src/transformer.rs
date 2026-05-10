use crate::achf::{aggregate_cache_stats_iter, AchfCacheStats, AchfLayer};
use crate::autograd::{Context, Tensor, TensorReadGuard};
use crate::config::{AchfConfig, Config};
use crate::dtype::{Dtype, Storage};
use crate::nn::{Linear, Module, RMSNorm};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
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

    pub fn from_config(config: &Config) -> Self {
        let num_heads = config.model_num_heads.max(1);
        Self {
            dim: config.model_hidden_dim,
            num_heads,
            q_lora_rank: 0,
            kv_lora_rank: config.model_kv_lora_rank,
            qk_rope_dim: config.model_qk_rope_dim,
            v_head_dim: config.model_hidden_dim / num_heads,
            max_seq_len: 256,
        }
    }
}

// --- MhcResidual: Multi-stream residual expansion (n x C) with doubly-stochastic mixing ---
#[derive(Clone, Serialize, Deserialize)]
pub struct MhcResidual {
    pub h_pre: Vec<Linear>,
    pub h_res: Linear,
    pub h_post: Vec<Linear>,
    pub n: usize,
}

impl MhcResidual {
    pub fn new(dim: usize, n: usize, seed: u64) -> Self {
        let mut h_pre = Vec::with_capacity(n);
        let mut h_post = Vec::with_capacity(n);
        for i in 0..n {
            h_pre.push(Linear::new(dim, dim, false, seed.wrapping_add(i as u64)));
            h_post.push(Linear::new(
                dim,
                dim,
                false,
                seed.wrapping_add(100 + i as u64),
            ));
        }
        let h_res = Linear::new(dim * n, dim * n, false, seed.wrapping_add(200));
        {
            let mut w = h_res.weight.data_write_f64();
            // Sinkhorn-Knopp requires strictly positive entries
            for v in w.iter_mut() {
                *v = v.abs();
            }
            crate::achf::sinkhorn_project(&mut w, dim * n, dim * n, 20, None, None);
        }
        Self {
            h_pre,
            h_res,
            h_post,
            n,
        }
    }

    /// Tensor-based forward (training). x shape: [..., dim].
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Expand to n streams
        let streams: Vec<Tensor> = self.h_pre.iter().map(|pre| pre.forward(x)).collect();
        // 2. Concatenate along last dim: [..., n*dim]
        let concat = concat_last_dim(&streams);
        // 3. Apply H_res mixing
        let mixed = self.h_res.forward(&concat);
        // 4. Split back to n streams
        let mixed_streams = split_last_dim(&mixed, self.n);
        // 5. Apply h_post to each stream and sum
        let mut out = Tensor::zeros(x.shape.clone());
        for (post, ms) in self.h_post.iter().zip(mixed_streams.iter()) {
            out = out + post.forward(ms);
        }
        out
    }

    #[allow(clippy::needless_range_loop)]
    /// Vec-based forward (inference). x: flat [num_positions * dim].
    pub fn forward_inference(&self, x: &[f64]) -> Vec<f64> {
        let dim = self.h_pre[0].in_features;
        let num_positions = x.len() / dim;
        let n = self.n;
        // 1. Expand: n streams
        let mut streams: Vec<Vec<f64>> = Vec::with_capacity(n);
        for pre in &self.h_pre {
            streams.push(pre.forward_inference(x));
        }
        // 2. Concatenate along last dim
        let mut concat = vec![0.0; num_positions * n * dim];
        for pos in 0..num_positions {
            for i in 0..n {
                let src_offset = pos * dim;
                let dst_offset = pos * n * dim + i * dim;
                concat[dst_offset..dst_offset + dim]
                    .copy_from_slice(&streams[i][src_offset..src_offset + dim]);
            }
        }
        // 3. Apply H_res
        let mixed = self.h_res.forward_inference(&concat);
        // 4. Split back to n streams
        let mut mixed_streams: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut ms = vec![0.0; num_positions * dim];
            for pos in 0..num_positions {
                let src_offset = pos * n * dim + i * dim;
                let dst_offset = pos * dim;
                ms[dst_offset..dst_offset + dim]
                    .copy_from_slice(&mixed[src_offset..src_offset + dim]);
            }
            mixed_streams.push(ms);
        }
        // 5. Apply h_post and sum
        let mut out = vec![0.0; x.len()];
        for (post, ms) in self.h_post.iter().zip(mixed_streams.iter()) {
            let post_out = post.forward_inference(ms);
            for i in 0..out.len() {
                out[i] += post_out[i];
            }
        }
        out
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for pre in &self.h_pre {
            p.extend(pre.parameters());
        }
        p.extend(self.h_res.parameters());
        for post in &self.h_post {
            p.extend(post.parameters());
        }
        p
    }

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        for pre in &mut self.h_pre {
            pre.to_cuda();
        }
        self.h_res.to_cuda();
        for post in &mut self.h_post {
            post.to_cuda();
        }
    }
}

/// Concatenate n tensors along the last dimension.
/// All tensors must share the same shape except the last dim.
fn concat_last_dim(tensors: &[Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "concat_last_dim: empty input");
    let first_shape = &tensors[0].shape;
    let prefix_len: usize = first_shape[..first_shape.len() - 1].iter().product();
    let last_dim = first_shape[first_shape.len() - 1];
    let n = tensors.len();
    let out_last_dim = last_dim * n;
    let total = prefix_len * out_last_dim;

    let mut out_data = vec![0.0; total];
    for t in tensors.iter() {
        assert_eq!(
            &t.shape[..t.shape.len() - 1],
            &first_shape[..first_shape.len() - 1],
            "concat_last_dim: prefix shapes must match"
        );
        assert_eq!(
            t.shape[t.shape.len() - 1],
            last_dim,
            "concat_last_dim: last dim must match"
        );
    }

    for p in 0..prefix_len {
        for (i, t) in tensors.iter().enumerate() {
            let t_data = t.data_f64();
            let src_start = p * last_dim;
            let dst_start = p * out_last_dim + i * last_dim;
            out_data[dst_start..dst_start + last_dim]
                .copy_from_slice(&t_data[src_start..src_start + last_dim]);
        }
    }

    let mut out_shape = first_shape.clone();
    *out_shape.last_mut().unwrap() = out_last_dim;
    Tensor::new(out_data, out_shape)
}

/// Split a tensor along the last dimension into n equal parts.
fn split_last_dim(tensor: &Tensor, n: usize) -> Vec<Tensor> {
    assert!(n > 0, "split_last_dim: n must be > 0");
    let shape = &tensor.shape;
    let prefix_len: usize = shape[..shape.len() - 1].iter().product();
    let last_dim = shape[shape.len() - 1];
    assert_eq!(
        last_dim % n,
        0,
        "split_last_dim: last dim {} not divisible by {}",
        last_dim,
        n
    );
    let split_dim = last_dim / n;
    let t_data = tensor.data_f64();

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut chunk_data = vec![0.0; prefix_len * split_dim];
        for p in 0..prefix_len {
            let src_start = p * last_dim + i * split_dim;
            let dst_start = p * split_dim;
            chunk_data[dst_start..dst_start + split_dim]
                .copy_from_slice(&t_data[src_start..src_start + split_dim]);
        }
        let mut chunk_shape = shape.clone();
        *chunk_shape.last_mut().unwrap() = split_dim;
        out.push(Tensor::new(chunk_data, chunk_shape));
    }
    out
}

// --- LuckTransformer (Transformer Backbone with MLA) ---
#[derive(Clone, Serialize, Deserialize)]
pub struct TransformerBlock {
    pub norm_1: RMSNorm,
    pub mla_layer: MultiHeadLatentAttention,
    pub mhc: Option<MhcResidual>,
    pub norm_2: RMSNorm,
    pub ffn_1: Linear,
    pub ffn_2: Linear,
    pub achf_ffn: Option<AchfLayer>,
}

#[derive(Default)]
struct TransformerStepScratch {
    h: Vec<f64>,
    norm1: Vec<f64>,
    attn: Vec<f64>,
    norm2: Vec<f64>,
    ffn1: Vec<f64>,
    ffn2: Vec<f64>,
    mhc_scratch: Vec<f64>,
}

thread_local! {
    static TRANSFORMER_STEP_SCRATCH: RefCell<TransformerStepScratch> =
        RefCell::new(TransformerStepScratch::default());
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LuckTransformer {
    pub embed: Linear,
    pub blocks: Vec<TransformerBlock>,
    pub norm_final: RMSNorm,
    pub out_proj: Linear,
}

impl LuckTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        _bias: bool,
        num_layers: usize,
        seed: u64,
        achf: &AchfConfig,
        mla_config: &MLAConfig,
        use_mhc: bool,
        mhc_factor: usize,
    ) -> Self {
        let num_layers = if num_layers == 0 { 2 } else { num_layers };
        let mut blocks = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let layer_seed = seed.wrapping_add((layer_idx as u64).wrapping_mul(200));
            let mhc = if use_mhc && mhc_factor > 1 {
                Some(MhcResidual::new(
                    hidden_dim,
                    mhc_factor,
                    layer_seed.wrapping_add(1000),
                ))
            } else {
                None
            };
            let achf_ffn = if achf.enabled && achf.apply_ffn {
                Some(AchfLayer::new(
                    hidden_dim * 2,
                    hidden_dim,
                    true,
                    achf.clone(),
                    layer_seed.wrapping_add(1100),
                ))
            } else {
                None
            };
            blocks.push(TransformerBlock {
                norm_1: RMSNorm::new(hidden_dim, 1e-5, layer_seed + 5),
                mla_layer: MultiHeadLatentAttention::new(mla_config.clone(), layer_seed + 10),
                mhc,
                norm_2: RMSNorm::new(hidden_dim, 1e-5, layer_seed + 15),
                ffn_1: Linear::new(hidden_dim, hidden_dim * 2, true, layer_seed + 20),
                ffn_2: Linear::new(hidden_dim * 2, hidden_dim, true, layer_seed + 30),
                achf_ffn,
            });
        }

        Self {
            embed: Linear::new(in_dim, hidden_dim, true, seed),
            blocks,
            norm_final: RMSNorm::new(hidden_dim, 1e-5, seed + 35),
            out_proj: Linear::new(hidden_dim, hidden_dim, true, seed + 40),
        }
    }

    /// Convenience constructor that builds MLAConfig from a global Config.
    pub fn new_with_config(config: &Config, seed: u64) -> Self {
        let mla = MLAConfig::from_config(config);
        Self::new(
            config.model_dim,
            config.model_hidden_dim,
            true,
            config.model_num_layers,
            seed,
            &config.achf,
            &mla,
            config.use_multi_stream,
            config.multi_stream_factor,
        )
    }

    /// Backward-compatible wrapper that uses hard-coded defaults for MLAConfig.
    #[allow(dead_code)]
    pub fn new_compat(
        in_dim: usize,
        hidden_dim: usize,
        bias: bool,
        num_layers: usize,
        seed: u64,
        achf: &AchfConfig,
    ) -> Self {
        let mla_config = MLAConfig {
            dim: hidden_dim,
            num_heads: 8,
            q_lora_rank: 0,
            kv_lora_rank: 128,
            qk_rope_dim: 64,
            v_head_dim: hidden_dim / 8,
            max_seq_len: 256,
        };
        Self::new(
            in_dim,
            hidden_dim,
            bias,
            num_layers,
            seed,
            achf,
            &mla_config,
            false,
            4,
        )
    }

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        self.embed.to_cuda();
        for block in &mut self.blocks {
            block.norm_1.to_cuda();
            block.mla_layer.to_cuda();
            if let Some(ref mut mhc) = block.mhc {
                mhc.to_cuda();
            }
            block.norm_2.to_cuda();
            block.ffn_1.to_cuda();
            block.ffn_2.to_cuda();
            if let Some(ref mut achf) = block.achf_ffn {
                achf.to_cuda();
            }
        }
        self.norm_final.to_cuda();
        self.out_proj.to_cuda();
    }

    pub fn forward(&self, x: &Tensor, _pity: &[usize]) -> Tensor {
        // x: [Batch, Seq, Dim]
        // Embed
        let mut h = self.embed.forward(x);

        for block in &self.blocks {
            // Block 1: MLA (Pre-Norm)
            let h_norm1 = block.norm_1.forward(&h);
            let attn_out = block.mla_layer.forward(&h_norm1);
            let h2 = if let Some(mhc) = &block.mhc {
                // mHC multi-stream residual replaces standard residual
                &h + &mhc.forward(&attn_out)
            } else {
                h.clone() + attn_out
            };

            // Block 2: FFN (Pre-Norm)
            let h_norm2 = block.norm_2.forward(&h2);
            let f1 = block.ffn_1.forward(&h_norm2).gelu();
            let f2 = if let Some(achf) = &block.achf_ffn {
                achf.forward(&f1)
            } else {
                block.ffn_2.forward(&f1)
            };
            let h3 = h2.clone() + f2;
            h = h3;
        }

        // Final Norm + Output
        let h_final = self.norm_final.forward(&h);
        self.out_proj.forward(&h_final)
    }

    pub fn last_token(&self, x: &Tensor) -> Tensor {
        // x: [Batch, Seq, Dim]
        // Return [Batch, Dim] (last token)
        let shape = &x.shape;
        let batch_size = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];

        let x_data = x.data_f64();
        let mut out_data = Vec::with_capacity(batch_size * dim);

        for b in 0..batch_size {
            let start = b * seq_len * dim + (seq_len - 1) * dim;
            out_data.extend_from_slice(&x_data[start..start + dim]);
        }

        Tensor::new(out_data, vec![batch_size, dim])
    }

    pub fn update_achf_after_backward(&self) {
        for block in &self.blocks {
            if let Some(achf) = &block.achf_ffn {
                achf.update_after_backward();
            }
        }
    }

    pub fn freeze_achf_for_inference(&self) {
        for block in &self.blocks {
            if let Some(achf) = &block.achf_ffn {
                achf.freeze_for_inference();
            }
        }
    }

    pub fn snapshot_achf(&self) -> Option<crate::achf::AchfStateSnapshot> {
        for block in &self.blocks {
            if let Some(achf) = &block.achf_ffn {
                return Some(achf.snapshot_state());
            }
        }
        None
    }

    /// Run inference forcing a specific ACHF path (0=Cached, 1=Sparse, 2=Dense).
    pub fn forward_inference_forced_path(&self, x: &[f64], forced_path: u8) -> Vec<f64> {
        use crate::simd::vector_gelu;
        let mut h = self.embed.forward_inference(x);
        for block in &self.blocks {
            let h_norm1 = block.norm_1.forward_inference(&h);
            let attn_out = block.mla_layer.forward_inference(&h_norm1);
            let mut h2 = vec![0.0; h.len()];
            if let Some(mhc) = &block.mhc {
                let mhc_out = mhc.forward_inference(&attn_out);
                for i in 0..h.len() {
                    h2[i] = h[i] + mhc_out[i];
                }
            } else {
                for i in 0..h.len() {
                    h2[i] = h[i] + attn_out[i];
                }
            }
            let h_norm2 = block.norm_2.forward_inference(&h2);
            let f1 = block.ffn_1.forward_inference(&h_norm2);
            let mut f1_gelu = vec![0.0; f1.len()];
            vector_gelu(&mut f1_gelu, &f1);
            let f2 = if let Some(achf) = &block.achf_ffn {
                achf.forward_inference_forced_path(&f1_gelu, forced_path)
            } else {
                block.ffn_2.forward_inference(&f1_gelu)
            };
            let mut h3 = vec![0.0; h2.len()];
            for i in 0..h2.len() {
                h3[i] = h2[i] + f2[i];
            }
            h = h3;
        }
        h
    }

    pub fn achf_cache_stats_iter(&self) -> impl Iterator<Item = AchfCacheStats> + '_ {
        self.blocks
            .iter()
            .flat_map(|block| block.achf_ffn.as_ref().map(|achf| achf.cache_stats()))
    }

    pub fn achf_cache_stats_aggregate(&self) -> AchfCacheStats {
        aggregate_cache_stats_iter(self.achf_cache_stats_iter())
    }

    pub fn achf_orthogonal_penalty(&self) -> Option<Tensor> {
        let mut reg: Option<Tensor> = None;
        for block in &self.blocks {
            if let Some(achf) = &block.achf_ffn {
                if let Some(val) = achf.orthogonal_penalty() {
                    reg = Some(match reg {
                        Some(r) => r + val,
                        None => val,
                    });
                }
            }
        }
        reg
    }

    pub fn forward_inference(&self, x: &[f64]) -> Vec<f64> {
        use crate::simd::{vector_add, vector_gelu};

        let mut h = self.embed.forward_inference(x);

        for block in &self.blocks {
            let h_norm1 = block.norm_1.forward_inference(&h);
            let attn_out = block.mla_layer.forward_inference(&h_norm1);

            let mut h2 = vec![0.0; h.len()];
            if let Some(mhc) = &block.mhc {
                let mhc_out = mhc.forward_inference(&attn_out);
                vector_add(&mut h2, &h, &mhc_out);
            } else {
                vector_add(&mut h2, &h, &attn_out);
            }

            let h_norm2 = block.norm_2.forward_inference(&h2);
            let f1 = block.ffn_1.forward_inference(&h_norm2);

            let mut f1_gelu = vec![0.0; f1.len()];
            vector_gelu(&mut f1_gelu, &f1);

            let f2 = if let Some(achf) = &block.achf_ffn {
                achf.forward_inference_residual(&f1_gelu)
            } else {
                block.ffn_2.forward_inference(&f1_gelu)
            };

            let mut h3 = vec![0.0; h2.len()];
            vector_add(&mut h3, &h2, &f2);
            h = h3;
        }

        let h_final = self.norm_final.forward_inference(&h);
        self.out_proj.forward_inference(&h_final)
    }

    pub fn last_token_inference(&self, x: &[f64]) -> Vec<f64> {
        let dim = self.out_proj.out_features;
        let seq_len = x.len() / dim;
        let start = (seq_len.saturating_sub(1)) * dim;
        x[start..start + dim].to_vec()
    }

    #[allow(dead_code)]
    pub fn forward_inference_step(
        &self,
        x: &[f64],
        kv_caches: &mut [KVCache],
        start_pos: usize,
    ) -> Vec<f64> {
        let mut out = Vec::new();
        self.forward_inference_step_into(x, kv_caches, start_pos, &mut out);
        out
    }

    pub fn forward_inference_step_into(
        &self,
        x: &[f64],
        kv_caches: &mut [KVCache],
        start_pos: usize,
        out: &mut Vec<f64>,
    ) {
        use crate::simd::{vector_gelu, vector_grad_acc};

        TRANSFORMER_STEP_SCRATCH.with(|scratch_cell| {
            let mut scratch = scratch_cell.borrow_mut();
            let TransformerStepScratch {
                h,
                norm1,
                attn,
                norm2,
                ffn1,
                ffn2,
                mhc_scratch,
            } = &mut *scratch;

            self.embed.forward_inference_into(x, h);

            let layer_count = self.blocks.len().min(kv_caches.len());
            for (i, kv_cache) in kv_caches.iter_mut().enumerate().take(layer_count) {
                let block = &self.blocks[i];
                block.norm_1.forward_inference_into(h, norm1);
                block
                    .mla_layer
                    .forward_inference_cached_into(norm1, kv_cache, start_pos, attn);

                if let Some(mhc) = &block.mhc {
                    *mhc_scratch = mhc.forward_inference(attn);
                    vector_grad_acc(h, mhc_scratch);
                } else {
                    vector_grad_acc(h, attn);
                }

                block.norm_2.forward_inference_into(h, norm2);
                block.ffn_1.forward_inference_into(norm2, ffn1);
                ffn2.resize(ffn1.len(), 0.0);
                vector_gelu(ffn2, ffn1);

                if let Some(achf) = &block.achf_ffn {
                    let achf_out = achf.forward_inference_residual(ffn2);
                    vector_grad_acc(h, &achf_out);
                } else {
                    block.ffn_2.forward_inference_into(ffn2, attn);
                    vector_grad_acc(h, attn);
                }
            }

            self.norm_final.forward_inference_into(h, norm1);
            self.out_proj.forward_inference_into(norm1, out);
        })
    }

    pub fn max_seq_len(&self) -> usize {
        self.blocks
            .first()
            .map(|b| b.mla_layer.config.max_seq_len)
            .unwrap_or(256)
    }

    pub fn prune_kv_cache(&self, kv_caches: &mut [KVCache], max_seq_len: usize) {
        let layer_count = self.blocks.len().min(kv_caches.len());
        for (i, kv_cache) in kv_caches.iter_mut().enumerate().take(layer_count) {
            self.blocks[i]
                .mla_layer
                .prune_kv_cache(kv_cache, max_seq_len);
        }
    }
}

impl Module for LuckTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input, &[]) // Default usage without pity for Module trait
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.embed.parameters();
        for block in &self.blocks {
            p.extend(block.norm_1.parameters());
            p.extend(block.mla_layer.parameters());
            if let Some(mhc) = &block.mhc {
                p.extend(mhc.parameters());
            }
            p.extend(block.norm_2.parameters());
            p.extend(block.ffn_1.parameters());
            p.extend(block.ffn_2.parameters());
            if let Some(achf) = &block.achf_ffn {
                p.extend(achf.parameters());
            }
        }
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
    pub cos_cache: Arc<Vec<f64>>,
    pub sin_cache: Arc<Vec<f64>>,
}

impl RoPE {
    pub fn new(dim: usize, max_len: usize) -> Self {
        let base: f64 = 10000.0;
        let half = dim / 2;
        let mut cos_cache = Vec::with_capacity(max_len * half);
        let mut sin_cache = Vec::with_capacity(max_len * half);

        for pos in 0..max_len {
            for i in 0..half {
                let theta = 1.0 / base.powf((2 * i) as f64 / dim as f64);
                let angle = pos as f64 * theta;
                cos_cache.push(angle.cos());
                sin_cache.push(angle.sin());
            }
        }

        Self {
            dim,
            base,
            cos_cache: Arc::new(cos_cache),
            sin_cache: Arc::new(sin_cache),
        }
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        // x: [Batch, Seq, Heads, Dim] or [Batch, Seq, Dim]
        // We assume x is [..., Seq, HeadDim]
        // RoPE applies to the last dimension.

        let shape = &x.shape;
        let dim = shape[shape.len() - 1];
        assert_eq!(dim, self.dim);
        let seq_len = shape[shape.len() - 2]; // Assumes ..., Seq, Dim
        let num_elements = x.data_f64().len();
        let total_batches = num_elements / (seq_len * dim);

        // GPU path: if tensor is on CUDA, use the optimized kernel
        #[cfg(cuda)]
        if x.device == crate::autograd::Device::Cuda {
            return x.rope_cuda(
                &self.cos_cache,
                &self.sin_cache,
                seq_len,
                dim,
                total_batches,
                start_pos,
            );
        }

        // CPU path
        let x_data = x.data_f64();
        let mut out_data = x_data.clone(); // Copy

        // Apply rotation
        // This is a naive CPU implementation

        for b in 0..total_batches {
            for t in 0..seq_len {
                let pos = start_pos + t;
                if pos * (self.dim / 2) >= self.cos_cache.len() {
                    continue;
                }
                let cache_idx = pos * (self.dim / 2);

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
        let cos_cache = Arc::clone(&self.cos_cache);
        let sin_cache = Arc::clone(&self.sin_cache);
        let dim = self.dim;
        let start_pos_cap = start_pos;

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(num_elements, Tensor::grad_dtype_for(Dtype::F64)),
            shape: shape.clone(),
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let mut inp_grad = input.grad_write_f64();
                    let shape = &input.shape;

                    let seq_len = shape[shape.len() - 2];
                    let total_batches = inp_grad.len() / (seq_len * dim);

                    for b in 0..total_batches {
                        for t in 0..seq_len {
                            let pos = start_pos_cap + t;
                            if pos * (dim / 2) >= cos_cache.len() {
                                continue;
                            }
                            let cache_idx = pos * (dim / 2);
                            let base_idx = b * (seq_len * dim) + t * dim;

                            for i in 0..dim / 2 {
                                let c = cos_cache[cache_idx + i];
                                let s = sin_cache[cache_idx + i];

                                let g1 = grad_out_f64[base_idx + 2 * i];
                                let g2 = grad_out_f64[base_idx + 2 * i + 1];

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

    #[allow(dead_code)]
    pub fn forward_inference(&self, x: &[f64], seq_len: usize, start_pos: usize) -> Vec<f64> {
        let dim = self.dim;
        let num_elements = x.len();
        let mut out = x.to_vec();

        let total_batches = num_elements / (seq_len * dim);

        for b in 0..total_batches {
            for t in 0..seq_len {
                let pos = start_pos + t;
                // Safety check for cache bounds
                if pos * (dim / 2) >= self.cos_cache.len() {
                    continue;
                }
                let cache_idx = pos * (dim / 2);
                let base_idx = b * (seq_len * dim) + t * dim;

                for i in 0..dim / 2 {
                    let c = self.cos_cache[cache_idx + i];
                    let s = self.sin_cache[cache_idx + i];

                    let r1 = x[base_idx + 2 * i];
                    let r2 = x[base_idx + 2 * i + 1];

                    out[base_idx + 2 * i] = r1 * c - r2 * s;
                    out[base_idx + 2 * i + 1] = r1 * s + r2 * c;
                }
            }
        }
        out
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

#[derive(Clone, Debug, Default)]
pub struct KVCache {
    // k_cache: [num_heads][seq_len * total_head_dim]
    pub k_cache: Vec<Vec<f64>>,
    // v_cache: [num_heads][seq_len * head_dim]
    pub v_cache: Vec<Vec<f64>>,
    scratch_scores: Vec<f64>,
    scratch_att_out: Vec<f64>,
    scratch_c_kv: Vec<f64>,
    scratch_k_c: Vec<f64>,
    scratch_v_c: Vec<f64>,
    scratch_k_r: Vec<f64>,
    scratch_q_r: Vec<f64>,
    scratch_q_c: Vec<f64>,
}

impl KVCache {
    pub fn new(num_heads: usize) -> Self {
        Self {
            k_cache: vec![Vec::new(); num_heads],
            v_cache: vec![Vec::new(); num_heads],
            scratch_scores: Vec::new(),
            scratch_att_out: Vec::new(),
            scratch_c_kv: Vec::new(),
            scratch_k_c: Vec::new(),
            scratch_v_c: Vec::new(),
            scratch_k_r: Vec::new(),
            scratch_q_r: Vec::new(),
            scratch_q_c: Vec::new(),
        }
    }

    /// Pre-allocate scratch buffers for inference to avoid reallocation overhead.
    /// Call this after construction where MLA config is available.
    pub fn preallocate(
        &mut self,
        num_heads: usize,
        kv_lora_rank: usize,
        v_head_dim: usize,
        qk_rope_dim: usize,
        max_seq_len: usize,
    ) {
        let k_cache_target = max_seq_len * (v_head_dim + qk_rope_dim);
        let v_cache_target = max_seq_len * v_head_dim;
        for k in &mut self.k_cache {
            k.reserve(k_cache_target);
        }
        for v in &mut self.v_cache {
            v.reserve(v_cache_target);
        }
        self.scratch_scores.reserve(max_seq_len * num_heads);
        self.scratch_att_out.reserve(num_heads * v_head_dim);
        self.scratch_c_kv.reserve(kv_lora_rank);
        self.scratch_k_c.reserve(num_heads * v_head_dim);
        self.scratch_v_c.reserve(num_heads * v_head_dim);
        self.scratch_k_r.reserve(num_heads * qk_rope_dim);
        self.scratch_q_r.reserve(num_heads * qk_rope_dim);
        self.scratch_q_c.reserve(num_heads * v_head_dim);
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for h in self.k_cache.iter_mut() {
            h.clear();
        }
        for h in self.v_cache.iter_mut() {
            h.clear();
        }
        self.scratch_scores.clear();
        self.scratch_att_out.clear();
        self.scratch_c_kv.clear();
        self.scratch_k_c.clear();
        self.scratch_v_c.clear();
        self.scratch_k_r.clear();
        self.scratch_q_r.clear();
        self.scratch_q_c.clear();
    }
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

    #[cfg(cuda)]
    pub fn to_cuda(&mut self) {
        self.w_dkv.to_cuda();
        self.w_uk.to_cuda();
        self.w_uv.to_cuda();
        self.w_q.to_cuda();
        self.w_kr.to_cuda();
        self.w_qr.to_cuda();
        self.w_o.to_cuda();
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

        // Causal mask: prevent attending to future positions
        let att_scores_masked = self.apply_causal_mask(&att_scores_scaled, seq_len);

        // Softmax (along last dim)
        let att_probs = self.softmax(&att_scores_masked, seq_len);

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

    pub fn forward_inference(&self, x: &[f64]) -> Vec<f64> {
        use crate::simd::{add_scaled_row, dot_product};
        use rayon::prelude::*;

        let dim = self.config.dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.v_head_dim;
        let rope_dim = self.config.qk_rope_dim;

        let num_elements = x.len();
        let seq_len = num_elements / dim;

        let c_kv = self.w_dkv.forward_inference(x);
        let k_c = self.w_uk.forward_inference(&c_kv);
        let v_c = self.w_uv.forward_inference(&c_kv);
        let k_r_flat = self.w_kr.forward_inference(x);
        let q_c = self.w_q.forward_inference(x);
        let q_r_flat = self.w_qr.forward_inference(x);

        let mut k_r_rot = k_r_flat.clone();
        let mut q_r_rot = q_r_flat.clone();

        for t in 0..seq_len {
            let pos = t;
            let cache_offset = pos * (rope_dim / 2);
            if cache_offset >= self.rope.cos_cache.len() {
                continue;
            }

            for h in 0..num_heads {
                let base = t * (num_heads * rope_dim) + h * rope_dim;
                for i in 0..rope_dim / 2 {
                    let c = self.rope.cos_cache[cache_offset + i];
                    let s = self.rope.sin_cache[cache_offset + i];

                    let idx1 = base + 2 * i;
                    let idx2 = base + 2 * i + 1;

                    let r1 = k_r_flat[idx1];
                    let r2 = k_r_flat[idx2];
                    k_r_rot[idx1] = r1 * c - r2 * s;
                    k_r_rot[idx2] = r1 * s + r2 * c;

                    let q1 = q_r_flat[idx1];
                    let q2 = q_r_flat[idx2];
                    q_r_rot[idx1] = q1 * c - q2 * s;
                    q_r_rot[idx2] = q1 * s + q2 * c;
                }
            }
        }

        let total_head_dim = head_dim + rope_dim;
        let mut q = vec![0.0; seq_len * num_heads * total_head_dim];
        let mut k = vec![0.0; seq_len * num_heads * total_head_dim];

        for t in 0..seq_len {
            for h in 0..num_heads {
                let dst_base = t * (num_heads * total_head_dim) + h * total_head_dim;

                let src_c_base = t * (num_heads * head_dim) + h * head_dim;
                q[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_c[src_c_base..src_c_base + head_dim]);
                k[dst_base..dst_base + head_dim]
                    .copy_from_slice(&k_c[src_c_base..src_c_base + head_dim]);

                let src_r_base = t * (num_heads * rope_dim) + h * rope_dim;
                let dst_r_base = dst_base + head_dim;
                q[dst_r_base..dst_r_base + rope_dim]
                    .copy_from_slice(&q_r_rot[src_r_base..src_r_base + rope_dim]);
                k[dst_r_base..dst_r_base + rope_dim]
                    .copy_from_slice(&k_r_rot[src_r_base..src_r_base + rope_dim]);
            }
        }

        let head_stride = seq_len * seq_len;
        let mut att_scores = vec![0.0; num_heads * head_stride];
        let scale = 1.0 / (total_head_dim as f64).sqrt();

        att_scores
            .par_chunks_mut(head_stride)
            .enumerate()
            .for_each(|(h, head_scores)| {
                for i in 0..seq_len {
                    let base_q = i * (num_heads * total_head_dim) + h * total_head_dim;
                    let q_slice = &q[base_q..base_q + total_head_dim];

                    for j in 0..seq_len {
                        let base_k = j * (num_heads * total_head_dim) + h * total_head_dim;
                        let k_slice = &k[base_k..base_k + total_head_dim];
                        head_scores[i * seq_len + j] = dot_product(q_slice, k_slice) * scale;
                    }

                    for j in (i + 1)..seq_len {
                        head_scores[i * seq_len + j] = f64::NEG_INFINITY;
                    }

                    let row = &mut head_scores[i * seq_len..(i + 1) * seq_len];
                    let sum = crate::simd::softmax_exp_sum(row);
                    crate::simd::vector_scale(row, 1.0 / sum);
                }
            });

        let mut att_out = vec![0.0; seq_len * num_heads * head_dim];

        let per_head_out: Vec<Vec<f64>> = (0..num_heads)
            .into_par_iter()
            .map(|h| {
                let mut head_out = vec![0.0; seq_len * head_dim];
                for i in 0..seq_len {
                    let out_slice = &mut head_out[i * head_dim..(i + 1) * head_dim];
                    for j in 0..seq_len {
                        let score = att_scores[h * head_stride + i * seq_len + j];
                        if score == 0.0 {
                            continue;
                        }
                        let base_v = j * (num_heads * head_dim) + h * head_dim;
                        let v_slice = &v_c[base_v..base_v + head_dim];
                        add_scaled_row(out_slice, v_slice, score);
                    }
                }
                head_out
            })
            .collect();

        for (h, head_buf) in per_head_out.iter().enumerate() {
            for i in 0..seq_len {
                let dst = i * (num_heads * head_dim) + h * head_dim;
                let src = i * head_dim;
                att_out[dst..dst + head_dim].copy_from_slice(&head_buf[src..src + head_dim]);
            }
        }

        self.w_o.forward_inference(&att_out)
    }

    // --- KV Cache Support ---
    #[allow(dead_code)]
    pub fn forward_inference_cached(
        &self,
        x: &[f64],
        kv_cache: &mut KVCache,
        start_pos: usize,
    ) -> Vec<f64> {
        let mut out = Vec::new();
        self.forward_inference_cached_into(x, kv_cache, start_pos, &mut out);
        out
    }

    pub fn forward_inference_cached_into(
        &self,
        x: &[f64],
        kv_cache: &mut KVCache,
        start_pos: usize,
        out: &mut Vec<f64>,
    ) {
        use crate::simd::{add_scaled_row, dot_product};

        let dim = self.config.dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.v_head_dim;
        let rope_dim = self.config.qk_rope_dim;

        // x is expected to be a single token or a short sequence
        let num_elements = x.len();
        let seq_len = num_elements / dim;

        if seq_len == 1 {
            self.forward_inference_cached_single_token_into(x, kv_cache, start_pos, out);
            return;
        }

        // 1. Projections
        let c_kv = self.w_dkv.forward_inference(x);
        let k_c = self.w_uk.forward_inference(&c_kv);
        let v_c = self.w_uv.forward_inference(&c_kv);
        let k_r_flat = self.w_kr.forward_inference(x);
        let q_c = self.w_q.forward_inference(x);
        let q_r_flat = self.w_qr.forward_inference(x);

        // 2. RoPE Rotation (in-place)
        let mut k_r_rot = k_r_flat;
        let mut q_r_rot = q_r_flat;

        for t in 0..seq_len {
            let pos = start_pos + t;
            let cache_offset = pos * (rope_dim / 2);
            if cache_offset >= self.rope.cos_cache.len() {
                continue;
            }

            for h in 0..num_heads {
                let base = t * (num_heads * rope_dim) + h * rope_dim;
                for i in 0..rope_dim / 2 {
                    let c = self.rope.cos_cache[cache_offset + i];
                    let s = self.rope.sin_cache[cache_offset + i];

                    let idx1 = base + 2 * i;
                    let idx2 = base + 2 * i + 1;

                    let r1 = k_r_rot[idx1];
                    let r2 = k_r_rot[idx2];
                    k_r_rot[idx1] = r1 * c - r2 * s;
                    k_r_rot[idx2] = r1 * s + r2 * c;

                    let q1 = q_r_rot[idx1];
                    let q2 = q_r_rot[idx2];
                    q_r_rot[idx1] = q1 * c - q2 * s;
                    q_r_rot[idx2] = q1 * s + q2 * c;
                }
            }
        }

        // 3. Assemble Q and K for current tokens
        let total_head_dim = head_dim + rope_dim;
        let mut q = vec![0.0; seq_len * num_heads * total_head_dim];
        let mut k = vec![0.0; seq_len * num_heads * total_head_dim];

        if seq_len > 0 {
            let target_k = self.config.max_seq_len * total_head_dim;
            let target_v = self.config.max_seq_len * head_dim;
            for h in 0..num_heads {
                let k_cache = &mut kv_cache.k_cache[h];
                if k_cache.is_empty() {
                    let cap = k_cache.capacity();
                    if cap < target_k {
                        k_cache.reserve(target_k - cap);
                    }
                }
                let v_cache = &mut kv_cache.v_cache[h];
                if v_cache.is_empty() {
                    let cap = v_cache.capacity();
                    if cap < target_v {
                        v_cache.reserve(target_v - cap);
                    }
                }
            }
        }

        for t in 0..seq_len {
            for h in 0..num_heads {
                let dst_base = t * (num_heads * total_head_dim) + h * total_head_dim;

                let src_c_base = t * (num_heads * head_dim) + h * head_dim;
                q[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_c[src_c_base..src_c_base + head_dim]);
                k[dst_base..dst_base + head_dim]
                    .copy_from_slice(&k_c[src_c_base..src_c_base + head_dim]);

                let src_r_base = t * (num_heads * rope_dim) + h * rope_dim;
                let dst_r_base = dst_base + head_dim;
                q[dst_r_base..dst_r_base + rope_dim]
                    .copy_from_slice(&q_r_rot[src_r_base..src_r_base + rope_dim]);
                k[dst_r_base..dst_r_base + rope_dim]
                    .copy_from_slice(&k_r_rot[src_r_base..src_r_base + rope_dim]);
            }
        }

        // 4. Update KV Cache
        // Append current k and v_c to cache
        // v_c needs reshaping to [seq_len, num_heads, head_dim] logic
        for t in 0..seq_len {
            for h in 0..num_heads {
                let k_start = t * (num_heads * total_head_dim) + h * total_head_dim;
                let k_slice = &k[k_start..k_start + total_head_dim];
                kv_cache.k_cache[h].extend_from_slice(k_slice);

                let v_start = t * (num_heads * head_dim) + h * head_dim;
                let v_slice = &v_c[v_start..v_start + head_dim];
                kv_cache.v_cache[h].extend_from_slice(v_slice);
            }
        }

        // 5. Attention with Cache
        // Q: [seq_len, num_heads, total_head_dim]
        // K_cache: [num_heads, cached_len + seq_len, total_head_dim]
        // V_cache: [num_heads, cached_len + seq_len, head_dim]

        let cached_len = kv_cache.k_cache[0].len() / total_head_dim;
        let head_stride = seq_len * cached_len;
        let mut att_scores = vec![0.0; num_heads * head_stride];
        let scale = 1.0 / (total_head_dim as f64).sqrt();

        let use_par = true;

        if use_par {
            use rayon::prelude::*;

            let k_caches: Vec<&[f64]> = kv_cache.k_cache.iter().map(|v| v.as_slice()).collect();

            att_scores
                .par_chunks_mut(head_stride)
                .enumerate()
                .for_each(|(h, head_scores)| {
                    let k_cache_head = k_caches[h];
                    for i in 0..seq_len {
                        let q_start = i * (num_heads * total_head_dim) + h * total_head_dim;
                        let q_vec = &q[q_start..q_start + total_head_dim];

                        for j in 0..cached_len {
                            let k_vec = &k_cache_head[j * total_head_dim..(j + 1) * total_head_dim];
                            head_scores[i * cached_len + j] = dot_product(q_vec, k_vec) * scale;
                        }

                        let abs_pos = start_pos + i;
                        for j in (abs_pos + 1)..cached_len {
                            head_scores[i * cached_len + j] = f64::NEG_INFINITY;
                        }

                        let row = &mut head_scores[i * cached_len..(i + 1) * cached_len];
                        let sum = crate::simd::softmax_exp_sum(row);
                        crate::simd::vector_scale(row, 1.0 / sum);
                    }
                });

            let v_caches: Vec<&[f64]> = kv_cache.v_cache.iter().map(|v| v.as_slice()).collect();

            let per_head_out: Vec<Vec<f64>> = (0..num_heads)
                .into_par_iter()
                .map(|h| {
                    let v_cache_head = v_caches[h];
                    let mut head_out = vec![0.0; seq_len * head_dim];
                    for i in 0..seq_len {
                        let out_slice = &mut head_out[i * head_dim..(i + 1) * head_dim];
                        for j in 0..cached_len {
                            let score = att_scores[h * head_stride + i * cached_len + j];
                            if score.abs() < 1e-9 {
                                continue;
                            }
                            let v_vec = &v_cache_head[j * head_dim..(j + 1) * head_dim];
                            add_scaled_row(out_slice, v_vec, score);
                        }
                    }
                    head_out
                })
                .collect();

            let mut att_out = vec![0.0; seq_len * num_heads * head_dim];
            for (h, head_buf) in per_head_out.iter().enumerate() {
                for i in 0..seq_len {
                    let dst = i * (num_heads * head_dim) + h * head_dim;
                    let src = i * head_dim;
                    att_out[dst..dst + head_dim].copy_from_slice(&head_buf[src..src + head_dim]);
                }
            }
            self.w_o.forward_inference_into(&att_out, out);
        } else {
            for h in 0..num_heads {
                let k_cache_head = &kv_cache.k_cache[h];
                for i in 0..seq_len {
                    let q_start = i * (num_heads * total_head_dim) + h * total_head_dim;
                    let q_vec = &q[q_start..q_start + total_head_dim];

                    for j in 0..cached_len {
                        let k_vec = &k_cache_head[j * total_head_dim..(j + 1) * total_head_dim];
                        att_scores[h * head_stride + i * cached_len + j] =
                            dot_product(q_vec, k_vec) * scale;
                    }

                    let abs_pos = start_pos + i;
                    for j in (abs_pos + 1)..cached_len {
                        att_scores[h * head_stride + i * cached_len + j] = f64::NEG_INFINITY;
                    }

                    let start = h * head_stride + i * cached_len;
                    let end = start + cached_len;
                    let slice = &mut att_scores[start..end];
                    let sum = crate::simd::softmax_exp_sum(slice);
                    crate::simd::vector_scale(slice, 1.0 / sum);
                }
            }

            let mut att_out = vec![0.0; seq_len * num_heads * head_dim];
            for h in 0..num_heads {
                let v_cache_head = &kv_cache.v_cache[h];
                for i in 0..seq_len {
                    let out_start = i * (num_heads * head_dim) + h * head_dim;
                    let out_slice = &mut att_out[out_start..out_start + head_dim];

                    for j in 0..cached_len {
                        let score = att_scores[h * head_stride + i * cached_len + j];
                        if score.abs() < 1e-9 {
                            continue;
                        }
                        let v_vec = &v_cache_head[j * head_dim..(j + 1) * head_dim];
                        add_scaled_row(out_slice, v_vec, score);
                    }
                }
            }
            self.w_o.forward_inference_into(&att_out, out);
        }
    }

    fn forward_inference_cached_single_token_into(
        &self,
        x: &[f64],
        kv_cache: &mut KVCache,
        start_pos: usize,
        out: &mut Vec<f64>,
    ) {
        use crate::simd::{add_scaled_row, dot_product, softmax_exp_sum, vector_scale};

        let dim = self.config.dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.v_head_dim;
        let rope_dim = self.config.qk_rope_dim;
        let total_head_dim = head_dim + rope_dim;

        debug_assert_eq!(x.len(), dim);

        self.w_dkv
            .forward_inference_into(x, &mut kv_cache.scratch_c_kv);
        {
            let c_kv = &kv_cache.scratch_c_kv;
            self.w_uk
                .forward_inference_into(c_kv, &mut kv_cache.scratch_k_c);
            self.w_uv
                .forward_inference_into(c_kv, &mut kv_cache.scratch_v_c);
        }
        self.w_kr
            .forward_inference_into(x, &mut kv_cache.scratch_k_r);
        self.w_qr
            .forward_inference_into(x, &mut kv_cache.scratch_q_r);
        self.w_q
            .forward_inference_into(x, &mut kv_cache.scratch_q_c);

        let cache_offset = start_pos * (rope_dim / 2);
        if cache_offset < self.rope.cos_cache.len() {
            for h in 0..num_heads {
                let base = h * rope_dim;
                for i in 0..rope_dim / 2 {
                    let c = self.rope.cos_cache[cache_offset + i];
                    let s = self.rope.sin_cache[cache_offset + i];

                    let idx1 = base + 2 * i;
                    let idx2 = base + 2 * i + 1;

                    let k1 = kv_cache.scratch_k_r[idx1];
                    let k2 = kv_cache.scratch_k_r[idx2];
                    kv_cache.scratch_k_r[idx1] = k1 * c - k2 * s;
                    kv_cache.scratch_k_r[idx2] = k1 * s + k2 * c;

                    let q1 = kv_cache.scratch_q_r[idx1];
                    let q2 = kv_cache.scratch_q_r[idx2];
                    kv_cache.scratch_q_r[idx1] = q1 * c - q2 * s;
                    kv_cache.scratch_q_r[idx2] = q1 * s + q2 * c;
                }
            }
        }

        let target_k = self.config.max_seq_len * total_head_dim;
        let target_v = self.config.max_seq_len * head_dim;
        for h in 0..num_heads {
            let k_cache = &mut kv_cache.k_cache[h];
            if k_cache.is_empty() {
                let cap = k_cache.capacity();
                if cap < target_k {
                    k_cache.reserve(target_k - cap);
                }
            }
            let v_cache = &mut kv_cache.v_cache[h];
            if v_cache.is_empty() {
                let cap = v_cache.capacity();
                if cap < target_v {
                    v_cache.reserve(target_v - cap);
                }
            }

            let content_base = h * head_dim;
            let rope_base = h * rope_dim;
            k_cache.extend_from_slice(&kv_cache.scratch_k_c[content_base..content_base + head_dim]);
            k_cache.extend_from_slice(&kv_cache.scratch_k_r[rope_base..rope_base + rope_dim]);
            v_cache.extend_from_slice(&kv_cache.scratch_v_c[content_base..content_base + head_dim]);
        }

        let cached_len = kv_cache.k_cache[0].len() / total_head_dim;
        kv_cache.scratch_scores.resize(cached_len, 0.0);
        kv_cache.scratch_att_out.resize(num_heads * head_dim, 0.0);
        kv_cache.scratch_att_out.fill(0.0);

        let scale = 1.0 / (total_head_dim as f64).sqrt();
        let (k_caches, v_caches, scratch_scores, scratch_att_out) = (
            &kv_cache.k_cache,
            &kv_cache.v_cache,
            &mut kv_cache.scratch_scores,
            &mut kv_cache.scratch_att_out,
        );

        for h in 0..num_heads {
            let q_content = &kv_cache.scratch_q_c[h * head_dim..(h + 1) * head_dim];
            let q_rope = &kv_cache.scratch_q_r[h * rope_dim..(h + 1) * rope_dim];
            let k_cache_head = &k_caches[h];

            for (j, score) in scratch_scores.iter_mut().enumerate().take(cached_len) {
                let base = j * total_head_dim;
                let k_content = &k_cache_head[base..base + head_dim];
                let k_rope = &k_cache_head[base + head_dim..base + total_head_dim];
                *score = (dot_product(q_content, k_content) + dot_product(q_rope, k_rope)) * scale;
            }

            for score in scratch_scores.iter_mut().skip(start_pos + 1) {
                *score = f64::NEG_INFINITY;
            }

            let sum = softmax_exp_sum(scratch_scores);
            vector_scale(scratch_scores, 1.0 / sum);

            let out_slice = &mut scratch_att_out[h * head_dim..(h + 1) * head_dim];
            let v_cache_head = &v_caches[h];
            for (j, &score) in scratch_scores.iter().enumerate().take(cached_len) {
                if score.abs() < 1e-9 {
                    continue;
                }
                let v_vec = &v_cache_head[j * head_dim..(j + 1) * head_dim];
                add_scaled_row(out_slice, v_vec, score);
            }
        }

        self.w_o.forward_inference_into(scratch_att_out, out);
    }

    pub fn prune_kv_cache(&self, kv_cache: &mut KVCache, max_seq_len: usize) {
        let head_dim = self.config.v_head_dim;
        let rope_dim = self.config.qk_rope_dim;
        let total_head_dim = head_dim + rope_dim;

        if kv_cache.k_cache.is_empty() {
            return;
        }
        let current_len = kv_cache.k_cache[0].len() / total_head_dim;

        if current_len > max_seq_len {
            let remove_count = current_len - max_seq_len;

            for h in 0..kv_cache.k_cache.len() {
                // Remove from front of K
                let k_remove_elements = remove_count * total_head_dim;
                if k_remove_elements < kv_cache.k_cache[h].len() {
                    kv_cache.k_cache[h].drain(0..k_remove_elements);
                } else {
                    kv_cache.k_cache[h].clear();
                }

                // Remove from front of V
                let v_remove_elements = remove_count * head_dim;
                if v_remove_elements < kv_cache.v_cache[h].len() {
                    kv_cache.v_cache[h].drain(0..v_remove_elements);
                } else {
                    kv_cache.v_cache[h].clear();
                }
            }
        }
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

        // Use batch lock for better performance
        let guards = TensorReadGuard::new(&[q, k]);
        let q_data = guards.get(0);
        let k_data = guards.get(1);

        let out_data: Vec<f64> = (0..b)
            .into_par_iter()
            .flat_map_iter(|batch_idx| {
                use crate::simd::dot_product;
                let base_q = batch_idx * seq * dim;
                let base_k = batch_idx * seq * dim;
                let mut batch_out = Vec::with_capacity(seq * seq);

                for i in 0..seq {
                    for j in 0..seq {
                        let q_slice = &q_data[base_q + i * dim..base_q + i * dim + dim];
                        let k_slice = &k_data[base_k + j * dim..base_k + j * dim + dim];
                        batch_out.push(dot_product(q_slice, k_slice));
                    }
                }
                batch_out
            })
            .collect();

        // Backward pass implementation
        let parents = vec![q.clone(), k.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(b * seq * seq, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![b, seq, seq],
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let q_in = &parents[0];
                    let k_in = &parents[1];
                    // Use batch lock for data read
                    let guards = TensorReadGuard::new(&[q_in, k_in]);
                    let q_data = guards.get(0);
                    let k_data = guards.get(1);

                    let mut q_grad = q_in.grad_write_f64();
                    let mut k_grad = k_in.grad_write_f64();

                    let chunk_size_grad = seq * dim;
                    let chunk_size_out = seq * seq;

                    q_grad
                        .par_chunks_mut(chunk_size_grad)
                        .zip(k_grad.par_chunks_mut(chunk_size_grad))
                        .zip(grad_out_f64.par_chunks(chunk_size_out))
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

        // GPU path: if both tensors are on CUDA, use the optimized kernel
        #[cfg(cuda)]
        if probs.device == crate::autograd::Device::Cuda
            && v.device == crate::autograd::Device::Cuda
        {
            return self.batched_matmul_probs_v_cuda(probs, v, b, seq, dim_v);
        }

        // Use batch lock for better performance
        let guards = TensorReadGuard::new(&[probs, v]);
        let p_data = guards.get(0);
        let v_data = guards.get(1);

        let out_data: Vec<f64> = (0..b)
            .into_par_iter()
            .flat_map_iter(|batch_idx| {
                use crate::simd::add_scaled_row;
                let base_p = batch_idx * seq * seq;
                let base_v = batch_idx * seq * dim_v;

                // Initialize output buffer for this batch
                let mut batch_out = vec![0.0; seq * dim_v];

                for i in 0..seq {
                    let out_row = &mut batch_out[i * dim_v..(i + 1) * dim_v];
                    for j in 0..seq {
                        let p_val = p_data[base_p + i * seq + j];
                        if p_val.abs() < 1e-9 {
                            continue;
                        } // Optimization for sparsity

                        let v_row = &v_data[base_v + j * dim_v..base_v + (j + 1) * dim_v];
                        add_scaled_row(out_row, v_row, p_val);
                    }
                }
                batch_out
            })
            .collect();

        // Backward pass implementation
        let parents = vec![probs.clone(), v.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(b * seq * dim_v, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![b, seq, dim_v],
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let p_in = &parents[0];
                    let v_in = &parents[1];
                    // Use batch lock for data read
                    let guards = TensorReadGuard::new(&[p_in, v_in]);
                    let p_data = guards.get(0);
                    let v_data = guards.get(1);

                    let mut p_grad = p_in.grad_write_f64();
                    let mut v_grad = v_in.grad_write_f64();

                    let chunk_size_p = seq * seq;
                    let chunk_size_v = seq * dim_v;
                    let chunk_size_grad = seq * dim_v;

                    p_grad
                        .par_chunks_mut(chunk_size_p)
                        .zip(v_grad.par_chunks_mut(chunk_size_v))
                        .zip(grad_out_f64.par_chunks(chunk_size_grad))
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

    /// GPU-accelerated batched matmul: probs * v -> out
    /// probs: [B, Seq, Seq], v: [B, Seq, DimV], out: [B, Seq, DimV]
    #[cfg(cuda)]
    fn batched_matmul_probs_v_cuda(
        &self,
        probs: &Tensor,
        v: &Tensor,
        b: usize,
        seq: usize,
        dim_v: usize,
    ) -> Tensor {
        use crate::cuda::kernels::attention_weighted_sum;
        use crate::cuda::memory::{alloc, copy_d2h};

        let _p_len = probs.data_f64().len();
        let _v_len = v.data_f64().len();
        let out_len = b * seq * dim_v;

        // Upload to GPU
        let d_probs = match probs.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((_stage, err)) => {
                log::warn!("[MLA] CUDA probs upload failed ({}), using CPU path", err);
                return self.batched_matmul_probs_v_cpu_fallback(probs, v, b, seq, dim_v);
            }
        };
        let d_v = match v.cuda_get_or_upload_buffer() {
            Ok(buf) => buf,
            Err((_stage, err)) => {
                log::warn!("[MLA] CUDA v upload failed ({}), using CPU path", err);
                return self.batched_matmul_probs_v_cpu_fallback(probs, v, b, seq, dim_v);
            }
        };
        let d_out = match alloc::<f64>(out_len) {
            Ok(buf) => buf,
            Err(err) => {
                log::warn!("[MLA] CUDA alloc failed ({})", err);
                return self.batched_matmul_probs_v_cpu_fallback(probs, v, b, seq, dim_v);
            }
        };

        if let Err(err) = attention_weighted_sum(&d_probs, &d_v, &d_out, b, seq, dim_v) {
            log::warn!("[MLA] CUDA attention_weighted_sum failed ({})", err);
            return self.batched_matmul_probs_v_cpu_fallback(probs, v, b, seq, dim_v);
        }

        let d_out = Arc::new(d_out);

        // Copy result back to CPU for backward pass and tensor ops
        let mut out_data = vec![0.0; out_len];
        if let Err(err) = copy_d2h(&mut out_data, &d_out) {
            log::warn!("[MLA] CUDA D2H copy failed ({})", err);
            return self.batched_matmul_probs_v_cpu_fallback(probs, v, b, seq, dim_v);
        }

        // Store probs and v data for backward
        let p_data = probs.data_f64().clone();
        let v_data = v.data_f64().clone();
        let _b_cap = b;
        let seq_cap = seq;
        let dim_v_cap = dim_v;

        let parents = vec![probs.clone(), v.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![b, seq, dim_v],
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let p_in = &parents[0];
                    let v_in = &parents[1];
                    let mut p_grad = p_in.grad_write_f64();
                    let mut v_grad = v_in.grad_write_f64();

                    let chunk_size_p = seq_cap * seq_cap;
                    let chunk_size_v = seq_cap * dim_v_cap;
                    let chunk_size_grad = seq_cap * dim_v_cap;

                    p_grad
                        .par_chunks_mut(chunk_size_p)
                        .zip(v_grad.par_chunks_mut(chunk_size_v))
                        .zip(grad_out_f64.par_chunks(chunk_size_grad))
                        .enumerate()
                        .for_each(|(batch_idx, ((p_g_chunk, v_g_chunk), g_out_chunk))| {
                            let base_p = batch_idx * seq_cap * seq_cap;
                            let base_v = batch_idx * seq_cap * dim_v_cap;
                            let p_slice = &p_data[base_p..base_p + chunk_size_p];
                            let v_slice = &v_data[base_v..base_v + chunk_size_v];

                            for i in 0..seq_cap {
                                for d in 0..dim_v_cap {
                                    let g = g_out_chunk[i * dim_v_cap + d];
                                    for j in 0..seq_cap {
                                        p_g_chunk[i * seq_cap + j] +=
                                            g * v_slice[j * dim_v_cap + d];
                                        v_g_chunk[j * dim_v_cap + d] +=
                                            g * p_slice[i * seq_cap + j];
                                    }
                                }
                            }
                        });
                }),
            })),
        }
    }

    /// CPU fallback for batched_matmul_probs_v (used when CUDA fails)
    #[cfg(cuda)]
    fn batched_matmul_probs_v_cpu_fallback(
        &self,
        probs: &Tensor,
        v: &Tensor,
        b: usize,
        seq: usize,
        dim_v: usize,
    ) -> Tensor {
        use crate::simd::add_scaled_row;
        let guards = TensorReadGuard::new(&[probs, v]);
        let p_data = guards.get(0);
        let v_data = guards.get(1);

        let out_data: Vec<f64> = (0..b)
            .into_par_iter()
            .flat_map_iter(|batch_idx| {
                let base_p = batch_idx * seq * seq;
                let base_v = batch_idx * seq * dim_v;
                let mut batch_out = vec![0.0; seq * dim_v];

                for i in 0..seq {
                    let out_row = &mut batch_out[i * dim_v..(i + 1) * dim_v];
                    for j in 0..seq {
                        let p_val = p_data[base_p + i * seq + j];
                        if p_val.abs() < 1e-9 {
                            continue;
                        }
                        let v_row = &v_data[base_v + j * dim_v..base_v + (j + 1) * dim_v];
                        add_scaled_row(out_row, v_row, p_val);
                    }
                }
                batch_out
            })
            .collect();

        let parents = vec![probs.clone(), v.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(b * seq * dim_v, Tensor::grad_dtype_for(Dtype::F64)),
            shape: vec![b, seq, dim_v],
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let p_in = &parents[0];
                    let v_in = &parents[1];
                    let guards = TensorReadGuard::new(&[p_in, v_in]);
                    let p_data = guards.get(0);
                    let v_data = guards.get(1);

                    let mut p_grad = p_in.grad_write_f64();
                    let mut v_grad = v_in.grad_write_f64();

                    let chunk_size_p = seq * seq;
                    let chunk_size_v = seq * dim_v;
                    let chunk_size_grad = seq * dim_v;

                    p_grad
                        .par_chunks_mut(chunk_size_p)
                        .zip(v_grad.par_chunks_mut(chunk_size_v))
                        .zip(grad_out_f64.par_chunks(chunk_size_grad))
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
                                        p_g_chunk[i * seq + j] += g * v_slice[j * dim_v + d];
                                        v_g_chunk[j * dim_v + d] += g * p_slice[i * seq + j];
                                    }
                                }
                            }
                        });
                }),
            })),
        }
    }

    // Apply causal mask: set positions where j > i to -inf (upper triangle excluding diagonal).
    // Input shape: [BH, Seq, Seq]. Backward zeroes out masked gradient positions.
    fn apply_causal_mask(&self, t: &Tensor, seq_len: usize) -> Tensor {
        let data = t.data_f64();
        let bh = data.len() / (seq_len * seq_len);
        let mut new_data = data.clone();

        for b in 0..bh {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    new_data[b * seq_len * seq_len + i * seq_len + j] = f64::NEG_INFINITY;
                }
            }
        }

        let parents = vec![t.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(new_data))),
            grad: Storage::zeros(data.len(), Tensor::grad_dtype_for(Dtype::F64)),
            shape: t.shape.clone(),
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_f64();
                    let bh = inp_grad.len() / (seq_len * seq_len);
                    for (idx, (&g, ig)) in grad_out_f64.iter().zip(inp_grad.iter_mut()).enumerate()
                    {
                        let local = idx % (seq_len * seq_len);
                        let i = local / seq_len;
                        let j = local % seq_len;
                        if j <= i {
                            *ig += g;
                        }
                        // masked positions (j > i): gradient is zero, skip
                    }
                    let _ = bh;
                }),
            })),
        }
    }

    fn scale_tensor(&self, t: &Tensor, scale: f64) -> Tensor {
        let data = t.data_f64();
        let new_data: Vec<f64> = data.par_iter().map(|&x| x * scale).collect();

        let parents = vec![t.clone()];
        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(new_data))),
            grad: Storage::zeros(data.len(), Tensor::grad_dtype_for(Dtype::F64)),
            shape: t.shape.clone(),
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_f64();
                    inp_grad
                        .par_iter_mut()
                        .zip(grad_out_f64.par_iter())
                        .for_each(|(ig, &g)| *ig += g * scale);
                }),
            })),
        }
    }

    fn softmax(&self, t: &Tensor, seq_len: usize) -> Tensor {
        // t: [B, Seq, Seq]
        // Softmax along last dimension
        // CUDA routing: if tensor is on GPU, use fused causal softmax kernel
        #[cfg(cuda)]
        if t.device == crate::autograd::Device::Cuda {
            return t.softmax_causal_cuda();
        }

        let data = t.data_f64();

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
            data: Storage::F64(Arc::new(RwLock::new(new_data))),
            grad: Storage::zeros(data.len(), Tensor::grad_dtype_for(Dtype::F64)),
            shape: t.shape.clone(),
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut inp_grad = parents[0].grad_write_f64();
                    let out_data = &out_data_clone;

                    inp_grad
                        .par_chunks_mut(seq_len)
                        .zip(grad_out_f64.par_chunks(seq_len))
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

        // Use batch lock for better performance
        let guards = TensorReadGuard::new(&[a, b]);
        let a_data = guards.get(0);
        let b_data = guards.get(1);

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
            data: Storage::F64(Arc::new(RwLock::new(new_data))),
            grad: Storage::zeros(
                total_elements * (last_dim_a + last_dim_b),
                Tensor::grad_dtype_for(Dtype::F64),
            ),
            shape: new_shape,
            device: crate::autograd::Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let mut a_grad = parents[0].grad_write_f64();
                    let mut b_grad = parents[1].grad_write_f64();

                    let stride = last_dim_a + last_dim_b;

                    a_grad
                        .par_chunks_mut(last_dim_a)
                        .zip(b_grad.par_chunks_mut(last_dim_b))
                        .zip(grad_out_f64.par_chunks(stride))
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

        // Zero-copy reshape: data is shared via Arc
        {
            let data_reshaped = t_reshaped.data_f64();
            assert_eq!(data_reshaped[0], 1.0);
            assert_eq!(data_reshaped[3], 4.0);
        }

        // Verify data is shared (zero-copy): mutating original is visible through reshaped
        {
            let mut data = t.data_write_f64();
            data[0] = 10.0;
        }
        {
            let data_reshaped_after = t_reshaped.data_f64();
            assert_eq!(data_reshaped_after[0], 10.0); // Shared �?reflects mutation
        }
    }

    #[test]
    fn test_tensor_transpose() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // [[1, 2, 3], [4, 5, 6]]
        let t_t = t.transpose(0, 1);
        // [[1, 4], [2, 5], [3, 6]]
        assert_eq!(t_t.shape, vec![3, 2]);
        let data = t_t.data_f64();
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
        let w_dkv_grad = mla.w_dkv.weight.grad_read_f64();
        let grad_sum: f64 = w_dkv_grad.iter().sum();
        assert!(grad_sum.abs() > 0.0, "Gradient should not be zero");

        // Check W_UK
        let w_uk_grad = mla.w_uk.weight.grad_read_f64();
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
        let x_grad = x.grad_read_f64();
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

        let x_grad = x.grad_read_f64();
        assert!(x_grad.iter().any(|&g| g.abs() > 1e-6));

        let w_grad = norm.weight.grad_read_f64();
        // Weight init is 1.0. Gradient should flow.
        assert!(w_grad.iter().any(|&g| g.abs() > 1e-6));
    }

    #[test]
    fn test_luck_transformer_integration() {
        // Use small dims for test
        let achf = crate::config::AchfConfig::default();
        let t = LuckTransformer::new_compat(8, 8, true, 2, 42, &achf);
        let x = Tensor::rand(vec![1, 5, 8], -0.1, 0.1, 123);

        let out = t.forward(&x, &[]);
        let loss = out.mean();
        loss.backward();

        // Check params count
        let params = t.parameters();
        assert!(params.len() > 10, "Should have many parameters");

        // Check if Embed gradients exist
        let embed_grad = t.embed.weight.grad_read_f64();
        assert!(
            embed_grad.iter().any(|&g: &f64| g.abs() > 0.0),
            "Embed grad missing"
        );

        // Check Norm grad
        let norm_grad = t.blocks[0].norm_1.weight.grad_read_f64();
        assert!(
            norm_grad.iter().any(|&g| g.abs() > 0.0),
            "Norm grad missing"
        );
    }

    // Verify causal masking: changing a future token must NOT affect earlier positions' output.
    #[test]
    fn test_mla_causal_mask_training_path() {
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

        let seq_len = 4;
        let dim = 16;

        // Run 1: input_a = [t0, t1, t2, t3]
        let input_a = Tensor::rand(vec![1, seq_len, dim], -0.1, 0.1, 100);
        let out_a = mla.forward(&input_a);
        let out_a_data = out_a.data_f64().clone();

        // Run 2: same t0, but t2 and t3 are different
        let mut input_b_raw = input_a.data_f64().clone();
        // Overwrite positions 2 and 3 (indices 2*dim .. 4*dim) with different values
        for val in input_b_raw.iter_mut().take(4 * dim).skip(2 * dim) {
            *val = 99.0 - *val;
        }
        let input_b = Tensor::new(input_b_raw, vec![1, seq_len, dim]);
        let out_b = mla.forward(&input_b);
        let out_b_data = out_b.data_f64().clone();

        // Position 0 output must be identical (only sees itself)
        for i in 0..dim {
            assert!(
                (out_a_data[i] - out_b_data[i]).abs() < 1e-10,
                "Causal violation at pos 0, dim {}: {} vs {}",
                i,
                out_a_data[i],
                out_b_data[i]
            );
        }

        // Position 1 output must be identical (sees pos 0 and 1, both unchanged)
        for i in dim..(2 * dim) {
            assert!(
                (out_a_data[i] - out_b_data[i]).abs() < 1e-10,
                "Causal violation at pos 1, dim {}: {} vs {}",
                i - dim,
                out_a_data[i],
                out_b_data[i]
            );
        }

        // Position 2 or 3 output SHOULD differ (they see changed tokens)
        let diff_pos2: f64 = (2 * dim..3 * dim)
            .map(|i| (out_a_data[i] - out_b_data[i]).abs())
            .sum();
        assert!(
            diff_pos2 > 1e-6,
            "Position 2 should differ when its own input changes"
        );
    }

    #[test]
    fn test_mla_causal_mask_inference_path() {
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

        let seq_len = 4;
        let dim = 16;

        // Run 1
        let input_a: Vec<f64> = {
            let t = Tensor::rand(vec![seq_len * dim], -0.1, 0.1, 200);
            let v = t.data_f64().clone();
            v
        };
        let out_a = mla.forward_inference(&input_a);

        // Run 2: change positions 2 and 3
        let mut input_b = input_a.clone();
        for val in input_b.iter_mut().take(4 * dim).skip(2 * dim) {
            *val = 99.0 - *val;
        }
        let out_b = mla.forward_inference(&input_b);

        // Position 0: must be identical
        for i in 0..dim {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-10,
                "Inference causal violation at pos 0, dim {}: {} vs {}",
                i,
                out_a[i],
                out_b[i]
            );
        }

        // Position 1: must be identical
        for i in dim..(2 * dim) {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-10,
                "Inference causal violation at pos 1, dim {}: {} vs {}",
                i - dim,
                out_a[i],
                out_b[i]
            );
        }

        // Position 2: should differ
        let diff: f64 = (2 * dim..3 * dim)
            .map(|i| (out_a[i] - out_b[i]).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "Inference pos 2 should differ when own input changes"
        );
    }

    #[test]
    fn test_mla_causal_mask_cached_path() {
        let config = MLAConfig {
            dim: 16,
            num_heads: 2,
            q_lora_rank: 0,
            kv_lora_rank: 8,
            qk_rope_dim: 4,
            v_head_dim: 8,
            max_seq_len: 10,
        };
        let mla = MultiHeadLatentAttention::new(config.clone(), 42);

        let dim = 16;

        // Prefill 2 tokens, then decode token 3 with two different values
        let prefill: Vec<f64> = {
            let t = Tensor::rand(vec![2 * dim], -0.1, 0.1, 300);
            let v = t.data_f64().clone();
            v
        };

        // Cache A: prefill then decode token_a
        let mut cache_a = KVCache::new(config.num_heads);
        let _ = mla.forward_inference_cached(&prefill, &mut cache_a, 0);

        let token_a: Vec<f64> = {
            let t = Tensor::rand(vec![dim], -0.1, 0.1, 400);
            let v = t.data_f64().clone();
            v
        };
        let out_a = mla.forward_inference_cached(&token_a, &mut cache_a, 2);

        // Cache B: same prefill, different decode token
        let mut cache_b = KVCache::new(config.num_heads);
        let _ = mla.forward_inference_cached(&prefill, &mut cache_b, 0);

        let token_b: Vec<f64> = token_a.iter().map(|&x| 99.0 - x).collect();
        let out_b = mla.forward_inference_cached(&token_b, &mut cache_b, 2);

        // Outputs SHOULD differ (different input at position 2)
        let diff: f64 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "Cached output should differ for different decode tokens"
        );

        // Verify prefill outputs match (same 2 tokens, same cache state)
        let mut cache_c = KVCache::new(config.num_heads);
        let prefill_out_c = mla.forward_inference_cached(&prefill, &mut cache_c, 0);
        let mut cache_d = KVCache::new(config.num_heads);
        let prefill_out_d = mla.forward_inference_cached(&prefill, &mut cache_d, 0);

        for (i, (&c, &d)) in prefill_out_c.iter().zip(prefill_out_d.iter()).enumerate() {
            assert!(
                (c - d).abs() < 1e-12,
                "Prefill should be deterministic at index {}",
                i
            );
        }

        let mut full_input = prefill.clone();
        full_input.extend_from_slice(&token_a);
        let full_out = mla.forward_inference(&full_input);
        let last_start = full_out.len() - dim;
        for (i, (&cached, &full)) in out_a.iter().zip(full_out[last_start..].iter()).enumerate() {
            assert!(
                (cached - full).abs() < 1e-10,
                "Cached decode mismatch at dim {}: {} vs {}",
                i,
                cached,
                full
            );
        }
    }

    #[test]
    fn test_luck_transformer_step_into_matches_allocating_path() {
        let achf = crate::config::AchfConfig::default();
        let model = LuckTransformer::new_compat(8, 8, true, 2, 42, &achf);
        let num_heads = model.blocks[0].mla_layer.config.num_heads;
        let mut base_cache = vec![KVCache::new(num_heads); model.blocks.len()];

        let prefill = Tensor::rand(vec![2 * 8], -0.1, 0.1, 500).data_f64().clone();
        let _ = model.forward_inference_step(&prefill, &mut base_cache, 0);

        let token = Tensor::rand(vec![8], -0.1, 0.1, 501).data_f64().clone();

        let mut cache_a = base_cache.clone();
        let expected = model.forward_inference_step(&token, &mut cache_a, 2);

        let mut cache_b = base_cache;
        let mut actual = vec![123.0; 3];
        model.forward_inference_step_into(&token, &mut cache_b, 2, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_mhc_preserves_dim() {
        let mhc = MhcResidual::new(8, 4, 42);
        let x = Tensor::rand(vec![2, 3, 8], -0.1, 0.1, 100);
        let out = mhc.forward(&x);
        assert_eq!(
            out.shape, x.shape,
            "MhcResidual output shape must match input"
        );
    }

    #[test]
    fn test_h_res_doubly_stochastic() {
        let dim = 4;
        let n = 2;
        let mhc = MhcResidual::new(dim, n, 42);
        let w = mhc.h_res.weight.data_f64();
        let total_dim = dim * n;
        // Verify row sums equal 1.0
        for r in 0..total_dim {
            let sum: f64 = (0..total_dim).map(|c| w[r * total_dim + c]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "H_res row {} sum = {}, expected 1.0",
                r,
                sum
            );
        }
        // Verify column sums equal 1.0
        for c in 0..total_dim {
            let sum: f64 = (0..total_dim).map(|r| w[r * total_dim + c]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "H_res col {} sum = {}, expected 1.0",
                c,
                sum
            );
        }
    }

    #[test]
    fn test_mhc_forward_inference_matches_tensor() {
        let mhc = MhcResidual::new(4, 2, 42);
        let x = Tensor::rand(vec![2, 4], -0.1, 0.1, 200);
        let x_data = x.data_f64().clone();
        let tensor_out = mhc.forward(&x);
        let tensor_data = tensor_out.data_f64().clone();
        let inference_out = mhc.forward_inference(&x_data);
        assert_eq!(tensor_data.len(), inference_out.len());
        for (a, b) in tensor_data.iter().zip(inference_out.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "Tensor and inference paths diverge: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_concat_split_last_dim_roundtrip() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let concat = concat_last_dim(&[a.clone(), b.clone()]);
        assert_eq!(concat.shape, vec![2, 4]);
        let split = split_last_dim(&concat, 2);
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].shape, vec![2, 2]);
        assert_eq!(split[1].shape, vec![2, 2]);
        let s0 = split[0].data_f64();
        let s1 = split[1].data_f64();
        assert_eq!(s0[0], 1.0);
        assert_eq!(s0[3], 4.0);
        assert_eq!(s1[0], 5.0);
        assert_eq!(s1[3], 8.0);
    }
}
