use crate::rng::Rng;

pub const DIM: usize = 8;
pub type Tensor = [f64; DIM];

#[inline(always)]
fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = [0.0; DIM];
    for i in 0..DIM {
        out[i] = a[i] + b[i];
    }
    out
}

#[inline(always)]
fn tensor_relu(t: &Tensor) -> Tensor {
    let mut out = [0.0; DIM];
    for i in 0..DIM {
        out[i] = if t[i] > 0.0 { t[i] } else { 0.0 };
    }
    out
}

#[inline(always)]
fn add_scaled_row(output: &mut Tensor, row: &[f64], scale: f64) {
    let len = output.len();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                add_scaled_row_avx2(output, row, scale);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            add_scaled_row_neon(output, row, scale);
        }
        return;
    }
    for i in 0..len {
        output[i] += scale * row[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_row_avx2(output: &mut [f64], row: &[f64], scale: f64) {
    use core::arch::x86_64::*;
    let len = output.len();
    let mut i = 0;
    let scale_vec = _mm256_set1_pd(scale);
    while i + 4 <= len {
        let out = _mm256_loadu_pd(output.as_ptr().add(i));
        let rowv = _mm256_loadu_pd(row.as_ptr().add(i));
        let prod = _mm256_mul_pd(rowv, scale_vec);
        let sum = _mm256_add_pd(out, prod);
        _mm256_storeu_pd(output.as_mut_ptr().add(i), sum);
        i += 4;
    }
    while i < len {
        *output.get_unchecked_mut(i) += scale * *row.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_scaled_row_neon(output: &mut [f64], row: &[f64], scale: f64) {
    use core::arch::aarch64::*;
    let len = output.len();
    let mut i = 0;
    let scale_vec = vdupq_n_f64(scale);
    while i + 2 <= len {
        let out = vld1q_f64(output.as_ptr().add(i));
        let rowv = vld1q_f64(row.as_ptr().add(i));
        let prod = vmulq_f64(rowv, scale_vec);
        let sum = vaddq_f64(out, prod);
        vst1q_f64(output.as_mut_ptr().add(i), sum);
        i += 2;
    }
    while i < len {
        *output.get_unchecked_mut(i) += scale * *row.get_unchecked(i);
        i += 1;
    }
}

#[derive(Clone)]
pub struct DenseLayer {
    pub weights: [f64; DIM * DIM],
    pub bias: [f64; DIM],
}

impl DenseLayer {
    pub fn new(rng_seed: u64) -> Self {
        let mut weights = [0.0; DIM * DIM];
        let mut bias = [0.0; DIM];
        
        // Initial random weights (using simple LCG for deterministic initialization from seed)
        let mut x = rng_seed;
        let mut next_rand = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (x as f64 / u64::MAX as f64) * 0.2 - 0.1
        };

        for weight in weights.iter_mut() {
            *weight = next_rand();
        }
        for b in bias.iter_mut() {
            *b = next_rand();
        }

        DenseLayer { weights, bias }
    }

    fn mutate(&mut self, rng: &mut Rng, rate: f64, scale: f64) {
        for w in self.weights.iter_mut() {
            if rng.next_f64() < rate {
                *w += (rng.next_f64() - 0.5) * 2.0 * scale;
            }
        }
        for b in self.bias.iter_mut() {
            if rng.next_f64() < rate {
                *b += (rng.next_f64() - 0.5) * 2.0 * scale;
            }
        }
    }

    #[inline(always)]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = self.bias;
        for (i, input_val) in input.iter().enumerate().take(DIM) {
            let row_start = i * DIM;
            let row = &self.weights[row_start..row_start + DIM];
            add_scaled_row(&mut output, row, *input_val);
        }
        output
    }

    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::new();
        self.write_params(&mut params);
        params
    }

    pub fn set_params(&mut self, params: &[f64]) -> bool {
        let mut idx = 0;
        if let Some(new_layer) = Self::read_params(params, &mut idx) {
            *self = new_layer;
            true
        } else {
            false
        }
    }

    pub fn write_params(&self, out: &mut Vec<f64>) {
        out.extend_from_slice(&self.weights);
        out.extend_from_slice(&self.bias);
    }

    pub fn read_params(data: &[f64], idx: &mut usize) -> Option<Self> {
        let total = DIM * DIM + DIM;
        if *idx + total > data.len() {
            return None;
        }
        let mut weights = [0.0; DIM * DIM];
        let mut bias = [0.0; DIM];
        weights.copy_from_slice(&data[*idx..*idx + DIM * DIM]);
        *idx += DIM * DIM;
        bias.copy_from_slice(&data[*idx..*idx + DIM]);
        *idx += DIM;
        Some(DenseLayer { weights, bias })
    }

    pub fn prune(&mut self, threshold: f64) {
        for w in self.weights.iter_mut() {
            if w.abs() < threshold {
                *w = 0.0;
            }
        }
    }

    pub fn count_active_params(&self) -> usize {
        let mut count = 0;
        for w in self.weights.iter() {
            if w.abs() > 1e-9 {
                count += 1;
            }
        }
        // Bias is usually not pruned, but let's count it if we want full sparsity
        // Usually we only prune weights.
        count
    }
}

#[derive(Clone)]
pub struct LayerNorm {
    pub gamma: [f64; DIM],
    pub beta: [f64; DIM],
    pub epsilon: f64,
}

impl LayerNorm {
    pub fn new() -> Self {
        LayerNorm {
            gamma: [1.0; DIM],
            beta: [0.0; DIM],
            epsilon: 1e-5,
        }
    }

    fn mutate(&mut self, rng: &mut Rng, rate: f64, scale: f64) {
        for g in self.gamma.iter_mut() {
            if rng.next_f64() < rate {
                *g += (rng.next_f64() - 0.5) * scale;
            }
        }
        for b in self.beta.iter_mut() {
            if rng.next_f64() < rate {
                *b += (rng.next_f64() - 0.5) * scale;
            }
        }
    }

    #[inline(always)]
    fn forward(&self, x: &Tensor) -> Tensor {
        let sum: f64 = x.iter().sum();
        let mean = sum / DIM as f64;

        let var_sum: f64 = x.iter().map(|v| {
            let d = v - mean;
            d * d
        }).sum();
        
        let var = var_sum / DIM as f64;
        let std_dev = (var + self.epsilon).sqrt();

        let mut output = [0.0; DIM];
        for i in 0..DIM {
            output[i] = self.gamma[i] * (x[i] - mean) / std_dev + self.beta[i];
        }
        output
    }

    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::new();
        self.write_params(&mut params);
        params
    }

    pub fn set_params(&mut self, params: &[f64]) -> bool {
        let mut idx = 0;
        if let Some(new_layer) = Self::read_params(params, &mut idx) {
            *self = new_layer;
            true
        } else {
            false
        }
    }

    pub fn write_params(&self, out: &mut Vec<f64>) {
        out.extend_from_slice(&self.gamma);
        out.extend_from_slice(&self.beta);
    }

    pub fn read_params(data: &[f64], idx: &mut usize) -> Option<Self> {
        let total = DIM + DIM;
        if *idx + total > data.len() {
            return None;
        }
        let mut gamma = [0.0; DIM];
        let mut beta = [0.0; DIM];
        gamma.copy_from_slice(&data[*idx..*idx + DIM]);
        *idx += DIM;
        beta.copy_from_slice(&data[*idx..*idx + DIM]);
        *idx += DIM;
        Some(LayerNorm { gamma, beta, epsilon: 1e-5 })
    }
}

#[derive(Clone)]
pub struct ResidualBlock {
    pub d1: DenseLayer,
    pub ln1: LayerNorm,
    pub d2: DenseLayer,
    pub ln2: LayerNorm,
}

impl ResidualBlock {
    pub fn new(seed: u64) -> Self {
        ResidualBlock {
            d1: DenseLayer::new(seed),
            ln1: LayerNorm::new(),
            d2: DenseLayer::new(seed.wrapping_add(1)),
            ln2: LayerNorm::new(),
        }
    }

    pub fn mutate(&mut self, rng: &mut Rng, rate: f64, scale: f64) {
        self.d1.mutate(rng, rate, scale);
        self.ln1.mutate(rng, rate, scale);
        self.d2.mutate(rng, rate, scale);
        self.ln2.mutate(rng, rate, scale);
    }

    #[inline(always)]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.d1.forward(x);
        let h1_norm = self.ln1.forward(&h1);
        let h1_act = tensor_relu(&h1_norm);
        
        let h2 = self.d2.forward(&h1_act);
        let h2_norm = self.ln2.forward(&h2);
        let h2_act = tensor_relu(&h2_norm);
        
        // Residual connection
        tensor_add(x, &h2_act)
    }

    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::new();
        self.write_params(&mut params);
        params
    }

    pub fn set_params(&mut self, params: &[f64]) -> bool {
        let mut idx = 0;
        if let Some(new_block) = Self::read_params(params, &mut idx) {
            *self = new_block;
            true
        } else {
            false
        }
    }

    pub fn write_params(&self, out: &mut Vec<f64>) {
        self.d1.write_params(out);
        self.ln1.write_params(out);
        self.d2.write_params(out);
        self.ln2.write_params(out);
    }

    pub fn read_params(data: &[f64], idx: &mut usize) -> Option<Self> {
        let d1 = DenseLayer::read_params(data, idx)?;
        let ln1 = LayerNorm::read_params(data, idx)?;
        let d2 = DenseLayer::read_params(data, idx)?;
        let ln2 = LayerNorm::read_params(data, idx)?;
        Some(ResidualBlock { d1, ln1, d2, ln2 })
    }

    pub fn prune(&mut self, threshold: f64) {
        self.d1.prune(threshold);
        self.d2.prune(threshold);
    }

    pub fn count_active_params(&self) -> usize {
        self.d1.count_active_params() + self.d2.count_active_params()
    }
}

#[derive(Clone)]
pub struct NeuralLuckOptimizer {
    pub res_block: ResidualBlock,
    pub linear_weights: [f64; DIM],
    pub linear_bias: f64,
}

impl NeuralLuckOptimizer {
    pub fn new(seed: u64) -> Self {
        let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut next_rand = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (x as f64 / u64::MAX as f64) * 0.02 - 0.01
        };
        let mut linear_weights = [0.0; DIM];
        for w in linear_weights.iter_mut() {
            *w = next_rand();
        }
        let linear_bias = next_rand();
        NeuralLuckOptimizer {
            res_block: ResidualBlock::new(seed), 
            linear_weights,
            linear_bias,
        }
    }

    pub fn mutate(&mut self, rng: &mut Rng, rate: f64, scale: f64) {
        self.res_block.mutate(rng, rate, scale);
    }

    pub fn set_linear_params(&mut self, weights: [f64; DIM], bias: f64) {
        self.linear_weights = weights;
        self.linear_bias = bias;
    }

    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::new();
        self.write_params(&mut params);
        params
    }

    pub fn set_params(&mut self, params: &[f64]) -> bool {
        let mut idx = 0;
        if let Some(new_opt) = Self::read_params(params, &mut idx) {
            *self = new_opt;
            true
        } else {
            false
        }
    }

    pub fn count_params_analysis() -> usize {
        // ResidualBlock: 2 * (DenseLayer + LayerNorm)
        // DenseLayer: DIM*DIM + DIM
        // LayerNorm: DIM + DIM
        2 * (DIM * DIM + DIM + 2 * DIM)
    }

    pub fn count_params_decision() -> usize {
        DIM + 1
    }

    pub fn write_params(&self, out: &mut Vec<f64>) {
        self.res_block.write_params(out);
        out.extend_from_slice(&self.linear_weights);
        out.push(self.linear_bias);
    }

    pub fn read_params(data: &[f64], idx: &mut usize) -> Option<Self> {
        let res_block = ResidualBlock::read_params(data, idx)?;
        if *idx + DIM + 1 > data.len() {
            return None;
        }
        let mut linear_weights = [0.0; DIM];
        linear_weights.copy_from_slice(&data[*idx..*idx + DIM]);
        *idx += DIM;
        let linear_bias = data[*idx];
        *idx += 1;
        Some(NeuralLuckOptimizer {
            res_block,
            linear_weights,
            linear_bias,
        })
    }

    pub fn prune(&mut self, threshold: f64) {
        self.res_block.prune(threshold);
        // Linear weights are part of the decision manifold, can also be pruned
        for w in self.linear_weights.iter_mut() {
            if w.abs() < threshold {
                *w = 0.0;
            }
        }
    }

    pub fn count_active_params(&self) -> usize {
        let mut count = self.res_block.count_active_params();
        for w in self.linear_weights.iter() {
            if w.abs() > 1e-9 {
                count += 1;
            }
        }
        count
    }

    fn param_count_v1() -> usize {
        2 * DIM * DIM + 6 * DIM
    }

    fn param_count_v2() -> usize {
        Self::param_count_v1() + DIM + 1
    }

    pub fn param_count() -> usize {
        Self::param_count_v2()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let count = Self::param_count() as u32;
        let mut params = Vec::with_capacity(count as usize);
        self.res_block.write_params(&mut params);
        params.extend_from_slice(&self.linear_weights);
        params.push(self.linear_bias);
        let mut out = Vec::with_capacity(8 + params.len() * 8);
        out.extend_from_slice(b"NLC2");
        out.extend_from_slice(&count.to_le_bytes());
        for v in params {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let magic = &bytes[0..4];
        let mut count_bytes = [0u8; 4];
        count_bytes.copy_from_slice(&bytes[4..8]);
        let count = u32::from_le_bytes(count_bytes) as usize;
        let expected_len = 8 + count * 8;
        if bytes.len() != expected_len {
            return None;
        }
        let mut data = Vec::with_capacity(count);
        let mut offset = 8;
        for _ in 0..count {
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&bytes[offset..offset + 8]);
            data.push(f64::from_le_bytes(chunk));
            offset += 8;
        }
        let mut idx = 0;
        if magic == b"NLC1" {
            if count != Self::param_count_v1() {
                return None;
            }
            let res_block = ResidualBlock::read_params(&data, &mut idx)?;
            return Some(NeuralLuckOptimizer {
                res_block,
                linear_weights: [0.0; DIM],
                linear_bias: 0.0,
            });
        }
        if magic != b"NLC2" {
            return None;
        }
        if count != Self::param_count_v2() {
            return None;
        }
        let res_block = ResidualBlock::read_params(&data, &mut idx)?;
        if idx + DIM + 1 > data.len() {
            return None;
        }
        let mut linear_weights = [0.0; DIM];
        linear_weights.copy_from_slice(&data[idx..idx + DIM]);
        idx += DIM;
        let linear_bias = data[idx];
        Some(NeuralLuckOptimizer { res_block, linear_weights, linear_bias })
    }

    #[inline(always)]
    pub fn predict(&self, x: &Tensor, dropout_seed: u64) -> f64 {
        // Dropout logic
        let dropout_val = (dropout_seed.wrapping_mul(6364136223846793005) >> 56) as f64 / 256.0;
        
        if dropout_val < 0.05 { // Reduced dropout slightly
            return 0.0;
        }

        let y = self.res_block.forward(x);

        // TRUE NEURAL IMPLEMENTATION:
        // We use the first output of the network as the bias directly.
        // We apply tanh to constrain it to [-1, 1], then scale it to a reasonable probability range (e.g. +/- 2%).
        // The network MUST learn to output positive values when the player is unlucky (loss_streak high)
        // and negative/zero values otherwise.
        
        let neural_output = y[0]; 
        
        // Tanh approximation for speed: x / (1 + |x|) or just clamp
        // Let's use standard clamp for simplicity but allow a wider range for the network to explore
        // Then we scale it down.
        
        let activation = neural_output.clamp(-1.0, 1.0); 
        let mut linear_sum = self.linear_bias;
        for (i, x_val) in x.iter().enumerate().take(DIM) {
            linear_sum += self.linear_weights[i] * x_val;
        }
        let linear_bias = linear_sum.clamp(-1.0, 1.0) * 0.01;
        (activation * 0.015 + linear_bias).clamp(-0.02, 0.02)
    }
}
