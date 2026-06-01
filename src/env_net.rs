//! EnvNet: Hand-rolled MLP to replace DBN for environment noise modeling.
//!
//! Architecture: 5 → 64 → 32 → 16 → 2
//! Input:  [rng, pity_6_norm, total_pulls_norm, streak_norm, loss_streak_norm]
//! Output: [env_noise, env_bias]
#![allow(dead_code, unused, clippy::needless_range_loop)]

use crate::config::Config;
use crate::rng::Rng;
use crate::simd::{add_scaled_row, dot_product};

// =============================================================================
// BatchNorm1d
// =============================================================================

/// Hand-rolled BatchNorm1d (no external crate).
///
/// During training: uses batch statistics (mean/var of current mini-batch).
/// During inference: uses running mean/var (exponential moving average).
pub struct BatchNorm1d {
    gamma: Vec<f64>, // scale parameters
    beta: Vec<f64>,  // shift parameters
    running_mean: Vec<f64>,
    running_var: Vec<f64>,
    momentum: f64,
    eps: f64,
    training: bool,
}

impl BatchNorm1d {
    /// Create a new BatchNorm1d with given feature dimension.
    pub fn new(dim: usize) -> Self {
        let gamma = vec![1.0; dim];
        let beta = vec![0.0; dim];
        let running_mean = vec![0.0; dim];
        let running_var = vec![1.0; dim];
        BatchNorm1d {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }

    /// Toggle training/evaluation mode.
    pub fn set_train(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward pass: applies BN to a 1D slice of features.
    ///
    /// Training: normalizes using batch mean/var, updates running stats.
    /// Inference: normalizes using running mean/var.
    pub fn forward(&mut self, x: &[f64]) -> Vec<f64> {
        let dim = x.len();
        assert_eq!(self.gamma.len(), dim);

        if self.training {
            // ---- Compute batch statistics ----
            let sum: f64 = x.iter().sum();
            let mean = sum / dim as f64;

            let var_sum = x
                .iter()
                .map(|v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>();
            let var = var_sum / dim as f64;
            let std_dev = (var + self.eps).sqrt();

            // ---- Update running statistics (EMA) ----
            for i in 0..dim {
                self.running_mean[i] =
                    self.momentum * self.running_mean[i] + (1.0 - self.momentum) * mean;
                self.running_var[i] =
                    self.momentum * self.running_var[i] + (1.0 - self.momentum) * var;
            }

            // ---- Normalize ----
            let mut output = Vec::with_capacity(dim);
            for i in 0..dim {
                let norm = (x[i] - mean) / std_dev;
                output.push(self.gamma[i] * norm + self.beta[i]);
            }
            output
        } else {
            // ---- Use running statistics ----
            let mut output = Vec::with_capacity(dim);
            for i in 0..dim {
                let std_dev = (self.running_var[i] + self.eps).sqrt();
                let norm = (x[i] - self.running_mean[i]) / std_dev;
                output.push(self.gamma[i] * norm + self.beta[i]);
            }
            output
        }
    }

    /// Forward pass writing into a pre-allocated buffer. Returns stats for backward.
    pub fn forward_into(&mut self, x: &[f64], output: &mut [f64]) -> BnStats {
        let dim = x.len();
        assert_eq!(self.gamma.len(), dim);
        assert_eq!(output.len(), dim);

        let sum: f64 = x.iter().sum();
        let mean = sum / dim as f64;

        let var_sum = x
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>();
        let var = var_sum / dim as f64;
        let std_dev = (var + self.eps).sqrt();

        // Update running stats
        for i in 0..dim {
            self.running_mean[i] =
                self.momentum * self.running_mean[i] + (1.0 - self.momentum) * mean;
            self.running_var[i] = self.momentum * self.running_var[i] + (1.0 - self.momentum) * var;
        }

        for i in 0..dim {
            let norm = (x[i] - mean) / std_dev;
            output[i] = self.gamma[i] * norm + self.beta[i];
        }

        BnStats { mean, var, std_dev }
    }

    /// Inference forward writing into a pre-allocated buffer.
    pub fn forward_infer_into(&self, x: &[f64], output: &mut [f64]) {
        let dim = x.len();
        assert_eq!(output.len(), dim);

        for i in 0..dim {
            let std_dev = (self.running_var[i] + self.eps).sqrt();
            let norm = (x[i] - self.running_mean[i]) / std_dev;
            output[i] = self.gamma[i] * norm + self.beta[i];
        }
    }

    /// Returns the current running mean (for inspection/serialization).
    #[allow(dead_code)]
    pub fn running_mean(&self) -> &[f64] {
        &self.running_mean
    }

    /// Returns the current running variance (for inspection/serialization).
    #[allow(dead_code)]
    pub fn running_var(&self) -> &[f64] {
        &self.running_var
    }

    /// Returns the number of parameters (gamma + beta).
    pub fn param_count(&self) -> usize {
        2 * self.gamma.len()
    }

    /// Serialize gamma and beta into a flat vector.
    pub fn write_params(&self, out: &mut Vec<f64>) {
        out.extend_from_slice(&self.gamma);
        out.extend_from_slice(&self.beta);
    }

    /// Deserialize gamma and beta from a flat vector (idx is advanced).
    pub fn read_params(data: &[f64], idx: &mut usize) -> Option<Self> {
        let dim = (data.len() - *idx) / 2;
        if dim == 0 {
            return None;
        }
        let mut gamma = vec![0.0; dim];
        let mut beta = vec![0.0; dim];
        gamma.copy_from_slice(&data[*idx..*idx + dim]);
        *idx += dim;
        beta.copy_from_slice(&data[*idx..*idx + dim]);
        *idx += dim;

        Some(BatchNorm1d {
            gamma,
            beta,
            running_mean: vec![0.0; dim],
            running_var: vec![1.0; dim],
            momentum: 0.1,
            eps: 1e-5,
            training: false,
        })
    }
}

// =============================================================================
// Adam Optimizer
// =============================================================================

/// Hand-rolled Adam optimizer with per-parameter state (m, v, t).
pub struct AdamOptimizer {
    /// Per-parameter first moment vectors (one per parameter).
    m: Vec<f64>,
    /// Per-parameter second moment vectors (one per parameter).
    v: Vec<f64>,
    /// Time step counter.
    t: usize,
    /// Learning rate.
    lr: f64,
    /// Beta1 decay rate.
    beta1: f64,
    /// Beta2 decay rate.
    beta2: f64,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer for a given number of parameters.
    ///
    /// Default hyperparameters:
    ///   lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
    pub fn new(param_count: usize) -> Self {
        AdamOptimizer {
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            t: 0,
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// One step of Adam.
    ///
    /// Updates params in-place using gradients.
    /// Standard Adam:
    ///   m_t = beta1 * m_{t-1} + (1 - beta1) * g
    ///   v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    ///   m_hat = m_t / (1 - beta1^t)
    ///   v_hat = v_t / (1 - beta2^t)
    ///   theta -= lr * m_hat / (sqrt(v_hat) + eps)
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), self.m.len());

        self.t += 1;
        let t_f64 = self.t as f64;

        let beta1_t = self.beta1.powi(t_f64 as i32);
        let beta2_t = self.beta2.powi(t_f64 as i32);

        let one_minus_beta1_t = 1.0 - beta1_t;
        let one_minus_beta2_t = 1.0 - beta2_t;

        for i in 0..params.len() {
            let g = grads[i];

            // m_t = beta1 * m_{t-1} + (1 - beta1) * g
            self.m[i] = self.beta1 * self.m[i] + one_minus_beta1_t * g;

            // v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            self.v[i] = self.beta2 * self.v[i] + one_minus_beta2_t * g * g;

            // m_hat = m_t / (1 - beta1^t)
            let m_hat = self.m[i] / one_minus_beta1_t;

            // v_hat = v_t / (1 - beta2^t)
            let v_hat = self.v[i] / one_minus_beta2_t;

            // theta -= lr * m_hat / (sqrt(v_hat) + eps)
            let denom = v_hat.sqrt() + self.eps;
            params[i] -= self.lr * m_hat / denom;
        }
    }

    /// Returns the current time step.
    #[allow(dead_code)]
    pub fn t(&self) -> usize {
        self.t
    }

    /// Serialize state (m, v, t) into a flat vector.
    pub fn write_state(&self, out: &mut Vec<f64>) {
        out.push(self.t as f64);
        out.extend_from_slice(&self.m);
        out.extend_from_slice(&self.v);
    }

    /// Deserialize state from a flat vector (idx is advanced).
    pub fn read_state(data: &[f64], idx: &mut usize) -> Option<Self> {
        if *idx >= data.len() {
            return None;
        }
        let t = data[*idx] as usize;
        *idx += 1;

        let m_len = data.len() - *idx;
        if m_len == 0 {
            return None;
        }
        let half = m_len / 2;
        if half == 0 {
            return None;
        }

        let mut m = vec![0.0; half];
        let mut v = vec![0.0; half];
        m.copy_from_slice(&data[*idx..*idx + half]);
        *idx += half;
        v.copy_from_slice(&data[*idx..*idx + half]);
        *idx += half;

        Some(AdamOptimizer {
            m,
            v,
            t,
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        })
    }
}

// =============================================================================
// Linear Layer
// =============================================================================

/// A fully-connected linear layer with Xavier/Glorot initialization.
#[derive(Clone)]
struct LinearLayer {
    weights: Vec<f64>, // [out_features, in_features] row-major
    bias: Vec<f64>,
    in_features: usize,
    out_features: usize,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier/Glorot initialization.
    ///
    /// Weights are initialized as: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    /// Bias is initialized to zeros.
    fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Self {
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();

        let mut weights = vec![0.0; in_features * out_features];
        for w in weights.iter_mut() {
            *w = (rng.next_f64() - 0.5) * 2.0 * scale;
        }

        let bias = vec![0.0; out_features];

        LinearLayer {
            weights,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: compute y = x @ W^T + b
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.in_features);

        let mut output = self.bias.clone();
        for (i, &in_val) in input.iter().enumerate() {
            if in_val == 0.0 {
                continue;
            }
            let row_start = i * self.out_features;
            let row = &self.weights[row_start..row_start + self.out_features];
            add_scaled_row(&mut output, row, in_val);
        }
        output
    }

    /// Forward pass writing into a pre-allocated buffer.
    fn forward_into(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), self.in_features);
        assert_eq!(output.len(), self.out_features);

        output.copy_from_slice(&self.bias);
        for (i, &in_val) in input.iter().enumerate() {
            if in_val == 0.0 {
                continue;
            }
            let row_start = i * self.out_features;
            let row = &self.weights[row_start..row_start + self.out_features];
            add_scaled_row(output, row, in_val);
        }
    }

    /// Compute gradients w.r.t. weights, bias, and input.
    ///
    /// Given upstream gradient dL/dy (same shape as output),
    /// returns (dL/dW, dL/db, dL/dx).
    fn backward(&self, input: &[f64], upstream: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        assert_eq!(upstream.len(), self.out_features);
        assert_eq!(input.len(), self.in_features);

        // dL/dW = x^T @ dL/dy  (in_features x out_features)
        let mut grad_w = vec![0.0; self.in_features * self.out_features];
        for i in 0..self.in_features {
            for j in 0..self.out_features {
                grad_w[i * self.out_features + j] = input[i] * upstream[j];
            }
        }

        // dL/db = dL/dy
        let grad_b = upstream.to_vec();

        // dL/dx = dL/dy @ W
        let mut grad_x = vec![0.0; self.in_features];
        for i in 0..self.in_features {
            let row_start = i * self.out_features;
            let row = &self.weights[row_start..row_start + self.out_features];
            grad_x[i] = dot_product(row, upstream);
        }

        (grad_w, grad_b, grad_x)
    }

    /// Backward pass writing into pre-allocated buffers.
    fn backward_into(
        &self,
        input: &[f64],
        upstream: &[f64],
        grad_w: &mut [f64],
        grad_b: &mut [f64],
        grad_x: &mut [f64],
    ) {
        assert_eq!(upstream.len(), self.out_features);
        assert_eq!(input.len(), self.in_features);
        assert_eq!(grad_w.len(), self.in_features * self.out_features);
        assert_eq!(grad_b.len(), self.out_features);
        assert_eq!(grad_x.len(), self.in_features);

        // dL/dW = x^T @ dL/dy
        grad_w.fill(0.0);
        for i in 0..self.in_features {
            let base = i * self.out_features;
            let xi = input[i];
            for j in 0..self.out_features {
                grad_w[base + j] = xi * upstream[j];
            }
        }

        // dL/db = dL/dy
        grad_b.copy_from_slice(upstream);

        // dL/dx = dL/dy @ W
        grad_x.fill(0.0);
        for i in 0..self.in_features {
            let row_start = i * self.out_features;
            let row = &self.weights[row_start..row_start + self.out_features];
            grad_x[i] = dot_product(row, upstream);
        }
    }

    /// Apply gradients to weights and bias in-place (no optimizer, raw SGD).
    fn apply_grad_raw(&mut self, grad_w: &[f64], grad_b: &[f64], lr: f64) {
        assert_eq!(grad_w.len(), self.weights.len());
        assert_eq!(grad_b.len(), self.bias.len());

        for i in 0..self.weights.len() {
            self.weights[i] -= lr * grad_w[i];
        }
        for i in 0..self.bias.len() {
            self.bias[i] -= lr * grad_b[i];
        }
    }

    pub fn param_count(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    pub fn write_params(&self, out: &mut Vec<f64>) {
        out.extend_from_slice(&self.weights);
        out.extend_from_slice(&self.bias);
    }
}

// =============================================================================
// EnvNet
// =============================================================================

/// Stats for BN backward (mean, var, std_dev computed during forward pass).
pub(crate) struct BnStats {
    mean: f64,
    var: f64,
    std_dev: f64,
}

/// Environment noise MLP.
///
/// Input (5 features):
///   - rng: random value in [0,1) from `rng.next_f64()`
///   - pity_6_norm: current pity normalized to [0,1]
///   - total_pulls_norm: total pulls normalized to [0,1]
///   - streak_norm: recent win streak normalized
///   - loss_streak_norm: recent loss streak normalized
///
/// Output (2 values):
///   - env_noise: additive noise to probability
///   - env_bias: bias term for probability adjustment
pub struct EnvNet {
    // Linear layers
    fc1: LinearLayer,
    fc2: LinearLayer,
    fc3: LinearLayer,
    fc4: LinearLayer,

    // BatchNorm layers
    bn1: BatchNorm1d,
    bn2: BatchNorm1d,
    bn3: BatchNorm1d,

    // Adam optimizers per layer (separate for weights and biases)
    opt1_w: AdamOptimizer,
    opt1_b: AdamOptimizer,
    opt2_w: AdamOptimizer,
    opt2_b: AdamOptimizer,
    opt3_w: AdamOptimizer,
    opt3_b: AdamOptimizer,
    opt4_w: AdamOptimizer,
    opt4_b: AdamOptimizer,
    opt_bn1: AdamOptimizer,
    opt_bn2: AdamOptimizer,
    opt_bn3: AdamOptimizer,

    training: bool,
}

impl EnvNet {
    /// Create a new EnvNet with random initialization.
    pub fn new(rng: &mut Rng) -> Self {
        let (fc1, bn1, opt_bn1) = Self::make_layer(5, 64, rng);
        let (fc2, bn2, opt_bn2) = Self::make_layer(64, 32, rng);
        let (fc3, bn3, opt_bn3) = Self::make_layer(32, 16, rng);

        let fc4 = LinearLayer::new(16, 2, rng);

        // Separate optimizers for weights and biases
        let opt1_w = AdamOptimizer::new(fc1.weights.len());
        let opt1_b = AdamOptimizer::new(fc1.bias.len());
        let opt2_w = AdamOptimizer::new(fc2.weights.len());
        let opt2_b = AdamOptimizer::new(fc2.bias.len());
        let opt3_w = AdamOptimizer::new(fc3.weights.len());
        let opt3_b = AdamOptimizer::new(fc3.bias.len());
        let opt4_w = AdamOptimizer::new(fc4.weights.len());
        let opt4_b = AdamOptimizer::new(fc4.bias.len());

        EnvNet {
            fc1,
            fc2,
            fc3,
            fc4,
            bn1,
            bn2,
            bn3,
            opt1_w,
            opt1_b,
            opt2_w,
            opt2_b,
            opt3_w,
            opt3_b,
            opt4_w,
            opt4_b,
            opt_bn1,
            opt_bn2,
            opt_bn3,
            training: true,
        }
    }

    fn make_layer(
        in_feat: usize,
        out_feat: usize,
        rng: &mut Rng,
    ) -> (LinearLayer, BatchNorm1d, AdamOptimizer) {
        let fc = LinearLayer::new(in_feat, out_feat, rng);
        let bn = BatchNorm1d::new(out_feat);
        // BN has gamma + beta = 2 * out_feat params, but we use separate optimizers for gamma and beta
        // so each optimizer handles out_feat parameters
        let opt = AdamOptimizer::new(out_feat);
        (fc, bn, opt)
    }

    /// Toggle training/evaluation mode.
    pub fn set_train(&mut self, training: bool) {
        self.training = training;
        self.bn1.set_train(training);
        self.bn2.set_train(training);
        self.bn3.set_train(training);
    }

    /// Forward pass: returns (env_noise, env_bias).
    ///
    /// Architecture:
    ///   Input(5) → fc1 → relu → bn1
    ///             → fc2 → relu → bn2
    ///             → fc3 → relu → bn3
    ///             → fc4 → output(2)
    pub fn forward(&self, input: &[f64]) -> (f64, f64) {
        assert_eq!(input.len(), 5);

        // Hidden 1: Linear(5→64) + ReLU + BN
        let h1 = self.fc1.forward(input);
        let h1_relu: Vec<f64> = h1.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        // Note: forward() clones for BN input since it needs &mut self
        // We'll call individual layer methods that take &self

        // We need a version that doesn't clone - use a separate forward path
        unreachable!("use forward_clipped instead")
    }

    /// Forward pass that returns owned output, suitable for training.
    fn forward_train(&self, input: &[f64]) -> Vec<f64> {
        // Hidden 1: Linear(5→64) + ReLU + BN
        let h1 = self.fc1.forward(input);
        let h1_relu: Vec<f64> = h1.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let h1_bn = Self::bn_forward(&self.bn1, &h1_relu);

        // Hidden 2: Linear(64→32) + ReLU + BN
        let h2 = self.fc2.forward(&h1_bn);
        let h2_relu: Vec<f64> = h2.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let h2_bn = Self::bn_forward(&self.bn2, &h2_relu);

        // Hidden 3: Linear(32→16) + ReLU + BN
        let h3 = self.fc3.forward(&h2_bn);
        let h3_relu: Vec<f64> = h3.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let h3_bn = Self::bn_forward(&self.bn3, &h3_relu);

        // Output: Linear(16→2)
        self.fc4.forward(&h3_bn)
    }

    /// Inference forward (no gradient tracking).
    pub fn forward_infer(&self, input: &[f64]) -> (f64, f64) {
        assert_eq!(input.len(), 5);

        // Hidden 1
        let h1 = self.fc1.forward(input);
        let h1_relu: Vec<f64> = h1.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let h1_bn = Self::bn_forward_infer(&self.bn1, &h1_relu);

        // Hidden 2
        let h2 = self.fc2.forward(&h1_bn);
        let h2_relu: Vec<f64> = h2.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let h2_bn = Self::bn_forward_infer(&self.bn2, &h2_relu);

        // Hidden 3
        let h3 = self.fc3.forward(&h2_bn);
        let h3_relu: Vec<f64> = h3.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let h3_bn = Self::bn_forward_infer(&self.bn3, &h3_relu);

        // Output
        let out = self.fc4.forward(&h3_bn);
        (out[0], out[1])
    }

    fn bn_forward(bn: &BatchNorm1d, x: &[f64]) -> Vec<f64> {
        // During training we need to use training stats
        // This is a const fn helper, but we actually need mutable access
        // So we do this in the forward path
        let dim = x.len();
        let sum: f64 = x.iter().sum();
        let mean = sum / dim as f64;
        let var_sum = x
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>();
        let var = var_sum / dim as f64;
        let std_dev = (var + bn.eps).sqrt();

        let mut output = Vec::with_capacity(dim);
        for i in 0..dim {
            let norm = (x[i] - mean) / std_dev;
            output.push(bn.gamma[i] * norm + bn.beta[i]);
        }
        output
    }

    fn bn_forward_infer(bn: &BatchNorm1d, x: &[f64]) -> Vec<f64> {
        let dim = x.len();
        let mut output = Vec::with_capacity(dim);
        for i in 0..dim {
            let std_dev = (bn.running_var[i] + bn.eps).sqrt();
            let norm = (x[i] - bn.running_mean[i]) / std_dev;
            output.push(bn.gamma[i] * norm + bn.beta[i]);
        }
        output
    }

    /// Training step: MSE loss + backward + Adam step.
    ///
    /// Returns the loss value for monitoring.
    pub fn train_step(&mut self, input: &[f64], target: &[f64]) -> f64 {
        assert_eq!(input.len(), 5);
        assert_eq!(target.len(), 2);
        self.set_train(true);

        // ---- Forward pass ----
        let h1 = self.fc1.forward(input);
        let h1_relu: Vec<f64> = h1.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let (h1_bn, bn1_stats) = Self::bn_forward_with_stats(&mut self.bn1, &h1_relu);

        let h2 = self.fc2.forward(&h1_bn);
        let h2_relu: Vec<f64> = h2.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let (h2_bn, bn2_stats) = Self::bn_forward_with_stats(&mut self.bn2, &h2_relu);

        let h3 = self.fc3.forward(&h2_bn);
        let h3_relu: Vec<f64> = h3.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
        let (h3_bn, bn3_stats) = Self::bn_forward_with_stats(&mut self.bn3, &h3_relu);

        let output = self.fc4.forward(&h3_bn);

        // ---- Compute MSE loss ----
        let mut loss = 0.0;
        let mut loss_grad = vec![0.0; 2];
        for i in 0..2 {
            let err = output[i] - target[i];
            loss += 0.5 * err * err;
            loss_grad[i] = err; // dL/dy_i = (y_i - t_i)
        }

        // ---- Backward pass ----
        // Output layer: dL/dh3_bn = dL/dy @ W4
        let (grad_w4, grad_b4, grad_h3_bn) = self.fc4.backward(&h3_bn, &loss_grad);

        // BN3 backward (through ReLU)
        let grad_h3_relu =
            Self::bn_backward_x(&self.bn3, &h3_relu, &h3_bn, &bn3_stats, &grad_h3_bn);
        let grad_h3 = Self::relu_backward(&h3, &grad_h3_relu);

        // FC3 backward
        let (grad_w3, grad_b3, grad_h2_bn) = self.fc3.backward(&h2_bn, &grad_h3);

        // BN2 backward
        let grad_h2_relu =
            Self::bn_backward_x(&self.bn2, &h2_relu, &h2_bn, &bn2_stats, &grad_h2_bn);
        let grad_h2 = Self::relu_backward(&h2, &grad_h2_relu);

        // FC2 backward
        let (grad_w2, grad_b2, grad_h1_bn) = self.fc2.backward(&h1_bn, &grad_h2);

        // BN1 backward
        let grad_h1_relu =
            Self::bn_backward_x(&self.bn1, &h1_relu, &h1_bn, &bn1_stats, &grad_h1_bn);
        let grad_h1 = Self::relu_backward(&h1, &grad_h1_relu);

        // FC1 backward
        let (grad_w1, grad_b1, _grad_input) = self.fc1.backward(input, &grad_h1);

        // ---- Compute BN gamma/beta gradients ----
        // dL/d_gamma = sum(dL/d_y * (x_normalized)), dL/d_beta = sum(dL/d_y)
        let grad_bn1 = Self::bn_backward_gamma_beta(&self.bn1, &h1_relu, &bn1_stats, &grad_h1_relu);
        let grad_bn2 = Self::bn_backward_gamma_beta(&self.bn2, &h2_relu, &bn2_stats, &grad_h2_relu);
        let grad_bn3 = Self::bn_backward_gamma_beta(&self.bn3, &h3_relu, &bn3_stats, &grad_h3_relu);

        // ---- Adam updates ----
        // FC layers: weights and bias have separate optimizers
        self.opt1_w.step(&mut self.fc1.weights, &grad_w1);
        self.opt1_b.step(&mut self.fc1.bias, &grad_b1);

        self.opt2_w.step(&mut self.fc2.weights, &grad_w2);
        self.opt2_b.step(&mut self.fc2.bias, &grad_b2);

        self.opt3_w.step(&mut self.fc3.weights, &grad_w3);
        self.opt3_b.step(&mut self.fc3.bias, &grad_b3);

        self.opt4_w.step(&mut self.fc4.weights, &grad_w4);
        self.opt4_b.step(&mut self.fc4.bias, &grad_b4);

        self.opt_bn1.step(&mut self.bn1.gamma, &grad_bn1.0);
        self.opt_bn1.step(&mut self.bn1.beta, &grad_bn1.1);

        self.opt_bn2.step(&mut self.bn2.gamma, &grad_bn2.0);
        self.opt_bn2.step(&mut self.bn2.beta, &grad_bn2.1);

        self.opt_bn3.step(&mut self.bn3.gamma, &grad_bn3.0);
        self.opt_bn3.step(&mut self.bn3.beta, &grad_bn3.1);

        loss
    }

    /// Training step for a mini-batch: MSE loss + backward + Adam step.
    ///
    /// Accumulates gradients over the entire batch, then performs a single Adam update.
    /// Returns the average loss value for monitoring.
    pub fn train_step_batch(&mut self, inputs: &[[f64; 5]], targets: &[[f64; 2]]) -> f64 {
        assert!(!inputs.is_empty());
        assert_eq!(inputs.len(), targets.len());
        self.set_train(true);

        // Reusable forward buffers
        let mut h1 = vec![0.0; 64];
        let mut h1_relu = vec![0.0; 64];
        let mut h1_bn = vec![0.0; 64];
        let mut h2 = vec![0.0; 32];
        let mut h2_relu = vec![0.0; 32];
        let mut h2_bn = vec![0.0; 32];
        let mut h3 = vec![0.0; 16];
        let mut h3_relu = vec![0.0; 16];
        let mut h3_bn = vec![0.0; 16];
        let mut output = vec![0.0; 2];

        // Reusable backward buffers
        let mut loss_grad = vec![0.0; 2];
        let mut grad_h3_bn = vec![0.0; 16];
        let mut grad_h3_relu = vec![0.0; 16];
        let mut grad_h3 = vec![0.0; 16];
        let mut grad_h2_bn = vec![0.0; 32];
        let mut grad_h2_relu = vec![0.0; 32];
        let mut grad_h2 = vec![0.0; 32];
        let mut grad_h1_bn = vec![0.0; 64];
        let mut grad_h1_relu = vec![0.0; 64];
        let mut grad_h1 = vec![0.0; 64];
        let mut grad_input = vec![0.0; 5];

        // Gradient accumulators
        let mut acc_w1 = vec![0.0; self.fc1.weights.len()];
        let mut acc_b1 = vec![0.0; self.fc1.bias.len()];
        let mut acc_w2 = vec![0.0; self.fc2.weights.len()];
        let mut acc_b2 = vec![0.0; self.fc2.bias.len()];
        let mut acc_w3 = vec![0.0; self.fc3.weights.len()];
        let mut acc_b3 = vec![0.0; self.fc3.bias.len()];
        let mut acc_w4 = vec![0.0; self.fc4.weights.len()];
        let mut acc_b4 = vec![0.0; self.fc4.bias.len()];
        let mut acc_bn1_gamma = vec![0.0; 64];
        let mut acc_bn1_beta = vec![0.0; 64];
        let mut acc_bn2_gamma = vec![0.0; 32];
        let mut acc_bn2_beta = vec![0.0; 32];
        let mut acc_bn3_gamma = vec![0.0; 16];
        let mut acc_bn3_beta = vec![0.0; 16];

        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            // ---- Forward pass ----
            self.fc1.forward_into(input, &mut h1);
            for i in 0..64 {
                h1_relu[i] = if h1[i] > 0.0 { h1[i] } else { 0.0 };
            }
            let bn1_stats = self.bn1.forward_into(&h1_relu, &mut h1_bn);

            self.fc2.forward_into(&h1_bn, &mut h2);
            for i in 0..32 {
                h2_relu[i] = if h2[i] > 0.0 { h2[i] } else { 0.0 };
            }
            let bn2_stats = self.bn2.forward_into(&h2_relu, &mut h2_bn);

            self.fc3.forward_into(&h2_bn, &mut h3);
            for i in 0..16 {
                h3_relu[i] = if h3[i] > 0.0 { h3[i] } else { 0.0 };
            }
            let bn3_stats = self.bn3.forward_into(&h3_relu, &mut h3_bn);

            self.fc4.forward_into(&h3_bn, &mut output);

            // ---- Compute MSE loss ----
            let mut loss = 0.0;
            for i in 0..2 {
                let err = output[i] - target[i];
                loss += 0.5 * err * err;
                loss_grad[i] = err;
            }
            total_loss += loss;

            // ---- Backward pass ----
            self.fc4.backward_into(
                &h3_bn,
                &loss_grad,
                &mut acc_w4,
                &mut acc_b4,
                &mut grad_h3_bn,
            );

            Self::bn_backward_x_buf(
                &self.bn3,
                &h3_relu,
                &bn3_stats,
                &grad_h3_bn,
                &mut grad_h3_relu,
            );
            for i in 0..16 {
                grad_h3[i] = if h3[i] > 0.0 { grad_h3_relu[i] } else { 0.0 };
            }

            self.fc3
                .backward_into(&h2_bn, &grad_h3, &mut acc_w3, &mut acc_b3, &mut grad_h2_bn);

            Self::bn_backward_x_buf(
                &self.bn2,
                &h2_relu,
                &bn2_stats,
                &grad_h2_bn,
                &mut grad_h2_relu,
            );
            for i in 0..32 {
                grad_h2[i] = if h2[i] > 0.0 { grad_h2_relu[i] } else { 0.0 };
            }

            self.fc2
                .backward_into(&h1_bn, &grad_h2, &mut acc_w2, &mut acc_b2, &mut grad_h1_bn);

            Self::bn_backward_x_buf(
                &self.bn1,
                &h1_relu,
                &bn1_stats,
                &grad_h1_bn,
                &mut grad_h1_relu,
            );
            for i in 0..64 {
                grad_h1[i] = if h1[i] > 0.0 { grad_h1_relu[i] } else { 0.0 };
            }

            self.fc1
                .backward_into(input, &grad_h1, &mut acc_w1, &mut acc_b1, &mut grad_input);

            // BN gamma/beta gradients
            let (g1_g, g1_b) =
                Self::bn_backward_gamma_beta_buf(&self.bn1, &h1_relu, &bn1_stats, &grad_h1_relu);
            let (g2_g, g2_b) =
                Self::bn_backward_gamma_beta_buf(&self.bn2, &h2_relu, &bn2_stats, &grad_h2_relu);
            let (g3_g, g3_b) =
                Self::bn_backward_gamma_beta_buf(&self.bn3, &h3_relu, &bn3_stats, &grad_h3_relu);

            for i in 0..64 {
                acc_bn1_gamma[i] += g1_g[i];
                acc_bn1_beta[i] += g1_b[i];
            }
            for i in 0..32 {
                acc_bn2_gamma[i] += g2_g[i];
                acc_bn2_beta[i] += g2_b[i];
            }
            for i in 0..16 {
                acc_bn3_gamma[i] += g3_g[i];
                acc_bn3_beta[i] += g3_b[i];
            }
        }

        // ---- Average gradients ----
        let n = inputs.len() as f64;
        for v in acc_w1.iter_mut() {
            *v /= n;
        }
        for v in acc_b1.iter_mut() {
            *v /= n;
        }
        for v in acc_w2.iter_mut() {
            *v /= n;
        }
        for v in acc_b2.iter_mut() {
            *v /= n;
        }
        for v in acc_w3.iter_mut() {
            *v /= n;
        }
        for v in acc_b3.iter_mut() {
            *v /= n;
        }
        for v in acc_w4.iter_mut() {
            *v /= n;
        }
        for v in acc_b4.iter_mut() {
            *v /= n;
        }
        for v in acc_bn1_gamma.iter_mut() {
            *v /= n;
        }
        for v in acc_bn1_beta.iter_mut() {
            *v /= n;
        }
        for v in acc_bn2_gamma.iter_mut() {
            *v /= n;
        }
        for v in acc_bn2_beta.iter_mut() {
            *v /= n;
        }
        for v in acc_bn3_gamma.iter_mut() {
            *v /= n;
        }
        for v in acc_bn3_beta.iter_mut() {
            *v /= n;
        }

        // ---- Adam updates ----
        self.opt1_w.step(&mut self.fc1.weights, &acc_w1);
        self.opt1_b.step(&mut self.fc1.bias, &acc_b1);
        self.opt2_w.step(&mut self.fc2.weights, &acc_w2);
        self.opt2_b.step(&mut self.fc2.bias, &acc_b2);
        self.opt3_w.step(&mut self.fc3.weights, &acc_w3);
        self.opt3_b.step(&mut self.fc3.bias, &acc_b3);
        self.opt4_w.step(&mut self.fc4.weights, &acc_w4);
        self.opt4_b.step(&mut self.fc4.bias, &acc_b4);
        self.opt_bn1.step(&mut self.bn1.gamma, &acc_bn1_gamma);
        self.opt_bn1.step(&mut self.bn1.beta, &acc_bn1_beta);
        self.opt_bn2.step(&mut self.bn2.gamma, &acc_bn2_gamma);
        self.opt_bn2.step(&mut self.bn2.beta, &acc_bn2_beta);
        self.opt_bn3.step(&mut self.bn3.gamma, &acc_bn3_gamma);
        self.opt_bn3.step(&mut self.bn3.beta, &acc_bn3_beta);

        total_loss / n
    }

    /// BN backward: compute gradient w.r.t. input (buffer version).
    fn bn_backward_x_buf(
        bn: &BatchNorm1d,
        x: &[f64],
        stats: &BnStats,
        grad_out: &[f64],
        grad_x: &mut [f64],
    ) {
        let dim = x.len();
        assert_eq!(grad_out.len(), dim);
        assert_eq!(bn.gamma.len(), dim);
        assert_eq!(grad_x.len(), dim);

        let sum_grad_out: f64 = grad_out.iter().sum();
        let mean_grad_out = sum_grad_out / dim as f64;

        let mean_grad_x = grad_out
            .iter()
            .zip(x.iter())
            .map(|(&go, &xi)| go * (xi - stats.mean))
            .sum::<f64>()
            / dim as f64;

        for i in 0..dim {
            let term1 = grad_out[i] - mean_grad_out;
            let term2 = (x[i] - stats.mean) * mean_grad_x / (stats.var + bn.eps);
            grad_x[i] = bn.gamma[i] / stats.std_dev * (term1 - term2);
        }
    }

    /// BN backward: compute gradients w.r.t. gamma and beta (buffer version).
    fn bn_backward_gamma_beta_buf(
        bn: &BatchNorm1d,
        x: &[f64],
        stats: &BnStats,
        grad_out: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = x.len();
        let mut grad_gamma = vec![0.0; dim];
        let mut grad_beta = vec![0.0; dim];
        for i in 0..dim {
            grad_gamma[i] = grad_out[i] * (x[i] - stats.mean) / stats.std_dev;
            grad_beta[i] = grad_out[i];
        }
        (grad_gamma, grad_beta)
    }

    /// Forward with stats collection (for training).
    fn bn_forward_with_stats(bn: &mut BatchNorm1d, x: &[f64]) -> (Vec<f64>, BnStats) {
        let dim = x.len();
        let sum: f64 = x.iter().sum();
        let mean = sum / dim as f64;
        let var_sum = x
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>();
        let var = var_sum / dim as f64;
        let std_dev = (var + bn.eps).sqrt();

        // Update running stats
        for i in 0..dim {
            bn.running_mean[i] = bn.momentum * bn.running_mean[i] + (1.0 - bn.momentum) * mean;
            bn.running_var[i] = bn.momentum * bn.running_var[i] + (1.0 - bn.momentum) * var;
        }

        let mut output = Vec::with_capacity(dim);
        for i in 0..dim {
            let norm = (x[i] - mean) / std_dev;
            output.push(bn.gamma[i] * norm + bn.beta[i]);
        }
        (output, BnStats { mean, var, std_dev })
    }

    fn relu_backward(x: &[f64], grad_out: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), grad_out.len());
        x.iter()
            .zip(grad_out.iter())
            .map(|(xi, &go)| if *xi > 0.0 { go } else { 0.0 })
            .collect()
    }

    /// BN backward: compute gradient w.r.t. input.
    ///
    /// Uses the chain rule through the BN forward:
    /// y = gamma * (x - mean) / std_dev + beta
    fn bn_backward_x(
        bn: &BatchNorm1d,
        x: &[f64],
        _y: &[f64],
        stats: &BnStats,
        grad_out: &[f64],
    ) -> Vec<f64> {
        let dim = x.len();
        assert_eq!(grad_out.len(), dim);
        assert_eq!(bn.gamma.len(), dim);

        // mean(dL/d_y)
        let sum_grad_out: f64 = grad_out.iter().sum();
        let mean_grad_out = sum_grad_out / dim as f64;

        // mean(dL/d_y * (x - mean))
        let mean_grad_x = grad_out
            .iter()
            .zip(x.iter())
            .map(|(&go, &xi)| go * (xi - stats.mean))
            .sum::<f64>()
            / dim as f64;

        // dL/d_x = (gamma / std_dev) * (dL/d_y - mean(dL/d_y) - (x-mean)*mean(dL/d_y*(x-mean))/(var+eps))
        let mut grad_x = Vec::with_capacity(dim);
        for i in 0..dim {
            let term1 = grad_out[i] - mean_grad_out;
            let term2 = (x[i] - stats.mean) * mean_grad_x / (stats.var + bn.eps);
            grad_x.push(bn.gamma[i] / stats.std_dev * (term1 - term2));
        }
        grad_x
    }

    /// BN backward: compute gradients w.r.t. gamma and beta.
    ///
    /// dL/d_gamma = sum(dL/d_y * (x - mean) / std_dev)
    /// dL/d_beta  = sum(dL/d_y)
    fn bn_backward_gamma_beta(
        bn: &BatchNorm1d,
        x: &[f64],
        stats: &BnStats,
        grad_out: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = x.len();
        let mut grad_gamma = vec![0.0; dim];
        let mut grad_beta = vec![0.0; dim];
        for i in 0..dim {
            grad_gamma[i] = grad_out[i] * (x[i] - stats.mean) / stats.std_dev;
            grad_beta[i] = grad_out[i];
        }
        (grad_gamma, grad_beta)
    }

    fn bn_backward(&mut self, bn: &BatchNorm1d, x: &[f64]) -> (Vec<f64>, (Vec<f64>, Vec<f64>)) {
        let dim = x.len();

        let sum_x: f64 = x.iter().sum();
        let mean_x = sum_x / dim as f64;

        let var_sum = x
            .iter()
            .map(|v| {
                let d = v - mean_x;
                d * d
            })
            .sum::<f64>();
        let var = var_sum / dim as f64;
        let std_dev = (var + bn.eps).sqrt();

        // Output normalized (no gamma/beta applied yet for gradient)
        let mut y_normalized = Vec::with_capacity(dim);
        for i in 0..dim {
            y_normalized.push((x[i] - mean_x) / std_dev);
        }

        // Gradients for gamma and beta (placeholder - real grad comes from upstream)
        let grad_gamma_placeholder = vec![0.0; dim];
        let grad_beta_placeholder = vec![0.0; dim];

        (
            y_normalized,
            (grad_gamma_placeholder, grad_beta_placeholder),
        )
    }

    /// Create EnvNet with sensible initialization from Config.
    pub fn from_config(config: &Config, rng: &mut Rng) -> Self {
        Self::new(rng)
    }

    /// Pre-train EnvNet by generating synthetic simulation data.
    /// Runs `num_episodes` gacha simulations, collects (state, residual) pairs,
    /// and trains EnvNet via supervised MSE loss.
    pub fn pretrain(&mut self, rng: &mut Rng, config: &Config, num_episodes: usize, epochs: usize) {
        use crate::sim::{env_net_env, prob_6, PullState};
        use rayon::prelude::*;

        const BATCH_SIZE: usize = 32;
        const EPISODE_PULLS: usize = 200;

        // ── 1. Generate training data in parallel ─────────────────────────
        let base_seed = rng.next_u64();
        let (inputs, targets) = {
            let net = &*self;
            type EpisodeData = Vec<(Vec<[f64; 5]>, Vec<[f64; 2]>)>;
            let episode_data: EpisodeData = (0..num_episodes)
                .into_par_iter()
                .map(|ep| {
                    let mut state = PullState::new(config);

                    let ep_seed = base_seed.wrapping_add(ep as u64 * 7919);
                    let mut local_rng = crate::rng::Rng::from_seed(ep_seed);

                    let mut episode_inputs = Vec::with_capacity(EPISODE_PULLS);
                    let mut signed_deviations = Vec::with_capacity(EPISODE_PULLS);

                    for _ in 0..EPISODE_PULLS {
                        let base_prob = prob_6(state.pity_6, config);

                        // Advance RNG state (env_net_env may consume randomness)
                        let _ = env_net_env(
                            net,
                            &mut local_rng,
                            state.pity_6,
                            state.total_pulls_in_pool,
                            state.streak_4_star,
                            state.loss_streak,
                        );

                        let rng_val = local_rng.next_f64();
                        let pity_norm = (state.pity_6 as f64 / 80.0).clamp(0.0, 2.0);
                        let total_norm =
                            ((state.total_pulls_in_pool % 180) as f64 / 180.0).clamp(0.0, 1.0);
                        let streak_norm = (state.streak_4_star as f64 / 20.0).clamp(0.0, 2.0);
                        let loss_norm = (state.loss_streak as f64 / 3.0).clamp(0.0, 2.0);

                        let input = [rng_val, pity_norm, total_norm, streak_norm, loss_norm];

                        let r = local_rng.next_f64();
                        let is_six = r < base_prob;

                        let p_observed = if is_six { 1.0 } else { 0.0 };
                        let p_expected = base_prob.clamp(1e-8, 1.0 - 1e-8);
                        let signed_dev = p_observed - p_expected;
                        signed_deviations.push(signed_dev);

                        episode_inputs.push(input);

                        // Advance state
                        if is_six {
                            state.pity_6 = 0;
                            state.streak_4_star = 0;
                            if config.up_rate > 0.0
                                && !config.up_six.is_empty()
                                && local_rng.next_f64() < config.up_rate
                            {
                                state.loss_streak = 0;
                            } else {
                                state.loss_streak += 1;
                            }
                        } else {
                            state.pity_6 += 1;
                            let force_5 = config.always_5_star
                                || (config.five_star_pity > 0
                                    && state.streak_4_star >= config.five_star_pity - 1);
                            if force_5 || r < (base_prob + config.prob_5_base).min(1.0) {
                                state.streak_4_star = 0;
                            } else {
                                state.streak_4_star += 1;
                            }
                        }
                        state.total_pulls_in_pool += 1;
                    }

                    let observed_bias = if signed_deviations.is_empty() {
                        0.0
                    } else {
                        signed_deviations.iter().sum::<f64>() / signed_deviations.len() as f64
                    };

                    let mut episode_targets = Vec::with_capacity(signed_deviations.len());
                    for &signed_dev in &signed_deviations {
                        let scaled_noise = signed_dev * 10.0;
                        episode_targets.push([scaled_noise, observed_bias * 10.0]);
                    }

                    (episode_inputs, episode_targets)
                })
                .collect();

            let total_samples = episode_data.iter().map(|(i, _)| i.len()).sum();
            let mut inputs = Vec::with_capacity(total_samples);
            let mut targets = Vec::with_capacity(total_samples);
            for (inp, tgt) in episode_data {
                inputs.extend(inp);
                targets.extend(tgt);
            }
            (inputs, targets)
        };

        if inputs.is_empty() {
            log::warn!("[EnvNet] No training data generated, skipping pre-training.");
            return;
        }

        // ── 2. Train for `epochs` epochs ──────────────────────────────────
        let data_len = inputs.len();
        let mut indices: Vec<usize> = (0..data_len).collect();
        let num_batches = data_len.div_ceil(BATCH_SIZE);

        for epoch in 0..epochs {
            // Shuffle indices each epoch
            for i in (1..data_len).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                indices.swap(i, j);
            }

            let mut epoch_loss = 0.0;

            for batch_idx in 0..num_batches {
                let start = batch_idx * BATCH_SIZE;
                let end = ((batch_idx + 1) * BATCH_SIZE).min(data_len);

                let batch_size = end - start;
                let mut batch_inputs = vec![[0.0; 5]; batch_size];
                let mut batch_targets = vec![[0.0; 2]; batch_size];
                for i in 0..batch_size {
                    batch_inputs[i] = inputs[indices[start + i]];
                    batch_targets[i] = targets[indices[start + i]];
                }

                let batch_loss = self.train_step_batch(&batch_inputs, &batch_targets);
                epoch_loss += batch_loss * batch_size as f64;
            }

            let avg_loss = epoch_loss / data_len as f64;
            if epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 10 == 0 {
                log::info!(
                    "[EnvNet] Pre-train epoch {}/{}: loss={:.6} ({} samples)",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    data_len
                );
            }
        }

        // Set to inference mode after training
        self.set_train(false);
    }

    /// Serialize the entire network to JSON.
    /// Schema version for cache compatibility.
    const SCHEMA_VERSION: u32 = 1;
    /// Architecture signature: "in_h1_h2_h3_out" (e.g., "5_64_32_16_2").
    const ARCH_SIG: &str = "5_64_32_16_2";

    pub fn to_json(&self) -> String {
        let mut obj = serde_json::Map::new();

        // Metadata for version/architecture validation
        obj.insert(
            "schema_version".to_string(),
            serde_json::to_value(Self::SCHEMA_VERSION).unwrap(),
        );
        obj.insert(
            "arch_sig".to_string(),
            serde_json::to_value(Self::ARCH_SIG).unwrap(),
        );

        // FC layers
        obj.insert(
            "fc1_w".to_string(),
            serde_json::to_value(&self.fc1.weights).unwrap(),
        );
        obj.insert(
            "fc1_b".to_string(),
            serde_json::to_value(&self.fc1.bias).unwrap(),
        );
        obj.insert(
            "fc2_w".to_string(),
            serde_json::to_value(&self.fc2.weights).unwrap(),
        );
        obj.insert(
            "fc2_b".to_string(),
            serde_json::to_value(&self.fc2.bias).unwrap(),
        );
        obj.insert(
            "fc3_w".to_string(),
            serde_json::to_value(&self.fc3.weights).unwrap(),
        );
        obj.insert(
            "fc3_b".to_string(),
            serde_json::to_value(&self.fc3.bias).unwrap(),
        );
        obj.insert(
            "fc4_w".to_string(),
            serde_json::to_value(&self.fc4.weights).unwrap(),
        );
        obj.insert(
            "fc4_b".to_string(),
            serde_json::to_value(&self.fc4.bias).unwrap(),
        );

        // BN parameters
        obj.insert(
            "bn1_gamma".to_string(),
            serde_json::to_value(&self.bn1.gamma).unwrap(),
        );
        obj.insert(
            "bn1_beta".to_string(),
            serde_json::to_value(&self.bn1.beta).unwrap(),
        );
        obj.insert(
            "bn1_rm".to_string(),
            serde_json::to_value(&self.bn1.running_mean).unwrap(),
        );
        obj.insert(
            "bn1_rv".to_string(),
            serde_json::to_value(&self.bn1.running_var).unwrap(),
        );

        obj.insert(
            "bn2_gamma".to_string(),
            serde_json::to_value(&self.bn2.gamma).unwrap(),
        );
        obj.insert(
            "bn2_beta".to_string(),
            serde_json::to_value(&self.bn2.beta).unwrap(),
        );
        obj.insert(
            "bn2_rm".to_string(),
            serde_json::to_value(&self.bn2.running_mean).unwrap(),
        );
        obj.insert(
            "bn2_rv".to_string(),
            serde_json::to_value(&self.bn2.running_var).unwrap(),
        );

        obj.insert(
            "bn3_gamma".to_string(),
            serde_json::to_value(&self.bn3.gamma).unwrap(),
        );
        obj.insert(
            "bn3_beta".to_string(),
            serde_json::to_value(&self.bn3.beta).unwrap(),
        );
        obj.insert(
            "bn3_rm".to_string(),
            serde_json::to_value(&self.bn3.running_mean).unwrap(),
        );
        obj.insert(
            "bn3_rv".to_string(),
            serde_json::to_value(&self.bn3.running_var).unwrap(),
        );

        // Adam optimizer state (separate for weights and biases)
        obj.insert(
            "opt1_w_m".to_string(),
            serde_json::to_value(&self.opt1_w.m).unwrap(),
        );
        obj.insert(
            "opt1_w_v".to_string(),
            serde_json::to_value(&self.opt1_w.v).unwrap(),
        );
        obj.insert(
            "opt1_w_t".to_string(),
            serde_json::to_value(self.opt1_w.t).unwrap(),
        );

        obj.insert(
            "opt1_b_m".to_string(),
            serde_json::to_value(&self.opt1_b.m).unwrap(),
        );
        obj.insert(
            "opt1_b_v".to_string(),
            serde_json::to_value(&self.opt1_b.v).unwrap(),
        );
        obj.insert(
            "opt1_b_t".to_string(),
            serde_json::to_value(self.opt1_b.t).unwrap(),
        );

        obj.insert(
            "opt2_w_m".to_string(),
            serde_json::to_value(&self.opt2_w.m).unwrap(),
        );
        obj.insert(
            "opt2_w_v".to_string(),
            serde_json::to_value(&self.opt2_w.v).unwrap(),
        );
        obj.insert(
            "opt2_w_t".to_string(),
            serde_json::to_value(self.opt2_w.t).unwrap(),
        );

        obj.insert(
            "opt2_b_m".to_string(),
            serde_json::to_value(&self.opt2_b.m).unwrap(),
        );
        obj.insert(
            "opt2_b_v".to_string(),
            serde_json::to_value(&self.opt2_b.v).unwrap(),
        );
        obj.insert(
            "opt2_b_t".to_string(),
            serde_json::to_value(self.opt2_b.t).unwrap(),
        );

        obj.insert(
            "opt3_w_m".to_string(),
            serde_json::to_value(&self.opt3_w.m).unwrap(),
        );
        obj.insert(
            "opt3_w_v".to_string(),
            serde_json::to_value(&self.opt3_w.v).unwrap(),
        );
        obj.insert(
            "opt3_w_t".to_string(),
            serde_json::to_value(self.opt3_w.t).unwrap(),
        );

        obj.insert(
            "opt3_b_m".to_string(),
            serde_json::to_value(&self.opt3_b.m).unwrap(),
        );
        obj.insert(
            "opt3_b_v".to_string(),
            serde_json::to_value(&self.opt3_b.v).unwrap(),
        );
        obj.insert(
            "opt3_b_t".to_string(),
            serde_json::to_value(self.opt3_b.t).unwrap(),
        );

        obj.insert(
            "opt4_w_m".to_string(),
            serde_json::to_value(&self.opt4_w.m).unwrap(),
        );
        obj.insert(
            "opt4_w_v".to_string(),
            serde_json::to_value(&self.opt4_w.v).unwrap(),
        );
        obj.insert(
            "opt4_w_t".to_string(),
            serde_json::to_value(self.opt4_w.t).unwrap(),
        );

        obj.insert(
            "opt4_b_m".to_string(),
            serde_json::to_value(&self.opt4_b.m).unwrap(),
        );
        obj.insert(
            "opt4_b_v".to_string(),
            serde_json::to_value(&self.opt4_b.v).unwrap(),
        );
        obj.insert(
            "opt4_b_t".to_string(),
            serde_json::to_value(self.opt4_b.t).unwrap(),
        );

        obj.insert(
            "opt_bn1_m".to_string(),
            serde_json::to_value(&self.opt_bn1.m).unwrap(),
        );
        obj.insert(
            "opt_bn1_v".to_string(),
            serde_json::to_value(&self.opt_bn1.v).unwrap(),
        );
        obj.insert(
            "opt_bn1_t".to_string(),
            serde_json::to_value(self.opt_bn1.t).unwrap(),
        );

        obj.insert(
            "opt_bn2_m".to_string(),
            serde_json::to_value(&self.opt_bn2.m).unwrap(),
        );
        obj.insert(
            "opt_bn2_v".to_string(),
            serde_json::to_value(&self.opt_bn2.v).unwrap(),
        );
        obj.insert(
            "opt_bn2_t".to_string(),
            serde_json::to_value(self.opt_bn2.t).unwrap(),
        );

        obj.insert(
            "opt_bn3_m".to_string(),
            serde_json::to_value(&self.opt_bn3.m).unwrap(),
        );
        obj.insert(
            "opt_bn3_v".to_string(),
            serde_json::to_value(&self.opt_bn3.v).unwrap(),
        );
        obj.insert(
            "opt_bn3_t".to_string(),
            serde_json::to_value(self.opt_bn3.t).unwrap(),
        );

        serde_json::to_string(&obj).unwrap()
    }

    /// Deserialize from JSON with version/architecture validation.
    pub fn from_json(json_str: &str, _rng: &mut Rng) -> Option<Self> {
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(json_str).ok()?;

        // ── Version / architecture validation ─────────────────────────────
        let schema_version = map.get("schema_version")?.as_u64()? as u32;
        if schema_version != Self::SCHEMA_VERSION {
            eprintln!(
                "[EnvNet] Cache schema version mismatch: expected {}, got {}. Rebuilding.",
                Self::SCHEMA_VERSION,
                schema_version
            );
            return None;
        }
        let arch_sig = map.get("arch_sig")?.as_str()?;
        if arch_sig != Self::ARCH_SIG {
            eprintln!(
                "[EnvNet] Cache architecture mismatch: expected '{}', got '{}'. Rebuilding.",
                Self::ARCH_SIG,
                arch_sig
            );
            return None;
        }

        macro_rules! get_vec {
            ($key:expr, $expected_len:expr) => {{
                let v: Vec<f64> = map
                    .get($key)?
                    .as_array()?
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .collect();
                if v.len() != $expected_len {
                    eprintln!(
                        "[EnvNet] Cache dimension mismatch for '{}': expected {}, got {}. Rebuilding.",
                        $key, $expected_len, v.len()
                    );
                    return None;
                }
                v
            }};
        }
        macro_rules! get_usize {
            ($key:expr) => {{
                map.get($key)?.as_u64()? as usize
            }};
        }

        let fc1_w = get_vec!("fc1_w", 5 * 64);
        let fc1_b = get_vec!("fc1_b", 64);
        let fc2_w = get_vec!("fc2_w", 64 * 32);
        let fc2_b = get_vec!("fc2_b", 32);
        let fc3_w = get_vec!("fc3_w", 32 * 16);
        let fc3_b = get_vec!("fc3_b", 16);
        let fc4_w = get_vec!("fc4_w", 16 * 2);
        let fc4_b = get_vec!("fc4_b", 2);

        let bn1_gamma = get_vec!("bn1_gamma", 64);
        let bn1_beta = get_vec!("bn1_beta", 64);
        let bn1_rm = get_vec!("bn1_rm", 64);
        let bn1_rv = get_vec!("bn1_rv", 64);

        let bn2_gamma = get_vec!("bn2_gamma", 32);
        let bn2_beta = get_vec!("bn2_beta", 32);
        let bn2_rm = get_vec!("bn2_rm", 32);
        let bn2_rv = get_vec!("bn2_rv", 32);

        let bn3_gamma = get_vec!("bn3_gamma", 16);
        let bn3_beta = get_vec!("bn3_beta", 16);
        let bn3_rm = get_vec!("bn3_rm", 16);
        let bn3_rv = get_vec!("bn3_rv", 16);

        // Build layers with validated dimensions
        let mut fc1 = LinearLayer::new(5, 64, _rng);
        fc1.weights.copy_from_slice(&fc1_w);
        fc1.bias.copy_from_slice(&fc1_b);

        let mut fc2 = LinearLayer::new(64, 32, _rng);
        fc2.weights.copy_from_slice(&fc2_w);
        fc2.bias.copy_from_slice(&fc2_b);

        let mut fc3 = LinearLayer::new(32, 16, _rng);
        fc3.weights.copy_from_slice(&fc3_w);
        fc3.bias.copy_from_slice(&fc3_b);

        let mut fc4 = LinearLayer::new(16, 2, _rng);
        fc4.weights.copy_from_slice(&fc4_w);
        fc4.bias.copy_from_slice(&fc4_b);

        let mut bn1 = BatchNorm1d::new(64);
        bn1.gamma.copy_from_slice(&bn1_gamma);
        bn1.beta.copy_from_slice(&bn1_beta);
        bn1.running_mean.copy_from_slice(&bn1_rm);
        bn1.running_var.copy_from_slice(&bn1_rv);

        let mut bn2 = BatchNorm1d::new(32);
        bn2.gamma.copy_from_slice(&bn2_gamma);
        bn2.beta.copy_from_slice(&bn2_beta);
        bn2.running_mean.copy_from_slice(&bn2_rm);
        bn2.running_var.copy_from_slice(&bn2_rv);

        let mut bn3 = BatchNorm1d::new(16);
        bn3.gamma.copy_from_slice(&bn3_gamma);
        bn3.beta.copy_from_slice(&bn3_beta);
        bn3.running_mean.copy_from_slice(&bn3_rm);
        bn3.running_var.copy_from_slice(&bn3_rv);

        // Adam optimizer states (validated dimensions)
        let opt1_w_m = get_vec!("opt1_w_m", 5 * 64);
        let opt1_w_v = get_vec!("opt1_w_v", 5 * 64);
        let opt1_w_t = get_usize!("opt1_w_t");

        let opt1_b_m = get_vec!("opt1_b_m", 64);
        let opt1_b_v = get_vec!("opt1_b_v", 64);
        let opt1_b_t = get_usize!("opt1_b_t");

        let opt2_w_m = get_vec!("opt2_w_m", 64 * 32);
        let opt2_w_v = get_vec!("opt2_w_v", 64 * 32);
        let opt2_w_t = get_usize!("opt2_w_t");

        let opt2_b_m = get_vec!("opt2_b_m", 32);
        let opt2_b_v = get_vec!("opt2_b_v", 32);
        let opt2_b_t = get_usize!("opt2_b_t");

        let opt3_w_m = get_vec!("opt3_w_m", 32 * 16);
        let opt3_w_v = get_vec!("opt3_w_v", 32 * 16);
        let opt3_w_t = get_usize!("opt3_w_t");

        let opt3_b_m = get_vec!("opt3_b_m", 16);
        let opt3_b_v = get_vec!("opt3_b_v", 16);
        let opt3_b_t = get_usize!("opt3_b_t");

        let opt4_w_m = get_vec!("opt4_w_m", 16 * 2);
        let opt4_w_v = get_vec!("opt4_w_v", 16 * 2);
        let opt4_w_t = get_usize!("opt4_w_t");

        let opt4_b_m = get_vec!("opt4_b_m", 2);
        let opt4_b_v = get_vec!("opt4_b_v", 2);
        let opt4_b_t = get_usize!("opt4_b_t");

        let opt_bn1_m = get_vec!("opt_bn1_m", 64);
        let opt_bn1_v = get_vec!("opt_bn1_v", 64);
        let opt_bn1_t = get_usize!("opt_bn1_t");

        let opt_bn2_m = get_vec!("opt_bn2_m", 32);
        let opt_bn2_v = get_vec!("opt_bn2_v", 32);
        let opt_bn2_t = get_usize!("opt_bn2_t");

        let opt_bn3_m = get_vec!("opt_bn3_m", 16);
        let opt_bn3_v = get_vec!("opt_bn3_v", 16);
        let opt_bn3_t = get_usize!("opt_bn3_t");

        fn make_opt(m: Vec<f64>, v: Vec<f64>, t: usize) -> AdamOptimizer {
            AdamOptimizer {
                m,
                v,
                t,
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            }
        }

        Some(EnvNet {
            fc1,
            fc2,
            fc3,
            fc4,
            bn1,
            bn2,
            bn3,
            opt1_w: make_opt(opt1_w_m, opt1_w_v, opt1_w_t),
            opt1_b: make_opt(opt1_b_m, opt1_b_v, opt1_b_t),
            opt2_w: make_opt(opt2_w_m, opt2_w_v, opt2_w_t),
            opt2_b: make_opt(opt2_b_m, opt2_b_v, opt2_b_t),
            opt3_w: make_opt(opt3_w_m, opt3_w_v, opt3_w_t),
            opt3_b: make_opt(opt3_b_m, opt3_b_v, opt3_b_t),
            opt4_w: make_opt(opt4_w_m, opt4_w_v, opt4_w_t),
            opt4_b: make_opt(opt4_b_m, opt4_b_v, opt4_b_t),
            opt_bn1: make_opt(opt_bn1_m, opt_bn1_v, opt_bn1_t),
            opt_bn2: make_opt(opt_bn2_m, opt_bn2_v, opt_bn2_t),
            opt_bn3: make_opt(opt_bn3_m, opt_bn3_v, opt_bn3_t),
            training: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchnorm_forward_training() {
        let mut bn = BatchNorm1d::new(4);
        bn.set_train(true);

        let input = [1.0, 2.0, 3.0, 4.0];
        let output = bn.forward(&input);

        // After training, running stats should be updated
        assert_eq!(output.len(), 4);
        // Running mean should be updated
        assert!(bn.running_mean()[0] > 0.0);
    }

    #[test]
    fn batchnorm_forward_inference() {
        let mut bn = BatchNorm1d::new(4);
        bn.set_train(false);
        bn.running_mean = vec![2.5; 4];
        bn.running_var = vec![1.25; 4];

        let input = [1.0, 2.0, 3.0, 4.0];
        let output = bn.forward(&input);

        assert_eq!(output.len(), 4);
        // Without gamma/beta adjustment, normalized values should be around [-1.34, -0.45, 0.45, 1.34]
        // with gamma=1, beta=0
        let std_dev = (1.25_f64 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (input[i] - 2.5) / std_dev;
            assert!((output[i] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn adam_step() {
        let mut opt = AdamOptimizer::new(3);

        let mut params = [1.0, 2.0, 3.0];
        let grads = [0.1, -0.2, 0.05];

        opt.step(&mut params, &grads);

        // After one step, params should have moved in the negative gradient direction
        assert!(params[0] < 1.0);
        assert!(params[1] > 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn adam_multiple_steps() {
        let mut opt = AdamOptimizer::new(2);
        let mut params = [0.0, 0.0];
        let grads = [1.0, 1.0];

        // Do several steps
        for _ in 0..10 {
            opt.step(&mut params, &grads);
        }

        // Should have moved in the negative gradient direction
        assert!(params[0] < 0.0);
        assert!(params[1] < 0.0);
        assert_eq!(opt.t(), 10);
    }

    #[test]
    fn linear_layer_forward() {
        let mut rng = Rng::new();
        let layer = LinearLayer::new(3, 2, &mut rng);

        let input = [1.0, 0.5, -0.5];
        let output = layer.forward(&input);

        assert_eq!(output.len(), 2);
        // Output = input @ W^T + b
        // Each output[j] = sum_i input[i] * W[i][j] + b[j]
    }

    #[test]
    fn envnet_forward_inference() {
        let mut rng = Rng::new();
        let mut net = EnvNet::new(&mut rng);
        net.set_train(false);

        let input = [0.5, 0.3, 0.7, 0.2, 0.8];
        let (env_noise, env_bias) = net.forward_infer(&input);

        // Just verify outputs are finite
        assert!(env_noise.is_finite());
        assert!(env_bias.is_finite());
    }

    #[test]
    fn envnet_train_step() {
        let mut rng = Rng::new();
        let mut net = EnvNet::new(&mut rng);

        let input = [0.5, 0.3, 0.7, 0.2, 0.8];
        let target = [0.1, -0.05];

        let loss = net.train_step(&input, &target);

        // Loss should be finite and positive (MSE of some non-zero error)
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn envnet_serialization_roundtrip() {
        let mut rng = Rng::new();
        let net = EnvNet::new(&mut rng);

        let json = net.to_json();
        let restored = EnvNet::from_json(&json, &mut rng);

        assert!(restored.is_some());
        // Verify a forward pass gives same result
        let input = [0.5, 0.3, 0.7, 0.2, 0.8];
        let (n1, b1) = net.forward_infer(&input);
        let (n2, b2) = restored.unwrap().forward_infer(&input);
        assert!((n1 - n2).abs() < 1e-9);
        assert!((b1 - b2).abs() < 1e-9);
    }

    #[test]
    fn linear_backward() {
        let mut rng = Rng::new();
        let layer = LinearLayer::new(3, 2, &mut rng);

        let input = [1.0, 0.5, -0.5];
        let upstream = [0.5, -0.25];

        let (grad_w, grad_b, grad_x) = layer.backward(&input, &upstream);

        assert_eq!(grad_w.len(), 6); // 3 * 2
        assert_eq!(grad_b.len(), 2);
        assert_eq!(grad_x.len(), 3);

        // dL/dW[i][j] = input[i] * upstream[j]
        // So grad_w[0] = input[0] * upstream[0] = 1.0 * 0.5 = 0.5
        assert!((grad_w[0] - 0.5).abs() < 1e-9);
        // grad_w[1] = input[0] * upstream[1] = 1.0 * -0.25 = -0.25
        assert!((grad_w[1] - (-0.25)).abs() < 1e-9);

        // dL/db = upstream
        assert!((grad_b[0] - 0.5).abs() < 1e-9);
        assert!((grad_b[1] - (-0.25)).abs() < 1e-9);

        // dL/dx[j] = sum_i W[i][j] * upstream[i]
        // So grad_x[0] = W[0][0]*upstream[0] + W[0][1]*upstream[1]
        // = layer.weights[0]*0.5 + layer.weights[1]*(-0.25)
        let expected_grad_x0 = layer.weights[0] * 0.5 + layer.weights[1] * (-0.25);
        assert!((grad_x[0] - expected_grad_x0).abs() < 1e-9);
    }
}
