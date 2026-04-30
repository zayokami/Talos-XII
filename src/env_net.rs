//! EnvNet: Hand-rolled MLP to replace DBN for environment noise modeling.
//!
//! Architecture: 5 → 64 → 32 → 16 → 2
//! Input:  [rng, pity_6_norm, total_pulls_norm, streak_norm, loss_streak_norm]
//! Output: [env_noise, env_bias]
#![allow(dead_code, unused, clippy::needless_range_loop)]

use crate::config::Config;
use crate::rng::Rng;

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
            let row_start = i * self.out_features;
            for j in 0..self.out_features {
                output[j] += in_val * self.weights[row_start + j];
            }
        }
        output
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
            for j in 0..self.out_features {
                grad_x[i] += self.weights[row_start + j] * upstream[j];
            }
        }

        (grad_w, grad_b, grad_x)
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
struct BnStats {
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

    /// Serialize the entire network to JSON.
    pub fn to_json(&self) -> String {
        let mut obj = serde_json::Map::new();

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

    /// Deserialize from JSON.
    pub fn from_json(json_str: &str, rng: &mut Rng) -> Option<Self> {
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(json_str).ok()?;

        macro_rules! get_vec {
            ($key:expr) => {{
                map.get($key)?
                    .as_array()?
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .collect::<Vec<f64>>()
            }};
        }
        macro_rules! get_usize {
            ($key:expr) => {{
                map.get($key)?.as_u64()? as usize
            }};
        }

        let fc1_w: Vec<f64> = get_vec!("fc1_w");
        let fc1_b: Vec<f64> = get_vec!("fc1_b");
        let fc2_w: Vec<f64> = get_vec!("fc2_w");
        let fc2_b: Vec<f64> = get_vec!("fc2_b");
        let fc3_w: Vec<f64> = get_vec!("fc3_w");
        let fc3_b: Vec<f64> = get_vec!("fc3_b");
        let fc4_w: Vec<f64> = get_vec!("fc4_w");
        let fc4_b: Vec<f64> = get_vec!("fc4_b");

        let bn1_gamma: Vec<f64> = get_vec!("bn1_gamma");
        let bn1_beta: Vec<f64> = get_vec!("bn1_beta");
        let bn1_rm: Vec<f64> = get_vec!("bn1_rm");
        let bn1_rv: Vec<f64> = get_vec!("bn1_rv");

        let bn2_gamma: Vec<f64> = get_vec!("bn2_gamma");
        let bn2_beta: Vec<f64> = get_vec!("bn2_beta");
        let bn2_rm: Vec<f64> = get_vec!("bn2_rm");
        let bn2_rv: Vec<f64> = get_vec!("bn2_rv");

        let bn3_gamma: Vec<f64> = get_vec!("bn3_gamma");
        let bn3_beta: Vec<f64> = get_vec!("bn3_beta");
        let bn3_rm: Vec<f64> = get_vec!("bn3_rm");
        let bn3_rv: Vec<f64> = get_vec!("bn3_rv");

        // Build layers
        let mut fc1 = LinearLayer::new(5, 64, rng);
        fc1.weights.copy_from_slice(&fc1_w);
        fc1.bias.copy_from_slice(&fc1_b);

        let mut fc2 = LinearLayer::new(64, 32, rng);
        fc2.weights.copy_from_slice(&fc2_w);
        fc2.bias.copy_from_slice(&fc2_b);

        let mut fc3 = LinearLayer::new(32, 16, rng);
        fc3.weights.copy_from_slice(&fc3_w);
        fc3.bias.copy_from_slice(&fc3_b);

        let mut fc4 = LinearLayer::new(16, 2, rng);
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

        // Adam optimizer states (separate for weights and biases)
        let opt1_w_m: Vec<f64> = get_vec!("opt1_w_m");
        let opt1_w_v: Vec<f64> = get_vec!("opt1_w_v");
        let opt1_w_t = get_usize!("opt1_w_t");

        let opt1_b_m: Vec<f64> = get_vec!("opt1_b_m");
        let opt1_b_v: Vec<f64> = get_vec!("opt1_b_v");
        let opt1_b_t = get_usize!("opt1_b_t");

        let opt2_w_m: Vec<f64> = get_vec!("opt2_w_m");
        let opt2_w_v: Vec<f64> = get_vec!("opt2_w_v");
        let opt2_w_t = get_usize!("opt2_w_t");

        let opt2_b_m: Vec<f64> = get_vec!("opt2_b_m");
        let opt2_b_v: Vec<f64> = get_vec!("opt2_b_v");
        let opt2_b_t = get_usize!("opt2_b_t");

        let opt3_w_m: Vec<f64> = get_vec!("opt3_w_m");
        let opt3_w_v: Vec<f64> = get_vec!("opt3_w_v");
        let opt3_w_t = get_usize!("opt3_w_t");

        let opt3_b_m: Vec<f64> = get_vec!("opt3_b_m");
        let opt3_b_v: Vec<f64> = get_vec!("opt3_b_v");
        let opt3_b_t = get_usize!("opt3_b_t");

        let opt4_w_m: Vec<f64> = get_vec!("opt4_w_m");
        let opt4_w_v: Vec<f64> = get_vec!("opt4_w_v");
        let opt4_w_t = get_usize!("opt4_w_t");

        let opt4_b_m: Vec<f64> = get_vec!("opt4_b_m");
        let opt4_b_v: Vec<f64> = get_vec!("opt4_b_v");
        let opt4_b_t = get_usize!("opt4_b_t");

        let opt_bn1_m: Vec<f64> = get_vec!("opt_bn1_m");
        let opt_bn1_v: Vec<f64> = get_vec!("opt_bn1_v");
        let opt_bn1_t = get_usize!("opt_bn1_t");

        let opt_bn2_m: Vec<f64> = get_vec!("opt_bn2_m");
        let opt_bn2_v: Vec<f64> = get_vec!("opt_bn2_v");
        let opt_bn2_t = get_usize!("opt_bn2_t");

        let opt_bn3_m: Vec<f64> = get_vec!("opt_bn3_m");
        let opt_bn3_v: Vec<f64> = get_vec!("opt_bn3_v");
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
