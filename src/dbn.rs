use crate::rng::Rng;

#[inline(always)]
fn add_scaled_row(output: &mut [f64], row: &[f64], scale: f64) {
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

#[inline(always)]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                return dot_product_avx2(a, b);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return dot_product_neon(a, b);
        }
    }
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f64], b: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    let len = a.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_pd();
    while i + 4 <= len {
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let bv = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(av, bv);
        acc = _mm256_add_pd(acc, prod);
        i += 4;
    }
    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f64], b: &[f64]) -> f64 {
    use core::arch::aarch64::*;
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f64(0.0);
    while i + 2 <= len {
        let av = vld1q_f64(a.as_ptr().add(i));
        let bv = vld1q_f64(b.as_ptr().add(i));
        let prod = vmulq_f64(av, bv);
        acc = vaddq_f64(acc, prod);
        i += 2;
    }
    let mut tmp = [0.0; 2];
    vst1q_f64(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1];
    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let len = rows.checked_mul(cols).expect("Matrix size overflow");
        Self {
            rows,
            cols,
            data: vec![0.0; len],
        }
    }

    pub fn random(rows: usize, cols: usize, rng: &mut Rng, scale: f64) -> Self {
        let mut m = Self::new(rows, cols);
        for x in m.data.iter_mut() {
            *x = (rng.next_f64() - 0.5) * 2.0 * scale;
        }
        m
    }

    #[allow(dead_code)]
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[allow(dead_code)]
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    #[inline(always)]
    pub fn affine_transform(input: &[f64], weights: &Matrix, bias: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), weights.rows);
        assert_eq!(bias.len(), weights.cols);
        
        let mut output = bias.to_vec();
        
        for (i, &in_val) in input.iter().enumerate().take(weights.rows) {
            if in_val == 0.0 { continue; }

            let row_start = i * weights.cols;
            let row = &weights.data[row_start..row_start + weights.cols];
            add_scaled_row(&mut output, row, in_val);
        }
        output
    }

    #[inline(always)]
    pub fn affine_transform_back(input: &[f64], weights: &Matrix, bias: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), weights.cols);
        assert_eq!(bias.len(), weights.rows);

        let mut output = vec![0.0; weights.rows];
        for (i, out_val) in output.iter_mut().enumerate().take(weights.rows) {
            let row_start = i * weights.cols;
            let row = &weights.data[row_start..row_start + weights.cols];
            *out_val = bias[i] + dot_product(input, row);
        }
        output
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnitType {
    Bernoulli,
    Gaussian,
}

#[derive(Clone)]
pub struct Rbm {
    pub visible_size: usize,
    pub hidden_size: usize,
    pub visible_type: UnitType,
    pub hidden_type: UnitType,
    
    pub weights: Matrix,
    pub vbias: Vec<f64>,
    pub hbias: Vec<f64>,
    
    pub d_weights: Matrix,
    pub d_vbias: Vec<f64>,
    pub d_hbias: Vec<f64>,
}

impl Rbm {
    pub fn new(visible: usize, hidden: usize, v_type: UnitType, h_type: UnitType, rng: &mut Rng) -> Self {
        let scale = (6.0 / (visible + hidden) as f64).sqrt();
        let weights = Matrix::random(visible, hidden, rng, scale);
        
        Self {
            visible_size: visible,
            hidden_size: hidden,
            visible_type: v_type,
            hidden_type: h_type,
            weights: weights.clone(),
            vbias: vec![0.0; visible],
            hbias: vec![0.0; hidden],
            d_weights: Matrix::new(visible, hidden),
            d_vbias: vec![0.0; visible],
            d_hbias: vec![0.0; hidden],
        }
    }

    #[inline]
    pub fn propup(&self, v: &[f64]) -> Vec<f64> {
        let act = Matrix::affine_transform(v, &self.weights, &self.hbias);
        match self.hidden_type {
            UnitType::Bernoulli => act.iter().map(|&x| sigmoid(x)).collect(),
            UnitType::Gaussian => act,
        }
    }

    #[inline]
    pub fn propdown(&self, h: &[f64]) -> Vec<f64> {
        let act = Matrix::affine_transform_back(h, &self.weights, &self.vbias);
        match self.visible_type {
            UnitType::Bernoulli => act.iter().map(|&x| sigmoid(x)).collect(),
            UnitType::Gaussian => act,
        }
    }

    #[inline]
    pub fn sample_h_given_v(&self, v: &[f64], rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let probs = self.propup(v);
        let samples = match self.hidden_type {
            UnitType::Bernoulli => probs.iter().map(|&p| if rng.next_f64() < p { 1.0 } else { 0.0 }).collect(),
            UnitType::Gaussian => probs.iter().map(|&mu| mu + rng.next_f64_normal()).collect(),
        };
        (probs, samples)
    }

    #[inline]
    pub fn sample_v_given_h(&self, h: &[f64], rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let probs = self.propdown(h);
        let samples = match self.visible_type {
            UnitType::Bernoulli => probs.iter().map(|&p| if rng.next_f64() < p { 1.0 } else { 0.0 }).collect(),
            UnitType::Gaussian => probs.iter().map(|&mu| mu + rng.next_f64_normal()).collect(),
        };
        (probs, samples)
    }

    pub fn cd_k_batch(&mut self, inputs: &[Vec<f64>], k: usize, lr: f64, momentum: f64, rng: &mut Rng) -> f64 {
        if inputs.is_empty() { return 0.0; }
        
        // Gradient Accumulators
        let mut grad_w = vec![0.0; self.weights.data.len()];
        let mut grad_vb = vec![0.0; self.visible_size];
        let mut grad_hb = vec![0.0; self.hidden_size];
        let mut total_error = 0.0;

        for input in inputs {
            assert_eq!(input.len(), self.visible_size, "Input size mismatch");

            let (h0_probs, h0_samples) = self.sample_h_given_v(input, rng);
            
            let mut hk_samples = h0_samples.clone();
            let mut vk_probs = vec![];
            let mut vk_samples = vec![];

            for _ in 0..k {
                let (vp, vs) = self.sample_v_given_h(&hk_samples, rng);
                vk_probs = vp;
                vk_samples = vs;
                let (_, hs) = self.sample_h_given_v(&vk_samples, rng);
                hk_samples = hs;
            }
            
            let (hk_probs, _) = self.sample_h_given_v(&vk_samples, rng);

            // Accumulate Gradients (Positive phase - Negative phase)
            for (i, &input_i) in input.iter().enumerate().take(self.visible_size) {
                let vk_sample_i = unsafe { *vk_samples.get_unchecked(i) };
                let row_offset = i * self.hidden_size;

                for j in 0..self.hidden_size {
                    unsafe {
                        let pos_grad = input_i * *h0_probs.get_unchecked(j);
                        let neg_grad = vk_sample_i * *hk_probs.get_unchecked(j);
                        grad_w[row_offset + j] += pos_grad - neg_grad;
                    }
                }
                grad_vb[i] += input_i - vk_sample_i;
            }

            for j in 0..self.hidden_size {
                unsafe {
                    grad_hb[j] += *h0_probs.get_unchecked(j) - *hk_probs.get_unchecked(j);
                }
            }

            // Reconstruction Error
            for (i, &input_i) in input.iter().enumerate().take(self.visible_size) {
                unsafe {
                    total_error += (input_i - *vk_probs.get_unchecked(i)).powi(2);
                }
            }
        }

        // Apply Gradients
        let batch_size = inputs.len() as f64;
        
        // Update Weights
        for (i, g) in grad_w.iter().enumerate() {
            unsafe {
                let avg_grad = g / batch_size;
                // Standard Momentum Update
                // v = mu * v + lr * grad
                // w = w + v
                // Note: Previous implementation was: d_weight = momentum * d_weight + lr * grad
                // which implies d_weight IS the update step.
                let d_weight = momentum * *self.d_weights.data.get_unchecked(i) + lr * avg_grad;
                *self.d_weights.data.get_unchecked_mut(i) = d_weight;
                *self.weights.data.get_unchecked_mut(i) += d_weight;
            }
        }

        // Update Visible Bias
        for (i, g) in grad_vb.iter().enumerate() {
            unsafe {
                let avg_grad = g / batch_size;
                let d_vbias = momentum * *self.d_vbias.get_unchecked(i) + lr * avg_grad;
                *self.d_vbias.get_unchecked_mut(i) = d_vbias;
                *self.vbias.get_unchecked_mut(i) += d_vbias;
            }
        }

        // Update Hidden Bias
        for (i, g) in grad_hb.iter().enumerate() {
            unsafe {
                let avg_grad = g / batch_size;
                let d_hbias = momentum * *self.d_hbias.get_unchecked(i) + lr * avg_grad;
                *self.d_hbias.get_unchecked_mut(i) = d_hbias;
                *self.hbias.get_unchecked_mut(i) += d_hbias;
            }
        }

        total_error / (batch_size * self.visible_size as f64)
    }
}

pub struct Dbn {
    pub rbms: Vec<Rbm>,
}

impl Dbn {
    pub fn new(layer_sizes: &[usize], rng: &mut Rng) -> Self {
        let mut rbms = Vec::new();
        if layer_sizes.len() < 2 {
            panic!("DBN must have at least 2 layers");
        }

        rbms.push(Rbm::new(layer_sizes[0], layer_sizes[1], UnitType::Gaussian, UnitType::Bernoulli, rng));

        for i in 1..layer_sizes.len() - 1 {
            rbms.push(Rbm::new(layer_sizes[i], layer_sizes[i + 1], UnitType::Bernoulli, UnitType::Bernoulli, rng));
        }
        
        Self { rbms }
    }
    
    pub fn train(&mut self, rng: &mut Rng, data_count: usize, epochs: usize) {
        let mut input_data: Vec<Vec<f64>> = (0..data_count)
            .map(|_| {
                (0..self.rbms[0].visible_size)
                    .map(|_| rng.next_f64_normal())
                    .collect()
            })
            .collect();

        println!("Training started...");
        let num_layers = self.rbms.len();
        let batch_size = 32;

        for i in 0..num_layers {
            let rbm = &mut self.rbms[i];
            println!("Layer {}", i+1);
            
            for epoch in 0..epochs {
                let mut error_sum = 0.0;
                
                // Shuffle data (optional but good practice)
                // For now, just chunk it.
                
                let mut batches = 0;
                for chunk in input_data.chunks(batch_size) {
                     error_sum += rbm.cd_k_batch(chunk, 1, 0.01, 0.5, rng);
                     batches += 1;
                }
                
                if epoch == 0 || (epoch + 1) == epochs {
                     println!("Epoch {}: Error = {:.5}", epoch + 1, error_sum / batches as f64);
                }
            }

            if i < num_layers - 1 {
                let mut next_layer_input = Vec::with_capacity(input_data.len());
                for v in &input_data {
                    let (_, h_samples) = rbm.sample_h_given_v(v, rng);
                    next_layer_input.push(h_samples);
                }
                input_data = next_layer_input;
            }
        }
        println!("Training Complete.");
    }

    pub fn sample(&self, rng: &mut Rng, gibbs_steps: usize) -> Vec<f64> {
        let top_layer = self.rbms.last().unwrap();
        
        let mut h = (0..top_layer.hidden_size)
            .map(|_| if rng.next_f64() < 0.5 { 1.0 } else { 0.0 })
            .collect::<Vec<f64>>();

        for _ in 0..gibbs_steps {
            let (_, v_samples) = top_layer.sample_v_given_h(&h, rng);
            let (_, h_samples) = top_layer.sample_h_given_v(&v_samples, rng);
            h = h_samples;
        }

        let mut current_activations = h;
        
        for i in (0..self.rbms.len()).rev() {
            let rbm = &self.rbms[i];
            let (_, v_samples) = rbm.sample_v_given_h(&current_activations, rng);
            current_activations = v_samples;
        }

        current_activations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_scaled_row_matches_scalar() {
        let mut out_simd = [0.0; 8];
        let mut out_scalar = [0.0; 8];
        let row = [1.0, 2.5, -3.0, 4.0, -5.5, 6.25, 7.75, -8.0];
        let scale = 0.75;
        add_scaled_row(&mut out_simd, &row, scale);
        for i in 0..row.len() {
            out_scalar[i] += scale * row[i];
        }
        for i in 0..row.len() {
            assert!((out_simd[i] - out_scalar[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn dot_product_matches_scalar() {
        let a = [1.5, -2.0, 3.25, 4.0, -5.0, 6.5, -7.25, 8.0];
        let b = [0.5, 1.25, -2.0, 3.0, 4.5, -5.5, 6.0, -7.0];
        let mut expected = 0.0;
        for i in 0..a.len() {
            expected += a[i] * b[i];
        }
        let got = dot_product(&a, &b);
        assert!((got - expected).abs() < 1e-9);
    }
}
