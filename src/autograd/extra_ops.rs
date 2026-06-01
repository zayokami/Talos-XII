use crate::autograd::{Context, Device, Tensor, PAR_THRESHOLD};
use crate::dtype::Storage;
use rayon::prelude::*;
use std::sync::Arc;

const DIV_EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
enum ReductionKind {
    Sum,
    Mean,
    Max,
}

#[inline]
fn safe_nonzero(x: f64) -> f64 {
    if x.abs() < DIV_EPS {
        if x.is_sign_negative() {
            -DIV_EPS
        } else {
            DIV_EPS
        }
    } else {
        x
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn checked_product(shape: &[usize], op: &'static str) -> usize {
    shape.iter().copied().fold(1usize, |acc, dim| {
        acc.checked_mul(dim)
            .unwrap_or_else(|| panic!("{op} element count overflow"))
    })
}

fn checked_mul(lhs: usize, rhs: usize, op: &'static str) -> usize {
    lhs.checked_mul(rhs)
        .unwrap_or_else(|| panic!("{op} element count overflow"))
}

// Abramowitz and Stegun 7.1.26. Maximum absolute error is about 1.5e-7.
#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

impl Tensor {
    fn unary_autograd<F, G>(&self, forward: F, derivative: G) -> Tensor
    where
        F: Fn(f64) -> f64 + Send + Sync,
        G: Fn(f64, f64) -> f64 + Send + Sync + 'static,
    {
        let input = self.data_as_f64_vec();
        let len = input.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            input.par_iter().map(|&x| forward(x)).collect()
        } else {
            input.iter().map(|&x| forward(x)).collect()
        };
        let input_cache = Arc::new(input);
        let output_cache = Arc::new(data.clone());
        let derivative = Arc::new(derivative);
        let dtype = self.dtype;

        Tensor {
            data: Storage::from_f64_vec(data, dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_grad = parents[0].grad_write_compat();
                    if len >= PAR_THRESHOLD {
                        input_grad
                            .par_iter_mut()
                            .zip(grad_out.par_iter())
                            .zip(input_cache.par_iter())
                            .zip(output_cache.par_iter())
                            .for_each(|(((dst, &grad), &x), &y)| {
                                *dst += grad * derivative(x, y);
                            });
                    } else {
                        for i in 0..len {
                            input_grad[i] +=
                                grad_out[i] * derivative(input_cache[i], output_cache[i]);
                        }
                    }
                }),
            })),
        }
    }

    fn binary_autograd<F, GL, GR>(
        &self,
        rhs: &Tensor,
        op_name: &'static str,
        forward: F,
        lhs_derivative: GL,
        rhs_derivative: GR,
    ) -> Tensor
    where
        F: Fn(f64, f64) -> f64 + Send + Sync,
        GL: Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
        GR: Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    {
        self.assert_same_numel(rhs, op_name);
        let lhs_data = self.data_as_f64_vec();
        let rhs_data = rhs.data_as_f64_vec();
        let len = lhs_data.len();
        let data: Vec<f64> = if len >= PAR_THRESHOLD {
            lhs_data
                .par_iter()
                .zip(rhs_data.par_iter())
                .map(|(&lhs, &rhs)| forward(lhs, rhs))
                .collect()
        } else {
            lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(&lhs, &rhs)| forward(lhs, rhs))
                .collect()
        };
        let lhs_cache = Arc::new(lhs_data);
        let rhs_cache = Arc::new(rhs_data);
        let out_cache = Arc::new(data.clone());
        let lhs_derivative = Arc::new(lhs_derivative);
        let rhs_derivative = Arc::new(rhs_derivative);
        let dtype = Tensor::binary_dtype(self.dtype, rhs.dtype);

        Tensor {
            data: Storage::from_f64_vec(data, dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), rhs.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    if parents[0].grad.ptr_eq(&parents[1].grad) {
                        let mut grad = parents[0].grad_write_compat();
                        for i in 0..len {
                            grad[i] += grad_out[i]
                                * (lhs_derivative(lhs_cache[i], rhs_cache[i], out_cache[i])
                                    + rhs_derivative(lhs_cache[i], rhs_cache[i], out_cache[i]));
                        }
                    } else {
                        let mut lhs_grad = parents[0].grad_write_compat();
                        let mut rhs_grad = parents[1].grad_write_compat();
                        for i in 0..len {
                            lhs_grad[i] += grad_out[i]
                                * lhs_derivative(lhs_cache[i], rhs_cache[i], out_cache[i]);
                            rhs_grad[i] += grad_out[i]
                                * rhs_derivative(lhs_cache[i], rhs_cache[i], out_cache[i]);
                        }
                    }
                }),
            })),
        }
    }

    fn reduce_dim_impl(&self, dim: usize, keepdim: bool, kind: ReductionKind) -> Tensor {
        assert!(dim < self.shape.len(), "reduction dim out of bounds");
        let input = self.data_as_f64_vec();
        let dim_size = self.shape[dim];
        assert!(dim_size > 0, "cannot reduce an empty dimension");
        let inner = checked_product(&self.shape[dim + 1..], "reduce");
        let outer = checked_product(&self.shape[..dim], "reduce");
        let out_len = checked_mul(outer, inner, "reduce");
        let mut output = match kind {
            ReductionKind::Max => vec![f64::NEG_INFINITY; out_len],
            _ => vec![0.0; out_len],
        };

        for outer_idx in 0..outer {
            for dim_idx in 0..dim_size {
                let input_base = (outer_idx * dim_size + dim_idx) * inner;
                let output_base = outer_idx * inner;
                for inner_idx in 0..inner {
                    let value = input[input_base + inner_idx];
                    let out = &mut output[output_base + inner_idx];
                    match kind {
                        ReductionKind::Sum | ReductionKind::Mean => *out += value,
                        ReductionKind::Max => *out = out.max(value),
                    }
                }
            }
        }
        if matches!(kind, ReductionKind::Mean) {
            for value in &mut output {
                *value /= dim_size as f64;
            }
        }

        let mut shape = self.shape.clone();
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
            if shape.is_empty() {
                shape.push(1);
            }
        }

        let input_cache = Arc::new(input);
        let output_cache = Arc::new(output.clone());
        let dtype = self.dtype;
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape,
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_grad = parents[0].grad_write_compat();
                    for outer_idx in 0..outer {
                        for inner_idx in 0..inner {
                            let out_idx = outer_idx * inner + inner_idx;
                            let grad = match kind {
                                ReductionKind::Sum => grad_out[out_idx],
                                ReductionKind::Mean => grad_out[out_idx] / dim_size as f64,
                                ReductionKind::Max => {
                                    let mut ties = 0usize;
                                    for dim_idx in 0..dim_size {
                                        let idx =
                                            (outer_idx * dim_size + dim_idx) * inner + inner_idx;
                                        if input_cache[idx] == output_cache[out_idx] {
                                            ties += 1;
                                        }
                                    }
                                    grad_out[out_idx] / ties.max(1) as f64
                                }
                            };
                            for dim_idx in 0..dim_size {
                                let idx = (outer_idx * dim_size + dim_idx) * inner + inner_idx;
                                if !matches!(kind, ReductionKind::Max)
                                    || input_cache[idx] == output_cache[out_idx]
                                {
                                    input_grad[idx] += grad;
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }

    fn boolean_reduce_dim(&self, dim: usize, keepdim: bool, reduce_all: bool) -> Tensor {
        assert!(dim < self.shape.len(), "reduction dim out of bounds");
        let input = self.data_as_f64_vec();
        let dim_size = self.shape[dim];
        assert!(dim_size > 0, "cannot reduce an empty dimension");
        let inner = checked_product(&self.shape[dim + 1..], "boolean_reduce");
        let outer = checked_product(&self.shape[..dim], "boolean_reduce");
        let out_len = checked_mul(outer, inner, "boolean_reduce");
        let mut output = vec![if reduce_all { 1.0 } else { 0.0 }; out_len];

        for outer_idx in 0..outer {
            for dim_idx in 0..dim_size {
                let input_base = (outer_idx * dim_size + dim_idx) * inner;
                let output_base = outer_idx * inner;
                for inner_idx in 0..inner {
                    let truthy = input[input_base + inner_idx] != 0.0;
                    let out = &mut output[output_base + inner_idx];
                    if reduce_all {
                        if !truthy {
                            *out = 0.0;
                        }
                    } else if truthy {
                        *out = 1.0;
                    }
                }
            }
        }

        let mut shape = self.shape.clone();
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
            if shape.is_empty() {
                shape.push(1);
            }
        }
        Tensor::with_dtype(output, shape, self.dtype)
    }

    pub fn acos(&self) -> Tensor {
        self.unary_autograd(f64::acos, |x, _| -1.0 / (1.0 - x * x).sqrt())
    }

    pub fn asin(&self) -> Tensor {
        self.unary_autograd(f64::asin, |x, _| 1.0 / (1.0 - x * x).sqrt())
    }

    pub fn asinh(&self) -> Tensor {
        self.unary_autograd(f64::asinh, |x, _| 1.0 / (1.0 + x * x).sqrt())
    }

    pub fn atan(&self) -> Tensor {
        self.unary_autograd(f64::atan, |x, _| 1.0 / (1.0 + x * x))
    }

    pub fn cosh(&self) -> Tensor {
        self.unary_autograd(f64::cosh, |x, _| x.sinh())
    }

    pub fn sinh(&self) -> Tensor {
        self.unary_autograd(f64::sinh, |x, _| x.cosh())
    }

    pub fn erf(&self) -> Tensor {
        self.unary_autograd(erf_approx, |x, _| {
            2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp()
        })
    }

    pub fn erfc(&self) -> Tensor {
        self.unary_autograd(
            |x| 1.0 - erf_approx(x),
            |x, _| -2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp(),
        )
    }

    pub fn expm1(&self) -> Tensor {
        self.unary_autograd(f64::exp_m1, |x, _| x.exp())
    }

    pub fn log1p(&self) -> Tensor {
        self.unary_autograd(f64::ln_1p, |x, _| 1.0 / safe_nonzero(1.0 + x))
    }

    pub fn reciprocal(&self) -> Tensor {
        self.unary_autograd(
            |x| 1.0 / x,
            |x, _| {
                let x = safe_nonzero(x);
                -1.0 / (x * x)
            },
        )
    }

    pub fn inv(&self) -> Tensor {
        self.reciprocal()
    }

    pub fn rsqrt(&self) -> Tensor {
        self.unary_autograd(|x| 1.0 / x.sqrt(), |x, _| -0.5 / (x * x.sqrt()))
    }

    pub fn square(&self) -> Tensor {
        self.unary_autograd(|x| x * x, |x, _| 2.0 * x)
    }

    pub fn ceil(&self) -> Tensor {
        self.unary_autograd(f64::ceil, |_, _| 0.0)
    }

    pub fn floor(&self) -> Tensor {
        self.unary_autograd(f64::floor, |_, _| 0.0)
    }

    pub fn round(&self) -> Tensor {
        self.unary_autograd(f64::round, |_, _| 0.0)
    }

    pub fn rint(&self) -> Tensor {
        self.unary_autograd(f64::round_ties_even, |_, _| 0.0)
    }

    pub fn sign(&self) -> Tensor {
        self.unary_autograd(
            |x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
            |_, _| 0.0,
        )
    }

    pub fn softplus(&self) -> Tensor {
        self.unary_autograd(|x| x.max(0.0) + (-x.abs()).exp().ln_1p(), |x, _| sigmoid(x))
    }

    pub fn softsign(&self) -> Tensor {
        self.unary_autograd(
            |x| x / (1.0 + x.abs()),
            |x, _| 1.0 / (1.0 + x.abs()).powi(2),
        )
    }

    pub fn elu(&self, alpha: f64) -> Tensor {
        self.unary_autograd(
            move |x| if x > 0.0 { x } else { alpha * x.exp_m1() },
            move |x, _| if x > 0.0 { 1.0 } else { alpha * x.exp() },
        )
    }

    pub fn selu(&self) -> Tensor {
        const ALPHA: f64 = 1.673_263_242_354_377_2;
        const SCALE: f64 = 1.050_700_987_355_480_5;
        self.unary_autograd(
            |x| {
                if x > 0.0 {
                    SCALE * x
                } else {
                    SCALE * ALPHA * x.exp_m1()
                }
            },
            |x, _| {
                if x > 0.0 {
                    SCALE
                } else {
                    SCALE * ALPHA * x.exp()
                }
            },
        )
    }

    pub fn relu6(&self) -> Tensor {
        self.unary_autograd(
            |x| x.clamp(0.0, 6.0),
            |x, _| if x > 0.0 && x < 6.0 { 1.0 } else { 0.0 },
        )
    }

    pub fn prelu(&self, weight: &Tensor) -> Tensor {
        assert!(
            weight.numel() == 1 || weight.numel() == self.numel(),
            "prelu weight must be scalar or elementwise"
        );
        let input = self.data_as_f64_vec();
        let weight_data = weight.data_as_f64_vec();
        let weight_is_scalar = weight_data.len() == 1;
        let len = input.len();
        let output: Vec<f64> = input
            .iter()
            .enumerate()
            .map(|(idx, &x)| {
                if x > 0.0 {
                    x
                } else {
                    x * weight_data[if weight_is_scalar { 0 } else { idx }]
                }
            })
            .collect();
        let input_cache = Arc::new(input);
        let weight_cache = Arc::new(weight_data);
        let dtype = Tensor::binary_dtype(self.dtype, weight.dtype);
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), weight.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_delta = vec![0.0; len];
                    let mut weight_delta = vec![0.0; weight_cache.len()];
                    for i in 0..len {
                        let x = input_cache[i];
                        if x > 0.0 {
                            input_delta[i] += grad_out[i];
                        } else {
                            let w_idx = if weight_is_scalar { 0 } else { i };
                            input_delta[i] += grad_out[i] * weight_cache[w_idx];
                            weight_delta[w_idx] += grad_out[i] * x;
                        }
                    }
                    {
                        let mut input_grad = parents[0].grad_write_compat();
                        for i in 0..len {
                            input_grad[i] += input_delta[i];
                        }
                    }
                    {
                        let mut weight_grad = parents[1].grad_write_compat();
                        for i in 0..weight_delta.len() {
                            weight_grad[i] += weight_delta[i];
                        }
                    }
                }),
            })),
        }
    }

    pub fn maximum(&self, rhs: &Tensor) -> Tensor {
        self.binary_autograd(
            rhs,
            "maximum",
            f64::max,
            |lhs, rhs, _| {
                if lhs > rhs {
                    1.0
                } else if lhs == rhs {
                    0.5
                } else {
                    0.0
                }
            },
            |lhs, rhs, _| {
                if rhs > lhs {
                    1.0
                } else if lhs == rhs {
                    0.5
                } else {
                    0.0
                }
            },
        )
    }

    pub fn modulo(&self, rhs: &Tensor) -> Tensor {
        self.binary_autograd(
            rhs,
            "modulo",
            |lhs, rhs| lhs.rem_euclid(rhs),
            |_, _, _| 1.0,
            |lhs, rhs, _| -(lhs / safe_nonzero(rhs)).floor(),
        )
    }

    pub fn equal(&self, rhs: &Tensor) -> Tensor {
        self.assert_same_numel(rhs, "equal");
        let lhs = self.data_as_f64_vec();
        let rhs = rhs.data_as_f64_vec();
        let output = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&lhs, &rhs)| if lhs == rhs { 1.0 } else { 0.0 })
            .collect();
        Tensor::with_dtype(output, self.shape.clone(), self.dtype)
    }

    pub fn ones_like(&self) -> Tensor {
        Tensor::with_dtype(vec![1.0; self.numel()], self.shape.clone(), self.dtype)
    }

    pub fn reduce_sum_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        self.reduce_dim_impl(dim, keepdim, ReductionKind::Sum)
    }

    pub fn reduce_mean_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        self.reduce_dim_impl(dim, keepdim, ReductionKind::Mean)
    }

    pub fn reduce_max_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        self.reduce_dim_impl(dim, keepdim, ReductionKind::Max)
    }

    pub fn max(&self) -> Tensor {
        let flattened = self.reshape(vec![self.numel()]);
        flattened.reduce_max_dim(0, false)
    }

    pub fn reduce_all_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        self.boolean_reduce_dim(dim, keepdim, true)
    }

    pub fn reduce_any_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        self.boolean_reduce_dim(dim, keepdim, false)
    }

    pub fn all(&self) -> Tensor {
        let flattened = self.reshape(vec![self.numel()]);
        flattened.reduce_all_dim(0, false)
    }

    pub fn any(&self) -> Tensor {
        let flattened = self.reshape(vec![self.numel()]);
        flattened.reduce_any_dim(0, false)
    }

    pub fn l2_normalize(&self, dim: usize, eps: f64) -> Tensor {
        assert!(dim < self.shape.len(), "l2_normalize dim out of bounds");
        assert!(
            eps > 0.0 && eps.is_finite(),
            "l2_normalize eps must be finite and positive"
        );
        let input = self.data_as_f64_vec();
        let dim_size = self.shape[dim];
        assert!(
            dim_size > 0,
            "l2_normalize cannot normalize an empty dimension"
        );
        let inner = checked_product(&self.shape[dim + 1..], "l2_normalize");
        let outer = checked_product(&self.shape[..dim], "l2_normalize");
        let norm_count = checked_mul(outer, inner, "l2_normalize");
        let mut output = vec![0.0; input.len()];
        let mut inverse_norms = vec![0.0; norm_count];
        let mut norm_is_data_dependent = vec![false; norm_count];

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let norm_idx = outer_idx * inner + inner_idx;
                let mut sum_sq = 0.0;
                for dim_idx in 0..dim_size {
                    let idx = (outer_idx * dim_size + dim_idx) * inner + inner_idx;
                    sum_sq += input[idx] * input[idx];
                }
                let inv_norm = 1.0 / sum_sq.max(eps).sqrt();
                inverse_norms[norm_idx] = inv_norm;
                norm_is_data_dependent[norm_idx] = sum_sq > eps;
                for dim_idx in 0..dim_size {
                    let idx = (outer_idx * dim_size + dim_idx) * inner + inner_idx;
                    output[idx] = input[idx] * inv_norm;
                }
            }
        }

        let output_cache = Arc::new(output.clone());
        let inverse_norms = Arc::new(inverse_norms);
        let norm_is_data_dependent = Arc::new(norm_is_data_dependent);
        let dtype = self.dtype;
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(input.len(), Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_grad = parents[0].grad_write_compat();
                    for outer_idx in 0..outer {
                        for inner_idx in 0..inner {
                            let norm_idx = outer_idx * inner + inner_idx;
                            let mut dot = 0.0;
                            for dim_idx in 0..dim_size {
                                let idx = (outer_idx * dim_size + dim_idx) * inner + inner_idx;
                                dot += grad_out[idx] * output_cache[idx];
                            }
                            let inv_norm = inverse_norms[norm_idx];
                            for dim_idx in 0..dim_size {
                                let idx = (outer_idx * dim_size + dim_idx) * inner + inner_idx;
                                input_grad[idx] += if norm_is_data_dependent[norm_idx] {
                                    inv_norm * (grad_out[idx] - output_cache[idx] * dot)
                                } else {
                                    inv_norm * grad_out[idx]
                                };
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn group_norm(&self, groups: usize, eps: f64) -> Tensor {
        assert!(self.shape.len() >= 2, "group_norm requires rank >= 2");
        assert!(groups > 0, "group_norm groups must be positive");
        assert!(
            eps > 0.0 && eps.is_finite(),
            "group_norm eps must be finite and positive"
        );
        let batch = self.shape[0];
        let channels = self.shape[1];
        assert!(batch > 0, "group_norm batch must be non-empty");
        assert!(channels > 0, "group_norm channels must be non-empty");
        assert!(
            channels.is_multiple_of(groups),
            "group_norm channels must be divisible by groups"
        );
        let spatial = checked_product(&self.shape[2..], "group_norm");
        let group_channels = channels / groups;
        let segment_len = checked_mul(group_channels, spatial, "group_norm");
        assert!(segment_len > 0, "group_norm segment must be non-empty");
        let segments = checked_mul(batch, groups, "group_norm");
        let input = self.data_as_f64_vec();
        let mut output = vec![0.0; input.len()];
        let mut means = vec![0.0; segments];
        let mut inv_stds = vec![0.0; segments];

        for b in 0..batch {
            for g in 0..groups {
                let segment = b * groups + g;
                let channel_start = g * group_channels;
                let mut sum = 0.0;
                for gc in 0..group_channels {
                    let c = channel_start + gc;
                    let base = (b * channels + c) * spatial;
                    for s in 0..spatial {
                        sum += input[base + s];
                    }
                }
                let mean = sum / segment_len as f64;
                means[segment] = mean;
                let mut sum_sq = 0.0;
                for gc in 0..group_channels {
                    let c = channel_start + gc;
                    let base = (b * channels + c) * spatial;
                    for s in 0..spatial {
                        let diff = input[base + s] - mean;
                        sum_sq += diff * diff;
                    }
                }
                let inv_std = 1.0 / (sum_sq / segment_len as f64 + eps).sqrt();
                inv_stds[segment] = inv_std;
                for gc in 0..group_channels {
                    let c = channel_start + gc;
                    let base = (b * channels + c) * spatial;
                    for s in 0..spatial {
                        output[base + s] = (input[base + s] - mean) * inv_std;
                    }
                }
            }
        }

        let input_cache = Arc::new(input);
        let output_cache = Arc::new(output.clone());
        let means = Arc::new(means);
        let inv_stds = Arc::new(inv_stds);
        let dtype = self.dtype;
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(input_cache.len(), Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_grad = parents[0].grad_write_compat();
                    let segment_len_f = segment_len as f64;
                    for b in 0..batch {
                        for g in 0..groups {
                            let segment = b * groups + g;
                            let channel_start = g * group_channels;
                            let inv_std = inv_stds[segment];
                            let mut grad_sum = 0.0;
                            let mut grad_norm_sum = 0.0;
                            for gc in 0..group_channels {
                                let c = channel_start + gc;
                                let base = (b * channels + c) * spatial;
                                for s in 0..spatial {
                                    grad_sum += grad_out[base + s];
                                    grad_norm_sum += grad_out[base + s] * output_cache[base + s];
                                }
                            }
                            for gc in 0..group_channels {
                                let c = channel_start + gc;
                                let base = (b * channels + c) * spatial;
                                for s in 0..spatial {
                                    let idx = base + s;
                                    let normalized = (input_cache[idx] - means[segment]) * inv_std;
                                    input_grad[idx] += inv_std
                                        * (grad_out[idx]
                                            - grad_sum / segment_len_f
                                            - normalized * grad_norm_sum / segment_len_f);
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }

    pub fn instance_norm(&self, eps: f64) -> Tensor {
        assert!(self.shape.len() >= 2, "instance_norm requires rank >= 2");
        self.group_norm(self.shape[1], eps)
    }

    pub fn batch_norm2d(&self, scale: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
        assert!(self.shape.len() >= 2, "batch_norm2d requires rank >= 2");
        assert!(
            eps > 0.0 && eps.is_finite(),
            "batch_norm2d eps must be finite and positive"
        );
        let batch = self.shape[0];
        let channels = self.shape[1];
        assert!(channels > 0, "batch_norm2d channels must be non-empty");
        assert_eq!(
            scale.numel(),
            channels,
            "batch_norm2d scale length mismatch"
        );
        assert_eq!(bias.numel(), channels, "batch_norm2d bias length mismatch");
        let spatial = checked_product(&self.shape[2..], "batch_norm2d");
        let samples = checked_mul(batch, spatial, "batch_norm2d");
        assert!(samples > 0, "batch_norm2d requires non-empty batch/spatial");
        let input = self.data_as_f64_vec();
        let scale_data = scale.data_as_f64_vec();
        let bias_data = bias.data_as_f64_vec();
        let mut means = vec![0.0; channels];
        let mut inv_stds = vec![0.0; channels];
        let mut normalized = vec![0.0; input.len()];
        let mut output = vec![0.0; input.len()];

        for c in 0..channels {
            let mut sum = 0.0;
            for b in 0..batch {
                let base = (b * channels + c) * spatial;
                for s in 0..spatial {
                    sum += input[base + s];
                }
            }
            let mean = sum / samples as f64;
            means[c] = mean;
            let mut sum_sq = 0.0;
            for b in 0..batch {
                let base = (b * channels + c) * spatial;
                for s in 0..spatial {
                    let diff = input[base + s] - mean;
                    sum_sq += diff * diff;
                }
            }
            let inv_std = 1.0 / (sum_sq / samples as f64 + eps).sqrt();
            inv_stds[c] = inv_std;
            for b in 0..batch {
                let base = (b * channels + c) * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    normalized[idx] = (input[idx] - mean) * inv_std;
                    output[idx] = normalized[idx] * scale_data[c] + bias_data[c];
                }
            }
        }

        let normalized = Arc::new(normalized);
        let inv_stds = Arc::new(inv_stds);
        let scale_data = Arc::new(scale_data);
        let dtype = Tensor::binary_dtype(Tensor::binary_dtype(self.dtype, scale.dtype), bias.dtype);
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(input.len(), Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), scale.clone(), bias.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_delta = vec![0.0; grad_out.len()];
                    let mut scale_delta = vec![0.0; channels];
                    let mut bias_delta = vec![0.0; channels];
                    let samples_f = samples as f64;
                    for c in 0..channels {
                        let mut sum_grad = 0.0;
                        let mut sum_grad_norm = 0.0;
                        for b in 0..batch {
                            let base = (b * channels + c) * spatial;
                            for s in 0..spatial {
                                let idx = base + s;
                                sum_grad += grad_out[idx];
                                sum_grad_norm += grad_out[idx] * normalized[idx];
                            }
                        }
                        scale_delta[c] += sum_grad_norm;
                        bias_delta[c] += sum_grad;
                        for b in 0..batch {
                            let base = (b * channels + c) * spatial;
                            for s in 0..spatial {
                                let idx = base + s;
                                input_delta[idx] += scale_data[c]
                                    * inv_stds[c]
                                    * (grad_out[idx]
                                        - sum_grad / samples_f
                                        - normalized[idx] * sum_grad_norm / samples_f);
                            }
                        }
                    }
                    {
                        let mut input_grad = parents[0].grad_write_compat();
                        for i in 0..input_delta.len() {
                            input_grad[i] += input_delta[i];
                        }
                    }
                    {
                        let mut scale_grad = parents[1].grad_write_compat();
                        for i in 0..channels {
                            scale_grad[i] += scale_delta[i];
                        }
                    }
                    {
                        let mut bias_grad = parents[2].grad_write_compat();
                        for i in 0..channels {
                            bias_grad[i] += bias_delta[i];
                        }
                    }
                }),
            })),
        }
    }

    pub fn bias_add(&self, bias: &Tensor) -> Tensor {
        let channels = *self.shape.last().expect("bias_add requires rank >= 1");
        assert_eq!(
            bias.numel(),
            channels,
            "bias_add bias length must match the last dimension"
        );
        let input = self.data_as_f64_vec();
        let bias_data = bias.data_as_f64_vec();
        let mut output = input.clone();
        for (idx, value) in output.iter_mut().enumerate() {
            *value += bias_data[idx % channels];
        }
        let len = output.len();
        let dtype = Tensor::binary_dtype(self.dtype, bias.dtype);
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(len, Tensor::grad_dtype_for(dtype)),
            shape: self.shape.clone(),
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), bias.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    if parents[0].grad.ptr_eq(&parents[1].grad) {
                        let mut grad_storage = parents[0].grad_write_compat();
                        for (idx, &grad) in grad_out.iter().enumerate() {
                            grad_storage[idx] += grad;
                            grad_storage[idx % channels] += grad;
                        }
                    } else {
                        let mut input_grad = parents[0].grad_write_compat();
                        let mut bias_grad = parents[1].grad_write_compat();
                        for (idx, &grad) in grad_out.iter().enumerate() {
                            input_grad[idx] += grad;
                            bias_grad[idx % channels] += grad;
                        }
                    }
                }),
            })),
        }
    }
}
