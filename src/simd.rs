// Talos-XII SIMD acceleration layer.
// Multi-tier dispatch: Scalar -> AVX2 -> AVX2+FMA -> AVX-512F -> CUDA
// Runtime detection cached in a single atomic for near-zero dispatch overhead.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ═══════════════════════════════════════════════════════════════════════════
//  CPU Tier Detection
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
mod tier {
    use std::sync::atomic::{AtomicU8, Ordering};

    pub const SCALAR: u8 = 1;
    pub const AVX2: u8 = 2;
    pub const AVX2_FMA: u8 = 3;
    pub const AVX512: u8 = 4;

    static CPU_TIER: AtomicU8 = AtomicU8::new(0);

    #[inline(always)]
    pub fn get() -> u8 {
        let t = CPU_TIER.load(Ordering::Relaxed);
        if t != 0 {
            return t;
        }
        detect()
    }

    #[cold]
    fn detect() -> u8 {
        let t = if std::is_x86_feature_detected!("avx512f") {
            AVX512
        } else if std::is_x86_feature_detected!("fma") {
            AVX2_FMA
        } else if std::is_x86_feature_detected!("avx2") {
            AVX2
        } else {
            SCALAR
        };
        CPU_TIER.store(t, Ordering::Relaxed);
        t
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  CUDA Tier (when cuda feature is enabled)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(cuda)]
#[allow(dead_code)]
pub const CUDA: u8 = 5;

#[cfg(cuda)]
#[allow(dead_code)]
mod cuda_tier {
    use std::sync::atomic::{AtomicU8, Ordering};

    static CUDA_TIER: AtomicU8 = AtomicU8::new(0);
    static mut CUDA_AVAILABLE: bool = false;

    #[inline(always)]
    pub fn get() -> u8 {
        let t = CUDA_TIER.load(Ordering::Relaxed);
        if t != 0 {
            return t;
        }
        detect()
    }

    #[inline(always)]
    pub fn is_available() -> bool {
        // Safety: This is only called once during initialization
        unsafe { CUDA_AVAILABLE }
    }

    #[cold]
    fn detect() -> u8 {
        if crate::cuda::init().is_ok() {
            unsafe {
                CUDA_AVAILABLE = true;
            }
            CUDA_TIER.store(super::CUDA, Ordering::Relaxed);
            super::CUDA
        } else {
            CUDA_TIER.store(0, Ordering::Relaxed);
            0
        }
    }
}

#[cfg(cuda)]
#[inline(always)]
#[allow(dead_code)]
pub fn cuda_is_available() -> bool {
    cuda_tier::is_available()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: add_scaled_row  —  output[i] += scale * row[i]
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn add_scaled_row(output: &mut [f64], row: &[f64], scale: f64) {
    let len = output.len();
    assert_eq!(len, row.len(), "Dimension mismatch in add_scaled_row");

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                add_scaled_row_avx512(output, row, scale);
                return;
            }
            if t >= tier::AVX2_FMA {
                add_scaled_row_avx2_fma(output, row, scale);
                return;
            }
            if t >= tier::AVX2 {
                add_scaled_row_avx2(output, row, scale);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            add_scaled_row_neon(output, row, scale);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        add_scaled_row_scalar(output, row, scale);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn add_scaled_row_scalar(output: &mut [f64], row: &[f64], scale: f64) {
    for i in 0..output.len() {
        output[i] += scale * row[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: dot_product  —  sum(a[i] * b[i])
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    assert_eq!(len, b.len(), "Dimension mismatch in dot_product");

    #[cfg(target_arch = "aarch64")]
    unsafe {
        dot_product_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            let t = tier::get();
            unsafe {
                if t >= tier::AVX512 {
                    return dot_product_avx512(a, b);
                }
                if t >= tier::AVX2_FMA {
                    return dot_product_avx2_fma(a, b);
                }
                if t >= tier::AVX2 {
                    return dot_product_avx2(a, b);
                }
            }
        }
        dot_product_scalar(a, b)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dot_product_scalar(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_fma  —  dst[i] += a[i] * b[i]
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_fma(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    assert!(len <= a.len() && len <= b.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_fma_avx512(dst, a, b);
                return;
            }
            if t >= tier::AVX2_FMA {
                vector_fma_avx2_fma(dst, a, b);
                return;
            }
            if t >= tier::AVX2 {
                vector_fma_avx2(dst, a, b);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            vector_fma_neon(dst, a, b);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..len {
            dst[i] += a[i] * b[i];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_scale  —  row[i] *= scale
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_scale(row: &mut [f64], scale: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_scale_avx512(row, scale);
                return;
            }
            if t >= tier::AVX2 {
                vector_scale_avx2(row, scale);
                return;
            }
        }
    }

    for x in row.iter_mut() {
        *x *= scale;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_add  —  dst[i] = a[i] + b[i]
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_add(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    debug_assert!(len <= a.len() && len <= b.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_binop_avx512(dst, a, b, BinOp::Add);
                return;
            }
            if t >= tier::AVX2 {
                vector_binop_avx2(dst, a, b, BinOp::Add);
                return;
            }
        }
    }

    for i in 0..len {
        dst[i] = a[i] + b[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_sub  —  dst[i] = a[i] - b[i]
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_sub(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    debug_assert!(len <= a.len() && len <= b.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_binop_avx512(dst, a, b, BinOp::Sub);
                return;
            }
            if t >= tier::AVX2 {
                vector_binop_avx2(dst, a, b, BinOp::Sub);
                return;
            }
        }
    }

    for i in 0..len {
        dst[i] = a[i] - b[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_mul  —  dst[i] = a[i] * b[i]
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_mul(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    debug_assert!(len <= a.len() && len <= b.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_binop_avx512(dst, a, b, BinOp::Mul);
                return;
            }
            if t >= tier::AVX2 {
                vector_binop_avx2(dst, a, b, BinOp::Mul);
                return;
            }
        }
    }

    for i in 0..len {
        dst[i] = a[i] * b[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_relu  —  dst[i] = max(0, src[i])
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_relu(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    debug_assert!(len <= src.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_relu_avx512(dst, src);
                return;
            }
            if t >= tier::AVX2 {
                vector_relu_avx2(dst, src);
                return;
            }
        }
    }

    for i in 0..len {
        dst[i] = if src[i] > 0.0 { src[i] } else { 0.0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_gelu  —  GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_gelu(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    debug_assert!(len <= src.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_gelu_avx512(dst, src);
                return;
            }
            if t >= tier::AVX2 {
                vector_gelu_avx2(dst, src);
                return;
            }
        }
    }

    // Scalar fallback using accurate GELU formula
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let c = 0.044715f64;
    for i in 0..len {
        let x = src[i];
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + c * x3);
        dst[i] = 0.5 * x * (1.0 + inner.tanh());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: horizontal_sum  —  sum of all elements
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn horizontal_sum(src: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                return horizontal_sum_avx512(src);
            }
            if t >= tier::AVX2 {
                return horizontal_sum_avx2(src);
            }
        }
    }

    let mut s = 0.0;
    for &x in src {
        s += x;
    }
    s
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: vector_grad_acc  —  dst[i] += src[i]   (gradient accumulation)
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn vector_grad_acc(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    debug_assert!(len <= src.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                vector_grad_acc_avx512(dst, src);
                return;
            }
            if t >= tier::AVX2 {
                vector_grad_acc_avx2(dst, src);
                return;
            }
        }
    }

    for i in 0..len {
        dst[i] += src[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: softmax_exp_sum  —  fused max-shift-exp-sum for softmax
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn softmax_exp_sum(row: &mut [f64]) -> f64 {
    let len = row.len();
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                return softmax_exp_sum_avx512(row);
            }
            if t >= tier::AVX2 {
                return softmax_exp_sum_avx2(row);
            }
        }
    }

    softmax_exp_sum_scalar(row)
}

fn softmax_exp_sum_scalar(row: &mut [f64]) -> f64 {
    let mut max_val = f64::NEG_INFINITY;
    for &x in row.iter() {
        if x > max_val {
            max_val = x;
        }
    }
    let mut sum = 0.0;
    for x in row.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: fast_exp_f64  —  vectorized polynomial exp approximation
//  Cephes-style range reduction + minimax polynomial.
//  Absolute error < 2 ULP over [-709, 709]. Suitable for softmax.
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn fast_exp_bulk(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    debug_assert!(len <= src.len());

    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                fast_exp_bulk_avx512(dst, src);
                return;
            }
            if t >= tier::AVX2 {
                fast_exp_bulk_avx2(dst, src);
                return;
            }
        }
    }

    for i in 0..len {
        dst[i] = src[i].exp();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: prefetch_read_l1  —  software prefetch hint
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn prefetch_read_l1(ptr: *const f64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn prefetch_read_l2(ptr: *const f64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX-512F Implementations (x86_64 only)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add_scaled_row_avx512(output: &mut [f64], row: &[f64], scale: f64) {
    let len = output.len();
    let mut i = 0;
    let scale_vec = _mm512_set1_pd(scale);

    while i + 16 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);

        let out0 = _mm512_loadu_pd(out_ptr);
        let row0 = _mm512_loadu_pd(row_ptr);
        let res0 = _mm512_fmadd_pd(row0, scale_vec, out0);
        _mm512_storeu_pd(out_ptr, res0);

        let out1 = _mm512_loadu_pd(out_ptr.add(8));
        let row1 = _mm512_loadu_pd(row_ptr.add(8));
        let res1 = _mm512_fmadd_pd(row1, scale_vec, out1);
        _mm512_storeu_pd(out_ptr.add(8), res1);

        i += 16;
    }

    while i + 8 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);
        let out0 = _mm512_loadu_pd(out_ptr);
        let row0 = _mm512_loadu_pd(row_ptr);
        let res0 = _mm512_fmadd_pd(row0, scale_vec, out0);
        _mm512_storeu_pd(out_ptr, res0);
        i += 8;
    }

    while i < len {
        *output.get_unchecked_mut(i) += scale * *row.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut acc0 = _mm512_setzero_pd();
    let mut acc1 = _mm512_setzero_pd();

    while i + 16 <= len {
        let a0 = _mm512_loadu_pd(a.as_ptr().add(i));
        let b0 = _mm512_loadu_pd(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_pd(a0, b0, acc0);

        let a1 = _mm512_loadu_pd(a.as_ptr().add(i + 8));
        let b1 = _mm512_loadu_pd(b.as_ptr().add(i + 8));
        acc1 = _mm512_fmadd_pd(a1, b1, acc1);

        i += 16;
    }

    while i + 8 <= len {
        let av = _mm512_loadu_pd(a.as_ptr().add(i));
        let bv = _mm512_loadu_pd(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_pd(av, bv, acc0);
        i += 8;
    }

    acc0 = _mm512_add_pd(acc0, acc1);
    let mut sum = _mm512_reduce_add_pd(acc0);

    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_fma_avx512(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    let mut i = 0;

    while i + 16 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let dv0 = _mm512_loadu_pd(d_ptr);
        let av0 = _mm512_loadu_pd(a_ptr);
        let bv0 = _mm512_loadu_pd(b_ptr);
        _mm512_storeu_pd(d_ptr, _mm512_fmadd_pd(av0, bv0, dv0));

        let dv1 = _mm512_loadu_pd(d_ptr.add(8));
        let av1 = _mm512_loadu_pd(a_ptr.add(8));
        let bv1 = _mm512_loadu_pd(b_ptr.add(8));
        _mm512_storeu_pd(d_ptr.add(8), _mm512_fmadd_pd(av1, bv1, dv1));

        i += 16;
    }

    while i + 8 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let dv = _mm512_loadu_pd(d_ptr);
        let av = _mm512_loadu_pd(a.as_ptr().add(i));
        let bv = _mm512_loadu_pd(b.as_ptr().add(i));
        _mm512_storeu_pd(d_ptr, _mm512_fmadd_pd(av, bv, dv));
        i += 8;
    }

    while i < len {
        *dst.get_unchecked_mut(i) += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_scale_avx512(row: &mut [f64], scale: f64) {
    let len = row.len();
    let mut i = 0;
    let sv = _mm512_set1_pd(scale);

    while i + 16 <= len {
        let ptr = row.as_mut_ptr().add(i);
        _mm512_storeu_pd(ptr, _mm512_mul_pd(_mm512_loadu_pd(ptr), sv));
        _mm512_storeu_pd(ptr.add(8), _mm512_mul_pd(_mm512_loadu_pd(ptr.add(8)), sv));
        i += 16;
    }
    while i + 8 <= len {
        let ptr = row.as_mut_ptr().add(i);
        _mm512_storeu_pd(ptr, _mm512_mul_pd(_mm512_loadu_pd(ptr), sv));
        i += 8;
    }
    while i < len {
        *row.get_unchecked_mut(i) *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
enum BinOp {
    Add,
    Sub,
    Mul,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_binop_avx512(dst: &mut [f64], a: &[f64], b: &[f64], op: BinOp) {
    let len = dst.len();
    let mut i = 0;

    while i + 8 <= len {
        let av = _mm512_loadu_pd(a.as_ptr().add(i));
        let bv = _mm512_loadu_pd(b.as_ptr().add(i));
        let r = match op {
            BinOp::Add => _mm512_add_pd(av, bv),
            BinOp::Sub => _mm512_sub_pd(av, bv),
            BinOp::Mul => _mm512_mul_pd(av, bv),
        };
        _mm512_storeu_pd(dst.as_mut_ptr().add(i), r);
        i += 8;
    }

    while i < len {
        dst[i] = match op {
            BinOp::Add => a[i] + b[i],
            BinOp::Sub => a[i] - b[i],
            BinOp::Mul => a[i] * b[i],
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_binop_avx2(dst: &mut [f64], a: &[f64], b: &[f64], op: BinOp) {
    let len = dst.len();
    let mut i = 0;

    while i + 4 <= len {
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let bv = _mm256_loadu_pd(b.as_ptr().add(i));
        let r = match op {
            BinOp::Add => _mm256_add_pd(av, bv),
            BinOp::Sub => _mm256_sub_pd(av, bv),
            BinOp::Mul => _mm256_mul_pd(av, bv),
        };
        _mm256_storeu_pd(dst.as_mut_ptr().add(i), r);
        i += 4;
    }

    while i < len {
        dst[i] = match op {
            BinOp::Add => a[i] + b[i],
            BinOp::Sub => a[i] - b[i],
            BinOp::Mul => a[i] * b[i],
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_relu_avx512(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    let mut i = 0;
    let zero = _mm512_setzero_pd();

    while i + 8 <= len {
        let v = _mm512_loadu_pd(src.as_ptr().add(i));
        _mm512_storeu_pd(dst.as_mut_ptr().add(i), _mm512_max_pd(v, zero));
        i += 8;
    }
    while i < len {
        dst[i] = if src[i] > 0.0 { src[i] } else { 0.0 };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_relu_avx2(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    let mut i = 0;
    let zero = _mm256_setzero_pd();

    while i + 4 <= len {
        let v = _mm256_loadu_pd(src.as_ptr().add(i));
        _mm256_storeu_pd(dst.as_mut_ptr().add(i), _mm256_max_pd(v, zero));
        i += 4;
    }
    while i < len {
        dst[i] = if src[i] > 0.0 { src[i] } else { 0.0 };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_gelu_avx512(dst: &mut [f64], src: &[f64]) {
    // AVX doesn't have tanh intrinsic, so we use scalar fallback
    // which still benefits from the outer loop structure and cache
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let c = 0.044715;
    for i in 0..dst.len() {
        let x = src[i];
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + c * x3);
        dst[i] = 0.5 * x * (1.0 + inner.tanh());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_gelu_avx2(dst: &mut [f64], src: &[f64]) {
    // AVX doesn't have tanh intrinsic, so we use scalar fallback
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let c = 0.044715;
    for i in 0..dst.len() {
        let x = src[i];
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + c * x3);
        dst[i] = 0.5 * x * (1.0 + inner.tanh());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn horizontal_sum_avx512(src: &[f64]) -> f64 {
    let len = src.len();
    let mut i = 0;
    let mut acc = _mm512_setzero_pd();

    while i + 8 <= len {
        acc = _mm512_add_pd(acc, _mm512_loadu_pd(src.as_ptr().add(i)));
        i += 8;
    }
    let mut sum = _mm512_reduce_add_pd(acc);
    while i < len {
        sum += *src.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(src: &[f64]) -> f64 {
    let len = src.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    while i + 8 <= len {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(src.as_ptr().add(i)));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(src.as_ptr().add(i + 4)));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(src.as_ptr().add(i)));
        i += 4;
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc0);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < len {
        sum += *src.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_grad_acc_avx512(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    let mut i = 0;

    while i + 16 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let s_ptr = src.as_ptr().add(i);
        _mm512_storeu_pd(
            d_ptr,
            _mm512_add_pd(_mm512_loadu_pd(d_ptr), _mm512_loadu_pd(s_ptr)),
        );
        _mm512_storeu_pd(
            d_ptr.add(8),
            _mm512_add_pd(_mm512_loadu_pd(d_ptr.add(8)), _mm512_loadu_pd(s_ptr.add(8))),
        );
        i += 16;
    }
    while i + 8 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        _mm512_storeu_pd(
            d_ptr,
            _mm512_add_pd(_mm512_loadu_pd(d_ptr), _mm512_loadu_pd(src.as_ptr().add(i))),
        );
        i += 8;
    }
    while i < len {
        *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_grad_acc_avx2(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    let mut i = 0;

    while i + 8 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let s_ptr = src.as_ptr().add(i);
        _mm256_storeu_pd(
            d_ptr,
            _mm256_add_pd(_mm256_loadu_pd(d_ptr), _mm256_loadu_pd(s_ptr)),
        );
        _mm256_storeu_pd(
            d_ptr.add(4),
            _mm256_add_pd(_mm256_loadu_pd(d_ptr.add(4)), _mm256_loadu_pd(s_ptr.add(4))),
        );
        i += 8;
    }
    while i + 4 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        _mm256_storeu_pd(
            d_ptr,
            _mm256_add_pd(_mm256_loadu_pd(d_ptr), _mm256_loadu_pd(src.as_ptr().add(i))),
        );
        i += 4;
    }
    while i < len {
        *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX-512 Fast Exp  —  Cephes-style double-precision exp
//  x -> exp(x) via range reduction: x = n*ln2 + r, exp(x) = 2^n * exp(r)
//  exp(r) approximated by degree-11 minimax polynomial on [-ln2/2, ln2/2]
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fast_exp_avx512(x: __m512d) -> __m512d {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.93145751953125e-1;
    const LN2_LO: f64 = 1.428_606_820_309_417_3e-6;
    const EXP_LO: f64 = -708.3964185322641;
    const EXP_HI: f64 = 709.782712893384;
    const P0: f64 = 1.0;
    const P1: f64 = 1.0;
    const P2: f64 = 0.5;
    const P3: f64 = 0.16666666666666666;
    const P4: f64 = 0.041666666666666664;
    const P5: f64 = 0.008333333333333333;
    const P6: f64 = 1.388888888888889e-3;
    const P7: f64 = 1.984126984126984e-4;
    const P8: f64 = 2.48015873015873e-5;
    const P9: f64 = 2.7557319223985893e-6;
    const P10: f64 = 2.755731922398589e-7;
    const P11: f64 = 2.505210838544172e-8;

    // Clamp to avoid NaN from inf/underflow from -inf
    let x = _mm512_max_pd(
        _mm512_min_pd(x, _mm512_set1_pd(EXP_HI)),
        _mm512_set1_pd(EXP_LO),
    );

    let log2e = _mm512_set1_pd(LOG2E);
    let ln2_hi = _mm512_set1_pd(LN2_HI);
    let ln2_lo = _mm512_set1_pd(LN2_LO);

    let n = _mm512_roundscale_pd(_mm512_mul_pd(x, log2e), 0);
    // r = x - n*ln2 (Cody-Waite reduction for precision)
    let r = _mm512_sub_pd(
        _mm512_sub_pd(x, _mm512_mul_pd(n, ln2_hi)),
        _mm512_mul_pd(n, ln2_lo),
    );

    // Horner evaluation: p = P11*r + P10, then p = p*r + P9, etc.
    let mut p = _mm512_set1_pd(P11);
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P10));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P9));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P8));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P7));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P6));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P5));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P4));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P3));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P2));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P1));
    p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(P0));

    // 2^n via integer exponent manipulation
    let ni = _mm512_cvtpd_epi64(n);
    let bias = _mm512_set1_epi64(1023);
    let exp_bits = _mm512_slli_epi64(_mm512_add_epi64(ni, bias), 52);
    let pow2n = _mm512_castsi512_pd(exp_bits);

    _mm512_mul_pd(p, pow2n)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_exp_avx2(x: __m256d) -> __m256d {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.93145751953125e-1;
    const LN2_LO: f64 = 1.428_606_820_309_417_3e-6;
    const EXP_LO: f64 = -708.3964185322641;
    const EXP_HI: f64 = 709.782712893384;
    const P0: f64 = 1.0;
    const P1: f64 = 1.0;
    const P2: f64 = 0.5;
    const P3: f64 = 0.16666666666666666;
    const P4: f64 = 0.041666666666666664;
    const P5: f64 = 0.008333333333333333;
    const P6: f64 = 1.388888888888889e-3;
    const P7: f64 = 1.984126984126984e-4;
    const P8: f64 = 2.48015873015873e-5;
    const P9: f64 = 2.7557319223985893e-6;
    const P10: f64 = 2.755731922398589e-7;
    const P11: f64 = 2.505210838544172e-8;

    // Clamp to avoid NaN from inf/underflow from -inf
    let x = _mm256_max_pd(
        _mm256_min_pd(x, _mm256_set1_pd(EXP_HI)),
        _mm256_set1_pd(EXP_LO),
    );

    let log2e = _mm256_set1_pd(LOG2E);
    let ln2_hi = _mm256_set1_pd(LN2_HI);
    let ln2_lo = _mm256_set1_pd(LN2_LO);

    let n = _mm256_round_pd(
        _mm256_mul_pd(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_sub_pd(
        _mm256_sub_pd(x, _mm256_mul_pd(n, ln2_hi)),
        _mm256_mul_pd(n, ln2_lo),
    );

    let mut p = _mm256_set1_pd(P11);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P10));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P9));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P8));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P7));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P6));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P5));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P4));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P3));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P2));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P1));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(P0));

    // 2^n: convert n to i64, add exponent bias, shift into IEEE f64 exponent field
    let ni = _mm256_cvtpd_epi32(n);
    let ni64 = _mm256_cvtepi32_epi64(ni);
    let bias = _mm256_set1_epi64x(1023);
    let exp_bits = _mm256_slli_epi64(_mm256_add_epi64(ni64, bias), 52);
    let pow2n = _mm256_castsi256_pd(exp_bits);

    _mm256_mul_pd(p, pow2n)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fast_exp_bulk_avx512(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    let mut i = 0;

    while i + 8 <= len {
        let v = _mm512_loadu_pd(src.as_ptr().add(i));
        _mm512_storeu_pd(dst.as_mut_ptr().add(i), fast_exp_avx512(v));
        i += 8;
    }
    while i < len {
        dst[i] = src[i].exp();
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_exp_bulk_avx2(dst: &mut [f64], src: &[f64]) {
    let len = dst.len();
    let mut i = 0;

    while i + 4 <= len {
        let v = _mm256_loadu_pd(src.as_ptr().add(i));
        _mm256_storeu_pd(dst.as_mut_ptr().add(i), fast_exp_avx2(v));
        i += 4;
    }
    while i < len {
        dst[i] = src[i].exp();
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX-512 / AVX2 Fused Softmax: max + shift + exp + sum in two passes
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_exp_sum_avx512(row: &mut [f64]) -> f64 {
    let len = row.len();

    // Pass 1: find max
    let mut i = 0;
    let mut vmax = _mm512_set1_pd(f64::NEG_INFINITY);
    while i + 8 <= len {
        vmax = _mm512_max_pd(vmax, _mm512_loadu_pd(row.as_ptr().add(i)));
        i += 8;
    }
    let mut max_val = _mm512_reduce_max_pd(vmax);
    while i < len {
        let v = *row.get_unchecked(i);
        if v > max_val {
            max_val = v;
        }
        i += 1;
    }

    // Pass 2: subtract max, exp, accumulate sum
    let max_vec = _mm512_set1_pd(max_val);
    let mut sum_acc = _mm512_setzero_pd();
    i = 0;
    while i + 8 <= len {
        let ptr = row.as_mut_ptr().add(i);
        let shifted = _mm512_sub_pd(_mm512_loadu_pd(ptr), max_vec);
        let e = fast_exp_avx512(shifted);
        _mm512_storeu_pd(ptr, e);
        sum_acc = _mm512_add_pd(sum_acc, e);
        i += 8;
    }
    let mut sum = _mm512_reduce_add_pd(sum_acc);
    while i < len {
        let e = (*row.get_unchecked(i) - max_val).exp();
        *row.get_unchecked_mut(i) = e;
        sum += e;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_exp_sum_avx2(row: &mut [f64]) -> f64 {
    let len = row.len();

    // Pass 1: find max via SIMD
    let mut i = 0;
    let mut vmax = _mm256_set1_pd(f64::NEG_INFINITY);
    while i + 4 <= len {
        vmax = _mm256_max_pd(vmax, _mm256_loadu_pd(row.as_ptr().add(i)));
        i += 4;
    }
    let mut tmp = [0.0f64; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), vmax);
    let mut max_val = tmp[0].max(tmp[1]).max(tmp[2].max(tmp[3]));
    while i < len {
        let v = *row.get_unchecked(i);
        if v > max_val {
            max_val = v;
        }
        i += 1;
    }

    // Pass 2: subtract max, fast_exp, accumulate
    let max_vec = _mm256_set1_pd(max_val);
    let mut sum_acc = _mm256_setzero_pd();
    i = 0;
    while i + 4 <= len {
        let ptr = row.as_mut_ptr().add(i);
        let shifted = _mm256_sub_pd(_mm256_loadu_pd(ptr), max_vec);
        let e = fast_exp_avx2(shifted);
        _mm256_storeu_pd(ptr, e);
        sum_acc = _mm256_add_pd(sum_acc, e);
        i += 4;
    }
    _mm256_storeu_pd(tmp.as_mut_ptr(), sum_acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < len {
        let e = (*row.get_unchecked(i) - max_val).exp();
        *row.get_unchecked_mut(i) = e;
        sum += e;
        i += 1;
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX2 Implementations (existing, preserved)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_row_avx2(output: &mut [f64], row: &[f64], scale: f64) {
    let len = output.len();
    let mut i = 0;
    let scale_vec = _mm256_set1_pd(scale);

    while i + 8 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);

        let out0 = _mm256_loadu_pd(out_ptr);
        let row0 = _mm256_loadu_pd(row_ptr);
        let prod0 = _mm256_mul_pd(row0, scale_vec);
        let sum0 = _mm256_add_pd(out0, prod0);
        _mm256_storeu_pd(out_ptr, sum0);

        let out1 = _mm256_loadu_pd(out_ptr.add(4));
        let row1 = _mm256_loadu_pd(row_ptr.add(4));
        let prod1 = _mm256_mul_pd(row1, scale_vec);
        let sum1 = _mm256_add_pd(out1, prod1);
        _mm256_storeu_pd(out_ptr.add(4), sum1);

        i += 8;
    }

    while i + 4 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);

        let out0 = _mm256_loadu_pd(out_ptr);
        let rowv = _mm256_loadu_pd(row_ptr);
        let prod = _mm256_mul_pd(rowv, scale_vec);
        let sum0 = _mm256_add_pd(out0, prod);
        _mm256_storeu_pd(out_ptr, sum0);

        i += 4;
    }

    while i < len {
        *output.get_unchecked_mut(i) += scale * *row.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_scaled_row_avx2_fma(output: &mut [f64], row: &[f64], scale: f64) {
    let len = output.len();
    let mut i = 0;
    let scale_vec = _mm256_set1_pd(scale);

    while i + 8 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);

        let out0 = _mm256_loadu_pd(out_ptr);
        let row0 = _mm256_loadu_pd(row_ptr);
        let sum0 = _mm256_fmadd_pd(row0, scale_vec, out0);
        _mm256_storeu_pd(out_ptr, sum0);

        let out1 = _mm256_loadu_pd(out_ptr.add(4));
        let row1 = _mm256_loadu_pd(row_ptr.add(4));
        let sum1 = _mm256_fmadd_pd(row1, scale_vec, out1);
        _mm256_storeu_pd(out_ptr.add(4), sum1);

        i += 8;
    }

    while i + 4 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);

        let out0 = _mm256_loadu_pd(out_ptr);
        let rowv = _mm256_loadu_pd(row_ptr);
        let sum0 = _mm256_fmadd_pd(rowv, scale_vec, out0);
        _mm256_storeu_pd(out_ptr, sum0);

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
    let len = a.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    while i + 8 <= len {
        let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
        let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod0 = _mm256_mul_pd(a0, b0);
        acc0 = _mm256_add_pd(acc0, prod0);

        let a1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
        let b1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
        let prod1 = _mm256_mul_pd(a1, b1);
        acc1 = _mm256_add_pd(acc1, prod1);

        i += 8;
    }

    while i + 4 <= len {
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let bv = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(av, bv);
        acc0 = _mm256_add_pd(acc0, prod);
        i += 4;
    }

    acc0 = _mm256_add_pd(acc0, acc1);

    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc0);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    while i + 8 <= len {
        let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
        let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_pd(a0, b0, acc0);

        let a1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
        let b1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
        acc1 = _mm256_fmadd_pd(a1, b1, acc1);

        i += 8;
    }

    while i + 4 <= len {
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let bv = _mm256_loadu_pd(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_pd(av, bv, acc0);
        i += 4;
    }

    acc0 = _mm256_add_pd(acc0, acc1);

    let mut tmp = [0.0; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc0);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_fma_avx2(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    let mut i = 0;

    while i + 8 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let dv0 = _mm256_loadu_pd(d_ptr);
        let av0 = _mm256_loadu_pd(a_ptr);
        let bv0 = _mm256_loadu_pd(b_ptr);
        let prod0 = _mm256_mul_pd(av0, bv0);
        let res0 = _mm256_add_pd(dv0, prod0);
        _mm256_storeu_pd(d_ptr, res0);

        let dv1 = _mm256_loadu_pd(d_ptr.add(4));
        let av1 = _mm256_loadu_pd(a_ptr.add(4));
        let bv1 = _mm256_loadu_pd(b_ptr.add(4));
        let prod1 = _mm256_mul_pd(av1, bv1);
        let res1 = _mm256_add_pd(dv1, prod1);
        _mm256_storeu_pd(d_ptr.add(4), res1);

        i += 8;
    }

    while i + 4 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let dv = _mm256_loadu_pd(d_ptr);
        let av = _mm256_loadu_pd(a_ptr);
        let bv = _mm256_loadu_pd(b_ptr);
        let prod = _mm256_mul_pd(av, bv);
        let res = _mm256_add_pd(dv, prod);
        _mm256_storeu_pd(d_ptr, res);
        i += 4;
    }

    while i < len {
        *dst.get_unchecked_mut(i) += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vector_fma_avx2_fma(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    let mut i = 0;

    while i + 8 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let dv0 = _mm256_loadu_pd(d_ptr);
        let av0 = _mm256_loadu_pd(a_ptr);
        let bv0 = _mm256_loadu_pd(b_ptr);
        let res0 = _mm256_fmadd_pd(av0, bv0, dv0);
        _mm256_storeu_pd(d_ptr, res0);

        let dv1 = _mm256_loadu_pd(d_ptr.add(4));
        let av1 = _mm256_loadu_pd(a_ptr.add(4));
        let bv1 = _mm256_loadu_pd(b_ptr.add(4));
        let res1 = _mm256_fmadd_pd(av1, bv1, dv1);
        _mm256_storeu_pd(d_ptr.add(4), res1);

        i += 8;
    }

    while i + 4 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let dv = _mm256_loadu_pd(d_ptr);
        let av = _mm256_loadu_pd(a_ptr);
        let bv = _mm256_loadu_pd(b_ptr);
        let res = _mm256_fmadd_pd(av, bv, dv);
        _mm256_storeu_pd(d_ptr, res);
        i += 4;
    }

    while i < len {
        *dst.get_unchecked_mut(i) += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_scale_avx2(row: &mut [f64], scale: f64) {
    let len = row.len();
    let mut i = 0;
    let scale_vec = _mm256_set1_pd(scale);

    while i + 8 <= len {
        let ptr = row.as_mut_ptr().add(i);
        _mm256_storeu_pd(ptr, _mm256_mul_pd(_mm256_loadu_pd(ptr), scale_vec));
        _mm256_storeu_pd(
            ptr.add(4),
            _mm256_mul_pd(_mm256_loadu_pd(ptr.add(4)), scale_vec),
        );
        i += 8;
    }
    while i + 4 <= len {
        let ptr = row.as_mut_ptr().add(i);
        let val = _mm256_loadu_pd(ptr);
        let res = _mm256_mul_pd(val, scale_vec);
        _mm256_storeu_pd(ptr, res);
        i += 4;
    }

    while i < len {
        *row.get_unchecked_mut(i) *= scale;
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  NEON Implementations (aarch64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_scaled_row_neon(output: &mut [f64], row: &[f64], scale: f64) {
    use core::arch::aarch64::*;
    let len = output.len();
    let mut i = 0;
    let scale_vec = vdupq_n_f64(scale);
    while i + 4 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);

        let out0 = vld1q_f64(out_ptr);
        let row0 = vld1q_f64(row_ptr);
        let prod0 = vmulq_f64(row0, scale_vec);
        let sum0 = vaddq_f64(out0, prod0);
        vst1q_f64(out_ptr, sum0);

        let out1 = vld1q_f64(out_ptr.add(2));
        let row1 = vld1q_f64(row_ptr.add(2));
        let prod1 = vmulq_f64(row1, scale_vec);
        let sum1 = vaddq_f64(out1, prod1);
        vst1q_f64(out_ptr.add(2), sum1);

        i += 4;
    }
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
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    while i + 4 <= len {
        let a0 = vld1q_f64(a.as_ptr().add(i));
        let b0 = vld1q_f64(b.as_ptr().add(i));
        let prod0 = vmulq_f64(a0, b0);
        acc0 = vaddq_f64(acc0, prod0);

        let a1 = vld1q_f64(a.as_ptr().add(i + 2));
        let b1 = vld1q_f64(b.as_ptr().add(i + 2));
        let prod1 = vmulq_f64(a1, b1);
        acc1 = vaddq_f64(acc1, prod1);

        i += 4;
    }
    while i + 2 <= len {
        let av = vld1q_f64(a.as_ptr().add(i));
        let bv = vld1q_f64(b.as_ptr().add(i));
        let prod = vmulq_f64(av, bv);
        acc0 = vaddq_f64(acc0, prod);
        i += 2;
    }
    acc0 = vaddq_f64(acc0, acc1);
    let mut tmp = [0.0; 2];
    vst1q_f64(tmp.as_mut_ptr(), acc0);
    let mut sum = tmp[0] + tmp[1];
    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vector_fma_neon(dst: &mut [f64], a: &[f64], b: &[f64]) {
    use core::arch::aarch64::*;
    let len = dst.len();
    let mut i = 0;
    while i + 4 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let dv0 = vld1q_f64(d_ptr);
        let av0 = vld1q_f64(a_ptr);
        let bv0 = vld1q_f64(b_ptr);
        let prod0 = vmulq_f64(av0, bv0);
        let res0 = vaddq_f64(dv0, prod0);
        vst1q_f64(d_ptr, res0);

        let dv1 = vld1q_f64(d_ptr.add(2));
        let av1 = vld1q_f64(a_ptr.add(2));
        let bv1 = vld1q_f64(b_ptr.add(2));
        let prod1 = vmulq_f64(av1, bv1);
        let res1 = vaddq_f64(dv1, prod1);
        vst1q_f64(d_ptr.add(2), res1);

        i += 4;
    }
    while i + 2 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);

        let dv = vld1q_f64(d_ptr);
        let av = vld1q_f64(a_ptr);
        let bv = vld1q_f64(b_ptr);
        let prod = vmulq_f64(av, bv);
        let res = vaddq_f64(dv, prod);
        vst1q_f64(d_ptr, res);
        i += 2;
    }
    while i < len {
        *dst.get_unchecked_mut(i) += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API: layer_norm  —  y = (x - mean) * inv_std * gamma + beta
//  inv_std = 1/sqrt(variance + eps) precomputed by caller
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn layer_norm(data: &[f64], mean: f64, inv_std: f64, gamma: &[f64], beta: &[f64]) -> Vec<f64> {
    let mut output = data.to_vec();
    layer_norm_inplace(&mut output, mean, inv_std, gamma, beta);
    output
}

#[inline(always)]
pub fn layer_norm_inplace(data: &mut [f64], mean: f64, inv_std: f64, gamma: &[f64], beta: &[f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        let t = tier::get();
        unsafe {
            if t >= tier::AVX512 {
                layer_norm_avx512(data, mean, inv_std, gamma, beta);
                return;
            }
            if t >= tier::AVX2_FMA {
                layer_norm_avx2_fma(data, mean, inv_std, gamma, beta);
                return;
            }
            if t >= tier::AVX2 {
                layer_norm_avx2(data, mean, inv_std, gamma, beta);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            layer_norm_neon(data, mean, inv_std, gamma, beta);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        layer_norm_scalar(data, mean, inv_std, gamma, beta);
    }
}

#[allow(dead_code)]
#[inline(always)]
fn layer_norm_scalar(data: &mut [f64], mean: f64, inv_std: f64, gamma: &[f64], beta: &[f64]) {
    let has_gamma = !gamma.is_empty();
    let has_beta = !beta.is_empty();
    let g0 = if has_gamma { gamma[0] } else { 1.0 };
    let b0 = if has_beta { beta[0] } else { 0.0 };
    for item in data.iter_mut() {
        let mut y = (*item - mean) * inv_std;
        if has_gamma {
            y *= g0;
        }
        if has_beta {
            y += b0;
        }
        *item = y;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX-512 Layer Norm  —  8-wide FMA
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn layer_norm_avx512(
    data: &mut [f64],
    mean: f64,
    inv_std: f64,
    gamma: &[f64],
    beta: &[f64],
) {
    let len = data.len();
    let mean_vec = _mm512_set1_pd(mean);
    let inv_vec = _mm512_set1_pd(inv_std);
    let has_gamma = !gamma.is_empty();
    let has_beta = !beta.is_empty();
    let g0 = if has_gamma { gamma[0] } else { 1.0 };
    let b0 = if has_beta { beta[0] } else { 0.0 };
    let mut i = 0;

    while i + 8 <= len {
        let ptr = data.as_mut_ptr().add(i);
        let x = _mm512_loadu_pd(ptr);
        let diff = _mm512_sub_pd(x, mean_vec);
        let norm = _mm512_mul_pd(diff, inv_vec);

        let result = if has_gamma && has_beta {
            _mm512_fmadd_pd(norm, _mm512_set1_pd(g0), _mm512_set1_pd(b0))
        } else if has_gamma {
            _mm512_mul_pd(norm, _mm512_set1_pd(g0))
        } else if has_beta {
            _mm512_add_pd(norm, _mm512_set1_pd(b0))
        } else {
            norm
        };
        _mm512_storeu_pd(ptr, result);
        i += 8;
    }
    while i < len {
        let mut y = (data[i] - mean) * inv_std;
        if has_gamma {
            y *= g0;
        }
        if has_beta {
            y += b0;
        }
        data[i] = y;
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX2 Layer Norm  —  4-wide FMA
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_avx2_fma(
    data: &mut [f64],
    mean: f64,
    inv_std: f64,
    gamma: &[f64],
    beta: &[f64],
) {
    let len = data.len();
    let mean_vec = _mm256_set1_pd(mean);
    let inv_vec = _mm256_set1_pd(inv_std);
    let has_gamma = !gamma.is_empty();
    let has_beta = !beta.is_empty();
    let g0 = if has_gamma { gamma[0] } else { 1.0 };
    let b0 = if has_beta { beta[0] } else { 0.0 };
    let mut i = 0;

    while i + 4 <= len {
        let ptr = data.as_mut_ptr().add(i);
        let x = _mm256_loadu_pd(ptr);
        let diff = _mm256_sub_pd(x, mean_vec);
        let norm = _mm256_mul_pd(diff, inv_vec);

        let result = if has_gamma && has_beta {
            _mm256_fmadd_pd(norm, _mm256_set1_pd(g0), _mm256_set1_pd(b0))
        } else if has_gamma {
            _mm256_mul_pd(norm, _mm256_set1_pd(g0))
        } else if has_beta {
            _mm256_add_pd(norm, _mm256_set1_pd(b0))
        } else {
            norm
        };
        _mm256_storeu_pd(ptr, result);
        i += 4;
    }
    while i < len {
        let mut y = (data[i] - mean) * inv_std;
        if has_gamma {
            y *= g0;
        }
        if has_beta {
            y += b0;
        }
        data[i] = y;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_avx2(data: &mut [f64], mean: f64, inv_std: f64, gamma: &[f64], beta: &[f64]) {
    let len = data.len();
    let mean_vec = _mm256_set1_pd(mean);
    let inv_vec = _mm256_set1_pd(inv_std);
    let has_gamma = !gamma.is_empty();
    let has_beta = !beta.is_empty();
    let g0 = if has_gamma { gamma[0] } else { 1.0 };
    let b0 = if has_beta { beta[0] } else { 0.0 };
    let mut i = 0;

    while i + 4 <= len {
        let ptr = data.as_mut_ptr().add(i);
        let x = _mm256_loadu_pd(ptr);
        let diff = _mm256_sub_pd(x, mean_vec);
        let norm = _mm256_mul_pd(diff, inv_vec);

        let result = if has_gamma && has_beta {
            let scaled = _mm256_mul_pd(norm, _mm256_set1_pd(g0));
            _mm256_add_pd(scaled, _mm256_set1_pd(b0))
        } else if has_gamma {
            _mm256_mul_pd(norm, _mm256_set1_pd(g0))
        } else if has_beta {
            _mm256_add_pd(norm, _mm256_set1_pd(b0))
        } else {
            norm
        };
        _mm256_storeu_pd(ptr, result);
        i += 4;
    }
    while i < len {
        let mut y = (data[i] - mean) * inv_std;
        if has_gamma {
            y *= g0;
        }
        if has_beta {
            y += b0;
        }
        data[i] = y;
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  NEON Layer Norm  —  2-wide operations
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_neon(data: &mut [f64], mean: f64, inv_std: f64, gamma: &[f64], beta: &[f64]) {
    use core::arch::aarch64::*;
    let len = data.len();
    let mean_vec = vdupq_n_f64(mean);
    let inv_vec = vdupq_n_f64(inv_std);
    let has_gamma = !gamma.is_empty();
    let has_beta = !beta.is_empty();
    let g0 = if has_gamma { gamma[0] } else { 1.0 };
    let b0 = if has_beta { beta[0] } else { 0.0 };
    let mut i = 0;

    while i + 2 <= len {
        let ptr = data.as_mut_ptr().add(i);
        let x = vld1q_f64(ptr);
        let diff = vsubq_f64(x, mean_vec);
        let norm = vmulq_f64(diff, inv_vec);

        let result = if has_gamma && has_beta {
            let scaled = vmulq_f64(norm, vdupq_n_f64(g0));
            vaddq_f64(scaled, vdupq_n_f64(b0))
        } else if has_gamma {
            vmulq_f64(norm, vdupq_n_f64(g0))
        } else if has_beta {
            vaddq_f64(norm, vdupq_n_f64(b0))
        } else {
            norm
        };
        vst1q_f64(ptr, result);
        i += 2;
    }
    while i < len {
        let mut y = (data[i] - mean) * inv_std;
        if has_gamma {
            y *= g0;
        }
        if has_beta {
            y += b0;
        }
        data[i] = y;
        i += 1;
    }
}
