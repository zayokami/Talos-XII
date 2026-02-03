
#[inline(always)]
pub fn add_scaled_row(output: &mut [f64], row: &[f64], scale: f64) {
    let len = output.len();
    assert_eq!(len, row.len(), "Dimension mismatch in add_scaled_row");
    
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
    
    // Fallback
    for i in 0..len {
        output[i] += scale * row[i];
    }
}

#[inline(always)]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    assert_eq!(len, b.len(), "Dimension mismatch in dot_product");
    
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
    
    // Fallback
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

// dst[i] += a[i] * b[i]
#[inline(always)]
pub fn vector_fma(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let len = dst.len();
    assert!(len <= a.len() && len <= b.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                vector_fma_avx2(dst, a, b);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
         unsafe {
            vector_fma_neon(dst, a, b);
        }
        return;
    }
    
    for i in 0..len {
        dst[i] += a[i] * b[i];
    }
}

// --- AVX2 Implementations ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_row_avx2(output: &mut [f64], row: &[f64], scale: f64) {
    use core::arch::x86_64::*;
    let len = output.len();
    let mut i = 0;
    let scale_vec = _mm256_set1_pd(scale);
    
    // Unroll 4x doubles (32 bytes)
    while i + 4 <= len {
        let out_ptr = output.as_mut_ptr().add(i);
        let row_ptr = row.as_ptr().add(i);
        
        let out = _mm256_loadu_pd(out_ptr);
        let rowv = _mm256_loadu_pd(row_ptr);
        let prod = _mm256_mul_pd(rowv, scale_vec);
        let sum = _mm256_add_pd(out, prod);
        _mm256_storeu_pd(out_ptr, sum);
        
        i += 4;
    }
    
    // Handle remainder
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
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    
    // Unroll 2x (8 doubles) for better pipeline usage
    while i + 8 <= len {
        let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
        let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_pd(a0, b0, acc0); // Requires FMA3 usually, but avx2 implies fma often. 
        // Wait, FMA3 is separate feature bit. _mm256_mul_pd + _mm256_add_pd is safer if FMA not checked.
        // Let's use mul+add to be safe with just avx2 check.
        
        let a1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
        let b1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
        let prod1 = _mm256_mul_pd(a1, b1);
        acc1 = _mm256_add_pd(acc1, prod1);
        
        i += 8;
    }
    
    // Handle remaining 4-blocks
    while i + 4 <= len {
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let bv = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(av, bv);
        acc0 = _mm256_add_pd(acc0, prod);
        i += 4;
    }
    
    // Reduce accumulators
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
    use core::arch::x86_64::*;
    let len = dst.len();
    let mut i = 0;
    
    while i + 4 <= len {
        let d_ptr = dst.as_mut_ptr().add(i);
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);
        
        let dv = _mm256_loadu_pd(d_ptr);
        let av = _mm256_loadu_pd(a_ptr);
        let bv = _mm256_loadu_pd(b_ptr);
        
        // d = d + a * b
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

// --- NEON Implementations ---

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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vector_fma_neon(dst: &mut [f64], a: &[f64], b: &[f64]) {
    use core::arch::aarch64::*;
    let len = dst.len();
    let mut i = 0;
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
