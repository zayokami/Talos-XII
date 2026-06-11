//! GPU-side ACHF weight projection (row/col, Sinkhorn, low-rank truncation).

use crate::achf::{low_rank_truncate, SinkhornProjectionStats};
use crate::autograd::{Device, Tensor};
use crate::cuda::memory::{alloc, copy_d2h, copy_h2d};
use crate::dtype::Dtype;

pub fn project_rowcol(weight: &Tensor, rows: usize, cols: usize) -> bool {
    if weight.device != Device::Cuda || weight.dtype != Dtype::F32 || !crate::cuda::is_available() {
        return false;
    }
    let d_weight = match weight.cuda_get_or_upload_buffer() {
        Ok(buf) => buf,
        Err(_) => return false,
    };
    let d_weight = match d_weight.as_f32() {
        Some(ptr) => ptr,
        None => return false,
    };
    if crate::cuda::kernels::achf_rowcol_project_f32(d_weight, rows, cols).is_err() {
        return false;
    }
    weight.cuda_clear_host_data_preserve_cache();
    true
}

pub fn project_sinkhorn(
    weight: &Tensor,
    rows: usize,
    cols: usize,
    steps: usize,
    row_scales: Option<&[f32]>,
    col_scales: Option<&[f32]>,
) -> Option<(Vec<f32>, Vec<f32>, SinkhornProjectionStats)> {
    if weight.device != Device::Cuda || weight.dtype != Dtype::F32 || !crate::cuda::is_available() {
        return None;
    }
    let d_weight = weight.cuda_get_or_upload_buffer().ok()?;
    let d_weight = d_weight.as_f32()?;
    let mut row_host = row_scales.map_or_else(|| vec![1.0f32; rows], |v| v.to_vec());
    let mut col_host = col_scales.map_or_else(|| vec![1.0f32; cols], |v| v.to_vec());
    if row_host.len() != rows || col_host.len() != cols {
        return None;
    }
    let d_row = alloc::<f32>(rows).ok()?;
    let d_col = alloc::<f32>(cols).ok()?;
    let warm_started = row_scales.is_some() || col_scales.is_some();
    let (iterations, _, _) = crate::cuda::kernels::achf_sinkhorn_project_f32(
        d_weight, &d_row, &d_col, rows, cols, steps, &row_host, &col_host,
    )
    .ok()?;
    copy_d2h(&mut row_host, &d_row).ok()?;
    copy_d2h(&mut col_host, &d_col).ok()?;
    weight.cuda_clear_host_data_preserve_cache();
    Some((
        row_host,
        col_host,
        SinkhornProjectionStats {
            iterations,
            warm_started,
            ..Default::default()
        },
    ))
}

pub fn apply_low_rank_truncation(
    weight: &Tensor,
    rows: usize,
    cols: usize,
    rank: usize,
    seed: u64,
) -> Option<f64> {
    if weight.device != Device::Cuda || weight.dtype != Dtype::F32 || !crate::cuda::is_available() {
        return None;
    }
    let n = rows.checked_mul(cols)?;
    let d_weight = weight.cuda_get_or_upload_buffer().ok()?;
    let d_weight = d_weight.as_f32()?;
    let d_backup = alloc::<f32>(n).ok()?;
    crate::cuda::kernels::achf_copy_f32(d_weight, &d_backup, n).ok()?;

    let r = rank;
    let mut rng = crate::rng::Rng::from_seed(seed);
    let mut omega = vec![0.0f32; cols * r];
    for v in omega.iter_mut() {
        *v = rng.next_f64_normal() as f32;
    }

    let d_omega = alloc::<f32>(cols * r).ok()?;
    copy_h2d(&d_omega, &omega).ok()?;
    let d_y = alloc::<f32>(rows * r).ok()?;
    let d_z = alloc::<f32>(cols * r).ok()?;
    let d_b = alloc::<f32>(r * cols).ok()?;

    let rows_i32 = i32::try_from(rows).ok()?;
    let cols_i32 = i32::try_from(cols).ok()?;
    let r_i32 = i32::try_from(r).ok()?;

    if crate::cuda::blas::gemm_thread_local_f32(
        false,
        false,
        rows_i32,
        r_i32,
        cols_i32,
        1.0,
        d_weight.as_raw(),
        cols_i32,
        d_omega.as_raw(),
        r_i32,
        0.0,
        d_y.as_raw(),
        r_i32,
    )
    .is_err()
    {
        return cpu_low_rank_fallback(weight, rows, cols, rank, seed);
    }

    const POWER_ITERATIONS: usize = 3;
    let mut y_host = vec![0.0f32; rows * r];
    copy_d2h(&mut y_host, &d_y).ok()?;
    orthonormalize_columns_f32(&mut y_host, rows, r);
    copy_h2d(&d_y, &y_host).ok()?;

    for _ in 0..POWER_ITERATIONS {
        if crate::cuda::blas::gemm_thread_local_f32(
            true,
            false,
            cols_i32,
            r_i32,
            rows_i32,
            1.0,
            d_weight.as_raw(),
            cols_i32,
            d_y.as_raw(),
            r_i32,
            0.0,
            d_z.as_raw(),
            r_i32,
        )
        .is_err()
        {
            return cpu_low_rank_fallback(weight, rows, cols, rank, seed);
        }
        let mut z_host = vec![0.0f32; cols * r];
        copy_d2h(&mut z_host, &d_z).ok()?;
        orthonormalize_columns_f32(&mut z_host, cols, r);
        copy_h2d(&d_z, &z_host).ok()?;

        if crate::cuda::blas::gemm_thread_local_f32(
            false,
            false,
            rows_i32,
            r_i32,
            cols_i32,
            1.0,
            d_weight.as_raw(),
            cols_i32,
            d_z.as_raw(),
            r_i32,
            0.0,
            d_y.as_raw(),
            r_i32,
        )
        .is_err()
        {
            return cpu_low_rank_fallback(weight, rows, cols, rank, seed);
        }
        y_host.fill(0.0);
        copy_d2h(&mut y_host, &d_y).ok()?;
        orthonormalize_columns_f32(&mut y_host, rows, r);
        copy_h2d(&d_y, &y_host).ok()?;
    }

    if crate::cuda::blas::gemm_thread_local_f32(
        true,
        false,
        r_i32,
        cols_i32,
        rows_i32,
        1.0,
        d_y.as_raw(),
        r_i32,
        d_weight.as_raw(),
        cols_i32,
        0.0,
        d_b.as_raw(),
        cols_i32,
    )
    .is_err()
    {
        return cpu_low_rank_fallback(weight, rows, cols, rank, seed);
    }

    if crate::cuda::blas::gemm_thread_local_f32(
        false,
        false,
        rows_i32,
        cols_i32,
        r_i32,
        1.0,
        d_y.as_raw(),
        r_i32,
        d_b.as_raw(),
        cols_i32,
        0.0,
        d_weight.as_raw(),
        cols_i32,
    )
    .is_err()
    {
        return cpu_low_rank_fallback(weight, rows, cols, rank, seed);
    }

    let rel_err = crate::cuda::kernels::achf_frobenius_rel_err_f32(&d_backup, d_weight, n).ok()?;
    weight.cuda_clear_host_data_preserve_cache();
    Some(rel_err)
}

fn cpu_low_rank_fallback(
    weight: &Tensor,
    rows: usize,
    cols: usize,
    rank: usize,
    seed: u64,
) -> Option<f64> {
    let mut w = weight.data_to_f32_vec();
    let rel_err = low_rank_truncate(&mut w, rows, cols, rank, seed);
    crate::achf::sync_weight_from_host_f32(weight, &w);
    Some(rel_err)
}

fn orthonormalize_columns_f32(m: &mut [f32], rows: usize, cols: usize) {
    for j in 0..cols {
        for k in 0..j {
            let mut dot = 0.0f32;
            for i in 0..rows {
                dot += m[i * cols + j] * m[i * cols + k];
            }
            for i in 0..rows {
                m[i * cols + j] -= dot * m[i * cols + k];
            }
        }
        let mut norm_sq = 0.0f32;
        for i in 0..rows {
            let v = m[i * cols + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for i in 0..rows {
                m[i * cols + j] /= norm;
            }
        } else {
            for i in 0..rows {
                m[i * cols + j] = 0.0;
            }
        }
    }
}

pub fn grad_mean_sq(weight_grad: &Tensor) -> Option<f64> {
    if weight_grad.device != Device::Cuda || !crate::cuda::is_available() {
        return None;
    }
    let grad = weight_grad.cuda_grad_get_or_upload_buffer().ok()?;
    let grad = grad.as_f32()?;
    let len = weight_grad.grad.len();
    if len == 0 {
        return Some(0.0);
    }
    crate::cuda::kernels::grad_mean_sq_f32(grad, len).ok()
}
