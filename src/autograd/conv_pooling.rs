use crate::autograd::{Context, Device, Tensor, TensorReadGuard};
use crate::dtype::{Dtype, Storage};
use crate::simd::vector_fma;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::{Arc, RwLock};

const MAX_IM2COL_TEMP_ELEMENTS: usize = 32 * 1024 * 1024;

fn checked_product(shape: &[usize], op: &'static str) -> usize {
    shape.iter().copied().fold(1usize, |acc, dim| {
        acc.checked_mul(dim)
            .unwrap_or_else(|| panic!("{op} output element count overflow"))
    })
}

fn assert_nonzero_dims(op: &'static str, dims: &[(&'static str, usize)]) {
    for (name, value) in dims {
        assert!(*value > 0, "{op} {name} must be positive");
        assert!(
            *value <= isize::MAX as usize,
            "{op} {name} is too large for index arithmetic"
        );
    }
}

fn padded_extent(input: usize, padding: usize, op: &'static str, axis: &'static str) -> usize {
    assert!(
        padding <= isize::MAX as usize,
        "{op} {axis} padding is too large for index arithmetic"
    );
    let double_padding = padding
        .checked_mul(2)
        .unwrap_or_else(|| panic!("{op} {axis} padding overflow"));
    let padded = input
        .checked_add(double_padding)
        .unwrap_or_else(|| panic!("{op} {axis} padded extent overflow"));
    assert!(
        padded <= isize::MAX as usize,
        "{op} {axis} padded extent is too large for index arithmetic"
    );
    padded
}

fn forward_out_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    op: &'static str,
    axis: &'static str,
) -> usize {
    assert_nonzero_dims(op, &[(axis, input), ("kernel", kernel), ("stride", stride)]);
    let padded = padded_extent(input, padding, op, axis);
    assert!(
        padded >= kernel,
        "{op} {axis} kernel larger than padded input"
    );
    (padded - kernel) / stride + 1
}

fn transposed_out_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    op: &'static str,
    axis: &'static str,
) -> usize {
    assert_nonzero_dims(op, &[(axis, input), ("kernel", kernel), ("stride", stride)]);
    let expanded = (input - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(kernel))
        .unwrap_or_else(|| panic!("{op} {axis} transposed extent overflow"));
    let crop = padding
        .checked_mul(2)
        .unwrap_or_else(|| panic!("{op} {axis} padding overflow"));
    assert!(
        expanded > crop,
        "{op} {axis} padding removes the whole transposed output"
    );
    let output = expanded - crop;
    assert!(
        output <= isize::MAX as usize,
        "{op} {axis} output is too large for index arithmetic"
    );
    output
}

fn window_intersection(
    start: isize,
    kernel: usize,
    input_len: usize,
    op: &'static str,
    axis: &'static str,
) -> Range<usize> {
    assert!(
        kernel <= isize::MAX as usize,
        "{op} {axis} kernel is too large for index arithmetic"
    );
    assert!(
        input_len <= isize::MAX as usize,
        "{op} {axis} input is too large for index arithmetic"
    );
    let end = start
        .checked_add(kernel as isize)
        .unwrap_or_else(|| panic!("{op} {axis} window extent overflow"));
    let lo = start.max(0);
    let hi = end.min(input_len as isize);
    assert!(
        lo < hi,
        "{op} {axis} pooling window does not overlap the input"
    );
    lo as usize..hi as usize
}

impl Tensor {
    // Winograd F(2x2, 3x3) implementation
    // Input tile: 4x4, Output tile: 2x2
    fn winograd_conv2d_3x3(&self, weight: &Tensor, padding: usize) -> Tensor {
        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, _, _, _) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );
        // h_out, w_out calculation for stride 1, kernel 3.
        let h_out = forward_out_dim(h_in, 3, 1, padding, "conv2d", "height");
        let w_out = forward_out_dim(w_in, 3, 1, padding, "conv2d", "width");

        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len = checked_product(&out_shape, "conv2d");
        let mut out_data = vec![0.0; out_len];

        // Standard Winograd F(2,3) matrices. Hardcoded for speed.
        // G (4x3), B^T (4x4), A^T (2x4)

        // We compute U = G * g * G^T per [k, c] 3x3 block.

        let u_len = checked_product(&[c_out, c_in, 16], "conv2d");
        let mut u_data = vec![0.0; u_len]; // [C_out, C_in, 4, 4]

        {
            let weight_data = weight.data_f64();

            // Precompute U. This transforms the kernel into Winograd domain.
            u_data
                .par_chunks_mut(16)
                .enumerate()
                .for_each(|(idx, u_block)| {
                    // idx corresponds to (k * c_in + c)
                    let k = idx / c_in;
                    let c = idx % c_in;

                    // Read 3x3 weight
                    let w_base = (k * c_in + c) * 9;
                    let g00 = weight_data[w_base];
                    let g01 = weight_data[w_base + 1];
                    let g02 = weight_data[w_base + 2];
                    let g10 = weight_data[w_base + 3];
                    let g11 = weight_data[w_base + 4];
                    let g12 = weight_data[w_base + 5];
                    let g20 = weight_data[w_base + 6];
                    let g21 = weight_data[w_base + 7];
                    let g22 = weight_data[w_base + 8];

                    // Compute U = G * g * G^T
                    // Unrolled manually to avoid allocation

                    // Tmp = g * G^T
                    let t00 = g00;
                    let t01 = 0.5 * (g00 + g01 + g02);
                    let t02 = 0.5 * (g00 - g01 + g02);
                    let t03 = g02;

                    let t10 = g10;
                    let t11 = 0.5 * (g10 + g11 + g12);
                    let t12 = 0.5 * (g10 - g11 + g12);
                    let t13 = g12;

                    let t20 = g20;
                    let t21 = 0.5 * (g20 + g21 + g22);
                    let t22 = 0.5 * (g20 - g21 + g22);
                    let t23 = g22;

                    // U = G * Tmp
                    u_block[0] = t00;
                    u_block[4] = 0.5 * (t00 + t10 + t20);
                    u_block[8] = 0.5 * (t00 - t10 + t20);
                    u_block[12] = t20;

                    u_block[1] = t01;
                    u_block[5] = 0.5 * (t01 + t11 + t21);
                    u_block[9] = 0.5 * (t01 - t11 + t21);
                    u_block[13] = t21;

                    u_block[2] = t02;
                    u_block[6] = 0.5 * (t02 + t12 + t22);
                    u_block[10] = 0.5 * (t02 - t12 + t22);
                    u_block[14] = t22;

                    u_block[3] = t03;
                    u_block[7] = 0.5 * (t03 + t13 + t23);
                    u_block[11] = 0.5 * (t03 - t13 + t23);
                    u_block[15] = t23;
                });
        }

        {
            let input_data = self.data_f64();

            // Output is computed in 2x2 blocks (tiles).
            let n_tiles_h = h_out.div_ceil(2);
            let n_tiles_w = w_out.div_ceil(2);
            let n_tiles = checked_product(&[n_tiles_h, n_tiles_w], "conv2d");

            let out_plane_len = checked_product(&[h_out, w_out], "conv2d");

            out_data
                .par_chunks_mut(c_out * out_plane_len)
                .enumerate()
                .for_each(|(b, out_batch)| {
                    // We could parallelize over tiles, but that requires atomic writes to output or careful locking.
                    // Easier to parallelize over Output Channels (C_out) since they are independent.

                    // First, transform input image into V domain: V = B^T d B.
                    // This is shared across all C_out, so we do it once per batch item.
                    // V: [Tiles, C_in, 4, 4]
                    let mut v_data = vec![0.0; checked_product(&[n_tiles, c_in, 16], "conv2d")];

                    // Parallelize V computation over (Tile, C_in)
                    v_data
                        .par_chunks_mut(16)
                        .enumerate()
                        .for_each(|(idx, v_block)| {
                            let tile_idx = idx / c_in;
                            let c = idx % c_in;

                            let th = tile_idx / n_tiles_w;
                            let tw = tile_idx % n_tiles_w;

                            let h_start = (th * 2) as isize - padding as isize;
                            let w_start = (tw * 2) as isize - padding as isize;

                            // Read 4x4 input tile d
                            let mut d = [0.0; 16];
                            for i in 0..4 {
                                for j in 0..4 {
                                    let ih = h_start + i as isize;
                                    let iw = w_start + j as isize;
                                    if ih >= 0
                                        && ih < h_in as isize
                                        && iw >= 0
                                        && iw < w_in as isize
                                    {
                                        d[i * 4 + j] = input_data[((b * c_in + c) * h_in
                                            + ih as usize)
                                            * w_in
                                            + iw as usize];
                                    }
                                }
                            }

                            // Compute V = B^T * d * B
                            // 1. Tmp = B^T * d
                            let mut tmp = [0.0; 16];
                            for j in 0..4 {
                                let d0 = d[j];
                                let d1 = d[4 + j];
                                let d2 = d[8 + j];
                                let d3 = d[12 + j];
                                tmp[j] = d0 - d2;
                                tmp[4 + j] = d1 + d2;
                                tmp[8 + j] = d2 - d1;
                                tmp[12 + j] = d1 - d3;
                            }

                            // 2. V = Tmp * B
                            for i in 0..4 {
                                // row i
                                let t0 = tmp[i * 4];
                                let t1 = tmp[i * 4 + 1];
                                let t2 = tmp[i * 4 + 2];
                                let t3 = tmp[i * 4 + 3];
                                v_block[i * 4] = t0 - t2;
                                v_block[i * 4 + 1] = t1 + t2;
                                v_block[i * 4 + 2] = t2 - t1;
                                v_block[i * 4 + 3] = t1 - t3;
                            }
                        });

                    // Now Compute M = U * V and Y = A^T M A
                    // This part is specific to each C_out.
                    out_batch
                        .par_chunks_mut(out_plane_len)
                        .enumerate()
                        .for_each(|(k, out_plane)| {
                            for t in 0..n_tiles {
                                let th = t / n_tiles_w;
                                let tw = t % n_tiles_w;

                                // M = Sum_c (U[k,c] .* V[t,c])
                                let mut m = [0.0; 16];
                                for c in 0..c_in {
                                    let u_ptr = &u_data[((k * c_in + c) * 16)..];
                                    let v_ptr = &v_data[((t * c_in + c) * 16)..];
                                    // Element-wise mul. Hot path!
                                    vector_fma(&mut m, &u_ptr[0..16], &v_ptr[0..16]);
                                }

                                // Y = A^T * m * A
                                // 1. Tmp = A^T * m
                                let mut tmp = [0.0; 8];
                                for j in 0..4 {
                                    let m0 = m[j];
                                    let m1 = m[4 + j];
                                    let m2 = m[8 + j];
                                    let m3 = m[12 + j];
                                    tmp[j] = m0 + m1 + m2;
                                    tmp[4 + j] = m1 - m2 - m3;
                                }

                                // 2. Y = Tmp * A
                                let t00 = tmp[0];
                                let t01 = tmp[1];
                                let t02 = tmp[2];
                                let t03 = tmp[3];
                                let t10 = tmp[4];
                                let t11 = tmp[5];
                                let t12 = tmp[6];
                                let t13 = tmp[7];

                                let y00 = t00 + t01 + t02;
                                let y01 = t01 - t02 - t03;
                                let y10 = t10 + t11 + t12;
                                let y11 = t11 - t12 - t13;

                                // Scatter write to output
                                let oh_base = th * 2;
                                let ow_base = tw * 2;

                                if oh_base < h_out && ow_base < w_out {
                                    out_plane[oh_base * w_out + ow_base] = y00;
                                }
                                if oh_base < h_out && ow_base + 1 < w_out {
                                    out_plane[oh_base * w_out + ow_base + 1] = y01;
                                }
                                if oh_base + 1 < h_out && ow_base < w_out {
                                    out_plane[(oh_base + 1) * w_out + ow_base] = y10;
                                }
                                if oh_base + 1 < h_out && ow_base + 1 < w_out {
                                    out_plane[(oh_base + 1) * w_out + ow_base + 1] = y11;
                                }
                            }
                        });
                });
        }

        // Backward pass: Use standard Im2Col gradient computation.
        // Winograd F(2x2, 3x3) is mathematically equivalent to standard conv2d,
        // so standard backward produces correct gradients for the forward result.

        let parents = vec![self.clone(), weight.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                // Using standard Im2Col backward pass logic.
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let weight = &parents[1];
                    // Use batch lock for better performance
                    let guards = TensorReadGuard::new(&[input, weight]);
                    let input_data = guards.get(0);
                    let weight_data = guards.get(1);

                    let (n, c_in, h_in, w_in) = (
                        input.shape[0],
                        input.shape[1],
                        input.shape[2],
                        input.shape[3],
                    );
                    let (c_out, _, k_h, k_w) = (
                        weight.shape[0],
                        weight.shape[1],
                        weight.shape[2],
                        weight.shape[3],
                    );

                    // Winograd F(2x2, 3x3) only supports stride=1.
                    // Stride and padding are captured; stride is hardcoded to 1.
                    let stride = 1;
                    let h_out = h_in + 2 * padding - 2;
                    let w_out = w_in + 2 * padding - 2; // k_h=3, k_w=3

                    // dL/dInput (Standard Col2Im)
                    {
                        let mut input_grad = input.grad_write_compat();
                        input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(
                            |(idx, in_plane)| {
                                let b = idx / c_in;
                                let c = idx % c_in;

                                for ih in 0..h_in {
                                    let oh_min = (ih + padding).saturating_sub(k_h - 1) / stride;
                                    let oh_max = ((ih + padding) / stride).min(h_out - 1);

                                    for iw in 0..w_in {
                                        let mut sum = 0.0;
                                        let ow_min =
                                            (iw + padding).saturating_sub(k_w - 1) / stride;
                                        let ow_max = ((iw + padding) / stride).min(w_out - 1);

                                        if oh_min <= oh_max && ow_min <= ow_max {
                                            for oh in oh_min..=oh_max {
                                                for ow in ow_min..=ow_max {
                                                    let kh = ih as isize - (oh * stride) as isize
                                                        + padding as isize;
                                                    let kw = iw as isize - (ow * stride) as isize
                                                        + padding as isize;

                                                    if kh >= 0
                                                        && kh < k_h as isize
                                                        && kw >= 0
                                                        && kw < k_w as isize
                                                    {
                                                        for k in 0..c_out {
                                                            let g = grad_out_f64[((b * c_out + k)
                                                                * h_out
                                                                + oh)
                                                                * w_out
                                                                + ow];
                                                            let w = weight_data[((k * c_in + c)
                                                                * k_h
                                                                + kh as usize)
                                                                * k_w
                                                                + kw as usize];
                                                            sum += g * w;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        in_plane[ih * w_in + iw] += sum;
                                    }
                                }
                            },
                        );
                    }

                    // dL/dWeight
                    {
                        let mut weight_grad = weight.grad_write_compat();
                        weight_grad.par_chunks_mut(k_h * k_w).enumerate().for_each(
                            |(idx, w_plane)| {
                                let k = idx / c_in;
                                let c = idx % c_in;

                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let mut sum = 0.0;
                                        for b in 0..n {
                                            for oh in 0..h_out {
                                                for ow in 0..w_out {
                                                    let h_in_idx = (oh * stride) as isize
                                                        - padding as isize
                                                        + kh as isize;
                                                    let w_in_idx = (ow * stride) as isize
                                                        - padding as isize
                                                        + kw as isize;

                                                    if h_in_idx >= 0
                                                        && h_in_idx < h_in as isize
                                                        && w_in_idx >= 0
                                                        && w_in_idx < w_in as isize
                                                    {
                                                        let val_in = input_data[((b * c_in + c)
                                                            * h_in
                                                            + h_in_idx as usize)
                                                            * w_in
                                                            + w_in_idx as usize];
                                                        let g_val = grad_out_f64[((b * c_out + k)
                                                            * h_out
                                                            + oh)
                                                            * w_out
                                                            + ow];
                                                        sum += val_in * g_val;
                                                    }
                                                }
                                            }
                                        }
                                        w_plane[kh * k_w + kw] += sum;
                                    }
                                }
                            },
                        );
                    }
                }),
            })),
        }
    }

    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        assert_eq!(weight.shape.len(), 4, "Weight must be 4D (OIHW)");
        assert!(stride > 0, "conv2d stride must be positive");

        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, c_in_k, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );

        assert_eq!(
            c_in, c_in_k,
            "Input channels must match weight input channels"
        );
        assert_nonzero_dims(
            "conv2d",
            &[
                ("batch", n),
                ("input channels", c_in),
                ("output channels", c_out),
                ("height", h_in),
                ("width", w_in),
                ("kernel height", k_h),
                ("kernel width", k_w),
                ("stride", stride),
            ],
        );
        let h_out = forward_out_dim(h_in, k_h, stride, padding, "conv2d", "height");
        let w_out = forward_out_dim(w_in, k_w, stride, padding, "conv2d", "width");

        // Use Winograd F(2x2, 3x3) for 3x3 kernel with stride 1
        if k_h == 3 && k_w == 3 && stride == 1 {
            return self.winograd_conv2d_3x3(weight, padding);
        }

        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len = checked_product(&out_shape, "conv2d");
        let mut out_data = vec![0.0; out_len];

        let k_len = checked_product(&[c_in, k_h, k_w], "conv2d");
        let out_plane_len = checked_product(&[h_out, w_out], "conv2d");
        let out_batch_len = checked_product(&[c_out, out_plane_len], "conv2d");
        let im2col_len = checked_product(&[k_len, out_plane_len], "conv2d");
        assert!(
            im2col_len <= MAX_IM2COL_TEMP_ELEMENTS,
            "conv2d im2col workspace too large; use smaller input/kernel or split the batch"
        );

        {
            let input_data = self.data_f64();
            let weight_data = weight.data_f64();

            // Standard Im2Col implementation. Memory hungry but fast.
            // Parallelize over Batch
            out_data
                .par_chunks_mut(out_batch_len)
                .enumerate()
                .for_each(|(b, out_batch)| {
                    // Im2Col: Input (C_in, H, W) -> Cols (K_len, Out_len)
                    let mut cols = vec![0.0; im2col_len];

                    // Parallelize filling cols (by kernel rows)
                    cols.par_chunks_mut(out_plane_len)
                        .enumerate()
                        .for_each(|(k_idx, col_row)| {
                            let c = k_idx / (k_h * k_w);
                            let rem = k_idx % (k_h * k_w);
                            let kh = rem / k_w;
                            let kw = rem % k_w;

                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let h_in_idx =
                                        (oh * stride) as isize - padding as isize + kh as isize;
                                    let w_in_idx =
                                        (ow * stride) as isize - padding as isize + kw as isize;

                                    if h_in_idx >= 0
                                        && h_in_idx < h_in as isize
                                        && w_in_idx >= 0
                                        && w_in_idx < w_in as isize
                                    {
                                        col_row[oh * w_out + ow] =
                                            input_data[((b * c_in + c) * h_in + h_in_idx as usize)
                                                * w_in
                                                + w_in_idx as usize];
                                    }
                                }
                            }
                        });

                    // GEMM: Weight (C_out, K_len) * Cols (K_len, Out_len) -> Out (C_out, Out_len)
                    // out_batch is already slice of size C_out * Out_len

                    // Iterate over output rows (C_out)
                    out_batch
                        .par_chunks_mut(out_plane_len)
                        .enumerate()
                        .for_each(|(out_c, out_row)| {
                            // For each output channel, dot product weight row with all cols
                            // weight row start: out_c * k_len
                            let w_row_start = out_c * k_len;
                            let w_row = &weight_data[w_row_start..w_row_start + k_len];

                            for i in 0..out_plane_len {
                                let mut sum = 0.0;
                                // This inner loop is the hot path.
                                // Vectorization potential here.
                                for k in 0..k_len {
                                    sum += w_row[k] * cols[k * out_plane_len + i];
                                }
                                out_row[i] = sum;
                            }
                        });
                });
        }

        let parents = vec![self.clone(), weight.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let weight = &parents[1];
                    // Use batch lock for better performance
                    let guards = TensorReadGuard::new(&[input, weight]);
                    let input_data = guards.get(0);
                    let weight_data = guards.get(1);

                    // dL/dInput
                    {
                        let mut input_grad = input.grad_write_compat();
                        // Parallel over Input (N, C_in)
                        input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(
                            |(idx, in_plane)| {
                                let b = idx / c_in;
                                let c = idx % c_in;

                                // Optimized Col2Im (Transposed Conv)
                                for ih in 0..h_in {
                                    // Pre-calculate bounds to avoid inner loop checks
                                    let oh_min = (ih + padding).saturating_sub(k_h - 1) / stride;
                                    let oh_max = ((ih + padding) / stride).min(h_out - 1);

                                    for iw in 0..w_in {
                                        let mut sum = 0.0;
                                        let ow_min =
                                            (iw + padding).saturating_sub(k_w - 1) / stride;
                                        let ow_max = ((iw + padding) / stride).min(w_out - 1);

                                        // Check if range is valid (could be empty if padding is small/large)
                                        if oh_min <= oh_max && ow_min <= ow_max {
                                            for oh in oh_min..=oh_max {
                                                for ow in ow_min..=ow_max {
                                                    // ih = oh*s - p + kh => kh = ih - oh*s + p
                                                    let kh = ih as isize - (oh * stride) as isize
                                                        + padding as isize;
                                                    let kw = iw as isize - (ow * stride) as isize
                                                        + padding as isize;

                                                    if kh >= 0
                                                        && kh < k_h as isize
                                                        && kw >= 0
                                                        && kw < k_w as isize
                                                    {
                                                        // Should always be true given bounds, but stride check needed?
                                                        // If we iterate oh, ow, kh is determined.

                                                        for k in 0..c_out {
                                                            let g = grad_out_f64[((b * c_out + k)
                                                                * h_out
                                                                + oh)
                                                                * w_out
                                                                + ow];
                                                            let w = weight_data[((k * c_in + c)
                                                                * k_h
                                                                + kh as usize)
                                                                * k_w
                                                                + kw as usize];
                                                            sum += g * w;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        in_plane[ih * w_in + iw] += sum;
                                    }
                                }
                            },
                        );
                    }

                    // dL/dWeight
                    {
                        let mut weight_grad = weight.grad_write_compat();
                        // dWeight = grad_out * Input_Cols^T
                        // Implemented via manual accumulation over batch

                        // Direct loop is safer for memory.

                        // Parallel over Weight (C_out, C_in, KH, KW)
                        weight_grad.par_chunks_mut(k_h * k_w).enumerate().for_each(
                            |(idx, w_plane)| {
                                let k = idx / c_in;
                                let c = idx % c_in;

                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let mut sum = 0.0;
                                        for b in 0..n {
                                            for oh in 0..h_out {
                                                for ow in 0..w_out {
                                                    let h_in_idx = (oh * stride) as isize
                                                        - padding as isize
                                                        + kh as isize;
                                                    let w_in_idx = (ow * stride) as isize
                                                        - padding as isize
                                                        + kw as isize;

                                                    if h_in_idx >= 0
                                                        && h_in_idx < h_in as isize
                                                        && w_in_idx >= 0
                                                        && w_in_idx < w_in as isize
                                                    {
                                                        let val_in = input_data[((b * c_in + c)
                                                            * h_in
                                                            + h_in_idx as usize)
                                                            * w_in
                                                            + w_in_idx as usize];
                                                        let g_val = grad_out_f64[((b * c_out + k)
                                                            * h_out
                                                            + oh)
                                                            * w_out
                                                            + ow];
                                                        sum += val_in * g_val;
                                                    }
                                                }
                                            }
                                        }
                                        w_plane[kh * k_w + kw] += sum;
                                    }
                                }
                            },
                        );
                    }
                }),
            })),
        }
    }

    /// Compatibility alias for Conv2DCompress-style operator names.
    ///
    /// This computes the same mathematical convolution as `conv2d`; the project
    /// does not currently maintain a separate compressed-weight storage format.
    pub fn conv2d_compress(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        self.conv2d(weight, stride, padding)
    }

    pub fn max_pool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        assert!(kernel_size > 0, "max_pool2d kernel_size must be positive");
        assert!(stride > 0, "max_pool2d stride must be positive");
        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        assert_nonzero_dims(
            "max_pool2d",
            &[
                ("batch", n),
                ("channels", c),
                ("height", h_in),
                ("width", w_in),
                ("kernel", kernel_size),
                ("stride", stride),
            ],
        );

        let h_out = forward_out_dim(h_in, kernel_size, stride, padding, "max_pool2d", "height");
        let w_out = forward_out_dim(w_in, kernel_size, stride, padding, "max_pool2d", "width");

        let out_shape = vec![n, c, h_out, w_out];
        let out_len = checked_product(&out_shape, "max_pool2d");
        let out_plane_len = checked_product(&[h_out, w_out], "max_pool2d");
        let mut out_data = vec![0.0; out_len];

        {
            let input_data = self.data_f64();
            // Parallelize over (N, C)
            out_data
                .par_chunks_mut(out_plane_len)
                .enumerate()
                .for_each(|(idx, out_plane)| {
                    let b = idx / c;
                    let ch = idx % c;

                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let h_start = (oh * stride) as isize - padding as isize;
                            let w_start = (ow * stride) as isize - padding as isize;

                            let mut max_val = f64::NEG_INFINITY;

                            let h_range = window_intersection(
                                h_start,
                                kernel_size,
                                h_in,
                                "max_pool2d",
                                "height",
                            );
                            let w_range = window_intersection(
                                w_start,
                                kernel_size,
                                w_in,
                                "max_pool2d",
                                "width",
                            );

                            for h_in_idx in h_range {
                                for w_in_idx in w_range.clone() {
                                    let val = input_data
                                        [((b * c + ch) * h_in + h_in_idx) * w_in + w_in_idx];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                            out_plane[oh * w_out + ow] = max_val;
                        }
                    }
                });
        }

        let parents = vec![self.clone()];

        Tensor {
            data: Storage::F64(Arc::new(RwLock::new(out_data))),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(Dtype::F64)),
            shape: out_shape,
            device: Device::Cpu,
            dtype: Dtype::F64,
            _ctx: Some(Arc::new(Context {
                parents,
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out_f64 = grad_out.to_f64_vec();
                    let input = &parents[0];
                    let input_data = input.data_f64();
                    let mut input_grad = input.grad_write_compat();

                    // Parallelize over Input (N, C)
                    input_grad.par_chunks_mut(h_in * w_in).enumerate().for_each(
                        |(idx, in_plane)| {
                            let b = idx / c;
                            let ch = idx % c;

                            for ih in 0..h_in {
                                for iw in 0..w_in {
                                    let mut grad_sum = 0.0;
                                    let val_in = input_data[((b * c + ch) * h_in + ih) * w_in + iw];

                                    // Determine possible output windows
                                    // ih = oh*s - p + kh  => oh*s = ih + p - kh
                                    // oh_min occurs when kh is max (k-1) -> oh*s = ih + p - (k-1)
                                    // oh_max occurs when kh is min (0)   -> oh*s = ih + p

                                    let oh_min =
                                        (ih + padding).saturating_sub(kernel_size - 1) / stride;
                                    let oh_max = ((ih + padding) / stride).min(h_out - 1);
                                    let ow_min =
                                        (iw + padding).saturating_sub(kernel_size - 1) / stride;
                                    let ow_max = ((iw + padding) / stride).min(w_out - 1);

                                    if oh_min <= oh_max && ow_min <= ow_max {
                                        for oh in oh_min..=oh_max {
                                            for ow in ow_min..=ow_max {
                                                // Check stride alignment effectively handled by division/range but:
                                                // We need to check if ih is actually in the window for this oh.
                                                // The range calculation above is necessary but not sufficient if stride > 1?
                                                // Actually integer division handles "floor".
                                                // Let's verify: oh*s <= ih+p < oh*s + k
                                                // oh*s - p <= ih < oh*s - p + k

                                                let h_start =
                                                    (oh * stride) as isize - padding as isize;
                                                let w_start =
                                                    (ow * stride) as isize - padding as isize;

                                                if (ih as isize) >= h_start
                                                    && (ih as isize)
                                                        < h_start + kernel_size as isize
                                                    && (iw as isize) >= w_start
                                                    && (iw as isize)
                                                        < w_start + kernel_size as isize
                                                {
                                                    // Re-find max
                                                    let mut max_val = f64::NEG_INFINITY;

                                                    let h_range = window_intersection(
                                                        h_start,
                                                        kernel_size,
                                                        h_in,
                                                        "max_pool2d",
                                                        "height",
                                                    );
                                                    let w_range = window_intersection(
                                                        w_start,
                                                        kernel_size,
                                                        w_in,
                                                        "max_pool2d",
                                                        "width",
                                                    );

                                                    for h_k in h_range {
                                                        for w_k in w_range.clone() {
                                                            let v = input_data[((b * c + ch)
                                                                * h_in
                                                                + h_k)
                                                                * w_in
                                                                + w_k];
                                                            if v > max_val {
                                                                max_val = v;
                                                            }
                                                        }
                                                    }

                                                    if (val_in - max_val).abs() < 1e-6 {
                                                        grad_sum += grad_out_f64[((b * c + ch)
                                                            * h_out
                                                            + oh)
                                                            * w_out
                                                            + ow];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    in_plane[ih * w_in + iw] += grad_sum;
                                }
                            }
                        },
                    );
                }),
            })),
        }
    }

    /// Average pooling over NCHW tensors.
    ///
    /// When `count_include_pad` is false, padded cells are excluded from the
    /// denominator. This matches the most common neural-network API behavior.
    pub fn avg_pool2d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        assert!(kernel_size > 0, "avg_pool2d kernel_size must be positive");
        assert!(stride > 0, "avg_pool2d stride must be positive");
        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        assert_nonzero_dims(
            "avg_pool2d",
            &[
                ("batch", n),
                ("channels", c),
                ("height", h_in),
                ("width", w_in),
                ("kernel", kernel_size),
                ("stride", stride),
            ],
        );
        let h_out = forward_out_dim(h_in, kernel_size, stride, padding, "avg_pool2d", "height");
        let w_out = forward_out_dim(w_in, kernel_size, stride, padding, "avg_pool2d", "width");
        let out_shape = vec![n, c, h_out, w_out];
        let out_len = checked_product(&out_shape, "avg_pool2d");
        let input = self.data_as_f64_vec();
        let mut output = vec![0.0; out_len];
        let plane_len = checked_product(&[h_out, w_out], "avg_pool2d");
        let mut denominators = vec![0usize; plane_len];
        let padded_denominator = kernel_size
            .checked_mul(kernel_size)
            .unwrap_or_else(|| panic!("avg_pool2d kernel area overflow"));

        for oh in 0..h_out {
            for ow in 0..w_out {
                let h_start = (oh * stride) as isize - padding as isize;
                let w_start = (ow * stride) as isize - padding as isize;
                let h_range =
                    window_intersection(h_start, kernel_size, h_in, "avg_pool2d", "height");
                let w_range =
                    window_intersection(w_start, kernel_size, w_in, "avg_pool2d", "width");
                let valid = checked_product(&[h_range.len(), w_range.len()], "avg_pool2d");
                denominators[oh * w_out + ow] = if count_include_pad {
                    padded_denominator
                } else {
                    valid
                };
            }
        }

        for batch in 0..n {
            for channel in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let h_start = (oh * stride) as isize - padding as isize;
                        let w_start = (ow * stride) as isize - padding as isize;
                        let mut sum = 0.0;
                        let h_range =
                            window_intersection(h_start, kernel_size, h_in, "avg_pool2d", "height");
                        let w_range =
                            window_intersection(w_start, kernel_size, w_in, "avg_pool2d", "width");
                        for ih in h_range {
                            for iw in w_range.clone() {
                                sum += input[((batch * c + channel) * h_in + ih) * w_in + iw];
                            }
                        }
                        let denom = denominators[oh * w_out + ow] as f64;
                        output[((batch * c + channel) * h_out + oh) * w_out + ow] = sum / denom;
                    }
                }
            }
        }

        let denominators = Arc::new(denominators);
        let dtype = self.dtype;
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape: out_shape,
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let mut input_grad = parents[0].grad_write_compat();
                    for batch in 0..n {
                        for channel in 0..c {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let h_start = (oh * stride) as isize - padding as isize;
                                    let w_start = (ow * stride) as isize - padding as isize;
                                    let grad = grad_out
                                        [((batch * c + channel) * h_out + oh) * w_out + ow]
                                        / denominators[oh * w_out + ow] as f64;
                                    let h_range = window_intersection(
                                        h_start,
                                        kernel_size,
                                        h_in,
                                        "avg_pool2d",
                                        "height",
                                    );
                                    let w_range = window_intersection(
                                        w_start,
                                        kernel_size,
                                        w_in,
                                        "avg_pool2d",
                                        "width",
                                    );
                                    for ih in h_range {
                                        for iw in w_range.clone() {
                                            input_grad[((batch * c + channel) * h_in + ih)
                                                * w_in
                                                + iw] += grad;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }),
            })),
        }
    }

    /// Transposed 2D convolution over NCHW tensors.
    ///
    /// Weight layout is [C_in, C_out, K_h, K_w].
    pub fn conv2d_transpose(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        assert_eq!(weight.shape.len(), 4, "Weight must be 4D (IOHW)");
        assert!(stride > 0, "conv2d_transpose stride must be positive");
        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (w_c_in, c_out, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );
        assert_eq!(c_in, w_c_in, "input channels must match weight channels");
        assert_nonzero_dims(
            "conv2d_transpose",
            &[
                ("batch", n),
                ("input channels", c_in),
                ("output channels", c_out),
                ("height", h_in),
                ("width", w_in),
                ("kernel height", k_h),
                ("kernel width", k_w),
                ("stride", stride),
            ],
        );
        let h_out = transposed_out_dim(h_in, k_h, stride, padding, "conv2d_transpose", "height");
        let w_out = transposed_out_dim(w_in, k_w, stride, padding, "conv2d_transpose", "width");
        let h_out_padded = h_out
            .checked_add(padding)
            .unwrap_or_else(|| panic!("conv2d_transpose height bound overflow"));
        let w_out_padded = w_out
            .checked_add(padding)
            .unwrap_or_else(|| panic!("conv2d_transpose width bound overflow"));
        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len = checked_product(&out_shape, "conv2d_transpose");
        let input = self.data_as_f64_vec();
        let weight_data = weight.data_as_f64_vec();
        let mut output = vec![0.0; out_len];

        for b in 0..n {
            for ic in 0..c_in {
                for ih in 0..h_in {
                    for iw in 0..w_in {
                        let value = input[((b * c_in + ic) * h_in + ih) * w_in + iw];
                        for oc in 0..c_out {
                            for kh in 0..k_h {
                                let oh = ih * stride + kh;
                                if oh < padding || oh >= h_out_padded {
                                    continue;
                                }
                                let oh = oh - padding;
                                for kw in 0..k_w {
                                    let ow = iw * stride + kw;
                                    if ow < padding || ow >= w_out_padded {
                                        continue;
                                    }
                                    let ow = ow - padding;
                                    let w = weight_data[((ic * c_out + oc) * k_h + kh) * k_w + kw];
                                    output[((b * c_out + oc) * h_out + oh) * w_out + ow] +=
                                        value * w;
                                }
                            }
                        }
                    }
                }
            }
        }

        let dtype = Tensor::binary_dtype(self.dtype, weight.dtype);
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape: out_shape,
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), weight.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let input = parents[0].data_as_f64_vec();
                    let weight_data = parents[1].data_as_f64_vec();
                    let mut input_delta = vec![0.0; input.len()];
                    let mut weight_delta = vec![0.0; weight_data.len()];
                    for b in 0..n {
                        for ic in 0..c_in {
                            for ih in 0..h_in {
                                for iw in 0..w_in {
                                    let input_idx = ((b * c_in + ic) * h_in + ih) * w_in + iw;
                                    let mut input_sum = 0.0;
                                    for oc in 0..c_out {
                                        for kh in 0..k_h {
                                            let oh = ih * stride + kh;
                                            if oh < padding || oh >= h_out_padded {
                                                continue;
                                            }
                                            let oh = oh - padding;
                                            for kw in 0..k_w {
                                                let ow = iw * stride + kw;
                                                if ow < padding || ow >= w_out_padded {
                                                    continue;
                                                }
                                                let ow = ow - padding;
                                                let out_idx =
                                                    ((b * c_out + oc) * h_out + oh) * w_out + ow;
                                                let w_idx =
                                                    ((ic * c_out + oc) * k_h + kh) * k_w + kw;
                                                let grad = grad_out[out_idx];
                                                input_sum += grad * weight_data[w_idx];
                                                weight_delta[w_idx] += grad * input[input_idx];
                                            }
                                        }
                                    }
                                    input_delta[input_idx] += input_sum;
                                }
                            }
                        }
                    }
                    {
                        let mut input_grad = parents[0].grad_write_compat();
                        for (dst, delta) in input_grad.iter_mut().zip(input_delta.iter()) {
                            *dst += *delta;
                        }
                    }
                    {
                        let mut weight_grad = parents[1].grad_write_compat();
                        for (dst, delta) in weight_grad.iter_mut().zip(weight_delta.iter()) {
                            *dst += *delta;
                        }
                    }
                }),
            })),
        }
    }

    pub fn deconvolution(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        self.conv2d_transpose(weight, stride, padding)
    }

    /// Depthwise 2D convolution over NCHW tensors.
    ///
    /// Weight layout is [C_in, channel_multiplier, K_h, K_w].
    pub fn depthwise_conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        assert_eq!(weight.shape.len(), 4, "Weight must be 4D");
        assert!(stride > 0, "depthwise_conv2d stride must be positive");
        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (w_c_in, multiplier, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );
        assert_eq!(c_in, w_c_in, "input channels must match weight channels");
        assert_nonzero_dims(
            "depthwise_conv2d",
            &[
                ("batch", n),
                ("input channels", c_in),
                ("channel multiplier", multiplier),
                ("height", h_in),
                ("width", w_in),
                ("kernel height", k_h),
                ("kernel width", k_w),
                ("stride", stride),
            ],
        );
        let c_out = c_in
            .checked_mul(multiplier)
            .unwrap_or_else(|| panic!("depthwise_conv2d output channels overflow"));
        let h_out = forward_out_dim(h_in, k_h, stride, padding, "depthwise_conv2d", "height");
        let w_out = forward_out_dim(w_in, k_w, stride, padding, "depthwise_conv2d", "width");
        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len = checked_product(&out_shape, "depthwise_conv2d");
        let input = self.data_as_f64_vec();
        let weight_data = weight.data_as_f64_vec();
        let mut output = vec![0.0; out_len];

        for b in 0..n {
            for ic in 0..c_in {
                for m in 0..multiplier {
                    let oc = ic * multiplier + m;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum = 0.0;
                            for kh in 0..k_h {
                                let ih = oh * stride + kh;
                                if ih < padding || ih >= h_in + padding {
                                    continue;
                                }
                                let ih = ih - padding;
                                for kw in 0..k_w {
                                    let iw = ow * stride + kw;
                                    if iw < padding || iw >= w_in + padding {
                                        continue;
                                    }
                                    let iw = iw - padding;
                                    sum += input[((b * c_in + ic) * h_in + ih) * w_in + iw]
                                        * weight_data
                                            [((ic * multiplier + m) * k_h + kh) * k_w + kw];
                                }
                            }
                            output[((b * c_out + oc) * h_out + oh) * w_out + ow] = sum;
                        }
                    }
                }
            }
        }

        let dtype = Tensor::binary_dtype(self.dtype, weight.dtype);
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape: out_shape,
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), weight.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let input = parents[0].data_as_f64_vec();
                    let weight_data = parents[1].data_as_f64_vec();
                    let mut input_delta = vec![0.0; input.len()];
                    let mut weight_delta = vec![0.0; weight_data.len()];
                    for b in 0..n {
                        for ic in 0..c_in {
                            for m in 0..multiplier {
                                let oc = ic * multiplier + m;
                                for oh in 0..h_out {
                                    for ow in 0..w_out {
                                        let grad =
                                            grad_out[((b * c_out + oc) * h_out + oh) * w_out + ow];
                                        for kh in 0..k_h {
                                            let ih = oh * stride + kh;
                                            if ih < padding || ih >= h_in + padding {
                                                continue;
                                            }
                                            let ih = ih - padding;
                                            for kw in 0..k_w {
                                                let iw = ow * stride + kw;
                                                if iw < padding || iw >= w_in + padding {
                                                    continue;
                                                }
                                                let iw = iw - padding;
                                                let input_idx =
                                                    ((b * c_in + ic) * h_in + ih) * w_in + iw;
                                                let weight_idx =
                                                    ((ic * multiplier + m) * k_h + kh) * k_w + kw;
                                                input_delta[input_idx] +=
                                                    grad * weight_data[weight_idx];
                                                weight_delta[weight_idx] += grad * input[input_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    {
                        let mut input_grad = parents[0].grad_write_compat();
                        for (dst, delta) in input_grad.iter_mut().zip(input_delta.iter()) {
                            *dst += *delta;
                        }
                    }
                    {
                        let mut weight_grad = parents[1].grad_write_compat();
                        for (dst, delta) in weight_grad.iter_mut().zip(weight_delta.iter()) {
                            *dst += *delta;
                        }
                    }
                }),
            })),
        }
    }

    /// 3D convolution over NCDHW tensors, weight layout [C_out, C_in, K_d, K_h, K_w].
    pub fn conv3d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 5, "Input must be 5D (NCDHW)");
        assert_eq!(weight.shape.len(), 5, "Weight must be 5D (OIDHW)");
        assert!(stride > 0, "conv3d stride must be positive");
        let (n, c_in, d_in, h_in, w_in) = (
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
            self.shape[4],
        );
        let (c_out, w_c_in, k_d, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
            weight.shape[4],
        );
        assert_eq!(c_in, w_c_in, "input channels must match weight channels");
        assert_nonzero_dims(
            "conv3d",
            &[
                ("batch", n),
                ("input channels", c_in),
                ("output channels", c_out),
                ("depth", d_in),
                ("height", h_in),
                ("width", w_in),
                ("kernel depth", k_d),
                ("kernel height", k_h),
                ("kernel width", k_w),
                ("stride", stride),
            ],
        );
        let d_out = forward_out_dim(d_in, k_d, stride, padding, "conv3d", "depth");
        let h_out = forward_out_dim(h_in, k_h, stride, padding, "conv3d", "height");
        let w_out = forward_out_dim(w_in, k_w, stride, padding, "conv3d", "width");
        let out_shape = vec![n, c_out, d_out, h_out, w_out];
        let out_len = checked_product(&out_shape, "conv3d");
        let input = self.data_as_f64_vec();
        let weight_data = weight.data_as_f64_vec();
        let mut output = vec![0.0; out_len];

        for b in 0..n {
            for oc in 0..c_out {
                for od in 0..d_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum = 0.0;
                            for ic in 0..c_in {
                                for kd in 0..k_d {
                                    let id = od * stride + kd;
                                    if id < padding || id >= d_in + padding {
                                        continue;
                                    }
                                    let id = id - padding;
                                    for kh in 0..k_h {
                                        let ih = oh * stride + kh;
                                        if ih < padding || ih >= h_in + padding {
                                            continue;
                                        }
                                        let ih = ih - padding;
                                        for kw in 0..k_w {
                                            let iw = ow * stride + kw;
                                            if iw < padding || iw >= w_in + padding {
                                                continue;
                                            }
                                            let iw = iw - padding;
                                            let input_idx =
                                                (((b * c_in + ic) * d_in + id) * h_in + ih) * w_in
                                                    + iw;
                                            let weight_idx =
                                                (((oc * c_in + ic) * k_d + kd) * k_h + kh) * k_w
                                                    + kw;
                                            sum += input[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                            output[(((b * c_out + oc) * d_out + od) * h_out + oh) * w_out + ow] =
                                sum;
                        }
                    }
                }
            }
        }

        let dtype = Tensor::binary_dtype(self.dtype, weight.dtype);
        Tensor {
            data: Storage::from_f64_vec(output, dtype),
            grad: Storage::zeros(out_len, Tensor::grad_dtype_for(dtype)),
            shape: out_shape,
            device: Device::Cpu,
            dtype,
            _ctx: Some(Arc::new(Context {
                parents: vec![self.clone(), weight.clone()],
                backward_op: Box::new(move |grad_out, parents| {
                    let grad_out = grad_out.to_f64_vec();
                    let input = parents[0].data_as_f64_vec();
                    let weight_data = parents[1].data_as_f64_vec();
                    let mut input_delta = vec![0.0; input.len()];
                    let mut weight_delta = vec![0.0; weight_data.len()];
                    for b in 0..n {
                        for oc in 0..c_out {
                            for od in 0..d_out {
                                for oh in 0..h_out {
                                    for ow in 0..w_out {
                                        let grad = grad_out[(((b * c_out + oc) * d_out + od)
                                            * h_out
                                            + oh)
                                            * w_out
                                            + ow];
                                        for ic in 0..c_in {
                                            for kd in 0..k_d {
                                                let id = od * stride + kd;
                                                if id < padding || id >= d_in + padding {
                                                    continue;
                                                }
                                                let id = id - padding;
                                                for kh in 0..k_h {
                                                    let ih = oh * stride + kh;
                                                    if ih < padding || ih >= h_in + padding {
                                                        continue;
                                                    }
                                                    let ih = ih - padding;
                                                    for kw in 0..k_w {
                                                        let iw = ow * stride + kw;
                                                        if iw < padding || iw >= w_in + padding {
                                                            continue;
                                                        }
                                                        let iw = iw - padding;
                                                        let input_idx =
                                                            (((b * c_in + ic) * d_in + id) * h_in
                                                                + ih)
                                                                * w_in
                                                                + iw;
                                                        let weight_idx =
                                                            (((oc * c_in + ic) * k_d + kd) * k_h
                                                                + kh)
                                                                * k_w
                                                                + kw;
                                                        input_delta[input_idx] +=
                                                            grad * weight_data[weight_idx];
                                                        weight_delta[weight_idx] +=
                                                            grad * input[input_idx];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    {
                        let mut input_grad = parents[0].grad_write_compat();
                        for (dst, delta) in input_grad.iter_mut().zip(input_delta.iter()) {
                            *dst += *delta;
                        }
                    }
                    {
                        let mut weight_grad = parents[1].grad_write_compat();
                        for (dst, delta) in weight_grad.iter_mut().zip(weight_delta.iter()) {
                            *dst += *delta;
                        }
                    }
                }),
            })),
        }
    }
}
