use crate::autograd::{Context, Device, Tensor, TensorReadGuard};
use crate::dtype::{Dtype, Storage};
use crate::simd::vector_fma;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

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
        // h_out, w_out calculation for stride 1, kernel 3
        let h_out = h_in + 2 * padding - 2;
        let w_out = w_in + 2 * padding - 2;

        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len: usize = out_shape.iter().product();
        let mut out_data = vec![0.0; out_len];

        // Standard Winograd F(2,3) matrices. Hardcoded for speed.
        // G (4x3), B^T (4x4), A^T (2x4)

        // We compute U = G * g * G^T per [k, c] 3x3 block.

        let u_len = c_out * c_in * 16;
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
            let n_tiles = n_tiles_h * n_tiles_w;

            let out_plane_len = h_out * w_out;

            out_data
                .par_chunks_mut(c_out * out_plane_len)
                .enumerate()
                .for_each(|(b, out_batch)| {
                    // We could parallelize over tiles, but that requires atomic writes to output or careful locking.
                    // Easier to parallelize over Output Channels (C_out) since they are independent.

                    // First, transform input image into V domain: V = B^T d B.
                    // This is shared across all C_out, so we do it once per batch item.
                    // V: [Tiles, C_in, 4, 4]
                    let mut v_data = vec![0.0; n_tiles * c_in * 16];

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

        // Use Winograd F(2x2, 3x3) for 3x3 kernel with stride 1
        if k_h == 3 && k_w == 3 && stride == 1 {
            return self.winograd_conv2d_3x3(weight, padding);
        }

        let h_out = (h_in + 2 * padding - k_h) / stride + 1;
        let w_out = (w_in + 2 * padding - k_w) / stride + 1;

        let out_shape = vec![n, c_out, h_out, w_out];
        let out_len: usize = out_shape.iter().product();
        let mut out_data = vec![0.0; out_len];

        let k_len = c_in * k_h * k_w;
        let out_plane_len = h_out * w_out;

        {
            let input_data = self.data_f64();
            let weight_data = weight.data_f64();

            // Standard Im2Col implementation. Memory hungry but fast.
            // Parallelize over Batch
            out_data
                .par_chunks_mut(c_out * out_plane_len)
                .enumerate()
                .for_each(|(b, out_batch)| {
                    // Im2Col: Input (C_in, H, W) -> Cols (K_len, Out_len)
                    let mut cols = vec![0.0; k_len * out_plane_len];

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

    pub fn max_pool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D (NCHW)");
        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);

        let h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

        let out_shape = vec![n, c, h_out, w_out];
        let out_len: usize = out_shape.iter().product();
        let mut out_data = vec![0.0; out_len];

        {
            let input_data = self.data_f64();
            // Parallelize over (N, C)
            out_data
                .par_chunks_mut(h_out * w_out)
                .enumerate()
                .for_each(|(idx, out_plane)| {
                    let b = idx / c;
                    let ch = idx % c;

                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let h_start = (oh * stride) as isize - padding as isize;
                            let w_start = (ow * stride) as isize - padding as isize;

                            let mut max_val = f64::NEG_INFINITY;

                            // Optimization: Optimized bounds
                            let kh_start = if h_start < 0 { (-h_start) as usize } else { 0 };
                            let kw_start = if w_start < 0 { (-w_start) as usize } else { 0 };
                            let kh_end = if h_start + kernel_size as isize > h_in as isize {
                                (h_in as isize - h_start) as usize
                            } else {
                                kernel_size
                            };
                            let kw_end = if w_start + kernel_size as isize > w_in as isize {
                                (w_in as isize - w_start) as usize
                            } else {
                                kernel_size
                            };

                            // Inner loops now guaranteed valid
                            for kh in kh_start..kh_end {
                                for kw in kw_start..kw_end {
                                    let h_in_idx = (h_start + kh as isize) as usize;
                                    let w_in_idx = (w_start + kw as isize) as usize;
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

                                                    // Optimized bounds for inner search
                                                    let kh_start = if h_start < 0 {
                                                        (-h_start) as usize
                                                    } else {
                                                        0
                                                    };
                                                    let kw_start = if w_start < 0 {
                                                        (-w_start) as usize
                                                    } else {
                                                        0
                                                    };
                                                    let kh_end = if h_start + kernel_size as isize
                                                        > h_in as isize
                                                    {
                                                        (h_in as isize - h_start) as usize
                                                    } else {
                                                        kernel_size
                                                    };
                                                    let kw_end = if w_start + kernel_size as isize
                                                        > w_in as isize
                                                    {
                                                        (w_in as isize - w_start) as usize
                                                    } else {
                                                        kernel_size
                                                    };

                                                    for kh in kh_start..kh_end {
                                                        for kw in kw_start..kw_end {
                                                            let h_k =
                                                                (h_start + kh as isize) as usize;
                                                            let w_k =
                                                                (w_start + kw as isize) as usize;
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
}
