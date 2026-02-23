use core::f32;
use cubecl::{cube, prelude::*};

/// Cast a `u32` to `f32` inside a CubeCL kernel.
///
/// Used to convert integer loop counters or indices to float for arithmetic
/// inside GPU kernels where mixed integer/float casts must be explicit.
#[cube]
pub fn u32_to_float(x: u32) -> f32 {
    f32::cast_from(x)
}

/// Sentinel "infinity" value used to initialise the k-NN scratch buffers.
///
/// Any real distance will be smaller than this, so all `n` candidates will
/// replace the sentinel during the first iteration of the insertion sort.
/// Equal to `f32::MAX` (≈ 3.4 × 10³⁸).
const INFINITY: f32 = 3.40282347e+38;

/// KNN forward kernel — one GPU thread per row.
///
/// `local_distances` and `local_indices` have shape **[n, k]** so every thread
/// uses its own contiguous slot `[row*k .. row*k+k]` — no data races.
#[cube(launch)]
pub fn knn_kernel<F: Float + CubePrimitive, I: Int>(
    pairwise_distances: &Tensor<F>,  // [n, n]
    k: u32,                          // number of nearest neighbours
    local_distances: &mut Tensor<F>, // [n, k] scratch — one slice per row
    local_indices: &mut Tensor<I>,   // [n, k] scratch — one slice per row
    indices: &mut Tensor<I>,         // [n, k] output
    distances: &mut Tensor<F>,       // [n, k] output
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = pairwise_distances.shape(0);
    let k_usize = k as usize;

    if row >= n {
        // Thread is out of bounds — do nothing.
    } else {
        // ── initialise this row's scratch with sentinel values ──────────────
        for i in 0..k_usize {
            local_distances[row * k_usize + i] = F::new(INFINITY);
            local_indices[row * k_usize + i] = I::cast_from(n); // invalid sentinel
        }

        // ── insertion-sort the k smallest distances for this row ─────────────
        for col in 0..n {
            if row != col {
                let dist = pairwise_distances[row * n + col];

                // Only consider inserting if smaller than the current worst neighbour.
                if dist < local_distances[row * k_usize + k_usize - 1] {
                    let mut i = k_usize - 1;

                    // Shift larger entries one slot to the right to make room.
                    while i > 0 {
                        if dist < local_distances[row * k_usize + i] {
                            local_distances[row * k_usize + i] =
                                local_distances[row * k_usize + i - 1];
                            local_indices[row * k_usize + i] =
                                local_indices[row * k_usize + i - 1];
                        } else {
                            break;
                        }
                        i -= 1;
                    }

                    // Insert the new distance at position i.
                    local_distances[row * k_usize + i] = dist;
                    local_indices[row * k_usize + i] = I::cast_from(col);
                }
            }
        }

        // ── copy scratch → output ────────────────────────────────────────────
        for i in 0..k_usize {
            distances[row * k_usize + i] = local_distances[row * k_usize + i];
            indices[row * k_usize + i] = local_indices[row * k_usize + i];
        }
    }
}

/// KNN backward kernel — one GPU thread per row.
///
/// Re-runs the forward KNN pass to recover which neighbours were selected, then
/// propagates gradients from `grad_output` back to `grad_pairwise_distances`.
#[cube(launch)]
pub fn knn_backward_kernel<F: Float + CubePrimitive>(
    pairwise_distances: &Tensor<F>,          // [n, n]
    k: u32,
    local_distances: &mut Tensor<F>,         // [n, k] scratch
    local_indices: &mut Tensor<F>,           // [n, k] scratch  (stored as F for compat)
    grad_output: &Tensor<F>,                 // [n, k] gradient w.r.t. KNN distances
    grad_pairwise_distances: &mut Tensor<F>, // [n, n] gradient output
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = pairwise_distances.shape(0);
    let k_usize = k as usize;

    if row >= n {
        // out of bounds
    } else {
        // ── initialise scratch ──────────────────────────────────────────────
        for i in 0..k_usize {
            local_distances[row * k_usize + i] = F::new(INFINITY);
            local_indices[row * k_usize + i] = F::new(0.0);
        }

        // ── re-run insertion sort to find the k nearest neighbours ───────────
        for col in 0..n {
            if row != col {
                let dist = pairwise_distances[row * n + col];

                if dist < local_distances[row * k_usize + k_usize - 1] {
                    let mut i = k_usize - 1;

                    while i > 0 {
                        if dist < local_distances[row * k_usize + i] {
                            local_distances[row * k_usize + i] =
                                local_distances[row * k_usize + i - 1];
                            local_indices[row * k_usize + i] =
                                local_indices[row * k_usize + i - 1];
                        } else {
                            break;
                        }
                        i -= 1;
                    }

                    local_distances[row * k_usize + i] = dist;
                    local_indices[row * k_usize + i] = F::cast_from(col);
                }
            }
        }

        // ── propagate gradients ──────────────────────────────────────────────
        // Each thread only writes to its own row of grad_pairwise_distances
        // (i.e. positions [row * n + neighbor_col]).  The symmetric entry
        // [neighbor_col * n + row] is owned by thread `neighbor_col` and will be
        // handled there if `row` happens to be one of that thread's k-neighbours.
        // Writing to both here would cause a GPU data race (non-atomic +=).
        let epsilon = F::new(1e-8);
        for i in 0..k_usize {
            let grad_value = grad_output[row * k_usize + i];

            if grad_value != F::new(0.0) {
                let dist = F::max(local_distances[row * k_usize + i], epsilon);
                let grad_pairwise = grad_value / dist;

                let neighbor_col = u32::cast_from(local_indices[row * k_usize + i]) as usize;
                if neighbor_col < n {
                    // Safe: only this thread (row) writes to row's entries.
                    grad_pairwise_distances[row * n + neighbor_col] += grad_pairwise;
                }
            }
        }
    }
}
