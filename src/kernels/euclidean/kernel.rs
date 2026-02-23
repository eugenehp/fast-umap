use cubecl::{cube, prelude::*};

/// Forward kernel — one thread per (row, col) upper-triangle pair.
///
/// Computes `||x[row] - x[col]||_2` and stores it (symmetrically) in `output`.
#[cube(launch)]
pub fn euclidean_pairwise_distance_kernel<F: Float>(
    x: &Tensor<F>,          // [n, d]
    output: &mut Tensor<F>, // [n, n]
) {
    let row = ABSOLUTE_POS_X as usize;
    let col = ABSOLUTE_POS_Y as usize;

    let n = x.shape(0);
    let d = x.shape(1);

    if row >= n || col >= n || row > col {
        // no-op — lower triangle and out-of-range threads
    } else if row == col {
        output[row * n + col] = F::new(0.0);
    } else {
        let mut sum = F::new(0.0);
        for i in 0..d {
            let diff = x[row * d + i] - x[col * d + i];
            sum += diff * diff;
        }
        let dist = F::sqrt(sum);
        output[row * n + col] = dist;
        output[col * n + row] = dist; // symmetry
    }
}

/// Backward kernel — one thread per (row, feat) element of grad_x.
///
/// `ABSOLUTE_POS_X` = sample row index  (0 .. n)
/// `ABSOLUTE_POS_Y` = feature index     (0 .. d)
///
/// Each thread accumulates contributions from every other sample `col` using
/// the *precomputed* pairwise distance (no inner d-loop needed), then writes
/// the result with `=` (no accumulation → no cross-thread races).
///
/// Total work: n·d threads × n iterations each = O(n²d) — same as forward.
#[cube(launch)]
pub fn euclidean_pairwise_distance_backward_kernel<F: Float>(
    x: &Tensor<F>,           // [n, d]  original input
    pairwise: &Tensor<F>,    // [n, n]  precomputed distances (from forward state)
    grad_pairwise: &Tensor<F>, // [n, n]  ∂loss/∂pairwise_distances
    grad_x: &mut Tensor<F>,  // [n, d]  ∂loss/∂x  — written by this kernel
) {
    let row = ABSOLUTE_POS_X as usize; // sample index
    let feat = ABSOLUTE_POS_Y as usize; // feature index

    let n = pairwise.shape(0);
    let d = x.shape(1);
    let epsilon = F::new(1e-8);

    if row >= n || feat >= d {
        // out-of-range thread
    } else {
        let x_row_feat = x[row * d + feat];
        let mut grad_sum = F::new(0.0);

        for col in 0..n {
            if col != row {
                let dist = F::max(pairwise[row * n + col], epsilon);

                // Gradient from pairwise[row, col]:
                //   ∂||x_row - x_col|| / ∂x[row, feat] = (x[row,feat] - x[col,feat]) / dist
                // Gradient from pairwise[col, row]  (symmetric entry, same distance):
                //   same formula — both symmetric entries carry the same gradient factor
                let g_rc = grad_pairwise[row * n + col];
                let g_cr = grad_pairwise[col * n + row];

                let diff = x_row_feat - x[col * d + feat];
                grad_sum += (g_rc + g_cr) * diff / dist;
            }
        }

        // Each (row, feat) pair is owned by exactly one thread — safe plain write.
        grad_x[row * d + feat] = grad_sum;
    }
}
