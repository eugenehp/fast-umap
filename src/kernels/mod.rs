use burn::tensor::ops::{FloatTensor, IntTensor};
use cubecl::CubeDim;

use crate::backend::Backend;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

mod euclidean;
mod knn;

/// Default work-group (cube) dimensions used by all custom CubeCL kernels.
///
/// A 32 × 32 × 1 cube means up to 1 024 threads share a work-group.  This
/// matches the typical warp/wavefront size on WGPU targets (Metal, Vulkan,
/// DX12) and maximises occupancy for the matrix-shaped dispatch patterns used
/// by the Euclidean and k-NN kernels.
pub const DEFAULT_CUBE_DIM: CubeDim = CubeDim { x: 32, y: 32, z: 1 };

/// GPU implementation of the fast-umap [`Backend`] trait for [`CubeBackend`].
///
/// Each method dispatches to the corresponding CubeCL kernel in the
/// [`euclidean`] or [`knn`] sub-modules.
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Backend
    for CubeBackend<R, F, I, BT>
{
    /// Launch the Euclidean pairwise distance forward kernel.
    ///
    /// Computes the full `[n, n]` symmetric distance matrix for an `[n, d]`
    /// input using one GPU thread per `(row, col)` upper-triangle pair.
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean::forward::forward::<R, F, I, BT>(x)
    }

    /// Launch the Euclidean pairwise distance backward kernel.
    ///
    /// Dispatches one GPU thread per `(row, feature)` element of `grad_x`.
    /// Each thread iterates over all `n` columns to accumulate gradient
    /// contributions from both symmetric entries `pairwise[row, col]` and
    /// `pairwise[col, row]`.
    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        euclidean::forward::backward::<R, F, I, BT>(grad_pairwise, x, pairwise)
    }

    /// Launch the k-NN forward kernel.
    ///
    /// Selects the `k` nearest neighbours for every row using an
    /// insertion-sort approach — one GPU thread per row.
    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        knn::forward::forward::<R, F, I, BT>(pairwise_distances, k)
    }

    /// Launch the k-NN backward kernel.
    ///
    /// Re-runs the forward sort to recover the selected neighbours, then
    /// propagates `grad_output` (shape `[n, k]`) back to
    /// `grad_pairwise_distances` (shape `[n, n]`).
    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        knn::forward::backward::<R, F, I, BT>(pairwise_distances, k, grad_output)
    }
}

// Note: The Autodiff<B, C> impl of Backend has been moved to
// src/autodiff_ops.rs so it is available for all backends (MLX, WGPU, etc.),
// not just CubeBackend.
