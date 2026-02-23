use burn::{
    backend::{autodiff::checkpoint::strategy::CheckpointStrategy, Autodiff},
    tensor::ops::{FloatTensor, IntTensor},
};
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

/// Autodiff implementation of the fast-umap [`Backend`] trait.
///
/// This wrapper records the computation graph so that gradients flow back
/// through the custom kernels during `loss.backward()`.
///
/// * `euclidean_pairwise_distance` — delegates to [`euclidean::backward`],
///   which checkpoints the forward input and registers the backward hook.
/// * `knn` — delegates to [`knn::backward`], which checkpoints the pairwise
///   distance matrix and registers the backward hook.
/// * The `*_backward` variants are only called on the inner [`CubeBackend`]
///   and are therefore `unimplemented!()` here.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    /// Register the Euclidean pairwise distance operation in the autodiff graph.
    ///
    /// The actual forward computation runs on the inner [`CubeBackend`]; this
    /// layer only hooks up the backward pass.
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean::backward::backward::<B, C>(x)
    }

    /// Not called on the [`Autodiff`] wrapper — only on the inner backend.
    fn euclidean_pairwise_distance_backward(
        _grad_pairwise: FloatTensor<Self>,
        _x: FloatTensor<Self>,
        _pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unimplemented!(
            "Called on inner CubeBackend only; Autodiff dispatches via euclidean::backward."
        );
    }

    /// Register the k-NN operation in the autodiff graph.
    ///
    /// The actual forward computation runs on the inner [`CubeBackend`]; this
    /// layer only hooks up the backward pass.
    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        knn::backward::backward::<B, C>(pairwise_distances, k)
    }

    /// Not called on the [`Autodiff`] wrapper — only on the inner backend.
    fn knn_backward(
        _pairwise_distances: FloatTensor<Self>,
        _k: u32,
        _grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unimplemented!(
            "Triggered on the inner CubeBackend only; \
             the Autodiff wrapper delegates via knn::backward."
        );
    }
}
