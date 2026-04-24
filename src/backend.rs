use burn::{
    backend::Autodiff,
    tensor::ops::{FloatTensor, IntTensor},
};
#[cfg(feature = "gpu")]
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

/// Extension of the Burn [`Backend`](burn::tensor::backend::Backend) trait that
/// adds the custom GPU kernels required by fast-umap.
///
/// Two kernels are exposed:
///
/// * **Euclidean pairwise distance** — computes the full `[n, n]` symmetric
///   distance matrix for an `[n, d]` input; differentiable via a custom
///   backward kernel.
/// * **k-NN** — selects the `k` nearest neighbours for every row of a
///   pairwise distance matrix; differentiable (backward propagates through
///   the selected entries).
///
/// Concrete implementations are provided for [`CubeBackend`] (GPU) and
/// [`Autodiff<B>`] (automatic differentiation wrapper).
pub trait Backend: burn::tensor::backend::Backend {
    /// Whether to use `gather` instead of `select` for row indexing.
    ///
    /// Some backends (e.g. MLX) have buggy `select_add` in their backward pass.
    /// When `true`, the training loop uses `gather` (correct everywhere but
    /// slightly slower) instead of `select` (faster but broken on some backends).
    const USE_GATHER_FOR_SELECT: bool = false;

    /// Compute the full `[n, n]` symmetric Euclidean pairwise distance matrix
    /// for an input tensor of shape `[n, d]`.
    ///
    /// The result satisfies `output[i, i] = 0` and
    /// `output[i, j] = ||x[i] − x[j]||₂` for `i ≠ j`.
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self>;

    /// Backward pass for [`euclidean_pairwise_distance`](Self::euclidean_pairwise_distance).
    ///
    /// Computes `∂loss/∂x` given:
    ///
    /// * `grad_pairwise` — upstream gradient `∂loss/∂pairwise_distances`, shape `[n, n]`
    /// * `x`             — original input saved from the forward pass, shape `[n, d]`
    /// * `pairwise`      — precomputed distance matrix saved from the forward pass, shape `[n, n]`
    ///
    /// Returns `∂loss/∂x`, shape `[n, d]`.
    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Select the `k` nearest neighbours for every row of `pairwise_distances`.
    ///
    /// Returns `(indices, distances)`:
    /// * `indices`   — `[n, k]` integer tensor of neighbour column indices
    /// * `distances` — `[n, k]` float tensor of corresponding distances
    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>);

    /// Backward pass for [`knn`](Self::knn).
    ///
    /// Re-runs the forward k-NN pass to recover which neighbours were selected,
    /// then propagates `grad_output` (shape `[n, k]`) back to a gradient with
    /// respect to `pairwise_distances` (shape `[n, n]`).
    ///
    /// Only called on the inner [`CubeBackend`]; the [`Autodiff`] wrapper
    /// delegates via `euclidean::backward`.
    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// Marker trait combining the fast-umap [`Backend`] extension with Burn's own
/// [`AutodiffBackend`](burn::tensor::backend::AutodiffBackend).
///
/// Implement this trait for any backend that should support end-to-end
/// gradient flow through the custom kernels.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

/// Blanket implementation of [`AutodiffBackend`] for any [`CubeBackend`]
/// wrapped in [`Autodiff`].
///
/// This covers the primary GPU training path:
/// `Autodiff<CubeBackend<WgpuRuntime, f32, i32, u32>>`.
#[cfg(feature = "gpu")]
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> AutodiffBackend
    for Autodiff<CubeBackend<R, F, I, BT>>
{
}

/// Blanket implementation of [`AutodiffBackend`] for [`burn_mlx::Mlx`]
/// wrapped in [`Autodiff`].
///
/// This covers the MLX (Apple Silicon) training path:
/// `Autodiff<burn_mlx::Mlx>`.
#[cfg(feature = "mlx")]
impl AutodiffBackend for Autodiff<burn_mlx::Mlx> {}

// TODO: Candle backend — blocked on burn-candle/candle-core DType compat
// #[cfg(feature = "candle")]
// impl<F: burn_candle::element::FloatCandleElement, I: burn_candle::element::IntCandleElement>
//     AutodiffBackend for Autodiff<burn_candle::Candle<F, I>>
// {
// }
