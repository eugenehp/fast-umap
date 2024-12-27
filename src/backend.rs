use burn::{
    backend::Autodiff,
    tensor::ops::{FloatTensor, IntTensor},
};
use burn_jit::JitBackend;
use cubecl::wgpu::WgpuRuntime;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self>;
    fn euclidean_pairwise_distance_backward(
        grad_x: FloatTensor<Self>,
        output: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    // TODO: return IntTensor for indices
    /// Returns indices, distances
    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>);

    fn knn_backward(
        pairwise_distances: FloatTensor<Self>, // Pairwise distance matrix (n, n)
        k: u32,                                // Number of nearest neighbors
        grad_output: FloatTensor<Self>,        // Gradient of the loss w.r.t the output
    ) -> FloatTensor<Self>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

// this line along with the `backward` module is what's needed to enable support for a particular device below
impl AutodiffBackend for Autodiff<JitBackend<WgpuRuntime, f32, i32>> {}
