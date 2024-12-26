use burn::{
    backend::{autodiff::checkpoint::strategy::CheckpointStrategy, Autodiff},
    tensor::ops::FloatTensor,
};
use burn_jit::{FloatElement, IntElement, JitBackend, JitRuntime};
use cubecl::CubeDim;

use crate::backend::Backend;

mod euclidean;
mod knn;

/// Example cube size
pub const DEFAULT_CUBE_DIM: CubeDim = CubeDim { x: 32, y: 32, z: 1 };

impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean::forward::forward::<R, F, I>(x)
    }

    fn euclidean_pairwise_distance_backward(
        grad_x: FloatTensor<Self>,
        output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // TODO: this is confusing naming, FIXME
        euclidean::forward::backward::<R, F, I>(grad_x, output)
    }

    fn knn(x: FloatTensor<Self>, k: u32) -> (FloatTensor<Self>, FloatTensor<Self>) {
        todo!()
    }
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean::backward::backward::<B, C>(x)
    }

    fn euclidean_pairwise_distance_backward(
        grad_x: FloatTensor<Self>,
        output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn knn(x: FloatTensor<Self>, k: u32) -> (FloatTensor<Self>, FloatTensor<Self>) {
        unimplemented!();
    }
}
