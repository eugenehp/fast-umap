use burn::{
    backend::{autodiff::checkpoint::strategy::CheckpointStrategy, Autodiff},
    tensor::ops::{FloatTensor, IntTensor},
};
use burn_jit::{FloatElement, IntElement, JitBackend, JitRuntime};
use cubecl::CubeDim;
use kernels::{knn_backward_launch, knn_pairwise_euclidean_launch};

use crate::backend::Backend;
mod kernels;

impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
    fn knn(
        input: FloatTensor<Self>,
        k: u32,
        min_dist: f32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        knn_pairwise_euclidean_launch(input, k, min_dist)
    }
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn knn(
        input: FloatTensor<Self>,
        k: u32,
        min_dist: f32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
    }
}
