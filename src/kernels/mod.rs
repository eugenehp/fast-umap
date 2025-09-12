use burn::{
    backend::{autodiff::checkpoint::strategy::CheckpointStrategy, Autodiff},
    tensor::ops::{FloatTensor, IntTensor},
};
// use burn_jit::{FloatElement, IntElement, JitBackend, JitRuntime};
use cubecl::CubeDim;

use crate::backend::Backend;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

mod euclidean;
mod knn;

/// Example cube size
pub const DEFAULT_CUBE_DIM: CubeDim = CubeDim { x: 32, y: 32, z: 1 };

// impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Backend
    for CubeBackend<R, F, I, BT>
{
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean::forward::forward::<R, F, I, BT>(x)
    }

    fn euclidean_pairwise_distance_backward(
        grad_x: FloatTensor<Self>,
        output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // TODO: this is confusing naming, FIXME
        euclidean::forward::backward::<R, F, I, BT>(grad_x, output)
    }

    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        knn::forward::forward::<R, F, I, BT>(pairwise_distances, k)
    }

    fn knn_backward(
        pairwise_distances: FloatTensor<Self>, // Pairwise distance matrix (n, n)
        k: u32,                                // Number of nearest neighbors
        grad_output: FloatTensor<Self>,        // Gradient of the loss w.r.t the output
    ) -> FloatTensor<Self> {
        knn::forward::backward::<R, F, I, BT>(pairwise_distances, k, grad_output)
    }
}

// Forward
// JitBackend -> euclidean_pairwise_distance

// Backward
// Autodiff -> euclidean_pairwise_distance -> JitBackend -> euclidean_pairwise_distance_backward

// TODO: FIXME

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean::backward::backward::<B, C>(x)
    }

    fn euclidean_pairwise_distance_backward(
        _grad_x: FloatTensor<Self>,
        _output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unimplemented!("We trigger this method in `JitBackend` above. Since I didn't find a nicer way to call kernel from the `euclidean_pairwise_distance` in `Autodiff` above.");
    }

    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        // todo!("We need to implement backward kernel for the KNN")
        // (pairwise_distances.clone(), pairwise_distances)
        knn::backward::backward::<B, C>(pairwise_distances, k)
    }

    fn knn_backward(
        _pairwise_distances: FloatTensor<Self>, // Pairwise distance matrix (n, n)
        _k: u32,                                // Number of nearest neighbors
        _grad_output: FloatTensor<Self>,        // Gradient of the loss w.r.t the output
    ) -> FloatTensor<Self> {
        unimplemented!("We trigger this method in `JitBackend` above. Since I didn't find a nicer way to call kernel from the `knn` in `Autodiff` above.");
    }
}
