use burn::tensor::Tensor;
mod euclidian;
mod kernel;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn euclidean_pairwise_distance(x: Tensor<Self, 2>) -> Tensor<Self, 1>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}
