use burn::{backend::Autodiff, tensor::ops::FloatTensor};
use burn_jit::JitBackend;
use cubecl::wgpu::WgpuRuntime;
mod backward;
mod forward;
mod kernel;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {
    // fn euclidean_pairwise_distance(x: Tensor<Self, 2>) -> Tensor<Self, 1>;
}

impl AutodiffBackend for Autodiff<JitBackend<WgpuRuntime, f32, i32>> {}
