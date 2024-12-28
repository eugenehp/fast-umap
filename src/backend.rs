use burn::{
    backend::Autodiff,
    tensor::ops::{FloatTensor, IntTensor},
};
use burn_jit::JitBackend;
use cubecl::wgpu::WgpuRuntime;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn knn(input: FloatTensor<Self>, k: u32, min_dist: f32)
        -> (IntTensor<Self>, FloatTensor<Self>);
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

// this line along with the `backward` module is what's needed to enable support for a particular device below
impl AutodiffBackend for Autodiff<JitBackend<WgpuRuntime, f32, i32>> {}
