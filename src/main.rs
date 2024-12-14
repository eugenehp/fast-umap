use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use fast_umap::model::{UMAPModel, UMAPModelConfigBuilder};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let config = UMAPModelConfigBuilder::default()
        .input_size(100)
        .hidden_size(100)
        .output_size(2)
        .build()
        .unwrap();

    let model: UMAPModel<MyBackend> = UMAPModel::new(&config, &device);

    println!("hello");
}
