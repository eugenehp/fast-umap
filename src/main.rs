use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use fast_umap::{
    model::{UMAPModel, UMAPModelConfigBuilder},
    train::TrainingConfig,
};

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

    let config = TrainingConfig::builder()
        .epochs(100)
        .batch_size(32)
        .learning_rate(0.001)
        .device(device) // Using GPU (CUDA) or Device::cpu() for CPU
        .beta1(0.9)
        .beta2(0.999)
        .build()
        .expect("Failed to build TrainingConfig"); // Expecting a valid config

    // Load your training data (make sure it's in the correct format)
    let train_data: Vec<Tensor<B, 2>> = load_data(); // Replace with your actual data loading logic

    // Start training with the configured parameters
    train(model, &train_data, &config);

    println!("hello");
}
