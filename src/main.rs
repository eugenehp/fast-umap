#[allow(unused_imports)]
use burn::{
    backend::{Autodiff, Wgpu},
    prelude::Backend,
    prelude::*,
    tensor::{Device, Tensor},
};
use fast_umap::{
    model::{UMAPModel, UMAPModelConfigBuilder},
    train::{train, TrainingConfig},
    utils::generate_test_data,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let batch_size = 1;
    let num_samples = 100;
    let num_features = 10;
    let output_size = 2;
    let hidden_size = 100;
    let learning_rate = 0.001;
    let epochs = 100;

    MyBackend::seed(9999);

    let train_data = generate_test_data(num_samples, num_features);

    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_size(hidden_size)
        .output_size(output_size)
        .build()
        .unwrap();

    let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);
    println!("{}", model);

    let config = TrainingConfig::<MyAutodiffBackend>::builder()
        .epochs(epochs)
        .batch_size(batch_size)
        .learning_rate(learning_rate)
        .device(device)
        .beta1(0.9)
        .beta2(0.999)
        .build()
        .expect("Failed to build TrainingConfig");

    // Start training with the configured parameters
    train::<MyAutodiffBackend>(model, num_samples, num_features, train_data, &config);
}
