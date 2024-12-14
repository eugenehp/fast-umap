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
    utils::{load_test_data, print_tensor},
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let batch_size = 1;
    let num_samples = 10;
    let num_features = 3;
    let output_size = 2;

    let train_data = load_test_data::<MyAutodiffBackend>(num_samples, num_features, &device);

    print_tensor(Some("Generated data"), &train_data);

    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_size(100)
        .output_size(output_size)
        .build()
        .unwrap();

    let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);

    let config = TrainingConfig::<MyAutodiffBackend>::builder()
        .epochs(100)
        .batch_size(batch_size)
        .learning_rate(0.001)
        .device(device)
        .beta1(0.9)
        .beta2(0.999)
        .build()
        .expect("Failed to build TrainingConfig");

    // Start training with the configured parameters
    train::<MyAutodiffBackend>(model, train_data, &config);
}
