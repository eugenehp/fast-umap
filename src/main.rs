use burn::{
    backend::{Autodiff, Wgpu},
    prelude::Backend,
    prelude::*,
    tensor::{Device, Tensor},
};
use fast_umap::{
    model::{UMAPModel, UMAPModelConfigBuilder},
    train::{train, TrainingConfig},
};
use rand::Rng;

// Define the function to generate random data in the format `Vec<Tensor<B, 2>>`.
fn load_data<B: Backend>(
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device<B>,  // Device to place the tensor (CPU, GPU)
) -> Vec<Tensor<B, 2>> {
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        // Generate random data for the tensor (size = num_features)
        let data: Vec<_> = (0..num_features)
            .map(|_| rng.gen::<f64>()) // Random number generation for each feature
            .collect();

        let tensor_data = TensorData::new(data, [1, num_features]);

        // Create a Tensor with the shape [1, num_features] (1 row, num_features columns)
        let tensor = Tensor::<B, 2>::from_data(tensor_data, device);

        result.push(tensor);
    }

    result
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let num_samples = 3;
    let num_features = 10;
    let train_data = load_data(num_samples, num_features, &device);

    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_size(100)
        .output_size(2)
        .build()
        .unwrap();

    let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);

    let config = TrainingConfig::<MyAutodiffBackend>::builder()
        .epochs(100)
        .batch_size(32)
        .learning_rate(0.001)
        .device(device) // Using GPU (CUDA) or Device::cpu() for CPU
        .beta1(0.9)
        .beta2(0.999)
        .build()
        .expect("Failed to build TrainingConfig"); // Expecting a valid config

    // Start training with the configured parameters
    train::<MyAutodiffBackend>(model, &train_data, &config);
}
