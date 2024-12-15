use burn::module::AutodiffModule;
#[allow(unused_imports)]
use burn::{
    backend::{Autodiff, Wgpu},
    prelude::Backend,
    prelude::*,
    tensor::{Device, Tensor},
};
use fast_umap::{
    chart,
    model::{UMAPModel, UMAPModelConfigBuilder},
    train::{train, TrainingConfig},
    utils::*,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let batch_size = 1;
    let num_samples = 1000;
    let num_features = 100;
    let output_size = 2;
    let hidden_size = 100;
    let learning_rate = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epochs = 400;
    let seed = 9999;
    let verbose = true; // enables the progress bar

    MyBackend::seed(seed);

    let train_data = generate_test_data(num_samples, num_features);

    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_size(hidden_size)
        .output_size(output_size)
        .build()
        .unwrap();

    let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);
    // println!("{}", model);

    let config = TrainingConfig::<MyAutodiffBackend>::builder()
        .with_epochs(epochs)
        .with_batch_size(batch_size)
        .with_learning_rate(learning_rate)
        .with_device(device)
        .with_beta1(beta1)
        .with_beta2(beta2)
        .with_verbose(verbose)
        .build()
        .expect("Failed to build TrainingConfig");

    // Start training with the configured parameters
    let model = train::<MyAutodiffBackend>(
        model,
        num_samples,
        num_features,
        train_data.clone(),
        &config,
    );

    let model = model.valid();
    let global = convert_vector_to_tensor(train_data, num_samples, num_features, &config.device);
    let local = model.forward(global.clone());

    // if verbose {
    //     print_tensor_with_title("global", &global);
    //     print_tensor_with_title("local", &local);
    // }
    chart::chart_tensor(local, None);
}
