use burn::module::AutodiffModule;

use burn::{
    backend::{Autodiff, Wgpu},
    prelude::*,
};
use fast_umap::{
    chart,
    model::{UMAPModel, UMAPModelConfigBuilder},
    prelude::*,
    train::train,
    utils::*,
};

fn main() {
    // Define the custom backend type using Wgpu with specific precision (f32) and integer type (i32)
    type MyBackend = Wgpu<f32, i32>;

    // Define the AutodiffBackend based on the custom MyBackend type
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // Initialize the device (GPU) for computation
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Set the training parameters
    let batch_size = 1; // Batch size for training
    let num_samples = 1000; //1000; // Number of samples in the dataset
    let num_features = 100; // 100; // Number of features (dimensions) for each sample
    let k_neighbors = 10; // should be smaller than k_neighbors
    let output_size = 2; // Number of output dimensions (e.g., 2 for 2D embeddings)
    let hidden_size = 100; // Size of the hidden layer in the neural network
    let learning_rate = 0.001; // Learning rate for optimization
    let beta1 = 0.9; // Beta1 parameter for Adam optimizer
    let beta2 = 0.999; // Beta2 parameter for Adam optimizer
    let epochs = 400; // Number of training epochs
    let seed = 9999; // Random seed for reproducibility
    let verbose = true; // Enables the progress bar for training
    let patience = 50; // Number of epochs with no improvement before stopping early
    let min_desired_loss = 0.001;
    let metric = Metric::Euclidean; // this also works
                                    // let metric = Metric::EuclideanKNN; // this also works
                                    // let metric = "euclidean_knn";

    // Seed the random number generator for reproducibility
    MyBackend::seed(seed);

    // Generate random test data for training
    let train_data = generate_test_data(num_samples, num_features);

    // Configure the model by setting input size, hidden size, and output size
    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_size(hidden_size)
        .output_size(output_size)
        .build()
        .unwrap();

    // Initialize the UMAP model with the specified configuration and device
    let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);

    // Set up the training configuration with the specified parameters
    let config = TrainingConfig::<MyAutodiffBackend>::builder()
        .with_epochs(epochs) // Set the number of training epochs
        .with_batch_size(batch_size) // Set the batch size
        .with_learning_rate(learning_rate) // Set the learning rate
        .with_device(device) // Set the computation device (GPU)
        .with_beta1(beta1) // Set the beta1 parameter for Adam optimizer
        .with_beta2(beta2) // Set the beta2 parameter for Adam optimizer
        .with_verbose(verbose) // Enable or disable the progress bar
        .with_patience(patience)
        .with_metric(metric.into())
        .with_k_neighbors(k_neighbors)
        .with_min_desired_loss(min_desired_loss)
        .build()
        .expect("Failed to build TrainingConfig");

    // Start training the model with the training data and configuration
    let model = train::<MyAutodiffBackend>(
        model,              // The model to train
        num_samples,        // Number of samples in the dataset
        num_features,       // Number of features per sample
        train_data.clone(), // The training data
        &config,            // The training configuration
    );

    // Validate the trained model
    let (model, _) = model.valid();

    // Convert the training data into a tensor for input to the model
    let global = convert_vector_to_tensor(train_data, num_samples, num_features, &config.device);

    // Perform the forward pass to get the low-dimensional (local) representation
    let local = model.forward(global.clone());

    // Optionally, print the global and local tensors for inspection (commented-out for now)
    // if verbose {
    //     print_tensor_with_title("global", &global);
    //     print_tensor_with_title("local", &local);
    // }

    // Visualize the reduced dimensions (2D embedding) using a chart
    chart::chart_tensor(local, None);
}
