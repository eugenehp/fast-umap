use burn::{module::*, prelude::*};
use crossbeam_channel::unbounded;
use cubecl::wgpu::WgpuRuntime;
use fast_umap::{chart, model::*, prelude::*, train::train, utils::*};

fn main() {
    let (exit_tx, exit_rx) = unbounded();

    ctrlc::set_handler(move || exit_tx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    type F = f32;
    // Define a custom backend type using Wgpu with 32-bit floating point precision and 32-bit integer type
    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, F, i32, u32>;

    // Define the AutodiffBackend based on the custom MyBackend type
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    // Initialize the GPU device for computation
    // let device = burn::backend::wgpu::WgpuDevice::default();
    let device = Default::default();

    // Set training hyperparameters
    let batch_size = 1000; // Number of samples per batch during training
    let num_samples = 1000; // Total number of samples in the dataset
    let num_features = 100; // Number of features (dimensions) for each sample
    let k_neighbors = 10; // Number of nearest neighbors for the UMAP algorithm
    let output_size = 2; // Number of output dimensions (e.g., 2D for embeddings)
    let hidden_sizes = vec![100, 100, 100]; // Size of the hidden layer in the neural network
    let learning_rate = 0.001; // Learning rate for optimization
    let beta1 = 0.9; // Beta1 parameter for the Adam optimizer
    let beta2 = 0.999; // Beta2 parameter for the Adam optimizer

    // let epochs = 400; // Number of training epochs
    let epochs = 100; // Number of training epochs
    let seed = 9999; // Random seed to ensure reproducibility
    let verbose = true; // Whether to enable the progress bar during training

    // let patience = 10; // Number of epochs without improvement before early stopping
    let min_desired_loss = 0.001; // Minimum loss threshold for early stopping
    let timeout = 60;

    // let metric = "euclidean_knn"; // Distance metric used for the nearest neighbor search
    let metric = Metric::Euclidean; // Alternative metric for neighbors search

    // Seed the random number generator to ensure reproducibility
    MyAutodiffBackend::seed(seed);

    // Generate random test data for training
    let train_data: Vec<F> = generate_test_data(num_samples, num_features);

    // Configure the UMAP model with the specified input size, hidden layer size, and output size
    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_sizes(hidden_sizes)
        .output_size(output_size)
        .build()
        .unwrap();

    // Initialize the UMAP model with the defined configuration and the selected device
    // let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);
    let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);

    // Set up the training configuration with the specified hyperparameters
    let config = TrainingConfig::builder()
        .with_epochs(epochs) // Set the number of epochs for training
        .with_batch_size(batch_size) // Set the batch size for training
        .with_learning_rate(learning_rate) // Set the learning rate for the optimizer
        .with_beta1(beta1) // Set the beta1 parameter for the Adam optimizer
        .with_beta2(beta2) // Set the beta2 parameter for the Adam optimizer
        .with_verbose(verbose) // Enable or disable the progress bar
        // .with_patience(patience) // Set the patience for early stopping
        .with_metric(metric.into()) // Set the metric for nearest neighbors (e.g., Euclidean)
        .with_k_neighbors(k_neighbors) // Set the number of neighbors to consider for UMAP
        .with_min_desired_loss(min_desired_loss) // Set the minimum desired loss for early stopping
        .with_timeout(timeout) // set timeout in seconds
        .build()
        .expect("Failed to build TrainingConfig");

    // Start training the UMAP model with the specified training data and configuration
    let (model, _, _) = train(
        "advanced",
        model,              // The model to train
        num_samples,        // Total number of training samples
        num_features,       // Number of features per sample
        train_data.clone(), // The training data
        &config,            // The training configuration
        device.clone(),
        exit_rx,
    );

    // Validate the trained model after training
    let model = model.valid();

    // Convert the training data into a tensor for model input
    let global = convert_vector_to_tensor(train_data, num_samples, num_features, &device);

    // Perform a forward pass through the model to obtain the low-dimensional (local) representation
    let local = model.forward(global.clone());

    // Optionally, print the global and local tensors for inspection (currently commented out)
    // if verbose {
    //     print_tensor_with_title("global", &global);
    //     print_tensor_with_title("local", &local);
    // }

    // Visualize the 2D embedding (local representation) using a chart
    chart::chart_tensor(local, None, None);
}
