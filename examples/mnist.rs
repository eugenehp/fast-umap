use burn::{backend::*, module::*, prelude::*};
use crossbeam_channel::unbounded;
use fast_umap::chart;
#[allow(unused)]
use fast_umap::{
    chart::*,
    model::*,
    prelude::*,
    train::{train, LossReduction},
    utils::*,
};
use mnist::*;
use wgpu::WgpuRuntime;

fn main() {
    let (exit_tx, exit_rx) = unbounded();

    ctrlc::set_handler(move || exit_tx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    // Define a custom backend type using Wgpu with 32-bit floating point precision and 32-bit integer type
    type MyBackend = burn::backend::wgpu::JitBackend<WgpuRuntime, f32, i32>;

    // Define the AutodiffBackend based on the custom MyBackend type
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    // Initialize the GPU device for computation
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Set training hyperparameters
    let batch_size = 1_000; // Number of samples per batch during training
    let num_samples = 10_000 as usize; // Total number of samples in the dataset

    // let num_samples = 50_000 as usize; // Total number of samples in the dataset

    let num_features = 28 * 28; // Number of features (dimensions) for each sample, size of each mnist image
    let k_neighbors = 15; // Number of nearest neighbors for the UMAP algorithm
    let output_size = 2; // Number of output dimensions (e.g., 2D for embeddings)
    let hidden_sizes = vec![1000]; // Size of the hidden layer in the neural network
    let learning_rate = 1e-4; // Learning rate for optimization
    let penalty = 1e-6; // penalty for the Adam optimizer
    let beta1 = 0.9; // Beta1 parameter for the Adam optimizer
    let beta2 = 0.999; // Beta2 parameter for the Adam optimizer
    let epochs = 1_000; // Number of training epochs
    let seed = 9999; // Random seed to ensure reproducibility
    let verbose = true; // Whether to enable the progress bar during training

    // let patience = 100; // Number of epochs without improvement before early stopping
    let min_desired_loss = 1e-4; // Minimum loss threshold for early stopping
    let metric = Metric::Euclidean; // Alternative metric for neighbors search
    let loss_reduction = LossReduction::Mean;
    // below 1.0 gives NaN loss. Mind that it's rounded to integer inside the function
    // let minkowski_p = 3.0; // 1 is manhattan, 2 is Euclidean
    let normalized = true; // to reduce math, and keep it at float

    // let timeout = 30; // timeout in seconds

    // Seed the random number generator to ensure reproducibility
    MyBackend::seed(seed);

    let Mnist {
        trn_img,
        trn_lbl,
        // tst_img,
        // tst_lbl,
        ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(num_samples as u32)
        // .validation_set_length(10_000)
        // .test_set_length(10_000)
        .finalize();

    // Generate random test data for training
    // let train_data = generate_test_data(num_samples, num_features);
    let train_data: Vec<f64> = trn_img.into_iter().map(|byte| byte as f64).collect();

    // Configure the UMAP model with the specified input size, hidden layer size, and output size
    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_sizes(hidden_sizes)
        .output_size(output_size)
        .build()
        .unwrap();

    // Initialize the UMAP model with the defined configuration and the selected device
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
        .with_loss_reduction(loss_reduction)
        // .with_timeout(timeout) // set timeout in seconds
        // .with_minkowski_p(minkowski_p)
        .with_normalized(normalized)
        .with_penalty(penalty)
        .build()
        .expect("Failed to build TrainingConfig");

    // Start training the UMAP model with the specified training data and configuration
    let (model, _, _) = train(
        "mnist",
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

    let chart_config = ChartConfigBuilder::default()
        .caption("MNIST")
        .path("mnist.png")
        .build();

    let labels: Vec<String> = trn_lbl.iter().map(|digit| format!("{digit}")).collect();

    // Visualize the 2D embedding (local representation) using a chart
    chart::chart_tensor(local, Some(labels), Some(chart_config));
}
