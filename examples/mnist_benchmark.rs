use burn::{backend::*, module::AutodiffModule as _, prelude::*};
use fast_umap::{backend::AutodiffBackend, chart};
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

fn generate_model_name(
    prefix: &str,
    learning_rate: f64,
    batch_size: usize,
    penalty: f64,
    hidden_sizes: &Vec<usize>,
    epochs: usize,
) -> String {
    let hidden_sizes = hidden_sizes
        .iter()
        .map(|size| format!("{size}"))
        .collect::<Vec<String>>()
        .join("_");
    format!(
        "{prefix}_lr_{:.0e}_bs_{:04}_pen_{:.0e}_hs_{hidden_sizes}_ep_{epochs}",
        learning_rate, batch_size, penalty
    )
}

fn execute<B: AutodiffBackend>(
    name: String,
    num_features: usize,
    num_samples: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
    device: Device<B>,
    train_data: Vec<f64>,
    labels: Vec<String>,
    config: TrainingConfig,
) -> f64 {
    // Configure the UMAP model with the specified input size, hidden layer size, and output size
    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_sizes(hidden_sizes)
        .output_size(output_size)
        .build()
        .unwrap();

    // Initialize the UMAP model with the defined configuration and the selected device
    let model: UMAPModel<B> = UMAPModel::new(&model_config, &device);

    // Start training the UMAP model with the specified training data and configuration
    let (model, losses) = train(
        name.as_str(),
        model,              // The model to train
        num_samples,        // Total number of training samples
        num_features,       // Number of features per sample
        train_data.clone(), // The training data
        &config,            // The training configuration
        device.clone(),
    );

    // Validate the trained model after training
    let model = model.valid();

    // Convert the training data into a tensor for model input
    let global = convert_vector_to_tensor(train_data, num_samples, num_features, &device);

    // Perform a forward pass through the model to obtain the low-dimensional (local) representation
    let local = model.forward(global.clone());

    let chart_config = ChartConfigBuilder::default()
        .caption(name.as_str())
        .path(format!("{name}.png").as_str())
        .build();

    // Visualize the 2D embedding (local representation) using a chart
    chart::chart_tensor(local, Some(labels), Some(chart_config));

    let min_loss = losses
        .into_iter()
        .filter(|loss| !loss.is_nan()) // TODO: check the kernels to prevent NaN in the loss!
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    min_loss
}

fn find_best_hyperparameters<B: AutodiffBackend>(
    num_features: usize,
    num_samples: usize,
    train_data: Vec<f64>,
    labels: Vec<String>,
    device: Device<B>,
    config: TrainingConfig,
    learning_rates: Vec<f64>,
    batch_sizes: Vec<usize>,
    penalties: Vec<f64>,
    hidden_size_options: Vec<Vec<usize>>,
    epochs_options: Vec<usize>, // Added epochs as an array of values
) -> (String, f64) {
    let mut best_loss = f64::MAX;
    let mut best_config = String::new();

    // let (exit_tx, exit_rx) = channel();

    // ctrlc::set_handler(move || exit_tx.send(()).expect("Could not send signal on channel."))
    //     .expect("Error setting Ctrl-C handler");

    // Iterate over all combinations of the hyperparameters
    for &learning_rate in &learning_rates {
        for &batch_size in &batch_sizes {
            for &penalty in &penalties {
                for hidden_sizes in &hidden_size_options {
                    for &epochs in &epochs_options {
                        // Generate a model name for the current combination
                        let model_name = generate_model_name(
                            "mnist",
                            learning_rate,
                            batch_size,
                            penalty,
                            hidden_sizes,
                            epochs,
                        );

                        println!("{model_name}");

                        // Modify the configuration with the current batch_size and epochs
                        let mut current_config = config.clone(); // Copy the config for each iteration
                        current_config.batch_size = batch_size;
                        current_config.epochs = epochs;

                        // Execute training and get the loss for this configuration
                        let loss = execute::<B>(
                            model_name.clone(),
                            num_features,
                            num_samples,
                            hidden_sizes.clone(),
                            2, // output size (2D for UMAP)
                            device.clone(),
                            train_data.clone(),
                            labels.clone(),
                            current_config,
                        );

                        // Update the best configuration if the current loss is smaller
                        if loss < best_loss {
                            best_loss = loss;
                            best_config = model_name;
                        }
                    }
                }
            }
        }
    }

    (best_config, best_loss)
}

// Example usage in main:

fn main() {
    // Define a custom backend type using Wgpu with 32-bit floating point precision and 32-bit integer type
    type MyBackend = burn::backend::wgpu::JitBackend<WgpuRuntime, f32, i32>;

    // Define the AutodiffBackend based on the custom MyBackend type
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    // Initialize the GPU device for computation
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Set training parameters and configuration
    let num_samples = 10_000 as usize;
    let num_features = 28 * 28;
    let k_neighbors = 15;
    let learning_rate = 1e-4;
    let penalty = 1e-6;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let seed = 9999;
    let verbose = true;
    let min_desired_loss = 1e-4;
    let metric = Metric::Euclidean;
    let loss_reduction = LossReduction::Mean;
    let normalized = true;

    // Seed the random number generator to ensure reproducibility
    MyBackend::seed(seed);

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(num_samples as u32)
        .finalize();

    let train_data: Vec<f64> = trn_img.into_iter().map(|byte| byte as f64).collect();
    let labels: Vec<String> = trn_lbl.iter().map(|digit| format!("{digit}")).collect();

    // Set up the training configuration with the specified hyperparameters
    let config = TrainingConfig::builder() // This is a default value; will be overridden in the search
        .with_learning_rate(learning_rate)
        .with_beta1(beta1)
        .with_beta2(beta2)
        .with_verbose(verbose)
        .with_metric(metric.into())
        .with_k_neighbors(k_neighbors)
        .with_min_desired_loss(min_desired_loss)
        .with_loss_reduction(loss_reduction)
        .with_normalized(normalized)
        .with_penalty(penalty)
        .build()
        .expect("Failed to build TrainingConfig");

    // Define the arrays of hyperparameters to search
    let learning_rates = vec![1e-4, 1e-3, 1e-5];
    let batch_sizes = vec![500, 1000, 2000];
    let penalties = vec![1e-6, 1e-7, 1e-8];
    let hidden_size_options = vec![
        // vec![100],
        // vec![200],
        // vec![300],
        vec![500],  // One hidden layer with 500 neurons
        vec![1000], // One hidden layer with 1000 neurons
        vec![1500], // One hidden layer with 1500 neurons
        vec![100, 100],
        vec![200, 200],
        vec![300, 300],
        vec![500, 500],   // Two hidden layers, each with 500 neurons
        vec![1000, 1000], // One hidden layer with 1000 neurons, another with 500
        vec![1500, 1500], // One hidden layer with 1000 neurons, another with 500
        vec![100, 100, 100],
        vec![200, 200, 200],
        vec![300, 300, 300],
        vec![500, 500, 500],
    ];
    // let epochs_options = vec![100, 200, 500, 1000, 2000, 5000]; // Different epochs options to test
    let epochs_options = vec![500];

    // Find the best hyperparameters
    let (best_config, best_loss) = find_best_hyperparameters::<MyAutodiffBackend>(
        num_features,
        num_samples,
        train_data,
        labels,
        device,
        config,
        learning_rates,
        batch_sizes,
        penalties,
        hidden_size_options,
        epochs_options,
    );

    // Print the best configuration and its corresponding loss
    println!("Best model configuration: {}", best_config);
    println!("Best loss: {}", best_loss);
}
