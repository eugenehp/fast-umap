mod config;
mod get_distance_by_metric;

use crate::{
    chart::{self, plot_loss, ChartConfigBuilder},
    distances::*,
    format_duration,
    kernels::AutodiffBackend,
    model::UMAPModel,
    normalize_data,
    utils::convert_vector_to_tensor,
};
use burn::{
    module::AutodiffModule,
    nn::loss::MseLoss,
    optim::{
        decay::WeightDecayConfig, AdamConfig, GradientsAccumulator, GradientsParams, Optimizer,
    },
    tensor::{cast::ToElement, Tensor},
};
pub use config::*;
use ctrlc;
use get_distance_by_metric::get_distance_by_metric;
use indicatif::{ProgressBar, ProgressStyle};
use num::{Float, FromPrimitive};
use std::{sync::mpsc::channel, time::Duration};
use std::{thread, time::Instant};

/// Train the UMAP model over multiple epochs.
///
/// This function trains the UMAP model by iterating over the dataset for the specified
/// number of epochs. The model's parameters are updated using the Adam optimizer with
/// the specified learning rate, weight decay, and beta parameters. The loss is computed
/// at each epoch, and progress is displayed via a progress bar if verbose mode is enabled.
///
/// # Arguments
/// * `model`: The UMAP model to be trained.
/// * `num_samples`: The number of samples in the training data.
/// * `num_features`: The number of features per sample (columns in the data).
/// * `data`: The training data as a flat `Vec<f64>`, where each sample is represented as a
///   sequence of `num_features` values.
/// * `config`: The `TrainingConfig` containing training hyperparameters and options.
pub fn train<B: AutodiffBackend, F: Float>(
    mut model: UMAPModel<B>,
    num_samples: usize,         // Number of samples in the dataset.
    num_features: usize,        // Number of features (columns) in each sample.
    mut data: Vec<F>,           // Training data.
    config: &TrainingConfig<B>, // Configuration parameters for training.
) -> (UMAPModel<B>, Vec<f64>)
where
    F: FromPrimitive + Send + Sync + burn::tensor::Element,
{
    if config.metric == Metric::EuclideanKNN && config.k_neighbors > num_samples {
        panic!("When using Euclidean KNN distance, k_neighbors should be smaller than number of samples!")
    }

    let (exit_tx, exit_rx) = channel();

    ctrlc::set_handler(move || exit_tx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    let batch_size = config.batch_size;

    // Normalize the input data (Z-score normalization).
    normalize_data(&mut data, num_samples, num_features);

    // Step 1: Split the data into batches (Vec<Vec<f64>>).
    let mut batches: Vec<Vec<F>> = Vec::new();
    for batch_start in (0..num_samples).step_by(config.batch_size) {
        let batch_end = std::cmp::min(batch_start + config.batch_size, num_samples);
        // Create a batch by extracting `batch_size * num_features` elements
        let mut batch = Vec::new();
        for i in batch_start..batch_end {
            let start_idx = i * num_features;
            let end_idx = start_idx + num_features;
            batch.extend_from_slice(&data[start_idx..end_idx]);
        }
        batches.push(batch);
    }

    // Step 2: Precompute the tensor representations and global distances for each batch.
    let mut tensor_batches: Vec<Tensor<B, 2>> = Vec::new();
    let mut global_distances_batches: Vec<Tensor<B, 1>> = Vec::new();

    for batch_data in &batches {
        // Convert each batch to tensor format.
        let tensor_batch =
            convert_vector_to_tensor(batch_data.clone(), batch_size, num_features, &config.device);

        tensor_batches.push(tensor_batch);

        // Compute the global distances for each batch (using the entire dataset).
        let global_tensor_data =
            convert_vector_to_tensor(data.clone(), batch_size, num_features, &config.device);
        let global_distances = get_distance_by_metric(global_tensor_data.clone(), config);
        global_distances_batches.push(global_distances);
    }

    let global_distances_all = Tensor::<B, 1>::cat(global_distances_batches, 0); // Concatenate along the 0-axis
    let tensor_batches_all = Tensor::<B, 2>::cat(tensor_batches, 0); // Concatenate along the 0-axis

    // Initialize the Adam optimizer with weight decay (L2 regularization).
    let config_optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.penalty)))
        .with_beta_1(config.beta1 as f32)
        .with_beta_2(config.beta2 as f32);
    let mut optim = config_optimizer.init();

    let mut accumulator = GradientsAccumulator::new();

    // Start the timer to track training duration.
    let start_time = Instant::now();

    // Initialize a progress bar for verbose output, if enabled.
    let pb = match config.verbose {
        true => {
            let pb = ProgressBar::new(config.epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{bar:40} | {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            Some(pb)
        }
        false => None,
    };

    let mut epoch = 0;
    let mut losses: Vec<f64> = vec![];
    let mut best_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0;

    let mse_loss = MseLoss::new();

    'main: loop {
        // println!("batch {}", format_duration(start_time.elapsed()));
        for (batch_idx, _) in batches.iter().enumerate() {
            if let Ok(_) = exit_rx.try_recv() {
                break 'main;
            }

            // let tensor_batch: &Tensor<B, 2> = &tensor_batches[batch_idx];
            // let global_distances: &Tensor<B, 1> = &global_distances_batches[batch_idx];

            // Slice the corresponding part of the global_distances_all tensor for this batch
            let start_idx = batch_idx * batch_size; // Calculate the starting index
            let end_idx = (batch_idx + 1) * batch_size; // Calculate the ending index
            let end_idx = end_idx.min(tensor_batches_all.shape().dims[0]); // Clip to the size of the tensor

            let tensor_batch = tensor_batches_all.clone().slice([start_idx..end_idx]); // Slice the tensor
            let global_distances = global_distances_all.clone().slice([start_idx..end_idx]); // Slice the tensor

            // Forward pass to get the local (low-dimensional) representation.
            let local = model.forward(tensor_batch.clone());

            // Compute the loss for the batch.
            let local_distances = get_distance_by_metric(local.clone(), config);
            let loss = mse_loss.forward(
                global_distances.clone(),
                local_distances,
                burn::nn::loss::Reduction::Mean,
            );

            let current_loss = loss.clone().into_scalar().to_f64();
            // Compute gradients and update the model parameters using the optimizer.
            losses.push(current_loss);

            let batch_grads = GradientsParams::from_grads(loss.backward(), &model);

            // Accumulate gradients.
            accumulator.accumulate(&model, batch_grads);
        }

        let current_loss = losses.last().unwrap().clone();

        let grads = accumulator.grads(); // Pop the accumulated gradients.

        // Perform an optimization step to update model parameters.
        model = optim.step(config.learning_rate, model, grads);

        // Track elapsed time and update the progress bar.
        let elapsed = start_time.elapsed();
        if let Some(pb) = &pb {
            pb.set_message(format!(
                "Elapsed: {} | Epoch: {} | Loss: {:.4} | Best loss: {:.4}",
                format_duration(elapsed),
                epoch,
                current_loss,
                best_loss,
            ));
        }

        if let Some(timeout) = config.timeout {
            if elapsed >= Duration::from_secs(timeout) {
                break;
            }
        }

        // Track improvements in loss for early stopping.
        if current_loss <= best_loss {
            best_loss = current_loss;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
        }

        // Check for early stopping based on patience or number of epochs.
        if let Some(patience) = config.patience {
            if epochs_without_improvement >= patience && epoch >= config.epochs {
                break; // Stop training if patience is exceeded.
            }
        } else if epoch >= config.epochs {
            break; // Stop training if the specified number of epochs is reached.
        }

        // Stop early if we reach the desired loss.
        if let Some(min_desired_loss) = config.min_desired_loss {
            if current_loss < min_desired_loss {
                break;
            }
        }

        const STEP: usize = 100;
        if epoch > 0 && epoch % STEP == 0 {
            let losses = losses.clone();
            let model = &model.valid();
            let tensor_data =
                convert_vector_to_tensor(data.clone(), num_samples, num_features, &config.device);
            // this is still slow

            let embeddings_for_entire_dataset = model.forward(tensor_data);
            thread::spawn(move || {
                let chart_config = ChartConfigBuilder::default()
                    .caption("MNIST")
                    .path(format!("mnist_{epoch}.png").as_str())
                    .build();

                // Visualize the 2D embedding (local representation) using a chart
                chart::chart_tensor(embeddings_for_entire_dataset, Some(chart_config));
                // Print only last losses
                plot_loss(losses.clone()[STEP..].to_vec(), "losses.png").unwrap();
            });
        }

        epoch += 1;

        // Check for early stopping based on patience or number of epochs.
        if let Some(patience) = config.patience {
            if epochs_without_improvement >= patience && epoch >= config.epochs {
                break; // Stop training if patience is exceeded.
            }
        } else if epoch >= config.epochs {
            break; // Stop training if the specified number of epochs is reached.
        }

        // If verbose mode is enabled, plot the loss curve after training.
        if config.verbose {
            plot_loss(losses.clone(), "losses.png").unwrap();
        }

        // Finish the progress bar if it was used.
        if let Some(pb) = &pb {
            pb.finish();
        }
    }

    // If verbose mode is enabled, plot the loss curve after training.
    if config.verbose {
        plot_loss(losses.clone(), "losses.png").unwrap();
    }

    // Finish the progress bar if it was used.
    if let Some(pb) = pb {
        pb.finish();
    }

    // Return the trained model.
    (model, losses)
}
