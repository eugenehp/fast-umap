mod config;
mod get_distance_by_metric;

#[cfg(feature = "verbose")]
use tracing::{debug, info, span, Level};

#[cfg(feature = "plots")]
use crate::chart::{self, plot_loss, ChartConfigBuilder};

use crate::{
    backend::AutodiffBackend, format_duration, model::UMAPModel, normalize_data,
    utils::convert_vector_to_tensor,
};
use burn::{
    module::Module,
    nn::loss::MseLoss,
    optim::{
        decay::WeightDecayConfig, AdamConfig, GradientsAccumulator, GradientsParams, Optimizer,
    },
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{cast::ToElement, Device, Shape, Tensor},
};
pub use config::*;

use crossbeam_channel::Receiver;
use get_distance_by_metric::*;
use indicatif::{ProgressBar, ProgressStyle};
use num::{Float, FromPrimitive};
#[cfg(feature = "plots")]
use std::thread;
use std::time::Duration;
use std::time::Instant;

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
///   sequence of `num_features` values. In row-major order.
/// * `config`: The `TrainingConfig` containing training hyperparameters and options.
pub fn train<B: AutodiffBackend, F: Float>(
    name: &str,
    mut model: UMAPModel<B>,
    num_samples: usize,      // Number of samples in the dataset.
    num_features: usize,     // Number of features (columns) in each sample.
    mut data: Vec<F>,        // Training data in row-major order.
    config: &TrainingConfig, // Configuration parameters for training.
    device: Device<B>,
    exit_rx: Receiver<()>,
) -> (UMAPModel<B>, Vec<F>, F)
where
    F: FromPrimitive + Send + Sync + burn::tensor::Element,
{
    #[cfg(feature = "verbose")]
    {
        // Initialize tracing subscriber only once
        use tracing_subscriber;
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();
    }

    if config.metric == Metric::EuclideanKNN && config.k_neighbors > num_samples {
        panic!("When using Euclidean KNN distance, k_neighbors should be smaller than number of samples!")
    }

    // you can also store in memory using BytesRecorder
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_path = format!("./{name}.bin");

    let batch_size = config.batch_size;

    #[cfg(feature = "verbose")]
    {
        println!("config - {config:#?}");
    }

    if batch_size == 1 {
        panic!("You can not have a batch size of 1, because UMAP doesn't have other samples to compare to!")
    }

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
    let mut global_distances_batches: Vec<Tensor<B, 2>> = Vec::new();

    // store the size of the Tensor after the distance has been calculated
    let mut global_distance_size: Shape = Shape::from([0, 0]);

    #[cfg(feature = "verbose")]
    info!("Preprocessing {} batches", batches.len());

    for (_batch_idx, batch_data) in batches.iter().enumerate() {
        #[cfg(feature = "verbose")]
        let _span = span!(Level::DEBUG, "preprocess_batch", batch = _batch_idx).entered();

        // Convert each batch to tensor format.
        let tensor_batch =
            convert_vector_to_tensor(batch_data.clone(), batch_size, num_features, &device);

        tensor_batches.push(tensor_batch);

        // Compute the global distances for each batch (in high-dimensional space).
        let global_tensor_data =
            convert_vector_to_tensor(batch_data.clone(), batch_size, num_features, &device);

        #[cfg(feature = "verbose")]
        debug!("Computing global distances for batch {}", _batch_idx);

        let global_distances =
            get_distance_by_metric(global_tensor_data.clone(), config, Some("global".into()));

        global_distance_size = global_distances.shape();
        global_distances_batches.push(global_distances);
    }

    #[cfg(feature = "verbose")]
    info!("Preprocessing complete, starting training loop");

    let global_distances_all = Tensor::<B, 2>::cat(global_distances_batches, 0); // Concatenate along the 0-axis
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
    let mut losses: Vec<F> = vec![];
    let mut best_loss = F::infinity();
    let mut epochs_without_improvement = 0;

    let mse_loss = MseLoss::new();

    'main: loop {
        #[cfg(feature = "verbose")]
        let _epoch_span = span!(Level::INFO, "epoch", epoch = epoch).entered();

        for (batch_idx, _) in batches.iter().enumerate() {
            #[cfg(feature = "verbose")]
            let _batch_span = span!(Level::DEBUG, "train_batch", batch = batch_idx).entered();
            if let Ok(_) = exit_rx.try_recv() {
                break 'main;
            }

            // Slice the corresponding part of the global_distances_all tensor for this batch
            let start_idx = batch_idx * batch_size * num_features; // Calculate the starting index
            let end_idx = (batch_idx + 1) * batch_size * num_features; // Calculate the ending index
            let end_idx = end_idx.min(tensor_batches_all.shape().dims[0]); // Clip to the size of the tensor

            // skip last batch
            if start_idx > end_idx {
                continue;
            }

            let batch_start_idx = batch_idx * batch_size;
            let batch_end_idx = (batch_idx + 1) * batch_size;

            let global_start_idx = batch_idx * global_distance_size.dims[0];
            let global_end_idx = (batch_idx + 1) * global_distance_size.dims[0];

            let global_distances = global_distances_all
                .clone()
                .slice([global_start_idx..global_end_idx]); // Slice the tensor

            let tensor_batch = tensor_batches_all
                .clone()
                .slice([batch_start_idx..batch_end_idx, 0..num_features]); // Slice the tensor

            // Forward pass to get the local (low-dimensional) representation.
            let local = model.forward(tensor_batch.clone());

            // Compute the loss for the batch.
            let local_distances =
                get_distance_by_metric(local.clone(), config, Some("local".into()));

            let loss = mse_loss.forward(
                global_distances.clone(),
                local_distances.clone(),
                burn::nn::loss::Reduction::Mean,
            );

            let current_loss = F::from(loss.clone().into_scalar().to_f64()).unwrap();

            #[cfg(feature = "verbose")]
            {
                debug!("Batch {} loss: {}", batch_idx, current_loss);
                if current_loss.is_nan() {
                    panic!("current loss is NaN")
                }
                if current_loss.is_infinite() {
                    debug!("Warning: infinite loss detected");
                }
            }

            losses.push(current_loss);

            // TODO: if loss is NaN, do something else. FIXME
            let grads = loss.backward();

            // Compute gradients and update the model parameters using the optimizer.
            let batch_grads = GradientsParams::from_grads(grads, &model);

            // Accumulate gradients.
            accumulator.accumulate(&model, batch_grads);
        }

        let current_loss = losses.last().unwrap().clone();

        #[cfg(feature = "verbose")]
        info!("Epoch {} completed, loss: {}", epoch, current_loss);

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

            model
                .clone()
                .save_file(model_path.clone(), &recorder)
                .expect("Should be able to save the model");
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
            if current_loss < F::from(min_desired_loss).unwrap() {
                break;
            }
        }

        #[cfg(feature = "plots")]
        let output_path = format!("losses_{name}.png");

        #[cfg(feature = "plots")]
        {
            use burn::module::AutodiffModule;
            const STEP: usize = 100;

            let name = name.to_string();
            if epoch > 0 && epoch % STEP == 0 {
                let losses = losses.clone();
                let model = &model.valid();
                let tensor_data =
                    convert_vector_to_tensor(data.clone(), num_samples, num_features, &device);
                // this is still slow

                let embeddings_for_entire_dataset = model.forward(tensor_data);
                let output_path = output_path.clone();

                thread::spawn(move || {
                    let chart_config = ChartConfigBuilder::default()
                        .caption(format!("{name}_{epoch}").as_str())
                        .path(format!("{name}_{epoch}.png").as_str())
                        .build();

                    // Visualize the 2D embedding (local representation) using a chart
                    chart::chart_tensor(embeddings_for_entire_dataset, None, Some(chart_config));
                    // Print only last losses
                    plot_loss(losses.clone()[STEP..].to_vec(), &output_path).unwrap();
                });
            }
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

        // If plots feature is enabled, plot the loss curve after training.
        #[cfg(feature = "plots")]
        if config.verbose {
            plot_loss(losses.clone(), &output_path).unwrap();
        }

        // Finish the progress bar if it was used.
        if let Some(pb) = &pb {
            pb.finish();
        }
    }

    // If plots feature is enabled, plot the loss curve after training.
    #[cfg(feature = "plots")]
    {
        let name = format!("losses_{name}.png");
        plot_loss(losses.clone(), name.as_str()).unwrap();
    }

    // Finish the progress bar if it was used.
    if let Some(pb) = pb {
        pb.finish();
    }

    // let record = BinFileRecorder::<FullPrecisionSettings>::default();
    model = model
        .load_file(model_path, &recorder, &device)
        .expect("Load model from the best weights file");

    // Return best trained model
    (model, losses, best_loss)
}
