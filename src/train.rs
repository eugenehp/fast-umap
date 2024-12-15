use std::time::Instant;

use crate::{
    chart::plot_loss,
    format_duration,
    loss::{pairwise_distance, umap_loss},
    model::UMAPModel,
    utils::convert_vector_to_tensor,
};
use burn::{
    nn::loss::MseLoss,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, cast::ToElement, Device},
};
use indicatif::{ProgressBar, ProgressStyle};

/// Configuration for training the UMAP model.
///
/// This struct holds the parameters required for training, including hyperparameters like
/// the learning rate, batch size, and optimizer settings. It also includes options for
/// verbosity and early stopping through patience.
#[derive(Debug)]
pub struct TrainingConfig<B: AutodiffBackend> {
    pub epochs: usize,         // Total number of training epochs.
    pub batch_size: usize,     // Size of each training batch.
    pub learning_rate: f64,    // Learning rate for the optimizer.
    pub device: Device<B>,     // Device to run the model on (e.g., CPU or GPU).
    pub beta1: f64,            // Beta1 parameter for the Adam optimizer.
    pub beta2: f64,            // Beta2 parameter for the Adam optimizer.
    pub penalty: f64,          // L2 regularization (weight decay) penalty.
    pub verbose: bool,         // Whether to show detailed progress information.
    pub patience: Option<i32>, // Early stopping patience. None disables early stopping.
}

impl<B: AutodiffBackend> TrainingConfig<B> {
    /// Creates a new builder for constructing a `TrainingConfig`.
    ///
    /// This method allows you to incrementally build a `TrainingConfig` by setting its fields.
    pub fn builder() -> TrainingConfigBuilder<B> {
        TrainingConfigBuilder::default()
    }
}

/// Builder pattern for constructing a `TrainingConfig` with optional parameters.
#[derive(Default)]
pub struct TrainingConfigBuilder<B: AutodiffBackend> {
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
    device: Option<Device<B>>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    penalty: Option<f64>,
    verbose: Option<bool>,
    patience: Option<i32>,
}

impl<B: AutodiffBackend> TrainingConfigBuilder<B> {
    /// Set the number of epochs for training.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    /// Set the batch size used during training.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the learning rate for the optimizer.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    /// Set the device on which to run the model (e.g., CPU or GPU).
    pub fn with_device(mut self, device: Device<B>) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the beta1 parameter for the Adam optimizer.
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = Some(beta1);
        self
    }

    /// Set the beta2 parameter for the Adam optimizer.
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = Some(beta2);
        self
    }

    /// Set the weight decay penalty (L2 regularization) for the optimizer.
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.penalty = Some(penalty);
        self
    }

    /// Set whether verbose output should be shown during training.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }

    /// Set the patience value for early stopping.
    ///
    /// If training does not improve the loss for `patience` consecutive epochs, training will stop.
    /// **Warning!** Setting patience will override the number of epochs.
    pub fn with_patience(mut self, patience: i32) -> Self {
        self.patience = Some(patience);
        self
    }

    /// Finalize and create a `TrainingConfig` with the specified options.
    ///
    /// This method returns an `Option<TrainingConfig>` where `None` indicates that
    /// not all required parameters have been set. Default values are used for missing fields.
    pub fn build(self) -> Option<TrainingConfig<B>> {
        Some(TrainingConfig {
            epochs: self.epochs?,
            batch_size: self.batch_size?,
            learning_rate: self.learning_rate.unwrap_or(0.001), // Default to 0.001 if not set
            device: self.device?,
            beta1: self.beta1.unwrap_or(0.9), // Default beta1 if not set
            beta2: self.beta2.unwrap_or(0.999), // Default beta2 if not set
            penalty: self.penalty.unwrap_or(5e-5), // Default penalty if not set
            verbose: self.verbose.unwrap_or(false),
            patience: self.patience,
        })
    }
}

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
pub fn train<B: AutodiffBackend>(
    mut model: UMAPModel<B>,
    num_samples: usize,         // Number of samples in the dataset.
    num_features: usize,        // Number of features (columns) in each sample.
    data: Vec<f64>,             // Training data.
    config: &TrainingConfig<B>, // Configuration parameters for training.
) -> UMAPModel<B> {
    // Convert the input data into a tensor for processing.
    let tensor_data =
        convert_vector_to_tensor::<B>(data.clone(), num_samples, num_features, &config.device);

    // Initialize the Adam optimizer with weight decay (L2 regularization).
    let config_optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.penalty)))
        .with_beta_1(config.beta1 as f32)
        .with_beta_2(config.beta2 as f32);
    let mut optim = config_optimizer.init();

    // Start the timer to track training duration.
    let start_time = Instant::now();

    // Initialize a progress bar for verbose output, if enabled.
    let pb = match config.verbose {
        true => {
            let pb = ProgressBar::new(config.epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{bar:40} {pos}/{len} Epochs | {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            Some(pb)
        }
        false => None,
    };

    // Precompute the pairwise distances in the global space for loss calculation.
    let global_distances = pairwise_distance(tensor_data.clone());

    let mut epoch = 0;
    let mut losses: Vec<f64> = vec![];
    let mut best_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0;

    let mse_loss = MseLoss::new();

    loop {
        // Forward pass to get the local (low-dimensional) representation.
        let local = model.forward(tensor_data.clone());

        let local_distances = pairwise_distance(local);

        let loss = mse_loss.forward(
            global_distances.clone(),
            local_distances,
            burn::nn::loss::Reduction::Sum,
        );

        // Compute gradients and update the model parameters using the optimizer.
        let grads = loss.backward();
        let current_loss = loss.clone().into_scalar().to_f64();
        losses.push(current_loss);

        let grads = GradientsParams::from_grads(grads, &model);

        // Perform an optimization step to update model parameters.
        model = optim.step(config.learning_rate, model, grads);

        // Track elapsed time and update the progress bar.
        let elapsed = start_time.elapsed();
        if let Some(ref pbb) = pb {
            pbb.set_message(format!(
                "Elapsed: {} | Loss: {:.3} | Best loss: {:.3}",
                format_duration(elapsed),
                current_loss,
                best_loss,
            ));
            pbb.inc(1);
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

        epoch += 1; // Increment epoch counter.
    }

    // If verbose mode is enabled, plot the loss curve after training.
    if config.verbose {
        plot_loss(losses, "losses.png").unwrap();
    }

    // Finish the progress bar if it was used.
    if let Some(pb) = pb {
        pb.finish();
    }

    // Return the trained model.
    model
}
