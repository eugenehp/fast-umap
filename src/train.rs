use std::{
    fmt,
    time::{Duration, Instant},
};

use crate::{
    chart::plot_loss, format_duration, loss::*, model::UMAPModel, utils::convert_vector_to_tensor,
};
use burn::{
    nn::loss::MseLoss,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, cast::ToElement, Device, Tensor},
};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug)]
pub enum LossReduction {
    Mean,
    Sum,
}

#[derive(Debug, PartialEq)]
pub enum Metric {
    Euclidean,
    EuclideanKNN,
}

// Implement From<&str> for Metric
impl From<&str> for Metric {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "euclidean" => Metric::Euclidean,
            "euclideanknn" | "euclidean_knn" => Metric::EuclideanKNN,
            _ => panic!("Invalid metric type: {}", s),
        }
    }
}

// Implement Display for Metric
impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Metric::Euclidean => write!(f, "Euclidean"),
            Metric::EuclideanKNN => write!(f, "Euclidean KNN"),
        }
    }
}

/// Configuration for training the UMAP model.
///
/// This struct contains the hyperparameters and settings required to train the UMAP model.
/// It includes options for the optimizer (e.g., learning rate, batch size, and beta parameters),
/// device configuration (e.g., CPU or GPU), and additional features like verbosity, early stopping,
/// and time limits for training.
#[derive(Debug)]
pub struct TrainingConfig<B: AutodiffBackend> {
    /// The distance metric to use for training the model (e.g., "euclidean", "manhattan").
    pub metric: Metric,

    /// The total number of epochs to run during training.
    pub epochs: usize,

    /// The number of samples to process in each training batch.
    pub batch_size: usize,

    /// The learning rate for the optimizer (controls the step size for parameter updates).
    pub learning_rate: f64,

    /// The device on which to run the model (e.g., CPU or GPU).
    pub device: Device<B>,

    /// The Beta1 parameter for the Adam optimizer (controls the first moment estimate).
    pub beta1: f64,

    /// The Beta2 parameter for the Adam optimizer (controls the second moment estimate).
    pub beta2: f64,

    /// The L2 regularization (weight decay) penalty to apply during training.
    pub penalty: f64,

    /// Whether to show detailed progress information during training (e.g., loss values, progress bars).
    pub verbose: bool,

    /// The number of epochs to wait for improvement before triggering early stopping.
    /// `None` disables early stopping.
    pub patience: Option<i32>,

    /// The method used to reduce the loss during training (e.g., mean or sum).
    pub loss_reduction: LossReduction,

    /// The number of nearest neighbors to consider in the UMAP algorithm.
    pub k_neighbors: usize,

    /// Optionally, the minimum desired loss to achieve before stopping early.
    pub min_desired_loss: Option<f64>,

    /// The maximum time (in seconds) to allow for training. If `None`, there is no time limit.
    pub timeout: Option<u64>,
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
    metric: Option<Metric>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
    device: Option<Device<B>>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    penalty: Option<f64>,
    verbose: Option<bool>,
    patience: Option<i32>,
    loss_reduction: Option<LossReduction>,
    k_neighbors: Option<usize>,
    min_desired_loss: Option<f64>,
    timeout: Option<u64>,
}

impl<B: AutodiffBackend> TrainingConfigBuilder<B> {
    /// Set the distance metric for training (e.g., "Euclidean", "Manhattan").
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Set the number of epochs to train the model.
    /// This defines how many times the entire dataset will be processed.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    /// Set the batch size used during training.
    /// The batch size determines how many samples are processed before updating the model weights.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the learning rate for the optimizer.
    /// The learning rate controls the step size for each parameter update during training.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    /// Set the device to run the model on (e.g., CPU or GPU).
    /// This specifies the hardware where the model will be trained.
    pub fn with_device(mut self, device: Device<B>) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the beta1 parameter for the Adam optimizer.
    /// Beta1 controls the moving average of the first moment (mean) of the gradients.
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = Some(beta1);
        self
    }

    /// Set the beta2 parameter for the Adam optimizer.
    /// Beta2 controls the moving average of the second moment (uncentered variance) of the gradients.
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = Some(beta2);
        self
    }

    /// Set the L2 regularization (weight decay) penalty for the optimizer.
    /// This helps prevent overfitting by penalizing large weights.
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.penalty = Some(penalty);
        self
    }

    /// Set whether verbose output should be shown during training.
    /// If `true`, detailed progress (e.g., loss, metrics) will be displayed during training.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }

    /// Set the patience value for early stopping.
    ///
    /// If training does not improve the loss for `patience` consecutive epochs, training will stop early.
    /// **Warning!** Setting a `patience` value will override the `epochs` parameter.
    pub fn with_patience(mut self, patience: i32) -> Self {
        self.patience = Some(patience);
        self
    }

    /// Set the loss reduction method.
    /// This defines how the loss is reduced across batches (e.g., sum or mean).
    pub fn with_loss_reduction(mut self, loss_reduction: LossReduction) -> Self {
        self.loss_reduction = Some(loss_reduction);
        self
    }

    /// Set the number of nearest neighbors to use in the UMAP algorithm.
    /// This parameter controls the neighborhood size used in the model's calculations.
    pub fn with_k_neighbors(mut self, k_neighbors: usize) -> Self {
        self.k_neighbors = Some(k_neighbors);
        self
    }

    /// Set the minimum desired loss for early stopping.
    /// If the model reaches this loss value, training will stop early.
    pub fn with_min_desired_loss(mut self, min_desired_loss: f64) -> Self {
        self.min_desired_loss = Some(min_desired_loss);
        self
    }

    /// Set the maximum training time in seconds.
    /// The training will be stopped once this time is exceeded.
    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Finalize and create a `TrainingConfig` with the specified options.
    ///
    /// This method returns an `Option<TrainingConfig>`. If any required parameters are missing,
    /// it returns `None`, and default values will be used for those parameters.
    pub fn build(self) -> Option<TrainingConfig<B>> {
        Some(TrainingConfig {
            metric: self.metric.unwrap_or(Metric::Euclidean), // Default to Euclidean if not set
            epochs: self.epochs?,                             // Will panic if not set
            batch_size: self.batch_size?,                     // Will panic if not set
            learning_rate: self.learning_rate.unwrap_or(0.001), // Default to 0.001 if not set
            device: self.device?,                             // Will panic if not set
            beta1: self.beta1.unwrap_or(0.9),                 // Default beta1 if not set
            beta2: self.beta2.unwrap_or(0.999),               // Default beta2 if not set
            penalty: self.penalty.unwrap_or(5e-5),            // Default penalty if not set
            verbose: self.verbose.unwrap_or(false),           // Default to false if not set
            patience: self.patience,                          // Optional, no default
            loss_reduction: self.loss_reduction.unwrap_or(LossReduction::Sum), // Default to Sum if not set
            k_neighbors: self.k_neighbors.unwrap_or(15), // Default to 15 if not set
            min_desired_loss: self.min_desired_loss,     // Optional, no default
            timeout: self.timeout,                       // Optional, no default
        })
    }
}

fn get_distance<B: AutodiffBackend>(
    data: Tensor<B, 2>,
    config: &TrainingConfig<B>,
) -> Tensor<B, 1> {
    match config.metric {
        Metric::Euclidean => euclidean(data),
        Metric::EuclideanKNN => euclidean_knn(data, config.k_neighbors),
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
) -> (UMAPModel<B>, Vec<f64>) {
    if config.metric == Metric::EuclideanKNN && config.k_neighbors > num_samples {
        panic!("When using Euclidean KNN distance, k_neighbors should be smaller than number of samples!")
    }

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
    let global_distances = get_distance(tensor_data.clone(), config);

    // print_tensor_with_title("global_distances", &global_distances);

    let mut epoch = 0;
    let mut losses: Vec<f64> = vec![];
    let mut best_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0;

    let mse_loss = MseLoss::new();

    loop {
        // Forward pass to get the local (low-dimensional) representation.
        let local = model.forward(tensor_data.clone());

        let local_distances = get_distance(local, config);

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

        // stop early, if we reached the desired loss
        if let Some(min_desired_loss) = config.min_desired_loss {
            if current_loss < min_desired_loss {
                break;
            }
        }

        // Check for timeout
        if let Some(timeout) = config.timeout {
            if elapsed >= Duration::new(timeout, 0) {
                println!(
                    "Training stopped due to timeout after {:.2?} seconds.",
                    elapsed
                );
                break; // Stop training if the elapsed time exceeds the timeout
            }
        }

        epoch += 1; // Increment epoch counter.
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
