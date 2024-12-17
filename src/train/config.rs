use burn::tensor::{backend::AutodiffBackend, Device};
use std::fmt;

#[derive(Debug)]
pub enum LossReduction {
    Mean,
    Sum,
}

#[derive(Debug, PartialEq)]
pub enum Metric {
    Euclidean,
    EuclideanKNN,
    // EuclideanWeighted,
    Manhattan,
    // Cosine,
    // Correlation,
    // Hamming,
    // Jaccard,
    // Minkowski,
    // Chebyshev,
    // Mahalnobis,
    // Spearman, // Spearman’s Rank Correlation Distance
}

// Implement From<&str> for Metric
impl From<&str> for Metric {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "euclidean" => Metric::Euclidean,
            "euclideanknn" | "euclidean_knn" => Metric::EuclideanKNN,
            "manhattan" => Metric::Manhattan,
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
            Metric::Manhattan => write!(f, "Manhattan"),
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
