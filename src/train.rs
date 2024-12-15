use crate::{
    chart::plot_loss,
    loss::{pairwise_distance, umap_loss},
    model::UMAPModel,
    utils::convert_vector_to_tensor,
};
use burn::{
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, cast::ToElement, Device},
};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug)]
pub struct TrainingConfig<B: AutodiffBackend> {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub device: Device<B>,
    pub beta1: f64,
    pub beta2: f64,
    pub penalty: f64,
    pub verbose: bool,
    pub patience: Option<i32>,
}

impl<B: AutodiffBackend> TrainingConfig<B> {
    // Constructor to initialize the builder
    pub fn builder() -> TrainingConfigBuilder<B> {
        TrainingConfigBuilder::default()
    }
}

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
    /// Set the number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    /// Set the device (e.g., CPU or GPU)
    pub fn with_device(mut self, device: Device<B>) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the beta1 value for the Adam optimizer
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = Some(beta1);
        self
    }

    /// Set the beta2 value for the Adam optimizer
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = Some(beta2);
        self
    }

    /// Set the penalty value for the Adam optimizer
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.penalty = Some(penalty);
        self
    }

    /// Set the verbose value
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }

    /// Set the patience value.
    ///
    /// **Warning!** Overwrites number of epochs, and tries to minimize the loss dynamically.
    pub fn with_patience(mut self, patience: i32) -> Self {
        self.patience = Some(patience);
        self
    }

    // Finalize and build the TrainingConfig
    pub fn build(self) -> Option<TrainingConfig<B>> {
        Some(TrainingConfig {
            epochs: self.epochs?,
            batch_size: self.batch_size?,
            learning_rate: self.learning_rate.unwrap_or(0.001), // Default to 0.001 if not set
            device: self.device?,
            beta1: self.beta1.unwrap_or(0.9), // Default beta1 if not set
            beta2: self.beta2.unwrap_or(0.999), // Default beta2 if not set
            penalty: self.penalty.unwrap_or(5e-5), // Default penalty is not set
            verbose: self.verbose.unwrap_or(false),
            patience: self.patience,
        })
    }
}

/// Train the UMAP model over multiple epochs
pub fn train<B: AutodiffBackend>(
    mut model: UMAPModel<B>,
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    data: Vec<f64>,
    config: &TrainingConfig<B>,
) -> UMAPModel<B> {
    let tensor_data =
        convert_vector_to_tensor::<B>(data.clone(), num_samples, num_features, &config.device);

    let config_optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.penalty)))
        .with_beta_1(config.beta1 as f32)
        .with_beta_2(config.beta2 as f32);
    let mut optim = config_optimizer.init();

    // Initialize the progress bar with the number of epochs
    let mut pb = match config.verbose {
        true => {
            let pb = ProgressBar::new(config.epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{bar:40} {pos}/{len} Epochs, Loss: {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            Some(pb)
        }
        false => None,
    };

    let global_distances = pairwise_distance(tensor_data.clone());

    let mut epoch = 0;
    let mut losses: Vec<f64> = vec![];
    let mut best_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0;

    loop {
        // Forward pass to get the low-dimensional (local) representation
        let local = model.forward(tensor_data.clone());

        // Compute the UMAP loss by comparing the pairwise distances
        let loss = umap_loss(global_distances.clone(), local);

        // Gradients for the current backward pass
        let grads = loss.backward();
        let current_loss = loss.clone().into_scalar().to_f64();
        losses.push(current_loss);

        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);

        // Update model parameters using the optimizer
        model = optim.step(config.learning_rate, model, grads);

        // Update the progress bar with the current loss
        if let Some(pbb) = pb {
            pbb.set_message(format!("{:.3} Best loss: {}", current_loss, best_loss));
            pbb.inc(1);
            pb = Some(pbb);
        }

        if current_loss <= best_loss {
            best_loss = current_loss;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
        }

        if let Some(patience) = config.patience {
            if epochs_without_improvement >= patience && epoch >= config.epochs {
                break;
            }
        } else if epoch >= config.epochs {
            break;
        }

        epoch += 1;
    }

    if config.verbose {
        plot_loss(losses, "losses.png").unwrap();
    }

    if let Some(pb) = pb {
        pb.finish();
    }

    model
}
