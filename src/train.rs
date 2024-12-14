use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Device, Tensor},
};

use crate::{loss::umap_loss, model::UMAPModel};

#[derive(Debug)]
pub struct TrainingConfig<B: AutodiffBackend> {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub device: Device<B>,
    pub beta1: f64,
    pub beta2: f64,
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
}

impl<B: AutodiffBackend> TrainingConfigBuilder<B> {
    // Set the number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    // Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    // Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    // Set the device (e.g., CPU or GPU)
    pub fn device(mut self, device: Device<B>) -> Self {
        self.device = Some(device);
        self
    }

    // Set the beta1 value for the Adam optimizer
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = Some(beta1);
        self
    }

    // Set the beta2 value for the Adam optimizer
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = Some(beta2);
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
        })
    }
}

/// Train the UMAP model over multiple epochs
pub fn train<B: AutodiffBackend>(
    model: UMAPModel<B>,
    train_data: &Vec<Tensor<B, 2>>, // Global representations (high-dimensional data)
    config: &TrainingConfig<B>,
) {
    let learning_rate: f64 = 0.001;
    let config_optimizer = AdamConfig::new();
    // Initialize an optimizer, e.g., Adam with a learning rate
    let mut optim = config_optimizer.init();

    for epoch in 0..config.epochs {
        let n_features = train_data[0].dims()[1];
        let mut total_loss = Tensor::<B, 2>::zeros([n_features, n_features], &config.device); // Initialize total_loss as scalar

        // Loop over batches of input data
        for (iteration, batch) in train_data.chunks(config.batch_size).enumerate() {
            for input_tensor in batch {
                // Forward pass to get the low-dimensional (local) representation
                let local = model.forward(input_tensor.clone());

                // Compute the UMAP loss by comparing the pairwise distances
                let loss = umap_loss(input_tensor, &local);

                // Backward pass: Compute gradients
                loss.backward();

                // Log training progress
                println!(
                    "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                    epoch,
                    iteration,
                    loss.clone().into_scalar(),
                );

                // Gradients for the current backward pass
                let grads = loss.backward();
                // Gradients linked to each parameter of the model.
                let grads = GradientsParams::from_grads(grads, &model);
                // Update model parameters using the optimizer
                optim.step(learning_rate, model.clone(), grads);

                // Accumulate the loss for this epoch
                total_loss = total_loss.add(loss.unsqueeze());
            }
        }

        // Log the average loss for the epoch
        println!(
            "Epoch {}: Loss = {:.3}",
            epoch,
            total_loss.clone().into_scalar()
        );
    }
}
