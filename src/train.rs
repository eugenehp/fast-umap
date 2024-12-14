use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Device, Tensor},
};

use crate::{loss::umap_loss, model::UMAPModel, utils::print_tensor_with_title};

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
    data: Tensor<B, 2>,
    config: &TrainingConfig<B>,
) {
    let config_optimizer = AdamConfig::new();
    let mut optim = config_optimizer.init();
    // let mut accumulator = GradientsAccumulator::new();

    let dims = data.dims();
    let n_samples = dims[0];
    let n_features = dims[1];
    println!("n_features - {n_features}, n_samples - {n_samples}");

    for epoch in 0..config.epochs {
        // print_tensor(&vec![total_loss.clone()]);

        // print_tensor_with_title(Some("Input tensor"), &data);

        // Forward pass to get the low-dimensional (local) representation
        let local = model.forward(data.clone());
        // print_tensor_with_title(Some("local"), &local);

        // Compute the UMAP loss by comparing the pairwise distances
        let loss = umap_loss(data.clone(), local);

        // print_tensor_with_title(Some("loss"), &loss);

        // Gradients for the current backward pass
        let grads = loss.backward();

        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);

        // Update model parameters using the optimizer
        optim.step(config.learning_rate, model.clone(), grads);

        // Log the average loss for the epoch
        println!("Epoch {}:\tLoss = {:.3}", epoch, loss.into_scalar());

        model.debug();
    }
}
