use burn::{
    module::AutodiffModule,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Tensor},
};
use plotters::chart;

use crate::{
    loss::umap_loss,
    model::UMAPModel,
    utils::{chart, load_test_data, print_tensor_with_title},
};

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
    mut model: UMAPModel<B>,
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    data: Vec<f64>,
    config: &TrainingConfig<B>,
) {
    let tensor_data = load_test_data::<B>(data.clone(), num_samples, num_features, &config.device);
    print_tensor_with_title(Some("Training data"), &tensor_data);

    // let config_optimizer = AdamConfig::new();
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let mut optim = config_optimizer.init();

    // let dims = data.dims();
    // let num_samples = dims[0];
    // let num_features = dims[1];
    println!("num_features - {num_features}, num_samples - {num_samples}");

    for epoch in 0..config.epochs {
        // Forward pass to get the low-dimensional (local) representation
        let local = model.forward(tensor_data.clone());
        // print_tensor_with_title(Some("local"), &local);

        // Compute the UMAP loss by comparing the pairwise distances
        let loss = umap_loss(tensor_data.clone(), local);

        // print_tensor_with_title(Some("loss"), &loss);

        // Gradients for the current backward pass
        let grads = loss.backward();

        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);

        // Update model parameters using the optimizer
        model = optim.step(config.learning_rate, model, grads);

        // Log the average loss for the epoch
        println!("Epoch {}:\tLoss = {:.3}", epoch, loss.into_scalar());

        // model.debug();
    }

    let model = model.valid();
    let tensor_data = load_test_data(data.clone(), num_samples, num_features, &config.device);
    let local = model.forward(tensor_data);
    print_tensor_with_title(Some("result"), &local);
    chart(local);
}
