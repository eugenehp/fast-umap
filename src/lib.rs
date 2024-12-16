use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use model::{UMAPModel, UMAPModelConfigBuilder};
use train::*;
use utils::*;

use burn::tensor::{Device, Tensor};

pub mod chart;
pub mod loss;
pub mod model;
pub mod train;
pub mod utils;

/// Struct representing the UMAP (Uniform Manifold Approximation and Projection) model.
///
/// This struct contains the model and the device (e.g., CPU or GPU) used for computation.
/// The `fit` method trains the model, and the `transform` method projects the data into a lower-dimensional space.
pub struct UMAP<B: AutodiffBackend> {
    model: UMAPModel<B::InnerBackend>, // UMAP model that performs dimensionality reduction
    device: Device<B>,                 // Device to run the computation on (CPU, GPU)
}

impl<B: AutodiffBackend> UMAP<B> {
    /// Trains the UMAP model on the given data and returns a fitted UMAP model.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors, where each inner vector represents a data sample with multiple features.
    /// * `device` - The device (CPU or GPU) on which to perform training.
    ///
    /// # Returns
    /// A trained `UMAP` model.
    ///
    /// This method initializes the model configuration, sets up the training parameters (like batch size, learning rate, etc.),
    /// and runs the training process using the provided data. It returns an instance of the `UMAP` struct containing
    /// the trained model and the device.
    pub fn fit<F>(data: Vec<Vec<F>>, device: Device<B>, output_size: usize) -> Self
    where
        F: From<f32> + From<f64> + Clone, // F can be either f32 or f64
        f64: From<F>,                     // Ensure F can be converted to f64
    {
        // Set training parameters
        let batch_size = 1;
        let num_samples = data.len();
        let num_features = data[0].len();
        // let output_size = 2; // UMAP typically reduces the data to 2 dimensions
        let hidden_sizes = vec![100]; // Size of the hidden layers in the model
        let learning_rate = 0.001; // Learning rate for optimization
        let beta1 = 0.9; // Beta1 parameter for Adam optimizer
        let beta2 = 0.999; // Beta2 parameter for Adam optimizer
        let epochs = 100; // Number of epochs for training
        let seed = 9999; // Random seed for reproducibility

        B::seed(seed); // Set the seed for the backend

        // Flatten the input data into a single vector of f64 values
        let train_data: Vec<f64> = data.into_iter().flatten().map(|f| f64::from(f)).collect();

        // Build the model configuration
        let model_config = UMAPModelConfigBuilder::default()
            .input_size(num_features)
            .hidden_sizes(hidden_sizes)
            .output_size(output_size)
            .build()
            .unwrap();

        // Initialize the UMAP model
        let model: UMAPModel<B> = UMAPModel::new(&model_config, &device);

        // Build the training configuration
        let config = TrainingConfig::builder()
            .with_epochs(epochs)
            .with_batch_size(batch_size)
            .with_learning_rate(learning_rate)
            .with_device(device.clone())
            .with_beta1(beta1)
            .with_beta2(beta2)
            .build()
            .expect("Failed to build TrainingConfig");

        // Start training
        let model = train(
            model,
            num_samples,
            num_features,
            train_data.clone(),
            &config,
        );

        // Validate the trained model
        let (model, _) = model.valid();

        // Return the fitted UMAP model wrapped in the UMAP struct
        let umap = UMAP { model, device };

        umap
    }

    /// Transforms the input data to a tensor using the trained UMAP model.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors, where each inner vector represents a data sample with multiple features.
    ///
    /// # Returns
    /// A tensor of shape `[num_samples, num_output_features]` representing the transformed data.
    ///
    /// This method converts the input data into a tensor, passes it through the model to obtain
    /// the low-dimensional representation (local space), and returns the result as a tensor.
    pub fn transform_to_tensor(&self, data: Vec<Vec<f64>>) -> Tensor<B::InnerBackend, 2> {
        let num_samples = data.len();
        let num_features = data[0].len();

        // Flatten the input data into a vector of f64 values
        let train_data: Vec<f64> = data.into_iter().flatten().map(|f| f64::from(f)).collect();

        // Convert the data into a tensor
        let global = convert_vector_to_tensor(train_data, num_samples, num_features, &self.device);

        // Perform the forward pass to get the local (low-dimensional) representation
        let local = self.model.forward(global);

        local
    }

    /// Transforms the input data into a lower-dimensional space and returns it as a vector of vectors.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors, where each inner vector represents a data sample with multiple features.
    ///
    /// # Returns
    /// A vector of vectors, where each inner vector represents a low-dimensional representation of a data sample.
    ///
    /// This method is a higher-level abstraction that calls `transform_to_tensor`, converts the result
    /// back to a vector format for easier inspection, and returns the transformed data.
    pub fn transform(&self, data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let local = self.transform_to_tensor(data);
        let result = convert_tensor_to_vector(local);

        result
    }
}

#[allow(unused)]
/// Predefined module for commonly used utilities like generating test data and charting functions.
pub mod prelude {
    use crate::{chart, train, utils, UMAP};
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;

    // Re-export common utilities for easier use
    pub use chart::{chart_tensor, chart_vector};
    pub use train::Metric;
    pub use train::{TrainingConfig, TrainingConfigBuilder};
    pub use utils::generate_test_data;

    /// Convenience function for running UMAP with the WGPU backend.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors, where each inner vector represents a data sample with multiple features.
    ///
    /// # Returns
    /// A trained `UMAP` model that has been fitted to the input data, using the WGPU backend for computation.
    ///
    /// This function wraps the `UMAP::fit` method and provides a simplified way to fit UMAP models with the WGPU backend.
    /// The resulting model will have 2-dimensional output by default.
    ///
    /// # Example
    /// ```rust
    /// let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    /// let model = umap(data);
    /// ```
    pub fn umap<F>(data: Vec<Vec<F>>) -> UMAP<Autodiff<Wgpu>>
    where
        F: From<f32> + From<f64> + Clone, // F can be either f32 or f64
        f64: From<F>,                     // Ensure F can be converted to f64
    {
        let output_size = 2;
        let model = UMAP::<Autodiff<Wgpu>>::fit(data, WgpuDevice::default(), output_size);
        model
    }

    /// Convenience function for running UMAP with the WGPU backend and a custom output size.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors, where each inner vector represents a data sample with multiple features.
    /// * `output_size` - The number of dimensions for the reduced output. This controls the dimensionality of the embedding space.
    ///
    /// # Returns
    /// A trained `UMAP` model that has been fitted to the input data, using the WGPU backend for computation and the specified output size.
    ///
    /// This function wraps the `UMAP::fit` method, providing a way to fit UMAP models with the WGPU backend and a customizable number of output dimensions.
    ///
    /// # Example
    /// ```rust
    /// let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    /// let output_size = 3;
    /// let model = umap_size(data, output_size);
    /// ```
    pub fn umap_size<F>(data: Vec<Vec<F>>, output_size: usize) -> UMAP<Autodiff<Wgpu>>
    where
        F: From<f32> + From<f64> + Clone, // F can be either f32 or f64
        f64: From<F>,                     // Ensure F can be converted to f64
    {
        let model = UMAP::<Autodiff<Wgpu>>::fit(data, WgpuDevice::default(), output_size);
        model
    }
}
