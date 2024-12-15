use burn::prelude::*;
use nn::{Linear, LinearConfig, Relu};
use serde::{Deserialize, Serialize};

/// A neural network model representing the UMAP architecture, consisting of four linear layers and ReLU activations
///
/// # Arguments
/// * `B` - The backend type for tensor operations (e.g., `AutodiffBackend`)
#[derive(Module, Debug)]
pub struct UMAPModel<B: Backend> {
    linear1: Linear<B>, // First linear layer
    linear2: Linear<B>, // Second linear layer
    linear3: Linear<B>, // Third linear layer
    linear4: Linear<B>, // Fourth linear layer
    activation: Relu,   // ReLU activation function
}

impl<B: Backend> UMAPModel<B> {
    /// Creates a new instance of `UMAPModel` with the specified configuration and device
    ///
    /// # Arguments
    /// * `config` - Configuration struct containing the input size, hidden size, and output size
    /// * `device` - The device on which the model should be initialized (e.g., CPU or GPU)
    ///
    /// # Returns
    /// A new `UMAPModel` instance initialized with the provided configuration
    pub fn new(config: &UMAPModelConfig, device: &Device<B>) -> Self {
        // Initialize the first linear layer with the input size and hidden size
        let linear1 = LinearConfig::new(config.input_size, config.hidden_size)
            .with_bias(true)
            .init(device);
        let linear2 = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(true)
            .init(device);
        let linear3 = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(true)
            .init(device);
        let linear4 = LinearConfig::new(config.hidden_size, config.output_size)
            .with_bias(true)
            .init(device);

        // Initialize ReLU activation function
        let activation = Relu::new();

        // Return the UMAPModel with the initialized layers
        UMAPModel {
            linear1,
            linear2,
            linear3,
            linear4,
            activation,
        }
    }

    /// Perform a forward pass through the model
    ///
    /// # Arguments
    /// * `input` - A 2D tensor representing the input data, with shape (n_samples, n_features)
    ///
    /// # Returns
    /// A 2D tensor representing the output after passing through all the layers and activations
    ///
    /// The input passes sequentially through the four linear layers, with ReLU activations applied between each layer.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input); // First linear transformation
        let x = self.activation.forward(x); // Apply ReLU
        let x = self.linear2.forward(x); // Second linear transformation
        let x = self.activation.forward(x); // Apply ReLU
        let x = self.linear3.forward(x); // Third linear transformation
        let x = self.activation.forward(x); // Apply ReLU
        let x = self.linear4.forward(x); // Fourth linear transformation
        x
    }
}

/// Configuration structure for creating a `UMAPModel`
///
/// # Fields
/// * `input_size` - Number of input features
/// * `hidden_size` - Size of the hidden layers
/// * `output_size` - Number of output features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UMAPModelConfig {
    pub input_size: usize,  // Number of input features
    pub hidden_size: usize, // Size of the hidden layer
    pub output_size: usize, // Number of output features
}

impl UMAPModelConfig {
    /// Creates a new builder for the `UMAPModelConfig`
    ///
    /// # Returns
    /// A new `UMAPModelConfigBuilder` to configure the model
    pub fn builder() -> UMAPModelConfigBuilder {
        UMAPModelConfigBuilder::default()
    }
}

/// Builder pattern for the `UMAPModelConfig` struct
///
/// # Fields
/// * `input_size` - Option for the number of input features
/// * `hidden_size` - Option for the size of the hidden layers
/// * `output_size` - Option for the number of output features
#[derive(Debug, Clone)]
pub struct UMAPModelConfigBuilder {
    input_size: Option<usize>,
    hidden_size: Option<usize>,
    output_size: Option<usize>,
}

impl Default for UMAPModelConfigBuilder {
    fn default() -> Self {
        UMAPModelConfigBuilder {
            input_size: Some(100),
            hidden_size: Some(100),
            output_size: Some(2),
        }
    }
}

impl UMAPModelConfigBuilder {
    /// Set the input size for the model
    ///
    /// # Arguments
    /// * `input_size` - The number of input features
    ///
    /// # Returns
    /// The updated `UMAPModelConfigBuilder` with the specified input size
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }

    /// Set the hidden layer size for the model
    ///
    /// # Arguments
    /// * `hidden_size` - The size of the hidden layers
    ///
    /// # Returns
    /// The updated `UMAPModelConfigBuilder` with the specified hidden size
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = Some(hidden_size);
        self
    }

    /// Set the output size for the model
    ///
    /// # Arguments
    /// * `output_size` - The number of output features
    ///
    /// # Returns
    /// The updated `UMAPModelConfigBuilder` with the specified output size
    pub fn output_size(mut self, output_size: usize) -> Self {
        self.output_size = Some(output_size);
        self
    }

    /// Build and return the final `UMAPModelConfig`
    ///
    /// # Returns
    /// A `Result` containing the built `UMAPModelConfig` or an error message if required fields are missing
    pub fn build(self) -> Result<UMAPModelConfig, String> {
        // Ensure that all required fields are set
        Ok(UMAPModelConfig {
            input_size: self
                .input_size
                .ok_or_else(|| "Input size must be set".to_string())?,
            hidden_size: self
                .hidden_size
                .ok_or_else(|| "Hidden size must be set".to_string())?,
            output_size: self
                .output_size
                .ok_or_else(|| "Output size must be set".to_string())?,
        })
    }
}
