use burn::{prelude::*, tensor::backend::AutodiffBackend};
use nn::{Linear, LinearConfig, Relu};
use serde::{Deserialize, Serialize};

#[derive(Module, Debug)]
pub struct UMAPModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> UMAPModel<B> {
    /// Create a new instance of UMAPModel from a given configuration
    pub fn new(config: &UMAPModelConfig, device: &Device<B>) -> Self {
        // Initialize the first linear layer with the input size and hidden size
        let linear1 = LinearConfig::new(config.input_size, config.hidden_size).init(device);

        // Initialize the second linear layer with the hidden size and output size
        let linear2 = LinearConfig::new(config.hidden_size, config.output_size).init(device);

        // Initialize ReLU activation function
        let activation = Relu::new();

        // Return the UMAPModel with the initialized layers
        UMAPModel {
            linear1,
            linear2,
            activation,
        }
    }

    /// Forward pass: Pass the input through the model layers
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input); // First linear transformation
        let x = self.activation.forward(x); // Apply ReLU
        let x = self.linear2.forward(x); // Second linear transformation
        x
    }
}

/// The configuration structure for the UMAPModel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UMAPModelConfig {
    pub input_size: usize,  // Number of input features
    pub hidden_size: usize, // Size of the hidden layer
    pub output_size: usize, // Number of output features
}

impl UMAPModelConfig {
    /// Create a new builder for the UMAPModelConfig
    pub fn builder() -> UMAPModelConfigBuilder {
        UMAPModelConfigBuilder::default()
    }
}

/// The builder for the UMAPModelConfig
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
    /// Set the input size
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }

    /// Set the hidden size
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = Some(hidden_size);
        self
    }

    /// Set the output size
    pub fn output_size(mut self, output_size: usize) -> Self {
        self.output_size = Some(output_size);
        self
    }

    /// Build the UMAPModelConfig
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
