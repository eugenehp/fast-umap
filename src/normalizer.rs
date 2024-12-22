use burn::prelude::*;

pub fn normalize<B: Backend>(input: Tensor<B, 2>) -> Tensor<B, 2> {
    let mean = input.clone().mean_dim(1); // Mean along the feature dimension
    let var = input.clone().var(1); // Variance along the feature dimension
    let std = var.sqrt() + 1e-5; // Standard deviation with epsilon for numerical stability

    let normalized = (input - mean) / std; // Normalize the input

    normalized
}
