use burn::prelude::*;

/// Apply layer normalisation to a 2-D tensor along the feature dimension (dim 1).
///
/// For each sample row `x[i]`, computes:
///
/// ```text
/// y[i] = (x[i] − mean(x[i])) / (std(x[i]) + ε)
/// ```
///
/// where `ε = 1e-5` is added for numerical stability.
///
/// # Arguments
///
/// * `input` — A 2-D tensor of shape `[batch, features]`.
///
/// # Returns
///
/// A 2-D tensor of the same shape with each row having zero mean and unit
/// standard deviation (up to the ε offset).
///
/// # Note
///
/// This is a simple per-sample normalisation, not the learnable Layer Norm
/// from the original paper (there are no `γ`/`β` scale and shift parameters).
pub fn normalize<B: Backend>(input: Tensor<B, 2>) -> Tensor<B, 2> {
    let mean = input.clone().mean_dim(1); // mean along the feature dimension
    let var = input.clone().var(1); // variance along the feature dimension
    let std = var.sqrt() + 1e-5; // standard deviation with ε for numerical stability

    (input - mean) / std
}
