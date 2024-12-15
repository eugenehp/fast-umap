use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Calculate the pairwise Euclidean distance matrix for a given 2D tensor
///
/// # Arguments
/// * `x` - A 2D tensor of shape (n_samples, n_features) where each row is a sample and each column is a feature
///
/// # Returns
/// A 1D tensor containing the pairwise distances (upper triangular part of the distance matrix) of shape (n_samples)
///
/// This function computes the pairwise Euclidean distance between samples by using broadcasting
/// to efficiently subtract the samples from each other, squaring the differences, and summing across the features.
pub fn pairwise_distance<B: AutodiffBackend>(x: Tensor<B, 2>) -> Tensor<B, 1> {
    let n_samples = x.dims()[0]; // Number of samples (rows)
    let _n_features = x.dims()[1]; // Number of features (columns)

    // Expand x to shapes that allow broadcasting for pairwise subtraction
    let x_expanded = x.clone().unsqueeze::<3>(); // Shape: (1, n_samples, n_features)
    let x_transposed = x.clone().unsqueeze_dim(1); // Shape: (n_samples, 1, n_features)

    // Compute pairwise differences using broadcasting
    let diff = x_expanded - x_transposed; // Shape: (n_samples, n_samples, n_features)

    // Square the differences element-wise using powi_scalar
    let squared_diff = diff.powi_scalar(2); // Element-wise squared differences

    // Sum across the feature dimension (axis 2), producing a shape of (n_samples, n_samples)
    let pairwise_squared_distances = squared_diff.sum_dim(2); // Sum across the feature dimension

    // Use `flatten()` to convert the upper triangular part (excluding the diagonal) into a 1D tensor
    let pairwise_distances = pairwise_squared_distances.triu(0); // Extract the upper triangular part (without diagonal)

    // Extract the first column (distances from the first sample to all others)
    let distances = pairwise_distances
        .slice([0..n_samples, 0..1])
        .reshape([n_samples]);

    distances
}

/// Compute the UMAP loss between global and local representations using pairwise distances
///
/// # Arguments
/// * `global_distances` - A 1D tensor containing pairwise distances in the high-dimensional (global) representation
/// * `local` - A 2D tensor containing the low-dimensional (local) representation of shape (n_samples, n_features)
///
/// # Returns
/// A 1D tensor representing the UMAP loss between global and local pairwise distances
///
/// This function computes the UMAP loss by calculating the difference between pairwise distances in
/// global and local representations, clamping the distances to avoid overflow, and summing the squared differences.
pub fn umap_loss<B: AutodiffBackend>(
    global_distances: Tensor<B, 1>, // Distances in the high-dimensional (global) representation
    local: Tensor<B, 2>,            // Low-dimensional (local) representation
) -> Tensor<B, 1> {
    // Compute pairwise distances for both global and local representations
    let local_distances = pairwise_distance(local);

    // we have to add these to prevent "attempt to subtract with overflow" error
    let max_distance = 1e6; // A reasonable upper bound
    let safe_global_distances = global_distances.clamp(0.0, max_distance);
    let safe_local_distances = local_distances.clamp(0.0, max_distance);

    // Compute the loss as the Frobenius norm (L2 loss) between the pairwise distance matrices
    let difference = (safe_global_distances - safe_local_distances)
        .powi_scalar(2)
        .sum();

    difference
}
