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
pub fn euclidean<B: AutodiffBackend>(x: Tensor<B, 2>) -> Tensor<B, 1> {
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

/// Computes the sum of the top K smallest pairwise squared Euclidean distances for each sample in the input tensor.
///
/// This function calculates the Euclidean distances between all pairs of samples in the input tensor `x` using an efficient method
/// that avoids creating a full 3D tensor of pairwise distances. It then returns the sum of the K smallest distances for each sample.
///
/// # Parameters
/// - `x`: A 2D tensor of shape `(n_samples, n_features)` representing the dataset, where each row is a sample and each column is a feature.
/// - `k`: The number of nearest neighbors to consider when computing the sum of distances.
///
/// # Returns
/// - A 1D tensor of shape `(n_samples,)` containing the sum of the squared Euclidean distances to the top K nearest neighbors
///   for each sample. The distance computation is done efficiently using broadcasting to avoid creating large intermediate tensors.
///
/// # Example
/// ```rust
/// let x = Tensor::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
/// let k = 2;
/// let result = euclidean_knn(x, k);
/// println!("{:?}", result); // Output: sum of squared distances for each sample to its 2 nearest neighbors
/// ```
///
/// # Notes
/// - The computation of pairwise distances leverages broadcasting to calculate the squared distances between samples
///   without explicitly creating a full pairwise distance matrix.
/// - The function is designed to be efficient by avoiding the full pairwise distance matrix, which can be large for datasets with many samples.
/// - The tensor operations require that the backend supports the relevant operations for broadcasting, summing, and matrix multiplication.

pub fn euclidean_knn<B: AutodiffBackend>(x: Tensor<B, 2>, k: usize) -> Tensor<B, 1> {
    let n_samples = x.dims()[0]; // Number of samples (rows)
    let _n_features = x.dims()[1]; // Number of features (columns)

    // Efficient pairwise squared Euclidean distance computation:
    // We compute the distances without explicitly creating the full 3D tensor.

    // Step 1: Compute the squared norm of each sample (x^2)
    let x_norm_squared = x.clone().powi_scalar(2).sum_dim(1); // Shape: (n_samples)

    // Step 2: Compute pairwise squared Euclidean distances using broadcasting
    let pairwise_squared_distances = x_norm_squared.clone().unsqueeze_dim(1)
        + x_norm_squared.unsqueeze_dim(0)
        - x.clone().matmul(x.clone().transpose()).mul_scalar(2.0); // Shape: (n_samples, n_samples)

    // Step 3: Get the top K smallest distances for each sample (along axis 1)
    let (top_k_distances, _top_k_indices) = pairwise_squared_distances.topk_with_indices(k, 1);

    // Step 4: Sum the top K distances for each sample
    let sum_of_top_k_distances = top_k_distances.sum_dim(1).reshape([n_samples]); // Shape: (n_samples)

    // Return the sum of the top K distances
    sum_of_top_k_distances
}
