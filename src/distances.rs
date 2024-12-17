use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};

use crate::print_tensor_with_title;

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
pub fn euclidean_knn<B: AutodiffBackend>(x: Tensor<B, 2>, k: usize) -> Tensor<B, 1> {
    let device = x.device();
    let n_samples = x.dims()[0]; // Number of samples (rows)

    // Expand x to shapes that allow broadcasting for pairwise subtraction:
    let x_expanded = x.clone().unsqueeze::<3>(); // Shape: (1, n_samples, n_features)
    let x_transposed = x.clone().unsqueeze_dim(1); // Shape: (n_samples, 1, n_features)

    // Compute pairwise differences using broadcasting:
    let diff = x_expanded - x_transposed;

    // Element-wise square the differences:
    let squared_diff = diff.powi_scalar(2); // Shape: (n_samples, n_samples, n_features)

    // Sum along the feature dimension (axis 2) to get squared Euclidean distance:
    let pairwise_squared_distances = squared_diff.sum_dim(2); // Shape: (n_samples, n_samples)

    // Extract the upper triangular part (without diagonal) for efficient KNN calculation:
    let pairwise_distances = pairwise_squared_distances.triu(0); // Shape: (n_samples, n_samples)

    // Get the top K smallest distances for each sample (along axis 1):
    let (top_k_distances, _top_k_indices) = pairwise_distances.topk_with_indices(k, 1);

    // Sum the top K distances for each sample:
    let sum_of_top_k_distances: Tensor<B, 1> = top_k_distances.sum_dim(1).reshape([n_samples]); // Shape: (n_samples)

    // Normalize the result using min-max normalization:
    let min_val = sum_of_top_k_distances.clone().min(); // Find the minimum value
    let max_val = sum_of_top_k_distances.clone().max(); // Find the maximum value

    // this is to prevent deleting by zero
    let offset_val = Tensor::<B, 1>::from_data(TensorData::new(vec![1e-6], [1]), &device);

    let are_equal = max_val
        .clone()
        .equal(min_val.clone())
        .to_data()
        .to_vec::<bool>()
        .unwrap();

    let are_equal = are_equal.first().unwrap();

    // Avoid division by zero by ensuring max_val != min_val
    let normalized_distances = if !are_equal {
        (sum_of_top_k_distances - min_val.clone()) / (max_val - min_val + offset_val)
    } else {
        sum_of_top_k_distances.clone() // If all values are the same, return the original
    };

    // print_tensor_with_title("normalized_distances", &normalized_distances);

    // Return the normalized sum of the top K distances
    normalized_distances
}
