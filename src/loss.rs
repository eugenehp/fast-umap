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

/// Computes the sum of the squared Euclidean distances to the `k` nearest neighbors for each sample.
///
/// # Parameters
/// - `x`: A 2D tensor of shape `(n_samples, n_features)`, where `n_samples` is the number of samples
///   and `n_features` is the number of features (dimensions) for each sample.
/// - `k`: The number of nearest neighbors to consider.
///
/// # Returns
/// A 1D tensor of shape `(n_samples,)`, where each element is the sum of squared Euclidean distances
/// to the `k` nearest neighbors for the corresponding sample.
///
/// # Description
/// This function computes the pairwise squared Euclidean distances between each sample in the input tensor
/// `x`, and then identifies the `k` smallest distances for each sample. It returns the sum of these `k` smallest
/// distances, which can be used in various applications such as k-NN classification or as part of a loss function.
///
/// # Example
/// ```rust
/// let x = Tensor::<f32>::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
/// let k = 2;
/// let result = euclidean_knn(x, k);
/// // result will contain the sum of squared distances to the 2 nearest neighbors for each sample
/// ```
pub fn euclidean_knn<B: AutodiffBackend>(x: Tensor<B, 2>, k: usize) -> Tensor<B, 1> {
    let n_samples = x.dims()[0]; // Number of samples (rows)
    let _n_features = x.dims()[1]; // Number of features (columns)

    // Expand x to shapes that allow broadcasting for pairwise subtraction:
    // Shape of x_expanded: (1, n_samples, n_features)
    let x_expanded = x.clone().unsqueeze::<3>();

    // Shape of x_transposed: (n_samples, 1, n_features)
    let x_transposed = x.clone().unsqueeze_dim(1);

    // Compute pairwise differences using broadcasting:
    // Shape: (n_samples, n_samples, n_features)
    let diff = x_expanded - x_transposed;

    // Element-wise square the differences:
    let squared_diff = diff.powi_scalar(2); // Shape: (n_samples, n_samples, n_features)

    // Sum along the feature dimension (axis 2) to get squared Euclidean distance:
    let pairwise_squared_distances = squared_diff.sum_dim(2); // Shape: (n_samples, n_samples)

    // Now, we have pairwise squared distances in `pairwise_squared_distances`
    // For each sample, we want to find the k nearest neighbors, so we do the following:

    // Get the top K smallest distances (along axis 1) and the corresponding indices:
    let (top_k_distances, _top_k_indices) = pairwise_squared_distances.topk_with_indices(k, 1);

    // Sort distances in ascending order to ensure we are taking the smallest ones:
    let top_k_distances = top_k_distances.sort_descending(1);

    // Sum up the top K distances (you could also take the mean, depending on your loss function):
    let sum_of_top_k_distances = top_k_distances.sum_dim(1).reshape([n_samples]); // Shape: (n_samples)

    // You can return the sum of distances to K nearest neighbors (or use this as part of your loss):
    sum_of_top_k_distances
}
