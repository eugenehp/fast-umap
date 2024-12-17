use burn::tensor::{backend::AutodiffBackend, Tensor};

#[allow(unused)]
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

    sum_of_top_k_distances
}

pub fn manhattan<B: AutodiffBackend>(tensor: Tensor<B, 2>) -> Tensor<B, 1> {
    let n_samples = tensor.dims()[0];
    // Sum the absolute difference along the rows (axis 1)
    let x = tensor
        .abs() // Take absolute value
        .sum_dim(1)
        .reshape([n_samples]); // Sum along axis 1 (columns)

    x
}

/// Computes the cosine similarity between each row of a 2D tensor and the first row.
///
/// This function calculates the cosine similarity between each sample (row) in the input tensor
/// and the first sample (first row). The cosine similarity is defined as the dot product of two
/// vectors divided by the product of their magnitudes (L2 norms). The result is a 1D tensor where
/// each element represents the cosine similarity between the corresponding row and the first row.
///
/// # Arguments
/// * `tensor` - A 2D tensor of shape `(n_samples, n_features)` representing the data. The function
///   computes cosine similarity between each row (sample) and the first row.
///
/// # Returns
/// A 1D tensor of shape `(n_samples,)` containing the cosine similarities between the first row and
/// each of the other rows in the input tensor. The values are in the range [-1, 1], where 1 indicates
/// identical orientation, 0 indicates orthogonality, and -1 indicates opposite orientation.
///
/// # Example
/// ```
/// let tensor = Tensor::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// let similarities = cosine(tensor);
/// // `similarities` is a 1D tensor of cosine similarities between the first row and all other rows
/// ```
///
/// # Notes
/// The function uses the following steps:
/// 1. Computes the L2 norm (magnitude) of the first row.
/// 2. Computes the dot product of each row with the first row.
/// 3. Computes the L2 norm of each row.
/// 4. Divides the dot product by the product of the norms to compute cosine similarity.
///
/// # Performance
/// This function clones the tensor multiple times, which may impact performance for large tensors.
/// Optimizations could be made to minimize memory allocations and cloning.
pub fn cosine<B: AutodiffBackend>(tensor: Tensor<B, 2>) -> Tensor<B, 1> {
    let n_samples = tensor.dims()[0];
    let n_features = tensor.dims()[1];
    // First, get the first row to compare to
    let first_row = tensor.clone().slice([0..1, 0..n_features]); // Select the first row

    // Compute L2 norm of the first row manually (sqrt(sum(x^2)))
    let first_row_norm = first_row.clone().powi_scalar(2).sum_dim(1).sqrt();

    // Compute dot product of each row with the first row
    let dot_product: Tensor<B, 2> = tensor.clone().mul(first_row.clone()); // Calculate dot product for each row
    let dot_product: Tensor<B, 3> = dot_product.unsqueeze_dim(2);
    let dot_product: Tensor<B, 2> = dot_product.sum_dim(1).reshape([n_samples, 1]); // Reshape to a column vector (1D)

    // Compute L2 norm (magnitude) of each row manually (sqrt(sum(x^2)))
    let row_norms = tensor.clone().powi_scalar(2).sum_dim(1).sqrt();

    // Compute cosine similarity
    let x = dot_product.div(row_norms).div(first_row_norm);

    x.reshape([n_samples])
}
