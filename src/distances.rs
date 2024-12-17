use burn::tensor::{backend::AutodiffBackend, Tensor};

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
// pub fn euclidean_knn<B: AutodiffBackend>(x: Tensor<B, 2>, k: usize) -> Tensor<B, 1> {
//     let n_samples = x.dims()[0]; // Number of samples (rows)
//     let _n_features = x.dims()[1]; // Number of features (columns)
//     let _n_features = x.dims()[1]; // Number of features (columns)

//     // Expand x to shapes that allow broadcasting for pairwise subtraction:
//     // Shape of x_expanded: (1, n_samples, n_features)
//     let x_expanded = x.clone().unsqueeze::<3>();

//     // Shape of x_transposed: (n_samples, 1, n_features)
//     let x_transposed = x.clone().unsqueeze_dim(1);

//     // Compute pairwise differences using broadcasting:
//     // Shape: (n_samples, n_samples, n_features)
//     let diff = x_expanded - x_transposed;

//     // Element-wise square the differences:
//     let squared_diff = diff.powi_scalar(2); // Shape: (n_samples, n_samples, n_features)

//     // Sum along the feature dimension (axis 2) to get squared Euclidean distance:
//     let pairwise_squared_distances = squared_diff.sum_dim(2); // Shape: (n_samples, n_samples)

//     // this speeds up the KNN calculations after this line
//     let pairwise_distances = pairwise_squared_distances.triu(0); // Extract the upper triangular part (without diagonal)

//     // Step 3: Get the top K smallest distances for each sample (along axis 1)
//     let (top_k_distances, _top_k_indices) = pairwise_distances.topk_with_indices(k, 1);

//     // Step 4: Sum the top K distances for each sample
//     let sum_of_top_k_distances = top_k_distances.sum_dim(1).reshape([n_samples]); // Shape: (n_samples)

//     // Return the sum of the top K distances
//     sum_of_top_k_distances
// }

pub fn euclidean_knn<B: AutodiffBackend>(x: Tensor<B, 2>, k: usize) -> Tensor<B, 1> {
    let n_samples = x.dims()[0]; // Number of samples (rows)

    // Calculate pairwise Euclidean distances
    // x has shape [n_samples, n_features]
    let x_squared = x.clone().powf_scalar(2.0).sum_dim(1); // Shape: [n_samples]
    let x_transposed = x.clone().transpose();
    // println!("x_transposed - {:?}", x_transposed.dims());
    let matmul = x.clone().matmul(x_transposed); // Shape: [n_samples, n_samples]
                                                 // println!("matmul - {:?}", matmul.dims());
    let neg = matmul.neg(); // Negate the dot products
                            // println!("neg - {:?}", neg.dims());
    let xx: Tensor<B, 3> = x_squared.clone().unsqueeze_dim(1); // Shape: [n_samples, 1]
    let xx: Tensor<B, 2> = xx.squeeze(2);
    // println!("xx - {:?}", xx.dims());
    let xy: Tensor<B, 3> = x_squared.clone().unsqueeze_dim(0); // Shape: [1, n_samples]
    let xy: Tensor<B, 2> = xy.squeeze(2);
    // println!("xy - {:?}", xy.dims());

    let pairwise_distances = neg + xx + xy;

    // Ensure no negative values due to floating-point precision issues
    let pairwise_distances = pairwise_distances.clamp_min(0.0);

    // Take the square root of the distances
    let pairwise_distances = pairwise_distances.sqrt(); // Shape: [n_samples, n_samples]

    // Sort the distances to find the k-nearest neighbors
    let sorted_indices = pairwise_distances.argsort(1); // Sort by distance in ascending order

    // Cast indices to float (f32) using the cast method
    let nearest_neighbors = sorted_indices.float(); // Cast indices to float (f32)

    // print_tensor_with_title("nearest_neighbors", &nearest_neighbors);

    // Get the k nearest neighbors' indices
    let nearest_neighbors = nearest_neighbors.slice([0..n_samples, 0..k]); // Shape: [n_samples, k]

    // Step 10: Sum the nearest neighbor indices along dimension 1 (sum across k nearest neighbors)
    let summed_neighbors = nearest_neighbors.sum_dim(1).squeeze(1); // Shape: [n_samples]

    // println!("summed_neighbors {:?}", summed_neighbors.dims());

    // Step 11: Return the summed tensor containing the sum of indices for the k-nearest neighbors
    summed_neighbors
}
