use burn::tensor::{backend::AutodiffBackend, Tensor};

use crate::utils::print_tensor_with_title;

// slow version
fn pairwise_distance_slow<B: AutodiffBackend>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
    let n_samples = x.dims()[0];
    let mut dist = Tensor::zeros([n_samples * (n_samples - 1) / 2], &x.device()); // To store the flattened pairwise distances

    // Calculate pairwise squared Euclidean distances
    let mut idx = 0;
    for i in 0..n_samples {
        for j in i + 1..n_samples {
            // Compute squared Euclidean distance between points i and j
            let diff = x.clone().slice([i..i + 1]) - x.clone().slice([j..j + 1]); // Compute the difference between the i-th and j-th row (points)
            let dist_ij = diff.powi_scalar(2).sum(); // Sum of squares of differences (Euclidean squared distance)
            dist = dist.slice_assign([idx..idx + 1], dist_ij); // Store the distance in the flattened tensor
            idx += 1;
        }
    }

    dist // Return the tensor containing pairwise distances
}

fn pairwise_distance<B: AutodiffBackend>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
    let n_samples = x.dims()[0]; // Number of samples (rows)
    let n_features = x.dims()[1]; // Number of features (columns)

    // Expand x to shapes that allow broadcasting for pairwise subtraction
    let x_expanded = x.clone().expand([1, n_samples, n_features]); // Shape: (1, n_samples, n_features)
    let x_transposed = x.clone().expand([n_samples, 1, n_features]); // Shape: (n_samples, 1, n_features)

    // Compute pairwise differences using broadcasting
    let diff = x_expanded - x_transposed; // Shape: (n_samples, n_samples, n_features)

    // Square the differences element-wise using powi_scalar
    let squared_diff = diff.powi_scalar(2); // Element-wise squared differences

    // Sum across the feature dimension (axis 2), producing a shape of (n_samples, n_samples)
    let pairwise_squared_distances = squared_diff.sum_dim(2); // Sum across the feature dimension

    // Use `flatten()` to convert the upper triangular part (excluding the diagonal) into a 1D tensor
    let pairwise_distances = pairwise_squared_distances.triu(1); // Extract the upper triangular part (without diagonal)
    let flattened_distances = pairwise_distances.flatten(0, 1); // Flatten the tensor to a 1D vector

    flattened_distances // Return the flattened tensor of pairwise distances
}

/// Calculate the UMAP loss by comparing pairwise distances between global and local representations
pub fn umap_loss<B: AutodiffBackend>(
    global: &Tensor<B, 2>, // High-dimensional (global) representation
    local: &Tensor<B, 2>,  // Low-dimensional (local) representation
) -> Tensor<B, 1> {
    println!("umap_loss");
    // Compute pairwise distances for both global and local representations
    let global_distances = pairwise_distance(global);
    print_tensor_with_title(Some("global_distances"), &global_distances);
    let local_distances = pairwise_distance(local);
    print_tensor_with_title(Some("local_distances"), &local_distances);

    // Compute the loss as the Frobenius norm (L2 loss) between the pairwise distance matrices
    let difference = (global_distances - local_distances).powi_scalar(2).sum();

    print_tensor_with_title(Some("difference"), &difference);

    difference
}
