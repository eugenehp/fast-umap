use burn::tensor::{backend::AutodiffBackend, Tensor};

#[allow(unused)]
use crate::utils::print_tensor_with_title;

fn pairwise_distance<B: AutodiffBackend>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
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

    // print_tensor_with_title(
    //     Some("pairwise_squared_distances"),
    //     &pairwise_squared_distances,
    // );
    // Use `flatten()` to convert the upper triangular part (excluding the diagonal) into a 1D tensor
    let pairwise_distances = pairwise_squared_distances.triu(0); // Extract the upper triangular part (without diagonal)
                                                                 // print_tensor_with_title(Some("pairwise_distances"), &pairwise_distances);

    // Extract the first column (distances from the first sample to all others)
    let distances = pairwise_distances
        .slice([0..n_samples, 0..1])
        .reshape([n_samples]);
    // print_tensor_with_title(Some("distances"), &distances);

    distances
}

/// Calculate the UMAP loss by comparing pairwise distances between global and local representations
pub fn umap_loss<B: AutodiffBackend>(
    global: &Tensor<B, 2>, // High-dimensional (global) representation
    local: &Tensor<B, 2>,  // Low-dimensional (local) representation
) -> Tensor<B, 0> {
    println!("umap_loss");
    // Compute pairwise distances for both global and local representations
    let global_distances = pairwise_distance(global);
    // print_tensor_with_title(Some("global_distances"), &global_distances);
    let local_distances = pairwise_distance(local);
    // print_tensor_with_title(Some("local_distances"), &local_distances);

    // Compute the loss as the Frobenius norm (L2 loss) between the pairwise distance matrices
    let difference = (global_distances - local_distances).powi_scalar(2).sum();
    print_tensor_with_title(Some("difference"), &difference);

    let difference: Tensor<B, 0> = difference.squeeze(0);

    difference
}
