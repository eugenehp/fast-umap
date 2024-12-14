use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute pairwise Euclidean distance matrix
fn pairwise_distance<B: AutodiffBackend>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
    let sq_x = x.clone().powi_scalar(2).sum().unsqueeze::<1>(); // sum of squares along the rows
    let sq_x_t = sq_x.clone().transpose(); // Transpose the sum for pairwise computation
    let transposed = x.clone().transpose();
    let product = x.clone().matmul(transposed);
    let dist = sq_x + sq_x_t - product.mul_scalar(2).squeeze(1); // Apply the squared Euclidean distance formula
    dist.clamp_min(0.0).sqrt() // Take square root to get Euclidean distance
}

/// Calculate the UMAP loss by comparing pairwise distances between global and local representations
pub fn umap_loss<B: AutodiffBackend>(
    global: &Tensor<B, 2>, // High-dimensional (global) representation
    local: &Tensor<B, 2>,  // Low-dimensional (local) representation
) -> Tensor<B, 0> {
    // Compute pairwise distances for both global and local representations
    let global_distances = pairwise_distance(global);
    let local_distances = pairwise_distance(local);

    // Compute the loss as the Frobenius norm (L2 loss) between the pairwise distance matrices
    (global_distances - local_distances)
        .powi_scalar(2)
        .sum()
        .unsqueeze()
}
