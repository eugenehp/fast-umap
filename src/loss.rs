use burn::tensor::{backend::AutodiffBackend, Tensor};

use crate::utils::{print_tensor, print_tensor_with_title};

// /// Compute pairwise Euclidean distance matrix
// fn pairwise_distance<B: AutodiffBackend>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
//     println!("pairwise_distance - 1");
//     let sq_x = x.clone().powi_scalar(2); //.sum().unsqueeze::<1>(); // sum of squares along the rows
//     print_tensor(&sq_x);
//     println!("pairwise_distance - 2");
//     let sq_x_t = sq_x.clone().transpose(); // Transpose the sum for pairwise computation
//     println!("pairwise_distance - 3");
//     let transposed = x.clone().transpose();
//     println!("pairwise_distance - 4");
//     let product = x.clone().matmul(transposed);
//     println!("pairwise_distance - 5");
//     let dist = sq_x + sq_x_t - product.mul_scalar(2).squeeze(1); // Apply the squared Euclidean distance formula
//     println!("pairwise_distance - 6");
//     // dist.clamp_min(0.0).sqrt() // Take square root to get Euclidean distance
//     dist.unsqueeze::<1>()
// }

fn pairwise_distance<B: AutodiffBackend>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
    let n_samples = x.dims()[0];
    let mut dist = Tensor::zeros([n_samples * (n_samples - 1) / 2], &x.device()); // To store the flattened pairwise distances

    // Calculate pairwise squared Euclidean distances
    let mut idx = 0;
    for i in 0..n_samples {
        for j in i + 1..n_samples {
            // Compute squared Euclidean distance between points i and j
            let diff = x.clone().slice([i..i + 1]) - x.clone().slice([j..j + 1]); // Compute the difference between the i-th and j-th row (points)
            let dist_ij = diff.powi_scalar(2).sum(); // Sum of squares of differences (Euclidean squared distance)
                                                     // dist[idx] = dist_ij; // Store the distance in the flattened tensor
            dist = dist.slice_assign([idx..idx + 1], dist_ij);
            idx += 1;
        }
    }

    dist // Return the tensor containing pairwise distances
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
