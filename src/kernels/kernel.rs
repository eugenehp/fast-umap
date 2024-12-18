use cubecl::{cube, prelude::*};

/// Euclidean pairwise distance kernel for a 2D tensor.
#[cube(launch)]
pub fn euclidean_pairwise_distance_kernel<F: Float>(
    a: &cubecl::prelude::Tensor<F>, // Input tensor of shape (n_samples, n_features)
    output: &mut cubecl::prelude::Tensor<F>, // Output tensor of shape (n_samples * (n_samples - 1) / 2)
) {
    let row = ABSOLUTE_POS_X; // Row index for the pair (i, j)
    let col = ABSOLUTE_POS_Y; // Column index for the pair (i, j)

    let n_samples = a.shape(a.rank() - 2); // Number of samples (n_samples)
    let n_features = a.shape(a.rank() - 1); // Number of features per sample (n_features)

    // Ensure that the row and column are within bounds
    if row >= n_samples || col >= n_samples || row >= col {
        return;
    }

    // Initialize the sum of squared differences
    let mut sum_sq_diff = F::new(0.0);

    // Compute the squared differences for each feature
    for f in 0..n_features {
        let diff = a[row * n_features + f] - a[col * n_features + f];
        sum_sq_diff += diff * diff;
    }

    // Calculate the Euclidean distance (square root of the sum of squared differences)
    let distance = F::sqrt(sum_sq_diff);

    // Compute the linear index for the output tensor
    let output_index = row * n_samples + col - (row * (row + 1)) / 2;

    // Store the computed distance in the output tensor
    output[output_index] = distance;
}
