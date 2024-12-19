use cubecl::{cube, prelude::*};

/// Euclidean pairwise distance kernel for a 2D tensor.
#[cube(launch)]
pub fn euclidean_pairwise_distance_kernel<F: Float>(x: &Tensor<F>, output: &mut Tensor<F>) {
    let row_i = ABSOLUTE_POS_X;
    let row_j = ABSOLUTE_POS_Y;

    let n = x.shape(x.rank() - 2); // Number of rows (samples)
    let d = x.shape(x.rank() - 1); // Dimensionality of each sample

    if row_i >= n || row_j >= n || row_i >= row_j {
        return; // Only compute the upper triangular part of the distance matrix
    }

    let mut sum = F::new(0.0);
    for k in 0..d {
        let diff = x[row_i * d + k] - x[row_j * d + k];
        sum += diff * diff;
    }

    // Store the square root of the sum of squared differences
    let index = (row_j * (row_j - 1)) / 2 + row_i; // Store in upper triangular form
    output[index] = F::sqrt(sum);
}

/// Euclidean pairwise distance kernel for a 2D tensor.
/// This kernel computes the sum of all pairwise Euclidean distances between rows in the input tensor.
#[cube(launch)]
pub fn euclidean_pairwise_distance_sum_kernel<F: Float>(x: &Tensor<F>, output: &mut Tensor<F>) {
    let row_i = ABSOLUTE_POS_X;
    let row_j = ABSOLUTE_POS_Y;

    let n = x.shape(x.rank() - 2); // Number of rows (samples)
    let d = x.shape(x.rank() - 1); // Dimensionality of each sample

    // Initialize the sum of pairwise distances
    let mut total_distance = F::new(0.0);

    // Only compute for pairs where i < j (upper triangular part)
    if row_i >= n || row_j >= n || row_i >= row_j {
        return; // Skip if out of bounds or on the diagonal
    }

    let mut sum = F::new(0.0);
    for k in 0..d {
        let diff = x[row_i * d + k] - x[row_j * d + k];
        sum += diff * diff;
    }

    // Add the square root of the sum of squared differences to the total distance
    total_distance += F::sqrt(sum);

    // Since this kernel computes only one scalar, store the result in the first position of the output tensor
    if row_i == 0 && row_j == 0 {
        output[0] = total_distance;
    }
}
