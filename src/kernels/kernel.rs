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
