use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn euclidean_pairwise_distance_kernel<F: Float>(
    x: &Tensor<F>, // Input tensor of shape (n, d) representing n vectors of dimension d
    output: &mut Tensor<F>, // Output tensor of shape (n, n) to store pairwise distances
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let col = ABSOLUTE_POS_Y; // Column index for the pairwise computation

    let n = output.shape(output.rank() - 2); // Number of vectors (rows) in the output tensor
    let d = x.shape(x.rank() - 1); // Dimension of each vector (features) in the input tensor

    if row >= n || col >= n {
        return; // Skip threads that are out of bounds
    }

    let mut sum = F::new(0.0); // Sum of squared differences

    // Compute the squared differences between vectors row and col
    for i in 0..d {
        let index_row = row * d + i; // Linear index for row, dimension i
        let index_col = col * d + i; // Linear index for col, dimension i

        let diff = x[index_row] - x[index_col];
        sum += diff * diff;
    }

    // Calculate Euclidean distance (square root of sum of squared differences)
    let dist = F::sqrt(sum);

    // Linear index for the output tensor
    let output_index = row * n + col;

    // Store the pairwise Euclidean distance in the output tensor
    output[output_index] = dist;

    // Symmetry: dist(i, j) = dist(j, i)
    if row != col {
        // Avoid redundant assignments when row == col
        let output_index_sym = col * n + row;
        output[output_index_sym] = dist;
    }
}
