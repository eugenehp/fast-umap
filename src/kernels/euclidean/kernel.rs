use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn euclidean_pairwise_distance_kernel<F: Float>(
    x: &Tensor<F>, // Input tensor of shape (n, d) representing n vectors of dimension d
    output: &mut Tensor<F>, // Output tensor of shape (n, n) to store pairwise distances
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let col = ABSOLUTE_POS_Y; // Column index for the pairwise computation

    let n = x.shape(0); // Number of vectors (rows) in the output tensor
    let d = x.shape(1); // Dimension of each vector (features) in the input tensor

    if row >= n || col >= n {
        return; // Skip threads that are out of bounds
    }

    // Edge case 1: Handle empty input tensor (n == 0 or d == 0)
    if n == 0 || d == 0 {
        return; // No computation needed for empty tensor
    }

    // Edge case 2: Handle single vector case (n == 1)
    if n == 1 {
        output[0] = F::new(0.0); // Distance between the only vector and itself is 0
        return;
    }

    // Edge case 3: Handle zero-dimensional vectors (d == 0)
    if d == 0 {
        // If vectors have 0 dimensions, the distance between any two vectors is trivially 0
        for i in 0..n {
            for j in i..n {
                output[i * n + j] = F::new(0.0);
                output[j * n + i] = F::new(0.0); // Symmetry: dist(i, j) = dist(j, i)
            }
        }
        return;
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

#[cube(launch)]
pub fn euclidean_pairwise_distance_backward_kernel<F: Float>(
    output: &Tensor<F>,          // Output tensor (n, d), pairwise distances
    grad_output: &mut Tensor<F>, // Gradient of the loss with respect to output tensor (n, d)
    grad_x: &Tensor<F>,          // Gradient of the loss with respect to input tensor (n, n)
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let col = ABSOLUTE_POS_Y; // Column index for the pairwise computation

    // Get the number of vectors (n) and the dimension (d) of each vector
    let n = output.shape(0); // Number of vectors (rows) in the input tensor
    let d = output.shape(1); // Dimension of each vector (features) in the input tensor

    // Edge case 1: Handle empty input tensor (n == 0 or d == 0)
    if n == 0 || d == 0 {
        return; // No computation needed for empty tensor
    }

    // Edge case 2: Handle zero-dimensional vectors (d == 0)
    if d == 0 {
        return; // grad_output should already be zeroed out
    }

    // Edge case: Ensure row and col are within bounds
    if row >= n || col >= n {
        return; // Skip threads that are out of bounds
    }

    // Get the pairwise distance between vectors row and col
    let dist = output[row * n + col];

    // Handle small distances (to avoid division by zero)
    let epsilon = F::cast_from(1e-8); // Define a small epsilon value
    let dist = F::max(dist, epsilon); // Ensure dist is never less than epsilon

    // Skip if the distance is 0 (identical vectors)
    if dist < epsilon {
        return; // No gradient to propagate for identical vectors
    }

    // Compute the gradient of the pairwise distance w.r.t the input vectors
    for i in 0..d {
        let index_row = row * d + i; // Linear index for row, dimension i
        let index_col = col * d + i; // Linear index for col, dimension i

        let diff = output[index_row] - output[index_col]; // Difference between the vectors

        // Gradient of the distance w.r.t x_{i,k}
        let grad_dist_i = grad_x[row * n + col] * (diff / dist); // Scale the gradient

        // Propagate the gradient to the input tensor
        // grad_output is the gradient of the loss with respect to input tensor
        grad_output[index_row] += grad_dist_i; // Gradient w.r.t row vector (x_i)
        grad_output[index_col] -= grad_dist_i; // Gradient w.r.t col vector (x_j)
    }
}
