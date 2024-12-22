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

#[cube(launch_unchecked)]
pub fn euclidean_pairwise_distance_backward_kernel<E: Numeric>(
    x: &Tensor<E>,               // Input tensor (n, d)
    output: &Tensor<E>,          // Output tensor (n, n), pairwise distances
    grad_output: &mut Tensor<E>, // Gradient of the loss with respect to output tensor (n, n)
    grad_x: &Tensor<E>,          // Gradient of the loss with respect to input tensor (n, d)
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let col = ABSOLUTE_POS_Y; // Column index for the pairwise computation

    // Get the number of vectors (n) and the dimension (d) of each vector
    let n = x.shape(0); // Number of vectors (rows) in the input tensor
    let d = x.shape(1); // Dimension of each vector (features) in the input tensor

    // Edge case 1: Handle empty input tensor (n == 0 or d == 0)
    if n == 0 || d == 0 {
        return; // No computation needed for empty tensor
    }

    // Edge case 2: Handle zero-dimensional vectors (d == 0)
    if d == 0 {
        return; // grad_x should already be zeroed out
    }

    // Edge case: Ensure row and col are within bounds
    if row >= n || col >= n {
        return; // Skip threads that are out of bounds
    }

    // Get the pairwise distance between vectors row and col
    let dist = output[row * n + col];

    // Handle small distances (to avoid division by zero)
    let epsilon = E::cast_from(1e-10); // Define a small epsilon value
    let dist = if dist < epsilon { epsilon } else { dist }; // Ensure dist is never less than epsilon

    // Skip if the distance is 0 (identical vectors)
    if dist == E::cast_from(0.0) {
        return; // No gradient to propagate for identical vectors
    }

    // Compute the gradient of the pairwise distance w.r.t the input vectors
    let grad_dist = grad_x[row * n + col]; // Gradient of the loss w.r.t dist(i, j)

    // Gradient of the distance with respect to the vectors (i, j)
    for i in 0..d {
        let index_row = row * d + i; // Linear index for row, dimension i
        let index_col = col * d + i; // Linear index for col, dimension i

        let diff = x[index_row] - x[index_col];
        let grad_dist_i = grad_dist * (diff / dist); // Gradient of distance w.r.t x_{i,k}

        // Propagate the gradient to the input tensor
        // Ensure grad_x has the correct shape: [n, d] (1000, 2)
        grad_output[row * d + i] += grad_dist_i; // Gradient w.r.t row vector (x_i)
        grad_output[col * d + i] -= grad_dist_i; // Gradient w.r.t col vector (x_j)
    }
}
