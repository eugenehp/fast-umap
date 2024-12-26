use core::f32;
use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn knn_kernel<F: Float + CubePrimitive>(
    pairwise_distances: &Tensor<F>,  // Pairwise distance matrix (n, n)
    k: u32,                          // Number of nearest neighbors to find
    local_distances: &mut Tensor<F>, // for local distances storage, size of k
    local_indices: &mut Tensor<F>,   // for local indices storage, size of k
    indices: &mut Tensor<F>, // Output tensor of shape (n, k) storing the indices of k nearest neighbors
    distances: &mut Tensor<F>, // Output tensor of shape (n, k) storing the distances of k nearest neighbors
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let n = pairwise_distances.shape(0); // Number of vectors

    // Edge case: skip if the row is out of bounds
    if row >= n {
        return;
    }

    // Pre-allocate arrays to store the k smallest distances and corresponding indices (as F)
    // let mut local_distances = Array::<F>::new(k); // Array for storing k smallest distances
    // let mut local_indices = Array::<F>::new(k); // Array for storing k smallest indices

    // Initialize arrays with values that will be replaced by actual data
    for i in 0..k {
        // Initialize distances to infinity and indices to an invalid value
        local_distances[i] = F::new(f32::INFINITY); // Use F::infinity() to represent infinity
        local_indices[i] = F::from_int(k as i64); // Set to an invalid index (out of range)
    }

    // Iterate through all the pairwise distances for the current row
    for col in 0..n {
        if row != col {
            // Skip self-comparison
            let dist = pairwise_distances[row * n + col];

            // Find where to insert this distance in the sorted array of top-k distances
            if dist < local_distances[k - 1] {
                let mut i = k - 1; // Start from the last index

                // Shift larger distances one step to the right to make space for the new distance
                while i > 0 {
                    if dist < local_distances[i] {
                        local_distances[i] = local_distances[i - 1];
                        local_indices[i] = local_indices[i - 1];
                    } else {
                        break;
                    }
                    i -= 1; // Move to the previous index
                }

                // Insert the new distance at the correct position
                local_distances[i] = dist;
                local_indices[i] = F::from_int(col as i64); // Store the corresponding index
            }
        }
    }

    // Copy the results from local arrays into the output tensors
    for i in 0..k {
        distances[row * k + i] = local_distances[i];
        indices[row * k + i] = local_indices[i];
    }
}

#[cube(launch)]
pub fn knn_backward_kernel<F: Float + CubePrimitive>(
    pairwise_distances: &Tensor<F>,  // Pairwise distance matrix (n, n)
    k: u32,                          // Number of nearest neighbors to find
    local_distances: &mut Tensor<F>, // for local distances storage, size of k
    local_indices: &mut Tensor<F>,   // for local indices storage, size of k
    grad_output: &Tensor<F>, // Gradient of the loss w.r.t the output (distances and indices)
    grad_pairwise_distances: &mut Tensor<F>, // Gradient of the loss w.r.t the input (pairwise distances)
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let n = pairwise_distances.shape(0); // Number of vectors

    // Edge case: skip if the row is out of bounds
    if row >= n {
        return;
    }

    // Pre-allocate arrays to store the k smallest distances and corresponding indices
    // let mut local_distances = Array::<F>::new(k); // Array for storing k smallest distances
    // let mut local_indices = Array::<F>::new(k); // Array for storing k smallest indices

    // Initialize arrays with values that will be replaced by actual data
    for i in 0..k {
        local_distances[i] = F::new(f32::INFINITY); // Use F::infinity() to represent infinity
        local_indices[i] = F::from_int(k as i64); // Set to an invalid index (out of range)
    }

    // Retrieve k nearest neighbors' indices and distances for the current row
    for col in 0..n {
        if row != col {
            // Skip self-comparison
            let dist = pairwise_distances[row * n + col];

            // Find where to insert this distance in the sorted array of top-k distances
            if dist < local_distances[k - 1] {
                let mut i = k - 1; // Start from the last index

                // Shift larger distances one step to the right to make space for the new distance
                while i > 0 {
                    if dist < local_distances[i] {
                        local_distances[i] = local_distances[i - 1];
                        local_indices[i] = local_indices[i - 1];
                    } else {
                        break;
                    }
                    i -= 1; // Move to the previous index
                }

                // Insert the new distance at the correct position
                local_distances[i] = dist;
                local_indices[i] = F::from_int(col as i64); // Store the corresponding index
            }
        }
    }

    // Compute gradients with respect to the pairwise distances
    for i in 0..k {
        let neighbor_index = local_indices[i]; // Get the index of the neighbor (column)
        let grad_value = grad_output[row * k + i]; // Get the gradient from the output tensor

        // TODO: once we move indices to IntTensor, refactor the types
        let neighbor_index: u32 = u32::cast_from(neighbor_index);

        // If grad_value is non-zero, propagate the gradient to the pairwise distance
        if grad_value != F::new(0.0) {
            let dist = local_distances[i]; // The distance between row and neighbor_index
            let epsilon = F::new(1e-8); // Small epsilon to avoid division by zero

            // To avoid division by zero, we ensure that the distance is never too small
            let dist = F::max(dist, epsilon);

            // Gradient of the pairwise distance with respect to the input tensor
            // The gradient is proportional to the inverse of the distance
            let grad_pairwise = grad_value / dist;

            // Propagate the gradient back to the pairwise distance matrix
            grad_pairwise_distances[row * n + neighbor_index] += grad_pairwise;
            grad_pairwise_distances[neighbor_index * n + row] += grad_pairwise; // Symmetry
        }
    }
}
