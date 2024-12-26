use core::f32;

use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn knn_kernel<F: Float + CubePrimitive>(
    x: &Tensor<F>, // Input tensor of shape (n, d) representing n vectors of dimension d
    pairwise_distances: &Tensor<F>, // Pairwise distance matrix (n, n)
    k: u32,        // Number of nearest neighbors to find
    indices: &mut Tensor<F>, // Output tensor of shape (n, k) storing the indices of k nearest neighbors
    distances: &mut Tensor<F>, // Output tensor of shape (n, k) storing the distances of k nearest neighbors
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let n = x.shape(0); // Number of vectors

    // Edge case: skip if the row is out of bounds
    if row >= n {
        return;
    }

    // Pre-allocate arrays to store the k smallest distances and corresponding indices (as F)
    let mut local_distances = Array::<F>::new(k); // Array for storing k smallest distances
    let mut local_indices = Array::<F>::new(k); // Array for storing k smallest indices

    // Initialize arrays with values that will be replaced by actual data
    for i in 0..k {
        // local_distances[i] = F::INFINITY; // <-- this breaks for some reason
        local_distances[i] = F::new(f32::INFINITY);
        local_indices[i] = F::from_int(k as i64);
        // Set to an invalid index (as F)
    }

    // Iterate through all the pairwise distances for the current row
    for col in 0..n {
        if row != col {
            // Skip self-comparison
            let dist = pairwise_distances[row * n + col];

            // Find where to insert this distance in the sorted array of top-k distances
            if dist < local_distances[k - 1] {
                // Shift larger distances to make space for the new distance
                // let mut inserted = false;

                // Manually iterate backwards through the array using u32 indices
                let mut i = k - 1; // Start from the last index

                while i > 0 {
                    if dist < local_distances[i] {
                        // Shift the larger distances one step to the right
                        local_distances[i] = local_distances[i - 1];
                        local_indices[i] = local_indices[i - 1];
                    } else {
                        break;
                    }
                    i -= 1; // Move to the previous index
                }

                // Insert the new distance at the correct position
                local_distances[i] = dist;
                local_indices[i] = F::from_int(col as i64);
            }
        }
    }

    // Copy the results from local arrays into the output tensors
    for i in 0..k {
        distances[row * k + i] = local_distances[i];
        indices[row * k + i] = local_indices[i];
    }
}
