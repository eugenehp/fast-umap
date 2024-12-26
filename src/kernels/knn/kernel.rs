use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn knn_kernel<F: Float + CubePrimitive>(
    pairwise_distances: &Tensor<F>, // Pairwise distance matrix (n, n)
    k: u32,                         // Number of nearest neighbors to find
    indices: &mut Tensor<F>, // Output tensor of shape (n, k) storing the indices of k nearest neighbors
    distances: &mut Tensor<F>, // Output tensor of shape (n, k) storing the distances of k nearest neighbors
) {
    let row = ABSOLUTE_POS_X; // Row index for the pairwise computation
    let n = pairwise_distances.shape(0); // Number of vectors

    // Edge case: skip if the row is out of bounds
    if row >= n {
        return;
    }

    // Pre-allocate arrays to store the k smallest distances and corresponding indices
    let mut local_distances = Array::<F>::new(k); // Array for storing k smallest distances
    let mut local_indices = Array::<u32>::new(k); // Array for storing k smallest indices

    // Initialize arrays with values that will be replaced by actual data
    for i in 0..k {
        local_distances[i] = F::INFINITY; // Set to infinity initially
        local_indices[i] = n as u32; // Set to an invalid index initially
    }

    // Iterate through all the pairwise distances for the current row
    for col in 0..n {
        if row != col {
            // Skip self-comparison
            let dist = pairwise_distances[row * n + col];

            // Find where to insert this distance in the sorted array of top-k distances
            if dist < local_distances[k - 1] {
                // Shift larger distances to make space for the new distance
                let mut inserted = false;
                for i in (0..k).rev() {
                    if dist < local_distances[i] {
                        if i < k - 1 {
                            local_distances[i + 1] = local_distances[i];
                            local_indices[i + 1] = local_indices[i];
                        }
                        local_distances[i] = dist;
                        local_indices[i] = col as u32;
                        inserted = true;
                        break;
                    }
                }
                // If the distance is greater than the largest, it won't be inserted
            }
        }
    }

    // Copy the results from local arrays into the output tensors
    for i in 0..k {
        distances[row * k + i] = local_distances[i];
        indices[row * k + i] = local_indices[i];
    }
}
