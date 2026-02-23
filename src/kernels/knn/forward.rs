use super::kernel::*;
use crate::kernels::DEFAULT_CUBE_DIM;
use burn::tensor::{ops::FloatTensor, Shape};
use burn_cubecl::{
    kernel::into_contiguous, tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::prelude::*;

/// Forward pass: select the `k` nearest neighbours for every row.
///
/// Allocates `[n, k]` output buffers for indices and distances, plus `[n, k]`
/// scratch buffers used inside the kernel for the per-row insertion sort.
/// Launches [`knn_kernel`] with one GPU thread per row of the `[n, n]`
/// `pairwise_distances` matrix.
///
/// # Arguments
///
/// * `pairwise_distances` — Symmetric `[n, n]` float distance matrix.
/// * `k`                  — Number of nearest neighbours to select per row.
///
/// # Returns
///
/// A tuple `(indices, distances)`:
/// * `indices`   — `[n, k]` integer tensor of neighbour column indices.
/// * `distances` — `[n, k]` float tensor of corresponding distances.
pub fn forward<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    pairwise_distances: FloatTensor<CubeBackend<R, F, I, BT>>,
    k: u32,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let pairwise_distances = into_contiguous(pairwise_distances.clone());
    let client = pairwise_distances.client.clone();
    let device = pairwise_distances.device.clone();
    let dims = pairwise_distances.shape.dims.clone();
    let n = dims[0];

    // Allocate output tensors for indices and distances
    let indices_shape = Shape::from(vec![n, k as usize]);
    let distances_shape = Shape::from(vec![n, k as usize]);

    let indices_buffer = client.empty(indices_shape.num_elements() * std::mem::size_of::<F>());
    let distances_buffer = client.empty(distances_shape.num_elements() * std::mem::size_of::<F>());

    let indices: CubeTensor<R> = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        indices_shape,
        indices_buffer,
        burn::tensor::DType::I64,
    );
    let distances: CubeTensor<R> = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        distances_shape,
        distances_buffer,
        F::dtype(),
    );

    // Each of the n threads needs its own scratch space of k slots — so shape is [n, k].
    let local_shape = Shape::from(vec![n, k as usize]);

    let local_dist_buffer = pairwise_distances
        .client
        .empty(local_shape.num_elements() * std::mem::size_of::<F>());
    let local_idx_buffer = pairwise_distances
        .client
        .empty(local_shape.num_elements() * std::mem::size_of::<i64>());

    let local_distances: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        local_shape.clone(),
        local_dist_buffer,
        F::dtype(),
    );

    let local_indices: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        local_shape,
        local_idx_buffer,
        burn::tensor::DType::I64,
    );

    // One thread per row; y-dimension is unused (set to 1).
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_needed_in_y = 1_u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    let vectorisation = 1;

    // Launch the k-NN kernel
    knn_kernel::launch::<F, I, R>(
        &client,
        cube_count,
        cube_dim,
        pairwise_distances.as_tensor_arg(vectorisation), // Pairwise distance matrix
        ScalarArg::new(k),                               // Number of neighbors
        local_distances.as_tensor_arg(vectorisation),
        local_indices.as_tensor_arg(vectorisation),
        indices.as_tensor_arg(vectorisation), // Indices tensor
        distances.as_tensor_arg(vectorisation), // Distances tensor
    )
    .expect("knn_kernel launch failed");

    (indices, distances)
}

/// Backward pass: propagate gradients through the k-NN selection.
///
/// Re-runs the forward insertion sort (via [`knn_backward_kernel`]) to
/// recover which neighbours were selected, then propagates `grad_output`
/// (shape `[n, k]`) back to `grad_pairwise_distances` (shape `[n, n]`).
///
/// The output gradient buffer is zero-initialised because the kernel uses
/// `+=` to accumulate contributions.
///
/// # Arguments
///
/// * `pairwise_distances` — Symmetric `[n, n]` distance matrix from the forward pass.
/// * `k`                  — Number of nearest neighbours.
/// * `grad_output`        — Upstream gradient `∂loss/∂knn_distances`, shape `[n, k]`.
///
/// # Returns
///
/// `∂loss/∂pairwise_distances`, shape `[n, n]`.
pub fn backward<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    pairwise_distances: FloatTensor<CubeBackend<R, F, I, BT>>,
    k: u32,
    grad_output: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    // Convert the output tensor to a contiguous format for efficient access
    let pairwise_distances = into_contiguous(pairwise_distances);
    let n = pairwise_distances.shape.dims[0]; // Number of vectors
    let grad_output_shape = Shape::from(vec![n, k as usize]); // Gradient output shape for k neighbors

    // grad_pairwise_distances uses += in the kernel, so must be zero-initialised.
    let zero_bytes = vec![0u8; grad_output_shape.num_elements() * std::mem::size_of::<F>()];
    let grad_buffer = pairwise_distances.client.create_from_slice(&zero_bytes);

    let grad_pairwise_distances: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        pairwise_distances.shape.clone(),
        grad_buffer,
        F::dtype(),
    );

    // Per-row scratch space of k slots — shape [n, k].
    let local_shape = Shape::from(vec![n, k as usize]);

    let local_dist_buffer = pairwise_distances
        .client
        .empty(local_shape.num_elements() * std::mem::size_of::<F>());
    let local_idx_buffer = pairwise_distances
        .client
        .empty(local_shape.num_elements() * std::mem::size_of::<F>());

    let local_distances: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        local_shape.clone(),
        local_dist_buffer,
        F::dtype(),
    );

    let local_indices: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        local_shape,
        local_idx_buffer,
        F::dtype(),
    );

    // One thread per row.
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_needed_in_y = 1_u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    let vectorization = 1; // Use 1 for no vectorization

    // Launch the KNN backward kernel
    knn_backward_kernel::launch::<F, R>(
        &pairwise_distances.client,
        cube_count,
        cube_dim,
        pairwise_distances.as_tensor_arg(vectorization),
        ScalarArg::new(k), // Pass the value of k as an argument
        local_distances.as_tensor_arg(vectorization),
        local_indices.as_tensor_arg(vectorization),
        grad_output.as_tensor_arg(vectorization),
        grad_pairwise_distances.as_tensor_arg(vectorization),
    )
    .expect("knn_backward_kernel launch failed");

    // Return the gradient of the pairwise distances
    grad_pairwise_distances
}
