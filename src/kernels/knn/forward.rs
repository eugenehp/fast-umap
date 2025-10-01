use super::kernel::*;
use crate::kernels::DEFAULT_CUBE_DIM;
use burn::tensor::{ops::FloatTensor, Shape};
use burn_cubecl::{
    kernel::into_contiguous, tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::prelude::*;

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

    let local_shape = Shape::from(vec![k as usize]); // Local shape for k neighbors

    // Create the buffer and grad_pairwise_distances tensor
    let local_buffer = pairwise_distances
        .client
        .empty(local_shape.num_elements() * std::mem::size_of::<F>());

    let local_distances: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        pairwise_distances.shape.clone(),
        local_buffer.clone(),
        F::dtype(),
    );

    let local_indices: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        pairwise_distances.shape.clone(),
        local_buffer,
        burn::tensor::DType::I64,
    );

    // Launch the k-NN kernel
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_needed_in_y = (k as f32 / cube_dim.y as f32).ceil() as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    let vectorisation = 1;

    // Launch the k-NN kernel
    knn_kernel::launch::<F, I, R>(
        &client,
        cube_count,
        cube_dim,
        pairwise_distances.as_tensor_arg::<F>(vectorisation), // Pairwise distance matrix
        ScalarArg::new(k),                                    // Number of neighbors
        local_distances.as_tensor_arg::<F>(vectorisation),
        local_indices.as_tensor_arg::<I>(vectorisation),
        indices.as_tensor_arg::<I>(vectorisation), // Indices tensor
        distances.as_tensor_arg::<F>(vectorisation), // Distances tensor
    );

    (indices, distances)
}

pub fn backward<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    pairwise_distances: FloatTensor<CubeBackend<R, F, I, BT>>, // Pairwise distance matrix (n, n)
    k: u32,                                                    // Number of nearest neighbors
    grad_output: FloatTensor<CubeBackend<R, F, I, BT>>, // Gradient of the loss w.r.t the output
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    // Convert the output tensor to a contiguous format for efficient access
    let pairwise_distances = into_contiguous(pairwise_distances);
    let n = pairwise_distances.shape.dims[0]; // Number of vectors
    let grad_output_shape = Shape::from(vec![n, k as usize]); // Gradient output shape for k neighbors

    // Create the buffer and grad_pairwise_distances tensor
    let buffer = pairwise_distances
        .client
        .empty(grad_output_shape.num_elements() * std::mem::size_of::<F>());

    let grad_pairwise_distances: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        pairwise_distances.shape.clone(),
        buffer,
        F::dtype(),
    );

    let local_shape = Shape::from(vec![k as usize]); // Local shape for k neighbors

    // Create the buffer and grad_pairwise_distances tensor
    let local_buffer = pairwise_distances
        .client
        .empty(local_shape.num_elements() * std::mem::size_of::<F>());

    let local_distances: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        pairwise_distances.shape.clone(),
        local_buffer.clone(),
        F::dtype(),
    );

    let local_indices: CubeTensor<R> = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        pairwise_distances.shape.clone(),
        local_buffer,
        F::dtype(),
    );

    // Calculate the number of blocks needed for the kernel launch
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_needed_in_y = (n as f32 / cube_dim.y as f32).ceil() as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    let vectorization = 1; // Use 1 for no vectorization

    // Launch the KNN backward kernel
    knn_backward_kernel::launch::<F, R>(
        &pairwise_distances.client,
        cube_count,
        cube_dim,
        pairwise_distances.as_tensor_arg::<F>(vectorization),
        ScalarArg::new(k), // Pass the value of k as an argument
        local_distances.as_tensor_arg::<F>(vectorization),
        local_indices.as_tensor_arg::<I>(vectorization),
        grad_output.as_tensor_arg::<F>(vectorization),
        grad_pairwise_distances.as_tensor_arg::<F>(vectorization),
    );

    // Return the gradient of the pairwise distances
    grad_pairwise_distances
}
