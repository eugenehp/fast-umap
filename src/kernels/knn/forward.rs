use super::kernel::*;
use crate::kernels::DEFAULT_CUBE_DIM;
use burn::tensor::{ops::FloatTensor, Shape};
use burn_jit::{
    kernel::into_contiguous, tensor::JitTensor, FloatElement, IntElement, JitBackend, JitRuntime,
};
use cubecl::prelude::*;

pub fn forward<R: JitRuntime, F: FloatElement, I: IntElement>(
    pairwise_distances: FloatTensor<JitBackend<R, F, I>>,
    k: u32,
) -> (
    FloatTensor<JitBackend<R, F, I>>,
    FloatTensor<JitBackend<R, F, I>>,
) {
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

    let indices = JitTensor::new_contiguous(
        client.clone(),
        device.clone(),
        indices_shape,
        indices_buffer,
    );
    let distances = JitTensor::new_contiguous(
        client.clone(),
        device.clone(),
        distances_shape,
        distances_buffer,
    );

    // Launch the k-NN kernel
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_needed_in_y = (k as f32 / cube_dim.y as f32).ceil() as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    // Launch the k-NN kernel
    knn_kernel::launch::<F, R>(
        &client,
        cube_count,
        cube_dim,
        pairwise_distances.as_tensor_arg(2), // Pairwise distance matrix
        ScalarArg::new(k),                   // Number of neighbors
        indices.as_tensor_arg(1),            // Indices tensor
        distances.as_tensor_arg(1),          // Distances tensor
    );

    (indices, distances)
}
