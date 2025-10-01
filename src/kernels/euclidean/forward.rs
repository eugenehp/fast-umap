use super::kernel::*;
use crate::kernels::DEFAULT_CUBE_DIM;
use burn::tensor::{ops::FloatTensor, Shape};
use burn_cubecl::{
    kernel::into_contiguous, tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::prelude::*;

pub fn forward<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    x: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let xx = into_contiguous(x.clone());
    let client = xx.client;
    let device = xx.device;
    let dims = xx.shape.dims;
    let n = dims[0];

    // Allocate output tensor of shape (N, N) to hold pairwise distances
    let output_shape = Shape::from(vec![n, n]);
    let buffer = client.empty(output_shape.num_elements() * std::mem::size_of::<F>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        output_shape,
        buffer,
        F::dtype(),
    );

    // Launch the Euclidean pairwise distance kernel
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_needed_in_y = (n as f32 / cube_dim.y as f32).ceil() as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    // Launch the kernel
    euclidean_pairwise_distance_kernel::launch::<F, R>(
        &client,
        cube_count,
        cube_dim,
        x.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
    );

    #[cfg(feature = "verbose")]
    {
        // println!("euclidean_pairwise_distance[{n}] x - {x:?}");
        // println!("euclidean_pairwise_distance[{n}] output - {output:?}");
    }

    output
}

pub fn backward<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    grad_x: FloatTensor<CubeBackend<R, F, I, BT>>,
    output: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    // println!("backend - euclidean_pairwise_distance_backward");
    let output = into_contiguous(output);
    let n = output.shape.dims[0];
    let d = output.shape.dims[1];

    let grad_output_shape = Shape::from(vec![n, d]);
    let buffer = output
        .client
        .empty(grad_output_shape.num_elements() * std::mem::size_of::<F>());
    let grad_output: CubeTensor<R> = CubeTensor::new_contiguous(
        output.client.clone(),
        output.device.clone(),
        grad_output_shape,
        buffer,
        F::dtype(),
    );

    // Launch the Euclidean pairwise distance kernel
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_needed_in_y = (d as f32 / cube_dim.y as f32).ceil() as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    let vectorisation = 1;

    // Launch the kernel
    euclidean_pairwise_distance_backward_kernel::launch::<F, R>(
        &output.client,
        cube_count,
        cube_dim,
        output.as_tensor_arg::<F>(vectorisation),
        grad_output.as_tensor_arg::<F>(vectorisation),
        grad_x.as_tensor_arg::<F>(vectorisation),
    );

    grad_output
}
