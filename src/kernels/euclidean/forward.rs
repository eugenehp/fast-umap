use super::kernel::*;
use crate::backend::Backend;
use burn::tensor::{ops::FloatTensor, Shape};
use burn_jit::{
    kernel::into_contiguous, tensor::JitTensor, FloatElement, IntElement, JitBackend, JitRuntime,
};
use cubecl::prelude::*;

const DEFAULT_CUBE_DIM: CubeDim = CubeDim { x: 32, y: 32, z: 1 }; // Example cube size

impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        let xx = into_contiguous(x.clone());
        let client = xx.client;
        let device = xx.device;
        let dims = xx.shape.dims;
        let n = dims[0];

        // Allocate output tensor of shape (N, N) to hold pairwise distances
        let output_shape = Shape::from(vec![n, n]);
        let buffer = client.empty(output_shape.num_elements() * std::mem::size_of::<F>());
        let output =
            JitTensor::new_contiguous(client.clone(), device.clone(), output_shape, buffer);

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
            x.as_tensor_arg(1),
            output.as_tensor_arg(1),
        );

        output
    }

    fn euclidean_pairwise_distance_backward(
        grad_x: FloatTensor<Self>,
        output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // println!("backend - euclidean_pairwise_distance_backward");
        let output = into_contiguous(output);
        let n = output.shape.dims[0];
        let d = output.shape.dims[1];

        let grad_output_shape = Shape::from(vec![n, d]);
        let buffer = output
            .client
            .empty(grad_output_shape.num_elements() * std::mem::size_of::<F>());
        let grad_output: JitTensor<R, F> = JitTensor::new_contiguous(
            output.client.clone(),
            output.device.clone(),
            grad_output_shape,
            buffer,
        );

        // Launch the Euclidean pairwise distance kernel
        let cube_dim = DEFAULT_CUBE_DIM;
        let cubes_needed_in_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
        let cubes_needed_in_y = (d as f32 / cube_dim.y as f32).ceil() as u32;
        let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

        let clip_value = 1e-20;
        let clip_value = ScalarArg::new(F::new(clip_value));

        // Launch the kernel
        euclidean_pairwise_distance_backward_kernel::launch::<F, R>(
            &output.client,
            cube_count,
            cube_dim,
            output.as_tensor_arg(1),
            grad_output.as_tensor_arg(1),
            grad_x.as_tensor_arg(1),
            clip_value,
        );

        grad_output
    }
}
