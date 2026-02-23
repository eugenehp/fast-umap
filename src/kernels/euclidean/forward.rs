use super::kernel::*;
use crate::kernels::DEFAULT_CUBE_DIM;
use burn::tensor::{ops::FloatTensor, Shape};
use burn_cubecl::{
    kernel::into_contiguous, tensor::CubeTensor, BoolElement, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::prelude::*;

/// Forward pass: compute the `[n, n]` Euclidean pairwise distance matrix.
///
/// Allocates an `[n, n]` output buffer on the GPU and launches
/// [`euclidean_pairwise_distance_kernel`] with a 2-D grid of
/// `⌈n/32⌉ × ⌈n/32⌉` work-groups, one thread per `(row, col)` pair.
/// The kernel writes both `output[row, col]` and the symmetric entry
/// `output[col, row]` from the upper-triangle threads.
///
/// # Arguments
///
/// * `x` — Input tensor of shape `[n, d]` on the target CubeBackend device.
///
/// # Returns
///
/// A contiguous `[n, n]` float tensor of pairwise Euclidean distances.
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
        x.as_tensor_arg(1),
        output.as_tensor_arg(1),
    )
    .expect("euclidean_pairwise_distance_kernel launch failed");

    #[cfg(feature = "verbose")]
    {
        // println!("euclidean_pairwise_distance[{n}] x - {x:?}");
        // println!("euclidean_pairwise_distance[{n}] output - {output:?}");
    }

    output
}

/// Backward pass for euclidean pairwise distances.
///
/// Dispatches one GPU thread per (row, feature) output element of grad_x.
/// Each thread iterates over all columns — O(n) work — to accumulate the
/// gradient contributions from both `pairwise[row, col]` and the symmetric
/// entry `pairwise[col, row]`, using the precomputed distance to avoid
/// recomputing O(d) inner products per pair.
///
/// The kernel uses `=` (write-once), so no zero-initialisation is needed.
pub fn backward<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    grad_pairwise: FloatTensor<CubeBackend<R, F, I, BT>>, // ∂loss/∂pairwise_distances [n,n]
    x: FloatTensor<CubeBackend<R, F, I, BT>>,             // original input x            [n,d]
    pairwise: FloatTensor<CubeBackend<R, F, I, BT>>,      // precomputed distances       [n,n]
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let x = into_contiguous(x);
    let pairwise = into_contiguous(pairwise);
    let grad_pairwise = into_contiguous(grad_pairwise);
    let n = x.shape.dims[0];
    let d = x.shape.dims[1];

    // Output: ∂loss/∂x, shape [n, d].  Kernel writes with `=`, no init needed.
    let grad_x_shape = Shape::from(vec![n, d]);
    let buffer = x
        .client
        .empty(grad_x_shape.num_elements() * std::mem::size_of::<F>());
    let grad_x: CubeTensor<R> = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        grad_x_shape,
        buffer,
        F::dtype(),
    );

    // Dispatch one thread per (sample, feature) = n×d grid.
    // ABSOLUTE_POS_X → sample row, ABSOLUTE_POS_Y → feature column.
    let cube_dim = DEFAULT_CUBE_DIM;
    let cubes_x = (n as f32 / cube_dim.x as f32).ceil() as u32;
    let cubes_y = (d as f32 / cube_dim.y as f32).ceil() as u32;
    let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);

    euclidean_pairwise_distance_backward_kernel::launch::<F, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg(1),
        pairwise.as_tensor_arg(1),
        grad_pairwise.as_tensor_arg(1),
        grad_x.as_tensor_arg(1),
    )
    .expect("euclidean_pairwise_distance_backward_kernel launch failed");

    grad_x
}
