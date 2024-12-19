use super::{kernel::euclidean_pairwise_distance_kernel, Backend};
use burn::tensor::{ops::FloatTensor, Shape};
use burn_jit::{
    kernel::into_contiguous, tensor::JitTensor, FloatElement, IntElement, JitBackend, JitRuntime,
};
use cubecl::{CubeCount, CubeDim};

impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
    fn euclidean_pairwise_distance(a: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = a.clone().client;
        let device = a.clone().device;

        // Ensure the tensor is contiguous in memory
        let a_contiguous = into_contiguous(a);

        // println!("a_contiguous {:?}", a_contiguous.shape);

        // Get the number of samples and features
        let n_samples = a_contiguous.shape.num_dims() - 1;
        let _n_features = a_contiguous.shape.dims[n_samples];

        let size = n_samples * (n_samples - 1) / 2;
        // if input shape has only 1 sample, i.e. [1,100], where n_samples = 1, and n_features = 100
        let size = match size {
            0 => 1,
            _ => size,
        };

        // Allocate memory for the output tensor (1D tensor of pairwise distances)
        let shape_out = Shape::from(vec![size]);
        let buffer_out = client
            .clone()
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        // println!("shape_out - {:?}", shape_out);

        // Create the output tensor primitive.
        let output =
            JitTensor::new_contiguous(client.clone(), device.clone(), shape_out, buffer_out);

        // Define the workgroup size (adjust based on hardware capabilities)
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        // Calculate how many cubes are needed for the pairwise computation
        let cubes_needed_in_x = (n_samples as f32 / cube_dim.x as f32).ceil() as u32;
        let cubes_needed_in_y = (n_samples as f32 / cube_dim.y as f32).ceil() as u32;
        let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

        // Launch the kernel to compute the pairwise Euclidean distances
        euclidean_pairwise_distance_kernel::launch::<F, R>(
            &client,
            cube_count,
            cube_dim,
            a_contiguous.as_tensor_arg(1),
            output.as_tensor_arg(1),
        );

        // Return the output tensor with the pairwise distances
        output

        // Tensor::from_primitive(TensorPrimitive::Float(output))
    }
}
