use super::{kernel::euclidean_pairwise_distance_kernel, Backend};
use burn::tensor::{ops::FloatTensor, Shape};
use burn_jit::{
    kernel::into_contiguous, tensor::JitTensor, FloatElement, IntElement, JitBackend, JitRuntime,
};
use cubecl::{CubeCount, CubeDim};

impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        // Define cube dim, hardcoded for simplicity.
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        // Ensure the input tensor is contiguous.
        let x = into_contiguous(x);

        let n = x.shape.num_dims();
        let num_samples = x.shape.dims[n - 2]; // Number of rows (samples)

        // The output is a 1D tensor with size N * (N - 1) / 2
        let output_size = num_samples * (num_samples - 1) / 2;
        let output_shape = Shape::from(vec![output_size]);

        // Create a buffer for the output tensor.
        let buffer = x
            .client
            .empty(output_shape.num_elements() * core::mem::size_of::<F>());

        // Create the output tensor primitive.
        let output =
            JitTensor::new_contiguous(x.client.clone(), x.device.clone(), output_shape, buffer);

        // Define the number of cubes in x, y, and z.
        let cubes_needed_in_x = f32::ceil(num_samples as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_samples as f32 / cube_dim.y as f32) as u32;
        let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1); // Only one batch.

        // Launch the kernel to compute pairwise distances.
        euclidean_pairwise_distance_kernel::launch::<F, R>(
            &x.client,
            cube_count,
            cube_dim,
            x.as_tensor_arg(1),
            output.as_tensor_arg(1),
        );

        // Return the output tensor (pairwise distance matrix).
        output
    }
}

// impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
//     fn euclidean_pairwise_distance(a: FloatTensor<Self>) -> FloatTensor<Self> {
//         let client = a.clone().client;
//         let device = a.clone().device;

//         // Ensure the tensor is contiguous in memory
//         let a_contiguous = into_contiguous(a);

//         // println!("a_contiguous {:?}", a_contiguous.shape);

//         // Get the number of samples and features
//         let n_samples = a_contiguous.shape.num_dims() - 1;
//         let _n_features = a_contiguous.shape.dims[n_samples];

//         let size = n_samples * (n_samples - 1) / 2;
//         // if input shape has only 1 sample, i.e. [1,100], where n_samples = 1, and n_features = 100
//         let size = match size {
//             0 => 1,
//             _ => size,
//         };

//         // Allocate memory for the output tensor (1D tensor of pairwise distances)
//         let shape_out = Shape::from(vec![size]);
//         let buffer_out = client
//             .clone()
//             .empty(shape_out.num_elements() * core::mem::size_of::<F>());

//         // println!("shape_out - {:?}", shape_out);

//         // Create the output tensor primitive.
//         let output =
//             JitTensor::new_contiguous(client.clone(), device.clone(), shape_out, buffer_out);

//         // Define the workgroup size (adjust based on hardware capabilities)
//         let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

//         // Calculate how many cubes are needed for the pairwise computation
//         let cubes_needed_in_x = (n_samples as f32 / cube_dim.x as f32).ceil() as u32;
//         let cubes_needed_in_y = (n_samples as f32 / cube_dim.y as f32).ceil() as u32;
//         let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

//         // Launch the kernel to compute the pairwise Euclidean distances
//         euclidean_pairwise_distance_kernel::launch::<F, R>(
//             &client,
//             cube_count,
//             cube_dim,
//             a_contiguous.as_tensor_arg(1),
//             output.as_tensor_arg(1),
//         );

//         // Return the output tensor with the pairwise distances
//         output

//         // Tensor::from_primitive(TensorPrimitive::Float(output))
//     }
// }
