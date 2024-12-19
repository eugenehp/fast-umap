use std::{fmt::Debug, ops::Range};

use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
            NodeID,
        },
        Autodiff,
    },
    tensor::{ops::FloatTensor, Float, Tensor, TensorPrimitive},
};

use super::Backend;

// Implement the backward operation for the `euclidean_pairwise_distance` function.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct EuclideanPairwiseDistanceBackward;

        impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
            type State = (NodeID, FloatTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                // Get the input tensor node
                let [node_x] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, output) = ops.state;
                let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

                // let shape = FloatTensorOps::float_shape(&x);
                let xx: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(x.clone()));
                let shape = xx.shape();

                let n = shape.dims[0];
                let d = shape.dims[1];

                // Compute the gradient of the pairwise Euclidean distance with respect to `x`
                let mut grad_x = B::float_zeros(shape, &Default::default());

                for i in 0..n {
                    for j in i + 1..n {
                        // Access the output tensor element for pair (i, j) using float_slice
                        let output_slice = B::float_slice(
                            output.clone(),
                            &[Range {
                                start: i * (n - 1) / 2 + j,
                                end: i * (n - 1) / 2 + j + 1,
                            }],
                        );

                        // Access the first element of the slice using float_slice
                        let dist =
                            B::float_slice(output_slice.clone(), &[Range { start: 0, end: 1 }]);

                        // Access the gradient tensor for the pair (i, j) using float_slice
                        let grad_slice = B::float_slice(
                            grad.clone(),
                            &[Range {
                                start: i * (n - 1) / 2 + j,
                                end: i * (n - 1) / 2 + j + 1,
                            }],
                        );

                        // Access the first element of the gradient slice using float_slice
                        let grad_val =
                            B::float_slice(grad_slice.clone(), &[Range { start: 0, end: 1 }]);

                        // Access input tensor slices for x[i] and x[j]
                        let x_i_slice = B::float_slice(
                            x.clone(),
                            &[Range {
                                start: i * d,
                                end: (i + 1) * d,
                            }],
                        );
                        let x_j_slice = B::float_slice(
                            x.clone(),
                            &[Range {
                                start: j * d,
                                end: (j + 1) * d,
                            }],
                        );

                        // Calculate the difference between x[i] and x[j]
                        let diff = B::float_sub(x_i_slice, x_j_slice); // Slicing results in a tensor, so take the first element

                        // Compute the gradient with respect to the pairwise distance
                        let grad_val = B::float_div(grad_val, dist);

                        // Update the gradients for x[i] and x[j] using `float_slice_assign`
                        grad_x = B::float_slice_assign(
                            grad_x,
                            &[Range {
                                start: i * d,
                                end: (i + 1) * d,
                            }],
                            B::float_mul(grad_val.clone(), diff.clone()),
                        );

                        grad_x = B::float_slice_assign(
                            grad_x,
                            &[Range {
                                start: j * d,
                                end: (j + 1) * d,
                            }],
                            B::float_mul(B::float_neg(grad_val), diff),
                        );
                    }
                }

                // Register the gradient for the input tensor
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x);
                }
            }
        }

        match EuclideanPairwiseDistanceBackward
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // If the input tensor is tracked, checkpoint its state and perform the backward pass
                let x_state = prep.checkpoint(&x);
                let output = B::euclidean_pairwise_distance(x.primitive.clone());

                let state = (x_state, output.clone());
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                // If no nodes are tracked, just compute the pairwise distance directly
                let output = B::euclidean_pairwise_distance(x.primitive.clone());
                prep.finish(output)
            }
        }
    }
}

// impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
//     fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
//         // Define a struct for the backward pass.
//         #[derive(Debug)]
//         struct EuclideanPairwiseDistanceBackward;

//         // Implement the backward trait for the given backend B
//         impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
//             type State = (NodeID, FloatTensor<B>, Shape);

//             fn backward(
//                 self,
//                 ops: Ops<Self::State, 1>,
//                 grads: &mut Gradients,
//                 checkpointer: &mut Checkpointer,
//             ) {
//                 // Retrieve the parent nodes
//                 let [node_x] = ops.parents;

//                 // Fetch the gradient for the current node
//                 let grad = grads.consume::<B>(&ops.node);

//                 // Retrieve the state
//                 let (x_state, output, shape_x) = ops.state;
//                 let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

//                 let x = Tensor::from_primitive(TensorPrimitive::Float(x.clone()));
//                 let shape = x.shape();
//                 let dims = shape.dims;
//                 let device = x.device();

//                 let zero = Tensor::zeros([1], device);

//                 // Compute the gradient for each pairwise distance entry
//                 let grad_output = grad.clone(); // Gradient for the output distances
//                 let grad_output =
//                     Tensor::from_primitive(TensorPrimitive::Float(grad_output.clone()));
//                 let grad_dims = grad_output.shape().dims;

//                 // Initialize the gradient tensor for x
//                 let mut grad_x = vec![0.0; (dims[0] * dims[1]) as usize];
//                 let n = dims[0];

//                 // Iterate over each pair (i, j)
//                 for i in 0..n {
//                     for j in i..n {
//                         // Use tensor operations to get rows x[i] and x[j]
//                         let xi = x.slice([i..i]);
//                         let xj = x.slice([j..j]);

//                         // Compute the difference between xi and xj
//                         let diff = xi - xj;

//                         // Compute the Euclidean distance (norm)
//                         let diff_squared = diff.powf_scalar(2.0); // Square each element in the diff tensor
//                         let sum_of_squares = diff_squared.sum(); // Sum of squared differences
//                         let distance = sum_of_squares.sqrt(); // Take the square root of the sum

//                         let eq = distance
//                             .equal(zero)
//                             .to_data()
//                             .to_vec::<bool>()
//                             .unwrap()
//                             .first()
//                             .unwrap();

//                         if !eq {
//                             // Calculate the gradient for the inputs x[i] and x[j]
//                             let grad_ij =
//                                 grad_output.slice([i..i + 1, 0..grad_dims[1]]) * diff / distance;
//                             let grad_ji =
//                                 grad_output.slice([j..j + 1, 0..grad_dims[1]]) * diff / distance;

//                             // Store gradients in grad_x_vec
//                             grad_x[i * n + j] += grad_ij.to_data().to_vec(); // Assumes grad_ij is a scalar
//                             grad_x[j * n + i] -= grad_ji[0]; // Assumes grad_ji is a scalar
//                         }
//                     }
//                 }

//                 // Convert the gradient to a tensor and register it for the input x
//                 let grad_tensor = FloatTensor::<B>::from_vec(grad_x, shape_x);
//                 if let Some(node) = node_x {
//                     grads.register::<B>(node.id, grad_tensor);
//                 }
//             }
//         }

//         // Prepare the operation and determine whether to track the backward pass
//         match EuclideanPairwiseDistanceBackward
//             .prepare::<C>([x.node.clone()])
//             .compute_bound()
//             .stateful()
//         {
//             OpsKind::Tracked(mut prep) => {
//                 let x_state = prep.checkpoint(&x);
//                 let output = B::euclidean_pairwise_distance(x.primitive.clone());
//                 let state = (x_state, output.clone(), x.primitive.shape());

//                 prep.finish(state, output)
//             }
//             OpsKind::UnTracked(prep) => {
//                 let output = B::euclidean_pairwise_distance(x.primitive);
//                 prep.finish(output)
//             }
//         }
//     }
// }
