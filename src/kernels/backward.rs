use std::fmt::Debug;

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
    tensor::{ops::FloatTensor, ElementConversion, Float, Shape, Tensor, TensorPrimitive},
};

use super::Backend;

const VERBOSE: bool = false;

// Implement our custom backend trait for any backend that also implements our custom backend trait.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        // Create zero-sized struct for backward computation
        #[derive(Debug)]
        struct EuclideanPairwiseDistanceBackward;

        // Implement the backward trait
        impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
            type State = (NodeID, FloatTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                // Get the nodes of each variable.
                let [node_x] = ops.parents;
                // Fetch the gradient for the current node.
                let grad = grads.consume::<B>(&ops.node);

                // Set our state.
                let (x_state, output) = ops.state;
                let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

                // Convert x to a 2D tensor (this step is important for broadcasting support)
                let xx: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(x.clone()));

                if VERBOSE {
                    println!("=========");
                    println!("x - {x:?}");
                    println!("grad - {grad:?}");
                    println!("output - {output:?}");
                    println!("=========");
                }

                // Fetch shapes of our tensor to support broadcasting.
                let shape_x = xx.shape();

                if VERBOSE {
                    println!("backward - 1");
                }

                // Compute the gradient of the output with respect to x (derivative of Euclidean distance).
                // Euclidean distance gradient with respect to x is: grad_output * (x / ||x||) (simplified form).
                // let grad_output = B::float_mul(grad, B::float_div(x.clone(), output)); // This will need adjustment based on your Euclidean implementation.

                // Ensure grad_output matches the shape of x
                let grad_output = B::float_mul(grad, B::float_div(x.clone(), output)); // Adjust this as needed for proper gradient computation

                if VERBOSE {
                    println!("backward - 2 - grad_output - {grad_output:?}");
                }

                // Reshape the grad_output if needed to match x's shape or the expected pairwise distance shape.
                let reshaped_grad_output = B::float_reshape(grad_output, shape_x.clone()); // Reshaping grad_output to match x's shape

                // Ensure that you reshape x for matrix multiplication correctly
                let reshaped_x = B::float_reshape(x.clone(), Shape::from([1000, 2])); // Adjust shape as necessary

                // Matrix multiplication of the gradient with the reshaped input tensor.
                let tensor = B::float_matmul(
                    reshaped_grad_output.clone(),
                    B::float_reshape(reshaped_x, Shape::from([1000, 2])),
                );

                if VERBOSE {
                    println!("backward - 3");
                }

                // Compute the lhs gradient, which is the derivative of matmul with support for broadcasting.
                let grad_x = B::float_reshape(tensor, shape_x);

                if VERBOSE {
                    println!("backward - 4");
                }

                // Register the gradient for each variable based on whether they are marked as
                // `tracked`.
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x);
                }
            }
        }

        // Prepare the stateful operation
        match EuclideanPairwiseDistanceBackward
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                if VERBOSE {
                    println!("Tracked");
                }
                let x_state = prep.checkpoint(&x);
                let output = B::euclidean_pairwise_distance(x.primitive.clone());
                let state = (x_state, output.clone());
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                if VERBOSE {
                    println!("UnTracked");
                }
                let output = B::euclidean_pairwise_distance(x.primitive);
                prep.finish(output)
            }
        }
    }
}

// use std::fmt::Debug;

// use burn::{
//     backend::{
//         autodiff::{
//             checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
//             grads::Gradients,
//             ops::{Backward, Ops, OpsKind},
//             NodeID,
//         },
//         Autodiff,
//     },
//     tensor::{ops::FloatTensor, ElementConversion, Float, Int, Shape, Tensor, TensorPrimitive},
// };

// use super::Backend;

// const VERBOSE: bool = false;

// // Implement our custom backend trait for any backend that also implements our custom backend trait.
// impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
//     fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
//         // Create zero-sized struct for backward computation
//         #[derive(Debug)]
//         struct EuclideanPairwiseDistanceBackward;

//         // Implement the backward trait
//         impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
//             type State = (NodeID, FloatTensor<B>);

//             fn backward(
//                 self,
//                 ops: Ops<Self::State, 1>,
//                 grads: &mut Gradients,
//                 checkpointer: &mut Checkpointer,
//             ) {
//                 if VERBOSE {
//                     println!("backward start");
//                 }

//                 let [node_x] = ops.parents;
//                 let grad = grads.consume::<B>(&ops.node);

//                 let (x_state, output) = ops.state;
//                 let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

//                 println!("output - {output:?}");
//                 println!("grad - {grad:?}");

//                 // Ensure xx is a 2D tensor (shape: [n, d])
//                 let xx: Tensor<B, 2, Float> =
//                     Tensor::from_primitive(TensorPrimitive::Float(x.clone()));
//                 let shape_x = xx.shape();
//                 let dims = &shape_x.dims;

//                 let n = dims[0]; // Number of rows (samples)
//                 let d = dims[1]; // Dimensionality

//                 if VERBOSE {
//                     println!("xx shape: {:?}", xx.shape());
//                 }

//                 // Step 1: Compute pairwise difference (x_i - x_j)
//                 let xx_exp: Tensor<B, 3> = xx.clone().expand(Shape::from([n, n, d])); // Shape [n, n, d]
//                 let xx_exp_transpose: Tensor<B, 3> =
//                     xx.clone().expand(Shape::from([n, n, d])).transpose(); // Shape [n, n, d]

//                 let diff = B::float_sub(
//                     xx_exp.into_primitive().tensor(),
//                     xx_exp_transpose.into_primitive().tensor(),
//                 ); // Shape [n, n, d]

//                 if VERBOSE {
//                     println!("diff shape: {:?}", diff);
//                 }

//                 // Step 2: Compute squared L2 norm (Euclidean distance) for each pair
//                 let squared = B::float_mul(diff.clone(), diff.clone()); // Shape [n, n, d]
//                 let squared: Tensor<B, 3, Float> =
//                     Tensor::from_primitive(TensorPrimitive::Float(squared));

//                 let sum_of_squares =
//                     B::float_sum(squared.slice([0..n, 0..n, 0..d]).into_primitive().tensor()); // Shape [n, n]

//                 let dist = B::float_sqrt(sum_of_squares); // Shape [n, n]

//                 let epsilon = 1e-12.elem();
//                 let dist_safe = B::float_clamp_max(dist.clone(), epsilon); // Shape [n, n]

//                 // Step 3: Compute the gradient of the distance with respect to x[i] and x[j]
//                 let grad_diff = B::float_div(
//                     diff.clone(),
//                     B::float_expand(dist_safe, Shape::from([n, n, d])),
//                 ); // Normalize by Euclidean distance

//                 // Step 4: Compute the gradient of x
//                 // Gradient of x[i] is the sum over all j, but remember to reverse the direction for x[j]
//                 let grad_x = B::float_sum_dim(grad_diff.clone(), 1); // Summing over j for x[i]
//                 let grad_x_transpose = B::float_sum_dim(grad_diff, 0); // Summing over i for x[j]

//                 // Step 5: Register the gradients for both x[i] and x[j]
//                 let grad_x = B::float_reshape(grad_x, xx.shape()); // Reshape to match the original shape of x
//                 let grad_x_transpose = B::float_reshape(grad_x_transpose, xx.shape()); // Reshape to match the original shape of x

//                 if let Some(node) = node_x {
//                     grads.register::<B>(node.id, grad_x.clone());
//                     grads.register::<B>(node.id, grad_x_transpose.clone());
//                 }

//                 if VERBOSE {
//                     println!("backward end {grad_x:?}");
//                 }
//             }
//         }

//         // Prepare the stateful operation
//         match EuclideanPairwiseDistanceBackward
//             .prepare::<C>([x.node.clone()])
//             .compute_bound()
//             .stateful()
//         {
//             OpsKind::Tracked(mut prep) => {
//                 if VERBOSE {
//                     println!("Tracked");
//                 }
//                 let x_state = prep.checkpoint(&x);
//                 let output = B::euclidean_pairwise_distance(x.primitive.clone());
//                 let state = (x_state, output.clone());
//                 prep.finish(state, output)
//             }
//             OpsKind::UnTracked(prep) => {
//                 if VERBOSE {
//                     println!("UnTracked");
//                 }
//                 let output = B::euclidean_pairwise_distance(x.primitive);
//                 prep.finish(output)
//             }
//         }
//     }
// }
