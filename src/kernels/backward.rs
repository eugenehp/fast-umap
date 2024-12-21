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

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        // Create zero-sized struct for backward computation
        #[derive(Debug)]
        struct EuclideanPairwiseDistanceBackward;

        // Implement the backward trait for the given backend B
        impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
            type State = (NodeID, Tensor<B, 2, Float>, Tensor<B, 2, Float>); // Include both original x and output tensor

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (node_x, x, output) = ops.state; // Retrieve x and output from the state
                println!("Retrieved x: {:?}", x.shape()); // Debug: Print shape of x
                println!("Retrieved output: {:?}", output.shape()); // Debug: Print shape of output

                // Fetch the gradient for the output of the pairwise distance function
                let grad_primitive = grads.consume::<B>(&ops.node); // Gradient of pairwise distance output tensor
                let grad: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(grad_primitive)); // Convert to Tensor
                println!("Gradient: {:?}", grad.shape()); // Debug: Print gradient shape

                // Get the number of vectors (rows) and the dimensionality (columns)
                let n = x.shape().dims[0]; // Number of vectors
                let d = x.shape().dims[1]; // Dimensionality of each vector
                println!("n (number of vectors): {}", n); // Debug: Print number of vectors
                println!("d (dimensionality of vectors): {}", d); // Debug: Print dimensionality

                // Reshape the gradient to match the shape [n, n]
                let grad_reshaped: Tensor<B, 2, Float> = grad.reshape([n, n]);
                println!("Reshaped gradient: {:?}", grad_reshaped.shape()); // Debug: Print reshaped gradient shape

                // Expand the input tensor x to compute pairwise distances
                let x_expanded: Tensor<B, 3, Float> = x.expand([n, n, d]); // Shape [n, n, d]
                let x_transposed: Tensor<B, 3, Float> = x_expanded.clone().transpose().transpose(); // Shape [n, n, d]
                println!("Expanded x: {:?}", x_expanded.shape()); // Debug: Print expanded x shape
                println!("Transposed x: {:?}", x_transposed.shape()); // Debug: Print transposed x shape

                // Calculate the difference between each pair of vectors
                let difference = x_expanded - x_transposed; // Shape [n, n, d]
                println!(
                    "Difference (expanded - transposed): {:?}",
                    difference.shape()
                ); // Debug: Print difference shape

                // Compute the pairwise distances (for normalization in the derivative)
                let pairwise_distances = (difference.clone().powi_scalar(2)).sum_dim(2); // Shape [n, n], sum across the last dimension (d)
                println!("Pairwise distances: {:?}", pairwise_distances.shape()); // Debug: Print pairwise distances shape

                // Avoid division by zero by adding a small epsilon to the pairwise distances
                let epsilon = 1e-8; // A small constant to prevent division by zero
                let pairwise_distances_with_epsilon = pairwise_distances + epsilon;
                println!(
                    "Pairwise distances with epsilon: {:?}",
                    pairwise_distances_with_epsilon.shape()
                ); // Debug: Print distances with epsilon

                // Calculate the gradients w.r.t. the input x
                let grad_x =
                    difference * grad_reshaped.unsqueeze_dim(2) / pairwise_distances_with_epsilon;
                // let grad_x = difference * grad_reshaped.expand([n, n, d])
                //     / pairwise_distances_with_epsilon.expand([n, n, d]);
                println!("Grad_x (before sum): {:?}", grad_x.shape()); // Debug: Print grad_x shape

                // Now sum across dimension 1 (rows) and then squeeze the dimensions
                let grad_x_final: Tensor<B, 2, Float> = grad_x.sum_dim(1).squeeze_dims(&[1]); // Sum across rows, squeeze dim 1
                println!(
                    "Final grad_x after sum_dim and squeeze_dims: {:?}",
                    grad_x_final.shape()
                ); // Debug: Print final grad_x shape

                // Register the gradient for the input tensor x
                grads.register::<B>(node_x, grad_x_final.into_primitive().tensor());
                println!("Gradient registered for x."); // Debug: Indicate gradient has been registered
            }
        }

        // Prepare the stateful operation
        match EuclideanPairwiseDistanceBackward
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                println!("Tracked");
                // When at least one node is tracked, register the backward function
                let x_state = prep.checkpoint(&x); // Checkpoint x for future retrieval during the backward pass
                println!("Checkpointed x for backward pass."); // Debug: Indicate checkpointing

                let output = B::euclidean_pairwise_distance(x.clone().primitive); // Forward pass calculation
                println!("Forward pass output: {:?}", output); // Debug: Print output shape

                let x = Tensor::from_primitive(TensorPrimitive::Float(x.clone().primitive));
                let output_tensor = Tensor::from_primitive(TensorPrimitive::Float(output.clone()));

                // The state now includes the checkpointed x and the output
                let state = (x_state, x, output_tensor); // Pass both x and output
                prep.finish(state, output) // Finish with the computed output
            }
            OpsKind::UnTracked(prep) => {
                // If no node is tracked, just do the forward calculation
                let output = B::euclidean_pairwise_distance(x.primitive);
                println!("Forward pass output (UnTracked): {:?}", output); // Debug: Print output shape
                prep.finish(output) // No need for state here
            }
        }
    }
}

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
//                 // Get the nodes of each variable.
//                 let [node_x] = ops.parents;
//                 // Fetch the gradient for the current node.
//                 let grad = grads.consume::<B>(&ops.node);

//                 // Set our state.
//                 let (x_state, output) = ops.state;
//                 let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

//                 // Convert x to a 2D tensor (this step is important for broadcasting support)
//                 let xx: Tensor<B, 2, Float> =
//                     Tensor::from_primitive(TensorPrimitive::Float(x.clone()));

//                 if VERBOSE {
//                     println!("=========");
//                     println!("x - {x:?}");
//                     println!("grad - {grad:?}");
//                     println!("output - {output:?}");
//                     println!("=========");
//                 }

//                 // Fetch shapes of our tensor to support broadcasting.
//                 let shape_x = xx.shape();

//                 if VERBOSE {
//                     println!("backward - 1");
//                 }

//                 // Compute the gradient of the output with respect to x (derivative of Euclidean distance).
//                 // Euclidean distance gradient with respect to x is: grad_output * (x / ||x||) (simplified form).
//                 // let grad_output = B::float_mul(grad, B::float_div(x.clone(), output)); // This will need adjustment based on your Euclidean implementation.

//                 // Ensure grad_output matches the shape of x
//                 // let grad_output = B::float_mul(grad, B::float_div(x.clone(), output)); // Adjust this as needed for proper gradient computation
//                 let grad_output = B::relu_backward(output, grad);

//                 if VERBOSE {
//                     println!("backward - 2 - grad_output - {grad_output:?}");
//                 }

//                 // Reshape the grad_output if needed to match x's shape or the expected pairwise distance shape.
//                 let reshaped_grad_output = B::float_reshape(grad_output, shape_x.clone()); // Reshaping grad_output to match x's shape

//                 // Ensure that you reshape x for matrix multiplication correctly
//                 let reshaped_x = B::float_reshape(x.clone(), shape_x.clone()); // Adjust shape as necessary

//                 // Matrix multiplication of the gradient with the reshaped input tensor.
//                 let tensor = B::float_matmul(reshaped_grad_output.clone(), reshaped_x);

//                 if VERBOSE {
//                     println!("backward - 3");
//                 }

//                 // Compute the lhs gradient, which is the derivative of matmul with support for broadcasting.
//                 let grad_x = B::float_reshape(tensor, shape_x);

//                 if VERBOSE {
//                     println!("backward - 4");
//                 }

//                 // Register the gradient for each variable based on whether they are marked as
//                 // `tracked`.
//                 if let Some(node) = node_x {
//                     grads.register::<B>(node.id, grad_x);
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
