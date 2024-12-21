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
    tensor::{
        cast::ToElement, ops::FloatTensor, ElementConversion, Float, Shape, Tensor, TensorPrimitive,
    },
};

use super::Backend;

const VERBOSE: bool = true;

// Implement our custom backend trait for any backend that also implements our custom backend trait.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C>
// where
//     TensorPrimitive<B>: std::ops::Mul<
//         <B as burn::prelude::Backend>::FloatTensorPrimitive,
//         Output = <B as burn::prelude::Backend>::FloatTensorPrimitive,
//     >,
{
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        // Create our zero-sized type that will implement the Backward trait.
        #[derive(Debug)]
        struct EuclideanPairwiseDistanceBackward;

        // Implement the backward trait for the given backend B, the node gradient
        // with three other gradients to calculate (lhs, rhs, and bias).
        impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
            // Our state that we must build during the forward pass to compute the backward pass.
            //
            // Note that we could improve the performance further by only keeping the state of
            // tensors that are tracked, improving memory management, but for simplicity, we avoid
            // that part.
            type State = (NodeID, FloatTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                if VERBOSE {
                    println!("backward start");
                }

                let [node_x] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, output) = ops.state;
                let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

                // Ensure xx is a 2D tensor (shape: [n, d])
                let xx: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(x.clone()));
                let shape_x = xx.shape();
                let dims = &shape_x.dims;

                let n = dims[0]; // Number of rows (samples)
                let d = dims[1]; // Dimensionality

                if VERBOSE {
                    println!("xx shape: {:?}", xx.shape());
                }

                // ReLU backward gradient (we only need it if ReLU was applied to output)
                let grad_output = B::relu_backward(output, grad.clone());
                let grad_output: Tensor<B, 1, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(grad_output));

                // grad_output has shape [499500] for the upper triangular part of the gradient matrix
                // This should be reshaped into a 2D tensor for use in pairwise calculations.
                let grad_output: Tensor<B, 2> =
                    grad_output.reshape(Shape::from([n * (n - 1) / 2, 1])); // Flattened upper triangular part

                if VERBOSE {
                    println!("grad_output reshaped: {:?}", grad_output.shape());
                }

                // Compute pairwise differences for each pair (i, j) where i != j
                let xx_exp: Tensor<B, 3> = xx.clone().expand(Shape::from([n, n, d])); // Shape [n, n, d]
                let xx_exp_transpose: Tensor<B, 3> =
                    xx.clone().expand(Shape::from([n, n, d])).transpose(); // Shape [n, n, d]

                // Compute pairwise differences for all pairs in one go (broadcasted)
                let diff = B::float_sub(
                    xx_exp.into_primitive().tensor(),
                    xx_exp_transpose.into_primitive().tensor(),
                ); // Shape [n, n, d]

                if VERBOSE {
                    println!("diff shape: {:?}", diff);
                }

                // Compute squared L2 norm (Euclidean distance) for each pair
                let squared = B::float_mul(diff.clone(), diff.clone()); // Shape [n, n, d]
                let squared: Tensor<B, 3, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(squared));

                // Sum squared differences over the last dimension (d)
                let sum_of_squares =
                    B::float_sum(squared.slice([0..n, 0..n, 0..d]).into_primitive().tensor()); // Shape [n, n]

                // Compute Euclidean distance (sqrt of sum of squares)
                let dist = B::float_sqrt(sum_of_squares); // Shape [n, n]

                let epsilon = 1e-12.elem();

                // Avoid division by zero using epsilon
                let dist_safe = B::float_clamp_max(dist.clone(), epsilon); // Shape [n, n]

                // Now we need to broadcast dist_safe to the shape of diff: [n, n, d]
                let dist_safe_broadcasted = B::float_expand(dist_safe, Shape::from([n, n, d])); // Broadcast to shape [n, n, d]

                // Compute the gradient of the distance with respect to x[i] and x[j]
                let grad_diff = B::float_div(diff.clone(), dist_safe_broadcasted.clone()); // Shape [n, n, d]

                if VERBOSE {
                    println!("grad_diff shape: {:?}", grad_diff);
                }

                // Gradients for `x` are accumulated by applying grad_output to grad_diff
                // Since grad_output represents the upper triangular part of the gradient matrix,
                // we need to apply it to the corresponding upper triangular indices.

                // Expand grad_output into a full n x n matrix (upper triangular part)
                let mut grad_matrix: Tensor<B, 2> =
                    Tensor::zeros(Shape::from([n, n]), &xx.device()); // Shape [n, n]

                // We will now use broadcasting to apply the gradients for each pair (i, j)
                // grad_output needs to be applied to each (i, j) pair in the upper triangle
                let grad_output_exp = grad_output.reshape(Shape::from([n, n])); // Shape [n, n]

                grad_matrix = grad_matrix.slice_assign([0..n, 0..n], grad_output_exp);

                // Now compute the final gradient for x by multiplying grad_matrix with grad_diff
                let grad_x = B::float_sum(B::float_mul(
                    grad_matrix.into_primitive().tensor(),
                    grad_diff,
                )); // Shape [n, d]

                // Register the gradient for x
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x);
                }

                if VERBOSE {
                    println!("backward end");
                }
            }
        }

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match EuclideanPairwiseDistanceBackward
            .prepare::<C>([x.node.clone()])
            // Marks the operation as compute bound, meaning it will save its
            // state instead of recomputing itself during checkpointing
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                if VERBOSE {
                    println!("Tracked");
                }
                // When at least one node is tracked, we should register our backward step.

                // The state consists of what will be needed for this operation's backward pass.
                // Since we need the parents' outputs, we must checkpoint their ids to retrieve
                // their node output at the beginning of the backward pass. We can also save
                // utilitary data such as the bias shape. If we also need this operation's output,
                // we can either save it in the state or recompute it.
                // during the backward pass. Here we choose to save it in the state because it's a
                // compute bound operation.
                let x_state = prep.checkpoint(&x);

                let output = B::euclidean_pairwise_distance(x.primitive.clone());

                let state = (x_state, output.clone());

                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                if VERBOSE {
                    println!("UnTracked");
                }
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                let output = B::euclidean_pairwise_distance(x.primitive);
                prep.finish(output)
            }
        }
    }
}
