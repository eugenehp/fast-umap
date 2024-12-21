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
    tensor::{ops::FloatTensor, Float, Tensor, TensorPrimitive},
};

use super::Backend;

const VERBOSE: bool = false;

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
                    println!("backward start")
                }

                let [node_x] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                let (x_state, output) = ops.state;
                let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

                let xx: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(x.clone()));
                let shape_x = xx.clone().shape();
                let dims = &shape_x.dims;
                let n = dims[0]; // Number of rows (samples)
                let d = dims[1]; // Dimensionality (2 in this case)

                let grad_output = B::relu_backward(output, grad.clone());
                let grad_output: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(grad_output));

                // Compute the pairwise difference matrix: diff[i,j] = x[i,:] - x[j,:]
                let mut diff = Tensor::zeros_like(&(xx.clone())); // Shape [n, n, d]

                for i in 0..n {
                    let xx = xx.clone();
                    let lhs = xx.clone().slice([i..i + 1, 0..d]); // Shape [1, d]
                    for j in 0..n {
                        let xx = xx.clone();
                        let rhs = xx.slice([j..j + 1, 0..d]); // Shape [1, d]

                        // Compute difference for pair (i, j)
                        let diff_ij = B::float_sub(
                            lhs.clone().into_primitive().tensor(),
                            rhs.clone().into_primitive().tensor(),
                        ); // Shape [1, d]

                        // Correct the slice range to match the shape of diff_ij ([1, d])
                        // Use the correct slice range for the 3D tensor `diff`: [i..i+1, j..j+1, 0..d]
                        diff = diff.slice_assign(
                            [i..i + 1, j..j + 1, 0..d], // Correct slice range
                            Tensor::from_primitive(TensorPrimitive::Float(diff_ij)),
                        );
                    }
                }

                // Compute the squared L2 norm (Euclidean distance) for each pair: ||diff[i,j]||^2
                let squared = B::float_mul(
                    diff.clone().into_primitive().tensor(),
                    diff.clone().into_primitive().tensor(),
                ); // Shape [n, n, d]
                let squared: Tensor<B, 3, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(squared));

                let sum_of_squares =
                    B::float_sum(squared.slice([0..n, 0..n, 0..d]).into_primitive().tensor()); // Sum over the last dimension, resulting in [n, n]
                let dist = B::float_sqrt(sum_of_squares); // Shape [n, n]

                // Avoid dividing by zero (set small values to a small epsilon)
                let dist_safe = B::float_max(dist.clone());
                let dist_safe: Tensor<B, 3, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(dist_safe));

                // Compute the gradient of the distance with respect to x[i] and x[j]
                let grad_diff = B::float_div(
                    diff.clone().into_primitive().tensor(),
                    dist_safe
                        .slice([0..n, 0..n, 0..d])
                        .into_primitive()
                        .tensor(),
                ); // Shape [n, n, d]

                // Compute gradients for all pairs at once
                let grad_x = B::float_sum(B::float_mul(
                    grad_output
                        .slice([0..n, 0..d, 0..1])
                        .into_primitive()
                        .tensor(),
                    grad_diff,
                )); // Shape [n, d]

                // Register the gradient for x
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x);
                }

                if VERBOSE {
                    println!("backward end")
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
