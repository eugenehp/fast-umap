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

                // ReLU backward gradient
                let grad_output = B::relu_backward(output, grad.clone());
                let grad_output: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(grad_output));

                if VERBOSE {
                    println!("grad_output - {grad_output:?}");
                }

                // Broadcasting xx to create pairwise differences
                let xx_exp: Tensor<B, 3> = xx.clone().expand(Shape::from([n, n, d])); // Shape [n, n, d]
                let diff = B::float_sub(
                    xx_exp.clone().into_primitive().tensor(),
                    xx_exp.transpose().into_primitive().tensor(),
                ); // Shape [n, n, d]

                if VERBOSE {
                    println!("diff - {diff:?}");
                }

                // Compute squared L2 norm (Euclidean distance) for each pair
                let squared = B::float_mul(diff.clone(), diff.clone()); // Shape [n, n, d]
                let squared: Tensor<B, 3, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(squared));

                if VERBOSE {
                    println!("squared - {squared:?}");
                }

                // Sum squared differences over the last dimension (d)
                let sum_of_squares =
                    B::float_sum(squared.slice([0..n, 0..n, 0..d]).into_primitive().tensor()); // Shape [n, n]

                if VERBOSE {
                    println!("sum_of_squares - {sum_of_squares:?}");
                }

                // Compute Euclidean distance (sqrt of sum of squares)
                let dist = B::float_sqrt(sum_of_squares); // Shape [n, n]

                if VERBOSE {
                    println!("dist - {dist:?}");
                }

                let epsilon = 1e-12.elem();

                // Avoid division by zero using epsilon
                let dist_safe = B::float_clamp_max(dist.clone(), epsilon);
                // let dist_safe = B::float_expand(dist_safe, Shape::from([n, n, d]));

                if VERBOSE {
                    println!("dist_safe - {:?}", dist_safe);
                }

                // Compute gradient of the distance with respect to x[i] and x[j]
                let grad_diff = B::float_div(diff.clone(), dist_safe.clone()); // Shape [n, n, d]

                if VERBOSE {
                    println!("grad_diff - {grad_diff:?}");
                }

                // Compute gradients for all pairs at once
                let grad_x = B::float_sum(B::float_mul(
                    grad_output
                        .slice([0..n, 0..d, 0..1])
                        .into_primitive()
                        .tensor(),
                    grad_diff,
                )); // Shape [n, d]

                if VERBOSE {
                    println!("grad_x - {grad_x:?}");
                }

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
