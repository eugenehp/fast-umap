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
    tensor::{ops::FloatTensor, ElementConversion, Float, Int, Shape, Tensor, TensorPrimitive},
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

                // Step 1: Compute pairwise difference (x_i - x_j)
                let xx_exp: Tensor<B, 3> = xx.clone().expand(Shape::from([n, n, d])); // Shape [n, n, d]
                let xx_exp_transpose: Tensor<B, 3> =
                    xx.clone().expand(Shape::from([n, n, d])).transpose(); // Shape [n, n, d]

                let diff = B::float_sub(
                    xx_exp.into_primitive().tensor(),
                    xx_exp_transpose.into_primitive().tensor(),
                ); // Shape [n, n, d]

                if VERBOSE {
                    println!("diff shape: {:?}", diff);
                }

                // Step 2: Compute squared L2 norm (Euclidean distance) for each pair
                let squared = B::float_mul(diff.clone(), diff.clone()); // Shape [n, n, d]
                let squared: Tensor<B, 3, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(squared));

                let sum_of_squares =
                    B::float_sum(squared.slice([0..n, 0..n, 0..d]).into_primitive().tensor()); // Shape [n, n]

                let dist = B::float_sqrt(sum_of_squares); // Shape [n, n]

                let epsilon = 1e-12.elem();
                let dist_safe = B::float_clamp_max(dist.clone(), epsilon); // Shape [n, n]

                // Step 3: Compute the gradient of the distance with respect to x[i] and x[j]
                let grad_diff = B::float_div(
                    diff.clone(),
                    B::float_expand(dist_safe, Shape::from([n, n, d])),
                ); // Normalize by Euclidean distance

                // Step 4: Compute the gradient of x
                // Gradient of x[i] is the sum over all j, but remember to reverse the direction for x[j]
                let grad_x = B::float_sum_dim(grad_diff.clone(), 1); // Summing over j for x[i]
                let grad_x_transpose = B::float_sum_dim(grad_diff, 0); // Summing over i for x[j]

                // Step 5: Register the gradients for both x[i] and x[j]
                let grad_x = B::float_reshape(grad_x, xx.shape()); // Reshape to match the original shape of x
                let grad_x_transpose = B::float_reshape(grad_x_transpose, xx.shape()); // Reshape to match the original shape of x

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x.clone());
                    grads.register::<B>(node.id, grad_x_transpose.clone());
                }

                if VERBOSE {
                    println!("backward end {grad_x:?}");
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
