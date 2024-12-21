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

const VERBOSE: bool = true;

// Implement our custom backend trait for any backend that also implements our custom backend trait.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
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

                // Step 1: Compute grad_output for ReLU backward (should have shape [499500])
                let grad_output = B::relu_backward(output, grad.clone());
                let grad_output: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(grad_output));

                // Step 2: Generate the upper triangular indices of the full n x n matrix
                let grad_output_exp = grad_output.clone(); // Gradient output stays the same, but we need to apply it into the gradient matrix

                // Step 3: Create an empty n x n gradient matrix and fill in the upper triangular values
                let mut grad_matrix: Tensor<B, 2> =
                    Tensor::zeros(Shape::from([n, n]), &xx.device()); // Shape [n, n]

                // Create a vector of pairs (i, j) for the upper triangular part
                let upper_triangular_indices: Vec<(usize, usize)> = (0..n)
                    .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                    .collect();

                // Convert to a format compatible with the backend
                let indices_data: Vec<usize> = upper_triangular_indices
                    .iter()
                    .flat_map(|(i, j)| vec![*i as usize, *j as usize]) // Convert to i32 for indices
                    .collect();

                // Create a tensor for indices with the correct backend-specific type
                let indices = Tensor::<B, 1, Int>::from_data(indices_data.as_slice(), &xx.device()); // Use from_data instead of from_primitive

                // Step 5: We need to select and assign the gradients using select_assign
                grad_matrix = grad_matrix.select_assign(0, indices.clone(), grad_output_exp); // This applies the gradient output to the grad_matrix

                // Step 6: Symmetrically assign values to the lower triangular part
                let grad_matrix_lower = grad_matrix.clone();
                grad_matrix = grad_matrix.select_assign(1, indices, grad_matrix_lower); // For the lower triangular part

                // Step 7: Compute pairwise differences (xx[i] - xx[j]) using broadcasting
                let xx_exp: Tensor<B, 3> = xx.clone().expand(Shape::from([n, n, d])); // Shape [n, n, d]
                let xx_exp_transpose: Tensor<B, 3> =
                    xx.clone().expand(Shape::from([n, n, d])).transpose(); // Shape [n, n, d]

                // Pairwise differences in a vectorized way (broadcasting automatically handles the loops)
                let diff = B::float_sub(
                    xx_exp.into_primitive().tensor(),
                    xx_exp_transpose.into_primitive().tensor(),
                ); // Shape [n, n, d]

                if VERBOSE {
                    println!("diff shape: {:?}", diff);
                }

                // Step 8: Compute squared L2 norm (Euclidean distance) for each pair
                let squared = B::float_mul(diff.clone(), diff.clone()); // Shape [n, n, d]
                let squared: Tensor<B, 3, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(squared));

                // Sum squared differences over the last dimension (d)
                let sum_of_squares =
                    B::float_sum(squared.slice([0..n, 0..n, 0..d]).into_primitive().tensor()); // Shape [n, n]

                // Step 9: Compute Euclidean distance (sqrt of sum of squares)
                let dist = B::float_sqrt(sum_of_squares); // Shape [n, n]

                let epsilon = 1e-12.elem();

                // Avoid division by zero using epsilon
                let dist_safe = B::float_clamp_max(dist.clone(), epsilon); // Shape [n, n]

                // Now we need to broadcast dist_safe to the shape of diff: [n, n, d]
                let dist_safe_broadcasted = B::float_expand(dist_safe, Shape::from([n, n, d])); // Broadcast to shape [n, n, d]

                // Step 10: Compute the gradient of the distance with respect to x[i] and x[j]
                let grad_diff = B::float_div(diff.clone(), dist_safe_broadcasted.clone()); // Shape [n, n, d]

                if VERBOSE {
                    println!("grad_diff shape: {:?}", grad_diff);
                }

                // Step 11: Apply gradients to the correct elements
                // We already have the pairwise gradients in grad_matrix, so we directly apply it.
                let grad_x = B::float_sum_dim(grad_matrix.into_primitive().tensor(), 1); // Summing over the rows (first dimension)

                let grad_x = B::float_reshape(grad_x, xx.shape());

                // Register the gradient for x
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x.clone());
                }

                if VERBOSE {
                    println!("backward end {grad_x:?}");
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
