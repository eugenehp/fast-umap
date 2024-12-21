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
                    println!("backward")
                }
                // Get the nodes of each variable.
                let [node_x] = ops.parents;
                // Fetch the gradient for the current node.
                if VERBOSE {
                    println!("backward - 1");
                }
                let grad = grads.consume::<B>(&ops.node);

                if VERBOSE {
                    println!("backward - 2");
                }
                // Set our state.
                let (x_state, output) = ops.state;
                if VERBOSE {
                    println!("backward - 3");
                }
                let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);

                if VERBOSE {
                    println!("backward - 4");
                }
                let xx: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(x.clone()));
                if VERBOSE {
                    println!("backward - 5");
                }
                let shape = xx.shape();
                if VERBOSE {
                    println!("backward - 6");
                }

                // Fetch shapes of our tensor to support broadcasting.
                // let shape_x = x.shape();
                let shape_x = shape;
                if VERBOSE {
                    println!("backward - 7");
                }

                let dims = &shape_x.dims;

                // Compute the gradient of the output using the already existing `relu_backward`
                // function in the basic Burn backend trait.
                let grad_output = B::relu_backward(output, grad.clone());
                if VERBOSE {
                    println!("backward - 8 {grad_output:?}");
                }
                if VERBOSE {
                    println!("x shape {:?}", shape_x);
                }

                // Compute the lhs gradient, which is the derivative of matmul with support for
                // broadcasting.
                // let grad_x = B::float_expand(
                //     B::float_matmul(grad_output.clone(), B::float_transpose(x)),
                //     shape_x,
                // );
                // let grad_x = B::float_expand(
                //     B::float_matmul(B::float_transpose(x), grad_output.clone()),
                //     shape_x,
                // );

                let n = dims[0]; // Number of rows (samples)
                let d = dims[1]; // Dimensionality (2 in this case)

                let mut grad_x = Tensor::zeros_like(&xx); // Initialize gradient tensor with the same shape as x

                if VERBOSE {
                    println!("backward - 9");
                }
                let grad: Tensor<B, 2, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(grad));

                if VERBOSE {
                    println!("backward - 10");
                }
                let mut grad_index = 0;
                for i in 0..n {
                    for j in i + 1..n {
                        println!("{i}/{j}");
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 1");
                        }
                        let lhs = xx.clone().slice([i..i + 1, 0..d]).into_primitive().tensor();
                        let rhs = xx.clone().slice([j..j + 1, 0..d]).into_primitive().tensor();
                        // Compute the gradient for pair (i, j)
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 2");
                        }
                        let diff = B::float_sub(lhs, rhs); // Difference between the two points

                        // let dist = diff.norm(); // Euclidean distance
                        // Step 1: Square each element of `diff`
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 3");
                        }
                        let squared = B::float_mul(diff.clone(), diff.clone());

                        // Step 2: Sum the squared elements (across the last dimension)
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 4");
                        }
                        let sum_of_squares = B::float_sum(squared); // Sum over the columns (dim 1)

                        // Step 3: Take the square root of the sum of squares
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 5");
                        }
                        let dist = B::float_sqrt(sum_of_squares); // Take the square root to get the Euclidean norm
                        let diff_shape = Tensor::<B, 1, Float>::from_primitive(
                            TensorPrimitive::Float(diff.clone()),
                        )
                        .shape();

                        let dist = B::float_expand(dist, diff_shape);
                        if VERBOSE {
                            println!("diff - {diff:?}, dist - {dist:?}");
                        }

                        if VERBOSE {
                            println!("backward - [{i},{j}] - 6");
                        }
                        let grad_i = B::float_div(diff, dist); // Gradient with respect to x_i
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 7");
                        }
                        let grad_j = B::float_neg(grad_i.clone()); // Gradient with respect to x_j (negative of grad_i)

                        // Access the gradient for this pair using slice
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 8 - {grad:?}");
                        }
                        if VERBOSE {
                            println!("grad_index - {grad_index}");
                        }
                        let grad_value = grad.clone().slice([grad_index..grad_index + 1]); // Get scalar value from flattened gradient tensor

                        if VERBOSE {
                            println!("backward - [{i},{j}] - 9");
                        }
                        if VERBOSE {
                            println!("grad_i - {grad_i:?}");
                        }
                        let grad_i_shape = Tensor::<B, 1, Float>::from_primitive(
                            TensorPrimitive::Float(grad_i.clone()),
                        )
                        .shape();
                        let grad_value =
                            B::float_expand(grad_value.into_primitive().tensor(), grad_i_shape);
                        if VERBOSE {
                            println!("grad_value - {grad_value:?}");
                        }
                        let values = Tensor::from_primitive(TensorPrimitive::Float(B::float_mul(
                            grad_i,
                            grad_value.clone(),
                            // grad_value.clone().into_primitive().tensor(),
                        )));
                        // Using slice_assign to update grad_x at the correct positions
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 10");
                        }
                        grad_x = grad_x.slice_assign([i..i + 1, 0..d], values);

                        if VERBOSE {
                            println!("backward - [{i},{j}] - 11");
                        }
                        let values = Tensor::from_primitive(TensorPrimitive::Float(B::float_mul(
                            grad_j, grad_value,
                        )));
                        if VERBOSE {
                            println!("backward - [{i},{j}] - 12");
                        }
                        grad_x = grad_x.slice_assign([j..j + 1, 0..d], values);

                        if VERBOSE {
                            println!("backward - [{i},{j}] - 13");
                        }
                        grad_index += 1;
                    }
                }

                // Register the gradient for each variable based on whether they are marked as
                // `tracked`.
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x.into_primitive().tensor());
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
