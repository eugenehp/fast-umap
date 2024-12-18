use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{broadcast_shape, Backward, Ops, OpsKind},
            NodeID,
        },
        Autodiff,
    },
    tensor::{ops::FloatTensor, Float, Tensor, TensorPrimitive},
};

use super::Backend;

// Implement our custom backend trait for any backend that also implements our custom backend trait.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        // Create our zero-sized type that will implement the Backward trait.
        #[derive(Debug)]
        struct FusedEuclideanDistanceBackward;

        // Implement the backward computation for the pairwise Euclidean distance
        impl<B: Backend> Backward<B, 1> for FusedEuclideanDistanceBackward {
            type State = (NodeID, FloatTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                // println!("euclidean_pairwise_distance[backward]");
                // Fetch the node and the gradient for the current node
                let [node_lhs] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                // Retrieve the state for each parent node
                let (lhs_state, output) = ops.state;
                let lhs: FloatTensor<B> = checkpointer.retrieve_node_output(lhs_state);

                // let lhs = Tensor::from_primitive(TensorPrimitive::Float(lhs.clone()));
                // let output = Tensor::from_primitive(TensorPrimitive::Float(output.clone()));
                // let grad = Tensor::from_primitive(TensorPrimitive::Float(grad.clone()));

                // // Fetch shapes of our tensor to support broadcasting.
                // let shape_lhs = lhs.shape();

                // // Add a small epsilon value to avoid division by zero
                // let epsilon = 1e-6; // Adjust this value as needed

                // // Clamp the output to a minimum value (epsilon) to avoid division by zero
                // let output_clamped = output.clone().clamp_min(epsilon);

                // // Now, we need to compute the gradient with respect to `lhs`
                // // For pairwise Euclidean distance, the gradient is:
                // // grad = (x_i - x_j) / distance(x_i, x_j)
                // // let grad_output: Tensor<B, 1, Float> = grad / output.clone();
                // let grad_output: Tensor<B, 1, Float> = grad / output_clamped;

                // let grad = grad_output.clone() * lhs.clone();

                // // Compute the gradient for the lhs tensor (x_i)
                // let grad_lhs = broadcast_shape::<B>(grad.into_primitive().tensor(), &shape_lhs);

                // Register the gradients
                if let Some(node) = node_lhs {
                    grads.register::<B>(node.id, lhs);
                    // grads.register::<B>(node.id, grad_lhs);
                }
            }
        }

        // Here, we prepare the operation and perform the forward pass
        match FusedEuclideanDistanceBackward
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // println!("euclidean_pairwise_distance - autodiff - Tracked");
                // Register the backward step if needed (if node is tracked)
                let lhs_state = prep.checkpoint(&x);

                // Perform the forward pass for the pairwise Euclidean distance
                let output = B::euclidean_pairwise_distance(x.into_primitive());

                let state = (lhs_state, output.clone());

                let x = prep.finish(state, output);
                x
            }
            OpsKind::UnTracked(prep) => {
                let x = x.into_primitive();
                // println!("euclidean_pairwise_distance - autodiff - UnTracked");
                // If not tracked, just perform the forward pass
                let output = B::euclidean_pairwise_distance(x);
                prep.finish(output)
            }
        }
    }
}
