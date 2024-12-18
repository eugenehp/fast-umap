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
    tensor::{ops::FloatTensor, Float, Shape, Tensor, TensorPrimitive},
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
            type State = (NodeID, NodeID, FloatTensor<B>, Shape);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                // Fetch the node and the gradient for the current node
                let [node_lhs] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);

                // Retrieve the state for each parent node
                let (lhs_state, _, output, _) = ops.state;
                let lhs: FloatTensor<B> = checkpointer.retrieve_node_output(lhs_state);

                let lhs = Tensor::from_primitive(TensorPrimitive::Float(lhs.clone()));
                let output = Tensor::from_primitive(TensorPrimitive::Float(output.clone()));
                let grad = Tensor::from_primitive(TensorPrimitive::Float(grad.clone()));
                // Fetch shapes of our tensor to support broadcasting.
                let shape_lhs = lhs.shape();

                // Now, we need to compute the gradient with respect to `lhs`
                // For pairwise Euclidean distance, the gradient is:
                // grad = (x_i - x_j) / distance(x_i, x_j)
                let grad_output: Tensor<B, 2, Float> = grad / output.clone();

                let grad = grad_output.clone() * lhs.clone();

                // Compute the gradient for the lhs tensor (x_i)
                let grad_lhs = broadcast_shape::<B>(grad.into_primitive().tensor(), &shape_lhs);

                // Register the gradients
                if let Some(node) = node_lhs {
                    grads.register::<B>(node.id, grad_lhs);
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
                // Register the backward step if needed (if node is tracked)
                let lhs_state = prep.checkpoint(&x);

                // Perform the forward pass for the pairwise Euclidean distance
                let output = Self::euclidean_pairwise_distance(x.clone());
                let output = output.into_primitive();

                let t: Tensor<B, 1, Float> =
                    Tensor::from_primitive(TensorPrimitive::Float(output.clone()));
                let shape = t.shape();

                let state = (lhs_state, lhs_state, output.clone(), shape);

                let x = prep.finish(state, output);
                x
            }
            OpsKind::UnTracked(prep) => {
                // If not tracked, just perform the forward pass
                let output = Self::euclidean_pairwise_distance(x.clone());
                let output = output.into_primitive();
                prep.finish(output)
            }
        }
    }
}
