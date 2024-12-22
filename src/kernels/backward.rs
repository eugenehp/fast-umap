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
    tensor::ops::FloatTensor,
};

use crate::print_if;

use super::Backend;

const VERBOSE: bool = false;

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        // Create zero-sized struct for backward computation
        #[derive(Debug)]
        struct EuclideanPairwiseDistanceBackward;

        // Implement the backward trait for the given backend B
        impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
            type State = NodeID;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let node_x = ops.state; // Retrieve x and output from the state

                // Fetch the gradient for the current node.
                let grad = grads.consume::<B>(&ops.node);
                let output: FloatTensor<B> = checkpointer.retrieve_node_output(node_x);

                print_if!(VERBOSE, "Grad {:?}", grad);
                print_if!(VERBOSE, "Output {:?}", output);

                let grad_x = B::float_matmul(grad, output);
                print_if!(VERBOSE, "grad_x {:?}", grad_x);

                grads.register::<B>(node_x, grad_x);
            }
        }

        // Prepare the stateful operation
        match EuclideanPairwiseDistanceBackward
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // When at least one node is tracked, register the backward function
                let x_state = prep.checkpoint(&x); // Checkpoint x for future retrieval during the backward pass

                let output = B::euclidean_pairwise_distance(x.clone().primitive); // Forward pass calculation
                print_if!(VERBOSE, "Forward pass output (Tracked): {:?}", output); // Debug: Print output shape

                // The state now includes the checkpointed x and the output
                prep.finish(x_state, output) // Finish with the computed output
            }
            OpsKind::UnTracked(prep) => {
                // If no node is tracked, just do the forward calculation
                let output = B::euclidean_pairwise_distance(x.primitive);
                print_if!(VERBOSE, "Forward pass output (UnTracked): {:?}", output); // Debug: Print output shape
                prep.finish(output) // No need for state here
            }
        }
    }
}
