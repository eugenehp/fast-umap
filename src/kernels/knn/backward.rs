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

use crate::{backend::Backend, print_if, print_primitive_tensor};

const VERBOSE: bool = false;

pub fn backward<B: Backend, C: CheckpointStrategy>(
    pairwise_distances: FloatTensor<Autodiff<B, C>>,
    k: u32, // Number of nearest neighbors
) -> (FloatTensor<Autodiff<B, C>>, FloatTensor<Autodiff<B, C>>) {
    // println!("knn_backward");
    // Create zero-sized struct for backward computation
    #[derive(Debug)]
    struct KnnBackward;

    // Implement the backward trait for the given backend B
    impl<B: Backend> Backward<B, 1> for KnnBackward {
        type State = (NodeID, u32); // FloatTensor<B>,

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let (node_pairwise_distances, k) = ops.state; // Retrieve pairwise_distances and output from the state

            // Fetch the gradient for the current node.
            let grad_output = grads.consume::<B>(&ops.node);
            let pairwise_distances: FloatTensor<B> =
                checkpointer.retrieve_node_output(node_pairwise_distances);

            if VERBOSE {
                println!("grad_output {grad_output:?}");
                print_primitive_tensor::<B>(&grad_output, 10, 10);
                println!("pairwise_distances {pairwise_distances:?}");
                print_primitive_tensor::<B>(&pairwise_distances, 10, 10);
            }

            // Perform the backward pass for the KNN operation
            let grad_pairwise_distances = B::knn_backward(pairwise_distances, k, grad_output);

            if VERBOSE {
                println!("===grad_pairwise_distances=== {grad_pairwise_distances:?}");
                print_primitive_tensor::<B>(&grad_pairwise_distances, 0, 0);
            }

            // Register the gradient for the pairwise_distances tensor
            grads.register::<B>(node_pairwise_distances, grad_pairwise_distances);
        }
    }

    // Prepare the stateful operation
    let indicies = match KnnBackward
        .prepare::<C>([pairwise_distances.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            // When at least one node is tracked, register the backward function
            let pairwise_distances_state = prep.checkpoint(&pairwise_distances); // Checkpoint pairwise_distances for future retrieval during the backward pass

            let output = B::knn(pairwise_distances.clone().primitive, k); // Forward pass calculation
            let (indicies, distances) = output;
            print_if!(VERBOSE, "Forward pass indicies (Tracked): {:?}", indicies); // Debug: Print indicies shape
            print_if!(VERBOSE, "Forward pass distances (Tracked): {:?}", distances); // Debug: Print distances shape

            let state = (pairwise_distances_state, k);

            // The state now includes the checkpointed pairwise_distances and the output
            let indicies = prep.finish(state, indicies); // Finish with the computed output

            indicies
        }
        OpsKind::UnTracked(prep) => {
            // If no node is tracked, just do the forward calculation
            let output = B::knn(pairwise_distances.clone().primitive, k); // Forward pass calculation
            let (indicies, distances) = output;

            print_if!(VERBOSE, "Forward pass indicies (UnTracked): {:?}", indicies); // Debug: Print indicies shape
            print_if!(
                VERBOSE,
                "Forward pass distances (UnTracked): {:?}",
                distances
            ); // Debug: Print distances shape

            let indicies = prep.finish(indicies);

            indicies
        }
    };

    let distances = match KnnBackward
        .prepare::<C>([pairwise_distances.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            // When at least one node is tracked, register the backward function
            let pairwise_distances_state = prep.checkpoint(&pairwise_distances); // Checkpoint pairwise_distances for future retrieval during the backward pass

            let output = B::knn(pairwise_distances.clone().primitive, k); // Forward pass calculation
            let (indicies, distances) = output;
            print_if!(VERBOSE, "Forward pass indicies (Tracked): {:?}", indicies); // Debug: Print indicies shape
            print_if!(VERBOSE, "Forward pass distances (Tracked): {:?}", distances); // Debug: Print distances shape

            let state = (pairwise_distances_state, k);

            // The state now includes the checkpointed pairwise_distances and the output
            let distances = prep.finish(state, distances); // Finish with the computed output

            distances
        }
        OpsKind::UnTracked(prep) => {
            // If no node is tracked, just do the forward calculation
            let output = B::knn(pairwise_distances.clone().primitive, k); // Forward pass calculation
            let (indicies, distances) = output;

            print_if!(VERBOSE, "Forward pass indicies (UnTracked): {:?}", indicies); // Debug: Print indicies shape
            print_if!(
                VERBOSE,
                "Forward pass distances (UnTracked): {:?}",
                distances
            ); // Debug: Print distances shape

            let distances = prep.finish(distances);

            distances
        }
    };

    (indicies, distances)
}
