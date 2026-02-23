//! Autodiff registration for the k-NN operation.
//!
//! [`backward`] wires up the Burn autodiff graph so that `loss.backward()`
//! correctly propagates gradients through the k-NN selection back to the
//! pairwise distance matrix — and from there, through the Euclidean distance
//! kernel, all the way to the neural-network embedding weights.

use std::fmt::Debug;

use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
            NodeId,
        },
        Autodiff,
    },
    tensor::ops::{FloatTensor, IntTensor},
};

use crate::{backend::*, print_if, print_primitive_tensor};

const VERBOSE: bool = false;

/// Register the k-NN operation in the Burn autodiff graph.
///
/// On the **tracked** path (at least one input requires a gradient) this
/// function:
/// 1. Checkpoints `pairwise_distances` for retrieval during the backward pass.
/// 2. Runs the forward k-NN on the inner backend.
/// 3. Registers [`KnnBackward`] as the backward hook via [`Ops::finish`].
///
/// On the **untracked** path it simply runs the forward pass and wraps the
/// result in an [`Autodiff`] tensor.
///
/// # Arguments
///
/// * `pairwise_distances` — `[n, n]` autodiff-wrapped distance matrix.
/// * `k`                  — Number of nearest neighbours.
///
/// # Returns
///
/// `(indices, distances)` — both wrapped in the [`Autodiff`] backend so that
/// gradients flow through them during `loss.backward()`.
pub fn backward<B: Backend, C: CheckpointStrategy>(
    pairwise_distances: FloatTensor<Autodiff<B, C>>,
    k: u32,
) -> (IntTensor<Autodiff<B, C>>, FloatTensor<Autodiff<B, C>>) {
    // println!("knn_backward");
    /// Zero-sized marker struct that implements Burn's [`Backward`] trait for
    /// the k-NN selection operation.
    #[derive(Debug)]
    struct KnnBackward;

    impl<B: Backend> Backward<B, 1> for KnnBackward {
        /// Backward state: the `NodeId` of `pairwise_distances` (for gradient
        /// registration) and the scalar `k` needed to re-run the forward sort.
        type State = (NodeId, u32);

        /// Called by Burn's autodiff engine during `loss.backward()`.
        ///
        /// Retrieves `pairwise_distances` from the checkpoint, consumes the
        /// upstream gradient `∂loss/∂knn_distances`, dispatches the backward
        /// GPU kernel via `B::knn_backward`, and registers the resulting
        /// `∂loss/∂pairwise_distances` gradient for further propagation.
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

            let (indicies, distances) = B::knn(pairwise_distances.clone().primitive, k); // Forward pass calculation
            print_if!(VERBOSE, "Forward pass indicies (Tracked): {:?}", indicies); // Debug: Print indicies shape
            print_if!(VERBOSE, "Forward pass distances (Tracked): {:?}", distances); // Debug: Print distances shape

            let state = (pairwise_distances_state, k);

            // TODO: this is a strange way to convert it
            let indicies = B::int_into_float(indicies);

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

            // TODO: this is a strange way to convert it
            let indicies = B::int_into_float(indicies);

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

            let (indicies, distances) = B::knn(pairwise_distances.clone().primitive, k); // Forward pass calculation
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

    // Extract the inner tensor
    let inner_tensor = indicies.into_primitive();
    let int_tensor = B::float_into_int(inner_tensor);

    // Convert the inner tensor to the autodiff backend
    let indicies: IntTensor<Autodiff<B, C>> = IntTensor::<Autodiff<B, C>>::from(int_tensor);

    (indicies, distances)
}
