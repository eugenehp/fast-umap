//! Autodiff registration for the Euclidean pairwise distance operation.
//!
//! [`backward`] wires the custom GPU kernel into Burn's autodiff graph so
//! that `loss.backward()` correctly computes `∂loss/∂x` through the
//! pairwise distance computation.

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
    tensor::ops::FloatTensor,
};

use crate::backend::Backend;

/// Register the Euclidean pairwise distance operation in the Burn autodiff graph.
///
/// On the **tracked** path this function:
/// 1. Checkpoints `x` (shape `[n, d]`) for retrieval during the backward pass.
/// 2. Runs `B::euclidean_pairwise_distance` on the inner backend to get the
///    `[n, n]` pairwise distance matrix.
/// 3. Stores the pairwise matrix in the op state (avoids recomputing it
///    during backward, which would require another O(n²d) kernel launch).
/// 4. Registers [`EuclideanPairwiseDistanceBackward`] as the backward hook.
///
/// On the **untracked** path it simply runs the forward computation.
///
/// # Arguments
///
/// * `x` — Input tensor of shape `[n, d]` wrapped in the [`Autodiff`] backend.
///
/// # Returns
///
/// The `[n, n]` pairwise distance matrix, wrapped in [`Autodiff`] so that
/// gradients can flow back through it.
pub fn backward<B: Backend, C: CheckpointStrategy>(
    x: FloatTensor<Autodiff<B, C>>,
) -> FloatTensor<Autodiff<B, C>> {
    /// Zero-sized marker struct that implements Burn's [`Backward`] trait for
    /// the Euclidean pairwise distance operation.
    #[derive(Debug)]
    struct EuclideanPairwiseDistanceBackward;

    impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
        /// Backward state: the `NodeId` of `x` (used to register the computed
        /// gradient) plus the precomputed `[n, n]` pairwise distance tensor
        /// saved from the forward pass (avoids recomputing O(n²d) distances).
        type State = (NodeId, FloatTensor<B>);

        /// Called by Burn's autodiff engine during `loss.backward()`.
        ///
        /// Retrieves `x` from the checkpoint, consumes the upstream gradient
        /// `∂loss/∂pairwise`, dispatches the backward GPU kernel via
        /// `B::euclidean_pairwise_distance_backward`, and registers the
        /// resulting `∂loss/∂x` gradient for further propagation.
        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let (node_x, pairwise) = ops.state;

            // Gradient of loss w.r.t. the pairwise distance matrix output.
            let grad_pairwise = grads.consume::<B>(&ops.node);

            // Retrieve the original input x from the checkpoint.
            let x: FloatTensor<B> = checkpointer.retrieve_node_output(node_x);

            // Compute gradient w.r.t. x using the precomputed pairwise distances.
            let grad_x =
                B::euclidean_pairwise_distance_backward(grad_pairwise, x, pairwise);

            grads.register::<B>(node_x, grad_x);
        }
    }

    match EuclideanPairwiseDistanceBackward
        .prepare::<C>([x.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            // Checkpoint x so we can retrieve it during backward.
            let x_state = prep.checkpoint(&x);

            // Run the forward kernel — result is the pairwise distance matrix.
            let pairwise = B::euclidean_pairwise_distance(x.clone().primitive);

            // Store both: the NodeId of x (for retrieval) and the pairwise
            // distances tensor (to avoid recomputing them in backward).
            let state = (x_state, pairwise.clone());

            prep.finish(state, pairwise)
        }
        OpsKind::UnTracked(prep) => {
            let output = B::euclidean_pairwise_distance(x.primitive);
            prep.finish(output)
        }
    }
}
