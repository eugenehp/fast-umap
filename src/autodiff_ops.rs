//! Autodiff registration for custom fast-umap operations.
//!
//! This module wires the custom [`Backend`] trait operations into Burn's
//! autodiff graph so that `loss.backward()` correctly computes gradients
//! through pairwise distance and k-NN computations.
//!
//! The code is generic over any `B: Backend` and is NOT tied to CubeCL or
//! any specific GPU runtime — it works with MLX, WGPU, NdArray, etc.

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

use crate::backend::Backend;
#[allow(unused_imports)]
use crate::{print_if, print_primitive_tensor};

// ── Euclidean pairwise distance autodiff ─────────────────────────────────────

/// Register the Euclidean pairwise distance operation in the Burn autodiff graph.
///
/// On the **tracked** path this function:
/// 1. Checkpoints `x` (shape `[n, d]`) for retrieval during the backward pass.
/// 2. Runs `B::euclidean_pairwise_distance` on the inner backend.
/// 3. Stores the pairwise matrix in the op state.
/// 4. Registers the backward hook.
///
/// On the **untracked** path it simply runs the forward computation.
pub fn euclidean_pairwise_distance_backward<B: Backend, C: CheckpointStrategy>(
    x: FloatTensor<Autodiff<B, C>>,
) -> FloatTensor<Autodiff<B, C>> {
    #[derive(Debug)]
    struct EuclideanPairwiseDistanceBackward;

    impl<B: Backend> Backward<B, 1> for EuclideanPairwiseDistanceBackward {
        type State = (NodeId, FloatTensor<B>);

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let (node_x, pairwise) = ops.state;
            let grad_pairwise = grads.consume::<B>(&ops.node);
            let x: FloatTensor<B> = checkpointer.retrieve_node_output(node_x);
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
            let x_state = prep.checkpoint(&x);
            let pairwise = B::euclidean_pairwise_distance(x.clone().primitive);
            let state = (x_state, pairwise.clone());
            prep.finish(state, pairwise)
        }
        OpsKind::UnTracked(prep) => {
            let output = B::euclidean_pairwise_distance(x.primitive);
            prep.finish(output)
        }
    }
}

// ── k-NN autodiff ────────────────────────────────────────────────────────────

/// Register the k-NN operation in the Burn autodiff graph.
///
/// On the **tracked** path: checkpoints `pairwise_distances`, runs forward
/// k-NN, and registers the backward hook.
///
/// On the **untracked** path: runs the forward pass only.
pub fn knn_backward<B: Backend, C: CheckpointStrategy>(
    pairwise_distances: FloatTensor<Autodiff<B, C>>,
    k: u32,
) -> (IntTensor<Autodiff<B, C>>, FloatTensor<Autodiff<B, C>>) {
    const VERBOSE: bool = false;

    #[derive(Debug)]
    struct KnnBackward;

    impl<B: Backend> Backward<B, 1> for KnnBackward {
        type State = (NodeId, u32);

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let (node_pairwise_distances, k) = ops.state;
            let grad_output = grads.consume::<B>(&ops.node);
            let pairwise_distances: FloatTensor<B> =
                checkpointer.retrieve_node_output(node_pairwise_distances);

            if VERBOSE {
                println!("grad_output {grad_output:?}");
                print_primitive_tensor::<B>(&grad_output, 10, 10);
                println!("pairwise_distances {pairwise_distances:?}");
                print_primitive_tensor::<B>(&pairwise_distances, 10, 10);
            }

            let grad_pairwise_distances = B::knn_backward(pairwise_distances, k, grad_output);

            if VERBOSE {
                println!("===grad_pairwise_distances=== {grad_pairwise_distances:?}");
                print_primitive_tensor::<B>(&grad_pairwise_distances, 0, 0);
            }

            grads.register::<B>(node_pairwise_distances, grad_pairwise_distances);
        }
    }

    // Indices branch
    let indicies = match KnnBackward
        .prepare::<C>([pairwise_distances.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            let pairwise_distances_state = prep.checkpoint(&pairwise_distances);
            let (indicies, _distances) = B::knn(pairwise_distances.clone().primitive, k);
            let state = (pairwise_distances_state, k);
            let indicies = B::int_into_float(indicies);
            prep.finish(state, indicies)
        }
        OpsKind::UnTracked(prep) => {
            let (indicies, _distances) = B::knn(pairwise_distances.clone().primitive, k);
            let indicies = B::int_into_float(indicies);
            prep.finish(indicies)
        }
    };

    // Distances branch
    let distances = match KnnBackward
        .prepare::<C>([pairwise_distances.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            let pairwise_distances_state = prep.checkpoint(&pairwise_distances);
            let (_indicies, distances) = B::knn(pairwise_distances.clone().primitive, k);
            let state = (pairwise_distances_state, k);
            prep.finish(state, distances)
        }
        OpsKind::UnTracked(prep) => {
            let (_indicies, distances) = B::knn(pairwise_distances.clone().primitive, k);
            prep.finish(distances)
        }
    };

    // Convert indices from float back to int
    let inner_tensor = indicies.into_primitive();
    let int_tensor = B::float_into_int(inner_tensor);
    let indicies: IntTensor<Autodiff<B, C>> = IntTensor::<Autodiff<B, C>>::from(int_tensor);

    (indicies, distances)
}

// ── Autodiff Backend impl ────────────────────────────────────────────────────

/// Implementation of the fast-umap [`Backend`] trait for [`Autodiff`]-wrapped
/// backends.
///
/// This is generic over any `B: Backend` — it works with CubeBackend (WGPU),
/// MLX, or any other backend that implements the custom trait.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    const USE_GATHER_FOR_SELECT: bool = B::USE_GATHER_FOR_SELECT;

    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean_pairwise_distance_backward::<B, C>(x)
    }

    fn euclidean_pairwise_distance_backward(
        _grad_pairwise: FloatTensor<Self>,
        _x: FloatTensor<Self>,
        _pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unimplemented!(
            "Called on inner backend only; Autodiff dispatches via euclidean_pairwise_distance_backward."
        );
    }

    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        knn_backward::<B, C>(pairwise_distances, k)
    }

    fn knn_backward(
        _pairwise_distances: FloatTensor<Self>,
        _k: u32,
        _grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unimplemented!(
            "Triggered on the inner backend only; \
             the Autodiff wrapper delegates via knn_backward."
        );
    }
}
