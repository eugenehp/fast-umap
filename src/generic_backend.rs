//! Generic implementations of the fast-umap [`Backend`](crate::backend::Backend)
//! trait operations using only standard Burn tensor ops.
//!
//! These free functions work with **any** Burn backend (MLX, NdArray, WGPU, etc.)
//! without requiring CubeCL or GPU-specific kernels.

use burn::tensor::{ops::FloatTensor, ops::IntTensor, Tensor, TensorData, TensorPrimitive};

/// Compute the full `[n, n]` Euclidean pairwise distance matrix using the
/// squared-expansion identity:
///
/// ```text
/// ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
/// ```
///
/// Uses only `matmul`, `sum_dim`, broadcasting, `clamp_min`, and `sqrt` —
/// all standard Burn ops available on every backend.
pub fn euclidean_pairwise_distance<B: burn::tensor::backend::Backend>(
    x: FloatTensor<B>,
) -> FloatTensor<B> {
    let tensor: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(x));
    let [n, _d] = tensor.dims();

    // ||x_i||^2 for each row → [n, 1]
    let sq_norms = tensor
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .reshape([n, 1]);

    // -2 * X @ X^T → [n, n]
    let cross = tensor.clone().matmul(tensor.transpose());
    let neg2_cross = cross.mul_scalar(-2.0f32);

    // ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
    // Broadcasting: [n,1] + [1,n] + [n,n]
    let sq_norms_t = sq_norms.clone().reshape([1, n]);
    let sq_dists = neg2_cross.add(sq_norms).add(sq_norms_t);

    // Numerical safety: clamp negatives from floating-point noise, then sqrt
    let dists = sq_dists.clamp_min(0.0f32).sqrt();

    dists.into_primitive().tensor()
}

/// Backward pass for `euclidean_pairwise_distance`.
///
/// Not needed in the sparse training path (precomputation only, no gradients).
/// Provided for completeness / dense training path.
pub fn euclidean_pairwise_distance_backward<B: burn::tensor::backend::Backend>(
    grad_pairwise: FloatTensor<B>,
    x: FloatTensor<B>,
    pairwise: FloatTensor<B>,
) -> FloatTensor<B> {
    // Analytical gradient:
    // d(dist_ij)/d(x_i_f) = (x_i_f - x_j_f) / dist_ij
    // grad_x[i,f] = sum_j [ (grad[i,j] + grad[j,i]) * (x[i,f] - x[j,f]) / max(dist[i,j], eps) ]
    //
    // We avoid materialising the full [n, n, d] tensor by looping over chunks
    // or using a batched approach. For safety on the sparse path (where this is
    // never called), we use a simple but memory-intensive broadcast approach.

    let x_t: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(x));
    let grad_t: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(grad_pairwise));
    let pw_t: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(pairwise));
    let [n, d] = x_t.dims();

    // Symmetrise the gradient: (grad + grad^T)
    let sym_grad = grad_t.clone().add(grad_t.transpose());

    // Scale by 1/dist (with epsilon for safety): [n, n]
    let inv_dist = pw_t.clamp_min(1e-8f32).recip();
    let scale = sym_grad.mul(inv_dist); // [n, n]

    // grad_x = scale @ x - diag(scale.sum(1)) @ x
    // = scale @ x - (scale.sum(1).unsqueeze(1)) * x
    let grad_x = scale.clone().matmul(x_t.clone())
        - scale.sum_dim(1).reshape([n, 1]).mul(x_t);

    grad_x.into_primitive().tensor()
}

/// CPU-side k-NN selection: pull pairwise distances to CPU, find k nearest
/// neighbours per row, push results back as tensors.
///
/// This reuses [`knn_from_pairwise_cpu`](crate::train::get_distance_by_metric::knn_from_pairwise_cpu).
pub fn knn<B: burn::tensor::backend::Backend>(
    pairwise_distances: FloatTensor<B>,
    k: u32,
) -> (IntTensor<B>, FloatTensor<B>) {
    let pw: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(pairwise_distances));
    let [n, _n2] = pw.dims();
    let device = pw.device();
    let k = k as usize;

    // Pull to CPU
    let flat: Vec<f32> = pw.to_data().to_vec::<f32>().unwrap();

    // Reuse existing CPU k-NN implementation
    let (idx_flat, dist_flat) = crate::train::get_distance_by_metric::knn_from_pairwise_cpu(
        &flat, n, k,
    );

    // Push back as tensors
    let indices: Tensor<B, 2, burn::tensor::Int> =
        Tensor::from_data(TensorData::new(idx_flat, [n, k]), &device);
    let distances: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(dist_flat, [n, k]), &device);

    (indices.into_primitive(), distances.into_primitive().tensor())
}

/// Backward pass for k-NN. Not called in the sparse training path.
pub fn knn_backward<B: burn::tensor::backend::Backend>(
    _pairwise_distances: FloatTensor<B>,
    _k: u32,
    _grad_output: FloatTensor<B>,
) -> FloatTensor<B> {
    unimplemented!(
        "knn_backward is not needed for the sparse training path. \
         Use the GPU (CubeCL) backend for the dense training path."
    );
}
