use burn::tensor::{Float, Tensor, TensorPrimitive};

use crate::backend::Backend;

use super::*;

/// Computes the distance metric for the given data.
///
/// This function calculates the distance between points in the provided `data` tensor
/// according to the metric specified in the `config`. It currently supports the following
/// metrics:
///
/// - `Euclidean`: Computes the Euclidean distance between points.
/// - `EuclideanKNN`: Computes the Euclidean distance using k-nearest neighbors (KNN), with `k_neighbors`
///   determining the number of nearest neighbors to consider.
///
/// # Parameters
/// - `data`: A 2D tensor representing the data points, where each row is a point and each column is a feature.
/// - `config`: The training configuration, which specifies the metric and other parameters like `k_neighbors`.
///
/// # Returns
/// A 1D tensor containing the computed distances for each point based on the selected metric.
///
/// # Type Parameters
/// - `B`: The backend type used for automatic differentiation (AutodiffBackend), which enables GPU or CPU computations.
///
/// # Example
/// ```rust
/// let data = Tensor::from(...); // Some 2D tensor of data points
/// let config = TrainingConfig { metric: Metric::Euclidean, k_neighbors: 5 };
/// let distances = get_distance_by_metric(data, &config);
/// ```
#[allow(unused)]
pub fn get_distance_by_metric<B: Backend>(
    data: Tensor<B, 2>,
    config: &TrainingConfig,
    verbose: Option<String>,
) -> Tensor<B, 2> {
    type F = f32;
    // let verbose = verbose.unwrap_or("".into());
    // let before = data.to_data().to_vec::<F>().unwrap();
    // println!(
    //     "get_distance_by_metric - before - {verbose} - {:?}",
    //     data.shape(),
    // );

    let distance = match config.metric {
        // Metric::Euclidean => euclidean(data),
        // Metric::Euclidean => {
        _ => {
            let x = data.clone().into_primitive().tensor();
            let pairwise_distances = B::euclidean_pairwise_distance(x);

            let (indices, distances) =
                B::knn(pairwise_distances.clone(), config.k_neighbors as u32);

            // TODO: don't clone later, to optimize the speed
            let pairwise_distances: Tensor<B, 2, Float> =
                Tensor::from_primitive(TensorPrimitive::Float(pairwise_distances));
            let distances = Tensor::from_primitive(TensorPrimitive::Float(distances));

            distances
        } // Metric::EuclideanKNN => euclidean_knn(data, config.k_neighbors),
          // Metric::Manhattan => manhattan(data),
          // Metric::Cosine => cosine(data),
          // Metric::Minkowski => minkowski(data, config.minkowski_p),
          // _ => euclidean(data),
    };

    // let after = distance.to_data().to_vec::<F>().unwrap();
    // println!(
    //     "get_distance_by_metric - after - {verbose} - {:?}",
    //     distance.shape(),
    // );

    // match config.normalized {
    //     true => normalize_tensor(distance),
    //     false => distance,
    // }
    distance
}
