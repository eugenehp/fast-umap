use burn::tensor::{Tensor, TensorPrimitive};

use crate::{kernels::AutodiffBackend, normalize_tensor};

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
pub fn get_distance_by_metric<B: AutodiffBackend>(
    data: Tensor<B, 2>,
    config: &TrainingConfig<B>,
) -> Tensor<B, 1> {
    let distance = match config.metric {
        // Metric::Euclidean => euclidean(data),
        Metric::Euclidean => {
            let x = data.into_primitive().tensor();
            let output = B::euclidean_pairwise_distance(x);
            Tensor::from_primitive(TensorPrimitive::Float(output))
        }
        Metric::EuclideanKNN => euclidean_knn(data, config.k_neighbors),
        Metric::Manhattan => manhattan(data),
        Metric::Cosine => cosine(data),
        Metric::Minkowski => minkowski(data, config.minkowski_p),
        // _ => euclidean(data),
    };

    match config.normalized {
        true => normalize_tensor(distance),
        false => distance,
    }
}
