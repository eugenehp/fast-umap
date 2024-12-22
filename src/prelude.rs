use crate::backend::AutodiffBackend;
use crate::{chart, train, utils, UMAP};

// Re-export common utilities for easier use
pub use chart::{chart_tensor, chart_vector};

use num::Float;
pub use train::Metric;
pub use train::{TrainingConfig, TrainingConfigBuilder};
pub use utils::generate_test_data;

/// Convenience function for running UMAP with the WGPU backend.
///
/// # Arguments
/// * `data` - A vector of vectors, where each inner vector represents a data sample with multiple features.
///
/// # Returns
/// A trained `UMAP` model that has been fitted to the input data, using the WGPU backend for computation.
///
/// This function wraps the `UMAP::fit` method and provides a simplified way to fit UMAP models with the WGPU backend.
/// The resulting model will have 2-dimensional output by default.
///
/// # Example
/// ```rust
/// let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let model = umap(data);
/// ```
pub fn umap<B: AutodiffBackend, F: Float>(data: Vec<Vec<F>>) -> UMAP<B>
where
    F: num::FromPrimitive + burn::tensor::Element,
{
    let output_size = 2;
    let device = Default::default();
    let model = UMAP::<B>::fit(data, device, output_size);
    model
}

/// Convenience function for running UMAP with the WGPU backend and a custom output size.
///
/// # Arguments
/// * `data` - A vector of vectors, where each inner vector represents a data sample with multiple features.
/// * `output_size` - The number of dimensions for the reduced output. This controls the dimensionality of the embedding space.
///
/// # Returns
/// A trained `UMAP` model that has been fitted to the input data, using the WGPU backend for computation and the specified output size.
///
/// This function wraps the `UMAP::fit` method, providing a way to fit UMAP models with the WGPU backend and a customizable number of output dimensions.
///
/// # Example
/// ```rust
/// let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let output_size = 3;
/// let model = umap_size(data, output_size);
/// ```
pub fn umap_size<B: AutodiffBackend, F: Float>(data: Vec<Vec<F>>, output_size: usize) -> UMAP<B>
where
    F: num::FromPrimitive + burn::tensor::Element,
{
    let device = Default::default();
    let model = UMAP::<B>::fit(data, device, output_size);
    model
}
