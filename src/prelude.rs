use crate::backend::AutodiffBackend;
use crate::{chart, train, utils, UMAP};

// Re-export common utilities for easier use
pub use chart::{chart_tensor, chart_vector};

use crossbeam_channel::unbounded;
use num::Float;

// Re-export the new umap-rs style API
pub use crate::{
    GraphParams, ManifoldParams, Metric, OptimizationParams, UmapConfig,
};
pub use crate::FittedUmap as FittedUmapExport;

// Re-export legacy types for backward compatibility
pub use train::{TrainingConfig, TrainingConfigBuilder};
pub use utils::generate_test_data;

/// Convenience function for running UMAP with the WGPU backend.
///
/// This creates a `Umap` with default configuration (2-D output) and fits it
/// to the data.
///
/// # Arguments
/// * `data` - A vector of vectors, where each inner vector represents a data sample.
///
/// # Returns
/// A trained `UMAP` model (legacy type). For the new API, use
/// `Umap::new(UmapConfig::default()).fit(data, None)`.
///
/// # Example
/// ```rust,ignore
/// let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let model = umap(data);
/// ```
pub fn umap<B: AutodiffBackend, F: Float>(data: Vec<Vec<F>>) -> UMAP<B>
where
    F: num::FromPrimitive + burn::tensor::Element,
{
    let (exit_tx, exit_rx) = unbounded();

    ctrlc::set_handler(move || exit_tx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    let output_size = 2;
    let device = Default::default();
    let model = UMAP::<B>::fit(data, device, output_size, exit_rx);
    model
}

/// Convenience function for running UMAP with a custom output size.
///
/// # Arguments
/// * `data` - A vector of vectors, where each inner vector represents a data sample.
/// * `output_size` - Number of dimensions for the reduced output.
///
/// # Returns
/// A trained `UMAP` model (legacy type). For the new API, use
/// `Umap::new(UmapConfig { n_components: output_size, ..Default::default() }).fit(data, None)`.
pub fn umap_size<B: AutodiffBackend, F: Float>(data: Vec<Vec<F>>, output_size: usize) -> UMAP<B>
where
    F: num::FromPrimitive + burn::tensor::Element,
{
    let (exit_tx, exit_rx) = unbounded();

    ctrlc::set_handler(move || exit_tx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    let device = Default::default();
    let model = UMAP::<B>::fit(data, device, output_size, exit_rx);
    model
}
