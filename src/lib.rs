//! # fast-umap
//!
//! GPU-accelerated parametric UMAP (Uniform Manifold Approximation and Projection)
//! in Rust, built on [burn](https://github.com/tracel-ai/burn) +
//! [CubeCL](https://github.com/tracel-ai/cubecl).
//!
//! **Up to 4.7× faster** than [umap-rs](https://crates.io/crates/umap-rs) on
//! datasets ≥ 10 000 samples, with the ability to [`transform()`](FittedUmap::transform)
//! new unseen data (something classical UMAP cannot do).
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use fast_umap::prelude::*;
//!
//! // 100 samples × 10 features
//! let data: Vec<Vec<f64>> = generate_test_data(100, 10)
//!     .chunks(10)
//!     .map(|c| c.iter().map(|&x: &f32| x as f64).collect())
//!     .collect();
//!
//! // Configure and fit UMAP
//! let config = UmapConfig::default();
//! // let umap = Umap::<MyAutodiffBackend>::new(config);
//! // let fitted = umap.fit(data.clone(), None);
//! // let embedding = fitted.embedding();
//! // let new_embedding = fitted.transform(new_data);
//! ```
//!
//! ## Interface
//!
//! The public API mirrors the [`umap-rs`](https://crates.io/crates/umap-rs) crate:
//!
//! * [`Umap`] — Main algorithm struct, created via `Umap::new(config)`
//! * [`FittedUmap`] — Fitted model returned from `umap.fit(data)`, with
//!   [`embedding()`](FittedUmap::embedding),
//!   [`into_embedding()`](FittedUmap::into_embedding),
//!   [`transform()`](FittedUmap::transform), and
//!   [`config()`](FittedUmap::config)
//! * [`UmapConfig`] — Configuration with nested [`GraphParams`] and [`OptimizationParams`]
//! * [`Metric`] — Distance metric enum (`Euclidean`, `Manhattan`, `Cosine`, …)
//!
//! ## Performance
//!
//! | Dataset | fast-umap | umap-rs | Speedup |
//! |---------|-----------|---------|---------|
//! | 5 000 × 100 | 6.75s | 2.31s | 0.34× *(umap-rs faster)* |
//! | 10 000 × 100 | 5.93s | 8.68s | **1.5× faster** |
//! | 20 000 × 100 | 7.32s | 34.10s | **4.7× faster** |
//!
//! Benchmarked on Apple M3 Max. Reproduce with
//! `cargo run --release --example crate_comparison`.
//!
//! ## Architecture
//!
//! The dimensionality reduction is performed by a small feed-forward neural
//! network (`UMAPModel`) trained with the UMAP cross-entropy loss using
//! sparse edge subsampling and negative sampling:
//!
//! ```text
//! attraction  =  mean_{sampled k-NN edges}   [ −log q_ij ]
//! repulsion   =  mean_{negative samples}     [ −log (1 − q_ij) ]
//! loss        =  attraction  +  repulsion_strength × repulsion
//! ```
//!
//! where `q_ij = 1 / (1 + a · d_ij^(2b))` is the UMAP kernel applied to
//! embedding distances (`a` and `b` are fitted from `min_dist` / `spread`).
//! Per-epoch cost is O(min(n·k, 50K)) regardless of dataset size.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`model`] | `UMAPModel` neural network and config builder |
//! | [`train`] | Training loop, `UmapConfig`, sparse training, loss computation |
//! | [`chart`] | 2-D scatter plots and loss curves (plotters) |
//! | [`utils`] | Data generation, tensor conversion, normalisation |
//! | [`kernels`] | Custom CubeCL GPU kernels (Euclidean distance, k-NN) |
//! | [`backend`] | Backend trait extension for custom kernel dispatch |
//! | [`distances`] | CPU-side distance functions (Euclidean, cosine, Minkowski…) |
//! | [`prelude`] | Re-exports of the most commonly used items |

use backend::AutodiffBackend;
use burn::module::AutodiffModule;

use crossbeam_channel::Receiver;
use model::{UMAPModel, UMAPModelConfigBuilder};
use num::Float;
use train::*;
use utils::*;

use burn::tensor::{Device, Tensor};

pub mod backend;
pub mod chart;
pub mod distances;
pub mod kernels;
pub mod macros;
pub mod model;
pub mod normalizer;
pub mod prelude;
pub mod train;
pub mod utils;

// Re-export config types at crate root for umap-rs style access
pub use train::{
    GraphParams, LossReduction, ManifoldParams, Metric, OptimizationParams, TrainingConfig,
    TrainingConfigBuilder, UmapConfig,
};

/// UMAP dimensionality reduction algorithm (GPU-accelerated, parametric).
///
/// This struct holds the configuration for UMAP. Use [`Umap::new`] to create
/// an instance, then call [`fit`](Umap::fit) to train and obtain a
/// [`FittedUmap`].
///
/// The interface mirrors [`umap-rs`](https://crates.io/crates/umap-rs), but
/// this implementation uses a parametric neural network trained on a GPU,
/// which means:
/// - It can [`transform`](FittedUmap::transform) new (unseen) data
/// - It leverages GPU acceleration via burn/CubeCL
/// - It does **not** require precomputed KNN (computed internally per batch)
///
/// # Example
///
/// ```ignore
/// use fast_umap::prelude::*;
///
/// let config = UmapConfig {
///     n_components: 2,
///     graph: GraphParams {
///         n_neighbors: 15,
///         ..Default::default()
///     },
///     optimization: OptimizationParams {
///         n_epochs: 200,
///         ..Default::default()
///     },
///     ..Default::default()
/// };
///
/// let umap = Umap::<MyAutodiffBackend>::new(config);
/// let fitted = umap.fit(data, None);
/// let embedding = fitted.embedding();
/// ```
pub struct Umap<B: AutodiffBackend> {
    config: UmapConfig,
    device: Device<B>,
}

impl<B: AutodiffBackend> Umap<B> {
    /// Create a new UMAP instance with the given configuration.
    ///
    /// Uses the default device for the backend.
    ///
    /// # Arguments
    ///
    /// * `config` - UMAP configuration parameters
    pub fn new(config: UmapConfig) -> Self {
        Self {
            config,
            device: Default::default(),
        }
    }

    /// Create a new UMAP instance with a specific device.
    ///
    /// # Arguments
    ///
    /// * `config` - UMAP configuration parameters
    /// * `device` - The device (CPU or GPU) on which to perform computation
    pub fn with_device(config: UmapConfig, device: Device<B>) -> Self {
        Self { config, device }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &UmapConfig {
        &self.config
    }

    /// Fit the UMAP model to the given data.
    ///
    /// Trains a parametric neural network to learn a low-dimensional embedding
    /// of the input data. The model can then be used to transform new data.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data as a vector of vectors (n_samples × n_features)
    /// * `labels` - Optional per-sample labels for coloured epoch snapshots
    ///
    /// # Returns
    ///
    /// A [`FittedUmap`] containing the trained model and embedding.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = UmapConfig::default();
    /// let umap = Umap::<MyAutodiffBackend>::new(config);
    /// let fitted = umap.fit(data, None);
    /// let embedding = fitted.embedding();
    /// ```
    pub fn fit<F: Float>(self, data: Vec<Vec<F>>, labels: Option<Vec<String>>) -> FittedUmap<B>
    where
        F: num::FromPrimitive + burn::tensor::Element,
    {
        let (exit_tx, exit_rx) = crossbeam_channel::unbounded();

        ctrlc::set_handler(move || {
            let _ = exit_tx.send(());
        })
        .ok(); // Ignore error if handler already set

        self.fit_with_signal(data, labels, exit_rx)
    }

    /// Fit the UMAP model with an external cancellation signal.
    ///
    /// Same as [`fit`](Umap::fit) but accepts an external `Receiver<()>` for
    /// graceful cancellation (e.g., from Ctrl-C).
    ///
    /// # Arguments
    ///
    /// * `data` - Input data as a vector of vectors (n_samples × n_features)
    /// * `labels` - Optional per-sample labels for coloured epoch snapshots
    /// * `exit_rx` - Channel receiver for cancellation signal
    ///
    /// # Returns
    ///
    /// A [`FittedUmap`] containing the trained model and embedding.
    pub fn fit_with_signal<F: Float>(
        self,
        data: Vec<Vec<F>>,
        labels: Option<Vec<String>>,
        exit_rx: Receiver<()>,
    ) -> FittedUmap<B>
    where
        F: num::FromPrimitive + burn::tensor::Element,
    {
        let default_name = "model";
        let num_samples = data.len();
        let num_features = data[0].len();
        let batch_size = num_samples;

        let seed = 9999;
        B::seed(&self.device, seed);

        // Flatten input data
        let train_data: Vec<F> = data.into_iter().flatten().collect();

        // Build model config
        let model_config = UMAPModelConfigBuilder::default()
            .input_size(num_features)
            .hidden_sizes(self.config.hidden_sizes.clone())
            .output_size(self.config.n_components)
            .build()
            .unwrap();

        let model: UMAPModel<B> = UMAPModel::new(&model_config, &self.device);

        // Build training config from UmapConfig
        let (kernel_a, kernel_b) = train::fit_ab(
            self.config.manifold.min_dist,
            self.config.manifold.spread,
        );
        let training_config = TrainingConfig {
            metric: self.config.graph.metric.clone(),
            epochs: self.config.optimization.n_epochs,
            batch_size,
            learning_rate: self.config.optimization.learning_rate,
            beta1: self.config.optimization.beta1,
            beta2: self.config.optimization.beta2,
            penalty: self.config.optimization.penalty,
            verbose: self.config.optimization.verbose,
            patience: self.config.optimization.patience,
            loss_reduction: self.config.optimization.loss_reduction.clone(),
            k_neighbors: self.config.graph.n_neighbors,
            min_desired_loss: self.config.optimization.min_desired_loss,
            timeout: self.config.optimization.timeout,
            normalized: self.config.graph.normalized,
            minkowski_p: self.config.graph.minkowski_p,
            repulsion_strength: self.config.optimization.repulsion_strength,
            kernel_a,
            kernel_b,
            neg_sample_rate: self.config.optimization.neg_sample_rate,
        };

        // Use the sparse training path (O(n·k) per epoch) by default.
        // Falls back to dense path only if explicitly requested in the future.
        let (model, _losses, _best_loss): (UMAPModel<B>, Vec<F>, F) = train::train_sparse(
            default_name,
            model,
            num_samples,
            num_features,
            train_data.clone(),
            &training_config,
            self.device.clone(),
            exit_rx,
            labels,
        );

        let model: UMAPModel<B::InnerBackend> = model.valid();

        // Compute the embedding for the training data
        let mut normalized_data = train_data;
        normalize_data(&mut normalized_data, num_samples, num_features);
        let global = convert_vector_to_tensor(
            normalized_data,
            num_samples,
            num_features,
            &self.device,
        );
        let embedding_tensor = model.forward(global);
        let embedding: Vec<Vec<f64>> = convert_tensor_to_vector(embedding_tensor);

        FittedUmap {
            model,
            device: self.device,
            config: self.config,
            embedding,
            num_features,
        }
    }
}

/// A fitted UMAP model containing the trained neural network and embedding.
///
/// This struct is returned by [`Umap::fit`] and provides methods to:
/// - Access the computed embedding via [`embedding`](FittedUmap::embedding)
///   and [`into_embedding`](FittedUmap::into_embedding)
/// - Transform new data via [`transform`](FittedUmap::transform) and
///   [`transform_to_tensor`](FittedUmap::transform_to_tensor)
/// - Access the configuration via [`config`](FittedUmap::config)
///
/// Because this is a parametric UMAP (neural network), it can project unseen
/// data through [`transform`](FittedUmap::transform) — unlike classical UMAP
/// implementations.
pub struct FittedUmap<B: AutodiffBackend> {
    model: UMAPModel<B::InnerBackend>,
    device: Device<B>,
    config: UmapConfig,
    embedding: Vec<Vec<f64>>,
    #[allow(dead_code)]
    num_features: usize,
}

impl<B: AutodiffBackend> FittedUmap<B> {
    /// Get the computed embedding for the training data.
    ///
    /// Returns a reference to the embedding coordinates. Each inner vector
    /// represents one input sample in the low-dimensional space.
    ///
    /// # Returns
    ///
    /// A reference to a `Vec<Vec<f64>>` of shape (n_samples, n_components).
    pub fn embedding(&self) -> &Vec<Vec<f64>> {
        &self.embedding
    }

    /// Consume the fitted model and return the embedding, avoiding a copy.
    ///
    /// # Returns
    ///
    /// The embedding as `Vec<Vec<f64>>` of shape (n_samples, n_components).
    pub fn into_embedding(self) -> Vec<Vec<f64>> {
        self.embedding
    }

    /// Get a reference to the configuration used for this fit.
    pub fn config(&self) -> &UmapConfig {
        &self.config
    }

    /// Transform new data points into the embedding space.
    ///
    /// Projects new, unseen data through the trained neural network to obtain
    /// low-dimensional coordinates. This is possible because fast-umap uses a
    /// parametric (neural network) approach.
    ///
    /// # Arguments
    ///
    /// * `data` - New data points as `Vec<Vec<f64>>` (n_new_samples × n_features)
    ///
    /// # Returns
    ///
    /// Embeddings for the new data points as `Vec<Vec<f64>>` (n_new_samples × n_components).
    pub fn transform(&self, data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let local = self.transform_to_tensor(data);
        convert_tensor_to_vector(local)
    }

    /// Transform new data into the embedding space, returning a tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - New data points as `Vec<Vec<f64>>` (n_new_samples × n_features)
    ///
    /// # Returns
    ///
    /// A 2D tensor of shape `[n_new_samples, n_components]`.
    pub fn transform_to_tensor(&self, data: Vec<Vec<f64>>) -> Tensor<B::InnerBackend, 2> {
        let num_samples = data.len();
        let num_features = data[0].len();

        let train_data: Vec<f64> = data.into_iter().flatten().collect();

        let global = convert_vector_to_tensor(train_data, num_samples, num_features, &self.device);

        self.model.forward(global)
    }
}

// ─── Backward-compatible UMAP type alias ─────────────────────────────────────

/// Legacy UMAP struct — **Deprecated**: Use [`Umap`] and [`FittedUmap`] instead.
///
/// This struct is provided for backward compatibility. New code should use
/// the [`Umap`] / [`FittedUmap`] API which mirrors `umap-rs`.
pub struct UMAP<B: AutodiffBackend> {
    model: UMAPModel<B::InnerBackend>,
    device: Device<B>,
}

impl<B: AutodiffBackend> UMAP<B> {
    /// Trains the UMAP model on the given data and returns a fitted UMAP model.
    ///
    /// **Deprecated**: Use `Umap::new(config).fit(data, None)` instead.
    pub fn fit<F: Float>(
        data: Vec<Vec<F>>,
        device: Device<B>,
        output_size: usize,
        exit_rx: Receiver<()>,
    ) -> Self
    where
        F: num::FromPrimitive + burn::tensor::Element,
    {
        let default_name = "model";
        let num_samples = data.len();
        let num_features = data[0].len();
        let batch_size = num_samples;
        let hidden_sizes = vec![100];
        let learning_rate = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epochs = 100;
        let seed = 9999;

        B::seed(&device, seed);

        let train_data: Vec<F> = data.into_iter().flatten().collect();

        let model_config = UMAPModelConfigBuilder::default()
            .input_size(num_features)
            .hidden_sizes(hidden_sizes)
            .output_size(output_size)
            .build()
            .unwrap();

        let model: UMAPModel<B> = UMAPModel::new(&model_config, &device);

        let config = TrainingConfig::builder()
            .with_epochs(epochs)
            .with_batch_size(batch_size)
            .with_learning_rate(learning_rate)
            .with_beta1(beta1)
            .with_beta2(beta2)
            .build()
            .expect("Failed to build TrainingConfig");

        let (model, _losses, _best_loss): (UMAPModel<B>, Vec<F>, F) = train(
            default_name,
            model,
            num_samples,
            num_features,
            train_data.clone(),
            &config,
            device.clone(),
            exit_rx,
            None,
        );

        let model: UMAPModel<B::InnerBackend> = model.valid();

        UMAP { model, device }
    }

    /// Transforms the input data to a tensor using the trained UMAP model.
    pub fn transform_to_tensor(&self, data: Vec<Vec<f64>>) -> Tensor<B::InnerBackend, 2> {
        let num_samples = data.len();
        let num_features = data[0].len();

        let train_data: Vec<f64> = data.into_iter().flatten().map(|f| f64::from(f)).collect();

        let global = convert_vector_to_tensor(train_data, num_samples, num_features, &self.device);

        let local = self.model.forward(global);

        local
    }

    /// Transforms the input data into a lower-dimensional space.
    pub fn transform(&self, data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let local = self.transform_to_tensor(data);
        convert_tensor_to_vector(local)
    }
}
