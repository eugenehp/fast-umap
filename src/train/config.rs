use serde::{Deserialize, Serialize};
use std::fmt;

// ─── UMAP a, b curve fitting ─────────────────────────────────────────────────

/// Fit the UMAP kernel parameters `a` and `b` from `min_dist` and `spread`.
///
/// The kernel is `phi(d) = 1 / (1 + a * d^(2b))`.
/// We fit it to the piecewise target:
///   - `phi(d) = 1.0`                              for `d <= min_dist`
///   - `phi(d) = exp(-(d - min_dist) / spread)`     for `d >  min_dist`
///
/// Uses simple grid search + refinement (runs once at init, ~ms).
pub fn fit_ab(min_dist: f32, spread: f32) -> (f32, f32) {
    // Generate target data
    let n = 300;
    let x_max = 3.0 * spread;
    let xs: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32 * x_max).collect();
    let ys: Vec<f32> = xs
        .iter()
        .map(|&x| {
            if x <= min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    // phi(d; a, b) = 1 / (1 + a * d^(2b))
    // Minimize sum of squared residuals
    let residual = |a: f32, b: f32| -> f32 {
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let pred = 1.0 / (1.0 + a * x.powf(2.0 * b));
                (pred - y) * (pred - y)
            })
            .sum::<f32>()
    };

    // Coarse grid search
    let mut best_a = 1.0f32;
    let mut best_b = 1.0f32;
    let mut best_err = f32::INFINITY;

    for ai in 1..=80 {
        let a = ai as f32 * 0.08;
        for bi in 1..=50 {
            let b = bi as f32 * 0.06;
            let err = residual(a, b);
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    // Fine refinement via coordinate descent
    for _ in 0..100 {
        let step_a = best_a * 0.02;
        let step_b = best_b * 0.02;
        for &da in &[-step_a, 0.0, step_a] {
            for &db in &[-step_b, 0.0, step_b] {
                let a = (best_a + da).max(1e-4);
                let b = (best_b + db).max(1e-4);
                let err = residual(a, b);
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_b = b;
                }
            }
        }
    }

    (best_a, best_b)
}

// ─── Metric ──────────────────────────────────────────────────────────────────

/// Distance metric used to build the high-dimensional k-NN graph during the
/// precomputation phase.
///
/// The choice of metric determines how "closeness" is measured in the original
/// feature space.  [`Euclidean`](Metric::Euclidean) (L2) is the default and
/// works well for most continuous data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Metric {
    /// Standard L2 (Euclidean) distance — default.
    Euclidean,
    /// Euclidean distance computed via the GPU k-NN kernel path.
    EuclideanKNN,
    /// L1 (Manhattan / taxicab) distance.
    Manhattan,
    /// Cosine dissimilarity `1 − cos(θ)`.
    Cosine,
    /// Generalised Minkowski distance of order `p`
    /// (`p = 1` → Manhattan, `p = 2` → Euclidean).
    Minkowski,
}

impl From<&str> for Metric {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "euclidean" => Metric::Euclidean,
            "euclideanknn" | "euclidean_knn" => Metric::EuclideanKNN,
            "manhattan" => Metric::Manhattan,
            "cosine" => Metric::Cosine,
            "minkowski" => Metric::Minkowski,
            _ => panic!("Invalid metric type: {}", s),
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Metric::Euclidean => write!(f, "Euclidean"),
            Metric::EuclideanKNN => write!(f, "Euclidean KNN"),
            Metric::Manhattan => write!(f, "Manhattan"),
            Metric::Cosine => write!(f, "cosine"),
            Metric::Minkowski => write!(f, "minkowski"),
        }
    }
}

// ─── LossReduction ───────────────────────────────────────────────────────────

/// How the per-sample losses are combined into a single scalar for
/// backpropagation.
///
/// * [`Mean`](LossReduction::Mean) - divide by the number of elements
///   (scale-invariant, recommended for most use cases).
/// * [`Sum`](LossReduction::Sum) - sum without normalisation (sensitive to
///   batch size; may require a lower learning rate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossReduction {
    /// Average the loss over all contributing pairs.
    Mean,
    /// Sum the loss over all contributing pairs without normalisation.
    Sum,
}

// ─── ManifoldParams ──────────────────────────────────────────────────────────

/// Configuration for manifold shape and embedding space properties.
///
/// These parameters control the geometric properties of the low-dimensional
/// embedding space and how the manifold is shaped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldParams {
    /// Minimum distance between points in the embedding space.
    ///
    /// Controls how tightly points can be packed together. Smaller values
    /// create more clustered embeddings, larger values spread points out more.
    ///
    /// Default: 0.1
    pub min_dist: f32,

    /// The effective scale of embedded points.
    ///
    /// Together with `min_dist`, this determines the embedding's overall spread.
    ///
    /// Default: 1.0
    pub spread: f32,
}

impl Default for ManifoldParams {
    fn default() -> Self {
        Self {
            min_dist: 0.1,
            spread: 1.0,
        }
    }
}

// ─── GraphParams ─────────────────────────────────────────────────────────────

/// Configuration for k-nearest neighbor graph construction.
///
/// These parameters control how the high-dimensional manifold structure
/// is captured via a fuzzy topological representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphParams {
    /// Number of nearest neighbors to use for manifold approximation.
    ///
    /// Larger values capture more global structure but may miss fine details.
    /// Smaller values focus on local structure but may fragment the manifold.
    ///
    /// Default: 15
    pub n_neighbors: usize,

    /// The distance metric to use for building the k-NN graph.
    ///
    /// Default: Euclidean
    pub metric: Metric,

    /// Whether to normalize distance outputs before use in the loss.
    ///
    /// Default: true
    pub normalized: bool,

    /// The Minkowski `p` parameter (only used when metric is Minkowski).
    ///
    /// Default: 1.0
    pub minkowski_p: f64,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            metric: Metric::Euclidean,
            normalized: true,
            minkowski_p: 1.0,
        }
    }
}

// ─── OptimizationParams ──────────────────────────────────────────────────────

/// Configuration for stochastic gradient descent optimization.
///
/// These parameters control the embedding optimization process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParams {
    /// Number of optimization epochs.
    ///
    /// Default: 100
    pub n_epochs: usize,

    /// The number of samples to process in each training batch.
    ///
    /// Default: 1000
    pub batch_size: usize,

    /// Initial learning rate for the Adam optimizer.
    ///
    /// Default: 0.001
    pub learning_rate: f64,

    /// Beta1 parameter for the Adam optimizer.
    ///
    /// Default: 0.9
    pub beta1: f64,

    /// Beta2 parameter for the Adam optimizer.
    ///
    /// Default: 0.999
    pub beta2: f64,

    /// L2 regularization (weight decay) penalty.
    ///
    /// Default: 1e-5
    pub penalty: f32,

    /// Weight applied to the repulsion term of the UMAP cross-entropy loss.
    ///
    /// Default: 1.0
    pub repulsion_strength: f32,

    /// Number of epochs to wait for improvement before triggering early stopping.
    /// `None` disables early stopping.
    ///
    /// Default: None
    pub patience: Option<i32>,

    /// The method used to reduce the loss (mean or sum).
    ///
    /// Default: Sum
    pub loss_reduction: LossReduction,

    /// Minimum desired loss to achieve before stopping early.
    ///
    /// Default: None
    pub min_desired_loss: Option<f64>,

    /// Maximum training time in seconds. `None` means no limit.
    ///
    /// Default: None
    pub timeout: Option<u64>,

    /// Whether to show detailed progress information during training.
    ///
    /// Default: false
    pub verbose: bool,

    /// Number of negative (repulsion) samples drawn per positive (attraction)
    /// edge each epoch.
    ///
    /// Higher values produce stronger repulsion and better cluster separation
    /// at the cost of more computation per epoch.
    ///
    /// Default: 5
    pub neg_sample_rate: usize,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            n_epochs: 100,
            batch_size: 1000,
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            penalty: 1e-5,
            repulsion_strength: 1.0,
            patience: None,
            loss_reduction: LossReduction::Sum,
            min_desired_loss: None,
            timeout: None,
            verbose: false,
            neg_sample_rate: 5,
        }
    }
}

// ─── UmapConfig ──────────────────────────────────────────────────────────────

/// Complete UMAP configuration.
///
/// Groups all parameters for dimensionality reduction into a coherent structure.
/// All parameter groups have sensible defaults and can be customized individually.
///
/// This struct mirrors the configuration style of
/// [`umap-rs`](https://crates.io/crates/umap-rs) with nested parameter groups.
///
/// # Example
///
/// ```ignore
/// use fast_umap::prelude::*;
///
/// // Use all defaults (2-D output, Euclidean metric)
/// let config = UmapConfig::default();
///
/// // Customize specific groups
/// let config = UmapConfig {
///     n_components: 3,
///     graph: GraphParams {
///         n_neighbors: 30,
///         ..Default::default()
///     },
///     optimization: OptimizationParams {
///         n_epochs: 500,
///         learning_rate: 1e-3,
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapConfig {
    /// Number of dimensions in the output embedding.
    ///
    /// Typically 2 for visualization or 3-50 for downstream ML tasks.
    ///
    /// Default: 2
    pub n_components: usize,

    /// Hidden layer sizes for the parametric neural network.
    ///
    /// Default: [100]
    pub hidden_sizes: Vec<usize>,

    /// Manifold shape configuration.
    pub manifold: ManifoldParams,

    /// Graph construction configuration.
    pub graph: GraphParams,

    /// Optimization configuration.
    pub optimization: OptimizationParams,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            hidden_sizes: vec![100],
            manifold: ManifoldParams::default(),
            graph: GraphParams::default(),
            optimization: OptimizationParams::default(),
        }
    }
}

// ─── TrainingConfig (backward compatibility) ─────────────────────────────────

/// Configuration for training the UMAP model.
///
/// **Deprecated**: Use [`UmapConfig`] instead. This type is provided for
/// backward compatibility and converts to/from `UmapConfig`.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// The distance metric to use for training the model.
    pub metric: Metric,
    /// The total number of epochs to run during training.
    pub epochs: usize,
    /// The number of samples to process in each training batch.
    pub batch_size: usize,
    /// The learning rate for the optimizer.
    pub learning_rate: f64,
    /// The Beta1 parameter for the Adam optimizer.
    pub beta1: f64,
    /// The Beta2 parameter for the Adam optimizer.
    pub beta2: f64,
    /// The L2 regularization (weight decay) penalty.
    pub penalty: f32,
    /// Whether to show detailed progress information during training.
    pub verbose: bool,
    /// The number of epochs to wait for improvement before triggering early stopping.
    pub patience: Option<i32>,
    /// The method used to reduce the loss during training.
    pub loss_reduction: LossReduction,
    /// The number of nearest neighbors to consider.
    pub k_neighbors: usize,
    /// Minimum desired loss to achieve before stopping early.
    pub min_desired_loss: Option<f64>,
    /// Maximum training time in seconds.
    pub timeout: Option<u64>,
    /// Normalize distance output.
    pub normalized: bool,
    /// Minkowski p parameter.
    pub minkowski_p: f64,
    /// Weight applied to the repulsion term.
    pub repulsion_strength: f32,
    /// UMAP kernel parameter `a`, fitted from `min_dist` and `spread`.
    /// Controls the width of the kernel: `q = 1 / (1 + a * d^(2b))`.
    pub kernel_a: f32,
    /// UMAP kernel parameter `b`, fitted from `min_dist` and `spread`.
    /// Controls the decay shape: `q = 1 / (1 + a * d^(2b))`.
    pub kernel_b: f32,
    /// Number of negative samples per positive edge per epoch.
    pub neg_sample_rate: usize,
}

impl TrainingConfig {
    /// Creates a new builder for constructing a `TrainingConfig`.
    pub fn builder() -> TrainingConfigBuilder {
        TrainingConfigBuilder::default()
    }
}

impl From<&UmapConfig> for TrainingConfig {
    fn from(config: &UmapConfig) -> Self {
        let (kernel_a, kernel_b) = fit_ab(config.manifold.min_dist, config.manifold.spread);
        TrainingConfig {
            metric: config.graph.metric.clone(),
            epochs: config.optimization.n_epochs,
            batch_size: config.optimization.batch_size,
            learning_rate: config.optimization.learning_rate,
            beta1: config.optimization.beta1,
            beta2: config.optimization.beta2,
            penalty: config.optimization.penalty,
            verbose: config.optimization.verbose,
            patience: config.optimization.patience,
            loss_reduction: config.optimization.loss_reduction.clone(),
            k_neighbors: config.graph.n_neighbors,
            min_desired_loss: config.optimization.min_desired_loss,
            timeout: config.optimization.timeout,
            normalized: config.graph.normalized,
            minkowski_p: config.graph.minkowski_p,
            repulsion_strength: config.optimization.repulsion_strength,
            kernel_a,
            kernel_b,
            neg_sample_rate: config.optimization.neg_sample_rate,
        }
    }
}

impl From<UmapConfig> for TrainingConfig {
    fn from(config: UmapConfig) -> Self {
        TrainingConfig::from(&config)
    }
}

impl From<&TrainingConfig> for UmapConfig {
    fn from(config: &TrainingConfig) -> Self {
        UmapConfig {
            n_components: 2,
            hidden_sizes: vec![100],
            manifold: ManifoldParams::default(),
            graph: GraphParams {
                n_neighbors: config.k_neighbors,
                metric: config.metric.clone(),
                normalized: config.normalized,
                minkowski_p: config.minkowski_p,
            },
            optimization: OptimizationParams {
                n_epochs: config.epochs,
                batch_size: config.batch_size,
                learning_rate: config.learning_rate,
                beta1: config.beta1,
                beta2: config.beta2,
                penalty: config.penalty,
                repulsion_strength: config.repulsion_strength,
                patience: config.patience,
                loss_reduction: config.loss_reduction.clone(),
                min_desired_loss: config.min_desired_loss,
                timeout: config.timeout,
                verbose: config.verbose,
                neg_sample_rate: config.neg_sample_rate,
            },
        }
    }
}

impl From<TrainingConfig> for UmapConfig {
    fn from(config: TrainingConfig) -> Self {
        UmapConfig::from(&config)
    }
}

/// Builder pattern for constructing a `TrainingConfig` with optional parameters.
#[derive(Default)]
pub struct TrainingConfigBuilder {
    metric: Option<Metric>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    penalty: Option<f32>,
    verbose: Option<bool>,
    patience: Option<i32>,
    loss_reduction: Option<LossReduction>,
    k_neighbors: Option<usize>,
    min_desired_loss: Option<f64>,
    timeout: Option<u64>,
    normalized: Option<bool>,
    minkowski_p: Option<f64>,
    repulsion_strength: Option<f32>,
    neg_sample_rate: Option<usize>,
}

impl TrainingConfigBuilder {
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = Some(metric);
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = Some(beta1);
        self
    }

    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = Some(beta2);
        self
    }

    pub fn with_penalty(mut self, penalty: f32) -> Self {
        self.penalty = Some(penalty);
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }

    pub fn with_patience(mut self, patience: i32) -> Self {
        self.patience = Some(patience);
        self
    }

    pub fn with_loss_reduction(mut self, loss_reduction: LossReduction) -> Self {
        self.loss_reduction = Some(loss_reduction);
        self
    }

    pub fn with_k_neighbors(mut self, k_neighbors: usize) -> Self {
        self.k_neighbors = Some(k_neighbors);
        self
    }

    pub fn with_min_desired_loss(mut self, min_desired_loss: f64) -> Self {
        self.min_desired_loss = Some(min_desired_loss);
        self
    }

    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = Some(normalized);
        self
    }

    pub fn with_minkowski_p(mut self, minkowski_p: f64) -> Self {
        self.minkowski_p = Some(minkowski_p);
        self
    }

    pub fn with_repulsion_strength(mut self, repulsion_strength: f32) -> Self {
        self.repulsion_strength = Some(repulsion_strength);
        self
    }

    pub fn with_neg_sample_rate(mut self, neg_sample_rate: usize) -> Self {
        self.neg_sample_rate = Some(neg_sample_rate);
        self
    }

    pub fn build(self) -> Option<TrainingConfig> {
        let defaults = ManifoldParams::default();
        let (kernel_a, kernel_b) = fit_ab(defaults.min_dist, defaults.spread);
        Some(TrainingConfig {
            metric: self.metric.unwrap_or(Metric::Euclidean),
            epochs: self.epochs.unwrap_or(1000),
            batch_size: self.batch_size.unwrap_or(1000),
            learning_rate: self.learning_rate.unwrap_or(0.001),
            beta1: self.beta1.unwrap_or(0.9),
            beta2: self.beta2.unwrap_or(0.999),
            penalty: self.penalty.unwrap_or(1e-5),
            verbose: self.verbose.unwrap_or(false),
            patience: self.patience,
            loss_reduction: self.loss_reduction.unwrap_or(LossReduction::Sum),
            k_neighbors: self.k_neighbors.unwrap_or(15),
            min_desired_loss: self.min_desired_loss,
            timeout: self.timeout,
            normalized: self.normalized.unwrap_or(true),
            minkowski_p: self.minkowski_p.unwrap_or(1.0),
            repulsion_strength: self.repulsion_strength.unwrap_or(1.0),
            kernel_a,
            kernel_b,
            neg_sample_rate: self.neg_sample_rate.unwrap_or(5),
        })
    }
}
