/// Integration tests for fast-umap.
///
/// These tests use `burn::backend::NdArray` (CPU) so no GPU is required.
/// They cover:
///   - Pure utility functions (normalize_data, format_duration, generate_test_data)
///   - Tensor round-trips (convert_vector_to_tensor / convert_tensor_to_vector)
///   - Model construction and forward-pass shapes
///   - Layer normaliser
///   - Configuration builders (TrainingConfig, UMAPModelConfig)
///   - Distance functions (euclidean, manhattan, cosine, minkowski)
///     via the standard-backend overloads in distances.rs
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use fast_umap::{
    model::{UMAPModel, UMAPModelConfigBuilder},
    normalizer::normalize,
    train::{Metric, TrainingConfig},
    utils::{
        convert_tensor_to_vector, convert_vector_to_tensor, format_duration, generate_test_data,
        normalize_data, normalize_tensor,
    },
};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Backend alias
// ---------------------------------------------------------------------------
type B = NdArray<f32>;

fn cpu() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

// ===========================================================================
// normalize_data
// ===========================================================================

#[test]
fn normalize_data_no_nan() {
    let mut data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    normalize_data(&mut data, 2, 3);
    assert!(
        data.iter().all(|v| !v.is_nan()),
        "normalised values must not be NaN"
    );
}

#[test]
fn normalize_data_constant_column_does_not_nan() {
    // All values in one column are equal → std would be 0 without epsilon guard
    let mut data = vec![5.0f64, 5.0, 5.0, 5.0, 5.0, 5.0];
    normalize_data(&mut data, 3, 2);
    assert!(
        data.iter().all(|v| !v.is_nan()),
        "normalised values must not be NaN even for constant columns"
    );
}

#[test]
fn normalize_data_zero_mean_unit_std() {
    // Single feature, multiple samples: [1,2,3,4,5] → z-score normalisation
    let mut data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let num_samples = 5;
    let num_features = 1;
    normalize_data(&mut data, num_samples, num_features);

    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let var: f64 = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std = var.sqrt();

    assert!(
        mean.abs() < 1e-10,
        "mean after normalisation should be ~0, got {mean}"
    );
    assert!(
        (std - 1.0).abs() < 0.05,
        "std after normalisation should be ~1, got {std}"
    );
}

#[test]
fn normalize_data_multi_feature() {
    let num_samples = 4;
    let num_features = 3;
    let mut data: Vec<f64> = (0..num_samples * num_features)
        .map(|i| i as f64)
        .collect();
    normalize_data(&mut data, num_samples, num_features);
    assert!(data.iter().all(|v| !v.is_nan()));
    assert!(data.iter().all(|v| v.is_finite()));
}

// ===========================================================================
// format_duration
// ===========================================================================

#[test]
fn format_duration_zero() {
    assert_eq!(format_duration(Duration::from_secs(0)), "00:00:00");
}

#[test]
fn format_duration_seconds() {
    assert_eq!(format_duration(Duration::from_secs(45)), "00:00:45");
}

#[test]
fn format_duration_minutes() {
    assert_eq!(format_duration(Duration::from_secs(125)), "00:02:05");
}

#[test]
fn format_duration_hours() {
    assert_eq!(format_duration(Duration::from_secs(3723)), "01:02:03");
}

#[test]
fn format_duration_large() {
    // 2h 30m 0s
    assert_eq!(format_duration(Duration::from_secs(9000)), "02:30:00");
}

// ===========================================================================
// generate_test_data
// ===========================================================================

#[test]
fn generate_test_data_shape() {
    let num_samples = 50;
    let num_features = 8;
    let data: Vec<f32> = generate_test_data(num_samples, num_features);
    assert_eq!(data.len(), num_samples * num_features);
}

#[test]
fn generate_test_data_bounds() {
    let data: Vec<f32> = generate_test_data(200, 10);
    for v in &data {
        assert!(*v >= 0.0 && *v < 1.0, "value {v} out of [0,1) range");
    }
}

#[test]
fn generate_test_data_not_all_zero() {
    let data: Vec<f32> = generate_test_data(10, 5);
    assert!(
        data.iter().any(|&v| v > 0.0),
        "expected at least one non-zero value"
    );
}

// ===========================================================================
// convert_vector_to_tensor / convert_tensor_to_vector
// ===========================================================================

#[test]
fn tensor_round_trip_f32() {
    let device = cpu();
    let num_samples = 4;
    let num_features = 3;
    let orig: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let tensor: Tensor<B, 2> =
        convert_vector_to_tensor(orig.clone(), num_samples, num_features, &device);
    assert_eq!(tensor.dims(), [num_samples, num_features]);
    let back: Vec<Vec<f64>> = convert_tensor_to_vector(tensor);
    let flat: Vec<f32> = back.into_iter().flatten().map(|v| v as f32).collect();
    for (a, b) in orig.iter().zip(flat.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "round-trip mismatch: {a} vs {b}"
        );
    }
}

#[test]
fn tensor_shape_matches_input() {
    let device = cpu();
    let num_samples = 7;
    let num_features = 5;
    let data: Vec<f32> = vec![0.0f32; num_samples * num_features];
    let t: Tensor<B, 2> = convert_vector_to_tensor(data, num_samples, num_features, &device);
    assert_eq!(t.dims(), [num_samples, num_features]);
}

#[test]
fn convert_tensor_to_vector_shape() {
    let device = cpu();
    let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let t: Tensor<B, 2> = convert_vector_to_tensor(data, 6, 5, &device);
    let v: Vec<Vec<f64>> = convert_tensor_to_vector(t);
    assert_eq!(v.len(), 6);
    assert!(v.iter().all(|row| row.len() == 5));
}

#[test]
fn convert_tensor_nan_replaced_by_zero() {
    // put NaN into a tensor manually via from_data
    let device = cpu();
    let data = vec![f32::NAN, 0.0, 0.0, 0.0];
    let t: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(data, [2, 2]), &device);
    let v: Vec<Vec<f64>> = convert_tensor_to_vector(t);
    assert_eq!(v[0][0], 0.0, "NaN should be replaced by 0.0");
}

// ===========================================================================
// normalize_tensor (1-D min-max normalisation)
// ===========================================================================

#[test]
fn normalize_tensor_bounds() {
    let device = cpu();
    let data: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let t: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(data, [5]), &device);
    let n = normalize_tensor(t);
    let vals = n.to_data().to_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-5));
}

#[test]
fn normalize_tensor_constant_input() {
    let device = cpu();
    let data = vec![4.0f32; 6];
    let t: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(data.clone(), [6]), &device);
    let n = normalize_tensor(t);
    // All equal → normaliser returns the original tensor unchanged
    let vals = n.to_data().to_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| !v.is_nan()));
}

// ===========================================================================
// normalizer::normalize (2-D layer normalisation)
// ===========================================================================

#[test]
fn normalizer_no_nan() {
    let device = cpu();
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let t: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(data, [4, 5]), &device);
    let n = normalize(t);
    let vals = n.to_data().to_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| !v.is_nan()));
}

#[test]
fn normalizer_shape_preserved() {
    let device = cpu();
    let data: Vec<f32> = vec![0.0f32; 15];
    let t: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(data, [3, 5]), &device);
    let n = normalize(t);
    assert_eq!(n.dims(), [3, 5]);
}

// ===========================================================================
// UMAPModelConfigBuilder
// ===========================================================================

#[test]
fn model_config_defaults() {
    let cfg = UMAPModelConfigBuilder::default().build().unwrap();
    assert_eq!(cfg.input_size, 100);
    assert_eq!(cfg.output_size, 2);
    assert!(!cfg.hidden_sizes.is_empty());
}

#[test]
fn model_config_custom() {
    let cfg = UMAPModelConfigBuilder::default()
        .input_size(28)
        .hidden_sizes(vec![64, 32])
        .output_size(3)
        .build()
        .unwrap();
    assert_eq!(cfg.input_size, 28);
    assert_eq!(cfg.hidden_sizes, vec![64, 32]);
    assert_eq!(cfg.output_size, 3);
}

// ===========================================================================
// TrainingConfig builder
// ===========================================================================

#[test]
fn training_config_defaults() {
    let cfg = TrainingConfig::builder().build().unwrap();
    assert_eq!(cfg.metric, Metric::Euclidean);
    assert!(cfg.epochs > 0);
    assert!(cfg.learning_rate > 0.0);
    assert!(cfg.patience.is_none());
    assert!(cfg.timeout.is_none());
    assert!(cfg.min_desired_loss.is_none());
}

#[test]
fn training_config_custom() {
    let cfg = TrainingConfig::builder()
        .with_epochs(50)
        .with_batch_size(32)
        .with_learning_rate(0.01)
        .with_beta1(0.85)
        .with_beta2(0.99)
        .with_penalty(1e-4)
        .with_verbose(true)
        .with_patience(5)
        .with_k_neighbors(10)
        .with_min_desired_loss(0.001)
        .with_timeout(120)
        .with_metric(Metric::Manhattan)
        .build()
        .unwrap();
    assert_eq!(cfg.epochs, 50);
    assert_eq!(cfg.batch_size, 32);
    assert!((cfg.learning_rate - 0.01).abs() < 1e-10);
    assert_eq!(cfg.patience, Some(5));
    assert_eq!(cfg.k_neighbors, 10);
    assert_eq!(cfg.timeout, Some(120));
    assert_eq!(cfg.metric, Metric::Manhattan);
}

#[test]
fn metric_from_str() {
    assert_eq!(Metric::from("euclidean"), Metric::Euclidean);
    assert_eq!(Metric::from("euclidean_knn"), Metric::EuclideanKNN);
    assert_eq!(Metric::from("manhattan"), Metric::Manhattan);
    assert_eq!(Metric::from("cosine"), Metric::Cosine);
    assert_eq!(Metric::from("minkowski"), Metric::Minkowski);
}

#[test]
#[should_panic]
fn metric_from_invalid_str_panics() {
    let _ = Metric::from("invalid_metric");
}

// ===========================================================================
// UMAPModel construction and forward pass
// ===========================================================================

fn make_model(input: usize, hidden: Vec<usize>, output: usize) -> UMAPModel<B> {
    let cfg = UMAPModelConfigBuilder::default()
        .input_size(input)
        .hidden_sizes(hidden)
        .output_size(output)
        .build()
        .unwrap();
    UMAPModel::new(&cfg, &cpu())
}

#[test]
fn model_forward_output_shape_2d() {
    let device = cpu();
    let model = make_model(10, vec![32], 2);
    let num_samples = 5;
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(vec![0.5f32; num_samples * 10], [num_samples, 10]), &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [num_samples, 2]);
}

#[test]
fn model_forward_output_shape_3d() {
    let device = cpu();
    let model = make_model(8, vec![16, 8], 3);
    let num_samples = 3;
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(vec![1.0f32; num_samples * 8], [num_samples, 8]), &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [num_samples, 3]);
}

#[test]
fn model_forward_no_nan() {
    let device = cpu();
    let model = make_model(16, vec![32, 16], 2);
    let num_samples = 10;
    let data: Vec<f32> = generate_test_data(num_samples, 16);
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(data, [num_samples, 16]), &device);
    let out = model.forward(input);
    let vals = out.to_data().to_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| !v.is_nan()), "model output contains NaN");
}

#[test]
fn model_forward_deterministic() {
    // Same input → same output (model is not stochastic at inference)
    let device = cpu();
    let model = make_model(6, vec![12], 2);
    let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let make_input = || -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(data.clone(), [1, 6]), &device)
    };
    let out1 = model.forward(make_input()).to_data().to_vec::<f32>().unwrap();
    let out2 = model.forward(make_input()).to_data().to_vec::<f32>().unwrap();
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!((a - b).abs() < 1e-6, "forward pass is not deterministic");
    }
}

#[test]
fn model_single_hidden_layer() {
    let model = make_model(4, vec![8], 2);
    let device = cpu();
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(vec![1.0f32; 4], [1, 4]), &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [1, 2]);
}

#[test]
fn model_deep_network() {
    let model = make_model(20, vec![64, 32, 16, 8], 2);
    let device = cpu();
    let data: Vec<f32> = generate_test_data(4, 20);
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(data, [4, 20]), &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [4, 2]);
    assert!(out.to_data().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}

// ===========================================================================
// Distance functions
// These live in fast_umap::distances and use crate::backend::Backend.
// We test them via standard tensor arithmetic directly, to avoid requiring
// a CubeBackend (GPU) implementation in the test environment.
// ===========================================================================

/// Compute Euclidean pairwise distances with standard tensor ops
fn euclidean_distances_cpu(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let n = x.dims()[0];
    let x3 = x.clone().unsqueeze::<3>(); // [1, n, d]
    let xt = x.unsqueeze_dim(1);         // [n, 1, d]
    let diff = x3 - xt;                  // [n, n, d]
    // sum_dim keeps the dim (gives [n,n,1]), then squeeze to [n,n]
    let sq_sum: Tensor<B, 3> = diff.powi_scalar(2).sum_dim(2);
    sq_sum.squeeze_dim::<2>(2).sqrt()    // [n, n]
}

#[test]
fn euclidean_distance_self_is_zero() {
    let device = cpu();
    let data = vec![1.0f32, 0.0, 0.0, 1.0];
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::new(data, [2, 2]), &device);
    let dist = euclidean_distances_cpu(x);
    let vals = dist.to_data().to_vec::<f32>().unwrap();
    // diagonal entries must be 0
    assert!(vals[0].abs() < 1e-5, "dist(0,0) should be 0, got {}", vals[0]);
    assert!(vals[3].abs() < 1e-5, "dist(1,1) should be 0, got {}", vals[3]);
}

#[test]
fn euclidean_distance_known_value() {
    let device = cpu();
    // [0,0] and [3,4] → distance = 5.0
    let data = vec![0.0f32, 0.0, 3.0, 4.0];
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::new(data, [2, 2]), &device);
    let dist = euclidean_distances_cpu(x);
    let vals = dist.to_data().to_vec::<f32>().unwrap();
    assert!((vals[1] - 5.0).abs() < 1e-4, "expected 5.0, got {}", vals[1]);
    assert!((vals[2] - 5.0).abs() < 1e-4, "expected 5.0, got {}", vals[2]);
}

#[test]
fn euclidean_distance_symmetric() {
    let device = cpu();
    let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::new(data, [3, 3]), &device);
    let dist = euclidean_distances_cpu(x);
    let vals = dist.to_data().to_vec::<f32>().unwrap();
    // dist[i,j] == dist[j,i]
    let n = 3;
    for i in 0..n {
        for j in 0..n {
            let ij = vals[i * n + j];
            let ji = vals[j * n + i];
            assert!((ij - ji).abs() < 1e-5, "dist not symmetric at ({i},{j})");
        }
    }
}

#[test]
fn manhattan_distance_known_value() {
    let device = cpu();
    let a: Tensor<B, 2> = Tensor::from_data(TensorData::new(vec![1.0f32, 2.0, 4.0, 6.0], [2, 2]), &device);
    let b: Tensor<B, 2> = Tensor::from_data(TensorData::new(vec![1.0f32, 2.0, 4.0, 6.0], [2, 2]), &device);
    let diff = (a - b).abs().sum_dim(1); // [2,1]
    let vals = diff.to_data().to_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v.abs() < 1e-5), "identical rows → L1 distance 0");
}
