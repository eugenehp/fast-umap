/// Criterion benchmarks for fast-umap.
///
/// All benchmarks run on the CPU (`burn::backend::NdArray`) so no GPU is
/// required and they integrate easily into CI.
///
/// Benchmark groups:
///   • normalize_data      – in-place Z-score normalisation
///   • generate_test_data  – random data generation
///   • tensor_convert      – Vec<F> ↔ Tensor round-trips
///   • model_forward       – UMAPModel forward pass at various sizes
///   • normalize_tensor    – 1-D min-max normalisation
///   • layer_normalize     – 2-D layer normalisation
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fast_umap::{
    model::{UMAPModel, UMAPModelConfigBuilder},
    normalizer::normalize as layer_normalize,
    utils::{
        convert_tensor_to_vector, convert_vector_to_tensor, generate_test_data, normalize_data,
        normalize_tensor,
    },
};

type B = NdArray<f32>;

fn cpu() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

// ---------------------------------------------------------------------------
// Helper: build a model
// ---------------------------------------------------------------------------
fn build_model(input: usize, hidden: &[usize], output: usize) -> UMAPModel<B> {
    let cfg = UMAPModelConfigBuilder::default()
        .input_size(input)
        .hidden_sizes(hidden.to_vec())
        .output_size(output)
        .build()
        .unwrap();
    UMAPModel::new(&cfg, &cpu())
}

// ---------------------------------------------------------------------------
// normalize_data
// ---------------------------------------------------------------------------
fn bench_normalize_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_data");

    for (samples, features) in [(100, 10), (500, 30), (1_000, 50), (5_000, 100)] {
        let id = BenchmarkId::from_parameter(format!("{samples}×{features}"));
        group.bench_with_input(id, &(samples, features), |b, &(s, f)| {
            b.iter(|| {
                let mut data: Vec<f64> = generate_test_data(s, f);
                normalize_data(&mut data, s, f);
                data
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// generate_test_data
// ---------------------------------------------------------------------------
fn bench_generate_test_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_test_data");

    for (samples, features) in [(100, 10), (500, 30), (1_000, 50), (5_000, 100)] {
        let id = BenchmarkId::from_parameter(format!("{samples}×{features}"));
        group.bench_with_input(id, &(samples, features), |b, &(s, f)| {
            b.iter(|| -> Vec<f32> { generate_test_data(s, f) });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// tensor round-trip: Vec → Tensor → Vec
// ---------------------------------------------------------------------------
fn bench_tensor_convert(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_convert");

    for (samples, features) in [(100, 10), (500, 30), (1_000, 50)] {
        let id = BenchmarkId::from_parameter(format!("{samples}×{features}"));
        group.bench_with_input(id, &(samples, features), |b, &(s, f)| {
            let data: Vec<f32> = generate_test_data(s, f);
            let device = cpu();
            b.iter(|| {
                let t: Tensor<B, 2> =
                    convert_vector_to_tensor(data.clone(), s, f, &device);
                let _v: Vec<Vec<f64>> = convert_tensor_to_vector(t);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// UMAPModel::forward
// ---------------------------------------------------------------------------
fn bench_model_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_forward");

    let configs: &[(&str, usize, &[usize], usize, usize)] = &[
        ("16s-10f-32h-2o",  16,   &[32],       10, 2),
        ("64s-50f-64h-2o",  64,   &[64],       50, 2),
        ("128s-50f-128h-2o", 128, &[128],      50, 2),
        ("64s-100f-128h-3o", 64,  &[128, 64],  100, 3),
        ("256s-100f-256h-2o",256, &[256, 128], 100, 2),
    ];

    for &(label, num_samples, hidden, num_features, output_size) in configs {
        let id = BenchmarkId::from_parameter(label);
        group.bench_with_input(id, &(num_samples, num_features), |b, &(s, f)| {
            let model = build_model(f, hidden, output_size);
            let device = cpu();
            let data: Vec<f32> = generate_test_data(s, f);
            let input: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(data, [s, f]), &device);
            b.iter(|| {
                let _out = model.forward(input.clone());
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// normalize_tensor (1-D min-max)
// ---------------------------------------------------------------------------
fn bench_normalize_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_tensor");

    for n in [64, 512, 4_096, 32_768] {
        let id = BenchmarkId::from_parameter(n);
        group.bench_with_input(id, &n, |b, &n| {
            let device = cpu();
            let data: Vec<f32> = generate_test_data(n, 1);
            b.iter(|| {
                let t: Tensor<B, 1> =
                    Tensor::from_data(TensorData::new(data.clone(), [n]), &device);
                normalize_tensor(t)
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Layer normalisation (2-D)
// ---------------------------------------------------------------------------
fn bench_layer_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_normalize");

    for (samples, features) in [(32, 16), (128, 64), (512, 128), (1_000, 256)] {
        let id = BenchmarkId::from_parameter(format!("{samples}×{features}"));
        group.bench_with_input(id, &(samples, features), |b, &(s, f)| {
            let device = cpu();
            let data: Vec<f32> = generate_test_data(s, f);
            let t: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(data, [s, f]), &device);
            b.iter(|| layer_normalize(t.clone()));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
criterion_group!(
    benches,
    bench_normalize_data,
    bench_generate_test_data,
    bench_tensor_convert,
    bench_model_forward,
    bench_normalize_tensor,
    bench_layer_normalize,
);
criterion_main!(benches);
