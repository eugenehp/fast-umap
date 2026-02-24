/// MNIST benchmark — trains UMAP on 10K MNIST digits and generates figures.
///
/// Outputs:
///     figures/mnist.png          — 2-D embedding coloured by digit class
///     figures/losses_mnist.png   — loss curve
///
/// Usage:
///     cargo run --release --example bench_mnist
use cubecl::wgpu::WgpuRuntime;
use fast_umap::{chart, prelude::*};
use mnist::*;
use std::time::Instant;

fn main() {
    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           MNIST Benchmark — fast-umap                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let num_samples = 10_000_usize;
    let num_features = 28 * 28; // 784

    println!("  Loading MNIST ({num_samples} samples, {num_features} features)…");
    let load_start = Instant::now();
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(num_samples as u32)
        .finalize();

    let data: Vec<Vec<f64>> = trn_img
        .chunks(num_features)
        .map(|chunk| chunk.iter().map(|&b| b as f64).collect())
        .collect();

    let labels: Vec<String> = trn_lbl.iter().map(|d| format!("{d}")).collect();
    println!("  Loaded in {:.2}s", load_start.elapsed().as_secs_f64());
    println!();

    // ── Train ────────────────────────────────────────────────────────────────
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![256],
        graph: GraphParams {
            n_neighbors: 15,
            metric: Metric::Euclidean,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 1_000,
            batch_size: 2_000,
            learning_rate: 1e-3,
            patience: Some(100),
            repulsion_strength: 3.0,
            penalty: 1e-6,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    println!("  Training UMAP ({} epochs, k={})…",
        config.optimization.n_epochs, config.graph.n_neighbors);
    let fit_start = Instant::now();

    let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);
    let fitted = umap.fit(data, Some(labels.clone()));

    let fit_secs = fit_start.elapsed().as_secs_f64();
    println!();
    println!("  Fit time: {fit_secs:.2}s");
    println!("  Embedding: {} × {}", fitted.embedding().len(), fitted.embedding()[0].len());

    // ── Generate figures ─────────────────────────────────────────────────────
    std::fs::create_dir_all("figures").unwrap();

    println!();
    println!("  Writing figures/mnist.png …");
    let chart_config = chart::ChartConfigBuilder::default()
        .caption("MNIST")
        .path("figures/mnist.png")
        .build();
    chart::chart_vector(fitted.embedding().clone(), Some(labels), Some(chart_config));

    // Loss plot is written by the training loop when verbose=true
    // (figures/losses_model.png). We also write a summary.
    println!("  ✓ figures/mnist.png");
    if std::path::Path::new("figures/losses_model.png").exists() {
        println!("  ✓ figures/losses_model.png");
    }
    println!();
    println!("✓  MNIST benchmark done ({fit_secs:.1}s)");
    println!();
}
