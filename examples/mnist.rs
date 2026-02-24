use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use mnist::*;

fn main() {
    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    let num_samples = 10_000_usize;
    let num_features = 28 * 28; // 784

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

    // ── New API (mirrors umap-rs) ────────────────────────────────────────────
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

    let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);
    let fitted = umap.fit(data, Some(labels.clone()));

    println!(
        "Embedding: {} samples × {} dims",
        fitted.embedding().len(),
        fitted.embedding()[0].len()
    );
    println!("Done. Run `./bench.sh` to generate figures.");
}
