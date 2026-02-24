use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;

fn main() {
    type F = f32;
    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, F, i32, u32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    // Set training hyperparameters
    let num_samples = 1000;
    let num_features = 100;

    // Generate random test data
    let train_data: Vec<F> = generate_test_data(num_samples, num_features);
    let data: Vec<Vec<F>> = train_data
        .chunks(num_features)
        .map(|c| c.to_vec())
        .collect();

    // ── New API (mirrors umap-rs) ────────────────────────────────────────────
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![100, 100, 100],
        graph: GraphParams {
            n_neighbors: 10,
            metric: Metric::Euclidean,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 100,
            batch_size: 1000,
            learning_rate: 0.001,
            verbose: true,
            min_desired_loss: Some(0.001),
            timeout: Some(60),
            ..Default::default()
        },
        ..Default::default()
    };

    let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);
    let fitted = umap.fit(data, None);

    // Access the embedding
    let embedding = fitted.embedding();
    println!(
        "Embedding: {} samples × {} dims",
        embedding.len(),
        embedding[0].len()
    );

    println!("Done.");
}
