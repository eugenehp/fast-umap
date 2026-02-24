use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use rand::Rng;

fn main() {
    // Number of samples in the dataset
    let num_samples = 100;

    // Number of features (dimensions) for each sample
    let num_features = 3;

    // Create a random number generator for generating random values
    let mut rng = rand::rng();

    // Generate a dataset of random values with `num_samples` rows and `num_features` columns
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();

    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    // ── New API (mirrors umap-rs) ────────────────────────────────────────────
    let config = UmapConfig::default(); // 2-D output, Euclidean, default hyperparameters
    let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);
    let fitted = umap.fit(data.clone(), None);

    // Get the embedding
    let embedding = fitted.embedding();
    println!("Embedding shape: {} × {}", embedding.len(), embedding[0].len());

    // Transform new data through the trained model
    let _new_embedding = fitted.transform(data.clone());
    println!("Transform done.");
}
