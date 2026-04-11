use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use fast_umap::{Umap, FittedUmap};
use std::path::PathBuf;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate some test data
    let data: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
        vec![4.0, 5.0, 6.0, 7.0],
        vec![5.0, 6.0, 7.0, 8.0],
    ];

    println!("Training UMAP model...");
    
    // Configure and fit UMAP
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![10], // Small network for quick demo
        graph: GraphParams {
            n_neighbors: 3, // Reduce neighbors for small dataset
            ..Default::default()
        },
        ..Default::default()
    };
    
    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data.clone(), None);
    
    println!("Original embedding:");
    let embedding = fitted.embedding();
    for (i, point) in embedding.iter().enumerate() {
        println!("  Point {}: {:?}", i, point);
    }

    // Save the model
    let model_path = PathBuf::from("umap_model.bin");
    println!("\nSaving model to {:?}...", model_path);
    fitted.save(&model_path)?;
    println!("Model saved successfully!");

    // Load the model
    println!("\nLoading model from {:?}...", model_path);
    let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
        &model_path,
        config,
        4, // Input size (number of features)
        Default::default(),
    )?;
    println!("Model loaded successfully!");

    // Test the loaded model with some new data
    let new_data: Vec<Vec<f64>> = vec![
        vec![1.5, 2.5, 3.5, 4.5],
        vec![2.5, 3.5, 4.5, 5.5],
    ];

    println!("\nTransforming new data with loaded model:");
    let new_embedding = loaded_fitted.transform(new_data);
    for (i, point) in new_embedding.iter().enumerate() {
        println!("  New point {}: {:?}", i, point);
    }

    // Clean up
    std::fs::remove_file(&model_path)?;
    println!("\nCleaned up model file.");

    Ok(())
}