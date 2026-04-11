use fast_umap::prelude::*;
use fast_umap::{Umap, FittedUmap};
use cubecl::wgpu::WgpuRuntime;
use std::path::PathBuf;
use std::time::Instant;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate larger test data for more realistic benchmarking
    let num_samples = 1000;
    let num_features = 50;
    
    println!("Generating {} samples with {} features each...", num_samples, num_features);
    
    let mut data = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let mut sample = Vec::with_capacity(num_features);
        for j in 0..num_features {
            // Create some structured data for better UMAP performance
            let base = (i as f64) / (num_samples as f64) * 10.0;
            let feature = base + ((j as f64) / (num_features as f64)) + 
                (rand::random::<f64>() - 0.5) * 0.1; // Small noise
            sample.push(feature);
        }
        data.push(sample);
    }

    println!("Data generation complete.");

    // Configure UMAP with reasonable settings for the dataset size
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![50, 30], // Slightly larger network
        graph: GraphParams {
            n_neighbors: 15,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 50, // Reduced for benchmarking
            ..Default::default()
        },
        ..Default::default()
    };

    println!("Training UMAP model...");
    let train_start = Instant::now();
    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data.clone(), None);
    let train_duration = train_start.elapsed();
    println!("Training completed in {:.2} seconds", train_duration.as_secs_f64());

    // Benchmark saving
    let model_path = PathBuf::from("benchmark_model.bin");
    
    println!("\n=== SAVE BENCHMARK ===");
    let save_start = Instant::now();
    fitted.save(&model_path)?;
    let save_duration = save_start.elapsed();
    
    // Get file size
    let file_metadata = std::fs::metadata(&model_path)?;
    let file_size_mb = file_metadata.len() as f64 / (1024.0 * 1024.0);
    
    println!("Save time: {:.3} seconds", save_duration.as_secs_f64());
    println!("File size: {:.3} MB", file_size_mb);
    println!("Save speed: {:.3} MB/s", file_size_mb / save_duration.as_secs_f64());

    // Benchmark loading
    println!("\n=== LOAD BENCHMARK ===");
    let load_start = Instant::now();
    let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
        &model_path,
        config,
        num_features,
        Default::default(),
    )?;
    let load_duration = load_start.elapsed();
    
    println!("Load time: {:.3} seconds", load_duration.as_secs_f64());
    println!("Load speed: {:.3} MB/s", file_size_mb / load_duration.as_secs_f64());

    // Test the loaded model
    println!("\n=== TRANSFORM BENCHMARK ===");
    let test_data: Vec<Vec<f64>> = (0..100).map(|i| {
        (0..num_features).map(|j| {
            let base = (i as f64) / 100.0 * 10.0;
            base + ((j as f64) / (num_features as f64)) + (rand::random::<f64>() - 0.5) * 0.1
        }).collect()
    }).collect();

    let transform_start = Instant::now();
    let _embedding = loaded_fitted.transform(test_data);
    let transform_duration = transform_start.elapsed();
    
    println!("Transform time (100 samples): {:.3} seconds", transform_duration.as_secs_f64());
    println!("Transform speed: {:.1} samples/second", 100.0 / transform_duration.as_secs_f64());

    // Summary
    println!("\n=== SUMMARY ===");
    println!("Training time: {:.2} seconds", train_duration.as_secs_f64());
    println!("Save time: {:.3} seconds ({:.3} MB at {:.1} MB/s)", 
             save_duration.as_secs_f64(), file_size_mb, file_size_mb / save_duration.as_secs_f64());
    println!("Load time: {:.3} seconds ({:.3} MB at {:.1} MB/s)", 
             load_duration.as_secs_f64(), file_size_mb, file_size_mb / load_duration.as_secs_f64());
    println!("Transform time: {:.3} seconds ({:.1} samples/sec)", 
             transform_duration.as_secs_f64(), 100.0 / transform_duration.as_secs_f64());

    // Clean up
    std::fs::remove_file(&model_path)?;
    println!("\nCleaned up model file.");

    Ok(())
}