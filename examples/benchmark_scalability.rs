use fast_umap::prelude::*;
use fast_umap::{Umap, FittedUmap};
use cubecl::wgpu::WgpuRuntime;
use std::path::PathBuf;
use std::time::Instant;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

fn generate_data(num_samples: usize, num_features: usize) -> Vec<Vec<f64>> {
    (0..num_samples).map(|i| {
        (0..num_features).map(|j| {
            let base = (i as f64) / (num_samples as f64) * 10.0;
            base + ((j as f64) / (num_features as f64)) + (rand::random::<f64>() - 0.5) * 0.1
        }).collect()
    }).collect()
}

fn benchmark_model(
    name: &str,
    num_samples: usize,
    num_features: usize,
    hidden_sizes: Vec<usize>,
    n_neighbors: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {} ===", name);
    
    let data = generate_data(num_samples, num_features);
    
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: hidden_sizes.clone(),
        graph: GraphParams {
            n_neighbors,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 20,
            ..Default::default()
        },
        ..Default::default()
    };

    // Train
    let train_start = Instant::now();
    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data, None);
    let train_duration = train_start.elapsed();
    
    // Calculate model size (approximate)
    let mut param_count = 0;
    let mut input_size = num_features;
    for &hidden_size in &hidden_sizes {
        param_count += input_size * hidden_size; // weights
        param_count += hidden_size; // biases
        input_size = hidden_size;
    }
    param_count += input_size * 2; // final layer weights + biases
    
    // Save
    let model_path = PathBuf::from(format!("scalability_{}.bin", name.to_lowercase().replace(' ', "_")));
    let save_start = Instant::now();
    fitted.save(&model_path)?;
    let save_duration = save_start.elapsed();
    
    // Get file size
    let file_metadata = std::fs::metadata(&model_path)?;
    let file_size_kb = file_metadata.len() as f64 / 1024.0;
    
    // Load
    let load_start = Instant::now();
    let _loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
        &model_path,
        config,
        num_features,
        Default::default(),
    )?;
    let load_duration = load_start.elapsed();

    // Clean up
    std::fs::remove_file(&model_path)?;

    println!("  Samples: {}, Features: {}, Hidden: {:?}", num_samples, num_features, hidden_sizes);
    println!("  Parameters: ~{}, File size: {:.1} KB", param_count, file_size_kb);
    println!("  Training: {:.2}s, Save: {:.3}s, Load: {:.3}s", 
             train_duration.as_secs_f64(), save_duration.as_secs_f64(), load_duration.as_secs_f64());
    println!("  Save speed: {:.1} KB/s, Load speed: {:.1} KB/s",
             file_size_kb / save_duration.as_secs_f64(),
             file_size_kb / load_duration.as_secs_f64());

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("UMAP Model Serialization Scalability Benchmark");
    println!("==============================================");

    // Small model
    benchmark_model("Small Model", 500, 20, vec![10], 10)?;
    
    // Medium model
    benchmark_model("Medium Model", 1000, 50, vec![50, 30], 15)?;
    
    // Large model
    benchmark_model("Large Model", 2000, 100, vec![100, 50, 20], 20)?;
    
    // Wide model (many features)
    benchmark_model("Wide Model", 500, 200, vec![100, 50], 15)?;
    
    // Deep model (many layers)
    benchmark_model("Deep Model", 1000, 50, vec![80, 60, 40, 20], 15)?;

    println!("\nBenchmark complete!");
    Ok(())
}