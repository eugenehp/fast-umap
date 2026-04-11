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

    // Choose backend: "cpu" or "gpu"
    let backend_choice = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gpu".to_string());
    
    println!("Running UMAP with {} backend", backend_choice);
    
    match backend_choice.as_str() {
        "cpu" => {
            #[cfg(feature = "cpu")]
            {
                // Use CPU backend with umap-rs fallback
                run_cpu_umap(data);
            }
            #[cfg(not(feature = "cpu"))]
            {
                eprintln!("CPU backend not available. Compile with --features cpu");
                std::process::exit(1);
            }
        }
        "gpu" => {
            // Use GPU backend (parametric UMAP with WGPU)
            run_gpu_umap(data);
        }
        _ => {
            eprintln!("Unknown backend. Please use 'cpu' or 'gpu'");
            std::process::exit(1);
        }
    }
}

fn run_gpu_umap(data: Vec<Vec<f64>>) {
    #[cfg(feature = "gpu")]
    {
        use burn::backend::Autodiff;
        
        // GPU backend uses parametric UMAP (neural network) with full autodiff support
        type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;
        
        let config = UmapConfig::default();
        let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);
        let fitted = umap.fit(data.clone(), None);
        
        let embedding = fitted.embedding();
        println!("GPU Embedding shape: {} × {}", embedding.len(), embedding[0].len());
        
        // GPU backend can transform new data through the trained neural network
        let _new_embedding = fitted.transform(data);
        println!("GPU backend successfully transformed new data");
    }
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("GPU backend not available. Compile with default features or --features gpu");
        std::process::exit(1);
    }
}

#[cfg(feature = "cpu")]
fn run_cpu_umap(data: Vec<Vec<f64>>) {
    use fast_umap::cpu_backend::api as cpu_api;
    
    // CPU backend uses classical UMAP (umap-rs) - no neural network training
    let config = UmapConfig::default();
    let fitted = cpu_api::fit_cpu(config, data.clone(), None);
    
    let embedding = fitted.embedding();
    println!("CPU Embedding shape: {} × {}", embedding.len(), embedding[0].len());
    
    // Note: CPU backend cannot transform new data (classical UMAP limitation)
    println!("Note: CPU backend does not support transforming new data");
}