//! GPU to CPU fallback demonstration
//!
//! This example shows how to implement a robust fallback mechanism
//! that automatically uses CPU backend when GPU is unavailable.

use rand::Rng;

fn main() {
    println!("=== GPU to CPU Fallback Demo ===");
    
    // Generate test data
    let num_samples = 100;
    let num_features = 3;
    let mut rng = rand::rng();
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    println!("Generated {} samples with {} features each", num_samples, num_features);
    println!();
    
    // Try GPU first, fall back to CPU if needed
    match try_gpu_backend(data.clone()) {
        Ok(gpu_result) => {
            println!("✓ GPU backend succeeded:");
            println!("  Embedding shape: {} × {}", gpu_result.len(), gpu_result[0].len());
            println!("  Transform support: ✅ Yes");
        }
        Err(gpu_error) => {
            println!("⚠ GPU backend failed: {}", gpu_error);
            println!("Falling back to CPU backend...");
            
            match try_cpu_backend(data.clone()) {
                Ok(cpu_result) => {
                    println!("✓ CPU backend succeeded:");
                    println!("  Embedding shape: {} × {}", cpu_result.len(), cpu_result[0].len());
                    println!("  Transform support: ❌ No (classical UMAP limitation)");
                }
                Err(cpu_error) => {
                    println!("❌ CPU backend also failed: {}", cpu_error);
                    println!("No working backend available!");
                    std::process::exit(1);
                }
            }
        }
    }
    
    println!();
    println!("=== Fallback Demo Complete ===");
    println!("This demonstrates robust error handling and automatic fallback.");
}

/// Try GPU backend with proper error handling
fn try_gpu_backend(data: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    #[cfg(feature = "gpu")]
    {
        use burn::backend::Autodiff;
        use cubecl::wgpu::WgpuRuntime;
        use fast_umap::prelude::*;
        
        println!("Attempting GPU backend...");
        
        type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;
        
        let config = UmapConfig::default();
        let umap = fast_umap::Umap::<MyAutodiffBackend>::new(config);
        
        // GPU fit can panic on errors, so we wrap in catch_unwind
        use std::panic::catch_unwind;
        
        match catch_unwind(|| umap.fit(data.clone(), None)) {
            Ok(fitted) => {
                let embedding = fitted.embedding().clone();
                // Test transform to ensure GPU is working properly
                let _transform_result = fitted.transform(data);
                Ok(embedding)
            }
            Err(e) => {
                if let Some(s) = e.downcast_ref::<&str>() {
                    Err(format!("GPU fitting failed: {}", s))
                } else if let Some(s) = e.downcast_ref::<String>() {
                    Err(format!("GPU fitting failed: {}", s))
                } else {
                    Err("GPU fitting failed with unknown error".to_string())
                }
            }
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        Err("GPU feature not compiled. Use --features gpu".to_string())
    }
}

/// Try CPU backend with proper error handling
fn try_cpu_backend(_data: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    #[cfg(feature = "cpu")]
    {
        use fast_umap::{prelude::*, cpu_backend::api as cpu_api};
        
        println!("Attempting CPU backend...");
        
        let config = UmapConfig::default();
        // CPU fit can also panic, so we wrap in catch_unwind
        use std::panic::catch_unwind;
        
        match catch_unwind(|| cpu_api::fit_cpu(config, data, None)) {
            Ok(fitted) => Ok(fitted.embedding().clone()),
            Err(e) => {
                if let Some(s) = e.downcast_ref::<&str>() {
                    Err(format!("CPU fitting failed: {}", s))
                } else if let Some(s) = e.downcast_ref::<String>() {
                    Err(format!("CPU fitting failed: {}", s))
                } else {
                    Err("CPU fitting failed with unknown error".to_string())
                }
            }
        }
    }
    
    #[cfg(not(feature = "cpu"))]
    {
        Err("CPU feature not compiled. Use --features cpu".to_string())
    }
}