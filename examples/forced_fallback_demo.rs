//! Forced fallback demonstration
//!
//! This example simulates GPU failure to demonstrate the fallback mechanism.

use rand::Rng;

fn main() {
    println!("=== Forced Fallback Demo ===");
    println!("This demo shows what happens when GPU fails and CPU fallback is used.");
    println!();
    
    // Generate test data
    let num_samples = 50;
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
    
    // Simulate GPU failure by forcing a panic
    println!("Simulating GPU backend failure...");
    let gpu_result = try_gpu_backend_with_forced_failure(data.clone());
    
    match gpu_result {
        Ok(embedding) => {
            println!("✓ GPU backend succeeded (unexpected!):");
            println!("  Embedding shape: {} × {}", embedding.len(), embedding[0].len());
        }
        Err(gpu_error) => {
            println!("⚠ GPU backend failed as expected: {}", gpu_error);
            println!("Falling back to CPU backend...");
            println!();
            
            match try_cpu_backend(data.clone()) {
                Ok(cpu_embedding) => {
                    println!("✓ CPU backend succeeded:");
                    println!("  Embedding shape: {} × {}", cpu_embedding.len(), cpu_embedding[0].len());
                    println!("  Transform support: ❌ No");
                    println!();
                    println!("✅ Fallback mechanism worked perfectly!");
                    println!("   GPU failed → CPU took over → Success!");
                }
                Err(cpu_error) => {
                    println!("❌ CPU backend also failed: {}", cpu_error);
                    println!("This should not happen in normal circumstances.");
                    std::process::exit(1);
                }
            }
        }
    }
    
    println!();
    println!("=== Forced Fallback Demo Complete ===");
}

/// Simulate GPU backend failure
fn try_gpu_backend_with_forced_failure(_data: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    #[cfg(feature = "gpu")]
    {
        // Simulate a GPU failure by forcing a panic
        // In real scenarios, this could be:
        // - GPU driver issues
        // - Out of GPU memory
        // - GPU initialization failure
        // - Unsupported GPU hardware
        
        println!("  GPU initialization... FAILED (simulated)");
        
        // Simulate failure
        Err("Simulated GPU failure: Could not initialize WGPU device".to_string())
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        Err("GPU feature not compiled. Use --features gpu".to_string())
    }
}

/// Try CPU backend (same as fallback_demo)
#[allow(unused_variables)]
fn try_cpu_backend(data: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    #[cfg(feature = "cpu")]
    {
        use fast_umap::{prelude::*, cpu_backend::api as cpu_api};
        use std::panic::catch_unwind;

        println!("  CPU initialization... SUCCESS");
        println!("  CPU processing... COMPLETE");

        let config = UmapConfig::default();
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