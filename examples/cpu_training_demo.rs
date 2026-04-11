//! CPU backend demonstration using umap-rs as fallback
//!
//! This example demonstrates the CPU backend capabilities when the "cpu" feature is enabled.
//! The CPU backend uses umap-rs for classical UMAP computation, providing full functionality
//! when GPU is not available.

use fast_umap::prelude::*;
use rand::Rng;

fn main() {
    println!("=== CPU Backend Demo with umap-rs ===");
    
    // Generate test data
    let num_samples = 100;
    let num_features = 5;
    
    let mut rng = rand::rng();
    #[allow(unused_variables)]
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    println!("Generated {} samples with {} features each", num_samples, num_features);
    
    #[cfg(feature = "cpu")]
    {
        println!("\nRunning CPU UMAP with umap-rs backend:");
        
        use fast_umap::cpu_backend::api as cpu_api;
        
        // Create configuration
        let config = UmapConfig {
            n_components: 2,
            graph: GraphParams {
                n_neighbors: 15,
                ..Default::default()
            },
            optimization: OptimizationParams {
                n_epochs: 100,
                verbose: true,
                ..Default::default()
            },
            ..Default::default()
        };
        
        // Fit UMAP using CPU backend
        let fitted = cpu_api::fit_cpu(config, data.clone(), None);
        
        let embedding = fitted.embedding();
        println!("✓ CPU UMAP completed successfully!");
        println!("Embedding shape: {} × {}", embedding.len(), embedding[0].len());
        
        // Show first few embedding points
        println!("\nFirst 5 embedding points:");
        for (i, point) in embedding.iter().take(5).enumerate() {
            println!("  Sample {}: {:?}", i, point);
        }
        
        // Note about CPU backend limitations
        println!("\n=== CPU Backend Characteristics ===");
        println!("✓ Full UMAP functionality using umap-rs");
        println!("✓ No GPU required");
        println!("✓ Same API as GPU backend");
        println!("❌ Cannot transform new data (classical UMAP limitation)");
        println!("  Note: For parametric UMAP with transform support, use GPU backend");
    }
    
    #[cfg(not(feature = "cpu"))]
    {
        println!("CPU backend not available.");
        println!("Please compile with: cargo build --release --features cpu");
    }
}