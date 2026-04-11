//! Feature-based compilation demonstration
//!
//! This example shows how to compile fast-umap with different feature combinations
//! to optimize for your specific needs and reduce compile times.

use rand::Rng;

fn main() {
    println!("=== Fast-UMAP Feature Demo ===");
    println!();
    println!("Available feature combinations:");
    println!();
    
    println!("1. Minimal GPU (default):");
    println!("   cargo build --release");
    println!("   Features: gpu, verbose");
    println!("   Best for: Production use with GPU");
    println!();
    
    println!("2. CPU-only:");
    println!("   cargo build --release --features cpu");
    println!("   Features: cpu, verbose");
    println!("   Best for: CPU-only environments");
    println!();
    
    println!("3. Minimal (no backend):");
    println!("   cargo build --release --no-default-features");
    println!("   Features: (none)");
    println!("   Best for: Library usage, custom backends");
    println!();
    
    println!("4. All features:");
    println!("   cargo build --release --features all");
    println!("   Features: gpu, cpu, verbose, plotters");
    println!("   Best for: Development and testing");
    println!();
    
    println!("5. Custom combination:");
    println!("   cargo build --release --features \"gpu,plotters\"");
    println!("   Features: gpu, plotters, verbose");
    println!("   Best for: GPU with visualization");
    println!();
    
    // Generate some test data
    let num_samples = 50;
    let num_features = 3;
    let mut rng = rand::rng();
    let _data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    println!("Generated test data: {} samples × {} features", num_samples, num_features);
    println!();
    
    // Test available backends
    #[cfg(feature = "gpu")]
    {
        println!("✓ GPU backend available");
        println!("  - Parametric UMAP with neural networks");
        println!("  - GPU acceleration via WGPU");
        println!("  - Transform new data support");
    }
    
    #[cfg(feature = "cpu")]
    {
        println!("✓ CPU backend available");
        println!("  - Classical UMAP algorithm");
        println!("  - No GPU required");
        println!("  - Full configuration support");
    }
    
    #[cfg(all(not(feature = "gpu"), not(feature = "cpu")))]
    {
        println!("⚠ No backend enabled");
        println!("  - Library mode only");
        println!("  - Use --features gpu or --features cpu");
    }
    
    #[cfg(feature = "verbose")]
    println!("✓ Verbose output enabled");
    
    #[cfg(feature = "plotters")]
    println!("✓ Plotting support enabled");
    
    println!();
    println!("=== Feature Demo Complete ===");
    println!();
    println!("Tip: Choose features based on your target environment:");
    println!("  - Cloud GPU instances: --features gpu");
    println!("  - CPU-only servers: --features cpu");
    println!("  - Edge devices: --features cpu --no-default-features");
    println!("  - Development: --features all");
}