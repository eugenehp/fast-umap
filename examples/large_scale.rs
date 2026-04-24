/// Test NN-Descent on a large dataset (50K samples).
/// This would OOM with the exact pairwise distance approach (10GB matrix).
///
/// Usage:
///     cargo run --release --features mlx --example large_scale
use std::time::Instant;

use fast_umap::{
    utils::generate_test_data,
    Umap, UmapConfig, GraphParams, OptimizationParams,
};

#[cfg(feature = "mlx")]
type Backend = burn::backend::Autodiff<burn_mlx::Mlx>;
#[cfg(all(feature = "gpu", not(feature = "mlx")))]
type Backend = burn::backend::Autodiff<
    burn::backend::wgpu::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
>;

fn main() {
    let n = 50_000;
    let d = 100;
    let epochs = 50;

    println!("=== Large-Scale UMAP: {n} samples x {d} features ===");
    println!("  (exact KNN would need {:.1} GB for pairwise matrix)",
        (n as f64 * n as f64 * 4.0) / 1e9);
    println!();

    let flat: Vec<f32> = generate_test_data(n, d);
    let data: Vec<Vec<f64>> = flat.chunks(d)
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect();

    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        graph: GraphParams {
            n_neighbors: 15,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: epochs,
            batch_size: n,
            learning_rate: 1e-3,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let start = Instant::now();
    let (_, exit_rx) = crossbeam_channel::unbounded();
    let umap = Umap::<Backend>::new(config);
    let fitted = umap.fit_with_signal(data, None, exit_rx);
    let elapsed = start.elapsed().as_secs_f64();

    let emb = fitted.embedding();
    println!("\nDone: {elapsed:.2}s, embedding {n} -> {}x{}", emb.len(), emb[0].len());
}
