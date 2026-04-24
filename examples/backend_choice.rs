/// Runtime backend selection: gpu, mlx, or cpu.
///
/// Usage:
///     cargo run --release --features gpu          --example backend_choice -- gpu
///     cargo run --release --features mlx          --example backend_choice -- mlx
///     cargo run --release --features gpu,cpu      --example backend_choice -- cpu
use fast_umap::prelude::*;
use rand::Rng;

fn main() {
    let num_samples = 100;
    let num_features = 3;

    let mut rng = rand::rng();
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();

    let backend_choice = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            // Auto-detect best available backend
            if cfg!(feature = "mlx") { "mlx" }
            else if cfg!(feature = "gpu") { "gpu" }
            else if cfg!(feature = "cpu") { "cpu" }
            else { "gpu" }
        }.to_string());

    println!("Running UMAP with {backend_choice} backend");

    match backend_choice.as_str() {
        "mlx" => run_mlx_umap(data),
        "gpu" | "wgpu" => run_gpu_umap(data),
        "cpu" => run_cpu_umap(data),
        _ => {
            eprintln!("Unknown backend '{backend_choice}'. Use: mlx, gpu, or cpu");
            std::process::exit(1);
        }
    }
}

fn run_mlx_umap(data: Vec<Vec<f64>>) {
    #[cfg(feature = "mlx")]
    {
        type B = burn::backend::Autodiff<burn_mlx::Mlx>;

        let config = UmapConfig::default();
        let umap = fast_umap::Umap::<B>::new(config);
        let fitted = umap.fit(data.clone(), None);

        let embedding = fitted.embedding();
        println!("MLX Embedding shape: {} x {}", embedding.len(), embedding[0].len());

        let _new = fitted.transform(data);
        println!("MLX backend successfully transformed new data");
    }
    #[cfg(not(feature = "mlx"))]
    {
        eprintln!("MLX backend not available. Compile with --features mlx");
        std::process::exit(1);
    }
}

fn run_gpu_umap(data: Vec<Vec<f64>>) {
    #[cfg(feature = "gpu")]
    {
        type B = burn::backend::Autodiff<
            burn::backend::wgpu::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
        >;

        let config = UmapConfig::default();
        let umap = fast_umap::Umap::<B>::new(config);
        let fitted = umap.fit(data.clone(), None);

        let embedding = fitted.embedding();
        println!("GPU Embedding shape: {} x {}", embedding.len(), embedding[0].len());

        let _new = fitted.transform(data);
        println!("GPU backend successfully transformed new data");
    }
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("GPU backend not available. Compile with --features gpu");
        std::process::exit(1);
    }
}

fn run_cpu_umap(data: Vec<Vec<f64>>) {
    #[cfg(feature = "cpu")]
    {
        use fast_umap::cpu_backend::api as cpu_api;

        let config = UmapConfig::default();
        let fitted = cpu_api::fit_cpu(config, data, None);

        let embedding = fitted.embedding();
        println!("CPU Embedding shape: {} x {}", embedding.len(), embedding[0].len());
        println!("Note: CPU backend does not support transforming new data");
    }
    #[cfg(not(feature = "cpu"))]
    {
        eprintln!("CPU backend not available. Compile with --features cpu");
        std::process::exit(1);
    }
}