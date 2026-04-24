/// Benchmark comparison between WGPU (CubeCL) and MLX backends on Apple Silicon.
///
/// Usage:
///     cargo run --release --features gpu,mlx --example backend_benchmark
///
/// Output:
///     figures/backend_benchmark.md — Markdown summary table
///
/// Both backends run the same fast-umap configuration on identical datasets.
/// Timing covers end-to-end fit (including KNN precomputation and training).
use std::{
    io::Write as _,
    time::Instant,
};

use fast_umap::{
    utils::generate_test_data,
    Umap, UmapConfig, GraphParams, OptimizationParams,
};

// ── Backend type aliases ────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
type WgpuBackend =
    burn::backend::wgpu::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
#[cfg(feature = "gpu")]
type WgpuAutodiff = burn::backend::Autodiff<WgpuBackend>;

#[cfg(feature = "mlx")]
type MlxAutodiff = burn::backend::Autodiff<burn_mlx::Mlx>;

// ── Configuration ───────────────────────────────────────────────────────────

#[derive(Clone)]
struct Scenario {
    label: String,
    n_samples: usize,
    n_features: usize,
    n_neighbors: usize,
    n_components: usize,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            label: "1000x50".into(),
            n_samples: 1_000,
            n_features: 50,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "5000x100".into(),
            n_samples: 5_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "10000x100".into(),
            n_samples: 10_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "20000x100".into(),
            n_samples: 20_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
        },
    ]
}

const N_EPOCHS: usize = 50;

// ── Data generation ─────────────────────────────────────────────────────────

fn gen_data(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    let flat: Vec<f32> = generate_test_data(n_samples, n_features);
    flat.chunks(n_features)
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect()
}

fn make_config(sc: &Scenario) -> UmapConfig {
    UmapConfig {
        n_components: sc.n_components,
        hidden_sizes: vec![128],
        graph: GraphParams {
            n_neighbors: sc.n_neighbors,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: N_EPOCHS,
            batch_size: sc.n_samples,
            learning_rate: 1e-3,
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

// ── Timing results ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct TimingResult {
    scenario: String,
    backend: String,
    total_secs: f64,
    fit_secs: f64,
}

// ── WGPU runner ─────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn run_wgpu(sc: &Scenario) -> TimingResult {
    let total_start = Instant::now();
    let data = gen_data(sc.n_samples, sc.n_features);

    let fit_start = Instant::now();
    let config = make_config(sc);
    let (_, exit_rx) = crossbeam_channel::unbounded();
    let umap = Umap::<WgpuAutodiff>::new(config);
    let fitted = umap.fit_with_signal(data, None, exit_rx);
    let fit_secs = fit_start.elapsed().as_secs_f64();

    let _embedding = fitted.into_embedding();
    let total_secs = total_start.elapsed().as_secs_f64();

    TimingResult {
        scenario: sc.label.clone(),
        backend: "WGPU".into(),
        total_secs,
        fit_secs,
    }
}

// ── MLX runner ──────────────────────────────────────────────────────────────

#[cfg(feature = "mlx")]
fn run_mlx(sc: &Scenario) -> TimingResult {
    let total_start = Instant::now();
    let data = gen_data(sc.n_samples, sc.n_features);

    let fit_start = Instant::now();
    let config = make_config(sc);
    let (_, exit_rx) = crossbeam_channel::unbounded();
    let umap = Umap::<MlxAutodiff>::new(config);
    let fitted = umap.fit_with_signal(data, None, exit_rx);
    let fit_secs = fit_start.elapsed().as_secs_f64();

    let _embedding = fitted.into_embedding();
    let total_secs = total_start.elapsed().as_secs_f64();

    TimingResult {
        scenario: sc.label.clone(),
        backend: "MLX".into(),
        total_secs,
        fit_secs,
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("===========================================================");
    println!("       WGPU vs MLX Backend Benchmark");
    println!("===========================================================");
    println!();
    println!("  Epochs: {}", N_EPOCHS);
    println!();

    let scenarios = scenarios();
    let mut results: Vec<TimingResult> = Vec::new();

    // ── Warmup ──────────────────────────────────────────────────────────────
    let warmup_sc = Scenario {
        label: "warmup".into(),
        n_samples: 100,
        n_features: 10,
        n_neighbors: 5,
        n_components: 2,
    };

    #[cfg(feature = "gpu")]
    {
        print!("  WGPU warmup ... ");
        std::io::stdout().flush().unwrap();
        let t = Instant::now();
        let _ = run_wgpu(&warmup_sc);
        println!("done ({:.2}s)", t.elapsed().as_secs_f64());
    }

    #[cfg(feature = "mlx")]
    {
        print!("  MLX warmup  ... ");
        std::io::stdout().flush().unwrap();
        let t = Instant::now();
        let _ = run_mlx(&warmup_sc);
        println!("done ({:.2}s)", t.elapsed().as_secs_f64());
    }

    println!();

    // ── Run scenarios ───────────────────────────────────────────────────────
    for sc in &scenarios {
        println!(
            "---  {} ({} samples x {} features)  ---",
            sc.label, sc.n_samples, sc.n_features
        );

        #[cfg(feature = "gpu")]
        {
            print!("  WGPU ... ");
            std::io::stdout().flush().unwrap();
            let r = run_wgpu(sc);
            println!("total={:.3}s  fit={:.3}s", r.total_secs, r.fit_secs);
            results.push(r);
        }

        #[cfg(feature = "mlx")]
        {
            print!("  MLX  ... ");
            std::io::stdout().flush().unwrap();
            let r = run_mlx(sc);
            println!("total={:.3}s  fit={:.3}s", r.total_secs, r.fit_secs);
            results.push(r);
        }

        // Print speedup
        let wgpu_total = results.iter().rev()
            .find(|r| r.backend == "WGPU" && r.scenario == sc.label)
            .map(|r| r.total_secs);
        let mlx_total = results.iter().rev()
            .find(|r| r.backend == "MLX" && r.scenario == sc.label)
            .map(|r| r.total_secs);

        if let (Some(w), Some(m)) = (wgpu_total, mlx_total) {
            let speedup = w / m;
            if speedup >= 1.0 {
                println!("  -> MLX is {:.2}x faster", speedup);
            } else {
                println!("  -> WGPU is {:.2}x faster", 1.0 / speedup);
            }
        }
        println!();
    }

    // ── Write markdown results ──────────────────────────────────────────────
    println!("---  Results  ---");
    println!();

    let mut md = String::new();
    md.push_str("# WGPU vs MLX Backend Benchmark\n\n");
    md.push_str(&format!("Epochs: {}\n\n", N_EPOCHS));

    md.push_str("| Dataset | WGPU Total | MLX Total | WGPU Fit | MLX Fit | Speedup (total) |\n");
    md.push_str("|---------|------------|-----------|----------|---------|----------------|\n");

    for sc in &scenarios {
        let wgpu = results.iter().find(|r| r.backend == "WGPU" && r.scenario == sc.label);
        let mlx = results.iter().find(|r| r.backend == "MLX" && r.scenario == sc.label);

        let wt = wgpu.map(|r| format!("{:.3}s", r.total_secs)).unwrap_or_else(|| "N/A".into());
        let mt = mlx.map(|r| format!("{:.3}s", r.total_secs)).unwrap_or_else(|| "N/A".into());
        let wf = wgpu.map(|r| format!("{:.3}s", r.fit_secs)).unwrap_or_else(|| "N/A".into());
        let mf = mlx.map(|r| format!("{:.3}s", r.fit_secs)).unwrap_or_else(|| "N/A".into());

        let speedup = match (wgpu, mlx) {
            (Some(w), Some(m)) => {
                let s = w.total_secs / m.total_secs;
                if s >= 1.0 {
                    format!("MLX {:.2}x faster", s)
                } else {
                    format!("WGPU {:.2}x faster", 1.0 / s)
                }
            }
            _ => "N/A".into(),
        };

        md.push_str(&format!("| {} | {} | {} | {} | {} | {} |\n",
            sc.label, wt, mt, wf, mf, speedup));
    }

    // Print to console
    print!("{}", md);

    // Write to file
    std::fs::create_dir_all("figures").unwrap();
    std::fs::write("figures/backend_benchmark.md", &md).unwrap();
    println!("\nWrote figures/backend_benchmark.md");
}
