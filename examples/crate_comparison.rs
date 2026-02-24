/// Benchmark comparison between `fast-umap` (GPU-accelerated, parametric) and
/// `umap-rs` (CPU, classical SGD) on identical datasets.
///
/// Usage:
///     cargo run --release --example crate_comparison
///
/// Output:
///     figures/crate_comparison.json   — raw timing data
///     figures/crate_comparison.svg     — bar chart
///     figures/crate_comparison.md      — Markdown summary
///
/// Each crate is timed end-to-end: data prep → fit → extract embedding.
/// The comparison is run across several dataset sizes.
use std::{
    fs,
    io::Write as _,
    path::Path,
    time::Instant,
};

use ndarray_017 as ndarray;

// ── fast-umap imports ────────────────────────────────────────────────────────
use fast_umap::{
    utils::generate_test_data,
    Umap, UmapConfig, GraphParams, OptimizationParams,
};

// ── umap-rs imports ──────────────────────────────────────────────────────────
use umap_rs::{Umap as UmapRs, UmapConfig as UmapRsConfig};

// ── burn backend ─────────────────────────────────────────────────────────────
type GpuBackend =
    burn::backend::wgpu::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<GpuBackend>;

// ─── Configuration ───────────────────────────────────────────────────────────

/// A single benchmark scenario.
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
            label: "500×50".into(),
            n_samples: 500,
            n_features: 50,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "1000×50".into(),
            n_samples: 1_000,
            n_features: 50,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "2000×100".into(),
            n_samples: 2_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "5000×100".into(),
            n_samples: 5_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "10000×100".into(),
            n_samples: 10_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
        },
        Scenario {
            label: "20000×100".into(),
            n_samples: 20_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
        },
    ]
}

// Number of epochs for each crate (keep small for benchmarking)
const FAST_UMAP_EPOCHS: usize = 50;
const UMAP_RS_EPOCHS: usize = 200; // classical UMAP uses more epochs by convention

// ─── Data generation ─────────────────────────────────────────────────────────

/// Generate data for fast-umap (Vec<Vec<f64>>)
fn gen_data_vv(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    let flat: Vec<f32> = generate_test_data(n_samples, n_features);
    flat.chunks(n_features)
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect()
}

/// Generate data as ndarray Array2<f32> for umap-rs
fn gen_data_ndarray(n_samples: usize, n_features: usize) -> ndarray::Array2<f32> {
    let flat: Vec<f32> = generate_test_data(n_samples, n_features);
    ndarray::Array2::from_shape_vec((n_samples, n_features), flat).unwrap()
}

// ─── Brute-force KNN (needed for umap-rs) ────────────────────────────────────

fn compute_knn(
    data: &ndarray::Array2<f32>,
    k: usize,
) -> (ndarray::Array2<u32>, ndarray::Array2<f32>) {
    let n = data.shape()[0];
    let mut knn_indices = ndarray::Array2::<u32>::zeros((n, k));
    let mut knn_dists = ndarray::Array2::<f32>::zeros((n, k));

    for i in 0..n {
        let point = data.row(i);
        let mut distances: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let other = data.row(j);
                let dist: f32 = point
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (j, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (ki, &(j, d)) in distances.iter().take(k).enumerate() {
            knn_indices[(i, ki)] = j as u32;
            knn_dists[(i, ki)] = d;
        }
    }

    (knn_indices, knn_dists)
}

/// Random initialization for umap-rs
fn random_init(n_samples: usize, n_components: usize) -> ndarray::Array2<f32> {
    use rand::Rng;
    let mut rng = rand::rng();
    ndarray::Array2::from_shape_fn((n_samples, n_components), |_| {
        rng.random_range(-10.0f32..10.0)
    })
}

// ─── Timing ──────────────────────────────────────────────────────────────────

#[derive(Clone, serde::Serialize)]
struct TimingResult {
    scenario: String,
    crate_name: String,
    /// Total wall-clock time in seconds (data prep + fit + extract)
    total_secs: f64,
    /// Fit-only wall-clock time in seconds
    fit_secs: f64,
    /// Number of epochs
    epochs: usize,
    n_samples: usize,
    n_features: usize,
}

fn run_fast_umap(sc: &Scenario) -> TimingResult {
    let total_start = Instant::now();

    // Data prep
    let data = gen_data_vv(sc.n_samples, sc.n_features);

    // Fit
    let fit_start = Instant::now();
    let config = UmapConfig {
        n_components: sc.n_components,
        hidden_sizes: vec![128],
        graph: GraphParams {
            n_neighbors: sc.n_neighbors,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: FAST_UMAP_EPOCHS,
            batch_size: sc.n_samples, // one batch
            learning_rate: 1e-3,
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let (exit_tx, exit_rx) = crossbeam_channel::unbounded();
    // Don't set ctrlc handler here — may already be set
    let umap = Umap::<MyAutodiffBackend>::new(config);
    let fitted = umap.fit_with_signal(data, None, exit_rx);
    let fit_secs = fit_start.elapsed().as_secs_f64();

    // Extract embedding
    let _embedding = fitted.into_embedding();
    let total_secs = total_start.elapsed().as_secs_f64();
    drop(exit_tx);

    TimingResult {
        scenario: sc.label.clone(),
        crate_name: "fast-umap".into(),
        total_secs,
        fit_secs,
        epochs: FAST_UMAP_EPOCHS,
        n_samples: sc.n_samples,
        n_features: sc.n_features,
    }
}

fn run_umap_rs(sc: &Scenario) -> TimingResult {
    let total_start = Instant::now();

    // Data prep + KNN (included in total time since fast-umap does it internally)
    let data = gen_data_ndarray(sc.n_samples, sc.n_features);
    let (knn_indices, knn_dists) = compute_knn(&data, sc.n_neighbors);
    let init = random_init(sc.n_samples, sc.n_components);

    // Fit
    let fit_start = Instant::now();
    let config = UmapRsConfig {
        n_components: sc.n_components,
        optimization: umap_rs::OptimizationParams {
            n_epochs: Some(UMAP_RS_EPOCHS),
            ..Default::default()
        },
        ..Default::default()
    };
    let umap = UmapRs::new(config);
    let fitted = umap.fit(
        data.view(),
        knn_indices.view(),
        knn_dists.view(),
        init.view(),
    );
    let fit_secs = fit_start.elapsed().as_secs_f64();

    // Extract embedding
    let _embedding = fitted.into_embedding();
    let total_secs = total_start.elapsed().as_secs_f64();

    TimingResult {
        scenario: sc.label.clone(),
        crate_name: "umap-rs".into(),
        total_secs,
        fit_secs,
        epochs: UMAP_RS_EPOCHS,
        n_samples: sc.n_samples,
        n_features: sc.n_features,
    }
}

// ─── JSON output ─────────────────────────────────────────────────────────────

fn write_json(path: &str, results: &[TimingResult]) {
    let json = serde_json::to_string_pretty(results).unwrap();
    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
    fs::write(path, &json).unwrap();
    println!("  Wrote {path}");
}

// ─── Markdown output ─────────────────────────────────────────────────────────

fn write_md(path: &str, results: &[TimingResult]) {
    let mut md = String::new();
    md.push_str("# fast-umap vs umap-rs — Crate Comparison\n\n");
    md.push_str(&format!(
        "> **Date:** {}  \n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    ));
    md.push_str(&format!(
        "> **fast-umap:** {} epochs (parametric, GPU)  \n",
        FAST_UMAP_EPOCHS
    ));
    md.push_str(&format!(
        "> **umap-rs:** {} epochs (classical, CPU)  \n",
        UMAP_RS_EPOCHS
    ));
    md.push_str("> **Reproduce:** `cargo run --release --example crate_comparison`\n\n");
    md.push_str("![chart](crate_comparison.svg)\n\n");

    md.push_str("## Total Time (data prep + fit + extract)\n\n");
    md.push_str(
        "| Dataset | fast-umap | umap-rs | Speedup |\n\
         |---------|-----------|---------|--------|\n",
    );

    let scenarios = scenarios();
    for sc in &scenarios {
        let fu = results.iter().find(|r| r.scenario == sc.label && r.crate_name == "fast-umap");
        let ur = results.iter().find(|r| r.scenario == sc.label && r.crate_name == "umap-rs");
        if let (Some(fu), Some(ur)) = (fu, ur) {
            let speedup = ur.total_secs / fu.total_secs;
            let icon = if speedup > 1.0 { "🚀" } else { "" };
            md.push_str(&format!(
                "| {} | {:.3}s | {:.3}s | {:.2}× {icon} |\n",
                sc.label, fu.total_secs, ur.total_secs, speedup,
            ));
        }
    }

    md.push_str("\n## Fit Time Only\n\n");
    md.push_str(
        "| Dataset | fast-umap | umap-rs | Speedup |\n\
         |---------|-----------|---------|--------|\n",
    );

    for sc in &scenarios {
        let fu = results.iter().find(|r| r.scenario == sc.label && r.crate_name == "fast-umap");
        let ur = results.iter().find(|r| r.scenario == sc.label && r.crate_name == "umap-rs");
        if let (Some(fu), Some(ur)) = (fu, ur) {
            let speedup = ur.fit_secs / fu.fit_secs;
            let icon = if speedup > 1.0 { "🚀" } else { "" };
            md.push_str(&format!(
                "| {} | {:.3}s | {:.3}s | {:.2}× {icon} |\n",
                sc.label, fu.fit_secs, ur.fit_secs, speedup,
            ));
        }
    }

    md.push_str("\n---\n\n");
    md.push_str("**Notes:**\n");
    md.push_str("- fast-umap is a *parametric* UMAP (neural network, GPU-accelerated via burn/CubeCL)\n");
    md.push_str("- umap-rs is a *classical* UMAP (SGD on embedding, CPU, multithreaded via rayon)\n");
    md.push_str("- fast-umap includes batch-local KNN computation; umap-rs requires precomputed KNN (included in total time)\n");
    md.push_str("- fast-umap can `transform()` new unseen data; umap-rs cannot (yet)\n");

    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
    fs::write(path, &md).unwrap();
    println!("  Wrote {path}");
}

// ─── SVG chart ───────────────────────────────────────────────────────────────

fn write_svg(path: &str, results: &[TimingResult]) {
    let scenarios = scenarios();
    let n = scenarios.len();

    // Layout
    const SVG_W: f64 = 900.0;
    const TITLE_H: f64 = 60.0;
    const LEGEND_H: f64 = 30.0;
    const CHART_LEFT: f64 = 130.0;
    const CHART_RIGHT: f64 = 100.0;
    const CHART_TOP: f64 = 10.0;
    const CHART_BOTTOM: f64 = 50.0;
    const BAR_H: f64 = 16.0;
    const PAIR_GAP: f64 = 3.0;
    const ENTRY_GAP: f64 = 18.0;

    let chart_w = SVG_W - CHART_LEFT - CHART_RIGHT;
    let chart_h = n as f64 * (2.0 * BAR_H + PAIR_GAP) + (n as f64 - 1.0) * ENTRY_GAP;
    let svg_h = TITLE_H + LEGEND_H + CHART_TOP + chart_h + CHART_BOTTOM + 30.0;

    let fu_colour = "#E45756"; // red/coral for fast-umap
    let ur_colour = "#4C78A8"; // blue for umap-rs

    // Collect all total times for scale
    let all_times: Vec<f64> = results.iter().map(|r| r.total_secs).collect();
    let max_time = all_times.iter().cloned().fold(0.0f64, f64::max);

    // Linear scale (times are already in seconds, no log needed for typical ranges)
    let x_of = |v: f64| -> f64 { (v / (max_time * 1.15)) * chart_w };

    let mut out: Vec<u8> = Vec::new();
    let w = &mut out;

    writeln!(w, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{SVG_W}\" height=\"{svg_h:.0}\" font-family=\"system-ui, -apple-system, sans-serif\">").unwrap();
    writeln!(w, "<rect width=\"{SVG_W}\" height=\"{svg_h:.0}\" fill=\"#FAFAFA\"/>").unwrap();

    // Title
    writeln!(
        w,
        "<text x=\"{:.0}\" y=\"30\" text-anchor=\"middle\" font-size=\"16\" font-weight=\"bold\" fill=\"#222\">fast-umap vs umap-rs \u{2014} Performance Comparison</text>",
        SVG_W / 2.0
    ).unwrap();
    writeln!(
        w,
        "<text x=\"{:.0}\" y=\"48\" text-anchor=\"middle\" font-size=\"11\" fill=\"#666\">Total wall-clock time (lower is better)</text>",
        SVG_W / 2.0
    ).unwrap();

    // Legend
    let ly = TITLE_H + 5.0;
    let lx = SVG_W / 2.0 - 130.0;
    writeln!(w, "<rect x=\"{lx:.0}\" y=\"{ly:.0}\" width=\"14\" height=\"12\" rx=\"2\" fill=\"{fu_colour}\"/>").unwrap();
    writeln!(w, "<text x=\"{:.0}\" y=\"{:.0}\" font-size=\"11\" fill=\"#333\">fast-umap (GPU, parametric)</text>", lx + 18.0, ly + 10.0).unwrap();
    writeln!(w, "<rect x=\"{:.0}\" y=\"{ly:.0}\" width=\"14\" height=\"12\" rx=\"2\" fill=\"{ur_colour}\"/>", lx + 200.0).unwrap();
    writeln!(w, "<text x=\"{:.0}\" y=\"{:.0}\" font-size=\"11\" fill=\"#333\">umap-rs (CPU, classical)</text>", lx + 218.0, ly + 10.0).unwrap();

    let cx = CHART_LEFT;
    let cy = TITLE_H + LEGEND_H + CHART_TOP;

    // X-axis grid lines
    let n_ticks = 6;
    for i in 0..=n_ticks {
        let frac = i as f64 / n_ticks as f64;
        let val = frac * max_time * 1.15;
        let tx = cx + frac * chart_w;
        writeln!(
            w,
            "<line x1=\"{tx:.1}\" y1=\"{cy:.1}\" x2=\"{tx:.1}\" y2=\"{:.1}\" stroke=\"#E8E8E8\" stroke-width=\"1\"/>",
            cy + chart_h
        ).unwrap();
        writeln!(
            w,
            "<text x=\"{tx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"10\" fill=\"#999\">{:.1}s</text>",
            cy + chart_h + 16.0,
            val,
        ).unwrap();
    }

    // Bars
    for (i, sc) in scenarios.iter().enumerate() {
        let fu = results
            .iter()
            .find(|r| r.scenario == sc.label && r.crate_name == "fast-umap");
        let ur = results
            .iter()
            .find(|r| r.scenario == sc.label && r.crate_name == "umap-rs");

        let pair_h = 2.0 * BAR_H + PAIR_GAP;
        let by_base = cy + i as f64 * (pair_h + ENTRY_GAP);

        // fast-umap bar
        if let Some(fu) = fu {
            let bw = x_of(fu.total_secs).max(2.0);
            writeln!(
                w,
                "<rect x=\"{cx:.1}\" y=\"{:.1}\" width=\"{bw:.1}\" height=\"{BAR_H:.0}\" rx=\"3\" fill=\"{fu_colour}\" opacity=\"0.85\"/>",
                by_base
            ).unwrap();
            writeln!(
                w,
                "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" fill=\"#555\">{:.3}s</text>",
                cx + bw + 5.0,
                by_base + BAR_H / 2.0 + 3.5,
                fu.total_secs,
            ).unwrap();
        }

        // umap-rs bar
        let ur_by = by_base + BAR_H + PAIR_GAP;
        if let Some(ur) = ur {
            let bw = x_of(ur.total_secs).max(2.0);
            writeln!(
                w,
                "<rect x=\"{cx:.1}\" y=\"{ur_by:.1}\" width=\"{bw:.1}\" height=\"{BAR_H:.0}\" rx=\"3\" fill=\"{ur_colour}\" opacity=\"0.85\"/>",
            ).unwrap();
            writeln!(
                w,
                "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" fill=\"#555\">{:.3}s</text>",
                cx + bw + 5.0,
                ur_by + BAR_H / 2.0 + 3.5,
                ur.total_secs,
            ).unwrap();
        }

        // Speedup label
        if let (Some(fu), Some(ur)) = (fu, ur) {
            let speedup = ur.total_secs / fu.total_secs;
            let (sp_text, sp_fill) = if speedup >= 2.0 {
                (format!("{speedup:.1}× 🚀"), "#c41e3a")
            } else if speedup >= 1.0 {
                (format!("{speedup:.2}×"), "#2a7ec7")
            } else {
                (format!("{:.2}× (umap-rs faster)", speedup), "#666")
            };
            writeln!(
                w,
                "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" font-weight=\"bold\" fill=\"{sp_fill}\">{sp_text}</text>",
                SVG_W - CHART_RIGHT + 8.0,
                by_base + pair_h / 2.0 + 3.5,
            ).unwrap();
        }

        // Y label (dataset size)
        writeln!(
            w,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-size=\"11\" font-weight=\"600\" fill=\"#333\">{}</text>",
            cx - 8.0,
            by_base + pair_h / 2.0 + 4.0,
            sc.label,
        ).unwrap();

        // Sub-label with details
        writeln!(
            w,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-size=\"8\" fill=\"#999\">k={}</text>",
            cx - 8.0,
            by_base + pair_h / 2.0 + 14.0,
            sc.n_neighbors,
        ).unwrap();
    }

    // X-axis baseline
    writeln!(
        w,
        "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#BBB\" stroke-width=\"1\"/>",
        cy + chart_h,
        cx + chart_w,
        cy + chart_h,
    ).unwrap();

    writeln!(w, "</svg>").unwrap();

    let svg = String::from_utf8(out).expect("valid UTF-8");
    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
    fs::write(path, &svg).unwrap();
    println!("  Wrote {path}");
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       fast-umap vs umap-rs — Crate Comparison          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  fast-umap: {} epochs (parametric, GPU)", FAST_UMAP_EPOCHS);
    println!("  umap-rs:   {} epochs (classical, CPU)", UMAP_RS_EPOCHS);
    println!();

    // ── GPU warmup ─────────────────────────────────────────────────────────
    // Run a tiny fast-umap fit to trigger WGPU shader compilation and device
    // init. Without this, the first real scenario pays ~0.3-2s JIT overhead.
    {
        print!("  GPU warmup … ");
        std::io::stdout().flush().unwrap();
        let warmup_start = Instant::now();
        let warmup_sc = Scenario {
            label: "warmup".into(),
            n_samples: 50,
            n_features: 10,
            n_neighbors: 5,
            n_components: 2,
        };
        let _ = run_fast_umap(&warmup_sc);
        println!("done ({:.2}s)", warmup_start.elapsed().as_secs_f64());
        println!();
    }

    let scenarios = scenarios();
    let mut results: Vec<TimingResult> = Vec::new();

    for sc in &scenarios {
        println!(
            "━━━  Scenario: {} ({} samples × {} features)  ━━━",
            sc.label, sc.n_samples, sc.n_features
        );

        // Run umap-rs first (CPU-only, no GPU warmup needed)
        print!("  umap-rs  … ");
        std::io::stdout().flush().unwrap();
        let ur = run_umap_rs(sc);
        println!("total={:.3}s  fit={:.3}s", ur.total_secs, ur.fit_secs);
        results.push(ur);

        // Run fast-umap
        print!("  fast-umap … ");
        std::io::stdout().flush().unwrap();
        let fu = run_fast_umap(sc);
        println!("total={:.3}s  fit={:.3}s", fu.total_secs, fu.fit_secs);
        results.push(fu);

        // Quick speedup
        let ur_total = results
            .iter()
            .rev()
            .find(|r| r.crate_name == "umap-rs" && r.scenario == sc.label)
            .unwrap()
            .total_secs;
        let fu_total = results
            .iter()
            .rev()
            .find(|r| r.crate_name == "fast-umap" && r.scenario == sc.label)
            .unwrap()
            .total_secs;
        let speedup = ur_total / fu_total;
        if speedup >= 1.0 {
            println!("  → fast-umap is {:.2}× faster", speedup);
        } else {
            println!("  → umap-rs is {:.2}× faster", 1.0 / speedup);
        }
        println!();
    }

    // ── Write results ────────────────────────────────────────────────────────
    let results_dir = "figures";
    println!("━━━  Writing results  ━━━");

    write_json(
        &format!("{results_dir}/crate_comparison.json"),
        &results,
    );
    write_md(
        &format!("{results_dir}/crate_comparison.md"),
        &results,
    );
    write_svg(
        &format!("{results_dir}/crate_comparison.svg"),
        &results,
    );

    println!();
    println!("✓  Done. Results in {results_dir}/crate_comparison.*");
    println!();
}
