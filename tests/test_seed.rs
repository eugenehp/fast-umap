#![cfg(feature = "gpu")]
//! Reproducibility: two fits with the same data, configuration and seed give
//! the same embedding; a different seed gives a different one.
//!
//! Both halves matter. Without the second, a seed that never reaches the
//! generators would still pass the first.

use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use fast_umap::Umap;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

/// Three well separated blobs, built without any random generator so that the
/// data itself cannot be a source of variation between runs.
fn blobs() -> Vec<Vec<f64>> {
    let (per_blob, dims) = (40usize, 8usize);
    (0..3 * per_blob)
        .map(|i| {
            let blob = (i / per_blob) as f64;
            let t = (i % per_blob) as f64;
            (0..dims)
                .map(|j| 10.0 * blob * ((j % 3) as f64 + 1.0) + ((t * 7.0 + j as f64 * 3.0) % 5.0) * 0.1)
                .collect()
        })
        .collect()
}

fn embedding(seed: u64) -> Vec<Vec<f64>> {
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![16],
        graph: GraphParams {
            n_neighbors: 8,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 40,
            verbose: false,
            ..Default::default()
        },
        seed,
        ..Default::default()
    };
    Umap::<MyAutodiffBackend>::new(config).fit(blobs(), None).into_embedding()
}

fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f64::max)
}

fn scale(a: &[Vec<f64>]) -> f64 {
    let all = a.iter().flatten();
    let max = all.clone().fold(f64::MIN, |m, &v| m.max(v));
    let min = all.fold(f64::MAX, |m, &v| m.min(v));
    max - min
}

/// One test, sequential on purpose: `Backend::seed` seeds the device's
/// generator, which every fit in the process shares. Two fits running at the
/// same time on different threads reseed each other's generator mid-training,
/// and the result looks like "the seed does nothing".
#[test]
fn same_seed_same_embedding_and_different_seed_different_embedding() {
    let a = embedding(7);
    let b = embedding(7);
    let c = embedding(8);
    let s = scale(&a);

    let d_same = max_abs_diff(&a, &b);
    assert_eq!(d_same, 0.0, "two fits with seed 7 differ by up to {d_same} (embedding scale {s})");

    let d_other = max_abs_diff(&a, &c);
    assert!(
        d_other > 0.01 * s,
        "seeds 7 and 8 give embeddings within {d_other} of each other (scale {s}): the seed is not reaching the generators"
    );
}
