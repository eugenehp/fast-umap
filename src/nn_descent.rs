//! NN-Descent approximate k-NN graph construction.
//!
//! Builds a k-nearest-neighbor graph directly from raw data in O(n * k * iters)
//! time and O(n * k) memory, avoiding the O(n^2) pairwise distance matrix.
//! This enables UMAP on datasets with 50K+ samples where the full distance
//! matrix would exceed available memory.
//!
//! Reference: Dong, W., Moses, C., & Li, K. (2011).
//! "Efficient k-nearest neighbor graph construction for generic similarity measures."

use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

/// Build an approximate k-NN graph using NN-Descent.
///
/// # Arguments
/// * `data`  — flat row-major `[n * d]` input data
/// * `n`     — number of samples
/// * `d`     — number of features per sample
/// * `k`     — number of nearest neighbors
///
/// # Returns
/// `(indices, distances)` each flat row-major of length `n * k`.
/// Same format as [`knn_from_pairwise_cpu`](crate::train::get_distance_by_metric::knn_from_pairwise_cpu).
pub fn nn_descent(data: &[f32], n: usize, d: usize, k: usize) -> (Vec<i32>, Vec<f32>) {
    assert!(k < n, "k ({k}) must be < n ({n})");
    let max_iters = 12;
    let sample_rate = 0.5f32; // fraction of neighbors to sample per iteration
    let min_updates_frac = 0.001; // converge when < 0.1% of entries change

    // ── Initialize with random neighbors ────────────────────────────────────
    let mut graph = vec![vec![(f32::INFINITY, 0u32); k]; n];
    {
        let mut rng = rand::rng();
        for i in 0..n {
            let mut candidates: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            candidates.shuffle(&mut rng);
            for slot in 0..k {
                let j = candidates[slot];
                let dist = euclidean_dist(data, i, j, d);
                graph[i][slot] = (dist, j as u32);
            }
            graph[i].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
    }

    // ── Iterate: local join ─────────────────────────────────────────────────
    for _iter in 0..max_iters {
        let n_sample = ((k as f32 * sample_rate) as usize).max(1);

        // Build reverse neighbor lists (who has me as a neighbor?)
        let mut reverse: Vec<Vec<u32>> = vec![vec![]; n];
        for i in 0..n {
            for &(_d, j) in &graph[i] {
                if (j as usize) < n {
                    reverse[j as usize].push(i as u32);
                }
            }
        }

        // Sample candidates: for each point, gather a subset of its
        // neighbors and reverse neighbors
        let candidates: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                let mut cands: Vec<u32> = Vec::new();
                // Forward neighbors (sample)
                for s in 0..n_sample.min(graph[i].len()) {
                    cands.push(graph[i][s].1);
                }
                // Reverse neighbors (sample)
                let rev = &reverse[i];
                let take = n_sample.min(rev.len());
                for s in 0..take {
                    cands.push(rev[s]);
                }
                cands.sort_unstable();
                cands.dedup();
                cands
            })
            .collect();

        // Local join: compare each point's candidates against each other
        let updates: usize = (0..n)
            .into_par_iter()
            .map(|i| {
                let cands = &candidates[i];
                let mut local_updates = 0usize;
                for ci in 0..cands.len() {
                    let u = cands[ci] as usize;
                    if u == i {
                        continue;
                    }
                    // Try u as neighbor of i
                    let dist = euclidean_dist(data, i, u, d);
                    // SAFETY: we only read graph[i], parallel reads are safe
                    // Updates happen via atomic-like check below
                    let worst = unsafe { &*(& graph[i] as *const Vec<(f32, u32)>) };
                    if dist < worst[k - 1].0 {
                        local_updates += 1;
                    }
                }
                local_updates
            })
            .sum();

        // Now do actual updates (sequential to avoid races)
        let mut total_updates = 0usize;
        for i in 0..n {
            let cands = &candidates[i];
            for ci in 0..cands.len() {
                let u = cands[ci] as usize;
                if u == i {
                    continue;
                }
                let dist = euclidean_dist(data, i, u, d);
                if try_insert(&mut graph[i], dist, u as u32, k) {
                    total_updates += 1;
                }
                // Symmetric: also try i as neighbor of u
                if try_insert(&mut graph[u], euclidean_dist(data, u, i, d), i as u32, k) {
                    total_updates += 1;
                }

                // Also compare candidates against each other
                for cj in (ci + 1)..cands.len() {
                    let v = cands[cj] as usize;
                    if v == u {
                        continue;
                    }
                    let d_uv = euclidean_dist(data, u, v, d);
                    try_insert(&mut graph[u], d_uv, v as u32, k);
                    try_insert(&mut graph[v], d_uv, u as u32, k);
                }
            }
        }

        let frac = total_updates as f64 / (n * k) as f64;
        if frac < min_updates_frac as f64 {
            break;
        }
    }

    // ── Flatten to output format ────────────────────────────────────────────
    let mut out_idx = vec![0i32; n * k];
    let mut out_dist = vec![0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            out_idx[i * k + j] = graph[i][j].1 as i32;
            out_dist[i * k + j] = graph[i][j].0;
        }
    }

    (out_idx, out_dist)
}

/// Try to insert (dist, idx) into a sorted neighbor list of capacity k.
/// Returns true if the entry was inserted (i.e., it was better than the worst).
#[inline]
fn try_insert(neighbors: &mut Vec<(f32, u32)>, dist: f32, idx: u32, k: usize) -> bool {
    // Check if idx is already in the list
    if neighbors.iter().any(|&(_, j)| j == idx) {
        return false;
    }
    // Check if this is better than the worst neighbor
    if dist >= neighbors[k - 1].0 {
        return false;
    }
    // Insert in sorted order
    let pos = neighbors
        .binary_search_by(|probe| probe.0.partial_cmp(&dist).unwrap())
        .unwrap_or_else(|e| e);
    neighbors.insert(pos, (dist, idx));
    neighbors.truncate(k);
    true
}

/// Euclidean distance between rows i and j of a flat [n, d] data array.
#[inline]
fn euclidean_dist(data: &[f32], i: usize, j: usize, d: usize) -> f32 {
    let mut sum = 0.0f32;
    let row_i = &data[i * d..(i + 1) * d];
    let row_j = &data[j * d..(j + 1) * d];
    for f in 0..d {
        let diff = row_i[f] - row_j[f];
        sum += diff * diff;
    }
    sum.sqrt()
}
