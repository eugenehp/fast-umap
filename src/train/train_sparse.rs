//! Sparse-loss UMAP training — O(n·k) per epoch instead of O(n²).
//!
//! Performance techniques:
//!
//! 1. **Global KNN computed once** — brute-force on GPU, CPU sort, stored as edges.
//! 2. **Edge subsampling** — cap edges per epoch so cost is O(E_max), not O(n·k).
//! 3. **Negative sampling** — `neg_rate` random non-neighbor pairs per sampled edge.
//! 4. **In-memory checkpointing** — best model weights kept in RAM, no disk I/O.
//! 5. **Pre-batched edge samples** — uploaded to GPU once, cycled per epoch.
//! 6. **Fused gather** — positive + negative indices merged into 2 selects (not 4).
//! 7. **Async loss readback** — GPU→CPU sync only every N epochs for progress display.
//! 8. **Lazy loss plotting** — charts rendered only every N epochs, not every epoch.

use crate::{
    backend::AutodiffBackend,
    chart::{self, plot_loss, ChartConfigBuilder},
    format_duration,
    model::UMAPModel,
    normalize_data,
    utils::convert_vector_to_tensor,
};
use burn::{
    module::{AutodiffModule, Module},
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    tensor::{cast::ToElement, Device, Int, Tensor, TensorData},
};

use super::get_distance_by_metric::*;
use super::config::*;

use crossbeam_channel::Receiver;
use indicatif::{ProgressBar, ProgressStyle};
use num::{Float, FromPrimitive};
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Duration;
use std::{thread, time::Instant};

/// Default number of negative samples per positive edge.
const DEFAULT_NEG_RATE: usize = 5;

/// Number of pre-generated edge sample batches kept on GPU.
/// More batches = more stochastic variety across epochs.
const EDGE_BATCH_COUNT: usize = 16;

/// How often to read loss back from GPU (every N epochs).
const LOSS_READBACK_INTERVAL: usize = 5;

/// How often to render loss plots when verbose (every N epochs).
const PLOT_INTERVAL: usize = 25;

/// Maximum number of positive edges sampled per epoch.
/// Caps the per-epoch cost regardless of dataset size.
/// At 20K samples with k=15 there are 300K edges — using all of them
/// makes each epoch O(300K). Capping at 50K makes epochs ~6× cheaper
/// while still covering the full graph over multiple epochs.
const MAX_POS_EDGES_PER_EPOCH: usize = 50_000;

/// Train the UMAP model using sparse edge-based loss with negative sampling.
///
/// Per-epoch cost is O(min(n·k, MAX_EDGES) + neg_samples).
pub fn train_sparse<B: AutodiffBackend, F: Float>(
    name: &str,
    mut model: UMAPModel<B>,
    num_samples: usize,
    num_features: usize,
    mut data: Vec<F>,
    config: &TrainingConfig,
    device: Device<B>,
    exit_rx: Receiver<()>,
    labels: Option<Vec<String>>,
) -> (UMAPModel<B>, Vec<F>, F)
where
    F: FromPrimitive + Send + Sync + burn::tensor::Element,
{
    std::fs::create_dir_all("figures").expect("Could not create figures/ directory");

    let k = config.k_neighbors;
    let repulsion_strength = config.repulsion_strength;
    let neg_rate = DEFAULT_NEG_RATE;

    if num_samples <= k {
        panic!("num_samples ({num_samples}) must be > k_neighbors ({k}).");
    }

    // ── Z-score normalise input features ─────────────────────────────────────
    normalize_data(&mut data, num_samples, num_features);

    // ── Precompute global KNN once ───────────────────────────────────────────
    println!(
        "[fast-umap] Precomputing global k-NN (k={k}) for {num_samples} point(s) …"
    );
    let knn_start = Instant::now();

    let all_data_tensor: Tensor<B, 2> =
        convert_vector_to_tensor(data.clone(), num_samples, num_features, &device);

    let knn_indices: Vec<i32>;
    {
        let hd_pairwise = pairwise_distances(all_data_tensor.clone());
        let flat: Vec<f32> = hd_pairwise.to_data().to_vec::<f32>().unwrap();
        let (idx, _dist) = knn_from_pairwise_cpu(&flat, num_samples, k);
        knn_indices = idx;
    }

    let n_all_edges = num_samples * k;

    // Build full positive edge arrays (used as pool for subsampling)
    let mut all_pos_edges: Vec<(i64, i64)> = Vec::with_capacity(n_all_edges);
    for i in 0..num_samples {
        for j in 0..k {
            all_pos_edges.push((i as i64, knn_indices[i * k + j] as i64));
        }
    }

    // Determine how many positive edges to sample per epoch
    let n_pos = n_all_edges.min(MAX_POS_EDGES_PER_EPOCH);
    let subsampling = n_pos < n_all_edges;
    let n_neg = n_pos * neg_rate / k.max(1); // scale neg samples with pos
    let n_neg = n_neg.max(num_samples); // at least 1 neg per point
    let n_total = n_pos + n_neg;

    // ── Pre-generate fused edge-sample batches on GPU ────────────────────────
    let mut rng = rand::rng();

    let fused_batches: Vec<(Tensor<B, 1, Int>, Tensor<B, 1, Int>, usize, usize)> =
        (0..EDGE_BATCH_COUNT)
            .map(|_| {
                // Sample positive edges (with replacement if subsampling)
                let pos_sample: Vec<&(i64, i64)> = if subsampling {
                    let mut indices: Vec<usize> = (0..n_all_edges).collect();
                    indices.shuffle(&mut rng);
                    indices.iter().take(n_pos).map(|&i| &all_pos_edges[i]).collect()
                } else {
                    all_pos_edges.iter().collect()
                };

                let actual_n_pos = pos_sample.len();

                let mut all_head: Vec<i64> = Vec::with_capacity(n_total);
                let mut all_tail: Vec<i64> = Vec::with_capacity(n_total);

                // Positive edges
                for &(h, t) in &pos_sample {
                    all_head.push(*h);
                    all_tail.push(*t);
                }

                // Negative samples
                let actual_n_neg = n_neg;
                for _ in 0..actual_n_neg {
                    let i = rng.random_range(0..num_samples);
                    let mut j = rng.random_range(0..num_samples - 1);
                    if j >= i {
                        j += 1;
                    }
                    all_head.push(i as i64);
                    all_tail.push(j as i64);
                }

                let total = actual_n_pos + actual_n_neg;
                let h = Tensor::from_data(
                    TensorData::new(all_head, [total]),
                    &device,
                );
                let t = Tensor::from_data(
                    TensorData::new(all_tail, [total]),
                    &device,
                );
                (h, t, actual_n_pos, total)
            })
            .collect();

    let knn_elapsed = knn_start.elapsed();
    if subsampling {
        println!(
            "[fast-umap] Global k-NN done in {:.2}s ({n_all_edges} edges, sampling {n_pos}+{n_neg} per epoch). Starting sparse training …",
            knn_elapsed.as_secs_f64()
        );
    } else {
        println!(
            "[fast-umap] Global k-NN done in {:.2}s ({n_all_edges} edges). Starting sparse training …",
            knn_elapsed.as_secs_f64()
        );
    }

    // ── Optimizer ─────────────────────────────────────────────────────────────
    let config_optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.penalty)))
        .with_beta_1(config.beta1 as f32)
        .with_beta_2(config.beta2 as f32);
    let mut optim = config_optimizer.init();

    let start_time = Instant::now();

    let pb = if config.verbose {
        let pb = ProgressBar::new(config.epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{bar:40} | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    let mut epoch = 0usize;
    let mut losses: Vec<F> = vec![];
    let mut best_loss = F::infinity();
    let mut last_read_loss = F::infinity();
    let mut epochs_without_improvement = 0i32;

    // ── In-memory best-model checkpoint ──────────────────────────────────────
    let mut best_record = model.clone().into_record();

    'main: loop {
        if exit_rx.try_recv().is_ok() {
            break 'main;
        }

        // ── Forward: embed ALL points at once ────────────────────────────────
        let embeddings = model.forward(all_data_tensor.clone()); // [n, output_dim]

        // ── Fused gather: 2 selects instead of 4 ────────────────────────────
        let (fused_h, fused_t, batch_n_pos, batch_n_total) =
            &fused_batches[epoch % EDGE_BATCH_COUNT];
        let batch_n_pos = *batch_n_pos;
        let batch_n_total = *batch_n_total;

        let all_head_emb = embeddings.clone().select(0, fused_h.clone());
        let all_tail_emb = embeddings.select(0, fused_t.clone());

        let diff = all_head_emb - all_tail_emb;
        let dist_sq = (diff.clone() * diff).sum_dim(1); // [batch_n_total, 1]

        // ── Split into positive (attraction) and negative (repulsion) ────────
        let dist_sq_pos = dist_sq.clone().slice([0..batch_n_pos]);
        let dist_sq_neg = dist_sq.slice([batch_n_pos..batch_n_total]);

        // Attraction: -log(q) where q = 1/(1+d²)
        let q_pos = (dist_sq_pos + 1.0f32).recip();
        let attraction = q_pos.clamp_min(1e-6f32).log().neg().mean();

        // Repulsion: -log(1-q) where 1-q = d²/(1+d²)
        let dist_sq_neg_safe = dist_sq_neg.clamp_min(1e-8f32);
        let one_minus_q_neg = dist_sq_neg_safe.clone() / (dist_sq_neg_safe + 1.0f32);
        let repulsion = one_minus_q_neg.clamp_min(1e-6f32).log().neg().mean();

        // ── UMAP cross-entropy loss ──────────────────────────────────────────
        let loss = attraction + repulsion_strength * repulsion;

        // ── Read loss from GPU only every N epochs ───────────────────────────
        let should_read = epoch % LOSS_READBACK_INTERVAL == 0
            || epoch >= config.epochs
            || epoch == 0;

        let current_loss = if should_read {
            let v = F::from(loss.clone().into_scalar().to_f64()).unwrap();
            last_read_loss = v;
            v
        } else {
            last_read_loss
        };

        if should_read && (current_loss.is_nan() || current_loss.is_infinite()) {
            eprintln!(
                "[fast-umap] WARNING: loss is {current_loss:.4} at epoch {epoch} — stopping early."
            );
            break 'main;
        }

        // ── Backward + optimizer step ────────────────────────────────────────
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.learning_rate, model, grads);

        // ── Epoch bookkeeping ────────────────────────────────────────────────
        losses.push(current_loss);

        let elapsed = start_time.elapsed();
        if let Some(pb) = &pb {
            pb.inc(1);
            if should_read {
                pb.set_message(format!(
                    "Elapsed: {} | Epoch: {epoch} | Loss: {current_loss:.6} | Best: {best_loss:.6}",
                    format_duration(elapsed),
                ));
            }
        }

        if let Some(timeout) = config.timeout {
            if elapsed >= Duration::from_secs(timeout) {
                break;
            }
        }

        if should_read && current_loss <= best_loss {
            best_loss = current_loss;
            epochs_without_improvement = 0;
            best_record = model.clone().into_record();
        } else if should_read {
            epochs_without_improvement += LOSS_READBACK_INTERVAL as i32;
        }

        if let Some(patience) = config.patience {
            if epochs_without_improvement >= patience {
                break;
            }
        }

        if let Some(min_desired_loss) = config.min_desired_loss {
            if should_read && current_loss < F::from(min_desired_loss).unwrap() {
                break;
            }
        }

        if epoch >= config.epochs {
            break;
        }

        // ── Periodic coloured snapshot (verbose feature flag) ─────────────────
        #[allow(unused_variables)]
        let loss_plot_path = format!("figures/losses_{name}.png");

        #[cfg(feature = "verbose")]
        {
            const STEP: usize = 100;
            if epoch > 0 && epoch % STEP == 0 {
                let losses_snap = losses.clone();
                let model_snap = model.valid();
                let tensor_data = convert_vector_to_tensor(
                    data.clone(),
                    num_samples,
                    num_features,
                    &device,
                );
                let embeddings = model_snap.forward(tensor_data);
                let lpath = loss_plot_path.clone();
                let caption = format!("{name}_{epoch}");
                let fig_path = format!("figures/{name}_{epoch}.png");
                let snap_labels = labels.clone();

                thread::spawn(move || {
                    let chart_config = ChartConfigBuilder::default()
                        .caption(&caption)
                        .path(&fig_path)
                        .build();
                    chart::chart_tensor(embeddings, snap_labels, Some(chart_config));
                    if losses_snap.len() > STEP {
                        plot_loss(losses_snap[STEP..].to_vec(), &lpath).unwrap();
                    }
                });
            }
        }

        if config.verbose && epoch % PLOT_INTERVAL == 0 {
            plot_loss(losses.clone(), &loss_plot_path).unwrap();
        }

        epoch += 1;
    }

    #[cfg(feature = "verbose")]
    {
        let path = format!("figures/losses_{name}.png");
        plot_loss(losses.clone(), &path).unwrap();
    }

    if config.verbose {
        let path = format!("figures/losses_{name}.png");
        plot_loss(losses.clone(), &path).unwrap();
    }

    if let Some(pb) = pb {
        pb.finish();
    }

    // ── Restore best model from in-memory record ─────────────────────────────
    model = model.load_record(best_record);

    (model, losses, best_loss)
}
