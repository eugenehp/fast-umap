mod config;
mod get_distance_by_metric;
mod train_sparse;

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
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{cast::ToElement, Device, Int, IndexingUpdateOp, Tensor, TensorData},
};
pub use config::*;
pub use train_sparse::train_sparse;

use crossbeam_channel::Receiver;
use get_distance_by_metric::*;
use indicatif::{ProgressBar, ProgressStyle};
use num::{Float, FromPrimitive};
use std::time::Duration;
use std::{thread, time::Instant};

/// Train the UMAP model using the UMAP cross-entropy loss.
///
/// # Loss function
///
/// For each mini-batch the model projects `bs` points to 2-D.
/// Pairwise Euclidean distances in the 2-D space are mapped through the
/// Student-t kernel `q_ij = 1 / (1 + d_ij²)`.
///
/// k-NN is computed **within each mini-batch** in the original high-dim space.
/// All `bs × k` neighbour pairs are guaranteed to be inside the batch
/// (no sparsity), giving a correct attraction/repulsion balance:
///
/// ```text
/// attraction  =  mean over k-NN pairs   [ −log( q_ij ) ]
/// repulsion   =  mean over non-NN pairs  [ −log( 1 − q_ij ) ]
/// loss        =  attraction + repulsion_strength × repulsion
/// ```
///
/// `−log(q)` → 0 as d → 0  (pull neighbours together).
/// `−log(1−q)` → 0 as d → ∞ (push non-neighbours apart).
pub fn train<B: AutodiffBackend, F: Float>(
    name: &str,
    mut model: UMAPModel<B>,
    num_samples: usize,
    num_features: usize,
    mut data: Vec<F>,
    config: &TrainingConfig,
    device: Device<B>,
    exit_rx: Receiver<()>,
    // Optional per-sample labels for coloured epoch snapshots.
    // Pass `None` for monochrome.
    labels: Option<Vec<String>>,
) -> (UMAPModel<B>, Vec<F>, F)
where
    F: FromPrimitive + Send + Sync + burn::tensor::Element,
{
    std::fs::create_dir_all("figures").expect("Could not create figures/ directory");

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_path = format!("/tmp/{name}.bin");

    let batch_size = config.batch_size;
    let k = config.k_neighbors;
    let repulsion_strength = config.repulsion_strength;

    if batch_size <= 1 {
        panic!("batch_size must be > 1");
    }
    if k >= batch_size {
        panic!(
            "k_neighbors ({k}) must be < batch_size ({batch_size}). \
             Increase batch_size or decrease k_neighbors."
        );
    }

    // ── Z-score normalise input features ─────────────────────────────────────
    normalize_data(&mut data, num_samples, num_features);

    // ─────────────────────────────────────────────────────────────────────────
    // Precompute per-batch artefacts (done once before the epoch loop)
    //
    // For each mini-batch of `bs` points we:
    //   1. Compute the [bs × bs] pairwise distance matrix in high-dim space
    //      on the GPU, then pull to CPU for sorting.
    //   2. Find the k nearest neighbours of each point *within the batch*
    //      (batch-local k-NN).  All k neighbours are guaranteed in-batch, so
    //      the attraction term always has bs × k pairs — never sparse.
    //   3. Build a dense [bs × bs] `knn_indicator` tensor and the
    //      complementary `non_neighbor_mask`.
    //   4. Store the input tensor slice for cheap reuse across epochs.
    // ─────────────────────────────────────────────────────────────────────────
    println!(
        "[fast-umap] Precomputing batch-local k-NN (k={k}) for {} point(s) …",
        num_samples
    );

    let mut batches_start: Vec<usize> = Vec::new();
    let mut tensor_batches: Vec<Tensor<B, 2>> = Vec::new();
    // Per-batch: (knn_indicator [bs,bs], within_knn_count, non_neighbor_mask [bs,bs])
    let mut batch_artefacts: Vec<(Tensor<B, 2>, usize, Tensor<B, 2>)> = Vec::new();

    for batch_start in (0..num_samples).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_samples);
        let bs = batch_end - batch_start;

        // Build raw batch slice
        let mut batch_vec: Vec<F> = Vec::with_capacity(bs * num_features);
        for i in batch_start..batch_end {
            let s = i * num_features;
            batch_vec.extend_from_slice(&data[s..s + num_features]);
        }
        let tensor_batch: Tensor<B, 2> =
            convert_vector_to_tensor(batch_vec, bs, num_features, &device);

        // (1) High-dim pairwise distances, pulled to CPU for sorting
        let hd_pairwise = pairwise_distances(tensor_batch.clone());
        let flat: Vec<f32> = hd_pairwise.to_data().to_vec::<f32>().unwrap();

        // (2) Batch-local k-NN sort (CPU)
        let (idx_flat, _) = knn_from_pairwise_cpu(&flat, bs, k);
        // idx_flat[i*k .. i*k+k] = local indices of k nearest neighbours of point i

        // (3a) Dense [bs, bs] knn_indicator
        let mut knn_ind_flat = vec![0.0f32; bs * bs];
        for local_i in 0..bs {
            for j in 0..k {
                let local_j = idx_flat[local_i * k + j] as usize;
                knn_ind_flat[local_i * bs + local_j] = 1.0;
            }
        }
        let knn_indicator: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(knn_ind_flat, [bs, bs]),
            &device,
        );
        let within_knn_count = bs * k; // all k-NN are in-batch by construction

        // (3b) Diagonal (self-pair) identity matrix — built on GPU via
        //      arange + scatter to avoid a large CPU allocation.
        let diag_idx: Tensor<B, 1, Int> =
            Tensor::arange(0i64..(bs as i64), &device);
        let eye: Tensor<B, 2> =
            Tensor::<B, 2>::zeros([bs, bs], &device).scatter(
                1,
                diag_idx.reshape([bs, 1]),
                Tensor::ones([bs, 1], &device),
                IndexingUpdateOp::Add,
            );

        // non_neighbor_mask[i,j] = 1  iff j is not a k-NN of i and i ≠ j
        let non_neighbor_mask =
            (Tensor::ones([bs, bs], &device) - knn_indicator.clone() - eye)
                .clamp_min(0.0f32);

        batches_start.push(batch_start);
        tensor_batches.push(tensor_batch);
        batch_artefacts.push((knn_indicator, within_knn_count, non_neighbor_mask));
    }

    let num_batches = batches_start.len();
    println!(
        "[fast-umap] Precomputation done ({num_batches} batch(es)). Starting training …"
    );

    // Concatenate all batch tensors for O(1) slicing in the epoch loop.
    let tensor_batches_all = Tensor::<B, 2>::cat(tensor_batches, 0);

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
    let mut epochs_without_improvement = 0i32;

    'main: loop {
        let mut epoch_loss_sum = F::zero();
        let mut num_batches_seen = 0usize;

        for batch_idx in 0..num_batches {
            if exit_rx.try_recv().is_ok() {
                break 'main;
            }

            let batch_start = batches_start[batch_idx];
            let batch_end = (batch_start + batch_size).min(num_samples);
            let bs = batch_end - batch_start;

            let tensor_batch = tensor_batches_all
                .clone()
                .slice([batch_start..batch_end, 0..num_features]);

            let (knn_indicator, within_knn_count, non_neighbor_mask) =
                batch_artefacts[batch_idx].clone();

            // ── Forward ───────────────────────────────────────────────────────
            let local = model.forward(tensor_batch); // [bs, output_size]

            // ── Squared pairwise distances in embedding space ─────────────────
            let local_pairwise = pairwise_distances(local); // [bs, bs]
            let local_sq = local_pairwise.clone() * local_pairwise; // d²

            // ── Student-t kernel  q = 1 / (1 + d²) ───────────────────────────
            let q = (local_sq.clone() + 1.0f32).recip(); // [bs, bs]

            // ── Attraction: mean over k-NN pairs of −log(q_ij) ───────────────
            // knn_indicator zeros out non-neighbour entries; we normalise by
            // the exact count of k-NN pairs.
            let log_q = q.clone().clamp_min(1e-6f32).log();
            let attraction =
                (log_q.neg() * knn_indicator).sum() / within_knn_count as f32;

            // ── Repulsion: mean over non-k-NN pairs of −log(1 − q_ij) ────────
            // 1 − q = d² / (1 + d²).  Clamp d² to keep gradients finite at
            // the diagonal; non_neighbor_mask zeros out self-pairs anyway.
            let local_sq_safe = local_sq.clamp_min(1e-8f32);
            let one_minus_q = local_sq_safe.clone() / (local_sq_safe + 1.0f32);
            let repulsion_per_pair = one_minus_q.clamp_min(1e-6f32).log().neg();

            let num_non_neighbors =
                (bs * (bs - 1)).saturating_sub(within_knn_count) as f32;
            let repulsion = (repulsion_per_pair * non_neighbor_mask).sum()
                / num_non_neighbors.max(1.0);

            // ── UMAP cross-entropy loss ───────────────────────────────────────
            let loss = attraction + repulsion_strength * repulsion;

            let current_loss = F::from(loss.clone().into_scalar().to_f64()).unwrap();

            if current_loss.is_nan() || current_loss.is_infinite() {
                eprintln!(
                    "[fast-umap] WARNING: loss is {current_loss:.4} at epoch {epoch}, \
                     batch {batch_idx} — stopping early."
                );
                break 'main;
            }

            epoch_loss_sum = epoch_loss_sum + current_loss;
            num_batches_seen += 1;

            // ── Backward + optimizer step ─────────────────────────────────────
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        // ── Epoch bookkeeping ─────────────────────────────────────────────────
        let epoch_loss = if num_batches_seen > 0 {
            epoch_loss_sum / F::from_usize(num_batches_seen).unwrap()
        } else {
            F::infinity()
        };
        losses.push(epoch_loss);

        let elapsed = start_time.elapsed();
        if let Some(pb) = &pb {
            pb.inc(1);
            pb.set_message(format!(
                "Elapsed: {} | Epoch: {epoch} | Loss: {epoch_loss:.6} | Best: {best_loss:.6}",
                format_duration(elapsed),
            ));
        }

        if let Some(timeout) = config.timeout {
            if elapsed >= Duration::from_secs(timeout) {
                break;
            }
        }

        if epoch_loss <= best_loss {
            best_loss = epoch_loss;
            epochs_without_improvement = 0;
            model
                .clone()
                .save_file(model_path.clone(), &recorder)
                .expect("Could not save model checkpoint");
        } else {
            epochs_without_improvement += 1;
        }

        if let Some(patience) = config.patience {
            if epochs_without_improvement >= patience {
                break;
            }
        }

        if let Some(min_desired_loss) = config.min_desired_loss {
            if epoch_loss < F::from(min_desired_loss).unwrap() {
                break;
            }
        }

        if epoch >= config.epochs {
            break;
        }

        // ── Periodic coloured snapshot (verbose feature flag) ─────────────────
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

        if config.verbose {
            plot_loss(losses.clone(), &loss_plot_path).unwrap();
        }

        epoch += 1;
    }

    #[cfg(feature = "verbose")]
    {
        let path = format!("figures/losses_{name}.png");
        plot_loss(losses.clone(), &path).unwrap();
    }

    if let Some(pb) = pb {
        pb.finish();
    }

    model = model
        .load_file(model_path, &recorder, &device)
        .expect("Could not load best model checkpoint");

    (model, losses, best_loss)
}
