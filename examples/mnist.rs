use burn::{backend::*, module::*, prelude::*};
use crossbeam_channel::unbounded;
use fast_umap::chart;
#[allow(unused)]
use fast_umap::{
    chart::*,
    model::*,
    prelude::*,
    train::{train, LossReduction},
    utils::*,
};
use mnist::*;
use wgpu::WgpuRuntime;

fn main() {
    let (exit_tx, exit_rx) = unbounded();

    ctrlc::set_handler(move || exit_tx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let num_samples = 10_000_usize;
    let num_features = 28 * 28; // 784

    // ── Hyperparameters ───────────────────────────────────────────────────────
    //
    // k-NN is computed batch-locally (within each mini-batch), so batch_size
    // directly controls the neighbourhood quality.  With 10 classes and
    // batch_size = 2 000, each batch contains ~200 examples per class, giving
    // the repulsion term a strong cross-class signal.  The [2000×2000] pairwise
    // matrix is only 16 MB — very fast on any GPU.
    let batch_size = 2_000;

    let k_neighbors = 15;
    let output_size = 2;
    let hidden_sizes = vec![256]; // single hidden layer — fast forward/backward
    let learning_rate = 1e-3;    // higher LR → escapes plateau much faster
    let penalty = 1e-6;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epochs = 1_000;
    let patience = 100; // stop if loss doesn't improve for this many consecutive epochs
    let seed = 9999;
    let verbose = true;
    let metric = Metric::Euclidean;
    let repulsion_strength = 3.0_f32; // strong repulsion → tight, separated clusters

    MyBackend::seed(&device, seed);

    let Mnist {
        trn_img,
        trn_lbl,
        ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(num_samples as u32)
        .finalize();

    let train_data: Vec<f64> = trn_img.into_iter().map(|b| b as f64).collect();

    // Build labels once — reused for coloured snapshots AND the final chart.
    let labels: Vec<String> = trn_lbl.iter().map(|d| format!("{d}")).collect();

    let model_config = UMAPModelConfigBuilder::default()
        .input_size(num_features)
        .hidden_sizes(hidden_sizes)
        .output_size(output_size)
        .build()
        .unwrap();

    let model: UMAPModel<MyAutodiffBackend> = UMAPModel::new(&model_config, &device);

    let config = TrainingConfig::builder()
        .with_epochs(epochs)
        .with_batch_size(batch_size)
        .with_learning_rate(learning_rate)
        .with_beta1(beta1)
        .with_beta2(beta2)
        .with_verbose(verbose)
        .with_metric(metric.into())
        .with_k_neighbors(k_neighbors)
        .with_patience(patience)
        .with_repulsion_strength(repulsion_strength)
        .with_penalty(penalty)
        .build()
        .expect("Failed to build TrainingConfig");

    let (model, _, _) = train(
        "mnist",
        model,
        num_samples,
        num_features,
        train_data.clone(), // train() normalises its own copy internally
        &config,
        device.clone(),
        exit_rx,
        Some(labels.clone()), // colour epoch snapshots by digit class
    );

    let model = model.valid();

    // ── Normalise before inference ────────────────────────────────────────────
    //
    // train() z-score-normalises the copy it receives, but the original
    // train_data here is still raw pixel values (0–255).  We must apply the
    // same normalisation so the model sees the same distribution it was
    // trained on (otherwise the output scale is wildly off).
    let mut train_data_norm = train_data;
    normalize_data(&mut train_data_norm, num_samples, num_features);

    let global =
        convert_vector_to_tensor(train_data_norm, num_samples, num_features, &device);
    let local = model.forward(global);

    let chart_config = ChartConfigBuilder::default()
        .caption("MNIST")
        .path("figures/mnist.png")
        .build();

    chart::chart_tensor(local, Some(labels), Some(chart_config));
}
