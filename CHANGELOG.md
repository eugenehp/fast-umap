# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] — 2026-02-24

### Added

- **New public API** mirroring [umap-rs](https://crates.io/crates/umap-rs):
  - `Umap<B>` — main algorithm struct (`Umap::new(config)`)
  - `FittedUmap<B>` — fitted model with `.embedding()`, `.into_embedding()`, `.transform()`, `.config()`
  - `UmapConfig` — configuration with nested `GraphParams` and `OptimizationParams`
  - `ManifoldParams`, `GraphParams`, `OptimizationParams` — structured config types (all `Serialize`/`Deserialize`)
  - `Metric` enum — `Euclidean`, `EuclideanKNN`, `Manhattan`, `Cosine`
- **Sparse training path** (`train_sparse`) — O(n·k) per epoch instead of O(n²):
  - Global KNN precomputed once on GPU
  - Edge subsampling — caps at 50K positive edges per epoch regardless of dataset size
  - Negative sampling — random non-neighbor pairs for repulsion term
  - Pre-batched index tensors — 16 shuffled batches uploaded to GPU once, cycled per epoch
  - Fused gather — 2 GPU `select()` calls instead of 4
  - Async loss readback — GPU→CPU sync every 5 epochs to avoid pipeline stalls
  - In-memory checkpointing — best model kept in RAM, no disk I/O during training
- **Crate comparison benchmark** (`examples/crate_comparison.rs`):
  - Benchmarks fast-umap vs umap-rs on 6 dataset sizes (500–20K samples)
  - Outputs `figures/crate_comparison.{json,svg,md}`
  - GPU warmup pass to exclude shader compilation from timings
- **MNIST benchmark** (`examples/bench_mnist.rs`):
  - Trains UMAP on 10K MNIST digits and generates `figures/mnist.png` + loss curve
- **Unified benchmark script** (`bench.sh`):
  - Runs all benchmarks: hardware micro-benchmarks, crate comparison, MNIST
  - Supports `--only <name>`, `--skip-mnist`, `--criterion` flags
  - Replaces old `generate_images.sh`
- **Prelude updated** — re-exports `UmapConfig`, `ManifoldParams`, `GraphParams`, `OptimizationParams`, `Metric`

### Changed

- **Default training path**: `Umap::fit()` uses sparse training (`train_sparse`); legacy `UMAP::fit()` still uses dense `train()` for backward compatibility
- **Examples no longer generate figures** — examples are lightweight smoke tests; use `./bench.sh` for figure generation
- `run_all.sh` simplified to only run examples (no figures); points to `bench.sh` for benchmarks
- Benchmark results and micro-benchmark tables in README updated to latest run

### Performance

Benchmarked on Apple M3 Max (fast-umap: 50 epochs GPU, umap-rs: 200 epochs CPU):

| Dataset | fast-umap | umap-rs | Speedup |
|---------|-----------|---------|---------|
| 500 × 50 | 0.22s | 0.06s | 0.29× *(umap-rs faster)* |
| 1 000 × 50 | 0.81s | 0.12s | 0.15× *(umap-rs faster)* |
| 2 000 × 100 | 0.92s | 0.44s | 0.48× *(umap-rs faster)* |
| 5 000 × 100 | 1.62s | 2.27s | **1.4× faster** |
| 10 000 × 100 | 2.06s | 8.67s | **4.2× faster** |
| 20 000 × 100 | 3.72s | 34.22s | **9.2× faster** |

### Removed

- `generate_images.sh` — replaced by `bench.sh`

---

## [1.0.0] — 2026-02-23

### Added

- **burn 0.20.1** upgrade (from 0.18); cubecl 0.9 (from 0.6)
- **36 unit tests** — covering normalization, tensor conversion, model shape, distance math (all CPU-only via NdArray)
- **Hardware-tagged benchmarks** (`cargo run --release --bin bench_report`):
  - Auto-detects CPU & GPU
  - Writes `.md` + `.svg` to `benches/results/`
  - CPU vs GPU comparison chart
- **MNIST example** (`examples/mnist.rs`) — 10K digits projected to 2-D with coloured class labels
- **Hyperparameter benchmark** (`examples/mnist_benchmark.rs`) — sweep over learning rates, batch sizes, penalties, hidden layer configs
- **Charting** — `chart` module with scatter plots and loss curves via plotters
- **Labels in plots** — coloured by class when labels provided
- **Early stopping** — `patience`, `min_desired_loss`, `timeout` parameters
- **Distance metrics** — `Euclidean`, `EuclideanKNN`, `Manhattan`, `Cosine`
- **Custom CubeCL GPU kernels** — Euclidean pairwise distance and KNN

### Fixed

- `Backend::seed` signature updated for burn 0.20 (`(&device, seed)`)
- `as_tensor_arg` no longer takes a generic type parameter
- `NodeID` renamed to `NodeId`
- `ABSOLUTE_POS_X/Y` cast to `usize` for shape indexing in CubeCL kernels
- `normalize_tensor` replaced `.to_vec::<bool>()` with f32 arithmetic (WGPU stores bools as u32)
- All 4 `unused Result` warnings from kernel launches resolved with `.expect()`

---

## [0.1.0] — 2024-12-01

### Added

- Initial release
- GPU-accelerated UMAP via burn + CubeCL
- `UMAP` struct with `.fit()` and `.transform()`
- `UMAPModel` neural network with configurable hidden layers
- `TrainingConfig` builder
- Basic examples (simple, advanced)
