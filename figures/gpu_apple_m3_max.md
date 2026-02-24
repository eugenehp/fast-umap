# fast-umap — Benchmark Results

> **Hardware:** GPU: Apple M3 Max  
> **Detail:** Apple M3 Max  
> **Backend:** burn 0.20.1  
> **Reproduce:** `cargo run --release --bin bench_report`

![chart](benchmark_results.svg)

All times are **[min, mean, max]** over multiple timed iterations.

---

## `model_forward (GPU / WGPU)`

| Input | Min | **Mean** | Max |
|-------|-----|----------|-----|
| `16s×10f [32]→2` | 369.67 µs | **482.31 µs** | 1.19 ms |
| `64s×50f [64]→2` | 349.33 µs | **462.40 µs** | 747.08 µs |
| `128s×50f [128]→2` | 423.04 µs | **614.21 µs** | 846.25 µs |
| `64s×100f [128,64]→3` | 504.79 µs | **642.49 µs** | 997.71 µs |
| `256s×100f [256,128]→2` | 460.79 µs | **530.90 µs** | 999.92 µs |
| `512s×100f [256,128]→2` | 870.88 µs | **1.05 ms** | 1.36 ms |

## `normalize_tensor (GPU / WGPU)`

| Input | Min | **Mean** | Max |
|-------|-----|----------|-----|
| `n=512` | 554.38 µs | **788.45 µs** | 1.36 ms |
| `n=4096` | 606.17 µs | **681.95 µs** | 852.54 µs |
| `n=32768` | 635.21 µs | **689.35 µs** | 779.08 µs |
| `n=262144` | 1.07 ms | **1.13 ms** | 1.31 ms |

## `layer_normalize (GPU / WGPU)`

| Input | Min | **Mean** | Max |
|-------|-----|----------|-----|
| `128×64` | 462.92 µs | **667.58 µs** | 798.12 µs |
| `512×128` | 486.08 µs | **626.13 µs** | 832.00 µs |
| `1000×256` | 618.00 µs | **667.16 µs** | 714.17 µs |
| `4000×512` | 1.55 ms | **1.66 ms** | 1.79 ms |

