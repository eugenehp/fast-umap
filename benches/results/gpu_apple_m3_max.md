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
| `16s×10f [32]→2` | 419.21 µs | **564.85 µs** | 770.33 µs |
| `64s×50f [64]→2` | 424.17 µs | **465.69 µs** | 585.42 µs |
| `128s×50f [128]→2` | 423.50 µs | **474.35 µs** | 551.38 µs |
| `64s×100f [128,64]→3` | 565.00 µs | **637.41 µs** | 824.96 µs |
| `256s×100f [256,128]→2` | 637.75 µs | **674.90 µs** | 723.71 µs |
| `512s×100f [256,128]→2` | 920.92 µs | **1.10 ms** | 1.32 ms |

## `normalize_tensor (GPU / WGPU)`

| Input | Min | **Mean** | Max |
|-------|-----|----------|-----|
| `n=512` | 560.38 µs | **659.77 µs** | 956.71 µs |
| `n=4096` | 554.62 µs | **650.43 µs** | 827.71 µs |
| `n=32768` | 642.21 µs | **711.39 µs** | 843.96 µs |
| `n=262144` | 1.06 ms | **1.13 ms** | 1.23 ms |

## `layer_normalize (GPU / WGPU)`

| Input | Min | **Mean** | Max |
|-------|-----|----------|-----|
| `128×64` | 365.42 µs | **418.61 µs** | 647.79 µs |
| `512×128` | 407.38 µs | **503.48 µs** | 1.46 ms |
| `1000×256` | 567.54 µs | **715.07 µs** | 1.67 ms |
| `4000×512` | 1.75 ms | **2.12 ms** | 3.01 ms |

