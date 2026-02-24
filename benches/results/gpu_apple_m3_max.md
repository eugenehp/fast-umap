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
| `16s×10f [32]→2` | 408.46 µs | **617.36 µs** | 893.83 µs |
| `64s×50f [64]→2` | 430.38 µs | **481.15 µs** | 776.46 µs |
| `128s×50f [128]→2` | 432.08 µs | **475.36 µs** | 576.12 µs |
| `64s×100f [128,64]→3` | 549.29 µs | **688.03 µs** | 1.82 ms |
| `256s×100f [256,128]→2` | 631.38 µs | **690.56 µs** | 827.75 µs |
| `512s×100f [256,128]→2` | 926.08 µs | **1.08 ms** | 1.42 ms |

## `normalize_tensor (GPU / WGPU)`

| Input | Min | **Mean** | Max |
|-------|-----|----------|-----|
| `n=512` | 572.08 µs | **695.04 µs** | 1.28 ms |
| `n=4096` | 590.00 µs | **661.72 µs** | 821.21 µs |
| `n=32768` | 629.46 µs | **712.00 µs** | 882.96 µs |
| `n=262144` | 1.08 ms | **1.12 ms** | 1.22 ms |

## `layer_normalize (GPU / WGPU)`

| Input | Min | **Mean** | Max |
|-------|-----|----------|-----|
| `128×64` | 437.29 µs | **470.85 µs** | 609.12 µs |
| `512×128` | 466.96 µs | **500.43 µs** | 647.67 µs |
| `1000×256` | 616.54 µs | **661.83 µs** | 776.67 µs |
| `4000×512` | 1.81 ms | **1.93 ms** | 2.15 ms |

