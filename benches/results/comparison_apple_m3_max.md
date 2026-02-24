# fast-umap — CPU vs GPU Comparison

> **CPU:** Apple M3 Max  
> **GPU:** Apple M3 Max (WGPU / Metal)  
> **Backend:** burn 0.20.1 · NdArray vs WGPU  
> **Reproduce:** `cargo run --release --bin bench_report`

![chart](comparison.svg)

Speedup = CPU mean ÷ GPU mean.  
> 1× = GPU faster, < 1× = CPU faster (e.g. small tensors where dispatch overhead dominates).

---

## `model_forward`

| Input | model_forward (CPU / NdArray) | model_forward (GPU / WGPU) | Speedup |
|-------|-------------------------------|----------------------------|----------|
| `16s×10f [32]→2` | 34.54 µs | 617.36 µs | 0.06× *(CPU faster)* |
| `64s×50f [64]→2` | 34.19 µs | 481.15 µs | 0.07× *(CPU faster)* |
| `128s×50f [128]→2` | 58.62 µs | 475.36 µs | 0.12× *(CPU faster)* |
| `64s×100f [128,64]→3` | 80.29 µs | 688.03 µs | 0.12× *(CPU faster)* |
| `256s×100f [256,128]→2` | 293.25 µs | 690.56 µs | 0.42× *(CPU faster)* |

## `normalize_tensor`

| Input | normalize_tensor (1-D min-max) | normalize_tensor (GPU / WGPU) | Speedup |
|-------|--------------------------------|-------------------------------|----------|
| `n=512` | 2.86 µs | 695.04 µs | 0.00× *(CPU faster)* |
| `n=4096` | 9.59 µs | 661.72 µs | 0.01× *(CPU faster)* |
| `n=32768` | 70.28 µs | 712.00 µs | 0.10× *(CPU faster)* |

## `layer_normalize`

| Input | layer_normalize (2-D) | layer_normalize (GPU / WGPU) | Speedup |
|-------|-----------------------|------------------------------|----------|
| `128×64` | 20.02 µs | 470.85 µs | 0.04× *(CPU faster)* |
| `512×128` | 117.17 µs | 500.43 µs | 0.23× *(CPU faster)* |
| `1000×256` | 420.11 µs | 661.83 µs | 0.63× *(CPU faster)* |

