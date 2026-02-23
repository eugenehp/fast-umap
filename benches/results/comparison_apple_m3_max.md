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
| `16s×10f [32]→2` | 36.03 µs | 564.85 µs | 0.06× *(CPU faster)* |
| `64s×50f [64]→2` | 34.38 µs | 465.69 µs | 0.07× *(CPU faster)* |
| `128s×50f [128]→2` | 57.50 µs | 474.35 µs | 0.12× *(CPU faster)* |
| `64s×100f [128,64]→3` | 74.67 µs | 637.41 µs | 0.12× *(CPU faster)* |
| `256s×100f [256,128]→2` | 248.14 µs | 674.90 µs | 0.37× *(CPU faster)* |

## `normalize_tensor`

| Input | normalize_tensor (1-D min-max) | normalize_tensor (GPU / WGPU) | Speedup |
|-------|--------------------------------|-------------------------------|----------|
| `n=512` | 2.92 µs | 659.77 µs | 0.00× *(CPU faster)* |
| `n=4096` | 9.70 µs | 650.43 µs | 0.01× *(CPU faster)* |
| `n=32768` | 70.30 µs | 711.39 µs | 0.10× *(CPU faster)* |

## `layer_normalize`

| Input | layer_normalize (2-D) | layer_normalize (GPU / WGPU) | Speedup |
|-------|-----------------------|------------------------------|----------|
| `128×64` | 18.11 µs | 418.61 µs | 0.04× *(CPU faster)* |
| `512×128` | 117.04 µs | 503.48 µs | 0.23× *(CPU faster)* |
| `1000×256` | 416.44 µs | 715.07 µs | 0.58× *(CPU faster)* |

