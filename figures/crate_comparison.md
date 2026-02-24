# fast-umap vs umap-rs — Crate Comparison

> **Date:** 2026-02-24 05:13 UTC  
> **fast-umap:** 50 epochs (parametric, GPU)  
> **umap-rs:** 200 epochs (classical, CPU)  
> **Reproduce:** `cargo run --release --example crate_comparison`

![chart](crate_comparison.svg)

## Total Time (data prep + fit + extract)

| Dataset | fast-umap | umap-rs | Speedup |
|---------|-----------|---------|--------|
| 500×50 | 0.220s | 0.064s | 0.29×  |
| 1000×50 | 0.814s | 0.119s | 0.15×  |
| 2000×100 | 0.920s | 0.438s | 0.48×  |
| 5000×100 | 1.623s | 2.272s | 1.40× 🚀 |
| 10000×100 | 2.062s | 8.666s | 4.20× 🚀 |
| 20000×100 | 3.717s | 34.216s | 9.21× 🚀 |

## Fit Time Only

| Dataset | fast-umap | umap-rs | Speedup |
|---------|-----------|---------|--------|
| 500×50 | 0.219s | 0.053s | 0.24×  |
| 1000×50 | 0.814s | 0.074s | 0.09×  |
| 2000×100 | 0.919s | 0.115s | 0.12×  |
| 5000×100 | 1.620s | 0.216s | 0.13×  |
| 10000×100 | 2.055s | 0.397s | 0.19×  |
| 20000×100 | 3.704s | 0.720s | 0.19×  |

---

**Notes:**
- fast-umap is a *parametric* UMAP (neural network, GPU-accelerated via burn/CubeCL)
- umap-rs is a *classical* UMAP (SGD on embedding, CPU, multithreaded via rayon)
- fast-umap includes batch-local KNN computation; umap-rs requires precomputed KNN (included in total time)
- fast-umap can `transform()` new unseen data; umap-rs cannot (yet)
