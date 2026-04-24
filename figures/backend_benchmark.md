# WGPU vs MLX Backend Benchmark

Epochs: 50

| Dataset | WGPU Total | MLX Total | WGPU Fit | MLX Fit | Speedup (total) |
|---------|------------|-----------|----------|---------|----------------|
| 1000x50 | 1.672s | 0.070s | 1.672s | 0.069s | MLX 23.98x faster |
| 5000x100 | 4.735s | 0.394s | 4.733s | 0.391s | MLX 12.02x faster |
| 10000x100 | 5.087s | 0.549s | 5.081s | 0.544s | MLX 9.26x faster |
| 20000x100 | 6.819s | 1.618s | 6.809s | 1.608s | MLX 4.21x faster |
