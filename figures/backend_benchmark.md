# WGPU vs MLX Backend Benchmark

Epochs: 50

| Dataset | WGPU Total | MLX Total | WGPU Fit | MLX Fit | Speedup (total) |
|---------|------------|-----------|----------|---------|----------------|
| 1000x50 | 1.812s | 0.073s | 1.811s | 0.073s | MLX 24.66x faster |
| 5000x100 | 5.448s | 0.279s | 5.446s | 0.278s | MLX 19.49x faster |
| 10000x100 | 5.780s | 0.709s | 5.777s | 0.705s | MLX 8.15x faster |
| 20000x100 | 7.243s | 1.582s | 7.236s | 1.574s | MLX 4.58x faster |
