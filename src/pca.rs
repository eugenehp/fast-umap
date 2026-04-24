//! PCA (Principal Component Analysis) for warm-starting UMAP embeddings.
//!
//! Computes the top-k principal components of centered data using the
//! covariance-eigendecomposition approach via power iteration. Works on
//! any Burn backend (CPU, WGPU, MLX) without external linear algebra deps.

use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Compute PCA projection of `data` [n, d] down to `n_components` dimensions.
///
/// Returns `(projected, components, mean)`:
/// - `projected`  — `[n, n_components]` PCA-projected data
/// - `components` — `[n_components, d]` principal component vectors (rows)
/// - `mean`       — `[1, d]` column means used for centering
pub fn pca<B: Backend>(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    device: &burn::tensor::Device<B>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // 1. Compute column means
    let mut mean = vec![0.0f64; n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            mean[j] += data[i * n_features + j] as f64;
        }
    }
    for j in 0..n_features {
        mean[j] /= n_samples as f64;
    }

    // 2. Center the data
    let mut centered = vec![0.0f32; n_samples * n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[i * n_features + j] = data[i * n_features + j] - mean[j] as f32;
        }
    }

    // 3. Compute covariance matrix C = X^T X / (n-1), shape [d, d]
    //    Use Burn tensors for the matmul (GPU-accelerated if available)
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(centered.clone(), [n_samples, n_features]),
        device,
    );
    let cov = x.clone().transpose().matmul(x).div_scalar((n_samples - 1) as f32);

    // 4. Extract eigenvectors via power iteration with deflation
    let mut cov_data: Vec<f32> = cov.to_data().to_vec::<f32>().unwrap();
    let mut components = Vec::with_capacity(n_components * n_features);

    for _comp in 0..n_components {
        // Power iteration to find dominant eigenvector
        let mut v = vec![0.0f32; n_features];
        // Initialize with a non-zero vector (alternating signs for stability)
        for j in 0..n_features {
            v[j] = if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        normalize_vec(&mut v);

        for _iter in 0..100 {
            // w = C @ v
            let mut w = vec![0.0f32; n_features];
            for i in 0..n_features {
                let mut s = 0.0f32;
                for j in 0..n_features {
                    s += cov_data[i * n_features + j] * v[j];
                }
                w[i] = s;
            }
            normalize_vec(&mut w);

            // Check convergence (dot product of old and new ~= 1)
            let dot: f32 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            v = w;
            if dot.abs() > 1.0 - 1e-8 {
                break;
            }
        }

        components.extend_from_slice(&v);

        // Deflate: C = C - eigenvalue * v * v^T
        // eigenvalue = v^T C v
        let mut eigenvalue = 0.0f32;
        for i in 0..n_features {
            let mut s = 0.0f32;
            for j in 0..n_features {
                s += cov_data[i * n_features + j] * v[j];
            }
            eigenvalue += v[i] * s;
        }

        for i in 0..n_features {
            for j in 0..n_features {
                cov_data[i * n_features + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    // 5. Project: projected = centered @ components^T, shape [n, n_components]
    let mut projected = vec![0.0f32; n_samples * n_components];
    for i in 0..n_samples {
        for c in 0..n_components {
            let mut s = 0.0f32;
            for j in 0..n_features {
                s += centered[i * n_features + j] * components[c * n_features + j];
            }
            projected[i * n_components + c] = s;
        }
    }

    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
    (projected, components, mean_f32)
}

fn normalize_vec(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
