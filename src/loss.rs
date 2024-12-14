use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Device, Tensor},
};

use crate::model::UMAPModel;

/// Compute pairwise Euclidean distance matrix
fn pairwise_distance<B: AutodiffBackend>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
    let sq_x = x.clone().powi_scalar(2).sum().unsqueeze::<1>(); // sum of squares along the rows
    let sq_x_t = sq_x.clone().transpose(); // Transpose the sum for pairwise computation
    let transposed = x.clone().transpose();
    let product = x.clone().matmul(transposed);
    let dist = sq_x + sq_x_t - product.mul_scalar(2).squeeze(1); // Apply the squared Euclidean distance formula
    dist.clamp_min(0.0).sqrt() // Take square root to get Euclidean distance
}

/// Calculate the UMAP loss by comparing pairwise distances between global and local representations
fn umap_loss<B: AutodiffBackend>(
    global: &Tensor<B, 2>, // High-dimensional (global) representation
    local: &Tensor<B, 2>,  // Low-dimensional (local) representation
) -> Tensor<B, 0> {
    // Compute pairwise distances for both global and local representations
    let global_distances = pairwise_distance(global);
    let local_distances = pairwise_distance(local);

    // Compute the loss as the Frobenius norm (L2 loss) between the pairwise distance matrices
    (global_distances - local_distances)
        .powi_scalar(2)
        .sum()
        .unsqueeze()
}

/// Train the UMAP model over multiple epochs
pub fn train<B: AutodiffBackend>(
    model: UMAPModel<B>,
    train_data: &Vec<Tensor<B, 2>>, // Global representations (high-dimensional data)
    device: &Device<B>,
    epochs: usize,
    batch_size: usize,
) {
    let learning_rate: f64 = 0.001;
    let config_optimizer = AdamConfig::new();
    // Initialize an optimizer, e.g., Adam with a learning rate
    let mut optim = config_optimizer.init();

    for epoch in 0..epochs {
        let n_features = train_data[0].dims()[1];
        let mut total_loss = Tensor::<B, 2>::zeros([n_features, n_features], device); // Initialize total_loss as scalar

        // Loop over batches of input data
        for (iteration, batch) in train_data.chunks(batch_size).enumerate() {
            for input_tensor in batch {
                // Forward pass to get the low-dimensional (local) representation
                let local = model.forward(input_tensor.clone());

                // Compute the UMAP loss by comparing the pairwise distances
                let loss = umap_loss(input_tensor, &local);

                // Backward pass: Compute gradients
                loss.backward();

                // Log training progress
                println!(
                    "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                    epoch,
                    iteration,
                    loss.clone().into_scalar(),
                );

                // Gradients for the current backward pass
                let grads = loss.backward();
                // Gradients linked to each parameter of the model.
                let grads = GradientsParams::from_grads(grads, &model);
                // Update model parameters using the optimizer
                optim.step(learning_rate, model.clone(), grads);

                // Accumulate the loss for this epoch
                total_loss = total_loss.add(loss.unsqueeze());
            }
        }

        // Log the average loss for the epoch
        println!(
            "Epoch {}: Loss = {:.3}",
            epoch,
            total_loss.clone().into_scalar()
        );
    }
}
