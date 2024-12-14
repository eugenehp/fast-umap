use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Device, Tensor},
};

use crate::{loss::umap_loss, model::UMAPModel};

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
