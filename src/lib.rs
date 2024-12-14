use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use model::{UMAPModel, UMAPModelConfigBuilder};
use train::*;
use utils::*;

use burn::{
    backend::{Autodiff, Wgpu},
    tensor::Device,
};

pub mod chart;
pub mod distance;
pub mod loss;
pub mod model;
pub mod train;
pub mod utils;

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

pub struct UMAP<B: AutodiffBackend> {
    model: UMAPModel<B::InnerBackend>,
    device: Device<B>,
}

impl<B: AutodiffBackend> UMAP<B> {
    pub fn fit<F>(data: Vec<Vec<F>>) -> Self
    where
        F: From<f32> + From<f64> + Clone,
        f64: From<F>,
        B: AutodiffBackend,
    {
        let device = burn::backend::wgpu::WgpuDevice::default();

        let batch_size = 1;
        let num_samples = data.len();
        let num_features = data[0].len();
        let output_size = 2;
        let hidden_size = 100;
        let learning_rate = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epochs = 400;
        let seed = 9999;

        B::seed(seed);

        let train_data: Vec<f64> = data.into_iter().flatten().map(|f| f64::from(f)).collect();

        let model_config = UMAPModelConfigBuilder::default()
            .input_size(num_features)
            .hidden_size(hidden_size)
            .output_size(output_size)
            .build()
            .unwrap();

        let model: UMAPModel<_> = UMAPModel::new(&model_config, &device);

        let config = TrainingConfig::<_>::builder()
            .epochs(epochs)
            .batch_size(batch_size)
            .learning_rate(learning_rate)
            .device(device.clone())
            .beta1(beta1)
            .beta2(beta2)
            .build()
            .expect("Failed to build TrainingConfig");

        // Start training with the configured parameters
        let model = train::<MyAutodiffBackend>(
            model,
            num_samples,
            num_features,
            train_data.clone(),
            &config,
        );

        // let model: UMAPModel<<B as AutodiffBackend>::InnerBackend> = model.valid();
        let model: UMAPModel<_> = model.valid();

        let umap = UMAP { model, device };

        umap
    }

    pub fn transform<F>(&self, data: Vec<Vec<F>>) -> Vec<Vec<F>>
    where
        F: From<f64> + From<f32>,
        f64: From<F>,
    {
        let num_samples = data.len();
        let num_features = data[0].len();

        let train_data: Vec<f64> = data.into_iter().flatten().map(|f| f64::from(f)).collect();

        let global = convert_vector_to_tensor(train_data, num_samples, num_features, &self.device);
        let local = self.model.forward(global);
        let result = convert_tensor_to_vector(local);

        result
    }
}
