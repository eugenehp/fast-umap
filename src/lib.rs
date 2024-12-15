use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use model::{UMAPModel, UMAPModelConfigBuilder};
use train::*;
use utils::*;

use burn::tensor::{Device, Tensor};

pub mod chart;
pub mod distance;
pub mod loss;
pub mod model;
pub mod train;
pub mod utils;

pub struct UMAP<B: AutodiffBackend> {
    model: UMAPModel<B::InnerBackend>,
    device: Device<B>,
}

impl<B: AutodiffBackend> UMAP<B> {
    pub fn fit<F>(data: Vec<Vec<F>>, device: Device<B>) -> Self
    where
        F: From<f32> + From<f64> + Clone,
        f64: From<F>,
    {
        // type MyBackend = Wgpu<f32, i32>;
        // type MyAutodiffBackend = Autodiff<MyBackend>;
        // let device = burn::backend::wgpu::WgpuDevice::default();

        let batch_size = 1;
        let num_samples = data.len();
        let num_features = data[0].len();
        let output_size = 2;
        let hidden_size = 100;
        let learning_rate = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epochs = 100;
        let seed = 9999;

        B::seed(seed);

        let train_data: Vec<f64> = data.into_iter().flatten().map(|f| f64::from(f)).collect();

        let model_config = UMAPModelConfigBuilder::default()
            .input_size(num_features)
            .hidden_size(hidden_size)
            .output_size(output_size)
            .build()
            .unwrap();

        let model: UMAPModel<B> = UMAPModel::new(&model_config, &device);

        let config = TrainingConfig::builder()
            .with_epochs(epochs)
            .with_batch_size(batch_size)
            .with_learning_rate(learning_rate)
            .with_device(device.clone())
            .with_beta1(beta1)
            .with_beta2(beta2)
            .build()
            .expect("Failed to build TrainingConfig");

        // Start training with the configured parameters
        let model = train(
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

    pub fn transform_to_tensor(&self, data: Vec<Vec<f64>>) -> Tensor<B::InnerBackend, 2> {
        let num_samples = data.len();
        let num_features = data[0].len();

        let train_data: Vec<f64> = data.into_iter().flatten().map(|f| f64::from(f)).collect();

        let global = convert_vector_to_tensor(train_data, num_samples, num_features, &self.device);
        let local = self.model.forward(global);
        local
    }

    pub fn transform(&self, data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let local = self.transform_to_tensor(data);
        let result = convert_tensor_to_vector(local);

        result
    }
}

#[allow(unused)]
pub mod prelude {
    use crate::{chart, utils, UMAP};
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;

    pub use chart::{chart_tensor, chart_vector};
    pub use utils::generate_test_data;

    pub fn umap<F>(data: Vec<Vec<F>>) -> UMAP<Autodiff<Wgpu>>
    where
        F: From<f32> + From<f64> + Clone,
        f64: From<F>,
    {
        let model = UMAP::<Autodiff<Wgpu>>::fit(data, WgpuDevice::default());
        model
    }
}
