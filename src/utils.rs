use core::f64;

use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use prettytable::{row, Table};
use rand::Rng;

pub fn generate_test_data(
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
) -> Vec<f64> {
    let mut rng = rand::thread_rng();

    // Generate random data for the tensor (size = num_features)
    let data: Vec<f64> = (0..num_samples * num_features)
        .map(|_| rng.gen::<f64>()) // Random number generation for each feature
        .collect();

    data
}

// Define the function to generate random data in the format `<Tensor<B, 2>`.
pub fn convert_vector_to_tensor<B: Backend>(
    data: Vec<f64>,
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device<B>,  // Device to place the tensor (CPU, GPU)
) -> Tensor<B, 2> {
    let tensor_data = TensorData::new(data, [num_samples, num_features]);
    Tensor::<B, 2>::from_data(tensor_data, device)
}
pub fn print_tensor<B: Backend, const D: usize>(data: &Tensor<B, D>) {
    let dims = data.dims();
    let n_samples = match dims.len() > 0 {
        true => dims[0],
        false => 0,
    };

    let mut table = Table::new();
    table.add_row(row!["Index", "Tensor"]);

    for index in 0..n_samples {
        let row = data.clone().slice([index..index + 1]);
        let row = row.to_data().to_vec::<f32>().unwrap();
        let row = format!("{row:?}");
        table.add_row(row![index, format!("{:?}", row)]);
    }

    if dims.len() == 0 {
        let row = data.to_data().to_vec::<f32>().unwrap();
        let row = row.get(0).unwrap();
        table.add_row(row![0, format!("{:?}", row)]);
    }

    table.printstd();
}

pub fn print_tensor_with_title<B: Backend, const D: usize>(
    title: Option<&str>,
    data: &Tensor<B, D>,
) {
    if let Some(title) = title {
        println!("{title}")
    }

    print_tensor(data);
}

pub fn convert_tensor_to_vector<B: Backend, F>(data: Tensor<B, 2>) -> Vec<Vec<F>>
where
    F: From<f32> + From<f64>, // Ensure that F can be constructed from f32
{
    let n_components = data.dims()[1]; // usually 2 dimensional
    let data = data.to_data().to_vec::<f32>().unwrap();
    let data: Vec<Vec<F>> = data
        .chunks(n_components)
        .map(|chunk| {
            chunk
                .to_vec()
                .into_iter()
                .map(|value| F::from(value))
                .collect()
        })
        .collect();
    data
}
