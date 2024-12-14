use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use prettytable::{row, Table};
use rand::Rng;

// Define the function to generate random data in the format `<Tensor<B, 2>`.
pub fn load_test_data<B: Backend>(
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device<B>,  // Device to place the tensor (CPU, GPU)
) -> Tensor<B, 2> {
    let mut rng = rand::thread_rng();

    // Generate random data for the tensor (size = num_features)
    let data: Vec<_> = (0..num_samples * num_features)
        .map(|_| rng.gen::<f64>()) // Random number generation for each feature
        .collect();

    let tensor_data = TensorData::new(data, [num_samples, num_features]);

    // Create a Tensor with the shape [1, num_features] (1 row, num_features columns)
    let tensor = Tensor::<B, 2>::from_data(tensor_data, device);

    tensor
}
pub fn print_tensor<B: Backend, const D: usize>(data: &Tensor<B, D>) {
    let n_samples = data.dims()[0];
    // let _n_features = data.dims()[1];

    let mut table = Table::new();
    table.add_row(row!["Index", "Tensor"]);

    for index in 0..n_samples {
        let row = data.clone().slice([index..index + 1]);
        let row = row.to_data().to_vec::<f32>().unwrap();
        let row = format!("{row:?}");
        table.add_row(row![index, format!("{:?}", row)]);
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
