use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use prettytable::{row, Table};
use rand::Rng;

// Define the function to generate random data in the format `Vec<Tensor<B, 2>>`.
pub fn load_test_data<B: Backend>(
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device<B>,  // Device to place the tensor (CPU, GPU)
) -> Vec<Tensor<B, 2>> {
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        // Generate random data for the tensor (size = num_features)
        let data: Vec<_> = (0..num_features)
            .map(|_| rng.gen::<f64>()) // Random number generation for each feature
            .collect();

        let tensor_data = TensorData::new(data, [1, num_features]);

        // Create a Tensor with the shape [1, num_features] (1 row, num_features columns)
        let tensor = Tensor::<B, 2>::from_data(tensor_data, device);

        result.push(tensor);
    }

    result
}

pub fn print_tensor<B: Backend>(title: Option<&str>, tensors: &Vec<Tensor<B, 2>>) {
    if let Some(title) = title {
        println!("{title}")
    }
    let mut table = Table::new();
    table.add_row(row!["Index", "Tensor"]);
    tensors
        .iter()
        .enumerate()
        .map(|(index, t)| {
            let row = t.to_data().to_vec::<f32>().unwrap();
            table.add_row(row![index, format!("{:?}", row)]);
        })
        .for_each(drop);
    table.printstd();
}
