use core::f64;

use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use prettytable::{row, Table};
use rand::Rng;

/// Generates random test data with the given number of samples and features.
///
/// # Arguments
/// * `num_samples` - The number of samples to generate.
/// * `num_features` - The number of features (columns) per sample.
///
/// # Returns
/// A `Vec<f64>` containing the randomly generated data.
///
/// This function uses the `rand` crate to generate a flat vector of random floating-point values.
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

/// Converts a vector of `f64` values into a `Tensor<B, 2>` for the specified backend.
///
/// # Arguments
/// * `data` - A vector of `f64` values representing the data to convert.
/// * `num_samples` - The number of samples (rows).
/// * `num_features` - The number of features (columns).
/// * `device` - The device to place the tensor on (e.g., CPU, GPU).
///
/// # Returns
/// A `Tensor<B, 2>` containing the data arranged as samples x features.
///
/// This function uses the `TensorData` struct to create a tensor from the given data, then places it
/// on the specified device (`CPU` or `GPU`).
pub fn convert_vector_to_tensor<B: Backend>(
    data: Vec<f64>,
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device<B>,  // Device to place the tensor (CPU, GPU)
) -> Tensor<B, 2> {
    let tensor_data = TensorData::new(data, [num_samples, num_features]);
    Tensor::<B, 2>::from_data(tensor_data, device)
}

/// Prints the content of a tensor in a table format with index and tensor values.
///
/// # Arguments
/// * `data` - The tensor to print, with a generic backend and dimensionality `D`.
///
/// This function prints the tensor's data in a table with each row corresponding to one sample.
/// The tensor data is printed in a format that makes it easy to inspect.
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

/// Prints the content of a tensor with a title.
///
/// # Arguments
/// * `title` - A string title to print before displaying the tensor data.
/// * `data` - The tensor to print.
///
/// This function is similar to `print_tensor`, but with an added title to help distinguish different tensor prints.
pub fn print_tensor_with_title<B: Backend, const D: usize>(title: &str, data: &Tensor<B, D>) {
    println!("{title}");
    print_tensor(data);
}

/// Converts a 2D tensor into a `Vec<Vec<f64>>` for easier inspection or manipulation.
///
/// # Arguments
/// * `data` - A 2D tensor (samples x features) to convert into a vector of vectors.
///
/// # Returns
/// A `Vec<Vec<f64>>` where each inner `Vec<f64>` represents a row (sample) of the tensor.
///
/// This function extracts the data from a tensor and converts it into a `Vec<Vec<f64>>` format. The conversion
/// assumes that the tensor is in a 2D shape and the precision is `f32` within the tensor.
pub fn convert_tensor_to_vector<B: Backend>(data: Tensor<B, 2>) -> Vec<Vec<f64>> {
    let n_components = data.dims()[1]; // usually 2 dimensional

    // Burn Tensor only has f32 precision inside the tensors, when you export to to_data
    let data = data.to_data().to_vec::<f32>().unwrap();

    let data: Vec<Vec<f64>> = data
        .chunks(n_components)
        .map(|chunk| {
            chunk
                .to_vec()
                .into_iter()
                .map(|value| f64::from(value))
                .collect()
        })
        .collect();
    data
}

/// Formats a `Duration` into a human-readable string in hours, minutes, and seconds format.
///
/// # Arguments
/// * `duration` - The duration to format.
///
/// # Returns
/// A formatted string representing the duration in the format `HH:MM:SS`.
///
/// This function is useful for displaying elapsed times or durations in a more readable format.
pub fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

// a constant used to offset division by zero in the normalization function below
const SMALL_STD_DEV: f64 = 1e-6;

/// Normalizes the given dataset by centering each feature (column) to have mean 0
/// and standard deviation 1.
///
/// # Arguments
/// * `data` - A mutable slice representing the dataset, where each row is a sample,
///   and each column represents a feature. The data is assumed to be stored in
///   row-major order (i.e., `data[sample_idx * num_features + feature_idx]`).
/// * `num_samples` - The number of samples (rows) in the dataset.
/// * `num_features` - The number of features (columns) in the dataset.
///
/// # Example
/// ```
/// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let num_samples = 2;
/// let num_features = 3;
/// normalize_data(&mut data, num_samples, num_features);
/// ```
/// The function will normalize each feature (column) across all samples (rows).
///
/// # Note
/// This function assumes that the dataset has at least one sample and one feature.
/// The data is normalized in-place, meaning the original data is modified directly.
pub fn normalize_data(data: &mut [f64], num_samples: usize, num_features: usize) {
    for feature_idx in 0..num_features {
        // Calculate mean and standard deviation for the current feature
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for sample_idx in 0..num_samples {
            let value = data[sample_idx * num_features + feature_idx];
            sum += value;
            sum_sq += value * value;
        }

        let mean = sum / num_samples as f64;
        let variance = (sum_sq / num_samples as f64) - (mean * mean);
        let std_dev = variance.sqrt();

        // Normalize the feature
        for sample_idx in 0..num_samples {
            let value = data[sample_idx * num_features + feature_idx];
            let normalized_value = (value - mean) / (std_dev + SMALL_STD_DEV);
            data[sample_idx * num_features + feature_idx] = normalized_value;
        }
    }
}
