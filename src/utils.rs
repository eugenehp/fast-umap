use core::f64;

use burn::{
    prelude::Backend,
    tensor::{ops::FloatTensor, Device, Tensor, TensorData, TensorPrimitive},
};
use num::{Float, FromPrimitive};
// use prettytable::{row, Table};
use rand::{distributions::uniform::SampleUniform, Rng};
use rayon::prelude::*;

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
pub fn generate_test_data<F: Float + FromPrimitive + SampleUniform>(
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
) -> Vec<F> {
    let mut rng = rand::thread_rng();

    // Define the range for random numbers (e.g., [0.0, 1.0))
    let zero = F::from_f64(0.0).unwrap(); // 0.0 as a `F` type
    let one = F::from_f64(1.0).unwrap(); // 1.0 as a `F` type

    // Generate random data for the tensor (size = num_samples * num_features)
    let data: Vec<F> = (0..num_samples * num_features)
        .map(|_| rng.gen_range(zero..one)) // Generate random number from the range [0.0, 1.0)
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
pub fn convert_vector_to_tensor<B: Backend, F: Float>(
    data: Vec<F>,
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device<B>,  // Device to place the tensor (CPU, GPU)
) -> Tensor<B, 2>
where
    F: burn::tensor::Element,
{
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
// pub fn print_tensor<B: Backend, const D: usize>(data: &Tensor<B, D>) {
//     let dims = data.dims();
//     let n_samples = match dims.len() > 0 {
//         true => dims[0],
//         false => 0,
//     };

//     let mut table = Table::new();
//     table.add_row(row!["Index", "Tensor"]);

//     for index in 0..n_samples {
//         let row = data.clone().slice([index..index + 1]);
//         let row = row.to_data().to_vec::<f32>().unwrap();
//         let row = format!("{row:?}");
//         table.add_row(row![index, format!("{:?}", row)]);
//     }

//     if dims.len() == 0 {
//         let row = data.to_data().to_vec::<f32>().unwrap();
//         let row = row.get(0).unwrap();
//         table.add_row(row![0, format!("{:?}", row)]);
//     }

//     table.printstd();
// }

/// Prints the content of a tensor with a title.
///
/// # Arguments
/// * `title` - A string title to print before displaying the tensor data.
/// * `data` - The tensor to print.
///
/// This function is similar to `print_tensor`, but with an added title to help distinguish different tensor prints.
// pub fn print_tensor_with_title<B: Backend, const D: usize>(title: &str, data: &Tensor<B, D>) {
//     println!("{title}");
//     print_tensor(data);
// }

/// Converts a 2D tensor into a `Vec<Vec<f64>>` for easier inspection or manipulation.
///
/// # Arguments
/// * `data` - A 2D tensor (samples x features) to convert into a vector of vectors.
///
/// # Returns
/// A `Vec<Vec<f64>>` where each inner `Vec<f64>` represents a row (sample) of the tensor.
///
/// This function extracts the data from a tensor and converts it into a `Vec<Vec<F>>` format. The conversion
/// assumes that the tensor is in a 2D shape and the precision is `f32` within the tensor.
pub fn convert_tensor_to_vector<B: Backend, F: Float>(data: Tensor<B, 2>) -> Vec<Vec<F>>
where
    F: burn::tensor::Element,
{
    let n_components = data.dims()[1]; // usually 2 dimensional

    // Burn Tensor only has f32 precision inside the tensors, when you export to to_data
    let data = data.to_data().to_vec::<f32>().unwrap();

    let data: Vec<Vec<f32>> = data
        .chunks(n_components)
        .map(|chunk| chunk.to_vec())
        .collect();

    let data: Vec<Vec<F>> = data
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| {
                    if v.is_nan() {
                        F::infinity() // if NaN variables, replaces them with Infinity
                    } else {
                        F::from(v).unwrap()
                    }
                })
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
pub fn normalize_data<F: Float>(data: &mut [F], num_samples: usize, num_features: usize)
where
    F: num::FromPrimitive + Send + Sync,
{
    // Parallelize the outer loop over features
    (0..num_features).into_iter().for_each(|feature_idx| {
        // Calculate mean and standard deviation for the current feature
        let (sum, sum_sq) = (0..num_samples)
            .into_par_iter()
            .fold(
                || (F::zero(), F::zero()), // Initial value for fold: (sum, sum_sq)
                |(acc_sum, acc_sum_sq), sample_idx| {
                    let value = data[sample_idx * num_features + feature_idx];
                    (acc_sum + value, acc_sum_sq + value * value)
                },
            )
            .reduce(
                || (F::zero(), F::zero()),
                |(sum1, sum_sq1), (sum2, sum_sq2)| (sum1 + sum2, sum_sq1 + sum_sq2),
            );

        let mean = sum / F::from_usize(num_samples).unwrap();
        let variance = (sum_sq / F::from_usize(num_samples).unwrap()) - (mean * mean);
        let std_dev = variance.sqrt();

        // Avoid division by zero by adding SMALL_STD_DEV
        let safe_std_dev = std_dev + F::from_f64(SMALL_STD_DEV).unwrap();

        // Normalize the feature in parallel
        (0..num_samples).into_iter().for_each(|sample_idx| {
            let idx = sample_idx * num_features + feature_idx;
            let value = data[idx];
            let normalized_value = (value - mean) / safe_std_dev;
            // Directly update the value in the `data` array
            data[idx] = normalized_value;
        });
    });
}

/// Normalizes a 1D tensor using min-max normalization.
///
/// This function performs min-max normalization on a 1D tensor, scaling its values
/// to a range between 0 and 1. If the minimum and maximum values of the tensor are
/// equal (i.e., the tensor has no variance), the original tensor is returned unmodified.
///
/// # Type Parameters
///
/// - `B`: The autodiff backend type. This should implement the `AutodiffBackend` trait,
///   which provides support for automatic differentiation.
///
/// # Arguments
///
/// - `tensor`: A 1D tensor of type `Tensor<B, 1>`, which represents the input data to be normalized.
///
/// # Returns
///
/// - A new tensor of the same type and shape as the input tensor, with normalized values.
///   If the minimum and maximum values of the tensor are equal, the original tensor is returned unchanged.
///
/// # Explanation
///
/// The normalization is performed using the following formula:
///
/// ```
/// normalized = (tensor - min) / (max - min + epsilon)
/// ```
///
/// Where:
/// - `min`: The minimum value in the tensor.
/// - `max`: The maximum value in the tensor.
/// - `epsilon`: A small constant (`1e-6`) added to prevent division by zero, ensuring numerical stability.
///
/// The function first checks if the minimum and maximum values are equal. If they are, it avoids division by zero
/// and simply returns the original tensor. If they are not equal, it applies the min-max normalization formula.
///
/// # Example
///
/// ```rust
/// let tensor = Tensor::<B, 1>::from_data(vec![1.0, 2.0, 3.0], &device);
/// let normalized_tensor = normalize_tensor(tensor);
/// ```
pub fn normalize_tensor<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
    let device = tensor.device();

    // Normalize the result using min-max normalization:
    let min_val = tensor.clone().min(); // Find the minimum value
    let max_val = tensor.clone().max(); // Find the maximum value

    // this is to prevent deleting by zero
    let offset_val = Tensor::<B, 1>::from_data(TensorData::new(vec![1e-6], [1]), &device);

    let are_equal = max_val
        .clone()
        .equal(min_val.clone())
        .to_data()
        .to_vec::<bool>()
        .unwrap();

    let are_equal = are_equal.first().unwrap();

    // Avoid division by zero by ensuring max_val != min_val
    let normalized = if !are_equal {
        (tensor - min_val.clone()) / (max_val - min_val + offset_val)
    } else {
        tensor.clone() // If all values are the same, return the original
    };

    // Return the normalized sum of the top K distances
    normalized
}

pub fn print_primitive_tensor<B: Backend>(tensor: &FloatTensor<B>, rows: usize, cols: usize) {
    let tensor: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(tensor.clone()));
    print_tensor(&tensor, rows, cols)
}

const MIN_SIZE: usize = 10;
pub fn print_tensor<B: Backend>(tensor: &Tensor<B, 2>, rows: usize, cols: usize) {
    let shape = tensor.shape().dims;
    let n = shape[0]; // Number of rows
    let d = shape[1]; // Number of columns

    let data = tensor.to_data().to_vec::<f32>().unwrap();

    let nn = n.min(rows.max(MIN_SIZE)); // Ensure nn is at least MIN_SIZE if rows > 0
    let dd = d.min(cols.max(MIN_SIZE)); // Ensure dd is at least MIN_SIZE if cols > 0

    // Print first few rows and columns with scientific notation
    for i in 0..nn {
        for j in 0..dd {
            // Print each element with scientific notation, limited to 3 decimal places
            print!("{:10.3e}", data[i * d + j]);
        }
        println!();
    }
}
