use burn::{prelude::*, tensor::cast::ToElement};

// Function to compute the Euclidean distance for any backend
pub fn euclidean_distance<B: Backend>(
    tensor_a: Tensor<B, 1, Float>,
    tensor_b: Tensor<B, 1, Float>,
) -> f64 {
    // Compute the element-wise difference between the tensors
    let diff = tensor_b - tensor_a; // Element-wise subtraction
    let squared_diff = diff.powi_scalar(2); // Element-wise squaring

    // Sum up the squared differences
    let sum_squared_diff = squared_diff.sum();

    // Compute the square root of the sum of squared differences to get the Euclidean distance
    let sum = sum_squared_diff.sum().into_scalar().to_f64();
    sum.sqrt()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_euclidean_distance_wgpu() {
        type MyBackend = burn::backend::Wgpu<f32, i32>;
        // Initialize the WgpuBackend
        let device = burn::backend::wgpu::WgpuDevice::default();

        let arr1 = [1.0_f32, 2.0, 3.0];
        let arr2 = [4.0_f32, 5.0, 6.0];

        // Create example tensors
        let tensor_a = Tensor::<MyBackend, 1>::from_floats(arr1, &device);
        let tensor_b = Tensor::<MyBackend, 1>::from_floats(arr2, &device);

        // Calculate the Euclidean distance
        let result = euclidean_distance(tensor_a, tensor_b);

        // Define the expected result (distance between [1, 2, 3] and [4, 5, 6])
        let first = ((arr2[0] - arr1[0]) as f64).powi(2);
        let third = ((arr2[1] - arr1[1]) as f64).powi(2);
        let second = ((arr2[2] - arr1[2]) as f64).powi(2);
        let expected_distance = (first + second + third).sqrt();

        // Assert that the result is within a small tolerance
        assert_eq!(
            result, expected_distance,
            "Expected distance: {}, but got: {}",
            expected_distance, result
        );
    }
}
