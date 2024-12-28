use core::f32;

use burn_jit::{tensor::JitTensor, FloatElement, IntElement, JitRuntime};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch)]
fn knn_pairwise_euclidean_kernel<F: Float, I: Int>(
    input: &Tensor<F>,
    k: I,
    min_dist: F,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
) {
    let k = u32::cast_from(k);

    if ABSOLUTE_POS >= input.len() {
        return;
    }

    let n = input.shape(0);
    let d = input.shape(1);
    let i = ABSOLUTE_POS / input.stride(0);

    // Compute pairwise Euclidean distances for the i-th point
    for j in 0..n {
        let mut dist = F::new(0.0);
        for l in 0..d {
            let diff = input[i * input.stride(0) + l * input.stride(1)]
                - input[j * input.stride(0) + l * input.stride(1)];
            dist = dist + diff * diff;
        }
        distances[i * n + j] = dist;
        indices[i * n + j] = I::cast_from(j);
    }

    // Sort distances and indices in-place to get the k nearest neighbors
    let mut dist_array = Array::new(n); // Fixed-size array for distances
    let mut index_array = Array::new(n); // Fixed-size array for indices

    // Copy distances and indices into the temporary arrays
    for j in 0..n {
        dist_array[j] = distances[i * n + j];
        index_array[j] = indices[i * n + j];
    }

    // Perform an in-place sort on the distances and corresponding indices (Selection Sort)
    for j in 0..n {
        let mut min_idx = j;
        for l in j + 1..n {
            if dist_array[l] < dist_array[min_idx] {
                min_idx = l;
            }
        }
        if min_idx != j {
            // Manually swap the elements at positions `j` and `min_idx`
            let temp_dist = dist_array[j];
            let temp_index = index_array[j];
            dist_array[j] = dist_array[min_idx];
            index_array[j] = index_array[min_idx];
            dist_array[min_idx] = temp_dist;
            index_array[min_idx] = temp_index;
        }
    }

    // Update the distances and indices for the k nearest neighbors
    for j in 0..k {
        if dist_array[j] >= min_dist {
            distances[i * k + j] = dist_array[j];
            indices[i * k + j] = index_array[j];
        } else {
            distances[i * k + j] = F::cast_from(f32::INFINITY);
            indices[i * k + j] = I::cast_from(k);
        }
    }
}

pub(crate) fn knn_pairwise_euclidean_launch<R: JitRuntime, F: FloatElement, I: IntElement>(
    input: JitTensor<R, F>,
    k: I,
    min_dist: F,
    indices: JitTensor<R, I>,
    distances: JitTensor<R, F>,
) -> (JitTensor<R, I>, JitTensor<R, F>) {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    knn_pairwise_euclidean_kernel::launch::<F, I, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        ScalarArg::new(k),
        ScalarArg::new(min_dist),
        indices.as_tensor_arg(1),
        distances.as_tensor_arg(1),
    );

    (indices, distances)
}

#[cube(launch)]
fn knn_backward_kernel<F: Float, I: Int>(
    input: &Tensor<F>,
    grad_output: &Tensor<F>,
    indices: &Tensor<I>,
    distances: &Tensor<F>,
    grad_input: &mut Tensor<F>,
) {
    if ABSOLUTE_POS >= input.len() {
        return;
    }

    let n = input.shape(0);
    let d = input.shape(1);
    let i = ABSOLUTE_POS / input.stride(0);

    for j in 0..n {
        let idx = indices[i * n + j];
        if idx != I::cast_from(i32::MAX) {
            let dist = distances[i * n + j];
            for l in 0..d {
                let input_idx = idx * I::cast_from(input.stride(0))
                    + I::cast_from(l) * I::cast_from(input.stride(1));
                let input_idx = u32::cast_from(input_idx);

                let diff = input[i * input.stride(0) + l * input.stride(1)] - input[input_idx];
                grad_input[i * input.stride(0) + l * input.stride(1)] +=
                    grad_output[i * input.stride(0)] * diff / dist;
            }
        }
    }
}

pub(crate) fn knn_backward_launch<R: JitRuntime, F: FloatElement, I: IntElement>(
    input: JitTensor<R, F>,
    grad_output: JitTensor<R, F>,
    indices: JitTensor<R, I>,
    distances: JitTensor<R, F>,
    grad_input: JitTensor<R, F>,
) -> JitTensor<R, F> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    knn_backward_kernel::launch::<F, I, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        grad_output.as_tensor_arg(1),
        indices.as_tensor_arg(1),
        distances.as_tensor_arg(1),
        grad_input.as_tensor_arg(1),
    );

    grad_input
}
