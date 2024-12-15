use fast_umap::prelude::*;

fn main() {
    let num_samples = 100;
    let num_features = 3;

    let data: Vec<Vec<f64>> = generate_test_data(num_samples, num_features)
        .chunks_exact(2)
        .map(|chunk| chunk.to_vec())
        .collect();

    let umap = umap(data.clone());

    let reduced_dimensions_vector = umap.transform(data.clone());
    chart_vector(reduced_dimensions_vector, None);

    // let reduced_dimensions_tensor = umap.transform_to_tensor(data.clone());
    // print_tensor_with_title("reduced_dimensions", &reduced_dimensions_tensor);
    // chart_tensor(reduced_dimensions_tensor, None);
}
