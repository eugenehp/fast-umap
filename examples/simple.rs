use fast_umap::prelude::*;
use rand::Rng;

fn main() {
    let num_samples = 100;
    let num_features = 3;
    let mut rng = rand::thread_rng();

    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.gen::<f64>()) // Random number generation for each feature
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();

    let umap = umap(data.clone());
    let reduced_dimensions_vector = umap.transform(data.clone());
    chart_vector(reduced_dimensions_vector, None);

    // let reduced_dimensions_tensor = umap.transform_to_tensor(data.clone());
    // print_tensor_with_title("reduced_dimensions", &reduced_dimensions_tensor);
    // chart_tensor(reduced_dimensions_tensor, None);
}
