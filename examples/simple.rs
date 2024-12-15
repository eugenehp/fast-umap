use fast_umap::{chart::chart_vector, umap, utils::*};

fn main() {
    let num_samples = 100;
    let num_features = 3;

    let data: Vec<Vec<f64>> = generate_test_data(num_samples, num_features)
        .chunks_exact(2)
        .map(|chunk| chunk.to_vec())
        .collect();

    let umap = umap(data.clone());
    let reduced_dimensions = umap.transform(data);

    // print_tensor_with_title(Some("local"), &local);
    chart_vector(reduced_dimensions, None);
}
