use fast_umap::{utils::*, UMAP};

fn main() {
    let num_samples = 100;
    let num_features = 3;

    let data: Vec<Vec<f64>> = generate_test_data(num_samples, num_features)
        .chunks_exact(2)
        .map(|chunk| chunk.to_vec())
        .collect();

    let umap = UMAP::fit(data.clone());
    let reduced_dimensions = umap.transform(data);

    // print_tensor_with_title(Some("local"), &local);
    chart(reduced_dimensions, None);
}
