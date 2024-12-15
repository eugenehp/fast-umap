use fast_umap::prelude::*;
use rand::Rng;

fn main() {
    // Number of samples in the dataset
    let num_samples = 100;

    // Number of features (dimensions) for each sample
    let num_features = 3;

    // Create a random number generator for generating random values
    let mut rng = rand::thread_rng();

    // Generate a dataset of random values with `num_samples` rows and `num_features` columns
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.gen::<f64>()) // Random number generation for each feature
        .collect::<Vec<f64>>() // Collect all random values into a vector
        .chunks_exact(num_features) // Chunk the vector into rows of length `num_features`
        .map(|chunk| chunk.to_vec()) // Convert each chunk into a Vec<f64>
        .collect(); // Collect the rows into a vector of vectors

    // Fit the UMAP model to the data and reduce the data to a lower-dimensional space (default: 2D)
    let umap = umap(data.clone());

    // Transform the data using the trained UMAP model to reduce its dimensions
    let reduced_dimensions_vector = umap.transform(data.clone());

    // Visualize the reduced dimensions as a vector
    chart_vector(reduced_dimensions_vector, None);

    // Optionally, you can also visualize the reduced dimensions as a tensor
    // let reduced_dimensions_tensor = umap.transform_to_tensor(data.clone());
    // print_tensor_with_title("reduced_dimensions", &reduced_dimensions_tensor);
    // chart_tensor(reduced_dimensions_tensor, None);
}
