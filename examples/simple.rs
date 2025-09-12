use cubecl::wgpu::WgpuRuntime;
use fast_umap::prelude::*;
use rand::Rng;

fn main() {
    // Number of samples in the dataset
    let num_samples = 100;

    // Number of features (dimensions) for each sample
    let num_features = 3;

    // Create a random number generator for generating random values
    let mut rng = rand::rng();

    // Generate a dataset of random values with `num_samples` rows and `num_features` columns
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>()) // Random number generation for each feature
        .collect::<Vec<f64>>() // Collect all random values into a vector
        .chunks_exact(num_features) // Chunk the vector into rows of length `num_features`
        .map(|chunk| chunk.to_vec()) // Convert each chunk into a Vec<f64>
        .collect(); // Collect the rows into a vector of vectors

    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    // Fit the UMAP model to the data and reduce the data to a lower-dimensional space (default: 2D)
    let umap: fast_umap::UMAP<MyAutodiffBackend> = umap(data.clone());
    // let umap = umap_size(data.clone(), 3); // where 3 is the output size of projected dimensions

    // Transform the data using the trained UMAP model to reduce its dimensions
    let reduced_dimensions_vector = umap.transform(data.clone());

    // Visualize the reduced dimensions as a vector, plots only 2D for now
    chart_vector(reduced_dimensions_vector, None, None);

    // Optionally, you can also visualize the reduced dimensions as a tensor
    // let reduced_dimensions_tensor = umap.transform_to_tensor(data.clone());
    // print_tensor_with_title("reduced_dimensions", &reduced_dimensions_tensor);
    // chart_tensor(reduced_dimensions_tensor, None);
}
