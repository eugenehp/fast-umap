use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use plotters::prelude::*;
use prettytable::{row, Table};
use rand::Rng;

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

// Define the function to generate random data in the format `<Tensor<B, 2>`.
pub fn convert_vector_to_tensor<B: Backend>(
    data: Vec<f64>,
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device<B>,  // Device to place the tensor (CPU, GPU)
) -> Tensor<B, 2> {
    let tensor_data = TensorData::new(data, [num_samples, num_features]);

    // Create a Tensor with the shape [1, num_features] (1 row, num_features columns)
    let tensor = Tensor::<B, 2>::from_data(tensor_data, device);

    tensor
}
pub fn print_tensor<B: Backend, const D: usize>(data: &Tensor<B, D>) {
    let dims = data.dims();
    let n_samples = match dims.len() > 0 {
        true => dims[0],
        false => 0,
    };
    // let _n_features = dims[1];

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

pub fn print_tensor_with_title<B: Backend, const D: usize>(
    title: Option<&str>,
    data: &Tensor<B, D>,
) {
    if let Some(title) = title {
        println!("{title}")
    }

    print_tensor(data);
}

pub fn chart<B: Backend>(data: Tensor<B, 2>) {
    let n_components = 2;
    let data = data.to_data().to_vec::<f32>().unwrap();
    let data: Vec<Vec<f32>> = data
        .chunks(n_components)
        .map(|chunk| chunk.to_vec())
        .collect();

    let (width, height) = (1000, 1000);

    // Calculate the min and max values for both X and Y axes
    let (min_x, max_x) = data
        .iter()
        .map(|point| point[0])
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
            (min.min(x), max.max(x))
        });
    let (min_y, max_y) = data
        .iter()
        .map(|point| point[1])
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), y| {
            (min.min(y), max.max(y))
        });

    // Scale the data points to fit within the drawing area
    let scaled_data: Vec<(f32, f32)> = data
        .iter()
        .map(|point| {
            let x_scaled = (point[0] - min_x) / (max_x - min_x) * width as f32;
            let y_scaled = height as f32 - (point[1] - min_y) / (max_y - min_y) * height as f32; // Invert Y to fit canvas
            (x_scaled, y_scaled)
        })
        .collect();

    // Add padding around the min and max values to ensure the points fit nicely in the chart
    let padding = 100.0; // Adjust the padding as needed
    let x_range = min_x - padding..max_x + padding;
    let y_range = min_y - padding..max_y + padding;

    // Create the drawing area (output to a PNG file)
    let root = BitMapBackend::new("scatter_plot.png", (width, height)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Create a chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Scatter Plot", ("sans-serif", 30))
        .build_cartesian_2d(x_range, y_range) // Dynamic range based on data min/max
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(10) // Set 10 labels on the X-axis
        .y_labels(10) // Set 10 labels on the Y-axis
        .x_desc("X Axis")
        .y_desc("Y Axis")
        .draw()
        .unwrap();

    // Draw the scatter plot
    chart
        .draw_series(scaled_data.into_iter().map(|coords| {
            Circle::new(
                coords,
                5,
                ShapeStyle {
                    filled: true,
                    color: RED.to_rgba(),
                    stroke_width: 0,
                },
            )
        }))
        .unwrap();

    // Save the chart to a file
    root.present().unwrap();
}
