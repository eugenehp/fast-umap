use crate::utils::*;
use burn::prelude::*;
use plotters::prelude::*;

const CAPTION: &str = "fast-umap";
const PATH: &str = "plot.png";

#[derive(Debug, Clone)]
pub struct ChartConfig {
    pub caption: String,
    pub path: String,
    pub width: u32,
    pub height: u32,
}

impl ChartConfig {
    // Builder pattern for configuring the chart
    pub fn builder() -> ChartConfigBuilder {
        ChartConfigBuilder {
            caption: Some(CAPTION.to_string()),
            path: Some(PATH.to_string()),
            width: Some(1000),
            height: Some(1000),
        }
    }
}

impl Default for ChartConfig {
    fn default() -> Self {
        ChartConfig {
            caption: CAPTION.to_string(),
            path: PATH.to_string(),
            width: 1000,
            height: 1000,
        }
    }
}

pub struct ChartConfigBuilder {
    caption: Option<String>,
    path: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
}

impl ChartConfigBuilder {
    // Set caption
    pub fn caption(mut self, caption: &str) -> Self {
        self.caption = Some(caption.to_string());
        self
    }

    // Set path
    pub fn path(mut self, path: &str) -> Self {
        self.path = Some(path.to_string());
        self
    }

    // Set width
    pub fn width(mut self, width: u32) -> Self {
        self.width = Some(width);
        self
    }

    // Set height
    pub fn height(mut self, height: u32) -> Self {
        self.height = Some(height);
        self
    }

    // Build the final config
    pub fn build(self) -> ChartConfig {
        ChartConfig {
            caption: self.caption.unwrap_or_else(|| CAPTION.to_string()),
            path: self.path.unwrap_or_else(|| PATH.to_string()),
            width: self.width.unwrap_or(1000),
            height: self.height.unwrap_or(1000),
        }
    }
}

type Float = f64;

pub fn chart_tensor<B: Backend>(data: Tensor<B, 2>, config: Option<ChartConfig>) {
    let data: Vec<Vec<Float>> = convert_tensor_to_vector(data);
    chart_vector(data, config);
}

pub fn chart_vector(data: Vec<Vec<Float>>, config: Option<ChartConfig>) {
    let config = config.unwrap_or(ChartConfig::default());

    // Create a drawing area with a size of 800x600 pixels
    let root = BitMapBackend::new(&config.path, (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Define the range for x and y axes (include negative values)
    let min_x = data
        .iter()
        .flat_map(|v| v.iter().step_by(2)) // x values are at even indices
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as Float;

    let max_x = data
        .iter()
        .flat_map(|v| v.iter().step_by(2)) // x values are at even indices
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as Float;

    let min_y = data
        .iter()
        .flat_map(|v| v.iter().skip(1).step_by(2)) // y values are at odd indices
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as Float;

    let max_y = data
        .iter()
        .flat_map(|v| v.iter().skip(1).step_by(2)) // y values are at odd indices
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as Float;

    // Create a chart builder with specific size and axis ranges
    let mut chart = ChartBuilder::on(&root)
        .caption(config.caption, ("sans-serif", 30))
        .margin(40)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)
        .unwrap();

    // Draw the x and y axis
    chart
        .configure_mesh()
        .x_desc("X Axis")
        .y_desc("Y Axis")
        .x_labels(10)
        .y_labels(10)
        .draw()
        .unwrap();

    // Plot each vector in the Vec<Vec<F>> as a series of dots
    chart
        .draw_series(data.iter().map(|values| {
            Circle::new(
                (values[0], values[1]),
                5,
                ShapeStyle {
                    color: RED.to_rgba(),
                    filled: true,
                    stroke_width: 1,
                },
            )
        }))
        .unwrap()
        .label("UMAP")
        .legend(move |(x, y)| {
            Circle::new(
                (x, y),
                5,
                ShapeStyle {
                    color: RED.to_rgba(),
                    filled: true,
                    stroke_width: 1,
                },
            )
        });

    // Draw the legend
    chart.configure_mesh().draw().unwrap();

    // Save the chart to a file
    root.present().unwrap();
}
