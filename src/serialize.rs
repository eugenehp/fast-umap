use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use std::path::Path;
use std::fs::File;
use std::io::{Write, Read};

use crate::backend::AutodiffBackend;
use crate::model::UMAPModel;

/// Save a UMAP model's weights to a file.
///
/// # Arguments
///
/// * `model` - The UMAP model to save
/// * `path` - Path to the output file
///
/// # Example
///
/// ```ignore
/// use fast_umap::prelude::*;
/// use burn::backend::Wgpu;
///
/// let config = UmapConfig::default();
/// let umap = Umap::<Wgpu>::new(config);
/// let fitted = umap.fit(data, None);
/// fitted.save("model.umap").expect("Failed to save model");
/// ```
pub fn save_model<B: AutodiffBackend>(
    model: &UMAPModel<B::InnerBackend>,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let path_buf = path.as_ref().to_path_buf();
    model.clone().save_file(path_buf, &recorder)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Load a UMAP model's weights from a file.
///
/// # Arguments
///
/// * `model` - The UMAP model to load weights into
/// * `path` - Path to the input file
/// * `device` - The device where the model should be loaded
///
/// # Example
///
/// ```ignore
/// use fast_umap::prelude::*;
/// use burn::backend::Wgpu;
///
/// let config = UmapConfig::default();
/// let umap = Umap::<Wgpu>::new(config);
/// let fitted = umap.fit(data, None);
/// fitted.save("model.umap").expect("Failed to save model");
/// 
/// // Later...
/// let loaded_fitted = FittedUmap::<Wgpu>::load("model.umap", config, device)?;
/// ```
pub fn load_model<B: AutodiffBackend>(
    model: &mut UMAPModel<B::InnerBackend>,
    path: impl AsRef<Path>,
    device: &Device<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let path_buf = path.as_ref().to_path_buf();
    *model = model.clone().load_file(path_buf, &recorder, device)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    Ok(())
}
