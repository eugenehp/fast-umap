use fast_umap::prelude::*;
use fast_umap::{Umap, FittedUmap};
use cubecl::wgpu::WgpuRuntime;
use std::path::PathBuf;

type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

fn generate_test_data(num_samples: usize, num_features: usize) -> Vec<Vec<f64>> {
    (0..num_samples).map(|i| {
        (0..num_features).map(|j| {
            let base = (i as f64) / (num_samples as f64) * 10.0;
            base + ((j as f64) / (num_features as f64)) + (rand::random::<f64>() - 0.5) * 0.1
        }).collect()
    }).collect()
}

#[test]
fn test_single_sample_transform() -> Result<(), Box<dyn std::error::Error>> {
    // Test transforming single sample
    let data = generate_test_data(30, 15);

    let config = UmapConfig {
        n_components: 3,
        hidden_sizes: vec![10],
        graph: GraphParams {
            n_neighbors: 3,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 15,
            ..Default::default()
        },
        ..Default::default()
    };

    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data, None);
    
    let model_path = PathBuf::from("/tmp/test_edge_single.bin");
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    
    fitted.save(&model_path)?;
    
    let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
        &model_path,
        config,
        15,
        Default::default(),
    )?;

    // Test with single sample
    let single_data: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];
    let single_embedding = loaded_fitted.transform(single_data);
    
    assert_eq!(single_embedding.len(), 1);
    assert_eq!(single_embedding[0].len(), 3);
    
    // Verify embedding is valid
    for &coord in &single_embedding[0] {
        assert!(coord.is_finite());
        assert!(!coord.is_nan());
    }

    // Clean up
    std::fs::remove_file(&model_path)?;

    Ok(())
}

#[test]
fn test_large_batch_transform() -> Result<(), Box<dyn std::error::Error>> {
    // Test transforming large batch
    let data = generate_test_data(100, 20);

    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![15, 10],
        graph: GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 25,
            ..Default::default()
        },
        ..Default::default()
    };

    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data, None);
    
    let model_path = PathBuf::from("/tmp/test_edge_large.bin");
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    
    fitted.save(&model_path)?;
    
    let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
        &model_path,
        config,
        20,
        Default::default(),
    )?;

    // Test with large batch
    let large_batch: Vec<Vec<f64>> = generate_test_data(500, 20);
    let large_embedding = loaded_fitted.transform(large_batch);
    
    assert_eq!(large_embedding.len(), 500);
    
    // Verify all embeddings are valid
    for point in &large_embedding {
        assert_eq!(point.len(), 2);
        for &coord in point {
            assert!(coord.is_finite());
            assert!(!coord.is_nan());
        }
    }

    // Clean up
    std::fs::remove_file(&model_path)?;

    Ok(())
}

#[test]
fn test_different_output_dimensions() -> Result<(), Box<dyn std::error::Error>> {
    // Test models with different output dimensions
    let data = generate_test_data(80, 12);

    for n_components in [1, 2, 3, 5] {
        let config = UmapConfig {
            n_components,
            hidden_sizes: vec![10],
            graph: GraphParams {
                n_neighbors: 4,
                ..Default::default()
            },
            optimization: OptimizationParams {
                n_epochs: 15,
                ..Default::default()
            },
            ..Default::default()
        };

        let umap = Umap::<MyAutodiffBackend>::new(config.clone());
        let fitted = umap.fit(data.clone(), None);
        
        let model_path = PathBuf::from(format!("/tmp/test_edge_{}d.bin", n_components));
        if model_path.exists() {
            std::fs::remove_file(&model_path)?;
        }
        
        fitted.save(&model_path)?;
        
        let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
            &model_path,
            config,
            12,
            Default::default(),
        )?;

        // Test transformation
        let test_data = vec![data[0].clone()];
        let embedding = loaded_fitted.transform(test_data);
        
        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), n_components);
        
        // Verify embedding is valid
        for &coord in &embedding[0] {
            assert!(coord.is_finite());
            assert!(!coord.is_nan());
        }

        // Clean up
        std::fs::remove_file(&model_path)?;
    }

    Ok(())
}

#[test]
fn test_overwrite_existing_file() -> Result<(), Box<dyn std::error::Error>> {
    // Test that saving overwrites existing files
    let data1 = generate_test_data(50, 10);
    let data2 = generate_test_data(60, 10); // Different data

    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![8],
        graph: GraphParams {
            n_neighbors: 3,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 15,
            ..Default::default()
        },
        ..Default::default()
    };

    let model_path = PathBuf::from("/tmp/test_edge_overwrite.bin");
    
    // Save first model
    let umap1 = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted1 = umap1.fit(data1, None);
    fitted1.save(&model_path)?;
    
    let file_size_1 = std::fs::metadata(&model_path)?.len();
    assert!(file_size_1 > 0);

    // Save second model (should overwrite)
    let umap2 = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted2 = umap2.fit(data2, None);
    fitted2.save(&model_path)?;
    
    let file_size_2 = std::fs::metadata(&model_path)?.len();
    assert!(file_size_2 > 0);

    // Load the second model
    let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
        &model_path,
        config,
        10,
        Default::default(),
    )?;

    // Test transformation
    let test_data = vec![vec![1.0; 10]]; // Neutral test data
    let embedding = loaded_fitted.transform(test_data);
    
    assert_eq!(embedding.len(), 1);
    assert_eq!(embedding[0].len(), 2);

    // Clean up
    std::fs::remove_file(&model_path)?;

    Ok(())
}