#![cfg(feature = "gpu")]

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
fn test_save_load_model() -> Result<(), Box<dyn std::error::Error>> {
    // Generate some test data
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
        vec![4.0, 5.0, 6.0, 7.0],
        vec![5.0, 6.0, 7.0, 8.0],
    ];

    // Configure and fit UMAP
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![10],
        graph: GraphParams {
            n_neighbors: 3,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 50, // More epochs for stability
            ..Default::default()
        },
        ..Default::default()
    };

    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data.clone(), None);
    
    // Save the model to a temporary file
    let model_path = PathBuf::from("/tmp/test_umap_model.bin");
    
    // Clean up any existing file
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    
    // Debug: Check if path is valid
    println!("Saving to: {:?}", model_path);
    assert!(model_path.parent().unwrap().exists(), "Parent directory should exist");
    
    let save_result = fitted.save(&model_path);
    if let Err(e) = save_result {
        panic!("Save failed with error: {:?}", e);
    }
    
    // Verify file exists and has reasonable size
    assert!(model_path.exists(), "File should exist after save");
    let file_metadata = std::fs::metadata(&model_path)?;
    println!("File size: {} bytes", file_metadata.len());
    assert!(file_metadata.len() > 0, "File should not be empty");
    assert!(file_metadata.len() < 100_000, "File should be less than 100KB: {} bytes", file_metadata.len());
    
    // Load the model
    let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
        &model_path,
        config,
        4, // Input size
        Default::default(),
    )?;

    // Test that the loaded model can transform data
    let new_data: Vec<Vec<f64>> = vec![
        vec![1.5, 2.5, 3.5, 4.5],
        vec![2.5, 3.5, 4.5, 5.5],
    ];

    let embedding = loaded_fitted.transform(new_data);
    
    // Basic checks
    assert_eq!(embedding.len(), 2); // Two new data points
    assert_eq!(embedding[0].len(), 2); // 2D embedding
    assert_eq!(embedding[1].len(), 2);
    
    // Check that the embedding values are reasonable (not NaN, not infinite)
    for point in &embedding {
        for &coord in point {
            assert!(coord.is_finite());
            assert!(!coord.is_nan());
        }
    }

    Ok(())
}

#[test]
fn test_load_with_sample() -> Result<(), Box<dyn std::error::Error>> {
    // Generate test data
    let data = generate_test_data(100, 20);
    let num_features = data[0].len();

    // Configure and fit UMAP
    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![15, 10],
        graph: GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 30,
            ..Default::default()
        },
        ..Default::default()
    };

    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data.clone(), None);
    
    // Save the model
    let model_path = PathBuf::from("/tmp/test_umap_model_2.bin");
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    fitted.save(&model_path)?;

    // Test load_with_sample method
    let sample_data = vec![0.0; num_features]; // Sample with correct dimensionality
    let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load_with_sample(
        &model_path,
        config,
        sample_data,
        Default::default(),
    )?;

    // Test transformation
    let test_data = vec![data[0].clone(), data[1].clone()];
    let embedding = loaded_fitted.transform(test_data);
    
    assert_eq!(embedding.len(), 2);
    assert_eq!(embedding[0].len(), 2);
    
    // Verify embeddings are valid
    for point in &embedding {
        for &coord in point {
            assert!(coord.is_finite());
            assert!(!coord.is_nan());
        }
    }

    Ok(())
}

#[test]
fn test_multiple_save_load_cycles() -> Result<(), Box<dyn std::error::Error>> {
    // Generate test data
    let data = generate_test_data(50, 10);

    // Configure and fit UMAP
    let config = UmapConfig {
        n_components: 3, // 3D embedding
        hidden_sizes: vec![8],
        graph: GraphParams {
            n_neighbors: 3,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 20,
            ..Default::default()
        },
        ..Default::default()
    };

    let umap = Umap::<MyAutodiffBackend>::new(config.clone());
    let fitted = umap.fit(data.clone(), None);
    
    // Save to temporary file
    let model_path = PathBuf::from("/tmp/test_umap_model_3.bin");
    if model_path.exists() {
        std::fs::remove_file(&model_path)?;
    }
    
    // Test multiple save/load cycles
    for i in 0..3 {
        // Save
        fitted.save(&model_path)?;
        
        // Verify file exists
        assert!(model_path.exists());
        
        // Load
        let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
            &model_path,
            config.clone(),
            10, // Input size
            Default::default(),
        )?;
        
        // Test transformation
        let test_data = vec![data[i as usize].clone()];
        let embedding = loaded_fitted.transform(test_data);
        
        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), 3); // 3D embedding
        
        // Verify embeddings are valid
        for &coord in &embedding[0] {
            assert!(coord.is_finite());
            assert!(!coord.is_nan());
        }
    }

    Ok(())
}

#[test]
fn test_different_configurations() -> Result<(), Box<dyn std::error::Error>> {
    let test_cases = vec![
        // (num_samples, num_features, hidden_sizes, n_neighbors)
        (50, 10, vec![5], 3),
        (100, 20, vec![10, 5], 5),
        (75, 15, vec![8, 4], 4),
    ];

    for (i, (num_samples, num_features, hidden_sizes, n_neighbors)) in test_cases.iter().enumerate() {
        println!("Testing configuration {}", i + 1);
        
        // Generate test data
        let data = generate_test_data(*num_samples, *num_features);

        // Configure UMAP
        let config = UmapConfig {
            n_components: 2,
            hidden_sizes: hidden_sizes.clone(),
            graph: GraphParams {
                n_neighbors: *n_neighbors,
                ..Default::default()
            },
            optimization: OptimizationParams {
                n_epochs: 15,
                ..Default::default()
            },
            ..Default::default()
        };

        // Train and save
        let umap = Umap::<MyAutodiffBackend>::new(config.clone());
        let fitted = umap.fit(data.clone(), None);
        
        let model_path = PathBuf::from(format!("/tmp/test_umap_model_{}.bin", i));
        if model_path.exists() {
            std::fs::remove_file(&model_path)?;
        }
        
        fitted.save(&model_path)?;
        
        // Load and test
        let model_path_ref = &model_path;
        let loaded_fitted = FittedUmap::<MyAutodiffBackend>::load(
            model_path_ref,
            config,
            *num_features,
            Default::default(),
        )?;

        // Test transformation
        let test_data = vec![data[0].clone()];
        let embedding = loaded_fitted.transform(test_data);
        
        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), 2);
        
        // Verify embeddings are valid
        for &coord in &embedding[0] {
            assert!(coord.is_finite());
            assert!(!coord.is_nan());
        }
    }

    Ok(())
}

#[test]
fn test_file_not_found_error() {
    let config = UmapConfig::default();
    let nonexistent_path = PathBuf::from("nonexistent_model.bin");
    
    let result = FittedUmap::<MyAutodiffBackend>::load(
        &nonexistent_path,
        config,
        10,
        Default::default(),
    );
    
    // Should return an error
    assert!(result.is_err());
}

#[test]
fn test_invalid_path_error() {
    let config = UmapConfig::default();
    let invalid_path = PathBuf::from("/invalid/directory/model.bin");
    
    let result = FittedUmap::<MyAutodiffBackend>::load(
        &invalid_path,
        config,
        10,
        Default::default(),
    );
    
    // Should return an error
    assert!(result.is_err());
}