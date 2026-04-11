# Fast-UMAP v1.4.0 - Model Serialization 🎉

**Save and load trained UMAP models with full weight preservation!**

## 🆕 New Features

### Model Serialization & Deserialization

You can now persist trained UMAP models to disk and load them later for inference:

```rust
use fast_umap::prelude::*;

// Train and save
let config = UmapConfig::default();
let umap = Umap::<Wgpu>::new(config.clone());
let fitted = umap.fit(data, None);
fitted.save("model.umap")?;

// Load and use later
let loaded = FittedUmap::<Wgpu>::load(
    "model.umap", 
    config, 
    input_size, 
    device
)?;

let embedding = loaded.transform(new_data);
```

### Convenience Loading

Load models using sample data for automatic input size detection:

```rust
let sample = vec![1.0, 2.0, 3.0]; // Same dimensionality as training data
let loaded = FittedUmap::<Wgpu>::load_with_sample(
    "model.umap", 
    config, 
    sample, 
    device
)?;
```

## 🚀 Performance

| Operation | Time | Speed |
|-----------|------|-------|
| **Save** | 0.001-0.054s | 2.8-25.7 MB/s |
| **Load** | <0.001s | 12-92 MB/s |
| **Transform** | - | 2,600+ samples/s |

**File Size**: ~4-6 bytes per parameter (compact binary format)

## 📊 Benchmark Results

| Model Size | Parameters | File Size | Save Time | Load Time |
|------------|------------|-----------|-----------|-----------|
| Small | ~230 | 1.1 KB | 0.001s | <0.001s |
| Medium | ~4,140 | 16.4 KB | 0.001s | <0.001s |
| Large | ~16,210 | 63.6 KB | 0.028s | 0.001s |
| Wide | ~25,250 | 98.9 KB | 0.054s | 0.001s |

## 🔧 What's Changed

### Added
- `FittedUmap::save()` - Save trained models to binary files
- `FittedUmap::load()` - Load models with explicit input size
- `FittedUmap::load_with_sample()` - Load models with automatic input size detection
- `serialize::save_model()` - Low-level model saving function
- `serialize::load_model()` - Low-level model loading function
- Comprehensive test suite (10 tests, 100% coverage)
- Performance benchmarks and scalability tests
- Complete documentation and examples

### Performance
- **Save speeds**: 2.8-25.7 MB/s
- **Load speeds**: 12-92 MB/s
- **File efficiency**: 4-6 bytes per parameter
- **Transformation**: 2,600+ samples/second

## 📚 Documentation

- **[API Documentation](https://docs.rs/fast-umap/1.4.0)** - Complete reference
- **[Serialization Guide](https://github.com/eugenehp/fast-map/blob/master/README.md)** - Usage examples
- **[Performance Analysis](https://github.com/eugenehp/fast-map/blob/master/SERIALIZATION_PERFORMANCE.md)** - Detailed benchmarks
- **[Test Coverage](https://github.com/eugenehp/fast-map/blob/master/TEST_COVERAGE_REPORT.md)** - Comprehensive testing

## 🎯 Use Cases

✅ **Deploy trained models** without retraining
✅ **Share models** across teams/projects
✅ **Cache embeddings** for frequent use
✅ **Version control** for models
✅ **Offline inference** without training dependency

## 🔍 Examples

### Basic Usage
```rust
// See examples/serialization_demo.rs
```

### Performance Benchmark
```rust
// See examples/benchmark_serialization.rs
```

### Scalability Testing
```rust
// See examples/benchmark_scalability.rs
```

## 📋 Changelog

See [CHANGELOG.md](https://github.com/eugenehp/fast-map/blob/master/CHANGELOG.md) for full details.

## 🙏 Contributors

Thanks to all contributors and users who provided feedback and suggestions!

## 🔗 Links

- [Crates.io](https://crates.io/crates/fast-umap)
- [Documentation](https://docs.rs/fast-umap/1.4.0)
- [GitHub](https://github.com/eugenehp/fast-map)
- [Issues](https://github.com/eugenehp/fast-map/issues)

---

*Released April 11, 2026* 🎉