# MLX Integration for Tektra AI Assistant

This document describes the MLX (Apple's Machine Learning framework) integration in Tektra, providing dual-backend support for both GGUF and MLX model formats.

## Overview

Tektra now supports two inference backends:
- **GGUF**: Cross-platform quantized model format (default)
- **MLX**: Apple Silicon optimized format for maximum performance on M1/M2/M3 Macs

## Architecture

### Backend Abstraction

```rust
pub trait InferenceBackend: Send + Sync {
    fn load_model(&mut self, model_path: &Path) -> Result<()>;
    fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String>;
    fn generate_with_metrics(&self, prompt: &str, config: &InferenceConfig) -> Result<(String, InferenceMetrics)>;
    fn name(&self) -> &str;
    fn is_available() -> bool;
}
```

### Backend Selection

The system supports three backend selection strategies:
- `Auto`: Automatically selects MLX on Apple Silicon if available, falls back to GGUF
- `MLX`: Forces MLX backend (fails if not available)
- `GGUF`: Forces GGUF backend

## Configuration

### Runtime Configuration

Backend selection can be configured via:

1. **Configuration file** (`~/.config/tektra/config.json`):
```json
{
  "inference": {
    "backend": "Auto",
    "benchmark_on_startup": false,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1
  }
}
```

2. **Environment variables**:
```bash
export TEKTRA_BACKEND=mlx    # or gguf, auto
export TEKTRA_BENCHMARK=true  # Run benchmarks on startup
export TEKTRA_THREADS=8       # Number of threads
```

### Programmatic Configuration

```rust
// Create AI manager with specific backend
let ai_manager = AIManager::with_backend(app_handle, BackendType::MLX)?;

// Get backend information
let info = ai_manager.get_backend_info().await;

// Run benchmarks
let results = ai_manager.benchmark_backends("Test prompt", 100).await?;
```

## Performance Benchmarking

The system includes built-in benchmarking utilities to compare backends:

```rust
pub struct InferenceMetrics {
    pub tokens_generated: usize,
    pub time_to_first_token_ms: f64,
    pub tokens_per_second: f64,
    pub total_time_ms: f64,
    pub peak_memory_mb: f64,
}
```

### Running Benchmarks

Via Tauri command:
```javascript
const results = await invoke('benchmark_backends', {
    prompt: "What is the capital of France?",
    maxTokens: 100
});
```

## MLX Models

### Supported Models

MLX models from HuggingFace's `mlx-community`:
- `mlx-community/gemma-3n-E2B-4bit`
- `mlx-community/gemma-3n-E4B-4bit`
- Other MLX-quantized models

### Model Format

MLX models use:
- `model.safetensors`: Model weights in SafeTensors format
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration

## Current Status

### Implemented ✅

1. **Backend abstraction layer** - Complete trait-based design
2. **GGUF backend stub** - Ready for llama.cpp integration
3. **MLX backend stub** - Ready for mlx-rs integration
4. **Runtime configuration** - File and environment variable support
5. **Benchmarking framework** - Performance comparison utilities
6. **Platform detection** - Automatic Apple Silicon detection
7. **Fallback logic** - Graceful degradation to GGUF

### Pending Implementation 🚧

1. **MLX Compilation** - Requires XCode Command Line Tools:
   ```bash
   xcode-select --install
   xcrun -find metal  # Verify Metal compiler
   ```

2. **Actual inference** - Both GGUF and MLX need inference implementation:
   - GGUF: Integration with llama-cpp-rs or candle
   - MLX: Integration with mlx-rs when Metal compiler available

3. **Model conversion** - Utilities to convert between formats

## Enabling MLX

To enable MLX support, you need the Metal compiler which requires:

### Option 1: Full XCode Installation (Recommended)
1. **Install XCode from App Store** (free, ~15GB)
2. **Launch XCode once** to complete setup
3. **Verify Metal compiler**:
   ```bash
   xcrun -find metal
   # Should output: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal
   ```

### Option 2: Metal Developer Tools Only
1. **Download Metal Developer Tools** from [Apple Developer](https://developer.apple.com/metal/)
2. **Install the Metal framework**
3. **Set developer directory** if needed:
   ```bash
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```

### Once Metal is Available:

1. **Uncomment MLX dependency** in `Cargo.toml`:
   ```toml
   [target.'cfg(target_os = "macos")'.dependencies]
   mlx-rs = { version = "0.25.0-alpha.1", features = ["metal", "accelerate"] }
   ```

2. **Rebuild the application**:
   ```bash
   cd src-tauri
   cargo build --release
   ```

### Alternative: Use GGUF Backend

If you don't want to install XCode, the GGUF backend provides good performance and is already available:
- Set `TEKTRA_BACKEND=gguf` environment variable
- Or use the default Auto mode which will fall back to GGUF

## API Usage

### Frontend Integration

```typescript
// Get backend information
const backendInfo = await invoke('get_backend_info');
console.log(backendInfo);

// Run benchmarks
const benchmarks = await invoke('benchmark_backends', {
    prompt: "Explain quantum computing",
    maxTokens: 200
});

benchmarks.forEach(([backend, metrics]) => {
    console.log(`${backend}: ${metrics.tokens_per_second} tokens/sec`);
});
```

### Testing

Run the integration test:
```bash
cd src-tauri
cargo test test_inference_manager
```

## Future Enhancements

1. **Dynamic backend switching** - Switch backends without restart
2. **Model caching** - Shared cache between GGUF and MLX
3. **Streaming generation** - Token-by-token streaming
4. **Batch inference** - Process multiple prompts efficiently
5. **Custom quantization** - Convert models on-device

## Troubleshooting

### MLX Not Available

If MLX is not detected on Apple Silicon:
1. Check XCode tools: `xcode-select --install`
2. Verify architecture: `uname -m` (should show `arm64`)
3. Check Metal: `xcrun -find metal`
4. Review logs for specific errors

### Performance Issues

1. Ensure you're using the correct backend for your hardware
2. Check thread count configuration
3. Monitor memory usage during inference
4. Use benchmarking to compare backends

## Summary

The MLX integration provides a foundation for high-performance inference on Apple Silicon while maintaining cross-platform compatibility through GGUF. The abstraction layer allows easy addition of new backends and runtime selection based on platform capabilities.