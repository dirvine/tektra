# Cross-Platform Inference Support

Tektra AI Assistant now supports cross-platform inference with automatic backend selection based on your hardware and operating system.

## Platform Support Matrix

| Platform | Architecture | Default Backend | Notes |
|----------|-------------|-----------------|-------|
| macOS | Apple Silicon (M1/M2/M3) | MLX | Optimized for Metal GPU acceleration |
| macOS | Intel (x86_64) | GGUF | MLX not supported on Intel Macs |
| Linux | Any | GGUF | Cross-platform compatibility |
| Windows | Any | GGUF | Cross-platform compatibility |

## Automatic Backend Selection

When running Tektra in default mode, it automatically selects the best backend for your platform:

```bash
# Automatic selection (recommended)
cargo run
# - Apple Silicon Mac → MLX backend
# - Intel Mac → GGUF backend  
# - Linux → GGUF backend
# - Windows → GGUF backend
```

## Manual Backend Selection

You can override the automatic selection using environment variables:

```bash
# Force GGUF backend (works everywhere)
TEKTRA_BACKEND=gguf cargo run

# Force MLX backend (Apple Silicon only)
TEKTRA_BACKEND=mlx cargo run

# Auto mode (default)
TEKTRA_BACKEND=auto cargo run
```

## Backend Features

### MLX Backend (Apple Silicon Only)
- **Performance**: Leverages Metal GPU for faster inference
- **Memory**: Efficient unified memory architecture
- **Models**: Supports MLX-optimized models from mlx-community
- **Requirements**: 
  - macOS on Apple Silicon (M1/M2/M3)
  - XCode Command Line Tools with Metal compiler

### GGUF Backend (All Platforms)
- **Compatibility**: Works on macOS, Linux, and Windows
- **Models**: Supports quantized GGUF models
- **Framework**: Uses Candle for CPU inference
- **Fallback**: Provides helpful responses even without model files

## Verifying Your Setup

Run the platform detection test to verify your configuration:

```bash
cargo run --bin test_platform
```

Expected output on Apple Silicon Mac:
```
=== Cross-Platform Inference Backend Test ===

Available Inference Backends:
- GGUF: Available (cross-platform - works on all systems)
- MLX: Available (Apple Silicon detected)

Current platform: macos aarch64 (Apple Silicon)
Default backend: MLX (optimized for your Apple Silicon)

1. Testing Auto mode:
   ✅ Auto mode selected: MLX
   ✅ Correctly selected MLX on Apple Silicon!

2. Testing GGUF mode:
   ✅ GGUF backend created successfully (GGUF (Candle))

3. Testing MLX mode:
   ✅ MLX backend created successfully on Apple Silicon!
```

Expected output on Linux/Windows:
```
=== Cross-Platform Inference Backend Test ===

Available Inference Backends:
- GGUF: Available (cross-platform - works on all systems)
- MLX: Not available (Apple Silicon only - using GGUF on Linux)

Current platform: linux x86_64 (Non-Apple Silicon)
Default backend: GGUF (cross-platform compatibility)

1. Testing Auto mode:
   ✅ Auto mode selected: GGUF (Candle)
   ✅ Correctly selected GGUF on Linux!
```

## Configuration

### Using Configuration File

Create a `tektra.toml` configuration file:

```toml
[inference]
backend = "auto"  # Options: "auto", "mlx", "gguf"
benchmark_on_startup = false
max_tokens = 512
temperature = 0.7
top_p = 0.9
repeat_penalty = 1.1
```

### Environment Variables

```bash
# Backend selection
export TEKTRA_BACKEND=auto

# Inference parameters
export TEKTRA_MAX_TOKENS=512
export TEKTRA_TEMPERATURE=0.7
export TEKTRA_TOP_P=0.9
```

## Building for Different Platforms

### macOS (Universal Binary)
```bash
# Build for both Intel and Apple Silicon
cargo build --release --target universal-apple-darwin
```

### Linux
```bash
# Build for Linux
cargo build --release
```

### Windows
```bash
# Build for Windows
cargo build --release
```

## Docker Support

For containerized deployments (Linux containers):

```dockerfile
FROM rust:1.75

WORKDIR /app
COPY . .

# GGUF backend will be used automatically
RUN cargo build --release

CMD ["./target/release/tektra"]
```

## Troubleshooting

### MLX Not Available on Apple Silicon
If MLX shows as unavailable on your Apple Silicon Mac:

1. Install XCode Command Line Tools:
   ```bash
   xcode-select --install
   ```

2. Verify Metal compiler:
   ```bash
   xcrun -find metal
   ```

3. Rebuild the application:
   ```bash
   cargo clean
   cargo build
   ```

### Performance Optimization

- **Apple Silicon**: MLX backend provides 2-5x faster inference
- **Other Platforms**: GGUF with CPU optimization flags:
  ```bash
  RUSTFLAGS="-C target-cpu=native" cargo build --release
  ```

## Model Compatibility

### MLX Models (Apple Silicon)
Download from mlx-community on HuggingFace:
- `mlx-community/gemma-2b-it-4bit`
- `mlx-community/Qwen2.5-7B-Instruct-4bit`

### GGUF Models (All Platforms)
Download from TheBloke or other GGUF providers:
- `TheBloke/gemma-2-2b-it-GGUF`
- `TheBloke/Qwen2.5-7B-Instruct-GGUF`

## Summary

Tektra's cross-platform support ensures:
- **Optimal Performance**: Automatically uses MLX on Apple Silicon
- **Universal Compatibility**: Falls back to GGUF on all other platforms
- **Seamless Experience**: Same API and features across all platforms
- **Future Ready**: Easy to add new backends (CUDA, ROCm, etc.)

The implementation prioritizes user experience by automatically selecting the best available backend while allowing manual override when needed.