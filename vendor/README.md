# Tektra Vendored Dependencies

This directory contains vendored and customized versions of critical dependencies for the Tektra AI assistant. This approach provides complete control over the inference stack while ensuring stability and enabling custom optimizations.

## Directory Structure

```
vendor/
├── mistralrs/           # Custom Tektra fork of mistral.rs
│   ├── Cargo.toml       # Tektra-specific configurations
│   ├── src/             # Source code with custom optimizations
│   ├── docs/            # Documentation for customizations
│   └── benchmarks/      # Performance benchmarks
├── candle/              # Forked candle ML framework
│   ├── candle-core/     # Core tensor operations
│   ├── candle-nn/       # Neural network layers
│   ├── candle-transformers/ # Transformer architectures
│   ├── candle-metal/    # Metal compute backend
│   └── tektra-optimizations/ # Tektra-specific optimizations
├── hf-hub-tektra/       # Enhanced HuggingFace Hub integration
│   ├── src/             # Enhanced download and caching logic
│   ├── progress/        # Granular progress tracking
│   └── tektra-models/   # Tektra model registry
└── tokenizers-custom/   # Custom tokenizer implementations
    ├── qwen-tokenizer/  # Optimized Qwen tokenizers
    ├── fast-tokenizer/  # Performance-optimized tokenizers
    └── streaming/       # Streaming tokenization
```

## Key Modifications

### mistralrs (Tektra Fork)
- **Enhanced Progress Tracking**: File-level progress with detailed status updates
- **Memory Optimization**: Smart caching and automatic cleanup
- **Apple Silicon**: Custom Metal kernels for M-series chips
- **Error Recovery**: Graceful degradation and resume capability
- **Tektra Integration**: Native event emission and state management

### candle (Custom Build)
- **Metal Optimizations**: Custom compute shaders for inference
- **Memory Management**: Pool allocators and automatic cleanup
- **Quantization**: Dynamic precision switching based on memory
- **Streaming**: Real-time inference kernels
- **Profiling**: Built-in performance monitoring

### hf-hub-tektra (Enhanced Hub Client)
- **Parallel Downloads**: Connection pooling and concurrent transfers
- **Resume Support**: Robust handling of interrupted downloads
- **Progress Granularity**: Per-file and per-chunk progress reporting
- **Local Registry**: Model versioning and dependency tracking
- **Tektra Models**: Custom model format support

### tokenizers-custom (Optimized Tokenizers)
- **Fast Tokenization**: SIMD-optimized string processing
- **Streaming Support**: Real-time tokenization for voice input
- **Custom Vocabularies**: Tektra-specific token extensions
- **Memory Efficiency**: Reduced memory footprint for mobile deployment

## Version Control Strategy

Each vendored dependency maintains:
1. **Original Git History**: Preserved for upstream merging
2. **Tektra Branch**: Custom modifications and optimizations
3. **Version Tags**: Stable releases with semantic versioning
4. **Patch Files**: Documented changes for easy review

## Build System Integration

The main `Cargo.toml` references these vendored dependencies using path dependencies:

```toml
[dependencies]
tektra-inference = { path = "vendor/mistralrs" }
tektra-candle = { path = "vendor/candle/candle-core" }
tektra-models = { path = "vendor/hf-hub-tektra" }
tektra-tokenizers = { path = "vendor/tokenizers-custom" }
```

## Development Workflow

### Adding New Optimizations
1. Create feature branch in vendored repository
2. Implement optimization with comprehensive tests
3. Benchmark against baseline performance
4. Document changes and integration points
5. Update Tektra integration layer

### Updating Upstream Dependencies
1. Review upstream changes for compatibility
2. Cherry-pick relevant improvements
3. Resolve conflicts with Tektra modifications
4. Run full test suite and benchmarks
5. Update version tags and documentation

### Testing Strategy
- **Unit Tests**: Each module has comprehensive test coverage
- **Integration Tests**: Cross-module compatibility verification
- **Performance Tests**: Benchmark suites with regression detection
- **Memory Tests**: Leak detection and profiling
- **Platform Tests**: macOS, Linux, and Windows compatibility

## Performance Benchmarks

Baseline performance comparisons are maintained in each module:

- **Inference Speed**: Tokens per second for different model sizes
- **Memory Usage**: Peak and steady-state memory consumption
- **Startup Time**: Model loading and initialization performance
- **Download Speed**: Model download and caching efficiency

## Security Considerations

All vendored dependencies undergo:
- **Code Review**: Line-by-line review of all modifications
- **Dependency Audit**: Regular security scanning and updates
- **Supply Chain**: Verified sources and cryptographic signatures
- **License Compliance**: Clear attribution and compatibility checks

## Maintenance

Regular maintenance tasks include:
- Monthly upstream synchronization
- Quarterly performance benchmarking
- Annual security audits
- Continuous integration testing

This vendored approach ensures Tektra maintains complete control over its AI inference stack while providing stability, performance, and security benefits.