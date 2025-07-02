# Tektra End-to-End Testing Guide

This guide explains how to run comprehensive end-to-end tests for Tektra, including tests that use actual AI models, camera, microphone, and document processing.

## Prerequisites

### 1. Install Ollama
Tektra uses Ollama for model inference. Install it from: https://ollama.ai

### 2. Pull Required Model
```bash
ollama pull gemma3n:e4b
```

### 3. Start Ollama Service
```bash
ollama serve
```

## Available Test Suites

### 1. Simple E2E Test (Recommended for Quick Testing)

The simple E2E test validates core functionality without requiring camera/microphone permissions:

```bash
cargo run --bin simple-e2e-test --features e2e-testing
```

This test covers:
- ✅ Basic text generation
- ✅ Document analysis (text/markdown)
- ✅ Image description (simulated)
- ✅ Performance benchmarking
- ✅ Output validation

### 2. Live Model Integration Tests

Run the integration tests that use actual model loading:

```bash
# Run all live model tests
cargo test --test live_model_test -- --ignored --nocapture

# Run specific test
cargo test --test live_model_test test_live_model_text_generation -- --ignored --nocapture
```

Available tests:
- `test_live_model_text_generation` - Basic text generation
- `test_live_model_with_image` - Image analysis
- `test_live_model_with_documents` - Document processing
- `test_live_model_multimodal_combined` - Combined image + audio
- `test_live_model_validation` - Output validation with secondary model
- `test_live_model_performance` - Performance benchmarking

### 3. Comprehensive E2E Test Runner (Full Testing)

The comprehensive test runner includes live camera and microphone testing:

```bash
# Run all tests including live camera/mic
cargo run --bin e2e-test-runner --features e2e-testing -- all --with-live

# Run specific test categories
cargo run --bin e2e-test-runner --features e2e-testing -- model
cargo run --bin e2e-test-runner --features e2e-testing -- documents
cargo run --bin e2e-test-runner --features e2e-testing -- multimodal --with-camera --with-mic
cargo run --bin e2e-test-runner --features e2e-testing -- validation
cargo run --bin e2e-test-runner --features e2e-testing -- benchmark --iterations 20
```

### 4. Unit and Integration Tests

Run standard tests (no model required):

```bash
# All tests
cargo test --workspace

# Library tests only
cargo test --lib

# Multimodal tests with real files
cargo test --test multimodal_with_real_files_test
```

## Test Data

Test data is located in `test_data/`:

```
test_data/
├── images/
│   ├── test_image.png      # Complex test pattern
│   ├── simple_shapes.png   # Basic shapes (red square, blue circle, green triangle)
│   ├── gradient.png        # Color gradient
│   └── pattern.png         # Geometric pattern
├── audio/
│   ├── test_speech.wav     # Sample speech
│   ├── command.wav         # Voice command
│   └── question.wav        # Question audio
└── documents/
    ├── sample.txt          # Plain text document
    ├── sample.md           # Markdown document
    └── technical_spec.md   # Technical specification
```

## Generating Test Data

Generate new test images:
```bash
cargo run --bin generate_test_images
```

## CI/CD Integration

For CI environments, use non-interactive mode:

```bash
# Skip permission prompts
cargo run --bin simple-e2e-test --features e2e-testing

# Run unit tests only (no Ollama required)
cargo test --lib
```

## Troubleshooting

### "Failed to connect to Ollama"
- Ensure Ollama is running: `ollama serve`
- Check if Ollama is accessible: `curl http://localhost:11434/api/tags`

### "Model gemma3n:e4b not found"
- Pull the model: `ollama pull gemma3n:e4b`
- Verify installation: `ollama list`

### Camera/Microphone Permission Denied
- On macOS: Grant Terminal/IDE camera and microphone permissions in System Preferences
- Run without live tests: `cargo run --bin e2e-test-runner --features e2e-testing -- all`

### Test Timeout
- Increase test timeout: `RUST_TEST_THREADS=1 cargo test -- --test-threads=1`
- Check model performance: Some tests may take longer on slower hardware

## Performance Expectations

Typical performance on Apple Silicon (M1/M2):
- Simple text generation: 20-50 tokens/second
- Document analysis: 2-5 seconds for 1000 words
- Image description: 3-8 seconds per image
- Multimodal processing: 5-15 seconds combined

## Adding New Tests

To add new E2E tests:

1. Add test functions to `tests/live_model_test.rs`
2. Use `#[ignore]` attribute for tests requiring Ollama
3. Follow naming convention: `test_live_model_*`
4. Include clear documentation and prerequisites

Example:
```rust
#[tokio::test]
#[ignore = "requires live Ollama instance"]
async fn test_live_model_custom_feature() {
    let ollama = OllamaInference::new(None).await.unwrap();
    // Test implementation
}
```

## Benchmarking

Run performance benchmarks:

```bash
# Quick benchmark (5 iterations)
cargo run --bin simple-e2e-test --features e2e-testing

# Detailed benchmark (custom iterations)
cargo run --bin e2e-test-runner --features e2e-testing -- benchmark --iterations 50
```

Benchmark results include:
- Average response time
- Tokens per second
- Min/max latency
- Memory usage (when available)