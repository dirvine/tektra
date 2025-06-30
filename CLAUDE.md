# Claude Code Development Guidelines for Tektra

This file contains the development standards and practices that Claude should follow when working on the Tektra AI Assistant codebase.

## Project Overview

Tektra is a voice-interactive robotics AI assistant that merges voice, vision, and action understanding using cutting-edge open models. It features:

- **Desktop Application**: Built with Rust/Tauri + React TypeScript
- **AI Engine**: Ollama-based inference with automatic fallback to bundled Ollama
- **Multimodal Capabilities**: Text, voice, vision, and robotic action control using Gemma3n:e4b model
- **Apple Silicon Optimization**: Optimized for performance across platforms
- **Autonomous Ollama Management**: Detects system Ollama or automatically downloads and manages bundled version

## Architecture Guidelines

### Frontend (React TypeScript)
```typescript
// Component structure
src/
├── components/          # React components
│   ├── Avatar.tsx      # Animated avatar with lip-sync
│   ├── Chat.tsx        # Chat interface
│   ├── Controls.tsx    # Model and system controls
│   └── LoadingScreen.tsx # Model download progress
├── avatar/             # Avatar animation logic
├── styles/             # CSS modules
└── main.tsx           # Application entry point
```

### Backend (Rust/Tauri)
```rust
// Rust module structure
src-tauri/src/
├── main.rs            # Application entry point
├── lib.rs             # Library exports
├── ai/mod.rs          # AI model management
├── audio/mod.rs       # Audio processing
├── vision/mod.rs      # Camera integration
├── robot/mod.rs       # Robot control
└── state/mod.rs       # Application state
```

## Ollama Integration Architecture

### Autonomous Ollama Management
Tektra includes a sophisticated Ollama management system that ensures users never need to manually install Ollama:

```rust
// Ollama detection and management in src-tauri/src/ai/ollama_inference.rs
pub async fn find_ollama() -> Result<OllamaExe> {
    // 1. Check for system Ollama installation
    if let Ok(output) = Command::new("ollama").arg("--version").output() {
        if output.status.success() {
            return Ok(OllamaExe::System(PathBuf::from("ollama")));
        }
    }
    
    // 2. Download and extract bundled Ollama using ollama_td crate
    let download_config = ollama_td::OllamaDownload::default();
    let downloaded_path = ollama_td::download(download_config).await?;
    
    // 3. Extract and configure for platform
    // 4. Return embedded Ollama instance
    Ok(OllamaExe::Embedded(ollama_binary))
}
```

### Key Features:
- **Zero Manual Setup**: Users never need to install Ollama separately
- **System Integration**: Automatically detects and uses system Ollama if available
- **Cross-Platform Support**: Works on macOS, Linux, and Windows
- **Model Management**: Automatically pulls `gemma3n:e4b` model if not available
- **Background Operation**: Ollama server runs transparently in background

### Implementation Details:
- Uses `ollama_td` crate for reliable Ollama downloads
- Extracts platform-specific binaries from archives
- Manages Ollama server lifecycle (start/stop)
- Provides fallback error handling for network issues
- Caches downloaded Ollama in user data directory

## Development Workflow

### 1. Specification-Driven Development
- **Always reference** `specifications.md` for feature requirements
- **Maintain compatibility** with the original vision while modernizing implementation
- **Request user acceptance** before major architectural changes

### 2. Rust Development Standards

#### AI Module Guidelines
```rust
// Use proper error handling - NO unwrap() or expect()
pub async fn load_model(&mut self, model_name: &str) -> Result<()> {
    let tokenizer = match Tokenizer::from_file(&path) {
        Ok(tok) => tok,
        Err(e) => return Err(anyhow::anyhow!("Failed to load tokenizer: {}", e)),
    };
    // Implementation...
}

// Emit progress events for long operations
let _ = self.app_handle.emit("model-loading-progress", json!({
    "progress": 85,
    "status": "Loading tokenizer...",
    "model_name": model_name
}));
```

#### Model Management Requirements
- **HuggingFace Hub Integration**: Use standard cache directory (`~/.cache/huggingface`)
- **Progress Tracking**: Emit real-time progress events during downloads
- **Caching**: Check for existing files before downloading
- **Error Recovery**: Graceful handling of network failures

### 3. Frontend Development Standards

#### Loading Screen Component
```typescript
// Progress tracking interface
interface LoadingScreenProps {
  isLoading: boolean;
  progress?: number;
  status?: string;
  modelName?: string;
}

// Event listening for progress updates
useEffect(() => {
  const setupEventListeners = async () => {
    await listen<{ progress: number; status: string }>('model-loading-progress', 
      (event) => {
        // Update loading state
      });
  };
}, []);
```

#### Avatar System
- **Canvas-based rendering** for smooth animations
- **Lip-sync integration** (prepare for future TTS integration)
- **Responsive design** for different screen sizes

### 4. Model Integration Standards

#### Supported Models
Follow the specification requirements:
- **Primary Models**: MLX-community quantized models for Apple Silicon
- **Model Types**: Instruction-tuned models (Qwen, Llama, Phi, SmolLM)
- **File Format**: SafeTensors with tokenizer.json support

#### Download Progress Implementation
```rust
// File download with progress tracking
async fn download_file_with_progress(
    &self,
    url: &str,
    local_path: &PathBuf,
    filename: &str,
    model_name: &str,
    base_progress: f64,
    progress_weight: f64,
) -> Result<()> {
    // Stream download with chunk-by-chunk progress updates
    // Emit progress events: "Downloading model.safetensors (45.2 MB / 127.8 MB)"
}
```

### 5. Future Feature Integration

#### Robot Control (Per Specification)
```rust
// Prepare for FAST token integration
pub struct RobotController {
    connection: Option<TcpStream>,
    current_state: RobotState,
}

// Action sequence handling
pub async fn execute_action_sequence(&mut self, tokens: Vec<u32>) -> Result<()> {
    // Decode FAST tokens to robot commands
    // Execute action sequence
    // Log episode data for fine-tuning
}
```

#### Voice Integration (Future)
```rust
// Audio processing pipeline
pub struct AudioManager {
    recording_buffer: Arc<Mutex<Vec<f32>>>,
}

// Real-time audio processing
pub async fn process_voice_input(&self) -> Result<String> {
    // Convert audio to text
    // Pass to AI model for processing
}
```

## Testing Requirements

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_download_progress() {
        // Test progress event emission
        // Verify file download completion
        // Check cache behavior
    }

    #[test]
    fn test_hf_cache_dir_detection() {
        // Test cache directory resolution
        // Verify cross-platform compatibility
    }
}
```

### Integration Tests
- **Model Loading**: Verify complete model download and loading pipeline
- **Progress Tracking**: Ensure progress events are emitted correctly
- **Error Handling**: Test network failures and recovery
- **Cache Management**: Verify existing file detection

## Documentation Standards

### Code Documentation
```rust
/// Downloads a model file with real-time progress tracking
/// 
/// # Arguments
/// * `url` - Direct download URL for the model file
/// * `local_path` - Target path in HuggingFace cache format
/// * `filename` - Display name for progress messages
/// * `model_name` - Model identifier for progress events
/// * `base_progress` - Starting progress percentage (0-100)
/// * `progress_weight` - Progress allocation for this file (0-100)
/// 
/// # Returns
/// * `Ok(())` - File downloaded successfully
/// * `Err(anyhow::Error)` - Download failed with detailed error
/// 
/// # Example
/// ```rust
/// manager.download_file_with_progress(
///     "https://huggingface.co/model/resolve/main/model.safetensors",
///     &cache_path,
///     "model.safetensors", 
///     "mlx-community/Qwen2.5-7B-Instruct-4bit",
///     25.0,
///     70.0
/// ).await?;
/// ```
async fn download_file_with_progress(/* params */) -> Result<()>
```

## Quality Gates

Before any code is considered complete:

### ✅ Functionality
- [ ] Follows specification requirements
- [ ] Maintains existing loading screen functionality  
- [ ] Implements real-time progress tracking
- [ ] Uses proper HuggingFace Hub integration
- [ ] Handles errors gracefully

### ✅ Code Quality
- [ ] No `unwrap()` or `expect()` in production code
- [ ] Comprehensive error handling with `Result<T, E>`
- [ ] Progress events emitted for long operations
- [ ] Proper async/await usage
- [ ] Type safety with proper trait bounds

### ✅ Performance
- [ ] Streaming downloads for large files
- [ ] Memory-efficient file processing
- [ ] Cache-aware to avoid re-downloads
- [ ] Apple Silicon optimization via Metal/Candle

### ✅ User Experience
- [ ] Loading screen shows real progress
- [ ] File sizes displayed in human-readable format
- [ ] Clear error messages for failures
- [ ] Responsive UI during operations

## File Organization

```
tektra/
├── src/                    # React frontend
│   ├── components/        # UI components
│   ├── avatar/           # Avatar animation system
│   └── styles/           # CSS styling
├── src-tauri/            # Rust backend
│   ├── src/             # Rust source code
│   ├── icons/           # Application icons
│   └── Cargo.toml       # Rust dependencies
├── specifications.md     # Original project specification
├── CLAUDE.md            # This development guide
├── README.md            # User documentation
└── package.json         # Node.js frontend dependencies
```

## Key Principles

1. **Specification Compliance**: Always refer to `specifications.md` for requirements
2. **Progress Transparency**: Users should see real-time feedback for long operations
3. **Rust Safety**: Never use unsafe operations without explicit justification
4. **Performance First**: Optimize for Apple Silicon while maintaining cross-platform support
5. **User Experience**: Prioritize smooth, responsive interactions
6. **Future Ready**: Design for easy integration of voice, vision, and robot control

## Example Implementation Pattern

```rust
// Complete model loading with progress tracking
pub async fn load_model(&mut self, model_name: &str) -> Result<()> {
    // 1. Emit initial progress
    self.emit_progress(0, "Starting model load...", model_name).await;
    
    // 2. Download with progress tracking
    let download_result = self.download_model(model_name, false).await?;
    
    // 3. Load tokenizer with progress
    self.emit_progress(85, "Loading tokenizer...", model_name).await;
    let tokenizer = self.load_tokenizer(model_name).await?;
    
    // 4. Complete and emit success
    self.current_model = Some(model_name.to_string());
    self.tokenizer = Some(Arc::new(Mutex::new(tokenizer)));
    self.emit_completion(true, None).await;
    
    Ok(())
}
```

This approach ensures robust, user-friendly model management that aligns with the Tektra specification while providing excellent developer and user experience.

- #remember and use UV at all times