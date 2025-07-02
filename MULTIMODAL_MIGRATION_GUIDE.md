# Tektra Multimodal Infrastructure Migration Guide

This guide explains how to migrate from the current Ollama-only backend to the new unified multimodal infrastructure.

## Overview

The new infrastructure provides:
- **Multiple inference backends**: mistral.rs (primary), llama.cpp (fallback), Ollama (compatibility)
- **Full multimodal support**: Text, image, audio, and video inputs
- **Flexible model management**: Automatic backend selection based on model capabilities
- **Template-based formatting**: Model-specific prompt templates
- **Configuration-driven**: models.toml for easy model and backend configuration

## Architecture Changes

### Before (Ollama-only)
```
User Input → Ollama → Gemma 3N → Response
```

### After (Unified Infrastructure)
```
User Input → UnifiedModelManager → Backend Selection → Model → Response
                                         ↓
                                   MistralRs / LlamaCpp / Ollama
```

## Migration Steps

### 1. Update Main Application State

Replace the current `AIManager` with `TektraModelIntegration`:

```rust
// Before
struct AppState {
    ai_manager: Arc<Mutex<AIManager>>,
    audio_recorder: Arc<Mutex<AudioRecorder>>,
    camera_manager: Arc<Mutex<CameraManager>>,
}

// After
struct AppState {
    model_integration: Arc<TektraModelIntegration>,
    audio_recorder: Arc<Mutex<AudioRecorder>>,
    camera_manager: Arc<Mutex<CameraManager>>,
}
```

### 2. Initialize the New Infrastructure

In `main.rs`:

```rust
// Before
let ai_manager = Arc::new(Mutex::new(
    AIManager::new(app_handle.clone()).await?
));

// After
let model_integration = Arc::new(
    TektraModelIntegration::new(app_handle.clone()).await?
);
```

### 3. Update Tauri Commands

#### Load Model Command

```rust
// Before
#[tauri::command]
async fn load_model(
    state: tauri::State<'_, AppState>,
    model_name: &str
) -> Result<String, String> {
    let ai_manager = state.ai_manager.lock().await;
    ai_manager.load_model(model_name).await
        .map_err(|e| e.to_string())?;
    Ok(format!("Model {} loaded", model_name))
}

// After
#[tauri::command]
async fn load_model(
    state: tauri::State<'_, AppState>,
    model_name: &str
) -> Result<String, String> {
    state.model_integration.load_model(model_name).await
        .map_err(|e| e.to_string())?;
    Ok(format!("Model {} loaded", model_name))
}
```

#### Process Input Command

```rust
// Before
#[tauri::command]
async fn process_input(
    state: tauri::State<'_, AppState>,
    input: String
) -> Result<String, String> {
    let mut ai_manager = state.ai_manager.lock().await;
    ai_manager.process_text(&input).await
        .map_err(|e| e.to_string())
}

// After
#[tauri::command]
async fn process_input(
    state: tauri::State<'_, AppState>,
    input: String
) -> Result<String, String> {
    state.model_integration.process_input(&input).await
        .map_err(|e| e.to_string())
}
```

#### New Multimodal Command

```rust
#[tauri::command]
async fn process_multimodal(
    state: tauri::State<'_, AppState>,
    text: Option<String>,
    image_base64: Option<String>,
    audio_base64: Option<String>
) -> Result<String, String> {
    let image_data = image_base64
        .map(|b64| base64::decode(b64))
        .transpose()
        .map_err(|e| e.to_string())?;
    
    let audio_data = audio_base64
        .map(|b64| base64::decode(b64))
        .transpose()
        .map_err(|e| e.to_string())?;
    
    state.model_integration.process_multimodal(
        text,
        image_data,
        audio_data
    ).await.map_err(|e| e.to_string())
}
```

### 4. Update Audio Processing

For conversational audio with multimodal support:

```rust
// In process_audio_input command
let response = if audio_data.len() > 1000 { // If we have significant audio
    // Use multimodal processing
    state.model_integration.process_audio_with_transcription(
        &transcription,
        audio_data
    ).await?
} else {
    // Text-only processing
    state.model_integration.process_input(&transcription).await?
};
```

### 5. Update Camera Integration

For image processing:

```rust
#[tauri::command]
async fn process_camera_frame(
    state: tauri::State<'_, AppState>,
    description: String,
    image_base64: String
) -> Result<String, String> {
    let image_data = base64::decode(image_base64)
        .map_err(|e| e.to_string())?;
    
    state.model_integration.process_image_with_description(
        &description,
        image_data
    ).await.map_err(|e| e.to_string())
}
```

### 6. Add New Commands

Add these new commands to expose the enhanced capabilities:

```rust
#[tauri::command]
async fn list_models(
    state: tauri::State<'_, AppState>
) -> Result<Vec<ModelInfo>, String> {
    state.model_integration.list_models().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn switch_backend(
    state: tauri::State<'_, AppState>,
    backend: String
) -> Result<(), String> {
    state.model_integration.switch_backend(&backend).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn set_system_prompt(
    state: tauri::State<'_, AppState>,
    prompt: String
) -> Result<(), String> {
    state.model_integration.set_system_prompt(&prompt).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn stream_response(
    state: tauri::State<'_, AppState>,
    input: String,
    window: tauri::Window
) -> Result<(), String> {
    let mut receiver = state.model_integration.stream_response(&input).await
        .map_err(|e| e.to_string())?;
    
    // Stream chunks to frontend
    tokio::spawn(async move {
        while let Some(chunk) = receiver.recv().await {
            let _ = window.emit("stream-chunk", chunk);
        }
        let _ = window.emit("stream-end", ());
    });
    
    Ok(())
}
```

### 7. Update Frontend Integration

Update the frontend to handle new capabilities:

```typescript
// List available models
const models = await invoke('list_models');

// Process multimodal input
const response = await invoke('process_multimodal', {
  text: "What's in this image?",
  image_base64: imageData,
  audio_base64: null
});

// Stream responses
await invoke('stream_response', { input: userInput });

// Listen for streaming chunks
await listen('stream-chunk', (event) => {
  appendToResponse(event.payload);
});

await listen('stream-end', () => {
  finalizeResponse();
});
```

## Configuration

### models.toml

The new system uses `models.toml` for configuration. Key sections:

1. **Backend Configuration**:
```toml
[backends]
default = ["mistral_rs", "llama_cpp", "ollama"]

[backends.mistral_rs]
enabled = true
device = "metal"  # For Apple Silicon
```

2. **Model Definitions**:
```toml
[[models]]
id = "gemma3n:e4b"
backend_preferences = ["ollama", "mistral_rs"]
multimodal = true
capabilities = ["text", "image", "audio", "video"]
```

3. **Performance Presets**:
```toml
[presets.fast]
max_tokens = 256
temperature = 0.7
```

## Backward Compatibility

The `TektraModelIntegration` maintains backward compatibility:

- `load_gemma3n()` - Loads "gemma3n:e4b" model
- `process_input()` - Works exactly like before
- Ollama backend still supported for existing models

## Benefits of Migration

1. **Performance**: mistral.rs is faster than Ollama for many models
2. **Flexibility**: Switch backends without changing code
3. **Multimodal**: Native support for images, audio, video
4. **Streaming**: Built-in streaming support
5. **Templates**: Proper formatting for different model families
6. **Configuration**: Easy model management via models.toml

## Testing the Migration

1. **Test existing functionality**:
```bash
# Should work exactly as before
cargo tauri dev
# Load Gemma 3N, process text
```

2. **Test new multimodal features**:
```bash
# Process image with text
# Stream responses
# Switch backends
```

3. **Validate configuration**:
```rust
let issues = state.model_integration.validate_config().await;
if !issues.is_empty() {
    warn!("Configuration issues: {:?}", issues);
}
```

## Troubleshooting

### Model not loading
- Check models.toml for correct model ID
- Ensure backend is enabled
- Verify model is supported by the backend

### Performance issues
- Adjust memory limits in models.toml
- Use quantized models for faster inference
- Switch to CPU if GPU issues occur

### Multimodal not working
- Check model capabilities in models.toml
- Ensure backend supports multimodal
- Verify input format is correct

## Future Enhancements

The new infrastructure enables:
- Adding new backends (Triton, vLLM)
- Custom model loaders
- Fine-tuning integration
- Model optimization tools
- A/B testing different backends

## Summary

The migration preserves all existing functionality while adding powerful new capabilities. The system remains simple to use while providing advanced features when needed. The configuration-driven approach makes it easy to add new models and backends without code changes.