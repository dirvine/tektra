# Tektra - New Multimodal AI Architecture

This document describes the complete rewrite of Tektra with a modern, scalable architecture using mistral.rs as the primary inference backend.

## 🎯 Overview

The new Tektra architecture provides:

- **Advanced Multimodal AI**: Full support for text, vision, audio, and document processing
- **Modern Inference Backend**: Built on mistral.rs with Qwen2.5-VL as the flagship model
- **Sophisticated Conversation Management**: Context-aware conversations with memory and personas
- **Model Abstraction Layer**: Unified interface supporting multiple AI backends
- **MCP Server Integration**: Model Context Protocol support for extensible tools
- **Production Ready**: Comprehensive error handling, testing, and performance optimization

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Tauri Frontend (React)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ IPC Commands
┌─────────────────────▼───────────────────────────────────────┐
│                  Main Application (Rust)                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Inference     │   Multimodal    │    Conversation         │
│   System        │   Processing    │    Management           │
├─────────────────┼─────────────────┼─────────────────────────┤
│ • Model Registry│ • Vision Proc   │ • Context Manager       │
│ • mistral.rs    │ • Audio Proc    │ • Memory Store          │
│ • Abstractions  │ • Document Proc │ • Prompt Templates      │
│ • Quantization  │ • Input Pipeline│ • Persona Manager       │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## 🚀 Key Features

### 1. Model Registry System

The Model Registry provides unified access to multiple AI models:

- **Qwen2.5-VL 7B** (Default): Flagship multimodal model with excellent vision and reasoning
- **Pixtral 12B**: Mistral's vision model for detailed image analysis
- **Llama 3.2 Vision 11B**: Meta's vision-capable model
- **Gemma 3 9B**: Google's efficient text model

```rust
// Example: Switch models dynamically
let mut registry = ModelRegistry::new();
registry.initialize().await?;
registry.switch_model("qwen2.5-vl-7b").await?;
```

### 2. Multimodal Processing Pipeline

Advanced processing for all input types:

- **Vision**: JPEG, PNG, GIF, WebP, BMP support with automatic resizing
- **Audio**: WAV, MP3, FLAC, OGG with speech recognition
- **Documents**: PDF, DOCX, TXT, Markdown, JSON with text extraction
- **Combined**: Multiple modalities in a single request

```rust
// Example: Process multiple modalities
let input = MultimodalInput::Combined {
    text: Some("Analyze these attachments".to_string()),
    images: vec![image_data],
    audio: Some(audio_data),
    documents: vec![document_data],
};
```

### 3. Conversation Management

Sophisticated conversation handling with:

- **Context Management**: Sliding window with token-aware truncation
- **Memory Store**: Long-term memory for facts and preferences
- **Personas**: Different assistant personalities (helpful, technical, creative, tutor)
- **Flow Control**: Intelligent conversation state tracking

```rust
// Example: Start a conversation with a specific persona
let manager = ConversationManager::new(Some(config))?;
let session_id = manager.start_session("user123", Some("technical")).await?;
```

### 4. Advanced Quantization Support

Optimized model performance with multiple quantization formats:

- **Q4_K_M**: Best balance of quality and speed
- **Q5_K_M**: Higher quality with moderate speed
- **Q6_K**: Near-original quality
- **Q8_0**: Minimal quality loss
- **F16/BF16**: Half precision formats

## 📋 Installation & Setup

### Prerequisites

- Rust 1.70+ with Cargo
- Node.js 18+ with npm
- Git
- 16GB+ RAM recommended for 7B models

### Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd tektra
   ```

2. **Build with new architecture**:
   ```bash
   ./build-new-architecture.sh
   ```

3. **For development**:
   ```bash
   npm run tauri dev
   ```

4. **For production bundle**:
   ```bash
   ./build-new-architecture.sh --bundle
   ```

### Configuration

Create `~/.tektra/config.toml`:

```toml
[inference]
backend = "mistral_rs"
default_model = "qwen2.5-vl-7b"
device = "auto"  # "cpu", "cuda", "metal", or "auto"

[conversation]
max_context_length = 32000
memory_enabled = true
auto_summarize = true

[models]
cache_dir = "~/.cache/tektra/models"
auto_download = true
```

## 🔧 Development

### Project Structure

```
src-tauri/src/
├── inference/              # AI model abstractions
│   ├── model_registry.rs   # Central model management
│   ├── mistralrs_backend.rs # mistral.rs integration
│   ├── quantization.rs     # Model optimization
│   └── streaming.rs        # Real-time responses
├── multimodal/             # Input processing
│   ├── vision_processor.rs # Image/video handling
│   ├── audio_processor.rs  # Audio processing
│   └── document_processor.rs # Document parsing
├── conversation/           # Chat management
│   ├── context_manager.rs  # Context handling
│   ├── memory_store.rs     # Long-term memory
│   ├── persona_manager.rs  # Assistant personalities
│   └── prompt_templates.rs # Model-specific formatting
└── main_new.rs             # New application entry point
```

### Running Tests

```bash
cd src-tauri
cargo test  # mistral-backend is included by default
```

### Available Tauri Commands

#### Model Management
- `initialize_new_model_system()`: Initialize the model registry
- `list_available_models()`: Get available models
- `switch_active_model(model_id)`: Change active model
- `get_model_status()`: Get current model info

#### Conversation
- `send_multimodal_message(...)`: Send message with attachments
- `start_conversation_session(...)`: Start new conversation
- `end_conversation_session(...)`: End conversation

#### Legacy Compatibility
- `send_message(...)`: Legacy text-only messages
- `get_chat_history()`: Get chat history
- `clear_chat_history()`: Clear history

## 🎮 Usage Examples

### Basic Text Conversation

```javascript
// Frontend: Send a simple text message
const response = await invoke('send_multimodal_message', {
  message: "Hello! How can you help me today?",
  sessionId: "user123"
});
```

### Image Analysis

```javascript
// Frontend: Send message with image
const response = await invoke('send_multimodal_message', {
  message: "What do you see in this image?",
  imageData: base64ImageData,
  sessionId: "user123"
});
```

### Model Switching

```javascript
// Frontend: Switch to technical expert persona
await invoke('switch_active_model', {
  modelId: "qwen2.5-vl-7b"
});

await invoke('start_conversation_session', {
  sessionId: "technical_session",
  persona: "technical"
});
```

### Document Processing

```javascript
// Frontend: Upload and analyze document
const response = await invoke('send_multimodal_message', {
  message: "Please summarize this document",
  fileAttachments: ["/path/to/document.pdf"],
  sessionId: "user123"
});
```

## 🔍 Performance & Optimization

### Model Performance

| Model | Size | VRAM | Speed | Vision | Quality |
|-------|------|------|-------|--------|---------|
| Qwen2.5-VL 7B (Q4_K_M) | ~4.5GB | 6GB | Fast | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Pixtral 12B (Q4_K_M) | ~7GB | 8GB | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Llama 3.2 Vision 11B (Q5_K_M) | ~7.5GB | 9GB | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Optimization Tips

1. **Choose appropriate quantization** based on your hardware
2. **Enable conversation memory** for better context awareness
3. **Use sliding window** for long conversations
4. **Cache frequently used models** to reduce load times

### System Requirements

- **Minimum**: 8GB RAM, CPU-only
- **Recommended**: 16GB RAM, Apple Silicon or CUDA GPU
- **Optimal**: 32GB RAM, dedicated GPU with 12GB+ VRAM

## 🤝 Migration from Legacy Tektra

The new architecture maintains backward compatibility:

1. **Existing commands** continue to work
2. **Chat history** is preserved
3. **Settings** are migrated automatically
4. **Gradual migration** - use new features when ready

Migration checklist:
- [ ] Update frontend to use `send_multimodal_message`
- [ ] Implement conversation sessions
- [ ] Configure personas for different use cases
- [ ] Test multimodal capabilities
- [ ] Optimize model selection for your hardware

## 🐛 Troubleshooting

### Common Issues

1. **Model loading fails**
   - Check available RAM/VRAM
   - Try a smaller quantization (Q4_K_M instead of Q6_K)
   - Ensure internet connection for model download

2. **Slow responses**
   - Use GPU acceleration if available
   - Choose faster quantization
   - Reduce context length

3. **Memory errors**
   - Close other applications
   - Use CPU-only mode
   - Choose smaller models

### Debug Mode

Enable detailed logging:
```bash
RUST_LOG=debug npm run tauri dev
# mistral-backend features are included by default
```

## 📊 Monitoring & Analytics

The new architecture includes built-in performance monitoring:

- **Token generation rate**
- **Memory usage tracking**
- **Model switching statistics**
- **Conversation flow analysis**

Access via:
```javascript
const stats = await invoke('get_model_status');
const multimodalStats = await invoke('get_multimodal_stats');
```

## 🔐 Security & Privacy

- **Local processing**: All AI runs locally, no cloud dependencies
- **Data isolation**: Conversations isolated per session
- **Memory protection**: Secure handling of multimodal data
- **No telemetry**: Optional usage analytics only

## 🛣️ Roadmap

### Phase 6: MCP Server Integration
- [ ] Implement Model Context Protocol server
- [ ] Add extensible tool system
- [ ] Web search and API integration
- [ ] Custom tool development framework

### Phase 7: Enhanced Audio Support  
- [ ] Real-time speech recognition
- [ ] Voice activity detection
- [ ] Speaker identification
- [ ] Audio generation capabilities

### Phase 8: Advanced Features
- [ ] Multi-agent conversations
- [ ] Custom model fine-tuning
- [ ] Plugin architecture
- [ ] Cloud model integration

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add comprehensive tests
5. Submit a pull request

### Code Standards

- **Rust**: Follow `rustfmt` and `clippy` guidelines
- **Error handling**: Use `Result<T, E>` consistently
- **Documentation**: Document all public APIs
- **Testing**: Maintain >80% test coverage

## 📜 License

MIT OR Apache-2.0

## 🙏 Acknowledgments

- **mistral.rs**: High-performance inference engine
- **Qwen team**: Excellent multimodal models
- **Tauri**: Modern desktop application framework
- **Rust community**: Amazing ecosystem and tools

---

For more information, see the [full documentation](./docs/) or join our [Discord community](https://discord.gg/tektra).