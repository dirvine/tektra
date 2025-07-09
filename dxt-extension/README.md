# Tektra Voice AI Assistant - DXT Extension

A comprehensive Desktop Extension (DXT) that provides voice-interactive AI capabilities with multimodal support using the Qwen2.5-VL model and Unmute voice pipeline.

## Features

- **Real-time Voice Conversation**: Natural speech-to-text and text-to-speech interaction
- **Multimodal AI**: Vision, audio, and text processing using Qwen2.5-VL-7B model
- **Local Inference**: Runs entirely on your device with Metal acceleration (macOS) or CUDA
- **Professional Voice Pipeline**: Integrates with Unmute for high-quality speech processing
- **Model Management**: Automatic downloading, caching, and switching between AI models
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Installation

1. **Download the DXT file** from the releases page
2. **Install via DXT-compatible client** (Claude Desktop, etc.)
3. **Grant necessary permissions** when prompted:
   - Audio input/output access
   - File system access for model caching
   - Network access for model downloads
   - Process spawning for voice services

## Prerequisites

The extension will automatically handle most dependencies, but you may need:

- **Node.js 18+** (for the MCP server)
- **Tektra executable** (can be built from source or installed separately)
- **System audio permissions** for voice functionality
- **Available disk space** (~10GB for AI models)

## Quick Start

Once installed, the extension provides these MCP tools:

### Start Voice Conversation
```typescript
// Start a voice conversation with the AI
start_voice_conversation({
  character: "friendly" // options: default, friendly, professional
})
```

### Process Multimodal Input
```typescript
// Send text, images, or audio to the AI
process_multimodal_input({
  input_type: "text_with_image",
  text: "What do you see in this image?",
  image: "base64_encoded_image_data"
})
```

### Load AI Models
```typescript
// Switch between available models
load_model({
  model_id: "qwen2.5-vl-7b" // options: qwen2.5-vl-7b, qwen2.5-7b, auto
})
```

## Configuration

The extension supports user configuration through the DXT manifest:

| Setting | Description | Default |
|---------|-------------|---------|
| `voice_character` | AI personality | `default` |
| `model_preference` | Preferred AI model | `qwen2.5-vl-7b` |
| `enable_gpu_acceleration` | Use GPU when available | `true` |
| `voice_sensitivity` | Speech detection sensitivity | `0.6` |
| `enable_interruption` | Allow voice interruption | `true` |
| `auto_start_services` | Start services automatically | `true` |
| `cache_directory` | Model storage location | `~/.cache/tektra-ai` |

## Available Tools

### Voice Management
- `start_voice_conversation` - Begin real-time voice interaction
- `stop_voice_conversation` - End voice session
- `get_voice_status` - Check service status
- `configure_voice_settings` - Adjust voice parameters

### AI Model Operations
- `load_model` - Load or switch AI models
- `get_model_info` - View available models
- `process_multimodal_input` - Send complex inputs to AI

### Pipeline Management
- `manage_voice_pipeline` - Control voice service lifecycle

## Architecture

The extension consists of:

1. **MCP Server** (Node.js) - Handles tool requests and manages the Tektra process
2. **Tektra Core** (Rust) - Main AI inference engine with multimodal capabilities
3. **Voice Pipeline** (Unmute) - Professional speech processing services
4. **Model Cache** - Local storage for downloaded AI models

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DXT Client    │ ── │   MCP Server    │ ── │  Tektra Core    │
│ (Claude, etc.)  │    │   (Node.js)     │    │    (Rust)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                │                       │
                        ┌─────────────────┐    ┌─────────────────┐
                        │ Voice Pipeline  │    │  AI Models      │
                        │   (Unmute)      │    │ (Qwen2.5-VL)    │
                        └─────────────────┘    └─────────────────┘
```

## Troubleshooting

### Voice Services Won't Start
1. Check audio permissions
2. Ensure ports 8000, 8089, 8090 are available
3. Try manually starting: `manage_voice_pipeline({ action: "restart" })`

### Model Loading Fails
1. Check internet connection for downloads
2. Verify disk space (~8GB needed for Qwen2.5-VL)
3. Check cache directory permissions

### Performance Issues
1. Enable GPU acceleration in settings
2. Close other resource-intensive applications
3. Consider switching to smaller model if needed

## Development

To build from source:

```bash
# Clone the repository
git clone https://github.com/dirvine/tektra.git
cd tektra

# Build Tektra core
cargo build --release

# Build DXT extension
cd dxt-extension/server
npm install
npm run build

# Package DXT
cd ..
zip -r tektra-voice-ai.dxt manifest.json server/ icon.png screenshots/
```

## Contributing

Contributions are welcome! Please see the main [Tektra repository](https://github.com/dirvine/tektra) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Privacy & Security

- **Local Processing**: All AI inference happens on your device
- **No Data Collection**: Voice and image data never leave your machine
- **Secure Connections**: Local-only WebSocket connections
- **Open Source**: Full source code available for audit

## Support

- **Issues**: Report bugs on the [GitHub repository](https://github.com/dirvine/tektra/issues)
- **Documentation**: See the main Tektra documentation
- **Community**: Join discussions in GitHub Discussions