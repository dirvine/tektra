# Tektra Multimodal Guide

## Overview

Tektra is designed as a comprehensive multimodal AI assistant that can process text, images, audio, and video. However, the actual multimodal capabilities depend on the AI model being used and the current limitations of the Ollama backend.

## Current Multimodal Support Status

### Text Input ✅
All models support text input and generation.

### Image Input 🖼️
The following models support image analysis in Ollama:

- **LLaMA 3.2 Vision** (`llama3.2-vision:11b` or `llama3.2-vision:90b`)
  - Latest and most capable vision model
  - Excellent for image description, visual Q&A, and analysis
  
- **LLaVA** (`llava:7b`, `llava:13b`, or `llava:34b`)
  - Popular and well-tested vision model
  - Good balance of performance and capability
  
- **Moondream** (`moondream:latest`)
  - Lightweight vision model
  - Faster inference, suitable for quick image analysis
  
- **BakLLaVA** (`bakllava:latest`)
  - Alternative vision implementation
  - Similar capabilities to LLaVA

### Audio Input 🎵
Currently, **no Ollama models support direct audio input**. Audio files are transcribed to text using Whisper before being sent to the model.

### Video Input 🎬
Currently, **no Ollama models support video input**. Video processing is planned for future updates.

## Important Note About Gemma 3N

**Gemma 3N** (`gemma3n:e4b`) is designed by Google as a state-of-the-art multimodal model with:
- MobileNet-V5 vision encoder for image understanding
- USM audio encoder for audio processing
- Support for video analysis

However, **Ollama's current implementation of Gemma 3N only supports text input**. The multimodal features are not yet exposed through Ollama's API. This is a limitation of the Ollama integration, not the model itself.

## How to Use Multimodal Features

### For Image Analysis

1. **Switch to a vision-capable model**:
   ```bash
   # Pull a vision model
   ollama pull llama3.2-vision:11b
   # or
   ollama pull llava:7b
   ```

2. **In Tektra, select the vision model** from the model dropdown

3. **Upload an image** using the paperclip icon or drag-and-drop

4. **Ask questions about the image** in your message

### For Text + Document Analysis

All models support text document analysis. Simply:
1. Upload a text file (.txt, .md, .json)
2. Type your question about the document
3. The document content will be included with your question

## Technical Implementation

Tektra's multimodal pipeline includes:

1. **Multimodal Processor** (`multimodal_processor.rs`)
   - Optimizes images for model input
   - Handles audio preprocessing (when supported)
   - Manages context window efficiently

2. **Unified API** (`process_multimodal_input`)
   - Single endpoint for all input types
   - Combines text, images, audio, and documents
   - Maintains conversation context

3. **Smart Model Detection**
   - Automatically detects model capabilities
   - Provides helpful messages for unsupported features
   - Suggests alternative models when needed

## Future Roadmap

1. **Gemma 3N Multimodal**: Waiting for Ollama to expose multimodal capabilities
2. **Audio Support**: Direct audio input when models become available
3. **Video Processing**: Frame extraction and analysis
4. **Custom Model Integration**: Support for locally trained multimodal models

## Troubleshooting

### "Model doesn't support vision" message
- You're using a text-only model
- Switch to llava, llama3.2-vision, or moondream

### Images not being processed
- Ensure you're using a vision-capable model
- Check that the image format is supported (PNG, JPEG, GIF, WebP)
- Verify the image file isn't corrupted

### Large files causing timeouts
- Tektra automatically resizes images for optimal performance
- For very large documents, consider splitting them
- The context window is limited to 32K tokens

## Model Recommendations

- **For General Use**: `gemma3n:e4b` (text-only currently)
- **For Image Analysis**: `llama3.2-vision:11b` or `llava:7b`
- **For Fast Inference**: `moondream:latest`
- **For Code + Images**: `llava:7b` (good at reading code in screenshots)

## API Example

```typescript
// Check model capabilities
const capabilities = await invoke('get_model_capabilities', { 
  modelName: 'llava:7b' 
});

// Send multimodal input
const response = await invoke('process_multimodal_input', {
  message: "What's in this image?",
  imageData: imageBytes,  // Uint8Array
  audioData: null,
  videoData: null
});
```