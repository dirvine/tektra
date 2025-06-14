# Tektra AI Assistant

A comprehensive AI assistant that integrates voice interaction, computer vision, and robotic control capabilities. Tektra uses the Qwen2.5-Omni-7B model for language processing and Pi0 for physical action understanding.

## Features

- **Multimodal Capabilities**:
  - Voice input via microphone
  - Camera input for vision
  - Voice output via text-to-speech
  - Robot control via FAST action tokens

- **Core AI Models**:
  - Qwen2.5-Omni-7B: Text, voice, and image processing
  - Pi0 with FAST processor: Physical task understanding

- **Advanced Features**:
  - Continuous fine-tuning pipeline using LoRA adapters
  - Episode logging for reinforcement learning
  - Apple Silicon MPS (Metal Performance Shaders) acceleration
  - Seamless input/output method switching

## Requirements

- Python 3.11 or higher
- UV package manager (https://github.com/astral-sh/uv)
- Dependencies (automatically installed by UV):
  - transformers, torch, accelerate, huggingface-hub
  - peft (for fine-tuning)
  - sounddevice, soundfile, scipy (for audio processing)
  - opencv-python (for camera input)
  - psutil, numpy, pillow, einops

## Installation & Usage

The script is self-contained and uses UV to manage dependencies. You can run it directly with the provided shell script:

```bash
# Clone the repository
git clone https://github.com/yourusername/tektra.git
cd tektra

# Make the run script executable
chmod +x run_tektra.sh

# Run with the interactive menu (default)
./run_tektra.sh

# Run with voice input and camera enabled
./run_tektra.sh --chat --voice-input

# Run with text input only (no voice or camera)
./run_tektra.sh --text-only

# Start continuous chat mode
./run_tektra.sh --continuous

# Fine-tune the model on collected action data
./run_tektra.sh --fine-tune

# Display system information
./run_tektra.sh --info

# Show all available options
./run_tektra.sh --help
```

Alternatively, you can run the script directly with UV:

```bash
# Run with the interactive menu
uv run tektra.py --menu

# Run with chat mode
uv run tektra.py --chat
```

### Troubleshooting Model Loading Issues

If you experience issues with model loading, try the following:

1. Run with the `--force-download` flag to re-download model files:
   ```bash
   ./run_tektra.sh --force-download --model Qwen/Qwen2.5-4B-Instruct
   ```

2. Try a smaller model if memory is limited:
   ```bash
   ./run_tektra.sh --model Qwen/Qwen2.5-4B-Instruct
   ```

3. Check for version compatibility issues in the logs. The script includes multiple fallback methods for loading models.

## Command-Line Options

- `--chat`: Start chat mode
- `--continuous`: Start continuous chat mode
- `--fine-tune`: Fine-tune model using collected robot episodes
- `--voice-input`: Use voice input (microphone) if available
- `--voice-output`: Use voice output (text-to-speech) if available
- `--no-camera`: Disable camera input
- `--text-only`: Use text input and output only, no camera
- `--model MODEL`: Specify Hugging Face model to use (default: Qwen/Qwen2.5-Omni-7B)
- `--force-download`: Force download of model even if in cache
- `--menu`: Launch interactive menu
- `--info`: Show system information

## Apple Silicon Support

Tektra is optimized for Apple Silicon Macs (M1/M2/M3 series) and automatically uses MPS (Metal Performance Shaders) for hardware acceleration, making even large models run efficiently. The script includes:

- Automatic MPS detection and configuration
- Proper handling of half-precision (float16) on Apple Silicon
- Environment variables in run_tektra.sh to optimize memory usage
- Multiple fallback mechanisms to handle edge cases in the MPS backend
- Special handling for model loading and pipeline creation on Apple Silicon

For optimal performance on MacBook Pro with 96GB RAM:
- The script defaults to Qwen2.5-Omni-7B which performs well on Apple Silicon
- Half-precision (float16) is automatically used when supported
- Memory usage is optimized for large models

## Robot Control

When Tektra detects a physical action request, it generates FAST tokens which are decoded into robot control commands. These can be sent to:

- ROS 2 (for robot arms or simulators)
- UART/GPIO interfaces (for direct hardware control)
- MQTT or WebSockets (for networked robots)

## Directory Structure

```
tektra/
├── tektra.py            # Main script
├── tektra_log.txt       # Interaction log
├── data/
│   ├── robot_episodes.json  # Fine-tuning dataset
│   └── images/              # Captured camera images 
└── models/
    ├── tektra/              # Downloaded models
    └── fine_tuned/          # Fine-tuned models
```

## License

This project is available under the MIT License.