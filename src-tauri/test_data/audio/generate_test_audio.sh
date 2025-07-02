#!/bin/bash

# Generate a simple test audio file using macOS's say command
# This creates a WAV file with speech for testing

echo "Generating test audio files..."

# Create audio directory if it doesn't exist
mkdir -p test_data/audio

# Generate speech samples
say -o test_data/audio/test_speech.aiff "Hello, this is a test audio file for the Tektra multimodal AI system. Can you hear me clearly?"

# Convert to WAV format (more universal)
if command -v ffmpeg &> /dev/null; then
    ffmpeg -i test_data/audio/test_speech.aiff -acodec pcm_s16le -ar 16000 test_data/audio/test_speech.wav -y
    echo "Generated: test_data/audio/test_speech.wav (16kHz WAV)"
else
    echo "FFmpeg not found. Generated AIFF file only: test_data/audio/test_speech.aiff"
fi

# Generate a short command
say -o test_data/audio/command.aiff "Turn on the lights"
if command -v ffmpeg &> /dev/null; then
    ffmpeg -i test_data/audio/command.aiff -acodec pcm_s16le -ar 16000 test_data/audio/command.wav -y
    echo "Generated: test_data/audio/command.wav"
fi

# Generate a question
say -o test_data/audio/question.aiff "What is the weather like today?"
if command -v ffmpeg &> /dev/null; then
    ffmpeg -i test_data/audio/question.aiff -acodec pcm_s16le -ar 16000 test_data/audio/question.wav -y
    echo "Generated: test_data/audio/question.wav"
fi

echo "Test audio files generated successfully!"