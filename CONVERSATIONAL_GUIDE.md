# Tektra Conversational Audio Guide

## Overview

Tektra features a sophisticated conversational audio system designed to provide natural, hands-free interaction. The system includes wake word detection, continuous conversation support, interruption handling, and text-to-speech output.

## Key Features

### 1. Wake Word Detection ("Tektra")
- **Always Listening**: Tektra continuously listens for its wake word in the background
- **Low Latency**: Wake word detection happens locally for instant response
- **Flexible Recognition**: Handles variations like "Hey Tektra" or "Okay Tektra"
- **Visual Feedback**: Pulse animation when wake word is detected

### 2. Conversation Modes

The system operates in several modes:

- **Idle**: Listening only for wake word
- **Wake Word Detected**: Tektra heard its name and is waiting for your command
- **Active Listening**: Recording and processing your speech
- **Processing**: Thinking about your request
- **Responding**: Speaking the answer
- **Waiting for User**: Ready for follow-up questions (continuous conversation)

### 3. Natural Conversation Flow

```
User: "Tektra"
Tektra: *chime* (listening)
User: "What's the weather like today?"
Tektra: "I'd be happy to help with weather information..."
User: "What about tomorrow?" (no wake word needed)
Tektra: "Tomorrow's forecast shows..."
```

### 4. Interruption Handling
- Users can interrupt Tektra while it's speaking
- Simply start talking and Tektra will stop and listen
- Useful for corrections or follow-up questions

## Technical Architecture

### Audio Pipeline

1. **Audio Capture** → **VAD** → **Buffer Management** → **Whisper STT** → **Conversation Manager** → **Gemma 3N** → **TTS** → **Audio Output**

### Components

#### 1. Audio Recorder (`audio/mod.rs`)
- Manages microphone input
- Implements Voice Activity Detection (VAD)
- Handles continuous audio streaming
- Integrates Whisper for speech-to-text

#### 2. Conversation Manager (`audio/conversation_manager.rs`)
- Tracks conversation state
- Detects wake word
- Manages turn-taking
- Handles interruptions
- Maintains conversation context

#### 3. Speech Processor (`ai/speech_processor.rs`)
- Detects speech boundaries
- Manages silence thresholds
- Handles pause detection
- Coordinates with Whisper STT

#### 4. TTS Manager (`audio/tts.rs`)
- Multiple backend support (system native, Web Speech API)
- Voice configuration
- Emotion and prosody control
- Speech queue management

### Wake Word Detection

The system uses a rolling buffer approach:
```rust
// Continuously check for wake word in transcribed text
if text.to_lowercase().contains("tektra") {
    // Activate conversation mode
    start_conversation();
}
```

### Conversation State Machine

```
         ┌──────┐
         │ Idle │◄─────────────────┐
         └───┬──┘                  │
             │ "Tektra"            │
             ▼                     │
    ┌─────────────────┐            │
    │ Wake Word       │            │
    │ Detected        │            │
    └────────┬────────┘            │
             │ User speaks         │
             ▼                     │
    ┌─────────────────┐            │
    │ Active          │            │
    │ Listening       │            │
    └────────┬────────┘            │
             │ Silence detected    │
             ▼                     │
    ┌─────────────────┐            │
    │ Processing      │            │
    └────────┬────────┘            │
             │ AI responds         │
             ▼                     │
    ┌─────────────────┐            │
    │ Responding      │            │
    └────────┬────────┘            │
             │ TTS complete        │
             ▼                     │
    ┌─────────────────┐            │
    │ Waiting for     │            │
    │ User            ├────────────┘
    └─────────────────┘  30s timeout
```

## Configuration

### Timing Parameters

```rust
ConversationConfig {
    wake_word: "tektra",
    wake_word_timeout_ms: 3000,      // Wait 3s after wake word
    turn_timeout_ms: 1500,           // 1.5s silence = end turn
    interrupt_threshold_ms: 500,     // 0.5s to interrupt
    continuous_conversation: true,    // Stay active after response
    auto_end_timeout_ms: 30000,      // End after 30s inactivity
}
```

### Audio Settings

- **Sample Rate**: 16kHz (optimal for Whisper)
- **VAD Threshold**: Configurable
- **Buffer Size**: 160ms chunks (matches USM encoder)

## UI Integration

### ConversationStatus Component

```typescript
// Visual feedback for conversation state
<ConversationStatus />

// Shows:
// - Current mode (idle, listening, thinking, speaking)
// - Wake word detection pulse
// - Audio level visualization
// - Transcribed text
// - AI response preview
```

### Events

Frontend can listen to these events:
- `conversation-wake-word-detected`
- `conversation-user-input`
- `conversation-ai-responding`
- `conversation-ended`
- `conversation-user-interrupted`
- `tts-speaking-started`
- `tts-speaking-finished`

## Usage Examples

### Basic Conversation
```
User: "Tektra"
Tektra: [listening chime]
User: "What can you help me with?"
Tektra: "I can help you with many things! I can answer questions, help with analysis..."
```

### Continuous Conversation
```
User: "Tektra, tell me about quantum computing"
Tektra: "Quantum computing is a type of computation that..."
User: "How is it different from classical computing?" (no wake word)
Tektra: "The main differences are..."
User: "Thanks"
Tektra: "You're welcome! Is there anything else?"
[30 seconds pass]
[Conversation ends automatically]
```

### Interruption
```
User: "Tektra, explain machine learning"
Tektra: "Machine learning is a subset of artificial intelligence that..."
User: "Actually, tell me about deep learning instead"
Tektra: [stops speaking] "Deep learning is a specialized form of machine learning..."
```

## Future Enhancements

### When Gemma 3N Gets Native Audio Support in Ollama

Currently, we process audio through Whisper → Text → Gemma 3N → Text → TTS. When Ollama adds native audio support for Gemma 3N:

1. **Direct Audio Input**: Send audio directly to Gemma 3N
2. **Audio Output**: Generate audio responses directly
3. **Emotion Recognition**: Understand tone and emotion from voice
4. **Voice Cloning**: Match user's speaking style
5. **Non-Verbal Understanding**: Process laughs, sighs, etc.

### Planned Features

1. **Multi-User Support**: Recognize different speakers
2. **Emotion-Aware Responses**: Adjust tone based on context
3. **Background Noise Handling**: Better performance in noisy environments
4. **Custom Wake Words**: Allow users to set their own wake word
5. **Voice Profiles**: Save user preferences and voice characteristics

## API Reference

### Rust Commands

```rust
// Start always-listening mode
start_always_listening() -> Result<()>

// Get current conversation mode
get_conversation_mode() -> Result<String>

// End conversation manually
end_conversation() -> Result<()>

// Speak text with TTS
speak_text(text: String) -> Result<()>

// Stop current speech
stop_speaking() -> Result<()>
```

### TypeScript/JavaScript

```typescript
// Start always-listening
await invoke('start_always_listening');

// Listen for wake word
await listen('conversation-wake-word-detected', (event) => {
  console.log('Wake word detected!');
});

// Get conversation mode
const mode = await invoke('get_conversation_mode');
```

## Troubleshooting

### Common Issues

1. **"Tektra" not recognized**
   - Speak clearly and naturally
   - Ensure microphone is working
   - Check Whisper model is loaded

2. **Conversation ends too quickly**
   - Adjust `turn_timeout_ms` for longer pauses
   - Speak within the timeout window

3. **TTS not working**
   - Check system audio permissions
   - Verify TTS backend is available
   - Try different voice settings

4. **High CPU usage**
   - Adjust VAD sensitivity
   - Increase processing loop delay
   - Use smaller Whisper model

## Best Practices

1. **Natural Speech**: Talk to Tektra like you would a person
2. **Clear Commands**: After wake word, pause briefly then speak clearly
3. **Interruptions**: Don't wait for Tektra to finish if you need to correct
4. **Context**: Leverage continuous conversation for follow-up questions
5. **Privacy**: Whisper runs locally, audio is not sent to cloud

## Privacy & Security

- **Local Processing**: All audio processing happens on device
- **No Cloud Storage**: Audio is not uploaded or stored
- **Wake Word Only**: Only processes audio after wake word detection
- **User Control**: Can disable always-listening at any time
- **Secure Pipeline**: Audio data is processed in memory only