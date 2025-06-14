# Tektra: Voice-Interactive Robotics AI Assistant

## Overview
**Tektra** is an advanced AI system that merges voice, vision, and action understanding using cutting-edge open models to enable real-time multimodal interaction and robotic control.

Tektra integrates:
- **[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)**: A compact, multimodal foundation model capable of processing text, images, audio, and outputting both speech and text.
- **[Pi0 (Physical Intelligence)](https://huggingface.co/lerobot/pi0fast_base)**: A vision-language-action model trained across real-world robot tasks using FAST (frequency-space action tokens).
- **[FAST Processor](https://huggingface.co/physical-intelligence/fast)**: Used for tokenising and decoding action sequences.

---

## Objectives
1. 🎤 Enable real-time **voice input** and **natural speech output**
2. 🤖 Support **robot action control** via FAST action tokens
3. 📷 Integrate **camera input** as part of perception
4. 🔄 Build a **continuous fine-tuning** pipeline using LoRA for action data
5. 📝 Implement logging and optional local/offline operation
6. 🧠 Package as a self-contained, `uv`-runnable Python script

---

## Architecture
```
Voice / Camera Input
      ↓
 Qwen2.5-Omni-7B (w/ Pi0 actions)
      ↓
Text / Audio / <ACTION> FAST tokens
      ↓
Speech Reply or Robotic Execution
```

- Voice is captured with `sounddevice`
- Camera input is taken via OpenCV
- Model response is inspected to determine:
  - `Text/Speech` → Natural reply
  - `<ACTION>` → Decode and execute as robot control sequence

---

## Merging Qwen with Pi0 Action Generation
Tektra combines the **perception and dialogue capabilities** of Qwen2.5-Omni-7B with the **task-oriented action generation** of Pi0 using these strategies:

### 🔧 Token Integration
- Extract the FAST token vocabulary used by Pi0
- Merge this into Qwen's tokenizer to unify output space
- Extend Qwen's output head to support prediction of FAST action tokens alongside text

### 🧠 Input Format
- Perception (image/audio) and current robot state are passed to Qwen as part of structured messages
- Qwen is instructed via system prompt to decide whether to respond with voice or action tokens

### 🔄 Multi-Mode Output Handling
- If the generated response begins with `<ACTION>`, Tektra parses the sequence as discrete FAST tokens
- These are decoded into control signals for the robot's actuation layer

---

## Action Data Collection Strategy
To continuously improve Tektra’s robotics performance, action data must be collected during real-world or simulated operation.

### 🎥 Episode Logging
- Each robot interaction (prompt, perception, state, and resulting action) is stored as an episode in `robot_episodes.json`
- Episodes include:
  ```json
  {
    "image": "frame.jpg",
    "instruction": "Pick up the red mug",
    "state": [0.2, 0.5, 0.1],
    "actions": [32, 44, 67, 18]  // FAST tokens
  }
  ```

### 🧠 Fine-Tuning
- These episodes are used to fine-tune Tektra using LoRA adapters
- Periodic training (e.g. nightly or on-demand) updates Tektra’s action policy

---

## Outputting Actions to Real Robots
Tektra will support direct connection to robotic systems through a modular control backend.

### 🔌 Output Interface Options
- UART, I2C or GPIO (e.g. Raspberry Pi)
- ROS 2 bridge (for simulation or real robot arms)
- MQTT or WebSocket (for networked robot platforms)

### 🧾 Action Execution Flow
1. Model generates FAST token sequence
2. A decoder maps tokens to robot-specific instructions (e.g. joint angles, cartesian moves)
3. Robot controller receives and executes the sequence
4. Result and optional sensory feedback can be logged for later replay/fine-tuning

---

## Features
### ✅ Default Behaviour
- User speaks
- Tektra replies with voice or emits `<ACTION>` FAST tokens

### 📸 Camera Integration
- Snapshots are taken at runtime and passed into the model

### 🔁 Fine-Tuning Pipeline
- Uses Hugging Face `Trainer` + `peft.LoRA` to adapt Qwen with robotics data
- Supports adapter-based continual learning

### 🧪 Inference Path
- Audio + camera + robot state → Model → Action or voice response

### 📂 Logging
- All conversations and outputs are stored in `tektra_log.txt`
- Timestamps and metadata included

### 🖥️ CLI Interface
Run Tektra via:
```bash
uv run tektra.py --chat        # Voice chat + action
uv run tektra.py --fine-tune   # Launch fine-tuning job
uv run tektra.py --menu        # Interactive terminal menu
```

---

## File Breakdown
| File             | Description                                           |
|------------------|-------------------------------------------------------|
| `tektra.py`      | Main controller: CLI, menu, voice/chat, fine-tuning   |
| `tektra_log.txt` | Interaction log file with timestamps                  |
| `models/tektra`  | Downloaded models and tokenizer state                 |
| `data/robot_episodes.json` | Fine-tuning dataset for robotics actions   |

---

## Next Steps
- [ ] Add FAST token decoding and robotic control hooks
- [ ] Integrate action simulator or real robot bridge (e.g., ROS, UART)
- [ ] Build optional macOS `.app` launcher
- [ ] Optionally bundle in local voice synthesis using `TTS` models

---

## Notes
- Designed for MacBook Pro (Apple Silicon, 96 GB RAM)
- Offline-capable and open-source model compliant
- Future integration possible with SAFENetwork or decentralised control

---

## Summary
Tektra bridges the gap between human-like interaction and real-world robotic control using an efficient open-model stack. Its modular architecture ensures future extensibility, continual learning, and deep integration with perception and actuation hardware.


