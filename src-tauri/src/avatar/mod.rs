use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvatarState {
    pub expression: String,
    pub mouth_open: f32, // 0.0 to 1.0 for lip sync
    pub eye_blink: f32,  // 0.0 to 1.0 for blinking
    pub head_tilt: f32,  // -1.0 to 1.0 for head movement
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipSyncFrame {
    pub timestamp: f32,
    pub mouth_open: f32,
}

pub struct AvatarManager {
    app_handle: AppHandle,
    current_state: Arc<Mutex<AvatarState>>,
    is_speaking: Arc<Mutex<bool>>,
}

impl AvatarManager {
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            app_handle,
            current_state: Arc::new(Mutex::new(AvatarState {
                expression: "neutral".to_string(),
                mouth_open: 0.0,
                eye_blink: 0.0,
                head_tilt: 0.0,
            })),
            is_speaking: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn set_expression(&self, expression: &str) -> anyhow::Result<()> {
        let mut state = self.current_state.lock().unwrap();
        state.expression = expression.to_string();
        
        // Emit state change
        let _ = self.app_handle.emit_all("avatar-state-changed", state.clone());
        
        info!("Avatar expression changed to: {}", expression);
        Ok(())
    }

    pub async fn start_speaking(&self, text: &str) -> anyhow::Result<Vec<LipSyncFrame>> {
        *self.is_speaking.lock().unwrap() = true;
        
        // Generate basic lip sync data based on text
        // In a real implementation, this would analyze phonemes
        let lip_sync_data = self.generate_basic_lip_sync(text);
        
        // Start lip sync animation
        self.animate_lip_sync(lip_sync_data.clone()).await;
        
        Ok(lip_sync_data)
    }

    pub async fn stop_speaking(&self) -> anyhow::Result<()> {
        *self.is_speaking.lock().unwrap() = false;
        
        // Close mouth
        let mut state = self.current_state.lock().unwrap();
        state.mouth_open = 0.0;
        let _ = self.app_handle.emit_all("avatar-state-changed", state.clone());
        
        Ok(())
    }

    fn generate_basic_lip_sync(&self, text: &str) -> Vec<LipSyncFrame> {
        let mut frames = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_time = 0.0;
        
        for word in words {
            // Simple approximation: each word takes about 0.3 seconds
            let word_duration = 0.3;
            
            // Open mouth for vowels
            for (i, char) in word.chars().enumerate() {
                let char_time = current_time + (i as f32 * word_duration / word.len() as f32);
                
                let mouth_open = match char.to_lowercase().next().unwrap_or(' ') {
                    'a' | 'e' | 'i' | 'o' | 'u' => 0.8,
                    'w' | 'm' | 'b' | 'p' => 0.3,
                    _ => 0.1,
                };
                
                frames.push(LipSyncFrame {
                    timestamp: char_time,
                    mouth_open,
                });
            }
            
            current_time += word_duration + 0.1; // Add pause between words
        }
        
        // Close mouth at the end
        frames.push(LipSyncFrame {
            timestamp: current_time,
            mouth_open: 0.0,
        });
        
        frames
    }

    async fn animate_lip_sync(&self, frames: Vec<LipSyncFrame>) {
        let state_clone = Arc::clone(&self.current_state);
        let app_handle_clone = self.app_handle.clone();
        let is_speaking_clone = Arc::clone(&self.is_speaking);
        
        tokio::spawn(async move {
            let start_time = std::time::Instant::now();
            
            for frame in frames {
                if !*is_speaking_clone.lock().unwrap() {
                    break;
                }
                
                // Wait until the frame's timestamp
                let elapsed = start_time.elapsed().as_secs_f32();
                if frame.timestamp > elapsed {
                    let wait_time = frame.timestamp - elapsed;
                    tokio::time::sleep(tokio::time::Duration::from_secs_f32(wait_time)).await;
                }
                
                // Update mouth position
                let mut state = state_clone.lock().unwrap();
                state.mouth_open = frame.mouth_open;
                let _ = app_handle_clone.emit_all("avatar-state-changed", state.clone());
            }
        });
    }

    pub async fn blink(&self) -> anyhow::Result<()> {
        let state_clone = Arc::clone(&self.current_state);
        let app_handle_clone = self.app_handle.clone();
        
        tokio::spawn(async move {
            // Close eyes
            {
                let mut state = state_clone.lock().unwrap();
                state.eye_blink = 1.0;
                let _ = app_handle_clone.emit_all("avatar-state-changed", state.clone());
            } // state lock is dropped here
            
            // Keep closed for 150ms
            tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
            
            // Open eyes
            {
                let mut state = state_clone.lock().unwrap();
                state.eye_blink = 0.0;
                let _ = app_handle_clone.emit_all("avatar-state-changed", state.clone());
            }
        });
        
        Ok(())
    }

    pub fn get_current_state(&self) -> AvatarState {
        self.current_state.lock().unwrap().clone()
    }
}

// Simple 2D avatar implementation notes:
// For the frontend, we can use:
// 1. Canvas-based rendering with bezier curves for smooth animations
// 2. SVG-based avatar with CSS animations
// 3. Pre-rendered sprite sheets for different expressions
// 4. Integration with libraries like:
//    - Live2D (2D avatar animation)
//    - Three.js with morph targets (3D option)
//    - Rive (interactive animations)

// For lip sync, we could enhance this with:
// 1. Phoneme detection from text
// 2. Audio amplitude analysis for more accurate mouth movements
// 3. Viseme mapping (visual phonemes)
// 4. Machine learning models for better lip sync