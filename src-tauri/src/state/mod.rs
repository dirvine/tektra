use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::AppHandle;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::ai::ModelManager;
use crate::audio::AudioManager;
use crate::vision::VisionManager;
use crate::robot::RobotController;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: i64,
}

pub struct AppState {
    app_handle: AppHandle,
    model_manager: Arc<RwLock<ModelManager>>,
    audio_manager: Arc<RwLock<AudioManager>>,
    vision_manager: Arc<RwLock<VisionManager>>,
    robot_controller: Arc<RwLock<RobotController>>,
    conversation_history: Arc<RwLock<Vec<Message>>>,
}

impl AppState {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        Ok(Self {
            app_handle: app_handle.clone(),
            model_manager: Arc::new(RwLock::new(ModelManager::new(app_handle.clone())?)),
            audio_manager: Arc::new(RwLock::new(AudioManager::new()?)),
            vision_manager: Arc::new(RwLock::new(VisionManager::new()?)),
            robot_controller: Arc::new(RwLock::new(RobotController::new()?)),
            conversation_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    pub async fn process_message(&self, message: String) -> Result<String> {
        // Add message to history
        let mut history = self.conversation_history.write().await;
        history.push(Message {
            role: "user".to_string(),
            content: message.clone(),
            timestamp: chrono::Utc::now().timestamp(),
        });
        
        // Process through AI model
        let model_manager = self.model_manager.read().await;
        let response = model_manager.generate_response(&message, &*history).await?;
        
        // Add response to history
        history.push(Message {
            role: "assistant".to_string(),
            content: response.clone(),
            timestamp: chrono::Utc::now().timestamp(),
        });
        
        // Check if response contains robot actions
        if response.contains("<ACTION>") {
            self.process_robot_action(&response).await?;
        }
        
        Ok(response)
    }
    
    pub async fn start_voice_capture(&self) -> Result<()> {
        let mut audio_manager = self.audio_manager.write().await;
        audio_manager.start_recording().await
    }
    
    pub async fn stop_voice_capture(&self) -> Result<()> {
        let mut audio_manager = self.audio_manager.write().await;
        audio_manager.stop_recording().await
    }
    
    pub async fn capture_camera_frame(&self) -> Result<Vec<u8>> {
        let vision_manager = self.vision_manager.read().await;
        vision_manager.capture_frame().await
    }
    
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        let mut model_manager = self.model_manager.write().await;
        model_manager.load_model(model_name).await
    }
    
    pub async fn get_model_status(&self) -> serde_json::Value {
        let model_manager = self.model_manager.read().await;
        model_manager.get_status().await
    }
    
    pub async fn download_model(&self, model_name: &str, force: bool) -> Result<serde_json::Value> {
        let model_manager = self.model_manager.read().await;
        model_manager.download_model(model_name, force).await
    }
    
    pub async fn list_cached_models(&self) -> Result<serde_json::Value> {
        let model_manager = self.model_manager.read().await;
        model_manager.list_cached_models().await
    }
    
    async fn process_robot_action(&self, response: &str) -> Result<()> {
        // Extract action tokens from response
        if let Some(start) = response.find("<ACTION>") {
            if let Some(end) = response.find("</ACTION>") {
                let action_str = &response[start + 8..end];
                let mut robot_controller = self.robot_controller.write().await;
                robot_controller.execute_action(action_str).await?;
            }
        }
        Ok(())
    }
}