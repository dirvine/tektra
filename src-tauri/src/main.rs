// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::Path;
use tauri::{Manager, State, Emitter};
use tokio::sync::Mutex;
use tracing::{info, error, warn};

mod ai;
// New modular architecture imports
mod inference;
mod multimodal;
mod conversation;
mod voice;

use inference::{ModelRegistry as InferenceModelRegistry, /* QwenOmniModel, */ MultimodalInput, ModelResponse};
use conversation::ConversationManager;
use voice::{VoicePipeline, VoiceCharacterConfig, VoicePipelineConfig, VoicePipelineEvent, UnmuteServiceManager, UnmuteConfig, ServiceEvent};
mod audio;
use audio::AudioRecorder;
mod vision;
use vision::VisionManager;
mod avatar;
use avatar::AvatarManager;
mod cli;
mod config;
use config::AppConfig;
mod vector_db;
use vector_db::VectorDB;
mod database;
use database::{Database, Document};
mod types;
use types::{ChatMessage, AppSettings};


type ChatHistory = Arc<Mutex<Vec<ChatMessage>>>;
type Settings = Arc<Mutex<AppSettings>>;
type ModelRegistry = Arc<Mutex<InferenceModelRegistry>>;
type ConversationMgr = Arc<Mutex<ConversationManager>>;
type AudioRec = Arc<Mutex<AudioRecorder>>;
type Vision = Arc<Mutex<VisionManager>>;
type Avatar = Arc<Mutex<AvatarManager>>;
type VectorStore = Arc<Mutex<VectorDB>>;
type DB = Arc<Database>;
type VoicePipelineMgr = Arc<Mutex<Option<VoicePipeline>>>;
type UnmuteServiceMgr = Arc<Mutex<Option<UnmuteServiceManager>>>;

#[tauri::command]
async fn start_audio_recording(audio: State<'_, AudioRec>) -> Result<bool, String> {
    let recorder = audio.lock().await;
    match recorder.start_recording().await {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Failed to start recording: {}", e)),
    }
}

#[tauri::command]
async fn stop_audio_recording(audio: State<'_, AudioRec>) -> Result<Vec<f32>, String> {
    let recorder = audio.lock().await;
    match recorder.stop_recording().await {
        Ok(audio_data) => Ok(audio_data),
        Err(e) => Err(format!("Failed to stop recording: {}", e)),
    }
}

#[tauri::command]
async fn is_recording(audio: State<'_, AudioRec>) -> Result<bool, String> {
    let recorder = audio.lock().await;
    Ok(recorder.is_recording())
}

#[tauri::command]
async fn process_audio_stream(audio: State<'_, AudioRec>) -> Result<(), String> {
    let mut recorder = audio.lock().await;
    match recorder.process_audio_stream().await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to process audio stream: {}", e)),
    }
}

#[tauri::command]
async fn initialize_whisper(audio: State<'_, AudioRec>) -> Result<bool, String> {
    let mut recorder = audio.lock().await;
    match recorder.initialize_whisper().await {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Failed to initialize Whisper: {}", e)),
    }
}

#[tauri::command]
async fn initialize_model(model_registry: State<'_, ModelRegistry>) -> Result<bool, String> {
    let registry = model_registry.lock().await;
    
    // Initialize the registry and load available models
    match registry.initialize().await {
        Ok(_) => {
            info!("Successfully initialized model registry with available models");
            Ok(true)
        }
        Err(e) => {
            error!("Failed to initialize model registry: {}", e);
            Err(format!("Failed to initialize models: {}", e))
        }
    }
}

#[tauri::command]
async fn send_message(
    message: String,
    chat_history: State<'_, ChatHistory>,
    model_registry: State<'_, ModelRegistry>,
    conversation_mgr: State<'_, ConversationMgr>,
    settings: State<'_, Settings>,
    vector_store: State<'_, VectorStore>,
) -> Result<String, String> {
    // Add user message to history
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: message.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(user_msg);

    // Get settings
    let settings_guard = settings.lock().await;
    let max_tokens = settings_guard.max_tokens;
    let system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Create multimodal input
    let input = MultimodalInput::Text(message.clone());

    // Generate response using new model registry
    let registry = model_registry.lock().await;
    
    let response = if registry.get_active_model_id().await.is_some() {
        match registry.generate(input).await {
            Ok(model_response) => model_response.text,
            Err(e) => {
                eprintln!("Error generating response: {}", e);
                format!("I apologize, but I encountered an error: {}. Please try again.", e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    
    drop(registry);
    
    // Add assistant response to history
    let assistant_msg = ChatMessage {
        role: "assistant".to_string(),
        content: response.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(assistant_msg);
    
    Ok(response)
}

#[tauri::command]
async fn send_message_with_camera(
    message: String,
    chat_history: State<'_, ChatHistory>,
    model_registry: State<'_, ModelRegistry>,
    conversation_mgr: State<'_, ConversationMgr>,
    vision: State<'_, Vision>,
    settings: State<'_, Settings>,
    vector_store: State<'_, VectorStore>,
) -> Result<String, String> {
    // Add user message to history
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: message.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(user_msg);

    // Get settings
    let settings_guard = settings.lock().await;
    let max_tokens = settings_guard.max_tokens;
    let system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Capture camera frame if camera is active
    let vision_manager = vision.lock().await;
    let frame_data: Option<Vec<u8>> = if vision_manager.is_capturing() {
        match vision_manager.capture_frame().await {
            Ok(frame) => {
                // Convert RGB to raw bytes for AI processing
                Some(frame.data)
            }
            Err(e) => {
                eprintln!("Failed to capture camera frame: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Generate response using new model registry
    let registry = model_registry.lock().await;
    
    let response = if registry.get_active_model_id().await.is_some() {
        if let Some(image_data) = frame_data {
            // Use vision-enabled response generation
            let input = MultimodalInput::TextWithImage {
                text: message.clone(),
                image: crate::inference::ImageData {
                    data: image_data,
                    format: crate::inference::ImageFormat::Jpeg,
                    width: Some(640),
                    height: Some(480),
                }
            };
            match registry.generate(input).await {
                Ok(model_response) => model_response.text,
                Err(e) => {
                    eprintln!("Error generating response with image: {}", e);
                    format!("I apologize, but I encountered an error processing the image: {}. Please try again.", e)
                }
            }
        } else {
            // Fall back to text-only response if no camera
            let input = MultimodalInput::Text(message.clone());
            match registry.generate(input).await {
                Ok(model_response) => model_response.text,
                Err(e) => {
                    eprintln!("Error generating response: {}", e);
                    format!("I apologize, but I encountered an error: {}. Please try again.", e)
                }
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    
    drop(registry);
    
    // Add assistant response to history
    let assistant_msg = ChatMessage {
        role: "assistant".to_string(),
        content: response.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(assistant_msg);
    
    Ok(response)
}

#[tauri::command]
async fn get_chat_history(chat_history: State<'_, ChatHistory>) -> Result<Vec<ChatMessage>, String> {
    Ok(chat_history.lock().await.clone())
}

#[tauri::command]
async fn clear_chat_history(chat_history: State<'_, ChatHistory>) -> Result<(), String> {
    chat_history.lock().await.clear();
    Ok(())
}

#[tauri::command]
async fn get_settings(settings: State<'_, Settings>) -> Result<AppSettings, String> {
    Ok(settings.lock().await.clone())
}

#[tauri::command]
async fn update_settings(
    new_settings: AppSettings,
    settings: State<'_, Settings>,
) -> Result<(), String> {
    *settings.lock().await = new_settings;
    Ok(())
}

#[tauri::command]
async fn check_model_status(model_registry: State<'_, ModelRegistry>) -> Result<bool, String> {
    let registry = model_registry.lock().await;
    Ok(registry.get_active_model_id().await.is_some())
}

#[tauri::command]
async fn get_available_models(model_registry: State<'_, ModelRegistry>) -> Result<Vec<serde_json::Value>, String> {
    use serde_json::json;
    
    let registry = model_registry.lock().await;
    let models = registry.list_models().await;
    
    // Convert to detailed model information
    let model_info: Vec<serde_json::Value> = models.into_iter()
        .map(|model| json!({
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "supports_vision": model.supports_vision,
            "supports_audio": model.supports_audio,
            "supports_documents": model.supports_documents,
            "context_window": model.context_window,
            "quantization": model.quantization,
            "model_id": model.model_id,
            "default": model.default,
            "recommended_for": model.recommended_for
        }))
        .collect();
    
    Ok(model_info)
}

#[tauri::command]
async fn get_model_capabilities(model_name: String, model_registry: State<'_, ModelRegistry>) -> Result<serde_json::Value, String> {
    use serde_json::json;
    
    let registry = model_registry.lock().await;
    let models = registry.list_models().await;
    
    // Find the specific model
    if let Some(model) = models.iter().find(|m| m.id == model_name || m.name == model_name) {
        let recommended_for = if model.supports_audio && model.supports_vision {
            vec!["multimodal conversation", "real-time interaction", "voice + vision", "comprehensive AI assistance"]
        } else if model.supports_vision {
            vec!["image analysis", "visual question answering", "OCR", "document analysis"]
        } else if model.supports_audio {
            vec!["voice interaction", "speech processing", "audio analysis"]
        } else {
            vec!["text generation", "conversation", "question answering", "code generation"]
        };
        
        Ok(json!({
            "model": model.name,
            "model_id": model.model_id,
            "supports_text": true,
            "supports_vision": model.supports_vision,
            "supports_audio": model.supports_audio,
            "supports_video": false, // Not in DefaultModelConfig yet
            "supports_documents": model.supports_documents,
            "supports_real_time": model.id.contains("omni"), // Omni models support real-time
            "thinker_talker_architecture": model.id.contains("omni"),
            "context_window": model.context_window,
            "quantization": model.quantization,
            "description": model.description,
            "recommended_for": recommended_for
        }))
    } else {
        Err(format!("Model '{}' not found", model_name))
    }
}

#[tauri::command]
async fn load_model(model_id: String, model_registry: State<'_, ModelRegistry>, app: tauri::AppHandle) -> Result<String, String> {
    info!("Loading model: {}", model_id);
    
    // Emit progress start
    let _ = app.emit("model-loading-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 0,
        "status": "Starting model load...",
        "stage": "initializing"
    }));
    
    let registry = model_registry.lock().await;
    
    // Emit progress for model discovery
    let _ = app.emit("model-loading-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 10,
        "status": "Checking model availability...",
        "stage": "checking"
    }));
    
    // Actually attempt to load the model through the registry
    match registry.load_model(&model_id).await {
        Ok(_) => {
            // Emit progress for activation
            let _ = app.emit("model-loading-progress", serde_json::json!({
                "model_id": model_id,
                "progress": 80,
                "status": "Activating model...",
                "stage": "activating"
            }));
            
            // Set as active model
            match registry.set_active_model(&model_id).await {
                Ok(_) => {
                    // Emit completion
                    let _ = app.emit("model-loading-progress", serde_json::json!({
                        "model_id": model_id,
                        "progress": 100,
                        "status": "Model loaded successfully",
                        "stage": "complete"
                    }));
                    
                    info!("Successfully loaded and activated model: {}", model_id);
                    Ok(format!("Model '{}' loaded successfully", model_id))
                }
                Err(e) => {
                    // Emit error
                    let _ = app.emit("model-loading-progress", serde_json::json!({
                        "model_id": model_id,
                        "progress": 0,
                        "status": format!("Failed to activate: {}", e),
                        "stage": "error"
                    }));
                    
                    error!("Failed to set active model {}: {}", model_id, e);
                    Err(format!("Failed to activate model '{}': {}", model_id, e))
                }
            }
        }
        Err(e) => {
            // Emit error
            let _ = app.emit("model-loading-progress", serde_json::json!({
                "model_id": model_id,
                "progress": 0,
                "status": format!("Failed to load: {}", e),
                "stage": "error"
            }));
            
            error!("Failed to load model {}: {}", model_id, e);
            Err(format!("Failed to load model '{}': {}", model_id, e))
        }
    }
}

#[tauri::command]
async fn get_current_model(model_registry: State<'_, ModelRegistry>) -> Result<Option<String>, String> {
    let registry = model_registry.lock().await;
    
    if let Some(active_id) = registry.get_active_model_id().await {
        // Get the available models to find the display name
        let models = registry.list_models().await;
        let current_model = models.iter().find(|m| m.id == active_id);
        
        Ok(Some(current_model.map(|m| m.name.clone()).unwrap_or(active_id)))
    } else {
        Ok(None)
    }
}

// Conversation commands
#[tauri::command]
async fn start_always_listening(audio: State<'_, AudioRec>) -> Result<(), String> {
    let audio_recorder = audio.lock().await;
    match audio_recorder.start_always_listening().await {
        Ok(_) => {
            info!("Always-listening mode started");
            Ok(())
        }
        Err(e) => Err(format!("Failed to start always-listening: {}", e)),
    }
}

#[tauri::command]
async fn get_conversation_mode(audio: State<'_, AudioRec>) -> Result<String, String> {
    let audio_recorder = audio.lock().await;
    let conversation_manager = audio_recorder.get_conversation_manager();
    let mode = conversation_manager.get_mode().await;
    Ok(format!("{:?}", mode))
}

#[tauri::command]
async fn end_conversation(audio: State<'_, AudioRec>) -> Result<(), String> {
    let audio_recorder = audio.lock().await;
    let conversation_manager = audio_recorder.get_conversation_manager();
    match conversation_manager.end_conversation().await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to end conversation: {}", e)),
    }
}

#[tauri::command]
async fn speak_text(text: String, audio: State<'_, AudioRec>) -> Result<(), String> {
    let audio_recorder = audio.lock().await;
    let tts_manager = audio_recorder.get_tts_manager();
    match tts_manager.speak(&text).await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to speak text: {}", e)),
    }
}

#[tauri::command]
async fn stop_speaking(audio: State<'_, AudioRec>) -> Result<(), String> {
    let audio_recorder = audio.lock().await;
    let tts_manager = audio_recorder.get_tts_manager();
    match tts_manager.stop_speaking().await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to stop speaking: {}", e)),
    }
}

// Camera commands
#[tauri::command]
async fn initialize_camera(_vision: State<'_, Vision>) -> Result<bool, String> {
    // Vision features are being developed with mistral.rs backend
    warn!("Camera initialization requested but vision features are still in development");
    Err("Vision features are currently in development. Please wait for future updates with full multimodal support.".to_string())
}

#[tauri::command]
async fn start_camera_capture(vision: State<'_, Vision>) -> Result<bool, String> {
    let vision_manager = vision.lock().await;
    match vision_manager.start_capture().await {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Failed to start camera capture: {}", e)),
    }
}

#[tauri::command]
async fn stop_camera_capture(vision: State<'_, Vision>) -> Result<bool, String> {
    let vision_manager = vision.lock().await;
    match vision_manager.stop_capture().await {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Failed to stop camera capture: {}", e)),
    }
}

#[tauri::command]
async fn get_camera_frame(vision: State<'_, Vision>) -> Result<String, String> {
    let vision_manager = vision.lock().await;
    match crate::vision::get_camera_frame_base64(&*vision_manager).await {
        Ok(frame) => Ok(frame),
        Err(e) => Err(format!("Failed to get camera frame: {}", e)),
    }
}

// Multimodal input commands
#[tauri::command]
async fn process_image_input(
    message: String,
    image_data: Vec<u8>,
    chat_history: State<'_, ChatHistory>,
    model_registry: State<'_, ModelRegistry>,
    conversation_mgr: State<'_, ConversationMgr>,
    settings: State<'_, Settings>,
) -> Result<String, String> {
    // Add user message to history with image indicator
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: format!("{} [Image attached]", message),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(user_msg);

    // Get settings
    let settings_guard = settings.lock().await;
    let max_tokens = settings_guard.max_tokens;
    let system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Generate response using new model registry with image
    let registry = model_registry.lock().await;
    
    let response = if registry.get_active_model_id().await.is_some() {
        let input = MultimodalInput::TextWithImage {
            text: message.clone(),
            image: crate::inference::ImageData {
                data: image_data,
                format: crate::inference::ImageFormat::Jpeg,
                width: None,
                height: None,
            }
        };
        match registry.generate(input).await {
            Ok(model_response) => model_response.text,
            Err(e) => {
                eprintln!("Error generating response with image: {}", e);
                format!("I can see the image you've shared, but I'm still learning to process visual information. Error: {}", e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    
    drop(registry);
    
    // Add assistant response to history
    let assistant_msg = ChatMessage {
        role: "assistant".to_string(),
        content: response.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(assistant_msg);
    
    Ok(response)
}

#[tauri::command]
async fn process_camera_feed(
    vision: State<'_, Vision>,
    model_registry: State<'_, ModelRegistry>,
    conversation_mgr: State<'_, ConversationMgr>,
    settings: State<'_, Settings>,
    chat_history: State<'_, ChatHistory>,
) -> Result<(), String> {
    let vision_manager = vision.lock().await;
    if !vision_manager.is_capturing() {
        return Err("Camera is not capturing".to_string());
    }

    let vision_processor = crate::vision::VisionProcessor::new_default().unwrap();

    loop {
        let vision_check = vision.lock().await;
        if !vision_check.is_capturing() {
            drop(vision_check);
            break;
        }
        drop(vision_check);

        match vision_manager.capture_frame().await {
            Ok(frame) => {
                match vision_processor.process_camera_frame(&frame) {
                    Ok(_) => {
                        let registry = model_registry.lock().await;
                        let settings_guard = settings.lock().await;
                        let max_tokens = settings_guard.max_tokens;
                        let system_prompt = settings_guard.system_prompt.clone();
                        drop(settings_guard);
                        
                        let response = if registry.get_active_model_id().await.is_some() {
                            let input = MultimodalInput::TextWithImage {
                                text: "Describe what you see.".to_string(),
                                image: crate::inference::ImageData {
                                    data: frame.data.clone(),
                                    format: crate::inference::ImageFormat::Jpeg,
                                    width: Some(640),
                                    height: Some(480),
                                }
                            };
                            match registry.generate(input).await {
                                Ok(model_response) => model_response.text,
                                Err(e) => {
                                    eprintln!("Error generating response with image: {}", e);
                                    format!("I apologize, but I encountered an error processing the image: {}. Please try again.", e)
                                }
                            }
                        } else {
                            "The AI model is still loading. Please wait a moment and try again.".to_string()
                        };
                        
                        drop(registry);
                        
                        let assistant_msg = ChatMessage {
                            role: "assistant".to_string(),
                            content: response.clone(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        };
                        
                        chat_history.lock().await.push(assistant_msg);
                    }
                    Err(e) => {
                        eprintln!("Failed to process camera frame: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to capture camera frame: {}", e);
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    }

    Ok(())
}

#[tauri::command]
async fn process_audio_input(
    message: String,
    audio_data: Vec<u8>,
    chat_history: State<'_, ChatHistory>,
    model_registry: State<'_, ModelRegistry>,
    _conversation_mgr: State<'_, ConversationMgr>,
    audio: State<'_, AudioRec>,
    settings: State<'_, Settings>,
) -> Result<String, String> {
    info!("Processing audio input: {} bytes of audio data", audio_data.len());
    
    // Get conversation manager
    let audio_recorder = audio.lock().await;
    let conversation_manager = audio_recorder.get_conversation_manager();
    let tts_manager = audio_recorder.get_tts_manager();
    
    // Notify conversation manager that AI is processing
    if let Err(e) = conversation_manager.start_ai_response("Processing...").await {
        error!("Failed to update conversation state: {}", e);
    }
    
    // Add user message to history
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: format!("{} [Audio: {:.2}s]", message, audio_data.len() as f32 / (16000.0 * 2.0)), // Approximate duration
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(user_msg);

    // Get settings
    let settings_guard = settings.lock().await;
    let _max_tokens = settings_guard.max_tokens;
    let _system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Generate response using new model registry with audio data
    let registry = model_registry.lock().await;
    
    let response = if registry.get_active_model_id().await.is_some() {
        // Use multimodal generation with audio data
        let input = MultimodalInput::TextWithAudio {
            text: message.clone(),
            audio: crate::inference::AudioData {
                data: audio_data,
                format: crate::inference::AudioFormat::Wav,
                sample_rate: Some(16000),
                channels: Some(1),
                duration: None,
            }
        };
        match registry.generate(input).await {
            Ok(model_response) => model_response.text,
            Err(e) => {
                error!("Error generating response with audio: {}", e);
                format!("I heard your audio input but encountered an error processing it: {}. Please try again.", e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    
    drop(registry);
    
    // Clone response for use in multiple places
    let response_for_tts = response.clone();
    
    // Update conversation manager with AI response
    if let Err(e) = conversation_manager.start_ai_response(&response).await {
        error!("Failed to update conversation with AI response: {}", e);
    }
    
    // Speak the response using TTS
    tokio::spawn(async move {
        if let Err(e) = tts_manager.speak(&response_for_tts).await {
            error!("Failed to speak response: {}", e);
        }
        
        // Notify conversation manager that AI finished responding
        if let Err(e) = conversation_manager.end_ai_response().await {
            error!("Failed to mark end of AI response: {}", e);
        }
    });
    
    // Add assistant response to history
    let assistant_msg = ChatMessage {
        role: "assistant".to_string(),
        content: response.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(assistant_msg);
    
    Ok(response)
}

#[tauri::command]
async fn process_multimodal_input(
    message: String,
    image_data: Option<Vec<u8>>,
    audio_data: Option<Vec<u8>>,
    video_data: Option<Vec<u8>>,
    chat_history: State<'_, ChatHistory>,
    model_registry: State<'_, ModelRegistry>,
    conversation_mgr: State<'_, ConversationMgr>,
    vision: State<'_, Vision>,
    settings: State<'_, Settings>,
) -> Result<String, String> {
    // Create a description of the multimodal input
    let mut input_description = message.clone();
    let mut modality_count = 0;
    
    if image_data.is_some() {
        input_description.push_str(" [Image attached]");
        modality_count += 1;
    }
    if audio_data.is_some() {
        input_description.push_str(" [Audio attached]");
        modality_count += 1;
    }
    if video_data.is_some() {
        input_description.push_str(" [Video attached]");
        modality_count += 1;
    }
    
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: input_description,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(user_msg);

    // Get settings
    let settings_guard = settings.lock().await;
    let max_tokens = settings_guard.max_tokens;
    let system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Build conversation context from recent chat history
    let recent_history = {
        let history = chat_history.lock().await;
        let context_messages = history.iter()
            .rev()
            .take(6) // Last 3 exchanges (6 messages)
            .rev()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n");
        if context_messages.is_empty() { None } else { Some(context_messages) }
    };

    // Process comprehensive multimodal input
    let registry = model_registry.lock().await;
    let vision_manager = vision.lock().await;
    
    let response = if registry.get_active_model_id().await.is_some() {
        // Handle camera capture separately to avoid lifetime issues
        let camera_frame_data = if image_data.is_none() && vision_manager.is_capturing() {
            // Capture current frame for live camera analysis
            match vision_manager.capture_frame().await {
                Ok(frame) => Some(frame.data),
                Err(e) => {
                    eprintln!("Failed to capture camera frame: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        drop(vision_manager);
        
        // Determine final image data to use
        let final_image_data = if let Some(ref img_data) = image_data {
            Some(img_data.clone())
        } else if let Some(ref cam_data) = camera_frame_data {
            Some(cam_data.clone())
        } else {
            None
        };
        
        // Create comprehensive multimodal input
        let input = MultimodalInput::MultimodalConversation {
            text: Some(message.clone()),
            images: final_image_data.map(|data| vec![crate::inference::ImageData {
                data,
                format: crate::inference::ImageFormat::Jpeg,
                width: None,
                height: None,
            }]).unwrap_or_default(),
            audio: audio_data.map(|data| crate::inference::AudioData {
                data,
                format: crate::inference::AudioFormat::Wav,
                sample_rate: Some(16000),
                channels: Some(1),
                duration: None,
            }),
            video: video_data.map(|data| crate::inference::VideoData {
                data,
                format: crate::inference::VideoFormat::Mp4,
                width: None,
                height: None,
                fps: None,
                duration: None,
                frames: None,
            }),
            documents: vec![],
            real_time: false,
            conversation_context: None,
        };
        
        // Use comprehensive multimodal generation
        match registry.generate(input).await {
            Ok(model_response) => {
                info!("Generated comprehensive multimodal response with {} modalities", modality_count);
                model_response.text
            }
            Err(e) => {
                error!("Error generating comprehensive multimodal response: {}", e);
                format!("I can see your multimodal input ({} modalities), but encountered an error: {}. Please try again.", modality_count, e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    
    drop(registry);
    
    // Add assistant response to history
    let assistant_msg = ChatMessage {
        role: "assistant".to_string(),
        content: response.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    chat_history.lock().await.push(assistant_msg);
    
    Ok(response)
}

// Avatar commands
#[tauri::command]
async fn set_avatar_expression(avatar: State<'_, Avatar>, expression: String) -> Result<(), String> {
    let avatar_manager = avatar.lock().await;
    match avatar_manager.set_expression(&expression).await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to set avatar expression: {}", e)),
    }
}

#[tauri::command]
async fn start_avatar_speaking(avatar: State<'_, Avatar>, text: String) -> Result<Vec<crate::avatar::LipSyncFrame>, String> {
    let avatar_manager = avatar.lock().await;
    match avatar_manager.start_speaking(&text).await {
        Ok(frames) => Ok(frames),
        Err(e) => Err(format!("Failed to start avatar speaking: {}", e)),
    }
}

#[tauri::command]
async fn stop_avatar_speaking(avatar: State<'_, Avatar>) -> Result<(), String> {
    let avatar_manager = avatar.lock().await;
    match avatar_manager.stop_speaking().await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to stop avatar speaking: {}", e)),
    }
}

#[tauri::command]
async fn avatar_blink(avatar: State<'_, Avatar>) -> Result<(), String> {
    let avatar_manager = avatar.lock().await;
    match avatar_manager.blink().await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to blink avatar: {}", e)),
    }
}

#[tauri::command]
async fn get_backend_info(model_registry: State<'_, ModelRegistry>) -> Result<String, String> {
    let registry = model_registry.lock().await;
    if let Some(model_id) = registry.get_active_model_id().await {
        Ok(format!("Active model: {} (Qwen2.5-Omni via mistral.rs backend)", model_id))
    } else {
        Ok("No active model (mistral.rs backend ready)".to_string())
    }
}

#[tauri::command]
async fn benchmark_backends(
    model_registry: State<'_, ModelRegistry>,
    prompt: Option<String>,
    max_tokens: Option<usize>,
) -> Result<Vec<(String, f64)>, String> {
    let registry = model_registry.lock().await;
    let test_prompt = prompt.unwrap_or_else(|| "What is the capital of France?".to_string());
    let _tokens = max_tokens.unwrap_or(100);
    
    // Simple benchmark of current model
    if let Some(model_id) = registry.get_active_model_id().await {
        let start = std::time::Instant::now();
        let input = MultimodalInput::Text(test_prompt);
        match registry.generate(input).await {
            Ok(_) => {
                let elapsed = start.elapsed().as_secs_f64();
                Ok(vec![(model_id, elapsed)])
            }
            Err(e) => Err(format!("Failed to benchmark model: {}", e)),
        }
    } else {
        Err("No active model to benchmark".to_string())
    }
}

#[tauri::command]
async fn restart_backend(model_registry: State<'_, ModelRegistry>) -> Result<String, String> {
    info!("Attempting to restart mistral.rs backend...");
    
    let registry = model_registry.lock().await;
    
    // Reset the model registry - unload current models
    if let Some(model_id) = registry.get_active_model_id().await {
        info!("Unloading active model: {}", model_id);
        match registry.unload_model(&model_id).await {
            Ok(_) => {
                drop(registry);
                Ok(format!("Backend restarted successfully. Model '{}' unloaded.", model_id))
            }
            Err(e) => {
                drop(registry);
                Err(format!("Failed to restart backend: {}", e))
            }
        }
    } else {
        drop(registry);
        Ok("Backend restarted successfully. No models were loaded.".to_string())
    }
}

// Project management commands
#[tauri::command]
async fn create_project(name: String, description: Option<String>, db: State<'_, DB>) -> Result<serde_json::Value, String> {
    info!("Creating project: {}", name);
    
    let project = db.create_project(name, description).await
        .map_err(|e| format!("Failed to create project: {}", e))?;
    
    // Convert to JSON value
    serde_json::to_value(&project)
        .map_err(|e| format!("Failed to serialize project: {}", e))
}

#[tauri::command]
async fn get_projects(db: State<'_, DB>) -> Result<Vec<serde_json::Value>, String> {
    let projects = db.get_projects().await
        .map_err(|e| format!("Failed to get projects: {}", e))?;
    
    // Convert to JSON values
    projects.into_iter()
        .map(|p| serde_json::to_value(&p).map_err(|e| format!("Failed to serialize project: {}", e)))
        .collect()
}

#[tauri::command]
async fn delete_project(project_id: String, db: State<'_, DB>) -> Result<(), String> {
    info!("Deleting project: {}", project_id);
    db.delete_project(project_id).await
        .map_err(|e| format!("Failed to delete project: {}", e))
}

#[tauri::command]
async fn toggle_project_star(project_id: String, db: State<'_, DB>) -> Result<(), String> {
    info!("Toggling star for project: {}", project_id);
    db.toggle_project_star(project_id).await
        .map_err(|e| format!("Failed to toggle project star: {}", e))
}

#[tauri::command]
async fn get_project_documents(project_id: String, db: State<'_, DB>) -> Result<Vec<serde_json::Value>, String> {
    info!("Getting documents for project: {}", project_id);
    let documents = db.get_project_documents(project_id).await
        .map_err(|e| format!("Failed to get project documents: {}", e))?;
    
    // Convert to JSON values
    documents.into_iter()
        .map(|d| serde_json::to_value(&d).map_err(|e| format!("Failed to serialize document: {}", e)))
        .collect()
}

#[tauri::command]
async fn upload_file_to_project(
    project_id: String,
    file_path: String,
    file_name: String,
    db: State<'_, DB>,
    model_registry: State<'_, ModelRegistry>,
) -> Result<serde_json::Value, String> {
    info!("Uploading file {} to project {}", file_name, project_id);
    
    // Read file metadata
    let metadata = match std::fs::metadata(&file_path) {
        Ok(meta) => meta,
        Err(e) => return Err(format!("Failed to read file metadata: {}", e)),
    };
    
    let file_size = metadata.len();
    let document_id = uuid::Uuid::new_v4().to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| format!("Failed to get timestamp: {}", e))?
        .as_secs();
    
    // Determine file type based on extension
    let file_type = if file_name.ends_with(".jpg") || file_name.ends_with(".jpeg") || 
                       file_name.ends_with(".png") || file_name.ends_with(".gif") ||
                       file_name.ends_with(".webp") || file_name.ends_with(".bmp") {
        "image"
    } else if file_name.ends_with(".mp4") || file_name.ends_with(".avi") || 
              file_name.ends_with(".mov") || file_name.ends_with(".webm") {
        "video"
    } else if file_name.ends_with(".mp3") || file_name.ends_with(".wav") || 
              file_name.ends_with(".flac") || file_name.ends_with(".ogg") {
        "audio"
    } else if file_name.ends_with(".txt") || file_name.ends_with(".md") || 
              file_name.ends_with(".doc") || file_name.ends_with(".docx") ||
              file_name.ends_with(".pdf") {
        "text"
    } else {
        "other"
    };
    
    // Process file content if it's a text document
    let (content, embeddings) = if file_type == "text" {
        match process_text_file(&file_path, &file_name, &model_registry).await {
            Ok((content, embeddings)) => (Some(content), Some(embeddings)),
            Err(e) => {
                warn!("Failed to process text file: {}", e);
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    
    // Create document
    let document = Document {
        id: document_id,
        project_id: project_id.clone(),
        name: file_name,
        doc_type: file_type.to_string(),
        size: file_size,
        path: file_path,
        uploaded_at: now,
        tags: vec![],
        content,
        embeddings,
    };
    
    // Add to database
    let document = db.add_document(project_id, document).await
        .map_err(|e| format!("Failed to add document: {}", e))?;
    
    // Convert to JSON value
    serde_json::to_value(&document)
        .map_err(|e| format!("Failed to serialize document: {}", e))
}

/// Helper function to process text files and extract content + embeddings
async fn process_text_file(
    file_path: &str,
    _file_name: &str,
    model_registry: &State<'_, ModelRegistry>,
) -> anyhow::Result<(String, Vec<f32>)> {
    // Read file content directly for now
    let content = std::fs::read_to_string(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))?;
    
    // Generate embeddings using the model registry
    let registry = model_registry.lock().await;
    let embeddings = if registry.get_active_model_id().await.is_some() {
        // For now, return dummy embeddings until we implement proper embedding generation
        vec![0.0; 768] // Standard embedding size
    } else {
        vec![]
    };
    drop(registry);
    
    Ok((content, embeddings))
}

#[tauri::command]
async fn semantic_search(
    query: String,
    project_id: Option<String>,
    limit: Option<usize>,
    vector_store: State<'_, VectorStore>,
    _model_registry: State<'_, ModelRegistry>,
) -> Result<Vec<serde_json::Value>, String> {
    info!("Performing semantic search for: '{}'", query);
    
    // Generate embedding for the query (simplified)
    let query_embedding = vector_db::generate_simple_embedding(&query);
    
    let vector_db = vector_store.lock().await;
    match vector_db.search(
        query_embedding, 
        project_id, 
        limit.unwrap_or(10), 
        0.1 // min similarity threshold
    ).await {
        Ok(results) => {
            let search_results: Vec<serde_json::Value> = results
                .into_iter()
                .map(|result| {
                    serde_json::json!({
                        "chunk": result.chunk,
                        "similarity_score": result.similarity_score,
                        "context_chunks": result.context_chunks
                    })
                })
                .collect();
            
            info!("Found {} semantic search results", search_results.len());
            Ok(search_results)
        }
        Err(e) => Err(format!("Search failed: {}", e))
    }
}

#[tauri::command]
async fn add_document_to_vector_db(
    document_id: String,
    project_id: String,
    content: String,
    vector_store: State<'_, VectorStore>,
) -> Result<usize, String> {
    info!("Adding document to vector database: {}", document_id);
    
    // Chunk the document
    let chunks = vector_db::chunk_text(&content, &document_id, &project_id);
    let chunk_count = chunks.len();
    
    let vector_db = vector_store.lock().await;
    
    // Add all chunks
    for chunk in chunks {
        if let Err(e) = vector_db.add_chunk(chunk).await {
            error!("Failed to add chunk: {}", e);
            return Err(format!("Failed to add chunk: {}", e));
        }
    }
    
    info!("Added {} chunks to vector database", chunk_count);
    Ok(chunk_count)
}

#[tauri::command]
async fn remove_document_from_vector_db(
    document_id: String,
    vector_store: State<'_, VectorStore>,
) -> Result<(), String> {
    info!("Removing document from vector database: {}", document_id);
    
    let vector_db = vector_store.lock().await;
    match vector_db.remove_document(&document_id).await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to remove document: {}", e))
    }
}

#[tauri::command]
async fn get_vector_db_stats(
    vector_store: State<'_, VectorStore>,
) -> Result<std::collections::HashMap<String, serde_json::Value>, String> {
    let vector_db = vector_store.lock().await;
    Ok(vector_db.get_stats().await)
}

#[tauri::command]
async fn process_file_content(
    file_name: String,
    file_content: Vec<u8>,
    file_type: String,
    chat_history: State<'_, ChatHistory>,
    model_registry: State<'_, ModelRegistry>,
    _conversation_mgr: State<'_, ConversationMgr>,
    settings: State<'_, Settings>,
) -> Result<String, String> {
    info!("Processing file content: {} ({} bytes)", file_name, file_content.len());
    
    // Convert content to string for text files
    let text_content = if file_type.starts_with("text/") || 
                          file_name.ends_with(".txt") || 
                          file_name.ends_with(".md") ||
                          file_name.ends_with(".json") {
        match String::from_utf8(file_content.clone()) {
            Ok(text) => text,
            Err(_) => return Err("Failed to parse file as UTF-8 text".to_string()),
        }
    } else {
        return Err(format!("File type '{}' is not supported. Please upload text files (.txt, .md, .json) only.", 
                          if file_name.contains('.') { 
                              file_name.split('.').last().unwrap_or("unknown") 
                          } else { 
                              "unknown" 
                          }));
    };
    
    info!("Processing file content: {} ({} characters)", file_name, text_content.len());
    
    // Limit file content size to prevent model hanging
    let content_to_send = if text_content.len() > 3000 {
        warn!("File content too large ({} chars), truncating to 3000 chars", text_content.len());
        format!("{}...\n[Content truncated - showing first 3000 characters]", &text_content[0..3000])
    } else {
        text_content.clone()
    };
    
    // Directly send the file content to the model for analysis
    // Create a clear message that includes the file content
    let file_analysis_message = format!(
        "I've uploaded a text file called '{}' with the following content:\n\n--- File Content ---\n{}\n--- End of File ---\n\nPlease analyze and describe this file for me.",
        file_name,
        content_to_send
    );
    
    // Add user message to chat history
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: format!("Uploaded file: {}", file_name),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    chat_history.lock().await.push(user_msg);

    // Get settings for AI generation
    let settings_guard = settings.lock().await;
    let max_tokens = settings_guard.max_tokens;
    let system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Send directly to model registry without any vector database processing
    let registry = model_registry.lock().await;
    let response = if registry.get_active_model_id().await.is_some() {
        let input = MultimodalInput::Text(file_analysis_message);
        match registry.generate(input).await {
            Ok(model_response) => {
                info!("Successfully generated response for file: {} characters", model_response.text.len());
                model_response.text
            },
            Err(e) => {
                error!("Error generating response for file: {}", e);
                format!("I apologize, but I encountered an error analyzing the file: {}. Please try again.", e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    drop(registry);
    
    // Add assistant response to chat history
    let assistant_msg = ChatMessage {
        role: "assistant".to_string(),
        content: response.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    chat_history.lock().await.push(assistant_msg);
    
    info!("Successfully processed file '{}' - content sent directly to model", file_name);
    Ok(format!("File '{}' has been uploaded and analyzed by the AI model. Check the chat for the analysis.", file_name))
}

#[tauri::command]
async fn process_uploaded_files(
    files: Vec<String>, // File paths
    _project_id: Option<String>,
    chat_history: State<'_, ChatHistory>,
    model_registry: State<'_, ModelRegistry>,
    _conversation_mgr: State<'_, ConversationMgr>,
    settings: State<'_, Settings>,
    _vector_store: State<'_, VectorStore>,
) -> Result<String, String> {
    info!("Processing {} uploaded files", files.len());
    
    if files.is_empty() {
        return Err("No files provided".to_string());
    }
    
    // For now, process the first file as an example
    let file_path = &files[0];
    
    // Read file content
    let file_content = match std::fs::read(file_path) {
        Ok(content) => content,
        Err(e) => return Err(format!("Failed to read file: {}", e)),
    };
    
    // Determine if it's an image or text file
    let file_name = std::path::Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    
    if file_name.ends_with(".jpg") || file_name.ends_with(".jpeg") || 
       file_name.ends_with(".png") || file_name.ends_with(".gif") {
        // Process as image
        return process_image_input(
            format!("I've uploaded an image: {}", file_name),
            file_content,
            chat_history,
            model_registry,
            _conversation_mgr,
            settings,
        ).await;
    } else if file_name.ends_with(".txt") || file_name.ends_with(".md") {
        // Process as text
        let text_content = match String::from_utf8(file_content) {
            Ok(text) => text,
            Err(_) => return Err("File is not valid UTF-8 text".to_string()),
        };
        
        info!("Processing file content: {} ({} bytes)", file_name, text_content.len());
        
        // Limit file content size to prevent model hanging
        let content_to_send = if text_content.len() > 3000 {
            warn!("File content too large ({} chars), truncating to 3000 chars", text_content.len());
            format!("{}...\n[Content truncated - showing first 3000 characters]", &text_content[0..3000])
        } else {
            text_content
        };
        
        // Directly send the file content to the model for analysis
        // Create a clear message that includes the file content
        let file_analysis_message = format!(
            "I've uploaded a text file called '{}' with the following content:\n\n--- File Content ---\n{}\n--- End of File ---\n\nPlease analyze and describe this file for me.",
            file_name,
            content_to_send
        );
        
        // Add user message to chat history
        let user_msg = ChatMessage {
            role: "user".to_string(),
            content: format!("Uploaded file: {}", file_name),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        chat_history.lock().await.push(user_msg);

        // Get settings for AI generation
        let settings_guard = settings.lock().await;
        let _max_tokens = settings_guard.max_tokens;
        let _system_prompt = settings_guard.system_prompt.clone();
        drop(settings_guard);

        // Send directly to model registry without any vector database processing
        let registry = model_registry.lock().await;
        let response = if registry.get_active_model_id().await.is_some() {
            let input = MultimodalInput::Text(file_analysis_message);
            match registry.generate(input).await {
                Ok(model_response) => model_response.text,
                Err(e) => {
                    error!("Error generating response for file: {}", e);
                    format!("I apologize, but I encountered an error analyzing the file: {}. Please try again.", e)
                }
            }
        } else {
            "The AI model is still loading. Please wait a moment and try again.".to_string()
        };
        drop(registry);

        // Add assistant response to history
        let assistant_msg = ChatMessage {
            role: "assistant".to_string(),
            content: response.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        chat_history.lock().await.push(assistant_msg);

        return Ok(response);
    } else {
        return Err(format!("Unsupported file type: {}", file_name));
    }
}

// Voice pipeline commands
#[tauri::command]
async fn start_voice_session(
    voice_pipeline: State<'_, VoicePipelineMgr>,
    model_registry: State<'_, ModelRegistry>,
    app: tauri::AppHandle,
) -> Result<String, String> {
    info!("Starting voice conversation session");
    
    let mut pipeline_guard = voice_pipeline.lock().await;
    
    // Create voice pipeline if not already initialized
    if pipeline_guard.is_none() {
        let registry_arc = model_registry.inner().clone();
        
        let config = VoicePipelineConfig {
            unmute_backend_url: "ws://localhost:8000".to_string(),
            voice_character: VoiceCharacterConfig {
                voice: "default".to_string(),
                instructions: Some("You are Tektra, a helpful multimodal AI assistant. Respond naturally and conversationally in a friendly, concise manner.".to_string()),
                allow_recording: true,
            },
            ..Default::default()
        };
        
        match VoicePipeline::new(registry_arc, config).await {
            Ok(mut pipeline) => {
                // Set up event channel for UI communication
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                pipeline.set_event_channel(tx);
                
                // Spawn task to handle voice events and emit them to frontend
                let app_handle = app.clone();
                tokio::spawn(async move {
                    while let Some(event) = rx.recv().await {
                        let _ = app_handle.emit("voice-pipeline-event", serde_json::to_value(&event).unwrap_or_default());
                    }
                });
                
                *pipeline_guard = Some(pipeline);
            }
            Err(e) => {
                error!("Failed to create voice pipeline: {}", e);
                return Err(format!("Failed to initialize voice pipeline: {}", e));
            }
        }
    }
    
    // Start the session
    if let Some(ref mut pipeline) = pipeline_guard.as_mut() {
        match pipeline.start_session().await {
            Ok(_) => Ok("Voice session started successfully".to_string()),
            Err(e) => {
                error!("Failed to start voice session: {}", e);
                Err(format!("Failed to start voice session: {}", e))
            }
        }
    } else {
        Err("Voice pipeline not initialized".to_string())
    }
}

#[tauri::command]
async fn stop_voice_session(voice_pipeline: State<'_, VoicePipelineMgr>) -> Result<String, String> {
    info!("Stopping voice conversation session");
    
    let mut pipeline_guard = voice_pipeline.lock().await;
    
    if let Some(ref mut pipeline) = pipeline_guard.as_mut() {
        match pipeline.stop_session().await {
            Ok(_) => Ok("Voice session stopped successfully".to_string()),
            Err(e) => {
                error!("Failed to stop voice session: {}", e);
                Err(format!("Failed to stop voice session: {}", e))
            }
        }
    } else {
        Err("Voice pipeline not initialized".to_string())
    }
}

#[tauri::command]
async fn get_voice_session_status(voice_pipeline: State<'_, VoicePipelineMgr>) -> Result<serde_json::Value, String> {
    let pipeline_guard = voice_pipeline.lock().await;
    
    if let Some(ref pipeline) = pipeline_guard.as_ref() {
        let is_active = pipeline.is_session_active().await;
        let conversation_state = pipeline.get_conversation_state().await;
        
        Ok(serde_json::json!({
            "is_active": is_active,
            "conversation_state": conversation_state,
            "pipeline_initialized": true
        }))
    } else {
        Ok(serde_json::json!({
            "is_active": false,
            "conversation_state": null,
            "pipeline_initialized": false
        }))
    }
}

#[tauri::command]
async fn update_voice_config(
    voice_pipeline: State<'_, VoicePipelineMgr>,
    config: VoicePipelineConfig,
) -> Result<String, String> {
    info!("Updating voice pipeline configuration");
    
    let mut pipeline_guard = voice_pipeline.lock().await;
    
    if let Some(ref mut pipeline) = pipeline_guard.as_mut() {
        match pipeline.update_config(config).await {
            Ok(_) => Ok("Voice configuration updated successfully".to_string()),
            Err(e) => {
                error!("Failed to update voice config: {}", e);
                Err(format!("Failed to update voice configuration: {}", e))
            }
        }
    } else {
        Err("Voice pipeline not initialized".to_string())
    }
}

#[tauri::command]
async fn get_voice_devices() -> Result<serde_json::Value, String> {
    use voice::RealtimeAudioManager;
    use voice::AudioSettings;
    
    // Create temporary audio manager to list devices
    let settings = AudioSettings {
        sample_rate: 24000,
        buffer_size: 1024,
        audio_format: "opus".to_string(),
        noise_reduction: true,
        auto_gain: true,
    };
    
    match RealtimeAudioManager::new(settings).await {
        Ok(audio_manager) => {
            let input_devices = audio_manager.list_input_devices().unwrap_or_default();
            let output_devices = audio_manager.list_output_devices().unwrap_or_default();
            
            Ok(serde_json::json!({
                "input_devices": input_devices,
                "output_devices": output_devices
            }))
        }
        Err(e) => {
            error!("Failed to create audio manager: {}", e);
            Err(format!("Failed to list audio devices: {}", e))
        }
    }
}

// Unmute service management commands
#[tauri::command]
async fn start_unmute_services(
    unmute_manager: State<'_, UnmuteServiceMgr>,
    model_registry: State<'_, ModelRegistry>,
    app: tauri::AppHandle,
) -> Result<String, String> {
    info!("Starting Unmute services");
    
    let mut manager_guard = unmute_manager.lock().await;
    
    // Initialize service manager if not already done
    if manager_guard.is_none() {
        let app_data_dir = app.path().app_data_dir()
            .map_err(|e| format!("Failed to get app data directory: {}", e))?;
        
        let unmute_dir = app_data_dir.join("unmute");
        
        // Get model registry for Rust backend
        let registry_arc = model_registry.inner().clone();
        
        match UnmuteServiceManager::new_with_model_registry(unmute_dir, None, registry_arc).await {
            Ok(mut manager) => {
                // Set up event channel for service events
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                manager.set_event_channel(tx);
                
                // Spawn task to handle service events and emit them to frontend
                let app_handle = app.clone();
                tokio::spawn(async move {
                    while let Some(event) = rx.recv().await {
                        let _ = app_handle.emit("unmute-service-event", serde_json::to_value(&event).unwrap_or_default());
                    }
                });
                
                *manager_guard = Some(manager);
            }
            Err(e) => {
                error!("Failed to create Unmute service manager: {}", e);
                return Err(format!("Failed to initialize Unmute service manager: {}", e));
            }
        }
    }
    
    // Start the services
    if let Some(ref mut manager) = manager_guard.as_mut() {
        match manager.start_services().await {
            Ok(_) => Ok("Unmute services started successfully".to_string()),
            Err(e) => {
                error!("Failed to start Unmute services: {}", e);
                Err(format!("Failed to start Unmute services: {}", e))
            }
        }
    } else {
        Err("Unmute service manager not initialized".to_string())
    }
}

#[tauri::command]
async fn stop_unmute_services(unmute_manager: State<'_, UnmuteServiceMgr>) -> Result<String, String> {
    info!("Stopping Unmute services");
    
    let mut manager_guard = unmute_manager.lock().await;
    
    if let Some(ref mut manager) = manager_guard.as_mut() {
        match manager.stop_services().await {
            Ok(_) => Ok("Unmute services stopped successfully".to_string()),
            Err(e) => {
                error!("Failed to stop Unmute services: {}", e);
                Err(format!("Failed to stop Unmute services: {}", e))
            }
        }
    } else {
        Err("Unmute service manager not initialized".to_string())
    }
}

#[tauri::command]
async fn get_unmute_services_status(unmute_manager: State<'_, UnmuteServiceMgr>) -> Result<serde_json::Value, String> {
    let manager_guard = unmute_manager.lock().await;
    
    if let Some(ref manager) = manager_guard.as_ref() {
        let is_running = manager.is_running().await;
        let service_status = manager.get_service_status().await;
        
        Ok(serde_json::json!({
            "is_running": is_running,
            "services": service_status,
            "manager_initialized": true
        }))
    } else {
        Ok(serde_json::json!({
            "is_running": false,
            "services": {},
            "manager_initialized": false
        }))
    }
}

#[tauri::command]
async fn get_unmute_service_status(unmute_manager: State<'_, UnmuteServiceMgr>) -> Result<std::collections::HashMap<String, bool>, String> {
    let manager_guard = unmute_manager.lock().await;
    
    if let Some(ref manager) = manager_guard.as_ref() {
        Ok(manager.get_service_status().await)
    } else {
        // Return default status when manager is not initialized
        let mut status = std::collections::HashMap::new();
        status.insert("backend".to_string(), false);
        status.insert("stt".to_string(), false);
        status.insert("tts".to_string(), false);
        Ok(status)
    }
}

#[tauri::command]
async fn update_unmute_config(
    unmute_manager: State<'_, UnmuteServiceMgr>,
    config: UnmuteConfig,
) -> Result<String, String> {
    info!("Updating Unmute service configuration");
    
    let mut manager_guard = unmute_manager.lock().await;
    
    if let Some(ref mut manager) = manager_guard.as_mut() {
        match manager.update_config(config).await {
            Ok(_) => Ok("Unmute configuration updated successfully".to_string()),
            Err(e) => {
                error!("Failed to update Unmute config: {}", e);
                Err(format!("Failed to update Unmute configuration: {}", e))
            }
        }
    } else {
        Err("Unmute service manager not initialized".to_string())
    }
}

#[tauri::command]
async fn check_unmute_dependencies(unmute_manager: State<'_, UnmuteServiceMgr>) -> Result<serde_json::Value, String> {
    let mut manager_guard = unmute_manager.lock().await;
    
    if manager_guard.is_none() {
        return Ok(serde_json::json!({
            "checked": false,
            "dependencies": {},
            "message": "Service manager not initialized"
        }));
    }
    
    if let Some(ref manager) = manager_guard.as_ref() {
        match manager.check_dependencies().await {
            Ok(_) => Ok(serde_json::json!({
                "checked": true,
                "dependencies": {
                    "git": "installed",
                    "uv": "installed", 
                    "cargo": "installed",
                    "pnpm": "installed"
                },
                "message": "All dependencies are available"
            })),
            Err(e) => Ok(serde_json::json!({
                "checked": false,
                "dependencies": {},
                "message": format!("Dependency check failed: {}", e)
            }))
        }
    } else {
        Err("Unmute service manager not initialized".to_string())
    }
}

// Simple command to check if Tauri backend is ready
#[tauri::command]
async fn app_ready() -> Result<bool, String> {
    Ok(true)
}


fn main() {
    // Check if running as CLI
    if let Err(e) = cli::run_cli() {
        eprintln!("CLI error: {}", e);
        std::process::exit(1);
    }
    
    // Check if we're in a special CLI mode that shouldn't start Tauri
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && matches!(args[1].as_str(), "help" | "--help" | "-h" | "version" | "--version" | "-v" | "dev") {
        return;
    }
    
    // Initialize tracing with clean format (no timestamps)
    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_level(true)
        .compact()
        .init();
    
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let app_handle = app.handle();
            
            // Load configuration
            let mut app_config = match AppConfig::get_config_path() {
                Ok(path) => AppConfig::load(&path).unwrap_or_default(),
                Err(_) => AppConfig::default(),
            };
            
            // Apply environment variable overrides
            app_config.apply_env_overrides();
            
            // Log configuration
            tracing::info!("Loaded configuration: backend={:?}, benchmark={}", 
                app_config.inference.backend, 
                app_config.inference.benchmark_on_startup
            );
            
            // Create new model registry and conversation manager (sync initialization)
            let model_registry = InferenceModelRegistry::new();
            
            let conversation_manager = ConversationManager::new(None)
                .map_err(|e| format!("Failed to create conversation manager: {}", e))?;
            
            let audio_recorder = AudioRecorder::new(app_handle.clone());
            let avatar_manager = AvatarManager::new(app_handle.clone());
            let vision_manager = VisionManager::new(app_handle.clone()).unwrap();
            let vector_db = VectorDB::new();
            let database = Database::new(&app_handle)
                .map_err(|e| format!("Failed to initialize database: {}", e))?;
            
            app.manage(ChatHistory::new(Mutex::new(Vec::new())));
            app.manage(Settings::new(Mutex::new(AppSettings::default())));
            app.manage(ModelRegistry::new(Mutex::new(model_registry)));
            app.manage(ConversationMgr::new(Mutex::new(conversation_manager)));
            app.manage(AudioRec::new(Mutex::new(audio_recorder)));
            app.manage(Vision::new(Mutex::new(vision_manager)));
            app.manage(Avatar::new(Mutex::new(avatar_manager)));
            app.manage(VectorStore::new(Mutex::new(vector_db)));
            app.manage(DB::new(database));
            app.manage(VoicePipelineMgr::new(Mutex::new(None)));
            app.manage(UnmuteServiceMgr::new(Mutex::new(None)));
            
            // Store config for later use
            app.manage(Arc::new(Mutex::new(app_config)));
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_ready,
            initialize_model,
            send_message,
            send_message_with_camera,
            get_chat_history,
            clear_chat_history,
            get_settings,
            update_settings,
            check_model_status,
            get_available_models,
            get_model_capabilities,
            load_model,
            get_current_model,
            start_audio_recording,
            stop_audio_recording,
            is_recording,
            process_audio_stream,
            initialize_whisper,
            start_always_listening,
            get_conversation_mode,
            end_conversation,
            speak_text,
            stop_speaking,
            initialize_camera,
            start_camera_capture,
            stop_camera_capture,
            get_camera_frame,
            process_image_input,
            process_camera_feed,
            process_audio_input,
            process_multimodal_input,
            set_avatar_expression,
            start_avatar_speaking,
            stop_avatar_speaking,
            avatar_blink,
            get_backend_info,
            benchmark_backends,
            restart_backend,
            start_voice_session,
            stop_voice_session,
            get_voice_session_status,
            update_voice_config,
            get_voice_devices,
            start_unmute_services,
            stop_unmute_services,
            get_unmute_services_status,
            get_unmute_service_status,
            update_unmute_config,
            check_unmute_dependencies,
            create_project,
            get_projects,
            delete_project,
            toggle_project_star,
            get_project_documents,
            upload_file_to_project,
            process_file_content,
            process_uploaded_files,
            semantic_search,
            add_document_to_vector_db,
            remove_document_from_vector_db,
            get_vector_db_stats
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}