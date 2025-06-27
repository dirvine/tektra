// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{Manager, State};
use tokio::sync::Mutex;

mod ai;
use ai::AIManager;
mod audio;
use audio::AudioRecorder;
mod vision;
use vision::VisionManager;
mod avatar;
use avatar::AvatarManager;
mod cli;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
    timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppSettings {
    model_name: String,
    max_tokens: usize,
    temperature: f32,
    voice_enabled: bool,
    auto_speech: bool,
    system_prompt: Option<String>,
    user_prefix: Option<String>,
    assistant_prefix: Option<String>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            model_name: "Gemma-3n-E2B".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            voice_enabled: true,
            auto_speech: false,
            system_prompt: Some("You are Tektra, a helpful AI assistant powered by the Gemma-3n model. You provide accurate, thoughtful, and detailed responses.".to_string()),
            user_prefix: Some("User: ".to_string()),
            assistant_prefix: Some("Assistant: ".to_string()),
        }
    }
}

type ChatHistory = Arc<Mutex<Vec<ChatMessage>>>;
type Settings = Arc<Mutex<AppSettings>>;
type AI = Arc<Mutex<AIManager>>;
type AudioRec = Arc<Mutex<AudioRecorder>>;
type Vision = Arc<Mutex<VisionManager>>;
type Avatar = Arc<Mutex<AvatarManager>>;

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
    let recorder = audio.lock().await;
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
async fn initialize_model(ai: State<'_, AI>) -> Result<bool, String> {
    let mut ai_manager = ai.lock().await;
    
    match ai_manager.load_model().await {
        Ok(_) => Ok(true),
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            Err(format!("Failed to load model: {}", e))
        }
    }
}

#[tauri::command]
async fn send_message(
    message: String,
    chat_history: State<'_, ChatHistory>,
    ai: State<'_, AI>,
    settings: State<'_, Settings>,
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

    // Generate response using AI
    let ai_manager = ai.lock().await;
    
    let response = if ai_manager.is_loaded() {
        match ai_manager.generate_response_with_system_prompt(&message, max_tokens, system_prompt).await {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Error generating response: {}", e);
                format!("I apologize, but I encountered an error: {}. Please try again.", e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    
    drop(ai_manager);
    
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
    ai: State<'_, AI>,
    vision: State<'_, Vision>,
    settings: State<'_, Settings>,
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
    let frame_data = if vision_manager.is_capturing() {
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
    drop(vision_manager);

    // Generate response using AI
    let ai_manager = ai.lock().await;
    
    let response = if ai_manager.is_loaded() {
        if let Some(image_data) = frame_data {
            // Use vision-enabled response generation
            match ai_manager.generate_response_with_image_and_system_prompt(&message, &image_data, max_tokens, system_prompt).await {
                Ok(resp) => resp,
                Err(e) => {
                    eprintln!("Error generating response with image: {}", e);
                    format!("I apologize, but I encountered an error processing the image: {}. Please try again.", e)
                }
            }
        } else {
            // Fall back to text-only response if no camera
            match ai_manager.generate_response_with_system_prompt(&message, max_tokens, system_prompt).await {
                Ok(resp) => resp,
                Err(e) => {
                    eprintln!("Error generating response: {}", e);
                    format!("I apologize, but I encountered an error: {}. Please try again.", e)
                }
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    
    drop(ai_manager);
    
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
async fn check_model_status(ai: State<'_, AI>) -> Result<bool, String> {
    Ok(ai.lock().await.is_loaded())
}

#[tauri::command]
async fn get_available_models() -> Result<Vec<String>, String> {
    Ok(vec![
        "TinyLlama-1.1B-Chat".to_string(),
        "TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
    ])
}

// Camera commands
#[tauri::command]
async fn initialize_camera(vision: State<'_, Vision>) -> Result<bool, String> {
    let vision_manager = vision.lock().await;
    match vision_manager.initialize_camera().await {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Failed to initialize camera: {}", e)),
    }
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
    
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    tauri::Builder::default()
        .setup(|app| {
            let app_handle = app.handle();
            let ai_manager = AIManager::new(app_handle.clone())
                .map_err(|e| format!("Failed to create AI manager: {}", e))?;
            
            let audio_recorder = AudioRecorder::new(app_handle.clone());
            let vision_manager = VisionManager::new(app_handle.clone());
            let avatar_manager = AvatarManager::new(app_handle.clone());
            
            app.manage(ChatHistory::new(Mutex::new(Vec::new())));
            app.manage(Settings::new(Mutex::new(AppSettings::default())));
            app.manage(AI::new(Mutex::new(ai_manager)));
            app.manage(AudioRec::new(Mutex::new(audio_recorder)));
            app.manage(Vision::new(Mutex::new(vision_manager)));
            app.manage(Avatar::new(Mutex::new(avatar_manager)));
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            initialize_model,
            send_message,
            send_message_with_camera,
            get_chat_history,
            clear_chat_history,
            get_settings,
            update_settings,
            check_model_status,
            get_available_models,
            start_audio_recording,
            stop_audio_recording,
            is_recording,
            process_audio_stream,
            initialize_whisper,
            initialize_camera,
            start_camera_capture,
            stop_camera_capture,
            get_camera_frame,
            set_avatar_expression,
            start_avatar_speaking,
            stop_avatar_speaking,
            avatar_blink
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}