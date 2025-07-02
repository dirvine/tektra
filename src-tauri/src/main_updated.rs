// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{Manager, State};
use tokio::sync::Mutex;
use tracing::{info, error, warn};

mod ai;
// Use the new TektraModelIntegration instead of AIManager
use ai::TektraModelIntegration;
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
            model_name: "gemma3n:e4b".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            voice_enabled: false,
            auto_speech: false,
            system_prompt: Some("You are Tektra, a helpful AI assistant. Provide clear, conversational responses. Use natural formatting with line breaks and structure your responses naturally. Be helpful and friendly in your interactions.".to_string()),
            user_prefix: Some("User: ".to_string()),
            assistant_prefix: Some("Assistant: ".to_string()),
        }
    }
}

type ChatHistory = Arc<Mutex<Vec<ChatMessage>>>;
type Settings = Arc<Mutex<AppSettings>>;
// Updated to use TektraModelIntegration
type AI = Arc<TektraModelIntegration>;
type AudioRec = Arc<Mutex<AudioRecorder>>;
type Vision = Arc<Mutex<VisionManager>>;
type Avatar = Arc<Mutex<AvatarManager>>;
type VectorStore = Arc<Mutex<VectorDB>>;

// Audio recording commands remain the same
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

// Updated model initialization to use new infrastructure
#[tauri::command]
async fn initialize_model(model_name: String, ai: State<'_, AI>, settings: State<'_, Settings>) -> Result<String, String> {
    info!("Initializing model: {}", model_name);
    
    // Update settings with new model name
    settings.lock().await.model_name = model_name.clone();
    
    // Load model using new infrastructure
    match ai.load_model(&model_name).await {
        Ok(_) => Ok(format!("Model {} loaded successfully", model_name)),
        Err(e) => Err(format!("Failed to load model: {}", e)),
    }
}

#[tauri::command]
async fn app_ready() -> Result<(), String> {
    info!("Application is ready, frontend connected");
    Ok(())
}

// Updated send_message to use new infrastructure
#[tauri::command]
async fn send_message(
    message: String,
    chat_history: State<'_, ChatHistory>,
    settings: State<'_, Settings>,
    ai: State<'_, AI>,
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
    let system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Set system prompt if provided
    if let Some(prompt) = system_prompt {
        let _ = ai.set_system_prompt(&prompt).await;
    }

    // Generate response using new infrastructure
    let response = match ai.process_input(&message).await {
        Ok(resp) => resp,
        Err(e) => {
            error!("Error generating response: {}", e);
            format!("I apologize, but I encountered an error: {}. Please try again.", e)
        }
    };
    
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

// Updated send_message_with_camera to use multimodal capabilities
#[tauri::command]
async fn send_message_with_camera(
    message: String,
    chat_history: State<'_, ChatHistory>,
    settings: State<'_, Settings>,
    ai: State<'_, AI>,
    vision: State<'_, Vision>,
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
    let system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Set system prompt if provided
    if let Some(prompt) = system_prompt {
        let _ = ai.set_system_prompt(&prompt).await;
    }

    // Capture camera frame if camera is active
    let vision_manager = vision.lock().await;
    let frame_data: Option<Vec<u8>> = if vision_manager.is_capturing() {
        match vision_manager.capture_frame().await {
            Ok(frame) => Some(frame.data),
            Err(e) => {
                error!("Failed to capture camera frame: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Generate response using multimodal capabilities
    let response = if let Some(image_data) = frame_data {
        match ai.process_image_with_description(&message, image_data).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Error generating response with image: {}", e);
                format!("I apologize, but I encountered an error processing the image: {}. Please try again.", e)
            }
        }
    } else {
        // Fall back to text-only if no camera
        match ai.process_input(&message).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Error generating response: {}", e);
                format!("I apologize, but I encountered an error: {}. Please try again.", e)
            }
        }
    };
    
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

// New command to list available models
#[tauri::command]
async fn list_models(ai: State<'_, AI>) -> Result<Vec<ai::ModelInfo>, String> {
    ai.list_models().await.map_err(|e| e.to_string())
}

// New command to switch backend
#[tauri::command]
async fn switch_backend(backend: String, ai: State<'_, AI>) -> Result<(), String> {
    ai.switch_backend(&backend).await.map_err(|e| e.to_string())
}

// New command for streaming responses
#[tauri::command]
async fn stream_response(
    message: String,
    window: tauri::Window,
    ai: State<'_, AI>,
) -> Result<(), String> {
    let mut receiver = ai.stream_response(&message).await
        .map_err(|e| e.to_string())?;
    
    // Stream chunks to frontend
    tokio::spawn(async move {
        while let Some(chunk) = receiver.recv().await {
            let _ = window.emit("stream-chunk", chunk);
        }
        let _ = window.emit("stream-end", ());
    });
    
    Ok(())
}

// Process multimodal input (new capability)
#[tauri::command]
async fn process_multimodal_input(
    text: Option<String>,
    image_base64: Option<String>,
    audio_base64: Option<String>,
    ai: State<'_, AI>,
) -> Result<String, String> {
    let image_data = if let Some(b64) = image_base64 {
        Some(base64::decode(b64).map_err(|e| e.to_string())?)
    } else {
        None
    };
    
    let audio_data = if let Some(b64) = audio_base64 {
        Some(base64::decode(b64).map_err(|e| e.to_string())?)
    } else {
        None
    };
    
    ai.process_multimodal(text, image_data, audio_data)
        .await
        .map_err(|e| e.to_string())
}

// Process audio input with transcription
#[tauri::command]
async fn process_audio_input(
    audio_data: Vec<u8>,
    ai: State<'_, AI>,
    audio: State<'_, AudioRec>,
    conversation_manager: State<'_, ConversationManager>,
    settings: State<'_, Settings>,
    window: tauri::Window,
) -> Result<String, String> {
    // Get Whisper transcription
    let mut audio_recorder = audio.lock().await;
    let transcription = match audio_recorder.transcribe_audio(&audio_data).await {
        Ok(text) => text,
        Err(e) => return Err(format!("Transcription failed: {}", e)),
    };
    
    // Update conversation manager
    let mut conv_manager = conversation_manager.lock().await;
    conv_manager.process_user_input(&transcription).await;
    
    // Process with AI using multimodal capabilities if available
    let response = ai.process_audio_with_transcription(&transcription, audio_data)
        .await
        .map_err(|e| e.to_string())?;
    
    // Update conversation state
    conv_manager.process_ai_response(&response).await;
    
    // Emit events
    let _ = window.emit("conversation-user-input", &transcription);
    let _ = window.emit("conversation-ai-responding", &response);
    
    Ok(response)
}

// Other commands remain mostly the same but can be enhanced with new capabilities...

#[tauri::command]
async fn get_chat_history(chat_history: State<'_, ChatHistory>) -> Result<Vec<ChatMessage>, String> {
    Ok(chat_history.lock().await.clone())
}

#[tauri::command]
async fn clear_chat_history(
    chat_history: State<'_, ChatHistory>,
    ai: State<'_, AI>
) -> Result<(), String> {
    chat_history.lock().await.clear();
    // Also clear conversation in AI integration
    ai.clear_conversation().await.map_err(|e| e.to_string())?;
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
    Ok(ai.is_loaded().await)
}

// New backend info command
#[tauri::command]
async fn get_backend_info(ai: State<'_, AI>) -> Result<String, String> {
    let backend = ai.current_model().await
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "None".to_string());
    
    let memory = ai.get_memory_usage().await / (1024 * 1024); // Convert to MB
    
    Ok(format!("Backend: {}\nMemory Usage: {} MB", backend, memory))
}

// ... (rest of the commands remain similar with minor adjustments for the new AI integration)

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
            
            // Create the new TektraModelIntegration
            let model_integration = tauri::async_runtime::block_on(async {
                TektraModelIntegration::new(app_handle.clone()).await
            }).map_err(|e| format!("Failed to create model integration: {}", e))?;
            
            let audio_recorder = AudioRecorder::new(app_handle.clone());
            let avatar_manager = AvatarManager::new(app_handle.clone());
            let vision_manager = VisionManager::new(app_handle.clone()).unwrap();
            let vector_db = VectorDB::new();
            
            app.manage(ChatHistory::new(Mutex::new(Vec::new())));
            app.manage(Settings::new(Mutex::new(AppSettings::default())));
            app.manage(AI::new(model_integration));
            app.manage(AudioRec::new(Mutex::new(audio_recorder)));
            app.manage(Vision::new(Mutex::new(vision_manager)));
            app.manage(Avatar::new(Mutex::new(avatar_manager)));
            app.manage(VectorStore::new(Mutex::new(vector_db)));
            
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
            list_models,
            switch_backend,
            stream_response,
            process_multimodal_input,
            get_backend_info,
            // ... rest of the commands ...
            start_audio_recording,
            stop_audio_recording,
            is_recording,
            process_audio_stream,
            initialize_whisper,
            process_audio_input,
            // Add other commands as needed
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}