// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::Path;
use tauri::{Manager, State};
use tokio::sync::Mutex;
use tracing::{info, error, warn};

// Import legacy modules for compatibility
mod ai;
use ai::AIManager;
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

// Import new architecture modules
mod inference;
use inference::{ModelRegistry, MultimodalInput, ModelResponse, ImageData, ImageFormat};
mod multimodal;
use multimodal::{MultimodalProcessor, ProcessingStats};
mod conversation;
use conversation::{ConversationManager, ConversationConfig};

// Type aliases for Tauri state management
type ChatHistory = Arc<Mutex<Vec<ChatMessage>>>;
type Settings = Arc<Mutex<AppSettings>>;
type AI = Arc<Mutex<AIManager>>;
type AudioRec = Arc<Mutex<AudioRecorder>>;
type Vision = Arc<Mutex<VisionManager>>;
type Avatar = Arc<Mutex<AvatarManager>>;
type VectorStore = Arc<Mutex<VectorDB>>;
type DB = Arc<Database>;

// New architecture state types
type ModelReg = Arc<Mutex<ModelRegistry>>;
type MultimodalProc = Arc<Mutex<MultimodalProcessor>>;
type ConversationMgr = Arc<Mutex<ConversationManager>>;

/// Enhanced Tauri commands for the new architecture

#[tauri::command]
async fn initialize_new_model_system(
    model_registry: State<'_, ModelReg>,
) -> Result<bool, String> {
    info!("Initializing new model system");
    
    let mut registry = model_registry.lock().await;
    match registry.initialize().await {
        Ok(_) => {
            info!("Model system initialized successfully");
            Ok(true)
        }
        Err(e) => {
            error!("Failed to initialize model system: {}", e);
            Err(format!("Failed to initialize model system: {}", e))
        }
    }
}

#[tauri::command]
async fn send_multimodal_message(
    message: String,
    image_data: Option<Vec<u8>>,
    file_attachments: Option<Vec<String>>,
    session_id: Option<String>,
    model_registry: State<'_, ModelReg>,
    conversation_mgr: State<'_, ConversationMgr>,
    multimodal_proc: State<'_, MultimodalProc>,
    chat_history: State<'_, ChatHistory>,
) -> Result<String, String> {
    info!("Processing multimodal message");
    
    let session_id = session_id.unwrap_or_else(|| "default".to_string());
    
    // Process multimodal input
    let multimodal_input = if let Some(img_data) = image_data {
        let processor = multimodal_proc.lock().await;
        match processor.process_file_data(&img_data, None).await {
            Ok(input) => input,
            Err(e) => {
                error!("Failed to process image data: {}", e);
                return Err(format!("Failed to process image: {}", e));
            }
        }
    } else if let Some(files) = file_attachments {
        if !files.is_empty() {
            let processor = multimodal_proc.lock().await;
            match processor.process_file(&files[0], &files[0]).await {
                Ok(input) => input,
                Err(e) => {
                    error!("Failed to process file: {}", e);
                    return Err(format!("Failed to process file: {}", e));
                }
            }
        } else {
            MultimodalInput::Text(message.clone())
        }
    } else {
        MultimodalInput::Text(message.clone())
    };
    
    // Process through conversation manager
    let mut conversation_manager = conversation_mgr.lock().await;
    let conversation_response = match conversation_manager.process_turn(
        &session_id,
        multimodal_input,
        None,
    ).await {
        Ok(response) => response,
        Err(e) => {
            error!("Conversation processing failed: {}", e);
            return Err(format!("Conversation processing failed: {}", e));
        }
    };
    
    // Generate response using model registry
    let model_registry_guard = model_registry.lock().await;
    let model_response = match model_registry_guard.generate(conversation_response.conversation_context.messages.into()).await {
        Ok(response) => response,
        Err(e) => {
            error!("Model generation failed: {}", e);
            return Err(format!("Model generation failed: {}", e));
        }
    };
    
    // Add to conversation history
    if let Err(e) = conversation_manager.add_assistant_response(&session_id, model_response.clone()).await {
        warn!("Failed to add assistant response to conversation: {}", e);
    }
    
    // Add to legacy chat history for compatibility
    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: message,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    let assistant_msg = ChatMessage {
        role: "assistant".to_string(),
        content: model_response.text.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    {
        let mut history = chat_history.lock().await;
        history.push(user_msg);
        history.push(assistant_msg);
    }
    
    Ok(model_response.text)
}

#[tauri::command]
async fn list_available_models(
    model_registry: State<'_, ModelReg>,
) -> Result<Vec<serde_json::Value>, String> {
    let registry = model_registry.lock().await;
    let models = registry.list_models().await;
    
    let model_info: Vec<serde_json::Value> = models.into_iter()
        .map(|model| serde_json::json!({
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "supports_vision": model.supports_vision,
            "supports_audio": model.supports_audio,
            "supports_documents": model.supports_documents,
            "default": model.default,
            "recommended_for": model.recommended_for,
        }))
        .collect();
    
    Ok(model_info)
}

#[tauri::command]
async fn switch_active_model(
    model_id: String,
    model_registry: State<'_, ModelReg>,
) -> Result<bool, String> {
    info!("Switching to model: {}", model_id);
    
    let mut registry = model_registry.lock().await;
    match registry.switch_model(&model_id).await {
        Ok(_) => {
            info!("Successfully switched to model: {}", model_id);
            Ok(true)
        }
        Err(e) => {
            error!("Failed to switch model: {}", e);
            Err(format!("Failed to switch model: {}", e))
        }
    }
}

#[tauri::command]
async fn get_model_status(
    model_registry: State<'_, ModelReg>,
) -> Result<serde_json::Value, String> {
    let registry = model_registry.lock().await;
    let stats = registry.get_stats().await;
    
    Ok(serde_json::json!({
        "active_model": stats.active_model,
        "loaded_models": stats.loaded_models,
        "total_memory_usage": stats.total_memory_usage,
        "loaded_model_ids": stats.loaded_model_ids,
    }))
}

#[tauri::command]
async fn get_multimodal_stats(
    multimodal_proc: State<'_, MultimodalProc>,
) -> Result<ProcessingStats, String> {
    let processor = multimodal_proc.lock().await;
    Ok(processor.get_stats())
}

#[tauri::command]
async fn start_conversation_session(
    session_id: Option<String>,
    persona: Option<String>,
    conversation_mgr: State<'_, ConversationMgr>,
) -> Result<String, String> {
    let session_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    
    let mut manager = conversation_mgr.lock().await;
    match manager.start_session(session_id.clone(), persona).await {
        Ok(id) => Ok(id),
        Err(e) => Err(format!("Failed to start conversation session: {}", e)),
    }
}

#[tauri::command]
async fn end_conversation_session(
    session_id: String,
    conversation_mgr: State<'_, ConversationMgr>,
) -> Result<(), String> {
    let mut manager = conversation_mgr.lock().await;
    match manager.end_session(&session_id).await {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Failed to end conversation session: {}", e)),
    }
}

// Legacy commands for backwards compatibility

#[tauri::command]
async fn send_message(
    message: String,
    chat_history: State<'_, ChatHistory>,
    ai: State<'_, AI>,
    settings: State<'_, Settings>,
) -> Result<String, String> {
    // Legacy implementation - delegate to new system in production
    warn!("Using legacy send_message command - consider migrating to send_multimodal_message");
    
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

    // Generate response using legacy AI
    let ai_manager = ai.lock().await;
    
    let response = if ai_manager.is_loaded() {
        match ai_manager.generate_response_with_system_prompt(&message, max_tokens, system_prompt).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Error generating response: {}", e);
                format!("I apologize, but I encountered an error: {}. Please try again.", e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
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

// Include other legacy commands for compatibility...
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
            info!("Loaded configuration: backend={:?}, benchmark={}", 
                app_config.inference.backend, 
                app_config.inference.benchmark_on_startup
            );
            
            // Initialize legacy systems for compatibility
            let ai_manager = AIManager::with_backend(app_handle.clone(), app_config.inference.backend)
                .map_err(|e| format!("Failed to create AI manager: {}", e))?;
            
            let audio_recorder = AudioRecorder::new(app_handle.clone());
            let avatar_manager = AvatarManager::new(app_handle.clone());
            let vision_manager = VisionManager::new(app_handle.clone()).unwrap();
            let vector_db = VectorDB::new();
            let database = Database::new(&app_handle)
                .map_err(|e| format!("Failed to initialize database: {}", e))?;
            
            // Initialize new architecture
            let model_registry = ModelRegistry::new();
            let multimodal_processor = MultimodalProcessor::new()
                .map_err(|e| format!("Failed to initialize multimodal processor: {}", e))?;
            let conversation_manager = ConversationManager::new(Some(ConversationConfig::default()))
                .map_err(|e| format!("Failed to initialize conversation manager: {}", e))?;
            
            // Legacy state management
            app.manage(ChatHistory::new(Mutex::new(Vec::new())));
            app.manage(Settings::new(Mutex::new(AppSettings::default())));
            app.manage(AI::new(Mutex::new(ai_manager)));
            app.manage(AudioRec::new(Mutex::new(audio_recorder)));
            app.manage(Vision::new(Mutex::new(vision_manager)));
            app.manage(Avatar::new(Mutex::new(avatar_manager)));
            app.manage(VectorStore::new(Mutex::new(vector_db)));
            app.manage(DB::new(database));
            
            // New architecture state management
            app.manage(ModelReg::new(Mutex::new(model_registry)));
            app.manage(MultimodalProc::new(Mutex::new(multimodal_processor)));
            app.manage(ConversationMgr::new(Mutex::new(conversation_manager)));
            
            // Store config for later use
            app.manage(Arc::new(Mutex::new(app_config)));
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // New architecture commands
            initialize_new_model_system,
            send_multimodal_message,
            list_available_models,
            switch_active_model,
            get_model_status,
            get_multimodal_stats,
            start_conversation_session,
            end_conversation_session,
            
            // Legacy commands for compatibility
            app_ready,
            send_message,
            get_chat_history,
            clear_chat_history,
            get_settings,
            update_settings,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}