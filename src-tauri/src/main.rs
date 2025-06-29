// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{Manager, State};
use tokio::sync::Mutex;
use tracing::{info, error};

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
            model_name: "gemma2:2b".to_string(),
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
        "gemma2:2b".to_string(),
        "gemma3n:e2b".to_string(),
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

// Multimodal input commands
#[tauri::command]
async fn process_image_input(
    message: String,
    image_data: Vec<u8>,
    chat_history: State<'_, ChatHistory>,
    ai: State<'_, AI>,
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

    // Generate response using AI with image
    let ai_manager = ai.lock().await;
    
    let response = if ai_manager.is_loaded() {
        match ai_manager.generate_response_with_image_and_system_prompt(&message, &image_data, max_tokens, system_prompt).await {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Error generating response with image: {}", e);
                format!("I can see the image you've shared, but I'm still learning to process visual information. Error: {}", e)
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
async fn process_camera_feed(
    vision: State<'_, Vision>,
    ai: State<'_, AI>,
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
                        let ai_manager = ai.lock().await;
                        let settings_guard = settings.lock().await;
                        let max_tokens = settings_guard.max_tokens;
                        let system_prompt = settings_guard.system_prompt.clone();
                        
                        let response = if ai_manager.is_loaded() {
                            match ai_manager.generate_response_with_image_and_system_prompt("Describe what you see.", &frame.data, max_tokens, system_prompt).await {
                                Ok(resp) => resp,
                                Err(e) => {
                                    eprintln!("Error generating response with image: {}", e);
                                    format!("I apologize, but I encountered an error processing the image: {}. Please try again.", e)
                                }
                            }
                        } else {
                            "The AI model is still loading. Please wait a moment and try again.".to_string()
                        };
                        
                        drop(ai_manager);
                        
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
    ai: State<'_, AI>,
    settings: State<'_, Settings>,
) -> Result<String, String> {
    info!("Processing audio input: {} bytes of audio data", audio_data.len());
    
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
    let max_tokens = settings_guard.max_tokens;
    let _system_prompt = settings_guard.system_prompt.clone();
    drop(settings_guard);

    // Generate response using AI with audio data
    let ai_manager = ai.lock().await;
    
    let response = if ai_manager.is_loaded() {
        // Use multimodal generation with audio data
        match ai_manager.generate_response_with_audio(&message, &audio_data, max_tokens).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Error generating response with audio: {}", e);
                format!("I heard your audio input but encountered an error processing it: {}. Please try again.", e)
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
async fn process_multimodal_input(
    message: String,
    image_data: Option<Vec<u8>>,
    audio_data: Option<Vec<u8>>,
    video_data: Option<Vec<u8>>,
    chat_history: State<'_, ChatHistory>,
    ai: State<'_, AI>,
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

    // Process multimodal input
    let ai_manager = ai.lock().await;
    
    let vision_manager = vision.lock().await;
    let response = if ai_manager.is_loaded() {
        if let Some(img_data) = image_data {
            // Process image input
            match ai_manager.generate_response_with_image_and_system_prompt(&message, &img_data, max_tokens, system_prompt).await {
                Ok(resp) => resp,
                Err(e) => {
                    eprintln!("Error generating multimodal response: {}", e);
                    format!("I can see your multimodal input ({} modalities), but I'm still learning to process all types. Error: {}", modality_count, e)
                }
            }
        } else if vision_manager.is_capturing() {
            // Process camera feed
            match vision_manager.capture_frame().await {
                Ok(frame) => {
                    match ai_manager.generate_response_with_image_and_system_prompt("Describe what you see.", &frame.data, max_tokens, system_prompt).await {
                        Ok(resp) => resp,
                        Err(e) => {
                            eprintln!("Error generating response with image: {}", e);
                            format!("I apologize, but I encountered an error processing the image: {}. Please try again.", e)
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to capture camera frame: {}", e);
                    "I couldn't capture a frame from the camera.".to_string()
                }
            }
        } else {
            // Fall back to text-only processing
            match ai_manager.generate_response_with_system_prompt(&message, max_tokens, system_prompt).await {
                Ok(resp) => {
                    if modality_count > 0 {
                        format!("{}\n\nNote: I received {} additional input modalities that I'm still learning to process fully.", resp, modality_count)
                    } else {
                        resp
                    }
                }
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
async fn get_backend_info(ai: State<'_, AI>) -> Result<String, String> {
    let ai_manager = ai.lock().await;
    Ok(ai_manager.get_backend_info().await)
}

#[tauri::command]
async fn benchmark_backends(
    ai: State<'_, AI>,
    prompt: Option<String>,
    max_tokens: Option<usize>,
) -> Result<Vec<(String, crate::ai::InferenceMetrics)>, String> {
    let ai_manager = ai.lock().await;
    let test_prompt = prompt.unwrap_or_else(|| "What is the capital of France?".to_string());
    let tokens = max_tokens.unwrap_or(100);
    
    match ai_manager.benchmark_backends(&test_prompt, tokens).await {
        Ok(results) => Ok(results),
        Err(e) => Err(format!("Failed to benchmark backends: {}", e)),
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
    
    // Initialize tracing with clean format (no timestamps)
    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_level(true)
        .compact()
        .init();
    
    tauri::Builder::default()
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
            
            // Create AI manager with configured backend
            let ai_manager = AIManager::with_backend(app_handle.clone(), app_config.inference.backend)
                .map_err(|e| format!("Failed to create AI manager: {}", e))?;
            
            let audio_recorder = AudioRecorder::new(app_handle.clone());
            let avatar_manager = AvatarManager::new(app_handle.clone());
            let vision_manager = VisionManager::new(app_handle.clone()).unwrap();
            
            app.manage(ChatHistory::new(Mutex::new(Vec::new())));
            app.manage(Settings::new(Mutex::new(AppSettings::default())));
            app.manage(AI::new(Mutex::new(ai_manager)));
            app.manage(AudioRec::new(Mutex::new(audio_recorder)));
            app.manage(Vision::new(Mutex::new(vision_manager)));
            app.manage(Avatar::new(Mutex::new(avatar_manager)));
            
            // Store config for later use
            app.manage(Arc::new(Mutex::new(app_config)));
            
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
            process_image_input,
            process_camera_feed,
            process_audio_input,
            process_multimodal_input,
            set_avatar_expression,
            start_avatar_speaking,
            stop_avatar_speaking,
            avatar_blink,
            get_backend_info,
            benchmark_backends
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}