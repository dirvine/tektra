// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{Manager, State};
use tokio::sync::Mutex;
use tracing::{info, error, warn};

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
            model_name: "gemma3:4b".to_string(),
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
type VectorStore = Arc<Mutex<VectorDB>>;

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

    // Search for relevant documents before generating response
    let mut context_documents = Vec::new();
    {
        let query_embedding = vector_db::generate_simple_embedding(&message);
        let vector_db = vector_store.lock().await;
        match vector_db.search(query_embedding, None, 5, 0.2).await { // Lower threshold for better recall
            Ok(results) => {
                info!("Vector search found {} results for query: '{}'", results.len(), message);
                for result in results {
                    info!("Found document chunk with similarity {}: {}", result.similarity_score, result.chunk.document_id);
                    // Include all results above low threshold - Gemma 3 can handle context well
                    context_documents.push(format!(
                        "Document: {}\nContent: {}\n",
                        result.chunk.document_id,
                        result.chunk.content
                    ));
                }
                info!("Added {} context documents to response", context_documents.len());
            }
            Err(e) => {
                info!("Document search failed: {}", e);
            }
        }
    }
    
    // Prepare enhanced message with document context (limit context size)
    let enhanced_message = if !context_documents.is_empty() {
        info!("Enhancing response with {} document contexts", context_documents.len());
        
        // Limit total context size to prevent model hanging
        let full_context = context_documents.join("\n");
        let context_to_use = if full_context.len() > 2000 {
            warn!("Document context too large ({} chars), truncating to 2000 chars to prevent model hanging", full_context.len());
            let truncated = &full_context[0..2000];
            format!("{}...\n[Context truncated due to length]", truncated)
        } else {
            info!("Using full document context ({} chars)", full_context.len());
            full_context
        };
        
        let enhanced = format!(
            "User Question: {}\n\nUploaded File Content:\n{}\n\nBased on the file content above, please provide a helpful response. You can describe, analyze, summarize, or answer questions about this content:",
            message,
            context_to_use
        );
        
        info!("Enhanced message length: {} characters", enhanced.len());
        enhanced
    } else {
        info!("No relevant documents found, responding without additional context");
        message.clone()
    };

    // Generate response using AI
    let ai_manager = ai.lock().await;
    
    let response = if ai_manager.is_loaded() {
        match ai_manager.generate_response_with_system_prompt(&enhanced_message, max_tokens, system_prompt).await {
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
        "gemma3:4b".to_string(),
        "gemma2:2b".to_string(),
        "qwen2.5:7b".to_string(),
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
    imageData: Vec<u8>,
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
        match ai_manager.generate_response_with_image_and_system_prompt(&message, &imageData, max_tokens, system_prompt).await {
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

// Project management commands
#[tauri::command]
async fn create_project(name: String, description: Option<String>) -> Result<serde_json::Value, String> {
    use std::collections::HashMap;
    
    let project_id = uuid::Uuid::new_v4().to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let mut project = HashMap::new();
    project.insert("id".to_string(), serde_json::Value::String(project_id));
    project.insert("name".to_string(), serde_json::Value::String(name));
    project.insert("description".to_string(), serde_json::Value::String(description.unwrap_or_default()));
    project.insert("createdAt".to_string(), serde_json::Value::String(now.to_string()));
    project.insert("updatedAt".to_string(), serde_json::Value::String(now.to_string()));
    project.insert("documentCount".to_string(), serde_json::Value::Number(0.into()));
    project.insert("tags".to_string(), serde_json::Value::Array(vec![]));
    project.insert("isStarred".to_string(), serde_json::Value::Bool(false));
    
    // TODO: Save to actual database
    Ok(serde_json::Value::Object(project.into_iter().collect()))
}

#[tauri::command]
async fn get_projects() -> Result<Vec<serde_json::Value>, String> {
    // TODO: Load from actual database
    // For now, return empty array
    Ok(vec![])
}

#[tauri::command]
async fn delete_project(project_id: String) -> Result<(), String> {
    info!("Deleting project: {}", project_id);
    // TODO: Implement actual deletion
    Ok(())
}

#[tauri::command]
async fn toggle_project_star(project_id: String) -> Result<(), String> {
    info!("Toggling star for project: {}", project_id);
    // TODO: Implement actual star toggle
    Ok(())
}

#[tauri::command]
async fn get_project_documents(project_id: String) -> Result<Vec<serde_json::Value>, String> {
    info!("Getting documents for project: {}", project_id);
    // TODO: Load from actual database
    Ok(vec![])
}

#[tauri::command]
async fn upload_file_to_project(
    project_id: String,
    file_path: String,
    file_name: String,
) -> Result<serde_json::Value, String> {
    use std::collections::HashMap;
    
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
        .unwrap()
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
    
    let mut document = HashMap::new();
    document.insert("id".to_string(), serde_json::Value::String(document_id));
    document.insert("projectId".to_string(), serde_json::Value::String(project_id));
    document.insert("name".to_string(), serde_json::Value::String(file_name));
    document.insert("type".to_string(), serde_json::Value::String(file_type.to_string()));
    document.insert("size".to_string(), serde_json::Value::Number(file_size.into()));
    document.insert("path".to_string(), serde_json::Value::String(file_path));
    document.insert("uploadedAt".to_string(), serde_json::Value::String(now.to_string()));
    document.insert("tags".to_string(), serde_json::Value::Array(vec![]));
    
    // TODO: Save to actual database and process file content
    Ok(serde_json::Value::Object(document.into_iter().collect()))
}

#[tauri::command]
async fn semantic_search(
    query: String,
    project_id: Option<String>,
    limit: Option<usize>,
    vector_store: State<'_, VectorStore>,
    _ai: State<'_, AI>,
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
    fileName: String,
    fileContent: Vec<u8>,
    fileType: String,
    vector_store: State<'_, VectorStore>,
) -> Result<String, String> {
    info!("Processing file content: {} ({} bytes)", fileName, fileContent.len());
    
    // Convert content to string for text files
    let text_content = if fileType.starts_with("text/") || 
                          fileName.ends_with(".txt") || 
                          fileName.ends_with(".md") ||
                          fileName.ends_with(".json") {
        match String::from_utf8(fileContent.clone()) {
            Ok(text) => text,
            Err(_) => return Err("Failed to parse file as UTF-8 text".to_string()),
        }
    } else {
        return Err(format!("File type '{}' is not supported. Please upload text files (.txt, .md, .json) only.", 
                          if fileName.contains('.') { 
                              fileName.split('.').last().unwrap_or("unknown") 
                          } else { 
                              "unknown" 
                          }));
    };
    
    // Add to vector database using the existing chunking system
    let document_id = format!("uploaded_file_{}", fileName);
    let project_id = "default".to_string(); // Default project for uploaded files
    
    // Chunk the document
    let chunks = vector_db::chunk_text(&text_content, &document_id, &project_id);
    let chunk_count = chunks.len();
    
    // Add all chunks to the vector database
    let vector_db = vector_store.lock().await;
    for chunk in chunks {
        if let Err(e) = vector_db.add_chunk(chunk).await {
            error!("Failed to add chunk to vector DB: {}", e);
            return Err(format!("Failed to process file chunk: {}", e));
        }
    }
    
    info!("Successfully added {} with {} chunks to vector database", fileName, chunk_count);
    Ok(format!("Successfully processed and indexed '{}' ({} characters, {} chunks)", fileName, text_content.len(), chunk_count))
}

#[tauri::command]
async fn process_uploaded_files(
    files: Vec<String>, // File paths
    _project_id: Option<String>,
    chat_history: State<'_, ChatHistory>,
    ai: State<'_, AI>,
    settings: State<'_, Settings>,
    vector_store: State<'_, VectorStore>,
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
            ai,
            settings,
        ).await;
    } else if file_name.ends_with(".txt") || file_name.ends_with(".md") {
        // Process as text
        let text_content = match String::from_utf8(file_content) {
            Ok(text) => text,
            Err(_) => return Err("File is not valid UTF-8 text".to_string()),
        };
        
        // Store the file content in vector database for future RAG queries
        let document_id = format!("uploaded_file_{}", file_name);
        let project_id = "default".to_string();
        
        info!("Processing file content: {} ({} bytes)", file_name, text_content.len());
        
        // Add to vector database
        match add_document_to_vector_db(
            document_id,
            project_id,
            text_content.clone(),
            vector_store.clone(),
        ).await {
            Ok(chunk_count) => {
                info!("Successfully added {} chunks to vector database for file: {}", chunk_count, file_name);
            }
            Err(e) => {
                error!("Failed to add file to vector database: {}", e);
                return Err(format!("Failed to process file: {}", e));
            }
        }
        
        // Add a file upload message to chat history
        let upload_msg = ChatMessage {
            role: "user".to_string(),
            content: format!("Uploaded file: {}", file_name),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        chat_history.lock().await.push(upload_msg);
        
        // Return success message - the actual file content will be retrieved via RAG when needed
        return Ok(format!("Successfully uploaded and processed '{}'. You can now ask questions about this file and I'll analyze its content.", file_name));
    } else {
        return Err(format!("Unsupported file type: {}", file_name));
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
            
            // Create AI manager with configured backend
            let ai_manager = AIManager::with_backend(app_handle.clone(), app_config.inference.backend)
                .map_err(|e| format!("Failed to create AI manager: {}", e))?;
            
            let audio_recorder = AudioRecorder::new(app_handle.clone());
            let avatar_manager = AvatarManager::new(app_handle.clone());
            let vision_manager = VisionManager::new(app_handle.clone()).unwrap();
            let vector_db = VectorDB::new();
            
            app.manage(ChatHistory::new(Mutex::new(Vec::new())));
            app.manage(Settings::new(Mutex::new(AppSettings::default())));
            app.manage(AI::new(Mutex::new(ai_manager)));
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
            benchmark_backends,
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