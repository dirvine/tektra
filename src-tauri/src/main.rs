// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::Path;
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
mod database;
use database::{Database, Document};
mod types;
use types::{ChatMessage, AppSettings};


type ChatHistory = Arc<Mutex<Vec<ChatMessage>>>;
type Settings = Arc<Mutex<AppSettings>>;
type AI = Arc<Mutex<AIManager>>;
type AudioRec = Arc<Mutex<AudioRecorder>>;
type Vision = Arc<Mutex<VisionManager>>;
type Avatar = Arc<Mutex<AvatarManager>>;
type VectorStore = Arc<Mutex<VectorDB>>;
type DB = Arc<Database>;

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

    // Send message directly to AI model without vector database processing
    let enhanced_message = message.clone();

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
        // Text-only models
        "gemma3n:e4b".to_string(),  // Gemma 3N (text-only in Ollama currently)
        "gemma2:2b".to_string(),
        "qwen2.5:7b".to_string(),
        "llama3.2:3b".to_string(),
        "phi3:mini".to_string(),
        
        // Vision-capable models
        "llama3.2-vision:11b".to_string(),  // Latest vision model
        "llava:7b".to_string(),              // Popular vision model
        "moondream:latest".to_string(),      // Lightweight vision model
        "bakllava:latest".to_string(),       // Alternative vision model
    ])
}

#[tauri::command]
async fn get_model_capabilities(model_name: String) -> Result<serde_json::Value, String> {
    use serde_json::json;
    
    // Check if this is a vision-capable model in Ollama
    let supports_vision = model_name.contains("llava") || 
                         model_name.contains("bakllava") || 
                         model_name.contains("moondream") ||
                         model_name.contains("llama3.2-vision") ||
                         model_name.contains("llama3.2:11b-vision") ||
                         model_name.contains("llama3.2:90b-vision");
    
    // Special note for Gemma 3N
    let note = if model_name.contains("gemma3n") {
        Some("Gemma 3N is designed as a multimodal model with vision, audio, and video capabilities. However, Ollama's current implementation only supports text input. Full multimodal support is expected in a future update.")
    } else {
        None
    };
    
    Ok(json!({
        "model": model_name,
        "supports_text": true,
        "supports_vision": supports_vision,
        "supports_audio": false,  // No Ollama models currently support direct audio input
        "supports_video": false,  // No Ollama models currently support video
        "note": note,
        "recommended_for": if supports_vision {
            vec!["image analysis", "visual question answering", "OCR", "image description"]
        } else if model_name.contains("gemma3n") {
            vec!["general conversation", "code generation", "reasoning", "creative writing"]
        } else {
            vec!["text generation", "conversation", "question answering"]
        }
    }))
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
    // For now, return a message about vision limitations
    warn!("Camera initialization requested but vision features are limited in current Ollama implementation");
    Err("Vision features are currently limited. Gemma 3N's multimodal capabilities are not yet available in Ollama. Please use llava or llama3.2-vision models for image analysis.".to_string())
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
    let ai_manager = ai.lock().await;
    let vision_manager = vision.lock().await;
    
    let response = if ai_manager.is_loaded() {
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
        
        // Determine final image data to use
        let final_image_data = if let Some(ref img_data) = image_data {
            Some(img_data.as_slice())
        } else if let Some(ref cam_data) = camera_frame_data {
            Some(cam_data.as_slice())
        } else {
            None
        };
        
        // Use comprehensive multimodal generation with context
        match ai_manager.generate_multimodal_response(
            &message,
            final_image_data,
            audio_data.as_deref(),
            system_prompt,
            recent_history.as_deref(),
            max_tokens
        ).await {
            Ok(resp) => {
                info!("Generated comprehensive multimodal response with {} modalities", modality_count);
                resp
            }
            Err(e) => {
                error!("Error generating comprehensive multimodal response: {}", e);
                format!("I can see your multimodal input ({} modalities), but encountered an error: {}. Please try again.", modality_count, e)
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
    ai: State<'_, AI>,
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
        match process_text_file(&file_path, &file_name, &ai).await {
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
    ai: &State<'_, AI>,
) -> anyhow::Result<(String, Vec<f32>)> {
    use crate::ai::document_processor::UnifiedDocumentProcessor;
    
    // Create document processor
    let processor = UnifiedDocumentProcessor::new();
    
    // Process the document
    let processed = processor.process_file(Path::new(file_path)).await?;
    
    // Extract text content from the processed document
    let content = processed.raw_text.clone();
    
    // Generate embeddings using the AI model
    let ai_manager = ai.lock().await;
    let embeddings = if ai_manager.is_loaded() {
        // For now, return dummy embeddings until we implement proper embedding generation
        vec![0.0; 768] // Standard embedding size
    } else {
        vec![]
    };
    drop(ai_manager);
    
    Ok((content, embeddings))
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
    file_name: String,
    file_content: Vec<u8>,
    file_type: String,
    chat_history: State<'_, ChatHistory>,
    ai: State<'_, AI>,
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

    // Send directly to AI model without any vector database processing
    let ai_manager = ai.lock().await;
    let response = if ai_manager.is_loaded() {
        match ai_manager.generate_response_with_system_prompt(&file_analysis_message, max_tokens, system_prompt).await {
            Ok(resp) => {
                info!("Successfully generated response for file: {} characters", resp.len());
                resp
            },
            Err(e) => {
                error!("Error generating response for file: {}", e);
                format!("I apologize, but I encountered an error analyzing the file: {}. Please try again.", e)
            }
        }
    } else {
        "The AI model is still loading. Please wait a moment and try again.".to_string()
    };
    drop(ai_manager);
    
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
        let max_tokens = settings_guard.max_tokens;
        let system_prompt = settings_guard.system_prompt.clone();
        drop(settings_guard);

        // Send directly to AI model without any vector database processing
        let ai_manager = ai.lock().await;
        let response = if ai_manager.is_loaded() {
            match ai_manager.generate_response_with_system_prompt(&file_analysis_message, max_tokens, system_prompt).await {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Error generating response for file: {}", e);
                    format!("I apologize, but I encountered an error analyzing the file: {}. Please try again.", e)
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

        return Ok(response);
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
            let database = Database::new(&app_handle)
                .map_err(|e| format!("Failed to initialize database: {}", e))?;
            
            app.manage(ChatHistory::new(Mutex::new(Vec::new())));
            app.manage(Settings::new(Mutex::new(AppSettings::default())));
            app.manage(AI::new(Mutex::new(ai_manager)));
            app.manage(AudioRec::new(Mutex::new(audio_recorder)));
            app.manage(Vision::new(Mutex::new(vision_manager)));
            app.manage(Avatar::new(Mutex::new(avatar_manager)));
            app.manage(VectorStore::new(Mutex::new(vector_db)));
            app.manage(DB::new(database));
            
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