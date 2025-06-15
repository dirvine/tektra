// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Manager, Emitter};
use tracing::{info, error};
use tracing_subscriber;

mod ai;
mod audio;
mod vision;
mod robot;
mod state;

use state::AppState;

#[tauri::command]
async fn send_message(
    message: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    info!("Received message: {}", message);
    
    // Process message through AI model
    match state.process_message(message).await {
        Ok(response) => Ok(response),
        Err(e) => {
            error!("Error processing message: {}", e);
            Err(e.to_string())
        }
    }
}

#[tauri::command]
async fn start_voice_input(
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state.start_voice_capture().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn stop_voice_input(
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state.stop_voice_capture().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn capture_image(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<u8>, String> {
    state.capture_camera_frame().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn load_model(
    model_name: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    info!("Loading model: {}", model_name);
    state.load_model(&model_name).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_model_status(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    Ok(state.get_model_status().await)
}

#[tauri::command]
async fn download_model(
    model_name: String,
    force: bool,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    info!("Downloading model: {} (force: {})", model_name, force);
    state.download_model(&model_name, force).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_cached_models(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    state.list_cached_models().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn transcribe_voice(
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    state.transcribe_voice_input().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn clear_voice_buffer(
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state.clear_voice_buffer().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn is_recording(
    state: tauri::State<'_, AppState>,
) -> Result<bool, String> {
    state.is_recording().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn speak_text(
    text: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state.speak_response(&text).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_camera_info(
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    state.get_camera_info().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_audio_info(
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    state.get_audio_info().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn cancel_download(
    app_handle: tauri::AppHandle,
    _state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    tracing::info!("Download cancellation requested - forcing stop");
    
    // Immediately emit cancellation event to frontend
    let _ = app_handle.emit("model-loading-complete", serde_json::json!({
        "success": false,
        "error": "Download cancelled by user"
    }));
    
    // Also emit a progress event to clear any loading indicators
    let _ = app_handle.emit("model-loading-progress", serde_json::json!({
        "progress": 0,
        "status": "Download cancelled",
        "model_name": ""
    }));
    
    tracing::info!("Cancellation events emitted");
    Ok(())
}

#[tauri::command]
async fn test_event_emission(
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    tracing::info!("Testing event emission");
    let _ = app_handle.emit("test-event", "Hello from backend!");
    Ok(())
}

#[tauri::command]
async fn get_last_selected_model(
    state: tauri::State<'_, AppState>,
) -> Result<Option<String>, String> {
    Ok(state.get_last_selected_model().await)
}

#[tauri::command]
async fn set_last_selected_model(
    model_name: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    state.set_last_selected_model(&model_name).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_available_models() -> Result<Vec<String>, String> {
    Ok(vec![
        "microsoft/phi-4".to_string(),
        "microsoft/Phi-4-mini-instruct".to_string(),
        "mlx-community/Phi-3.5-mini-instruct-4bit".to_string(),
        "mlx-community/SmolLM2-1.7B-Instruct-4bit".to_string(),
        "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        "distilbert-base-uncased".to_string(),
    ])
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("Starting Tektra AI Assistant");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Initialize application state
            let app_state = AppState::new(app.handle().clone())?;
            app.manage(app_state);
            
            info!("Application initialized successfully");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            send_message,
            start_voice_input,
            stop_voice_input,
            capture_image,
            load_model,
            get_model_status,
            download_model,
            list_cached_models,
            transcribe_voice,
            clear_voice_buffer,
            is_recording,
            speak_text,
            get_camera_info,
            get_audio_info,
            cancel_download,
            test_event_emission,
            get_last_selected_model,
            set_last_selected_model,
            get_available_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}