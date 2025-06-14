// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}