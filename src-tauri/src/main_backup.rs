// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tauri::State;
use tokio::sync::Mutex;

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
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            model_name: "Built-in Assistant".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            voice_enabled: true,
            auto_speech: false,
        }
    }
}

// Simple built-in responses for demonstration
struct LocalAssistant {
    responses: HashMap<String, Vec<String>>,
}

impl LocalAssistant {
    fn new() -> Self {
        let mut responses = HashMap::new();
        
        // Greetings
        responses.insert("hello".to_string(), vec![
            "Hello! I'm your local AI assistant. How can I help you today?".to_string(),
            "Hi there! What would you like to know?".to_string(),
            "Greetings! I'm here to assist you.".to_string(),
        ]);
        
        responses.insert("hi".to_string(), vec![
            "Hi! How can I assist you?".to_string(),
            "Hello! What can I do for you?".to_string(),
        ]);
        
        // Questions
        responses.insert("how are you".to_string(), vec![
            "I'm doing well, thank you! I'm a local AI assistant running entirely on your device.".to_string(),
            "I'm great! No internet required - I'm running completely offline on your machine.".to_string(),
        ]);
        
        responses.insert("what are you".to_string(), vec![
            "I'm Project Tektra, a local AI assistant built with Rust and Tauri. I run completely offline!".to_string(),
            "I'm your personal AI assistant that works entirely on your device without needing internet.".to_string(),
        ]);
        
        responses.insert("what can you do".to_string(), vec![
            "I can chat with you, answer questions, and help with various tasks. I'm completely self-contained and work offline!".to_string(),
            "I can have conversations, provide assistance, and help you with information. Best of all, I work without internet!".to_string(),
        ]);
        
        // Default responses
        responses.insert("default".to_string(), vec![
            "That's an interesting question! As a local AI assistant, I'm designed to help with various tasks.".to_string(),
            "I understand you're asking about that topic. While I'm a simplified local model, I'm happy to discuss it with you.".to_string(),
            "Thanks for sharing that with me! I'm here to help and chat about whatever you'd like.".to_string(),
            "I appreciate you talking with me! As your local AI assistant, I'm always ready to help.".to_string(),
            "That's a great point! I may be a simpler model, but I enjoy our conversations.".to_string(),
        ]);
        
        Self { responses }
    }
    
    fn generate_response(&self, input: &str) -> String {
        let input_lower = input.to_lowercase();
        
        // Find the best matching response category
        let category = if input_lower.contains("hello") || input_lower.contains("hi") {
            if input_lower.contains("hello") { "hello" } else { "hi" }
        } else if input_lower.contains("how are you") || input_lower.contains("how r u") {
            "how are you"
        } else if input_lower.contains("what are you") || input_lower.contains("who are you") {
            "what are you"
        } else if input_lower.contains("what can you do") || input_lower.contains("help me") {
            "what can you do"
        } else {
            "default"
        };
        
        // Get responses for the category
        let responses = self.responses.get(category).unwrap_or_else(|| {
            self.responses.get("default").unwrap()
        });
        
        // Return a random response
        let index = (input.len() + input_lower.chars().count()) % responses.len();
        responses[index].clone()
    }
}

type ChatHistory = Mutex<Vec<ChatMessage>>;
type Settings = Mutex<AppSettings>;
type Assistant = Mutex<LocalAssistant>;

#[tauri::command]
async fn initialize_model() -> Result<bool, String> {
    // For the built-in assistant, we don't need to download anything
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await; // Simulate loading
    Ok(true)
}

#[tauri::command]
async fn send_message(
    message: String,
    chat_history: State<'_, ChatHistory>,
    assistant: State<'_, Assistant>,
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

    // Generate response using the local assistant
    let assistant_guard = assistant.lock().await;
    let response = assistant_guard.generate_response(&message);
    drop(assistant_guard);
    
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
    
    // Simulate some processing time for realism
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
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
async fn check_model_status() -> Result<bool, String> {
    Ok(true) // Built-in model is always available
}

#[tauri::command]
async fn get_available_models() -> Result<Vec<String>, String> {
    Ok(vec![
        "Built-in Assistant".to_string(),
        "Local Pattern Matcher".to_string(),
    ])
}

fn main() {
    tauri::Builder::default()
        .manage(ChatHistory::new(Vec::new()))
        .manage(Settings::new(AppSettings::default()))
        .manage(Assistant::new(LocalAssistant::new()))
        .invoke_handler(tauri::generate_handler![
            initialize_model,
            send_message,
            get_chat_history,
            clear_chat_history,
            get_settings,
            update_settings,
            check_model_status,
            get_available_models
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}