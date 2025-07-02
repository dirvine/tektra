use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub model_name: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub voice_enabled: bool,
    pub auto_speech: bool,
    pub system_prompt: Option<String>,
    pub user_prefix: Option<String>,
    pub assistant_prefix: Option<String>,
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