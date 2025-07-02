use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// Represents a chat message with role and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Template for formatting prompts for different model families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Name of the template (e.g., "gemma", "llama", "mistral")
    pub name: String,
    
    /// System prompt prefix
    pub system_prefix: Option<String>,
    
    /// System prompt suffix
    pub system_suffix: Option<String>,
    
    /// User message prefix
    pub user_prefix: String,
    
    /// User message suffix
    pub user_suffix: String,
    
    /// Assistant message prefix
    pub assistant_prefix: String,
    
    /// Assistant message suffix
    pub assistant_suffix: String,
    
    /// Whether to include a BOS token
    pub add_bos: bool,
    
    /// Whether to include an EOS token
    pub add_eos: bool,
    
    /// Special tokens for multimodal inputs
    pub multimodal_markers: MultimodalMarkers,
    
    /// Whether this template supports system messages
    pub supports_system: bool,
    
    /// Custom stop sequences for this template
    pub stop_sequences: Vec<String>,
}

/// Markers for different modalities in prompts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalMarkers {
    /// Marker for image inputs (e.g., "<image>", "[IMG]", etc.)
    pub image_start: String,
    pub image_end: String,
    
    /// Marker for audio inputs
    pub audio_start: String,
    pub audio_end: String,
    
    /// Marker for video inputs
    pub video_start: String,
    pub video_end: String,
    
    /// Whether to replace content or wrap it
    pub replace_content: bool,
}

impl Default for MultimodalMarkers {
    fn default() -> Self {
        Self {
            image_start: "<image>".to_string(),
            image_end: "</image>".to_string(),
            audio_start: "<audio>".to_string(),
            audio_end: "</audio>".to_string(),
            video_start: "<video>".to_string(),
            video_end: "</video>".to_string(),
            replace_content: false,
        }
    }
}

/// Manager for handling model-specific prompt templates
#[derive(Clone)]
pub struct TemplateManager {
    templates: Arc<RwLock<HashMap<String, PromptTemplate>>>,
    model_mappings: Arc<RwLock<HashMap<String, String>>>,
}

impl TemplateManager {
    pub fn new() -> Self {
        let manager = Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            model_mappings: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Clone for the async task
        let manager_clone = manager.clone();
        
        // Initialize with default templates
        tokio::spawn(async move {
            if let Err(e) = manager_clone.load_default_templates().await {
                warn!("Failed to load default templates: {}", e);
            }
        });
        
        manager
    }
    
    /// Load default templates for common model families
    async fn load_default_templates(&self) -> Result<()> {
        // Gemma template
        self.add_template(PromptTemplate {
            name: "gemma".to_string(),
            system_prefix: Some("<start_of_turn>system\n".to_string()),
            system_suffix: Some("<end_of_turn>\n".to_string()),
            user_prefix: "<start_of_turn>user\n".to_string(),
            user_suffix: "<end_of_turn>\n".to_string(),
            assistant_prefix: "<start_of_turn>model\n".to_string(),
            assistant_suffix: "<end_of_turn>\n".to_string(),
            add_bos: true,
            add_eos: false,
            multimodal_markers: MultimodalMarkers::default(),
            supports_system: true,
            stop_sequences: vec!["<end_of_turn>".to_string()],
        }).await?;
        
        // Llama template
        self.add_template(PromptTemplate {
            name: "llama".to_string(),
            system_prefix: Some("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n".to_string()),
            system_suffix: Some("<|eot_id|>".to_string()),
            user_prefix: "<|start_header_id|>user<|end_header_id|>\n\n".to_string(),
            user_suffix: "<|eot_id|>".to_string(),
            assistant_prefix: "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string(),
            assistant_suffix: "<|eot_id|>".to_string(),
            add_bos: false,
            add_eos: false,
            multimodal_markers: MultimodalMarkers {
                image_start: "<|image|>".to_string(),
                image_end: "".to_string(),
                replace_content: true,
                ..Default::default()
            },
            supports_system: true,
            stop_sequences: vec!["<|eot_id|>".to_string()],
        }).await?;
        
        // Mistral/Mixtral template
        self.add_template(PromptTemplate {
            name: "mistral".to_string(),
            system_prefix: Some("[INST] ".to_string()),
            system_suffix: Some(" [/INST]".to_string()),
            user_prefix: "[INST] ".to_string(),
            user_suffix: " [/INST]".to_string(),
            assistant_prefix: "".to_string(),
            assistant_suffix: "</s>".to_string(),
            add_bos: true,
            add_eos: false,
            multimodal_markers: MultimodalMarkers::default(),
            supports_system: false, // Mistral merges system into first user message
            stop_sequences: vec!["</s>".to_string()],
        }).await?;
        
        // ChatML template (used by many models)
        self.add_template(PromptTemplate {
            name: "chatml".to_string(),
            system_prefix: Some("<|im_start|>system\n".to_string()),
            system_suffix: Some("<|im_end|>\n".to_string()),
            user_prefix: "<|im_start|>user\n".to_string(),
            user_suffix: "<|im_end|>\n".to_string(),
            assistant_prefix: "<|im_start|>assistant\n".to_string(),
            assistant_suffix: "<|im_end|>\n".to_string(),
            add_bos: false,
            add_eos: false,
            multimodal_markers: MultimodalMarkers::default(),
            supports_system: true,
            stop_sequences: vec!["<|im_end|>".to_string()],
        }).await?;
        
        // Phi template
        self.add_template(PromptTemplate {
            name: "phi".to_string(),
            system_prefix: Some("System: ".to_string()),
            system_suffix: Some("\n\n".to_string()),
            user_prefix: "User: ".to_string(),
            user_suffix: "\n".to_string(),
            assistant_prefix: "Assistant: ".to_string(),
            assistant_suffix: "\n".to_string(),
            add_bos: false,
            add_eos: false,
            multimodal_markers: MultimodalMarkers::default(),
            supports_system: true,
            stop_sequences: vec!["\n\n".to_string()],
        }).await?;
        
        // Qwen template
        self.add_template(PromptTemplate {
            name: "qwen".to_string(),
            system_prefix: Some("<|im_start|>system\n".to_string()),
            system_suffix: Some("<|im_end|>\n".to_string()),
            user_prefix: "<|im_start|>user\n".to_string(),
            user_suffix: "<|im_end|>\n".to_string(),
            assistant_prefix: "<|im_start|>assistant\n".to_string(),
            assistant_suffix: "<|im_end|>\n".to_string(),
            add_bos: false,
            add_eos: false,
            multimodal_markers: MultimodalMarkers {
                image_start: "<img>".to_string(),
                image_end: "</img>".to_string(),
                ..Default::default()
            },
            supports_system: true,
            stop_sequences: vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()],
        }).await?;
        
        // Add model mappings
        self.add_model_mapping("gemma", vec![
            "gemma", "gemma2", "gemma3n"
        ]).await?;
        
        self.add_model_mapping("llama", vec![
            "llama", "llama2", "llama3", "codellama"
        ]).await?;
        
        self.add_model_mapping("mistral", vec![
            "mistral", "mixtral", "mistral-nemo"
        ]).await?;
        
        self.add_model_mapping("phi", vec![
            "phi", "phi-2", "phi-3"
        ]).await?;
        
        self.add_model_mapping("qwen", vec![
            "qwen", "qwen2", "qwen2.5"
        ]).await?;
        
        info!("Loaded {} default templates", self.templates.read().await.len());
        Ok(())
    }
    
    /// Add a new template
    pub async fn add_template(&self, template: PromptTemplate) -> Result<()> {
        let name = template.name.clone();
        self.templates.write().await.insert(name.clone(), template);
        debug!("Added template: {}", name);
        Ok(())
    }
    
    /// Add model to template mappings
    pub async fn add_model_mapping(&self, template_name: &str, model_patterns: Vec<&str>) -> Result<()> {
        let mut mappings = self.model_mappings.write().await;
        for pattern in model_patterns {
            mappings.insert(pattern.to_string(), template_name.to_string());
        }
        Ok(())
    }
    
    /// Get template by name
    pub async fn get_template(&self, name: &str) -> Option<PromptTemplate> {
        self.templates.read().await.get(name).cloned()
    }
    
    /// Get template for a specific model
    pub async fn get_template_for_model(&self, model_id: &str) -> Option<PromptTemplate> {
        // First, check exact match
        if let Some(template) = self.get_template(model_id).await {
            return Some(template);
        }
        
        // Check model mappings
        let mappings = self.model_mappings.read().await;
        let model_lower = model_id.to_lowercase();
        
        // Find best matching pattern
        for (pattern, template_name) in mappings.iter() {
            if model_lower.contains(pattern) {
                if let Some(template) = self.get_template(template_name).await {
                    return Some(template);
                }
            }
        }
        
        // Default to ChatML if no match found
        self.get_template("chatml").await
    }
    
    /// Format a conversation into a prompt using the specified template
    pub fn format_prompt(
        &self,
        template: &PromptTemplate,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> String {
        let mut prompt = String::new();
        
        if template.add_bos {
            prompt.push_str("<s>");
        }
        
        let mut system_message = None;
        let mut conversation = Vec::new();
        
        // Separate system message if present
        for message in messages {
            match message.role {
                MessageRole::System => system_message = Some(message.content.clone()),
                _ => conversation.push(message.clone()),
            }
        }
        
        // Handle system message
        if let Some(system_content) = system_message {
            if template.supports_system {
                if let Some(prefix) = &template.system_prefix {
                    prompt.push_str(prefix);
                }
                prompt.push_str(&system_content);
                if let Some(suffix) = &template.system_suffix {
                    prompt.push_str(suffix);
                }
            } else {
                // Merge system message into first user message for templates that don't support it
                if let Some(first_user) = conversation.iter_mut().find(|m| m.role == MessageRole::User) {
                    first_user.content = format!("{}\n\n{}", system_content, first_user.content);
                }
            }
        }
        
        // Format conversation messages
        for message in &conversation {
            match message.role {
                MessageRole::User => {
                    prompt.push_str(&template.user_prefix);
                    prompt.push_str(&message.content);
                    prompt.push_str(&template.user_suffix);
                }
                MessageRole::Assistant => {
                    prompt.push_str(&template.assistant_prefix);
                    prompt.push_str(&message.content);
                    prompt.push_str(&template.assistant_suffix);
                }
                MessageRole::System => {} // Already handled
            }
        }
        
        // Add generation prompt if requested
        if add_generation_prompt {
            prompt.push_str(&template.assistant_prefix);
        }
        
        prompt
    }
    
    /// Format multimodal content with appropriate markers
    pub fn format_multimodal_content(
        &self,
        template: &PromptTemplate,
        text: &str,
        has_image: bool,
        has_audio: bool,
        has_video: bool,
    ) -> String {
        let mut content = String::new();
        let markers = &template.multimodal_markers;
        
        // Add multimodal markers
        if has_image {
            if markers.replace_content {
                content.push_str(&markers.image_start);
            } else {
                content.push_str(&markers.image_start);
                content.push_str("IMAGE_DATA");
                content.push_str(&markers.image_end);
                content.push('\n');
            }
        }
        
        if has_audio {
            content.push_str(&markers.audio_start);
            if !markers.replace_content {
                content.push_str("AUDIO_DATA");
            }
            content.push_str(&markers.audio_end);
            content.push('\n');
        }
        
        if has_video {
            content.push_str(&markers.video_start);
            if !markers.replace_content {
                content.push_str("VIDEO_DATA");
            }
            content.push_str(&markers.video_end);
            content.push('\n');
        }
        
        // Add text content
        content.push_str(text);
        
        content
    }
    
    /// Load templates from a configuration file
    pub async fn load_from_file(&self, path: &str) -> Result<()> {
        let content = tokio::fs::read_to_string(path).await?;
        let templates: Vec<PromptTemplate> = serde_json::from_str(&content)?;
        
        let count = templates.len();
        for template in templates {
            self.add_template(template).await?;
        }
        
        info!("Loaded {} templates from {}", count, path);
        Ok(())
    }
    
    /// Save current templates to a configuration file
    pub async fn save_to_file(&self, path: &str) -> Result<()> {
        let templates = self.templates.read().await;
        let templates_vec: Vec<&PromptTemplate> = templates.values().collect();
        let content = serde_json::to_string_pretty(&templates_vec)?;
        
        tokio::fs::write(path, content).await?;
        
        info!("Saved {} templates to {}", templates.len(), path);
        Ok(())
    }
    
    /// List all available templates
    pub async fn list_templates(&self) -> Vec<String> {
        self.templates.read().await.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_template_formatting() {
        let manager = TemplateManager::new();
        
        // Wait for default templates to load
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Test Gemma template
        let gemma_template = manager.get_template("gemma").await.unwrap();
        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: MessageRole::User,
                content: "Hello!".to_string(),
            },
        ];
        
        let prompt = manager.format_prompt(&gemma_template, &messages, true);
        assert!(prompt.contains("<start_of_turn>system"));
        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }
    
    #[tokio::test]
    async fn test_model_mapping() {
        let manager = TemplateManager::new();
        
        // Wait for default templates to load
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Test model mapping
        let template = manager.get_template_for_model("gemma3n:e4b").await.unwrap();
        assert_eq!(template.name, "gemma");
        
        let template = manager.get_template_for_model("llama3-8b-instruct").await.unwrap();
        assert_eq!(template.name, "llama");
    }
    
    #[tokio::test]
    async fn test_multimodal_formatting() {
        let manager = TemplateManager::new();
        
        // Wait for default templates to load
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let template = manager.get_template("gemma").await.unwrap();
        let content = manager.format_multimodal_content(
            &template,
            "What's in this image?",
            true,
            false,
            false,
        );
        
        assert!(content.contains("<image>"));
        assert!(content.contains("</image>"));
        assert!(content.contains("What's in this image?"));
    }
}