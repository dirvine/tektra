use super::*;
use anyhow::Result;
use std::collections::HashMap;

/// Prompt template management for different models and use cases
pub struct PromptTemplateManager {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptTemplateManager {
    pub fn new() -> Result<Self> {
        let mut templates = HashMap::new();
        
        // Add default templates
        templates.insert("qwen2.5-vl".to_string(), PromptTemplate {
            name: "Qwen2.5-VL".to_string(),
            system_prefix: "<|im_start|>system\n".to_string(),
            system_suffix: "<|im_end|>\n".to_string(),
            user_prefix: "<|im_start|>user\n".to_string(),
            user_suffix: "<|im_end|>\n".to_string(),
            assistant_prefix: "<|im_start|>assistant\n".to_string(),
            assistant_suffix: "<|im_end|>\n".to_string(),
            supports_system: true,
            supports_images: true,
            image_token: "<image>".to_string(),
            stop_sequences: vec!["<|im_end|>".to_string()],
        });
        
        templates.insert("llama3.2-vision".to_string(), PromptTemplate {
            name: "Llama 3.2 Vision".to_string(),
            system_prefix: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n".to_string(),
            system_suffix: "<|eot_id|>".to_string(),
            user_prefix: "<|start_header_id|>user<|end_header_id|>\n\n".to_string(),
            user_suffix: "<|eot_id|>".to_string(),
            assistant_prefix: "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string(),
            assistant_suffix: "<|eot_id|>".to_string(),
            supports_system: true,
            supports_images: true,
            image_token: "<image>".to_string(),
            stop_sequences: vec!["<|eot_id|>".to_string()],
        });
        
        Ok(Self { templates })
    }
    
    pub fn get_template(&self, model_id: &str) -> Option<&PromptTemplate> {
        // Try exact match first
        if let Some(template) = self.templates.get(model_id) {
            return Some(template);
        }
        
        // Try partial matches
        for (key, template) in &self.templates {
            if model_id.contains(key) {
                return Some(template);
            }
        }
        
        None
    }
    
    pub fn format_conversation(&self, template: &PromptTemplate, context: &ConversationContext) -> String {
        let mut formatted = String::new();
        
        // Add system prompt if supported and present
        if template.supports_system {
            if let Some(system_prompt) = &context.system_prompt {
                formatted.push_str(&template.system_prefix);
                formatted.push_str(system_prompt);
                formatted.push_str(&template.system_suffix);
            }
        }
        
        // Add conversation messages
        for message in &context.messages {
            match message.role {
                MessageRole::User => {
                    formatted.push_str(&template.user_prefix);
                    formatted.push_str(&message.content);
                    formatted.push_str(&template.user_suffix);
                }
                MessageRole::Assistant => {
                    formatted.push_str(&template.assistant_prefix);
                    formatted.push_str(&message.content);
                    formatted.push_str(&template.assistant_suffix);
                }
                MessageRole::System => {
                    if template.supports_system {
                        formatted.push_str(&template.system_prefix);
                        formatted.push_str(&message.content);
                        formatted.push_str(&template.system_suffix);
                    }
                }
            }
        }
        
        // Add assistant prefix for generation
        formatted.push_str(&template.assistant_prefix);
        
        formatted
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub name: String,
    pub system_prefix: String,
    pub system_suffix: String,
    pub user_prefix: String,
    pub user_suffix: String,
    pub assistant_prefix: String,
    pub assistant_suffix: String,
    pub supports_system: bool,
    pub supports_images: bool,
    pub image_token: String,
    pub stop_sequences: Vec<String>,
}