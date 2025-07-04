use super::*;
use crate::inference::{ModelResponse, Token, FinishReason, UsageStats};
use anyhow::Result;
use serde_json::Value;

/// Output formatter for model responses
pub struct OutputFormatter {
    max_response_length: usize,
}

impl OutputFormatter {
    pub fn new() -> Self {
        Self {
            max_response_length: 50_000, // 50k characters max
        }
    }

    /// Format a model response for display
    pub fn format_response(&self, response: &ModelResponse) -> Result<FormattedResponse> {
        let mut formatted_text = response.text.clone();
        
        // Truncate if too long
        if formatted_text.len() > self.max_response_length {
            formatted_text.truncate(self.max_response_length - 3);
            formatted_text.push_str("...");
        }
        
        // Clean up formatting
        formatted_text = self.clean_markdown(&formatted_text);
        
        Ok(FormattedResponse {
            text: formatted_text,
            metadata: self.create_metadata(response),
            tokens: response.tokens.clone(),
            finish_reason: response.finish_reason.clone(),
        })
    }

    /// Clean up markdown formatting
    fn clean_markdown(&self, text: &str) -> String {
        // Remove excessive whitespace
        let text = text.trim();
        
        // Normalize line breaks
        let text = text.replace("\r\n", "\n");
        let text = text.replace('\r', "\n");
        
        // Remove excessive empty lines
        let lines: Vec<&str> = text.lines().collect();
        let mut cleaned_lines = Vec::new();
        let mut empty_line_count = 0;
        
        for line in lines {
            if line.trim().is_empty() {
                empty_line_count += 1;
                if empty_line_count <= 2 {
                    cleaned_lines.push(line);
                }
            } else {
                empty_line_count = 0;
                cleaned_lines.push(line);
            }
        }
        
        cleaned_lines.join("\n")
    }

    /// Create metadata for the response
    fn create_metadata(&self, response: &ModelResponse) -> ResponseMetadata {
        ResponseMetadata {
            token_count: response.tokens.len(),
            character_count: response.text.len(),
            word_count: response.text.split_whitespace().count(),
            finish_reason: format!("{:?}", response.finish_reason),
            usage_stats: response.usage.clone(),
        }
    }

    /// Format streaming token for display
    pub fn format_streaming_token(&self, token: &Token) -> Result<String> {
        // For streaming, just return the token text
        // Could add special formatting for special tokens in the future
        if token.special {
            Ok(String::new()) // Don't display special tokens
        } else {
            Ok(token.text.clone())
        }
    }

    /// Format error for display
    pub fn format_error(&self, error: &anyhow::Error) -> FormattedError {
        FormattedError {
            message: format!("Error: {}", error),
            error_type: "InferenceError".to_string(),
            recoverable: true, // Most errors are recoverable
        }
    }

    /// Extract structured data from response
    pub fn extract_structured_data(&self, response: &ModelResponse) -> Option<Value> {
        // Try to parse JSON from code blocks
        let text = &response.text;
        
        // Look for JSON code blocks
        if let Some(json_start) = text.find("```json") {
            if let Some(json_end) = text[json_start..].find("```") {
                let json_content = &text[json_start + 7..json_start + json_end];
                if let Ok(value) = serde_json::from_str::<Value>(json_content) {
                    return Some(value);
                }
            }
        }
        
        // Look for inline JSON
        if text.trim().starts_with('{') && text.trim().ends_with('}') {
            if let Ok(value) = serde_json::from_str::<Value>(text.trim()) {
                return Some(value);
            }
        }
        
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedResponse {
    pub text: String,
    pub metadata: ResponseMetadata,
    pub tokens: Vec<Token>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub token_count: usize,
    pub character_count: usize,
    pub word_count: usize,
    pub finish_reason: String,
    pub usage_stats: UsageStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedError {
    pub message: String,
    pub error_type: String,
    pub recoverable: bool,
}