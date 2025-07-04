use super::*;
use std::collections::HashMap;

/// Token estimation and text processing utilities
pub struct TokenEstimator {
    model_specific_rates: HashMap<String, f32>,
    default_rate: f32,
}

impl TokenEstimator {
    pub fn new() -> Self {
        let mut model_specific_rates = HashMap::new();
        
        // Common token-to-character ratios for different model families
        model_specific_rates.insert("llama".to_string(), 0.25);    // ~4 chars per token
        model_specific_rates.insert("mistral".to_string(), 0.25);  // ~4 chars per token
        model_specific_rates.insert("qwen".to_string(), 0.3);      // ~3.3 chars per token
        model_specific_rates.insert("gemma".to_string(), 0.25);    // ~4 chars per token
        model_specific_rates.insert("phi".to_string(), 0.25);      // ~4 chars per token
        
        Self {
            model_specific_rates,
            default_rate: 0.25, // Default ~4 characters per token
        }
    }
    
    /// Estimate token count for text
    pub fn estimate_tokens(&self, text: &str) -> usize {
        self.estimate_tokens_for_model(text, None)
    }
    
    /// Estimate token count for specific model
    pub fn estimate_tokens_for_model(&self, text: &str, model_id: Option<&str>) -> usize {
        let rate = if let Some(model) = model_id {
            self.get_rate_for_model(model)
        } else {
            self.default_rate
        };
        
        // Basic estimation: character count * rate
        // More sophisticated estimation could use actual tokenizer
        let char_count = text.chars().count() as f32;
        (char_count * rate).ceil() as usize
    }
    
    /// Estimate tokens for multimodal input
    pub fn estimate_multimodal_tokens(&self, input: &MultimodalInput, model_id: Option<&str>) -> usize {
        let mut total_tokens = 0;
        
        match input {
            MultimodalInput::Text(text) => {
                total_tokens += self.estimate_tokens_for_model(text, model_id);
            }
            MultimodalInput::TextWithImage { text, image } => {
                total_tokens += self.estimate_tokens_for_model(text, model_id);
                total_tokens += self.estimate_image_tokens(image);
            }
            MultimodalInput::TextWithAudio { text, audio } => {
                total_tokens += self.estimate_tokens_for_model(text, model_id);
                total_tokens += self.estimate_audio_tokens(audio);
            }
            MultimodalInput::TextWithDocument { text, document } => {
                total_tokens += self.estimate_tokens_for_model(text, model_id);
                total_tokens += self.estimate_document_tokens(document);
            }
            MultimodalInput::Combined { text, images, audio, documents } => {
                if let Some(text) = text {
                    total_tokens += self.estimate_tokens_for_model(text, model_id);
                }
                for image in images {
                    total_tokens += self.estimate_image_tokens(image);
                }
                if let Some(audio) = audio {
                    total_tokens += self.estimate_audio_tokens(audio);
                }
                for document in documents {
                    total_tokens += self.estimate_document_tokens(document);
                }
            }
            MultimodalInput::TextWithVideo { text, video: _ } => {
                total_tokens += self.estimate_tokens_for_model(text, model_id);
                total_tokens += 512; // Rough estimate for video processing
            }
            MultimodalInput::RealTimeAudio { audio_stream: _ } => {
                total_tokens += 256; // Rough estimate for audio stream processing
            }
            MultimodalInput::MultimodalConversation { 
                text, images, audio, video: _, documents, .. 
            } => {
                if let Some(text) = text {
                    total_tokens += self.estimate_tokens_for_model(text, model_id);
                }
                for image in images {
                    total_tokens += self.estimate_image_tokens(image);
                }
                if let Some(audio) = audio {
                    total_tokens += self.estimate_audio_tokens(audio);
                }
                for document in documents {
                    total_tokens += self.estimate_document_tokens(document);
                }
                // Add overhead for conversation context
                total_tokens += 128;
            }
        }
        
        total_tokens
    }
    
    /// Estimate tokens for image input
    fn estimate_image_tokens(&self, image: &ImageData) -> usize {
        // Vision models typically use a fixed number of tokens per image
        // This varies by model but is often around 256-1024 tokens
        let base_tokens = 256;
        
        // Adjust based on image size if available
        if let (Some(width), Some(height)) = (image.width, image.height) {
            let pixels = width * height;
            
            // Larger images may use more tokens (up to a limit)
            let size_multiplier = if pixels > 1_000_000 {
                4.0 // High resolution
            } else if pixels > 500_000 {
                2.0 // Medium resolution
            } else {
                1.0 // Standard resolution
            };
            
            (base_tokens as f32 * size_multiplier) as usize
        } else {
            base_tokens
        }
    }
    
    /// Estimate tokens for audio input
    fn estimate_audio_tokens(&self, audio: &AudioData) -> usize {
        // Audio is typically converted to text first, so estimate based on duration
        if let Some(duration) = audio.duration {
            // Estimate ~150 words per minute of speech
            // ~1.3 tokens per word average
            let words_per_second = 150.0 / 60.0;
            let estimated_words = duration * words_per_second;
            (estimated_words * 1.3) as usize
        } else {
            // Default estimate for unknown duration
            100
        }
    }
    
    /// Estimate tokens for document input
    fn estimate_document_tokens(&self, document: &DocumentData) -> usize {
        // Try to extract text from document and estimate tokens
        match document.format {
            DocumentFormat::Txt | DocumentFormat::Markdown => {
                if let Ok(text) = std::str::from_utf8(&document.data) {
                    self.estimate_tokens(text)
                } else {
                    0
                }
            }
            DocumentFormat::Json => {
                if let Ok(text) = std::str::from_utf8(&document.data) {
                    // JSON has more overhead, so multiply by factor
                    (self.estimate_tokens(text) as f32 * 1.2) as usize
                } else {
                    0
                }
            }
            DocumentFormat::Pdf | DocumentFormat::Docx => {
                // For binary formats, estimate based on file size
                // Very rough approximation: 1 token per 5 bytes
                document.data.len() / 5
            }
        }
    }
    
    /// Truncate text to fit within token limit
    pub fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> String {
        self.truncate_to_tokens_for_model(text, max_tokens, None)
    }
    
    /// Truncate text to fit within token limit for specific model
    pub fn truncate_to_tokens_for_model(&self, text: &str, max_tokens: usize, model_id: Option<&str>) -> String {
        let current_tokens = self.estimate_tokens_for_model(text, model_id);
        
        if current_tokens <= max_tokens {
            return text.to_string();
        }
        
        // Calculate how much text to keep
        let ratio = max_tokens as f32 / current_tokens as f32;
        let chars_to_keep = (text.chars().count() as f32 * ratio) as usize;
        
        if chars_to_keep < 10 {
            return "...".to_string();
        }
        
        // Truncate and add ellipsis
        let truncated: String = text.chars().take(chars_to_keep - 3).collect();
        format!("{}...", truncated)
    }
    
    /// Split text into chunks that fit within token limits
    pub fn chunk_text(&self, text: &str, max_tokens_per_chunk: usize, overlap_tokens: usize, model_id: Option<&str>) -> Vec<String> {
        let total_tokens = self.estimate_tokens_for_model(text, model_id);
        
        if total_tokens <= max_tokens_per_chunk {
            return vec![text.to_string()];
        }
        
        let mut chunks = Vec::new();
        let rate = self.get_rate_for_model(model_id.unwrap_or(""));
        
        // Calculate characters per chunk
        let chars_per_chunk = (max_tokens_per_chunk as f32 / rate) as usize;
        let overlap_chars = (overlap_tokens as f32 / rate) as usize;
        
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        
        while start < chars.len() {
            let end = std::cmp::min(start + chars_per_chunk, chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);
            
            if end >= chars.len() {
                break;
            }
            
            // Move start position with overlap
            start = end - overlap_chars;
        }
        
        chunks
    }
    
    /// Get context window utilization
    pub fn get_utilization(&self, input: &MultimodalInput, context_window: usize, model_id: Option<&str>) -> ContextUtilization {
        let used_tokens = self.estimate_multimodal_tokens(input, model_id);
        let utilization_percentage = (used_tokens as f32 / context_window as f32 * 100.0).min(100.0);
        
        let status = if utilization_percentage >= 95.0 {
            UtilizationStatus::Critical
        } else if utilization_percentage >= 80.0 {
            UtilizationStatus::High
        } else if utilization_percentage >= 50.0 {
            UtilizationStatus::Medium
        } else {
            UtilizationStatus::Low
        };
        
        ContextUtilization {
            used_tokens,
            total_tokens: context_window,
            remaining_tokens: context_window.saturating_sub(used_tokens),
            utilization_percentage,
            status,
        }
    }
    
    /// Get token rate for specific model
    fn get_rate_for_model(&self, model_id: &str) -> f32 {
        let model_lower = model_id.to_lowercase();
        
        for (family, rate) in &self.model_specific_rates {
            if model_lower.contains(family) {
                return *rate;
            }
        }
        
        self.default_rate
    }
    
    /// Calculate processing cost estimation (for future API usage tracking)
    pub fn estimate_processing_cost(&self, input: &MultimodalInput, model_id: Option<&str>) -> ProcessingCostEstimate {
        let input_tokens = self.estimate_multimodal_tokens(input, model_id);
        
        // Rough estimates - would need to be calibrated with actual model performance
        let base_cost_per_token = 0.0001; // $0.0001 per token (hypothetical)
        let vision_multiplier = 10.0; // Vision processing is more expensive
        let audio_multiplier = 5.0;   // Audio processing multiplier
        
        let mut total_cost = input_tokens as f32 * base_cost_per_token;
        
        // Add multimodal processing costs
        match input {
            MultimodalInput::TextWithImage { image, .. } => {
                let image_tokens = self.estimate_image_tokens(image);
                total_cost += image_tokens as f32 * base_cost_per_token * vision_multiplier;
            }
            MultimodalInput::TextWithAudio { audio, .. } => {
                let audio_tokens = self.estimate_audio_tokens(audio);
                total_cost += audio_tokens as f32 * base_cost_per_token * audio_multiplier;
            }
            MultimodalInput::Combined { images, audio, .. } => {
                for image in images {
                    let image_tokens = self.estimate_image_tokens(image);
                    total_cost += image_tokens as f32 * base_cost_per_token * vision_multiplier;
                }
                if let Some(audio) = audio {
                    let audio_tokens = self.estimate_audio_tokens(audio);
                    total_cost += audio_tokens as f32 * base_cost_per_token * audio_multiplier;
                }
            }
            _ => {}
        }
        
        ProcessingCostEstimate {
            input_tokens,
            estimated_cost: total_cost,
            cost_breakdown: HashMap::new(), // Could be expanded with detailed breakdown
        }
    }
}

impl Default for TokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextUtilization {
    pub used_tokens: usize,
    pub total_tokens: usize,
    pub remaining_tokens: usize,
    pub utilization_percentage: f32,
    pub status: UtilizationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UtilizationStatus {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingCostEstimate {
    pub input_tokens: usize,
    pub estimated_cost: f32,
    pub cost_breakdown: HashMap<String, f32>,
}

/// Text processing utilities for optimizing model input
pub struct TextProcessor;

impl TextProcessor {
    /// Clean and normalize text for better token efficiency
    pub fn normalize_text(text: &str) -> String {
        // Remove excessive whitespace
        let cleaned = text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        
        // Normalize common unicode characters
        let normalized = cleaned
            .replace("'", "'")
            .replace("'", "'")
            .replace("\u{201C}", "\"")
            .replace("\u{201D}", "\"")
            .replace("–", "-")
            .replace("—", "-");
        
        normalized
    }
    
    /// Extract key content from text while preserving meaning
    pub fn extract_key_content(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            return Self::normalize_text(text);
        }
        
        let sentences: Vec<&str> = text.split('.').collect();
        let mut result = String::new();
        
        for sentence in sentences {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                let potential_addition = if result.is_empty() {
                    format!("{}.", trimmed)
                } else {
                    format!(" {}.", trimmed)
                };
                
                if result.len() + potential_addition.len() <= max_length {
                    result.push_str(&potential_addition);
                } else {
                    break;
                }
            }
        }
        
        if result.is_empty() {
            // If no complete sentences fit, truncate at word boundary
            let words: Vec<&str> = text.split_whitespace().collect();
            for word in words {
                let potential_addition = if result.is_empty() {
                    word.to_string()
                } else {
                    format!(" {}", word)
                };
                
                if result.len() + potential_addition.len() <= max_length - 3 {
                    result.push_str(&potential_addition);
                } else {
                    result.push_str("...");
                    break;
                }
            }
        }
        
        Self::normalize_text(&result)
    }
    
    /// Summarize text to reduce token usage
    pub fn summarize_for_context(text: &str, target_ratio: f32) -> String {
        if target_ratio >= 1.0 {
            return Self::normalize_text(text);
        }
        
        let target_length = (text.len() as f32 * target_ratio) as usize;
        Self::extract_key_content(text, target_length)
    }
}

/// Advanced token management for conversation contexts
pub struct ConversationTokenManager {
    estimator: TokenEstimator,
    max_context_tokens: usize,
    reserved_tokens: usize,
}

impl ConversationTokenManager {
    pub fn new(max_context_tokens: usize) -> Self {
        Self {
            estimator: TokenEstimator::new(),
            max_context_tokens,
            reserved_tokens: max_context_tokens / 4, // Reserve 25% for response
        }
    }
    
    /// Manage conversation history to fit within context limits
    pub fn manage_conversation_context(
        &self,
        conversation_history: &[String],
        new_input: &str,
        model_id: Option<&str>,
    ) -> Vec<String> {
        let available_tokens = self.max_context_tokens - self.reserved_tokens;
        let new_input_tokens = self.estimator.estimate_tokens_for_model(new_input, model_id);
        
        if new_input_tokens >= available_tokens {
            // New input alone exceeds context, truncate it
            let truncated_input = self.estimator.truncate_to_tokens_for_model(
                new_input,
                available_tokens,
                model_id,
            );
            return vec![truncated_input];
        }
        
        let mut result = Vec::new();
        let mut used_tokens = new_input_tokens;
        
        // Add messages from most recent to oldest
        for message in conversation_history.iter().rev() {
            let message_tokens = self.estimator.estimate_tokens_for_model(message, model_id);
            
            if used_tokens + message_tokens <= available_tokens {
                result.insert(0, message.clone());
                used_tokens += message_tokens;
            } else {
                // Try to fit a truncated version of this message
                let remaining_tokens = available_tokens - used_tokens;
                if remaining_tokens > 50 { // Only if we have reasonable space left
                    let truncated = self.estimator.truncate_to_tokens_for_model(
                        message,
                        remaining_tokens,
                        model_id,
                    );
                    result.insert(0, truncated);
                }
                break;
            }
        }
        
        // Add the new input
        result.push(new_input.to_string());
        
        result
    }
    
    /// Get recommendations for context optimization
    pub fn get_optimization_recommendations(&self, input: &MultimodalInput, model_id: Option<&str>) -> Vec<String> {
        let utilization = self.estimator.get_utilization(input, self.max_context_tokens, model_id);
        let mut recommendations = Vec::new();
        
        match utilization.status {
            UtilizationStatus::Critical => {
                recommendations.push("Context utilization is critical (>95%). Consider reducing input size.".to_string());
                recommendations.push("Try summarizing or chunking your input.".to_string());
            }
            UtilizationStatus::High => {
                recommendations.push("Context utilization is high (>80%). Monitor for potential issues.".to_string());
                recommendations.push("Consider optimizing text for better token efficiency.".to_string());
            }
            UtilizationStatus::Medium => {
                recommendations.push("Context utilization is moderate. You have room for longer conversations.".to_string());
            }
            UtilizationStatus::Low => {
                recommendations.push("Context utilization is low. You can add more context if needed.".to_string());
            }
        }
        
        recommendations
    }
}