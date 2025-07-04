use super::*;
use anyhow::Result;
use std::collections::HashMap;
use tracing::{info, warn, debug};

/// Advanced context management for maintaining conversation coherence
pub struct ContextManager {
    max_context_length: usize,
    token_estimator: TokenEstimator,
}

impl ContextManager {
    pub fn new(max_context_length: usize) -> Result<Self> {
        info!("Initializing context manager with max length: {}", max_context_length);
        
        Ok(Self {
            max_context_length,
            token_estimator: TokenEstimator::new(),
        })
    }
    
    /// Prepare conversation context for model input
    pub async fn prepare_context(
        &self,
        session: &ConversationSession,
        memory_context: Option<&MemoryContext>,
        model_name: Option<&str>,
    ) -> Result<ConversationContext> {
        debug!("Preparing conversation context for session: {}", session.id);
        
        let mut context = ConversationContext {
            messages: Vec::new(),
            system_prompt: None,
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            streaming: false,
            tools: Vec::new(),
            memory_context: memory_context.cloned(),
        };
        
        // Add system prompt based on persona
        if let Some(system_prompt) = &session.persona.system_prompt {
            context.system_prompt = Some(system_prompt.clone());
        }
        
        // Add memory context if available
        if let Some(memory) = memory_context {
            if !memory.relevant_history.is_empty() {
                // Add memory as system context
                let memory_prompt = self.format_memory_context(memory);
                if let Some(existing_prompt) = &context.system_prompt {
                    context.system_prompt = Some(format!("{}\n\n{}", existing_prompt, memory_prompt));
                } else {
                    context.system_prompt = Some(memory_prompt);
                }
            }
        }
        
        // Convert conversation history to messages
        let recent_history = session.get_recent_context();
        context.messages = self.convert_turns_to_messages(recent_history).await?;
        
        // Apply token budget management
        self.apply_token_budget(&mut context, model_name).await?;
        
        // Configure generation parameters based on persona
        self.apply_persona_parameters(&mut context, &session.persona);
        
        debug!("Context prepared with {} messages", context.messages.len());
        Ok(context)
    }
    
    /// Apply sliding window to manage context length
    pub async fn apply_sliding_window(
        &self,
        session: &mut ConversationSession,
        config: &ConversationConfig,
    ) -> Result<()> {
        info!("Applying sliding window to session: {}", session.id);
        
        let current_length = session.history.len();
        if current_length <= config.sliding_window_size {
            return Ok(());
        }
        
        // Calculate how many turns to remove
        let turns_to_remove = current_length - config.sliding_window_size;
        
        // Always preserve system messages if configured
        let mut preserved_turns = Vec::new();
        let mut regular_turns = Vec::new();
        
        for turn in session.history.iter() {
            match &turn.content {
                TurnContent::System(_) if config.preserve_system_messages => {
                    preserved_turns.push(turn.clone());
                }
                _ => {
                    regular_turns.push(turn.clone());
                }
            }
        }
        
        // Remove oldest regular turns
        if regular_turns.len() > turns_to_remove {
            regular_turns.drain(0..turns_to_remove);
        }
        
        // Reconstruct history with preserved system messages at the beginning
        session.history.clear();
        for turn in preserved_turns {
            session.history.push_back(turn);
        }
        for turn in regular_turns {
            session.history.push_back(turn);
        }
        
        // Generate summary of removed content if auto-summarize is enabled
        if config.auto_summarize && turns_to_remove > 0 {
            let summary = self.generate_context_summary(&session.history, turns_to_remove).await?;
            if !summary.is_empty() {
                let summary_turn = ConversationTurn {
                    turn_index: 0, // Special index for summary
                    role: TurnRole::System,
                    content: TurnContent::System(format!("Previous conversation summary: {}", summary)),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_secs(),
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("type".to_string(), "context_summary".to_string());
                        metadata.insert("summarized_turns".to_string(), turns_to_remove.to_string());
                        metadata
                    },
                };
                session.history.push_front(summary_turn);
            }
        }
        
        info!("Sliding window applied, removed {} turns", turns_to_remove);
        Ok(())
    }
    
    /// Estimate token count for context
    pub async fn estimate_token_count(&self, context: &ConversationContext) -> usize {
        let mut total_tokens = 0;
        
        // System prompt tokens
        if let Some(system_prompt) = &context.system_prompt {
            total_tokens += self.token_estimator.estimate_tokens(system_prompt);
        }
        
        // Message tokens
        for message in &context.messages {
            total_tokens += self.token_estimator.estimate_message_tokens(message);
        }
        
        total_tokens
    }
    
    // Private helper methods
    
    async fn convert_turns_to_messages(&self, turns: Vec<ConversationTurn>) -> Result<Vec<ContextMessage>> {
        let mut messages = Vec::new();
        
        for turn in turns {
            let message = match turn.content {
                TurnContent::Input(input) => {
                    ContextMessage {
                        role: MessageRole::User,
                        content: self.format_multimodal_input(input),
                        timestamp: Some(turn.timestamp),
                        metadata: turn.metadata,
                    }
                }
                TurnContent::Response(response) => {
                    ContextMessage {
                        role: MessageRole::Assistant,
                        content: response.text,
                        timestamp: Some(turn.timestamp),
                        metadata: turn.metadata,
                    }
                }
                TurnContent::System(text) => {
                    ContextMessage {
                        role: MessageRole::System,
                        content: text,
                        timestamp: Some(turn.timestamp),
                        metadata: turn.metadata,
                    }
                }
            };
            messages.push(message);
        }
        
        Ok(messages)
    }
    
    fn format_multimodal_input(&self, input: MultimodalInput) -> String {
        match input {
            MultimodalInput::Text(text) => text,
            MultimodalInput::TextWithImage { text, .. } => {
                format!("{} [Image attached]", text)
            }
            MultimodalInput::TextWithAudio { text, .. } => {
                format!("{} [Audio attached]", text)
            }
            MultimodalInput::TextWithDocument { text, .. } => {
                format!("{} [Document attached]", text)
            }
            MultimodalInput::Combined { text, images, audio, documents } => {
                let mut content = text.unwrap_or_default();
                if !images.is_empty() {
                    content.push_str(&format!(" [Attachments: {} images", images.len()));
                }
                if audio.is_some() {
                    content.push_str(" + audio");
                }
                if !documents.is_empty() {
                    content.push_str(&format!(" + {} documents", documents.len()));
                }
                if !images.is_empty() || audio.is_some() || !documents.is_empty() {
                    content.push(']');
                }
                content
            }
            // Handle new Omni input types
            MultimodalInput::TextWithVideo { text, .. } => {
                format!("{} [Video attached]", text)
            }
            MultimodalInput::RealTimeAudio { .. } => {
                "[Real-time audio stream]".to_string()
            }
            MultimodalInput::MultimodalConversation { text, images, audio, video, .. } => {
                let mut content = text.unwrap_or_default();
                let mut attachments = Vec::new();
                
                if !images.is_empty() {
                    attachments.push(format!("{} images", images.len()));
                }
                if audio.is_some() {
                    attachments.push("audio".to_string());
                }
                if video.is_some() {
                    attachments.push("video".to_string());
                }
                
                if !attachments.is_empty() {
                    content.push_str(&format!(" [Attachments: {}]", attachments.join(" + ")));
                }
                content
            }
        }
    }
    
    fn format_memory_context(&self, memory: &MemoryContext) -> String {
        let mut context_parts = Vec::new();
        
        if !memory.relevant_history.is_empty() {
            context_parts.push("Previous conversation context:".to_string());
            for item in &memory.relevant_history {
                context_parts.push(format!("- {}", item.content));
            }
        }
        
        if !memory.facts.is_empty() {
            context_parts.push("Relevant facts:".to_string());
            for fact in &memory.facts {
                context_parts.push(format!("- {}", fact));
            }
        }
        
        if !memory.preferences.is_empty() {
            context_parts.push("User preferences:".to_string());
            for (key, value) in &memory.preferences {
                context_parts.push(format!("- {}: {}", key, value));
            }
        }
        
        context_parts.join("\n")
    }
    
    async fn apply_token_budget(&self, context: &mut ConversationContext, model_name: Option<&str>) -> Result<()> {
        let estimated_tokens = self.estimate_token_count(context).await;
        
        if estimated_tokens > self.max_context_length {
            warn!("Context too long ({} tokens), applying token budget", estimated_tokens);
            
            // Remove oldest messages until we're within budget
            while !context.messages.is_empty() && self.estimate_token_count(context).await > self.max_context_length {
                context.messages.remove(0);
            }
            
            // If still too long, truncate system prompt
            if self.estimate_token_count(context).await > self.max_context_length {
                if let Some(system_prompt) = &context.system_prompt {
                    let truncated = self.token_estimator.truncate_to_tokens(
                        system_prompt,
                        self.max_context_length / 4 // Reserve 25% for system prompt
                    );
                    context.system_prompt = Some(truncated);
                }
            }
        }
        
        debug!("Token budget applied, final context: {} estimated tokens", 
               self.estimate_token_count(context).await);
        Ok(())
    }
    
    fn apply_persona_parameters(&self, context: &mut ConversationContext, persona: &PersonaConfig) {
        if let Some(temp) = persona.temperature {
            context.temperature = temp;
        }
        if let Some(top_p) = persona.top_p {
            context.top_p = top_p;
        }
        if let Some(max_tokens) = persona.max_tokens {
            context.max_tokens = max_tokens;
        }
    }
    
    async fn generate_context_summary(&self, _remaining_history: &VecDeque<ConversationTurn>, _removed_count: usize) -> Result<String> {
        // In a full implementation, this would use the AI model to generate a summary
        // For now, return a simple placeholder
        Ok("Earlier conversation covered various topics and interactions.".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    pub messages: Vec<ContextMessage>,
    pub system_prompt: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub streaming: bool,
    pub tools: Vec<Tool>,
    pub memory_context: Option<MemoryContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: Option<u64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Token estimation for context management
pub struct TokenEstimator {
    average_tokens_per_char: f32,
}

impl TokenEstimator {
    pub fn new() -> Self {
        Self {
            average_tokens_per_char: 0.25, // Rough estimate: 4 characters per token
        }
    }
    
    pub fn estimate_tokens(&self, text: &str) -> usize {
        (text.len() as f32 * self.average_tokens_per_char).ceil() as usize
    }
    
    pub fn estimate_message_tokens(&self, message: &ContextMessage) -> usize {
        let content_tokens = self.estimate_tokens(&message.content);
        // Add overhead for role and formatting
        content_tokens + 10
    }
    
    pub fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> String {
        let max_chars = (max_tokens as f32 / self.average_tokens_per_char) as usize;
        
        if text.len() <= max_chars {
            text.to_string()
        } else {
            // Try to truncate at word boundary
            let truncated = &text[..max_chars];
            if let Some(last_space) = truncated.rfind(' ') {
                format!("{}...", &truncated[..last_space])
            } else {
                format!("{}...", truncated)
            }
        }
    }
}