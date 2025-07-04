use super::*;
use anyhow::Result;
use std::collections::HashMap;
use tracing::{info, debug};

/// Long-term memory storage for conversation context and user preferences
pub struct MemoryStore {
    conversations: HashMap<String, Vec<MemoryEntry>>,
    facts: HashMap<String, Vec<String>>,
    preferences: HashMap<String, HashMap<String, String>>,
    entry_count: usize,
}

impl MemoryStore {
    pub fn new() -> Result<Self> {
        info!("Initializing memory store");
        
        Ok(Self {
            conversations: HashMap::new(),
            facts: HashMap::new(),
            preferences: HashMap::new(),
            entry_count: 0,
        })
    }
    
    /// Store a conversation turn in long-term memory
    pub async fn store_conversation_turn(
        &mut self,
        session_id: &str,
        context: &[ConversationTurn],
        response: &ModelResponse,
    ) -> Result<()> {
        debug!("Storing conversation turn in memory for session: {}", session_id);
        
        let entry = MemoryEntry {
            id: format!("{}_{}", session_id, self.entry_count),
            session_id: session_id.to_string(),
            content: response.text.clone(),
            context: self.summarize_context(context),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            relevance_score: 1.0, // Initial relevance
            tags: self.extract_tags(&response.text),
        };
        
        self.conversations
            .entry(session_id.to_string())
            .or_default()
            .push(entry);
        
        self.entry_count += 1;
        Ok(())
    }
    
    /// Search for relevant conversation context
    pub async fn search_relevant_context(
        &self,
        session_id: &str,
        input: &MultimodalInput,
    ) -> Result<Option<MemoryContext>> {
        debug!("Searching for relevant context for session: {}", session_id);
        
        let query = self.extract_query_from_input(input);
        if query.is_empty() {
            return Ok(None);
        }
        
        let mut relevant_history = Vec::new();
        let mut all_facts = Vec::new();
        let session_preferences = self.preferences.get(session_id).cloned().unwrap_or_default();
        
        // Search conversation history
        for (sid, entries) in &self.conversations {
            for entry in entries {
                let relevance = self.calculate_relevance(&query, entry);
                if relevance > 0.3 {
                    relevant_history.push(MemoryItem {
                        content: entry.content.clone(),
                        context: entry.context.clone(),
                        relevance_score: relevance,
                        timestamp: entry.timestamp,
                    });
                }
            }
        }
        
        // Search facts
        for (topic, topic_facts) in &self.facts {
            if query.to_lowercase().contains(&topic.to_lowercase()) {
                all_facts.extend_from_slice(topic_facts);
            }
        }
        
        // Sort by relevance and limit results
        relevant_history.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        relevant_history.truncate(5);
        
        if relevant_history.is_empty() && all_facts.is_empty() && session_preferences.is_empty() {
            Ok(None)
        } else {
            Ok(Some(MemoryContext {
                relevant_history,
                facts: all_facts,
                preferences: session_preferences,
            }))
        }
    }
    
    /// Store session summary for long-term retention
    pub async fn store_session_summary(&mut self, session: &ConversationSession) -> Result<()> {
        info!("Storing session summary for: {}", session.id);
        
        let summary = self.generate_session_summary(session).await?;
        
        let entry = MemoryEntry {
            id: format!("{}_summary", session.id),
            session_id: session.id.clone(),
            content: summary,
            context: "Session summary".to_string(),
            timestamp: session.last_activity,
            relevance_score: 0.8,
            tags: vec!["summary".to_string()],
        };
        
        self.conversations
            .entry(session.id.clone())
            .or_default()
            .push(entry);
        
        Ok(())
    }
    
    /// Add a fact to memory
    pub async fn add_fact(&mut self, topic: &str, fact: &str) -> Result<()> {
        info!("Adding fact to memory - topic: {}", topic);
        
        self.facts
            .entry(topic.to_string())
            .or_default()
            .push(fact.to_string());
        
        Ok(())
    }
    
    /// Update user preference
    pub async fn set_preference(&mut self, session_id: &str, key: &str, value: &str) -> Result<()> {
        info!("Setting preference for {}: {} = {}", session_id, key, value);
        
        self.preferences
            .entry(session_id.to_string())
            .or_default()
            .insert(key.to_string(), value.to_string());
        
        Ok(())
    }
    
    /// Search conversations across all sessions
    pub async fn search_conversations(&self, query: &str, limit: usize) -> Result<Vec<ConversationSearchResult>> {
        debug!("Searching conversations with query: {}", query);
        
        let mut results = Vec::new();
        
        for (session_id, entries) in &self.conversations {
            for (index, entry) in entries.iter().enumerate() {
                let relevance = self.calculate_relevance(query, entry);
                if relevance > 0.2 {
                    results.push(ConversationSearchResult {
                        session_id: session_id.clone(),
                        turn_index: index,
                        content: entry.content.clone(),
                        timestamp: entry.timestamp,
                        relevance_score: relevance,
                    });
                }
            }
        }
        
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        results.truncate(limit);
        
        Ok(results)
    }
    
    /// Get memory entry count
    pub async fn get_entry_count(&self) -> usize {
        self.entry_count
    }
    
    /// Clear all memory for a session
    pub async fn clear_session_memory(&mut self, session_id: &str) -> Result<()> {
        info!("Clearing memory for session: {}", session_id);
        
        self.conversations.remove(session_id);
        self.preferences.remove(session_id);
        
        Ok(())
    }
    
    /// Export memory data
    pub async fn export_memory(&self, session_id: Option<&str>) -> Result<MemoryExport> {
        let conversations = if let Some(sid) = session_id {
            self.conversations.get(sid).cloned().unwrap_or_default()
        } else {
            self.conversations.values().flatten().cloned().collect()
        };
        
        let preferences = if let Some(sid) = session_id {
            self.preferences.get(sid).cloned().unwrap_or_default()
        } else {
            HashMap::new() // Don't export all preferences for privacy
        };
        
        Ok(MemoryExport {
            conversations,
            facts: self.facts.clone(),
            preferences,
            export_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        })
    }
    
    // Private helper methods
    
    fn extract_query_from_input(&self, input: &MultimodalInput) -> String {
        match input {
            MultimodalInput::Text(text) => text.clone(),
            MultimodalInput::TextWithImage { text, .. } => text.clone(),
            MultimodalInput::TextWithAudio { text, .. } => text.clone(),
            MultimodalInput::TextWithDocument { text, .. } => text.clone(),
            MultimodalInput::Combined { text, .. } => text.clone().unwrap_or_default(),
            // Handle new Omni input types
            MultimodalInput::TextWithVideo { text, .. } => text.clone(),
            MultimodalInput::RealTimeAudio { .. } => "real-time audio input".to_string(),
            MultimodalInput::MultimodalConversation { text, .. } => text.clone().unwrap_or_default(),
        }
    }
    
    fn calculate_relevance(&self, query: &str, entry: &MemoryEntry) -> f32 {
        let query_lower = query.to_lowercase();
        let content_lower = entry.content.to_lowercase();
        
        let mut relevance = 0.0;
        
        // Exact phrase match
        if content_lower.contains(&query_lower) {
            relevance += 0.8;
        }
        
        // Word overlap
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let content_words: Vec<&str> = content_lower.split_whitespace().collect();
        
        let mut common_words = 0;
        for word in &query_words {
            if word.len() > 3 && content_words.contains(word) {
                common_words += 1;
            }
        }
        
        if !query_words.is_empty() {
            relevance += (common_words as f32 / query_words.len() as f32) * 0.5;
        }
        
        // Tag matching
        for tag in &entry.tags {
            if query_lower.contains(&tag.to_lowercase()) {
                relevance += 0.3;
            }
        }
        
        // Time decay (more recent entries are more relevant)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let age_days = (now - entry.timestamp) / (24 * 3600);
        let time_factor = 1.0 / (1.0 + age_days as f32 / 30.0); // Decay over 30 days
        
        relevance * time_factor * entry.relevance_score
    }
    
    fn extract_tags(&self, text: &str) -> Vec<String> {
        let mut tags = Vec::new();
        let text_lower = text.to_lowercase();
        
        // Simple keyword-based tagging
        let keywords = [
            ("question", "question"),
            ("help", "assistance"),
            ("code", "programming"),
            ("explain", "explanation"),
            ("how", "tutorial"),
            ("what", "information"),
            ("error", "problem"),
            ("fix", "solution"),
        ];
        
        for (keyword, tag) in keywords {
            if text_lower.contains(keyword) {
                tags.push(tag.to_string());
            }
        }
        
        tags
    }
    
    fn summarize_context(&self, context: &[ConversationTurn]) -> String {
        if context.is_empty() {
            return "No context".to_string();
        }
        
        // Simple context summarization
        let mut summary_parts = Vec::new();
        
        for turn in context.iter().take(3) { // Last 3 turns
            match &turn.content {
                TurnContent::Input(input) => {
                    let text = self.extract_query_from_input(input);
                    if !text.is_empty() {
                        summary_parts.push(format!("User: {}", text.chars().take(50).collect::<String>()));
                    }
                }
                TurnContent::Response(response) => {
                    summary_parts.push(format!("Assistant: {}", response.text.chars().take(50).collect::<String>()));
                }
                TurnContent::System(text) => {
                    summary_parts.push(format!("System: {}", text.chars().take(50).collect::<String>()));
                }
            }
        }
        
        summary_parts.join(" | ")
    }
    
    async fn generate_session_summary(&self, session: &ConversationSession) -> Result<String> {
        let turns = session.get_history();
        if turns.is_empty() {
            return Ok("Empty conversation".to_string());
        }
        
        // Simple session summarization
        let mut summary = format!("Conversation with {} persona, {} turns", 
                                  session.persona.name, turns.len());
        
        // Add key topics if available
        let mut topics = Vec::new();
        for turn in &turns {
            match &turn.content {
                TurnContent::Input(input) => {
                    let text = self.extract_query_from_input(input);
                    let tags = self.extract_tags(&text);
                    topics.extend(tags);
                }
                _ => {}
            }
        }
        
        topics.sort();
        topics.dedup();
        
        if !topics.is_empty() {
            summary.push_str(&format!(", topics: {}", topics.join(", ")));
        }
        
        Ok(summary)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub session_id: String,
    pub content: String,
    pub context: String,
    pub timestamp: u64,
    pub relevance_score: f32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub relevant_history: Vec<MemoryItem>,
    pub facts: Vec<String>,
    pub preferences: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub content: String,
    pub context: String,
    pub relevance_score: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryExport {
    pub conversations: Vec<MemoryEntry>,
    pub facts: HashMap<String, Vec<String>>,
    pub preferences: HashMap<String, String>,
    pub export_timestamp: u64,
}