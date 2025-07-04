use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug};

pub mod context_manager;
pub mod memory_store;
pub mod prompt_templates;
pub mod conversation_flow;
pub mod persona_manager;
// pub mod enhanced_manager;  // Temporarily disabled due to UnifiedMultimodalInterface dependency
pub mod intelligent_orchestrator;

pub use context_manager::*;
pub use memory_store::*;
pub use prompt_templates::*;
pub use conversation_flow::*;
pub use persona_manager::*;
// pub use enhanced_manager::*;  // Temporarily disabled
pub use intelligent_orchestrator::*;

use crate::inference::{MultimodalInput, ModelResponse};

/// Advanced conversation management with context, memory, and flow control
pub struct ConversationManager {
    context_manager: Arc<ContextManager>,
    memory_store: Arc<tokio::sync::Mutex<MemoryStore>>,
    prompt_templates: Arc<PromptTemplateManager>,
    conversation_flow: Arc<ConversationFlow>,
    persona_manager: Arc<PersonaManager>,
    active_sessions: Arc<RwLock<HashMap<String, ConversationSession>>>,
    default_config: ConversationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    pub max_context_length: usize,
    pub memory_enabled: bool,
    pub sliding_window_size: usize,
    pub auto_summarize: bool,
    pub preserve_system_messages: bool,
    pub enable_memory_search: bool,
    pub conversation_timeout_minutes: u64,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_context_length: 32000,
            memory_enabled: true,
            sliding_window_size: 20,
            auto_summarize: true,
            preserve_system_messages: true,
            enable_memory_search: true,
            conversation_timeout_minutes: 60,
        }
    }
}

impl ConversationManager {
    pub fn new(config: Option<ConversationConfig>) -> Result<Self> {
        info!("Initializing conversation manager");
        
        let config = config.unwrap_or_default();
        
        Ok(Self {
            context_manager: Arc::new(ContextManager::new(config.max_context_length)?),
            memory_store: Arc::new(tokio::sync::Mutex::new(MemoryStore::new()?)),
            prompt_templates: Arc::new(PromptTemplateManager::new()?),
            conversation_flow: Arc::new(ConversationFlow::new()?),
            persona_manager: Arc::new(PersonaManager::new()?),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            default_config: config,
        })
    }
    
    /// Start a new conversation session
    pub async fn start_session(&self, session_id: String, persona: Option<String>) -> Result<String> {
        info!("Starting conversation session: {}", session_id);
        
        let persona_config = if let Some(persona_name) = persona {
            self.persona_manager.get_persona(&persona_name).await?
        } else {
            self.persona_manager.get_default_persona().await?
        };
        
        let session = ConversationSession::new(
            session_id.clone(),
            persona_config,
            self.default_config.clone(),
        )?;
        
        self.active_sessions.write().insert(session_id.clone(), session);
        
        Ok(session_id)
    }
    
    /// Process a conversation turn
    pub async fn process_turn(
        &self,
        session_id: &str,
        input: MultimodalInput,
        model_name: Option<String>,
    ) -> Result<ConversationResponse> {
        debug!("Processing conversation turn for session: {}", session_id);
        
        // Get or create session
        let mut sessions = self.active_sessions.write();
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        // Add user input to conversation history
        session.add_user_turn(input.clone()).await?;
        
        // Retrieve relevant memory context
        let memory_context = if self.default_config.memory_enabled {
            self.memory_store.lock().await.search_relevant_context(session_id, &input).await?
        } else {
            None
        };
        
        // Prepare conversation context
        let conversation_context = self.context_manager.prepare_context(
            session,
            memory_context.as_ref(),
            model_name.as_deref(),
        ).await?;
        
        // Apply conversation flow logic
        let flow_result = self.conversation_flow.process_turn(
            session,
            &input,
            &conversation_context,
        ).await?;
        
        // Update session state
        session.update_flow_state(flow_result.new_state.clone()).await?;
        
        Ok(ConversationResponse {
            session_id: session_id.to_string(),
            conversation_context,
            flow_result,
            memory_context,
        })
    }
    
    /// Add assistant response to conversation
    pub async fn add_assistant_response(
        &self,
        session_id: &str,
        response: ModelResponse,
    ) -> Result<()> {
        debug!("Adding assistant response to session: {}", session_id);
        
        let mut sessions = self.active_sessions.write();
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        // Add assistant response to session
        session.add_assistant_turn(response.clone()).await?;
        
        // Store in long-term memory if enabled
        if self.default_config.memory_enabled {
            self.memory_store.lock().await.store_conversation_turn(
                session_id,
                &session.get_recent_context(),
                &response,
            ).await?;
        }
        
        // Apply sliding window if context is getting too long
        if session.needs_context_management(&self.default_config) {
            self.context_manager.apply_sliding_window(session, &self.default_config).await?;
        }
        
        Ok(())
    }
    
    /// Get conversation history for a session
    pub async fn get_conversation_history(&self, session_id: &str) -> Result<Vec<ConversationTurn>> {
        let sessions = self.active_sessions.read();
        let session = sessions.get(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        Ok(session.get_history())
    }
    
    /// Clear conversation history but preserve session
    pub async fn clear_history(&self, session_id: &str) -> Result<()> {
        info!("Clearing history for session: {}", session_id);
        
        let mut sessions = self.active_sessions.write();
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        session.clear_history().await?;
        
        Ok(())
    }
    
    /// End a conversation session
    pub async fn end_session(&self, session_id: &str) -> Result<()> {
        info!("Ending conversation session: {}", session_id);
        
        // Store final session state in memory if enabled
        if self.default_config.memory_enabled {
            if let Some(session) = self.active_sessions.read().get(session_id) {
                self.memory_store.lock().await.store_session_summary(session).await?;
            }
        }
        
        // Remove from active sessions
        self.active_sessions.write().remove(session_id);
        
        Ok(())
    }
    
    /// Get active session count
    pub async fn get_active_session_count(&self) -> usize {
        self.active_sessions.read().len()
    }
    
    /// List all active sessions
    pub async fn list_active_sessions(&self) -> Vec<String> {
        self.active_sessions.read().keys().cloned().collect()
    }
    
    /// Update conversation configuration
    pub async fn update_config(&mut self, new_config: ConversationConfig) -> Result<()> {
        info!("Updating conversation configuration");
        self.default_config = new_config;
        Ok(())
    }
    
    /// Get conversation statistics
    pub async fn get_stats(&self) -> ConversationStats {
        let sessions = self.active_sessions.read();
        let total_turns: usize = sessions.values()
            .map(|session| session.get_turn_count())
            .sum();
        
        ConversationStats {
            active_sessions: sessions.len(),
            total_conversation_turns: total_turns,
            memory_entries: self.memory_store.lock().await.get_entry_count().await,
            average_turns_per_session: if sessions.is_empty() {
                0.0
            } else {
                total_turns as f32 / sessions.len() as f32
            },
        }
    }
    
    /// Search conversation history across all sessions
    pub async fn search_conversations(&self, query: &str, limit: usize) -> Result<Vec<ConversationSearchResult>> {
        self.memory_store.lock().await.search_conversations(query, limit).await
    }
    
    /// Export conversation history
    pub async fn export_session(&self, session_id: &str) -> Result<ConversationExport> {
        let sessions = self.active_sessions.read();
        let session = sessions.get(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        Ok(ConversationExport {
            session_id: session_id.to_string(),
            persona: session.persona.clone(),
            created_at: session.created_at,
            turns: session.get_history(),
            metadata: session.get_metadata(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ConversationResponse {
    pub session_id: String,
    pub conversation_context: ConversationContext,
    pub flow_result: FlowResult,
    pub memory_context: Option<MemoryContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStats {
    pub active_sessions: usize,
    pub total_conversation_turns: usize,
    pub memory_entries: usize,
    pub average_turns_per_session: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSearchResult {
    pub session_id: String,
    pub turn_index: usize,
    pub content: String,
    pub timestamp: u64,
    pub relevance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationExport {
    pub session_id: String,
    pub persona: PersonaConfig,
    pub created_at: u64,
    pub turns: Vec<ConversationTurn>,
    pub metadata: HashMap<String, String>,
}

/// Individual conversation session
#[derive(Debug, Clone)]
pub struct ConversationSession {
    pub id: String,
    pub persona: PersonaConfig,
    pub config: ConversationConfig,
    pub created_at: u64,
    pub last_activity: u64,
    pub flow_state: FlowState,
    history: VecDeque<ConversationTurn>,
    metadata: HashMap<String, String>,
    turn_count: usize,
}

impl ConversationSession {
    pub fn new(id: String, persona: PersonaConfig, config: ConversationConfig) -> Result<Self> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        
        Ok(Self {
            id,
            persona,
            config,
            created_at: now,
            last_activity: now,
            flow_state: FlowState::default(),
            history: VecDeque::new(),
            metadata: HashMap::new(),
            turn_count: 0,
        })
    }
    
    pub async fn add_user_turn(&mut self, input: MultimodalInput) -> Result<()> {
        let turn = ConversationTurn {
            turn_index: self.turn_count,
            role: TurnRole::User,
            content: TurnContent::Input(input),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        self.history.push_back(turn);
        self.turn_count += 1;
        self.update_activity_timestamp();
        
        Ok(())
    }
    
    pub async fn add_assistant_turn(&mut self, response: ModelResponse) -> Result<()> {
        let turn = ConversationTurn {
            turn_index: self.turn_count,
            role: TurnRole::Assistant,
            content: TurnContent::Response(response),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            metadata: HashMap::new(),
        };
        
        self.history.push_back(turn);
        self.turn_count += 1;
        self.update_activity_timestamp();
        
        Ok(())
    }
    
    pub fn get_history(&self) -> Vec<ConversationTurn> {
        self.history.iter().cloned().collect()
    }
    
    pub fn get_recent_context(&self) -> Vec<ConversationTurn> {
        self.history.iter()
            .rev()
            .take(self.config.sliding_window_size)
            .rev()
            .cloned()
            .collect()
    }
    
    pub fn get_turn_count(&self) -> usize {
        self.turn_count
    }
    
    pub fn get_metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }
    
    pub async fn clear_history(&mut self) -> Result<()> {
        self.history.clear();
        self.turn_count = 0;
        self.update_activity_timestamp();
        Ok(())
    }
    
    pub fn needs_context_management(&self, config: &ConversationConfig) -> bool {
        self.history.len() > config.sliding_window_size
    }
    
    pub async fn update_flow_state(&mut self, new_state: FlowState) -> Result<()> {
        self.flow_state = new_state;
        self.update_activity_timestamp();
        Ok(())
    }
    
    fn update_activity_timestamp(&mut self) {
        self.last_activity = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub turn_index: usize,
    pub role: TurnRole,
    pub content: TurnContent,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurnRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurnContent {
    Input(MultimodalInput),
    Response(ModelResponse),
    System(String),
}

impl ConversationTurn {
    /// Check if this turn contains multimodal content
    pub fn has_multimodal_content(&self) -> bool {
        match &self.content {
            TurnContent::Input(input) => match input {
                MultimodalInput::Text(_) => false,
                _ => true, // All other variants contain multimodal content
            },
            TurnContent::Response(response) => response.audio.is_some(),
            TurnContent::System(_) => false,
        }
    }
}