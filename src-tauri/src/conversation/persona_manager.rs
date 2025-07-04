use super::*;
use anyhow::Result;
use std::collections::HashMap;

/// Persona management for different assistant personalities and behaviors
pub struct PersonaManager {
    personas: HashMap<String, PersonaConfig>,
    default_persona: String,
}

impl PersonaManager {
    pub fn new() -> Result<Self> {
        let mut personas = HashMap::new();
        
        // Default helpful assistant
        personas.insert("default".to_string(), PersonaConfig {
            name: "Helpful Assistant".to_string(),
            system_prompt: Some("You are a helpful, harmless, and honest AI assistant. You aim to be informative, accurate, and engaging while maintaining a friendly and professional tone.".to_string()),
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(2048),
            personality_traits: vec![
                "helpful".to_string(),
                "informative".to_string(),
                "friendly".to_string(),
                "professional".to_string(),
            ],
            response_style: ResponseStyle::Balanced,
            specializations: vec!["general knowledge".to_string(), "problem solving".to_string()],
        });
        
        // Technical expert
        personas.insert("technical".to_string(), PersonaConfig {
            name: "Technical Expert".to_string(),
            system_prompt: Some("You are a technical expert AI assistant with deep knowledge in programming, engineering, and technology. You provide detailed, accurate technical explanations and solutions. You use precise terminology and can dive into implementation details when needed.".to_string()),
            temperature: Some(0.3),
            top_p: Some(0.8),
            max_tokens: Some(3072),
            personality_traits: vec![
                "precise".to_string(),
                "analytical".to_string(),
                "thorough".to_string(),
                "expert".to_string(),
            ],
            response_style: ResponseStyle::Technical,
            specializations: vec![
                "programming".to_string(),
                "software engineering".to_string(),
                "system architecture".to_string(),
                "debugging".to_string(),
            ],
        });
        
        // Creative assistant
        personas.insert("creative".to_string(), PersonaConfig {
            name: "Creative Assistant".to_string(),
            system_prompt: Some("You are a creative and imaginative AI assistant. You excel at brainstorming, storytelling, creative writing, and artistic endeavors. You think outside the box and offer innovative solutions and ideas.".to_string()),
            temperature: Some(0.9),
            top_p: Some(0.95),
            max_tokens: Some(2048),
            personality_traits: vec![
                "creative".to_string(),
                "imaginative".to_string(),
                "inspiring".to_string(),
                "artistic".to_string(),
            ],
            response_style: ResponseStyle::Creative,
            specializations: vec![
                "creative writing".to_string(),
                "brainstorming".to_string(),
                "storytelling".to_string(),
                "artistic concepts".to_string(),
            ],
        });
        
        // Educational tutor
        personas.insert("tutor".to_string(), PersonaConfig {
            name: "Educational Tutor".to_string(),
            system_prompt: Some("You are an educational tutor AI assistant. You excel at explaining complex concepts in simple terms, creating learning materials, and helping students understand difficult topics. You're patient, encouraging, and adapt your explanations to the student's level.".to_string()),
            temperature: Some(0.5),
            top_p: Some(0.85),
            max_tokens: Some(2048),
            personality_traits: vec![
                "patient".to_string(),
                "encouraging".to_string(),
                "clear".to_string(),
                "adaptive".to_string(),
            ],
            response_style: ResponseStyle::Educational,
            specializations: vec![
                "teaching".to_string(),
                "explanation".to_string(),
                "learning theory".to_string(),
                "curriculum design".to_string(),
            ],
        });
        
        Ok(Self {
            personas,
            default_persona: "default".to_string(),
        })
    }
    
    pub async fn get_persona(&self, name: &str) -> Result<PersonaConfig> {
        self.personas.get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Persona not found: {}", name))
    }
    
    pub async fn get_default_persona(&self) -> Result<PersonaConfig> {
        self.get_persona(&self.default_persona).await
    }
    
    pub async fn list_personas(&self) -> Vec<String> {
        self.personas.keys().cloned().collect()
    }
    
    pub async fn add_custom_persona(&mut self, persona: PersonaConfig) -> Result<()> {
        let name = persona.name.to_lowercase().replace(' ', "_");
        self.personas.insert(name, persona);
        Ok(())
    }
    
    pub async fn remove_persona(&mut self, name: &str) -> Result<()> {
        if name == &self.default_persona {
            return Err(anyhow::anyhow!("Cannot remove default persona"));
        }
        
        self.personas.remove(name)
            .ok_or_else(|| anyhow::anyhow!("Persona not found: {}", name))?;
        
        Ok(())
    }
    
    pub async fn set_default_persona(&mut self, name: &str) -> Result<()> {
        if !self.personas.contains_key(name) {
            return Err(anyhow::anyhow!("Persona not found: {}", name));
        }
        
        self.default_persona = name.to_string();
        Ok(())
    }
    
    pub async fn get_persona_suggestions(&self, context: &str) -> Vec<PersonaSuggestion> {
        let context_lower = context.to_lowercase();
        let mut suggestions = Vec::new();
        
        for (id, persona) in &self.personas {
            let relevance = self.calculate_persona_relevance(&context_lower, persona);
            if relevance > 0.3 {
                suggestions.push(PersonaSuggestion {
                    persona_id: id.clone(),
                    persona_name: persona.name.clone(),
                    relevance_score: relevance,
                    reason: self.generate_suggestion_reason(&context_lower, persona),
                });
            }
        }
        
        suggestions.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        suggestions
    }
    
    fn calculate_persona_relevance(&self, context: &str, persona: &PersonaConfig) -> f32 {
        let mut relevance: f32 = 0.0;
        
        // Check specializations
        for spec in &persona.specializations {
            if context.contains(&spec.to_lowercase()) {
                relevance += 0.5;
            }
        }
        
        // Check personality traits
        for trait_name in &persona.personality_traits {
            if context.contains(&trait_name.to_lowercase()) {
                relevance += 0.2;
            }
        }
        
        // Check response style keywords
        match persona.response_style {
            ResponseStyle::Technical => {
                if context.contains("code") || context.contains("technical") || context.contains("programming") {
                    relevance += 0.3;
                }
            }
            ResponseStyle::Creative => {
                if context.contains("creative") || context.contains("story") || context.contains("art") {
                    relevance += 0.3;
                }
            }
            ResponseStyle::Educational => {
                if context.contains("learn") || context.contains("teach") || context.contains("explain") {
                    relevance += 0.3;
                }
            }
            _ => {}
        }
        
        relevance.min(1.0)
    }
    
    fn generate_suggestion_reason(&self, context: &str, persona: &PersonaConfig) -> String {
        let mut reasons = Vec::new();
        
        for spec in &persona.specializations {
            if context.contains(&spec.to_lowercase()) {
                reasons.push(format!("Expert in {}", spec));
            }
        }
        
        if reasons.is_empty() {
            format!("Good fit for {} tasks", persona.response_style.to_string().to_lowercase())
        } else {
            reasons.join(", ")
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    pub name: String,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub personality_traits: Vec<String>,
    pub response_style: ResponseStyle,
    pub specializations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStyle {
    Concise,
    Detailed,
    Conversational,
    Professional,
    Technical,
    Creative,
    Educational,
    Balanced,
}

impl ToString for ResponseStyle {
    fn to_string(&self) -> String {
        match self {
            ResponseStyle::Concise => "Concise".to_string(),
            ResponseStyle::Detailed => "Detailed".to_string(),
            ResponseStyle::Conversational => "Conversational".to_string(),
            ResponseStyle::Professional => "Professional".to_string(),
            ResponseStyle::Technical => "Technical".to_string(),
            ResponseStyle::Creative => "Creative".to_string(),
            ResponseStyle::Educational => "Educational".to_string(),
            ResponseStyle::Balanced => "Balanced".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSuggestion {
    pub persona_id: String,
    pub persona_name: String,
    pub relevance_score: f32,
    pub reason: String,
}