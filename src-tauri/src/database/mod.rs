use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use tauri::Manager;

/// Project data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "createdAt")]
    pub created_at: u64,
    #[serde(rename = "updatedAt")]
    pub updated_at: u64,
    #[serde(rename = "documentCount")]
    pub document_count: u32,
    pub tags: Vec<String>,
    #[serde(rename = "isStarred")]
    pub is_starred: bool,
}

/// Document data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    #[serde(rename = "projectId")]
    pub project_id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub size: u64,
    pub path: String,
    #[serde(rename = "uploadedAt")]
    pub uploaded_at: u64,
    pub tags: Vec<String>,
    pub content: Option<String>, // Extracted text content
    pub embeddings: Option<Vec<f32>>, // Vector embeddings for semantic search
}

/// Database manager for projects and documents
pub struct Database {
    data_dir: PathBuf,
    projects: Arc<Mutex<HashMap<String, Project>>>,
    documents: Arc<Mutex<HashMap<String, Vec<Document>>>>,
}

impl Database {
    /// Create a new database instance
    /// 
    /// Initializes a file-based database in the app data directory.
    /// Automatically loads existing data if present.
    /// 
    /// # Arguments
    /// * `app_handle` - Tauri application handle for path resolution
    /// 
    /// # Returns
    /// * `Result<Self>` - Database instance or error
    /// 
    /// # Errors
    /// - Failed to create data directory
    /// - Failed to load existing data
    pub fn new(app_handle: &tauri::AppHandle) -> Result<Self> {
        let data_dir = app_handle
            .path()
            .app_data_dir()
            .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
        
        // Ensure data directory exists
        fs::create_dir_all(&data_dir)?;
        
        let mut db = Self {
            data_dir: data_dir.clone(),
            projects: Arc::new(Mutex::new(HashMap::new())),
            documents: Arc::new(Mutex::new(HashMap::new())),
        };
        
        // Load existing data
        db.load_data()?;
        
        Ok(db)
    }
    
    /// Load data from disk
    fn load_data(&mut self) -> Result<()> {
        let projects_file = self.data_dir.join("projects.json");
        let documents_file = self.data_dir.join("documents.json");
        
        // Load projects
        if projects_file.exists() {
            let data = fs::read_to_string(&projects_file)?;
            if let Ok(projects) = serde_json::from_str::<HashMap<String, Project>>(&data) {
                *self.projects.blocking_lock() = projects;
                info!("Loaded {} projects from disk", self.projects.blocking_lock().len());
            }
        }
        
        // Load documents
        if documents_file.exists() {
            let data = fs::read_to_string(&documents_file)?;
            if let Ok(documents) = serde_json::from_str::<HashMap<String, Vec<Document>>>(&data) {
                *self.documents.blocking_lock() = documents;
                info!("Loaded documents from disk");
            }
        }
        
        Ok(())
    }
    
    /// Save data to disk
    async fn save_data(&self) -> Result<()> {
        let projects_file = self.data_dir.join("projects.json");
        let documents_file = self.data_dir.join("documents.json");
        
        // Save projects
        let projects = self.projects.lock().await;
        let projects_json = serde_json::to_string_pretty(&*projects)?;
        tokio::fs::write(&projects_file, projects_json).await?;
        
        // Save documents
        let documents = self.documents.lock().await;
        let documents_json = serde_json::to_string_pretty(&*documents)?;
        tokio::fs::write(&documents_file, documents_json).await?;
        
        Ok(())
    }
    
    /// Create a new project
    /// 
    /// Creates a project with auto-generated ID and timestamps.
    /// Persists immediately to disk.
    /// 
    /// # Arguments
    /// * `name` - Project name
    /// * `description` - Optional project description
    /// 
    /// # Returns
    /// * `Result<Project>` - Created project or error
    /// 
    /// # Example
    /// ```rust
    /// let project = db.create_project(
    ///     "My AI Project".to_string(),
    ///     Some("Research on multimodal AI".to_string())
    /// ).await?;
    /// ```
    pub async fn create_project(&self, name: String, description: Option<String>) -> Result<Project> {
        let project_id = uuid::Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        
        let project = Project {
            id: project_id.clone(),
            name,
            description: description.unwrap_or_default(),
            created_at: now,
            updated_at: now,
            document_count: 0,
            tags: vec![],
            is_starred: false,
        };
        
        // Add to storage
        self.projects.lock().await.insert(project_id.clone(), project.clone());
        
        // Initialize empty document list for this project
        self.documents.lock().await.insert(project_id, vec![]);
        
        // Save to disk
        self.save_data().await?;
        
        Ok(project)
    }
    
    /// Get all projects sorted by update time
    /// 
    /// # Returns
    /// * `Result<Vec<Project>>` - List of projects (newest first)
    pub async fn get_projects(&self) -> Result<Vec<Project>> {
        let projects = self.projects.lock().await;
        let mut project_list: Vec<Project> = projects.values().cloned().collect();
        
        // Sort by updated_at descending (newest first)
        project_list.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        
        Ok(project_list)
    }
    
    /// Delete a project
    pub async fn delete_project(&self, project_id: String) -> Result<()> {
        // Remove project
        let removed = self.projects.lock().await.remove(&project_id);
        if removed.is_none() {
            return Err(anyhow!("Project not found"));
        }
        
        // Remove associated documents
        self.documents.lock().await.remove(&project_id);
        
        // Save to disk
        self.save_data().await?;
        
        info!("Deleted project: {}", project_id);
        Ok(())
    }
    
    /// Toggle project star status
    pub async fn toggle_project_star(&self, project_id: String) -> Result<()> {
        let mut projects = self.projects.lock().await;
        
        if let Some(project) = projects.get_mut(&project_id) {
            project.is_starred = !project.is_starred;
            project.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            
            drop(projects); // Release lock before saving
            self.save_data().await?;
            
            info!("Toggled star for project: {}", project_id);
            Ok(())
        } else {
            Err(anyhow!("Project not found"))
        }
    }
    
    /// Get documents for a project
    pub async fn get_project_documents(&self, project_id: String) -> Result<Vec<Document>> {
        let documents = self.documents.lock().await;
        
        if let Some(docs) = documents.get(&project_id) {
            Ok(docs.clone())
        } else {
            Ok(vec![])
        }
    }
    
    /// Add a document to a project
    /// 
    /// Adds document to project and updates project metadata.
    /// Persists changes immediately.
    /// 
    /// # Arguments
    /// * `project_id` - ID of the parent project
    /// * `document` - Document to add
    /// 
    /// # Returns
    /// * `Result<Document>` - Added document or error
    /// 
    /// # Errors
    /// - Project not found
    /// - Save operation failed
    pub async fn add_document(&self, project_id: String, document: Document) -> Result<Document> {
        // Update document count in project
        let mut projects = self.projects.lock().await;
        if let Some(project) = projects.get_mut(&project_id) {
            project.document_count += 1;
            project.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
        } else {
            return Err(anyhow!("Project not found"));
        }
        drop(projects);
        
        // Add document
        let mut documents = self.documents.lock().await;
        documents.entry(project_id.clone())
            .or_insert_with(Vec::new)
            .push(document.clone());
        drop(documents);
        
        // Save to disk
        self.save_data().await?;
        
        info!("Added document {} to project {}", document.name, project_id);
        Ok(document)
    }
    
    /// Delete a document from a project
    pub async fn delete_document(&self, project_id: String, document_id: String) -> Result<()> {
        // Remove document
        let mut documents = self.documents.lock().await;
        if let Some(docs) = documents.get_mut(&project_id) {
            let original_len = docs.len();
            docs.retain(|d| d.id != document_id);
            
            if docs.len() == original_len {
                return Err(anyhow!("Document not found"));
            }
        } else {
            return Err(anyhow!("Project not found"));
        }
        drop(documents);
        
        // Update document count in project
        let mut projects = self.projects.lock().await;
        if let Some(project) = projects.get_mut(&project_id) {
            if project.document_count > 0 {
                project.document_count -= 1;
            }
            project.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
        }
        drop(projects);
        
        // Save to disk
        self.save_data().await?;
        
        info!("Deleted document {} from project {}", document_id, project_id);
        Ok(())
    }
    
    /// Update document content and embeddings
    pub async fn update_document_content(
        &self, 
        project_id: String, 
        document_id: String,
        content: Option<String>,
        embeddings: Option<Vec<f32>>
    ) -> Result<()> {
        let mut documents = self.documents.lock().await;
        
        if let Some(docs) = documents.get_mut(&project_id) {
            if let Some(doc) = docs.iter_mut().find(|d| d.id == document_id) {
                doc.content = content;
                doc.embeddings = embeddings;
                
                drop(documents); // Release lock before saving
                self.save_data().await?;
                
                info!("Updated content for document: {}", document_id);
                Ok(())
            } else {
                Err(anyhow!("Document not found"))
            }
        } else {
            Err(anyhow!("Project not found"))
        }
    }
}