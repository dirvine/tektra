pub mod ai;
pub mod vector_db;
pub mod database;
pub mod types;

// Re-export commonly used types for tests
pub use crate::types::{ChatMessage, AppSettings};