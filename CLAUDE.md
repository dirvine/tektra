# Claude Code Development Guidelines for Tektra

This file contains the development standards and practices that Claude should follow when working on the Tektra AI Assistant codebase.

[... existing content remains unchanged ...]

## Development Workflow

### 1. Specification-Driven Development
- **Always reference** `specifications.md` for feature requirements
- **Maintain compatibility** with the original vision while modernizing implementation
- **Request user acceptance** before major architectural changes

### 2. Rust Development Standards

#### AI Module Guidelines
```rust
// Use proper error handling - NO unwrap() or expect()
pub async fn load_model(&mut self, model_name: &str) -> Result<()> {
    let tokenizer = match Tokenizer::from_file(&path) {
        Ok(tok) => tok,
        Err(e) => return Err(anyhow::anyhow!("Failed to load tokenizer: {}", e)),
    };
    // Implementation...
}

// Emit progress events for long operations
let _ = self.app_handle.emit("model-loading-progress", json!({
    "progress": 85,
    "status": "Loading tokenizer...",
    "model_name": model_name
}));
```

- **remember and use UV at all times**
- **Never use placeholders or dummy code, no mocks**

[... rest of the existing content remains unchanged ...]