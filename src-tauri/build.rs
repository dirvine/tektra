use std::fs;
use std::path::Path;

fn main() {
    // For cargo publish scenarios, create minimal frontend if dist doesn't exist
    let dist_dir = Path::new("../dist");
    if !dist_dir.exists() {
        println!("cargo:warning=Creating minimal frontend for packaging");
        create_minimal_frontend();
    }
    
    tauri_build::build()
}

fn create_minimal_frontend() {
    let dist_dir = Path::new("../dist");
    let assets_dir = dist_dir.join("assets");
    
    // Create directories
    fs::create_dir_all(&assets_dir).ok();
    
    // Create index.html with working Tektra UI
    let index_html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tektra - AI Voice Assistant</title>
    <link rel="stylesheet" href="assets/index.css">
</head>
<body>
    <div id="root">
        <div class="app">
            <div class="header">
                <h1>🎙️ Tektra</h1>
                <p>AI Voice Assistant</p>
            </div>
            <div class="chat-container">
                <div id="messages" class="messages"></div>
                <div class="input-container">
                    <input type="text" id="messageInput" placeholder="Type your message..." />
                    <button id="sendButton">Send</button>
                </div>
            </div>
            <div class="status" id="status">Ready</div>
        </div>
    </div>
    <script src="assets/index.js"></script>
</body>
</html>"#;

    // Create CSS with modern styling
    let main_css = r#"
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.app {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    width: 90%;
    max-width: 800px;
    height: 600px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 20px;
    text-align: center;
}

.header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
.header p { opacity: 0.9; }

.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 20px;
}

.messages {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 20px;
    padding: 10px;
    border: 2px solid #f0f0f0;
    border-radius: 10px;
    background: #fafafa;
}

.message {
    margin: 10px 0;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 80%;
}

.message.user {
    background: #667eea;
    color: white;
    margin-left: auto;
}

.message.assistant {
    background: #e9ecef;
    color: #333;
}

.input-container {
    display: flex;
    gap: 10px;
}

#messageInput {
    flex: 1;
    padding: 12px 16px;
    border: 2px solid #e0e0e0;
    border-radius: 25px;
    font-size: 14px;
    outline: none;
}

#messageInput:focus { border-color: #667eea; }

#sendButton {
    padding: 12px 24px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-weight: 600;
}

#sendButton:hover { opacity: 0.9; }

.status {
    padding: 10px 20px;
    background: #f8f9fa;
    border-top: 1px solid #e0e0e0;
    text-align: center;
    font-size: 12px;
    color: #666;
}

.loading { opacity: 0.7; }
"#;

    // Create JavaScript with Tauri integration
    let main_js = r#"
const { invoke } = window.__TAURI__.tauri;

const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const statusDiv = document.getElementById('status');

let isLoading = false;

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.textContent = content;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function setStatus(text) {
    statusDiv.textContent = text;
}

function setLoading(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
    sendButton.textContent = loading ? 'Sending...' : 'Send';
    document.body.className = loading ? 'loading' : '';
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    addMessage('user', message);
    messageInput.value = '';
    setLoading(true);
    setStatus('Thinking...');

    try {
        const response = await invoke('send_message', { message });
        addMessage('assistant', response);
        setStatus('Ready');
    } catch (error) {
        addMessage('assistant', `Error: ${error}`);
        setStatus('Error occurred');
    } finally {
        setLoading(false);
    }
}

sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Initialize the model on startup
async function initializeModel() {
    setStatus('Initializing AI model...');
    try {
        await invoke('initialize_model');
        setStatus('Ready');
        addMessage('assistant', 'Hello! I\'m Tektra, your AI voice assistant. How can I help you today?');
    } catch (error) {
        setStatus('Failed to initialize model');
        addMessage('assistant', 'Welcome to Tektra! Note: AI model initialization failed, but I\'m still here to help with basic functions.');
    }
}

// Start initialization when page loads
window.addEventListener('DOMContentLoaded', initializeModel);
"#;

    // Write files
    fs::write(dist_dir.join("index.html"), index_html).ok();
    fs::write(assets_dir.join("index.css"), main_css).ok();
    fs::write(assets_dir.join("index.js"), main_js).ok();
    
    println!("cargo:warning=Created minimal frontend with working Tektra UI");
}