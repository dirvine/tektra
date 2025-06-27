import { invoke } from '@tauri-apps/api/tauri';

class ProjectTektra {
    constructor() {
        this.isListening = false;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        
        this.initializeElements();
        this.attachEventListeners();
        this.initializeApp();
        this.setupSpeechRecognition();
    }

    initializeElements() {
        this.elements = {
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            voiceBtn: document.getElementById('voiceBtn'),
            clearBtn: document.getElementById('clearBtn'),
            refreshBtn: document.getElementById('refreshBtn'),
            settingsBtn: document.getElementById('settingsBtn'),
            modelName: document.getElementById('modelName'),
            modelStatus: document.getElementById('modelStatus'),
            voiceIndicator: document.getElementById('voiceIndicator'),
            voiceEnabled: document.getElementById('voiceEnabled'),
            autoSpeech: document.getElementById('autoSpeech'),
            
            // Settings modal
            settingsModal: document.getElementById('settingsModal'),
            closeSettings: document.getElementById('closeSettings'),
            ollamaModel: document.getElementById('ollamaModel'),
            ollamaUrl: document.getElementById('ollamaUrl'),
            settingsVoiceEnabled: document.getElementById('settingsVoiceEnabled'),
            saveSettings: document.getElementById('saveSettings')
        };
    }

    attachEventListeners() {
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        this.elements.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        this.elements.voiceBtn.addEventListener('click', () => this.toggleVoiceRecognition());
        this.elements.clearBtn.addEventListener('click', () => this.clearChat());
        this.elements.refreshBtn.addEventListener('click', () => this.refreshStatus());
        this.elements.settingsBtn.addEventListener('click', () => this.showSettings());
        
        this.elements.closeSettings.addEventListener('click', () => this.hideSettings());
        this.elements.saveSettings.addEventListener('click', () => this.saveSettings());
        
        // Close modal when clicking outside
        this.elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.hideSettings();
            }
        });
    }

    async initializeApp() {
        this.updateStatus('Loading AI Model...', false);
        
        try {
            // Load settings
            await this.loadSettings();
            
            // Initialize the embedded model
            this.addMessage('system', 'Initializing local AI assistant...');
            const modelLoaded = await invoke('initialize_model');
            
            if (modelLoaded) {
                await this.loadAvailableModels();
                this.updateStatus('AI Ready', true);
                await this.loadChatHistory();
                this.addMessage('system', 'Local AI assistant ready! No internet required - everything runs on your device.');
            } else {
                this.updateStatus('AI Offline', false);
                this.addMessage('system', 'Failed to initialize local AI assistant.');
            }
        } catch (error) {
            console.error('Initialization error:', error);
            this.updateStatus('Error', false);
            this.addMessage('system', `Error: ${error}`);
        }
    }

    setupSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onstart = () => {
                console.log('Speech recognition started');
                this.elements.voiceIndicator.classList.remove('hidden');
                this.elements.voiceBtn.classList.add('listening');
                this.addMessage('system', '🎤 Listening...');
            };

            this.recognition.onresult = (event) => {
                console.log('Speech recognition result:', event.results);
                const transcript = event.results[0][0].transcript;
                console.log('Transcript:', transcript);
                this.elements.messageInput.value = transcript;
                this.sendMessage();
            };

            this.recognition.onend = () => {
                console.log('Speech recognition ended');
                this.isListening = false;
                this.elements.voiceIndicator.classList.add('hidden');
                this.elements.voiceBtn.classList.remove('listening');
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.isListening = false;
                this.elements.voiceIndicator.classList.add('hidden');
                this.elements.voiceBtn.classList.remove('listening');
                
                let errorMsg = 'Microphone error: ';
                switch(event.error) {
                    case 'not-allowed':
                        errorMsg += 'Microphone permission denied. Please allow microphone access.';
                        break;
                    case 'no-speech':
                        errorMsg += 'No speech detected. Try speaking louder.';
                        break;
                    case 'network':
                        errorMsg += 'Network error occurred.';
                        break;
                    default:
                        errorMsg += event.error;
                }
                this.addMessage('system', errorMsg);
            };
        } else {
            this.elements.voiceBtn.style.display = 'none';
            console.warn('Speech recognition not supported');
            this.addMessage('system', 'Speech recognition not supported in this browser');
        }
    }

    async sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message) return;

        // Clear input and disable send button
        this.elements.messageInput.value = '';
        this.elements.sendBtn.disabled = true;
        
        // Add user message to UI
        this.addMessage('user', message);
        
        // Show typing indicator
        const typingId = this.addMessage('assistant', 'Thinking...');

        try {
            // Send to Rust backend
            const response = await invoke('send_message', { message });
            
            // Remove typing indicator and add response
            this.removeMessage(typingId);
            this.addMessage('assistant', response);
            
            // Text-to-speech if enabled
            if (this.elements.autoSpeech.checked) {
                this.speak(response);
            }
            
        } catch (error) {
            console.error('Send message error:', error);
            this.removeMessage(typingId);
            this.addMessage('system', `Error: ${error}`);
        } finally {
            this.elements.sendBtn.disabled = false;
            this.elements.messageInput.focus();
        }
    }

    addMessage(type, content) {
        const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.id = messageId;
        
        const timestamp = new Date().toLocaleTimeString();
        
        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="message-content">${this.escapeHtml(content)}</div>
                <div class="message-time">${timestamp}</div>
            `;
        } else if (type === 'assistant') {
            messageDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-body">
                    <div class="message-content">${this.escapeHtml(content)}</div>
                    <div class="message-time">${timestamp}</div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">${this.escapeHtml(content)}</div>
            `;
        }

        this.elements.chatMessages.appendChild(messageDiv);
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        
        return messageId;
    }

    removeMessage(messageId) {
        const element = document.getElementById(messageId);
        if (element) {
            element.remove();
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async toggleVoiceRecognition() {
        if (!this.recognition) return;

        if (this.isListening) {
            this.recognition.stop();
        } else {
            try {
                // Request microphone permission first
                await navigator.mediaDevices.getUserMedia({ audio: true });
                this.isListening = true;
                this.recognition.start();
            } catch (error) {
                console.error('Microphone permission error:', error);
                this.addMessage('system', 'Microphone access denied. Please check your browser settings and allow microphone access for this app.');
            }
        }
    }

    speak(text) {
        if (this.synthesis && this.synthesis.speaking) {
            this.synthesis.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        
        this.synthesis.speak(utterance);
    }

    async clearChat() {
        try {
            await invoke('clear_chat_history');
            this.elements.chatMessages.innerHTML = `
                <div class="message system">Chat cleared. How can I help you?</div>
            `;
        } catch (error) {
            console.error('Clear chat error:', error);
        }
    }

    async refreshStatus() {
        await this.initializeApp();
    }

    async loadChatHistory() {
        try {
            const history = await invoke('get_chat_history');
            this.elements.chatMessages.innerHTML = '';
            
            if (history.length === 0) {
                this.addMessage('system', 'Welcome to Project Tektra! How can I assist you today?');
            } else {
                history.forEach(msg => {
                    this.addMessage(msg.role, msg.content);
                });
            }
        } catch (error) {
            console.error('Load chat history error:', error);
        }
    }

    async loadSettings() {
        try {
            const settings = await invoke('get_settings');
            this.elements.ollamaModel.value = settings.model_name;
            this.elements.ollamaUrl.style.display = 'none'; // Hide URL setting for embedded model
            this.elements.settingsVoiceEnabled.checked = settings.voice_enabled;
            this.elements.voiceEnabled.checked = settings.voice_enabled;
            this.elements.autoSpeech.checked = settings.auto_speech;
            
            this.elements.modelName.textContent = settings.model_name;
        } catch (error) {
            console.error('Load settings error:', error);
        }
    }

    async loadAvailableModels() {
        try {
            const models = await invoke('get_available_models');
            this.elements.ollamaModel.innerHTML = '';
            
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                this.elements.ollamaModel.appendChild(option);
            });
        } catch (error) {
            console.error('Load models error:', error);
        }
    }

    showSettings() {
        this.elements.settingsModal.classList.remove('hidden');
    }

    hideSettings() {
        this.elements.settingsModal.classList.add('hidden');
    }

    async saveSettings() {
        try {
            const settings = {
                model_name: this.elements.ollamaModel.value,
                max_tokens: 512,
                temperature: 0.7,
                voice_enabled: this.elements.settingsVoiceEnabled.checked,
                auto_speech: this.elements.autoSpeech.checked
            };

            await invoke('update_settings', { newSettings: settings });
            
            // Update UI
            this.elements.modelName.textContent = settings.model_name;
            this.elements.voiceEnabled.checked = settings.voice_enabled;
            
            this.hideSettings();
            this.addMessage('system', 'Settings saved successfully!');
            
        } catch (error) {
            console.error('Save settings error:', error);
            this.addMessage('system', `Error saving settings: ${error}`);
        }
    }

    updateStatus(text, connected) {
        this.elements.statusText.textContent = text;
        this.elements.modelStatus.textContent = text;
        
        if (connected) {
            this.elements.statusDot.classList.add('connected');
        } else {
            this.elements.statusDot.classList.remove('connected');
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ProjectTektra();
});

// Add styles
const style = document.createElement('style');
style.textContent = `
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 100vh;
        overflow: hidden;
    }

    .header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }

    .header h1 {
        font-size: 1.5rem;
        font-weight: 600;
    }

    .status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ff4444;
        animation: pulse 2s infinite;
    }

    .status-dot.connected {
        background: #44ff44;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .icon-btn {
        background: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        color: white;
        cursor: pointer;
        font-size: 1.2rem;
        transition: all 0.3s ease;
    }

    .icon-btn:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.1);
    }

    .main-container {
        display: flex;
        height: calc(100vh - 80px);
    }

    .chat-section {
        flex: 1;
        display: flex;
        flex-direction: column;
        padding: 1rem;
    }

    .chat-messages {
        flex: 1;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .message {
        margin-bottom: 1rem;
        padding: 0.75rem;
        border-radius: 8px;
        word-wrap: break-word;
    }

    .message.user {
        background: rgba(102, 126, 234, 0.3);
        color: white;
        margin-left: 20%;
        text-align: right;
    }

    .message.assistant {
        background: rgba(118, 75, 162, 0.3);
        color: white;
        margin-right: 20%;
        display: flex;
        gap: 0.5rem;
    }

    .message.system {
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.8);
        font-style: italic;
        text-align: center;
    }

    .message-avatar {
        font-size: 1.5rem;
        flex-shrink: 0;
    }

    .message-body {
        flex: 1;
    }

    .message-time {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }

    .chat-input-container {
        display: flex;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    #messageInput {
        flex: 1;
        padding: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        font-size: 1rem;
    }

    #messageInput::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }

    .send-btn, .voice-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        transition: all 0.3s ease;
        min-width: 50px;
    }

    .send-btn:hover, .voice-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .voice-btn.listening {
        background: #ff6b6b;
        animation: pulse 1s infinite;
    }

    .sidebar {
        width: 300px;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .info-panel, .voice-panel {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }

    .info-panel h3, .voice-panel h3 {
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }

    .model-info {
        margin-bottom: 1rem;
    }

    .model-name {
        font-weight: 600;
        font-size: 1rem;
    }

    .model-status {
        font-size: 0.9rem;
        opacity: 0.8;
    }

    .action-btn {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        cursor: pointer;
        margin-bottom: 0.5rem;
        width: 100%;
        transition: all 0.3s ease;
    }

    .action-btn:hover {
        background: rgba(255, 255, 255, 0.3);
    }

    .voice-controls {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .toggle {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
    }

    .toggle input[type="checkbox"] {
        display: none;
    }

    .slider {
        width: 40px;
        height: 20px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        position: relative;
        transition: all 0.3s ease;
    }

    .slider::before {
        content: '';
        position: absolute;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: white;
        top: 2px;
        left: 2px;
        transition: all 0.3s ease;
    }

    .toggle input[type="checkbox"]:checked + .slider {
        background: #667eea;
    }

    .toggle input[type="checkbox"]:checked + .slider::before {
        transform: translateX(20px);
    }

    .voice-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    .wave-animation {
        display: flex;
        gap: 2px;
    }

    .wave {
        width: 3px;
        height: 15px;
        background: #667eea;
        animation: wave 1s infinite ease-in-out;
    }

    .wave:nth-child(2) { animation-delay: -0.9s; }
    .wave:nth-child(3) { animation-delay: -0.8s; }

    @keyframes wave {
        0%, 40%, 100% { transform: scaleY(0.4); }
        20% { transform: scaleY(1); }
    }

    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }

    .modal-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        width: 500px;
        max-width: 90vw;
        max-height: 80vh;
        overflow: auto;
    }

    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    }

    .modal-header h2 {
        margin: 0;
        color: #333;
    }

    .close-btn {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #666;
    }

    .modal-body {
        padding: 1.5rem;
    }

    .setting-group {
        margin-bottom: 1.5rem;
    }

    .setting-group label {
        display: block;
        margin-bottom: 0.5rem;
        color: #333;
        font-weight: 500;
    }

    .setting-group input, .setting-group select {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #ddd;
        border-radius: 6px;
        font-size: 1rem;
    }

    .modal-footer {
        padding: 1.5rem;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
        text-align: right;
    }

    .save-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 600;
    }

    .hidden {
        display: none !important;
    }

    /* Scrollbar styles */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
`;
document.head.appendChild(style);