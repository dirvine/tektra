import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { Avatar2D } from './avatar.js';

class ProjectTektra {
    constructor() {
        this.isListening = false;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.avatar = null;
        this.cameraActive = false;
        this.cameraStream = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.initializeApp();
        this.setupSpeechRecognition();
        this.initializeAvatar();
        this.initializeCamera();
    }

    initializeElements() {
        this.elements = {
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
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
            modelSelector: document.getElementById('modelSelector'),
            modelUrl: document.getElementById('modelUrl'),
            settingsVoiceEnabled: document.getElementById('settingsVoiceEnabled'),
            saveSettings: document.getElementById('saveSettings'),
            systemPrompt: document.getElementById('systemPrompt'),
            userPrefix: document.getElementById('userPrefix'),
            assistantPrefix: document.getElementById('assistantPrefix'),
            
            // Loading overlay
            loadingOverlay: document.getElementById('loadingOverlay'),
            progressFill: document.getElementById('progressFill'),
            progressStatus: document.getElementById('progressStatus'),
            progressPercent: document.getElementById('progressPercent'),
            
            // Camera elements
            cameraToggleBtn: document.getElementById('cameraToggleBtn'),
            cameraPreview: document.getElementById('cameraPreview'),
            cameraCanvas: document.getElementById('cameraCanvas'),
            
            // Avatar elements
            avatarCanvas: document.getElementById('avatarCanvas'),
            avatarExpression: document.getElementById('avatarExpression')
        };
    }

    attachEventListeners() {
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        this.elements.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        this.elements.clearBtn.addEventListener('click', () => this.clearChat());
        this.elements.refreshBtn.addEventListener('click', () => this.refreshStatus());
        this.elements.settingsBtn.addEventListener('click', () => this.showSettings());
        
        // Voice enabled toggle - controls always listening
        this.elements.voiceEnabled.addEventListener('change', () => this.toggleAlwaysListening());
        
        this.elements.closeSettings.addEventListener('click', () => this.hideSettings());
        this.elements.saveSettings.addEventListener('click', () => this.saveSettings());
        
        // Close modal when clicking outside
        this.elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.hideSettings();
            }
        });
        
        // Camera toggle
        this.elements.cameraToggleBtn.addEventListener('click', () => this.toggleCamera());
        
        // Avatar expression change
        this.elements.avatarExpression.addEventListener('change', (e) => {
            if (this.avatar) {
                this.avatar.setExpression(e.target.value);
            }
        });
    }

    async initializeApp() {
        this.updateStatus('Loading AI Model...', false);
        
        // Show loading overlay
        this.elements.loadingOverlay.classList.remove('hidden');
        
        // Track which models are being loaded
        this.modelsLoading = new Set();
        this.modelProgress = {};
        
        // Listen for model loading progress
        const unlisten = await listen('model-loading-progress', (event) => {
            const { progress, status, model_name } = event.payload;
            
            // Track this model
            this.modelsLoading.add(model_name);
            this.modelProgress[model_name] = progress;
            
            // Calculate overall progress
            const totalProgress = Object.values(this.modelProgress).reduce((sum, p) => sum + p, 0) / Object.keys(this.modelProgress).length || 0;
            
            // Update progress bar
            this.elements.progressFill.style.width = `${totalProgress}%`;
            this.elements.progressStatus.textContent = status;
            this.elements.progressPercent.textContent = `${Math.round(totalProgress)}%`;
            
            // Update status text
            this.updateStatus(`${status} (${Math.round(totalProgress)}%)`, false);
            
            if (progress === 100 && model_name) {
                this.addMessage('system', `✅ ${model_name} loaded successfully!`);
            }
        });
        
        // Listen for completion
        const unlistenComplete = await listen('model-loading-complete', (event) => {
            const { success } = event.payload;
            if (success && Object.values(this.modelProgress).every(p => p === 100)) {
                // All models loaded
                setTimeout(() => {
                    this.elements.loadingOverlay.classList.add('hidden');
                }, 500);
                
                this.updateStatus('AI Ready', true);
            }
        });
        
        try {
            // Load settings
            await this.loadSettings();
            
            // Initialize the AI model
            this.addMessage('system', 'Initializing Qwen2.5-VL-7B model...');
            const modelLoaded = await invoke('initialize_model');
            
            if (modelLoaded) {
                await this.loadAvailableModels();
                this.updateStatus('AI Ready', true);
                await this.loadChatHistory();
                
                // Tektra introduces itself on startup
                const introduction = "Hello! I'm Tektra, your AI assistant powered by the Qwen2.5-VL model. I'm here to help you with questions, tasks, and conversations. I'm always listening when you enable the microphone toggle, so just speak naturally and I'll assist you. How can I help you today?";
                this.addMessage('assistant', introduction);
                
                // Auto-speak the introduction if voice is enabled
                if (this.elements.autoSpeech.checked) {
                    this.speak(introduction);
                }
                
                // Initialize Whisper after main model
                this.addMessage('system', 'Initializing Whisper speech-to-text...');
                try {
                    const whisperLoaded = await invoke('initialize_whisper');
                    if (!whisperLoaded) {
                        this.addMessage('system', '⚠️ Whisper initialization failed. Voice input may not work properly.');
                    }
                } catch (whisperError) {
                    console.error('Whisper initialization error:', whisperError);
                    this.addMessage('system', `⚠️ Whisper error: ${whisperError}`);
                }
                
                // Hide loading overlay if not already hidden
                setTimeout(() => {
                    this.elements.loadingOverlay.classList.add('hidden');
                }, 500);
            } else {
                this.updateStatus('AI Offline', false);
                this.addMessage('system', 'Failed to initialize AI model.');
                this.elements.loadingOverlay.classList.add('hidden');
            }
        } catch (error) {
            console.error('Initialization error:', error);
            this.updateStatus('Error', false);
            this.addMessage('system', `Error: ${error}`);
            this.elements.loadingOverlay.classList.add('hidden');
        }
    }

    setupSpeechRecognition() {
        // Use native Tauri audio recording
        console.log('Setting up native audio recording...');
        
        // Update UI with more informative message
        this.addMessage('system', '🎤 Voice input is ready! When you enable "Always Listening", I\'ll transcribe your speech and respond automatically.');
        
        // Setup native audio recording
        this.isRecording = false;
        this.useNativeAudio = true;
        this.continuousListening = false;
        
        // Listen for recording events
        this.setupAudioEventListeners();
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Don't initialize any Web Speech API stuff
        this.recognition = null;
        this.mediaRecorder = null;
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
            // Send to Rust backend - use camera-enabled command if camera is active
            const response = this.cameraActive 
                ? await invoke('send_message_with_camera', { message })
                : await invoke('send_message', { message });
            
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
        
        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="message-content">${this.escapeHtml(content)}</div>
            `;
        } else if (type === 'assistant') {
            messageDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-body">
                    <div class="message-content">${this.formatResponse(content)}</div>
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

    formatResponse(text) {
        // Keep basic HTML escaping but preserve line breaks and formatting
        let formatted = this.escapeHtml(text);
        
        // Convert line breaks to <br> tags
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Convert double line breaks to paragraph breaks
        formatted = formatted.replace(/(<br>){2,}/g, '</p><p>');
        
        // Wrap in paragraphs if there are paragraph breaks
        if (formatted.includes('</p><p>')) {
            formatted = '<p>' + formatted + '</p>';
        }
        
        // Handle basic markdown-style formatting
        // Bold text **text**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic text *text*
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Code blocks ```code```
        formatted = formatted.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
        
        // Inline code `code`
        formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Handle numbered lists (1. item)
        formatted = formatted.replace(/^(\d+)\.\s(.+)$/gm, '<li>$2</li>');
        if (formatted.includes('<li>')) {
            formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
        }
        
        // Handle bullet lists (- item or * item)
        formatted = formatted.replace(/^[-*]\s(.+)$/gm, '<li>$1</li>');
        if (formatted.includes('<li>') && !formatted.includes('<ol>')) {
            formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        }
        
        return formatted;
    }

    
    async setupAudioEventListeners() {
        const { listen } = await import('@tauri-apps/api/event');
        
        // Listen for recording started
        await listen('recording-started', () => {
            console.log('Recording started');
            this.elements.voiceIndicator.classList.remove('hidden');
        });
        
        // Listen for recording stopped
        await listen('recording-stopped', (event) => {
            console.log('Recording stopped, samples:', event.payload);
            this.elements.voiceIndicator.classList.add('hidden');
        });
        
        // Listen for speech transcription
        await listen('speech-transcribed', (event) => {
            console.log('Speech transcribed:', event.payload);
            const text = event.payload;
            
            // Add to message input or send directly
            if (text && text.trim().length > 0) {
                // Show what was transcribed
                this.addMessage('system', `🎤 Heard: "${text}"`);
                
                this.elements.messageInput.value = text;
                
                // Auto-send if continuous listening is enabled
                console.log('Continuous listening enabled:', this.continuousListening);
                console.log('Voice enabled checkbox:', this.elements.voiceEnabled.checked);
                
                if (this.continuousListening && this.elements.voiceEnabled.checked) {
                    console.log('Auto-sending transcribed message:', text);
                    // Small delay to ensure UI is updated
                    setTimeout(() => {
                        this.sendMessage();
                    }, 100);
                } else {
                    console.log('Not auto-sending - continuous listening disabled or voice not enabled');
                }
            }
        });
        
        // Listen for audio data for multimodal processing
        await listen('audio-for-processing', async (event) => {
            console.log('Received audio data for multimodal processing:', event.payload);
            const { audio_data, sample_rate, duration_secs, prompt } = event.payload;
            
            try {
                // Show that we're processing audio
                this.addMessage('system', `🎤 Processing ${duration_secs.toFixed(1)}s audio with Qwen2.5-VL...`);
                
                // Show typing indicator
                const typingId = this.addMessage('assistant', 'Listening and understanding...');
                
                // Send audio data to multimodal AI endpoint
                const response = await invoke('process_audio_input', {
                    message: prompt,
                    audioData: audio_data
                });
                
                // Remove typing indicator and add response
                this.removeMessage(typingId);
                this.addMessage('assistant', response);
                
                // Text-to-speech if enabled
                if (this.elements.autoSpeech.checked) {
                    this.speak(response);
                }
                
            } catch (error) {
                console.error('Audio processing error:', error);
                this.addMessage('system', `❌ Audio processing failed: ${error}`);
            }
        });
        
        // Listen for assistant interruption
        await listen('interrupt-assistant', () => {
            console.log('Interrupting assistant');
            // Stop any ongoing speech synthesis
            if (this.synthesis && this.synthesis.speaking) {
                this.synthesis.cancel();
            }
        });
    }

    async toggleAlwaysListening() {
        if (this.elements.voiceEnabled.checked) {
            // Start continuous listening
            this.continuousListening = true;
            this.addMessage('system', '🎤 Always listening mode activated. I\'ll transcribe your speech and respond automatically.');
            await this.startContinuousRecording();
        } else {
            // Stop continuous listening
            this.continuousListening = false;
            this.addMessage('system', '🎤 Always listening mode deactivated.');
            if (this.isRecording) {
                await this.stopContinuousRecording();
            }
        }
    }
    
    async startContinuousRecording() {
        try {
            const started = await invoke('start_audio_recording');
            if (started) {
                this.isRecording = true;
                this.elements.voiceIndicator.classList.remove('hidden');
                console.log('Continuous recording started');
                
                // Set up interval to process audio chunks
                this.recordingInterval = setInterval(async () => {
                    if (this.continuousListening && this.isRecording) {
                        try {
                            await invoke('process_audio_stream');
                        } catch (error) {
                            console.error('Error processing audio stream:', error);
                        }
                    }
                }, 100); // Process every 100ms for real-time response
            } else {
                this.addMessage('system', 'Failed to start recording. Check microphone permissions in System Preferences.');
                this.elements.voiceEnabled.checked = false;
            }
        } catch (error) {
            console.error('Recording error:', error);
            this.addMessage('system', `Recording error: ${error}`);
            this.elements.voiceEnabled.checked = false;
        }
    }
    
    async stopContinuousRecording() {
        try {
            if (this.recordingInterval) {
                clearInterval(this.recordingInterval);
            }
            
            const audioData = await invoke('stop_audio_recording');
            console.log('Recording stopped, total samples:', audioData.length);
            
            this.isRecording = false;
            this.elements.voiceIndicator.classList.add('hidden');
        } catch (error) {
            console.error('Stop recording error:', error);
        }
    }

    async speak(text) {
        if (this.synthesis && this.synthesis.speaking) {
            this.synthesis.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        
        // Start avatar lip sync
        if (this.avatar) {
            try {
                const lipSyncFrames = await invoke('start_avatar_speaking', { text });
                this.avatar.animateLipSync(lipSyncFrames);
            } catch (error) {
                console.error('Avatar lip sync error:', error);
            }
        }
        
        // Handle speech end
        utterance.onend = async () => {
            if (this.avatar) {
                try {
                    await invoke('stop_avatar_speaking');
                } catch (error) {
                    console.error('Avatar stop speaking error:', error);
                }
            }
        };
        
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
            this.elements.modelSelector.value = settings.model_name;
            this.elements.modelUrl.style.display = 'none'; // Hide URL setting for embedded model
            this.elements.settingsVoiceEnabled.checked = settings.voice_enabled;
            this.elements.voiceEnabled.checked = settings.voice_enabled;
            this.elements.autoSpeech.checked = settings.auto_speech;
            
            // Load prompt settings
            this.elements.systemPrompt.value = settings.system_prompt || 'You are Tektra, a helpful AI assistant powered by the Qwen2.5-VL model. You provide accurate, thoughtful, and detailed responses.';
            this.elements.userPrefix.value = settings.user_prefix || 'User: ';
            this.elements.assistantPrefix.value = settings.assistant_prefix || 'Assistant: ';
            
            this.elements.modelName.textContent = settings.model_name;
        } catch (error) {
            console.error('Load settings error:', error);
        }
    }

    async loadAvailableModels() {
        try {
            const models = await invoke('get_available_models');
            this.elements.modelSelector.innerHTML = '';
            
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                this.elements.modelSelector.appendChild(option);
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
                model_name: this.elements.modelSelector.value,
                max_tokens: 512,
                temperature: 0.7,
                voice_enabled: this.elements.settingsVoiceEnabled.checked,
                auto_speech: this.elements.autoSpeech.checked,
                system_prompt: this.elements.systemPrompt.value,
                user_prefix: this.elements.userPrefix.value,
                assistant_prefix: this.elements.assistantPrefix.value
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
            this.elements.statusDot.classList.add('online');
            this.elements.statusDot.classList.remove('offline');
        } else {
            this.elements.statusDot.classList.remove('online');
            this.elements.statusDot.classList.add('offline');
        }
    }
    
    // Avatar methods
    initializeAvatar() {
        if (this.elements.avatarCanvas) {
            this.avatar = new Avatar2D(this.elements.avatarCanvas);
            
            // Listen for avatar state changes from backend
            listen('avatar-state-changed', (event) => {
                if (this.avatar) {
                    this.avatar.setState(event.payload);
                }
            });
            
            // Set up periodic blinking
            setInterval(() => {
                if (this.avatar && Math.random() < 0.3) {
                    invoke('avatar_blink');
                }
            }, 3000);
        }
    }
    
    // Camera methods
    async initializeCamera() {
        try {
            // Listen for camera events
            await listen('camera-ready', () => {
                this.addMessage('system', '📷 Camera initialized successfully');
            });
            
            await listen('camera-error', (event) => {
                this.addMessage('system', `📷 Camera error: ${event.payload}`);
            });
            
            await listen('camera-capture-started', () => {
                this.cameraActive = true;
                this.elements.cameraToggleBtn.textContent = 'Disable Camera';
                this.elements.cameraPreview.classList.remove('hidden');
                this.startCameraPreview();
            });
            
            await listen('camera-capture-stopped', () => {
                this.cameraActive = false;
                this.elements.cameraToggleBtn.textContent = 'Enable Camera';
                this.elements.cameraPreview.classList.add('hidden');
                this.stopCameraPreview();
            });
        } catch (error) {
            console.error('Camera initialization error:', error);
        }
    }
    
    async toggleCamera() {
        try {
            if (!this.cameraActive) {
                // Initialize camera if not already done
                const initialized = await invoke('initialize_camera');
                if (initialized) {
                    await invoke('start_camera_capture');
                }
            } else {
                await invoke('stop_camera_capture');
            }
        } catch (error) {
            console.error('Camera toggle error:', error);
            this.addMessage('system', `📷 Camera error: ${error}`);
        }
    }
    
    async startCameraPreview() {
        const canvas = this.elements.cameraCanvas;
        const ctx = canvas.getContext('2d');
        
        const updateFrame = async () => {
            if (!this.cameraActive) return;
            
            try {
                const frameData = await invoke('get_camera_frame');
                
                // Create an image from the base64 data
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = frameData;
                
                // Continue updating
                if (this.cameraActive) {
                    requestAnimationFrame(updateFrame);
                }
            } catch (error) {
                console.error('Camera frame error:', error);
            }
        };
        
        updateFrame();
    }
    
    stopCameraPreview() {
        // Clear the canvas
        const canvas = this.elements.cameraCanvas;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ProjectTektra();
});

// Styles are now loaded from external styles.css file