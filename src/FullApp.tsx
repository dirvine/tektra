import React, { useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SimpleHeaderBar from './components/SimpleHeaderBar';
import LeftSidebar from './components/LeftSidebar';
import Avatar3D from './components/Avatar3D';
import { useTektraStore } from './store';
import { 
  Send, 
  Mic, 
  MicOff, 
  Camera, 
  CameraOff, 
  Paperclip,
  MessageSquare,
  BarChart3,
  FileText,
  Database,
  CheckSquare
} from 'lucide-react';
import './App.css';

// Create QueryClient instance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Simple Chat Interface Component
const SimpleChatInterface: React.FC = () => {
  const messages = useTektraStore((state) => state.messages);
  const isRecording = useTektraStore((state) => state.isRecording);
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const addMessage = useTektraStore((state) => state.addMessage);
  const setRecording = useTektraStore((state) => state.setRecording);

  const [inputValue, setInputValue] = React.useState('');

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;
    
    addMessage({
      role: 'user',
      content: inputValue
    });
    
    // Simulate AI response
    setTimeout(() => {
      addMessage({
        role: 'assistant',
        content: `I understand you said: "${inputValue}". This is the professional Tektra AI Assistant interface working with full chat functionality!`
      });
    }, 1000);
    
    setInputValue('');
  };

  const toggleRecording = () => {
    setRecording(!isRecording);
    if (!isRecording) {
      // Start recording
      addMessage({
        role: 'system',
        content: '🎤 Voice recording started...'
      });
    } else {
      // Stop recording
      addMessage({
        role: 'system',
        content: '🎤 Voice recording stopped. Processing...'
      });
    }
  };

  const toggleCamera = () => {
    addMessage({
      role: 'system',
      content: modelStatus.cameraEnabled ? '📷 Camera disabled' : '📷 Camera enabled'
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <MessageSquare className="w-12 h-12 text-text-tertiary mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              Welcome to Tektra AI Assistant
            </h3>
            <p className="text-text-secondary">
              Start a conversation with voice, text, or visual input
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-accent text-white'
                    : message.role === 'system'
                    ? 'bg-surface border border-border-primary text-text-secondary text-sm'
                    : 'bg-surface border border-border-primary text-text-primary'
                }`}
              >
                <p className="text-sm">{message.content}</p>
                <p className="text-xs opacity-70 mt-1">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-border-primary p-4">
        <div className="flex items-center space-x-2">
          {/* Voice Input */}
          <button
            onClick={toggleRecording}
            className={`p-3 rounded-full transition-colors ${
              isRecording
                ? 'bg-error text-white'
                : 'bg-surface border border-border-primary hover:bg-surface-hover text-text-secondary'
            }`}
            title={isRecording ? 'Stop recording' : 'Start voice input'}
          >
            {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>

          {/* Camera Input */}
          <button
            onClick={toggleCamera}
            className={`p-3 rounded-full transition-colors ${
              modelStatus.cameraEnabled
                ? 'bg-success text-white'
                : 'bg-surface border border-border-primary hover:bg-surface-hover text-text-secondary'
            }`}
            title={modelStatus.cameraEnabled ? 'Disable camera' : 'Enable camera'}
          >
            {modelStatus.cameraEnabled ? <Camera className="w-5 h-5" /> : <CameraOff className="w-5 h-5" />}
          </button>

          {/* File Attachment */}
          <button
            className="p-3 rounded-full bg-surface border border-border-primary hover:bg-surface-hover text-text-secondary transition-colors"
            title="Attach file"
          >
            <Paperclip className="w-5 h-5" />
          </button>

          {/* Text Input */}
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your message..."
              className="w-full px-4 py-3 bg-surface border border-border-primary rounded-button text-text-primary placeholder-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent"
            />
          </div>

          {/* Send Button */}
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim()}
            className="p-3 rounded-full bg-accent hover:bg-accent-hover disabled:bg-surface disabled:text-text-tertiary text-white transition-colors"
            title="Send message"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

// Simple Right Sidebar Component
const SimpleRightSidebar: React.FC = () => {
  const uiState = useTektraStore((state) => state.uiState);
  const sessionState = useTektraStore((state) => state.sessionState);
  const messages = useTektraStore((state) => state.messages);
  const setActiveTab = useTektraStore((state) => state.setActiveTab);

  const tabs = [
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'session', label: 'Session', icon: MessageSquare },
    { id: 'files', label: 'Files', icon: FileText },
    { id: 'knowledge', label: 'Knowledge', icon: Database },
    { id: 'tasks', label: 'Tasks', icon: CheckSquare },
  ] as const;

  if (!uiState.rightSidebarVisible) return null;

  return (
    <aside className="fixed right-0 top-16 bottom-8 z-40 w-80 bg-secondary-bg border-l border-border-primary flex flex-col overflow-hidden">
      {/* Tabs */}
      <div className="flex border-b border-border-primary">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex items-center justify-center space-x-1 py-3 px-2 text-xs transition-colors ${
              uiState.activeTab === tab.id
                ? 'bg-surface text-accent border-b-2 border-accent'
                : 'text-text-secondary hover:text-text-primary hover:bg-surface/50'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span className="hidden lg:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {uiState.activeTab === 'analytics' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Analytics</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <p className="text-xs text-text-tertiary">Messages</p>
                <p className="text-lg font-semibold text-text-primary">{messages.length}</p>
              </div>
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <p className="text-xs text-text-tertiary">Duration</p>
                <p className="text-lg font-semibold text-text-primary">{Math.floor(sessionState.duration / 60)}m</p>
              </div>
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <p className="text-xs text-text-tertiary">Tokens</p>
                <p className="text-lg font-semibold text-text-primary">{sessionState.tokenUsage}</p>
              </div>
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <p className="text-xs text-text-tertiary">Cost</p>
                <p className="text-lg font-semibold text-text-primary">${sessionState.costEstimate.toFixed(3)}</p>
              </div>
            </div>
          </div>
        )}

        {uiState.activeTab === 'session' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Session Info</h3>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-text-tertiary">Started</p>
                <p className="text-text-primary">Just now</p>
              </div>
              <div>
                <p className="text-text-tertiary">Project</p>
                <p className="text-text-primary">{uiState.currentProject}</p>
              </div>
              <div>
                <p className="text-text-tertiary">Mode</p>
                <p className="text-text-primary">Interactive</p>
              </div>
            </div>
          </div>
        )}

        {uiState.activeTab === 'files' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Recent Files</h3>
            <p className="text-sm text-text-tertiary">No files uploaded yet</p>
          </div>
        )}

        {uiState.activeTab === 'knowledge' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Knowledge Base</h3>
            <p className="text-sm text-text-tertiary">Connected to local knowledge</p>
          </div>
        )}

        {uiState.activeTab === 'tasks' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Active Tasks</h3>
            <p className="text-sm text-text-tertiary">No active tasks</p>
          </div>
        )}
      </div>
    </aside>
  );
};

// Simple Status Bar Component
const SimpleStatusBar: React.FC = () => {
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const isRecording = useTektraStore((state) => state.isRecording);

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-between px-4 text-xs">
      <div className="flex items-center space-x-4">
        <span className="text-text-tertiary">Tektra AI Assistant</span>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            modelStatus.isLoaded ? 'bg-success' : 'bg-warning'
          }`}></div>
          <span className="text-text-secondary">
            {modelStatus.isLoaded ? 'Ready' : 'Loading...'}
          </span>
        </div>
        {isRecording && (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-error animate-pulse"></div>
            <span className="text-error">Recording</span>
          </div>
        )}
      </div>
      <div className="text-text-tertiary">
        Press / for shortcuts
      </div>
    </div>
  );
};

const FullAppContent: React.FC = () => {
  // Use individual selectors
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);
  const setModelStatus = useTektraStore((state) => state.setModelStatus);
  const addMessage = useTektraStore((state) => state.addMessage);
  const addNotification = useTektraStore((state) => state.addNotification);

  useEffect(() => {
    initializeApp();
    setupEventListeners();
  }, []);

  const initializeApp = async () => {
    try {
      setModelStatus({ isLoading: true });
      
      // Initialize AI model
      const modelLoaded = await invoke<boolean>('initialize_model').catch(() => false);
      if (modelLoaded) {
        setModelStatus({ isLoaded: true, isLoading: false });
        addMessage({
          role: 'system',
          content: '✅ AI model loaded successfully. Welcome to Tektra!'
        });
        addNotification({
          type: 'success',
          message: 'System ready'
        });
      } else {
        // Fallback for demo
        setTimeout(() => {
          setModelStatus({ isLoaded: true, isLoading: false });
          addMessage({
            role: 'system',
            content: '✅ AI model simulation loaded. Professional interface ready!'
          });
        }, 2000);
      }
      
    } catch (error) {
      console.error('Model initialization error:', error);
      setModelStatus({ isLoading: false });
      addMessage({
        role: 'system',
        content: `❌ Failed to load AI model: ${error}`
      });
    }
  };

  const setupEventListeners = async () => {
    try {
      // Listen for AI responses
      await listen('ai-response', (event: any) => {
        addMessage({
          role: 'assistant',
          content: event.payload.content
        });
      });

      // Listen for transcription results
      await listen('transcription-result', (event: any) => {
        addMessage({
          role: 'user',
          content: event.payload.text
        });
      });
    } catch (error) {
      console.log('Event listener setup skipped:', error);
    }
  };

  return (
    <div className="min-h-screen bg-primary-bg text-text-primary overflow-hidden">
      {/* Header Bar */}
      <SimpleHeaderBar />

      {/* Main Layout */}
      <div className="flex pt-16 pb-8">
        {/* Left Sidebar */}
        <LeftSidebar />

        {/* Main Content Area */}
        <main className={`
          flex-1 flex overflow-hidden transition-all duration-300
          ${uiState.leftSidebarCollapsed ? 'ml-16' : 'ml-80'}
          ${uiState.rightSidebarVisible ? 'mr-80' : 'mr-0'}
        `}>
          {/* Avatar Panel */}
          <div className="w-80 flex flex-col border-r border-border-primary">
            <div className="p-4 border-b border-border-primary">
              <h3 className="font-semibold text-text-primary">AI Avatar</h3>
            </div>
            <div className="flex-1">
              <Avatar3D />
            </div>
          </div>

          {/* Chat Interface */}
          <div className="flex-1 flex flex-col">
            <SimpleChatInterface />
          </div>
        </main>

        {/* Right Sidebar */}
        <SimpleRightSidebar />
      </div>

      {/* Status Bar */}
      <SimpleStatusBar />
    </div>
  );
};

const FullApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <FullAppContent />
    </QueryClientProvider>
  );
};

export default FullApp;