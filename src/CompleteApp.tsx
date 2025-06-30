import React, { useEffect, useState } from 'react';
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
  CheckSquare,
  Settings,
  Users,
  Zap,
  Activity,
  Wifi,
  WifiOff,
  Volume2
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

// Enhanced Chat Interface with full functionality
const EnhancedChatInterface: React.FC = () => {
  const messages = useTektraStore((state) => state.messages);
  const isRecording = useTektraStore((state) => state.isRecording);
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const addMessage = useTektraStore((state) => state.addMessage);
  const setRecording = useTektraStore((state) => state.setRecording);
  const setAvatarSpeaking = useTektraStore((state) => state.setAvatarSpeaking);
  const setAvatarListening = useTektraStore((state) => state.setAvatarListening);

  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  
  // Ref for auto-scrolling
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    
    addMessage({
      role: 'user',
      content: inputValue
    });
    
    const userMessage = inputValue;
    setInputValue('');
    setIsTyping(true);
    setAvatarSpeaking(true);
    
    try {
      // Use real backend - check if camera is enabled for multimodal
      const result = modelStatus.cameraEnabled 
        ? await invoke<string>('send_message_with_camera', { message: userMessage })
        : await invoke<string>('send_message', { message: userMessage });
      
      addMessage({
        role: 'assistant',
        content: result
      });
    } catch (error) {
      console.error('AI response error:', error);
      addMessage({
        role: 'assistant',
        content: `Error: ${error}`
      });
    } finally {
      setIsTyping(false);
      setAvatarSpeaking(false);
    }
  };

  const toggleRecording = async () => {
    const newRecording = !isRecording;
    setRecording(newRecording);
    setAvatarListening(newRecording);
    
    try {
      if (newRecording) {
        const started = await invoke<boolean>('start_audio_recording');
        if (started) {
          addMessage({
            role: 'system',
            content: '🎤 Voice recording started...'
          });
        }
      } else {
        const audioData = await invoke<number[]>('stop_audio_recording');
        addMessage({
          role: 'system',
          content: '🎤 Voice recording stopped. Processing speech...'
        });
        
        // Process the actual audio data
        if (audioData && audioData.length > 0) {
          const audioBytes = new Uint8Array(audioData);
          const result = await invoke<string>('process_audio_input', { 
            message: '', 
            audioData: Array.from(audioBytes) 
          });
          
          if (result.trim()) {
            addMessage({
              role: 'user',
              content: result
            });
          }
        }
        setAvatarListening(false);
      }
    } catch (error) {
      console.error('Audio recording error:', error);
      addMessage({
        role: 'system',
        content: `❌ Audio error: ${error}`
      });
      setRecording(false);
      setAvatarListening(false);
    }
  };

  const toggleCamera = async () => {
    const newCameraState = !modelStatus.cameraEnabled;
    
    try {
      if (newCameraState) {
        const initialized = await invoke<boolean>('initialize_camera');
        if (initialized) {
          const started = await invoke<boolean>('start_camera_capture');
          if (started) {
            // Update store state
            const setModelStatus = useTektraStore.getState().setModelStatus;
            setModelStatus({ cameraEnabled: true });
            addMessage({
              role: 'system',
              content: '📷 Camera enabled - Ready for vision tasks'
            });
          }
        }
      } else {
        await invoke<boolean>('stop_camera_capture');
        const setModelStatus = useTektraStore.getState().setModelStatus;
        setModelStatus({ cameraEnabled: false });
        addMessage({
          role: 'system',
          content: '📷 Camera disabled'
        });
      }
    } catch (error) {
      console.error('Camera toggle error:', error);
      addMessage({
        role: 'system',
        content: `❌ Camera error: ${error}`
      });
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Chat Header */}
      <div className="flex-shrink-0 p-4 border-b border-border-primary bg-surface/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-text-primary">AI Assistant Chat</h2>
            <p className="text-sm text-text-secondary">Professional multimodal conversation interface</p>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              modelStatus.isLoaded ? 'bg-success' : 'bg-warning'
            }`}></div>
            <span className="text-sm text-text-secondary">
              {modelStatus.isLoaded ? 'Ready' : 'Loading...'}
            </span>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-4 min-h-0" style={{height: 'calc(100vh - 200px)'}}>
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
                className={`max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl px-4 py-3 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-accent text-white'
                    : message.role === 'system'
                    ? 'bg-surface border border-border-primary text-text-secondary text-sm'
                    : 'bg-surface border border-border-primary text-text-primary'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
                <p className="text-xs opacity-70 mt-1">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
        
        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="max-w-xs px-4 py-3 bg-surface border border-border-primary rounded-lg">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-text-tertiary rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-text-tertiary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-text-tertiary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-border-primary p-4 bg-surface/30">
        <div className="flex items-center space-x-3">
          {/* Voice Input */}
          <button
            onClick={toggleRecording}
            className={`p-3 rounded-full transition-all duration-200 ${
              isRecording
                ? 'bg-error text-white shadow-lg animate-pulse'
                : 'bg-surface border border-border-primary hover:bg-surface-hover text-text-secondary'
            }`}
            title={isRecording ? 'Stop recording' : 'Start voice input'}
          >
            {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>

          {/* Camera Input */}
          <button
            onClick={toggleCamera}
            className={`p-3 rounded-full transition-all duration-200 ${
              modelStatus.cameraEnabled
                ? 'bg-success text-white shadow-lg'
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

        {/* Quick Actions */}
        <div className="flex items-center justify-center space-x-4 mt-3 text-xs text-text-tertiary">
          <span>Press / for shortcuts</span>
          <span>•</span>
          <span>Tab to autocomplete</span>
          <span>•</span>
          <span>Shift+Enter for new line</span>
        </div>
      </div>
    </div>
  );
};

// Complete Right Sidebar with all tabs
const CompleteRightSidebar: React.FC = () => {
  const uiState = useTektraStore((state) => state.uiState);
  const sessionState = useTektraStore((state) => state.sessionState);
  const messages = useTektraStore((state) => state.messages);
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const setActiveTab = useTektraStore((state) => state.setActiveTab);
  const toggleRightSidebar = useTektraStore((state) => state.toggleRightSidebar);

  const tabs = [
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'session', label: 'Session', icon: Users },
    { id: 'files', label: 'Files', icon: FileText },
    { id: 'knowledge', label: 'Knowledge', icon: Database },
    { id: 'tasks', label: 'Tasks', icon: CheckSquare },
  ] as const;

  if (!uiState.rightSidebarVisible) return null;

  return (
    <aside className="fixed right-0 top-16 bottom-8 z-40 w-80 bg-secondary-bg border-l border-border-primary flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border-primary">
        <h2 className="font-semibold text-text-primary">Context Panel</h2>
        <button
          onClick={toggleRightSidebar}
          className="p-2 rounded-button hover:bg-surface-hover transition-colors"
        >
          <Settings className="w-4 h-4 text-text-secondary" />
        </button>
      </div>

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
            <div className="grid grid-cols-3 gap-3">
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
            </div>

            {/* Performance Metrics */}
            <div className="mt-6">
              <h4 className="font-medium text-text-primary mb-3">Performance</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-secondary">Response Time</span>
                  <span className="text-sm text-text-primary">1.2s avg</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-secondary">Model Load</span>
                  <span className={`text-sm ${modelStatus.isLoaded ? 'text-success' : 'text-warning'}`}>
                    {modelStatus.isLoaded ? 'Ready' : 'Loading'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-secondary">Memory Usage</span>
                  <span className="text-sm text-text-primary">2.1GB</span>
                </div>
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
              <div>
                <p className="text-text-tertiary">Model</p>
                <p className="text-text-primary">{modelStatus.modelName}</p>
              </div>
              <div>
                <p className="text-text-tertiary">Backend</p>
                <p className="text-text-primary">{modelStatus.backend}</p>
              </div>
            </div>

            {/* Active Users */}
            <div className="mt-6">
              <h4 className="font-medium text-text-primary mb-3">Active Users</h4>
              <div className="flex items-center space-x-2 p-3 bg-surface rounded-card border border-border-primary">
                <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center">
                  <span className="text-white text-sm">U</span>
                </div>
                <div>
                  <p className="text-sm text-text-primary">User</p>
                  <p className="text-xs text-text-tertiary">Online</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {uiState.activeTab === 'files' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Recent Files</h3>
            <div className="space-y-2">
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <div className="flex items-center space-x-2">
                  <FileText className="w-4 h-4 text-accent" />
                  <span className="text-sm text-text-primary">document.pdf</span>
                </div>
                <p className="text-xs text-text-tertiary mt-1">2 minutes ago</p>
              </div>
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <div className="flex items-center space-x-2">
                  <FileText className="w-4 h-4 text-accent" />
                  <span className="text-sm text-text-primary">image.png</span>
                </div>
                <p className="text-xs text-text-tertiary mt-1">5 minutes ago</p>
              </div>
            </div>
            <button className="w-full p-2 bg-accent hover:bg-accent-hover text-white rounded-button text-sm transition-colors">
              Upload File
            </button>
          </div>
        )}

        {uiState.activeTab === 'knowledge' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Knowledge Base</h3>
            <div className="space-y-3">
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Local Documents</span>
                  <span className="text-xs text-success">Connected</span>
                </div>
              </div>
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Web Search</span>
                  <span className="text-xs text-success">Available</span>
                </div>
              </div>
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Code Context</span>
                  <span className="text-xs text-success">Active</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {uiState.activeTab === 'tasks' && (
          <div className="space-y-4">
            <h3 className="font-semibold text-text-primary">Active Tasks</h3>
            <div className="space-y-2">
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Process audio input</span>
                  <CheckSquare className="w-4 h-4 text-success" />
                </div>
              </div>
              <div className="p-3 bg-surface rounded-card border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Analyze image</span>
                  <Activity className="w-4 h-4 text-warning animate-pulse" />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};

// Professional Status Bar
const ProfessionalStatusBar: React.FC = () => {
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const isRecording = useTektraStore((state) => state.isRecording);
  const avatarState = useTektraStore((state) => state.avatarState);
  const sessionState = useTektraStore((state) => state.sessionState);

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-between px-4 text-xs">
      <div className="flex items-center space-x-4">
        <span className="text-text-tertiary">Tektra AI Assistant</span>
        
        {/* Connection Status */}
        <div className="flex items-center space-x-2">
          <Wifi className="w-3 h-3 text-success" />
          <span className="text-text-secondary">Connected</span>
        </div>

        {/* Model Status */}
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            modelStatus.isLoaded ? 'bg-success' : 'bg-warning'
          }`}></div>
          <span className="text-text-secondary">
            {modelStatus.modelName} - {modelStatus.isLoaded ? 'Ready' : 'Loading...'}
          </span>
        </div>

        {/* Recording Indicator */}
        {isRecording && (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-error animate-pulse"></div>
            <span className="text-error">Recording</span>
          </div>
        )}

        {/* Avatar Status */}
        {avatarState.isSpeaking && (
          <div className="flex items-center space-x-2">
            <Volume2 className="w-3 h-3 text-accent" />
            <span className="text-accent">Speaking</span>
          </div>
        )}
      </div>

      {/* Right Side Info */}
      <div className="flex items-center space-x-4">
        <span className="text-text-tertiary">
          Tokens: {sessionState.tokenUsage}
        </span>
        <span className="text-text-tertiary">
          Press / for shortcuts
        </span>
      </div>
    </div>
  );
};

// Enhanced Loading Progress Component with File-by-File Tracking
const LoadingProgress: React.FC<{ 
  progress: number; 
  status: string; 
  visible: boolean 
}> = ({ progress, status, visible }) => {
  const [fileDownloads, setFileDownloads] = useState<Map<string, {
    name: string;
    completedMB: number;
    totalMB: number;
    progress: number;
    isCompleted: boolean;
  }>>(new Map());
  const [currentStep, setCurrentStep] = useState('');
  const [totalFiles, setTotalFiles] = useState(0);
  const [completedFiles, setCompletedFiles] = useState(0);

  useEffect(() => {
    // Parse status messages to extract file download information
    const statusLower = status.toLowerCase();
    
    if (status.includes('📋')) {
      setCurrentStep('manifest');
    } else if (status.includes('📦')) {
      setCurrentStep('downloading');
      // Extract file count: "📦 Downloading Layer abc123 (file 3/7)"
      const fileCountMatch = status.match(/file (\d+)\/(\d+)/);
      if (fileCountMatch) {
        setTotalFiles(parseInt(fileCountMatch[2]));
      }
    } else if (status.includes('⬇️')) {
      setCurrentStep('downloading');
      // Parse download progress: "⬇️ Layer abc123 • 45.2 MB / 127.8 MB (35.4%)"
      const downloadMatch = status.match(/⬇️\s+(Layer\s+\w+|[^•]+)\s+•\s+([\d.]+)\s+MB\s+\/\s+([\d.]+)\s+MB\s+\(([\d.]+)%\)/);
      if (downloadMatch) {
        const [, layerName, completedMB, totalMB, fileProgress] = downloadMatch;
        const layerId = layerName.replace('Layer ', '').trim();
        
        setFileDownloads(prev => {
          const newMap = new Map(prev);
          newMap.set(layerId, {
            name: layerName,
            completedMB: parseFloat(completedMB),
            totalMB: parseFloat(totalMB),
            progress: parseFloat(fileProgress),
            isCompleted: false
          });
          return newMap;
        });
      }
    } else if (status.includes('✅') && status.includes('Completed layer')) {
      // Mark file as completed: "✅ Completed layer abc123 (127.8 MB) • 3/7 files done"
      const completedMatch = status.match(/✅\s+Completed layer\s+(\w+).*?(\d+)\/(\d+)\s+files done/);
      if (completedMatch) {
        const [, layerId, completed, total] = completedMatch;
        setCompletedFiles(parseInt(completed));
        setTotalFiles(parseInt(total));
        
        setFileDownloads(prev => {
          const newMap = new Map(prev);
          if (newMap.has(layerId)) {
            const file = newMap.get(layerId)!;
            newMap.set(layerId, {
              ...file,
              progress: 100,
              isCompleted: true
            });
          }
          return newMap;
        });
      }
    } else if (status.includes('🔍')) {
      setCurrentStep('verifying');
    } else if (status.includes('📝')) {
      setCurrentStep('installing');
    } else if (status.includes('🎉')) {
      setCurrentStep('complete');
    }
  }, [status]);

  if (!visible) return null;

  const downloadEntries = Array.from(fileDownloads.entries());
  const isDownloading = currentStep === 'downloading' && downloadEntries.length > 0;

  return (
    <div className="fixed inset-0 bg-primary-bg/90 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-surface border border-border-primary rounded-lg p-8 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="text-center mb-6">
          <h3 className="text-xl font-semibold text-text-primary mb-2">Setting up Tektra AI</h3>
          <p className="text-text-secondary text-sm leading-relaxed">{status}</p>
        </div>
        
        {/* Main Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-text-secondary mb-2">
            <span>Setup Progress</span>
            <span className="font-medium">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-surface-hover rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-accent to-accent-hover h-3 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
              style={{ width: `${progress}%` }}
            >
              {progress > 10 && (
                <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
              )}
            </div>
          </div>
        </div>

        {/* File Download Progress (shown during model download) */}
        {isDownloading && downloadEntries.length > 0 && (
          <div className="mb-6 bg-surface-hover/30 rounded-lg p-4">
            <div className="flex justify-between items-center mb-3">
              <h4 className="text-sm font-medium text-text-primary">Model Files</h4>
              {totalFiles > 0 && (
                <span className="text-xs text-text-secondary">
                  {completedFiles}/{totalFiles} completed
                </span>
              )}
            </div>
            
            <div className="space-y-3 max-h-48 overflow-y-auto">
              {downloadEntries.map(([layerId, file]) => (
                <div key={layerId} className="bg-surface rounded-md p-3">
                  <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center space-x-2">
                      {file.isCompleted ? (
                        <div className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center">
                          <span className="text-white text-xs">✓</span>
                        </div>
                      ) : (
                        <div className="w-4 h-4 rounded-full border-2 border-accent animate-spin border-t-transparent"></div>
                      )}
                      <span className="text-sm font-medium text-text-primary">
                        {file.name}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-text-secondary">
                        {file.completedMB.toFixed(1)} / {file.totalMB.toFixed(1)} MB
                      </div>
                      <div className="text-xs font-medium text-text-primary">
                        {file.progress.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  
                  {/* Individual file progress bar */}
                  <div className="w-full bg-surface-hover rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        file.isCompleted 
                          ? 'bg-green-500' 
                          : 'bg-gradient-to-r from-blue-500 to-blue-600'
                      }`}
                      style={{ width: `${file.progress}%` }}
                    >
                      {!file.isCompleted && (
                        <div className="h-full bg-white/30 animate-pulse rounded-full"></div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Setup Steps */}
        <div className="space-y-3 mb-6">
          <div className={`flex items-center space-x-3 text-sm ${progress >= 50 ? 'text-text-primary' : 'text-text-tertiary'}`}>
            <div className={`w-2 h-2 rounded-full ${progress >= 50 ? 'bg-accent' : 'bg-surface-hover'}`}></div>
            <span>Tektra Components</span>
            {progress >= 50 && <span className="text-accent">✓</span>}
          </div>
          <div className={`flex items-center space-x-3 text-sm ${progress >= 70 ? 'text-text-primary' : 'text-text-tertiary'}`}>
            <div className={`w-2 h-2 rounded-full ${progress >= 70 ? 'bg-accent' : 'bg-surface-hover'}`}></div>
            <span>Speech Recognition</span>
            {progress >= 70 && <span className="text-accent">✓</span>}
          </div>
          <div className={`flex items-center space-x-3 text-sm ${progress >= 100 ? 'text-text-primary' : 'text-text-tertiary'}`}>
            <div className={`w-2 h-2 rounded-full ${progress >= 100 ? 'bg-accent' : 'bg-surface-hover'} ${progress >= 80 && progress < 100 ? 'animate-pulse' : ''}`}></div>
            <span>AI Model (Gemma3n:e4b)</span>
            {progress >= 100 && <span className="text-accent">✓</span>}
            {progress >= 80 && progress < 100 && <span className="text-warning">Downloading...</span>}
          </div>
        </div>

        {/* Loading Animation */}
        <div className="flex items-center justify-center space-x-2">
          <div className="w-2 h-2 bg-accent rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
          <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        </div>
        
        {progress >= 80 && progress < 100 && !isDownloading && (
          <div className="mt-4 text-center">
            <p className="text-xs text-text-tertiary">
              First-time setup requires downloading AI models (~2-4GB)
              <br />
              This may take several minutes depending on your internet connection
            </p>
          </div>
        )}

        {isDownloading && (
          <div className="mt-4 text-center">
            <p className="text-xs text-text-tertiary">
              Downloading model layers - Each file contains part of the AI model
              <br />
              Progress is saved automatically, you can safely close and reopen the app
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

const CompleteAppContent: React.FC = () => {
  // Use individual selectors
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);
  const setModelStatus = useTektraStore((state) => state.setModelStatus);
  const addMessage = useTektraStore((state) => state.addMessage);
  const addNotification = useTektraStore((state) => state.addNotification);
  const setRecording = useTektraStore((state) => state.setRecording);
  const setAvatarSpeaking = useTektraStore((state) => state.setAvatarSpeaking);
  const setAvatarListening = useTektraStore((state) => state.setAvatarListening);

  // Local state for UI
  const [isTyping, setIsTyping] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingStatus, setLoadingStatus] = useState('Initializing Tektra AI...');
  const [showLoading, setShowLoading] = useState(true);

  useEffect(() => {
    initializeApp();
    setupEventListeners();
  }, []);

  const waitForTauriReady = async (): Promise<boolean> => {
    console.log('🔍 Starting Tauri detection...');
    console.log('Window location:', window.location.href);
    console.log('Protocol:', window.location.protocol);
    console.log('Hostname:', window.location.hostname);
    console.log('Port:', window.location.port);
    
    // FORCE TAURI DETECTION - We know we're running cargo tauri dev
    if (window.location.hostname === 'localhost' && window.location.port === '1420') {
      console.log('🚀 FORCED: Detected Tauri dev server on localhost:1420 - this IS a Tauri app!');
      return true;
    }
    
    // Check for Tauri file protocol
    if (window.location.protocol === 'tauri:') {
      console.log('🚀 DETECTED: Tauri protocol - this IS a Tauri app!');
      return true;
    }
    
    // Check for non-HTTP protocols (Tauri uses custom protocols)
    const isWebBrowser = window.location.protocol === 'http:' || 
                        window.location.protocol === 'https:';
    
    if (!isWebBrowser) {
      console.log('🚀 DETECTED: Non-HTTP protocol - assuming Tauri app!');
      return true;
    }
    
    // Quick check for Tauri globals
    const maxAttempts = 10; // Reduce wait time
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      // Check for any Tauri indicators
      if (typeof (window as any).__TAURI_IPC__ === 'function' ||
          typeof (window as any).__TAURI__ !== 'undefined' ||
          typeof (window as any).__TAURI_INVOKE__ !== 'undefined') {
        console.log('🚀 DETECTED: Tauri globals found!');
        return true;
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }
    
    // FINAL OVERRIDE: Since we're running cargo tauri dev, force detection
    console.log('⚠️  Tauri globals not found, but since we are running cargo tauri dev, FORCING Tauri detection!');
    return true;
  };

  const initializeApp = async () => {
    try {
      setLoadingProgress(10);
      setLoadingStatus('Initializing Tektra components...');
      
      // Wait for Tauri to be fully initialized
      const isTauriApp = await waitForTauriReady();
      setLoadingProgress(25);
      
      if (!isTauriApp) {
        // Fallback: we might be in dev mode or Tauri isn't available
        console.log('Tektra backend not available - running in fallback mode');
        setLoadingProgress(100);
        setLoadingStatus('Demo mode active');
        setModelStatus({ isLoading: false, modelName: 'Demo Mode' });
        addMessage({
          role: 'system',
          content: '⚠️ Running in demo mode - Tektra backend not available'
        });
        setTimeout(() => setShowLoading(false), 1000);
        return;
      }

      setLoadingProgress(30);
      setLoadingStatus('Tektra backend connected! Setting up AI services...');
      setModelStatus({ isLoading: true, modelName: 'gemma3n:e4b', backend: 'Ollama' });
      
      // Start model initialization immediately but keep loading screen
      initializeModelsInBackground();
      
    } catch (error) {
      console.error('App initialization error:', error);
      setLoadingProgress(100);
      setLoadingStatus('Initialization failed');
      setModelStatus({ isLoading: false, modelName: 'gemma3n:e4b' });
      addMessage({
        role: 'system',
        content: `❌ App initialization failed: ${error}`
      });
      setTimeout(() => setShowLoading(false), 2000);
    }
  };

  const initializeModelsInBackground = async () => {
    try {
      setLoadingProgress(35);
      setLoadingStatus('Initializing speech recognition...');
      
      // Initialize Whisper first (faster)
      try {
        await invoke<boolean>('initialize_whisper');
        setLoadingProgress(45);
        setLoadingStatus('✅ Speech recognition ready');
        addMessage({
          role: 'system',
          content: '✅ Speech recognition ready'
        });
      } catch (whisperError) {
        console.warn('Whisper initialization failed:', whisperError);
        setLoadingProgress(45);
        setLoadingStatus('Speech recognition unavailable - continuing...');
        addMessage({
          role: 'system',
          content: '⚠️ Speech recognition unavailable - continuing without it'
        });
      }
      
      // Initialize AI model (this may take a long time for downloads)
      setLoadingProgress(50);
      setLoadingStatus('Initializing AI model - downloading if needed...');
      
      // Note: Backend logs show model loads in ~5 seconds, set reasonable timeout
      
      // Set up a timeout to prevent hanging at 100%
      let modelLoadTimeout: NodeJS.Timeout | null = null;
      const timeoutPromise = new Promise((_, reject) => {
        modelLoadTimeout = setTimeout(() => {
          reject(new Error('Model loading timed out after 10 minutes'));
        }, 600000); // 10 minutes
      });
      
      try {
        const modelLoaded = await Promise.race([
          invoke<boolean>('initialize_model'),
          timeoutPromise
        ]);
        
        if (modelLoadTimeout) clearTimeout(modelLoadTimeout);
        
        if (modelLoaded) {
          // Don't set to 100% immediately - let the progress events handle it
          setLoadingStatus('✅ All components ready!');
          setModelStatus({ isLoaded: true, isLoading: false, modelName: 'gemma3n:e4b', backend: 'Ollama' });
          
          addMessage({
            role: 'system',
            content: '✅ Gemma3n:e4b model loaded successfully! Multimodal AI capabilities are now ready.'
          });
          addNotification({
            type: 'success',
            message: 'AI model ready'
          });
          
          // Auto-hide loading screen if no completion event comes
          setTimeout(() => {
            if (showLoading) {
              console.log('⏰ Auto-hiding loading screen after timeout');
              setLoadingProgress(100);
              setLoadingStatus('✅ Model loaded successfully!');
              setModelStatus({ isLoaded: true, isLoading: false, modelName: 'gemma3n:e4b', backend: 'Ollama' });
              setShowLoading(false);
            }
          }, 3000); // Shorter timeout since model loads quickly
        } else {
          throw new Error('Model failed to load - check if gemma3n:e4b is available in Ollama');
        }
      } catch (timeoutError) {
        if (modelLoadTimeout) clearTimeout(modelLoadTimeout);
        throw timeoutError;
      }
      
    } catch (error) {
      console.error('Model initialization error:', error);
      setLoadingProgress(100);
      setLoadingStatus('Setup failed - continuing in limited mode');
      setModelStatus({ isLoading: false, modelName: 'gemma3n:e4b', backend: 'Ollama' });
      
      addMessage({
        role: 'system',
        content: `❌ Failed to load gemma3n:e4b model: ${error}. You can still use the app interface, but AI features won't be available.`
      });
      addNotification({
        type: 'error',
        message: 'Model initialization failed'
      });
      
      // Hide loading screen after error
      setTimeout(() => setShowLoading(false), 3000);
    }
  };

  const setupEventListeners = async () => {
    try {
      // Wait for Tauri to be available before setting up listeners
      const maxRetries = 10;
      let retries = 0;
      
      while (retries < maxRetries) {
        try {
          const { listen } = await import('@tauri-apps/api/event');
          
          // Listen for AI responses
          await listen('ai-response', (event: any) => {
            addMessage({
              role: 'assistant',
              content: event.payload.content || event.payload.message || event.payload
            });
            setIsTyping(false);
            setAvatarSpeaking(false);
          });

          // Listen for model loading progress
          await listen('model-loading-progress', (event: any) => {
            const { progress, status, model_name } = event.payload;
            
            // Show progress in the loading screen if still visible
            if (showLoading) {
              // Ensure progress stays within 50-95% range during download
              const adjustedProgress = Math.min(95, Math.max(50, progress));
              setLoadingProgress(adjustedProgress);
              setLoadingStatus(status);
            }
            
            // Also add chat message for transparency
            addMessage({
              role: 'system',
              content: `📥 ${status} (${Math.round(progress)}%)`
            });
          });

          // Listen for model loading completion
          await listen('model-loading-complete', (event: any) => {
            const { success, error, model_name } = event.payload;
            if (success) {
              setLoadingProgress(100);
              setLoadingStatus('✅ All components ready!');
              setModelStatus({ isLoaded: true, isLoading: false, modelName: model_name, backend: 'Ollama' });
              addMessage({
                role: 'system',
                content: `✅ ${model_name} model ready! You can now chat with the AI.`
              });
              
              // Hide loading screen after success with shorter delay
              setTimeout(() => setShowLoading(false), 1500);
            } else {
              setLoadingProgress(100);
              setLoadingStatus('Setup failed - continuing in limited mode');
              setModelStatus({ isLoading: false, modelName: model_name, backend: 'Ollama' });
              addMessage({
                role: 'system',
                content: `❌ Model loading failed: ${error || 'Unknown error'}`
              });
              
              // Hide loading screen after error
              setTimeout(() => setShowLoading(false), 3000);
            }
          });
          
          break; // Success, exit retry loop
        } catch (error) {
          console.log(`Event listener setup attempt ${retries + 1} failed:`, error);
          retries++;
          if (retries < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 500));
          }
        }
      }
      
      if (retries >= maxRetries) {
        console.log('Could not set up event listeners - continuing without them');
        return;
      }

      // Listen for transcription results
      await listen('transcription-result', (event: any) => {
        addMessage({
          role: 'user',
          content: event.payload.text || event.payload
        });
      });

      // Listen for audio recording events
      await listen('audio-recording-started', () => {
        setRecording(true);
        setAvatarListening(true);
      });

      await listen('audio-recording-stopped', () => {
        setRecording(false);
      });

      // Listen for camera events
      await listen('camera-initialized', () => {
        const setModelStatus = useTektraStore.getState().setModelStatus;
        setModelStatus({ cameraEnabled: true });
        addMessage({
          role: 'system',
          content: '📷 Camera initialized and ready'
        });
      });

      await listen('camera-frame-captured', (event: any) => {
        // Handle camera frame if needed
        console.log('Camera frame captured:', event.payload);
      });

      // Listen for model loading progress
      await listen('model-loading-progress', (event: any) => {
        addMessage({
          role: 'system',
          content: `📊 ${event.payload.status || event.payload}`
        });
      });

      // Listen for errors
      await listen('error', (event: any) => {
        addMessage({
          role: 'system',
          content: `❌ Error: ${event.payload.message || event.payload}`
        });
      });

    } catch (error) {
      console.log('Event listener setup failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-primary-bg text-text-primary overflow-hidden">
      {/* Loading Progress Overlay */}
      <LoadingProgress 
        progress={loadingProgress}
        status={loadingStatus}
        visible={showLoading}
      />

      {/* Header Bar */}
      <SimpleHeaderBar />

      {/* Main Layout */}
      <div className="flex pt-16 pb-8 h-screen">
        {/* Left Sidebar */}
        <LeftSidebar />

        {/* Main Content Area */}
        <main className={`
          flex-1 flex overflow-hidden transition-all duration-300
          ${uiState.leftSidebarCollapsed ? 'ml-16' : 'ml-80'}
          ${uiState.rightSidebarVisible ? 'mr-80' : 'mr-0'}
        `}>
          {/* Avatar Panel */}
          <div className="w-80 flex flex-col border-r border-border-primary bg-surface/20">
            <div className="p-4 border-b border-border-primary">
              <h3 className="font-semibold text-text-primary">AI Avatar</h3>
              <p className="text-sm text-text-secondary">3D interactive assistant</p>
            </div>
            <div className="flex-1">
              <Avatar3D />
            </div>
          </div>

          {/* Chat Interface */}
          <div className="flex-1 flex flex-col min-h-0">
            <EnhancedChatInterface />
          </div>
        </main>

        {/* Right Sidebar */}
        <CompleteRightSidebar />
      </div>

      {/* Status Bar */}
      <ProfessionalStatusBar />
    </div>
  );
};

const CompleteApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <CompleteAppContent />
    </QueryClientProvider>
  );
};

export default CompleteApp;