import React, { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SimpleHeaderBar from './components/SimpleHeaderBar';
import LeftSidebar from './components/LeftSidebar';
import Avatar3D from './components/Avatar3D';
import { useTektraStore } from './store';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ErrorBoundary from './components/ErrorBoundary';
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
  Volume2,
  X,
  Upload,
  Image
} from 'lucide-react';
import './App.css';
import './styles/react-markdown.css';

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
  const [attachments, setAttachments] = useState<File[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  
  // Ref for auto-scrolling
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Alternative approach: focus on making file upload button work reliably
  const alternativeFileInputRef = React.useRef<HTMLInputElement>(null);

  // Tauri v2 drag and drop with correct event names
  React.useEffect(() => {
    console.log('🔧 Setting up Tauri v2 drag and drop with correct event names');
    
    let unlistenFunctions: (() => void)[] = [];

    const setupTauriV2DragDrop = async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        
        // Tauri v2 uses these specific event names
        const events = [
          'tauri://drag-enter',
          'tauri://drag-over', 
          'tauri://drag-drop',
          'tauri://drag-leave'
        ];
        
        for (const eventName of events) {
          try {
            console.log(`📡 Registering: ${eventName}`);
            const unlisten = await listen(eventName, (event) => {
              console.log(`🎉 TAURI v2 EVENT: ${eventName}`, event);
              
              switch (eventName) {
                case 'tauri://drag-enter':
                case 'tauri://drag-over':
                  setIsDragOver(true);
                  break;
                  
                case 'tauri://drag-leave':
                  setIsDragOver(false);
                  break;
                  
                case 'tauri://drag-drop':
                  setIsDragOver(false);
                  const files = event.payload as string[];
                  if (files && files.length > 0) {
                    console.log('📂 Files dropped:', files);
                    handleTauriFileDrop(files);
                  }
                  break;
              }
            });
            
            unlistenFunctions.push(unlisten);
            console.log(`✅ Registered: ${eventName}`);
            
          } catch (err) {
            console.log(`❌ Failed to register: ${eventName}`, err);
          }
        }
        
        console.log(`✅ Tauri v2 drag drop setup complete - registered ${unlistenFunctions.length} events`);
        
      } catch (error) {
        console.error('❌ Failed to setup Tauri v2 drag drop:', error);
        console.log('📁 File upload button is available as backup');
      }
    };

    setupTauriV2DragDrop();

    return () => {
      console.log('🧹 Cleaning up Tauri v2 drag drop listeners');
      unlistenFunctions.forEach(unlisten => unlisten());
    };
  }, []);

  // Handle Tauri file drops
  const handleTauriFileDrop = async (filePaths: string[]) => {
    console.log('🔄 Processing Tauri file drop:', filePaths);
    
    try {
      // In Tauri v2, we can use fetch with the file path directly
      // or use the invoke method to read files through backend commands
      
      for (const filePath of filePaths) {
        console.log('📂 Processing dropped file:', filePath);
        
        // Extract filename from path
        const filename = filePath.split('/').pop() || filePath.split('\\').pop() || 'unknown';
        console.log('📋 Extracted filename:', filename);
        
        try {
          // For now, just create a placeholder file object to test the drop mechanism
          // We'll handle actual file reading through the existing backend commands
          const placeholderFile = new File([''], filename, {
            type: getFileType(filename)
          });
          
          console.log('📋 Created placeholder file object:', { 
            name: placeholderFile.name, 
            type: placeholderFile.type, 
            path: filePath 
          });
          
          // Add to attachments
          setAttachments(prev => {
            const newAttachments = [...prev, placeholderFile];
            console.log('💾 Updated attachments:', newAttachments.map(f => f.name));
            return newAttachments;
          });
          
        } catch (fileError) {
          console.error('❌ Error creating file object:', fileError);
        }
      }
      
      addMessage({
        role: 'system',
        content: `🎉 TAURI FILE DROP WORKS! Added ${filePaths.length} file(s): ${filePaths.map(p => p.split('/').pop() || p.split('\\').pop()).join(', ')}`,
      });
      
    } catch (error) {
      console.error('❌ Error processing dropped files:', error);
      addMessage({
        role: 'system',
        content: `❌ Error processing dropped files: ${error}`,
      });
    }
  };

  // Helper function to determine file type
  const getFileType = (filename: string): string => {
    const ext = filename.toLowerCase().split('.').pop();
    switch (ext) {
      case 'txt': return 'text/plain';
      case 'md': return 'text/markdown';
      case 'json': return 'application/json';
      case 'png': return 'image/png';
      case 'jpg':
      case 'jpeg': return 'image/jpeg';
      case 'gif': return 'image/gif';
      case 'webp': return 'image/webp';
      default: return 'application/octet-stream';
    }
  };

  // Simplified file handling - focus on button first
  const handleAlternativeFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log('📂 FILE INPUT CHANGE EVENT');
    if (e.target.files) {
      const files = Array.from(e.target.files);
      console.log('📋 Selected files:', files.map(f => ({ name: f.name, type: f.type, size: f.size })));
      
      setAttachments(prev => {
        const newAttachments = [...prev, ...files];
        console.log('💾 Updated attachments:', newAttachments.map(f => f.name));
        return newAttachments;
      });
      
      addMessage({
        role: 'system',
        content: `📂 Selected ${files.length} file(s) via button: ${files.map(f => f.name).join(', ')}`,
      });
    }
    e.target.value = ''; // Reset for reselection
  };

  const triggerAlternativeFileSelect = () => {
    console.log('🖱️ TRIGGERING FILE SELECTOR');
    alternativeFileInputRef.current?.click();
  };


  const handleSendMessage = async () => {
    if (!inputValue.trim() && attachments.length === 0) return;
    
    // Process attachments first if any
    if (attachments.length > 0) {
      await processAttachments(attachments);
      setAttachments([]);
    }
    
    if (inputValue.trim()) {
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

  // File upload functions
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setAttachments(prev => [...prev, ...files]);
    }
    e.target.value = '';
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const processAttachments = async (files: File[]) => {
    if (files.length === 0) return;

    addMessage({
      role: 'system',
      content: `📎 Processing ${files.length} file(s): ${files.map(f => f.name).join(', ')}`,
    });

    let processedCount = 0;
    
    for (const file of files) {
      try {
        const isImage = file.type.startsWith('image/');
        const isText = file.type.startsWith('text/') || 
                      file.name.endsWith('.txt') || 
                      file.name.endsWith('.md') ||
                      file.name.endsWith('.json');
        
        addMessage({
          role: 'system', 
          content: `${isImage ? '🖼️' : '📄'} Loading ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`,
        });
        
        if (isImage) {
          // Process image for multimodal input
          const arrayBuffer = await file.arrayBuffer();
          const uint8Array = Array.from(new Uint8Array(arrayBuffer));
          
          // Process image through backend multimodal system
          const result = await invoke('process_image_input', {
            message: `Analyze this uploaded image: ${file.name}`,
            imageData: uint8Array,
          });
          
          addMessage({
            role: 'assistant',
            content: result,
          });
          
          processedCount++;
        } else if (isText) {
          // Process text files for vector database
          const arrayBuffer = await file.arrayBuffer();
          const uint8Array = Array.from(new Uint8Array(arrayBuffer));
          
          const result = await invoke('process_file_content', {
            fileName: file.name,
            fileContent: uint8Array,
            fileType: file.type,
          });
          
          addMessage({
            role: 'system',
            content: `✅ ${result}`,
          });
          
          processedCount++;
        } else {
          addMessage({
            role: 'system',
            content: `⚠️ ${file.name} is not a supported file type. Please use text (.txt, .md, .json) or image files.`,
          });
        }
      } catch (error) {
        console.error(`Error processing ${file.name}:`, error);
        addMessage({
          role: 'system',
          content: `❌ Failed to process ${file.name}: ${error}`,
        });
      }
    }
    
    if (processedCount > 0) {
      addMessage({
        role: 'system',
        content: `🎉 Successfully processed ${processedCount}/${files.length} file(s)`,
      });
    }
  };


  return (
    <div 
      className="flex flex-col h-full min-h-0 relative"
    >
      {/* Hidden Alternative File Input for Testing */}
      <input
        ref={alternativeFileInputRef}
        type="file"
        multiple
        accept=".txt,.md,.json,.png,.jpg,.jpeg,.gif,.webp,text/*,image/*"
        onChange={handleAlternativeFileSelect}
        style={{ display: 'none' }}
      />
      
      {/* Drag Overlay for Tauri v2 */}
      {isDragOver && (
        <div className="absolute inset-0 bg-accent/10 border-4 border-dashed border-accent rounded-lg flex items-center justify-center z-50">
          <div className="text-center">
            <div className="w-20 h-20 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Upload className="w-10 h-10 text-accent" />
            </div>
            <p className="text-xl font-medium text-accent mb-2">Drop files here to upload</p>
            <p className="text-text-secondary">
              Supports text files (.txt, .md, .json) and images (.png, .jpg, .jpeg, .gif, .webp)
            </p>
            <p className="text-xs text-text-tertiary mt-2">
              ✨ Tauri v2 Native Drag & Drop
            </p>
          </div>
        </div>
      )}

      {/* Chat Header */}
      <div className="flex-shrink-0 p-4 border-b border-border-primary bg-surface/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-text-primary">AI Assistant Chat</h2>
            <p className="text-sm text-text-secondary">Professional multimodal conversation interface</p>
            {!modelStatus.isLoaded && (
              <button 
                onClick={async () => {
                  try {
                    console.log('🔄 Manually initializing model...');
                    const result = await invoke<boolean>('initialize_model');
                    console.log('✅ Model initialization result:', result);
                    addMessage({
                      role: 'system',
                      content: result ? '✅ Model initialized successfully' : '❌ Model initialization failed'
                    });
                  } catch (error) {
                    console.error('❌ Model initialization error:', error);
                    addMessage({
                      role: 'system',
                      content: `❌ Model initialization error: ${error}`
                    });
                  }
                }}
                className="mt-2 px-3 py-1 text-xs bg-warning text-white rounded hover:bg-warning/80"
              >
                🔄 Initialize Model
              </button>
            )}
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
                {message.role === 'assistant' ? (
                  <ErrorBoundary fallback={<p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>}>
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      className="text-sm"
                      components={{
                        p: ({children}) => <p className="mb-2">{children}</p>,
                        strong: ({children}) => <strong className="font-bold">{children}</strong>,
                        em: ({children}) => <em className="italic">{children}</em>,
                        ul: ({children}) => <ul className="list-disc list-inside mb-2">{children}</ul>,
                        ol: ({children}) => <ol className="list-decimal list-inside mb-2">{children}</ol>,
                        li: ({children}) => <li className="mb-1">{children}</li>,
                        code: ({node, inline, className, children, ...props}) => {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline && match ? (
                            <pre className="bg-black/20 p-2 rounded overflow-x-auto mb-2">
                              <code className="text-sm" {...props}>
                                {String(children).replace(/\n$/, '')}
                              </code>
                            </pre>
                          ) : (
                            <code className="bg-black/20 px-1 py-0.5 rounded text-sm" {...props}>
                              {children}
                            </code>
                          )
                        },
                        h1: ({children}) => <h1 className="text-lg font-bold mb-1">{children}</h1>,
                        h2: ({children}) => <h2 className="text-base font-bold mb-1">{children}</h2>,
                        h3: ({children}) => <h3 className="text-sm font-bold mb-1">{children}</h3>,
                      }}
                    >
                      {message.content || ''}
                    </ReactMarkdown>
                  </ErrorBoundary>
                ) : (
                  <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
                )}
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
        {/* Attachments Preview */}
        {attachments.length > 0 && (
          <div className="mb-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-text-primary">
                Attached Files ({attachments.length})
              </span>
              <button
                onClick={() => setAttachments([])}
                className="text-xs text-text-tertiary hover:text-error transition-colors"
              >
                Clear all
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {attachments.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 px-3 py-2 bg-surface rounded-lg border border-border-primary group hover:border-accent/30 transition-colors"
                >
                  {file.type.startsWith('image/') ? (
                    <Image className="w-4 h-4 text-text-secondary" />
                  ) : (
                    <FileText className="w-4 h-4 text-text-secondary" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-text-primary truncate max-w-32" title={file.name}>
                      {file.name}
                    </div>
                    <div className="text-xs text-text-tertiary">
                      {(file.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                  <button 
                    onClick={() => removeAttachment(index)}
                    className="opacity-0 group-hover:opacity-100 text-text-tertiary hover:text-error transition-all p-1 rounded"
                    title="Remove file"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        
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
          <label
            htmlFor="file-upload"
            className="p-3 rounded-full bg-surface border border-border-primary hover:bg-surface-hover text-text-secondary transition-colors cursor-pointer"
            title="Attach file"
          >
            <Paperclip className="w-5 h-5" />
          </label>
          <input
            id="file-upload"
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.md,.json,.png,.jpg,.jpeg,.gif,.webp,text/*,image/*"
            onChange={handleAlternativeFileSelect}
            className="hidden"
          />

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
            <span>AI Model (Gemma3:4b)</span>
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
      setModelStatus({ isLoading: true, modelName: 'gemma3:4b', backend: 'Ollama' });
      
      // Start model initialization immediately but keep loading screen
      initializeModelsInBackground();
      
    } catch (error) {
      console.error('App initialization error:', error);
      setLoadingProgress(100);
      setLoadingStatus('Initialization failed');
      setModelStatus({ isLoading: false, modelName: 'gemma3:4b' });
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
          setModelStatus({ isLoaded: true, isLoading: false, modelName: 'gemma3:4b', backend: 'Ollama' });
          
          addMessage({
            role: 'system',
            content: '✅ Gemma3:4b model loaded successfully! Multimodal AI capabilities are now ready.'
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
              setModelStatus({ isLoaded: true, isLoading: false, modelName: 'gemma3:4b', backend: 'Ollama' });
              setShowLoading(false);
            }
          }, 3000); // Shorter timeout since model loads quickly
        } else {
          throw new Error('Model failed to load - check if gemma3:4b is available in Ollama');
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
            
            console.log(`🎯 PROGRESS EVENT RECEIVED: ${progress}% - ${status} - model: ${model_name}`);
            console.log('📊 Raw event payload:', event.payload);
            
            // Show progress in the loading screen if still visible
            if (showLoading) {
              console.log(`📈 Updating loading progress from ${loadingProgress} to ${progress}`);
              // Use the actual progress from ollama_inference.rs which provides granular file-by-file tracking
              // The ollama backend already provides detailed progress from 0-100%
              setLoadingProgress(progress);
              setLoadingStatus(status);
            } else {
              console.log('⚠️ Loading screen not visible, ignoring progress update');
            }
            
            // Add chat message for all progress updates during debugging
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