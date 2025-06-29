import React, { useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import HeaderBar from './components/HeaderBar';
import LeftSidebar from './components/LeftSidebar';
import RightSidebar from './components/RightSidebar';
import EnhancedChatInterface from './components/EnhancedChatInterface';
import Avatar3D from './components/Avatar3D';
import StatusBar from './components/StatusBar';
import { useTektraStore } from './store';
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

const AppContent: React.FC = () => {
  // Use individual selectors instead of destructuring - this is more reliable
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);
  const setModelStatus = useTektraStore((state) => state.setModelStatus);
  const addMessage = useTektraStore((state) => state.addMessage);
  const setAvatarSpeaking = useTektraStore((state) => state.setAvatarSpeaking);
  const setAvatarListening = useTektraStore((state) => state.setAvatarListening);
  const addNotification = useTektraStore((state) => state.addNotification);

  useEffect(() => {
    initializeApp();
    setupEventListeners();
  }, []);

  const initializeApp = async () => {
    try {
      setModelStatus({ isLoading: true });
      
      // Initialize AI model
      const modelLoaded = await invoke<boolean>('initialize_model');
      if (modelLoaded) {
        setModelStatus({ 
          isLoaded: true, 
          isLoading: false 
        });
        
        addMessage({
          role: 'system',
          content: 'AI model loaded successfully'
        });
        
        addNotification({
          type: 'success',
          message: 'AI model ready'
        });
        
        // Initialize Whisper
        try {
          const whisperLoaded = await invoke<boolean>('initialize_whisper');
          setModelStatus({ 
            whisperReady: whisperLoaded 
          });
          
          if (whisperLoaded) {
            addMessage({
              role: 'system',
              content: 'Whisper speech-to-text ready'
            });
          } else {
            addMessage({
              role: 'system',
              content: '⚠️ Whisper initialization failed'
            });
          }
        } catch (error) {
          console.error('Whisper initialization error:', error);
          addMessage({
            role: 'system',
            content: `⚠️ Whisper error: ${error}`
          });
        }
      }
    } catch (error) {
      console.error('Model initialization error:', error);
      setModelStatus({ isLoading: false });
      
      // Provide specific error messaging for common issues
      const errorMessage = String(error);
      if (errorMessage.includes('Ollama') || errorMessage.includes('connection') || errorMessage.includes('download')) {
        addMessage({
          role: 'system',
          content: `🔧 Setting up AI backend...\n\nTektra is downloading and installing the necessary AI components. This may take a few minutes on first run.\n\nError details: ${error}`
        });
        
        addNotification({
          type: 'info',
          message: 'Downloading AI components - please wait...'
        });
      } else {
        addMessage({
          role: 'system',
          content: `❌ Failed to load AI model: ${error}`
        });
        
        addNotification({
          type: 'error',
          message: 'Failed to initialize AI model'
        });
      }
    }
  };

  const setupEventListeners = async () => {
    // Listen for speech transcription
    await listen<{ text: string }>('speech-transcribed', (event) => {
      const { text } = event.payload;
      if (text.trim()) {
        addMessage({
          role: 'user',
          content: text
        });
        // Auto-send transcribed message
        // sendMessage(text);
      }
    });

    // Listen for model loading progress
    await listen<{ progress: number; status: string; model_name: string }>('model-loading-progress', (event) => {
      const { status } = event.payload;
      addMessage({
        role: 'system',
        content: status
      });
    });

    // Listen for avatar state changes
    await listen<{ speaking: boolean }>('avatar-speaking', (event) => {
      setAvatarSpeaking(event.payload.speaking);
    });

    await listen<{ listening: boolean }>('avatar-listening', (event) => {
      setAvatarListening(event.payload.listening);
    });
  };

  return (
    <div className="min-h-screen bg-primary-bg text-text-primary overflow-hidden">
      {/* Header Bar */}
      <HeaderBar />

      {/* Main Layout */}
      <div className="flex pt-16 pb-8">
        {/* Left Sidebar */}
        <LeftSidebar />

        {/* Main Content Area */}
        <main className={`
          flex-1 flex flex-col overflow-hidden transition-all duration-300
          ${uiState.leftSidebarCollapsed ? 'ml-16' : 'ml-80'}
          ${uiState.rightSidebarVisible ? 'mr-80' : 'mr-0'}
        `}>
          {/* Avatar Container */}
          <div className="p-4 pb-0">
            <Avatar3D />
          </div>

          {/* Chat Interface */}
          <div className="flex-1 p-4 pt-0">
            <EnhancedChatInterface className="h-full" />
          </div>
        </main>

        {/* Right Sidebar */}
        <RightSidebar />
      </div>

      {/* Status Bar */}
      <StatusBar />
    </div>
  );
};

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
};

export default App;