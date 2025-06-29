import React, { useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import HeaderBar from './components/HeaderBar';
import LeftSidebar from './components/LeftSidebar';
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

const ProgressiveAppContent: React.FC = () => {
  // Use individual selectors instead of destructuring - this is more reliable
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);
  const setModelStatus = useTektraStore((state) => state.setModelStatus);
  const addMessage = useTektraStore((state) => state.addMessage);
  const addNotification = useTektraStore((state) => state.addNotification);

  useEffect(() => {
    const initializeApp = async () => {
      try {
        setModelStatus({ isLoading: true });
        
        // Simulate model loading
        setTimeout(() => {
          setModelStatus({ 
            isLoaded: true, 
            isLoading: false 
          });
          
          addMessage({
            role: 'system',
            content: 'Professional AI Assistant interface loaded successfully'
          });
          
          addNotification({
            type: 'success',
            message: 'System ready'
          });
        }, 1000);
        
      } catch (error) {
        console.error('Model initialization error:', error);
        setModelStatus({ isLoading: false });
        addMessage({
          role: 'system',
          content: `❌ Failed to load AI model: ${error}`
        });
      }
    };

    initializeApp();
  }, []);

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
          <div className="flex-1 p-8 flex items-center justify-center">
            <div className="text-center max-w-2xl">
              <h1 className="text-4xl font-bold text-text-primary mb-6">
                Tektra AI Assistant
              </h1>
              <p className="text-lg text-text-secondary mb-8">
                Professional multimodal AI assistant with voice, vision, and action understanding
              </p>
              
              <div className="grid grid-cols-2 gap-6 mb-8">
                <div className="p-6 bg-surface rounded-card border border-border-primary">
                  <h3 className="text-lg font-semibold text-text-primary mb-3">
                    ✅ Core Systems
                  </h3>
                  <div className="text-left space-y-2 text-sm text-text-secondary">
                    <p>• React + TypeScript Interface</p>
                    <p>• Zustand State Management</p>
                    <p>• Tailwind Design System</p>
                    <p>• Professional Layout</p>
                  </div>
                </div>
                
                <div className="p-6 bg-surface rounded-card border border-border-primary">
                  <h3 className="text-lg font-semibold text-text-primary mb-3">
                    🔧 Backend Integration
                  </h3>
                  <div className="text-left space-y-2 text-sm text-text-secondary">
                    <p>• Model: {modelStatus.modelName}</p>
                    <p>• Backend: {modelStatus.backend}</p>
                    <p>• Status: {modelStatus.isLoaded ? 'Ready' : 'Loading...'}</p>
                    <p>• Whisper: {modelStatus.whisperReady ? 'Ready' : 'Disabled'}</p>
                  </div>
                </div>
              </div>

              <div className="p-6 bg-gradient-to-r from-accent/10 to-accent-light/10 rounded-card border border-accent/20">
                <p className="text-accent font-medium">
                  🎉 Professional interface successfully loaded!
                </p>
                <p className="text-text-tertiary text-sm mt-2">
                  HeaderBar + LeftSidebar working perfectly with store integration
                </p>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Simple Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-center px-4 text-xs">
        <span className="text-text-tertiary">
          Progressive Interface Test - HeaderBar + LeftSidebar + Main Content
        </span>
      </div>
    </div>
  );
};

const ProgressiveApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ProgressiveAppContent />
    </QueryClientProvider>
  );
};

export default ProgressiveApp;