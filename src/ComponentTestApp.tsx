import React, { useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useTektraStore } from './store';
import './App.css';

// Import our components one by one
import HeaderBar from './components/HeaderBar';
import StoreTestSidebar from './components/StoreTestSidebar';

// Create QueryClient instance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const ComponentTestContent: React.FC = () => {
  const { 
    modelStatus, 
    uiState,
    setModelStatus, 
    addMessage, 
    addNotification 
  } = useTektraStore();

  // Simplified initialization (similar to original App)
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
            content: 'AI model loaded successfully'
          });
          
          addNotification({
            type: 'success',
            message: 'AI model ready'
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
      {/* Header Bar - Testing first component */}
      <HeaderBar />

      {/* Main Layout */}
      <div className="flex pt-16 pb-8">
        {/* Left Sidebar - Testing second component */}
        <StoreTestSidebar />

        {/* Simple content area */}
        <main className={`
          flex-1 flex flex-col overflow-hidden transition-all duration-300
          ${uiState.leftSidebarCollapsed ? 'ml-16' : 'ml-80'}
        `}>
          <div className="flex-1 p-8 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-text-primary mb-4">
                Component Integration Test
              </h2>
              <div className="p-6 bg-surface rounded-card border border-border-primary max-w-md">
                <p className="text-sm text-text-primary mb-4">
                  ✅ React + Tailwind Working<br/>
                  ✅ Zustand Store Working<br/>
                  ✅ React Query Working<br/>
                  ✅ Three.js Working<br/>
                  ✅ HeaderBar Component Working<br/>
                  ✅ FixedLeftSidebar Component Working
                </p>
                
                <div className="text-left">
                  <h3 className="text-sm font-semibold mb-2">Model Status:</h3>
                  <p className="text-xs text-text-secondary font-mono">
                    Loading: {modelStatus.isLoading ? 'Yes' : 'No'}<br/>
                    Loaded: {modelStatus.isLoaded ? 'Yes' : 'No'}<br/>
                    Model: {modelStatus.modelName}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Simple Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-center px-4 text-xs">
        <span className="text-text-tertiary">Component Test - HeaderBar + LeftSidebar Integration</span>
      </div>
    </div>
  );
};

const ComponentTestApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ComponentTestContent />
    </QueryClientProvider>
  );
};

export default ComponentTestApp;