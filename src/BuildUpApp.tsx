import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SimpleHeaderBar from './components/SimpleHeaderBar';
import LeftSidebar from './components/LeftSidebar';
import { useTektraStore } from './store';
import { Send, Mic, Camera, MessageSquare } from 'lucide-react';
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

const BuildUpContent: React.FC = () => {
  // Use individual selectors - this is the only pattern that works
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);

  return (
    <div className="min-h-screen bg-primary-bg text-text-primary">
      {/* Professional Header */}
      <SimpleHeaderBar />

      {/* Layout with Sidebar */}
      <div className="flex pt-16 pb-8">
        {/* Left Sidebar */}
        <LeftSidebar />

        {/* Main content with proper padding */}
        <main className={`
          flex-1 flex flex-col transition-all duration-300
          ${uiState.leftSidebarCollapsed ? 'ml-16' : 'ml-80'}
        `}>
          {/* Chat Interface */}
          <div className="flex-1 flex flex-col">
            {/* Chat Header */}
            <div className="p-4 border-b border-border-primary">
              <h2 className="text-lg font-semibold text-text-primary">AI Assistant Chat</h2>
              <p className="text-sm text-text-secondary">Professional multimodal conversation interface</p>
            </div>

            {/* Chat Messages Area */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="max-w-3xl mx-auto space-y-4">
                {/* Welcome Message */}
                <div className="flex justify-start">
                  <div className="max-w-md p-4 bg-surface border border-border-primary rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <MessageSquare className="w-4 h-4 text-accent" />
                      <span className="text-sm font-medium text-text-primary">Tektra AI</span>
                    </div>
                    <p className="text-sm text-text-secondary">
                      Hello! I'm your AI assistant. I can help with text, voice, and vision tasks. 
                      The professional interface is now fully operational with HeaderBar + Sidebar working perfectly!
                    </p>
                    <p className="text-xs text-text-tertiary mt-2">Just now</p>
                  </div>
                </div>

                {/* System Status */}
                <div className="flex justify-center">
                  <div className="px-4 py-2 bg-surface/50 border border-border-primary rounded-full text-xs text-text-secondary">
                    ✅ Model: {modelStatus.modelName} • Status: {modelStatus.isLoaded ? 'Ready' : 'Loading'} • Interface: Active
                  </div>
                </div>
              </div>
            </div>

            {/* Chat Input */}
            <div className="border-t border-border-primary p-4">
              <div className="max-w-3xl mx-auto">
                <div className="flex items-center space-x-3">
                  {/* Voice Button */}
                  <button className="p-3 bg-surface border border-border-primary hover:bg-surface-hover rounded-full transition-colors">
                    <Mic className="w-5 h-5 text-text-secondary" />
                  </button>

                  {/* Camera Button */}
                  <button className="p-3 bg-surface border border-border-primary hover:bg-surface-hover rounded-full transition-colors">
                    <Camera className="w-5 h-5 text-text-secondary" />
                  </button>

                  {/* Text Input */}
                  <div className="flex-1">
                    <input
                      type="text"
                      placeholder="Type your message..."
                      className="w-full px-4 py-3 bg-surface border border-border-primary rounded-button text-text-primary placeholder-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent"
                    />
                  </div>

                  {/* Send Button */}
                  <button className="p-3 bg-accent hover:bg-accent-hover text-white rounded-full transition-colors">
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
          </div>
        </main>
      </div>
    </div>
  );
};

const BuildUpApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BuildUpContent />
    </QueryClientProvider>
  );
};

export default BuildUpApp;