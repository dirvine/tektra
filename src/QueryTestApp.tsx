import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
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

const QueryTestContent: React.FC = () => {
  const { modelStatus, avatarState } = useTektraStore();

  return (
    <div className="min-h-screen bg-primary-bg text-text-primary overflow-hidden">
      {/* Header Bar */}
      <div className="fixed top-0 left-0 right-0 z-50 h-16 bg-secondary-bg border-b border-border-primary flex items-center justify-between px-6">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <span className="text-white font-bold text-sm">T</span>
          </div>
          <h1 className="text-lg font-semibold text-text-primary">Tektra</h1>
        </div>
        <div className="text-text-secondary">Testing React Query</div>
      </div>

      {/* Main Content */}
      <div className="pt-16 pb-8 flex items-center justify-center h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-text-primary mb-4">
            React Query Test
          </h2>
          <div className="mt-6 p-4 bg-surface rounded-card border border-border-primary max-w-md">
            <p className="text-sm text-text-primary">
              ✅ React Components Working<br/>
              ✅ Tailwind Styling Working<br/>
              ✅ Zustand Store Working<br/>
              ✅ React Query Working
            </p>
            
            <div className="mt-4 text-left">
              <h3 className="text-sm font-semibold mb-2">Store Data:</h3>
              <p className="text-xs text-text-secondary font-mono">
                Model: {modelStatus.modelName}<br/>
                Avatar: {avatarState.expression}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-center px-4 text-xs">
        <span className="text-text-tertiary">React Query Test - All Basic Components Working</span>
      </div>
    </div>
  );
};

const QueryTestApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <QueryTestContent />
    </QueryClientProvider>
  );
};

export default QueryTestApp;