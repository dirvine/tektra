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

const UltraMinimalContent: React.FC = () => {
  // Use individual selectors - this is the only pattern that works
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);

  return (
    <div className="min-h-screen bg-primary-bg text-text-primary p-8">
      {/* Ultra simple header - no Framer Motion */}
      <div className="fixed top-0 left-0 right-0 z-50 h-16 bg-secondary-bg border-b border-border-primary flex items-center justify-between px-6">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <span className="text-white font-bold text-sm">T</span>
          </div>
          <h1 className="text-lg font-semibold text-text-primary">Tektra</h1>
        </div>
        <div className="text-text-secondary">Ultra Minimal Test</div>
      </div>

      {/* Main content with top padding */}
      <div className="pt-20">
        <h1 className="text-4xl font-bold text-text-primary mb-6 text-center">
          Ultra Minimal Interface Test
        </h1>
        
        <div className="max-w-2xl mx-auto space-y-6">
          <div className="p-6 bg-surface rounded-card border border-border-primary">
            <h2 className="text-xl font-semibold text-text-primary mb-4">Store Data Test</h2>
            <div className="space-y-2 text-sm">
              <p className="text-text-secondary">Model: {modelStatus.modelName}</p>
              <p className="text-text-secondary">Backend: {modelStatus.backend}</p>
              <p className="text-text-secondary">Loaded: {modelStatus.isLoaded ? 'Yes' : 'No'}</p>
              <p className="text-text-secondary">Sidebar Collapsed: {uiState.leftSidebarCollapsed ? 'Yes' : 'No'}</p>
              <p className="text-text-secondary">Theme: {uiState.theme}</p>
            </div>
          </div>

          <div className="p-6 bg-gradient-to-r from-accent/20 to-accent-light/20 rounded-card border border-accent/30">
            <p className="text-accent font-medium text-center">
              ✅ If you can see this, React + Store + Tailwind are all working!
            </p>
          </div>

          <div className="text-center">
            <p className="text-text-tertiary text-sm">
              No Framer Motion • No Complex Components • Pure HTML + CSS + React
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const UltraMinimalApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <UltraMinimalContent />
    </QueryClientProvider>
  );
};

export default UltraMinimalApp;