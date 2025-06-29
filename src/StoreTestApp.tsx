import React from 'react';
import { useTektraStore } from './store';
import './App.css';

const StoreTestApp: React.FC = () => {
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
        <div className="text-text-secondary">Testing Store</div>
      </div>

      {/* Main Content */}
      <div className="pt-16 pb-8 flex items-center justify-center h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-text-primary mb-4">
            Store Test
          </h2>
          <div className="mt-6 p-4 bg-surface rounded-card border border-border-primary max-w-md">
            <h3 className="text-lg font-semibold mb-2">Model Status:</h3>
            <p className="text-sm text-text-secondary">
              Loaded: {modelStatus.isLoaded ? '✅' : '❌'}<br/>
              Model: {modelStatus.modelName}<br/>
              Backend: {modelStatus.backend}
            </p>
            
            <h3 className="text-lg font-semibold mb-2 mt-4">Avatar State:</h3>
            <p className="text-sm text-text-secondary">
              Visible: {avatarState.isVisible ? '✅' : '❌'}<br/>
              Expression: {avatarState.expression}<br/>
              Speaking: {avatarState.isSpeaking ? '✅' : '❌'}
            </p>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-center px-4 text-xs">
        <span className="text-text-tertiary">Store Test - Zustand Working</span>
      </div>
    </div>
  );
};

export default StoreTestApp;