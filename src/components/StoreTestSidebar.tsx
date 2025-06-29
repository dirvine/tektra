import React from 'react';
import {
  Brain,
  User,
  Mic,
  PanelLeftClose,
} from 'lucide-react';
import { useTektraStore } from '../store';

const StoreTestSidebar: React.FC = () => {
  // Test minimal store access
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const avatarState = useTektraStore((state) => state.avatarState);
  const uiState = useTektraStore((state) => state.uiState);

  return (
    <aside className="fixed left-0 top-16 bottom-8 z-40 w-80 bg-secondary-bg border-r border-border-primary flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border-primary">
        <h2 className="font-semibold text-text-primary">Configuration</h2>
        <button className="p-2 rounded-button hover:bg-surface-hover transition-colors">
          <PanelLeftClose className="w-4 h-4 text-text-secondary" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {/* AI Model Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 p-4">
          <div className="flex items-center space-x-3 mb-3">
            <Brain className="w-5 h-5 text-accent" />
            <span className="font-medium text-text-primary">AI Model</span>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-text-secondary">
              Model: {modelStatus.modelName}
            </p>
            <p className="text-sm text-text-secondary">
              Backend: {modelStatus.backend}
            </p>
            <p className="text-sm text-text-secondary">
              Status: {modelStatus.isLoaded ? 'Loaded' : 'Not Loaded'}
            </p>
          </div>
        </div>

        {/* Avatar Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 p-4">
          <div className="flex items-center space-x-3 mb-3">
            <User className="w-5 h-5 text-accent" />
            <span className="font-medium text-text-primary">Avatar</span>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-text-secondary">
              Expression: {avatarState.expression}
            </p>
            <p className="text-sm text-text-secondary">
              Style: {avatarState.appearance.style}
            </p>
            <p className="text-sm text-text-secondary">
              Speaking: {avatarState.isSpeaking ? 'Yes' : 'No'}
            </p>
          </div>
        </div>

        {/* Input Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 p-4">
          <div className="flex items-center space-x-3 mb-3">
            <Mic className="w-5 h-5 text-accent" />
            <span className="font-medium text-text-primary">Input</span>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-text-secondary">
              Whisper: {modelStatus.whisperReady ? 'Ready' : 'Not Ready'}
            </p>
            <p className="text-sm text-text-secondary">
              Camera: {modelStatus.cameraEnabled ? 'Enabled' : 'Disabled'}
            </p>
          </div>
        </div>

        {/* UI State */}
        <div className="border border-border-primary rounded-card bg-surface/50 p-4">
          <h3 className="font-medium text-text-primary mb-3">UI State</h3>
          <div className="space-y-2">
            <p className="text-sm text-text-secondary">
              Sidebar Collapsed: {uiState.leftSidebarCollapsed ? 'Yes' : 'No'}
            </p>
            <p className="text-sm text-text-secondary">
              Theme: {uiState.theme}
            </p>
            <p className="text-sm text-text-secondary">
              Project: {uiState.currentProject}
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default StoreTestSidebar;