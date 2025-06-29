import React, { useState } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Settings,
  User,
  Mic,
  Camera,
  Brain,
  PanelLeft,
  PanelLeftClose,
} from 'lucide-react';
import { useTektraStore } from '../store';

const SimpleLeftSidebar: React.FC = () => {
  const { uiState, toggleLeftSidebar, modelStatus } = useTektraStore();
  const [openSections, setOpenSections] = useState<Set<string>>(new Set(['ai-model']));

  const toggleSection = (sectionId: string) => {
    const newOpenSections = new Set(openSections);
    if (newOpenSections.has(sectionId)) {
      newOpenSections.delete(sectionId);
    } else {
      newOpenSections.add(sectionId);
    }
    setOpenSections(newOpenSections);
  };

  if (uiState.leftSidebarCollapsed) {
    return (
      <aside className="fixed left-0 top-16 bottom-8 z-40 w-16 bg-secondary-bg border-r border-border-primary flex flex-col items-center py-4 space-y-4">
        <button
          onClick={toggleLeftSidebar}
          className="p-3 rounded-button hover:bg-surface-hover transition-colors"
          title="Expand sidebar"
        >
          <PanelLeft className="w-5 h-5 text-text-secondary" />
        </button>
        
        <div className="space-y-3">
          <div className="p-2 rounded-button" title="AI Model">
            <Brain className="w-5 h-5 text-accent" />
          </div>
          <div className="p-2 rounded-button" title="Avatar">
            <User className="w-5 h-5 text-text-secondary" />
          </div>
          <div className="p-2 rounded-button" title="Input">
            <Mic className="w-5 h-5 text-text-secondary" />
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="fixed left-0 top-16 bottom-8 z-40 w-80 bg-secondary-bg border-r border-border-primary flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border-primary">
        <h2 className="font-semibold text-text-primary">Configuration</h2>
        <button
          onClick={toggleLeftSidebar}
          className="p-2 rounded-button hover:bg-surface-hover transition-colors"
        >
          <PanelLeftClose className="w-4 h-4 text-text-secondary" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {/* AI Model Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('ai-model')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <Brain className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">AI Model</span>
            </div>
            {openSections.has('ai-model') ? (
              <ChevronDown className="w-4 h-4 text-text-secondary" />
            ) : (
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            )}
          </button>
          
          {openSections.has('ai-model') && (
            <div className="p-4 pt-0 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Current Model
                </label>
                <div className="p-3 bg-surface rounded-button border border-border-primary">
                  <p className="text-sm text-text-primary">{modelStatus.modelName}</p>
                  <p className="text-xs text-text-tertiary">{modelStatus.backend}</p>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Status
                </label>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    modelStatus.isLoaded ? 'bg-success' : 'bg-error'
                  }`} />
                  <span className="text-sm text-text-secondary">
                    {modelStatus.isLoaded ? 'Ready' : 'Not Loaded'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Avatar Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('avatar')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <User className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">Avatar</span>
            </div>
            {openSections.has('avatar') ? (
              <ChevronDown className="w-4 h-4 text-text-secondary" />
            ) : (
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            )}
          </button>
          
          {openSections.has('avatar') && (
            <div className="p-4 pt-0 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Style
                </label>
                <select className="w-full p-2 bg-surface border border-border-primary rounded-button text-text-primary">
                  <option>Realistic</option>
                  <option>Stylized</option>
                  <option>Minimal</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Input Modes Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('input')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <Mic className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">Input Modes</span>
            </div>
            {openSections.has('input') ? (
              <ChevronDown className="w-4 h-4 text-text-secondary" />
            ) : (
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            )}
          </button>
          
          {openSections.has('input') && (
            <div className="p-4 pt-0 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-primary">Voice Input</span>
                <input 
                  type="checkbox" 
                  defaultChecked={modelStatus.whisperReady}
                  className="w-4 h-4"
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-primary">Camera Input</span>
                <input 
                  type="checkbox" 
                  defaultChecked={modelStatus.cameraEnabled}
                  className="w-4 h-4"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

export default SimpleLeftSidebar;