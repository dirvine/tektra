import React, { useState } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Brain,
  User,
  Mic,
  Camera,
  Tool,
  PanelLeft,
  PanelLeftClose,
} from 'lucide-react';
import { useTektraStore } from '../store';

const WorkingLeftSidebar: React.FC = () => {
  const { 
    uiState, 
    modelStatus, 
    avatarState,
    toggleLeftSidebar,
    setModelStatus,
    setAvatarState 
  } = useTektraStore();
  
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(['ai-model'])
  );

  const toggleSection = (sectionId: string) => {
    const newOpenSections = new Set(openSections);
    if (newOpenSections.has(sectionId)) {
      newOpenSections.delete(sectionId);
    } else {
      newOpenSections.add(sectionId);
    }
    setOpenSections(newOpenSections);
  };

  // Collapsed sidebar
  if (uiState.leftSidebarCollapsed) {
    return (
      <aside className="fixed left-0 top-16 bottom-8 z-40 w-16 bg-secondary-bg border-r border-border-primary flex flex-col items-center py-4 space-y-4">
        <button
          onClick={toggleLeftSidebar}
          className="p-3 rounded-button hover:bg-surface-hover transition-colors group"
          title="Expand sidebar"
        >
          <PanelLeft className="w-5 h-5 text-text-secondary group-hover:text-text-primary transition-colors" />
        </button>
        
        <div className="space-y-3">
          <div className="p-2 rounded-button hover:bg-surface-hover transition-colors" title="AI Model">
            <Brain className="w-5 h-5 text-accent" />
          </div>
          <div className="p-2 rounded-button hover:bg-surface-hover transition-colors" title="Avatar">
            <User className="w-5 h-5 text-text-secondary" />
          </div>
          <div className="p-2 rounded-button hover:bg-surface-hover transition-colors" title="Input">
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
            <div className={`transform transition-transform duration-200 ${
              openSections.has('ai-model') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('ai-model') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-4">
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

              <button className="w-full p-2 bg-accent hover:bg-accent-hover text-white rounded-button transition-colors text-sm">
                Change Model
              </button>
            </div>
          </div>
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
            <div className={`transform transition-transform duration-200 ${
              openSections.has('avatar') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('avatar') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Expression
                </label>
                <select 
                  value={avatarState.expression}
                  onChange={(e) => setAvatarState({ expression: e.target.value as any })}
                  className="w-full p-2 bg-surface border border-border-primary rounded-button text-text-primary"
                >
                  <option value="neutral">Neutral</option>
                  <option value="happy">Happy</option>
                  <option value="thinking">Thinking</option>
                  <option value="surprised">Surprised</option>
                  <option value="friendly">Friendly</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Style
                </label>
                <select 
                  value={avatarState.appearance.style}
                  onChange={(e) => setAvatarState({ 
                    appearance: { ...avatarState.appearance, style: e.target.value as any }
                  })}
                  className="w-full p-2 bg-surface border border-border-primary rounded-button text-text-primary"
                >
                  <option value="realistic">Realistic</option>
                  <option value="stylized">Stylized</option>
                  <option value="minimal">Minimal</option>
                </select>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-text-primary">Eye Tracking</span>
                <input 
                  type="checkbox" 
                  checked={avatarState.animation.eyeTracking}
                  onChange={(e) => setAvatarState({ 
                    animation: { ...avatarState.animation, eyeTracking: e.target.checked }
                  })}
                  className="w-4 h-4"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Input Modes */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('input-modes')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <Mic className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">Input Modes</span>
            </div>
            <div className={`transform transition-transform duration-200 ${
              openSections.has('input-modes') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('input-modes') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Mic className="w-4 h-4 text-accent" />
                  <span className="text-sm text-text-primary">Voice Input</span>
                </div>
                <input 
                  type="checkbox" 
                  checked={modelStatus.whisperReady}
                  onChange={(e) => setModelStatus({ whisperReady: e.target.checked })}
                  className="w-4 h-4"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Camera className="w-4 h-4 text-accent" />
                  <span className="text-sm text-text-primary">Camera Input</span>
                </div>
                <input 
                  type="checkbox" 
                  checked={modelStatus.cameraEnabled}
                  onChange={(e) => setModelStatus({ cameraEnabled: e.target.checked })}
                  className="w-4 h-4"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Tools Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('tools')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <Tool className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">Tools</span>
            </div>
            <div className={`transform transition-transform duration-200 ${
              openSections.has('tools') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('tools') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-3">
              <div className="p-3 bg-surface rounded-button border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Web Search</span>
                  <span className="text-xs text-success">Available</span>
                </div>
              </div>

              <div className="p-3 bg-surface rounded-button border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Code Execution</span>
                  <span className="text-xs text-success">Available</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default WorkingLeftSidebar;