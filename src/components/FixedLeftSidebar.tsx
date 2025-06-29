import React, { useState } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Settings,
  User,
  Mic,
  Camera,
  FileText,
  Zap,
  Volume2,
  Sliders,
  Palette,
  Brain,
  Eye,
  Upload,
  Link,
  Tool,
  BarChart3,
  PanelLeft,
  PanelLeftClose,
} from 'lucide-react';
import { useTektraStore } from '../store';

interface CollapsibleSectionProps {
  title: string;
  icon: React.ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

// Fixed CollapsibleSection without problematic animations
const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  icon,
  isOpen,
  onToggle,
  children,
}) => (
  <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
    >
      <div className="flex items-center space-x-3">
        <div className="text-accent">{icon}</div>
        <span className="font-medium text-text-primary">{title}</span>
      </div>
      <div className={`transform transition-transform duration-200 ${isOpen ? 'rotate-90' : 'rotate-0'}`}>
        <ChevronRight className="w-4 h-4 text-text-secondary" />
      </div>
    </button>
    
    {/* Simple CSS transition instead of Framer Motion */}
    <div className={`transition-all duration-200 ease-in-out ${
      isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
    } overflow-hidden border-t border-border-primary`}>
      <div className="p-4 space-y-4">
        {children}
      </div>
    </div>
  </div>
);

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  unit?: string;
}

const SliderControl: React.FC<SliderControlProps> = ({
  label,
  value,
  min,
  max,
  step = 0.1,
  onChange,
  unit = '',
}) => (
  <div className="space-y-2">
    <div className="flex justify-between items-center">
      <label className="text-sm text-text-secondary">{label}</label>
      <span className="text-sm text-text-primary font-mono">
        {value}{unit}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full"
    />
  </div>
);

const FixedLeftSidebar: React.FC = () => {
  const { 
    uiState, 
    modelStatus, 
    avatarState,
    toggleLeftSidebar,
    setModelStatus,
    setAvatarState 
  } = useTektraStore();
  
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(['ai-model', 'avatar'])
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
          <div className="p-2 rounded-button hover:bg-surface-hover transition-colors" title="Tools">
            <Tool className="w-5 h-5 text-text-secondary" />
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
        {/* AI Model Configuration */}
        <CollapsibleSection
          title="AI Model"
          icon={<Brain className="w-5 h-5" />}
          isOpen={openSections.has('ai-model')}
          onToggle={() => toggleSection('ai-model')}
        >
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
        </CollapsibleSection>

        {/* Avatar Customization */}
        <CollapsibleSection
          title="Avatar"
          icon={<User className="w-5 h-5" />}
          isOpen={openSections.has('avatar')}
          onToggle={() => toggleSection('avatar')}
        >
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
        </CollapsibleSection>

        {/* Input Modes */}
        <CollapsibleSection
          title="Input Modes"
          icon={<Mic className="w-5 h-5" />}
          isOpen={openSections.has('input-modes')}
          onToggle={() => toggleSection('input-modes')}
        >
          <div className="space-y-3">
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

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4 text-accent" />
                <span className="text-sm text-text-primary">File Upload</span>
              </div>
              <input type="checkbox" defaultChecked className="w-4 h-4" />
            </div>
          </div>
        </CollapsibleSection>

        {/* Tools & Integrations */}
        <CollapsibleSection
          title="Tools & Integrations"
          icon={<Tool className="w-5 h-5" />}
          isOpen={openSections.has('tools')}
          onToggle={() => toggleSection('tools')}
        >
          <div className="space-y-3">
            <div className="p-3 bg-surface rounded-button border border-border-primary">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-primary">Web Search</span>
                <span className="text-xs text-success">Connected</span>
              </div>
            </div>

            <div className="p-3 bg-surface rounded-button border border-border-primary">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-primary">Code Execution</span>
                <span className="text-xs text-text-tertiary">Available</span>
              </div>
            </div>

            <button className="w-full p-2 bg-surface hover:bg-surface-hover border border-border-primary rounded-button text-text-primary text-sm transition-colors">
              + Add Integration
            </button>
          </div>
        </CollapsibleSection>
      </div>
    </aside>
  );
};

export default FixedLeftSidebar;