import React from 'react';
import {
  Brain,
  User,
  Mic,
  PanelLeft,
  PanelLeftClose,
} from 'lucide-react';

const MinimalLeftSidebar: React.FC = () => {
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
        {/* Simple static sections */}
        <div className="border border-border-primary rounded-card bg-surface/50 p-4">
          <div className="flex items-center space-x-3 mb-3">
            <Brain className="w-5 h-5 text-accent" />
            <span className="font-medium text-text-primary">AI Model</span>
          </div>
          <p className="text-sm text-text-secondary">Model configuration options</p>
        </div>

        <div className="border border-border-primary rounded-card bg-surface/50 p-4">
          <div className="flex items-center space-x-3 mb-3">
            <User className="w-5 h-5 text-accent" />
            <span className="font-medium text-text-primary">Avatar</span>
          </div>
          <p className="text-sm text-text-secondary">Avatar customization options</p>
        </div>

        <div className="border border-border-primary rounded-card bg-surface/50 p-4">
          <div className="flex items-center space-x-3 mb-3">
            <Mic className="w-5 h-5 text-accent" />
            <span className="font-medium text-text-primary">Input Modes</span>
          </div>
          <p className="text-sm text-text-secondary">Voice and camera input settings</p>
        </div>
      </div>
    </aside>
  );
};

export default MinimalLeftSidebar;