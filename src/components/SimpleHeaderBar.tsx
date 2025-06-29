import React from 'react';
import { 
  Settings, 
  Bell, 
  User, 
  ChevronDown, 
  Activity,
  Zap,
  Brain,
  Eye
} from 'lucide-react';
import { useTektraStore } from '../store';

interface HeaderBarProps {
  className?: string;
}

const SimpleHeaderBar: React.FC<HeaderBarProps> = ({ className = '' }) => {
  // Use individual selectors - this is the only pattern that works
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);

  const getStatusColor = () => {
    if (modelStatus.isLoading) return 'text-warning';
    if (modelStatus.isLoaded) return 'text-success';
    return 'text-error';
  };

  const getStatusText = () => {
    if (modelStatus.isLoading) return 'Loading...';
    if (modelStatus.isLoaded) return 'Ready';
    return 'Offline';
  };

  return (
    <header className={`
      fixed top-0 left-0 right-0 z-50 h-16
      bg-secondary-bg border-b border-border-primary
      flex items-center justify-between px-6
      ${className}
    `}>
      {/* Left Section - Logo and Brand */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-3">
          {/* Logo */}
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-accent-light flex items-center justify-center shadow-lg">
            <Brain className="w-6 h-6 text-white" />
          </div>
          
          {/* Brand */}
          <div>
            <h1 className="text-xl font-bold text-text-primary">Tektra</h1>
            <p className="text-xs text-text-tertiary -mt-1">AI Assistant</p>
          </div>
        </div>

        {/* Project Selector */}
        <div className="flex items-center space-x-2 px-3 py-1.5 bg-surface rounded-button border border-border-primary hover:bg-surface-hover transition-colors cursor-pointer">
          <span className="text-sm text-text-primary">{uiState.currentProject}</span>
          <ChevronDown className="w-4 h-4 text-text-secondary" />
        </div>
      </div>

      {/* Center Section - Status */}
      <div className="flex items-center space-x-4">
        {/* Model Status */}
        <div className="flex items-center space-x-2 px-3 py-1.5 bg-surface/50 rounded-button border border-border-primary">
          <Activity className={`w-4 h-4 ${getStatusColor()}`} />
          <span className="text-sm text-text-primary">{modelStatus.modelName}</span>
          <span className={`text-xs ${getStatusColor()}`}>{getStatusText()}</span>
        </div>

        {/* Capabilities */}
        <div className="flex items-center space-x-2">
          <div className={`p-2 rounded-full ${
            modelStatus.whisperReady ? 'bg-success/20 text-success' : 'bg-surface text-text-tertiary'
          }`} title="Voice Input">
            <Zap className="w-4 h-4" />
          </div>
          <div className={`p-2 rounded-full ${
            modelStatus.cameraEnabled ? 'bg-success/20 text-success' : 'bg-surface text-text-tertiary'
          }`} title="Vision">
            <Eye className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* Right Section - User Controls */}
      <div className="flex items-center space-x-3">
        {/* Notifications */}
        <button className="p-2 rounded-button hover:bg-surface-hover transition-colors relative">
          <Bell className="w-5 h-5 text-text-secondary" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent rounded-full"></div>
        </button>

        {/* Settings */}
        <button className="p-2 rounded-button hover:bg-surface-hover transition-colors">
          <Settings className="w-5 h-5 text-text-secondary" />
        </button>

        {/* User Profile */}
        <div className="flex items-center space-x-2 px-3 py-1.5 bg-surface rounded-button border border-border-primary hover:bg-surface-hover transition-colors cursor-pointer">
          <div className="w-6 h-6 rounded-full bg-accent flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          <ChevronDown className="w-3 h-3 text-text-secondary" />
        </div>
      </div>
    </header>
  );
};

export default SimpleHeaderBar;