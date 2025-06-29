import React from 'react';
import { motion } from 'framer-motion';
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

const HeaderBar: React.FC<HeaderBarProps> = ({ className = '' }) => {
  // Use individual selectors instead of destructuring - this is more reliable
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const uiState = useTektraStore((state) => state.uiState);
  const setActiveTab = useTektraStore((state) => state.setActiveTab);

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
      glass-dark border-b border-border-primary
      flex items-center justify-between px-6
      ${className}
    `}>
      {/* Left Section - Logo and Brand */}
      <motion.div 
        className="flex items-center space-x-4"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center space-x-3">
          {/* Logo */}
          <div className="w-8 h-8 bg-gradient-to-br from-accent to-accent-light rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          
          {/* Brand Name */}
          <div className="flex flex-col">
            <h1 className="text-xl font-bold text-text-primary font-sans tracking-tight">
              Tektra
            </h1>
            <p className="text-xs text-text-tertiary -mt-1">
              AI Assistant
            </p>
          </div>
        </div>
      </motion.div>

      {/* Center Section - Project Selector and Status */}
      <motion.div 
        className="flex items-center space-x-6"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        {/* Project Selector */}
        <div className="flex items-center space-x-2 px-4 py-2 rounded-card bg-surface border border-border-primary hover:border-border-secondary transition-colors cursor-pointer group">
          <span className="text-text-primary font-medium">
            {uiState.currentProject}
          </span>
          <ChevronDown className="w-4 h-4 text-text-secondary group-hover:text-text-primary transition-colors" />
        </div>

        {/* Status Indicators */}
        <div className="flex items-center space-x-4">
          {/* AI Model Status */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              modelStatus.isLoaded ? 'bg-success' : 
              modelStatus.isLoading ? 'bg-warning animate-pulse' : 'bg-error'
            }`} />
            <span className="text-sm text-text-secondary">
              {modelStatus.modelName}
            </span>
            <span className={`text-xs ${getStatusColor()}`}>
              {getStatusText()}
            </span>
          </div>

          {/* Whisper Status */}
          {modelStatus.whisperReady && (
            <div className="flex items-center space-x-1">
              <Activity className="w-3 h-3 text-success" />
              <span className="text-xs text-text-tertiary">STT</span>
            </div>
          )}

          {/* Camera Status */}
          {modelStatus.cameraEnabled && (
            <div className="flex items-center space-x-1">
              <Eye className="w-3 h-3 text-success" />
              <span className="text-xs text-text-tertiary">Vision</span>
            </div>
          )}
        </div>
      </motion.div>

      {/* Right Section - User Controls */}
      <motion.div 
        className="flex items-center space-x-3"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
      >
        {/* Performance Indicator */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-button bg-surface/50 border border-border-primary">
          <Zap className="w-3 h-3 text-accent" />
          <span className="text-xs text-text-secondary font-mono">
            60fps
          </span>
        </div>

        {/* Notifications */}
        <button className="relative p-2 rounded-button hover:bg-surface-hover transition-colors group">
          <Bell className="w-5 h-5 text-text-secondary group-hover:text-text-primary transition-colors" />
          {uiState.notifications.length > 0 && (
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-error rounded-full flex items-center justify-center">
              <span className="text-xs text-white font-bold">
                {uiState.notifications.length}
              </span>
            </div>
          )}
        </button>

        {/* Settings */}
        <button className="p-2 rounded-button hover:bg-surface-hover transition-colors group">
          <Settings className="w-5 h-5 text-text-secondary group-hover:text-text-primary transition-colors" />
        </button>

        {/* User Profile */}
        <button className="flex items-center space-x-2 p-2 rounded-button hover:bg-surface-hover transition-colors group">
          <div className="w-6 h-6 bg-gradient-to-br from-accent to-accent-light rounded-full flex items-center justify-center">
            <User className="w-3 h-3 text-white" />
          </div>
          <ChevronDown className="w-3 h-3 text-text-secondary group-hover:text-text-primary transition-colors" />
        </button>
      </motion.div>
    </header>
  );
};

export default HeaderBar;