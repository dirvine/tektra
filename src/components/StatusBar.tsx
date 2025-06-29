import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Wifi,
  WifiOff,
  Zap,
  Clock,
  DollarSign,
  Keyboard,
  Monitor,
  Camera,
  Mic,
  Brain,
  Activity,
} from 'lucide-react';
import { useTektraStore } from '../store';

interface ConnectionStatus {
  status: 'connected' | 'disconnected' | 'connecting';
  latency: number;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
}

const StatusBar: React.FC = () => {
  const { modelStatus, sessionState, uiState } = useTektraStore();
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    status: 'connected',
    latency: 45,
    quality: 'excellent',
  });
  const [fps, setFps] = useState(60);
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Simulate connection monitoring
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate latency fluctuation
      const newLatency = Math.random() * 100 + 20;
      const quality = newLatency < 50 ? 'excellent' : 
                     newLatency < 100 ? 'good' : 
                     newLatency < 150 ? 'fair' : 'poor';
      
      setConnectionStatus(prev => ({
        ...prev,
        latency: Math.round(newLatency),
        quality,
      }));

      // Simulate FPS monitoring
      setFps(Math.round(60 - Math.random() * 5));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getConnectionIcon = () => {
    if (connectionStatus.status === 'connecting') {
      return <Wifi className="w-3 h-3 text-warning animate-pulse" />;
    } else if (connectionStatus.status === 'connected') {
      return <Wifi className="w-3 h-3 text-success" />;
    } else {
      return <WifiOff className="w-3 h-3 text-error" />;
    }
  };

  const getQualityColor = () => {
    switch (connectionStatus.quality) {
      case 'excellent': return 'text-success';
      case 'good': return 'text-success';
      case 'fair': return 'text-warning';
      case 'poor': return 'text-error';
      default: return 'text-text-tertiary';
    }
  };

  const getCurrentMode = () => {
    if (modelStatus.cameraEnabled && modelStatus.whisperReady) return 'Multimodal';
    if (modelStatus.cameraEnabled) return 'Vision';
    if (modelStatus.whisperReady) return 'Voice';
    return 'Chat';
  };

  const formatCost = (cost: number) => {
    return cost < 0.01 ? '<$0.01' : `$${cost.toFixed(3)}`;
  };

  const shortcuts = [
    { key: 'Ctrl+N', action: 'New conversation' },
    { key: 'Ctrl+K', action: 'Command palette' },
    { key: 'Space', action: 'Push to talk' },
    { key: 'Ctrl+/', action: 'Toggle shortcuts' },
    { key: 'Ctrl+B', action: 'Toggle sidebar' },
    { key: 'Ctrl+.', action: 'Toggle settings' },
  ];

  return (
    <>
      {/* Status Bar */}
      <footer className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-between px-4 text-xs">
        {/* Left Section - Connection & Performance */}
        <div className="flex items-center space-x-4">
          {/* Connection Status */}
          <div className="flex items-center space-x-1">
            {getConnectionIcon()}
            <span className="text-text-tertiary">
              {connectionStatus.status === 'connected' ? 'Connected' : 
               connectionStatus.status === 'connecting' ? 'Connecting...' : 'Offline'}
            </span>
          </div>

          {/* Latency */}
          <div className="flex items-center space-x-1">
            <Activity className="w-3 h-3 text-text-tertiary" />
            <span className={getQualityColor()}>
              {connectionStatus.latency}ms
            </span>
          </div>

          {/* FPS Counter */}
          <div className="flex items-center space-x-1">
            <Monitor className="w-3 h-3 text-text-tertiary" />
            <span className={fps >= 55 ? 'text-success' : fps >= 30 ? 'text-warning' : 'text-error'}>
              {fps}fps
            </span>
          </div>

          {/* Memory Usage (simulated) */}
          <div className="flex items-center space-x-1">
            <Brain className="w-3 h-3 text-text-tertiary" />
            <span className="text-text-tertiary">
              2.1GB
            </span>
          </div>
        </div>

        {/* Center Section - Current Mode */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            {getCurrentMode() === 'Multimodal' && (
              <>
                <Camera className="w-3 h-3 text-accent" />
                <Mic className="w-3 h-3 text-accent" />
              </>
            )}
            {getCurrentMode() === 'Vision' && <Camera className="w-3 h-3 text-accent" />}
            {getCurrentMode() === 'Voice' && <Mic className="w-3 h-3 text-accent" />}
            {getCurrentMode() === 'Chat' && <Brain className="w-3 h-3 text-text-tertiary" />}
          </div>
          <span className="text-text-primary font-medium">
            {getCurrentMode()}
          </span>
          <div className="w-px h-3 bg-border-primary" />
          <span className="text-text-tertiary">
            {modelStatus.modelName}
          </span>
        </div>

        {/* Right Section - Usage & Shortcuts */}
        <div className="flex items-center space-x-4">
          {/* Token Usage */}
          <div className="flex items-center space-x-1">
            <Zap className="w-3 h-3 text-text-tertiary" />
            <span className="text-text-tertiary">
              {sessionState.tokenUsage.toLocaleString()} tokens
            </span>
          </div>

          {/* Cost Estimate */}
          <div className="flex items-center space-x-1">
            <DollarSign className="w-3 h-3 text-text-tertiary" />
            <span className="text-text-tertiary">
              {formatCost(sessionState.costEstimate)}
            </span>
          </div>

          {/* Session Duration */}
          <div className="flex items-center space-x-1">
            <Clock className="w-3 h-3 text-text-tertiary" />
            <span className="text-text-tertiary">
              {Math.floor(sessionState.duration / 60)}:{(sessionState.duration % 60).toString().padStart(2, '0')}
            </span>
          </div>

          {/* Keyboard Shortcuts Hint */}
          <button
            onClick={() => setShowShortcuts(!showShortcuts)}
            className="flex items-center space-x-1 hover:text-text-primary transition-colors"
          >
            <Keyboard className="w-3 h-3 text-text-tertiary" />
            <span className="text-text-tertiary">Ctrl+/</span>
          </button>
        </div>
      </footer>

      {/* Keyboard Shortcuts Modal */}
      {showShortcuts && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[100] flex items-center justify-center"
          onClick={() => setShowShortcuts(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-secondary-bg border border-border-primary rounded-modal p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-text-primary">Keyboard Shortcuts</h3>
              <button
                onClick={() => setShowShortcuts(false)}
                className="text-text-tertiary hover:text-text-primary transition-colors"
              >
                ×
              </button>
            </div>
            
            <div className="space-y-3">
              {shortcuts.map((shortcut, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-text-secondary">{shortcut.action}</span>
                  <kbd className="px-2 py-1 bg-surface border border-border-primary rounded text-xs font-mono text-text-primary">
                    {shortcut.key}
                  </kbd>
                </div>
              ))}
            </div>
            
            <div className="mt-6 pt-4 border-t border-border-primary">
              <p className="text-xs text-text-tertiary text-center">
                Press <kbd className="px-1 py-0.5 bg-surface border border-border-primary rounded text-xs">Esc</kbd> or click outside to close
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </>
  );
};

export default StatusBar;