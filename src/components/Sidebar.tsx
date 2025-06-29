import React from 'react';
import { 
  Mic, 
  MicOff, 
  Camera, 
  CameraOff, 
  Bot, 
  Settings, 
  CheckCircle, 
  AlertCircle,
  Loader,
  Cpu,
  Volume2
} from 'lucide-react';
import { ModelStatus } from '../types';

interface SidebarProps {
  modelStatus: ModelStatus;
  onToggleRecording: () => void;
  onToggleCamera: () => void;
  isRecording: boolean;
  showCamera: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  modelStatus,
  onToggleRecording,
  onToggleCamera,
  isRecording,
  showCamera
}) => {
  const getStatusIcon = (ready: boolean, loading?: boolean) => {
    if (loading) return <Loader className="w-4 h-4 animate-spin text-blue-500" />;
    return ready ? 
      <CheckCircle className="w-4 h-4 text-green-500" /> : 
      <AlertCircle className="w-4 h-4 text-red-500" />;
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo">
          <Bot className="w-8 h-8 text-blue-500" />
          <h1 className="text-xl font-bold">Tektra</h1>
        </div>
        <p className="subtitle">AI Assistant</p>
      </div>

      <div className="sidebar-section">
        <h3 className="section-title">
          <Cpu className="w-4 h-4" />
          Model Status
        </h3>
        
        <div className="status-grid">
          <div className="status-item">
            <div className="status-label">
              <span>AI Model</span>
              {getStatusIcon(modelStatus.isLoaded, modelStatus.isLoading)}
            </div>
            <div className="status-details">
              <span className="model-name">{modelStatus.modelName}</span>
              <span className="backend-name">{modelStatus.backend}</span>
            </div>
          </div>

          <div className="status-item">
            <div className="status-label">
              <span>Speech Recognition</span>
              {getStatusIcon(modelStatus.whisperReady)}
            </div>
            <div className="status-details">
              <span className={modelStatus.whisperReady ? 'text-green-600' : 'text-red-600'}>
                {modelStatus.whisperReady ? 'Whisper Ready' : 'Not Available'}
              </span>
            </div>
          </div>

          <div className="status-item">
            <div className="status-label">
              <span>Camera</span>
              {getStatusIcon(modelStatus.cameraEnabled)}
            </div>
            <div className="status-details">
              <span className={modelStatus.cameraEnabled ? 'text-green-600' : 'text-gray-600'}>
                {modelStatus.cameraEnabled ? 'Active' : 'Disabled'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="sidebar-section">
        <h3 className="section-title">
          <Settings className="w-4 h-4" />
          Controls
        </h3>
        
        <div className="controls-grid">
          <button
            onClick={onToggleRecording}
            disabled={!modelStatus.whisperReady}
            className={`control-button ${isRecording ? 'recording' : ''}`}
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
          >
            {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
          </button>

          <button
            onClick={onToggleCamera}
            className={`control-button ${showCamera ? 'active' : ''}`}
            title={showCamera ? 'Disable Camera' : 'Enable Camera'}
          >
            {showCamera ? <CameraOff className="w-5 h-5" /> : <Camera className="w-5 h-5" />}
            <span>{showCamera ? 'Disable Camera' : 'Enable Camera'}</span>
          </button>
        </div>
      </div>

      <div className="sidebar-section">
        <h3 className="section-title">
          <Volume2 className="w-4 h-4" />
          Audio Features
        </h3>
        
        <div className="feature-list">
          <div className="feature-item">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span>Real-time Transcription</span>
          </div>
          <div className="feature-item">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span>Voice Activity Detection</span>
          </div>
          <div className="feature-item">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span>Continuous Listening</span>
          </div>
        </div>
      </div>

      <div className="sidebar-footer">
        <div className="version-info">
          <span className="version">v0.2.0</span>
          <span className="build">Production Build</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;