import React from 'react';
import '../styles/LoadingScreen.css';

interface LoadingScreenProps {
  isLoading: boolean;
  progress?: number;
  status?: string;
  modelName?: string;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({ 
  isLoading, 
  progress = 0, 
  status = 'Initializing...', 
  modelName 
}) => {
  if (!isLoading) return null;

  return (
    <div className="loading-overlay">
      <div className="loading-container">
        <div className="loading-header">
          <div className="loading-icon">
            <div className="spinner"></div>
          </div>
          <h2>Loading AI Model</h2>
          {modelName && <p className="model-name">{modelName}</p>}
        </div>
        
        <div className="loading-content">
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${Math.min(progress, 100)}%` }}
              ></div>
            </div>
            <div className="progress-text">
              {progress > 0 ? `${Math.round(progress)}%` : '...'}
            </div>
          </div>
          
          <div className="status-text">
            {status}
          </div>
          
          <div className="loading-details">
            <p>This may take a few minutes on first run.</p>
            <p>Models are cached for faster future loading.</p>
          </div>
        </div>
      </div>
    </div>
  );
};