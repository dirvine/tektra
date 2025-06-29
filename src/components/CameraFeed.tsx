import React, { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { Camera, RefreshCw, AlertCircle } from 'lucide-react';

const CameraFeed: React.FC = () => {
  const [cameraFrame, setCameraFrame] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    startCameraFeed();
    return () => {
      stopCameraFeed();
    };
  }, []);

  const startCameraFeed = () => {
    // Update camera feed every 100ms for smooth video
    intervalRef.current = setInterval(async () => {
      try {
        const frame = await invoke<string>('get_camera_frame');
        setCameraFrame(frame);
        setIsLoading(false);
        setError(null);
      } catch (err) {
        console.error('Camera frame error:', err);
        setError(err as string);
        setIsLoading(false);
      }
    }, 100);
  };

  const stopCameraFeed = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const refreshCamera = () => {
    setIsLoading(true);
    setError(null);
    stopCameraFeed();
    setTimeout(startCameraFeed, 500);
  };

  if (error) {
    return (
      <div className="camera-feed error">
        <div className="camera-header">
          <Camera className="w-5 h-5" />
          <h3>Camera Feed</h3>
          <button onClick={refreshCamera} className="refresh-button" title="Refresh Camera">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        
        <div className="camera-error">
          <AlertCircle className="w-8 h-8 text-red-500" />
          <p>Camera Error</p>
          <span className="error-text">{error}</span>
          <button onClick={refreshCamera} className="retry-button">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="camera-feed">
      <div className="camera-header">
        <Camera className="w-5 h-5" />
        <h3>Camera Feed</h3>
        <div className="camera-controls">
          <div className="recording-indicator">
            <div className="recording-dot" />
            <span>Live</span>
          </div>
          <button onClick={refreshCamera} className="refresh-button" title="Refresh Camera">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="camera-viewport">
        {isLoading ? (
          <div className="camera-loading">
            <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
            <p>Initializing camera...</p>
          </div>
        ) : cameraFrame ? (
          <img
            src={cameraFrame}
            alt="Camera feed"
            className="camera-image"
          />
        ) : (
          <div className="camera-placeholder">
            <Camera className="w-12 h-12 text-gray-400" />
            <p>No camera feed available</p>
          </div>
        )}
      </div>

      <div className="camera-info">
        <div className="info-item">
          <span className="label">Status:</span>
          <span className={`value ${cameraFrame ? 'active' : 'inactive'}`}>
            {cameraFrame ? 'Active' : 'Inactive'}
          </span>
        </div>
        <div className="info-item">
          <span className="label">Resolution:</span>
          <span className="value">640x480</span>
        </div>
        <div className="info-item">
          <span className="label">FPS:</span>
          <span className="value">~10</span>
        </div>
      </div>
    </div>
  );
};

export default CameraFeed;