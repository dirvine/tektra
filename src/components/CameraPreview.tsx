import React, { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Camera,
  CameraOff,
  Video,
  VideoOff,
  Settings,
  Maximize2,
  Minimize2,
  RotateCcw,
  Eye,
  EyeOff,
  Zap,
  Activity,
} from 'lucide-react';
import { useTektraStore } from '../store';

interface CameraPreviewProps {
  className?: string;
  onFrameCapture?: (frameData: string) => void;
}

const CameraPreview: React.FC<CameraPreviewProps> = ({ className = '', onFrameCapture }) => {
  const { modelStatus, setModelStatus } = useTektraStore();
  const [isCapturing, setIsCapturing] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  
  const videoRef = useRef<HTMLDivElement>(null);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const fpsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastFrameTime = useRef<number>(Date.now());

  // Initialize camera when component mounts
  useEffect(() => {
    initializeCamera();
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
      }
      stopCapture();
    };
  }, []);

  // Listen for camera events
  useEffect(() => {
    const setupEventListeners = async () => {
      await listen('camera-capture-started', () => {
        setIsCapturing(true);
        setCameraError(null);
      });
      
      await listen('camera-capture-stopped', () => {
        setIsCapturing(false);
        setCurrentFrame(null);
      });
      
      await listen('camera-error', (event: any) => {
        setCameraError(event.payload);
        setIsCapturing(false);
      });
    };

    setupEventListeners();
  }, []);

  const initializeCamera = async () => {
    try {
      await invoke('initialize_camera');
    } catch (error) {
      console.error('Failed to initialize camera:', error);
      setCameraError(`Camera initialization failed: ${error}`);
    }
  };

  const startCapture = async () => {
    try {
      const success = await invoke<boolean>('start_camera_capture');
      if (success) {
        setIsCapturing(true);
        setCameraError(null);
        
        // Update model status to enable camera
        setModelStatus({
          ...modelStatus,
          cameraEnabled: true,
        });
        
        // Start frame capture loop
        captureIntervalRef.current = setInterval(captureFrame, 100); // 10 FPS
        
        // Start FPS counter
        fpsIntervalRef.current = setInterval(() => {
          const now = Date.now();
          const timeDiff = now - lastFrameTime.current;
          const currentFps = timeDiff > 0 ? Math.round(1000 / timeDiff) : 0;
          setFps(currentFps);
        }, 1000);
      }
    } catch (error) {
      console.error('Failed to start camera capture:', error);
      setCameraError(`Failed to start camera: ${error}`);
    }
  };

  const stopCapture = async () => {
    try {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
      
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
        fpsIntervalRef.current = null;
      }
      
      await invoke('stop_camera_capture');
      setIsCapturing(false);
      setCurrentFrame(null);
      setFps(0);
      
      // Update model status to disable camera
      setModelStatus({
        ...modelStatus,
        cameraEnabled: false,
      });
    } catch (error) {
      console.error('Failed to stop camera capture:', error);
      setCameraError(`Failed to stop camera: ${error}`);
    }
  };

  const captureFrame = async () => {
    try {
      const frameData = await invoke<string>('get_camera_frame');
      setCurrentFrame(frameData);
      setFrameCount(prev => prev + 1);
      lastFrameTime.current = Date.now();
      
      // Callback for parent component
      if (onFrameCapture) {
        onFrameCapture(frameData);
      }
    } catch (error) {
      console.error('Failed to capture frame:', error);
    }
  };

  const toggleCapture = () => {
    if (isCapturing) {
      stopCapture();
    } else {
      startCapture();
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  return (
    <div 
      className={`
        relative bg-surface border border-border-primary rounded-card overflow-hidden
        transition-all duration-300 ease-in-out
        ${isFullscreen ? 'fixed inset-4 z-50' : 'w-full h-64'}
        ${className}
      `}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      {/* Camera Feed */}
      <div ref={videoRef} className="relative w-full h-full bg-black">
        {currentFrame ? (
          <motion.img
            key={frameCount}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.1 }}
            src={currentFrame}
            alt="Camera feed"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-text-tertiary">
            {cameraError ? (
              <div className="text-center p-4">
                <CameraOff className="w-12 h-12 mx-auto mb-2 text-error" />
                <p className="text-sm text-error mb-2">Camera Error</p>
                <p className="text-xs text-text-tertiary max-w-xs">{cameraError}</p>
                <button
                  onClick={initializeCamera}
                  className="mt-2 px-3 py-1 text-xs bg-accent text-white rounded-button hover:bg-accent-hover transition-colors"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="text-center">
                <Camera className="w-12 h-12 mx-auto mb-2" />
                <p className="text-sm">Camera Preview</p>
                <p className="text-xs text-text-tertiary">Click to start</p>
              </div>
            )}
          </div>
        )}

        {/* Recording Indicator */}
        {isCapturing && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="absolute top-3 left-3 flex items-center space-x-2 px-2 py-1 bg-error/90 text-white text-xs rounded-full"
          >
            <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
            <span>LIVE</span>
          </motion.div>
        )}

        {/* FPS Counter */}
        {isCapturing && (
          <div className="absolute top-3 right-3 px-2 py-1 bg-black/70 text-white text-xs rounded-full">
            {fps} FPS
          </div>
        )}

        {/* Controls Overlay */}
        <AnimatePresence>
          {showControls && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/20 flex items-end justify-between p-4"
            >
              {/* Bottom Controls */}
              <div className="flex items-center space-x-2">
                {/* Capture Toggle */}
                <button
                  onClick={toggleCapture}
                  className={`
                    p-3 rounded-full transition-all duration-200
                    ${isCapturing 
                      ? 'bg-error text-white hover:bg-error/80' 
                      : 'bg-accent text-white hover:bg-accent-hover'
                    }
                  `}
                  title={isCapturing ? 'Stop camera' : 'Start camera'}
                >
                  {isCapturing ? (
                    <VideoOff className="w-5 h-5" />
                  ) : (
                    <Video className="w-5 h-5" />
                  )}
                </button>

                {/* Settings */}
                <button
                  className="p-2 bg-surface/80 hover:bg-surface text-text-primary rounded-full transition-colors"
                  title="Camera settings"
                >
                  <Settings className="w-4 h-4" />
                </button>

                {/* AI Vision Toggle */}
                <button
                  onClick={() => setModelStatus({
                    ...modelStatus,
                    visionEnabled: !modelStatus.visionEnabled
                  })}
                  className={`
                    p-2 rounded-full transition-colors
                    ${modelStatus.visionEnabled 
                      ? 'bg-accent text-white' 
                      : 'bg-surface/80 hover:bg-surface text-text-secondary'
                    }
                  `}
                  title={modelStatus.visionEnabled ? 'Disable AI vision' : 'Enable AI vision'}
                >
                  {modelStatus.visionEnabled ? (
                    <Eye className="w-4 h-4" />
                  ) : (
                    <EyeOff className="w-4 h-4" />
                  )}
                </button>
              </div>

              {/* Top Right Controls */}
              <div className="flex items-center space-x-2">
                {/* Performance Indicator */}
                {isCapturing && (
                  <div className="flex items-center space-x-1 px-2 py-1 bg-surface/80 rounded-full text-xs text-text-secondary">
                    <Activity className="w-3 h-3" />
                    <span>{frameCount}</span>
                  </div>
                )}

                {/* Fullscreen Toggle */}
                <button
                  onClick={toggleFullscreen}
                  className="p-2 bg-surface/80 hover:bg-surface text-text-primary rounded-full transition-colors"
                  title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                  {isFullscreen ? (
                    <Minimize2 className="w-4 h-4" />
                  ) : (
                    <Maximize2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Status Bar */}
      <div className="absolute bottom-0 left-0 right-0 bg-black/80 text-white text-xs px-3 py-1 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <span className={`
            w-2 h-2 rounded-full
            ${isCapturing ? 'bg-success animate-pulse' : 'bg-text-tertiary'}
          `} />
          <span>
            {isCapturing ? 'Camera Active' : 'Camera Inactive'}
          </span>
          {modelStatus.visionEnabled && (
            <>
              <span>•</span>
              <div className="flex items-center space-x-1">
                <Zap className="w-3 h-3 text-accent" />
                <span>AI Vision</span>
              </div>
            </>
          )}
        </div>
        
        {isCapturing && (
          <div className="text-text-tertiary">
            Frame #{frameCount}
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraPreview;