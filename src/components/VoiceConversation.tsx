import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Settings,
  Power,
  Activity,
  AlertCircle,
  CheckCircle,
  Loader2,
  Headphones,
  Speaker,
  Waveform,
  Play,
  Pause,
  Square,
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useTektraStore } from '../store';

// Voice conversation state types
interface VoiceServiceStatus {
  backend: boolean;
  stt: boolean;
  tts: boolean;
  overall: boolean;
}

interface VoiceMetrics {
  audioLevel: number;
  speechDetected: boolean;
  processingLatency: number;
  transcriptionAccuracy: number;
  conversationTurns: number;
}

interface VoiceEvent {
  type: string;
  data?: any;
  timestamp: number;
}

// Voice Conversation Component
const VoiceConversation: React.FC = () => {
  // State management
  const [isVoiceSessionActive, setIsVoiceSessionActive] = useState(false);
  const [isServicesStarting, setIsServicesStarting] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<VoiceServiceStatus>({
    backend: false,
    stt: false,
    tts: false,
    overall: false,
  });
  const [voiceMetrics, setVoiceMetrics] = useState<VoiceMetrics>({
    audioLevel: 0,
    speechDetected: false,
    processingLatency: 0,
    transcriptionAccuracy: 0,
    conversationTurns: 0,
  });
  const [currentTranscription, setCurrentTranscription] = useState('');
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // Store hooks
  const { avatarState, setAvatarListening, setAvatarSpeaking, addMessage, addNotification } = useTektraStore();

  // Refs for audio visualization
  const audioVisualizationRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  // Initialize voice services
  const initializeVoiceServices = async () => {
    setIsServicesStarting(true);
    setLastError(null);

    try {
      addNotification({
        type: 'info',
        message: 'Starting voice services...',
      });

      // Start Unmute services
      const result = await invoke<string>('start_unmute_services');
      console.log('Voice services startup result:', result);

      addNotification({
        type: 'success',
        message: 'Voice services started successfully',
      });

      // Check service status
      await checkServiceStatus();

    } catch (error) {
      console.error('Failed to start voice services:', error);
      setLastError(String(error));
      
      addNotification({
        type: 'error',
        message: `Failed to start voice services: ${error}`,
      });
    } finally {
      setIsServicesStarting(false);
    }
  };

  // Check status of all voice services
  const checkServiceStatus = async () => {
    try {
      const status = await invoke<{[key: string]: boolean}>('get_unmute_service_status');
      
      const newStatus: VoiceServiceStatus = {
        backend: status.backend || false,
        stt: status.stt || false,
        tts: status.tts || false,
        overall: (status.backend && status.stt && status.tts) || false,
      };

      setServiceStatus(newStatus);
      return newStatus.overall;
    } catch (error) {
      console.error('Failed to check service status:', error);
      return false;
    }
  };

  // Start voice conversation session
  const startVoiceSession = async () => {
    try {
      if (!serviceStatus.overall) {
        await initializeVoiceServices();
        
        // Wait for services to be ready
        let attempts = 0;
        while (attempts < 10) {
          const isReady = await checkServiceStatus();
          if (isReady) break;
          
          await new Promise(resolve => setTimeout(resolve, 1000));
          attempts++;
        }
        
        if (!serviceStatus.overall) {
          throw new Error('Voice services failed to start properly');
        }
      }

      // Start the voice pipeline session
      await invoke('start_voice_session');
      
      setIsVoiceSessionActive(true);
      setAvatarListening(true);
      
      addMessage({
        role: 'system',
        content: '🎤 Voice conversation started. You can now speak naturally!',
      });

      addNotification({
        type: 'success',
        message: 'Voice conversation active',
      });

      // Start audio visualization
      startAudioVisualization();

    } catch (error) {
      console.error('Failed to start voice session:', error);
      setLastError(String(error));
      
      addNotification({
        type: 'error',
        message: `Failed to start voice session: ${error}`,
      });
    }
  };

  // Stop voice conversation session
  const stopVoiceSession = async () => {
    try {
      await invoke('stop_voice_session');
      
      setIsVoiceSessionActive(false);
      setAvatarListening(false);
      setAvatarSpeaking(false);
      setCurrentTranscription('');
      
      // Stop audio visualization
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      addMessage({
        role: 'system',
        content: '🛑 Voice conversation ended.',
      });

      addNotification({
        type: 'info',
        message: 'Voice conversation stopped',
      });

    } catch (error) {
      console.error('Failed to stop voice session:', error);
      addNotification({
        type: 'error',
        message: `Failed to stop voice session: ${error}`,
      });
    }
  };

  // Stop voice services
  const stopVoiceServices = async () => {
    try {
      await invoke('stop_unmute_services');
      
      setServiceStatus({
        backend: false,
        stt: false,
        tts: false,
        overall: false,
      });

      addNotification({
        type: 'info',
        message: 'Voice services stopped',
      });

    } catch (error) {
      console.error('Failed to stop voice services:', error);
      addNotification({
        type: 'error',
        message: `Failed to stop voice services: ${error}`,
      });
    }
  };

  // Audio visualization
  const startAudioVisualization = () => {
    const canvas = audioVisualizationRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      if (!isVoiceSessionActive) return;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw audio waveform simulation
      const centerY = canvas.height / 2;
      const barWidth = 4;
      const spacing = 2;
      const numBars = Math.floor(canvas.width / (barWidth + spacing));
      
      for (let i = 0; i < numBars; i++) {
        const x = i * (barWidth + spacing);
        const height = (Math.random() * voiceMetrics.audioLevel * canvas.height) / 2;
        
        // Color based on activity
        const intensity = voiceMetrics.speechDetected ? 1 : 0.3;
        const hue = isAiSpeaking ? 200 : 120; // Blue for AI, green for user
        
        ctx.fillStyle = `hsla(${hue}, 70%, 50%, ${intensity})`;
        ctx.fillRect(x, centerY - height / 2, barWidth, height);
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();
  };

  // Event listeners setup
  useEffect(() => {
    const setupEventListeners = async () => {
      // Listen for voice pipeline events
      const unlistenVoiceEvents = await listen<VoiceEvent>('voice-pipeline-event', (event) => {
        const { type, data } = event.payload;
        
        switch (type) {
          case 'speech-started':
            setCurrentTranscription('');
            setVoiceMetrics(prev => ({ ...prev, speechDetected: true }));
            setAvatarListening(true);
            break;
            
          case 'speech-stopped':
            setVoiceMetrics(prev => ({ ...prev, speechDetected: false }));
            break;
            
          case 'transcription-delta':
            setCurrentTranscription(data.text);
            break;
            
          case 'transcription-complete':
            setCurrentTranscription('');
            addMessage({
              role: 'user',
              content: data.text,
            });
            setVoiceMetrics(prev => ({ 
              ...prev, 
              conversationTurns: prev.conversationTurns + 1 
            }));
            break;
            
          case 'response-started':
            setIsAiSpeaking(true);
            setAvatarSpeaking(true);
            break;
            
          case 'response-text-complete':
            addMessage({
              role: 'assistant',
              content: data.text,
            });
            break;
            
          case 'audio-synthesis-complete':
            setIsAiSpeaking(false);
            setAvatarSpeaking(false);
            setAvatarListening(true);
            break;
            
          case 'interrupted-by-vad':
            setIsAiSpeaking(false);
            setAvatarSpeaking(false);
            setAvatarListening(true);
            break;
            
          case 'error':
            setLastError(data.message);
            addNotification({
              type: 'error',
              message: `Voice error: ${data.message}`,
            });
            break;
        }
      });

      // Listen for service events
      const unlistenServiceEvents = await listen<{type: string, service?: string}>('unmute-service-event', (event) => {
        const { type, service } = event.payload;
        
        if (type === 'all-services-ready') {
          checkServiceStatus();
        } else if (type.includes('service-started') && service) {
          setServiceStatus(prev => ({ ...prev, [service]: true }));
        } else if (type.includes('service-stopped') && service) {
          setServiceStatus(prev => ({ ...prev, [service]: false }));
        }
      });

      return () => {
        unlistenVoiceEvents();
        unlistenServiceEvents();
      };
    };

    setupEventListeners();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Service status indicator
  const ServiceIndicator: React.FC<{ name: string; active: boolean }> = ({ name, active }) => (
    <div className="flex items-center space-x-2">
      <div className={`w-2 h-2 rounded-full ${active ? 'bg-success' : 'bg-error'}`} />
      <span className={`text-xs ${active ? 'text-success' : 'text-error'}`}>
        {name}
      </span>
    </div>
  );

  return (
    <div className="bg-secondary-bg rounded-card border border-border-primary p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-accent to-accent-light rounded-full flex items-center justify-center">
            <Headphones className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary">Voice Conversation</h3>
            <p className="text-sm text-text-secondary">Real-time speech interaction</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-button hover:bg-surface-hover transition-colors"
            title="Voice settings"
          >
            <Settings className="w-4 h-4 text-text-secondary" />
          </button>
        </div>
      </div>

      {/* Service Status */}
      <div className="mb-6 p-4 bg-surface rounded-card">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-text-primary">Voice Services</h4>
          <div className="flex items-center space-x-2">
            {serviceStatus.overall ? (
              <CheckCircle className="w-4 h-4 text-success" />
            ) : (
              <AlertCircle className="w-4 h-4 text-warning" />
            )}
            <span className={`text-xs font-medium ${serviceStatus.overall ? 'text-success' : 'text-warning'}`}>
              {serviceStatus.overall ? 'Ready' : 'Not Ready'}
            </span>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <ServiceIndicator name="Backend" active={serviceStatus.backend} />
          <ServiceIndicator name="STT" active={serviceStatus.stt} />
          <ServiceIndicator name="TTS" active={serviceStatus.tts} />
        </div>
        
        {!serviceStatus.overall && (
          <div className="mt-3 flex space-x-2">
            <button
              onClick={initializeVoiceServices}
              disabled={isServicesStarting}
              className={`flex items-center space-x-2 px-3 py-2 rounded-button transition-colors ${
                isServicesStarting
                  ? 'bg-surface text-text-tertiary cursor-not-allowed'
                  : 'bg-accent text-white hover:bg-accent-hover'
              }`}
            >
              {isServicesStarting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Power className="w-4 h-4" />
              )}
              <span className="text-sm">
                {isServicesStarting ? 'Starting...' : 'Start Services'}
              </span>
            </button>
            
            {serviceStatus.backend && (
              <button
                onClick={stopVoiceServices}
                className="flex items-center space-x-2 px-3 py-2 bg-error text-white rounded-button hover:bg-error/80 transition-colors"
              >
                <Square className="w-4 h-4" />
                <span className="text-sm">Stop Services</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* Main Voice Controls */}
      <div className="mb-6">
        <div className="flex items-center justify-center space-x-6">
          {/* Main conversation control */}
          <motion.button
            onClick={isVoiceSessionActive ? stopVoiceSession : startVoiceSession}
            disabled={!serviceStatus.overall}
            className={`
              w-20 h-20 rounded-full flex items-center justify-center transition-all
              ${isVoiceSessionActive
                ? 'bg-error text-white hover:bg-error/80 animate-pulse'
                : serviceStatus.overall
                ? 'bg-accent text-white hover:bg-accent-hover'
                : 'bg-surface text-text-tertiary cursor-not-allowed'
              }
            `}
            whileHover={{ scale: serviceStatus.overall ? 1.05 : 1 }}
            whileTap={{ scale: serviceStatus.overall ? 0.95 : 1 }}
          >
            {isVoiceSessionActive ? (
              <MicOff className="w-8 h-8" />
            ) : (
              <Mic className="w-8 h-8" />
            )}
          </motion.button>
          
          {/* Mute/unmute */}
          <button
            className="p-3 rounded-button hover:bg-surface-hover transition-colors"
            title="Toggle audio output"
          >
            <Volume2 className="w-5 h-5 text-text-secondary" />
          </button>
        </div>
        
        <div className="text-center mt-4">
          <p className="text-sm text-text-secondary">
            {isVoiceSessionActive
              ? 'Voice conversation active - speak naturally'
              : serviceStatus.overall
              ? 'Click to start voice conversation'
              : 'Start voice services first'
            }
          </p>
        </div>
      </div>

      {/* Audio Visualization */}
      {isVoiceSessionActive && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-6"
        >
          <div className="bg-surface rounded-card p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium text-text-primary">Audio Activity</h4>
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-accent" />
                <span className="text-xs text-text-secondary">
                  {voiceMetrics.speechDetected ? 'Speech detected' : 'Listening...'}
                </span>
              </div>
            </div>
            
            <canvas
              ref={audioVisualizationRef}
              width={320}
              height={60}
              className="w-full h-15 bg-primary-bg rounded"
            />
          </div>
        </motion.div>
      )}

      {/* Current Transcription */}
      <AnimatePresence>
        {currentTranscription && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-6 p-4 bg-accent/10 border border-accent/20 rounded-card"
          >
            <div className="flex items-center space-x-2 mb-2">
              <Waveform className="w-4 h-4 text-accent" />
              <span className="text-sm font-medium text-accent">Transcribing...</span>
            </div>
            <p className="text-text-primary italic">"{currentTranscription}"</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* AI Speaking Indicator */}
      <AnimatePresence>
        {isAiSpeaking && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-6 p-4 bg-success/10 border border-success/20 rounded-card"
          >
            <div className="flex items-center space-x-2">
              <Speaker className="w-4 h-4 text-success animate-pulse" />
              <span className="text-sm font-medium text-success">AI is responding...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Metrics */}
      {isVoiceSessionActive && (
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-surface p-3 rounded-card">
            <div className="text-xs text-text-tertiary mb-1">Conversation Turns</div>
            <div className="text-lg font-semibold text-text-primary">
              {voiceMetrics.conversationTurns}
            </div>
          </div>
          <div className="bg-surface p-3 rounded-card">
            <div className="text-xs text-text-tertiary mb-1">Processing Latency</div>
            <div className="text-lg font-semibold text-text-primary">
              {voiceMetrics.processingLatency}ms
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      <AnimatePresence>
        {lastError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-4 p-4 bg-error/10 border border-error/20 rounded-card"
          >
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-error flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-error mb-1">Voice Error</p>
                <p className="text-xs text-text-secondary">{lastError}</p>
              </div>
              <button
                onClick={() => setLastError(null)}
                className="text-error hover:text-error/80 transition-colors"
              >
                ×
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-border-primary pt-4"
          >
            <h4 className="text-sm font-medium text-text-primary mb-3">Voice Settings</h4>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-text-secondary">Voice Character</label>
                <select className="w-full mt-1 px-3 py-2 bg-surface border border-border-primary rounded text-sm text-text-primary">
                  <option>Default Assistant</option>
                  <option>Friendly Guide</option>
                  <option>Professional Expert</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-text-secondary">Speech Sensitivity</label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  defaultValue="0.6"
                  className="w-full mt-1"
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-secondary">Enable Interruption</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" defaultChecked className="sr-only peer" />
                  <div className="w-11 h-6 bg-surface peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
                </label>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default VoiceConversation;