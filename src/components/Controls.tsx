import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import "../styles/Controls.css";

interface CachedModel {
  name: string;
  path: string;
  size_bytes: number;
  size_gb: number;
}

interface ControlsProps {
  modelStatus: { loaded: boolean; model: string; device: string } | null;
  onLoadModel: (modelName: string) => void;
  onToggleVoice: () => void;
  onCaptureImage: () => void;
  isVoiceActive: boolean;
  isLoading: boolean;
}

// Models will be loaded from backend
const DEFAULT_MODELS = [
  "mlx-community/SmolLM2-1.7B-Instruct-4bit",
];

const Controls: React.FC<ControlsProps> = ({
  modelStatus,
  onLoadModel,
  onToggleVoice,
  onCaptureImage,
  isVoiceActive,
  isLoading,
}) => {
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODELS[0]);
  const [availableModels, setAvailableModels] = useState<string[]>(DEFAULT_MODELS);
  const [showSettings, setShowSettings] = useState(false);
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);
  const [audioInfo, setAudioInfo] = useState<string>("");

  const handleLoadModel = () => {
    onLoadModel(selectedModel);
  };

  const loadCachedModels = async () => {
    try {
      console.log('🗂️ CONTROLS: Loading cached models...');
      const result = await invoke<{ success: boolean; models: CachedModel[] }>(
        'list_cached_models'
      );
      console.log('📋 CONTROLS: Cached models result:', result);
      if (result.success) {
        console.log('Controls: Found', result.models.length, 'cached models');
        setCachedModels(result.models);
      } else {
        console.warn('Controls: Failed to load cached models - success=false');
      }
    } catch (error) {
      console.error('Controls: Failed to load cached models:', error);
    }
  };

  const loadAudioInfo = async () => {
    try {
      console.log('🎵 CONTROLS: Loading audio info...');
      const info = await invoke<string>('get_audio_info');
      console.log('🎵 CONTROLS: Audio info received:', info);
      setAudioInfo(info);
    } catch (error) {
      console.error('Controls: Failed to load audio info:', error);
      setAudioInfo("❌ Audio system not available");
    }
  };

  const loadAvailableModels = async () => {
    try {
      const models = await invoke<string[]>('get_available_models');
      setAvailableModels(models);
      
      // Update selected model if current one is not in the new list
      if (!models.includes(selectedModel)) {
        setSelectedModel(models[0] || DEFAULT_MODELS[0]);
      }
    } catch (error) {
      console.error('Controls: Failed to load available models:', error);
      setAvailableModels(DEFAULT_MODELS);
    }
  };


  // Load available models and last selected model on component mount
  useEffect(() => {
    const initializeModels = async () => {
      if (!isInitialized) {
        try {
          // First load available models
          console.log('🤖 CONTROLS: Loading available models from backend...');
          const models = await invoke<string[]>('get_available_models');
          console.log('🤖 CONTROLS: Available models received:', models);
          setAvailableModels(models);
          
          // Then load last selected model
          console.log('🔄 CONTROLS: Loading last selected model...');
          const lastModel = await invoke<string | null>('get_last_selected_model');
          console.log('📋 CONTROLS: Last selected model:', lastModel);
          
          if (lastModel && models.includes(lastModel)) {
            console.log('✅ CONTROLS: Setting selected model to:', lastModel);
            setSelectedModel(lastModel);
          } else {
            console.log('ℹ️ CONTROLS: Using default model (no valid last selection)');
            setSelectedModel(models[0] || DEFAULT_MODELS[0]);
          }
        } catch (error) {
          console.error('❌ CONTROLS: Failed to initialize models:', error);
          setAvailableModels(DEFAULT_MODELS);
        } finally {
          setIsInitialized(true);
        }
      }
    };

    initializeModels();
  }, [isInitialized]);

  useEffect(() => {
    if (showSettings) {
      loadCachedModels();
      loadAudioInfo();
      loadAvailableModels();
    }
  }, [showSettings]);

  useEffect(() => {
    // Listen for model load completion to refresh cached models
    const handleModelLoadComplete = () => {
      console.log('Controls: Model load completed, refreshing cached models');
      loadCachedModels();
    };

    // Listen for close settings event
    const handleCloseSettings = () => {
      console.log('Controls: Closing settings panel after model load');
      setShowSettings(false);
    };

    window.addEventListener('modelLoadComplete', handleModelLoadComplete);
    window.addEventListener('closeSettings', handleCloseSettings);
    
    return () => {
      window.removeEventListener('modelLoadComplete', handleModelLoadComplete);
      window.removeEventListener('closeSettings', handleCloseSettings);
    };
  }, []);

  return (
    <div className="controls-container">
      <div className="controls-main">
        <button
          onClick={onToggleVoice}
          className={`control-button ${isVoiceActive ? "active" : ""}`}
          disabled={isLoading}
        >
          {isVoiceActive ? "🎙️ Stop Voice" : "🎙️ Start Voice"}
        </button>
        
        <button
          onClick={onCaptureImage}
          className="control-button"
          disabled={isLoading}
        >
          📷 Capture Image
        </button>
        
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="control-button"
        >
          ⚙️ Settings
        </button>
        
      </div>
      
      {showSettings && (
        <div className="settings-panel">
          <h3>Model Settings</h3>
          <div className="model-selector">
            <select
              value={selectedModel}
              onChange={async (e) => {
                const newModel = e.target.value;
                console.log('🔄 CONTROLS: Model selection changed to:', newModel);
                setSelectedModel(newModel);
                
                // Persist the selection immediately
                try {
                  await invoke('set_last_selected_model', { modelName: newModel });
                  console.log('💾 CONTROLS: Saved model selection:', newModel);
                } catch (error) {
                  console.error('❌ CONTROLS: Failed to save model selection:', error);
                }
              }}
              disabled={isLoading}
            >
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <button
              onClick={handleLoadModel}
              disabled={isLoading || (modelStatus?.model === selectedModel)}
              className="load-model-button"
            >
              Load Model
            </button>
          </div>
          
          {modelStatus && (
            <div className="model-info">
              <p>Current Model: {modelStatus.model}</p>
              <p>Device: {modelStatus.device}</p>
              <p>Status: {modelStatus.loaded ? "✅ Loaded" : "❌ Not Loaded"}</p>
            </div>
          )}
          
          <div className="cached-models-section">
            <h4>Cached Models ({cachedModels.length})</h4>
            {cachedModels.length === 0 ? (
              <p>No models cached yet. Load a model to cache it.</p>
            ) : (
              <div className="cached-models-list">
                {cachedModels.map((model) => (
                  <div key={model.path} className="cached-model-item">
                    <div className="model-details">
                      <strong>{model.name}</strong>
                      <span className="model-size">{model.size_gb.toFixed(1)} GB</span>
                    </div>
                    <div className="model-actions">
                      <button
                        onClick={async () => {
                          console.log('🔄 CONTROLS: Loading cached model:', model.name);
                          setSelectedModel(model.name);
                          
                          // Persist the selection
                          try {
                            await invoke('set_last_selected_model', { modelName: model.name });
                            console.log('💾 CONTROLS: Saved cached model selection:', model.name);
                          } catch (error) {
                            console.error('❌ CONTROLS: Failed to save cached model selection:', error);
                          }
                          
                          onLoadModel(model.name);
                        }}
                        disabled={isLoading || modelStatus?.model === model.name}
                        className="load-cached-button"
                      >
                        {modelStatus?.model === model.name ? "✅ Loaded" : "Load"}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          
          <div className="audio-status-section">
            <h4>Audio System Status</h4>
            {audioInfo ? (
              <div className="audio-info">
                {audioInfo.split('\n').map((line, index) => (
                  <p key={index} className="audio-info-line">{line}</p>
                ))}
              </div>
            ) : (
              <p>Loading audio information...</p>
            )}
          </div>
        </div>
      )}
      
    </div>
  );
};

export default Controls;