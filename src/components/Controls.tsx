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

const AVAILABLE_MODELS = [
  "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "mlx-community/Phi-3.5-mini-instruct-4bit",
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
  const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0]);
  const [showSettings, setShowSettings] = useState(false);
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);

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


  // Load last selected model on component mount
  useEffect(() => {
    const loadLastSelectedModel = async () => {
      if (!isInitialized) {
        try {
          console.log('🔄 CONTROLS: Loading last selected model...');
          const lastModel = await invoke<string | null>('get_last_selected_model');
          console.log('📋 CONTROLS: Last selected model:', lastModel);
          
          if (lastModel && AVAILABLE_MODELS.includes(lastModel)) {
            console.log('✅ CONTROLS: Setting selected model to:', lastModel);
            setSelectedModel(lastModel);
          } else {
            console.log('ℹ️ CONTROLS: Using default model (no valid last selection)');
          }
        } catch (error) {
          console.error('❌ CONTROLS: Failed to load last selected model:', error);
        } finally {
          setIsInitialized(true);
        }
      }
    };

    loadLastSelectedModel();
  }, [isInitialized]);

  useEffect(() => {
    if (showSettings) {
      loadCachedModels();
    }
  }, [showSettings]);

  useEffect(() => {
    // Listen for model load completion to refresh cached models
    const handleModelLoadComplete = () => {
      console.log('Controls: Model load completed, refreshing cached models');
      loadCachedModels();
    };

    window.addEventListener('modelLoadComplete', handleModelLoadComplete);
    return () => window.removeEventListener('modelLoadComplete', handleModelLoadComplete);
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
              {AVAILABLE_MODELS.map((model) => (
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
        </div>
      )}
      
    </div>
  );
};

export default Controls;