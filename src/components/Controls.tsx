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
  const [downloadProgress, setDownloadProgress] = useState<{
    model: string;
    downloading: boolean;
  } | null>(null);
  const [showDownloadPanel, setShowDownloadPanel] = useState(false);

  const handleLoadModel = () => {
    onLoadModel(selectedModel);
  };

  const loadCachedModels = async () => {
    try {
      const result = await invoke<{ success: boolean; models: CachedModel[] }>(
        'list_cached_models'
      );
      if (result.success) {
        setCachedModels(result.models);
      }
    } catch (error) {
      console.error('Failed to load cached models:', error);
    }
  };

  const handleDownloadModel = async (modelName: string, force = false) => {
    try {
      setDownloadProgress({ model: modelName, downloading: true });
      
      const result = await invoke<{ success: boolean; message?: string; error?: string }>(
        'download_model',
        { modelName, force }
      );
      
      if (result.success) {
        console.log(`Model downloaded: ${result.message}`);
        await loadCachedModels(); // Refresh cached models list
      } else {
        console.error(`Download failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Download error:', error);
    } finally {
      setDownloadProgress(null);
    }
  };

  useEffect(() => {
    if (showSettings) {
      loadCachedModels();
    }
  }, [showSettings]);

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
        
        <button
          onClick={() => setShowDownloadPanel(!showDownloadPanel)}
          className="control-button"
        >
          📥 Models
        </button>
      </div>
      
      {showSettings && (
        <div className="settings-panel">
          <h3>Model Settings</h3>
          <div className="model-selector">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
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
        </div>
      )}
      
      {showDownloadPanel && (
        <div className="download-panel">
          <h3>Model Management</h3>
          
          <div className="download-section">
            <h4>Download New Model</h4>
            <div className="download-controls">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={downloadProgress?.downloading}
              >
                {AVAILABLE_MODELS.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <button
                onClick={() => handleDownloadModel(selectedModel)}
                disabled={downloadProgress?.downloading}
                className="download-button"
              >
                {downloadProgress?.model === selectedModel && downloadProgress.downloading
                  ? "Downloading..."
                  : "Download"}
              </button>
            </div>
            {downloadProgress?.downloading && (
              <div className="download-progress">
                <p>Downloading {downloadProgress.model}...</p>
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
              </div>
            )}
          </div>
          
          <div className="cached-models-section">
            <h4>Cached Models ({cachedModels.length})</h4>
            {cachedModels.length === 0 ? (
              <p>No models cached yet. Download a model to get started.</p>
            ) : (
              <div className="cached-models-list">
                {cachedModels.map((model) => (
                  <div key={model.path} className="cached-model-item">
                    <div className="model-details">
                      <strong>{model.name}</strong>
                      <span className="model-size">{model.size_gb} GB</span>
                    </div>
                    <div className="model-actions">
                      <button
                        onClick={() => onLoadModel(model.name)}
                        disabled={isLoading || modelStatus?.model === model.name}
                        className="load-cached-button"
                      >
                        {modelStatus?.model === model.name ? "Loaded" : "Load"}
                      </button>
                      <button
                        onClick={() => handleDownloadModel(model.name, true)}
                        disabled={downloadProgress?.downloading}
                        className="redownload-button"
                        title="Force re-download"
                      >
                        🔄
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