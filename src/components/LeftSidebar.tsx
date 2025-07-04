import React, { useState, useEffect } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Brain,
  User,
  Mic,
  Camera,
  Settings,
  PanelLeft,
  PanelLeftClose,
  X,
  Check,
  Loader2,
} from 'lucide-react';
import { useTektraStore } from '../store';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

const LeftSidebar: React.FC = () => {
  // Use individual selectors instead of destructuring - this is more reliable
  const uiState = useTektraStore((state) => state.uiState);
  const modelStatus = useTektraStore((state) => state.modelStatus);
  const avatarState = useTektraStore((state) => state.avatarState);
  const toggleLeftSidebar = useTektraStore((state) => state.toggleLeftSidebar);
  const setModelStatus = useTektraStore((state) => state.setModelStatus);
  const setAvatarState = useTektraStore((state) => state.setAvatarState);
  
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(['ai-model'])
  );
  
  // Model selection state
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  
  // Progress tracking state
  const [downloadProgress, setDownloadProgress] = useState<{
    modelId: string | null;
    progress: number;
    status: string;
    stage: string;
  }>({
    modelId: null,
    progress: 0,
    status: '',
    stage: 'idle'
  });

  const toggleSection = (sectionId: string) => {
    const newOpenSections = new Set(openSections);
    if (newOpenSections.has(sectionId)) {
      newOpenSections.delete(sectionId);
    } else {
      newOpenSections.add(sectionId);
    }
    setOpenSections(newOpenSections);
  };

  // Load available models when component mounts and setup event listeners
  useEffect(() => {
    loadCurrentModel();
    
    // Setup progress event listener
    const setupProgressListener = async () => {
      await listen<{
        model_id: string;
        progress: number;
        status: string;
        stage: string;
      }>('model-loading-progress', (event) => {
        const { model_id, progress, status, stage } = event.payload;
        setDownloadProgress({
          modelId: model_id,
          progress,
          status,
          stage
        });
        
        // Update model status based on progress
        if (stage === 'complete') {
          // Find the model name for this ID
          const selectedModel = availableModels.find(m => m.id === model_id);
          if (selectedModel) {
            setModelStatus({
              ...modelStatus,
              modelName: selectedModel.name,
              isLoaded: true,
              isLoading: false,
              backend: 'mistral.rs'
            });
          }
          setLoadingModel(null);
          setShowModelSelector(false);
        } else if (stage === 'error') {
          setModelStatus({
            ...modelStatus,
            isLoading: false,
            isLoaded: false
          });
          setLoadingModel(null);
        }
      });
    };
    
    setupProgressListener();
  }, [availableModels, modelStatus, setModelStatus]);

  const loadCurrentModel = async () => {
    try {
      const currentModelId = await invoke<string | null>('get_current_model');
      if (currentModelId) {
        // Get the available models to find the display name
        const models = await invoke<any[]>('get_available_models');
        const currentModel = models.find(m => m.id === currentModelId);
        
        setModelStatus({
          ...modelStatus,
          modelName: currentModel ? currentModel.name : currentModelId,
          isLoaded: true,
          backend: 'mistral.rs'
        });
      } else {
        // No active model, set default state
        setModelStatus({
          ...modelStatus,
          modelName: 'No model loaded',
          isLoaded: false,
          backend: 'mistral.rs'
        });
      }
    } catch (error) {
      console.error('Failed to load current model:', error);
      setModelStatus({
        ...modelStatus,
        modelName: 'Error loading model',
        isLoaded: false,
        backend: 'mistral.rs'
      });
    }
  };

  const fetchAvailableModels = async () => {
    setLoadingModels(true);
    try {
      const models = await invoke<any[]>('get_available_models');
      setAvailableModels(models);
    } catch (error) {
      console.error('Failed to fetch available models:', error);
      setAvailableModels([]);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleChangeModel = async () => {
    setShowModelSelector(true);
    await fetchAvailableModels();
  };

  const handleSelectModel = async (modelId: string, modelName: string) => {
    setLoadingModel(modelId);
    try {
      // Update UI immediately to show loading
      setModelStatus({
        ...modelStatus,
        modelName: modelName,
        isLoading: true,
        isLoaded: false
      });

      await invoke('load_model', { model_id: modelId });
      
      // Update final state
      setModelStatus({
        ...modelStatus,
        modelName: modelName,
        isLoaded: true,
        isLoading: false,
        backend: 'mistral.rs'
      });

      setShowModelSelector(false);
    } catch (error) {
      console.error('Failed to load model:', error);
      // Revert UI state on error
      setModelStatus({
        ...modelStatus,
        isLoading: false,
        isLoaded: false
      });
    } finally {
      setLoadingModel(null);
    }
  };

  // Collapsed sidebar
  if (uiState.leftSidebarCollapsed) {
    return (
      <aside className="fixed left-0 top-16 bottom-8 z-40 w-16 bg-secondary-bg border-r border-border-primary flex flex-col items-center py-4 space-y-4">
        <button
          onClick={toggleLeftSidebar}
          className="p-3 rounded-button hover:bg-surface-hover transition-colors group"
          title="Expand sidebar"
        >
          <PanelLeft className="w-5 h-5 text-text-secondary group-hover:text-text-primary transition-colors" />
        </button>
        
        <div className="space-y-3">
          <div className="p-2 rounded-button hover:bg-surface-hover transition-colors" title="AI Model">
            <Brain className="w-5 h-5 text-accent" />
          </div>
          <div className="p-2 rounded-button hover:bg-surface-hover transition-colors" title="Avatar">
            <User className="w-5 h-5 text-text-secondary" />
          </div>
          <div className="p-2 rounded-button hover:bg-surface-hover transition-colors" title="Input">
            <Mic className="w-5 h-5 text-text-secondary" />
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="fixed left-0 top-16 bottom-8 z-40 w-80 bg-secondary-bg border-r border-border-primary flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border-primary">
        <h2 className="font-semibold text-text-primary">Configuration</h2>
        <button
          onClick={toggleLeftSidebar}
          className="p-2 rounded-button hover:bg-surface-hover transition-colors"
        >
          <PanelLeftClose className="w-4 h-4 text-text-secondary" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {/* AI Model Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('ai-model')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <Brain className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">AI Model</span>
            </div>
            <div className={`transform transition-transform duration-200 ${
              openSections.has('ai-model') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('ai-model') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Current Model
                </label>
                <div className="p-3 bg-surface rounded-button border border-border-primary">
                  <p className="text-sm text-text-primary">{modelStatus.modelName}</p>
                  <p className="text-xs text-text-tertiary">{modelStatus.backend}</p>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Status
                </label>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      modelStatus.isLoaded ? 'bg-success' : 
                      modelStatus.isLoading || downloadProgress.stage !== 'idle' ? 'bg-accent' : 'bg-error'
                    }`} />
                    <span className="text-sm text-text-secondary">
                      {modelStatus.isLoaded ? 'Ready' : 
                       modelStatus.isLoading || downloadProgress.stage !== 'idle' ? 'Loading' : 'Not Loaded'}
                    </span>
                  </div>
                  
                  {/* Progress bar when loading */}
                  {(modelStatus.isLoading || downloadProgress.stage !== 'idle') && (
                    <div className="space-y-1">
                      <div className="w-full bg-surface rounded-full h-2">
                        <div 
                          className="bg-accent h-2 rounded-full transition-all duration-300" 
                          style={{ width: `${downloadProgress.progress}%` }}
                        />
                      </div>
                      <div className="text-xs text-text-tertiary">
                        {downloadProgress.status || 'Loading...'}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <button 
                onClick={handleChangeModel}
                disabled={modelStatus.isLoading}
                className="w-full p-2 bg-accent hover:bg-accent-hover disabled:bg-accent/50 text-white rounded-button transition-colors text-sm flex items-center justify-center gap-2"
              >
                {modelStatus.isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading...
                  </>
                ) : (
                  'Change Model'
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Avatar Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('avatar')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <User className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">Avatar</span>
            </div>
            <div className={`transform transition-transform duration-200 ${
              openSections.has('avatar') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('avatar') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Expression
                </label>
                <select 
                  value={avatarState.expression}
                  onChange={(e) => setAvatarState({ expression: e.target.value as any })}
                  className="w-full p-2 bg-surface border border-border-primary rounded-button text-text-primary"
                >
                  <option value="neutral">Neutral</option>
                  <option value="happy">Happy</option>
                  <option value="thinking">Thinking</option>
                  <option value="surprised">Surprised</option>
                  <option value="friendly">Friendly</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Style
                </label>
                <select 
                  value={avatarState.appearance.style}
                  onChange={(e) => setAvatarState({ 
                    appearance: { ...avatarState.appearance, style: e.target.value as any }
                  })}
                  className="w-full p-2 bg-surface border border-border-primary rounded-button text-text-primary"
                >
                  <option value="realistic">Realistic</option>
                  <option value="stylized">Stylized</option>
                  <option value="minimal">Minimal</option>
                </select>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-text-primary">Eye Tracking</span>
                <input 
                  type="checkbox" 
                  checked={avatarState.animation.eyeTracking}
                  onChange={(e) => setAvatarState({ 
                    animation: { ...avatarState.animation, eyeTracking: e.target.checked }
                  })}
                  className="w-4 h-4"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Input Modes */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('input-modes')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <Mic className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">Input Modes</span>
            </div>
            <div className={`transform transition-transform duration-200 ${
              openSections.has('input-modes') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('input-modes') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Mic className="w-4 h-4 text-accent" />
                  <span className="text-sm text-text-primary">Voice Input</span>
                </div>
                <input 
                  type="checkbox" 
                  checked={modelStatus.whisperReady}
                  onChange={(e) => setModelStatus({ whisperReady: e.target.checked })}
                  className="w-4 h-4"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Camera className="w-4 h-4 text-accent" />
                  <span className="text-sm text-text-primary">Camera Input</span>
                </div>
                <input 
                  type="checkbox" 
                  checked={modelStatus.cameraEnabled}
                  onChange={(e) => setModelStatus({ cameraEnabled: e.target.checked })}
                  className="w-4 h-4"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Tools Section */}
        <div className="border border-border-primary rounded-card bg-surface/50 overflow-hidden">
          <button
            onClick={() => toggleSection('tools')}
            className="w-full flex items-center justify-between p-4 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-center space-x-3">
              <Settings className="w-5 h-5 text-accent" />
              <span className="font-medium text-text-primary">Tools</span>
            </div>
            <div className={`transform transition-transform duration-200 ${
              openSections.has('tools') ? 'rotate-90' : 'rotate-0'
            }`}>
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </div>
          </button>
          
          <div className={`transition-all duration-200 ease-in-out ${
            openSections.has('tools') ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
          } overflow-hidden border-t border-border-primary`}>
            <div className="p-4 space-y-3">
              <div className="p-3 bg-surface rounded-button border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Web Search</span>
                  <span className="text-xs text-success">Available</span>
                </div>
              </div>

              <div className="p-3 bg-surface rounded-button border border-border-primary">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary">Code Execution</span>
                  <span className="text-xs text-success">Available</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Selection Modal */}
      {showModelSelector && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-primary-bg border border-border-primary rounded-card w-96 max-h-96 overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-border-primary">
              <h3 className="text-lg font-semibold text-text-primary">Select Model</h3>
              <button
                onClick={() => setShowModelSelector(false)}
                className="p-1 hover:bg-surface-hover rounded-button transition-colors"
              >
                <X className="w-5 h-5 text-text-secondary" />
              </button>
            </div>
            
            <div className="max-h-80 overflow-y-auto">
              {loadingModels ? (
                <div className="flex items-center justify-center p-8">
                  <Loader2 className="w-8 h-8 animate-spin text-accent" />
                  <span className="ml-2 text-text-secondary">Loading models...</span>
                </div>
              ) : availableModels.length > 0 ? (
                <div className="p-2 space-y-2">
                  {availableModels.map((model) => (
                    <div
                      key={model.id}
                      className="border border-border-primary rounded-button hover:bg-surface-hover transition-colors"
                    >
                      <button
                        onClick={() => handleSelectModel(model.id, model.name)}
                        disabled={loadingModel === model.id}
                        className="w-full p-3 text-left disabled:opacity-50"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium text-text-primary">{model.name}</h4>
                              {model.default && (
                                <span className="text-xs bg-accent text-white px-2 py-1 rounded">
                                  Default
                                </span>
                              )}
                              {modelStatus.modelName === model.name && (
                                <Check className="w-4 h-4 text-success" />
                              )}
                            </div>
                            <p className="text-sm text-text-secondary mt-1">
                              {model.description}
                            </p>
                            <div className="flex flex-wrap gap-1 mt-2">
                              {model.supports_vision && (
                                <span className="text-xs bg-surface text-text-secondary px-2 py-1 rounded">
                                  Vision
                                </span>
                              )}
                              {model.supports_audio && (
                                <span className="text-xs bg-surface text-text-secondary px-2 py-1 rounded">
                                  Audio
                                </span>
                              )}
                              {model.supports_documents && (
                                <span className="text-xs bg-surface text-text-secondary px-2 py-1 rounded">
                                  Documents
                                </span>
                              )}
                            </div>
                          </div>
                          {loadingModel === model.id && (
                            <Loader2 className="w-5 h-5 animate-spin text-accent ml-2" />
                          )}
                        </div>
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center p-8">
                  <span className="text-text-secondary">No models available</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};

export default LeftSidebar;