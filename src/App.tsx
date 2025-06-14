import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import Chat from "./components/Chat";
import Avatar from "./components/Avatar";
import Controls from "./components/Controls";
import { LoadingScreen } from "./components/LoadingScreen";
import "./styles/App.css";

interface ModelStatus {
  loaded: boolean;
  model: string;
  device: string;
}

interface LoadingState {
  isVisible: boolean;
  progress: number;
  status: string;
  modelName?: string;
}

function App() {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [currentViseme] = useState("neutral");
  const [loadingState, setLoadingState] = useState<LoadingState>({
    isVisible: false,
    progress: 0,
    status: "Initializing...",
  });

  useEffect(() => {
    // Check model status on load
    checkModelStatus();
    
    // Listen for model loading progress events
    const setupEventListeners = async () => {
      await listen<{ progress: number; status: string; model_name?: string }>(
        'model-loading-progress', 
        (event) => {
          setLoadingState({
            isVisible: true,
            progress: event.payload.progress,
            status: event.payload.status,
            modelName: event.payload.model_name,
          });
        }
      );

      await listen<{ success: boolean; error?: string }>(
        'model-loading-complete',
        (event) => {
          if (event.payload.success) {
            setLoadingState(prev => ({
              ...prev,
              isVisible: false,
              progress: 100,
              status: "Complete!",
            }));
            // Refresh model status
            checkModelStatus();
          } else {
            setLoadingState(prev => ({
              ...prev,
              status: `Error: ${event.payload.error || 'Unknown error'}`,
            }));
          }
        }
      );
    };

    setupEventListeners();
  }, []);

  const checkModelStatus = async () => {
    try {
      const status = await invoke<ModelStatus>("get_model_status");
      setModelStatus(status);
    } catch (error) {
      console.error("Failed to get model status:", error);
    }
  };

  const loadModel = async (modelName: string) => {
    setLoadingState({
      isVisible: true,
      progress: 0,
      status: "Starting model download...",
      modelName,
    });
    
    try {
      await invoke("load_model", { modelName });
      // The completion will be handled by the event listener
    } catch (error) {
      console.error("Failed to load model:", error);
      setLoadingState(prev => ({
        ...prev,
        status: `Error: ${error}`,
      }));
    }
  };

  const sendMessage = async (message: string) => {
    // Add user message to chat
    setMessages(prev => [...prev, { role: "user", content: message }]);
    setIsLoading(true);

    try {
      const response = await invoke<string>("send_message", { message });
      
      // Add AI response to chat
      setMessages(prev => [...prev, { role: "assistant", content: response }]);
      
      // TODO: Process response for TTS and lip-sync
      // This would trigger avatar animation
      
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages(prev => [...prev, { 
        role: "system", 
        content: "Error: Failed to process message" 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleVoiceInput = async () => {
    try {
      if (isVoiceActive) {
        // Stop recording and transcribe
        await invoke("stop_voice_input");
        
        // Get transcription and process it
        const transcription = await invoke<string>("transcribe_voice");
        if (transcription && transcription.trim()) {
          await sendMessage(transcription);
        }
        
        // Clear the voice buffer for next recording
        await invoke("clear_voice_buffer");
      } else {
        // Start recording
        await invoke("start_voice_input");
      }
      setIsVoiceActive(!isVoiceActive);
    } catch (error) {
      console.error("Failed to toggle voice input:", error);
    }
  };

  const captureImage = async () => {
    try {
      const imageData = await invoke<number[]>("capture_image");
      // TODO: Process captured image
      console.log("Image captured:", imageData.length, "bytes");
    } catch (error) {
      console.error("Failed to capture image:", error);
    }
  };

  return (
    <div className="app">
      <LoadingScreen
        isLoading={loadingState.isVisible}
        progress={loadingState.progress}
        status={loadingState.status}
        modelName={loadingState.modelName}
      />
      
      <div className="app-header">
        <h1>Tektra AI Assistant</h1>
        {modelStatus && (
          <div className="model-status">
            Model: {modelStatus.model} | Device: {modelStatus.device}
          </div>
        )}
      </div>
      
      <div className="app-content">
        <div className="avatar-section">
          <Avatar 
            currentViseme={currentViseme}
            isListening={isVoiceActive}
            isSpeaking={isLoading}
          />
        </div>
        
        <div className="chat-section">
          <Chat 
            messages={messages}
            onSendMessage={sendMessage}
            isLoading={isLoading}
          />
        </div>
      </div>
      
      <Controls
        modelStatus={modelStatus}
        onLoadModel={loadModel}
        onToggleVoice={toggleVoiceInput}
        onCaptureImage={captureImage}
        isVoiceActive={isVoiceActive}
        isLoading={isLoading}
      />
    </div>
  );
}

export default App;