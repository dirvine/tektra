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
    console.log('🚀 REACT APP STARTED - App.tsx useEffect running');
    
    // Check model status on load
    checkModelStatus();
    
    let progressUnlisten: (() => void) | null = null;
    let completeUnlisten: (() => void) | null = null;
    
    // Set up event listeners
    const setupListeners = async () => {
      try {
        progressUnlisten = await listen<{ progress: number; status: string; model_name?: string }>(
          'model-loading-progress', 
          (event) => {
            console.log('FRONTEND: Progress event received:', event.payload);
            
            setLoadingState(prev => ({
              ...prev,
              isVisible: true,
              progress: event.payload.progress,
              status: event.payload.status,
              modelName: event.payload.model_name,
            }));
          }
        );

        completeUnlisten = await listen<{ success: boolean; error?: string }>(
          'model-loading-complete',
          (event) => {
            console.log('FRONTEND: Completion event received:', event.payload);
            
            if (event.payload.success) {
              console.log("FRONTEND: SUCCESS - Starting hide sequence");
              
              // Show completion status briefly
              setLoadingState(prev => ({
                ...prev,
                progress: 100,
                status: "Complete!",
              }));
              
              // Hide loading screen after brief delay
              setTimeout(() => {
                console.log("FRONTEND: Hiding loading screen now");
                setLoadingState(prev => ({
                  ...prev,
                  isVisible: false,
                }));
                
                // Refresh model status
                checkModelStatus();
                
                // Trigger a refresh of cached models in controls
                window.dispatchEvent(new CustomEvent('modelLoadComplete'));
                
                // Close settings panel after successful model load
                window.dispatchEvent(new CustomEvent('closeSettings'));
              }, 1500);
              
            } else {
              console.log("FRONTEND: ERROR - Showing error");
              setLoadingState(prev => ({
                ...prev,
                status: `Error: ${event.payload.error || 'Unknown error'}`,
                progress: 0,
              }));
            }
          }
        );
        
        console.log('App: Event listeners set up successfully');
      } catch (error) {
        console.error('App: Failed to set up event listeners:', error);
      }
    };
    
    setupListeners();
    
    // Cleanup function
    return () => {
      console.log('App: Cleaning up event listeners');
      if (progressUnlisten) progressUnlisten();
      if (completeUnlisten) completeUnlisten();
    };
  }, []);

  const checkModelStatus = async () => {
    try {
      console.log('📡 Checking model status...');
      const status = await invoke<ModelStatus>("get_model_status");
      console.log('✅ Model status received:', status);
      setModelStatus(status);
    } catch (error) {
      console.error("❌ Failed to get model status:", error);
    }
  };

  const loadModel = async (modelName: string) => {
    console.log("🔄 LOAD MODEL CALLED:", modelName);
    
    setLoadingState({
      isVisible: true,
      progress: 0,
      status: "Initializing...",
      modelName,
    });
    
    // Set up a timeout to hide loading screen if event doesn't come through
    const timeoutId = setTimeout(() => {
      console.log("⏰ FRONTEND: Timeout reached, forcing loading screen to close");
      setLoadingState(prev => ({
        ...prev,
        isVisible: false,
      }));
      checkModelStatus();
    }, 600000); // 10 minute timeout for large models
    
    try {
      console.log("App: Invoking load_model command");
      await invoke("load_model", { modelName });
      console.log("App: load_model command completed - waiting for events");
      
      // If the model was already cached, the events might not fire reliably
      // Check status after a short delay to ensure UI updates
      setTimeout(async () => {
        await checkModelStatus();
        const status = await invoke<ModelStatus>("get_model_status");
        if (status.loaded && status.model === modelName) {
          console.log("🔄 FRONTEND: Model loaded, clearing loading screen as backup");
          clearTimeout(timeoutId);
          setLoadingState(prev => ({
            ...prev,
            isVisible: false,
          }));
        }
      }, 2000);
      
    } catch (error) {
      console.error("App: Failed to load model:", error);
      clearTimeout(timeoutId);
      setLoadingState(prev => ({
        ...prev,
        status: `Error: ${error}`,
        isVisible: false,
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
          console.log("🎤 Voice transcription:", transcription);
          await sendMessage(transcription);
        } else {
          console.log("🔇 No speech detected in recording");
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