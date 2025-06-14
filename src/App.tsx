import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import Chat from "./components/Chat";
import Avatar from "./components/Avatar";
import Controls from "./components/Controls";
import "./styles/App.css";

interface ModelStatus {
  loaded: boolean;
  model: string;
  device: string;
}

function App() {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [currentViseme] = useState("neutral");

  useEffect(() => {
    // Check model status on load
    checkModelStatus();
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
    setIsLoading(true);
    try {
      await invoke("load_model", { modelName });
      await checkModelStatus();
    } catch (error) {
      console.error("Failed to load model:", error);
    } finally {
      setIsLoading(false);
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
        await invoke("stop_voice_input");
      } else {
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