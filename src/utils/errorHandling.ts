import { invoke } from '@tauri-apps/api/core';

export interface ErrorInfo {
  isConnectionError: boolean;
  isOllamaError: boolean;
  needsRestart: boolean;
  userMessage: string;
  technicalDetails?: string;
}

export function analyzeError(error: any): ErrorInfo {
  const errorStr = String(error).toLowerCase();
  
  // Check for Ollama connection errors
  const isConnectionError = 
    errorStr.includes('reqwest') || 
    errorStr.includes('connection') ||
    errorStr.includes('refused') ||
    errorStr.includes('broken pipe') ||
    errorStr.includes('timeout') ||
    errorStr.includes('ollama connection');
    
  const isOllamaError = 
    errorStr.includes('ollama') ||
    errorStr.includes('model inference failed');
    
  const needsRestart = 
    errorStr.includes('restart may be needed') ||
    errorStr.includes('server may need restart');
  
  let userMessage = '';
  
  if (isConnectionError && isOllamaError) {
    userMessage = `🔌 Connection Issue: It looks like the AI server isn't responding. 

This can happen when:
• The Ollama server stopped unexpectedly
• Your computer went to sleep
• Port 11434 is blocked

Would you like me to try restarting the AI server?`;
  } else if (isOllamaError) {
    userMessage = `⚠️ AI Server Error: ${error}

The AI model encountered an issue. This might be temporary.`;
  } else {
    userMessage = `❌ Error: ${error}`;
  }
  
  return {
    isConnectionError,
    isOllamaError,
    needsRestart: needsRestart || (isConnectionError && isOllamaError),
    userMessage,
    technicalDetails: String(error)
  };
}

export async function attemptOllamaRestart(): Promise<{ success: boolean; message: string }> {
  try {
    await invoke('restart_ollama');
    return {
      success: true,
      message: '✅ AI server restart initiated. Please wait a moment...'
    };
  } catch (error) {
    return {
      success: false,
      message: String(error)
    };
  }
}

export function createRestartButton(onRestart: () => void): React.ReactElement {
  return (
    <button
      onClick={onRestart}
      className="mt-3 px-4 py-2 bg-accent text-white rounded-button hover:bg-accent-hover transition-colors flex items-center space-x-2"
    >
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
      <span>Restart AI Server</span>
    </button>
  );
}