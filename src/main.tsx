import React from 'react';
import ReactDOM from 'react-dom/client';
import CompleteApp from './CompleteApp';

// Wait for DOM and Tauri to be ready
function initApp() {
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
    return;
  }

  // Mount React app immediately - let the app handle Tauri initialization internally
  ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
      <CompleteApp />
    </React.StrictMode>
  );
}

// Initialize immediately or when DOM is ready
initApp();