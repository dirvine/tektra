// Tauri IPC type declarations
declare global {
  interface Window {
    __TAURI_IPC__?: (params: any) => Promise<any>;
  }
}

export {};