import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // Vite options tailored for Tauri development
  clearScreen: false,
  
  // Tauri uses a fixed localhost URL
  server: {
    port: 1420,
    strictPort: true,
  },
  
  // Add environmental variables
  envPrefix: ["VITE_", "TAURI_"],
  
  build: {
    // Tauri uses a consistent build output directory
    outDir: "dist",
    
    // Don't minify for debugging in production
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    
    // Produce sourcemaps for debugging
    sourcemap: !!process.env.TAURI_DEBUG,
  },
});