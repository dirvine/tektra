import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],
  // Vite options tailored for Tauri development
  // prevent vite from obscuring rust errors
  clearScreen: false,
  // Set base path for production builds
  base: "./",
  // Configure build options
  build: {
    // Tauri expects these in the dist folder
    outDir: "dist",
    emptyOutDir: true,
    // Ensure assets are properly bundled
    assetsDir: "assets",
    // Generate source maps for debugging
    sourcemap: false,
    // Optimize for production
    minify: "esbuild",
    // Target modern browsers
    target: "esnext",
    rollupOptions: {
      // Ensure proper asset handling
      output: {
        assetFileNames: "assets/[name]-[hash][extname]",
        chunkFileNames: "assets/[name]-[hash].js",
        entryFileNames: "assets/[name]-[hash].js",
      },
    },
  },
  // tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // 3. tell vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));