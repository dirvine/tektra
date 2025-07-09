import '@testing-library/jest-dom'
import { beforeAll, afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

// Mock Tauri API
vi.mock('@tauri-apps/api/tauri', () => ({
  invoke: vi.fn(),
}))

vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(),
  emit: vi.fn(),
}))

vi.mock('@tauri-apps/api/window', () => ({
  appWindow: {
    listen: vi.fn(),
    emit: vi.fn(),
  },
}))

// Mock Web Audio API
Object.defineProperty(window, 'AudioContext', {
  writable: true,
  value: vi.fn().mockImplementation(() => ({
    createGain: vi.fn(() => ({
      connect: vi.fn(),
      gain: { value: 1 },
    })),
    createOscillator: vi.fn(() => ({
      connect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      frequency: { value: 440 },
    })),
    destination: {},
  })),
})

// Mock Three.js WebGL context
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  getExtension: vi.fn(),
  getParameter: vi.fn(),
  createProgram: vi.fn(),
  createShader: vi.fn(),
  shaderSource: vi.fn(),
  compileShader: vi.fn(),
  attachShader: vi.fn(),
  linkProgram: vi.fn(),
  useProgram: vi.fn(),
  getProgramParameter: vi.fn(() => true),
  getShaderParameter: vi.fn(() => true),
  viewport: vi.fn(),
  clear: vi.fn(),
  clearColor: vi.fn(),
}))

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Cleanup after each test
afterEach(() => {
  cleanup()
})

// Global test setup
beforeAll(() => {
  // Add any global setup here
})