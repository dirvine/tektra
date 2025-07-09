import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { EventEmitter } from 'events';

// Mock the MCP SDK
jest.mock('@modelcontextprotocol/sdk/server/index.js', () => ({
  Server: jest.fn()
}));

jest.mock('@modelcontextprotocol/sdk/server/stdio.js', () => ({
  StdioServerTransport: jest.fn()
}));

jest.mock('@modelcontextprotocol/sdk/types.js', () => ({
  CallToolRequestSchema: 'CallToolRequestSchema',
  ErrorCode: { MethodNotFound: 'MethodNotFound', InternalError: 'InternalError' },
  ListToolsRequestSchema: 'ListToolsRequestSchema',
  McpError: class McpError extends Error {
    constructor(code, message) {
      super(message);
      this.code = code;
    }
  }
}));

// Create a mock process that extends EventEmitter
const createMockProcess = () => {
  const mockProcess = new EventEmitter();
  mockProcess.pid = 12345;
  mockProcess.stdout = new EventEmitter();
  mockProcess.stderr = new EventEmitter();
  mockProcess.kill = jest.fn();
  return mockProcess;
};

// Create a mock WebSocket that extends EventEmitter
const createMockWebSocket = () => {
  const mockWS = new EventEmitter();
  mockWS.readyState = 1; // WebSocket.OPEN
  mockWS.close = jest.fn();
  mockWS.send = jest.fn();
  return mockWS;
};

// Mock the main module dynamically
let TektraAIServer;

beforeEach(async () => {
  // Clear module cache
  jest.resetModules();
  
  // Get the mocked functions
  const { spawn } = await import('child_process');
  const { promises: fs } = await import('fs');
  const { WebSocket } = await import('ws');
  const fetch = (await import('node-fetch')).default;
  
  // Setup mock implementations
  spawn.mockImplementation(() => createMockProcess());
  fs.access.mockResolvedValue(undefined);
  WebSocket.mockImplementation(() => createMockWebSocket());
  fetch.mockImplementation(() => Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ success: true }),
    text: () => Promise.resolve('OK')
  }));
  
  // Import the class after mocking
  const module = await import('../index.js');
  TektraAIServer = module.TektraAIServer;
});

afterEach(() => {
  jest.clearAllMocks();
});

describe('TektraAIServer', () => {
  let server;

  beforeEach(() => {
    server = new TektraAIServer();
  });

  afterEach(async () => {
    if (server) {
      await server.shutdown();
    }
  });

  describe('Constructor', () => {
    it('should initialize with default configuration', () => {
      expect(server.config).toBeDefined();
      expect(server.config.voiceCharacter).toBe('default');
      expect(server.config.modelPreference).toBe('qwen2.5-vl-7b');
      expect(server.config.enableGpuAcceleration).toBe(true);
      expect(server.config.voiceSensitivity).toBe(0.6);
      expect(server.config.enableInterruption).toBe(true);
      expect(server.config.autoStartServices).toBe(true);
    });

    it('should initialize with default state', () => {
      expect(server.state).toBeDefined();
      expect(server.state.voiceServicesRunning).toBe(false);
      expect(server.state.currentModel).toBeNull();
      expect(server.state.modelLoading).toBe(false);
      expect(server.state.conversationActive).toBe(false);
      expect(server.state.tektraProcess).toBeNull();
      expect(server.state.serviceConnections).toBeInstanceOf(Map);
      expect(server.state.lastError).toBeNull();
    });
  });

  describe('Configuration', () => {
    it('should parse environment variables', () => {
      const originalEnv = process.env;
      process.env = {
        ...originalEnv,
        TEKTRA_VOICE_CHARACTER: 'professional',
        TEKTRA_MODEL_PREFERENCE: 'qwen2.5-7b',
        TEKTRA_ENABLE_GPU: 'false',
        TEKTRA_VOICE_SENSITIVITY: '0.8',
        TEKTRA_CACHE_DIR: '/custom/cache'
      };

      const testServer = new TektraAIServer();
      
      expect(testServer.config.voiceCharacter).toBe('professional');
      expect(testServer.config.modelPreference).toBe('qwen2.5-7b');
      expect(testServer.config.enableGpuAcceleration).toBe(false);
      expect(testServer.config.voiceSensitivity).toBe(0.8);
      expect(testServer.config.cacheDirectory).toBe('/custom/cache');

      process.env = originalEnv;
    });

    it('should handle invalid environment variables gracefully', () => {
      const originalEnv = process.env;
      process.env = {
        ...originalEnv,
        TEKTRA_VOICE_SENSITIVITY: 'invalid_number'
      };

      const testServer = new TektraAIServer();
      
      expect(testServer.config.voiceSensitivity).toBe(0.6); // Should use default

      process.env = originalEnv;
    });
  });

  describe('Process Management', () => {
    it('should find tektra executable', async () => {
      const { promises: fs } = await import('fs');
      fs.access.mockResolvedValueOnce(undefined);
      
      const path = await server.findTektraExecutable();
      
      expect(path).toBeDefined();
      expect(fs.access).toHaveBeenCalled();
    });

    it('should return null if no executable found', async () => {
      const { promises: fs } = await import('fs');
      fs.access.mockRejectedValue(new Error('File not found'));
      
      const path = await server.findTektraExecutable();
      
      expect(path).toBeNull();
    });

    it('should start tektra process successfully', async () => {
      const { spawn } = await import('child_process');
      const { promises: fs } = await import('fs');
      const fetch = (await import('node-fetch')).default;
      
      const mockProcess = createMockProcess();
      spawn.mockReturnValue(mockProcess);
      fs.access.mockResolvedValue(undefined);
      
      // Mock successful health check
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' })
      });

      const result = await server.startTektraProcess();
      
      expect(spawn).toHaveBeenCalledWith(
        expect.any(String),
        [],
        expect.objectContaining({
          stdio: ['ignore', 'pipe', 'pipe'],
          env: expect.objectContaining({
            TEKTRA_HEADLESS: 'true',
            TEKTRA_CONFIG_DIR: server.config.cacheDirectory
          })
        })
      );
      
      expect(result).toEqual({
        success: true,
        pid: mockProcess.pid
      });
      
      expect(server.state.tektraProcess).toBe(mockProcess);
    });

    it('should handle process startup failure', async () => {
      const { promises: fs } = await import('fs');
      fs.access.mockRejectedValue(new Error('Executable not found'));
      
      await expect(server.startTektraProcess()).rejects.toThrow('Tektra executable not found');
      expect(server.state.lastError).toBe('Tektra executable not found. Please ensure Tektra is installed.');
    });

    it('should wait for services to be ready', async () => {
      const fetch = (await import('node-fetch')).default;
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' })
      });

      const result = await server.waitForServicesReady(5000);
      
      expect(result).toBe(true);
      expect(server.state.voiceServicesRunning).toBe(true);
      expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/health', {
        timeout: 1000
      });
    });

    it('should timeout if services not ready', async () => {
      const fetch = (await import('node-fetch')).default;
      fetch.mockRejectedValue(new Error('Connection refused'));

      await expect(server.waitForServicesReady(1000)).rejects.toThrow(
        'Tektra services failed to start within timeout period'
      );
    });

    it('should shutdown gracefully', async () => {
      const mockProcess = createMockProcess();
      const mockWS = createMockWebSocket();
      
      server.state.tektraProcess = mockProcess;
      server.state.serviceConnections.set('test', mockWS);
      server.state.voiceServicesRunning = true;
      server.state.conversationActive = true;

      await server.shutdown();
      
      expect(mockProcess.kill).toHaveBeenCalledWith('SIGTERM');
      expect(mockWS.close).toHaveBeenCalled();
      expect(server.state.tektraProcess).toBeNull();
      expect(server.state.serviceConnections.size).toBe(0);
      expect(server.state.voiceServicesRunning).toBe(false);
      expect(server.state.conversationActive).toBe(false);
    });
  });

  describe('WebSocket Management', () => {
    it('should connect to voice service', async () => {
      const { WebSocket } = await import('ws');
      const mockWS = createMockWebSocket();
      WebSocket.mockImplementation(() => {
        setTimeout(() => mockWS.emit('open'), 10);
        return mockWS;
      });

      const ws = await server.connectToVoiceService('test', 8000);
      
      expect(WebSocket).toHaveBeenCalledWith('ws://127.0.0.1:8000');
      expect(ws).toBe(mockWS);
      expect(server.state.serviceConnections.get('test')).toBe(mockWS);
    });

    it('should handle connection errors', async () => {
      const mockWS = createMockWebSocket();
      WebSocket.mockImplementation(() => {
        setTimeout(() => mockWS.emit('error', new Error('Connection failed')), 10);
        return mockWS;
      });

      await expect(server.connectToVoiceService('test', 8000)).rejects.toThrow('Connection failed');
    });

    it('should handle incoming messages', async () => {
      const mockWS = createMockWebSocket();
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      WebSocket.mockImplementation(() => {
        setTimeout(() => {
          mockWS.emit('open');
          mockWS.emit('message', Buffer.from('test message'));
        }, 10);
        return mockWS;
      });

      await server.connectToVoiceService('test', 8000);
      
      expect(consoleSpy).toHaveBeenCalledWith('test service message:', 'test message');
      
      consoleSpy.mockRestore();
    });
  });

  describe('API Integration', () => {
    it('should process multimodal input', async () => {
      const mockResponse = {
        ok: true,
        json: () => Promise.resolve({ text: 'AI response', confidence: 0.95 })
      };
      fetch.mockResolvedValue(mockResponse);

      const input = {
        text: 'Hello, AI!',
        type: 'text'
      };

      const result = await server.processMultimodalInput(input);
      
      expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/api/v1/multimodal', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input,
          config: {
            model: server.config.modelPreference,
            enableGpu: server.config.enableGpuAcceleration
          }
        })
      });
      
      expect(result).toEqual({ text: 'AI response', confidence: 0.95 });
    });

    it('should handle API errors', async () => {
      const mockResponse = {
        ok: false,
        statusText: 'Internal Server Error'
      };
      fetch.mockResolvedValue(mockResponse);

      await expect(server.processMultimodalInput({ text: 'test' })).rejects.toThrow(
        'Multimodal processing failed: Internal Server Error'
      );
    });

    it('should get model info', async () => {
      const mockModels = {
        available: ['qwen2.5-vl-7b', 'qwen2.5-7b'],
        current: 'qwen2.5-vl-7b'
      };
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockModels)
      });

      const result = await server.getModelInfo();
      
      expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/api/v1/models');
      expect(result).toEqual(mockModels);
    });

    it('should load model', async () => {
      const mockResult = {
        success: true,
        model: 'qwen2.5-vl-7b',
        message: 'Model loaded successfully'
      };
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResult)
      });

      const result = await server.loadModel('qwen2.5-vl-7b');
      
      expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/api/v1/models/load', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelId: 'qwen2.5-vl-7b',
          config: {
            enableGpu: server.config.enableGpuAcceleration,
            cacheDir: server.config.cacheDirectory
          }
        })
      });
      
      expect(result).toEqual(mockResult);
      expect(server.state.currentModel).toBe('qwen2.5-vl-7b');
      expect(server.state.modelLoading).toBe(false);
    });

    it('should handle model loading errors', async () => {
      fetch.mockRejectedValue(new Error('Network error'));
      
      await expect(server.loadModel('qwen2.5-vl-7b')).rejects.toThrow('Network error');
      expect(server.state.modelLoading).toBe(false);
    });

    it('should set model loading state', async () => {
      const mockPromise = new Promise((resolve) => {
        setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({ success: true })
        }), 100);
      });
      fetch.mockReturnValue(mockPromise);

      const loadPromise = server.loadModel('qwen2.5-vl-7b');
      
      expect(server.state.modelLoading).toBe(true);
      
      await loadPromise;
      
      expect(server.state.modelLoading).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should handle uncaught exceptions', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      const error = new Error('Test error');
      process.emit('uncaughtException', error);
      
      expect(consoleSpy).toHaveBeenCalledWith('Uncaught exception:', error);
      expect(server.state.lastError).toBe('Test error');
      
      consoleSpy.mockRestore();
    });

    it('should handle process signals', () => {
      const shutdownSpy = jest.spyOn(server, 'shutdown').mockImplementation();
      
      process.emit('SIGINT');
      expect(shutdownSpy).toHaveBeenCalled();
      
      process.emit('SIGTERM');
      expect(shutdownSpy).toHaveBeenCalledTimes(2);
      
      shutdownSpy.mockRestore();
    });
  });

  describe('State Management', () => {
    it('should track service connections', async () => {
      const mockWS = createMockWebSocket();
      WebSocket.mockImplementation(() => {
        setTimeout(() => mockWS.emit('open'), 10);
        return mockWS;
      });

      await server.connectToVoiceService('backend', 8000);
      await server.connectToVoiceService('stt', 8090);
      
      expect(server.state.serviceConnections.size).toBe(2);
      expect(server.state.serviceConnections.has('backend')).toBe(true);
      expect(server.state.serviceConnections.has('stt')).toBe(true);
    });

    it('should update conversation state', () => {
      expect(server.state.conversationActive).toBe(false);
      
      server.state.conversationActive = true;
      expect(server.state.conversationActive).toBe(true);
      
      server.state.conversationActive = false;
      expect(server.state.conversationActive).toBe(false);
    });

    it('should track current model', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ success: true })
      });

      expect(server.state.currentModel).toBeNull();
      
      await server.loadModel('qwen2.5-vl-7b');
      expect(server.state.currentModel).toBe('qwen2.5-vl-7b');
    });

    it('should track last error', () => {
      expect(server.state.lastError).toBeNull();
      
      server.state.lastError = 'Test error';
      expect(server.state.lastError).toBe('Test error');
    });
  });
});