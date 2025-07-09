import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { EventEmitter } from 'events';

// Mock MCP SDK
const mockServer = {
  setRequestHandler: jest.fn(),
  connect: jest.fn()
};

const mockTransport = {
  connect: jest.fn()
};

jest.mock('@modelcontextprotocol/sdk/server/index.js', () => ({
  Server: jest.fn(() => mockServer)
}));

jest.mock('@modelcontextprotocol/sdk/server/stdio.js', () => ({
  StdioServerTransport: jest.fn(() => mockTransport)
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

// Mock process and WebSocket
const createMockProcess = () => {
  const mockProcess = new EventEmitter();
  mockProcess.pid = 12345;
  mockProcess.stdout = new EventEmitter();
  mockProcess.stderr = new EventEmitter();
  mockProcess.kill = jest.fn();
  return mockProcess;
};

const createMockWebSocket = () => {
  const mockWS = new EventEmitter();
  mockWS.readyState = 1;
  mockWS.close = jest.fn();
  mockWS.send = jest.fn();
  return mockWS;
};

describe('MCP Integration', () => {
  let toolHandlers;
  let listToolsHandler;

  beforeEach(async () => {
    jest.clearAllMocks();
    
    // Get mocked dependencies
    const { spawn } = await import('child_process');
    const { promises: fs } = await import('fs');
    const { WebSocket } = await import('ws');
    const fetch = (await import('node-fetch')).default;
    
    // Mock dependencies
    spawn.mockImplementation(() => createMockProcess());
    fs.access.mockResolvedValue(undefined);
    WebSocket.mockImplementation(() => createMockWebSocket());
    fetch.mockImplementation(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ success: true }),
      text: () => Promise.resolve('OK')
    }));
    
    // Reset handlers
    toolHandlers = {};
    listToolsHandler = null;
    
    // Capture handlers when they're set
    mockServer.setRequestHandler.mockImplementation((schema, handler) => {
      if (schema === 'ListToolsRequestSchema') {
        listToolsHandler = handler;
      } else if (schema === 'CallToolRequestSchema') {
        toolHandlers.callTool = handler;
      }
    });
    
    // Import after mocking
    await import('../index.js');
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Tool Registration', () => {
    it('should register all required tools', async () => {
      expect(listToolsHandler).toBeDefined();
      
      const result = await listToolsHandler();
      
      expect(result).toHaveProperty('tools');
      expect(result.tools).toBeInstanceOf(Array);
      expect(result.tools.length).toBe(8);
      
      const toolNames = result.tools.map(tool => tool.name);
      expect(toolNames).toContain('start_voice_conversation');
      expect(toolNames).toContain('stop_voice_conversation');
      expect(toolNames).toContain('load_model');
      expect(toolNames).toContain('get_voice_status');
      expect(toolNames).toContain('process_multimodal_input');
      expect(toolNames).toContain('manage_voice_pipeline');
      expect(toolNames).toContain('get_model_info');
      expect(toolNames).toContain('configure_voice_settings');
    });

    it('should have correct tool schemas', async () => {
      const result = await listToolsHandler();
      
      const startVoiceTool = result.tools.find(tool => tool.name === 'start_voice_conversation');
      expect(startVoiceTool).toBeDefined();
      expect(startVoiceTool.description).toBe('Start a real-time voice conversation with the AI assistant');
      expect(startVoiceTool.inputSchema).toHaveProperty('type', 'object');
      expect(startVoiceTool.inputSchema.properties).toHaveProperty('character');
      
      const loadModelTool = result.tools.find(tool => tool.name === 'load_model');
      expect(loadModelTool).toBeDefined();
      expect(loadModelTool.inputSchema.required).toContain('model_id');
      expect(loadModelTool.inputSchema.properties.model_id.enum).toContain('qwen2.5-vl-7b');
      
      const multimodalTool = result.tools.find(tool => tool.name === 'process_multimodal_input');
      expect(multimodalTool).toBeDefined();
      expect(multimodalTool.inputSchema.required).toContain('input_type');
      expect(multimodalTool.inputSchema.properties.input_type.enum).toContain('text');
      expect(multimodalTool.inputSchema.properties.input_type.enum).toContain('text_with_image');
    });
  });

  describe('Tool Execution', () => {
    it('should start voice conversation', async () => {
      const { WebSocket } = await import('ws');
      const fetch = (await import('node-fetch')).default;
      
      const mockWS = createMockWebSocket();
      WebSocket.mockImplementation(() => {
        setTimeout(() => mockWS.emit('open'), 10);
        return mockWS;
      });
      
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' })
      });

      const request = {
        params: {
          name: 'start_voice_conversation',
          arguments: { character: 'professional' }
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content).toBeInstanceOf(Array);
      expect(result.content[0]).toHaveProperty('type', 'text');
      expect(result.content[0].text).toContain('Voice conversation started');
      expect(result.content[0].text).toContain('professional');
    });

    it('should stop voice conversation', async () => {
      const request = {
        params: {
          name: 'stop_voice_conversation',
          arguments: {}
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content[0].text).toContain('No active voice conversation to stop');
    });

    it('should load model', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ 
          success: true,
          message: 'Model loaded successfully' 
        })
      });

      const request = {
        params: {
          name: 'load_model',
          arguments: { model_id: 'qwen2.5-vl-7b' }
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content[0].text).toContain('Successfully loaded model: qwen2.5-vl-7b');
    });

    it('should get voice status', async () => {
      const request = {
        params: {
          name: 'get_voice_status',
          arguments: {}
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content[0].text).toContain('Voice Status:');
      expect(result.content[0].text).toContain('services_running');
      expect(result.content[0].text).toContain('conversation_active');
    });

    it('should process multimodal input', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ 
          text: 'AI response to multimodal input',
          confidence: 0.95
        })
      });

      const request = {
        params: {
          name: 'process_multimodal_input',
          arguments: {
            input_type: 'text',
            text: 'Hello, can you help me?'
          }
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content[0].text).toContain('Multimodal processing result:');
      expect(result.content[0].text).toContain('AI response to multimodal input');
    });

    it('should manage voice pipeline', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' })
      });

      const request = {
        params: {
          name: 'manage_voice_pipeline',
          arguments: { action: 'status' }
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content[0].text).toContain('Voice Pipeline Status:');
      expect(result.content[0].text).toContain('running');
    });

    it('should get model info', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          available: ['qwen2.5-vl-7b', 'qwen2.5-7b'],
          current: 'qwen2.5-vl-7b',
          loaded: true
        })
      });

      const request = {
        params: {
          name: 'get_model_info',
          arguments: {}
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content[0].text).toContain('Available Models:');
      expect(result.content[0].text).toContain('qwen2.5-vl-7b');
    });

    it('should configure voice settings', async () => {
      const request = {
        params: {
          name: 'configure_voice_settings',
          arguments: {
            character: 'friendly',
            sensitivity: 0.8,
            enableInterruption: true
          }
        }
      };

      const result = await toolHandlers.callTool(request);
      
      expect(result).toHaveProperty('content');
      expect(result.content[0].text).toContain('Voice settings updated:');
      expect(result.content[0].text).toContain('friendly');
      expect(result.content[0].text).toContain('0.8');
      expect(result.content[0].text).toContain('true');
    });
  });

  describe('Error Handling', () => {
    it('should handle unknown tool names', async () => {
      const request = {
        params: {
          name: 'unknown_tool',
          arguments: {}
        }
      };

      await expect(toolHandlers.callTool(request)).rejects.toThrow('Unknown tool: unknown_tool');
    });

    it('should handle tool execution errors', async () => {
      fetch.mockRejectedValue(new Error('Network error'));

      const request = {
        params: {
          name: 'load_model',
          arguments: { model_id: 'qwen2.5-vl-7b' }
        }
      };

      await expect(toolHandlers.callTool(request)).rejects.toThrow('Tool execution failed: Network error');
    });

    it('should handle missing required parameters', async () => {
      const request = {
        params: {
          name: 'load_model',
          arguments: {} // missing model_id
        }
      };

      await expect(toolHandlers.callTool(request)).rejects.toThrow();
    });

    it('should handle API errors gracefully', async () => {
      fetch.mockResolvedValue({
        ok: false,
        statusText: 'Internal Server Error'
      });

      const request = {
        params: {
          name: 'get_model_info',
          arguments: {}
        }
      };

      await expect(toolHandlers.callTool(request)).rejects.toThrow('Tool execution failed: Failed to get model info: Internal Server Error');
    });
  });

  describe('State Management', () => {
    it('should maintain conversation state across tool calls', async () => {
      const mockWS = createMockWebSocket();
      WebSocket.mockImplementation(() => {
        setTimeout(() => mockWS.emit('open'), 10);
        return mockWS;
      });
      
      fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' })
      });

      // Start conversation
      const startRequest = {
        params: {
          name: 'start_voice_conversation',
          arguments: { character: 'default' }
        }
      };

      await toolHandlers.callTool(startRequest);

      // Check status
      const statusRequest = {
        params: {
          name: 'get_voice_status',
          arguments: {}
        }
      };

      const statusResult = await toolHandlers.callTool(statusRequest);
      expect(statusResult.content[0].text).toContain('"conversation_active": true');

      // Stop conversation
      const stopRequest = {
        params: {
          name: 'stop_voice_conversation',
          arguments: {}
        }
      };

      await toolHandlers.callTool(stopRequest);

      // Check status again
      const statusResult2 = await toolHandlers.callTool(statusRequest);
      expect(statusResult2.content[0].text).toContain('"conversation_active": false');
    });

    it('should track model loading state', async () => {
      let resolvePromise;
      const loadingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      fetch.mockReturnValue(loadingPromise.then(() => ({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })));

      const loadRequest = {
        params: {
          name: 'load_model',
          arguments: { model_id: 'qwen2.5-vl-7b' }
        }
      };

      const loadPromise = toolHandlers.callTool(loadRequest);

      // Check status while loading
      const statusRequest = {
        params: {
          name: 'get_voice_status',
          arguments: {}
        }
      };

      const statusResult = await toolHandlers.callTool(statusRequest);
      expect(statusResult.content[0].text).toContain('"model_loading": true');

      // Complete loading
      resolvePromise();
      await loadPromise;

      // Check status after loading
      const statusResult2 = await toolHandlers.callTool(statusRequest);
      expect(statusResult2.content[0].text).toContain('"model_loading": false');
      expect(statusResult2.content[0].text).toContain('"current_model": "qwen2.5-vl-7b"');
    });

    it('should prevent concurrent model loading', async () => {
      let resolvePromise;
      const loadingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      fetch.mockReturnValue(loadingPromise.then(() => ({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })));

      const loadRequest1 = {
        params: {
          name: 'load_model',
          arguments: { model_id: 'qwen2.5-vl-7b' }
        }
      };

      const loadRequest2 = {
        params: {
          name: 'load_model',
          arguments: { model_id: 'qwen2.5-7b' }
        }
      };

      const loadPromise1 = toolHandlers.callTool(loadRequest1);
      
      // Second request should fail
      await expect(toolHandlers.callTool(loadRequest2)).rejects.toThrow('Another model is currently loading');

      // Complete first request
      resolvePromise();
      await loadPromise1;
    });
  });
});