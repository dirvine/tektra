#!/usr/bin/env node

/**
 * Tektra Voice AI Assistant - MCP Server
 * 
 * A comprehensive MCP server that provides voice-interactive AI capabilities
 * with multimodal support using Qwen2.5-VL model and Unmute voice pipeline.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn, execSync } from 'child_process';
import { WebSocket } from 'ws';
import { promises as fs } from 'fs';
import { homedir, platform } from 'os';
import { join, resolve } from 'path';
import fetch from 'node-fetch';

/**
 * Configuration and state management
 */
class TektraAIServer {
  constructor() {
    this.config = this.loadUserConfig();
    this.state = {
      voiceServicesRunning: false,
      currentModel: null,
      modelLoading: false,
      conversationActive: false,
      tektraProcess: null,
      serviceConnections: new Map(),
      lastError: null
    };
    
    this.setupSignalHandlers();
  }

  /**
   * Load user configuration from manifest user_config
   */
  loadUserConfig() {
    const defaultConfig = {
      voiceCharacter: 'default',
      modelPreference: 'qwen2.5-vl-7b',
      enableGpuAcceleration: true,
      voiceSensitivity: 0.6,
      enableInterruption: true,
      autoStartServices: true,
      cacheDirectory: join(homedir(), '.cache', 'tektra-ai')
    };

    // In a real implementation, this would read from environment variables
    // or user configuration files provided by the DXT runtime
    return {
      ...defaultConfig,
      ...this.parseConfigFromEnv()
    };
  }

  /**
   * Parse configuration from environment variables
   */
  parseConfigFromEnv() {
    const config = {};
    
    if (process.env.TEKTRA_VOICE_CHARACTER) {
      config.voiceCharacter = process.env.TEKTRA_VOICE_CHARACTER;
    }
    if (process.env.TEKTRA_MODEL_PREFERENCE) {
      config.modelPreference = process.env.TEKTRA_MODEL_PREFERENCE;
    }
    if (process.env.TEKTRA_ENABLE_GPU !== undefined) {
      config.enableGpuAcceleration = process.env.TEKTRA_ENABLE_GPU === 'true';
    }
    if (process.env.TEKTRA_VOICE_SENSITIVITY) {
      config.voiceSensitivity = parseFloat(process.env.TEKTRA_VOICE_SENSITIVITY);
    }
    if (process.env.TEKTRA_CACHE_DIR) {
      config.cacheDirectory = process.env.TEKTRA_CACHE_DIR;
    }

    return config;
  }

  /**
   * Setup signal handlers for graceful shutdown
   */
  setupSignalHandlers() {
    process.on('SIGINT', () => this.shutdown());
    process.on('SIGTERM', () => this.shutdown());
    process.on('uncaughtException', (error) => {
      console.error('Uncaught exception:', error);
      this.state.lastError = error.message;
    });
  }

  /**
   * Start Tektra application process
   */
  async startTektraProcess() {
    try {
      // First check if Tektra is available in the system
      const tektraPath = await this.findTektraExecutable();
      
      if (!tektraPath) {
        throw new Error('Tektra executable not found. Please ensure Tektra is installed.');
      }

      console.log(`Starting Tektra process: ${tektraPath}`);
      
      const tektraProcess = spawn(tektraPath, [], {
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
          ...process.env,
          TEKTRA_HEADLESS: 'true',
          TEKTRA_CONFIG_DIR: this.config.cacheDirectory
        }
      });

      tektraProcess.stdout.on('data', (data) => {
        console.log(`Tektra stdout: ${data}`);
      });

      tektraProcess.stderr.on('data', (data) => {
        console.error(`Tektra stderr: ${data}`);
      });

      tektraProcess.on('close', (code) => {
        console.log(`Tektra process exited with code ${code}`);
        this.state.tektraProcess = null;
        this.state.voiceServicesRunning = false;
      });

      this.state.tektraProcess = tektraProcess;
      
      // Wait for services to start up
      await this.waitForServicesReady();
      
      return { success: true, pid: tektraProcess.pid };
    } catch (error) {
      console.error('Failed to start Tektra process:', error);
      this.state.lastError = error.message;
      throw error;
    }
  }

  /**
   * Find Tektra executable
   */
  async findTektraExecutable() {
    const possiblePaths = [
      // Check if built locally
      resolve(__dirname, '../../target/release/tektra'),
      resolve(__dirname, '../../target/debug/tektra'),
      // Check system PATH
      'tektra',
      // Check common installation directories
      '/usr/local/bin/tektra',
      '/opt/tektra/bin/tektra',
      join(homedir(), '.local/bin/tektra')
    ];

    for (const path of possiblePaths) {
      try {
        if (path === 'tektra') {
          // Check if it's in PATH
          execSync('which tektra', { stdio: 'ignore' });
          return 'tektra';
        } else {
          await fs.access(path);
          return path;
        }
      } catch {
        continue;
      }
    }

    return null;
  }

  /**
   * Wait for Tektra services to be ready
   */
  async waitForServicesReady(maxWaitMs = 30000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitMs) {
      try {
        // Try to connect to the backend service
        const response = await fetch('http://127.0.0.1:8000/health', {
          timeout: 1000
        });
        
        if (response.ok) {
          this.state.voiceServicesRunning = true;
          return true;
        }
      } catch {
        // Service not ready yet, continue waiting
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error('Tektra services failed to start within timeout period');
  }

  /**
   * Stop all services and processes
   */
  async shutdown() {
    console.log('Shutting down Tektra AI server...');
    
    if (this.state.tektraProcess) {
      this.state.tektraProcess.kill('SIGTERM');
      this.state.tektraProcess = null;
    }
    
    // Close WebSocket connections
    for (const [name, ws] of this.state.serviceConnections) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
    this.state.serviceConnections.clear();
    
    this.state.voiceServicesRunning = false;
    this.state.conversationActive = false;
  }

  /**
   * Connect to voice service WebSocket
   */
  async connectToVoiceService(serviceType, port) {
    try {
      const ws = new WebSocket(`ws://127.0.0.1:${port}`);
      
      return new Promise((resolve, reject) => {
        ws.on('open', () => {
          console.log(`Connected to ${serviceType} service on port ${port}`);
          this.state.serviceConnections.set(serviceType, ws);
          resolve(ws);
        });
        
        ws.on('error', (error) => {
          console.error(`${serviceType} service connection error:`, error);
          reject(error);
        });
        
        ws.on('message', (data) => {
          console.log(`${serviceType} service message:`, data.toString());
        });
      });
    } catch (error) {
      console.error(`Failed to connect to ${serviceType} service:`, error);
      throw error;
    }
  }

  /**
   * Process multimodal input through Tektra
   */
  async processMultimodalInput(input) {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/multimodal', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input,
          config: {
            model: this.config.modelPreference,
            enableGpu: this.config.enableGpuAcceleration
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Multimodal processing failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Multimodal processing error:', error);
      throw error;
    }
  }

  /**
   * Get current model information
   */
  async getModelInfo() {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/models');
      
      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting model info:', error);
      throw error;
    }
  }

  /**
   * Load or switch AI model
   */
  async loadModel(modelId) {
    try {
      this.state.modelLoading = true;
      
      const response = await fetch('http://127.0.0.1:8000/api/v1/models/load', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelId,
          config: {
            enableGpu: this.config.enableGpuAcceleration,
            cacheDir: this.config.cacheDirectory
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Model loading failed: ${response.statusText}`);
      }

      const result = await response.json();
      this.state.currentModel = modelId;
      this.state.modelLoading = false;
      
      return result;
    } catch (error) {
      this.state.modelLoading = false;
      console.error('Model loading error:', error);
      throw error;
    }
  }
}

/**
 * Create and configure the MCP server
 */
const tektraServer = new TektraAIServer();

const server = new Server(
  {
    name: 'tektra-voice-ai',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

/**
 * Tool handlers
 */

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'start_voice_conversation',
        description: 'Start a real-time voice conversation with the AI assistant',
        inputSchema: {
          type: 'object',
          properties: {
            character: {
              type: 'string',
              description: 'Voice character to use (default, friendly, professional)',
              default: tektraServer.config.voiceCharacter
            }
          }
        },
      },
      {
        name: 'stop_voice_conversation',
        description: 'Stop the current voice conversation session',
        inputSchema: {
          type: 'object',
          properties: {}
        },
      },
      {
        name: 'load_model',
        description: 'Load or switch AI models (Qwen2.5-VL, other supported models)',
        inputSchema: {
          type: 'object',
          properties: {
            model_id: {
              type: 'string',
              description: 'Model identifier to load',
              enum: ['qwen2.5-vl-7b', 'qwen2.5-7b', 'auto']
            }
          },
          required: ['model_id']
        },
      },
      {
        name: 'get_voice_status',
        description: 'Get current status of voice services (STT, TTS, backend)',
        inputSchema: {
          type: 'object',
          properties: {}
        },
      },
      {
        name: 'process_multimodal_input',
        description: 'Process text, images, audio, or combined multimodal inputs',
        inputSchema: {
          type: 'object',
          properties: {
            text: {
              type: 'string',
              description: 'Text input or prompt'
            },
            image: {
              type: 'string',
              description: 'Base64 encoded image data'
            },
            audio: {
              type: 'string', 
              description: 'Base64 encoded audio data'
            },
            input_type: {
              type: 'string',
              description: 'Type of multimodal input',
              enum: ['text', 'text_with_image', 'text_with_audio', 'combined']
            }
          },
          required: ['input_type']
        },
      },
      {
        name: 'manage_voice_pipeline',
        description: 'Start/stop/configure the Unmute voice processing pipeline',
        inputSchema: {
          type: 'object',
          properties: {
            action: {
              type: 'string',
              description: 'Action to perform',
              enum: ['start', 'stop', 'restart', 'status']
            },
            config: {
              type: 'object',
              description: 'Optional configuration for voice pipeline',
              properties: {
                sensitivity: { type: 'number', minimum: 0.1, maximum: 1.0 },
                enableInterruption: { type: 'boolean' }
              }
            }
          },
          required: ['action']
        },
      },
      {
        name: 'get_model_info',
        description: 'Get information about available and loaded AI models',
        inputSchema: {
          type: 'object',
          properties: {}
        },
      },
      {
        name: 'configure_voice_settings',
        description: 'Configure voice processing parameters and preferences',
        inputSchema: {
          type: 'object',
          properties: {
            character: {
              type: 'string',
              enum: ['default', 'friendly', 'professional']
            },
            sensitivity: {
              type: 'number',
              minimum: 0.1,
              maximum: 1.0
            },
            enableInterruption: {
              type: 'boolean'
            }
          }
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'start_voice_conversation': {
        if (!tektraServer.state.voiceServicesRunning) {
          await tektraServer.startTektraProcess();
        }
        
        if (tektraServer.state.conversationActive) {
          throw new Error('Voice conversation is already active');
        }

        // Configure voice character if specified
        if (args.character) {
          tektraServer.config.voiceCharacter = args.character;
        }

        // Connect to voice services
        await tektraServer.connectToVoiceService('backend', 8000);
        await tektraServer.connectToVoiceService('stt', 8090);
        await tektraServer.connectToVoiceService('tts', 8089);

        tektraServer.state.conversationActive = true;

        return {
          content: [
            {
              type: 'text',
              text: `Voice conversation started with ${args.character || 'default'} character. All voice services are running and ready for real-time interaction.`
            }
          ]
        };
      }

      case 'stop_voice_conversation': {
        if (!tektraServer.state.conversationActive) {
          return {
            content: [
              {
                type: 'text',
                text: 'No active voice conversation to stop.'
              }
            ]
          };
        }

        tektraServer.state.conversationActive = false;
        
        // Close voice service connections
        for (const [name, ws] of tektraServer.state.serviceConnections) {
          if (ws.readyState === WebSocket.OPEN) {
            ws.close();
          }
        }
        tektraServer.state.serviceConnections.clear();

        return {
          content: [
            {
              type: 'text',
              text: 'Voice conversation stopped. All voice service connections closed.'
            }
          ]
        };
      }

      case 'load_model': {
        const { model_id } = args;
        
        if (tektraServer.state.modelLoading) {
          throw new Error('Another model is currently loading');
        }

        const result = await tektraServer.loadModel(model_id);
        
        return {
          content: [
            {
              type: 'text',
              text: `Successfully loaded model: ${model_id}. ${result.message || 'Model is ready for inference.'}`
            }
          ]
        };
      }

      case 'get_voice_status': {
        const status = {
          services_running: tektraServer.state.voiceServicesRunning,
          conversation_active: tektraServer.state.conversationActive,
          current_model: tektraServer.state.currentModel,
          model_loading: tektraServer.state.modelLoading,
          connected_services: Array.from(tektraServer.state.serviceConnections.keys()),
          last_error: tektraServer.state.lastError
        };

        return {
          content: [
            {
              type: 'text',
              text: `Voice Status:\n${JSON.stringify(status, null, 2)}`
            }
          ]
        };
      }

      case 'process_multimodal_input': {
        if (!tektraServer.state.voiceServicesRunning) {
          await tektraServer.startTektraProcess();
        }

        const result = await tektraServer.processMultimodalInput(args);
        
        return {
          content: [
            {
              type: 'text',
              text: `Multimodal processing result:\n${result.text || result.response || JSON.stringify(result)}`
            }
          ]
        };
      }

      case 'manage_voice_pipeline': {
        const { action, config } = args;
        
        switch (action) {
          case 'start':
            if (!tektraServer.state.voiceServicesRunning) {
              await tektraServer.startTektraProcess();
              return {
                content: [{ type: 'text', text: 'Voice pipeline started successfully.' }]
              };
            } else {
              return {
                content: [{ type: 'text', text: 'Voice pipeline is already running.' }]
              };
            }
            
          case 'stop':
            await tektraServer.shutdown();
            return {
              content: [{ type: 'text', text: 'Voice pipeline stopped.' }]
            };
            
          case 'restart':
            await tektraServer.shutdown();
            await tektraServer.startTektraProcess();
            return {
              content: [{ type: 'text', text: 'Voice pipeline restarted successfully.' }]
            };
            
          case 'status':
            const pipelineStatus = {
              running: tektraServer.state.voiceServicesRunning,
              process_id: tektraServer.state.tektraProcess?.pid || null,
              services: Array.from(tektraServer.state.serviceConnections.keys())
            };
            return {
              content: [
                {
                  type: 'text',
                  text: `Voice Pipeline Status:\n${JSON.stringify(pipelineStatus, null, 2)}`
                }
              ]
            };
            
          default:
            throw new Error(`Unknown pipeline action: ${action}`);
        }
      }

      case 'get_model_info': {
        const modelInfo = await tektraServer.getModelInfo();
        
        return {
          content: [
            {
              type: 'text',
              text: `Available Models:\n${JSON.stringify(modelInfo, null, 2)}`
            }
          ]
        };
      }

      case 'configure_voice_settings': {
        if (args.character) {
          tektraServer.config.voiceCharacter = args.character;
        }
        if (args.sensitivity) {
          tektraServer.config.voiceSensitivity = args.sensitivity;
        }
        if (args.enableInterruption !== undefined) {
          tektraServer.config.enableInterruption = args.enableInterruption;
        }

        return {
          content: [
            {
              type: 'text',
              text: `Voice settings updated:\n${JSON.stringify({
                character: tektraServer.config.voiceCharacter,
                sensitivity: tektraServer.config.voiceSensitivity,
                enableInterruption: tektraServer.config.enableInterruption
              }, null, 2)}`
            }
          ]
        };
      }

      default:
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${name}`
        );
    }
  } catch (error) {
    console.error(`Error executing tool ${name}:`, error);
    tektraServer.state.lastError = error.message;
    
    throw new McpError(
      ErrorCode.InternalError,
      `Tool execution failed: ${error.message}`
    );
  }
});

/**
 * Start the MCP server
 */
async function main() {
  console.log('Starting Tektra Voice AI MCP Server...');
  
  // Auto-start services if configured
  if (tektraServer.config.autoStartServices) {
    try {
      console.log('Auto-starting voice services...');
      await tektraServer.startTektraProcess();
      console.log('Voice services started successfully.');
    } catch (error) {
      console.error('Failed to auto-start services:', error);
      console.log('Services can be started manually using the start_voice_conversation tool.');
    }
  }

  // Connect via stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);
  
  console.log('Tektra Voice AI MCP Server running on stdio transport');
}

// Handle shutdown gracefully
process.on('SIGINT', async () => {
  console.log('Received SIGINT, shutting down gracefully...');
  await tektraServer.shutdown();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('Received SIGTERM, shutting down gracefully...');
  await tektraServer.shutdown();
  process.exit(0);
});

// Start the server
main().catch((error) => {
  console.error('Failed to start MCP server:', error);
  process.exit(1);
});