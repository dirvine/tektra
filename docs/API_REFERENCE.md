# Tektra AI Assistant - API Reference

## Overview

The Tektra AI Assistant provides a comprehensive REST API for creating, managing, and interacting with AI agents. This reference covers all available endpoints, request/response formats, authentication, and integration examples.

## Base URL

```
Production: https://api.tektra.ai/v1
Staging: https://staging-api.tektra.ai/v1
Development: http://localhost:8000/v1
```

## Authentication

### API Key Authentication

```bash
# Include API key in header
curl -H "Authorization: Bearer your_api_key" \
     https://api.tektra.ai/v1/agents
```

### Session Authentication

```bash
# Login to get session token
curl -X POST https://api.tektra.ai/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "password": "pass"}'

# Use session token
curl -H "Authorization: Bearer session_token" \
     https://api.tektra.ai/v1/agents
```

## Core Endpoints

### Agents

#### Create Agent

Create a new AI agent with specified capabilities.

```http
POST /v1/agents
```

**Request Body:**
```json
{
  "name": "Code Assistant",
  "description": "AI agent specialized in code analysis and generation",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "system_prompt": "You are a helpful coding assistant...",
  "tools": ["python_executor", "file_reader", "web_search"],
  "security_level": "medium",
  "max_memory_mb": 2048,
  "timeout_seconds": 300,
  "metadata": {
    "department": "engineering",
    "project": "code_review"
  }
}
```

**Response:**
```json
{
  "agent_id": "agent_abc123",
  "name": "Code Assistant",
  "status": "created",
  "created_at": "2024-01-01T12:00:00Z",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "capabilities": {
    "text_generation": true,
    "code_execution": true,
    "vision": true,
    "tool_use": true
  },
  "resource_usage": {
    "memory_mb": 1024,
    "cpu_percent": 0,
    "gpu_memory_mb": 2048
  }
}
```

#### List Agents

Get a list of all agents with optional filtering.

```http
GET /v1/agents?status=active&limit=50&offset=0
```

**Query Parameters:**
- `status`: Filter by status (active, inactive, error)
- `model`: Filter by model name
- `security_level`: Filter by security level
- `limit`: Number of results (default: 50, max: 200)
- `offset`: Pagination offset (default: 0)

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "agent_abc123",
      "name": "Code Assistant",
      "status": "active",
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "created_at": "2024-01-01T12:00:00Z",
      "last_used": "2024-01-01T14:30:00Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

#### Get Agent Details

Retrieve detailed information about a specific agent.

```http
GET /v1/agents/{agent_id}
```

**Response:**
```json
{
  "agent_id": "agent_abc123",
  "name": "Code Assistant",
  "description": "AI agent specialized in code analysis",
  "status": "active",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "system_prompt": "You are a helpful coding assistant...",
  "tools": ["python_executor", "file_reader"],
  "security_level": "medium",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T14:30:00Z",
  "statistics": {
    "total_requests": 150,
    "successful_requests": 145,
    "error_rate": 0.033,
    "average_response_time_ms": 250,
    "total_tokens_processed": 50000
  },
  "resource_usage": {
    "memory_mb": 1024,
    "cpu_percent": 15,
    "gpu_memory_mb": 2048
  }
}
```

#### Update Agent

Update agent configuration or settings.

```http
PUT /v1/agents/{agent_id}
```

**Request Body:**
```json
{
  "name": "Advanced Code Assistant",
  "system_prompt": "Updated system prompt...",
  "tools": ["python_executor", "file_reader", "web_search", "database_query"],
  "max_memory_mb": 4096
}
```

#### Delete Agent

Remove an agent and free its resources.

```http
DELETE /v1/agents/{agent_id}
```

**Response:**
```json
{
  "message": "Agent deleted successfully",
  "agent_id": "agent_abc123",
  "deleted_at": "2024-01-01T15:00:00Z"
}
```

### Conversations

#### Start Conversation

Begin a new conversation with an agent.

```http
POST /v1/agents/{agent_id}/conversations
```

**Request Body:**
```json
{
  "message": "Hello! Can you help me analyze this Python code?",
  "context": {
    "user_id": "user_123",
    "session_id": "session_456",
    "metadata": {
      "source": "web_ui",
      "ip_address": "192.168.1.100"
    }
  },
  "attachments": [
    {
      "type": "text",
      "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    }
  ]
}
```

**Response:**
```json
{
  "conversation_id": "conv_xyz789",
  "agent_id": "agent_abc123",
  "message": "I'd be happy to help analyze your Python code! Looking at your fibonacci function, I can see it's a recursive implementation...",
  "status": "completed",
  "created_at": "2024-01-01T15:30:00Z",
  "processing_time_ms": 1250,
  "token_usage": {
    "prompt_tokens": 120,
    "completion_tokens": 85,
    "total_tokens": 205
  },
  "tools_used": [
    {
      "tool": "code_analyzer",
      "execution_time_ms": 450,
      "result": "analysis_complete"
    }
  ]
}
```

#### Continue Conversation

Send a follow-up message in an existing conversation.

```http
POST /v1/conversations/{conversation_id}/messages
```

#### Get Conversation History

Retrieve the complete conversation history.

```http
GET /v1/conversations/{conversation_id}
```

**Response:**
```json
{
  "conversation_id": "conv_xyz789",
  "agent_id": "agent_abc123",
  "created_at": "2024-01-01T15:30:00Z",
  "updated_at": "2024-01-01T15:35:00Z",
  "message_count": 4,
  "messages": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "Hello! Can you help me analyze this Python code?",
      "timestamp": "2024-01-01T15:30:00Z"
    },
    {
      "id": "msg_002",
      "role": "assistant",
      "content": "I'd be happy to help analyze your Python code!...",
      "timestamp": "2024-01-01T15:30:15Z",
      "tools_used": ["code_analyzer"]
    }
  ]
}
```

### Tool Management

#### List Available Tools

Get a list of all available tools.

```http
GET /v1/tools
```

**Response:**
```json
{
  "tools": [
    {
      "tool_id": "python_executor",
      "name": "Python Code Executor",
      "description": "Execute Python code in a secure sandbox",
      "category": "code_execution",
      "security_level": "medium",
      "parameters": {
        "code": {
          "type": "string",
          "required": true,
          "description": "Python code to execute"
        },
        "timeout": {
          "type": "integer",
          "required": false,
          "default": 30,
          "description": "Execution timeout in seconds"
        }
      }
    }
  ]
}
```

#### Execute Tool

Execute a tool directly (without agent).

```http
POST /v1/tools/{tool_id}/execute
```

**Request Body:**
```json
{
  "parameters": {
    "code": "print('Hello, World!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')",
    "timeout": 10
  },
  "security_context": {
    "user_id": "user_123",
    "security_level": "medium"
  }
}
```

**Response:**
```json
{
  "execution_id": "exec_123456",
  "tool_id": "python_executor",
  "status": "completed",
  "result": {
    "output": "Hello, World!\n2 + 2 = 4",
    "exit_code": 0,
    "execution_time_ms": 150
  },
  "resource_usage": {
    "memory_mb": 12,
    "cpu_time_ms": 45
  },
  "security_events": []
}
```

### Models

#### List Available Models

Get information about available AI models.

```http
GET /v1/models
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
      "name": "Qwen 2.5 VL 7B Instruct",
      "description": "Multimodal model with vision and language capabilities",
      "type": "multimodal",
      "capabilities": ["text", "vision", "reasoning"],
      "parameters": "7B",
      "memory_requirements": {
        "minimum_mb": 8192,
        "recommended_mb": 16384
      },
      "status": "available",
      "last_updated": "2024-01-01T10:00:00Z"
    }
  ]
}
```

#### Get Model Status

Check the status and performance of a specific model.

```http
GET /v1/models/{model_id}/status
```

### System

#### Health Check

Check system health and status.

```http
GET /v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-01T16:00:00Z",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "models": "healthy",
    "security": "healthy"
  },
  "metrics": {
    "active_agents": 5,
    "total_requests_today": 1250,
    "average_response_time_ms": 320,
    "memory_usage_percent": 65,
    "cpu_usage_percent": 45
  }
}
```

#### System Metrics

Get detailed system metrics.

```http
GET /v1/metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-01T16:00:00Z",
  "system": {
    "memory": {
      "total_mb": 65536,
      "used_mb": 42598,
      "available_mb": 22938,
      "usage_percent": 65.0
    },
    "cpu": {
      "cores": 16,
      "usage_percent": 45.2,
      "load_average": [2.1, 1.8, 1.5]
    },
    "gpu": {
      "devices": [
        {
          "id": 0,
          "name": "NVIDIA RTX 4090",
          "memory_total_mb": 24564,
          "memory_used_mb": 16384,
          "utilization_percent": 85
        }
      ]
    }
  },
  "application": {
    "agents": {
      "total": 8,
      "active": 5,
      "idle": 3,
      "error": 0
    },
    "requests": {
      "total_today": 1250,
      "successful": 1198,
      "failed": 52,
      "error_rate": 0.042
    },
    "performance": {
      "average_response_time_ms": 320,
      "p95_response_time_ms": 850,
      "p99_response_time_ms": 1200
    }
  }
}
```

## WebSocket API

### Real-time Conversations

Connect to WebSocket for real-time agent conversations.

```javascript
// Connect to WebSocket
const ws = new WebSocket('wss://api.tektra.ai/v1/ws/conversations/{conversation_id}');

// Send message
ws.send(JSON.stringify({
  type: 'message',
  content: 'Hello, how can you help me today?',
  metadata: {
    timestamp: new Date().toISOString()
  }
}));

// Receive messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Streaming Responses

Get streaming responses for long-running operations.

```javascript
const ws = new WebSocket('wss://api.tektra.ai/v1/ws/stream/{agent_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'token':
      // Partial response token
      console.log('Token:', data.content);
      break;
    case 'tool_execution':
      // Tool execution status
      console.log('Tool:', data.tool, 'Status:', data.status);
      break;
    case 'complete':
      // Response complete
      console.log('Final response:', data.content);
      break;
  }
};
```

## SDKs and Client Libraries

### Python SDK

```bash
pip install tektra-sdk
```

```python
from tektra import TektraClient

# Initialize client
client = TektraClient(api_key="your_api_key")

# Create agent
agent = client.create_agent(
    name="Code Assistant",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    tools=["python_executor", "file_reader"]
)

# Start conversation
response = client.chat(
    agent_id=agent.id,
    message="Help me optimize this Python function",
    attachments=[{"type": "code", "content": "def slow_function():..."}]
)

print(response.message)
```

### JavaScript SDK

```bash
npm install tektra-sdk
```

```javascript
import { TektraClient } from 'tektra-sdk';

// Initialize client
const client = new TektraClient({ apiKey: 'your_api_key' });

// Create agent
const agent = await client.createAgent({
  name: 'Code Assistant',
  model: 'Qwen/Qwen2.5-VL-7B-Instruct',
  tools: ['python_executor', 'file_reader']
});

// Start conversation
const response = await client.chat({
  agentId: agent.id,
  message: 'Help me optimize this Python function',
  attachments: [{ type: 'code', content: 'def slow_function():...' }]
});

console.log(response.message);
```

## Error Handling

### HTTP Status Codes

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Permission denied
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service overloaded

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is invalid",
    "details": {
      "field": "model",
      "reason": "Model not found"
    },
    "request_id": "req_123456",
    "timestamp": "2024-01-01T16:00:00Z"
  }
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_API_KEY` | API key is invalid or expired | Check API key |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Implement backoff |
| `AGENT_NOT_FOUND` | Agent ID doesn't exist | Verify agent ID |
| `MODEL_UNAVAILABLE` | Model is not loaded | Wait or use different model |
| `INSUFFICIENT_RESOURCES` | Not enough memory/CPU | Scale up or retry later |
| `SECURITY_VIOLATION` | Security policy violation | Review request content |
| `TOOL_EXECUTION_FAILED` | Tool execution error | Check tool parameters |

## Rate Limits

### Default Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| Agent Creation | 10 requests | 1 minute |
| Conversations | 100 requests | 1 minute |
| Tool Execution | 50 requests | 1 minute |
| Model Queries | 200 requests | 1 minute |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
X-RateLimit-RetryAfter: 60
```

## Webhook Integration

### Configure Webhooks

Set up webhooks to receive real-time notifications.

```http
POST /v1/webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhooks/tektra",
  "events": [
    "agent.created",
    "agent.deleted",
    "conversation.completed",
    "tool.executed",
    "security.violation"
  ],
  "secret": "webhook_secret_key"
}
```

### Webhook Payload

```json
{
  "event": "conversation.completed",
  "timestamp": "2024-01-01T16:00:00Z",
  "data": {
    "conversation_id": "conv_xyz789",
    "agent_id": "agent_abc123",
    "user_id": "user_123",
    "message_count": 5,
    "total_tokens": 1250,
    "duration_ms": 3500
  },
  "signature": "sha256=webhook_signature"
}
```

## Best Practices

### Authentication Security

```python
# Use environment variables for API keys
import os
api_key = os.getenv('TEKTRA_API_KEY')

# Rotate API keys regularly
# Implement proper key storage (e.g., AWS Secrets Manager)
```

### Error Handling

```python
import time
from tektra import TektraClient, TektraError

client = TektraClient(api_key=api_key)

def robust_chat(agent_id, message, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat(agent_id=agent_id, message=message)
        except TektraError as e:
            if e.status_code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise e
    raise Exception("Max retries exceeded")
```

### Resource Management

```python
# Monitor resource usage
def monitor_agent_resources(agent_id):
    agent = client.get_agent(agent_id)
    memory_usage = agent.resource_usage.memory_mb
    
    if memory_usage > 8192:  # 8GB threshold
        # Scale up or optimize
        client.update_agent(agent_id, max_memory_mb=16384)
```

## Migration Guide

### From v0.x to v1.0

1. Update API base URL
2. Add authentication headers
3. Update request/response formats
4. Handle new error codes
5. Update SDK versions

See the complete migration guide at [docs.tektra.ai/migration](https://docs.tektra.ai/migration).