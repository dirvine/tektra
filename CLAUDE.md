# Claude Code Development Guidelines for Tektra

This file contains the development standards and practices that Claude should follow when working on the Tektra AI Assistant codebase.

## Project Overview

Tektra is a **production-ready, enterprise-grade AI assistant platform** built with Python. The system integrates SmolAgents for AI capabilities, comprehensive security frameworks, performance optimization, and enterprise deployment features.

### Key Technologies
- **Python 3.11+** with UV package manager
- **SmolAgents** for AI agent execution
- **Docker & Kubernetes** for deployment
- **PostgreSQL & Redis** for data storage
- **Prometheus & Grafana** for monitoring

## Development Workflow

### 1. Architecture-Driven Development
- **Reference documentation** in `docs/` for system architecture and API specifications
- **Maintain enterprise standards** with security, performance, and scalability focus
- **Follow existing patterns** established in the current codebase

### 2. Python Development Standards

#### Core Module Guidelines
```python
# Use proper async/await patterns
async def create_agent(self, config: AgentConfig) -> Agent:
    """Create and initialize a new agent."""
    try:
        # Validate configuration
        await self._validate_agent_config(config)
        
        # Create security context
        security_context = await self.security_manager.create_context(config)
        
        # Initialize agent
        agent = await self._initialize_agent(config, security_context)
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise AgentCreationError(f"Agent creation failed: {e}") from e

# Use comprehensive logging
logger.info(f"Creating agent {config.name} with model {config.model}")
```

#### Security and Error Handling
- **Never use mock implementations** in production code
- **Always use proper exception handling** with custom exception types
- **Implement comprehensive logging** with structured data
- **Follow security-first design** with validation and sandboxing

#### SmolAgents Integration
```python
# Proper SmolAgent integration
from smolagents import CodeAgent, Tool

async def create_smolagent(self, config: AgentConfig) -> CodeAgent:
    """Create SmolAgent with Tektra security integration."""
    # Convert Tektra tools to SmolAgent tools
    tools = await self._convert_tools(config.tools)
    
    # Create agent with security wrapper
    agent = CodeAgent(
        model=config.model,
        tools=tools,
        system_prompt=config.system_prompt
    )
    
    return agent
```