"""
Real SmolAgents Integration

This module provides real SmolAgents integration with Tektra's AI backend systems.
It bridges SmolAgents with our Qwen backend and provides comprehensive tool support.

Key Features:
- Real code execution via SmolAgents
- Integration with Qwen 2.5-VL backend
- Comprehensive tool ecosystem
- Memory system integration
- Secure sandboxed execution
- Async support for Tektra architecture
"""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

# Import SmolAgents components
try:
    from smolagents import (
        CodeAgent,
        ToolCallingAgent,
        Tool,
        LiteLLMModel,
    )
    
    # Import available tools (some may not be available in all versions)
    available_tools = []
    
    try:
        from smolagents import DuckDuckGoSearchTool
        available_tools.append("DuckDuckGoSearchTool")
    except ImportError:
        DuckDuckGoSearchTool = None
    
    try:
        from smolagents.default_tools import PythonInterpreterTool
        available_tools.append("PythonInterpreterTool")
    except ImportError:
        PythonInterpreterTool = None
    
    try:
        from smolagents.default_tools import WebSearchTool
        available_tools.append("WebSearchTool")
    except ImportError:
        WebSearchTool = None
    
    SMOLAGENTS_AVAILABLE = True
    logger.success("Real SmolAgents library loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import SmolAgents: {e}")
    SMOLAGENTS_AVAILABLE = False
    
    # Create mock classes for graceful degradation
    class CodeAgent:
        def __init__(self, *args, **kwargs):
            pass
    
    class Tool:
        def __init__(self, *args, **kwargs):
            pass
    
    class CodeAgentError(Exception):
        pass
    
    class AgentError(Exception):
        pass


class SmolAgentsManager:
    """
    Manager for SmolAgents integration.
    
    Handles creation and management of SmolAgents-based AI agents.
    """
    
    def __init__(self, qwen_backend=None):
        """Initialize SmolAgents manager."""
        self.qwen_backend = qwen_backend
        self.available = SMOLAGENTS_AVAILABLE
        self.agents = {}
        
        if not self.available:
            logger.warning("SmolAgents not available - using fallback implementations")
    
    async def create_agent(self, agent_config: dict) -> str:
        """Create a new agent."""
        agent_id = str(uuid.uuid4())[:8]
        
        if self.available:
            # Create real SmolAgent
            try:
                agent = TektraCodeAgent(
                    qwen_backend=self.qwen_backend,
                    system_prompt=agent_config.get("system_prompt", "You are a helpful AI assistant."),
                    tools=agent_config.get("tools", []),
                    max_iterations=agent_config.get("max_iterations", 10)
                )
                self.agents[agent_id] = agent
                logger.info(f"Created SmolAgent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to create SmolAgent: {e}")
                raise AgentError(f"Agent creation failed: {e}")
        else:
            # Create mock agent
            self.agents[agent_id] = {
                "id": agent_id,
                "config": agent_config,
                "status": "ready"
            }
            logger.info(f"Created mock agent: {agent_id}")
        
        return agent_id
    
    async def execute_agent(self, agent_id: str, task: str) -> str:
        """Execute a task with an agent."""
        if agent_id not in self.agents:
            raise AgentError(f"Agent not found: {agent_id}")
        
        agent = self.agents[agent_id]
        
        if self.available and hasattr(agent, 'execute'):
            try:
                return await agent.execute(task)
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                raise AgentError(f"Execution failed: {e}")
        else:
            # Mock execution
            return f"Mock agent {agent_id} would execute: {task}"
    
    def list_agents(self) -> list:
        """List all agents."""
        return [
            {
                "id": agent_id,
                "status": "ready" if self.available else "mock",
                "available": self.available
            }
            for agent_id in self.agents.keys()
        ]
    
    async def cleanup(self):
        """Clean up agents."""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up agent {agent_id}: {e}")
        
        self.agents.clear()
        logger.info("SmolAgents manager cleaned up")


class QwenModelAdapter:
    """
    Adapter to make Qwen backend compatible with SmolAgents.
    
    SmolAgents expects specific model interfaces, but we want to use
    our Qwen backend. This adapter bridges the gap.
    """
    
    def __init__(self, qwen_backend):
        """Initialize with Qwen backend."""
        self.qwen_backend = qwen_backend
        self.model_name = qwen_backend.config.model_name if qwen_backend else "qwen-2.5-vl"
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Qwen backend.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response string
        """
        if not self.qwen_backend:
            raise RuntimeError("Qwen backend not available")
            
        if not self.qwen_backend.is_initialized:
            raise RuntimeError("Qwen backend not initialized - call initialize() first")
            
        try:
            # Prepare context from kwargs
            context = kwargs.get('context', {})
            
            # Add generation parameters to context if provided
            if kwargs:
                context.update({
                    'generation_params': {
                        k: v for k, v in kwargs.items() 
                        if k not in ['context']
                    }
                })
            
            # Use Qwen backend for generation
            response = await self.qwen_backend.generate_response(prompt, context)
            
            logger.debug(f"QwenModelAdapter generated response: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Error in Qwen generation: {e}")
            raise
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Synchronous interface for SmolAgents compatibility."""
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - need to use thread pool
                import concurrent.futures
                import threading
                
                logger.debug("Running Qwen generation in thread pool (async context detected)")
                
                def run_async():
                    return asyncio.run(self.generate(prompt, **kwargs))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_async)
                    result = future.result(timeout=30)  # 30 second timeout
                    return result
                    
            except RuntimeError:
                # No running loop - we can use asyncio.run directly
                logger.debug("Running Qwen generation with asyncio.run")
                return asyncio.run(self.generate(prompt, **kwargs))
                
        except Exception as e:
            error_msg = f"QwenModelAdapter error: {e}"
            logger.error(error_msg)
            return f"Error: {e}"
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model information for debugging."""
        return {
            "model_name": self.model_name,
            "backend_available": self.qwen_backend is not None,
            "backend_initialized": self.qwen_backend.is_initialized if self.qwen_backend else False,
            "is_vision_model": self.qwen_backend.is_vision_model if self.qwen_backend else False,
        }


class TektraCodeAgent:
    """
    Tektra-enhanced CodeAgent with memory integration and async support.
    
    This wraps SmolAgents CodeAgent with our specific requirements:
    - Memory system integration
    - Async-first design
    - Enhanced security
    - Better error handling
    """
    
    def __init__(
        self,
        qwen_backend=None,
        tools=None,
        memory_manager=None,
        system_prompt=None,
        max_iterations=10,
        executor_type="local",
        additional_authorized_imports=None,
    ):
        """
        Initialize Tektra CodeAgent.
        
        Args:
            qwen_backend: Qwen backend for model inference
            tools: List of tools available to the agent
            memory_manager: Memory system for context and learning
            system_prompt: Custom system prompt
            max_iterations: Maximum execution iterations
            executor_type: Code execution environment
            additional_authorized_imports: Additional Python imports allowed
        """
        self.qwen_backend = qwen_backend
        self.memory_manager = memory_manager
        self.max_iterations = max_iterations
        
        # Create model adapter
        if qwen_backend:
            self.model = QwenModelAdapter(qwen_backend)
        else:
            # Fallback to basic LiteLLM model
            self.model = LiteLLMModel(model_id="gpt-3.5-turbo")
            logger.warning("No Qwen backend provided, using fallback model")
        
        # Setup tools
        self.tools = tools or self._get_default_tools()
        
        # Initialize SmolAgents CodeAgent
        try:
            # For now, don't use custom prompt templates due to SmolAgents requirements
            # TODO: Investigate how to properly customize system prompt in SmolAgents
            prompt_templates = None
            if system_prompt:
                # Store system prompt for potential future use
                self.system_prompt = system_prompt
                logger.debug(f"Stored system prompt for future use: {system_prompt[:50]}...")
            else:
                self.system_prompt = None
            
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.model,
                prompt_templates=prompt_templates,
                executor_type=executor_type,
                additional_authorized_imports=additional_authorized_imports or [],
            )
            logger.info("CodeAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CodeAgent: {e}")
            self.agent = None
    
    def _get_default_tools(self) -> list[Tool]:
        """Get default tool set for agents."""
        default_tools = []
        
        # Add available SmolAgents tools
        if DuckDuckGoSearchTool:
            try:
                default_tools.append(DuckDuckGoSearchTool())
                logger.debug("Added DuckDuckGoSearchTool")
            except Exception as e:
                logger.warning(f"Could not add DuckDuckGoSearchTool: {e}")
        
        if PythonInterpreterTool:
            try:
                default_tools.append(PythonInterpreterTool())
                logger.debug("Added PythonInterpreterTool")
            except Exception as e:
                logger.warning(f"Could not add PythonInterpreterTool: {e}")
        
        # Add custom Tektra tools (disabled temporarily due to SmolAgents Tool interface issues)
        # TODO: Fix custom tool creation to work with SmolAgents Tool class
        # default_tools.extend(self._get_tektra_tools())
        
        logger.info(f"Loaded {len(default_tools)} tools for agent")
        return default_tools
    
    def _get_tektra_tools(self) -> list[Tool]:
        """Get Tektra-specific tools."""
        tools = []
        
        # Memory tool
        if self.memory_manager:
            tools.append(self._create_memory_tool())
        
        # File operations tool
        tools.append(self._create_file_tool())
        
        # Configuration tool
        tools.append(self._create_config_tool())
        
        return tools
    
    def _create_memory_tool(self) -> Tool:
        """Create memory interaction tool."""
        async def search_memory(query: str, max_results: int = 5) -> str:
            """Search agent memory for relevant information."""
            try:
                if not self.memory_manager:
                    return "Memory system not available"
                
                # Search memory
                from ..memory.memory_types import MemoryContext
                context = MemoryContext(
                    query=query,
                    max_results=max_results,
                    min_relevance=0.3,
                )
                
                result = await self.memory_manager.search_memories(context)
                
                if result.entries:
                    memories = []
                    for entry in result.entries:
                        memories.append(f"[{entry.timestamp}] {entry.content}")
                    return f"Found {len(memories)} relevant memories:\n" + "\n".join(memories)
                else:
                    return "No relevant memories found"
                    
            except Exception as e:
                logger.error(f"Memory search error: {e}")
                return f"Memory search failed: {e}"
        
        async def save_memory(content: str, importance: float = 0.7) -> str:
            """Save information to agent memory."""
            try:
                if not self.memory_manager:
                    return "Memory system not available"
                
                # Save to memory
                await self.memory_manager.add_agent_context(
                    agent_id="current",  # Will be set properly by the system
                    context=content,
                    importance=importance
                )
                
                return f"Saved to memory with importance {importance}"
                
            except Exception as e:
                logger.error(f"Memory save error: {e}")
                return f"Memory save failed: {e}"
        
        # Wrap async functions for SmolAgents
        def memory_search_sync(query: str, max_results: int = 5) -> str:
            return asyncio.run(search_memory(query, max_results))
        
        def memory_save_sync(content: str, importance: float = 0.7) -> str:
            return asyncio.run(save_memory(content, importance))
        
        # Create compound memory tool
        def memory_tool(action: str, **kwargs) -> str:
            """
            Interact with agent memory system.
            
            Actions:
            - search: Search memory for information (query, max_results)
            - save: Save information to memory (content, importance)
            """
            if action == "search":
                return memory_search_sync(
                    kwargs.get("query", ""),
                    kwargs.get("max_results", 5)
                )
            elif action == "save":
                return memory_save_sync(
                    kwargs.get("content", ""),
                    kwargs.get("importance", 0.7)
                )
            else:
                return f"Unknown memory action: {action}. Use 'search' or 'save'"
        
        # Create tool using SmolAgents Tool class properly
        try:
            return Tool(
                name="memory",
                description="Search and save information in agent memory. Use action='search' with query parameter, or action='save' with content and importance parameters.",
                inputs={
                    "action": {
                        "type": "string", 
                        "description": "Action to perform: 'search' or 'save'"
                    },
                    "query": {
                        "type": "string", 
                        "description": "Query for search action (optional)"
                    },
                    "content": {
                        "type": "string", 
                        "description": "Content to save for save action (optional)"
                    },
                    "importance": {
                        "type": "number", 
                        "description": "Importance score for save action (optional, default 0.7)"
                    }
                },
                output_type="string",
            )
        except Exception as e:
            logger.warning(f"Could not create memory tool with inputs schema: {e}")
            # Fallback to simpler tool creation
            return Tool(
                name="memory", 
                description="Search and save information in agent memory",
                function=memory_tool
            )
    
    def _create_file_tool(self) -> Tool:
        """Create file operations tool."""
        def file_operations(action: str, path: str = "", content: str = "") -> str:
            """
            Perform file operations safely.
            
            Actions:
            - read: Read file content
            - write: Write content to file
            - list: List files in directory
            """
            try:
                file_path = Path(path)
                
                # Security: Restrict to safe directories
                safe_dirs = [Path.cwd(), Path(tempfile.gettempdir())]
                if not any(file_path.is_relative_to(safe_dir) for safe_dir in safe_dirs):
                    return f"Access denied: Path not in safe directories"
                
                if action == "read":
                    if file_path.exists() and file_path.is_file():
                        return file_path.read_text()
                    else:
                        return f"File not found: {path}"
                
                elif action == "write":
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)
                    return f"Successfully wrote to {path}"
                
                elif action == "list":
                    if file_path.exists() and file_path.is_dir():
                        files = [f.name for f in file_path.iterdir()]
                        return f"Files in {path}: {', '.join(files)}"
                    else:
                        return f"Directory not found: {path}"
                
                else:
                    return f"Unknown file action: {action}"
                    
            except Exception as e:
                logger.error(f"File operation error: {e}")
                return f"File operation failed: {e}"
        
        # Create tool using SmolAgents Tool class properly
        try:
            return Tool(
                name="file_ops",
                description="Perform safe file operations. Use action='read/write/list' with path parameter, and content for write operations.",
                inputs={
                    "action": {
                        "type": "string", 
                        "description": "Action: 'read', 'write', or 'list'"
                    },
                    "path": {
                        "type": "string", 
                        "description": "File or directory path"
                    },
                    "content": {
                        "type": "string", 
                        "description": "Content for write operations (optional)"
                    }
                },
                output_type="string",
            )
        except Exception as e:
            logger.warning(f"Could not create file tool with inputs schema: {e}")
            # Fallback to simpler tool creation
            return Tool(
                name="file_ops", 
                description="Perform safe file operations",
                function=file_operations
            )
    
    def _create_config_tool(self) -> Tool:
        """Create configuration access tool."""
        def get_config(key: str = "") -> str:
            """Get configuration information."""
            try:
                # This would integrate with Tektra's config system
                config_info = {
                    "app_name": "Tektra AI Assistant",
                    "version": "0.1.0",
                    "environment": "development",
                    "features": ["voice", "vision", "agents", "memory"],
                }
                
                if key:
                    return str(config_info.get(key, f"Unknown config key: {key}"))
                else:
                    return json.dumps(config_info, indent=2)
                    
            except Exception as e:
                logger.error(f"Config access error: {e}")
                return f"Config access failed: {e}"
        
        # Create tool using SmolAgents Tool class properly
        try:
            return Tool(
                name="config",
                description="Access Tektra configuration information. Provide key parameter for specific config values.",
                inputs={
                    "key": {
                        "type": "string", 
                        "description": "Configuration key to retrieve (optional, returns all if empty)"
                    }
                },
                output_type="string",
            )
        except Exception as e:
            logger.warning(f"Could not create config tool with inputs schema: {e}")
            # Fallback to simpler tool creation
            return Tool(
                name="config", 
                description="Access Tektra configuration information",
                function=get_config
            )
    
    async def run(self, task: str, agent_id: str = None) -> dict[str, Any]:
        """
        Run the agent with a task.
        
        Args:
            task: The task for the agent to perform
            agent_id: Optional agent ID for memory context
            
        Returns:
            Dictionary with execution results
        """
        if not self.agent:
            return {
                "success": False,
                "error": "CodeAgent not properly initialized",
                "task": task,
            }
        
        try:
            logger.info(f"CodeAgent executing task: {task[:100]}...")
            
            # Add memory context if available
            enhanced_task = task
            if self.memory_manager and agent_id:
                try:
                    # Search for relevant context
                    from ..memory.memory_types import MemoryContext
                    context = MemoryContext(
                        agent_id=agent_id,
                        query=task,
                        max_results=3,
                        min_relevance=0.5,
                    )
                    
                    memory_result = await self.memory_manager.search_memories(context)
                    if memory_result.entries:
                        context_info = []
                        for entry in memory_result.entries:
                            context_info.append(f"- {entry.content}")
                        
                        enhanced_task = f"""
Previous context:
{chr(10).join(context_info)}

Current task: {task}
"""
                except Exception as e:
                    logger.warning(f"Could not enhance task with memory: {e}")
            
            # Execute the agent
            result = self.agent.run(enhanced_task)
            
            # Save result to memory if available
            if self.memory_manager and agent_id:
                try:
                    await self.memory_manager.add_task_result(
                        task_description=task,
                        result=str(result),
                        success=True,
                        agent_id=agent_id,
                    )
                except Exception as e:
                    logger.warning(f"Could not save result to memory: {e}")
            
            return {
                "success": True,
                "result": result,
                "task": task,
                "agent_type": "CodeAgent",
                "enhanced_task": enhanced_task != task,
            }
            
        except Exception as e:
            logger.error(f"CodeAgent execution failed: {e}")
            
            # Save failure to memory
            if self.memory_manager and agent_id:
                try:
                    await self.memory_manager.add_task_result(
                        task_description=task,
                        result=str(e),
                        success=False,
                        agent_id=agent_id,
                    )
                except Exception as save_e:
                    logger.warning(f"Could not save failure to memory: {save_e}")
            
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "agent_type": "CodeAgent",
            }


class TektraToolCallingAgent:
    """
    Tektra-enhanced ToolCallingAgent with memory integration and async support.
    
    Similar to TektraCodeAgent but uses tool calling approach instead of code generation.
    """
    
    def __init__(
        self,
        qwen_backend=None,
        tools=None,
        memory_manager=None,
        system_prompt=None,
        max_tool_threads=4,
    ):
        """Initialize Tektra ToolCallingAgent."""
        self.qwen_backend = qwen_backend
        self.memory_manager = memory_manager
        
        # Create model adapter
        if qwen_backend:
            self.model = QwenModelAdapter(qwen_backend)
        else:
            self.model = LiteLLMModel(model_id="gpt-3.5-turbo")
            logger.warning("No Qwen backend provided, using fallback model")
        
        # Setup tools (reuse CodeAgent tool logic)
        self.tools = tools or self._get_default_tools()
        
        # Initialize SmolAgents ToolCallingAgent
        try:
            # For now, don't use custom prompt templates due to SmolAgents requirements
            # TODO: Investigate how to properly customize system prompt in SmolAgents
            prompt_templates = None
            if system_prompt:
                # Store system prompt for potential future use
                self.system_prompt = system_prompt
                logger.debug(f"Stored system prompt for future use: {system_prompt[:50]}...")
            else:
                self.system_prompt = None
            
            self.agent = ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                prompt_templates=prompt_templates,
                max_tool_threads=max_tool_threads,
            )
            logger.info("ToolCallingAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ToolCallingAgent: {e}")
            self.agent = None
    
    def _get_default_tools(self) -> list[Tool]:
        """Get default tool set (reuse CodeAgent implementation)."""
        # For simplicity, reuse the same tool creation logic
        code_agent = TektraCodeAgent(memory_manager=self.memory_manager)
        return code_agent._get_default_tools()
    
    async def run(self, task: str, agent_id: str = None) -> dict[str, Any]:
        """
        Run the tool calling agent with a task.
        
        Args:
            task: The task for the agent to perform
            agent_id: Optional agent ID for memory context
            
        Returns:
            Dictionary with execution results
        """
        if not self.agent:
            return {
                "success": False,
                "error": "ToolCallingAgent not properly initialized",
                "task": task,
            }
        
        try:
            logger.info(f"ToolCallingAgent executing task: {task[:100]}...")
            
            # Execute the agent
            result = self.agent.run(task)
            
            # Save result to memory if available
            if self.memory_manager and agent_id:
                try:
                    await self.memory_manager.add_task_result(
                        task_description=task,
                        result=str(result),
                        success=True,
                        agent_id=agent_id,
                    )
                except Exception as e:
                    logger.warning(f"Could not save result to memory: {e}")
            
            return {
                "success": True,
                "result": result,
                "task": task,
                "agent_type": "ToolCallingAgent",
            }
            
        except Exception as e:
            logger.error(f"ToolCallingAgent execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "agent_type": "ToolCallingAgent",
            }


# Compatibility layer for existing code
def get_real_smolagents():
    """Get real SmolAgents implementation with Tektra enhancements."""
    import types
    
    module = types.ModuleType("smolagents")
    module.CodeAgent = TektraCodeAgent
    module.ToolCallingAgent = TektraToolCallingAgent
    module.Tool = Tool if SMOLAGENTS_AVAILABLE else None
    module.QwenModelAdapter = QwenModelAdapter
    
    return module


# Main module logic - try real SmolAgents, enhance with Tektra features
if SMOLAGENTS_AVAILABLE:
    logger.success("Using real SmolAgents with Tektra enhancements")
    # Export enhanced versions
    smolagents = get_real_smolagents()
else:
    logger.error("SmolAgents not available - this should not happen in production")
    smolagents = None