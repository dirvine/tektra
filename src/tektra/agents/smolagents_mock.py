"""
Mock SmolAgents Implementation

This provides a mock implementation of SmolAgents for development
until the real SmolAgents library is available.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import asyncio
import json
from loguru import logger


@dataclass
class Tool:
    """Mock tool for agents."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = None


class BaseAgent:
    """Base class for mock agents."""
    
    def __init__(
        self,
        tools: List[Tool] = None,
        model: Any = None,
        system_prompt: str = "",
        memory: List[Dict[str, Any]] = None
    ):
        self.tools = tools or []
        self.model = model
        self.system_prompt = system_prompt
        self.memory = memory or []
        
    async def run(self, task: str) -> Any:
        """Run the agent with a task."""
        logger.info(f"Mock agent running task: {task}")
        
        # Simple mock response
        return {
            "task": task,
            "result": f"Mock result for: {task}",
            "status": "success",
            "agent_type": self.__class__.__name__
        }


class CodeAgent(BaseAgent):
    """Mock implementation of CodeAgent."""
    
    async def run(self, task: str) -> Any:
        """Run code agent."""
        logger.info(f"CodeAgent executing: {task}")
        
        # Mock code generation
        code = f"""
# Generated code for: {task}
def solution():
    # Mock implementation
    return "Task completed"
    
result = solution()
"""
        
        return {
            "task": task,
            "code": code,
            "result": "Task completed (mock)",
            "status": "success"
        }


class ToolCallingAgent(BaseAgent):
    """Mock implementation of ToolCallingAgent."""
    
    async def run(self, task: str) -> Any:
        """Run tool calling agent."""
        logger.info(f"ToolCallingAgent executing: {task}")
        
        # Mock tool calls
        tool_calls = [
            {
                "tool": "search",
                "arguments": {"query": task},
                "result": "Mock search results"
            }
        ]
        
        return {
            "task": task,
            "tool_calls": tool_calls,
            "result": "Task completed via tool calls (mock)",
            "status": "success"
        }


class InferenceClientModel:
    """Mock model for agents."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        
    async def generate(self, prompt: str) -> str:
        """Generate response."""
        return f"Mock response to: {prompt}"
    
    def get_tool_call(self, prompt: str) -> Dict[str, Any]:
        """Get tool call from prompt."""
        return {
            "tool": "mock_tool",
            "arguments": {"input": prompt}
        }


# Mock the smolagents imports
def get_mock_smolagents():
    """Get mock smolagents module."""
    import types
    
    module = types.ModuleType('smolagents')
    module.CodeAgent = CodeAgent
    module.ToolCallingAgent = ToolCallingAgent
    module.Tool = Tool
    module.InferenceClientModel = InferenceClientModel
    
    return module


# Try to import real smolagents, fall back to mock
try:
    import smolagents
    logger.info("Using real SmolAgents library")
except ImportError:
    logger.warning("SmolAgents not available, using mock implementation")
    import sys
    sys.modules['smolagents'] = get_mock_smolagents()
    smolagents = sys.modules['smolagents']