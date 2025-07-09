"""
Tektra Agent System

This module provides the agent orchestration capabilities for Tektra,
enabling users to create, deploy, and manage AI agents through natural language.

The agent system is built on SmolAgents and provides:
- Natural language agent creation
- Secure sandboxed execution
- Agent lifecycle management
- Inter-agent communication
- Persistent agent state
"""

from .builder import AgentBuilder, AgentSpecification
from .runtime import AgentRuntime, AgentSandbox, SandboxType
from .registry import AgentRegistry, AgentStatus
from .templates import AgentTemplate, TemplateLibrary

__all__ = [
    'AgentBuilder',
    'AgentSpecification', 
    'AgentRuntime',
    'AgentSandbox',
    'SandboxType',
    'AgentRegistry',
    'AgentStatus',
    'AgentTemplate',
    'TemplateLibrary'
]