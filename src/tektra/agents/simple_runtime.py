"""
Simple Agent Runtime

This module provides a simplified runtime for executing agents without the
complexity of Docker or advanced sandboxing. It focuses on basic functionality
that works reliably.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from .builder import AgentSpecification
from .simple_agent import PythonAgent, SimpleAgentFactory, AgentStatus, AgentResult


class SimpleAgentState(Enum):
    """Simple agent states."""
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RunningAgent:
    """Information about a running agent."""
    id: str
    spec: AgentSpecification
    agent: PythonAgent
    state: SimpleAgentState
    created_at: datetime
    last_activity: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.spec.name,
            "description": self.spec.description,
            "type": self.spec.type.value,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }


class SimpleAgentRuntime:
    """
    Simplified agent runtime for basic agent execution.
    
    This runtime manages Python agents without Docker or complex sandboxing,
    focusing on reliability and ease of use.
    """

    def __init__(self, llm_backend=None, memory_manager=None):
        """
        Initialize the simple agent runtime.
        
        Args:
            llm_backend: LLM backend for agent intelligence
            memory_manager: Memory manager for context (optional)
        """
        self.llm_backend = llm_backend
        self.memory_manager = memory_manager
        self.running_agents: Dict[str, RunningAgent] = {}
        self.agent_factory = SimpleAgentFactory()
        
        logger.info("Simple Agent Runtime initialized")

    async def deploy_agent(self, spec: AgentSpecification) -> str:
        """
        Deploy an agent from specification.
        
        Args:
            spec: Agent specification
            
        Returns:
            str: Agent ID
        """
        try:
            # Create a Python agent
            python_agent = self.agent_factory.create_python_agent(
                name=spec.name,
                description=spec.description
            )
            
            # Create running agent record
            running_agent = RunningAgent(
                id=spec.id,
                spec=spec,
                agent=python_agent,
                state=SimpleAgentState.CREATED,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Register the agent
            self.running_agents[spec.id] = running_agent
            
            logger.info(f"Agent deployed: {spec.name} ({spec.id})")
            return spec.id
            
        except Exception as e:
            logger.error(f"Failed to deploy agent: {e}")
            raise

    async def execute_agent_task(self, agent_id: str, task: str) -> AgentResult:
        """
        Execute a task with a specific agent.
        
        Args:
            agent_id: Agent ID
            task: Task to execute
            
        Returns:
            AgentResult: Execution result
        """
        if agent_id not in self.running_agents:
            return AgentResult(
                success=False,
                output="",
                error=f"Agent {agent_id} not found"
            )
        
        try:
            running_agent = self.running_agents[agent_id]
            running_agent.state = SimpleAgentState.RUNNING
            running_agent.last_activity = datetime.now()
            
            # Execute the task
            result = await running_agent.agent.run_natural_language_task(
                task=task,
                llm_backend=self.llm_backend
            )
            
            # Update state based on result
            if result.success:
                running_agent.state = SimpleAgentState.STOPPED
            else:
                running_agent.state = SimpleAgentState.ERROR
            
            running_agent.last_activity = datetime.now()
            
            logger.info(f"Agent {agent_id} executed task: {task[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing agent task: {e}")
            
            if agent_id in self.running_agents:
                self.running_agents[agent_id].state = SimpleAgentState.ERROR
            
            return AgentResult(
                success=False,
                output="",
                error=f"Execution error: {e}"
            )

    async def execute_agent_code(self, agent_id: str, code: str) -> AgentResult:
        """
        Execute Python code with a specific agent.
        
        Args:
            agent_id: Agent ID
            code: Python code to execute
            
        Returns:
            AgentResult: Execution result
        """
        if agent_id not in self.running_agents:
            return AgentResult(
                success=False,
                output="",
                error=f"Agent {agent_id} not found"
            )
        
        try:
            running_agent = self.running_agents[agent_id]
            running_agent.state = SimpleAgentState.RUNNING
            running_agent.last_activity = datetime.now()
            
            # Execute the code
            result = await running_agent.agent.execute_code(code)
            
            # Update state based on result
            if result.success:
                running_agent.state = SimpleAgentState.STOPPED
            else:
                running_agent.state = SimpleAgentState.ERROR
            
            running_agent.last_activity = datetime.now()
            
            logger.info(f"Agent {agent_id} executed code")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing agent code: {e}")
            
            if agent_id in self.running_agents:
                self.running_agents[agent_id].state = SimpleAgentState.ERROR
            
            return AgentResult(
                success=False,
                output="",
                error=f"Execution error: {e}"
            )

    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a running agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            bool: True if stopped successfully
        """
        if agent_id not in self.running_agents:
            return False
        
        try:
            running_agent = self.running_agents[agent_id]
            running_agent.state = SimpleAgentState.STOPPED
            running_agent.last_activity = datetime.now()
            
            logger.info(f"Agent {agent_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping agent: {e}")
            return False

    async def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the runtime.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            bool: True if removed successfully
        """
        if agent_id not in self.running_agents:
            return False
        
        try:
            running_agent = self.running_agents[agent_id]
            
            # Clean up agent resources
            running_agent.agent.cleanup()
            
            # Remove from registry
            del self.running_agents[agent_id]
            
            logger.info(f"Agent {agent_id} removed")
            return True
            
        except Exception as e:
            logger.error(f"Error removing agent: {e}")
            return False

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            dict: Agent information or None if not found
        """
        if agent_id not in self.running_agents:
            return None
        
        running_agent = self.running_agents[agent_id]
        
        info = running_agent.to_dict()
        
        # Add execution history
        info["execution_history"] = [
            {
                "success": result.success,
                "output": result.output[:100] + "..." if len(result.output) > 100 else result.output,
                "error": result.error,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            }
            for result in running_agent.agent.get_execution_history()
        ]
        
        return info

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all running agents.
        
        Returns:
            list: List of agent information
        """
        return [running_agent.to_dict() for running_agent in self.running_agents.values()]

    def get_agent_count(self) -> int:
        """Get the number of running agents."""
        return len(self.running_agents)

    def get_running_agents(self) -> List[Dict[str, Any]]:
        """Get list of currently running agents."""
        return [
            running_agent.to_dict() 
            for running_agent in self.running_agents.values()
            if running_agent.state == SimpleAgentState.RUNNING
        ]

    async def cleanup(self):
        """Clean up all agents and resources."""
        try:
            agent_ids = list(self.running_agents.keys())
            
            for agent_id in agent_ids:
                await self.remove_agent(agent_id)
            
            logger.info("Simple Agent Runtime cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the runtime.
        
        Returns:
            dict: Health information
        """
        return {
            "status": "healthy",
            "total_agents": len(self.running_agents),
            "running_agents": len(self.get_running_agents()),
            "llm_backend_available": self.llm_backend is not None and getattr(self.llm_backend, 'is_initialized', False),
            "memory_manager_available": self.memory_manager is not None,
        }