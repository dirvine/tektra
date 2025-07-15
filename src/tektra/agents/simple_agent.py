"""
Simple Agent Implementation

This module provides a basic, working agent implementation that can execute
Python code safely. This is a simplified version of the agent system that
focuses on functionality over complexity.
"""

import asyncio
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class AgentStatus(Enum):
    """Agent execution status."""
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SimpleAgentSpec:
    """Specification for a simple agent."""
    id: str
    name: str
    description: str
    agent_type: str
    code: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PythonAgent:
    """
    A simple Python code execution agent.
    
    This agent can execute Python code safely in a subprocess with timeouts
    and basic safety measures.
    """

    def __init__(self, spec: SimpleAgentSpec):
        """
        Initialize the Python agent.
        
        Args:
            spec: Agent specification
        """
        self.spec = spec
        self.status = AgentStatus.CREATED
        self.execution_history = []
        self.working_directory = None
        
        # Safety settings
        self.timeout = 30  # Maximum execution time in seconds
        self.max_output_size = 1024 * 1024  # 1MB max output
        
        # Create working directory
        self._setup_working_directory()
        
        logger.info(f"Python agent '{spec.name}' initialized")

    def _setup_working_directory(self):
        """Set up a temporary working directory for the agent."""
        try:
            self.working_directory = Path(tempfile.mkdtemp(prefix=f"agent_{self.spec.id}_"))
            logger.debug(f"Working directory: {self.working_directory}")
        except Exception as e:
            logger.error(f"Failed to create working directory: {e}")
            self.working_directory = Path("/tmp")

    async def execute_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            context: Optional execution context
            
        Returns:
            AgentResult with execution results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.status = AgentStatus.RUNNING
            
            # Validate code safety
            if not self._validate_code_safety(code):
                return AgentResult(
                    success=False,
                    output="",
                    error="Code validation failed: potentially unsafe operations detected",
                    execution_time=0.0
                )
            
            # Write code to temporary file
            code_file = self.working_directory / f"execution_{uuid.uuid4()}.py"
            
            # Wrap code with safety measures
            wrapped_code = self._wrap_code_with_safety(code)
            
            with open(code_file, 'w') as f:
                f.write(wrapped_code)
            
            # Execute code in subprocess
            process = await asyncio.create_subprocess_exec(
                'python', str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_directory),
                limit=self.max_output_size
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Process results
                output = stdout.decode('utf-8', errors='ignore')
                error = stderr.decode('utf-8', errors='ignore')
                
                success = process.returncode == 0
                
                result = AgentResult(
                    success=success,
                    output=output,
                    error=error if error else None,
                    execution_time=execution_time
                )
                
                # Clean up
                try:
                    code_file.unlink()
                except:
                    pass
                
                # Update status
                self.status = AgentStatus.STOPPED if success else AgentStatus.ERROR
                
                # Store in history
                self.execution_history.append(result)
                
                logger.info(f"Code execution {'succeeded' if success else 'failed'} in {execution_time:.2f}s")
                
                return result
                
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                
                return AgentResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout} seconds",
                    execution_time=self.timeout
                )
                
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Code execution error: {e}")
            
            self.status = AgentStatus.ERROR
            
            return AgentResult(
                success=False,
                output="",
                error=f"Execution error: {e}",
                execution_time=execution_time
            )

    def _validate_code_safety(self, code: str) -> bool:
        """
        Validate code for basic safety.
        
        Args:
            code: Code to validate
            
        Returns:
            bool: True if code appears safe
        """
        # Basic blacklist of dangerous operations
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'import shutil',
            'import socket',
            'import urllib',
            'import requests',
            'import http',
            'open(',
            'exec(',
            'eval(',
            '__import__',
            'globals()',
            'locals()',
            'dir(',
            'file(',
            'input(',
            'raw_input(',
            'exit(',
            'quit(',
            'reload(',
            'compile(',
            'delattr(',
            'setattr(',
            'getattr(',
            'hasattr(',
        ]
        
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                logger.warning(f"Code validation failed: found dangerous pattern '{pattern}'")
                return False
        
        return True

    def _wrap_code_with_safety(self, code: str) -> str:
        """
        Wrap code with safety measures.
        
        Args:
            code: Original code
            
        Returns:
            str: Wrapped code with safety measures
        """
        wrapped = f"""
# Safety wrapper for agent code execution
import sys
import signal
import time
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = captured_output = StringIO()

# Set up timeout handler
def timeout_handler(signum, frame):
    print("ERROR: Code execution timed out", file=sys.stderr)
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.timeout})

try:
    # Execute user code
{self._indent_code(code, 4)}
    
    # Restore stdout and print captured output
    sys.stdout = old_stdout
    print(captured_output.getvalue(), end='')
    
except Exception as e:
    sys.stdout = old_stdout
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
finally:
    signal.alarm(0)  # Cancel the alarm
"""
        return wrapped

    def _indent_code(self, code: str, spaces: int) -> str:
        """
        Indent code by specified number of spaces.
        
        Args:
            code: Code to indent
            spaces: Number of spaces to indent
            
        Returns:
            str: Indented code
        """
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in code.split('\n'))

    async def run_natural_language_task(self, task: str, llm_backend) -> AgentResult:
        """
        Execute a natural language task by converting it to Python code.
        
        Args:
            task: Natural language description of the task
            llm_backend: LLM backend for code generation
            
        Returns:
            AgentResult with execution results
        """
        try:
            # Generate Python code from natural language
            code_prompt = f"""
Convert this natural language task into Python code:

Task: {task}

Requirements:
- Use only safe Python operations
- No file I/O, network operations, or system calls
- Focus on data manipulation, calculations, and simple logic
- Include print statements to show results
- Keep it simple and safe

Python code:
"""
            
            if llm_backend and llm_backend.is_initialized:
                generated_code = await llm_backend.generate_response(
                    code_prompt,
                    max_tokens=256,
                    temperature=0.3
                )
                
                # Clean up the response to extract just the code
                code_lines = []
                in_code_block = False
                
                for line in generated_code.split('\n'):
                    if line.strip().startswith('```python'):
                        in_code_block = True
                        continue
                    elif line.strip().startswith('```'):
                        in_code_block = False
                        continue
                    elif in_code_block or (not line.startswith('#') and line.strip()):
                        code_lines.append(line)
                
                code = '\n'.join(code_lines).strip()
                
                if code:
                    logger.info(f"Generated code for task '{task}': {code[:100]}...")
                    return await self.execute_code(code)
                else:
                    return AgentResult(
                        success=False,
                        output="",
                        error="Failed to generate valid Python code from task"
                    )
            else:
                return AgentResult(
                    success=False,
                    output="",
                    error="LLM backend not available for code generation"
                )
                
        except Exception as e:
            logger.error(f"Error in natural language task execution: {e}")
            return AgentResult(
                success=False,
                output="",
                error=f"Task execution error: {e}"
            )

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information.
        
        Returns:
            dict: Agent information
        """
        return {
            "id": self.spec.id,
            "name": self.spec.name,
            "description": self.spec.description,
            "type": self.spec.agent_type,
            "status": self.status.value,
            "created_at": self.spec.created_at.isoformat(),
            "execution_count": len(self.execution_history),
            "working_directory": str(self.working_directory)
        }

    def get_execution_history(self) -> List[AgentResult]:
        """Get the execution history."""
        return self.execution_history.copy()

    def cleanup(self):
        """Clean up agent resources."""
        try:
            if self.working_directory and self.working_directory.exists():
                import shutil
                shutil.rmtree(self.working_directory)
                logger.debug(f"Cleaned up working directory: {self.working_directory}")
        except Exception as e:
            logger.warning(f"Error cleaning up working directory: {e}")


class SimpleAgentFactory:
    """Factory for creating simple agents."""

    @staticmethod
    def create_python_agent(name: str, description: str) -> PythonAgent:
        """
        Create a Python execution agent.
        
        Args:
            name: Agent name
            description: Agent description
            
        Returns:
            PythonAgent: Ready-to-use Python agent
        """
        spec = SimpleAgentSpec(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            agent_type="python",
            code=""  # Code will be generated dynamically
        )
        
        return PythonAgent(spec)

    @staticmethod
    def create_from_description(description: str) -> PythonAgent:
        """
        Create an agent from a natural language description.
        
        Args:
            description: Natural language description
            
        Returns:
            PythonAgent: Ready-to-use agent
        """
        # Extract name from description (simple heuristic)
        words = description.split()
        name = " ".join(words[:3]) if len(words) >= 3 else description
        
        if len(name) > 30:
            name = name[:30] + "..."
        
        return SimpleAgentFactory.create_python_agent(name, description)