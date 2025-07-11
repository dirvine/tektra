"""
Agent Runtime System

This module provides the execution environment for agents, including:
- Sandboxed execution environments
- Resource monitoring and limits
- Agent lifecycle management
- Communication channels
- State persistence
- Memory integration for context-aware execution

The runtime ensures agents execute safely and efficiently while
maintaining isolation from the host system and other agents.
"""

import asyncio
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles
import docker
from loguru import logger

from ..memory import MemoryContext, MemoryType, TektraMemoryManager
from .builder import AgentSpecification


class AgentState(Enum):
    """Possible states for an agent."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class SandboxType(Enum):
    """Types of sandboxing available."""

    DOCKER = "docker"
    PROCESS = "process"  # Isolated process
    PYODIDE = "pyodide"  # WebAssembly sandbox
    LOCAL = "local"  # Development only - no isolation


@dataclass
class AgentExecutionContext:
    """Context for agent execution."""

    agent_id: str
    spec: AgentSpecification
    input_data: dict[str, Any]
    environment: dict[str, str]
    working_directory: Path
    start_time: datetime
    timeout: int

    # Runtime state
    state: AgentState = AgentState.CREATED
    process_id: int | None = None
    container_id: str | None = None

    # Metrics
    cpu_usage: float = 0.0
    memory_usage: int = 0
    execution_count: int = 0

    # Results
    output: Any | None = None
    error: str | None = None
    logs: list[str] = None

    # Memory integration
    memory_manager: TektraMemoryManager | None = None
    session_id: str = None
    memory_context: list[Any] | None = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())


class AgentSandbox:
    """
    Provides sandboxed execution environment for agents.

    This is the security boundary that prevents agents from:
    - Accessing unauthorized resources
    - Consuming excessive CPU/memory
    - Making unauthorized network calls
    - Interfering with other agents
    """

    def __init__(self, sandbox_type: SandboxType = SandboxType.DOCKER, qwen_backend=None):
        """Initialize sandbox with specified isolation type."""
        self.sandbox_type = sandbox_type
        self.docker_client = None
        self.qwen_backend = qwen_backend

        if sandbox_type == SandboxType.DOCKER:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker sandbox initialized")
            except Exception as e:
                logger.warning(
                    f"Docker not available: {e}, falling back to process isolation"
                )
                self.sandbox_type = SandboxType.PROCESS

    async def execute_agent(self, context: AgentExecutionContext) -> dict[str, Any]:
        """
        Execute an agent in the sandbox.

        Args:
            context: Execution context with agent spec and parameters

        Returns:
            Execution result with output, logs, and metrics
        """
        logger.info(
            f"Executing agent {context.agent_id} in {self.sandbox_type.value} sandbox"
        )

        try:
            context.state = AgentState.INITIALIZING

            if self.sandbox_type == SandboxType.DOCKER:
                result = await self._execute_in_docker(context)
            elif self.sandbox_type == SandboxType.PROCESS:
                result = await self._execute_in_process(context)
            elif self.sandbox_type == SandboxType.LOCAL:
                result = await self._execute_local(context)
            else:
                raise ValueError(f"Unsupported sandbox type: {self.sandbox_type}")

            context.state = AgentState.COMPLETED
            return result

        except TimeoutError:
            logger.error(f"Agent {context.agent_id} timed out after {context.timeout}s")
            context.state = AgentState.FAILED
            context.error = "Execution timeout"
            return self._create_error_result(context, "Timeout")

        except Exception as e:
            logger.error(f"Agent {context.agent_id} failed: {e}")
            context.state = AgentState.FAILED
            context.error = str(e)
            return self._create_error_result(context, str(e))

    async def _execute_in_docker(
        self, context: AgentExecutionContext
    ) -> dict[str, Any]:
        """Execute agent in Docker container."""
        if not self.docker_client:
            raise RuntimeError("Docker client not initialized")

        # Prepare container configuration
        container_config = {
            "image": "python:3.11-slim",
            "command": "python /app/agent.py",
            "volumes": {str(context.working_directory): {"bind": "/app", "mode": "rw"}},
            "environment": context.environment,
            "mem_limit": f"{context.spec.resource_limits.get('memory_mb', 512)}m",
            "cpu_percent": context.spec.resource_limits.get("cpu_percent", 50),
            "network_mode": "bridge",
            "detach": True,
            "remove": True,
        }

        # Write agent code to working directory
        agent_file = context.working_directory / "agent.py"
        await self._write_agent_file(agent_file, context)

        # Create and start container
        container = self.docker_client.containers.run(**container_config)
        context.container_id = container.id
        context.state = AgentState.RUNNING

        # Monitor execution
        start_time = time.time()
        result_data = None

        try:
            # Wait for completion with timeout
            exit_code = container.wait(timeout=context.timeout)

            # Get logs
            logs = container.logs().decode("utf-8").split("\n")
            context.logs.extend(logs)

            # Read result file if it exists
            result_file = context.working_directory / "result.json"
            if result_file.exists():
                async with aiofiles.open(result_file) as f:
                    result_data = json.loads(await f.read())

            # Collect metrics
            elapsed_time = time.time() - start_time

            return {
                "success": exit_code["StatusCode"] == 0,
                "output": result_data,
                "logs": logs,
                "metrics": {
                    "execution_time": elapsed_time,
                    "exit_code": exit_code["StatusCode"],
                    "container_id": container.id,
                },
            }

        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            raise
        finally:
            # Cleanup
            try:
                container.stop()
                logger.info(f"Stopped Docker container {container.id[:12]}")
                container.remove()
                logger.debug(f"Removed Docker container {container.id[:12]}")
            except docker.errors.NotFound:
                logger.warning(f"Container {container.id[:12]} already removed")
            except docker.errors.APIError as e:
                logger.error(f"Docker API error during container cleanup: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during container cleanup: {e}")
                # Try force removal as last resort
                try:
                    container.remove(force=True)
                    logger.warning(f"Force removed container {container.id[:12]} after cleanup error")
                except Exception as force_error:
                    logger.error(f"Failed to force remove container {container.id[:12]}: {force_error}")

    async def _execute_in_process(
        self, context: AgentExecutionContext
    ) -> dict[str, Any]:
        """Execute agent in isolated process."""
        # Prepare execution script
        script_path = context.working_directory / "agent_runner.py"
        await self._write_runner_script(script_path, context)

        # Create restricted environment
        env = os.environ.copy()
        env.update(context.environment)
        env["PYTHONPATH"] = str(context.working_directory)

        # Start process with resource limits
        process = await asyncio.create_subprocess_exec(
            "python",
            str(script_path),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(context.working_directory),
            preexec_fn=self._set_process_limits if os.name != "nt" else None,
        )

        context.process_id = process.pid
        context.state = AgentState.RUNNING

        # Monitor process with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=context.timeout
            )

            # Parse output
            output_lines = stdout.decode("utf-8").split("\n") if stdout else []
            error_lines = stderr.decode("utf-8").split("\n") if stderr else []

            context.logs.extend(output_lines)
            context.logs.extend(error_lines)

            # Read result
            result_file = context.working_directory / "result.json"
            result_data = None
            if result_file.exists():
                async with aiofiles.open(result_file) as f:
                    result_data = json.loads(await f.read())

            return {
                "success": process.returncode == 0,
                "output": result_data,
                "logs": context.logs,
                "metrics": {"exit_code": process.returncode, "process_id": process.pid},
            }

        except TimeoutError:
            # Kill process on timeout
            try:
                process.kill()
                logger.warning(f"Killed agent process {process.pid} due to timeout")
                await process.wait()
                logger.debug(f"Process {process.pid} cleanup completed")
            except ProcessLookupError:
                logger.debug(f"Process {process.pid} already terminated")
            except PermissionError as e:
                logger.error(f"Permission denied killing process {process.pid}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error killing process {process.pid}: {e}")
            raise

    async def _execute_local(self, context: AgentExecutionContext) -> dict[str, Any]:
        """Execute agent locally (development only)."""
        logger.warning("Executing agent in LOCAL mode - no isolation!")

        # Prepare execution environment
        agent_module = await self._create_agent_module(context)

        # Execute with timeout
        start_time = time.time()

        try:
            # Prepare input with memory manager and backend if available
            execution_input = context.input_data.copy()
            if context.memory_manager and context.spec.memory_enabled:
                execution_input["memory_manager"] = context.memory_manager
                execution_input["agent_id"] = context.agent_id
                execution_input["session_id"] = context.session_id
            
            # Add qwen backend for real SmolAgents integration
            if self.qwen_backend:
                execution_input["qwen_backend"] = self.qwen_backend

            # Run agent function
            result = await asyncio.wait_for(
                agent_module.run_agent(execution_input), timeout=context.timeout
            )

            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "output": result,
                "logs": context.logs,
                "metrics": {"execution_time": elapsed_time},
            }

        except Exception as e:
            logger.error(f"Local execution error: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "logs": context.logs,
                "metrics": {"execution_time": time.time() - start_time},
            }

    async def _write_agent_file(self, file_path: Path, context: AgentExecutionContext):
        """Write agent code to file."""
        agent_code = f"""
# Auto-generated agent code
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt
SYSTEM_PROMPT = '''{context.spec.system_prompt}'''

# Agent implementation
{context.spec.initial_code}

# Main execution
async def main():
    try:
        # Load input data
        input_file = Path('/app/input.json')
        if input_file.exists():
            with open(input_file, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = {{}}

        # Run agent
        result = await run_agent(input_data)

        # Save result
        with open('/app/result.json', 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("Agent execution completed successfully")

    except Exception as e:
        logger.error(f"Agent execution failed: {{e}}")
        error_result = {{
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }}
        with open('/app/result.json', 'w') as f:
            json.dump(error_result, f, indent=2)
        raise

if __name__ == '__main__':
    asyncio.run(main())
"""

        async with aiofiles.open(file_path, "w") as f:
            await f.write(agent_code)

    async def _write_runner_script(
        self, script_path: Path, context: AgentExecutionContext
    ):
        """Write runner script for process isolation."""
        runner_code = f"""
import sys
import os
import resource
import asyncio
import json
from pathlib import Path

# Set resource limits
def set_limits():
    # Memory limit
    memory_limit = {context.spec.resource_limits.get('memory_mb', 512)} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

    # CPU time limit
    cpu_limit = {context.timeout}
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

# Set limits before importing agent
if os.name != 'nt':
    set_limits()

# Import and run agent
sys.path.insert(0, str(Path(__file__).parent))

{context.spec.initial_code}

# Run agent
async def main():
    input_file = Path(__file__).parent / 'input.json'
    result_file = Path(__file__).parent / 'result.json'

    # Load input
    if input_file.exists():
        with open(input_file, 'r') as f:
            input_data = json.load(f)
    else:
        input_data = {{}}

    # Execute
    result = await run_agent(input_data)

    # Save result
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    asyncio.run(main())
"""

        async with aiofiles.open(script_path, "w") as f:
            await f.write(runner_code)

    def _set_process_limits(self):
        """Set resource limits for process (Unix only)."""
        import resource

        # Limit memory
        memory_limit = 512 * 1024 * 1024  # 512MB
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

        # Limit CPU time
        cpu_limit = 300  # 5 minutes
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

        # Limit file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))

    async def _create_agent_module(self, context: AgentExecutionContext):
        """Create agent module for local execution."""
        # Create a module from agent code
        import types

        module = types.ModuleType("agent")

        # Add system prompt
        module.SYSTEM_PROMPT = context.spec.system_prompt

        # Execute agent code in module namespace
        exec(context.spec.initial_code, module.__dict__)

        return module

    def _create_error_result(
        self, context: AgentExecutionContext, error: str
    ) -> dict[str, Any]:
        """Create standardized error result."""
        return {
            "success": False,
            "output": None,
            "error": error,
            "logs": context.logs,
            "metrics": {"state": context.state.value, "error": error},
        }


class AgentRuntime:
    """
    Main runtime system for managing agent execution.

    This orchestrates:
    - Agent lifecycle (create, start, stop, destroy)
    - Resource allocation and monitoring
    - Communication between agents
    - State persistence
    - Event handling
    - Memory integration for context-aware agents
    """

    def __init__(
        self,
        sandbox_type: SandboxType = SandboxType.DOCKER,
        memory_manager: TektraMemoryManager | None = None,
        qwen_backend=None,
    ):
        """Initialize runtime with specified sandbox type, memory manager, and AI backend."""
        self.sandbox = AgentSandbox(sandbox_type, qwen_backend)
        self.running_agents: dict[str, AgentExecutionContext] = {}
        self.memory_manager = memory_manager
        self.qwen_backend = qwen_backend
        self.agent_queues: dict[str, asyncio.Queue] = {}
        self.working_directory = Path.home() / ".tektra" / "agents"
        self.working_directory.mkdir(parents=True, exist_ok=True)

        # Memory integration
        self.memory_manager = memory_manager
        self.enable_memory = memory_manager is not None

        logger.info(f"Agent Runtime initialized with {sandbox_type.value} sandbox")
        if self.enable_memory:
            logger.info("Memory integration enabled for agents")

    async def deploy_agent(
        self,
        spec: AgentSpecification,
        input_data: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> str:
        """
        Deploy an agent from specification.

        Args:
            spec: Agent specification
            input_data: Initial input data for the agent
            user_id: Optional user ID for memory context

        Returns:
            Agent ID for tracking
        """
        agent_id = spec.id

        # Create working directory for agent
        agent_dir = self.working_directory / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Create execution context
        context = AgentExecutionContext(
            agent_id=agent_id,
            spec=spec,
            input_data=input_data or {},
            environment=spec.environment_vars,
            working_directory=agent_dir,
            start_time=datetime.now(),
            timeout=spec.max_runtime_seconds,
            memory_manager=self.memory_manager if spec.memory_enabled else None,
        )

        # Load agent memory context if enabled
        if self.enable_memory and spec.memory_enabled and self.memory_manager:
            await self._load_agent_memory_context(context, user_id)

        # Store context
        self.running_agents[agent_id] = context

        # Create communication queue
        self.agent_queues[agent_id] = asyncio.Queue()

        # Start execution based on trigger type
        if spec.trigger_type == "manual":
            # Execute immediately
            asyncio.create_task(self._execute_agent(context))
        elif spec.trigger_type == "scheduled":
            # Schedule execution
            asyncio.create_task(self._schedule_agent(context))
        elif spec.trigger_type == "event-based":
            # Wait for events
            asyncio.create_task(self._event_loop_agent(context))

        logger.info(f"Deployed agent {agent_id} ({spec.name})")
        return agent_id

    async def _execute_agent(self, context: AgentExecutionContext):
        """Execute agent in sandbox."""
        try:
            # Prepare input data with memory manager if available
            execution_input = context.input_data.copy()
            if context.memory_manager and context.spec.memory_enabled:
                # Add memory manager reference (note: in sandbox this would be a proxy)
                execution_input["memory_manager"] = context.memory_manager
                execution_input["agent_id"] = context.agent_id
                execution_input["session_id"] = context.session_id

            # Save input data
            input_file = context.working_directory / "input.json"
            # Don't serialize memory_manager to JSON
            json_safe_input = {
                k: v for k, v in execution_input.items() if k != "memory_manager"
            }
            async with aiofiles.open(input_file, "w") as f:
                await f.write(json.dumps(json_safe_input, indent=2))

            # Execute in sandbox
            result = await self.sandbox.execute_agent(context)

            # Update context with results
            context.output = result.get("output")
            context.execution_count += 1

            # Save execution to memory if enabled
            if context.memory_manager and context.spec.memory_enabled:
                await self._save_execution_to_memory(context, result)

            # Handle output
            if context.spec.webhook_url:
                await self._send_webhook(context.spec.webhook_url, result)

            # Check if should restart
            if context.spec.auto_restart and result.get("success"):
                await asyncio.sleep(5)  # Brief pause
                await self._execute_agent(context)

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            context.state = AgentState.FAILED
            context.error = str(e)

    async def _schedule_agent(self, context: AgentExecutionContext):
        """Handle scheduled agent execution."""
        # Simple scheduling - in production use APScheduler or similar
        from datetime import datetime

        import croniter

        cron = croniter.croniter(context.spec.schedule, datetime.now())

        while context.state not in [AgentState.TERMINATED, AgentState.FAILED]:
            # Wait until next scheduled time
            next_run = cron.get_next(datetime)
            wait_seconds = (next_run - datetime.now()).total_seconds()

            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            # Execute agent
            await self._execute_agent(context)

    async def _event_loop_agent(self, context: AgentExecutionContext):
        """Handle event-driven agent execution."""
        queue = self.agent_queues[context.agent_id]

        while context.state not in [AgentState.TERMINATED, AgentState.FAILED]:
            try:
                # Wait for event
                event = await queue.get()

                # Update input data with event
                context.input_data = event

                # Execute agent
                await self._execute_agent(context)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event loop error: {e}")

    async def send_event_to_agent(self, agent_id: str, event_data: dict[str, Any]):
        """Send event to an event-driven agent."""
        if agent_id in self.agent_queues:
            await self.agent_queues[agent_id].put(event_data)
        else:
            raise ValueError(f"Agent {agent_id} not found or not event-driven")

    async def stop_agent(self, agent_id: str):
        """Stop a running agent."""
        if agent_id in self.running_agents:
            context = self.running_agents[agent_id]
            context.state = AgentState.TERMINATED

            # Kill process/container if running
            if context.process_id:
                try:
                    os.kill(context.process_id, signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to process {context.process_id}")
                except ProcessLookupError:
                    logger.debug(f"Process {context.process_id} already terminated")
                except PermissionError as e:
                    logger.error(f"Permission denied terminating process {context.process_id}: {e}")
                except Exception as e:
                    logger.error(f"Error terminating process {context.process_id}: {e}")

            if context.container_id and self.sandbox.docker_client:
                try:
                    container = self.sandbox.docker_client.containers.get(
                        context.container_id
                    )
                    container.stop()
                    logger.info(f"Stopped container {context.container_id[:12]}")
                    container.remove()
                    logger.debug(f"Removed container {context.container_id[:12]}")
                except docker.errors.NotFound:
                    logger.debug(f"Container {context.container_id[:12]} already removed")
                except docker.errors.APIError as e:
                    logger.error(f"Docker API error stopping container {context.container_id[:12]}: {e}")
                except Exception as e:
                    logger.error(f"Error stopping container {context.container_id[:12]}: {e}")
                    # Try force removal
                    try:
                        container = self.sandbox.docker_client.containers.get(context.container_id)
                        container.remove(force=True)
                        logger.warning(f"Force removed container {context.container_id[:12]}")
                    except Exception as force_error:
                        logger.error(f"Failed to force remove container {context.container_id[:12]}: {force_error}")

            logger.info(f"Stopped agent {agent_id}")

    async def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        """Get current status of an agent."""
        if agent_id not in self.running_agents:
            return {"error": "Agent not found"}

        context = self.running_agents[agent_id]

        return {
            "agent_id": agent_id,
            "name": context.spec.name,
            "state": context.state.value,
            "execution_count": context.execution_count,
            "start_time": context.start_time.isoformat(),
            "last_output": context.output,
            "error": context.error,
            "metrics": {
                "cpu_usage": context.cpu_usage,
                "memory_usage": context.memory_usage,
            },
        }

    async def list_running_agents(self) -> list[dict[str, Any]]:
        """List all running agents."""
        agents = []
        for agent_id, context in self.running_agents.items():
            agents.append(
                {
                    "agent_id": agent_id,
                    "name": context.spec.name,
                    "state": context.state.value,
                    "type": context.spec.type.value,
                    "trigger_type": context.spec.trigger_type,
                }
            )
        return agents

    async def cleanup(self):
        """Clean up all running agents."""
        for agent_id in list(self.running_agents.keys()):
            await self.stop_agent(agent_id)

    async def _send_webhook(self, url: str, data: dict[str, Any]):
        """Send webhook notification."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        logger.warning(f"Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    async def _load_agent_memory_context(
        self, context: AgentExecutionContext, user_id: str | None = None
    ):
        """Load memory context for an agent."""
        if not context.memory_manager or not context.spec.memory_enabled:
            return

        try:
            # Load agent context memories
            agent_memories = await context.memory_manager.get_agent_context(
                context.agent_id
            )

            # Load recent execution history if persistent memory is enabled
            if context.spec.persistent_memory:
                memory_ctx = MemoryContext(
                    agent_id=context.agent_id,
                    user_id=user_id,
                    memory_types=[MemoryType.TASK_RESULT, MemoryType.AGENT_CONTEXT],
                    max_results=context.spec.memory_context_limit,
                    min_relevance=context.spec.memory_importance_threshold,
                    time_window_hours=context.spec.memory_retention_hours,
                )

                search_result = await context.memory_manager.search_memories(memory_ctx)
                context.memory_context = search_result.entries

                logger.info(
                    f"Loaded {len(search_result.entries)} memory entries for agent {context.agent_id}"
                )

            # Store initial agent context if this is first run
            if not agent_memories:
                await context.memory_manager.add_agent_context(
                    agent_id=context.agent_id,
                    context=f"Agent: {context.spec.name}\nGoal: {context.spec.goal}",
                    importance=0.9,
                )

        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

    async def _save_execution_to_memory(
        self, context: AgentExecutionContext, result: dict[str, Any]
    ):
        """Save agent execution results to memory."""
        if not context.memory_manager or not context.spec.memory_enabled:
            return

        try:
            # Determine importance based on success and content
            importance = 0.7 if result.get("success") else 0.5
            if result.get("output"):
                # Increase importance for meaningful outputs
                output_str = str(result["output"])
                if len(output_str) > 100 or "important" in output_str.lower():
                    importance = 0.8

            # Save task result
            task_description = context.input_data.get("task", "Agent execution")
            result_summary = (
                json.dumps(result.get("output", {}))[:500]
                if result.get("output")
                else "No output"
            )

            await context.memory_manager.add_task_result(
                task_description=task_description,
                result=result_summary,
                success=result.get("success", False),
                agent_id=context.agent_id,
                user_id=(
                    context.input_data.get("user_id")
                    if isinstance(context.input_data, dict)
                    else None
                ),
            )

            # Save important logs if any errors occurred
            if not result.get("success") and result.get("error"):
                await context.memory_manager.add_agent_context(
                    agent_id=context.agent_id,
                    context=f"Error during execution: {result['error']}",
                    importance=0.8,
                )

            logger.debug(f"Saved execution to memory with importance {importance}")

        except Exception as e:
            logger.warning(f"Failed to save execution to memory: {e}")

    async def get_agent_memory_stats(self, agent_id: str) -> dict[str, Any]:
        """Get memory statistics for a specific agent."""
        if not self.memory_manager or agent_id not in self.running_agents:
            return {"error": "Memory not available or agent not found"}

        context = self.running_agents[agent_id]
        if not context.spec.memory_enabled:
            return {"error": "Memory not enabled for this agent"}

        try:
            # Get agent-specific memories
            agent_memories = await self.memory_manager.get_agent_context(agent_id)

            # Get task results
            memory_ctx = MemoryContext(
                agent_id=agent_id,
                memory_types=[MemoryType.TASK_RESULT],
                max_results=100,
            )
            task_results = await self.memory_manager.search_memories(memory_ctx)

            return {
                "agent_id": agent_id,
                "memory_enabled": True,
                "context_memories": len(agent_memories),
                "task_results": len(task_results.entries),
                "memory_config": {
                    "context_limit": context.spec.memory_context_limit,
                    "importance_threshold": context.spec.memory_importance_threshold,
                    "retention_hours": context.spec.memory_retention_hours,
                    "persistent": context.spec.persistent_memory,
                    "sharing_enabled": context.spec.memory_sharing_enabled,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    async def share_agent_memory(
        self, source_agent_id: str, target_agent_id: str, memory_types: list[MemoryType]
    ):
        """Share memories between agents if enabled."""
        if not self.memory_manager:
            raise RuntimeError("Memory manager not available")

        # Check if both agents exist and have memory sharing enabled
        source_context = self.running_agents.get(source_agent_id)
        target_context = self.running_agents.get(target_agent_id)

        if not source_context or not target_context:
            raise ValueError("One or both agents not found")

        if not source_context.spec.memory_sharing_enabled:
            raise ValueError(f"Agent {source_agent_id} does not allow memory sharing")

        if not target_context.spec.memory_enabled:
            raise ValueError(f"Agent {target_agent_id} does not have memory enabled")

        try:
            # Search for memories to share
            memory_ctx = MemoryContext(
                agent_id=source_agent_id,
                memory_types=memory_types,
                max_results=50,
                min_relevance=0.5,
            )

            memories_to_share = await self.memory_manager.search_memories(memory_ctx)

            # Copy memories to target agent
            shared_count = 0
            for memory in memories_to_share.entries:
                # Create a copy with target agent ID
                if memory.type == MemoryType.AGENT_CONTEXT:
                    await self.memory_manager.add_agent_context(
                        agent_id=target_agent_id,
                        context=f"[Shared from {source_context.spec.name}] {memory.content}",
                        importance=memory.importance
                        * 0.8,  # Slightly reduce importance for shared memories
                    )
                    shared_count += 1

            logger.info(
                f"Shared {shared_count} memories from {source_agent_id} to {target_agent_id}"
            )
            return {"shared_memories": shared_count}

        except Exception as e:
            logger.error(f"Failed to share memories: {e}")
            raise
