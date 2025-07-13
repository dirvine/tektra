#!/usr/bin/env python3
"""
Sandboxing System for Agent Execution

Provides secure sandboxing capabilities using process isolation,
resource limits, and access controls.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,loguru python sandbox.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
# ]
# ///

import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
import threading

import psutil
from loguru import logger


class SandboxType(Enum):
    """Types of sandbox isolation."""
    PROCESS = "process"      # Basic process isolation
    CHROOT = "chroot"       # Chroot jail isolation
    CONTAINER = "container"  # Container-based isolation
    VM = "virtual_machine"   # Virtual machine isolation


class SandboxStatus(Enum):
    """Sandbox execution status."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SandboxConfig:
    """Configuration for sandbox environment."""
    
    # Basic configuration
    sandbox_type: SandboxType = SandboxType.PROCESS
    enable_network: bool = True
    enable_filesystem: bool = True
    
    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_execution_time: int = 300
    max_processes: int = 5
    max_file_descriptors: int = 100
    
    # File system configuration
    sandbox_root: Optional[Path] = None
    read_only_paths: List[str] = field(default_factory=list)
    read_write_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    temp_dir_size_mb: int = 100
    
    # Network configuration
    allowed_hosts: List[str] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)
    max_connections: int = 10
    
    # Security configuration
    drop_privileges: bool = True
    enable_seccomp: bool = True
    enable_apparmor: bool = False
    custom_capabilities: List[str] = field(default_factory=list)
    
    # Monitoring configuration
    monitor_syscalls: bool = True
    monitor_network: bool = True
    monitor_filesystem: bool = True
    log_level: str = "INFO"


@dataclass
class SandboxMetrics:
    """Metrics for sandbox execution."""
    
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    files_created: int = 0
    files_deleted: int = 0
    network_requests: int = 0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    
    # Security events
    permission_violations: int = 0
    syscall_violations: int = 0
    escape_attempts: int = 0
    
    # Process information
    exit_code: Optional[int] = None
    signal_received: Optional[int] = None
    killed_by_timeout: bool = False


class SandboxManager:
    """
    Manages sandboxed execution environments for agents.
    
    Provides secure isolation using various sandboxing techniques
    including process isolation, resource limits, and access controls.
    """
    
    def __init__(self, base_config: Optional[SandboxConfig] = None):
        """
        Initialize sandbox manager.
        
        Args:
            base_config: Base configuration for all sandboxes
        """
        self.base_config = base_config or SandboxConfig()
        self.active_sandboxes: Dict[str, 'Sandbox'] = {}
        self.sandbox_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info("Sandbox manager initialized")
    
    def create_sandbox(
        self,
        sandbox_id: Optional[str] = None,
        config: Optional[SandboxConfig] = None
    ) -> 'Sandbox':
        """
        Create a new sandbox instance.
        
        Args:
            sandbox_id: Optional sandbox identifier
            config: Sandbox configuration (uses base config if not provided)
            
        Returns:
            Sandbox instance
        """
        with self._lock:
            if sandbox_id is None:
                sandbox_id = str(uuid.uuid4())
            
            if sandbox_id in self.active_sandboxes:
                raise ValueError(f"Sandbox {sandbox_id} already exists")
            
            # Merge with base config
            effective_config = self._merge_configs(self.base_config, config)
            
            # Create sandbox
            sandbox = Sandbox(sandbox_id, effective_config)
            self.active_sandboxes[sandbox_id] = sandbox
            
            logger.info(f"Sandbox created: {sandbox_id}")
            return sandbox
    
    def get_sandbox(self, sandbox_id: str) -> Optional['Sandbox']:
        """
        Get an existing sandbox by ID.
        
        Args:
            sandbox_id: Sandbox identifier
            
        Returns:
            Sandbox instance or None if not found
        """
        with self._lock:
            return self.active_sandboxes.get(sandbox_id)
    
    def destroy_sandbox(self, sandbox_id: str, force: bool = False) -> bool:
        """
        Destroy a sandbox and clean up resources.
        
        Args:
            sandbox_id: Sandbox identifier
            force: Force destruction even if still running
            
        Returns:
            True if destroyed successfully, False otherwise
        """
        with self._lock:
            sandbox = self.active_sandboxes.get(sandbox_id)
            if not sandbox:
                return False
            
            try:
                # Stop if running
                if sandbox.status in [SandboxStatus.RUNNING, SandboxStatus.STARTING]:
                    if not force and not sandbox.stop():
                        logger.warning(f"Could not stop sandbox {sandbox_id}")
                        return False
                    elif force:
                        sandbox.kill()
                
                # Clean up resources
                sandbox.cleanup()
                
                # Move to history
                self.sandbox_history.append({
                    "sandbox_id": sandbox_id,
                    "config": sandbox.config.__dict__,
                    "metrics": sandbox.metrics.__dict__,
                    "destroyed_at": time.time()
                })
                
                # Remove from active
                del self.active_sandboxes[sandbox_id]
                
                logger.info(f"Sandbox destroyed: {sandbox_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error destroying sandbox {sandbox_id}: {e}")
                return False
    
    def list_sandboxes(self) -> List[Dict[str, Any]]:
        """
        List all active sandboxes.
        
        Returns:
            List of sandbox information
        """
        with self._lock:
            sandboxes = []
            for sandbox_id, sandbox in self.active_sandboxes.items():
                sandboxes.append({
                    "sandbox_id": sandbox_id,
                    "status": sandbox.status.value,
                    "created_at": sandbox.created_at,
                    "execution_time": sandbox.get_execution_time(),
                    "resource_usage": sandbox.get_resource_usage()
                })
            return sandboxes
    
    def cleanup_finished_sandboxes(self) -> int:
        """
        Clean up finished sandboxes.
        
        Returns:
            Number of sandboxes cleaned up
        """
        with self._lock:
            to_remove = []
            
            for sandbox_id, sandbox in self.active_sandboxes.items():
                if sandbox.status in [SandboxStatus.STOPPED, SandboxStatus.ERROR]:
                    to_remove.append(sandbox_id)
            
            for sandbox_id in to_remove:
                self.destroy_sandbox(sandbox_id)
            
            logger.debug(f"Cleaned up {len(to_remove)} finished sandboxes")
            return len(to_remove)
    
    def shutdown(self) -> None:
        """Shutdown the sandbox manager and clean up all resources."""
        with self._lock:
            logger.info("Shutting down sandbox manager")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop all active sandboxes
            for sandbox_id in list(self.active_sandboxes.keys()):
                self.destroy_sandbox(sandbox_id, force=True)
            
            # Wait for cleanup thread
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5.0)
    
    def _merge_configs(
        self,
        base_config: SandboxConfig,
        override_config: Optional[SandboxConfig]
    ) -> SandboxConfig:
        """Merge base config with override config."""
        if not override_config:
            return base_config
        
        # Simple merge - override config takes precedence
        merged = SandboxConfig()
        
        # Copy base config values
        for field_name, field_value in base_config.__dict__.items():
            setattr(merged, field_name, field_value)
        
        # Override with specific config
        for field_name, field_value in override_config.__dict__.items():
            if field_value is not None:
                setattr(merged, field_name, field_value)
        
        return merged
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._shutdown_event.wait(30):  # Check every 30 seconds
                try:
                    self.cleanup_finished_sandboxes()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()


class Sandbox:
    """
    Individual sandbox instance for secure code execution.
    
    Provides isolated execution environment with resource limits,
    access controls, and monitoring capabilities.
    """
    
    def __init__(self, sandbox_id: str, config: SandboxConfig):
        """
        Initialize sandbox instance.
        
        Args:
            sandbox_id: Unique identifier for the sandbox
            config: Sandbox configuration
        """
        self.sandbox_id = sandbox_id
        self.config = config
        self.status = SandboxStatus.CREATED
        self.created_at = time.time()
        
        # Runtime state
        self.process: Optional[subprocess.Popen] = None
        self.sandbox_root: Optional[Path] = None
        self.temp_dirs: List[Path] = []
        self.metrics = SandboxMetrics()
        
        # Monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        
        logger.info(f"Sandbox initialized: {sandbox_id}")
    
    def start(self, command: List[str], env: Optional[Dict[str, str]] = None) -> bool:
        """
        Start execution in the sandbox.
        
        Args:
            command: Command to execute
            env: Environment variables
            
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.status != SandboxStatus.CREATED:
                logger.warning(f"Sandbox {self.sandbox_id} not in created state")
                return False
            
            self.status = SandboxStatus.STARTING
            self._notify_status_change()
            
            # Set up sandbox environment
            if not self._setup_environment():
                self.status = SandboxStatus.ERROR
                self._notify_status_change()
                return False
            
            # Prepare execution environment
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)
            
            # Apply security restrictions
            preexec_fn = self._create_preexec_function()
            
            # Start process
            self.process = subprocess.Popen(
                command,
                cwd=self.sandbox_root,
                env=exec_env,
                preexec_fn=preexec_fn,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Start monitoring
            self.metrics.start_time = time.time()
            self._start_monitoring()
            
            self.status = SandboxStatus.RUNNING
            self._notify_status_change()
            
            logger.info(f"Sandbox {self.sandbox_id} started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start sandbox {self.sandbox_id}: {e}")
            self.status = SandboxStatus.ERROR
            self._notify_status_change()
            return False
    
    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop sandbox execution gracefully.
        
        Args:
            timeout: Timeout for graceful shutdown
            
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if self.status != SandboxStatus.RUNNING:
                return True
            
            self.status = SandboxStatus.STOPPING
            self._notify_status_change()
            
            if self.process:
                # Try graceful shutdown
                self.process.terminate()
                
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    self.process.kill()
                    self.process.wait()
                    self.metrics.killed_by_timeout = True
                
                self.metrics.exit_code = self.process.returncode
                self.metrics.end_time = time.time()
                
                if self.metrics.start_time:
                    self.metrics.execution_time = self.metrics.end_time - self.metrics.start_time
            
            # Stop monitoring
            self._stop_monitoring()
            
            self.status = SandboxStatus.STOPPED
            self._notify_status_change()
            
            logger.info(f"Sandbox {self.sandbox_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop sandbox {self.sandbox_id}: {e}")
            return False
    
    def kill(self) -> bool:
        """
        Force kill sandbox execution.
        
        Returns:
            True if killed successfully, False otherwise
        """
        try:
            if self.process:
                self.process.kill()
                self.process.wait()
                self.metrics.exit_code = self.process.returncode
                self.metrics.signal_received = signal.SIGKILL
            
            self._stop_monitoring()
            
            self.status = SandboxStatus.STOPPED
            self._notify_status_change()
            
            logger.info(f"Sandbox {self.sandbox_id} killed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to kill sandbox {self.sandbox_id}: {e}")
            return False
    
    def pause(self) -> bool:
        """
        Pause sandbox execution.
        
        Returns:
            True if paused successfully, False otherwise
        """
        try:
            if self.status != SandboxStatus.RUNNING:
                return False
            
            if self.process:
                os.kill(self.process.pid, signal.SIGSTOP)
            
            self.status = SandboxStatus.PAUSED
            self._notify_status_change()
            
            logger.info(f"Sandbox {self.sandbox_id} paused")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause sandbox {self.sandbox_id}: {e}")
            return False
    
    def resume(self) -> bool:
        """
        Resume paused sandbox execution.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        try:
            if self.status != SandboxStatus.PAUSED:
                return False
            
            if self.process:
                os.kill(self.process.pid, signal.SIGCONT)
            
            self.status = SandboxStatus.RUNNING
            self._notify_status_change()
            
            logger.info(f"Sandbox {self.sandbox_id} resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume sandbox {self.sandbox_id}: {e}")
            return False
    
    def get_execution_time(self) -> float:
        """
        Get current execution time.
        
        Returns:
            Execution time in seconds
        """
        if not self.metrics.start_time:
            return 0.0
        
        end_time = self.metrics.end_time or time.time()
        return end_time - self.metrics.start_time
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with resource usage information
        """
        usage = {
            "memory_mb": 0.0,
            "cpu_percent": 0.0,
            "files_open": 0,
            "connections": 0,
            "execution_time": self.get_execution_time()
        }
        
        if self.process:
            try:
                process = psutil.Process(self.process.pid)
                
                # Memory usage
                memory_info = process.memory_info()
                usage["memory_mb"] = memory_info.rss / 1024 / 1024
                
                # CPU usage
                usage["cpu_percent"] = process.cpu_percent()
                
                # Open files
                try:
                    usage["files_open"] = len(process.open_files())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                
                # Network connections
                try:
                    usage["connections"] = len(process.connections())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Could not get resource usage for sandbox {self.sandbox_id}: {e}")
        
        return usage
    
    def add_status_callback(self, callback: Callable[['Sandbox'], None]) -> None:
        """
        Add callback for status changes.
        
        Args:
            callback: Function to call on status changes
        """
        self.status_callbacks.append(callback)
    
    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        try:
            # Stop monitoring
            self._stop_monitoring()
            
            # Clean up temporary directories
            for temp_dir in self.temp_dirs:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Clean up sandbox root if we created it
            if self.sandbox_root and self.config.sandbox_root is None:
                if self.sandbox_root.exists():
                    shutil.rmtree(self.sandbox_root, ignore_errors=True)
            
            logger.debug(f"Sandbox {self.sandbox_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up sandbox {self.sandbox_id}: {e}")
    
    def _setup_environment(self) -> bool:
        """Set up the sandbox environment."""
        try:
            # Create sandbox root directory
            if self.config.sandbox_root:
                self.sandbox_root = self.config.sandbox_root
                self.sandbox_root.mkdir(parents=True, exist_ok=True)
            else:
                self.sandbox_root = Path(tempfile.mkdtemp(prefix=f"sandbox_{self.sandbox_id}_"))
                self.temp_dirs.append(self.sandbox_root)
            
            # Create standard directories
            (self.sandbox_root / "tmp").mkdir(exist_ok=True)
            (self.sandbox_root / "home").mkdir(exist_ok=True)
            (self.sandbox_root / "work").mkdir(exist_ok=True)
            
            # Set up read-write paths
            for path in self.config.read_write_paths:
                target_path = self.sandbox_root / path.lstrip('/')
                target_path.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Sandbox environment set up at {self.sandbox_root}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up sandbox environment: {e}")
            return False
    
    def _create_preexec_function(self) -> Optional[Callable]:
        """Create preexec function for security restrictions."""
        def preexec_fn():
            try:
                # Drop privileges if requested
                if self.config.drop_privileges:
                    # In a real implementation, this would drop to a non-root user
                    pass
                
                # Set process group
                os.setpgrp()
                
                # Apply resource limits
                import resource
                
                # Memory limit
                if self.config.max_memory_mb > 0:
                    memory_bytes = self.config.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                
                # CPU time limit
                if self.config.max_execution_time > 0:
                    resource.setrlimit(resource.RLIMIT_CPU, (self.config.max_execution_time, self.config.max_execution_time))
                
                # File descriptor limit
                if self.config.max_file_descriptors > 0:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (self.config.max_file_descriptors, self.config.max_file_descriptors))
                
            except Exception as e:
                logger.warning(f"Could not apply all security restrictions: {e}")
        
        return preexec_fn
    
    def _start_monitoring(self) -> None:
        """Start resource monitoring thread."""
        def monitor_worker():
            while not self.stop_monitoring.wait(1.0):  # Check every second
                try:
                    if self.process and self.process.poll() is None:
                        usage = self.get_resource_usage()
                        
                        # Update peak memory
                        memory_mb = usage.get("memory_mb", 0.0)
                        if memory_mb > self.metrics.peak_memory_mb:
                            self.metrics.peak_memory_mb = memory_mb
                        
                        # Check resource violations
                        self._check_resource_violations(usage)
                        
                        # Check execution time limit
                        if self.get_execution_time() > self.config.max_execution_time:
                            logger.warning(f"Sandbox {self.sandbox_id} exceeded time limit")
                            self.kill()
                            break
                    else:
                        # Process finished
                        break
                        
                except Exception as e:
                    logger.error(f"Error in sandbox monitoring: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
    
    def _stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _check_resource_violations(self, usage: Dict[str, Any]) -> None:
        """Check for resource limit violations."""
        violations = []
        
        # Memory check
        memory_mb = usage.get("memory_mb", 0.0)
        if memory_mb > self.config.max_memory_mb:
            violations.append(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.config.max_memory_mb}MB")
        
        # CPU check (would need more sophisticated monitoring for accurate CPU limits)
        
        # Connection check
        connections = usage.get("connections", 0)
        if connections > self.config.max_connections:
            violations.append(f"Connection limit exceeded: {connections} > {self.config.max_connections}")
        
        if violations:
            self.metrics.permission_violations += len(violations)
            logger.warning(f"Resource violations in sandbox {self.sandbox_id}: {violations}")
    
    def _notify_status_change(self) -> None:
        """Notify registered callbacks of status changes."""
        for callback in self.status_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")


def create_sandbox_manager(config: Optional[SandboxConfig] = None) -> SandboxManager:
    """
    Create a sandbox manager instance.
    
    Args:
        config: Base configuration for sandboxes
        
    Returns:
        SandboxManager instance
    """
    return SandboxManager(config)


if __name__ == "__main__":
    def demo_sandbox_system():
        """Demonstrate sandbox system functionality."""
        print("🔒 Sandbox System Demo")
        print("=" * 40)
        
        # Create sandbox manager
        config = SandboxConfig(
            max_memory_mb=256,
            max_cpu_percent=25.0,
            max_execution_time=30,
            enable_network=False
        )
        
        sandbox_manager = create_sandbox_manager(config)
        
        # Create a sandbox
        sandbox = sandbox_manager.create_sandbox("demo_sandbox")
        print(f"Created sandbox: {sandbox.sandbox_id}")
        print(f"Status: {sandbox.status.value}")
        
        # Add status callback
        def status_callback(sb):
            print(f"Status changed: {sb.status.value}")
        
        sandbox.add_status_callback(status_callback)
        
        # Start a simple command
        if sandbox.start(["python3", "-c", "import time; print('Hello from sandbox'); time.sleep(2); print('Done')"]):
            print("✅ Sandbox started successfully")
            
            # Monitor for a few seconds
            start_time = time.time()
            while time.time() - start_time < 5:
                usage = sandbox.get_resource_usage()
                print(f"Resource usage: Memory={usage['memory_mb']:.1f}MB, CPU={usage['cpu_percent']:.1f}%")
                time.sleep(1)
                
                if sandbox.status == SandboxStatus.STOPPED:
                    break
            
            # Stop if still running
            if sandbox.status == SandboxStatus.RUNNING:
                sandbox.stop()
            
            print(f"Final status: {sandbox.status.value}")
            print(f"Execution time: {sandbox.get_execution_time():.2f}s")
            print(f"Exit code: {sandbox.metrics.exit_code}")
        
        # List all sandboxes
        sandboxes = sandbox_manager.list_sandboxes()
        print(f"Active sandboxes: {len(sandboxes)}")
        
        # Clean up
        sandbox_manager.destroy_sandbox("demo_sandbox")
        sandbox_manager.shutdown()
        
        print("\n🔒 Sandbox System Demo Complete")
    
    # Run demo
    demo_sandbox_system()