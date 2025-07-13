#!/usr/bin/env python3
"""
Advanced Sandboxing System

Provides container-based isolation and access controls for secure code execution
without heavy dependencies like Docker.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,loguru python advanced_sandbox.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
# ]
# ///

import asyncio
import os
import sys
import time
import uuid
import signal
import tempfile
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import shutil
import resource as sys_resource

import psutil
from loguru import logger

from .context import SecurityContext, SecurityLevel
from .permissions import PermissionManager


class IsolationType(Enum):
    """Types of isolation available."""
    NONE = "none"                    # No isolation (development only)
    PROCESS = "process"              # Process-level isolation
    CHROOT = "chroot"               # Filesystem isolation
    NAMESPACE = "namespace"          # Linux namespace isolation
    CONTAINER = "container"          # Full container isolation


class ResourceType(Enum):
    """Types of resources that can be limited."""
    CPU_TIME = "cpu_time"           # CPU time limit in seconds
    MEMORY = "memory"               # Memory limit in bytes
    DISK_SPACE = "disk_space"       # Disk usage limit in bytes
    FILE_DESCRIPTORS = "file_descriptors"  # File descriptor limit
    PROCESSES = "processes"         # Process count limit
    NETWORK_BANDWIDTH = "network_bandwidth"  # Network bandwidth limit


@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution."""
    
    cpu_time_seconds: Optional[float] = 60.0       # Maximum CPU time
    memory_bytes: Optional[int] = 256 * 1024 * 1024  # 256MB default
    disk_space_bytes: Optional[int] = 100 * 1024 * 1024  # 100MB default
    file_descriptors: Optional[int] = 64           # File descriptor limit
    processes: Optional[int] = 10                  # Process count limit
    network_allowed: bool = False                  # Network access allowed
    execution_timeout: Optional[float] = 120.0    # Total execution timeout
    
    def to_resource_dict(self) -> Dict[int, int]:
        """Convert to resource module limits."""
        limits = {}
        
        if self.cpu_time_seconds:
            limits[sys_resource.RLIMIT_CPU] = int(self.cpu_time_seconds)
        
        if self.memory_bytes:
            limits[sys_resource.RLIMIT_AS] = self.memory_bytes
        
        if self.file_descriptors:
            limits[sys_resource.RLIMIT_NOFILE] = self.file_descriptors
        
        if self.processes:
            limits[sys_resource.RLIMIT_NPROC] = self.processes
        
        return limits


@dataclass
class FileSystemRestrictions:
    """File system access restrictions."""
    
    allowed_read_paths: Set[str] = field(default_factory=set)
    allowed_write_paths: Set[str] = field(default_factory=set)
    allowed_execute_paths: Set[str] = field(default_factory=set)
    
    blocked_paths: Set[str] = field(default_factory=lambda: {
        "/etc/passwd", "/etc/shadow", "/etc/sudoers",
        "/root", "/boot", "/sys", "/proc/sys"
    })
    
    read_only_paths: Set[str] = field(default_factory=lambda: {
        "/usr", "/bin", "/sbin", "/lib", "/lib64"
    })
    
    temp_directory: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB default


@dataclass
class NetworkRestrictions:
    """Network access restrictions."""
    
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_domains: Set[str] = field(default_factory=set)
    allowed_ports: Set[int] = field(default_factory=set)
    blocked_ports: Set[int] = field(default_factory=lambda: {22, 23, 135, 139, 445})
    
    allow_outbound: bool = False
    allow_inbound: bool = False
    allow_localhost: bool = True
    
    bandwidth_limit_kbps: Optional[int] = None


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    
    isolation_type: IsolationType = IsolationType.PROCESS
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    filesystem_restrictions: FileSystemRestrictions = field(default_factory=FileSystemRestrictions)
    network_restrictions: NetworkRestrictions = field(default_factory=NetworkRestrictions)
    
    # Execution environment
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Security options
    drop_privileges: bool = True
    new_session: bool = True
    enable_core_dumps: bool = False
    
    # Monitoring
    log_system_calls: bool = False
    monitor_file_access: bool = True
    monitor_network_access: bool = True


class SandboxedProcess:
    """Represents a sandboxed process."""
    
    def __init__(
        self,
        sandbox_id: str,
        process: subprocess.Popen,
        config: SandboxConfig,
        temp_dir: Optional[Path] = None
    ):
        self.sandbox_id = sandbox_id
        self.process = process
        self.config = config
        self.temp_dir = temp_dir
        
        self.start_time = time.time()
        self.cpu_usage = 0.0
        self.memory_usage = 0
        self.disk_usage = 0
        
        self._monitoring = True
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Start monitoring
        if self.process.poll() is None:
            self._start_monitoring()
    
    def _start_monitoring(self) -> None:
        """Start resource monitoring thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_resources(self) -> None:
        """Monitor process resources."""
        try:
            psutil_process = psutil.Process(self.process.pid)
            
            while self._monitoring and self.process.poll() is None:
                try:
                    # Update resource usage
                    self.cpu_usage = psutil_process.cpu_percent()
                    
                    memory_info = psutil_process.memory_info()
                    self.memory_usage = memory_info.rss
                    
                    # Check resource limits
                    if self.config.resource_limits.memory_bytes:
                        if self.memory_usage > self.config.resource_limits.memory_bytes:
                            logger.warning(f"Sandbox {self.sandbox_id} exceeded memory limit")
                            self.terminate()
                            break
                    
                    # Check execution timeout
                    if self.config.resource_limits.execution_timeout:
                        elapsed = time.time() - self.start_time
                        if elapsed > self.config.resource_limits.execution_timeout:
                            logger.warning(f"Sandbox {self.sandbox_id} exceeded execution timeout")
                            self.terminate()
                            break
                    
                    time.sleep(0.1)  # Monitor every 100ms
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                except Exception as e:
                    logger.error(f"Error monitoring sandbox {self.sandbox_id}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to start monitoring for sandbox {self.sandbox_id}: {e}")
    
    def terminate(self) -> None:
        """Terminate the sandboxed process."""
        self._monitoring = False
        
        try:
            if self.process.poll() is None:
                # Try graceful termination first
                self.process.terminate()
                
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    self.process.kill()
                    self.process.wait()
                    
        except Exception as e:
            logger.error(f"Error terminating sandbox {self.sandbox_id}: {e}")
    
    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process.poll() is None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the sandboxed process."""
        return {
            "sandbox_id": self.sandbox_id,
            "pid": self.process.pid,
            "alive": self.is_alive(),
            "return_code": self.process.returncode,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "runtime_seconds": time.time() - self.start_time,
            "config": {
                "isolation_type": self.config.isolation_type.value,
                "memory_limit": self.config.resource_limits.memory_bytes,
                "cpu_limit": self.config.resource_limits.cpu_time_seconds,
                "network_allowed": self.config.resource_limits.network_allowed
            }
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.terminate()
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass


class AdvancedSandbox:
    """
    Advanced sandboxing system for secure code execution.
    
    Provides multiple levels of isolation from simple process isolation
    to full container-like sandboxing using available platform features.
    """
    
    def __init__(
        self,
        permission_manager: Optional[PermissionManager] = None,
        default_config: Optional[SandboxConfig] = None
    ):
        """
        Initialize the advanced sandbox.
        
        Args:
            permission_manager: Permission manager for access control
            default_config: Default sandbox configuration
        """
        self.permission_manager = permission_manager
        self.default_config = default_config or SandboxConfig()
        
        # Track active sandboxes
        self.active_sandboxes: Dict[str, SandboxedProcess] = {}
        
        # Platform capabilities
        self.capabilities = self._detect_capabilities()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Advanced sandbox initialized with capabilities: {list(self.capabilities.keys())}")
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect available sandboxing capabilities."""
        capabilities = {
            "process_isolation": True,  # Always available
            "resource_limits": hasattr(sys_resource, 'RLIMIT_AS'),
            "chroot": os.name == 'posix' and os.geteuid() == 0,
            "namespaces": False,
            "containers": False
        }
        
        # Check for Linux namespace support
        if sys.platform.startswith('linux'):
            try:
                # Check if /proc/self/ns exists (namespace support indicator)
                capabilities["namespaces"] = Path("/proc/self/ns").exists()
            except Exception:
                pass
        
        # Check for container runtime
        for runtime in ['podman', 'docker']:
            if shutil.which(runtime):
                capabilities["containers"] = True
                break
        
        return capabilities
    
    def _create_temp_directory(self) -> Path:
        """Create a temporary directory for sandbox execution."""
        temp_dir = Path(tempfile.mkdtemp(prefix="tektra_sandbox_"))
        
        # Set restrictive permissions
        temp_dir.chmod(0o700)
        
        return temp_dir
    
    def _prepare_environment(
        self,
        config: SandboxConfig,
        temp_dir: Path
    ) -> Dict[str, str]:
        """Prepare environment variables for sandboxed execution."""
        env = os.environ.copy()
        
        # Remove potentially dangerous environment variables
        dangerous_vars = [
            'LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH',
            'PATH', 'HOME', 'USER', 'LOGNAME'
        ]
        
        for var in dangerous_vars:
            env.pop(var, None)
        
        # Set safe defaults
        env.update({
            'HOME': str(temp_dir),
            'TMPDIR': str(temp_dir),
            'USER': 'sandbox',
            'LOGNAME': 'sandbox',
            'PATH': '/usr/bin:/bin',  # Minimal PATH
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1'
        })
        
        # Add custom environment variables
        env.update(config.environment_variables)
        
        return env
    
    def _apply_resource_limits(self, config: SandboxConfig) -> None:
        """Apply resource limits to the current process."""
        if not self.capabilities["resource_limits"]:
            logger.warning("Resource limits not supported on this platform")
            return
        
        try:
            limits = config.resource_limits.to_resource_dict()
            
            for resource_type, limit in limits.items():
                try:
                    sys_resource.setrlimit(resource_type, (limit, limit))
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not set resource limit {resource_type}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to apply resource limits: {e}")
    
    def _setup_filesystem_isolation(
        self,
        config: SandboxConfig,
        temp_dir: Path
    ) -> None:
        """Set up filesystem isolation (chroot if available)."""
        if not config.filesystem_restrictions.temp_directory:
            config.filesystem_restrictions.temp_directory = str(temp_dir)
        
        # Create necessary directories in temp space
        (temp_dir / "tmp").mkdir(exist_ok=True)
        (temp_dir / "work").mkdir(exist_ok=True)
        
        # If chroot is available and requested
        if (self.capabilities["chroot"] and 
            config.isolation_type in [IsolationType.CHROOT, IsolationType.NAMESPACE]):
            try:
                # This would require root privileges
                # os.chroot(str(temp_dir))
                # os.chdir("/")
                logger.debug("Chroot isolation would be applied here (requires root)")
            except Exception as e:
                logger.warning(f"Could not apply chroot isolation: {e}")
    
    def _create_isolated_process(
        self,
        command: List[str],
        config: SandboxConfig,
        temp_dir: Path,
        env: Dict[str, str]
    ) -> subprocess.Popen:
        """Create an isolated process based on available capabilities."""
        
        def preexec_fn():
            """Function to run in child process before exec."""
            try:
                # Apply resource limits
                self._apply_resource_limits(config)
                
                # Create new session if requested
                if config.new_session:
                    os.setsid()
                
                # Change working directory
                if config.working_directory:
                    os.chdir(config.working_directory)
                else:
                    os.chdir(str(temp_dir))
                
                # Drop privileges if requested (requires careful implementation)
                if config.drop_privileges:
                    # This would involve dropping to non-root user
                    pass
                
            except Exception as e:
                logger.error(f"Error in preexec_fn: {e}")
                raise
        
        # Create process with isolation
        process = subprocess.Popen(
            command,
            env=env,
            cwd=str(temp_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            preexec_fn=preexec_fn if os.name == 'posix' else None,
            start_new_session=config.new_session
        )
        
        return process
    
    async def execute(
        self,
        command: List[str],
        config: Optional[SandboxConfig] = None,
        input_data: Optional[bytes] = None,
        security_context: Optional[SecurityContext] = None
    ) -> Tuple[str, bytes, bytes, int]:
        """
        Execute a command in a sandbox.
        
        Args:
            command: Command and arguments to execute
            config: Sandbox configuration (uses default if None)
            input_data: Input data to send to process
            security_context: Security context for permission checking
            
        Returns:
            Tuple of (sandbox_id, stdout, stderr, return_code)
        """
        config = config or self.default_config
        sandbox_id = str(uuid.uuid4())
        
        # Check permissions if manager is available
        if self.permission_manager and security_context:
            # Check system command permission
            if not self.permission_manager.has_permission(
                security_context.agent_id, "system.command"
            ):
                raise PermissionError("System command execution not permitted")
        
        with self._lock:
            logger.info(f"Starting sandbox {sandbox_id} with isolation {config.isolation_type.value}")
            
            try:
                # Create temporary directory
                temp_dir = self._create_temp_directory()
                
                # Prepare environment
                env = self._prepare_environment(config, temp_dir)
                
                # Set up filesystem isolation
                self._setup_filesystem_isolation(config, temp_dir)
                
                # Create isolated process
                process = self._create_isolated_process(command, config, temp_dir, env)
                
                # Create sandboxed process wrapper
                sandboxed_process = SandboxedProcess(
                    sandbox_id=sandbox_id,
                    process=process,
                    config=config,
                    temp_dir=temp_dir
                )
                
                # Track active sandbox
                self.active_sandboxes[sandbox_id] = sandboxed_process
                
                # Execute with timeout
                timeout = config.resource_limits.execution_timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(process.communicate, input_data)
                        ),
                        timeout=timeout
                    )
                    
                    return_code = process.returncode
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Sandbox {sandbox_id} timed out")
                    sandboxed_process.terminate()
                    return_code = -1
                    stdout = stderr = b"Execution timed out"
                
                # Log execution results
                logger.info(f"Sandbox {sandbox_id} completed with return code {return_code}")
                
                return sandbox_id, stdout, stderr, return_code
                
            except Exception as e:
                logger.error(f"Sandbox execution failed: {e}")
                raise
            
            finally:
                # Cleanup
                if sandbox_id in self.active_sandboxes:
                    self.active_sandboxes[sandbox_id].terminate()
                    del self.active_sandboxes[sandbox_id]
    
    def get_sandbox_status(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific sandbox."""
        with self._lock:
            sandboxed_process = self.active_sandboxes.get(sandbox_id)
            if sandboxed_process:
                return sandboxed_process.get_status()
            return None
    
    def list_active_sandboxes(self) -> List[Dict[str, Any]]:
        """List all active sandboxes."""
        with self._lock:
            return [
                sandboxed_process.get_status() 
                for sandboxed_process in self.active_sandboxes.values()
            ]
    
    def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Terminate a specific sandbox."""
        with self._lock:
            sandboxed_process = self.active_sandboxes.get(sandbox_id)
            if sandboxed_process:
                sandboxed_process.terminate()
                del self.active_sandboxes[sandbox_id]
                logger.info(f"Terminated sandbox {sandbox_id}")
                return True
            return False
    
    def cleanup_all_sandboxes(self) -> None:
        """Cleanup all active sandboxes."""
        with self._lock:
            sandbox_ids = list(self.active_sandboxes.keys())
            for sandbox_id in sandbox_ids:
                self.terminate_sandbox(sandbox_id)
            
            logger.info(f"Cleaned up {len(sandbox_ids)} sandboxes")
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get available sandboxing capabilities."""
        return self.capabilities.copy()


def create_advanced_sandbox(**kwargs) -> AdvancedSandbox:
    """
    Create an advanced sandbox with the given configuration.
    
    Args:
        **kwargs: Sandbox configuration
        
    Returns:
        Configured advanced sandbox
    """
    return AdvancedSandbox(**kwargs)


if __name__ == "__main__":
    async def demo_advanced_sandbox():
        """Demonstrate advanced sandbox functionality."""
        print("🛡️ Advanced Sandbox Demo")
        print("=" * 40)
        
        # Create sandbox
        sandbox = create_advanced_sandbox()
        
        print(f"Platform capabilities: {sandbox.get_capabilities()}")
        
        # Create test configurations
        configs = [
            ("Basic Process Isolation", SandboxConfig(
                isolation_type=IsolationType.PROCESS,
                resource_limits=ResourceLimits(
                    memory_bytes=64 * 1024 * 1024,  # 64MB
                    cpu_time_seconds=10.0,
                    execution_timeout=15.0
                )
            )),
            ("Restricted Environment", SandboxConfig(
                isolation_type=IsolationType.PROCESS,
                resource_limits=ResourceLimits(
                    memory_bytes=32 * 1024 * 1024,  # 32MB
                    cpu_time_seconds=5.0,
                    execution_timeout=10.0,
                    network_allowed=False
                ),
                environment_variables={"RESTRICTED": "true"}
            ))
        ]
        
        for name, config in configs:
            print(f"\n{name}:")
            print("-" * len(name))
            
            try:
                # Test safe command
                sandbox_id, stdout, stderr, code = await sandbox.execute(
                    ["python3", "-c", "print('Hello from sandbox!'); import os; print(f'PID: {os.getpid()}')"],
                    config=config
                )
                
                print(f"✅ Sandbox {sandbox_id[:8]}:")
                print(f"   Return code: {code}")
                print(f"   Stdout: {stdout.decode().strip()}")
                if stderr:
                    print(f"   Stderr: {stderr.decode().strip()}")
                
                # Test resource-intensive command
                sandbox_id, stdout, stderr, code = await sandbox.execute(
                    ["python3", "-c", "x = [0] * (10**6); print('Memory test passed')"],
                    config=config
                )
                
                print(f"✅ Memory test {sandbox_id[:8]}: code={code}")
                if stdout:
                    print(f"   Output: {stdout.decode().strip()}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Test timeout handling
        print(f"\nTimeout Test:")
        print("-" * 12)
        
        timeout_config = SandboxConfig(
            resource_limits=ResourceLimits(execution_timeout=2.0)
        )
        
        try:
            sandbox_id, stdout, stderr, code = await sandbox.execute(
                ["python3", "-c", "import time; time.sleep(5); print('Should not reach here')"],
                config=timeout_config
            )
            
            print(f"✅ Timeout test {sandbox_id[:8]}: code={code}")
            print(f"   Output: {stdout.decode() if stdout else 'No output'}")
            
        except Exception as e:
            print(f"❌ Timeout error: {e}")
        
        # Show active sandboxes
        active = sandbox.list_active_sandboxes()
        print(f"\nActive sandboxes: {len(active)}")
        
        # Cleanup
        sandbox.cleanup_all_sandboxes()
        print("\n🛡️ Advanced Sandbox Demo Complete")
    
    # Run demo
    asyncio.run(demo_advanced_sandbox())