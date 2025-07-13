#!/usr/bin/env python3
"""
Resource Pool Management System

Provides efficient pooling and lifecycle management for expensive resources
like models, connections, and compute resources.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,loguru python resource_pool.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
# ]
# ///

import asyncio
import gc
import time
import uuid
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union
import threading
from contextlib import asynccontextmanager, contextmanager

import psutil
from loguru import logger


class ResourceType(Enum):
    """Types of pooled resources."""
    MODEL = "model"
    MEMORY = "memory"
    CONNECTION = "connection"
    THREAD = "thread"
    PROCESS = "process"
    GPU_CONTEXT = "gpu_context"
    CUSTOM = "custom"


class ResourceState(Enum):
    """States of pooled resources."""
    IDLE = "idle"                  # Available for use
    RESERVED = "reserved"          # Reserved but not yet in use
    ACTIVE = "active"             # Currently in use
    LOADING = "loading"           # Being initialized
    UNLOADING = "unloading"       # Being cleaned up
    ERROR = "error"               # In error state
    RETIRED = "retired"           # Marked for removal


@dataclass
class ResourceMetrics:
    """Metrics for resource usage."""
    total_acquisitions: int = 0
    total_releases: int = 0
    total_errors: int = 0
    total_evictions: int = 0
    
    acquisition_time_ms: List[float] = field(default_factory=list)
    hold_time_ms: List[float] = field(default_factory=list)
    
    peak_usage: int = 0
    current_usage: int = 0
    
    last_acquisition: Optional[float] = None
    last_release: Optional[float] = None
    last_error: Optional[float] = None
    
    def record_acquisition(self, time_ms: float) -> None:
        """Record resource acquisition."""
        self.total_acquisitions += 1
        self.acquisition_time_ms.append(time_ms)
        self.last_acquisition = time.time()
        self.current_usage += 1
        self.peak_usage = max(self.peak_usage, self.current_usage)
        
        # Keep only last 1000 measurements
        if len(self.acquisition_time_ms) > 1000:
            self.acquisition_time_ms = self.acquisition_time_ms[-1000:]
    
    def record_release(self, hold_time_ms: float) -> None:
        """Record resource release."""
        self.total_releases += 1
        self.hold_time_ms.append(hold_time_ms)
        self.last_release = time.time()
        self.current_usage = max(0, self.current_usage - 1)
        
        # Keep only last 1000 measurements
        if len(self.hold_time_ms) > 1000:
            self.hold_time_ms = self.hold_time_ms[-1000:]
    
    def record_error(self) -> None:
        """Record resource error."""
        self.total_errors += 1
        self.last_error = time.time()
    
    def record_eviction(self) -> None:
        """Record resource eviction."""
        self.total_evictions += 1
    
    @property
    def average_acquisition_time(self) -> float:
        """Get average acquisition time in milliseconds."""
        if not self.acquisition_time_ms:
            return 0.0
        return sum(self.acquisition_time_ms) / len(self.acquisition_time_ms)
    
    @property
    def average_hold_time(self) -> float:
        """Get average hold time in milliseconds."""
        if not self.hold_time_ms:
            return 0.0
        return sum(self.hold_time_ms) / len(self.hold_time_ms)


T = TypeVar('T')


@dataclass
class PooledResource(Generic[T]):
    """Wrapper for a pooled resource."""
    
    resource_id: str
    resource: T
    resource_type: ResourceType
    state: ResourceState = ResourceState.IDLE
    
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    error_count: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_callback: Optional[Callable[[T], None]] = None
    
    # Lifecycle tracking
    acquired_at: Optional[float] = None
    acquired_by: Optional[str] = None
    
    def acquire(self, acquired_by: str = "unknown") -> None:
        """Mark resource as acquired."""
        self.state = ResourceState.ACTIVE
        self.acquired_at = time.time()
        self.acquired_by = acquired_by
        self.last_accessed = time.time()
        self.access_count += 1
    
    def release(self) -> None:
        """Mark resource as released."""
        self.state = ResourceState.IDLE
        self.acquired_at = None
        self.acquired_by = None
        self.last_accessed = time.time()
    
    def mark_error(self) -> None:
        """Mark resource as having an error."""
        self.state = ResourceState.ERROR
        self.error_count += 1
        self.last_accessed = time.time()
    
    def cleanup(self) -> None:
        """Clean up the resource."""
        if self.cleanup_callback:
            try:
                self.cleanup_callback(self.resource)
            except Exception as e:
                logger.error(f"Error cleaning up resource {self.resource_id}: {e}")
    
    @property
    def age_seconds(self) -> float:
        """Get age of resource in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        if self.state != ResourceState.IDLE:
            return 0.0
        return time.time() - self.last_accessed


class ResourcePool(Generic[T]):
    """
    Generic resource pool for efficient resource management.
    
    Features:
    - Configurable pool size limits
    - Lazy resource creation
    - Resource lifecycle management
    - Health checking and eviction
    - Metrics and monitoring
    - Thread-safe operations
    """
    
    def __init__(
        self,
        resource_type: ResourceType,
        factory: Callable[[], T],
        cleanup: Optional[Callable[[T], None]] = None,
        min_size: int = 0,
        max_size: int = 10,
        max_idle_seconds: float = 300.0,
        max_age_seconds: Optional[float] = None,
        health_check: Optional[Callable[[T], bool]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize resource pool.
        
        Args:
            resource_type: Type of resources in the pool
            factory: Function to create new resources
            cleanup: Function to clean up resources
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_seconds: Maximum idle time before eviction
            max_age_seconds: Maximum age before eviction
            health_check: Function to check resource health
            name: Pool name for identification
        """
        self.resource_type = resource_type
        self.factory = factory
        self.cleanup_callback = cleanup
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_seconds = max_idle_seconds
        self.max_age_seconds = max_age_seconds
        self.health_check = health_check
        self.name = name or f"{resource_type.value}_pool"
        
        # Resource storage
        self.resources: Dict[str, PooledResource[T]] = {}
        self.idle_queue: deque[str] = deque()
        self.waiters: deque[asyncio.Future] = deque()
        
        # Metrics
        self.metrics = ResourceMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        
        # Initialize minimum resources
        self._initialize_min_resources()
        
        logger.info(f"Resource pool '{self.name}' initialized (min={min_size}, max={max_size})")
    
    def _initialize_min_resources(self) -> None:
        """Initialize minimum number of resources."""
        for _ in range(self.min_size):
            try:
                self._create_resource()
            except Exception as e:
                logger.warning(f"Failed to create initial resource: {e}")
    
    def _create_resource(self) -> str:
        """Create a new resource and add to pool."""
        with self._lock:
            if len(self.resources) >= self.max_size:
                raise RuntimeError(f"Pool '{self.name}' is at maximum capacity ({self.max_size})")
            
            resource_id = str(uuid.uuid4())
            
            try:
                # Create the resource
                resource = self.factory()
                
                # Wrap in pooled resource
                pooled = PooledResource(
                    resource_id=resource_id,
                    resource=resource,
                    resource_type=self.resource_type,
                    state=ResourceState.IDLE,
                    cleanup_callback=self.cleanup_callback
                )
                
                # Add to pool
                self.resources[resource_id] = pooled
                self.idle_queue.append(resource_id)
                
                logger.debug(f"Created resource {resource_id} in pool '{self.name}'")
                return resource_id
                
            except Exception as e:
                logger.error(f"Failed to create resource in pool '{self.name}': {e}")
                self.metrics.record_error()
                raise
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None, acquired_by: str = "unknown"):
        """
        Acquire a resource from the pool (async context manager).
        
        Args:
            timeout: Maximum time to wait for resource
            acquired_by: Identifier of who is acquiring the resource
            
        Yields:
            The acquired resource
        """
        resource_id = await self.acquire_resource(timeout, acquired_by)
        
        try:
            yield self.resources[resource_id].resource
        finally:
            self.release_resource(resource_id)
    
    @contextmanager
    def acquire_sync(self, timeout: Optional[float] = None, acquired_by: str = "unknown"):
        """
        Acquire a resource from the pool (sync context manager).
        
        Args:
            timeout: Maximum time to wait for resource
            acquired_by: Identifier of who is acquiring the resource
            
        Yields:
            The acquired resource
        """
        resource_id = self.acquire_resource_sync(timeout, acquired_by)
        
        try:
            yield self.resources[resource_id].resource
        finally:
            self.release_resource(resource_id)
    
    async def acquire_resource(
        self,
        timeout: Optional[float] = None,
        acquired_by: str = "unknown"
    ) -> str:
        """
        Acquire a resource from the pool.
        
        Args:
            timeout: Maximum time to wait for resource
            acquired_by: Identifier of who is acquiring the resource
            
        Returns:
            Resource ID
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if self._shutdown:
                    raise RuntimeError(f"Pool '{self.name}' is shutting down")
                
                # Try to get an idle resource
                resource_id = self._try_acquire_idle_resource()
                
                if resource_id:
                    # Found an idle resource
                    pooled = self.resources[resource_id]
                    pooled.acquire(acquired_by)
                    
                    acquisition_time = (time.time() - start_time) * 1000
                    self.metrics.record_acquisition(acquisition_time)
                    
                    logger.debug(f"Acquired resource {resource_id} from pool '{self.name}'")
                    return resource_id
                
                # Try to create a new resource if under limit
                if len(self.resources) < self.max_size:
                    try:
                        resource_id = self._create_resource()
                        pooled = self.resources[resource_id]
                        
                        # Remove from idle queue and acquire
                        self.idle_queue.remove(resource_id)
                        pooled.acquire(acquired_by)
                        
                        acquisition_time = (time.time() - start_time) * 1000
                        self.metrics.record_acquisition(acquisition_time)
                        
                        return resource_id
                        
                    except Exception as e:
                        logger.warning(f"Failed to create new resource: {e}")
                
                # Need to wait for a resource
                if timeout is not None and (time.time() - start_time) >= timeout:
                    raise asyncio.TimeoutError(f"Timeout waiting for resource from pool '{self.name}'")
                
                # Create a future to wait on
                future = asyncio.Future()
                self.waiters.append(future)
            
            # Wait outside the lock
            try:
                await asyncio.wait_for(future, timeout=1.0 if timeout is None else min(1.0, timeout - (time.time() - start_time)))
            except asyncio.TimeoutError:
                with self._lock:
                    if future in self.waiters:
                        self.waiters.remove(future)
                continue
    
    def acquire_resource_sync(
        self,
        timeout: Optional[float] = None,
        acquired_by: str = "unknown"
    ) -> str:
        """
        Synchronously acquire a resource from the pool.
        
        Args:
            timeout: Maximum time to wait for resource
            acquired_by: Identifier of who is acquiring the resource
            
        Returns:
            Resource ID
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if self._shutdown:
                    raise RuntimeError(f"Pool '{self.name}' is shutting down")
                
                # Try to get an idle resource
                resource_id = self._try_acquire_idle_resource()
                
                if resource_id:
                    pooled = self.resources[resource_id]
                    pooled.acquire(acquired_by)
                    
                    acquisition_time = (time.time() - start_time) * 1000
                    self.metrics.record_acquisition(acquisition_time)
                    
                    return resource_id
                
                # Try to create a new resource if under limit
                if len(self.resources) < self.max_size:
                    try:
                        resource_id = self._create_resource()
                        pooled = self.resources[resource_id]
                        
                        # Remove from idle queue and acquire
                        self.idle_queue.remove(resource_id)
                        pooled.acquire(acquired_by)
                        
                        acquisition_time = (time.time() - start_time) * 1000
                        self.metrics.record_acquisition(acquisition_time)
                        
                        return resource_id
                        
                    except Exception as e:
                        logger.warning(f"Failed to create new resource: {e}")
                
                # Check timeout
                if timeout is not None and (time.time() - start_time) >= timeout:
                    raise TimeoutError(f"Timeout waiting for resource from pool '{self.name}'")
            
            # Wait a bit before retrying
            time.sleep(0.1)
    
    def _try_acquire_idle_resource(self) -> Optional[str]:
        """Try to acquire an idle resource from the pool."""
        while self.idle_queue:
            resource_id = self.idle_queue.popleft()
            
            if resource_id not in self.resources:
                continue
            
            pooled = self.resources[resource_id]
            
            # Check if resource is healthy
            if self.health_check and pooled.resource:
                try:
                    if not self.health_check(pooled.resource):
                        logger.warning(f"Resource {resource_id} failed health check")
                        self._evict_resource(resource_id)
                        continue
                except Exception as e:
                    logger.error(f"Health check error for resource {resource_id}: {e}")
                    self._evict_resource(resource_id)
                    continue
            
            # Check if resource is still valid
            if pooled.state == ResourceState.IDLE:
                return resource_id
        
        return None
    
    def release_resource(self, resource_id: str) -> None:
        """
        Release a resource back to the pool.
        
        Args:
            resource_id: ID of the resource to release
        """
        with self._lock:
            if resource_id not in self.resources:
                logger.warning(f"Attempted to release unknown resource {resource_id}")
                return
            
            pooled = self.resources[resource_id]
            
            if pooled.state != ResourceState.ACTIVE:
                logger.warning(f"Attempted to release non-active resource {resource_id}")
                return
            
            # Record hold time
            if pooled.acquired_at:
                hold_time = (time.time() - pooled.acquired_at) * 1000
                self.metrics.record_release(hold_time)
            
            # Release the resource
            pooled.release()
            
            # Add back to idle queue
            self.idle_queue.append(resource_id)
            
            # Notify any waiters
            if self.waiters:
                future = self.waiters.popleft()
                if not future.done():
                    future.set_result(None)
            
            logger.debug(f"Released resource {resource_id} to pool '{self.name}'")
    
    def mark_unhealthy(self, resource_id: str) -> None:
        """
        Mark a resource as unhealthy and remove from pool.
        
        Args:
            resource_id: ID of the resource to mark unhealthy
        """
        with self._lock:
            if resource_id in self.resources:
                pooled = self.resources[resource_id]
                pooled.mark_error()
                self._evict_resource(resource_id)
                logger.warning(f"Marked resource {resource_id} as unhealthy")
    
    def _evict_resource(self, resource_id: str) -> None:
        """Evict a resource from the pool."""
        if resource_id not in self.resources:
            return
        
        pooled = self.resources[resource_id]
        
        # Remove from idle queue if present
        if resource_id in self.idle_queue:
            self.idle_queue.remove(resource_id)
        
        # Clean up the resource
        pooled.cleanup()
        
        # Remove from pool
        del self.resources[resource_id]
        
        self.metrics.record_eviction()
        logger.debug(f"Evicted resource {resource_id} from pool '{self.name}'")
    
    async def maintain(self) -> None:
        """Run maintenance tasks on the pool."""
        with self._lock:
            current_time = time.time()
            to_evict = []
            
            # Check each resource
            for resource_id, pooled in self.resources.items():
                # Skip active resources
                if pooled.state == ResourceState.ACTIVE:
                    continue
                
                # Check idle timeout
                if pooled.idle_seconds > self.max_idle_seconds:
                    to_evict.append(resource_id)
                    logger.debug(f"Resource {resource_id} exceeded idle timeout")
                    continue
                
                # Check age timeout
                if self.max_age_seconds and pooled.age_seconds > self.max_age_seconds:
                    to_evict.append(resource_id)
                    logger.debug(f"Resource {resource_id} exceeded age limit")
                    continue
                
                # Check health
                if self.health_check and pooled.state == ResourceState.IDLE:
                    try:
                        if not self.health_check(pooled.resource):
                            to_evict.append(resource_id)
                            logger.debug(f"Resource {resource_id} failed health check")
                    except Exception as e:
                        logger.error(f"Health check error for resource {resource_id}: {e}")
                        to_evict.append(resource_id)
            
            # Evict resources
            for resource_id in to_evict:
                self._evict_resource(resource_id)
            
            # Ensure minimum pool size
            while len(self.resources) < self.min_size:
                try:
                    self._create_resource()
                except Exception as e:
                    logger.warning(f"Failed to maintain minimum pool size: {e}")
                    break
            
            # Log pool status
            idle_count = len(self.idle_queue)
            active_count = len(self.resources) - idle_count
            logger.debug(f"Pool '{self.name}' status: {active_count} active, {idle_count} idle, {len(to_evict)} evicted")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status and metrics."""
        with self._lock:
            idle_count = len(self.idle_queue)
            active_count = len([r for r in self.resources.values() if r.state == ResourceState.ACTIVE])
            error_count = len([r for r in self.resources.values() if r.state == ResourceState.ERROR])
            
            return {
                "name": self.name,
                "resource_type": self.resource_type.value,
                "total_resources": len(self.resources),
                "idle_resources": idle_count,
                "active_resources": active_count,
                "error_resources": error_count,
                "min_size": self.min_size,
                "max_size": self.max_size,
                "metrics": {
                    "total_acquisitions": self.metrics.total_acquisitions,
                    "total_releases": self.metrics.total_releases,
                    "total_errors": self.metrics.total_errors,
                    "total_evictions": self.metrics.total_evictions,
                    "average_acquisition_time_ms": self.metrics.average_acquisition_time,
                    "average_hold_time_ms": self.metrics.average_hold_time,
                    "peak_usage": self.metrics.peak_usage,
                    "current_usage": self.metrics.current_usage,
                },
                "waiters": len(self.waiters),
                "shutdown": self._shutdown
            }
    
    async def start_maintenance(self) -> None:
        """Start background maintenance task."""
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            logger.info(f"Started maintenance for pool '{self.name}'")
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while not self._shutdown:
            try:
                await self.maintain()
                await asyncio.sleep(10)  # Run maintenance every 10 seconds
            except Exception as e:
                logger.error(f"Error in maintenance loop for pool '{self.name}': {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the resource pool."""
        logger.info(f"Shutting down pool '{self.name}'")
        
        with self._lock:
            self._shutdown = True
            
            # Cancel all waiters
            for future in self.waiters:
                if not future.done():
                    future.cancel()
            self.waiters.clear()
            
            # Stop maintenance task
            if self._maintenance_task:
                self._maintenance_task.cancel()
                try:
                    await self._maintenance_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up all resources
            for resource_id in list(self.resources.keys()):
                self._evict_resource(resource_id)
        
        logger.info(f"Pool '{self.name}' shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'resources'):
            with self._lock:
                for resource_id in list(self.resources.keys()):
                    try:
                        self._evict_resource(resource_id)
                    except Exception:
                        pass


def create_resource_pool(
    resource_type: ResourceType,
    factory: Callable[[], T],
    **kwargs
) -> ResourcePool[T]:
    """
    Create a resource pool with the given configuration.
    
    Args:
        resource_type: Type of resources to pool
        factory: Function to create new resources
        **kwargs: Additional pool configuration
        
    Returns:
        Configured resource pool
    """
    return ResourcePool(resource_type=resource_type, factory=factory, **kwargs)


if __name__ == "__main__":
    import random
    
    async def demo_resource_pool():
        """Demonstrate resource pool functionality."""
        print("🔧 Resource Pool Demo")
        print("=" * 40)
        
        # Mock resource class
        class MockResource:
            def __init__(self):
                self.id = str(uuid.uuid4())[:8]
                self.created = time.time()
                print(f"   Created resource {self.id}")
            
            def __del__(self):
                print(f"   Cleaned up resource {self.id}")
            
            def do_work(self):
                time.sleep(random.uniform(0.1, 0.3))
                return f"Work done by {self.id}"
        
        # Create resource pool
        pool = create_resource_pool(
            resource_type=ResourceType.CUSTOM,
            factory=MockResource,
            min_size=2,
            max_size=5,
            max_idle_seconds=30.0,
            name="demo_pool"
        )
        
        # Start maintenance
        await pool.start_maintenance()
        
        print(f"Pool created with {len(pool.resources)} initial resources")
        
        # Simulate concurrent resource usage
        async def use_resource(task_id: int):
            try:
                async with pool.acquire(timeout=5.0, acquired_by=f"task_{task_id}") as resource:
                    print(f"Task {task_id} acquired resource {resource.id}")
                    result = resource.do_work()
                    print(f"Task {task_id}: {result}")
            except asyncio.TimeoutError:
                print(f"Task {task_id}: Timeout waiting for resource")
        
        # Run multiple tasks concurrently
        print("\nRunning concurrent tasks...")
        tasks = []
        for i in range(8):
            tasks.append(asyncio.create_task(use_resource(i)))
        
        await asyncio.gather(*tasks)
        
        # Check pool status
        print("\nPool status:")
        status = pool.get_status()
        print(f"   Total resources: {status['total_resources']}")
        print(f"   Active resources: {status['active_resources']}")
        print(f"   Idle resources: {status['idle_resources']}")
        print(f"   Total acquisitions: {status['metrics']['total_acquisitions']}")
        print(f"   Average acquisition time: {status['metrics']['average_acquisition_time_ms']:.2f}ms")
        print(f"   Average hold time: {status['metrics']['average_hold_time_ms']:.2f}ms")
        
        # Shutdown pool
        await pool.shutdown()
        
        print("\n🔧 Resource Pool Demo Complete")
    
    # Run demo
    asyncio.run(demo_resource_pool())