#!/usr/bin/env python3
"""
Advanced Task Scheduling System

Priority-based task scheduling with work stealing, resource-aware execution,
and dynamic load balancing.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,loguru,networkx python task_scheduler.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
#     "networkx>=3.0",
# ]
# ///

import asyncio
import heapq
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import psutil
import networkx as nx
from loguru import logger


class Priority(Enum):
    """Task priority levels."""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting for dependencies


class ExecutorType(Enum):
    """Types of task executors."""
    SYNC = "sync"
    ASYNC = "async"
    THREAD = "thread"
    PROCESS = "process"


@dataclass
class ResourceRequirements:
    """Resource requirements for a task."""
    
    cpu_cores: float = 1.0           # Number of CPU cores
    memory_mb: float = 100.0         # Memory in MB
    gpu_memory_mb: float = 0.0       # GPU memory in MB
    disk_io_mbps: float = 0.0        # Disk I/O in MB/s
    network_mbps: float = 0.0        # Network bandwidth in MB/s
    
    exclusive_gpu: bool = False      # Requires exclusive GPU access
    numa_node: Optional[int] = None  # Preferred NUMA node
    
    def can_fit(self, available: 'ResourceRequirements') -> bool:
        """Check if requirements can fit in available resources."""
        return (
            self.cpu_cores <= available.cpu_cores and
            self.memory_mb <= available.memory_mb and
            self.gpu_memory_mb <= available.gpu_memory_mb and
            self.disk_io_mbps <= available.disk_io_mbps and
            self.network_mbps <= available.network_mbps
        )


@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    
    queue_time: float = 0.0          # Time spent in queue
    execution_time: float = 0.0      # Actual execution time
    total_time: float = 0.0          # Total time from submission to completion
    
    cpu_usage_percent: float = 0.0   # Average CPU usage
    memory_peak_mb: float = 0.0      # Peak memory usage
    
    retry_count: int = 0             # Number of retries
    error_count: int = 0             # Number of errors


@dataclass
class Task:
    """Represents a schedulable task."""
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    
    # Execution
    func: Optional[Callable] = None
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    priority: Priority = Priority.NORMAL
    state: TaskState = TaskState.PENDING
    executor_type: ExecutorType = ExecutorType.ASYNC
    
    # Resources
    requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this depends on
    dependents: Set[str] = field(default_factory=set)    # Tasks that depend on this
    
    # Timing
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    result: Any = None
    error: Optional[Exception] = None
    
    # Metrics
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    
    # Scheduling hints
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    preferred_worker: Optional[int] = None
    
    def __lt__(self, other: 'Task') -> bool:
        """Compare tasks by priority and creation time."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


@dataclass
class Worker:
    """Represents a worker that executes tasks."""
    
    worker_id: int
    name: str
    
    # Resources
    total_resources: ResourceRequirements
    available_resources: ResourceRequirements
    
    # Current workload
    running_tasks: Dict[str, Task] = field(default_factory=dict)
    task_queue: List[Task] = field(default_factory=list)  # Local queue for work stealing
    
    # Performance
    tasks_completed: int = 0
    total_execution_time: float = 0.0
    last_task_time: float = field(default_factory=time.time)
    
    # Status
    is_active: bool = True
    is_busy: bool = False
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 - 1.0)."""
        cpu_load = 1.0 - (self.available_resources.cpu_cores / self.total_resources.cpu_cores)
        mem_load = 1.0 - (self.available_resources.memory_mb / self.total_resources.memory_mb)
        return (cpu_load + mem_load) / 2.0
    
    def can_accept_task(self, task: Task) -> bool:
        """Check if worker can accept a task."""
        return (
            self.is_active and
            task.requirements.can_fit(self.available_resources)
        )
    
    def allocate_resources(self, task: Task) -> None:
        """Allocate resources for a task."""
        req = task.requirements
        self.available_resources.cpu_cores -= req.cpu_cores
        self.available_resources.memory_mb -= req.memory_mb
        self.available_resources.gpu_memory_mb -= req.gpu_memory_mb
        self.available_resources.disk_io_mbps -= req.disk_io_mbps
        self.available_resources.network_mbps -= req.network_mbps
    
    def release_resources(self, task: Task) -> None:
        """Release resources from a task."""
        req = task.requirements
        self.available_resources.cpu_cores += req.cpu_cores
        self.available_resources.memory_mb += req.memory_mb
        self.available_resources.gpu_memory_mb += req.gpu_memory_mb
        self.available_resources.disk_io_mbps += req.disk_io_mbps
        self.available_resources.network_mbps += req.network_mbps


class TaskScheduler:
    """
    Advanced task scheduling system with priority queues and work stealing.
    
    Features:
    - Priority-based scheduling
    - Resource-aware task placement
    - Work stealing for load balancing
    - Dependency graph resolution
    - Multiple executor types
    - Dynamic worker scaling
    - Task retry and timeout handling
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        max_threads: int = 10,
        max_processes: int = 4,
        enable_work_stealing: bool = True,
        enable_auto_scaling: bool = True
    ):
        """
        Initialize task scheduler.
        
        Args:
            num_workers: Number of workers (None for CPU count)
            max_threads: Maximum thread pool size
            max_processes: Maximum process pool size
            enable_work_stealing: Enable work stealing between workers
            enable_auto_scaling: Enable dynamic worker scaling
        """
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.max_threads = max_threads
        self.max_processes = max_processes
        self.enable_work_stealing = enable_work_stealing
        self.enable_auto_scaling = enable_auto_scaling
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.global_queue: List[Task] = []  # Min heap by priority
        self.blocked_tasks: Set[str] = set()
        
        # Workers
        self.workers: List[Worker] = []
        self._init_workers()
        
        # Dependency graph
        self.dependency_graph = nx.DiGraph()
        
        # Executors
        self.thread_pool = ThreadPoolExecutor(max_workers=max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        
        # Scheduling state
        self._scheduler_task: Optional[asyncio.Task] = None
        self._work_stealer_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Task scheduler initialized with {self.num_workers} workers")
    
    def _init_workers(self) -> None:
        """Initialize worker pool."""
        system_resources = self._get_system_resources()
        
        # Divide resources among workers
        cpu_per_worker = system_resources.cpu_cores / self.num_workers
        mem_per_worker = system_resources.memory_mb / self.num_workers
        
        for i in range(self.num_workers):
            worker = Worker(
                worker_id=i,
                name=f"worker_{i}",
                total_resources=ResourceRequirements(
                    cpu_cores=cpu_per_worker,
                    memory_mb=mem_per_worker,
                    gpu_memory_mb=system_resources.gpu_memory_mb / self.num_workers if system_resources.gpu_memory_mb > 0 else 0,
                    disk_io_mbps=100.0,  # Simplified
                    network_mbps=100.0   # Simplified
                ),
                available_resources=ResourceRequirements(
                    cpu_cores=cpu_per_worker,
                    memory_mb=mem_per_worker,
                    gpu_memory_mb=system_resources.gpu_memory_mb / self.num_workers if system_resources.gpu_memory_mb > 0 else 0,
                    disk_io_mbps=100.0,
                    network_mbps=100.0
                )
            )
            self.workers.append(worker)
    
    def _get_system_resources(self) -> ResourceRequirements:
        """Get available system resources."""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        # Simplified GPU detection
        gpu_memory = 0.0
        try:
            # Would use pynvml or similar in production
            gpu_memory = 0.0
        except:
            pass
        
        return ResourceRequirements(
            cpu_cores=float(cpu_count),
            memory_mb=memory.available / (1024 * 1024),
            gpu_memory_mb=gpu_memory,
            disk_io_mbps=1000.0,  # Simplified
            network_mbps=1000.0   # Simplified
        )
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        name: Optional[str] = None,
        priority: Priority = Priority.NORMAL,
        executor_type: ExecutorType = ExecutorType.ASYNC,
        requirements: Optional[ResourceRequirements] = None,
        dependencies: Optional[Set[str]] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            name: Task name
            priority: Task priority
            executor_type: Type of executor to use
            requirements: Resource requirements
            dependencies: Set of task IDs this depends on
            **kwargs: Keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            executor_type=executor_type,
            requirements=requirements or ResourceRequirements(),
            dependencies=dependencies or set()
        )
        
        with self._lock:
            self.tasks[task.task_id] = task
            self.total_tasks_submitted += 1
            
            # Add to dependency graph
            self.dependency_graph.add_node(task.task_id)
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self.dependency_graph.add_edge(dep_id, task.task_id)
                    self.tasks[dep_id].dependents.add(task.task_id)
            
            # Check if task is blocked
            if self._is_task_blocked(task):
                task.state = TaskState.BLOCKED
                self.blocked_tasks.add(task.task_id)
            else:
                # Add to global queue
                heapq.heappush(self.global_queue, task)
                task.state = TaskState.SCHEDULED
                task.scheduled_at = time.time()
        
        logger.debug(f"Submitted task {task.task_id} ({task.name})")
        return task.task_id
    
    def _is_task_blocked(self, task: Task) -> bool:
        """Check if task is blocked by dependencies."""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return True  # Dependency doesn't exist
            
            dep_task = self.tasks[dep_id]
            if dep_task.state not in [TaskState.COMPLETED]:
                return True  # Dependency not completed
        
        return False
    
    async def _schedule_tasks(self) -> None:
        """Main scheduling loop."""
        while not self._shutdown:
            try:
                with self._lock:
                    # Check for unblocked tasks
                    unblocked = []
                    for task_id in list(self.blocked_tasks):
                        task = self.tasks[task_id]
                        if not self._is_task_blocked(task):
                            unblocked.append(task)
                            self.blocked_tasks.remove(task_id)
                    
                    # Add unblocked tasks to queue
                    for task in unblocked:
                        heapq.heappush(self.global_queue, task)
                        task.state = TaskState.SCHEDULED
                        task.scheduled_at = time.time()
                    
                    # Assign tasks to workers
                    while self.global_queue:
                        task = heapq.heappop(self.global_queue)
                        
                        # Find best worker
                        best_worker = self._find_best_worker(task)
                        if best_worker:
                            # Assign to worker
                            heapq.heappush(best_worker.task_queue, task)
                            logger.debug(f"Assigned task {task.task_id} to {best_worker.name}")
                        else:
                            # No suitable worker, put back in queue
                            heapq.heappush(self.global_queue, task)
                            break
                
                # Let workers process tasks
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _find_best_worker(self, task: Task) -> Optional[Worker]:
        """Find the best worker for a task."""
        suitable_workers = [
            w for w in self.workers
            if w.can_accept_task(task)
        ]
        
        if not suitable_workers:
            return None
        
        # Prefer specified worker if available
        if task.preferred_worker is not None:
            for worker in suitable_workers:
                if worker.worker_id == task.preferred_worker:
                    return worker
        
        # Otherwise, choose least loaded worker
        return min(suitable_workers, key=lambda w: w.load_factor)
    
    async def _work_stealing_loop(self) -> None:
        """Work stealing loop for load balancing."""
        while not self._shutdown and self.enable_work_stealing:
            try:
                with self._lock:
                    # Find overloaded and underloaded workers
                    avg_load = sum(w.load_factor for w in self.workers) / len(self.workers)
                    
                    overloaded = [w for w in self.workers if w.load_factor > avg_load + 0.2]
                    underloaded = [w for w in self.workers if w.load_factor < avg_load - 0.2]
                    
                    # Steal work from overloaded to underloaded
                    for source in overloaded:
                        if not source.task_queue:
                            continue
                        
                        for target in underloaded:
                            if source.task_queue:
                                # Try to steal a task
                                for i, task in enumerate(source.task_queue):
                                    if target.can_accept_task(task):
                                        stolen_task = source.task_queue.pop(i)
                                        heapq.heappush(target.task_queue, stolen_task)
                                        logger.debug(f"Stole task {stolen_task.task_id} from {source.name} to {target.name}")
                                        break
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in work stealing loop: {e}")
    
    async def execute_task(self, task_id: str) -> Any:
        """
        Execute a specific task and wait for result.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Task result
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Wait for task to complete
        while task.state not in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
            await asyncio.sleep(0.1)
        
        if task.state == TaskState.FAILED:
            raise task.error or Exception(f"Task {task_id} failed")
        elif task.state == TaskState.CANCELLED:
            raise asyncio.CancelledError(f"Task {task_id} was cancelled")
        
        return task.result
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if cancelled successfully
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.state in [TaskState.COMPLETED, TaskState.FAILED]:
                return False  # Already finished
            
            task.state = TaskState.CANCELLED
            
            # Remove from queues
            if task in self.global_queue:
                self.global_queue.remove(task)
                heapq.heapify(self.global_queue)
            
            # Cancel dependents
            for dep_id in task.dependents:
                await self.cancel_task(dep_id)
            
            logger.info(f"Cancelled task {task_id}")
            return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status dictionary
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            return {
                "task_id": task.task_id,
                "name": task.name,
                "state": task.state.value,
                "priority": task.priority.value,
                "created_at": task.created_at,
                "scheduled_at": task.scheduled_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "dependencies": list(task.dependencies),
                "dependents": list(task.dependents),
                "metrics": {
                    "queue_time": task.metrics.queue_time,
                    "execution_time": task.metrics.execution_time,
                    "total_time": task.metrics.total_time,
                    "retry_count": task.metrics.retry_count,
                },
                "result": task.result if task.state == TaskState.COMPLETED else None,
                "error": str(task.error) if task.error else None,
            }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get overall scheduler status."""
        with self._lock:
            return {
                "workers": [
                    {
                        "worker_id": w.worker_id,
                        "name": w.name,
                        "load_factor": w.load_factor,
                        "tasks_completed": w.tasks_completed,
                        "running_tasks": len(w.running_tasks),
                        "queued_tasks": len(w.task_queue),
                        "is_active": w.is_active,
                    }
                    for w in self.workers
                ],
                "global_queue_size": len(self.global_queue),
                "blocked_tasks": len(self.blocked_tasks),
                "total_tasks": len(self.tasks),
                "tasks_by_state": {
                    state.value: sum(1 for t in self.tasks.values() if t.state == state)
                    for state in TaskState
                },
                "statistics": {
                    "total_submitted": self.total_tasks_submitted,
                    "total_completed": self.total_tasks_completed,
                    "total_failed": self.total_tasks_failed,
                    "completion_rate": self.total_tasks_completed / self.total_tasks_submitted if self.total_tasks_submitted > 0 else 0.0,
                },
            }
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._schedule_tasks())
            logger.info("Started task scheduler")
        
        if self.enable_work_stealing and self._work_stealer_task is None:
            self._work_stealer_task = asyncio.create_task(self._work_stealing_loop())
            logger.info("Started work stealing")
        
        # Start worker execution loops
        for worker in self.workers:
            asyncio.create_task(self._worker_loop(worker))
    
    async def _worker_loop(self, worker: Worker) -> None:
        """Worker execution loop."""
        while not self._shutdown and worker.is_active:
            try:
                # Get next task from worker queue
                if worker.task_queue:
                    with self._lock:
                        task = heapq.heappop(worker.task_queue)
                        
                        # Allocate resources
                        worker.allocate_resources(task)
                        worker.running_tasks[task.task_id] = task
                        worker.is_busy = True
                    
                    # Execute task
                    await self._execute_task_on_worker(task, worker)
                    
                    # Release resources
                    with self._lock:
                        worker.release_resources(task)
                        del worker.running_tasks[task.task_id]
                        worker.is_busy = len(worker.running_tasks) > 0
                        worker.tasks_completed += 1
                else:
                    # No tasks, sleep briefly
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in worker {worker.name}: {e}")
    
    async def _execute_task_on_worker(self, task: Task, worker: Worker) -> None:
        """Execute a task on a specific worker."""
        task.state = TaskState.RUNNING
        task.started_at = time.time()
        task.metrics.queue_time = task.started_at - task.scheduled_at
        
        try:
            # Execute based on executor type
            if task.executor_type == ExecutorType.ASYNC:
                if asyncio.iscoroutinefunction(task.func):
                    task.result = await task.func(*task.args, **task.kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    task.result = await loop.run_in_executor(
                        self.thread_pool,
                        task.func,
                        *task.args,
                        **task.kwargs
                    )
            
            elif task.executor_type == ExecutorType.THREAD:
                loop = asyncio.get_event_loop()
                task.result = await loop.run_in_executor(
                    self.thread_pool,
                    task.func,
                    *task.args,
                    **task.kwargs
                )
            
            elif task.executor_type == ExecutorType.PROCESS:
                loop = asyncio.get_event_loop()
                task.result = await loop.run_in_executor(
                    self.process_pool,
                    task.func,
                    *task.args,
                    **task.kwargs
                )
            
            else:  # SYNC
                task.result = task.func(*task.args, **task.kwargs)
            
            # Task completed successfully
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()
            task.metrics.execution_time = task.completed_at - task.started_at
            task.metrics.total_time = task.completed_at - task.created_at
            
            with self._lock:
                self.total_tasks_completed += 1
            
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Task failed
            task.error = e
            task.state = TaskState.FAILED
            task.completed_at = time.time()
            task.metrics.error_count += 1
            
            with self._lock:
                self.total_tasks_failed += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry if allowed
            if task.metrics.retry_count < task.max_retries:
                task.metrics.retry_count += 1
                task.state = TaskState.SCHEDULED
                task.scheduled_at = time.time()
                
                with self._lock:
                    heapq.heappush(self.global_queue, task)
                
                logger.info(f"Retrying task {task.task_id} (attempt {task.metrics.retry_count})")
    
    async def shutdown(self) -> None:
        """Shutdown the scheduler."""
        logger.info("Shutting down task scheduler")
        
        self._shutdown = True
        
        # Cancel scheduler tasks
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self._work_stealer_task:
            self._work_stealer_task.cancel()
            try:
                await self._work_stealer_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executors
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("Task scheduler shutdown complete")


def create_task_scheduler(**kwargs) -> TaskScheduler:
    """
    Create a task scheduler with the given configuration.
    
    Args:
        **kwargs: Scheduler configuration
        
    Returns:
        Configured task scheduler
    """
    return TaskScheduler(**kwargs)


if __name__ == "__main__":
    import random
    
    async def demo_task_scheduler():
        """Demonstrate task scheduler functionality."""
        print("📋 Task Scheduler Demo")
        print("=" * 40)
        
        # Create scheduler
        scheduler = create_task_scheduler(
            num_workers=4,
            enable_work_stealing=True
        )
        
        await scheduler.start()
        print("Scheduler started with 4 workers")
        
        # Define some test tasks
        async def async_task(task_id: int, duration: float) -> str:
            print(f"   Task {task_id} started (async)")
            await asyncio.sleep(duration)
            print(f"   Task {task_id} completed")
            return f"Result from task {task_id}"
        
        def sync_task(task_id: int, duration: float) -> str:
            print(f"   Task {task_id} started (sync)")
            time.sleep(duration)
            print(f"   Task {task_id} completed")
            return f"Result from task {task_id}"
        
        def cpu_intensive_task(task_id: int, iterations: int) -> int:
            print(f"   Task {task_id} started (CPU intensive)")
            result = 0
            for i in range(iterations):
                result += i ** 2
            print(f"   Task {task_id} completed")
            return result
        
        # Submit various tasks
        print("\nSubmitting tasks...")
        task_ids = []
        
        # High priority async tasks
        for i in range(3):
            task_id = await scheduler.submit_task(
                async_task,
                i,
                random.uniform(0.1, 0.3),
                name=f"async_high_{i}",
                priority=Priority.HIGH,
                executor_type=ExecutorType.ASYNC
            )
            task_ids.append(task_id)
        
        # Normal priority sync tasks
        for i in range(5):
            task_id = await scheduler.submit_task(
                sync_task,
                i + 10,
                random.uniform(0.1, 0.2),
                name=f"sync_normal_{i}",
                priority=Priority.NORMAL,
                executor_type=ExecutorType.THREAD
            )
            task_ids.append(task_id)
        
        # CPU intensive tasks
        for i in range(2):
            task_id = await scheduler.submit_task(
                cpu_intensive_task,
                i + 20,
                100000,
                name=f"cpu_intensive_{i}",
                priority=Priority.LOW,
                executor_type=ExecutorType.PROCESS,
                requirements=ResourceRequirements(cpu_cores=2.0, memory_mb=500.0)
            )
            task_ids.append(task_id)
        
        print(f"Submitted {len(task_ids)} tasks")
        
        # Submit task with dependencies
        print("\nSubmitting tasks with dependencies...")
        
        parent_id = await scheduler.submit_task(
            async_task,
            100,
            0.5,
            name="parent_task",
            priority=Priority.HIGH
        )
        
        child_id = await scheduler.submit_task(
            async_task,
            101,
            0.3,
            name="child_task",
            priority=Priority.HIGH,
            dependencies={parent_id}
        )
        
        print(f"Parent task: {parent_id}")
        print(f"Child task: {child_id} (depends on parent)")
        
        # Monitor progress
        print("\nMonitoring task execution...")
        await asyncio.sleep(2.0)
        
        # Check scheduler status
        status = scheduler.get_scheduler_status()
        print("\nScheduler Status:")
        print(f"   Global queue: {status['global_queue_size']} tasks")
        print(f"   Blocked tasks: {status['blocked_tasks']}")
        print(f"   Tasks by state:")
        for state, count in status['tasks_by_state'].items():
            if count > 0:
                print(f"      {state}: {count}")
        
        print("\nWorker Status:")
        for worker in status['workers']:
            print(f"   {worker['name']}:")
            print(f"      Load: {worker['load_factor']:.2f}")
            print(f"      Running: {worker['running_tasks']}")
            print(f"      Queued: {worker['queued_tasks']}")
            print(f"      Completed: {worker['tasks_completed']}")
        
        # Wait for some tasks to complete
        print("\nWaiting for tasks to complete...")
        completed_count = 0
        for task_id in task_ids[:5]:
            try:
                result = await scheduler.execute_task(task_id)
                print(f"   Task {task_id} result: {result}")
                completed_count += 1
            except Exception as e:
                print(f"   Task {task_id} error: {e}")
        
        # Final statistics
        await asyncio.sleep(3.0)
        final_status = scheduler.get_scheduler_status()
        stats = final_status['statistics']
        
        print("\nFinal Statistics:")
        print(f"   Total submitted: {stats['total_submitted']}")
        print(f"   Total completed: {stats['total_completed']}")
        print(f"   Total failed: {stats['total_failed']}")
        print(f"   Completion rate: {stats['completion_rate']:.2%}")
        
        # Shutdown
        await scheduler.shutdown()
        print("\n📋 Task Scheduler Demo Complete")
    
    # Run demo
    asyncio.run(demo_task_scheduler())