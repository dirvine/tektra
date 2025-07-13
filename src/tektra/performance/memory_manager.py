#!/usr/bin/env python3
"""
Memory Management System

Provides efficient memory allocation, pooling, and optimization including:
- Memory pool management
- Zero-copy operations
- Memory mapping
- Garbage collection optimization
- Memory pressure monitoring

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,loguru,numpy python memory_manager.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
#     "numpy>=1.24.0",
# ]
# ///

import gc
import mmap
import os
import sys
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import threading

import numpy as np
import psutil
from loguru import logger


class MemoryPressure(Enum):
    """System memory pressure levels."""
    LOW = "low"          # < 50% usage
    MODERATE = "moderate" # 50-70% usage
    HIGH = "high"        # 70-85% usage
    CRITICAL = "critical" # > 85% usage


@dataclass
class MemoryBlock:
    """Represents a block of managed memory."""
    
    block_id: str
    size_bytes: int
    data: Union[bytes, bytearray, memoryview, np.ndarray]
    
    allocated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    is_pinned: bool = False  # Pinned blocks won't be evicted
    is_shared: bool = False  # Shared across processes
    
    owner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPool:
    """Pool of pre-allocated memory blocks."""
    
    name: str
    block_size: int
    max_blocks: int
    
    free_blocks: deque = field(default_factory=deque)
    used_blocks: Dict[str, MemoryBlock] = field(default_factory=dict)
    
    total_allocated: int = 0
    peak_usage: int = 0
    
    allocations: int = 0
    deallocations: int = 0
    
    def __post_init__(self):
        """Pre-allocate initial blocks."""
        self._lock = threading.RLock()
        self._preallocate_blocks(min(self.max_blocks // 4, 10))
    
    def _preallocate_blocks(self, count: int) -> None:
        """Pre-allocate memory blocks."""
        for _ in range(count):
            if self.total_allocated >= self.max_blocks:
                break
            
            try:
                # Allocate block
                if self.block_size < 1024 * 1024:  # < 1MB use bytes
                    data = bytearray(self.block_size)
                else:  # >= 1MB use numpy for efficiency
                    data = np.zeros(self.block_size, dtype=np.uint8)
                
                block = MemoryBlock(
                    block_id=f"{self.name}_{self.total_allocated}",
                    size_bytes=self.block_size,
                    data=data
                )
                
                self.free_blocks.append(block)
                self.total_allocated += 1
                
            except (MemoryError, Exception) as e:
                logger.warning(f"Failed to pre-allocate block in pool {self.name}: {e}")
                break
    
    def allocate(self, owner: str) -> Optional[MemoryBlock]:
        """Allocate a block from the pool."""
        with self._lock:
            # Try to get from free blocks
            if self.free_blocks:
                block = self.free_blocks.popleft()
                block.owner = owner
                block.last_accessed = time.time()
                block.access_count += 1
                
                self.used_blocks[block.block_id] = block
                self.allocations += 1
                
                # Update peak usage
                current_usage = len(self.used_blocks)
                self.peak_usage = max(self.peak_usage, current_usage)
                
                return block
            
            # Try to allocate new block if under limit
            if self.total_allocated < self.max_blocks:
                self._preallocate_blocks(1)
                if self.free_blocks:
                    return self.allocate(owner)
            
            return None
    
    def deallocate(self, block_id: str) -> bool:
        """Return a block to the pool."""
        with self._lock:
            if block_id not in self.used_blocks:
                return False
            
            block = self.used_blocks.pop(block_id)
            block.owner = None
            
            # Clear data for security
            if isinstance(block.data, (bytes, bytearray)):
                block.data[:] = b'\0' * len(block.data)
            elif isinstance(block.data, np.ndarray):
                block.data.fill(0)
            
            self.free_blocks.append(block)
            self.deallocations += 1
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "name": self.name,
                "block_size": self.block_size,
                "total_blocks": self.total_allocated,
                "used_blocks": len(self.used_blocks),
                "free_blocks": len(self.free_blocks),
                "max_blocks": self.max_blocks,
                "allocations": self.allocations,
                "deallocations": self.deallocations,
                "peak_usage": self.peak_usage,
                "memory_mb": (self.total_allocated * self.block_size) / (1024 * 1024)
            }


class MemoryManager:
    """
    Comprehensive memory management system.
    
    Features:
    - Multiple memory pools with different block sizes
    - Zero-copy operations
    - Memory mapping for large allocations
    - Automatic garbage collection
    - Memory pressure monitoring
    - Allocation tracking and profiling
    """
    
    def __init__(
        self,
        max_memory_mb: float = 1024.0,
        enable_pools: bool = True,
        enable_profiling: bool = True,
        gc_threshold_mb: float = 100.0
    ):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum memory to manage (MB)
            enable_pools: Enable memory pooling
            enable_profiling: Enable allocation profiling
            gc_threshold_mb: Threshold for triggering GC (MB)
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.enable_pools = enable_pools
        self.enable_profiling = enable_profiling
        self.gc_threshold_mb = gc_threshold_mb
        
        # Memory pools for different sizes
        self.pools: Dict[int, MemoryPool] = {}
        if enable_pools:
            self._initialize_pools()
        
        # Direct allocations (not from pools)
        self.direct_allocations: Dict[str, MemoryBlock] = {}
        
        # Memory-mapped files
        self.mmap_files: Dict[str, mmap.mmap] = {}
        
        # Tracking
        self.total_allocated = 0
        self.total_freed = 0
        self.allocation_count = 0
        self.last_gc_time = time.time()
        self.last_gc_freed = 0
        
        # Profiling
        self.allocation_profile: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_bytes": 0,
            "peak_bytes": 0,
            "current_bytes": 0
        })
        
        # Weak references for automatic cleanup
        self._weak_refs: Dict[str, weakref.ReferenceType] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # System memory monitoring
        self.memory_info = psutil.virtual_memory()
        self._last_memory_check = time.time()
        
        logger.info(f"Memory manager initialized with {max_memory_mb}MB limit")
    
    def _initialize_pools(self) -> None:
        """Initialize memory pools for common sizes."""
        # Define pool configurations (size -> max_blocks)
        pool_configs = [
            (1024, 1000),           # 1KB blocks
            (4096, 500),            # 4KB blocks
            (16384, 200),           # 16KB blocks
            (65536, 100),           # 64KB blocks
            (262144, 50),           # 256KB blocks
            (1048576, 20),          # 1MB blocks
            (4194304, 10),          # 4MB blocks
        ]
        
        for block_size, max_blocks in pool_configs:
            pool_name = self._size_to_pool_name(block_size)
            self.pools[block_size] = MemoryPool(
                name=pool_name,
                block_size=block_size,
                max_blocks=max_blocks
            )
            logger.debug(f"Created memory pool: {pool_name}")
    
    def _size_to_pool_name(self, size: int) -> str:
        """Convert size to human-readable pool name."""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size // 1024}KB"
        else:
            return f"{size // (1024 * 1024)}MB"
    
    def allocate(
        self,
        size: int,
        owner: str = "unknown",
        use_pool: bool = True,
        zero_init: bool = True
    ) -> Tuple[str, Union[bytes, bytearray, memoryview, np.ndarray]]:
        """
        Allocate memory block.
        
        Args:
            size: Size in bytes
            owner: Owner identifier
            use_pool: Try to use memory pool
            zero_init: Initialize memory to zero
            
        Returns:
            Tuple of (allocation_id, memory_data)
        """
        with self._lock:
            # Check memory pressure
            if self._should_trigger_gc():
                self._run_gc()
            
            # Check if we're over limit
            if self.total_allocated + size > self.max_memory_bytes:
                self._try_free_memory(size)
                
                if self.total_allocated + size > self.max_memory_bytes:
                    raise MemoryError(f"Cannot allocate {size} bytes: would exceed limit")
            
            # Try pool allocation first
            if use_pool and self.enable_pools:
                pool_size = self._find_pool_size(size)
                if pool_size and pool_size in self.pools:
                    pool = self.pools[pool_size]
                    block = pool.allocate(owner)
                    
                    if block:
                        self.total_allocated += block.size_bytes
                        self.allocation_count += 1
                        
                        # Profile
                        if self.enable_profiling:
                            self._update_profile(owner, block.size_bytes, True)
                        
                        return block.block_id, block.data
            
            # Direct allocation
            allocation_id = f"direct_{self.allocation_count}"
            
            try:
                # Allocate memory
                if size < 1024 * 1024:  # < 1MB
                    if zero_init:
                        data = bytearray(size)
                    else:
                        data = bytearray(os.urandom(size))
                else:  # >= 1MB use numpy
                    if zero_init:
                        data = np.zeros(size, dtype=np.uint8)
                    else:
                        data = np.empty(size, dtype=np.uint8)
                
                block = MemoryBlock(
                    block_id=allocation_id,
                    size_bytes=size,
                    data=data,
                    owner=owner
                )
                
                self.direct_allocations[allocation_id] = block
                self.total_allocated += size
                self.allocation_count += 1
                
                # Profile
                if self.enable_profiling:
                    self._update_profile(owner, size, True)
                
                logger.debug(f"Allocated {size} bytes for {owner}")
                return allocation_id, data
                
            except (MemoryError, Exception) as e:
                logger.error(f"Failed to allocate {size} bytes: {e}")
                raise
    
    def allocate_shared(
        self,
        size: int,
        name: str,
        owner: str = "unknown"
    ) -> Tuple[str, memoryview]:
        """
        Allocate shared memory using memory mapping.
        
        Args:
            size: Size in bytes
            name: Name for the shared memory
            owner: Owner identifier
            
        Returns:
            Tuple of (allocation_id, memory_view)
        """
        with self._lock:
            # Create temporary file for mmap
            temp_file = Path(f"/tmp/tektra_mmap_{name}")
            
            try:
                # Create and resize file
                with open(temp_file, 'wb') as f:
                    f.seek(size - 1)
                    f.write(b'\0')
                
                # Create memory map
                with open(temp_file, 'r+b') as f:
                    mmap_obj = mmap.mmap(f.fileno(), size)
                
                self.mmap_files[name] = mmap_obj
                
                # Create block for tracking
                block = MemoryBlock(
                    block_id=f"mmap_{name}",
                    size_bytes=size,
                    data=memoryview(mmap_obj),
                    owner=owner,
                    is_shared=True
                )
                
                self.direct_allocations[block.block_id] = block
                self.total_allocated += size
                
                logger.info(f"Allocated {size} bytes of shared memory: {name}")
                return block.block_id, block.data
                
            except Exception as e:
                logger.error(f"Failed to allocate shared memory: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                raise
    
    def deallocate(self, allocation_id: str) -> bool:
        """
        Deallocate memory block.
        
        Args:
            allocation_id: ID of the allocation
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Check pools first
            for pool in self.pools.values():
                if allocation_id in pool.used_blocks:
                    block = pool.used_blocks[allocation_id]
                    size = block.size_bytes
                    owner = block.owner or "unknown"
                    
                    if pool.deallocate(allocation_id):
                        self.total_allocated -= size
                        self.total_freed += size
                        
                        # Profile
                        if self.enable_profiling:
                            self._update_profile(owner, size, False)
                        
                        return True
                    return False
            
            # Check direct allocations
            if allocation_id in self.direct_allocations:
                block = self.direct_allocations.pop(allocation_id)
                size = block.size_bytes
                owner = block.owner or "unknown"
                
                # Clean up mmap if needed
                if block.is_shared and allocation_id.startswith("mmap_"):
                    name = allocation_id[5:]  # Remove "mmap_" prefix
                    if name in self.mmap_files:
                        self.mmap_files[name].close()
                        del self.mmap_files[name]
                        
                        # Remove temp file
                        temp_file = Path(f"/tmp/tektra_mmap_{name}")
                        if temp_file.exists():
                            temp_file.unlink()
                
                self.total_allocated -= size
                self.total_freed += size
                
                # Profile
                if self.enable_profiling:
                    self._update_profile(owner, size, False)
                
                logger.debug(f"Deallocated {size} bytes from {owner}")
                return True
            
            return False
    
    def create_zero_copy_view(
        self,
        allocation_id: str,
        offset: int = 0,
        size: Optional[int] = None
    ) -> Optional[memoryview]:
        """
        Create a zero-copy view of allocated memory.
        
        Args:
            allocation_id: ID of the allocation
            offset: Offset in bytes
            size: Size of view (None for remaining)
            
        Returns:
            Memory view or None if not found
        """
        with self._lock:
            # Find the block
            block = None
            
            # Check pools
            for pool in self.pools.values():
                if allocation_id in pool.used_blocks:
                    block = pool.used_blocks[allocation_id]
                    break
            
            # Check direct allocations
            if not block and allocation_id in self.direct_allocations:
                block = self.direct_allocations[allocation_id]
            
            if not block:
                return None
            
            # Update access tracking
            block.last_accessed = time.time()
            block.access_count += 1
            
            # Create view
            if isinstance(block.data, memoryview):
                data = block.data
            else:
                data = memoryview(block.data)
            
            if size is None:
                size = len(data) - offset
            
            return data[offset:offset + size]
    
    def _find_pool_size(self, requested_size: int) -> Optional[int]:
        """Find appropriate pool size for requested allocation."""
        for pool_size in sorted(self.pools.keys()):
            if pool_size >= requested_size:
                # Don't use pool if requested size is less than 50% of pool size
                if requested_size >= pool_size * 0.5:
                    return pool_size
        return None
    
    def _should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        # Time-based check
        if time.time() - self.last_gc_time < 10.0:  # Min 10 seconds between GCs
            return False
        
        # Memory-based check
        if self.total_allocated > self.gc_threshold_mb * 1024 * 1024:
            return True
        
        # System memory pressure check
        if self._get_memory_pressure() in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            return True
        
        return False
    
    def _run_gc(self) -> None:
        """Run garbage collection."""
        logger.debug("Running garbage collection")
        
        start_allocated = self.total_allocated
        
        # Python garbage collection
        gc.collect()
        
        # Clean up weak references
        dead_refs = []
        for ref_id, weak_ref in self._weak_refs.items():
            if weak_ref() is None:
                dead_refs.append(ref_id)
        
        for ref_id in dead_refs:
            del self._weak_refs[ref_id]
            # Try to deallocate if still tracked
            self.deallocate(ref_id)
        
        # Record results
        self.last_gc_time = time.time()
        self.last_gc_freed = start_allocated - self.total_allocated
        
        if self.last_gc_freed > 0:
            logger.info(f"GC freed {self.last_gc_freed / (1024 * 1024):.1f}MB")
    
    def _try_free_memory(self, needed_bytes: int) -> None:
        """Try to free memory to accommodate allocation."""
        logger.debug(f"Trying to free {needed_bytes} bytes")
        
        # First run GC
        self._run_gc()
        
        # If still not enough, evict unpinned allocations
        if self.total_allocated + needed_bytes > self.max_memory_bytes:
            # Collect all unpinned allocations
            candidates = []
            
            for pool in self.pools.values():
                for block_id, block in pool.used_blocks.items():
                    if not block.is_pinned:
                        candidates.append((block.last_accessed, block_id, "pool"))
            
            for block_id, block in self.direct_allocations.items():
                if not block.is_pinned:
                    candidates.append((block.last_accessed, block_id, "direct"))
            
            # Sort by last accessed (oldest first)
            candidates.sort()
            
            # Evict until we have enough space
            for _, block_id, alloc_type in candidates:
                if self.total_allocated + needed_bytes <= self.max_memory_bytes:
                    break
                
                logger.debug(f"Evicting {block_id} to free memory")
                self.deallocate(block_id)
    
    def _get_memory_pressure(self) -> MemoryPressure:
        """Get current system memory pressure."""
        # Cache for 1 second
        if time.time() - self._last_memory_check < 1.0:
            percent = self.memory_info.percent
        else:
            self.memory_info = psutil.virtual_memory()
            self._last_memory_check = time.time()
            percent = self.memory_info.percent
        
        if percent < 50:
            return MemoryPressure.LOW
        elif percent < 70:
            return MemoryPressure.MODERATE
        elif percent < 85:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL
    
    def _update_profile(self, owner: str, size: int, is_allocation: bool) -> None:
        """Update allocation profile."""
        profile = self.allocation_profile[owner]
        
        if is_allocation:
            profile["count"] += 1
            profile["total_bytes"] += size
            profile["current_bytes"] += size
            profile["peak_bytes"] = max(profile["peak_bytes"], profile["current_bytes"])
        else:
            profile["current_bytes"] = max(0, profile["current_bytes"] - size)
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory manager status."""
        with self._lock:
            status = {
                "total_allocated_mb": self.total_allocated / (1024 * 1024),
                "total_freed_mb": self.total_freed / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "usage_percent": (self.total_allocated / self.max_memory_bytes) * 100,
                "allocation_count": self.allocation_count,
                "direct_allocations": len(self.direct_allocations),
                "memory_pressure": self._get_memory_pressure().value,
                "system_memory": {
                    "total_mb": self.memory_info.total / (1024 * 1024),
                    "available_mb": self.memory_info.available / (1024 * 1024),
                    "percent": self.memory_info.percent
                },
                "gc_stats": {
                    "last_gc_time": self.last_gc_time,
                    "last_gc_freed_mb": self.last_gc_freed / (1024 * 1024)
                },
                "pools": {}
            }
            
            # Add pool statistics
            for size, pool in self.pools.items():
                status["pools"][self._size_to_pool_name(size)] = pool.get_stats()
            
            # Add top allocators if profiling enabled
            if self.enable_profiling:
                top_allocators = sorted(
                    self.allocation_profile.items(),
                    key=lambda x: x[1]["current_bytes"],
                    reverse=True
                )[:5]
                
                status["top_allocators"] = {
                    owner: {
                        "current_mb": stats["current_bytes"] / (1024 * 1024),
                        "peak_mb": stats["peak_bytes"] / (1024 * 1024),
                        "allocations": stats["count"]
                    }
                    for owner, stats in top_allocators
                }
            
            return status
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        with self._lock:
            initial_allocated = self.total_allocated
            
            # Run GC
            self._run_gc()
            
            # Compact pools
            compacted_blocks = 0
            for pool in self.pools.values():
                # Move blocks to minimize fragmentation
                if len(pool.free_blocks) > pool.max_blocks // 2:
                    # Too many free blocks, release some
                    to_release = len(pool.free_blocks) - pool.max_blocks // 4
                    for _ in range(to_release):
                        if pool.free_blocks:
                            pool.free_blocks.popleft()
                            pool.total_allocated -= 1
                            compacted_blocks += 1
            
            # System-level optimization
            if sys.platform == "linux":
                try:
                    # Advise kernel to free cached memory
                    os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
                except Exception:
                    pass
            
            return {
                "initial_allocated_mb": initial_allocated / (1024 * 1024),
                "final_allocated_mb": self.total_allocated / (1024 * 1024),
                "freed_mb": (initial_allocated - self.total_allocated) / (1024 * 1024),
                "compacted_blocks": compacted_blocks,
                "gc_collections": gc.get_count()
            }


def create_memory_manager(**kwargs) -> MemoryManager:
    """
    Create a memory manager with the given configuration.
    
    Args:
        **kwargs: Memory manager configuration
        
    Returns:
        Configured memory manager
    """
    return MemoryManager(**kwargs)


if __name__ == "__main__":
    def demo_memory_manager():
        """Demonstrate memory manager functionality."""
        print("💾 Memory Manager Demo")
        print("=" * 40)
        
        # Create memory manager
        manager = create_memory_manager(
            max_memory_mb=100.0,
            enable_pools=True,
            enable_profiling=True
        )
        
        print("Memory manager initialized")
        
        # Test allocations
        allocations = []
        
        # Small allocations (should use pools)
        print("\nSmall allocations (using pools):")
        for i in range(5):
            size = 4096  # 4KB
            alloc_id, data = manager.allocate(size, owner=f"test_{i}")
            allocations.append(alloc_id)
            print(f"   Allocated {size} bytes: {alloc_id}")
        
        # Large allocation (direct)
        print("\nLarge allocation (direct):")
        large_size = 5 * 1024 * 1024  # 5MB
        large_id, large_data = manager.allocate(large_size, owner="large_test")
        allocations.append(large_id)
        print(f"   Allocated {large_size} bytes: {large_id}")
        
        # Shared memory allocation
        print("\nShared memory allocation:")
        shared_id, shared_view = manager.allocate_shared(
            1024 * 1024,  # 1MB
            name="shared_test",
            owner="shared_owner"
        )
        print(f"   Allocated shared memory: {shared_id}")
        
        # Check status
        print("\nMemory status:")
        status = manager.get_status()
        print(f"   Total allocated: {status['total_allocated_mb']:.2f}MB")
        print(f"   Usage: {status['usage_percent']:.1f}%")
        print(f"   Memory pressure: {status['memory_pressure']}")
        
        # Pool statistics
        print("\nPool statistics:")
        for pool_name, pool_stats in status['pools'].items():
            if pool_stats['used_blocks'] > 0:
                print(f"   {pool_name}: {pool_stats['used_blocks']}/{pool_stats['total_blocks']} blocks used")
        
        # Create zero-copy view
        print("\nZero-copy view:")
        if allocations:
            view = manager.create_zero_copy_view(allocations[0], offset=0, size=100)
            if view:
                print(f"   Created view of {len(view)} bytes")
        
        # Test deallocation
        print("\nDeallocating:")
        for alloc_id in allocations[:3]:
            if manager.deallocate(alloc_id):
                print(f"   Deallocated: {alloc_id}")
        
        # Optimize memory
        print("\nOptimizing memory:")
        optimization = manager.optimize()
        print(f"   Freed: {optimization['freed_mb']:.2f}MB")
        print(f"   Compacted blocks: {optimization['compacted_blocks']}")
        
        # Final status
        print("\nFinal status:")
        final_status = manager.get_status()
        print(f"   Total allocated: {final_status['total_allocated_mb']:.2f}MB")
        
        if final_status.get('top_allocators'):
            print("\nTop allocators:")
            for owner, stats in final_status['top_allocators'].items():
                print(f"   {owner}: {stats['current_mb']:.2f}MB current, {stats['peak_mb']:.2f}MB peak")
        
        print("\n💾 Memory Manager Demo Complete")
    
    # Run demo
    demo_memory_manager()