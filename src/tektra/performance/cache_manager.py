#!/usr/bin/env python3
"""
Advanced Cache Management System

Multi-level caching with intelligent eviction policies, compression,
and persistence support for optimal performance.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with redis,lz4,msgpack,psutil,loguru python cache_manager.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "redis>=5.0.0",
#     "lz4>=4.3.0",
#     "msgpack>=1.0.0",
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
# ]
# ///

import asyncio
import gc
import hashlib
import json
import pickle
import time
import weakref
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
import zlib

import lz4.frame
import msgpack
import psutil
import redis
from loguru import logger


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"      # In-process memory cache
    L2_SHARED = "l2_shared"       # Shared memory cache
    L3_DISK = "l3_disk"           # Disk-based cache
    L4_REMOTE = "l4_remote"       # Remote cache (Redis)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    FIFO = "fifo"                 # First In First Out
    ADAPTIVE = "adaptive"         # Adaptive based on access patterns
    TTL = "ttl"                   # Time To Live based


class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    AUTO = "auto"                 # Choose based on data characteristics


@dataclass
class CacheEntry:
    """Represents a cached item."""
    
    key: str
    value: Any
    size_bytes: int
    
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    ttl_seconds: Optional[float] = None
    expires_at: Optional[float] = None
    
    compression: CompressionType = CompressionType.NONE
    compressed_size: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate expiration time if TTL is set."""
        if self.ttl_seconds:
            self.expires_at = self.created_at + self.ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def access(self) -> None:
        """Record an access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


@dataclass
class CacheStats:
    """Cache statistics and metrics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    
    bytes_written: int = 0
    bytes_read: int = 0
    
    compression_ratio: float = 1.0
    
    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1
    
    def record_eviction(self) -> None:
        """Record an eviction."""
        self.evictions += 1
    
    def record_expiration(self) -> None:
        """Record an expiration."""
        self.expirations += 1
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int):
        """Initialize LRU cache."""
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry = self.cache[key]
            entry.access()
            
            return entry
    
    def put(self, entry: CacheEntry) -> Optional[str]:
        """Put item in cache, returns evicted key if any."""
        with self._lock:
            evicted_key = None
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and entry.key not in self.cache:
                # Evict least recently used (first item)
                evicted_key, _ = self.cache.popitem(last=False)
            
            # Add or update entry
            self.cache[entry.key] = entry
            self.cache.move_to_end(entry.key)
            
            return evicted_key
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
    
    def items(self) -> List[Tuple[str, CacheEntry]]:
        """Get all cache items."""
        with self._lock:
            return list(self.cache.items())


class LFUCache:
    """Least Frequently Used cache implementation."""
    
    def __init__(self, max_size: int):
        """Initialize LFU cache."""
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.freq_map: defaultdict[int, OrderedDict[str, bool]] = defaultdict(OrderedDict)
        self.min_freq = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            self._update_frequency(entry)
            entry.access()
            
            return entry
    
    def _update_frequency(self, entry: CacheEntry) -> None:
        """Update frequency tracking for an entry."""
        freq = entry.access_count
        
        # Remove from current frequency bucket
        if freq > 0 and entry.key in self.freq_map[freq - 1]:
            del self.freq_map[freq - 1][entry.key]
            if not self.freq_map[freq - 1] and self.min_freq == freq - 1:
                self.min_freq = freq
        
        # Add to new frequency bucket
        self.freq_map[freq][entry.key] = True
    
    def put(self, entry: CacheEntry) -> Optional[str]:
        """Put item in cache, returns evicted key if any."""
        with self._lock:
            evicted_key = None
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and entry.key not in self.cache:
                # Evict least frequently used
                if self.freq_map[self.min_freq]:
                    evicted_key, _ = self.freq_map[self.min_freq].popitem(last=False)
                    del self.cache[evicted_key]
            
            # Add or update entry
            self.cache[entry.key] = entry
            self.freq_map[0][entry.key] = True
            self.min_freq = 0
            
            return evicted_key
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            freq = entry.access_count
            
            del self.cache[key]
            if key in self.freq_map[freq]:
                del self.freq_map[freq][key]
            
            return True
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.freq_map.clear()
            self.min_freq = 0
    
    def items(self) -> List[Tuple[str, CacheEntry]]:
        """Get all cache items."""
        with self._lock:
            return list(self.cache.items())


class CacheManager:
    """
    Advanced multi-level cache management system.
    
    Features:
    - Multi-level cache hierarchy (L1-L4)
    - Multiple eviction policies
    - Automatic compression
    - Persistence support
    - Cache warming and prefetching
    - Distributed caching with Redis
    """
    
    def __init__(
        self,
        l1_size_mb: float = 100.0,
        l2_size_mb: float = 500.0,
        l3_size_mb: float = 2000.0,
        l4_enabled: bool = False,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        compression: CompressionType = CompressionType.AUTO,
        cache_dir: Optional[Path] = None,
        redis_url: Optional[str] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            l1_size_mb: L1 memory cache size in MB
            l2_size_mb: L2 shared memory cache size in MB
            l3_size_mb: L3 disk cache size in MB
            l4_enabled: Enable L4 remote cache
            eviction_policy: Default eviction policy
            compression: Compression type
            cache_dir: Directory for disk cache
            redis_url: Redis connection URL
        """
        self.l1_size_bytes = int(l1_size_mb * 1024 * 1024)
        self.l2_size_bytes = int(l2_size_mb * 1024 * 1024)
        self.l3_size_bytes = int(l3_size_mb * 1024 * 1024)
        self.l4_enabled = l4_enabled
        self.eviction_policy = eviction_policy
        self.compression = compression
        self.cache_dir = cache_dir or Path.home() / ".cache" / "tektra"
        
        # Initialize cache levels
        self._init_caches()
        
        # Redis connection for L4
        self.redis_client: Optional[redis.Redis] = None
        if l4_enabled and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for L4 cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.l4_enabled = False
        
        # Statistics per level
        self.stats: Dict[CacheLevel, CacheStats] = {
            level: CacheStats() for level in CacheLevel
        }
        
        # Weak references for automatic cleanup
        self._weak_refs: Dict[str, weakref.ReferenceType] = {}
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Cache manager initialized with {len(CacheLevel)} levels")
    
    def _init_caches(self) -> None:
        """Initialize cache storage for each level."""
        # L1: In-memory cache
        if self.eviction_policy == EvictionPolicy.LRU:
            self.l1_cache = LRUCache(max_size=1000)  # Entries, not bytes for now
        elif self.eviction_policy == EvictionPolicy.LFU:
            self.l1_cache = LFUCache(max_size=1000)
        else:
            self.l1_cache = LRUCache(max_size=1000)  # Default to LRU
        
        # L2: Shared memory cache (simplified for now)
        self.l2_cache: Dict[str, bytes] = {}
        self.l2_size_used = 0
        
        # L3: Disk cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.l3_index: Dict[str, Dict[str, Any]] = {}
        self._load_l3_index()
    
    def _load_l3_index(self) -> None:
        """Load L3 disk cache index."""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.l3_index = json.load(f)
                logger.debug(f"Loaded L3 index with {len(self.l3_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load L3 index: {e}")
                self.l3_index = {}
    
    def _save_l3_index(self) -> None:
        """Save L3 disk cache index."""
        index_file = self.cache_dir / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.l3_index, f)
        except Exception as e:
            logger.error(f"Failed to save L3 index: {e}")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        try:
            # Try msgpack first (faster)
            return msgpack.packb(value)
        except:
            # Fall back to pickle
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        try:
            # Try msgpack first
            return msgpack.unpackb(data)
        except:
            # Fall back to pickle
            return pickle.loads(data)
    
    def _compress(self, data: bytes) -> Tuple[bytes, CompressionType]:
        """Compress data based on policy."""
        if self.compression == CompressionType.NONE:
            return data, CompressionType.NONE
        
        original_size = len(data)
        
        # Auto-select compression
        if self.compression == CompressionType.AUTO:
            if original_size < 1024:  # Don't compress small data
                return data, CompressionType.NONE
            
            # Try LZ4 first (faster)
            try:
                compressed = lz4.frame.compress(data)
                if len(compressed) < original_size * 0.9:  # 10% improvement
                    return compressed, CompressionType.LZ4
            except:
                pass
            
            # Try zlib
            compressed = zlib.compress(data, level=6)
            if len(compressed) < original_size * 0.9:
                return compressed, CompressionType.ZLIB
            
            return data, CompressionType.NONE
        
        # Specific compression
        elif self.compression == CompressionType.LZ4:
            return lz4.frame.compress(data), CompressionType.LZ4
        elif self.compression == CompressionType.ZLIB:
            return zlib.compress(data, level=6), CompressionType.ZLIB
        
        return data, CompressionType.NONE
    
    def _decompress(self, data: bytes, compression: CompressionType) -> bytes:
        """Decompress data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        elif compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        else:
            return data
    
    def _hash_key(self, key: str) -> str:
        """Hash key for consistent storage."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    async def get(
        self,
        key: str,
        default: Any = None,
        promote: bool = True
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            promote: Promote to higher cache levels if found in lower levels
            
        Returns:
            Cached value or default
        """
        with self._lock:
            # Check L1
            entry = self.l1_cache.get(key)
            if entry and not entry.is_expired():
                self.stats[CacheLevel.L1_MEMORY].record_hit()
                return entry.value
            elif entry:
                # Expired
                self.l1_cache.remove(key)
                self.stats[CacheLevel.L1_MEMORY].record_expiration()
            
            # Check L2
            if key in self.l2_cache:
                self.stats[CacheLevel.L2_SHARED].record_hit()
                try:
                    value = self._deserialize(self.l2_cache[key])
                    
                    # Promote to L1
                    if promote:
                        await self._set_l1(key, value)
                    
                    return value
                except Exception as e:
                    logger.error(f"Failed to deserialize L2 value: {e}")
                    del self.l2_cache[key]
            
            # Check L3 (disk)
            if key in self.l3_index:
                self.stats[CacheLevel.L3_DISK].record_hit()
                value = await self._get_l3(key)
                
                if value is not None and promote:
                    # Promote to higher levels
                    await self._set_l1(key, value)
                    await self._set_l2(key, value)
                
                return value if value is not None else default
            
            # Check L4 (Redis)
            if self.l4_enabled and self.redis_client:
                try:
                    data = self.redis_client.get(f"cache:{key}")
                    if data:
                        self.stats[CacheLevel.L4_REMOTE].record_hit()
                        value = self._deserialize(data)
                        
                        if promote:
                            # Promote to higher levels
                            await self._set_l1(key, value)
                            await self._set_l2(key, value)
                            await self._set_l3(key, value)
                        
                        return value
                except Exception as e:
                    logger.error(f"Redis get error: {e}")
            
            # Cache miss at all levels
            for level in CacheLevel:
                self.stats[level].record_miss()
            
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        write_through: bool = True
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            write_through: Write to all cache levels
        """
        with self._lock:
            # Always write to L1
            await self._set_l1(key, value, ttl)
            
            if write_through:
                # Write to other levels
                await self._set_l2(key, value, ttl)
                await self._set_l3(key, value, ttl)
                
                if self.l4_enabled and self.redis_client:
                    await self._set_l4(key, value, ttl)
    
    async def _set_l1(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in L1 cache."""
        try:
            # Estimate size (simplified)
            size = len(self._serialize(value))
            
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size,
                ttl_seconds=ttl
            )
            
            evicted = self.l1_cache.put(entry)
            if evicted:
                self.stats[CacheLevel.L1_MEMORY].record_eviction()
            
            self.stats[CacheLevel.L1_MEMORY].bytes_written += size
            
        except Exception as e:
            logger.error(f"Failed to set L1 cache: {e}")
    
    async def _set_l2(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in L2 cache."""
        try:
            data = self._serialize(value)
            
            # Simple size management
            if self.l2_size_used + len(data) > self.l2_size_bytes:
                # Evict oldest entries (simplified)
                to_remove = []
                for k in list(self.l2_cache.keys())[:10]:  # Remove 10 oldest
                    to_remove.append(k)
                
                for k in to_remove:
                    if k in self.l2_cache:
                        self.l2_size_used -= len(self.l2_cache[k])
                        del self.l2_cache[k]
                        self.stats[CacheLevel.L2_SHARED].record_eviction()
            
            self.l2_cache[key] = data
            self.l2_size_used += len(data)
            self.stats[CacheLevel.L2_SHARED].bytes_written += len(data)
            
        except Exception as e:
            logger.error(f"Failed to set L2 cache: {e}")
    
    async def _set_l3(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in L3 disk cache."""
        try:
            data = self._serialize(value)
            compressed_data, compression_type = self._compress(data)
            
            # Generate filename
            file_hash = self._hash_key(key)
            file_path = self.cache_dir / f"{file_hash}.cache"
            
            # Write to disk
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
            
            # Update index
            self.l3_index[key] = {
                "file": file_hash,
                "size": len(compressed_data),
                "compression": compression_type.value,
                "created": time.time(),
                "ttl": ttl
            }
            
            self._save_l3_index()
            self.stats[CacheLevel.L3_DISK].bytes_written += len(compressed_data)
            
        except Exception as e:
            logger.error(f"Failed to set L3 cache: {e}")
    
    async def _set_l4(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in L4 Redis cache."""
        try:
            data = self._serialize(value)
            redis_key = f"cache:{key}"
            
            if ttl:
                self.redis_client.setex(redis_key, int(ttl), data)
            else:
                self.redis_client.set(redis_key, data)
            
            self.stats[CacheLevel.L4_REMOTE].bytes_written += len(data)
            
        except Exception as e:
            logger.error(f"Failed to set L4 cache: {e}")
    
    async def _get_l3(self, key: str) -> Optional[Any]:
        """Get value from L3 disk cache."""
        try:
            info = self.l3_index.get(key)
            if not info:
                return None
            
            # Check TTL
            if info.get("ttl"):
                if time.time() > info["created"] + info["ttl"]:
                    # Expired
                    await self.delete(key)
                    self.stats[CacheLevel.L3_DISK].record_expiration()
                    return None
            
            # Read from disk
            file_path = self.cache_dir / f"{info['file']}.cache"
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress and deserialize
            compression = CompressionType(info.get("compression", "none"))
            data = self._decompress(compressed_data, compression)
            value = self._deserialize(data)
            
            self.stats[CacheLevel.L3_DISK].bytes_read += len(compressed_data)
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get L3 cache: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from all cache levels.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted from any level
        """
        deleted = False
        
        with self._lock:
            # Delete from L1
            if self.l1_cache.remove(key):
                deleted = True
            
            # Delete from L2
            if key in self.l2_cache:
                self.l2_size_used -= len(self.l2_cache[key])
                del self.l2_cache[key]
                deleted = True
            
            # Delete from L3
            if key in self.l3_index:
                info = self.l3_index[key]
                file_path = self.cache_dir / f"{info['file']}.cache"
                if file_path.exists():
                    file_path.unlink()
                del self.l3_index[key]
                self._save_l3_index()
                deleted = True
            
            # Delete from L4
            if self.l4_enabled and self.redis_client:
                try:
                    if self.redis_client.delete(f"cache:{key}"):
                        deleted = True
                except Exception as e:
                    logger.error(f"Redis delete error: {e}")
        
        return deleted
    
    async def clear(self, level: Optional[CacheLevel] = None) -> None:
        """
        Clear cache at specified level or all levels.
        
        Args:
            level: Specific cache level to clear, or None for all
        """
        with self._lock:
            if level is None or level == CacheLevel.L1_MEMORY:
                self.l1_cache.clear()
                logger.info("Cleared L1 cache")
            
            if level is None or level == CacheLevel.L2_SHARED:
                self.l2_cache.clear()
                self.l2_size_used = 0
                logger.info("Cleared L2 cache")
            
            if level is None or level == CacheLevel.L3_DISK:
                # Clear disk cache
                for info in self.l3_index.values():
                    file_path = self.cache_dir / f"{info['file']}.cache"
                    if file_path.exists():
                        file_path.unlink()
                self.l3_index.clear()
                self._save_l3_index()
                logger.info("Cleared L3 cache")
            
            if level is None or level == CacheLevel.L4_REMOTE:
                if self.l4_enabled and self.redis_client:
                    try:
                        # Clear all cache keys
                        for key in self.redis_client.scan_iter("cache:*"):
                            self.redis_client.delete(key)
                        logger.info("Cleared L4 cache")
                    except Exception as e:
                        logger.error(f"Failed to clear L4 cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = sum(s.hits for s in self.stats.values())
            total_misses = sum(s.misses for s in self.stats.values())
            
            return {
                "levels": {
                    level.value: {
                        "hits": self.stats[level].hits,
                        "misses": self.stats[level].misses,
                        "hit_rate": self.stats[level].hit_rate,
                        "evictions": self.stats[level].evictions,
                        "expirations": self.stats[level].expirations,
                        "bytes_written": self.stats[level].bytes_written,
                        "bytes_read": self.stats[level].bytes_read,
                    }
                    for level in CacheLevel
                },
                "total": {
                    "hits": total_hits,
                    "misses": total_misses,
                    "hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
                },
                "sizes": {
                    "l1_entries": len(self.l1_cache.cache) if hasattr(self.l1_cache, 'cache') else 0,
                    "l2_bytes": self.l2_size_used,
                    "l3_entries": len(self.l3_index),
                }
            }
    
    async def prefetch(self, keys: List[str]) -> None:
        """
        Prefetch multiple keys into cache.
        
        Args:
            keys: List of keys to prefetch
        """
        tasks = [self.get(key) for key in keys]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def warm_cache(self, data: Dict[str, Any]) -> None:
        """
        Warm cache with provided data.
        
        Args:
            data: Dictionary of key-value pairs to cache
        """
        tasks = [self.set(key, value) for key, value in data.items()]
        await asyncio.gather(*tasks, return_exceptions=True)


def create_cache_manager(**kwargs) -> CacheManager:
    """
    Create a cache manager with the given configuration.
    
    Args:
        **kwargs: Cache manager configuration
        
    Returns:
        Configured cache manager
    """
    return CacheManager(**kwargs)


if __name__ == "__main__":
    async def demo_cache_manager():
        """Demonstrate cache manager functionality."""
        print("💾 Cache Manager Demo")
        print("=" * 40)
        
        # Create cache manager
        cache = create_cache_manager(
            l1_size_mb=10.0,
            l2_size_mb=50.0,
            l3_size_mb=200.0,
            l4_enabled=False,  # Redis disabled for demo
            eviction_policy=EvictionPolicy.LRU,
            compression=CompressionType.AUTO
        )
        
        print("Cache manager initialized")
        
        # Test basic operations
        test_data = {
            "user_123": {"name": "Alice", "age": 30, "data": "x" * 1000},
            "config_app": {"theme": "dark", "language": "en", "settings": {"a": 1, "b": 2}},
            "result_42": list(range(1000)),
        }
        
        # Write data
        print("\nWriting test data to cache...")
        for key, value in test_data.items():
            await cache.set(key, value, ttl=300)  # 5 minute TTL
            print(f"   Cached: {key}")
        
        # Read data (should hit L1)
        print("\nReading from cache (L1 hits expected)...")
        for key in test_data.keys():
            value = await cache.get(key)
            print(f"   Retrieved: {key} (found: {value is not None})")
        
        # Clear L1 to test L2/L3 promotion
        await cache.clear(CacheLevel.L1_MEMORY)
        print("\nCleared L1 cache")
        
        # Read again (should hit L2/L3 and promote)
        print("\nReading again (L2/L3 hits with promotion)...")
        for key in test_data.keys():
            value = await cache.get(key)
            print(f"   Retrieved: {key} (found: {value is not None})")
        
        # Test prefetching
        print("\nTesting prefetch...")
        new_keys = ["item_1", "item_2", "item_3"]
        for i, key in enumerate(new_keys):
            await cache.set(key, f"value_{i}")
        
        await cache.clear(CacheLevel.L1_MEMORY)
        await cache.prefetch(new_keys)
        print(f"   Prefetched {len(new_keys)} keys")
        
        # Test cache warming
        print("\nTesting cache warming...")
        warm_data = {f"warm_{i}": f"data_{i}" * 100 for i in range(5)}
        await cache.warm_cache(warm_data)
        print(f"   Warmed cache with {len(warm_data)} entries")
        
        # Show statistics
        print("\nCache Statistics:")
        stats = cache.get_stats()
        
        for level, level_stats in stats["levels"].items():
            print(f"\n   {level}:")
            print(f"      Hits: {level_stats['hits']}")
            print(f"      Misses: {level_stats['misses']}")
            print(f"      Hit rate: {level_stats['hit_rate']:.2%}")
            print(f"      Evictions: {level_stats['evictions']}")
        
        print(f"\n   Total hit rate: {stats['total']['hit_rate']:.2%}")
        
        # Test deletion
        print("\nTesting deletion...")
        deleted = await cache.delete("user_123")
        print(f"   Deleted user_123: {deleted}")
        
        value = await cache.get("user_123")
        print(f"   Retrieve after delete: {value is None}")
        
        print("\n💾 Cache Manager Demo Complete")
    
    # Run demo
    asyncio.run(demo_cache_manager())