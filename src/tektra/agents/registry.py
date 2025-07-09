"""
Agent Registry and Management

This module provides centralized management for all agents in the system:
- Agent registration and discovery
- State persistence
- Agent relationships and hierarchies
- Performance metrics and history
- Agent search and filtering
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import sqlite3
import aiosqlite

from loguru import logger

from .builder import AgentSpecification, AgentType, AgentCapability


def _serialize_agent_spec(spec: AgentSpecification) -> Dict[str, Any]:
    """Custom serialization for AgentSpecification that handles enums."""
    spec_dict = asdict(spec)
    
    # Convert enums to their string values
    if 'type' in spec_dict:
        spec_dict['type'] = spec_dict['type'].value if hasattr(spec_dict['type'], 'value') else spec_dict['type']
    
    if 'capabilities' in spec_dict:
        spec_dict['capabilities'] = [
            cap.value if hasattr(cap, 'value') else cap 
            for cap in spec_dict['capabilities']
        ]
    
    # Handle datetime objects
    if 'created_at' in spec_dict and hasattr(spec_dict['created_at'], 'isoformat'):
        spec_dict['created_at'] = spec_dict['created_at'].isoformat()
    
    return spec_dict


def _deserialize_agent_spec(spec_dict: Dict[str, Any]) -> AgentSpecification:
    """Custom deserialization for AgentSpecification that handles enums."""
    # Convert type back to enum
    if 'type' in spec_dict and isinstance(spec_dict['type'], str):
        spec_dict['type'] = AgentType(spec_dict['type'])
    
    # Convert capabilities back to enums
    if 'capabilities' in spec_dict:
        spec_dict['capabilities'] = [
            AgentCapability(cap) if isinstance(cap, str) else cap
            for cap in spec_dict['capabilities']
        ]
    
    # Handle datetime
    if 'created_at' in spec_dict and isinstance(spec_dict['created_at'], str):
        spec_dict['created_at'] = datetime.fromisoformat(spec_dict['created_at'])
    
    return AgentSpecification(**spec_dict)


class AgentStatus(Enum):
    """Overall status of an agent."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_runtime_seconds: float = 0.0
    average_runtime_seconds: float = 0.0
    last_execution_time: Optional[datetime] = None
    error_rate: float = 0.0
    
    # Resource usage
    peak_memory_mb: int = 0
    average_cpu_percent: float = 0.0
    
    # Communication
    messages_sent: int = 0
    messages_received: int = 0
    events_processed: int = 0


@dataclass
class AgentRecord:
    """Complete record of an agent in the registry."""
    specification: AgentSpecification
    status: AgentStatus
    metrics: AgentMetrics
    
    # Relationships
    parent_agents: List[str] = None
    child_agents: List[str] = None
    collaborating_agents: List[str] = None
    
    # History
    created_at: datetime = None
    last_modified: datetime = None
    version_history: List[Dict[str, Any]] = None
    
    # Tags and metadata
    tags: List[str] = None
    custom_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parent_agents is None:
            self.parent_agents = []
        if self.child_agents is None:
            self.child_agents = []
        if self.collaborating_agents is None:
            self.collaborating_agents = []
        if self.version_history is None:
            self.version_history = []
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_modified is None:
            self.last_modified = datetime.now()


class AgentRegistry:
    """
    Central registry for all agents in the Tektra system.
    
    Provides:
    - Agent CRUD operations
    - Search and discovery
    - Relationship management
    - Performance tracking
    - Persistence to database
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize registry with optional database path."""
        if db_path is None:
            db_path = Path.home() / '.tektra' / 'agents' / 'registry.db'
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.agents: Dict[str, AgentRecord] = {}
        self.agent_index: Dict[str, Set[str]] = {
            'by_type': {},
            'by_capability': {},
            'by_status': {},
            'by_tag': {}
        }
        
        # Initialize database
        asyncio.create_task(self._init_database())
        
        logger.info(f"Agent Registry initialized with database at {self.db_path}")
    
    async def _init_database(self):
        """Initialize SQLite database for persistence."""
        async with aiosqlite.connect(self.db_path) as db:
            # Create tables
            await db.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    specification TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    relationships TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    last_modified TIMESTAMP,
                    version INTEGER DEFAULT 1
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS agent_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    specification TEXT NOT NULL,
                    changed_at TIMESTAMP,
                    changed_by TEXT,
                    change_description TEXT,
                    FOREIGN KEY (agent_id) REFERENCES agents(id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS agent_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    success BOOLEAN,
                    output TEXT,
                    error TEXT,
                    metrics TEXT,
                    FOREIGN KEY (agent_id) REFERENCES agents(id)
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_status ON agents(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_created ON agents(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_execution_agent ON agent_executions(agent_id)")
            
            await db.commit()
            
        # Load existing agents
        await self._load_agents_from_db()
    
    async def register_agent(self, spec: AgentSpecification) -> str:
        """
        Register a new agent in the registry.
        
        Args:
            spec: Agent specification
            
        Returns:
            Agent ID
        """
        agent_id = spec.id
        
        # Check if already exists
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already exists")
        
        # Create agent record
        record = AgentRecord(
            specification=spec,
            status=AgentStatus.INACTIVE,
            metrics=AgentMetrics()
        )
        
        # Store in memory
        self.agents[agent_id] = record
        self._update_indexes(agent_id, record)
        
        # Persist to database
        await self._save_agent_to_db(record)
        
        logger.info(f"Registered agent {agent_id} ({spec.name})")
        return agent_id
    
    async def update_agent(
        self, 
        agent_id: str, 
        spec: Optional[AgentSpecification] = None,
        status: Optional[AgentStatus] = None,
        metrics: Optional[AgentMetrics] = None
    ):
        """Update agent information."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        record = self.agents[agent_id]
        
        # Track version history if spec changed
        if spec and spec != record.specification:
            version_entry = {
                'version': record.specification.version,
                'specification': asdict(record.specification),
                'changed_at': datetime.now().isoformat(),
                'changed_by': 'system'
            }
            record.version_history.append(version_entry)
            
            record.specification = spec
            record.specification.version += 1
        
        if status:
            record.status = status
        
        if metrics:
            record.metrics = metrics
        
        record.last_modified = datetime.now()
        
        # Update indexes
        self._update_indexes(agent_id, record)
        
        # Persist changes
        await self._save_agent_to_db(record)
        
        logger.debug(f"Updated agent {agent_id}")
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """Get agent record by ID."""
        return self.agents.get(agent_id)
    
    async def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AgentRecord]:
        """
        List agents with optional filtering.
        
        Args:
            status: Filter by status
            agent_type: Filter by agent type
            capabilities: Filter by required capabilities
            tags: Filter by tags
            limit: Maximum results
            offset: Results offset for pagination
            
        Returns:
            List of matching agent records
        """
        results = []
        
        # Start with all agents
        candidates = set(self.agents.keys())
        
        # Apply filters
        if status:
            status_agents = self.agent_index['by_status'].get(status.value, set())
            candidates &= status_agents
        
        if agent_type:
            type_agents = self.agent_index['by_type'].get(agent_type.value, set())
            candidates &= type_agents
        
        if capabilities:
            for cap in capabilities:
                cap_agents = self.agent_index['by_capability'].get(cap.value, set())
                candidates &= cap_agents
        
        if tags:
            for tag in tags:
                tag_agents = self.agent_index['by_tag'].get(tag, set())
                candidates &= tag_agents
        
        # Sort by creation date
        sorted_candidates = sorted(
            candidates,
            key=lambda aid: self.agents[aid].created_at,
            reverse=True
        )
        
        # Apply pagination
        for agent_id in sorted_candidates[offset:offset + limit]:
            results.append(self.agents[agent_id])
        
        return results
    
    async def search_agents(self, query: str) -> List[AgentRecord]:
        """
        Search agents by name or description.
        
        Args:
            query: Search query
            
        Returns:
            Matching agent records
        """
        query_lower = query.lower()
        results = []
        
        for agent_id, record in self.agents.items():
            spec = record.specification
            
            # Search in name and description
            if (query_lower in spec.name.lower() or 
                query_lower in spec.description.lower() or
                query_lower in spec.goal.lower()):
                results.append(record)
        
        return results
    
    async def add_agent_relationship(
        self,
        parent_id: str,
        child_id: str,
        relationship_type: str = "parent-child"
    ):
        """Add relationship between agents."""
        if parent_id not in self.agents or child_id not in self.agents:
            raise ValueError("One or both agents not found")
        
        parent_record = self.agents[parent_id]
        child_record = self.agents[child_id]
        
        if relationship_type == "parent-child":
            if child_id not in parent_record.child_agents:
                parent_record.child_agents.append(child_id)
            if parent_id not in child_record.parent_agents:
                child_record.parent_agents.append(parent_id)
        elif relationship_type == "collaboration":
            if child_id not in parent_record.collaborating_agents:
                parent_record.collaborating_agents.append(child_id)
            if parent_id not in child_record.collaborating_agents:
                child_record.collaborating_agents.append(parent_id)
        
        # Update both records
        await self._save_agent_to_db(parent_record)
        await self._save_agent_to_db(child_record)
    
    async def record_execution(
        self,
        agent_id: str,
        started_at: datetime,
        completed_at: datetime,
        success: bool,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Record an agent execution."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        record = self.agents[agent_id]
        agent_metrics = record.metrics
        
        # Update metrics
        agent_metrics.total_executions += 1
        if success:
            agent_metrics.successful_executions += 1
        else:
            agent_metrics.failed_executions += 1
        
        runtime = (completed_at - started_at).total_seconds()
        agent_metrics.total_runtime_seconds += runtime
        agent_metrics.average_runtime_seconds = (
            agent_metrics.total_runtime_seconds / agent_metrics.total_executions
        )
        agent_metrics.last_execution_time = completed_at
        agent_metrics.error_rate = (
            agent_metrics.failed_executions / agent_metrics.total_executions
        )
        
        # Update resource metrics if provided
        if metrics:
            if 'memory_mb' in metrics:
                agent_metrics.peak_memory_mb = max(
                    agent_metrics.peak_memory_mb,
                    metrics['memory_mb']
                )
            if 'cpu_percent' in metrics:
                # Running average
                prev_avg = agent_metrics.average_cpu_percent
                n = agent_metrics.total_executions
                agent_metrics.average_cpu_percent = (
                    (prev_avg * (n - 1) + metrics['cpu_percent']) / n
                )
        
        # Save execution to database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO agent_executions 
                (agent_id, started_at, completed_at, success, output, error, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id,
                started_at.isoformat(),
                completed_at.isoformat(),
                success,
                json.dumps(output) if output else None,
                error,
                json.dumps(metrics) if metrics else None
            ))
            await db.commit()
        
        # Update agent record
        await self.update_agent(agent_id, metrics=agent_metrics)
    
    async def get_agent_history(
        self, 
        agent_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get execution history for an agent."""
        history = []
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT started_at, completed_at, success, output, error, metrics
                FROM agent_executions
                WHERE agent_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (agent_id, limit)) as cursor:
                async for row in cursor:
                    history.append({
                        'started_at': row[0],
                        'completed_at': row[1],
                        'success': bool(row[2]),
                        'output': json.loads(row[3]) if row[3] else None,
                        'error': row[4],
                        'metrics': json.loads(row[5]) if row[5] else None
                    })
        
        return history
    
    async def archive_agent(self, agent_id: str):
        """Archive an agent (soft delete)."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        await self.update_agent(agent_id, status=AgentStatus.ARCHIVED)
        logger.info(f"Archived agent {agent_id}")
    
    async def delete_agent(self, agent_id: str):
        """Permanently delete an agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Remove from memory
        record = self.agents.pop(agent_id)
        self._remove_from_indexes(agent_id, record)
        
        # Remove from database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            await db.execute("DELETE FROM agent_history WHERE agent_id = ?", (agent_id,))
            await db.execute("DELETE FROM agent_executions WHERE agent_id = ?", (agent_id,))
            await db.commit()
        
        logger.info(f"Deleted agent {agent_id}")
    
    def _update_indexes(self, agent_id: str, record: AgentRecord):
        """Update internal indexes."""
        spec = record.specification
        
        # Remove from old indexes
        self._remove_from_indexes(agent_id, record)
        
        # Add to new indexes
        # By type
        type_key = spec.type.value
        if type_key not in self.agent_index['by_type']:
            self.agent_index['by_type'][type_key] = set()
        self.agent_index['by_type'][type_key].add(agent_id)
        
        # By capability
        for cap in spec.capabilities:
            cap_key = cap.value
            if cap_key not in self.agent_index['by_capability']:
                self.agent_index['by_capability'][cap_key] = set()
            self.agent_index['by_capability'][cap_key].add(agent_id)
        
        # By status
        status_key = record.status.value
        if status_key not in self.agent_index['by_status']:
            self.agent_index['by_status'][status_key] = set()
        self.agent_index['by_status'][status_key].add(agent_id)
        
        # By tag
        for tag in record.tags:
            if tag not in self.agent_index['by_tag']:
                self.agent_index['by_tag'][tag] = set()
            self.agent_index['by_tag'][tag].add(agent_id)
    
    def _remove_from_indexes(self, agent_id: str, record: AgentRecord):
        """Remove agent from all indexes."""
        # Remove from all index sets
        for index_type, index_dict in self.agent_index.items():
            for key, agent_set in index_dict.items():
                agent_set.discard(agent_id)
    
    async def _save_agent_to_db(self, record: AgentRecord):
        """Save agent record to database."""
        agent_id = record.specification.id
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO agents
                (id, specification, status, metrics, relationships, tags, metadata, 
                 created_at, last_modified, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id,
                json.dumps(_serialize_agent_spec(record.specification)),
                record.status.value,
                json.dumps(asdict(record.metrics)),
                json.dumps({
                    'parent_agents': record.parent_agents,
                    'child_agents': record.child_agents,
                    'collaborating_agents': record.collaborating_agents
                }),
                json.dumps(record.tags),
                json.dumps(record.custom_metadata),
                record.created_at.isoformat(),
                record.last_modified.isoformat(),
                record.specification.version
            ))
            await db.commit()
    
    async def _load_agents_from_db(self):
        """Load all agents from database."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, specification, status, metrics, relationships, 
                       tags, metadata, created_at, last_modified
                FROM agents
            """) as cursor:
                async for row in cursor:
                    try:
                        # Reconstruct agent record
                        spec_data = json.loads(row[1])
                        spec = _deserialize_agent_spec(spec_data)
                        
                        metrics_data = json.loads(row[2])
                        metrics = AgentMetrics(**metrics_data)
                        
                        relationships = json.loads(row[3]) if row[3] else {}
                        tags = json.loads(row[4]) if row[4] else []
                        metadata = json.loads(row[5]) if row[5] else {}
                        
                        record = AgentRecord(
                            specification=spec,
                            status=AgentStatus(row[2]),
                            metrics=metrics,
                            parent_agents=relationships.get('parent_agents', []),
                            child_agents=relationships.get('child_agents', []),
                            collaborating_agents=relationships.get('collaborating_agents', []),
                            tags=tags,
                            custom_metadata=metadata,
                            created_at=datetime.fromisoformat(row[7]),
                            last_modified=datetime.fromisoformat(row[8])
                        )
                        
                        self.agents[spec.id] = record
                        self._update_indexes(spec.id, record)
                        
                    except Exception as e:
                        logger.error(f"Error loading agent {row[0]}: {e}")
        
        logger.info(f"Loaded {len(self.agents)} agents from database")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about agents."""
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE])
        
        # Type distribution
        type_distribution = {}
        for type_key, agent_ids in self.agent_index['by_type'].items():
            type_distribution[type_key] = len(agent_ids)
        
        # Capability distribution
        capability_distribution = {}
        for cap_key, agent_ids in self.agent_index['by_capability'].items():
            capability_distribution[cap_key] = len(agent_ids)
        
        # Performance metrics
        total_executions = sum(a.metrics.total_executions for a in self.agents.values())
        avg_success_rate = (
            sum(a.metrics.successful_executions for a in self.agents.values()) / 
            total_executions if total_executions > 0 else 0
        )
        
        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'type_distribution': type_distribution,
            'capability_distribution': capability_distribution,
            'total_executions': total_executions,
            'average_success_rate': avg_success_rate,
            'registry_size_mb': self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0
        }