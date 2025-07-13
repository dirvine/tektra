#!/usr/bin/env python3
"""
Security Audit and Logging System

Comprehensive audit logging and security event tracking.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru python audit.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
# ]
# ///

import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
import threading

from loguru import logger


class AuditEventType(Enum):
    """Types of audit events."""
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    THREAT_DETECTED = "threat_detected"
    PERMISSION_DENIED = "permission_denied"
    SANDBOX_ESCAPE = "sandbox_escape"
    
    # Agent lifecycle events
    AGENT_CREATED = "agent_created"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_FAILED = "agent_failed"
    
    # Permission events
    PERMISSION_REQUESTED = "permission_requested"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    
    # Tool events
    TOOL_VALIDATED = "tool_validated"
    TOOL_APPROVED = "tool_approved"
    TOOL_BLOCKED = "tool_blocked"
    TOOL_EXECUTED = "tool_executed"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    
    # Network events
    NETWORK_REQUEST = "network_request"
    NETWORK_BLOCKED = "network_blocked"
    
    # File system events
    FILE_ACCESS = "file_access"
    FILE_MODIFICATION = "file_modification"
    FILE_DELETION = "file_deletion"


class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a security audit event."""
    
    event_id: str
    event_type: AuditEventType
    level: AuditLevel
    timestamp: float
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    source_component: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)
        data['tags'] = list(data['tags'])  # Convert set to list for JSON serialization
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary."""
        data = data.copy()
        data['event_type'] = AuditEventType(data['event_type'])
        data['level'] = AuditLevel(data['level'])
        data['tags'] = set(data.get('tags', []))
        return cls(**data)


@dataclass
class AuditFilter:
    """Filter for audit events."""
    
    event_types: Optional[Set[AuditEventType]] = None
    levels: Optional[Set[AuditLevel]] = None
    agent_ids: Optional[Set[str]] = None
    time_range: Optional[tuple[float, float]] = None  # (start_time, end_time)
    tags: Optional[Set[str]] = None
    text_search: Optional[str] = None
    
    def matches(self, event: AuditEvent) -> bool:
        """Check if event matches filter criteria."""
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.levels and event.level not in self.levels:
            return False
        
        if self.agent_ids and event.agent_id not in self.agent_ids:
            return False
        
        if self.time_range:
            start_time, end_time = self.time_range
            if not (start_time <= event.timestamp <= end_time):
                return False
        
        if self.tags and not self.tags.intersection(event.tags):
            return False
        
        if self.text_search:
            search_text = self.text_search.lower()
            searchable_content = f"{event.description} {json.dumps(event.details)}".lower()
            if search_text not in searchable_content:
                return False
        
        return True


class SecurityAuditor:
    """
    Security audit and logging system.
    
    Provides comprehensive audit trail functionality including:
    - Event logging and storage
    - Query and filtering capabilities
    - Real-time monitoring
    - Compliance reporting
    - Alert generation
    """
    
    def __init__(self, storage_path: Optional[Path] = None, max_memory_events: int = 10000):
        """
        Initialize security auditor.
        
        Args:
            storage_path: Path for persistent storage of audit logs
            max_memory_events: Maximum events to keep in memory
        """
        self.storage_path = storage_path or Path.home() / ".cache" / "tektra" / "audit"
        self.max_memory_events = max_memory_events
        
        # In-memory storage
        self.events: Dict[str, AuditEvent] = {}
        self.events_by_time: deque = deque(maxlen=max_memory_events)
        self.events_by_agent: Dict[str, List[str]] = defaultdict(list)
        self.events_by_type: Dict[AuditEventType, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_level": defaultdict(int),
            "events_by_type": defaultdict(int),
            "agents_monitored": set(),
            "start_time": time.time(),
        }
        
        # Real-time monitoring
        self.event_handlers: List[Callable[[AuditEvent], None]] = []
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Threading
        self._lock = threading.RLock()
        self._storage_thread: Optional[threading.Thread] = None
        self._storage_queue: deque = deque()
        self._shutdown_event = threading.Event()
        
        # Create storage directory
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._start_storage_thread()
        
        logger.info(f"Security auditor initialized with storage at {self.storage_path}")
    
    def log_event(
        self,
        event_type: AuditEventType,
        level: AuditLevel = AuditLevel.INFO,
        description: str = "",
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source_component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a security audit event.
        
        Args:
            event_type: Type of the event
            level: Severity level
            description: Human-readable description
            agent_id: Associated agent ID
            user_id: Associated user ID
            source_component: Component that generated the event
            details: Additional event details
            tags: Event tags for categorization
            correlation_id: ID for correlating related events
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            level=level,
            timestamp=time.time(),
            agent_id=agent_id,
            user_id=user_id,
            source_component=source_component,
            description=description,
            details=details or {},
            tags=tags or set(),
            correlation_id=correlation_id
        )
        
        with self._lock:
            # Store in memory
            self.events[event_id] = event
            self.events_by_time.append(event_id)
            
            if agent_id:
                self.events_by_agent[agent_id].append(event_id)
                self.stats["agents_monitored"].add(agent_id)
            
            self.events_by_type[event_type].append(event_id)
            
            # Update statistics
            self.stats["total_events"] += 1
            self.stats["events_by_level"][level.value] += 1
            self.stats["events_by_type"][event_type.value] += 1
            
            # Queue for storage
            if self.storage_path:
                self._storage_queue.append(event)
        
        # Trigger real-time handlers
        self._trigger_event_handlers(event)
        
        # Check alert rules
        self._check_alert_rules(event)
        
        # Log to standard logger
        log_func = {
            AuditLevel.DEBUG: logger.debug,
            AuditLevel.INFO: logger.info,
            AuditLevel.WARNING: logger.warning,
            AuditLevel.ERROR: logger.error,
            AuditLevel.CRITICAL: logger.critical
        }[level]
        
        log_func(f"AUDIT [{event_type.value}]: {description}")
        
        return event_id
    
    def query_events(
        self,
        filter_criteria: Optional[AuditFilter] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[AuditEvent]:
        """
        Query audit events with filtering.
        
        Args:
            filter_criteria: Filter criteria for events
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of matching audit events
        """
        with self._lock:
            # Get all events sorted by timestamp (newest first)
            all_events = []
            for event_id in reversed(list(self.events_by_time)):
                if event_id in self.events:
                    all_events.append(self.events[event_id])
            
            # Apply filters
            if filter_criteria:
                filtered_events = []
                for event in all_events:
                    if filter_criteria.matches(event):
                        filtered_events.append(event)
                all_events = filtered_events
            
            # Apply pagination
            start_idx = offset
            end_idx = start_idx + limit if limit else len(all_events)
            
            return all_events[start_idx:end_idx]
    
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get a specific audit event by ID.
        
        Args:
            event_id: Event identifier
            
        Returns:
            AuditEvent if found, None otherwise
        """
        with self._lock:
            return self.events.get(event_id)
    
    def get_agent_events(
        self,
        agent_id: str,
        limit: Optional[int] = None
    ) -> List[AuditEvent]:
        """
        Get all events for a specific agent.
        
        Args:
            agent_id: Agent identifier
            limit: Maximum number of events to return
            
        Returns:
            List of audit events for the agent
        """
        filter_criteria = AuditFilter(agent_ids={agent_id})
        return self.query_events(filter_criteria, limit)
    
    def get_events_by_type(
        self,
        event_type: AuditEventType,
        limit: Optional[int] = None
    ) -> List[AuditEvent]:
        """
        Get all events of a specific type.
        
        Args:
            event_type: Type of events to retrieve
            limit: Maximum number of events to return
            
        Returns:
            List of audit events of the specified type
        """
        filter_criteria = AuditFilter(event_types={event_type})
        return self.query_events(filter_criteria, limit)
    
    def get_recent_events(
        self,
        hours: float = 24.0,
        level: Optional[AuditLevel] = None
    ) -> List[AuditEvent]:
        """
        Get recent audit events.
        
        Args:
            hours: Number of hours to look back
            level: Minimum severity level
            
        Returns:
            List of recent audit events
        """
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        filter_criteria = AuditFilter(
            time_range=(start_time, end_time),
            levels={level} if level else None
        )
        
        return self.query_events(filter_criteria)
    
    def add_event_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """
        Add real-time event handler.
        
        Args:
            handler: Function to handle audit events
        """
        self.event_handlers.append(handler)
        logger.info("Audit event handler added")
    
    def add_alert_rule(
        self,
        rule_name: str,
        event_types: List[AuditEventType],
        min_level: AuditLevel,
        threshold: int = 1,
        time_window: float = 3600.0,  # 1 hour
        action: Callable[[List[AuditEvent]], None] = None
    ) -> None:
        """
        Add alert rule for automatic threat detection.
        
        Args:
            rule_name: Name of the alert rule
            event_types: Types of events to monitor
            min_level: Minimum severity level
            threshold: Number of events to trigger alert
            time_window: Time window in seconds
            action: Action to take when rule is triggered
        """
        rule = {
            "name": rule_name,
            "event_types": set(event_types),
            "min_level": min_level,
            "threshold": threshold,
            "time_window": time_window,
            "action": action or self._default_alert_action,
            "last_triggered": 0.0
        }
        
        self.alert_rules.append(rule)
        logger.info(f"Alert rule added: {rule_name}")
    
    def generate_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Generate audit report for a time period.
        
        Args:
            start_time: Report start time (None for beginning)
            end_time: Report end time (None for now)
            include_details: Include detailed event information
            
        Returns:
            Audit report dictionary
        """
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = self.stats["start_time"]
        
        filter_criteria = AuditFilter(time_range=(start_time, end_time))
        events = self.query_events(filter_criteria)
        
        # Calculate statistics
        report = {
            "report_period": {
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": (end_time - start_time) / 3600
            },
            "summary": {
                "total_events": len(events),
                "events_by_level": defaultdict(int),
                "events_by_type": defaultdict(int),
                "agents_involved": set(),
                "components_involved": set()
            },
            "security_highlights": {
                "critical_events": 0,
                "security_violations": 0,
                "threats_detected": 0,
                "permissions_denied": 0
            }
        }
        
        # Analyze events
        for event in events:
            report["summary"]["events_by_level"][event.level.value] += 1
            report["summary"]["events_by_type"][event.event_type.value] += 1
            
            if event.agent_id:
                report["summary"]["agents_involved"].add(event.agent_id)
            if event.source_component:
                report["summary"]["components_involved"].add(event.source_component)
            
            # Security highlights
            if event.level == AuditLevel.CRITICAL:
                report["security_highlights"]["critical_events"] += 1
            if event.event_type == AuditEventType.SECURITY_VIOLATION:
                report["security_highlights"]["security_violations"] += 1
            if event.event_type == AuditEventType.THREAT_DETECTED:
                report["security_highlights"]["threats_detected"] += 1
            if event.event_type == AuditEventType.PERMISSION_DENIED:
                report["security_highlights"]["permissions_denied"] += 1
        
        # Convert sets to lists for JSON serialization
        report["summary"]["agents_involved"] = list(report["summary"]["agents_involved"])
        report["summary"]["components_involved"] = list(report["summary"]["components_involved"])
        
        # Include detailed events if requested
        if include_details:
            report["events"] = [event.to_dict() for event in events]
        
        return report
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit system statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = self.stats.copy()
            stats["agents_monitored"] = list(stats["agents_monitored"])
            stats["uptime_hours"] = (time.time() - stats["start_time"]) / 3600
            stats["memory_events"] = len(self.events)
            stats["storage_queue_size"] = len(self._storage_queue)
            
            return stats
    
    def shutdown(self) -> None:
        """Shutdown the audit system and flush pending events."""
        logger.info("Shutting down security auditor")
        
        self._shutdown_event.set()
        
        if self._storage_thread:
            self._storage_thread.join(timeout=5.0)
        
        # Flush remaining events
        if self.storage_path:
            self._flush_storage_queue()
    
    def _start_storage_thread(self) -> None:
        """Start background storage thread."""
        def storage_worker():
            while not self._shutdown_event.wait(1.0):  # Check every second
                try:
                    self._flush_storage_queue()
                except Exception as e:
                    logger.error(f"Error in storage thread: {e}")
        
        self._storage_thread = threading.Thread(target=storage_worker, daemon=True)
        self._storage_thread.start()
    
    def _flush_storage_queue(self) -> None:
        """Flush events from storage queue to disk."""
        if not self.storage_path:
            return
        
        events_to_store = []
        
        with self._lock:
            while self._storage_queue:
                events_to_store.append(self._storage_queue.popleft())
        
        if not events_to_store:
            return
        
        # Store events to daily log files
        events_by_date = defaultdict(list)
        for event in events_to_store:
            date_str = time.strftime("%Y-%m-%d", time.localtime(event.timestamp))
            events_by_date[date_str].append(event)
        
        for date_str, events in events_by_date.items():
            log_file = self.storage_path / f"audit_{date_str}.jsonl"
            
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    for event in events:
                        f.write(json.dumps(event.to_dict()) + '\n')
            except Exception as e:
                logger.error(f"Failed to write audit events to {log_file}: {e}")
    
    def _trigger_event_handlers(self, event: AuditEvent) -> None:
        """Trigger real-time event handlers."""
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in audit event handler: {e}")
    
    def _check_alert_rules(self, event: AuditEvent) -> None:
        """Check if event triggers any alert rules."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            # Check if event matches rule criteria
            if event.event_type not in rule["event_types"]:
                continue
            
            if event.level.value not in [level.value for level in AuditLevel 
                                       if list(AuditLevel).index(level) >= list(AuditLevel).index(rule["min_level"])]:
                continue
            
            # Check time window
            window_start = current_time - rule["time_window"]
            matching_events = []
            
            for event_id in reversed(list(self.events_by_time)):
                if event_id not in self.events:
                    continue
                
                stored_event = self.events[event_id]
                if stored_event.timestamp < window_start:
                    break
                
                if (stored_event.event_type in rule["event_types"] and
                    list(AuditLevel).index(stored_event.level) >= list(AuditLevel).index(rule["min_level"])):
                    matching_events.append(stored_event)
            
            # Check threshold
            if len(matching_events) >= rule["threshold"]:
                # Avoid triggering too frequently
                if current_time - rule["last_triggered"] > 300:  # 5 minutes cooldown
                    rule["last_triggered"] = current_time
                    
                    try:
                        rule["action"](matching_events)
                    except Exception as e:
                        logger.error(f"Error in alert rule action: {e}")
    
    def _default_alert_action(self, events: List[AuditEvent]) -> None:
        """Default action for triggered alert rules."""
        logger.warning(f"Alert triggered: {len(events)} matching events detected")
        for event in events[:3]:  # Show first 3 events
            logger.warning(f"  - {event.event_type.value}: {event.description}")


def create_security_auditor(storage_path: Optional[Path] = None) -> SecurityAuditor:
    """
    Create a security auditor instance.
    
    Args:
        storage_path: Path for persistent audit log storage
        
    Returns:
        SecurityAuditor instance
    """
    return SecurityAuditor(storage_path)


if __name__ == "__main__":
    def demo_security_auditor():
        """Demonstrate security auditor functionality."""
        print("📋 Security Auditor Demo")
        print("=" * 40)
        
        # Create auditor
        auditor = create_security_auditor()
        
        # Add event handler
        def event_handler(event: AuditEvent):
            print(f"📝 Event: {event.event_type.value} - {event.description}")
        
        auditor.add_event_handler(event_handler)
        
        # Add alert rule
        auditor.add_alert_rule(
            rule_name="security_violations",
            event_types=[AuditEventType.SECURITY_VIOLATION, AuditEventType.THREAT_DETECTED],
            min_level=AuditLevel.WARNING,
            threshold=2,
            time_window=60.0
        )
        
        # Log various events
        print("Logging audit events...")
        
        # Normal events
        auditor.log_event(
            AuditEventType.AGENT_CREATED,
            AuditLevel.INFO,
            "Agent created successfully",
            agent_id="demo_agent_1"
        )
        
        auditor.log_event(
            AuditEventType.TOOL_VALIDATED,
            AuditLevel.INFO,
            "Tool validation completed",
            details={"tool_name": "calculator", "result": "safe"}
        )
        
        # Security events
        auditor.log_event(
            AuditEventType.SECURITY_VIOLATION,
            AuditLevel.WARNING,
            "Unauthorized file access attempt",
            agent_id="demo_agent_1",
            details={"file_path": "/etc/passwd", "action": "read"}
        )
        
        auditor.log_event(
            AuditEventType.THREAT_DETECTED,
            AuditLevel.ERROR,
            "Malicious code pattern detected",
            agent_id="demo_agent_1",
            details={"pattern": "eval()", "risk_score": 0.8}
        )
        
        # Query events
        print("\nQuerying events...")
        
        # Get all events
        all_events = auditor.query_events(limit=10)
        print(f"Total events: {len(all_events)}")
        
        # Get security-related events
        security_filter = AuditFilter(
            event_types={AuditEventType.SECURITY_VIOLATION, AuditEventType.THREAT_DETECTED}
        )
        security_events = auditor.query_events(security_filter)
        print(f"Security events: {len(security_events)}")
        
        # Get agent events
        agent_events = auditor.get_agent_events("demo_agent_1")
        print(f"Agent events: {len(agent_events)}")
        
        # Generate report
        print("\nGenerating audit report...")
        report = auditor.generate_report(include_details=False)
        print(f"Report period: {report['report_period']['duration_hours']:.2f} hours")
        print(f"Total events: {report['summary']['total_events']}")
        print(f"Critical events: {report['security_highlights']['critical_events']}")
        print(f"Security violations: {report['security_highlights']['security_violations']}")
        
        # Get statistics
        stats = auditor.get_statistics()
        print(f"\nSystem statistics:")
        print(f"  Uptime: {stats['uptime_hours']:.2f} hours")
        print(f"  Memory events: {stats['memory_events']}")
        print(f"  Agents monitored: {len(stats['agents_monitored'])}")
        
        # Shutdown
        auditor.shutdown()
        
        print("\n📋 Security Auditor Demo Complete")
    
    # Run demo
    demo_security_auditor()