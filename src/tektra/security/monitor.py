#!/usr/bin/env python3
"""
Security Monitoring System

Real-time security monitoring and threat detection for agent execution.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with psutil,loguru python monitor.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
# ]
# ///

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
import statistics

import psutil
from loguru import logger


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    RESOURCE_ABUSE = "resource_abuse"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SANDBOX_ESCAPE = "sandbox_escape"
    MALICIOUS_NETWORK = "malicious_network"
    SUSPICIOUS_FILESYSTEM = "suspicious_filesystem"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class ThreatEvent:
    """Represents a detected security threat."""
    
    threat_id: str
    agent_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    description: str
    timestamp: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    false_positive: bool = False


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    
    # Threat statistics
    total_threats: int = 0
    threats_by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    threats_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Resource monitoring
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    total_network_requests: int = 0
    total_file_operations: int = 0
    
    # Time windows
    last_24h_threats: int = 0
    last_hour_threats: int = 0
    
    # Response metrics
    average_detection_time: float = 0.0
    average_response_time: float = 0.0


class SecurityMonitor:
    """
    Security monitoring system for agent execution.
    
    Provides real-time threat detection, anomaly detection,
    and security event monitoring with automated response capabilities.
    """
    
    def __init__(self):
        """Initialize security monitor."""
        self.active = False
        self.threats: Dict[str, ThreatEvent] = {}
        self.metrics = SecurityMetrics()
        
        # Monitoring state
        self.agent_baselines: Dict[str, Dict[str, Any]] = {}
        self.recent_activities: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thresholds and limits
        self.thresholds = {
            "cpu_spike_threshold": 80.0,
            "memory_spike_threshold": 80.0,
            "network_rate_threshold": 100,  # requests per minute
            "file_rate_threshold": 50,      # operations per minute
            "anomaly_score_threshold": 0.8,
        }
        
        # Callbacks
        self.threat_handlers: List[Callable[[ThreatEvent], None]] = []
        self.metrics_callbacks: List[Callable[[SecurityMetrics], None]] = []
        
        # Threading
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        
        logger.info("Security monitor initialized")
    
    def start_monitoring(self) -> bool:
        """
        Start security monitoring.
        
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self.active:
                logger.warning("Security monitor already active")
                return False
            
            try:
                self.active = True
                self.stop_event.clear()
                
                # Start monitoring thread
                self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitor_thread.start()
                
                logger.info("Security monitoring started")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start security monitoring: {e}")
                self.active = False
                return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop security monitoring.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if not self.active:
                return True
            
            try:
                self.stop_event.set()
                
                if self.monitor_thread:
                    self.monitor_thread.join(timeout=5.0)
                
                self.active = False
                logger.info("Security monitoring stopped")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop security monitoring: {e}")
                return False
    
    def register_agent(self, agent_id: str, baseline_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an agent for monitoring.
        
        Args:
            agent_id: Agent identifier
            baseline_data: Baseline behavior data for anomaly detection
        """
        with self._lock:
            if baseline_data:
                self.agent_baselines[agent_id] = baseline_data
            else:
                # Initialize with default baseline
                self.agent_baselines[agent_id] = {
                    "typical_cpu_usage": 10.0,
                    "typical_memory_usage": 100.0,
                    "typical_network_rate": 5.0,
                    "typical_file_rate": 2.0,
                    "behavioral_patterns": []
                }
            
            logger.info(f"Agent registered for monitoring: {agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from monitoring.
        
        Args:
            agent_id: Agent identifier
        """
        with self._lock:
            self.agent_baselines.pop(agent_id, None)
            self.recent_activities.pop(agent_id, None)
            
            logger.info(f"Agent unregistered from monitoring: {agent_id}")
    
    def report_activity(
        self,
        agent_id: str,
        activity_type: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Report agent activity for monitoring.
        
        Args:
            agent_id: Agent identifier
            activity_type: Type of activity
            details: Activity details
        """
        with self._lock:
            activity = {
                "timestamp": time.time(),
                "type": activity_type,
                "details": details
            }
            
            self.recent_activities[agent_id].append(activity)
            
            # Check for immediate threats
            self._analyze_activity(agent_id, activity)
    
    def add_threat_handler(self, handler: Callable[[ThreatEvent], None]) -> None:
        """
        Add threat detection handler.
        
        Args:
            handler: Function to handle threat events
        """
        self.threat_handlers.append(handler)
        logger.info("Threat handler added")
    
    def add_metrics_callback(self, callback: Callable[[SecurityMetrics], None]) -> None:
        """
        Add metrics callback.
        
        Args:
            callback: Function to handle metrics updates
        """
        self.metrics_callbacks.append(callback)
        logger.info("Metrics callback added")
    
    def get_agent_threats(self, agent_id: str) -> List[ThreatEvent]:
        """
        Get threats associated with a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of threat events for the agent
        """
        with self._lock:
            return [threat for threat in self.threats.values() if threat.agent_id == agent_id]
    
    def get_active_threats(self) -> List[ThreatEvent]:
        """
        Get all active (unresolved) threats.
        
        Returns:
            List of active threat events
        """
        with self._lock:
            return [threat for threat in self.threats.values() if not threat.resolved]
    
    def resolve_threat(self, threat_id: str, resolution_note: str = "") -> bool:
        """
        Mark a threat as resolved.
        
        Args:
            threat_id: Threat identifier
            resolution_note: Optional resolution details
            
        Returns:
            True if resolved successfully, False otherwise
        """
        with self._lock:
            if threat_id not in self.threats:
                return False
            
            threat = self.threats[threat_id]
            threat.resolved = True
            threat.evidence["resolution_note"] = resolution_note
            threat.evidence["resolved_at"] = time.time()
            
            logger.info(f"Threat resolved: {threat_id}")
            return True
    
    def mark_false_positive(self, threat_id: str) -> bool:
        """
        Mark a threat as false positive.
        
        Args:
            threat_id: Threat identifier
            
        Returns:
            True if marked successfully, False otherwise
        """
        with self._lock:
            if threat_id not in self.threats:
                return False
            
            threat = self.threats[threat_id]
            threat.false_positive = True
            threat.resolved = True
            threat.evidence["marked_false_positive_at"] = time.time()
            
            logger.info(f"Threat marked as false positive: {threat_id}")
            return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get comprehensive security status.
        
        Returns:
            Dictionary with security status information
        """
        with self._lock:
            active_threats = self.get_active_threats()
            
            status = {
                "monitoring_active": self.active,
                "registered_agents": len(self.agent_baselines),
                "total_threats": len(self.threats),
                "active_threats": len(active_threats),
                "critical_threats": len([t for t in active_threats if t.threat_level == ThreatLevel.CRITICAL]),
                "high_threats": len([t for t in active_threats if t.threat_level == ThreatLevel.HIGH]),
                "metrics": self.metrics.__dict__,
                "recent_threat_summary": self._get_recent_threat_summary()
            }
            
            return status
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Security monitoring loop started")
        
        while not self.stop_event.wait(5.0):  # Check every 5 seconds
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Analyze agent behaviors
                self._analyze_agent_behaviors()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Update time-based metrics
                self._update_time_metrics()
                
                # Notify metrics callbacks
                for callback in self.metrics_callbacks:
                    try:
                        callback(self.metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {e}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
        
        logger.info("Security monitoring loop stopped")
    
    def _analyze_activity(self, agent_id: str, activity: Dict[str, Any]) -> None:
        """Analyze individual activity for threats."""
        activity_type = activity["type"]
        details = activity["details"]
        
        # Check for specific threat patterns
        threats = []
        
        # Resource abuse detection
        if activity_type == "resource_usage":
            cpu_percent = details.get("cpu_percent", 0)
            memory_mb = details.get("memory_mb", 0)
            
            baseline = self.agent_baselines.get(agent_id, {})
            typical_cpu = baseline.get("typical_cpu_usage", 10.0)
            typical_memory = baseline.get("typical_memory_usage", 100.0)
            
            if cpu_percent > self.thresholds["cpu_spike_threshold"]:
                threats.append(self._create_threat(
                    agent_id=agent_id,
                    threat_type=ThreatType.RESOURCE_ABUSE,
                    threat_level=ThreatLevel.HIGH if cpu_percent > 90 else ThreatLevel.MEDIUM,
                    description=f"High CPU usage detected: {cpu_percent:.1f}%",
                    evidence={"cpu_percent": cpu_percent, "threshold": self.thresholds["cpu_spike_threshold"]}
                ))
            
            if memory_mb > self.thresholds["memory_spike_threshold"]:
                threats.append(self._create_threat(
                    agent_id=agent_id,
                    threat_type=ThreatType.RESOURCE_ABUSE,
                    threat_level=ThreatLevel.HIGH,
                    description=f"High memory usage detected: {memory_mb:.1f}MB",
                    evidence={"memory_mb": memory_mb, "threshold": self.thresholds["memory_spike_threshold"]}
                ))
        
        # Network activity monitoring
        elif activity_type == "network_request":
            url = details.get("url", "")
            
            # Check for suspicious domains
            suspicious_domains = ["pastebin.com", "0bin.net", "temp-mail.org"]
            for domain in suspicious_domains:
                if domain in url.lower():
                    threats.append(self._create_threat(
                        agent_id=agent_id,
                        threat_type=ThreatType.MALICIOUS_NETWORK,
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Request to suspicious domain: {domain}",
                        evidence={"url": url, "suspicious_domain": domain}
                    ))
        
        # File system activity monitoring
        elif activity_type == "file_operation":
            file_path = details.get("file_path", "")
            operation = details.get("operation", "")
            
            # Check for sensitive file access
            sensitive_paths = ["/etc/passwd", "/etc/shadow", "/.ssh/", "/.aws/"]
            for path in sensitive_paths:
                if path in file_path:
                    threats.append(self._create_threat(
                        agent_id=agent_id,
                        threat_type=ThreatType.PRIVILEGE_ESCALATION,
                        threat_level=ThreatLevel.HIGH,
                        description=f"Access to sensitive file: {file_path}",
                        evidence={"file_path": file_path, "operation": operation}
                    ))
        
        # Process threats
        for threat in threats:
            self._process_threat(threat)
    
    def _analyze_agent_behaviors(self) -> None:
        """Analyze overall agent behaviors for anomalies."""
        for agent_id in list(self.agent_baselines.keys()):
            try:
                self._analyze_agent_behavior(agent_id)
            except Exception as e:
                logger.error(f"Error analyzing behavior for agent {agent_id}: {e}")
    
    def _analyze_agent_behavior(self, agent_id: str) -> None:
        """Analyze behavior patterns for a specific agent."""
        activities = self.recent_activities.get(agent_id, deque())
        if len(activities) < 10:  # Need minimum activities for analysis
            return
        
        recent_activities = list(activities)[-10:]  # Last 10 activities
        
        # Calculate activity rates
        now = time.time()
        hour_ago = now - 3600
        
        network_requests = len([a for a in recent_activities 
                              if a["type"] == "network_request" and a["timestamp"] > hour_ago])
        file_operations = len([a for a in recent_activities 
                             if a["type"] == "file_operation" and a["timestamp"] > hour_ago])
        
        baseline = self.agent_baselines.get(agent_id, {})
        typical_network_rate = baseline.get("typical_network_rate", 5.0)
        typical_file_rate = baseline.get("typical_file_rate", 2.0)
        
        # Check for anomalous rates
        if network_requests > typical_network_rate * 5:  # 5x normal rate
            threat = self._create_threat(
                agent_id=agent_id,
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.MEDIUM,
                description=f"Unusually high network activity: {network_requests} requests/hour",
                evidence={
                    "network_requests_per_hour": network_requests,
                    "typical_rate": typical_network_rate,
                    "anomaly_factor": network_requests / max(typical_network_rate, 1)
                }
            )
            self._process_threat(threat)
        
        if file_operations > typical_file_rate * 5:  # 5x normal rate
            threat = self._create_threat(
                agent_id=agent_id,
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.MEDIUM,
                description=f"Unusually high file activity: {file_operations} operations/hour",
                evidence={
                    "file_operations_per_hour": file_operations,
                    "typical_rate": typical_file_rate,
                    "anomaly_factor": file_operations / max(typical_file_rate, 1)
                }
            )
            self._process_threat(threat)
    
    def _create_threat(
        self,
        agent_id: str,
        threat_type: ThreatType,
        threat_level: ThreatLevel,
        description: str,
        evidence: Dict[str, Any]
    ) -> ThreatEvent:
        """Create a new threat event."""
        threat_id = f"threat_{int(time.time() * 1000)}_{agent_id}"
        
        return ThreatEvent(
            threat_id=threat_id,
            agent_id=agent_id,
            threat_type=threat_type,
            threat_level=threat_level,
            description=description,
            timestamp=time.time(),
            evidence=evidence
        )
    
    def _process_threat(self, threat: ThreatEvent) -> None:
        """Process a detected threat."""
        with self._lock:
            # Store threat
            self.threats[threat.threat_id] = threat
            
            # Update metrics
            self.metrics.total_threats += 1
            self.metrics.threats_by_level[threat.threat_level.value] += 1
            self.metrics.threats_by_type[threat.threat_type.value] += 1
            
            # Log threat
            log_level = {
                ThreatLevel.INFO: logger.info,
                ThreatLevel.LOW: logger.info,
                ThreatLevel.MEDIUM: logger.warning,
                ThreatLevel.HIGH: logger.warning,
                ThreatLevel.CRITICAL: logger.error
            }[threat.threat_level]
            
            log_level(f"Threat detected: {threat.description} (Level: {threat.threat_level.value})")
            
            # Notify handlers
            for handler in self.threat_handlers:
                try:
                    handler(threat)
                except Exception as e:
                    logger.error(f"Error in threat handler: {e}")
    
    def _update_system_metrics(self) -> None:
        """Update system-wide security metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.metrics.peak_cpu_usage:
                self.metrics.peak_cpu_usage = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = (memory.total - memory.available) / 1024 / 1024
            if memory_mb > self.metrics.peak_memory_usage:
                self.metrics.peak_memory_usage = memory_mb
            
        except Exception as e:
            logger.debug(f"Error updating system metrics: {e}")
    
    def _update_time_metrics(self) -> None:
        """Update time-based metrics."""
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400
        
        with self._lock:
            # Count recent threats
            recent_threats = [t for t in self.threats.values() if t.timestamp > hour_ago]
            daily_threats = [t for t in self.threats.values() if t.timestamp > day_ago]
            
            self.metrics.last_hour_threats = len(recent_threats)
            self.metrics.last_24h_threats = len(daily_threats)
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        cutoff_time = time.time() - 86400  # 24 hours ago
        
        with self._lock:
            # Clean up old activities
            for agent_id in self.recent_activities:
                activities = self.recent_activities[agent_id]
                while activities and activities[0]["timestamp"] < cutoff_time:
                    activities.popleft()
    
    def _get_recent_threat_summary(self) -> Dict[str, Any]:
        """Get summary of recent threats."""
        now = time.time()
        hour_ago = now - 3600
        
        recent_threats = [t for t in self.threats.values() if t.timestamp > hour_ago]
        
        return {
            "last_hour_total": len(recent_threats),
            "last_hour_by_level": {
                level.value: len([t for t in recent_threats if t.threat_level == level])
                for level in ThreatLevel
            },
            "last_hour_by_type": {
                ttype.value: len([t for t in recent_threats if t.threat_type == ttype])
                for ttype in ThreatType
            }
        }


def create_security_monitor() -> SecurityMonitor:
    """
    Create a security monitor instance.
    
    Returns:
        SecurityMonitor instance
    """
    return SecurityMonitor()


if __name__ == "__main__":
    def demo_security_monitor():
        """Demonstrate security monitor functionality."""
        print("🛡️ Security Monitor Demo")
        print("=" * 40)
        
        # Create security monitor
        monitor = create_security_monitor()
        
        # Add threat handler
        def threat_handler(threat: ThreatEvent):
            print(f"🚨 THREAT: {threat.description} (Level: {threat.threat_level.value})")
        
        monitor.add_threat_handler(threat_handler)
        
        # Start monitoring
        if monitor.start_monitoring():
            print("✅ Security monitoring started")
            
            # Register an agent
            monitor.register_agent("demo_agent")
            
            # Simulate some activities
            print("Simulating agent activities...")
            
            # Normal activity
            monitor.report_activity("demo_agent", "resource_usage", {
                "cpu_percent": 15.0,
                "memory_mb": 120.0
            })
            
            # Suspicious network activity
            monitor.report_activity("demo_agent", "network_request", {
                "url": "https://pastebin.com/suspicious",
                "method": "GET"
            })
            
            # High resource usage
            monitor.report_activity("demo_agent", "resource_usage", {
                "cpu_percent": 85.0,
                "memory_mb": 512.0
            })
            
            # Sensitive file access
            monitor.report_activity("demo_agent", "file_operation", {
                "file_path": "/etc/passwd",
                "operation": "read"
            })
            
            # Let monitoring run for a bit
            time.sleep(2)
            
            # Check security status
            status = monitor.get_security_status()
            print(f"Security status: {status['active_threats']} active threats")
            
            # Get active threats
            active_threats = monitor.get_active_threats()
            for threat in active_threats:
                print(f"  - {threat.description} (Type: {threat.threat_type.value})")
            
            # Resolve a threat
            if active_threats:
                monitor.resolve_threat(active_threats[0].threat_id, "Investigated and resolved")
                print(f"Resolved threat: {active_threats[0].threat_id}")
            
            # Stop monitoring
            monitor.stop_monitoring()
            print("✅ Security monitoring stopped")
        
        print("\n🛡️ Security Monitor Demo Complete")
    
    # Run demo
    demo_security_monitor()