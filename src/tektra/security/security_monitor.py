#!/usr/bin/env python3
"""
Security Monitoring and Logging System

Comprehensive security event logging, threat detection, anomaly analysis,
and real-time alerting for the Tektra AI Assistant security framework.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru,numpy,scikit-learn,prometheus-client python security_monitor.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "prometheus-client>=0.19.0",
#     "pydantic>=2.0.0",
#     "psutil>=5.9.0",
# ]
# ///

import asyncio
import time
import uuid
import json
import hashlib
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, AsyncIterable
from pathlib import Path
import threading
import queue
import re
from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from pydantic import BaseModel, Field

from .context import SecurityContext, SecurityLevel
from .permissions import PermissionManager
from .tool_validator import ValidationResult, ThreatLevel
from .consent_framework import ConsentResponse, ConsentAction


class EventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    PERMISSION_REQUEST = "permission_request"
    CONSENT_DECISION = "consent_decision"
    TOOL_VALIDATION = "tool_validation"
    SANDBOX_EXECUTION = "sandbox_execution"
    SYSTEM_ACCESS = "system_access"
    NETWORK_ACCESS = "network_access"
    FILE_ACCESS = "file_access"
    PROCESS_CREATION = "process_creation"
    SECURITY_VIOLATION = "security_violation"
    ANOMALY_DETECTED = "anomaly_detected"
    THREAT_DETECTED = "threat_detected"
    POLICY_VIOLATION = "policy_violation"
    RESOURCE_ABUSE = "resource_abuse"


class EventSeverity(Enum):
    """Security event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    MALWARE = "malware"
    INTRUSION = "intrusion"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    POLICY_VIOLATION = "policy_violation"
    RESOURCE_ABUSE = "resource_abuse"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ANOMALOUS_ACTIVITY = "anomalous_activity"


@dataclass
class SecurityEvent:
    """Represents a security event."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: EventType = EventType.SYSTEM_ACCESS
    severity: EventSeverity = EventSeverity.INFO
    
    # Core event data
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    
    # Event details
    message: str = ""
    description: str = ""
    resource: Optional[str] = None
    action: Optional[str] = None
    
    # Context and metadata
    security_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Threat information
    threat_indicators: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 to 1.0
    
    # Correlation
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "message": self.message,
            "description": self.description,
            "resource": self.resource,
            "action": self.action,
            "security_context": self.security_context,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "threat_indicators": self.threat_indicators,
            "risk_score": self.risk_score,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityEvent":
        """Create event from dictionary."""
        event = cls()
        event.event_id = data.get("event_id", event.event_id)
        event.timestamp = datetime.fromisoformat(data.get("timestamp", event.timestamp.isoformat()))
        event.event_type = EventType(data.get("event_type", event.event_type.value))
        event.severity = EventSeverity(data.get("severity", event.severity.value))
        event.agent_id = data.get("agent_id")
        event.user_id = data.get("user_id")
        event.session_id = data.get("session_id")
        event.source_ip = data.get("source_ip")
        event.message = data.get("message", "")
        event.description = data.get("description", "")
        event.resource = data.get("resource")
        event.action = data.get("action")
        event.security_context = data.get("security_context")
        event.metadata = data.get("metadata", {})
        event.tags = set(data.get("tags", []))
        event.threat_indicators = data.get("threat_indicators", [])
        event.risk_score = data.get("risk_score", 0.0)
        event.correlation_id = data.get("correlation_id")
        event.parent_event_id = data.get("parent_event_id")
        return event


@dataclass
class ThreatAlert:
    """Represents a security threat alert."""
    
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    threat_type: ThreatType = ThreatType.SUSPICIOUS_BEHAVIOR
    severity: EventSeverity = EventSeverity.WARNING
    
    # Alert details
    title: str = ""
    description: str = ""
    confidence: float = 0.5  # 0.0 to 1.0
    
    # Related events
    triggering_events: List[str] = field(default_factory=list)  # Event IDs
    
    # Threat analysis
    indicators: List[str] = field(default_factory=list)
    attack_vector: Optional[str] = None
    potential_impact: str = ""
    
    # Response information
    recommended_actions: List[str] = field(default_factory=list)
    auto_response_taken: Optional[str] = None
    
    # Status
    status: str = "active"  # active, investigating, resolved, false_positive
    assigned_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "triggering_events": self.triggering_events,
            "indicators": self.indicators,
            "attack_vector": self.attack_vector,
            "potential_impact": self.potential_impact,
            "recommended_actions": self.recommended_actions,
            "auto_response_taken": self.auto_response_taken,
            "status": self.status,
            "assigned_to": self.assigned_to
        }


class EventPattern:
    """Pattern for detecting threat sequences."""
    
    def __init__(
        self,
        pattern_id: str,
        name: str,
        description: str,
        event_sequence: List[Dict[str, Any]],
        time_window: timedelta,
        threshold: int = 1,
        threat_type: ThreatType = ThreatType.SUSPICIOUS_BEHAVIOR
    ):
        self.pattern_id = pattern_id
        self.name = name
        self.description = description
        self.event_sequence = event_sequence
        self.time_window = time_window
        self.threshold = threshold
        self.threat_type = threat_type
        
        # Pattern matching state
        self.matched_sequences: Dict[str, List[SecurityEvent]] = {}
    
    def matches_event(self, event: SecurityEvent, step: Dict[str, Any]) -> bool:
        """Check if an event matches a pattern step."""
        # Check event type
        if "event_type" in step:
            if event.event_type.value != step["event_type"]:
                return False
        
        # Check severity
        if "min_severity" in step:
            severity_order = ["debug", "info", "warning", "error", "critical"]
            event_idx = severity_order.index(event.severity.value)
            min_idx = severity_order.index(step["min_severity"])
            if event_idx < min_idx:
                return False
        
        # Check metadata patterns
        if "metadata_patterns" in step:
            for key, pattern in step["metadata_patterns"].items():
                if key not in event.metadata:
                    return False
                if not re.search(pattern, str(event.metadata[key])):
                    return False
        
        # Check tags
        if "required_tags" in step:
            if not all(tag in event.tags for tag in step["required_tags"]):
                return False
        
        return True
    
    def process_event(self, event: SecurityEvent) -> List[ThreatAlert]:
        """Process an event against this pattern."""
        alerts = []
        
        # Clean up old sequences
        cutoff_time = datetime.now() - self.time_window
        for key in list(self.matched_sequences.keys()):
            sequence = self.matched_sequences[key]
            if sequence and sequence[0].timestamp < cutoff_time:
                del self.matched_sequences[key]
        
        # Check if event matches any step in the sequence
        for step_idx, step in enumerate(self.event_sequence):
            if self.matches_event(event, step):
                # Find or create sequence
                sequence_key = f"{event.agent_id}_{event.session_id}"
                if sequence_key not in self.matched_sequences:
                    self.matched_sequences[sequence_key] = []
                
                sequence = self.matched_sequences[sequence_key]
                
                # Check if this is the next expected step
                if len(sequence) == step_idx:
                    sequence.append(event)
                    
                    # Check if we completed the pattern
                    if len(sequence) >= len(self.event_sequence):
                        # Pattern matched - create alert
                        alert = ThreatAlert(
                            threat_type=self.threat_type,
                            severity=EventSeverity.WARNING,
                            title=f"Pattern Detected: {self.name}",
                            description=self.description,
                            confidence=0.8,
                            triggering_events=[e.event_id for e in sequence],
                            indicators=[f"Pattern: {self.pattern_id}"],
                            recommended_actions=[
                                "Investigate sequence of events",
                                "Review agent behavior",
                                "Check for policy violations"
                            ]
                        )
                        alerts.append(alert)
                        
                        # Reset sequence
                        del self.matched_sequences[sequence_key]
        
        return alerts


class AnomalyDetector:
    """Machine learning-based anomaly detection."""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        
        # Feature extractors
        self.feature_names = [
            'hour_of_day', 'day_of_week', 'events_per_minute',
            'unique_agents', 'unique_resources', 'avg_risk_score',
            'error_rate', 'permission_requests', 'validation_failures'
        ]
        
        # Training data buffer
        self.training_buffer = deque(maxlen=1000)
        self.last_training = None
    
    def extract_features(self, events: List[SecurityEvent], time_window: timedelta) -> np.ndarray:
        """Extract features from events for anomaly detection."""
        if not events:
            return np.zeros(len(self.feature_names))
        
        now = datetime.now()
        window_start = now - time_window
        recent_events = [e for e in events if e.timestamp >= window_start]
        
        if not recent_events:
            return np.zeros(len(self.feature_names))
        
        features = []
        
        # Temporal features
        features.append(now.hour)  # hour_of_day
        features.append(now.weekday())  # day_of_week
        
        # Event volume features
        features.append(len(recent_events) / time_window.total_seconds() * 60)  # events_per_minute
        features.append(len(set(e.agent_id for e in recent_events if e.agent_id)))  # unique_agents
        features.append(len(set(e.resource for e in recent_events if e.resource)))  # unique_resources
        
        # Risk features
        risk_scores = [e.risk_score for e in recent_events if e.risk_score > 0]
        features.append(statistics.mean(risk_scores) if risk_scores else 0)  # avg_risk_score
        
        # Error rate
        error_events = len([e for e in recent_events if e.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]])
        features.append(error_events / len(recent_events))  # error_rate
        
        # Permission and validation features
        permission_events = len([e for e in recent_events if e.event_type == EventType.PERMISSION_REQUEST])
        validation_failures = len([e for e in recent_events if e.event_type == EventType.TOOL_VALIDATION and "failed" in e.message.lower()])
        
        features.append(permission_events)  # permission_requests
        features.append(validation_failures)  # validation_failures
        
        return np.array(features)
    
    def add_training_data(self, events: List[SecurityEvent]) -> None:
        """Add events to training data buffer."""
        features = self.extract_features(events, timedelta(minutes=5))
        self.training_buffer.append(features)
    
    def train(self) -> bool:
        """Train the anomaly detection model."""
        if len(self.training_buffer) < 50:  # Need minimum data
            return False
        
        try:
            # Prepare training data
            X = np.array(list(self.training_buffer))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled)
            self.trained = True
            self.last_training = datetime.now()
            
            logger.info(f"Anomaly detection model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detection model: {e}")
            return False
    
    def detect_anomaly(self, events: List[SecurityEvent]) -> Tuple[bool, float]:
        """Detect anomaly in current events."""
        if not self.trained:
            return False, 0.0
        
        try:
            features = self.extract_features(events, timedelta(minutes=5))
            features_scaled = self.scaler.transform([features])
            
            # Get anomaly score
            anomaly_score = self.model.decision_function(features_scaled)[0]
            is_anomaly = self.model.predict(features_scaled)[0] == -1
            
            # Convert to probability (0-1)
            anomaly_probability = max(0.0, min(1.0, (0.5 - anomaly_score) * 2))
            
            return is_anomaly, anomaly_probability
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False, 0.0


class SecurityMonitor:
    """
    Comprehensive security monitoring and logging system.
    
    Provides real-time security event logging, threat detection,
    anomaly analysis, and alerting capabilities.
    """
    
    def __init__(
        self,
        permission_manager: Optional[PermissionManager] = None,
        enable_prometheus: bool = True,
        prometheus_port: int = 8090
    ):
        """Initialize the security monitor."""
        self.permission_manager = permission_manager
        
        # Event storage and processing
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.alerts: List[ThreatAlert] = []
        self.event_queue: queue.Queue = queue.Queue()
        
        # Threat detection
        self.threat_patterns: Dict[str, EventPattern] = {}
        self.anomaly_detector = AnomalyDetector()
        
        # Alert handlers
        self.alert_handlers: List[Callable[[ThreatAlert], None]] = []
        
        # Configuration
        self.monitoring_enabled = True
        self.auto_training_enabled = True
        self.min_confidence_alert = 0.6
        
        # Threading
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Metrics (Prometheus)
        self.enable_prometheus = enable_prometheus
        if enable_prometheus:
            self._setup_prometheus_metrics(prometheus_port)
        
        # Initialize default patterns
        self._initialize_threat_patterns()
        
        # Start processing
        self._start_processing()
        
        logger.info("Security monitor initialized")
    
    def _setup_prometheus_metrics(self, port: int) -> None:
        """Set up Prometheus metrics."""
        try:
            # Event metrics
            self.event_counter = Counter(
                'security_events_total',
                'Total number of security events',
                ['event_type', 'severity', 'agent_id']
            )
            
            self.alert_counter = Counter(
                'security_alerts_total',
                'Total number of security alerts',
                ['threat_type', 'severity']
            )
            
            self.risk_score_histogram = Histogram(
                'security_risk_score',
                'Distribution of security event risk scores'
            )
            
            self.active_alerts_gauge = Gauge(
                'security_active_alerts',
                'Number of active security alerts'
            )
            
            # Start Prometheus server
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
            
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
            self.enable_prometheus = False
    
    def _initialize_threat_patterns(self) -> None:
        """Initialize default threat detection patterns."""
        # Rapid permission escalation pattern
        escalation_pattern = EventPattern(
            pattern_id="permission_escalation",
            name="Rapid Permission Escalation",
            description="Multiple permission requests in short time",
            event_sequence=[
                {"event_type": "permission_request", "min_severity": "info"},
                {"event_type": "permission_request", "min_severity": "info"},
                {"event_type": "permission_request", "min_severity": "info"}
            ],
            time_window=timedelta(minutes=2),
            threat_type=ThreatType.PRIVILEGE_ESCALATION
        )
        
        # Validation failure pattern
        validation_pattern = EventPattern(
            pattern_id="validation_failures",
            name="Multiple Validation Failures",
            description="Repeated tool validation failures indicating malicious code",
            event_sequence=[
                {"event_type": "tool_validation", "metadata_patterns": {"result": ".*failed.*"}},
                {"event_type": "tool_validation", "metadata_patterns": {"result": ".*failed.*"}}
            ],
            time_window=timedelta(minutes=5),
            threat_type=ThreatType.MALWARE
        )
        
        # Resource abuse pattern
        resource_pattern = EventPattern(
            pattern_id="resource_abuse",
            name="Resource Abuse Sequence",
            description="Pattern indicating resource exhaustion attack",
            event_sequence=[
                {"event_type": "resource_abuse", "min_severity": "warning"},
                {"event_type": "sandbox_execution", "metadata_patterns": {"timeout": "true"}}
            ],
            time_window=timedelta(minutes=3),
            threat_type=ThreatType.DENIAL_OF_SERVICE
        )
        
        self.threat_patterns["permission_escalation"] = escalation_pattern
        self.threat_patterns["validation_failures"] = validation_pattern
        self.threat_patterns["resource_abuse"] = resource_pattern
    
    def _start_processing(self) -> None:
        """Start the event processing thread."""
        self._processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self._processing_thread.start()
    
    def _process_events(self) -> None:
        """Main event processing loop."""
        while not self._stop_event.is_set():
            try:
                # Process queued events
                events_processed = 0
                while not self.event_queue.empty() and events_processed < 100:
                    try:
                        event = self.event_queue.get_nowait()
                        self._process_single_event(event)
                        events_processed += 1
                    except queue.Empty:
                        break
                
                # Periodic tasks
                if events_processed == 0:
                    # Anomaly detection
                    self._run_anomaly_detection()
                    
                    # Cleanup old data
                    self._cleanup_old_data()
                    
                    # Auto-training
                    if self.auto_training_enabled:
                        self._auto_train_models()
                    
                    # Sleep briefly if no events processed
                    self._stop_event.wait(1.0)
                
            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                self._stop_event.wait(5.0)
    
    def _process_single_event(self, event: SecurityEvent) -> None:
        """Process a single security event."""
        with self._lock:
            # Store event
            self.events.append(event)
            
            # Update metrics
            if self.enable_prometheus:
                self.event_counter.labels(
                    event_type=event.event_type.value,
                    severity=event.severity.value,
                    agent_id=event.agent_id or "unknown"
                ).inc()
                
                if event.risk_score > 0:
                    self.risk_score_histogram.observe(event.risk_score)
            
            # Run threat detection
            alerts = self._detect_threats(event)
            
            # Process alerts
            for alert in alerts:
                self._handle_alert(alert)
            
            # Add to training data
            if self.auto_training_enabled:
                recent_events = list(self.events)[-100:]  # Last 100 events
                self.anomaly_detector.add_training_data(recent_events)
    
    def _detect_threats(self, event: SecurityEvent) -> List[ThreatAlert]:
        """Run threat detection on an event."""
        alerts = []
        
        # Pattern-based detection
        for pattern in self.threat_patterns.values():
            pattern_alerts = pattern.process_event(event)
            alerts.extend(pattern_alerts)
        
        # Rule-based detection
        rule_alerts = self._run_rule_based_detection(event)
        alerts.extend(rule_alerts)
        
        return alerts
    
    def _run_rule_based_detection(self, event: SecurityEvent) -> List[ThreatAlert]:
        """Run rule-based threat detection."""
        alerts = []
        
        # High risk score events
        if event.risk_score > 0.8:
            alerts.append(ThreatAlert(
                threat_type=ThreatType.SUSPICIOUS_BEHAVIOR,
                severity=EventSeverity.WARNING,
                title="High Risk Event",
                description=f"Event with high risk score: {event.risk_score:.2f}",
                confidence=event.risk_score,
                triggering_events=[event.event_id],
                indicators=[f"Risk score: {event.risk_score}"],
                recommended_actions=["Investigate event details", "Review agent behavior"]
            ))
        
        # Critical severity events
        if event.severity == EventSeverity.CRITICAL:
            alerts.append(ThreatAlert(
                threat_type=ThreatType.SUSPICIOUS_BEHAVIOR,
                severity=EventSeverity.ERROR,
                title="Critical Security Event",
                description=f"Critical event detected: {event.message}",
                confidence=0.9,
                triggering_events=[event.event_id],
                indicators=["Critical severity"],
                recommended_actions=["Immediate investigation required"]
            ))
        
        # Specific threat indicators
        for indicator in event.threat_indicators:
            if indicator.lower() in ["malware", "exploit", "backdoor"]:
                alerts.append(ThreatAlert(
                    threat_type=ThreatType.MALWARE,
                    severity=EventSeverity.ERROR,
                    title=f"Threat Indicator: {indicator}",
                    description=f"Event contains threat indicator: {indicator}",
                    confidence=0.8,
                    triggering_events=[event.event_id],
                    indicators=[indicator],
                    recommended_actions=["Block agent", "Investigate code"]
                ))
        
        return alerts
    
    def _run_anomaly_detection(self) -> None:
        """Run anomaly detection on recent events."""
        if not self.anomaly_detector.trained:
            return
        
        recent_events = list(self.events)[-100:]  # Last 100 events
        is_anomaly, confidence = self.anomaly_detector.detect_anomaly(recent_events)
        
        if is_anomaly and confidence > self.min_confidence_alert:
            alert = ThreatAlert(
                threat_type=ThreatType.ANOMALOUS_ACTIVITY,
                severity=EventSeverity.WARNING,
                title="Anomalous Activity Detected",
                description=f"ML model detected anomalous pattern (confidence: {confidence:.2f})",
                confidence=confidence,
                indicators=["Anomaly detection"],
                recommended_actions=[
                    "Review recent activity patterns",
                    "Check for unusual agent behavior",
                    "Investigate system changes"
                ]
            )
            self._handle_alert(alert)
    
    def _handle_alert(self, alert: ThreatAlert) -> None:
        """Handle a security alert."""
        with self._lock:
            # Store alert
            self.alerts.append(alert)
            
            # Update metrics
            if self.enable_prometheus:
                self.alert_counter.labels(
                    threat_type=alert.threat_type.value,
                    severity=alert.severity.value
                ).inc()
                
                self.active_alerts_gauge.set(len([a for a in self.alerts if a.status == "active"]))
            
            # Run alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            # Log alert
            logger.warning(
                f"Security Alert: {alert.title} "
                f"(Type: {alert.threat_type.value}, Confidence: {alert.confidence:.2f})"
            )
    
    def _cleanup_old_data(self) -> None:
        """Clean up old alerts and training data."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        with self._lock:
            # Remove old resolved alerts
            self.alerts = [
                alert for alert in self.alerts
                if alert.timestamp > cutoff_time or alert.status == "active"
            ]
    
    def _auto_train_models(self) -> None:
        """Automatically train ML models if needed."""
        # Train anomaly detector if not trained or training is old
        if (not self.anomaly_detector.trained or
            (self.anomaly_detector.last_training and
             datetime.now() - self.anomaly_detector.last_training > timedelta(hours=6))):
            
            if len(self.anomaly_detector.training_buffer) >= 50:
                self.anomaly_detector.train()
    
    def log_event(
        self,
        event_type: EventType,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            agent_id=agent_id,
            **kwargs
        )
        
        # Queue for processing
        self.event_queue.put(event)
        
        return event.event_id
    
    def log_authentication_event(
        self,
        agent_id: str,
        success: bool,
        source_ip: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log an authentication event."""
        return self.log_event(
            event_type=EventType.AUTHENTICATION,
            severity=EventSeverity.INFO if success else EventSeverity.WARNING,
            message=f"Authentication {'successful' if success else 'failed'} for agent {agent_id}",
            agent_id=agent_id,
            source_ip=source_ip,
            metadata={"success": success, **kwargs.get("metadata", {})}
        )
    
    def log_permission_request(
        self,
        agent_id: str,
        permission: str,
        granted: bool,
        justification: str = "",
        **kwargs
    ) -> str:
        """Log a permission request event."""
        return self.log_event(
            event_type=EventType.PERMISSION_REQUEST,
            severity=EventSeverity.INFO,
            message=f"Permission '{permission}' {'granted' if granted else 'denied'} for agent {agent_id}",
            agent_id=agent_id,
            resource=permission,
            action="grant" if granted else "deny",
            metadata={
                "permission": permission,
                "granted": granted,
                "justification": justification,
                **kwargs.get("metadata", {})
            }
        )
    
    def log_consent_decision(
        self,
        agent_id: str,
        consent_response: ConsentResponse,
        **kwargs
    ) -> str:
        """Log a consent decision event."""
        return self.log_event(
            event_type=EventType.CONSENT_DECISION,
            severity=EventSeverity.INFO,
            message=f"Consent {consent_response.action.value} for agent {agent_id}",
            agent_id=agent_id,
            action=consent_response.action.value,
            metadata={
                "request_id": consent_response.request_id,
                "action": consent_response.action.value,
                "granted_permissions": consent_response.granted_permissions,
                "reason": consent_response.reason,
                **kwargs.get("metadata", {})
            }
        )
    
    def log_tool_validation(
        self,
        agent_id: str,
        tool_id: str,
        validation_result: ValidationResult,
        **kwargs
    ) -> str:
        """Log a tool validation event."""
        severity = EventSeverity.INFO
        if not validation_result.is_safe:
            severity = EventSeverity.WARNING
        if validation_result.overall_threat_level == ThreatLevel.CRITICAL:
            severity = EventSeverity.ERROR
        
        return self.log_event(
            event_type=EventType.TOOL_VALIDATION,
            severity=severity,
            message=f"Tool validation for {tool_id}: {'passed' if validation_result.is_safe else 'failed'}",
            agent_id=agent_id,
            resource=tool_id,
            action="validate",
            risk_score=self._threat_level_to_risk_score(validation_result.overall_threat_level),
            threat_indicators=[f.vulnerability_type.value for f in validation_result.findings],
            metadata={
                "tool_id": tool_id,
                "is_safe": validation_result.is_safe,
                "threat_level": validation_result.overall_threat_level.value,
                "findings_count": len(validation_result.findings),
                **kwargs.get("metadata", {})
            }
        )
    
    def log_sandbox_execution(
        self,
        agent_id: str,
        sandbox_id: str,
        command: List[str],
        return_code: int,
        execution_time: float,
        **kwargs
    ) -> str:
        """Log a sandbox execution event."""
        severity = EventSeverity.INFO
        if return_code != 0:
            severity = EventSeverity.WARNING
        if execution_time > 30.0:  # Long execution
            severity = EventSeverity.WARNING
        
        return self.log_event(
            event_type=EventType.SANDBOX_EXECUTION,
            severity=severity,
            message=f"Sandbox execution {sandbox_id} completed with code {return_code}",
            agent_id=agent_id,
            resource=sandbox_id,
            action="execute",
            metadata={
                "sandbox_id": sandbox_id,
                "command": command,
                "return_code": return_code,
                "execution_time": execution_time,
                "timeout": execution_time > 30.0,
                **kwargs.get("metadata", {})
            }
        )
    
    def log_security_violation(
        self,
        agent_id: str,
        violation_type: str,
        description: str,
        severity: EventSeverity = EventSeverity.ERROR,
        **kwargs
    ) -> str:
        """Log a security violation event."""
        return self.log_event(
            event_type=EventType.SECURITY_VIOLATION,
            severity=severity,
            message=f"Security violation: {violation_type}",
            description=description,
            agent_id=agent_id,
            action=violation_type,
            risk_score=0.8,  # High risk by default
            threat_indicators=[violation_type],
            tags={"security_violation", violation_type},
            metadata={
                "violation_type": violation_type,
                "description": description,
                **kwargs.get("metadata", {})
            }
        )
    
    def _threat_level_to_risk_score(self, threat_level: ThreatLevel) -> float:
        """Convert threat level to risk score."""
        mapping = {
            ThreatLevel.SAFE: 0.0,
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 1.0
        }
        return mapping.get(threat_level, 0.5)
    
    def add_alert_handler(self, handler: Callable[[ThreatAlert], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.debug(f"Added alert handler: {handler.__name__}")
    
    def add_threat_pattern(self, pattern: EventPattern) -> None:
        """Add a custom threat detection pattern."""
        self.threat_patterns[pattern.pattern_id] = pattern
        logger.info(f"Added threat pattern: {pattern.name}")
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get security events with filtering."""
        with self._lock:
            events = list(self.events)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_alerts(
        self,
        threat_type: Optional[ThreatType] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50
    ) -> List[ThreatAlert]:
        """Get security alerts with filtering."""
        with self._lock:
            alerts = list(self.alerts)
        
        # Apply filters
        if threat_type:
            alerts = [a for a in alerts if a.threat_type == threat_type]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            events = list(self.events)
            alerts = list(self.alerts)
        
        # Event statistics
        event_types = defaultdict(int)
        severities = defaultdict(int)
        agents = defaultdict(int)
        
        for event in events:
            event_types[event.event_type.value] += 1
            severities[event.severity.value] += 1
            if event.agent_id:
                agents[event.agent_id] += 1
        
        # Alert statistics
        threat_types = defaultdict(int)
        alert_severities = defaultdict(int)
        alert_statuses = defaultdict(int)
        
        for alert in alerts:
            threat_types[alert.threat_type.value] += 1
            alert_severities[alert.severity.value] += 1
            alert_statuses[alert.status] += 1
        
        return {
            "total_events": len(events),
            "total_alerts": len(alerts),
            "active_alerts": len([a for a in alerts if a.status == "active"]),
            "events_by_type": dict(event_types),
            "events_by_severity": dict(severities),
            "events_by_agent": dict(agents),
            "alerts_by_threat_type": dict(threat_types),
            "alerts_by_severity": dict(alert_severities),
            "alerts_by_status": dict(alert_statuses),
            "anomaly_detector_trained": self.anomaly_detector.trained,
            "threat_patterns_count": len(self.threat_patterns),
            "monitoring_enabled": self.monitoring_enabled
        }
    
    def shutdown(self) -> None:
        """Shutdown the security monitor."""
        logger.info("Shutting down security monitor")
        self._stop_event.set()
        
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        
        self.monitoring_enabled = False


def create_security_monitor(**kwargs) -> SecurityMonitor:
    """
    Create a security monitor with the given configuration.
    
    Args:
        **kwargs: Monitor configuration
        
    Returns:
        Configured security monitor
    """
    return SecurityMonitor(**kwargs)


# Default alert handlers
def console_alert_handler(alert: ThreatAlert) -> None:
    """Simple console alert handler."""
    print(f"\n🚨 SECURITY ALERT 🚨")
    print(f"Type: {alert.threat_type.value}")
    print(f"Severity: {alert.severity.value}")
    print(f"Title: {alert.title}")
    print(f"Description: {alert.description}")
    print(f"Confidence: {alert.confidence:.2f}")
    print(f"Timestamp: {alert.timestamp}")
    
    if alert.recommended_actions:
        print("Recommended Actions:")
        for action in alert.recommended_actions:
            print(f"  - {action}")
    print("-" * 50)


def email_alert_handler(alert: ThreatAlert) -> None:
    """Email alert handler (placeholder)."""
    # In production, this would send actual emails
    logger.warning(f"EMAIL ALERT: {alert.title} - {alert.description}")


if __name__ == "__main__":
    async def demo_security_monitor():
        """Demonstrate security monitoring functionality."""
        print("🔍 Security Monitor Demo")
        print("=" * 40)
        
        # Create monitor
        monitor = create_security_monitor(enable_prometheus=False)
        
        # Add alert handlers
        monitor.add_alert_handler(console_alert_handler)
        monitor.add_alert_handler(email_alert_handler)
        
        print("\n📝 Logging Security Events:")
        print("-" * 30)
        
        # Log various security events
        events = [
            ("Authentication", lambda: monitor.log_authentication_event("agent_001", True, "192.168.1.100")),
            ("Permission Request", lambda: monitor.log_permission_request("agent_001", "system.admin.users", False, "User management")),
            ("Tool Validation Failure", lambda: monitor.log_event(
                EventType.TOOL_VALIDATION,
                "Tool validation failed - malware detected",
                EventSeverity.ERROR,
                agent_id="agent_002",
                threat_indicators=["malware", "suspicious_code"],
                risk_score=0.9
            )),
            ("Multiple Permission Requests", lambda: [
                monitor.log_permission_request("agent_003", "filesystem.write.temp", True),
                monitor.log_permission_request("agent_003", "network.http.request", True),
                monitor.log_permission_request("agent_003", "system.admin.users", False)
            ]),
            ("Security Violation", lambda: monitor.log_security_violation(
                "agent_004", "unauthorized_access", "Agent attempted to access restricted resource"
            ))
        ]
        
        for name, event_func in events:
            print(f"  {name}...")
            event_func()
            time.sleep(0.5)  # Give processing time
        
        # Wait for processing
        print("\n⏳ Processing events...")
        time.sleep(3)
        
        # Show statistics
        print(f"\n📊 Monitoring Statistics:")
        print("-" * 25)
        
        stats = monitor.get_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # Show recent events
        print(f"\n📋 Recent Events:")
        print("-" * 15)
        
        recent_events = monitor.get_events(limit=5)
        for event in recent_events:
            print(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.event_type.value}: {event.message}")
        
        # Show alerts
        print(f"\n🚨 Security Alerts:")
        print("-" * 17)
        
        alerts = monitor.get_alerts()
        if alerts:
            for alert in alerts:
                print(f"  {alert.timestamp.strftime('%H:%M:%S')} - {alert.threat_type.value}: {alert.title}")
        else:
            print("  No alerts generated")
        
        # Test anomaly detection
        print(f"\n🤖 Testing Anomaly Detection:")
        print("-" * 30)
        
        # Generate training data
        print("  Generating training data...")
        for i in range(60):
            monitor.log_event(
                EventType.SYSTEM_ACCESS,
                f"Normal system access {i}",
                agent_id=f"agent_{i % 3}",
                risk_score=0.1 + (i % 10) * 0.01
            )
        
        time.sleep(2)
        
        # Train model
        if monitor.anomaly_detector.train():
            print("  ✅ Anomaly detection model trained")
            
            # Generate anomalous activity
            print("  Generating anomalous activity...")
            for i in range(20):
                monitor.log_event(
                    EventType.SECURITY_VIOLATION,
                    f"Suspicious activity {i}",
                    EventSeverity.ERROR,
                    agent_id="suspicious_agent",
                    risk_score=0.9,
                    threat_indicators=["anomaly"]
                )
            
            time.sleep(3)
            
            # Check for anomaly alerts
            anomaly_alerts = monitor.get_alerts(threat_type=ThreatType.ANOMALOUS_ACTIVITY)
            if anomaly_alerts:
                print(f"  ✅ {len(anomaly_alerts)} anomaly alerts generated")
            else:
                print("  ⚠️ No anomaly alerts (may need more diverse data)")
        else:
            print("  ⚠️ Not enough data to train anomaly detection")
        
        # Final statistics
        print(f"\n📈 Final Statistics:")
        print("-" * 18)
        
        final_stats = monitor.get_statistics()
        print(f"  Total Events: {final_stats['total_events']}")
        print(f"  Total Alerts: {final_stats['total_alerts']}")
        print(f"  Active Alerts: {final_stats['active_alerts']}")
        
        # Cleanup
        print(f"\n🧹 Shutting down monitor...")
        monitor.shutdown()
        
        print("\n🔍 Security Monitor Demo Complete")
    
    # Run demo
    asyncio.run(demo_security_monitor())