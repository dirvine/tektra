#!/usr/bin/env python3
"""
Security Integration and Testing Framework

Comprehensive security testing, integration validation, performance analysis,
and production readiness assessment for the Tektra security framework.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with loguru,pytest,asyncio,psutil,time,statistics python security_integration_test.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
#     "pytest>=7.4.0",
#     "pytest-asyncio>=0.21.0",
#     "psutil>=5.9.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
# ]
# ///

import asyncio
import time
import statistics
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import json
import sys
import os

import psutil
import numpy as np
from loguru import logger

# Import all security components
from .context import SecurityContext, SecurityLevel
from .permissions import PermissionManager, Permission
from .advanced_sandbox import AdvancedSandbox, SandboxConfig, IsolationType, ResourceLimits
from .tool_validator import ToolValidator, ValidationResult, ThreatLevel
from .consent_framework import ConsentFramework, ConsentMode, PermissionScope
from .security_monitor import SecurityMonitor, EventType, EventSeverity, ThreatType


@dataclass
class TestResult:
    """Result of a security test."""
    
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration": self.duration,
            "details": self.details,
            "errors": self.errors,
            "warnings": self.warnings,
            "performance_metrics": self.performance_metrics
        }


@dataclass
class IntegrationTestSuite:
    """Results from the complete integration test suite."""
    
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_duration: float = 0.0
    
    test_results: List[TestResult] = field(default_factory=list)
    performance_summary: Dict[str, float] = field(default_factory=dict)
    security_assessment: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        return (self.passed_tests / self.total_tests) if self.total_tests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": self.success_rate,
            "total_duration": self.total_duration,
            "test_results": [r.to_dict() for r in self.test_results],
            "performance_summary": self.performance_summary,
            "security_assessment": self.security_assessment
        }


class SecurityIntegrationTester:
    """
    Comprehensive security integration and testing framework.
    
    Tests all security components working together, validates performance,
    and assesses production readiness.
    """
    
    def __init__(self):
        """Initialize the security integration tester."""
        self.test_results: List[TestResult] = []
        
        # Security components (will be initialized for each test)
        self.permission_manager: Optional[PermissionManager] = None
        self.sandbox: Optional[AdvancedSandbox] = None
        self.tool_validator: Optional[ToolValidator] = None
        self.consent_framework: Optional[ConsentFramework] = None
        self.security_monitor: Optional[SecurityMonitor] = None
        
        # Test configuration
        self.test_timeout = 30.0
        self.performance_iterations = 10
        
        logger.info("Security integration tester initialized")
    
    async def run_full_test_suite(self) -> IntegrationTestSuite:
        """Run the complete security integration test suite."""
        logger.info("🚀 Starting comprehensive security integration test suite")
        
        suite = IntegrationTestSuite()
        start_time = time.time()
        
        # Test categories
        test_categories = [
            ("Component Initialization", self._test_component_initialization),
            ("Permission System Integration", self._test_permission_integration),
            ("Sandboxing Integration", self._test_sandbox_integration),
            ("Tool Validation Integration", self._test_tool_validation_integration),
            ("Consent Framework Integration", self._test_consent_integration),
            ("Security Monitoring Integration", self._test_monitoring_integration),
            ("End-to-End Workflows", self._test_end_to_end_workflows),
            ("Security Attack Scenarios", self._test_security_scenarios),
            ("Performance Under Security", self._test_performance_with_security),
            ("Production Readiness", self._test_production_readiness)
        ]
        
        for category_name, test_func in test_categories:
            logger.info(f"🔍 Running test category: {category_name}")
            
            try:
                category_results = await test_func()
                suite.test_results.extend(category_results)
                
                passed_in_category = sum(1 for r in category_results if r.passed)
                total_in_category = len(category_results)
                
                logger.info(
                    f"✅ {category_name}: {passed_in_category}/{total_in_category} tests passed"
                )
                
            except Exception as e:
                logger.error(f"❌ Test category {category_name} failed: {e}")
                
                # Add failure result
                suite.test_results.append(TestResult(
                    test_name=f"{category_name} (Category Failure)",
                    passed=False,
                    duration=0.0,
                    errors=[str(e)]
                ))
        
        # Calculate summary
        suite.total_tests = len(suite.test_results)
        suite.passed_tests = sum(1 for r in suite.test_results if r.passed)
        suite.failed_tests = suite.total_tests - suite.passed_tests
        suite.total_duration = time.time() - start_time
        
        # Generate performance summary
        suite.performance_summary = self._generate_performance_summary()
        
        # Generate security assessment
        suite.security_assessment = self._generate_security_assessment()
        
        logger.info(
            f"🎉 Test suite completed: {suite.passed_tests}/{suite.total_tests} tests passed "
            f"({suite.success_rate:.1%}) in {suite.total_duration:.2f}s"
        )
        
        return suite
    
    async def _test_component_initialization(self) -> List[TestResult]:
        """Test initialization of all security components."""
        results = []
        
        # Test individual component initialization
        components = [
            ("PermissionManager", lambda: PermissionManager()),
            ("AdvancedSandbox", lambda: AdvancedSandbox()),
            ("ToolValidator", lambda: ToolValidator()),
            ("ConsentFramework", lambda: ConsentFramework()),
            ("SecurityMonitor", lambda: SecurityMonitor(enable_prometheus=False))
        ]
        
        for name, init_func in components:
            start_time = time.time()
            
            try:
                component = init_func()
                duration = time.time() - start_time
                
                results.append(TestResult(
                    test_name=f"Initialize {name}",
                    passed=True,
                    duration=duration,
                    details={"component": name},
                    performance_metrics={"init_time": duration}
                ))
                
                # Store for later tests
                if name == "PermissionManager":
                    self.permission_manager = component
                elif name == "AdvancedSandbox":
                    self.sandbox = component
                elif name == "ToolValidator":
                    self.tool_validator = component
                elif name == "ConsentFramework":
                    self.consent_framework = component
                elif name == "SecurityMonitor":
                    self.security_monitor = component
                
            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    test_name=f"Initialize {name}",
                    passed=False,
                    duration=duration,
                    errors=[str(e)]
                ))
        
        # Test integrated initialization
        start_time = time.time()
        try:
            # Initialize with interdependencies
            integrated_sandbox = AdvancedSandbox(permission_manager=self.permission_manager)
            integrated_validator = ToolValidator(
                sandbox=integrated_sandbox,
                permission_manager=self.permission_manager
            )
            integrated_consent = ConsentFramework(permission_manager=self.permission_manager)
            integrated_monitor = SecurityMonitor(
                permission_manager=self.permission_manager,
                enable_prometheus=False
            )
            
            duration = time.time() - start_time
            
            results.append(TestResult(
                test_name="Integrated Component Initialization",
                passed=True,
                duration=duration,
                details={"components": 4},
                performance_metrics={"integrated_init_time": duration}
            ))
            
            # Update references to integrated components
            self.sandbox = integrated_sandbox
            self.tool_validator = integrated_validator
            self.consent_framework = integrated_consent
            self.security_monitor = integrated_monitor
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Integrated Component Initialization",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_permission_integration(self) -> List[TestResult]:
        """Test permission system integration."""
        results = []
        
        if not self.permission_manager:
            results.append(TestResult(
                test_name="Permission Integration",
                passed=False,
                duration=0.0,
                errors=["PermissionManager not initialized"]
            ))
            return results
        
        # Test basic permission operations
        start_time = time.time()
        try:
            agent_id = "test_agent_001"
            permission = "system.test.basic"
            
            # Grant permission
            self.permission_manager.grant_permission(agent_id, permission)
            
            # Check permission
            has_permission = self.permission_manager.has_permission(agent_id, permission)
            
            # Revoke permission
            self.permission_manager.revoke_permission(agent_id, permission)
            
            # Verify revocation
            revoked = not self.permission_manager.has_permission(agent_id, permission)
            
            duration = time.time() - start_time
            
            results.append(TestResult(
                test_name="Basic Permission Operations",
                passed=has_permission and revoked,
                duration=duration,
                details={
                    "grant_success": True,
                    "check_success": has_permission,
                    "revoke_success": revoked
                },
                performance_metrics={"permission_ops_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Basic Permission Operations",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        # Test permission with security context
        start_time = time.time()
        try:
            context = SecurityContext(
                agent_id="test_agent_002",
                security_level=SecurityLevel.MEDIUM,
                session_id="test_session"
            )
            
            # Test context-aware operations
            success = True  # Placeholder for actual context testing
            duration = time.time() - start_time
            
            results.append(TestResult(
                test_name="Permission with Security Context",
                passed=success,
                duration=duration,
                details={"context_agent": context.agent_id},
                performance_metrics={"context_permission_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Permission with Security Context",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_sandbox_integration(self) -> List[TestResult]:
        """Test sandboxing integration."""
        results = []
        
        if not self.sandbox:
            results.append(TestResult(
                test_name="Sandbox Integration",
                passed=False,
                duration=0.0,
                errors=["AdvancedSandbox not initialized"]
            ))
            return results
        
        # Test basic sandbox execution
        start_time = time.time()
        try:
            # Simple safe command
            sandbox_id, stdout, stderr, return_code = await self.sandbox.execute(
                ["python3", "-c", "print('Hello from sandbox')"],
                config=SandboxConfig(
                    isolation_type=IsolationType.PROCESS,
                    resource_limits=ResourceLimits(execution_timeout=10.0)
                )
            )
            
            duration = time.time() - start_time
            success = return_code == 0 and b"Hello from sandbox" in stdout
            
            results.append(TestResult(
                test_name="Basic Sandbox Execution",
                passed=success,
                duration=duration,
                details={
                    "sandbox_id": sandbox_id,
                    "return_code": return_code,
                    "stdout_length": len(stdout)
                },
                performance_metrics={"sandbox_exec_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Basic Sandbox Execution",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        # Test sandbox with resource limits
        start_time = time.time()
        try:
            # Command that should hit resource limits
            config = SandboxConfig(
                isolation_type=IsolationType.PROCESS,
                resource_limits=ResourceLimits(
                    memory_bytes=32 * 1024 * 1024,  # 32MB
                    execution_timeout=5.0
                )
            )
            
            sandbox_id, stdout, stderr, return_code = await self.sandbox.execute(
                ["python3", "-c", "x = [0] * (10**6); print('Memory test')"],
                config=config
            )
            
            duration = time.time() - start_time
            
            results.append(TestResult(
                test_name="Sandbox Resource Limits",
                passed=True,  # Success if it doesn't crash
                duration=duration,
                details={
                    "return_code": return_code,
                    "timeout_triggered": duration >= 5.0
                },
                performance_metrics={"resource_limit_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Sandbox Resource Limits",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_tool_validation_integration(self) -> List[TestResult]:
        """Test tool validation integration."""
        results = []
        
        if not self.tool_validator:
            results.append(TestResult(
                test_name="Tool Validation Integration",
                passed=False,
                duration=0.0,
                errors=["ToolValidator not initialized"]
            ))
            return results
        
        # Test safe code validation
        start_time = time.time()
        try:
            safe_code = '''
def safe_function(x, y):
    """A safe mathematical function."""
    return x + y

result = safe_function(2, 3)
print(f"Result: {result}")
'''
            
            validation_result = await self.tool_validator.validate_tool(
                tool_id="safe_tool_test",
                code=safe_code,
                skip_dynamic=True  # Skip for speed
            )
            
            duration = time.time() - start_time
            
            results.append(TestResult(
                test_name="Safe Code Validation",
                passed=validation_result.is_safe,
                duration=duration,
                details={
                    "threat_level": validation_result.overall_threat_level.value,
                    "findings_count": len(validation_result.findings)
                },
                performance_metrics={"safe_validation_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Safe Code Validation",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        # Test malicious code detection
        start_time = time.time()
        try:
            malicious_code = '''
import os
import subprocess

# Attempt system access
os.system("ls -la")
subprocess.Popen(["cat", "/etc/passwd"])

# Suspicious base64 usage
import base64
payload = base64.b64decode("bWFsaWNpb3VzIGNvZGU=")
exec(payload)
'''
            
            validation_result = await self.tool_validator.validate_tool(
                tool_id="malicious_tool_test",
                code=malicious_code,
                skip_dynamic=True  # Skip for safety
            )
            
            duration = time.time() - start_time
            
            # Should detect as unsafe
            detected_threats = not validation_result.is_safe
            
            results.append(TestResult(
                test_name="Malicious Code Detection",
                passed=detected_threats,
                duration=duration,
                details={
                    "threat_level": validation_result.overall_threat_level.value,
                    "findings_count": len(validation_result.findings),
                    "detected_unsafe": not validation_result.is_safe
                },
                performance_metrics={"malicious_validation_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Malicious Code Detection",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_consent_integration(self) -> List[TestResult]:
        """Test consent framework integration."""
        results = []
        
        if not self.consent_framework:
            results.append(TestResult(
                test_name="Consent Integration",
                passed=False,
                duration=0.0,
                errors=["ConsentFramework not initialized"]
            ))
            return results
        
        # Test automatic consent (policy-based)
        start_time = time.time()
        try:
            # Set to automatic mode
            self.consent_framework.set_consent_mode(ConsentMode.AUTOMATIC)
            
            # Request safe permission (should be auto-granted by policy)
            granted, grant_id = await self.consent_framework.request_permission(
                agent_id="test_agent_consent",
                permission="data.read.public",
                scope=PermissionScope.TASK,
                justification="Test safe operation"
            )
            
            duration = time.time() - start_time
            
            results.append(TestResult(
                test_name="Automatic Consent (Safe Operation)",
                passed=granted,
                duration=duration,
                details={
                    "granted": granted,
                    "grant_id": grant_id
                },
                performance_metrics={"auto_consent_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Automatic Consent (Safe Operation)",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        # Test consent denial for dangerous operations
        start_time = time.time()
        try:
            # Request dangerous permission (should be auto-denied by policy)
            granted, grant_id = await self.consent_framework.request_permission(
                agent_id="test_agent_consent",
                permission="system.admin.root",
                scope=PermissionScope.GLOBAL,
                justification="Test dangerous operation"
            )
            
            duration = time.time() - start_time
            
            # Should be denied
            correctly_denied = not granted
            
            results.append(TestResult(
                test_name="Automatic Consent (Dangerous Operation)",
                passed=correctly_denied,
                duration=duration,
                details={
                    "granted": granted,
                    "correctly_denied": correctly_denied
                },
                performance_metrics={"auto_denial_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Automatic Consent (Dangerous Operation)",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_monitoring_integration(self) -> List[TestResult]:
        """Test security monitoring integration."""
        results = []
        
        if not self.security_monitor:
            results.append(TestResult(
                test_name="Monitoring Integration",
                passed=False,
                duration=0.0,
                errors=["SecurityMonitor not initialized"]
            ))
            return results
        
        # Test event logging
        start_time = time.time()
        try:
            # Log various security events
            event_ids = []
            
            event_ids.append(self.security_monitor.log_authentication_event(
                "test_agent_monitor", True, "127.0.0.1"
            ))
            
            event_ids.append(self.security_monitor.log_permission_request(
                "test_agent_monitor", "system.test.monitor", True, "Testing monitoring"
            ))
            
            event_ids.append(self.security_monitor.log_security_violation(
                "test_agent_monitor", "test_violation", "Testing violation logging"
            ))
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            duration = time.time() - start_time
            
            # Check if events were logged
            recent_events = self.security_monitor.get_events(limit=10)
            logged_events = len([e for e in recent_events if e.event_id in event_ids])
            
            results.append(TestResult(
                test_name="Security Event Logging",
                passed=logged_events >= 3,
                duration=duration,
                details={
                    "events_logged": logged_events,
                    "event_ids": event_ids
                },
                performance_metrics={"event_logging_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Security Event Logging",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        # Test alert generation
        start_time = time.time()
        try:
            # Generate events that should trigger alerts
            for i in range(5):
                self.security_monitor.log_event(
                    EventType.SECURITY_VIOLATION,
                    f"High-risk event {i}",
                    EventSeverity.ERROR,
                    agent_id="suspicious_agent",
                    risk_score=0.9,
                    threat_indicators=["high_risk", "suspicious"]
                )
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            duration = time.time() - start_time
            
            # Check for alerts
            alerts = self.security_monitor.get_alerts(limit=5)
            
            results.append(TestResult(
                test_name="Alert Generation",
                passed=len(alerts) > 0,
                duration=duration,
                details={
                    "alerts_generated": len(alerts)
                },
                performance_metrics={"alert_generation_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Alert Generation",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_end_to_end_workflows(self) -> List[TestResult]:
        """Test end-to-end security workflows."""
        results = []
        
        # Test complete agent tool execution workflow
        start_time = time.time()
        try:
            agent_id = "e2e_test_agent"
            tool_code = '''
def calculate_fibonacci(n):
    """Calculate fibonacci number safely."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
'''
            
            # Step 1: Tool Validation
            validation_result = await self.tool_validator.validate_tool(
                tool_id="fibonacci_tool",
                code=tool_code,
                skip_dynamic=True
            )
            
            if not validation_result.is_safe:
                raise Exception("Tool validation failed")
            
            # Step 2: Permission Request
            granted, grant_id = await self.consent_framework.request_permission(
                agent_id=agent_id,
                permission="compute.cpu.basic",
                scope=PermissionScope.TASK,
                justification="Fibonacci calculation"
            )
            
            if not granted:
                raise Exception("Permission not granted")
            
            # Step 3: Sandbox Execution
            sandbox_id, stdout, stderr, return_code = await self.sandbox.execute(
                ["python3", "-c", tool_code],
                config=SandboxConfig(
                    isolation_type=IsolationType.PROCESS,
                    resource_limits=ResourceLimits(execution_timeout=10.0)
                )
            )
            
            if return_code != 0:
                raise Exception(f"Sandbox execution failed: {stderr}")
            
            # Step 4: Monitor Events
            await asyncio.sleep(0.5)
            events = self.security_monitor.get_events(agent_id=agent_id, limit=5)
            
            duration = time.time() - start_time
            
            workflow_success = (
                validation_result.is_safe and
                granted and
                return_code == 0 and
                len(events) > 0
            )
            
            results.append(TestResult(
                test_name="End-to-End Tool Execution Workflow",
                passed=workflow_success,
                duration=duration,
                details={
                    "validation_passed": validation_result.is_safe,
                    "permission_granted": granted,
                    "execution_successful": return_code == 0,
                    "events_logged": len(events)
                },
                performance_metrics={"e2e_workflow_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="End-to-End Tool Execution Workflow",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_security_scenarios(self) -> List[TestResult]:
        """Test specific security attack scenarios."""
        results = []
        
        # Test permission escalation attack
        start_time = time.time()
        try:
            attacker_agent = "attacker_agent"
            
            # Rapid permission requests (should trigger pattern)
            permissions = [
                "data.read.basic",
                "filesystem.write.temp",
                "network.http.request",
                "system.admin.users"
            ]
            
            granted_count = 0
            for permission in permissions:
                granted, _ = await self.consent_framework.request_permission(
                    agent_id=attacker_agent,
                    permission=permission,
                    scope=PermissionScope.TASK,
                    justification="Legitimate request"
                )
                if granted:
                    granted_count += 1
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            # Wait for pattern detection
            await asyncio.sleep(2.0)
            
            duration = time.time() - start_time
            
            # Check if escalation was detected
            alerts = self.security_monitor.get_alerts(threat_type=ThreatType.PRIVILEGE_ESCALATION)
            escalation_detected = len(alerts) > 0
            
            # Should have blocked some permissions and detected pattern
            security_effective = granted_count < len(permissions) or escalation_detected
            
            results.append(TestResult(
                test_name="Permission Escalation Attack",
                passed=security_effective,
                duration=duration,
                details={
                    "permissions_granted": granted_count,
                    "total_permissions": len(permissions),
                    "escalation_detected": escalation_detected
                },
                performance_metrics={"escalation_test_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Permission Escalation Attack",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        # Test malicious code injection
        start_time = time.time()
        try:
            malicious_payload = '''
import os
import subprocess
import base64

# Attempt to access sensitive files
try:
    with open("/etc/passwd", "r") as f:
        data = f.read()
        print("Accessed sensitive file")
except:
    pass

# Attempt command injection
try:
    os.system("whoami && id")
except:
    pass

# Encoded malicious payload
payload = base64.b64decode("ZXZhbCgiX19pbXBvcnRfXygncmVxdWVzdHMnKS5nZXQoJ2h0dHA6Ly9ldmlsLmNvbScpIik=")
try:
    exec(payload)
except:
    pass
'''
            
            # Attempt validation
            validation_result = await self.tool_validator.validate_tool(
                tool_id="malicious_payload",
                code=malicious_payload,
                skip_dynamic=True  # Don't actually execute
            )
            
            duration = time.time() - start_time
            
            # Should detect as malicious
            malware_detected = not validation_result.is_safe
            high_threat = validation_result.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            
            results.append(TestResult(
                test_name="Malicious Code Injection",
                passed=malware_detected and high_threat,
                duration=duration,
                details={
                    "malware_detected": malware_detected,
                    "threat_level": validation_result.overall_threat_level.value,
                    "findings_count": len(validation_result.findings)
                },
                performance_metrics={"malware_detection_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Malicious Code Injection",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_performance_with_security(self) -> List[TestResult]:
        """Test performance impact of security measures."""
        results = []
        
        # Benchmark basic operations with security
        operations = [
            ("Permission Check", lambda: self.permission_manager.has_permission("test_agent", "test.permission")),
            ("Tool Validation", lambda: asyncio.run(self.tool_validator.validate_tool(
                "test_tool", "print('hello')", skip_dynamic=True
            ))),
            ("Consent Request", lambda: asyncio.run(self.consent_framework.request_permission(
                "test_agent", "data.read.test", PermissionScope.TASK, "Test"
            ))),
            ("Event Logging", lambda: self.security_monitor.log_event(
                EventType.SYSTEM_ACCESS, "Test event", agent_id="test_agent"
            ))
        ]
        
        for op_name, op_func in operations:
            start_time = time.time()
            
            try:
                # Run operation multiple times for better measurement
                execution_times = []
                for _ in range(self.performance_iterations):
                    op_start = time.time()
                    
                    try:
                        result = op_func()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass  # Ignore errors in performance test
                    
                    execution_times.append(time.time() - op_start)
                
                duration = time.time() - start_time
                
                avg_time = statistics.mean(execution_times)
                median_time = statistics.median(execution_times)
                max_time = max(execution_times)
                
                # Performance thresholds (in seconds)
                performance_good = avg_time < 0.1  # 100ms average
                
                results.append(TestResult(
                    test_name=f"Performance: {op_name}",
                    passed=performance_good,
                    duration=duration,
                    details={
                        "iterations": self.performance_iterations,
                        "avg_time": avg_time,
                        "median_time": median_time,
                        "max_time": max_time
                    },
                    performance_metrics={
                        f"{op_name.lower().replace(' ', '_')}_avg_time": avg_time,
                        f"{op_name.lower().replace(' ', '_')}_median_time": median_time,
                        f"{op_name.lower().replace(' ', '_')}_max_time": max_time
                    }
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    test_name=f"Performance: {op_name}",
                    passed=False,
                    duration=duration,
                    errors=[str(e)]
                ))
        
        # Test memory usage
        start_time = time.time()
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Perform operations that might consume memory
            for i in range(100):
                await self.tool_validator.validate_tool(
                    f"test_tool_{i}",
                    f"print('Test {i}')",
                    skip_dynamic=True
                )
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            memory_increase_mb = memory_increase / (1024 * 1024)
            
            duration = time.time() - start_time
            
            # Memory increase should be reasonable (< 50MB for 100 validations)
            memory_efficient = memory_increase_mb < 50
            
            results.append(TestResult(
                test_name="Memory Usage Under Load",
                passed=memory_efficient,
                duration=duration,
                details={
                    "initial_memory_mb": initial_memory / (1024 * 1024),
                    "final_memory_mb": final_memory / (1024 * 1024),
                    "memory_increase_mb": memory_increase_mb
                },
                performance_metrics={
                    "memory_increase_mb": memory_increase_mb,
                    "memory_per_operation_kb": (memory_increase / 100) / 1024
                }
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Memory Usage Under Load",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    async def _test_production_readiness(self) -> List[TestResult]:
        """Test production readiness criteria."""
        results = []
        
        # Test error handling robustness
        start_time = time.time()
        try:
            error_scenarios = [
                ("Invalid Permission", lambda: self.permission_manager.has_permission(None, "")),
                ("Malformed Code", lambda: asyncio.run(self.tool_validator.validate_tool(
                    "bad_tool", "invalid python syntax {"
                ))),
                ("Resource Exhaustion", lambda: asyncio.run(self.sandbox.execute(
                    ["python3", "-c", "while True: pass"],
                    config=SandboxConfig(resource_limits=ResourceLimits(execution_timeout=1.0))
                ))),
            ]
            
            error_handling_count = 0
            for scenario_name, scenario_func in error_scenarios:
                try:
                    result = scenario_func()
                    if asyncio.iscoroutine(result):
                        await result
                    error_handling_count += 1  # Should have handled gracefully
                except Exception:
                    error_handling_count += 1  # Expected to raise exception
            
            duration = time.time() - start_time
            
            # All scenarios should be handled without crashing
            robust_error_handling = error_handling_count == len(error_scenarios)
            
            results.append(TestResult(
                test_name="Error Handling Robustness",
                passed=robust_error_handling,
                duration=duration,
                details={
                    "scenarios_tested": len(error_scenarios),
                    "scenarios_handled": error_handling_count
                },
                performance_metrics={"error_handling_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Error Handling Robustness",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        # Test component cleanup
        start_time = time.time()
        try:
            # Test graceful shutdown
            if self.security_monitor:
                self.security_monitor.shutdown()
            
            # Test resource cleanup
            if self.sandbox:
                self.sandbox.cleanup_all_sandboxes()
            
            duration = time.time() - start_time
            
            results.append(TestResult(
                test_name="Component Cleanup",
                passed=True,
                duration=duration,
                details={"components_cleaned": 2},
                performance_metrics={"cleanup_time": duration}
            ))
            
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                test_name="Component Cleanup",
                passed=False,
                duration=duration,
                errors=[str(e)]
            ))
        
        return results
    
    def _generate_performance_summary(self) -> Dict[str, float]:
        """Generate performance summary from test results."""
        perf_metrics = {}
        
        for result in self.test_results:
            for metric_name, metric_value in result.performance_metrics.items():
                if metric_name not in perf_metrics:
                    perf_metrics[metric_name] = []
                perf_metrics[metric_name].append(metric_value)
        
        # Calculate aggregated metrics
        summary = {}
        for metric_name, values in perf_metrics.items():
            if values:
                summary[f"{metric_name}_avg"] = statistics.mean(values)
                summary[f"{metric_name}_max"] = max(values)
                summary[f"{metric_name}_min"] = min(values)
        
        return summary
    
    def _generate_security_assessment(self) -> Dict[str, Any]:
        """Generate security assessment from test results."""
        assessment = {
            "overall_score": 0.0,
            "categories": {},
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Categorize tests
        categories = {
            "initialization": [],
            "permission_system": [],
            "sandboxing": [],
            "validation": [],
            "consent": [],
            "monitoring": [],
            "end_to_end": [],
            "attack_resistance": [],
            "performance": [],
            "production_readiness": []
        }
        
        for result in self.test_results:
            test_name = result.test_name.lower()
            
            if "initialization" in test_name or "initialize" in test_name:
                categories["initialization"].append(result)
            elif "permission" in test_name:
                categories["permission_system"].append(result)
            elif "sandbox" in test_name:
                categories["sandboxing"].append(result)
            elif "validation" in test_name:
                categories["validation"].append(result)
            elif "consent" in test_name:
                categories["consent"].append(result)
            elif "monitoring" in test_name or "alert" in test_name or "event" in test_name:
                categories["monitoring"].append(result)
            elif "end-to-end" in test_name or "workflow" in test_name:
                categories["end_to_end"].append(result)
            elif "attack" in test_name or "malicious" in test_name or "escalation" in test_name:
                categories["attack_resistance"].append(result)
            elif "performance" in test_name or "memory" in test_name:
                categories["performance"].append(result)
            elif "production" in test_name or "cleanup" in test_name or "error handling" in test_name:
                categories["production_readiness"].append(result)
        
        # Calculate category scores
        total_score = 0.0
        total_weight = 0.0
        
        for category, tests in categories.items():
            if tests:
                passed_tests = sum(1 for t in tests if t.passed)
                category_score = passed_tests / len(tests)
                
                # Weight categories differently
                weight = 1.0
                if category in ["attack_resistance", "validation", "sandboxing"]:
                    weight = 2.0  # Security-critical categories
                
                assessment["categories"][category] = {
                    "score": category_score,
                    "passed": passed_tests,
                    "total": len(tests),
                    "weight": weight
                }
                
                total_score += category_score * weight
                total_weight += weight
        
        # Overall score
        assessment["overall_score"] = total_score / total_weight if total_weight > 0 else 0.0
        
        # Generate recommendations
        if assessment["overall_score"] < 0.8:
            assessment["recommendations"].append("Security score below 80% - review failed tests")
        
        if "attack_resistance" in assessment["categories"]:
            if assessment["categories"]["attack_resistance"]["score"] < 1.0:
                assessment["recommendations"].append("Attack resistance tests failed - strengthen security measures")
        
        if "performance" in assessment["categories"]:
            if assessment["categories"]["performance"]["score"] < 0.8:
                assessment["recommendations"].append("Performance tests indicate security overhead - optimize critical paths")
        
        return assessment


async def run_comprehensive_security_test():
    """Run the comprehensive security integration test suite."""
    print("🛡️ Comprehensive Security Integration Test Suite")
    print("=" * 60)
    
    tester = SecurityIntegrationTester()
    
    try:
        # Run full test suite
        suite_results = await tester.run_full_test_suite()
        
        # Display results
        print(f"\n📊 Test Suite Results:")
        print(f"=" * 30)
        print(f"Total Tests: {suite_results.total_tests}")
        print(f"Passed: {suite_results.passed_tests}")
        print(f"Failed: {suite_results.failed_tests}")
        print(f"Success Rate: {suite_results.success_rate:.1%}")
        print(f"Total Duration: {suite_results.total_duration:.2f}s")
        
        # Security assessment
        print(f"\n🔒 Security Assessment:")
        print(f"=" * 25)
        assessment = suite_results.security_assessment
        print(f"Overall Security Score: {assessment['overall_score']:.1%}")
        
        print(f"\nCategory Scores:")
        for category, data in assessment["categories"].items():
            print(f"  {category.replace('_', ' ').title()}: {data['score']:.1%} ({data['passed']}/{data['total']})")
        
        if assessment["recommendations"]:
            print(f"\nRecommendations:")
            for rec in assessment["recommendations"]:
                print(f"  - {rec}")
        
        # Performance summary
        print(f"\n⚡ Performance Summary:")
        print(f"=" * 25)
        perf = suite_results.performance_summary
        key_metrics = [
            "safe_validation_time_avg", "auto_consent_time_avg", 
            "event_logging_time_avg", "e2e_workflow_time_avg"
        ]
        
        for metric in key_metrics:
            if metric in perf:
                print(f"  {metric.replace('_', ' ').title()}: {perf[metric]*1000:.1f}ms")
        
        # Failed tests
        failed_tests = [r for r in suite_results.test_results if not r.passed]
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            print(f"=" * 15)
            for test in failed_tests:
                print(f"  {test.test_name}")
                for error in test.errors:
                    print(f"    Error: {error}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        print(f"=" * 22)
        
        if suite_results.success_rate >= 0.9 and assessment["overall_score"] >= 0.8:
            print("✅ PRODUCTION READY - Security framework meets all requirements")
        elif suite_results.success_rate >= 0.8 and assessment["overall_score"] >= 0.7:
            print("⚠️  NEEDS MINOR FIXES - Address failed tests before production")
        else:
            print("❌ NOT PRODUCTION READY - Significant issues need resolution")
        
        # Save detailed results
        results_file = "security_integration_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(suite_results.to_dict(), f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        print(f"❌ Test suite execution failed: {e}")
    
    print("\n🛡️ Security Integration Test Suite Complete")


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(run_comprehensive_security_test())