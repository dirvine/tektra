#!/usr/bin/env python3
"""
Tool Validation System

Static and dynamic analysis of agent tools for security validation.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with ast,loguru python validator.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "loguru>=0.7.0",
# ]
# ///

import ast
import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
import threading

from loguru import logger


class ValidationResult(Enum):
    """Tool validation results."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


class ThreatCategory(Enum):
    """Categories of security threats in tools."""
    MALICIOUS_CODE = "malicious_code"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    NETWORK_ABUSE = "network_abuse"
    RESOURCE_ABUSE = "resource_abuse"
    OBFUSCATION = "obfuscation"
    SUSPICIOUS_IMPORTS = "suspicious_imports"
    DANGEROUS_FUNCTIONS = "dangerous_functions"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in code."""
    
    category: ThreatCategory
    severity: str  # "low", "medium", "high", "critical"
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0
    mitigation: Optional[str] = None


@dataclass
class ToolValidationReport:
    """Comprehensive validation report for a tool."""
    
    tool_name: str
    validation_result: ValidationResult
    issues: List[ValidationIssue] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 to 1.0
    execution_time: float = 0.0
    code_hash: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    validation_timestamp: float = field(default_factory=time.time)
    
    @property
    def is_safe(self) -> bool:
        """Check if tool is considered safe."""
        return self.validation_result == ValidationResult.SAFE
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical severity issues."""
        return [issue for issue in self.issues if issue.severity == "critical"]
    
    @property
    def high_issues(self) -> List[ValidationIssue]:
        """Get high severity issues."""
        return [issue for issue in self.issues if issue.severity == "high"]


class ToolValidator:
    """
    Static and dynamic analysis validator for agent tools.
    
    Provides comprehensive security validation including:
    - Static code analysis
    - Malicious pattern detection
    - Import validation
    - Function call analysis
    - Risk assessment
    """
    
    def __init__(self):
        """Initialize tool validator."""
        self.validation_cache: Dict[str, ToolValidationReport] = {}
        self.approved_tools: Set[str] = set()
        self.blocked_tools: Set[str] = set()
        self._lock = threading.RLock()
        
        # Dangerous patterns and functions
        self.dangerous_imports = {
            "os": ["system", "exec", "spawn", "popen", "remove", "rmdir"],
            "subprocess": ["call", "run", "Popen", "check_call", "check_output"],
            "shutil": ["rmtree", "move", "copy2"],
            "pickle": ["load", "loads"],  # Deserialization attacks
            "eval": ["eval", "exec", "compile"],
            "importlib": ["import_module", "__import__"],
            "ctypes": ["*"],  # Low-level system access
            "socket": ["socket", "bind", "listen", "connect"],
            "urllib": ["urlopen", "urlretrieve"],
            "requests": ["get", "post", "put", "delete"],
            "tempfile": ["mktemp"],  # Insecure temp files
        }
        
        self.dangerous_functions = {
            "eval", "exec", "compile", "__import__",
            "open", "input", "raw_input",
            "getattr", "setattr", "delattr", "hasattr",
            "globals", "locals", "vars", "dir",
            "exit", "quit", "reload"
        }
        
        self.suspicious_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"compile\s*\(",
            r"base64\.decode",
            r"\.encode\s*\(\s*['\"]hex['\"]",
            r"\.decode\s*\(\s*['\"]hex['\"]",
            r"chr\s*\(\s*\d+\s*\)",
            r"ord\s*\(",
            r"\.join\s*\(\s*map\s*\(",
            r"pickle\.loads",
            r"marshal\.loads",
            r"subprocess\.",
            r"os\.system",
            r"os\.popen",
            r"shell\s*=\s*True",
            r"password|passwd|secret|key|token",
            r"127\.0\.0\.1|localhost|0\.0\.0\.0",
        ]
        
        # File extension patterns
        self.executable_extensions = {".py", ".sh", ".bat", ".exe", ".com", ".scr"}
        
        logger.info("Tool validator initialized")
    
    def validate_tool(
        self,
        tool_code: str,
        tool_name: str = "unknown",
        force_revalidate: bool = False
    ) -> ToolValidationReport:
        """
        Validate a tool's code for security issues.
        
        Args:
            tool_code: Source code of the tool
            tool_name: Name/identifier of the tool
            force_revalidate: Force revalidation even if cached
            
        Returns:
            ToolValidationReport with validation results
        """
        start_time = time.time()
        
        # Calculate code hash for caching
        code_hash = hashlib.sha256(tool_code.encode()).hexdigest()
        
        with self._lock:
            # Check cache
            if not force_revalidate and code_hash in self.validation_cache:
                cached_report = self.validation_cache[code_hash]
                logger.debug(f"Using cached validation for tool {tool_name}")
                return cached_report
            
            # Check if tool is pre-approved or blocked
            if tool_name in self.blocked_tools:
                report = ToolValidationReport(
                    tool_name=tool_name,
                    validation_result=ValidationResult.BLOCKED,
                    code_hash=code_hash,
                    execution_time=time.time() - start_time
                )
                report.issues.append(ValidationIssue(
                    category=ThreatCategory.MALICIOUS_CODE,
                    severity="critical",
                    description="Tool is in blocked list",
                    confidence=1.0
                ))
                return report
            
            if tool_name in self.approved_tools:
                report = ToolValidationReport(
                    tool_name=tool_name,
                    validation_result=ValidationResult.SAFE,
                    code_hash=code_hash,
                    execution_time=time.time() - start_time
                )
                return report
        
        # Perform validation
        issues = []
        
        try:
            # Static code analysis
            issues.extend(self._analyze_code_structure(tool_code))
            issues.extend(self._analyze_imports(tool_code))
            issues.extend(self._analyze_function_calls(tool_code))
            issues.extend(self._analyze_patterns(tool_code))
            issues.extend(self._analyze_strings(tool_code))
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(issues)
            
            # Determine validation result
            validation_result = self._determine_result(risk_score, issues)
            
            # Create report
            report = ToolValidationReport(
                tool_name=tool_name,
                validation_result=validation_result,
                issues=issues,
                risk_score=risk_score,
                code_hash=code_hash,
                execution_time=time.time() - start_time
            )
            
            # Cache result
            with self._lock:
                self.validation_cache[code_hash] = report
            
            logger.info(f"Tool validation completed: {tool_name} -> {validation_result.value} (Risk: {risk_score:.2f})")
            return report
            
        except Exception as e:
            logger.error(f"Error validating tool {tool_name}: {e}")
            
            # Return dangerous result on error
            report = ToolValidationReport(
                tool_name=tool_name,
                validation_result=ValidationResult.DANGEROUS,
                code_hash=code_hash,
                execution_time=time.time() - start_time
            )
            report.issues.append(ValidationIssue(
                category=ThreatCategory.MALICIOUS_CODE,
                severity="high",
                description=f"Validation error: {str(e)}",
                confidence=0.8
            ))
            return report
    
    def approve_tool(self, tool_name: str, approved_by: str) -> bool:
        """
        Approve a tool for execution.
        
        Args:
            tool_name: Name of the tool to approve
            approved_by: Identifier of who approved the tool
            
        Returns:
            True if approved successfully, False otherwise
        """
        with self._lock:
            self.approved_tools.add(tool_name)
            self.blocked_tools.discard(tool_name)
            
            # Update cached reports
            for report in self.validation_cache.values():
                if report.tool_name == tool_name:
                    report.validation_result = ValidationResult.SAFE
                    report.approved_by = approved_by
                    report.approved_at = time.time()
            
            logger.info(f"Tool approved: {tool_name} by {approved_by}")
            return True
    
    def block_tool(self, tool_name: str, reason: str = "") -> bool:
        """
        Block a tool from execution.
        
        Args:
            tool_name: Name of the tool to block
            reason: Reason for blocking
            
        Returns:
            True if blocked successfully, False otherwise
        """
        with self._lock:
            self.blocked_tools.add(tool_name)
            self.approved_tools.discard(tool_name)
            
            # Update cached reports
            for report in self.validation_cache.values():
                if report.tool_name == tool_name:
                    report.validation_result = ValidationResult.BLOCKED
                    if reason:
                        report.issues.append(ValidationIssue(
                            category=ThreatCategory.MALICIOUS_CODE,
                            severity="critical",
                            description=f"Blocked: {reason}",
                            confidence=1.0
                        ))
            
            logger.warning(f"Tool blocked: {tool_name} - {reason}")
            return True
    
    def get_validation_status(self, tool_name: str) -> Optional[str]:
        """
        Get validation status for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Validation status or None if not found
        """
        with self._lock:
            if tool_name in self.approved_tools:
                return "approved"
            elif tool_name in self.blocked_tools:
                return "blocked"
            else:
                return "unknown"
    
    def _analyze_code_structure(self, code: str) -> List[ValidationIssue]:
        """Analyze code structure using AST."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.dangerous_functions:
                            issues.append(ValidationIssue(
                                category=ThreatCategory.DANGEROUS_FUNCTIONS,
                                severity="high" if func_name in ["eval", "exec"] else "medium",
                                description=f"Dangerous function call: {func_name}",
                                line_number=getattr(node, 'lineno', None),
                                confidence=0.9
                            ))
                
                # Check for dynamic imports
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    if hasattr(node, 'names'):
                        for alias in node.names:
                            module_name = alias.name
                            if module_name in ["importlib", "imp"]:
                                issues.append(ValidationIssue(
                                    category=ThreatCategory.SUSPICIOUS_IMPORTS,
                                    severity="medium",
                                    description=f"Dynamic import capability: {module_name}",
                                    line_number=getattr(node, 'lineno', None),
                                    confidence=0.7
                                ))
                
                # Check for attribute access that might be dangerous
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id == "__builtins__":
                            issues.append(ValidationIssue(
                                category=ThreatCategory.PRIVILEGE_ESCALATION,
                                severity="high",
                                description="Access to __builtins__",
                                line_number=getattr(node, 'lineno', None),
                                confidence=0.8
                            ))
        
        except SyntaxError as e:
            issues.append(ValidationIssue(
                category=ThreatCategory.MALICIOUS_CODE,
                severity="medium",
                description=f"Syntax error in code: {str(e)}",
                confidence=0.6
            ))
        
        return issues
    
    def _analyze_imports(self, code: str) -> List[ValidationIssue]:
        """Analyze import statements for dangerous modules."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        self._check_dangerous_import(module_name, issues, getattr(node, 'lineno', None))
                
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    
                    # Check the module itself
                    self._check_dangerous_import(module_name, issues, getattr(node, 'lineno', None))
                    
                    # Check specific imports
                    for alias in node.names:
                        import_name = alias.name
                        if module_name in self.dangerous_imports:
                            dangerous_funcs = self.dangerous_imports[module_name]
                            if "*" in dangerous_funcs or import_name in dangerous_funcs:
                                severity = "critical" if "*" in dangerous_funcs else "high"
                                issues.append(ValidationIssue(
                                    category=ThreatCategory.DANGEROUS_FUNCTIONS,
                                    severity=severity,
                                    description=f"Dangerous import: {module_name}.{import_name}",
                                    line_number=getattr(node, 'lineno', None),
                                    confidence=0.9
                                ))
        
        except SyntaxError:
            pass  # Already handled in structure analysis
        
        return issues
    
    def _check_dangerous_import(self, module_name: str, issues: List[ValidationIssue], line_number: Optional[int]) -> None:
        """Check if a module import is dangerous."""
        if module_name in self.dangerous_imports:
            severity = "critical" if module_name in ["os", "subprocess"] else "medium"
            issues.append(ValidationIssue(
                category=ThreatCategory.SUSPICIOUS_IMPORTS,
                severity=severity,
                description=f"Potentially dangerous module import: {module_name}",
                line_number=line_number,
                confidence=0.7
            ))
    
    def _analyze_function_calls(self, code: str) -> List[ValidationIssue]:
        """Analyze function calls for dangerous patterns."""
        issues = []
        
        # Pattern-based analysis for function calls
        patterns = {
            r"subprocess\.": ("DANGEROUS_FUNCTIONS", "high", "Subprocess call detected"),
            r"os\.system\s*\(": ("DANGEROUS_FUNCTIONS", "critical", "os.system call detected"),
            r"eval\s*\(": ("DANGEROUS_FUNCTIONS", "critical", "eval() call detected"),
            r"exec\s*\(": ("DANGEROUS_FUNCTIONS", "critical", "exec() call detected"),
            r"open\s*\([^)]*['\"]w": ("SUSPICIOUS_IMPORTS", "medium", "File write operation"),
            r"requests\.(get|post|put|delete)": ("NETWORK_ABUSE", "medium", "HTTP request detected"),
            r"socket\.": ("NETWORK_ABUSE", "high", "Socket operation detected"),
        }
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, (category, severity, description) in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(ValidationIssue(
                        category=ThreatCategory[category],
                        severity=severity,
                        description=description,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        confidence=0.8
                    ))
        
        return issues
    
    def _analyze_patterns(self, code: str) -> List[ValidationIssue]:
        """Analyze code for suspicious patterns."""
        issues = []
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in self.suspicious_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Determine category and severity based on pattern
                    if "eval" in pattern or "exec" in pattern:
                        category = ThreatCategory.DANGEROUS_FUNCTIONS
                        severity = "critical"
                    elif "base64" in pattern or "encode" in pattern:
                        category = ThreatCategory.OBFUSCATION
                        severity = "medium"
                    elif "subprocess" in pattern or "system" in pattern:
                        category = ThreatCategory.DANGEROUS_FUNCTIONS
                        severity = "high"
                    elif "password" in pattern or "secret" in pattern:
                        category = ThreatCategory.DATA_EXFILTRATION
                        severity = "medium"
                    else:
                        category = ThreatCategory.MALICIOUS_CODE
                        severity = "low"
                    
                    issues.append(ValidationIssue(
                        category=category,
                        severity=severity,
                        description=f"Suspicious pattern detected: {match.group()}",
                        line_number=line_num,
                        code_snippet=line.strip(),
                        confidence=0.6
                    ))
        
        return issues
    
    def _analyze_strings(self, code: str) -> List[ValidationIssue]:
        """Analyze string literals for suspicious content."""
        issues = []
        
        # Look for encoded strings, URLs, file paths, etc.
        patterns = {
            r"['\"][A-Za-z0-9+/]{20,}={0,2}['\"]": ("Base64-encoded string", "medium"),
            r"['\"]https?://[^'\"]+['\"]": ("HTTP URL in string", "low"),
            r"['\"][a-fA-F0-9]{32,}['\"]": ("Hexadecimal string (possible hash)", "low"),
            r"['\"]/(etc|usr|var|tmp)/[^'\"]*['\"]": ("System path reference", "medium"),
            r"['\"]\\x[0-9a-fA-F]{2}": ("Hex-encoded string", "medium"),
        }
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, (description, severity) in patterns.items():
                matches = re.finditer(pattern, line)
                for match in matches:
                    issues.append(ValidationIssue(
                        category=ThreatCategory.OBFUSCATION,
                        severity=severity,
                        description=description,
                        line_number=line_num,
                        code_snippet=match.group(),
                        confidence=0.5
                    ))
        
        return issues
    
    def _calculate_risk_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall risk score based on issues."""
        if not issues:
            return 0.0
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }
        
        total_score = 0.0
        for issue in issues:
            weight = severity_weights.get(issue.severity, 0.1)
            confidence = issue.confidence
            total_score += weight * confidence
        
        # Normalize to 0-1 range
        max_possible_score = len(issues) * 1.0
        normalized_score = min(1.0, total_score / max(max_possible_score, 1.0))
        
        return normalized_score
    
    def _determine_result(self, risk_score: float, issues: List[ValidationIssue]) -> ValidationResult:
        """Determine validation result based on risk score and issues."""
        # Check for critical issues
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        if critical_issues:
            return ValidationResult.DANGEROUS
        
        # Check for high-risk issues
        high_issues = [issue for issue in issues if issue.severity == "high"]
        if len(high_issues) >= 3 or risk_score > 0.8:
            return ValidationResult.DANGEROUS
        
        # Check for suspicious patterns
        if risk_score > 0.5 or len(high_issues) > 0:
            return ValidationResult.SUSPICIOUS
        
        # Check for medium-risk issues
        medium_issues = [issue for issue in issues if issue.severity == "medium"]
        if len(medium_issues) >= 5 or risk_score > 0.3:
            return ValidationResult.SUSPICIOUS
        
        return ValidationResult.SAFE


def create_tool_validator() -> ToolValidator:
    """
    Create a tool validator instance.
    
    Returns:
        ToolValidator instance
    """
    return ToolValidator()


if __name__ == "__main__":
    def demo_tool_validator():
        """Demonstrate tool validator functionality."""
        print("🔍 Tool Validator Demo")
        print("=" * 40)
        
        # Create validator
        validator = create_tool_validator()
        
        # Test cases
        test_cases = [
            # Safe code
            ("safe_tool", """
def calculate_sum(a, b):
    return a + b

def main():
    result = calculate_sum(5, 3)
    print(f"Result: {result}")
"""),
            
            # Suspicious code
            ("suspicious_tool", """
import requests
import base64

def fetch_data(url):
    response = requests.get(url)
    return response.text

def decode_secret(encoded):
    return base64.b64decode(encoded)
"""),
            
            # Dangerous code
            ("dangerous_tool", """
import os
import subprocess

def execute_command(cmd):
    return os.system(cmd)

def run_script(script):
    return subprocess.call(script, shell=True)

def dangerous_eval(code):
    return eval(code)
"""),
        ]
        
        for tool_name, code in test_cases:
            print(f"\nValidating: {tool_name}")
            print("-" * 30)
            
            report = validator.validate_tool(code, tool_name)
            
            print(f"Result: {report.validation_result.value}")
            print(f"Risk Score: {report.risk_score:.2f}")
            print(f"Issues Found: {len(report.issues)}")
            
            # Show issues
            for issue in report.issues:
                print(f"  - {issue.severity.upper()}: {issue.description}")
                if issue.line_number:
                    print(f"    Line {issue.line_number}: {issue.code_snippet}")
            
            # Test approval
            if report.validation_result == ValidationResult.SUSPICIOUS:
                print(f"Approving {tool_name}...")
                validator.approve_tool(tool_name, "admin")
                status = validator.get_validation_status(tool_name)
                print(f"Status after approval: {status}")
        
        print("\n🔍 Tool Validator Demo Complete")
    
    # Run demo
    demo_tool_validator()