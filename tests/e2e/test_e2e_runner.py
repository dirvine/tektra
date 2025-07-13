#!/usr/bin/env python3
"""
Tektra AI Assistant - E2E Test Suite Runner

Comprehensive end-to-end test suite runner that orchestrates all testing phases
including system integration, security compliance, performance benchmarks, and
production deployment validation.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with pytest,pytest-asyncio,pytest-html,pytest-cov,loguru python test_e2e_runner.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest>=7.4.0",
#     "pytest-asyncio>=0.21.0",
#     "pytest-html>=3.2.0",
#     "pytest-cov>=4.1.0",
#     "pytest-xdist>=3.3.0",
#     "loguru>=0.7.0",
#     "rich>=13.0.0",
#     "tabulate>=0.9.0",
# ]
# ///

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pytest

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from tabulate import tabulate


@dataclass
class TestResult:
    """Individual test result data structure."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration_seconds: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Test suite result data structure."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    coverage_percent: Optional[float] = None
    test_results: List[TestResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests
    
    @property
    def status(self) -> str:
        """Get overall suite status."""
        if self.errors > 0:
            return "ERROR"
        elif self.failed > 0:
            return "FAILED"
        elif self.skipped == self.total_tests:
            return "SKIPPED"
        elif self.passed == self.total_tests:
            return "PASSED"
        else:
            return "PARTIAL"


@dataclass
class E2ETestReport:
    """Complete E2E test report."""
    start_time: datetime
    end_time: Optional[datetime] = None
    environment: str = "development"
    total_duration_seconds: float = 0.0
    suite_results: List[TestSuiteResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tests(self) -> int:
        return sum(suite.total_tests for suite in self.suite_results)
    
    @property
    def total_passed(self) -> int:
        return sum(suite.passed for suite in self.suite_results)
    
    @property
    def total_failed(self) -> int:
        return sum(suite.failed for suite in self.suite_results)
    
    @property
    def total_skipped(self) -> int:
        return sum(suite.skipped for suite in self.suite_results)
    
    @property
    def total_errors(self) -> int:
        return sum(suite.errors for suite in self.suite_results)
    
    @property
    def overall_success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.total_passed / self.total_tests
    
    @property
    def overall_status(self) -> str:
        if self.total_errors > 0:
            return "ERROR"
        elif self.total_failed > 0:
            return "FAILED"
        elif self.total_skipped == self.total_tests:
            return "SKIPPED"
        elif self.total_passed == self.total_tests:
            return "PASSED"
        else:
            return "PARTIAL"


class E2ETestRunner:
    """Comprehensive E2E test suite runner."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.console = Console()
        self.report = E2ETestReport(start_time=datetime.now())
        
        # Test suite configuration
        self.test_suites = {
            "system_integration": {
                "file": "test_complete_system_integration.py",
                "description": "Complete system integration tests",
                "required": True,
                "timeout": 300,
                "markers": ["integration"]
            },
            "security_compliance": {
                "file": "test_security_compliance.py", 
                "description": "Security compliance and penetration tests",
                "required": True,
                "timeout": 600,
                "markers": ["security", "compliance"]
            },
            "performance_benchmarks": {
                "file": "test_performance_benchmarks.py",
                "description": "Performance benchmarks and load tests",
                "required": False,
                "timeout": 900,
                "markers": ["performance", "benchmark"]
            },
            "production_deployment": {
                "file": "test_production_deployment.py",
                "description": "Production deployment validation",
                "required": True,
                "timeout": 300,
                "markers": ["deployment", "production"]
            }
        }
        
        # Initialize system info
        self._collect_system_info()
    
    def _collect_system_info(self):
        """Collect system information for the report."""
        import platform
        import psutil
        
        self.report.system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
            "hostname": platform.node(),
            "architecture": platform.architecture()[0]
        }
    
    def _display_header(self):
        """Display test runner header."""
        header_text = """
🌟 Tektra AI Assistant - End-to-End Test Suite
Complete system validation including integration, security, performance, and deployment
        """
        
        self.console.print(Panel(
            header_text.strip(),
            title="E2E Test Runner",
            border_style="bright_blue",
            padding=(1, 2)
        ))
        
        # Display system information
        system_table = Table(title="System Information", show_header=False)
        system_table.add_column("Property", style="cyan")
        system_table.add_column("Value", style="white")
        
        for key, value in self.report.system_info.items():
            system_table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(system_table)
        self.console.print()
    
    def _run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> TestSuiteResult:
        """Run a single test suite."""
        test_file = self.project_root / "tests" / "e2e" / suite_config["file"]
        
        if not test_file.exists():
            logger.warning(f"Test file not found: {test_file}")
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=1,
                errors=0,
                duration_seconds=0.0
            )
        
        # Prepare pytest arguments
        pytest_args = [
            str(test_file),
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            f"--timeout={suite_config.get('timeout', 300)}",
            "--html=reports/pytest_report.html",
            "--self-contained-html",
            "--cov=tektra",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage"
        ]
        
        # Add markers if specified
        if "markers" in suite_config:
            for marker in suite_config["markers"]:
                pytest_args.extend(["-m", marker])
        
        logger.info(f"Running test suite: {suite_name}")
        start_time = time.time()
        
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest"
        ] + pytest_args, 
        capture_output=True, 
        text=True,
        cwd=self.project_root
        )
        
        duration = time.time() - start_time
        
        # Parse pytest output for results
        suite_result = self._parse_pytest_output(
            suite_name, 
            result.stdout, 
            result.stderr, 
            result.returncode,
            duration
        )
        
        return suite_result
    
    def _parse_pytest_output(self, suite_name: str, stdout: str, stderr: str, 
                           returncode: int, duration: float) -> TestSuiteResult:
        """Parse pytest output to extract test results."""
        # Default result
        suite_result = TestSuiteResult(
            suite_name=suite_name,
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration_seconds=duration
        )
        
        # Parse output for test counts
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for result summary line
            if "failed" in line and "passed" in line:
                # Example: "1 failed, 4 passed, 2 skipped in 10.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed," and i > 0:
                        suite_result.failed = int(parts[i-1])
                    elif part == "passed," and i > 0:
                        suite_result.passed = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        suite_result.skipped = int(parts[i-1])
                    elif part == "error" and i > 0:
                        suite_result.errors = int(parts[i-1])
            
            elif "passed in" in line:
                # Example: "5 passed in 10.5s"
                parts = line.split()
                if len(parts) >= 1 and parts[0].isdigit():
                    suite_result.passed = int(parts[0])
            
            # Extract coverage information
            elif "TOTAL" in line and "%" in line:
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            suite_result.coverage_percent = float(part[:-1])
                        except ValueError:
                            pass
        
        # Calculate total tests
        suite_result.total_tests = (
            suite_result.passed + 
            suite_result.failed + 
            suite_result.skipped + 
            suite_result.errors
        )
        
        # If we couldn't parse anything and there was an error, mark as error
        if suite_result.total_tests == 0 and returncode != 0:
            suite_result.errors = 1
            suite_result.total_tests = 1
        
        return suite_result
    
    def _display_suite_progress(self, suite_name: str, suite_config: Dict[str, Any]):
        """Display progress for current test suite."""
        description = suite_config.get("description", suite_name)
        required = "Required" if suite_config.get("required", False) else "Optional"
        
        self.console.print(f"\n🧪 Running: {description}")
        self.console.print(f"   Status: {required}")
        self.console.print(f"   Timeout: {suite_config.get('timeout', 300)}s")
    
    def _display_suite_result(self, result: TestSuiteResult):
        """Display results for a completed test suite."""
        # Status emoji
        status_emoji = {
            "PASSED": "✅",
            "FAILED": "❌", 
            "ERROR": "💥",
            "SKIPPED": "⏭️",
            "PARTIAL": "⚠️"
        }
        
        emoji = status_emoji.get(result.status, "❓")
        
        self.console.print(f"{emoji} {result.suite_name}: {result.status}")
        self.console.print(f"   Tests: {result.total_tests} total, "
                          f"{result.passed} passed, "
                          f"{result.failed} failed, "
                          f"{result.skipped} skipped")
        self.console.print(f"   Duration: {result.duration_seconds:.2f}s")
        self.console.print(f"   Success Rate: {result.success_rate:.1%}")
        
        if result.coverage_percent is not None:
            self.console.print(f"   Coverage: {result.coverage_percent:.1f}%")
    
    def _display_final_report(self):
        """Display final test report."""
        self.console.print("\n" + "="*60)
        self.console.print()
        
        # Overall status
        status_emoji = {
            "PASSED": "✅",
            "FAILED": "❌",
            "ERROR": "💥", 
            "SKIPPED": "⏭️",
            "PARTIAL": "⚠️"
        }
        
        emoji = status_emoji.get(self.report.overall_status, "❓")
        
        final_panel = Panel(
            f"{emoji} Overall Status: {self.report.overall_status}\n"
            f"Success Rate: {self.report.overall_success_rate:.1%}\n"
            f"Total Duration: {self.report.total_duration_seconds:.2f}s",
            title="E2E Test Results",
            border_style="bright_green" if self.report.overall_status == "PASSED" else "bright_red"
        )
        
        self.console.print(final_panel)
        
        # Detailed results table
        results_table = Table(title="Test Suite Results")
        results_table.add_column("Suite", style="cyan")
        results_table.add_column("Status", style="white")
        results_table.add_column("Tests", justify="right")
        results_table.add_column("Passed", justify="right", style="green")
        results_table.add_column("Failed", justify="right", style="red")
        results_table.add_column("Skipped", justify="right", style="yellow")
        results_table.add_column("Duration", justify="right")
        results_table.add_column("Success Rate", justify="right")
        
        for suite_result in self.report.suite_results:
            results_table.add_row(
                suite_result.suite_name,
                suite_result.status,
                str(suite_result.total_tests),
                str(suite_result.passed),
                str(suite_result.failed),
                str(suite_result.skipped),
                f"{suite_result.duration_seconds:.2f}s",
                f"{suite_result.success_rate:.1%}"
            )
        
        self.console.print("\n")
        self.console.print(results_table)
        
        # Summary statistics
        summary_table = Table(title="Summary Statistics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Test Suites", str(len(self.report.suite_results)))
        summary_table.add_row("Total Tests", str(self.report.total_tests))
        summary_table.add_row("Total Passed", str(self.report.total_passed))
        summary_table.add_row("Total Failed", str(self.report.total_failed))
        summary_table.add_row("Total Skipped", str(self.report.total_skipped))
        summary_table.add_row("Total Errors", str(self.report.total_errors))
        summary_table.add_row("Overall Success Rate", f"{self.report.overall_success_rate:.1%}")
        summary_table.add_row("Total Duration", f"{self.report.total_duration_seconds:.2f}s")
        
        self.console.print("\n")
        self.console.print(summary_table)
    
    def _save_report(self):
        """Save test report to files."""
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_report = {
            "start_time": self.report.start_time.isoformat(),
            "end_time": self.report.end_time.isoformat() if self.report.end_time else None,
            "environment": self.report.environment,
            "total_duration_seconds": self.report.total_duration_seconds,
            "overall_status": self.report.overall_status,
            "overall_success_rate": self.report.overall_success_rate,
            "total_tests": self.report.total_tests,
            "total_passed": self.report.total_passed,
            "total_failed": self.report.total_failed,
            "total_skipped": self.report.total_skipped,
            "total_errors": self.report.total_errors,
            "system_info": self.report.system_info,
            "suite_results": [
                {
                    "suite_name": suite.suite_name,
                    "status": suite.status,
                    "total_tests": suite.total_tests,
                    "passed": suite.passed,
                    "failed": suite.failed,
                    "skipped": suite.skipped,
                    "errors": suite.errors,
                    "duration_seconds": suite.duration_seconds,
                    "success_rate": suite.success_rate,
                    "coverage_percent": suite.coverage_percent
                }
                for suite in self.report.suite_results
            ]
        }
        
        json_file = reports_dir / "e2e_test_report.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"JSON report saved to: {json_file}")
        
        # Save markdown report
        markdown_content = self._generate_markdown_report()
        markdown_file = reports_dir / "e2e_test_report.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved to: {markdown_file}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown test report."""
        md_content = f"""# Tektra AI Assistant - E2E Test Report

**Generated:** {datetime.now().isoformat()}  
**Environment:** {self.report.environment}  
**Overall Status:** {self.report.overall_status}  
**Success Rate:** {self.report.overall_success_rate:.1%}  
**Total Duration:** {self.report.total_duration_seconds:.2f}s  

## Summary

- **Total Test Suites:** {len(self.report.suite_results)}
- **Total Tests:** {self.report.total_tests}
- **Passed:** {self.report.total_passed}
- **Failed:** {self.report.total_failed}
- **Skipped:** {self.report.total_skipped}
- **Errors:** {self.report.total_errors}

## System Information

| Property | Value |
|----------|-------|
"""
        
        for key, value in self.report.system_info.items():
            md_content += f"| {key.replace('_', ' ').title()} | {value} |\n"
        
        md_content += "\n## Test Suite Results\n\n"
        
        # Results table
        headers = ["Suite", "Status", "Tests", "Passed", "Failed", "Skipped", "Duration", "Success Rate"]
        table_data = []
        
        for suite in self.report.suite_results:
            table_data.append([
                suite.suite_name,
                suite.status,
                str(suite.total_tests),
                str(suite.passed),
                str(suite.failed),
                str(suite.skipped),
                f"{suite.duration_seconds:.2f}s",
                f"{suite.success_rate:.1%}"
            ])
        
        md_content += tabulate(table_data, headers=headers, tablefmt="github")
        
        # Detailed results
        md_content += "\n\n## Detailed Results\n\n"
        
        for suite in self.report.suite_results:
            status_emoji = {
                "PASSED": "✅",
                "FAILED": "❌",
                "ERROR": "💥",
                "SKIPPED": "⏭️", 
                "PARTIAL": "⚠️"
            }
            
            emoji = status_emoji.get(suite.status, "❓")
            
            md_content += f"### {emoji} {suite.suite_name}\n\n"
            md_content += f"- **Status:** {suite.status}\n"
            md_content += f"- **Tests:** {suite.total_tests} total\n"
            md_content += f"- **Passed:** {suite.passed}\n"
            md_content += f"- **Failed:** {suite.failed}\n"
            md_content += f"- **Skipped:** {suite.skipped}\n"
            md_content += f"- **Duration:** {suite.duration_seconds:.2f}s\n"
            md_content += f"- **Success Rate:** {suite.success_rate:.1%}\n"
            
            if suite.coverage_percent is not None:
                md_content += f"- **Coverage:** {suite.coverage_percent:.1f}%\n"
            
            md_content += "\n"
        
        return md_content
    
    def run_all_suites(self, include_optional: bool = False, 
                      specific_suites: Optional[List[str]] = None) -> bool:
        """Run all test suites."""
        self._display_header()
        
        # Determine which suites to run
        suites_to_run = {}
        
        if specific_suites:
            # Run only specified suites
            for suite_name in specific_suites:
                if suite_name in self.test_suites:
                    suites_to_run[suite_name] = self.test_suites[suite_name]
                else:
                    logger.warning(f"Unknown test suite: {suite_name}")
        else:
            # Run based on required/optional flag
            for suite_name, suite_config in self.test_suites.items():
                if suite_config.get("required", False) or include_optional:
                    suites_to_run[suite_name] = suite_config
        
        if not suites_to_run:
            self.console.print("❌ No test suites to run")
            return False
        
        self.console.print(f"🚀 Running {len(suites_to_run)} test suite(s)")
        
        # Run each test suite
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Running test suites...", total=len(suites_to_run))
            
            for suite_name, suite_config in suites_to_run.items():
                progress.update(task, description=f"Running {suite_name}...")
                
                self._display_suite_progress(suite_name, suite_config)
                
                suite_result = self._run_test_suite(suite_name, suite_config)
                self.report.suite_results.append(suite_result)
                
                self._display_suite_result(suite_result)
                
                progress.advance(task)
        
        # Finalize report
        self.report.end_time = datetime.now()
        self.report.total_duration_seconds = (
            self.report.end_time - self.report.start_time
        ).total_seconds()
        
        # Display and save final report
        self._display_final_report()
        self._save_report()
        
        # Return success status
        return self.report.overall_status in ["PASSED", "PARTIAL"]


def main():
    """Main entry point for E2E test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tektra E2E Test Suite Runner")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional test suites (e.g., performance benchmarks)"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        help="Specific test suites to run",
        choices=["system_integration", "security_compliance", "performance_benchmarks", "production_deployment"]
    )
    parser.add_argument(
        "--environment",
        default="development",
        help="Test environment (development, staging, production)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = E2ETestRunner(project_root=args.project_root)
    runner.report.environment = args.environment
    
    # Run test suites
    success = runner.run_all_suites(
        include_optional=args.include_optional,
        specific_suites=args.suites
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()