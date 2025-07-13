#!/usr/bin/env python3
"""
Tektra AI Assistant - E2E Test Suite Runner Script

Convenient script to run the complete end-to-end test suite with proper
environment setup and comprehensive reporting.

Usage:
    python run_e2e_tests.py                    # Run required tests only
    python run_e2e_tests.py --all              # Run all tests including optional
    python run_e2e_tests.py --security         # Run security tests only
    python run_e2e_tests.py --performance      # Run performance tests only
    python run_e2e_tests.py --help             # Show help

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run python run_e2e_tests.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest>=7.4.0",
#     "pytest-asyncio>=0.21.0",
#     "pytest-html>=3.2.0",
#     "pytest-cov>=4.1.0",
#     "rich>=13.0.0",
# ]
# ///

import argparse
import sys
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def run_test_command(args: List[str], description: str) -> bool:
    """Run a test command and return success status."""
    console.print(f"\n🚀 {description}")
    console.print(f"Command: {' '.join(args)}")
    
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        console.print("✅ Success", style="green")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Failed with exit code {e.returncode}", style="red")
        if e.stdout:
            console.print("STDOUT:", style="yellow")
            console.print(e.stdout)
        if e.stderr:
            console.print("STDERR:", style="red")
            console.print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Tektra AI Assistant E2E Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_e2e_tests.py                    # Run required tests only
  python run_e2e_tests.py --all              # Run all tests including optional
  python run_e2e_tests.py --security         # Run security tests only
  python run_e2e_tests.py --performance      # Run performance tests only
  python run_e2e_tests.py --fast             # Run quick tests only
  python run_e2e_tests.py --integration      # Run integration tests only
        """
    )
    
    # Test selection options
    parser.add_argument("--all", action="store_true",
                       help="Run all tests including optional/slow tests")
    parser.add_argument("--security", action="store_true",
                       help="Run security and compliance tests only")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance and benchmark tests only")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests only")
    parser.add_argument("--deployment", action="store_true",
                       help="Run deployment tests only")
    parser.add_argument("--fast", action="store_true",
                       help="Run fast tests only (skip slow/heavy tests)")
    
    # Test configuration options
    parser.add_argument("--no-cov", action="store_true",
                       help="Disable coverage reporting")
    parser.add_argument("--no-html", action="store_true",
                       help="Disable HTML report generation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--parallel", "-n", type=int, default=1,
                       help="Number of parallel workers (default: 1)")
    
    # Environment options
    parser.add_argument("--environment", default="testing",
                       choices=["testing", "development", "staging"],
                       help="Test environment (default: testing)")
    parser.add_argument("--no-heavy-models", action="store_true",
                       help="Skip tests requiring heavy AI models")
    
    args = parser.parse_args()
    
    # Display header
    header_text = Text("Tektra AI Assistant - E2E Test Suite", style="bold blue")
    console.print(Panel(header_text, expand=False))
    
    # Determine project root
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    # Base pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("--tb=short")
    
    # Add parallel execution
    if args.parallel > 1:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage if not disabled
    if not args.no_cov:
        pytest_cmd.extend([
            "--cov=tektra",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage"
        ])
    
    # Add HTML report if not disabled
    if not args.no_html:
        pytest_cmd.extend([
            "--html=reports/test_report.html",
            "--self-contained-html"
        ])
    
    # Add asyncio mode
    pytest_cmd.append("--asyncio-mode=auto")
    
    # Add environment variables
    env = os.environ.copy()
    env["TEKTRA_ENV"] = args.environment
    
    if args.no_heavy_models:
        pytest_cmd.append("--no-heavy-models")
    
    # Determine which tests to run
    test_selections = []
    
    if args.security:
        test_selections.append("security")
    if args.performance:
        test_selections.append("performance")
    if args.integration:
        test_selections.append("integration")
    if args.deployment:
        test_selections.append("deployment")
    
    success_results = []
    
    if args.fast:
        # Run fast tests only
        cmd = pytest_cmd + [
            "-m", "not slow and not heavy",
            "tests/"
        ]
        success = run_test_command(cmd, "Running Fast Tests")
        success_results.append(("Fast Tests", success))
    
    elif test_selections:
        # Run specific test categories
        for selection in test_selections:
            if selection == "security":
                cmd = pytest_cmd + [
                    "-m", "security",
                    "tests/e2e/test_security_compliance.py"
                ]
                success = run_test_command(cmd, "Running Security & Compliance Tests")
                success_results.append(("Security Tests", success))
            
            elif selection == "performance":
                cmd = pytest_cmd + [
                    "-m", "performance or benchmark",
                    "tests/e2e/test_performance_benchmarks.py"
                ]
                if not args.all:
                    cmd.extend(["--benchmark-skip"])
                success = run_test_command(cmd, "Running Performance & Benchmark Tests")
                success_results.append(("Performance Tests", success))
            
            elif selection == "integration":
                cmd = pytest_cmd + [
                    "-m", "integration or e2e",
                    "tests/e2e/test_complete_system_integration.py"
                ]
                success = run_test_command(cmd, "Running Integration Tests")
                success_results.append(("Integration Tests", success))
            
            elif selection == "deployment":
                cmd = pytest_cmd + [
                    "-m", "deployment",
                    "tests/e2e/test_production_deployment.py"
                ]
                success = run_test_command(cmd, "Running Deployment Tests")
                success_results.append(("Deployment Tests", success))
    
    else:
        # Run the comprehensive E2E test suite
        if args.all:
            # Run all tests including optional ones
            cmd = pytest_cmd + [
                "--include-slow",
                "tests/e2e/"
            ]
            success = run_test_command(cmd, "Running Complete E2E Test Suite (All Tests)")
            success_results.append(("Complete E2E Suite", success))
        else:
            # Run required tests only
            required_tests = [
                ("System Integration", "tests/e2e/test_complete_system_integration.py"),
                ("Security Compliance", "tests/e2e/test_security_compliance.py"),
                ("Production Deployment", "tests/e2e/test_production_deployment.py")
            ]
            
            for test_name, test_file in required_tests:
                cmd = pytest_cmd + [test_file]
                success = run_test_command(cmd, f"Running {test_name} Tests")
                success_results.append((test_name, success))
    
    # Also run the E2E test runner if it exists
    e2e_runner = project_root / "tests" / "e2e" / "test_e2e_runner.py"
    if e2e_runner.exists() and not test_selections and not args.fast:
        console.print("\n🎯 Running E2E Test Runner")
        
        runner_args = [sys.executable, str(e2e_runner)]
        if args.all:
            runner_args.append("--include-optional")
        if test_selections:
            runner_args.extend(["--suites"] + test_selections)
        runner_args.extend(["--environment", args.environment])
        
        try:
            result = subprocess.run(runner_args, check=True, env=env)
            console.print("✅ E2E Test Runner completed successfully", style="green")
            success_results.append(("E2E Test Runner", True))
        except subprocess.CalledProcessError as e:
            console.print(f"❌ E2E Test Runner failed with exit code {e.returncode}", style="red")
            success_results.append(("E2E Test Runner", False))
    
    # Display final results
    console.print("\n" + "="*60)
    console.print("📊 Test Results Summary", style="bold")
    console.print("="*60)
    
    all_passed = True
    for test_name, success in success_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        style = "green" if success else "red"
        console.print(f"{test_name:.<50} {status}", style=style)
        if not success:
            all_passed = False
    
    console.print("="*60)
    
    if all_passed:
        console.print("🎉 All tests passed successfully!", style="bold green")
        console.print("\n📁 Reports generated:")
        console.print("  - HTML Report: reports/test_report.html")
        console.print("  - Coverage Report: reports/coverage/index.html")
        console.print("  - E2E Report: reports/e2e_test_report.md")
        
        return 0
    else:
        console.print("💥 Some tests failed!", style="bold red")
        console.print("\n🔍 Check the output above for details")
        console.print("📁 Reports may still be available in the reports/ directory")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())