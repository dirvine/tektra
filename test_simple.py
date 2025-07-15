#!/usr/bin/env python3
"""
Test basic agent functionality without dependencies.
"""

import asyncio
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

class AgentStatus(Enum):
    """Agent execution status."""
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SimpleAgentSpec:
    """Specification for a simple agent."""
    id: str
    name: str
    description: str
    agent_type: str
    code: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class TestPythonAgent:
    """Simplified test version of PythonAgent."""

    def __init__(self, spec: SimpleAgentSpec):
        self.spec = spec
        self.status = AgentStatus.CREATED
        self.timeout = 30
        self.working_directory = Path(tempfile.mkdtemp(prefix=f"agent_{self.spec.id}_"))
        
        print(f"Test Python agent '{spec.name}' initialized")

    def _validate_code_safety(self, code: str) -> bool:
        """Validate code for basic safety."""
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'open(',
            'exec(',
            'eval(',
        ]
        
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                print(f"Code validation failed: found dangerous pattern '{pattern}'")
                return False
        
        return True

    async def execute_code(self, code: str) -> AgentResult:
        """Execute Python code safely."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.status = AgentStatus.RUNNING
            
            # Validate code safety
            if not self._validate_code_safety(code):
                return AgentResult(
                    success=False,
                    output="",
                    error="Code validation failed: potentially unsafe operations detected",
                    execution_time=0.0
                )
            
            # Simple execution test (just check syntax)
            try:
                compile(code, '<string>', 'exec')
                execution_time = asyncio.get_event_loop().time() - start_time
                
                return AgentResult(
                    success=True,
                    output="Code compiled successfully (test mode)",
                    error=None,
                    execution_time=execution_time
                )
            except SyntaxError as e:
                return AgentResult(
                    success=False,
                    output="",
                    error=f"Syntax error: {e}",
                    execution_time=0.0
                )
                
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            print(f"Code execution error: {e}")
            
            self.status = AgentStatus.ERROR
            
            return AgentResult(
                success=False,
                output="",
                error=f"Execution error: {e}",
                execution_time=execution_time
            )

    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "id": self.spec.id,
            "name": self.spec.name,
            "description": self.spec.description,
            "type": self.spec.agent_type,
            "status": self.status.value,
            "created_at": self.spec.created_at.isoformat(),
        }

async def test_agent_functionality():
    """Test basic agent functionality."""
    print("🧪 Testing Agent Functionality")
    print("=" * 30)
    
    # Create agent spec
    spec = SimpleAgentSpec(
        id=str(uuid.uuid4()),
        name="Test Calculator",
        description="A simple calculator agent",
        agent_type="python",
        code=""
    )
    
    # Create agent
    agent = TestPythonAgent(spec)
    print(f"✅ Agent created: {agent.spec.name}")
    
    # Test agent info
    info = agent.get_info()
    print(f"✅ Agent info: {info['name']} - {info['status']}")
    
    # Test safe code
    safe_code = """
result = 2 + 2
print(f"2 + 2 = {result}")

# Calculate factorial
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(f"5! = {factorial(5)}")
"""
    
    print("\n📊 Testing safe code execution...")
    result = await agent.execute_code(safe_code)
    print(f"✅ Safe code result: {result.success}")
    print(f"   Output: {result.output}")
    if result.error:
        print(f"   Error: {result.error}")
    
    # Test unsafe code
    unsafe_code = """
import os
os.system("echo 'This should be blocked'")
"""
    
    print("\n🔒 Testing unsafe code blocking...")
    result = await agent.execute_code(unsafe_code)
    print(f"✅ Unsafe code blocked: {not result.success}")
    if result.error:
        print(f"   Error: {result.error}")
    
    # Test syntax error
    syntax_error_code = """
def broken_function(
    print("This has a syntax error")
"""
    
    print("\n❌ Testing syntax error handling...")
    result = await agent.execute_code(syntax_error_code)
    print(f"✅ Syntax error caught: {not result.success}")
    if result.error:
        print(f"   Error: {result.error}")
    
    print("\n🎉 All agent tests completed!")

if __name__ == "__main__":
    asyncio.run(test_agent_functionality())