"""
Agent Builder Module

This module provides the core functionality for creating AI agents from
natural language descriptions. It leverages Qwen's code generation capabilities
to transform user intent into executable SmolAgents.

The builder follows a multi-stage process:
1. Intent Analysis - Understanding what the user wants
2. Specification Generation - Creating a formal agent spec
3. Code Generation - Writing the agent implementation
4. Validation - Ensuring the agent is safe and correct
5. Deployment - Creating the agent instance
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid

from loguru import logger
from pydantic import BaseModel, Field

# Import SmolAgents (with fallback to mock)
from .smolagents_mock import smolagents
CodeAgent = smolagents.CodeAgent
ToolCallingAgent = smolagents.ToolCallingAgent
Tool = smolagents.Tool

from ..ai.qwen_backend import QwenBackend
from ..ai.smart_router import SmartRouter


class AgentType(Enum):
    """Types of agents that can be created."""
    CODE = "code"  # Writes Python code to solve tasks
    TOOL_CALLING = "tool_calling"  # Uses JSON tool calls
    HYBRID = "hybrid"  # Can use both approaches
    MONITOR = "monitor"  # Continuous monitoring agents
    WORKFLOW = "workflow"  # Multi-step workflow agents


class AgentCapability(Enum):
    """Capabilities that agents can have."""
    WEB_SEARCH = "web_search"
    FILE_ACCESS = "file_access"
    DATABASE = "database"
    API_CALLS = "api_calls"
    CODE_EXECUTION = "code_execution"
    EMAIL = "email"
    SCHEDULING = "scheduling"
    DATA_ANALYSIS = "data_analysis"
    IMAGE_PROCESSING = "image_processing"
    NOTIFICATIONS = "notifications"


@dataclass
class AgentSpecification:
    """
    Complete specification for an agent.
    
    This is the intermediate representation between natural language
    and the actual agent implementation.
    """
    # Basic metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    type: AgentType = AgentType.CODE
    
    # Capabilities and permissions
    capabilities: List[AgentCapability] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    # Behavioral specification
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Implementation details
    system_prompt: str = ""
    initial_code: str = ""
    required_packages: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    # Scheduling and lifecycle
    trigger_type: str = "manual"  # manual, scheduled, event-based
    schedule: Optional[str] = None  # cron expression
    max_runtime_seconds: int = 300
    auto_restart: bool = False
    
    # Communication
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    
    # Memory configuration
    memory_enabled: bool = True
    memory_context_limit: int = 10  # Number of previous interactions to remember
    memory_importance_threshold: float = 0.5  # Minimum importance to store
    memory_retention_hours: int = 168  # 7 days default
    memory_sharing_enabled: bool = False  # Allow other agents to access this agent's memory
    memory_search_enabled: bool = True  # Enable semantic search in memory
    persistent_memory: bool = True  # Keep memory between restarts
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "user"
    version: int = 1
    parent_agent_id: Optional[str] = None  # For agent hierarchies


class AgentBuilder:
    """
    Builds agents from natural language descriptions.
    
    This is the core orchestrator that transforms user intent into
    working AI agents using Qwen's capabilities.
    """
    
    def __init__(self, qwen_backend: QwenBackend):
        """Initialize the agent builder with AI backend."""
        self.qwen = qwen_backend
        self.validation_rules = self._load_validation_rules()
        self.template_library = self._load_templates()
        
        logger.info("Agent Builder initialized")
    
    async def create_agent_from_description(
        self, 
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentSpecification:
        """
        Create an agent specification from natural language description.
        
        This is the main entry point for agent creation. It takes a user's
        natural language request and transforms it into a complete agent spec.
        
        Args:
            description: Natural language description of desired agent
            context: Optional context (previous conversation, files, etc.)
            
        Returns:
            AgentSpecification ready for deployment
        """
        try:
            logger.info(f"Creating agent from description: {description[:100]}...")
            
            # Stage 1: Analyze user intent
            intent_analysis = await self._analyze_intent(description, context)
            logger.debug(f"Intent analysis: {intent_analysis}")
            
            # Stage 2: Generate specification
            spec = await self._generate_specification(intent_analysis, description)
            logger.debug(f"Generated spec: {spec.name} ({spec.type.value})")
            
            # Stage 3: Generate implementation
            implementation = await self._generate_implementation(spec)
            spec.initial_code = implementation['code']
            spec.system_prompt = implementation['system_prompt']
            
            # Stage 4: Validate specification
            validation_result = await self._validate_specification(spec)
            if not validation_result['is_valid']:
                raise ValueError(f"Agent validation failed: {validation_result['errors']}")
            
            # Stage 5: Optimize and finalize
            spec = await self._optimize_specification(spec)
            
            logger.success(f"Successfully created agent specification: {spec.name}")
            return spec
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    async def _analyze_intent(
        self, 
        description: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user intent to understand what kind of agent they want.
        
        Uses Qwen to extract:
        - Primary goal
        - Required capabilities
        - Constraints and limitations
        - Success criteria
        """
        prompt = f"""Analyze this agent creation request and extract key information.

User Request: "{description}"

Context: {json.dumps(context or {}, indent=2)}

Extract and return a JSON object with:
1. primary_goal: Main objective of the agent
2. agent_type: "code", "tool_calling", "monitor", "workflow", or "hybrid"
3. capabilities_needed: List of required capabilities
4. constraints: Any limitations or rules the agent must follow
5. success_criteria: How to measure if the agent succeeded
6. suggested_name: A short, descriptive name for the agent
7. complexity_level: "simple", "moderate", or "complex"
8. estimated_tools: List of tools/APIs the agent might need
9. memory_requirements: Object with memory configuration:
   - needs_memory: boolean - whether the agent needs memory
   - context_limit: number - how many previous interactions to remember (1-50)
   - importance_threshold: number - minimum importance to store (0.0-1.0)
   - retention_hours: number - how long to keep memories (1-8760 hours)
   - needs_persistent_memory: boolean - keep memory between restarts
   - needs_shared_memory: boolean - allow other agents to access memories

Be specific and actionable. Return only valid JSON."""

        response = await self.qwen.generate_response(prompt)
        
        try:
            # Parse the JSON response
            intent = json.loads(response)
            return intent
        except json.JSONDecodeError:
            # Fallback to basic parsing if JSON fails
            logger.warning("Failed to parse JSON, using fallback")
            return {
                "primary_goal": description,
                "agent_type": "code",
                "capabilities_needed": ["code_execution"],
                "constraints": [],
                "success_criteria": ["Complete the requested task"],
                "suggested_name": "Custom Agent",
                "complexity_level": "moderate",
                "estimated_tools": [],
                "memory_requirements": {
                    "needs_memory": True,
                    "context_limit": 10,
                    "importance_threshold": 0.5,
                    "retention_hours": 168,
                    "needs_persistent_memory": True,
                    "needs_shared_memory": False
                }
            }
    
    async def _generate_specification(
        self, 
        intent_analysis: Dict[str, Any],
        original_description: str
    ) -> AgentSpecification:
        """Generate a complete agent specification from intent analysis."""
        spec = AgentSpecification()
        
        # Map intent to specification
        spec.name = intent_analysis.get('suggested_name', 'Unnamed Agent')
        spec.description = original_description
        spec.goal = intent_analysis.get('primary_goal', '')
        spec.constraints = intent_analysis.get('constraints', [])
        spec.success_criteria = intent_analysis.get('success_criteria', [])
        
        # Determine agent type
        agent_type_str = intent_analysis.get('agent_type', 'code')
        spec.type = AgentType(agent_type_str)
        
        # Map capabilities
        capability_mapping = {
            'web': AgentCapability.WEB_SEARCH,
            'file': AgentCapability.FILE_ACCESS,
            'database': AgentCapability.DATABASE,
            'api': AgentCapability.API_CALLS,
            'code': AgentCapability.CODE_EXECUTION,
            'email': AgentCapability.EMAIL,
            'schedule': AgentCapability.SCHEDULING,
            'data': AgentCapability.DATA_ANALYSIS,
            'image': AgentCapability.IMAGE_PROCESSING,
            'notify': AgentCapability.NOTIFICATIONS
        }
        
        capabilities_needed = intent_analysis.get('capabilities_needed', [])
        for cap_str in capabilities_needed:
            for key, capability in capability_mapping.items():
                if key in cap_str.lower():
                    spec.capabilities.append(capability)
        
        # Set resource limits based on complexity
        complexity = intent_analysis.get('complexity_level', 'moderate')
        if complexity == 'simple':
            spec.max_runtime_seconds = 60
            spec.resource_limits = {'memory_mb': 256, 'cpu_percent': 25}
        elif complexity == 'complex':
            spec.max_runtime_seconds = 600
            spec.resource_limits = {'memory_mb': 1024, 'cpu_percent': 75}
        else:  # moderate
            spec.max_runtime_seconds = 300
            spec.resource_limits = {'memory_mb': 512, 'cpu_percent': 50}
        
        # Determine trigger type
        if any(word in original_description.lower() for word in ['monitor', 'watch', 'continuously']):
            spec.trigger_type = 'scheduled'
            spec.schedule = '*/5 * * * *'  # Every 5 minutes by default
        elif any(word in original_description.lower() for word in ['when', 'if', 'on event']):
            spec.trigger_type = 'event-based'
        
        # Configure memory settings
        memory_reqs = intent_analysis.get('memory_requirements', {})
        spec.memory_enabled = memory_reqs.get('needs_memory', True)
        spec.memory_context_limit = max(1, min(50, memory_reqs.get('context_limit', 10)))
        spec.memory_importance_threshold = max(0.0, min(1.0, memory_reqs.get('importance_threshold', 0.5)))
        spec.memory_retention_hours = max(1, min(8760, memory_reqs.get('retention_hours', 168)))
        spec.persistent_memory = memory_reqs.get('needs_persistent_memory', True)
        spec.memory_sharing_enabled = memory_reqs.get('needs_shared_memory', False)
        
        # Adjust memory settings based on agent type
        if spec.type == AgentType.MONITOR:
            # Monitoring agents need more memory for trend analysis
            spec.memory_context_limit = max(spec.memory_context_limit, 20)
            spec.memory_retention_hours = max(spec.memory_retention_hours, 720)  # 30 days
        elif spec.type == AgentType.WORKFLOW:
            # Workflow agents need persistent memory for multi-step processes
            spec.persistent_memory = True
            spec.memory_context_limit = max(spec.memory_context_limit, 15)
        
        return spec
    
    async def _generate_implementation(self, spec: AgentSpecification) -> Dict[str, Any]:
        """
        Generate the actual implementation code for the agent.
        
        This is where Qwen's code generation capabilities shine.
        """
        # Build a comprehensive prompt for code generation
        prompt = f"""Generate a SmolAgents implementation for this agent specification:

Name: {spec.name}
Goal: {spec.goal}
Type: {spec.type.value}
Capabilities: {[cap.value for cap in spec.capabilities]}
Constraints: {spec.constraints}
Success Criteria: {spec.success_criteria}

Memory Configuration:
- Memory Enabled: {spec.memory_enabled}
- Context Limit: {spec.memory_context_limit} interactions
- Importance Threshold: {spec.memory_importance_threshold}
- Retention Hours: {spec.memory_retention_hours}
- Persistent Memory: {spec.persistent_memory}
- Memory Sharing: {spec.memory_sharing_enabled}

Generate:
1. A system prompt that will guide the agent's behavior
2. Initial Python code that implements the agent's core logic
3. Any helper functions needed

The agent should:
- Be focused on its specific goal
- Respect all constraints
- Use SmolAgents CodeAgent or ToolCallingAgent appropriately
- Include error handling
- Log important actions
- Use memory system for context-aware responses when memory is enabled
- Save important information to memory with appropriate importance scores

Return a JSON object with:
- system_prompt: String with the system prompt
- code: String with the Python implementation
- imports: List of required imports
- helper_functions: Any additional functions needed"""

        response = await self.qwen.generate_response(prompt)
        
        try:
            implementation = json.loads(response)
            
            # Ensure we have all required fields
            if 'system_prompt' not in implementation:
                implementation['system_prompt'] = self._get_default_system_prompt(spec)
            if 'code' not in implementation:
                implementation['code'] = self._get_default_code(spec)
                
            return implementation
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse implementation JSON, using defaults")
            return {
                'system_prompt': self._get_default_system_prompt(spec),
                'code': self._get_default_code(spec),
                'imports': ['from smolagents import CodeAgent', 'import asyncio'],
                'helper_functions': []
            }
    
    def _get_default_system_prompt(self, spec: AgentSpecification) -> str:
        """Generate a default system prompt for the agent."""
        memory_instructions = ""
        if spec.memory_enabled:
            memory_instructions = f"""

Memory System:
- You have access to a memory system that remembers past interactions
- Store important information with importance scores from 0.0 to 1.0
- Use minimum importance of {spec.memory_importance_threshold} for storage
- Remember up to {spec.memory_context_limit} previous interactions
- Memory is {'persistent' if spec.persistent_memory else 'temporary'} between sessions
- Memory sharing is {'enabled' if spec.memory_sharing_enabled else 'disabled'} with other agents

When using memory:
- Search for relevant past information before responding
- Save important results, learnings, and user preferences
- Use context from memory to provide personalized responses
- Assign higher importance (0.8-1.0) to critical information
- Use moderate importance (0.5-0.7) for useful context
- Use lower importance (0.3-0.5) for routine interactions"""
        
        return f"""You are {spec.name}, an AI agent with a specific purpose.

Your Goal: {spec.goal}

Your Constraints:
{chr(10).join(f'- {c}' for c in spec.constraints)}

Success Criteria:
{chr(10).join(f'- {c}' for c in spec.success_criteria)}

You have access to the following capabilities:
{chr(10).join(f'- {cap.value}' for cap in spec.capabilities)}{memory_instructions}

Always:
1. Stay focused on your goal
2. Respect all constraints
3. Work efficiently and safely
4. Log important actions
5. Handle errors gracefully
6. Use memory system effectively when available"""
    
    def _get_default_code(self, spec: AgentSpecification) -> str:
        """Generate default implementation code."""
        memory_code = ""
        if spec.memory_enabled:
            memory_code = f"""
        # Memory system integration
        memory_manager = input_data.get('memory_manager')
        if memory_manager:
            # Search for relevant context
            memory_context = await memory_manager.search_memories({{
                'agent_id': '{spec.id}',
                'query': input_data.get('task', ''),
                'max_results': {spec.memory_context_limit},
                'min_relevance': {spec.memory_importance_threshold}
            }})
            
            # Add context to task if available
            if memory_context.entries:
                context_str = '\\n'.join([entry.content for entry in memory_context.entries])
                task_with_context = f"Context from memory:\\n{{context_str}}\\n\\nTask: {{input_data.get('task', '')}}"
            else:
                task_with_context = input_data.get('task', '')
        else:
            task_with_context = input_data.get('task', '')"""
        else:
            memory_code = """
        task_with_context = input_data.get('task', '')"""
        
        memory_save_code = ""
        if spec.memory_enabled:
            memory_save_code = f"""
        # Save result to memory if important
        if memory_manager and result and isinstance(result, str):
            importance = 0.7 if 'success' in str(result).lower() else 0.5
            await memory_manager.add_agent_context(
                agent_id='{spec.id}',
                context=f"Task: {{input_data.get('task', '')}}\\nResult: {{result}}",
                importance=importance
            )"""
        
        if spec.type == AgentType.CODE:
            return f"""
async def run_agent(input_data):
    '''Main agent execution function'''
    try:
        # Initialize the agent
        from smolagents import CodeAgent{memory_code}
        
        agent = CodeAgent(
            model=get_model(),
            system_prompt=SYSTEM_PROMPT,
            tools=get_available_tools()
        )
        
        # Execute the agent's task
        result = await agent.run(task_with_context){memory_save_code}
        
        return {{
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }}
    except Exception as e:
        logger.error(f"Agent execution failed: {{e}}")
        return {{
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }}
"""
        else:
            return """
async def run_agent(input_data):
    '''Main agent execution function'''
    # Tool-calling agent implementation
    from smolagents import ToolCallingAgent
    # ... implementation
    pass
"""
    
    async def _validate_specification(self, spec: AgentSpecification) -> Dict[str, Any]:
        """
        Validate the agent specification for safety and correctness.
        
        Checks:
        - No malicious code patterns
        - Resource limits are reasonable
        - Required capabilities match implementation
        - No unauthorized access attempts
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'security_score': 100
        }
        
        # Check for dangerous code patterns
        dangerous_patterns = [
            'eval(', 'exec(', '__import__', 'subprocess', 'os.system',
            'open(', 'file(', 'compile(', 'globals()', 'locals()'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in spec.initial_code:
                validation_result['errors'].append(f"Dangerous pattern detected: {pattern}")
                validation_result['is_valid'] = False
                validation_result['security_score'] -= 20
        
        # Validate resource limits
        if spec.max_runtime_seconds > 3600:  # 1 hour
            validation_result['warnings'].append("Very long runtime limit")
            
        if spec.resource_limits.get('memory_mb', 0) > 2048:
            validation_result['warnings'].append("High memory limit")
        
        # Check capability alignment
        if AgentCapability.FILE_ACCESS in spec.capabilities and 'file' not in spec.initial_code.lower():
            validation_result['warnings'].append("File access capability requested but not used")
        
        # Validate scheduling
        if spec.trigger_type == 'scheduled' and not spec.schedule:
            validation_result['errors'].append("Scheduled agent missing schedule")
            validation_result['is_valid'] = False
        
        # Validate memory configuration
        if spec.memory_enabled:
            if not 1 <= spec.memory_context_limit <= 50:
                validation_result['errors'].append("Memory context limit must be between 1 and 50")
                validation_result['is_valid'] = False
            
            if not 0.0 <= spec.memory_importance_threshold <= 1.0:
                validation_result['errors'].append("Memory importance threshold must be between 0.0 and 1.0")
                validation_result['is_valid'] = False
            
            if not 1 <= spec.memory_retention_hours <= 8760:
                validation_result['errors'].append("Memory retention hours must be between 1 and 8760 (1 year)")
                validation_result['is_valid'] = False
        
        return validation_result
    
    async def _optimize_specification(self, spec: AgentSpecification) -> AgentSpecification:
        """
        Optimize the agent specification for performance and efficiency.
        
        This includes:
        - Minimizing resource usage
        - Optimizing code structure
        - Adding caching where beneficial
        - Improving error handling
        """
        # For now, return as-is
        # Future: Use Qwen to suggest optimizations
        return spec
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for agent creation."""
        return {
            'max_code_length': 10000,
            'max_capabilities': 10,
            'forbidden_imports': ['ctypes', 'win32api', 'socket'],
            'required_error_handling': True
        }
    
    def _load_templates(self) -> Dict[str, AgentSpecification]:
        """Load pre-built agent templates."""
        # This will be expanded with actual templates
        return {}
    
    async def modify_agent(
        self, 
        agent_id: str, 
        modification_request: str
    ) -> AgentSpecification:
        """
        Modify an existing agent based on natural language request.
        
        Args:
            agent_id: ID of the agent to modify
            modification_request: Natural language description of changes
            
        Returns:
            Updated agent specification
        """
        # Implementation for modifying existing agents
        raise NotImplementedError("Agent modification not yet implemented")
    
    async def create_agent_from_template(
        self,
        template_name: str,
        customizations: Dict[str, Any]
    ) -> AgentSpecification:
        """
        Create an agent from a pre-built template.
        
        Args:
            template_name: Name of the template to use
            customizations: Custom parameters for the template
            
        Returns:
            Agent specification based on template
        """
        if template_name not in self.template_library:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Clone template and apply customizations
        template = self.template_library[template_name]
        spec = AgentSpecification(**template.__dict__)
        
        # Apply customizations
        for key, value in customizations.items():
            if hasattr(spec, key):
                setattr(spec, key, value)
        
        # Regenerate ID for new instance
        spec.id = str(uuid.uuid4())
        spec.created_at = datetime.now()
        
        return spec


class AgentPromptGenerator:
    """
    Specialized class for generating high-quality prompts for agent creation.
    
    This separates the prompt engineering from the main builder logic.
    """
    
    @staticmethod
    def generate_analysis_prompt(description: str, context: Dict[str, Any]) -> str:
        """Generate prompt for intent analysis."""
        # Detailed prompt generation
        pass
    
    @staticmethod
    def generate_implementation_prompt(spec: AgentSpecification) -> str:
        """Generate prompt for code generation."""
        # Detailed prompt generation
        pass