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

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

# Import SmolAgents (real implementation)
try:
    from .smolagents_real import smolagents
    
    if smolagents is None:
        # Fallback to mock if real implementation failed
        logger.warning("Real SmolAgents failed, falling back to mock")
        from .smolagents_mock import smolagents
    
    CodeAgent = smolagents.CodeAgent
    ToolCallingAgent = smolagents.ToolCallingAgent
    Tool = smolagents.Tool
    
    logger.info("SmolAgents integration loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import SmolAgents: {e}")
    # Fallback to mock
    from .smolagents_mock import smolagents
    CodeAgent = smolagents.CodeAgent
    ToolCallingAgent = smolagents.ToolCallingAgent
    Tool = smolagents.Tool

from ..ai.qwen_backend import QwenBackend


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
    capabilities: list[AgentCapability] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    resource_limits: dict[str, Any] = field(default_factory=dict)

    # Behavioral specification
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    # Implementation details
    system_prompt: str = ""
    initial_code: str = ""
    required_packages: list[str] = field(default_factory=list)
    environment_vars: dict[str, str] = field(default_factory=dict)

    # Scheduling and lifecycle
    trigger_type: str = "manual"  # manual, scheduled, event-based
    schedule: str | None = None  # cron expression
    max_runtime_seconds: int = 300
    auto_restart: bool = False

    # Communication
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    webhook_url: str | None = None

    # Memory configuration
    memory_enabled: bool = True
    memory_context_limit: int = 10  # Number of previous interactions to remember
    memory_importance_threshold: float = 0.5  # Minimum importance to store
    memory_retention_hours: int = 168  # 7 days default
    memory_sharing_enabled: bool = (
        False  # Allow other agents to access this agent's memory
    )
    memory_search_enabled: bool = True  # Enable semantic search in memory
    persistent_memory: bool = True  # Keep memory between restarts

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "user"
    version: int = 1
    parent_agent_id: str | None = None  # For agent hierarchies


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
        self, description: str, context: dict[str, Any] | None = None
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
            spec.initial_code = implementation["code"]
            spec.system_prompt = implementation["system_prompt"]

            # Stage 4: Validate specification
            validation_result = await self._validate_specification(spec)
            if not validation_result["is_valid"]:
                raise ValueError(
                    f"Agent validation failed: {validation_result['errors']}"
                )

            # Stage 5: Optimize and finalize
            spec = await self._optimize_specification(spec)

            logger.success(f"Successfully created agent specification: {spec.name}")
            return spec

        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise

    async def _analyze_intent(
        self, description: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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
                    "needs_shared_memory": False,
                },
            }

    async def _generate_specification(
        self, intent_analysis: dict[str, Any], original_description: str
    ) -> AgentSpecification:
        """Generate a complete agent specification from intent analysis."""
        spec = AgentSpecification()

        # Map intent to specification
        spec.name = intent_analysis.get("suggested_name", "Unnamed Agent")
        spec.description = original_description
        spec.goal = intent_analysis.get("primary_goal", "")
        spec.constraints = intent_analysis.get("constraints", [])
        spec.success_criteria = intent_analysis.get("success_criteria", [])

        # Determine agent type
        agent_type_str = intent_analysis.get("agent_type", "code")
        spec.type = AgentType(agent_type_str)

        # Map capabilities
        capability_mapping = {
            "web": AgentCapability.WEB_SEARCH,
            "file": AgentCapability.FILE_ACCESS,
            "database": AgentCapability.DATABASE,
            "api": AgentCapability.API_CALLS,
            "code": AgentCapability.CODE_EXECUTION,
            "email": AgentCapability.EMAIL,
            "schedule": AgentCapability.SCHEDULING,
            "data": AgentCapability.DATA_ANALYSIS,
            "image": AgentCapability.IMAGE_PROCESSING,
            "notify": AgentCapability.NOTIFICATIONS,
        }

        capabilities_needed = intent_analysis.get("capabilities_needed", [])
        for cap_str in capabilities_needed:
            for key, capability in capability_mapping.items():
                if key in cap_str.lower():
                    spec.capabilities.append(capability)

        # Set resource limits based on complexity
        complexity = intent_analysis.get("complexity_level", "moderate")
        if complexity == "simple":
            spec.max_runtime_seconds = 60
            spec.resource_limits = {"memory_mb": 256, "cpu_percent": 25}
        elif complexity == "complex":
            spec.max_runtime_seconds = 600
            spec.resource_limits = {"memory_mb": 1024, "cpu_percent": 75}
        else:  # moderate
            spec.max_runtime_seconds = 300
            spec.resource_limits = {"memory_mb": 512, "cpu_percent": 50}

        # Determine trigger type
        if any(
            word in original_description.lower()
            for word in ["monitor", "watch", "continuously"]
        ):
            spec.trigger_type = "scheduled"
            spec.schedule = "*/5 * * * *"  # Every 5 minutes by default
        elif any(
            word in original_description.lower() for word in ["when", "if", "on event"]
        ):
            spec.trigger_type = "event-based"

        # Configure memory settings
        memory_reqs = intent_analysis.get("memory_requirements", {})
        spec.memory_enabled = memory_reqs.get("needs_memory", True)
        spec.memory_context_limit = max(
            1, min(50, memory_reqs.get("context_limit", 10))
        )
        spec.memory_importance_threshold = max(
            0.0, min(1.0, memory_reqs.get("importance_threshold", 0.5))
        )
        spec.memory_retention_hours = max(
            1, min(8760, memory_reqs.get("retention_hours", 168))
        )
        spec.persistent_memory = memory_reqs.get("needs_persistent_memory", True)
        spec.memory_sharing_enabled = memory_reqs.get("needs_shared_memory", False)

        # Adjust memory settings based on agent type
        if spec.type == AgentType.MONITOR:
            # Monitoring agents need more memory for trend analysis
            spec.memory_context_limit = max(spec.memory_context_limit, 20)
            spec.memory_retention_hours = max(
                spec.memory_retention_hours, 720
            )  # 30 days
        elif spec.type == AgentType.WORKFLOW:
            # Workflow agents need persistent memory for multi-step processes
            spec.persistent_memory = True
            spec.memory_context_limit = max(spec.memory_context_limit, 15)

        return spec

    async def _generate_implementation(
        self, spec: AgentSpecification
    ) -> dict[str, Any]:
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
            if "system_prompt" not in implementation:
                implementation["system_prompt"] = self._get_default_system_prompt(spec)
            if "code" not in implementation:
                implementation["code"] = self._get_default_code(spec)

            return implementation

        except json.JSONDecodeError:
            logger.warning("Failed to parse implementation JSON, using defaults")
            return {
                "system_prompt": self._get_default_system_prompt(spec),
                "code": self._get_default_code(spec),
                "imports": ["from smolagents import CodeAgent", "import asyncio"],
                "helper_functions": [],
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
        # Import Tektra SmolAgents implementation
        from tektra.agents.smolagents_real import TektraCodeAgent
        from tektra.ai.qwen_backend import QwenBackend
        import asyncio
        from datetime import datetime
        from loguru import logger{memory_code}

        # Get backend and memory manager from input
        qwen_backend = input_data.get('qwen_backend')
        memory_manager = input_data.get('memory_manager')
        agent_id = input_data.get('agent_id', '{spec.id}')
        
        # Initialize the Tektra CodeAgent
        agent = TektraCodeAgent(
            qwen_backend=qwen_backend,
            memory_manager=memory_manager,
            system_prompt=SYSTEM_PROMPT,
            max_iterations=10,
            executor_type='local'
        )

        # Execute the agent's task
        result = await agent.run(task_with_context, agent_id=agent_id){memory_save_code}

        return result

    except Exception as e:
        logger.error(f"Agent execution failed: {{e}}")
        return {{
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'agent_type': 'CodeAgent'
        }}
"""
        else:
            return f"""
async def run_agent(input_data):
    '''Main agent execution function for tool-calling agent'''
    try:
        # Import Tektra SmolAgents implementation
        from tektra.agents.smolagents_real import TektraToolCallingAgent
        from tektra.ai.qwen_backend import QwenBackend
        import asyncio
        from datetime import datetime
        from loguru import logger{memory_code}

        # Get backend and memory manager from input
        qwen_backend = input_data.get('qwen_backend')
        memory_manager = input_data.get('memory_manager')
        agent_id = input_data.get('agent_id', '{spec.id}')
        
        # Initialize the Tektra ToolCallingAgent
        agent = TektraToolCallingAgent(
            qwen_backend=qwen_backend,
            memory_manager=memory_manager,
            system_prompt=SYSTEM_PROMPT,
            max_tool_threads=4
        )

        # Execute the agent's task using tool calls
        result = await agent.run(task_with_context, agent_id=agent_id){memory_save_code}

        return result

    except Exception as e:
        logger.error(f\"Tool-calling agent execution failed: {{e}}\")
        return {{
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'agent_type': 'ToolCallingAgent'
        }}
"""

    async def _validate_specification(self, spec: AgentSpecification) -> dict[str, Any]:
        """
        Validate the agent specification for safety and correctness.

        Checks:
        - No malicious code patterns
        - Resource limits are reasonable
        - Required capabilities match implementation
        - No unauthorized access attempts
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "security_score": 100,
        }

        # Check for dangerous code patterns
        dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "subprocess",
            "os.system",
            "open(",
            "file(",
            "compile(",
            "globals()",
            "locals()",
        ]

        for pattern in dangerous_patterns:
            if pattern in spec.initial_code:
                validation_result["errors"].append(
                    f"Dangerous pattern detected: {pattern}"
                )
                validation_result["is_valid"] = False
                validation_result["security_score"] -= 20

        # Validate resource limits
        if spec.max_runtime_seconds > 3600:  # 1 hour
            validation_result["warnings"].append("Very long runtime limit")

        if spec.resource_limits.get("memory_mb", 0) > 2048:
            validation_result["warnings"].append("High memory limit")

        # Check capability alignment
        if (
            AgentCapability.FILE_ACCESS in spec.capabilities
            and "file" not in spec.initial_code.lower()
        ):
            validation_result["warnings"].append(
                "File access capability requested but not used"
            )

        # Validate scheduling
        if spec.trigger_type == "scheduled" and not spec.schedule:
            validation_result["errors"].append("Scheduled agent missing schedule")
            validation_result["is_valid"] = False

        # Validate memory configuration
        if spec.memory_enabled:
            if not 1 <= spec.memory_context_limit <= 50:
                validation_result["errors"].append(
                    "Memory context limit must be between 1 and 50"
                )
                validation_result["is_valid"] = False

            if not 0.0 <= spec.memory_importance_threshold <= 1.0:
                validation_result["errors"].append(
                    "Memory importance threshold must be between 0.0 and 1.0"
                )
                validation_result["is_valid"] = False

            if not 1 <= spec.memory_retention_hours <= 8760:
                validation_result["errors"].append(
                    "Memory retention hours must be between 1 and 8760 (1 year)"
                )
                validation_result["is_valid"] = False

        return validation_result

    async def _optimize_specification(
        self, spec: AgentSpecification
    ) -> AgentSpecification:
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

    def _load_validation_rules(self) -> dict[str, Any]:
        """Load validation rules for agent creation."""
        return {
            "max_code_length": 10000,
            "max_capabilities": 10,
            "forbidden_imports": ["ctypes", "win32api", "socket"],
            "required_error_handling": True,
        }

    def _load_templates(self) -> dict[str, AgentSpecification]:
        """Load pre-built agent templates."""
        # This will be expanded with actual templates
        return {}

    async def _analyze_modification_intent(
        self, existing_spec: AgentSpecification, modification_request: str
    ) -> dict[str, Any]:
        """
        Analyze modification request to understand what changes are needed.

        Uses Qwen to extract modification intent and scope.
        """
        prompt = f"""Analyze this agent modification request and extract key information.

Current Agent Details:
- Name: {existing_spec.name}
- Type: {existing_spec.type.value}
- Goal: {existing_spec.goal}
- Capabilities: {[cap.value for cap in existing_spec.capabilities]}
- Current Tools: {existing_spec.allowed_tools}
- Constraints: {existing_spec.constraints}

Modification Request: "{modification_request}"

Extract and return a JSON object with:
1. modification_type: "capability_change", "behavior_change", "constraint_change", "tool_change", "schedule_change", or "complete_redesign"
2. scope: "minor", "moderate", or "major" - how extensive are the changes
3. capabilities_to_add: List of new capabilities needed
4. capabilities_to_remove: List of capabilities to remove
5. tools_to_add: List of new tools/APIs needed
6. tools_to_remove: List of tools to remove
7. behavior_changes: Object describing how behavior should change:
   - new_goal: Updated primary goal (if changed)
   - additional_constraints: New constraints to add
   - removed_constraints: Constraints to remove
8. schedule_changes: Object with scheduling updates (if any):
   - new_trigger_type: "manual", "scheduled", "event-based"
   - new_schedule: cron expression (if scheduled)
9. needs_code_update: boolean - whether implementation code needs regeneration
10. risk_level: "low", "medium", "high" - potential impact of changes
11. compatibility: "full", "partial", "breaking" - compatibility with existing agent
12. summary: Brief description of what will be changed

Be specific and actionable. Return only valid JSON."""

        response = await self.qwen.generate_response(prompt)

        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse modification analysis: {e}")
            return {
                "modification_type": "behavior_change",
                "scope": "moderate",
                "needs_code_update": True,
                "risk_level": "medium",
                "compatibility": "partial",
                "summary": f"Unable to parse modification request: {modification_request}",
                "capabilities_to_add": [],
                "capabilities_to_remove": [],
                "tools_to_add": [],
                "tools_to_remove": [],
                "behavior_changes": {},
                "schedule_changes": {}
            }

    async def _validate_modification(
        self, existing_spec: AgentSpecification, modification_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate that the requested modification is safe and feasible.
        """
        errors = []
        warnings = []

        # Check risk level
        risk_level = modification_analysis.get("risk_level", "medium")
        if risk_level == "high":
            warnings.append("High-risk modification detected - review carefully")

        # Validate compatibility
        compatibility = modification_analysis.get("compatibility", "partial")
        if compatibility == "breaking":
            warnings.append("Breaking changes detected - agent behavior will change significantly")

        # Check scope restrictions
        scope = modification_analysis.get("scope", "moderate")
        if scope == "major" and modification_analysis.get("modification_type") == "complete_redesign":
            errors.append("Complete redesign not supported - create a new agent instead")

        # Validate capability changes
        caps_to_add = modification_analysis.get("capabilities_to_add", [])
        caps_to_remove = modification_analysis.get("capabilities_to_remove", [])
        
        # Ensure we don't remove all capabilities
        current_caps = set(cap.value for cap in existing_spec.capabilities)
        if set(caps_to_remove) >= current_caps and not caps_to_add:
            errors.append("Cannot remove all capabilities without adding new ones")

        # Validate tool changes
        tools_to_add = modification_analysis.get("tools_to_add", [])
        dangerous_tools = ["shell", "exec", "eval", "file_delete", "system"]
        if any(tool in dangerous_tools for tool in tools_to_add):
            errors.append(f"Cannot add dangerous tools: {[t for t in tools_to_add if t in dangerous_tools]}")

        # Check behavior changes
        behavior_changes = modification_analysis.get("behavior_changes", {})
        if behavior_changes.get("new_goal") and len(behavior_changes["new_goal"]) < 10:
            errors.append("New goal must be at least 10 characters long")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "risk_assessment": {
                "risk_level": risk_level,
                "compatibility": compatibility,
                "scope": scope
            }
        }

    async def _apply_modifications(
        self, existing_spec: AgentSpecification, modification_analysis: dict[str, Any]
    ) -> AgentSpecification:
        """
        Apply the analyzed modifications to create a new agent specification.
        """
        # Create a copy of the existing specification
        import copy
        modified_spec = copy.deepcopy(existing_spec)

        # Apply capability changes
        caps_to_add = modification_analysis.get("capabilities_to_add", [])
        caps_to_remove = modification_analysis.get("capabilities_to_remove", [])
        
        # Remove capabilities
        modified_spec.capabilities = [
            cap for cap in modified_spec.capabilities 
            if cap.value not in caps_to_remove
        ]
        
        # Add new capabilities
        from .registry import AgentCapability
        for cap_name in caps_to_add:
            try:
                new_cap = AgentCapability(cap_name)
                if new_cap not in modified_spec.capabilities:
                    modified_spec.capabilities.append(new_cap)
            except ValueError:
                logger.warning(f"Unknown capability: {cap_name}")

        # Apply tool changes
        tools_to_add = modification_analysis.get("tools_to_add", [])
        tools_to_remove = modification_analysis.get("tools_to_remove", [])
        
        # Remove tools
        modified_spec.allowed_tools = [
            tool for tool in modified_spec.allowed_tools 
            if tool not in tools_to_remove
        ]
        
        # Add new tools
        for tool in tools_to_add:
            if tool not in modified_spec.allowed_tools:
                modified_spec.allowed_tools.append(tool)

        # Apply behavior changes
        behavior_changes = modification_analysis.get("behavior_changes", {})
        if behavior_changes.get("new_goal"):
            modified_spec.goal = behavior_changes["new_goal"]
        
        if behavior_changes.get("additional_constraints"):
            modified_spec.constraints.extend(behavior_changes["additional_constraints"])
        
        if behavior_changes.get("removed_constraints"):
            modified_spec.constraints = [
                constraint for constraint in modified_spec.constraints
                if constraint not in behavior_changes["removed_constraints"]
            ]

        # Apply schedule changes
        schedule_changes = modification_analysis.get("schedule_changes", {})
        if schedule_changes.get("new_trigger_type"):
            modified_spec.trigger_type = schedule_changes["new_trigger_type"]
        
        if schedule_changes.get("new_schedule"):
            modified_spec.schedule = schedule_changes["new_schedule"]

        # Update description to reflect changes
        summary = modification_analysis.get("summary", "Agent modified")
        modified_spec.description = f"{existing_spec.description}\n\nModifications: {summary}"

        return modified_spec

    async def modify_agent(
        self, existing_spec: AgentSpecification, modification_request: str
    ) -> AgentSpecification:
        """
        Modify an existing agent based on natural language request.

        Args:
            existing_spec: Current agent specification to modify
            modification_request: Natural language description of changes

        Returns:
            Updated agent specification

        Raises:
            ValueError: If modification request is invalid or unsafe
            RuntimeError: If modification fails during processing
        """
        try:
            logger.info(f"Modifying agent '{existing_spec.name}': {modification_request[:100]}...")

            # Stage 1: Analyze modification intent
            modification_analysis = await self._analyze_modification_intent(
                existing_spec, modification_request
            )
            logger.debug(f"Modification analysis: {modification_analysis}")

            # Stage 2: Validate modification is safe and feasible
            validation_result = await self._validate_modification(
                existing_spec, modification_analysis
            )
            if not validation_result["is_valid"]:
                raise ValueError(
                    f"Modification validation failed: {validation_result['errors']}"
                )

            # Stage 3: Apply modifications to create new specification
            modified_spec = await self._apply_modifications(
                existing_spec, modification_analysis
            )

            # Stage 4: Generate updated implementation if code changes requested
            if modification_analysis.get("needs_code_update", False):
                implementation = await self._generate_implementation(modified_spec)
                modified_spec.initial_code = implementation["code"]
                if implementation.get("system_prompt"):
                    modified_spec.system_prompt = implementation["system_prompt"]

            # Stage 5: Final validation of modified agent
            final_validation = await self._validate_specification(modified_spec)
            if not final_validation["is_valid"]:
                raise RuntimeError(
                    f"Modified agent validation failed: {final_validation['errors']}"
                )

            # Stage 6: Optimize modified specification
            modified_spec = await self._optimize_specification(modified_spec)

            logger.success(f"Successfully modified agent: {modified_spec.name}")
            return modified_spec

        except Exception as e:
            logger.error(f"Failed to modify agent: {e}")
            raise RuntimeError(f"Agent modification failed: {e}") from e

    async def create_agent_from_template(
        self, template_name: str, customizations: dict[str, Any]
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
    def generate_analysis_prompt(description: str, context: dict[str, Any]) -> str:
        """Generate prompt for intent analysis."""
        return f"""Analyze this agent creation request and extract key information.

User Request: "{description}"

Context: {json.dumps(context, indent=2)}

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

    @staticmethod
    def generate_implementation_prompt(spec: AgentSpecification) -> str:
        """Generate prompt for code generation."""
        return f"""Generate a SmolAgents implementation for this agent specification:

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
