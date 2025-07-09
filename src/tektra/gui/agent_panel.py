"""
Agent Management Panel

This module provides the UI for creating, managing, and monitoring agents
within the Tektra application. It includes:
- Agent creation wizard
- Agent dashboard
- Template browser
- Execution monitoring
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import json

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from loguru import logger

from ..agents import (
    AgentBuilder, 
    AgentRuntime, 
    AgentRegistry,
    TemplateLibrary,
    AgentStatus,
    SandboxType
)


class AgentCreationWizard:
    """
    Wizard interface for creating agents from natural language or templates.
    """
    
    def __init__(
        self,
        on_agent_created: Optional[Callable] = None,
        agent_builder: Optional[AgentBuilder] = None
    ):
        """Initialize the agent creation wizard."""
        self.on_agent_created = on_agent_created
        self.agent_builder = agent_builder
        self.template_library = TemplateLibrary()
        
        # UI state
        self.current_step = 0
        self.creation_method = None  # 'natural_language' or 'template'
        self.selected_template = None
        self.agent_description = ""
        self.customizations = {}
        
        # UI components
        self.container = None
        self.step_container = None
        self.navigation_container = None
        
        self._build_ui()
    
    def _build_ui(self) -> toga.Box:
        """Build the wizard UI."""
        self.container = toga.Box(style=Pack(
            direction=COLUMN,
            background_color="#f8f9fa",
            padding=20
        ))
        
        # Header
        header = toga.Label(
            "Create New Agent",
            style=Pack(
                font_size=20,
                font_weight="bold",
                margin_bottom=20,
                color="#2c3e50"
            )
        )
        self.container.add(header)
        
        # Step indicator
        self.step_indicator = toga.Label(
            "Step 1 of 3: Choose Creation Method",
            style=Pack(
                font_size=14,
                margin_bottom=15,
                color="#7f8c8d"
            )
        )
        self.container.add(self.step_indicator)
        
        # Step container (content changes based on step)
        self.step_container = toga.Box(style=Pack(
            direction=COLUMN,
            min_height=300,
            margin_bottom=20
        ))
        self.container.add(self.step_container)
        
        # Navigation buttons
        self.navigation_container = toga.Box(style=Pack(
            direction=ROW,
            align_items="center"
        ))
        
        self.back_button = toga.Button(
            "← Back",
            on_press=self._on_back,
            style=Pack(
                margin_right=10,
                width=100
            ),
            enabled=False
        )
        self.navigation_container.add(self.back_button)
        
        # Spacer
        spacer = toga.Box(style=Pack(flex=1))
        self.navigation_container.add(spacer)
        
        self.next_button = toga.Button(
            "Next →",
            on_press=self._on_next,
            style=Pack(
                background_color="#007bff",
                color="#ffffff",
                width=100
            )
        )
        self.navigation_container.add(self.next_button)
        
        self.container.add(self.navigation_container)
        
        # Show first step
        self._show_step(0)
        
        return self.container
    
    def _show_step(self, step: int):
        """Show the specified wizard step."""
        self.current_step = step
        
        # Clear step container
        for child in list(self.step_container.children):
            self.step_container.remove(child)
        
        # Update navigation
        self.back_button.enabled = step > 0
        
        # Show appropriate step
        if step == 0:
            self._show_method_selection()
            self.step_indicator.text = "Step 1 of 3: Choose Creation Method"
            self.next_button.text = "Next →"
        elif step == 1:
            if self.creation_method == 'natural_language':
                self._show_natural_language_input()
                self.step_indicator.text = "Step 2 of 3: Describe Your Agent"
            else:
                self._show_template_selection()
                self.step_indicator.text = "Step 2 of 3: Choose a Template"
            self.next_button.text = "Next →"
        elif step == 2:
            if self.creation_method == 'natural_language':
                self._show_confirmation()
                self.step_indicator.text = "Step 3 of 3: Confirm and Create"
            else:
                self._show_template_customization()
                self.step_indicator.text = "Step 3 of 3: Customize Template"
            self.next_button.text = "Create Agent"
    
    def _show_method_selection(self):
        """Show creation method selection step."""
        # Title
        title = toga.Label(
            "How would you like to create your agent?",
            style=Pack(
                font_size=16,
                margin_bottom=20
            )
        )
        self.step_container.add(title)
        
        # Natural language option
        nl_box = toga.Box(style=Pack(
            direction=COLUMN,
            background_color="#ffffff",
            padding=15,
            margin_bottom=15
        ))
        
        nl_radio = toga.Selection(
            items=["Natural Language Description"],
            on_change=lambda w: setattr(self, 'creation_method', 'natural_language'),
            style=Pack(margin_bottom=10)
        )
        nl_box.add(nl_radio)
        
        nl_desc = toga.Label(
            "Describe what you want your agent to do in plain English",
            style=Pack(
                font_size=12,
                color="#6c757d"
            )
        )
        nl_box.add(nl_desc)
        
        self.step_container.add(nl_box)
        
        # Template option
        template_box = toga.Box(style=Pack(
            direction=COLUMN,
            background_color="#ffffff",
            padding=15
        ))
        
        template_radio = toga.Selection(
            items=["Use a Template"],
            on_change=lambda w: setattr(self, 'creation_method', 'template'),
            style=Pack(margin_bottom=10)
        )
        template_box.add(template_radio)
        
        template_desc = toga.Label(
            "Start with a pre-built template and customize it",
            style=Pack(
                font_size=12,
                color="#6c757d"
            )
        )
        template_box.add(template_desc)
        
        self.step_container.add(template_box)
    
    def _show_natural_language_input(self):
        """Show natural language description input."""
        # Instructions
        instructions = toga.Label(
            "Describe what you want your agent to do:",
            style=Pack(
                font_size=14,
                margin_bottom=10
            )
        )
        self.step_container.add(instructions)
        
        # Text input
        self.description_input = toga.MultilineTextInput(
            placeholder="Example: Create an agent that monitors my GitHub repos and sends me a daily summary of issues and pull requests...",
            style=Pack(
                min_height=150,
                margin_bottom=15,
                font_size=12
            )
        )
        if self.agent_description:
            self.description_input.value = self.agent_description
        
        self.step_container.add(self.description_input)
        
        # Examples
        examples_label = toga.Label(
            "Example descriptions:",
            style=Pack(
                font_size=12,
                font_weight="bold",
                margin_bottom=5
            )
        )
        self.step_container.add(examples_label)
        
        examples = [
            "• Monitor stock prices and alert me when they change by more than 5%",
            "• Analyze my sales data every morning and email me insights",
            "• Watch for new papers about machine learning on arXiv",
            "• Organize my downloads folder by file type every day"
        ]
        
        for example in examples:
            ex_label = toga.Label(
                example,
                style=Pack(
                    font_size=11,
                    color="#6c757d",
                    margin_bottom=3
                )
            )
            self.step_container.add(ex_label)
    
    def _show_template_selection(self):
        """Show template selection step."""
        # Categories
        categories = self.template_library.get_categories()
        
        category_label = toga.Label(
            "Select a category:",
            style=Pack(
                font_size=14,
                margin_bottom=10
            )
        )
        self.step_container.add(category_label)
        
        self.category_selection = toga.Selection(
            items=categories,
            on_change=self._on_category_change,
            style=Pack(margin_bottom=15)
        )
        self.step_container.add(self.category_selection)
        
        # Template list
        template_label = toga.Label(
            "Choose a template:",
            style=Pack(
                font_size=14,
                margin_bottom=10
            )
        )
        self.step_container.add(template_label)
        
        # Template container with scroll
        self.template_list_container = toga.Box(style=Pack(
            direction=COLUMN
        ))
        
        self.template_scroll = toga.ScrollContainer(
            content=self.template_list_container,
            style=Pack(
                height=200
            )
        )
        self.step_container.add(self.template_scroll)
        
        # Show templates for first category
        if categories:
            self._show_templates_for_category(categories[0])
    
    def _on_category_change(self, widget):
        """Handle category selection change."""
        category = widget.value
        self._show_templates_for_category(category)
    
    def _show_templates_for_category(self, category: str):
        """Show templates for the selected category."""
        # Clear existing templates
        for child in list(self.template_list_container.children):
            self.template_list_container.remove(child)
        
        # Get templates for category
        templates = self.template_library.list_templates(category)
        
        for template in templates:
            # Template card
            card = toga.Box(style=Pack(
                direction=ROW,
                background_color="#ffffff",
                padding=10,
                margin_bottom=10,
                align_items="center"
            ))
            
            # Icon
            icon_label = toga.Label(
                template.icon,
                style=Pack(
                    font_size=24,
                    margin_right=15
                )
            )
            card.add(icon_label)
            
            # Info
            info_box = toga.Box(style=Pack(
                direction=COLUMN,
                flex=1
            ))
            
            name_label = toga.Label(
                template.name,
                style=Pack(
                    font_weight="bold",
                    margin_bottom=3
                )
            )
            info_box.add(name_label)
            
            desc_label = toga.Label(
                template.description,
                style=Pack(
                    font_size=11,
                    color="#6c757d"
                )
            )
            info_box.add(desc_label)
            
            card.add(info_box)
            
            # Select button
            select_btn = toga.Button(
                "Select",
                on_press=lambda w, t=template: self._select_template(t),
                style=Pack(
                    width=80,
                    font_size=12
                )
            )
            card.add(select_btn)
            
            self.template_list_container.add(card)
    
    def _select_template(self, template):
        """Select a template."""
        self.selected_template = template
        logger.info(f"Selected template: {template.name}")
        # Move to next step
        self._show_step(2)
    
    def _show_template_customization(self):
        """Show template customization step."""
        if not self.selected_template:
            return
        
        # Template info
        info_box = toga.Box(style=Pack(
            direction=ROW,
            margin_bottom=15,
            align_items="center"
        ))
        
        icon_label = toga.Label(
            self.selected_template.icon,
            style=Pack(
                font_size=20,
                margin_right=10
            )
        )
        info_box.add(icon_label)
        
        name_label = toga.Label(
            self.selected_template.name,
            style=Pack(
                font_weight="bold"
            )
        )
        info_box.add(name_label)
        
        self.step_container.add(info_box)
        
        # Parameters
        params_label = toga.Label(
            "Customize parameters:",
            style=Pack(
                font_size=14,
                margin_bottom=10,
                font_weight="bold"
            )
        )
        self.step_container.add(params_label)
        
        # Parameter inputs
        self.param_inputs = {}
        
        for param_name, param_info in self.selected_template.parameters.items():
            # Parameter container
            param_box = toga.Box(style=Pack(
                direction=COLUMN,
                margin_bottom=15
            ))
            
            # Label
            label = toga.Label(
                f"{param_info['description']}:",
                style=Pack(
                    font_size=12,
                    margin_bottom=5
                )
            )
            param_box.add(label)
            
            # Input based on type
            if param_info['type'] == 'string':
                input_widget = toga.TextInput(
                    placeholder=str(param_info.get('default', '')),
                    style=Pack(width=300)
                )
            elif param_info['type'] == 'enum':
                input_widget = toga.Selection(
                    items=param_info['values'],
                    style=Pack(width=300)
                )
                # Set default
                if 'default' in param_info:
                    input_widget.value = param_info['default']
            elif param_info['type'] == 'boolean':
                input_widget = toga.Switch(
                    text="",
                    value=param_info.get('default', False)
                )
            elif param_info['type'] == 'integer':
                input_widget = toga.NumberInput(
                    value=param_info.get('default', 0),
                    style=Pack(width=100)
                )
            else:
                # Default to text input
                input_widget = toga.TextInput(
                    style=Pack(width=300)
                )
            
            self.param_inputs[param_name] = input_widget
            param_box.add(input_widget)
            
            self.step_container.add(param_box)
    
    def _show_confirmation(self):
        """Show confirmation step for natural language creation."""
        # Summary
        summary_label = toga.Label(
            "Agent Summary:",
            style=Pack(
                font_size=16,
                font_weight="bold",
                margin_bottom=10
            )
        )
        self.step_container.add(summary_label)
        
        # Description
        desc_box = toga.Box(style=Pack(
            background_color="#f8f9fa",
            padding=15,
            margin_bottom=20
        ))
        
        desc_label = toga.Label(
            self.agent_description or self.description_input.value,
            style=Pack(
                font_size=12
            )
        )
        desc_box.add(desc_label)
        
        self.step_container.add(desc_box)
        
        # What happens next
        next_label = toga.Label(
            "What happens next:",
            style=Pack(
                font_size=14,
                font_weight="bold",
                margin_bottom=10
            )
        )
        self.step_container.add(next_label)
        
        steps = [
            "1. Tektra will analyze your description",
            "2. Generate an agent specification",
            "3. Create the implementation code",
            "4. Deploy the agent in a secure sandbox",
            "5. You can monitor and control your agent"
        ]
        
        for step in steps:
            step_label = toga.Label(
                step,
                style=Pack(
                    font_size=12,
                    margin_bottom=5,
                    color="#495057"
                )
            )
            self.step_container.add(step_label)
    
    async def _on_next(self, widget):
        """Handle next button press."""
        if self.current_step == 0:
            # Validate method selection
            if not self.creation_method:
                # Show error
                return
            self._show_step(1)
        
        elif self.current_step == 1:
            if self.creation_method == 'natural_language':
                # Save description
                self.agent_description = self.description_input.value
                if not self.agent_description.strip():
                    # Show error
                    return
            else:
                # Validate template selection
                if not self.selected_template:
                    # Show error
                    return
            self._show_step(2)
        
        elif self.current_step == 2:
            # Create agent
            await self._create_agent()
    
    async def _on_back(self, widget):
        """Handle back button press."""
        if self.current_step > 0:
            self._show_step(self.current_step - 1)
    
    async def _create_agent(self):
        """Create the agent based on user input."""
        try:
            # Disable buttons during creation
            self.next_button.enabled = False
            self.back_button.enabled = False
            self.next_button.text = "Creating..."
            
            if self.creation_method == 'natural_language':
                # Create from description
                if self.agent_builder:
                    spec = await self.agent_builder.create_agent_from_description(
                        self.agent_description
                    )
                else:
                    # Mock creation for demo
                    from ..agents.builder import AgentSpecification
                    spec = AgentSpecification(
                        name="Custom Agent",
                        description=self.agent_description
                    )
            else:
                # Create from template
                customizations = {}
                for param_name, input_widget in self.param_inputs.items():
                    if hasattr(input_widget, 'value'):
                        customizations[param_name] = input_widget.value
                
                spec = self.template_library.create_agent_from_template(
                    self.selected_template.name,
                    customizations
                )
            
            # Notify callback
            if self.on_agent_created:
                await self._safe_callback(self.on_agent_created, spec)
            
            # Show success
            self.next_button.text = "✓ Agent Created!"
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            self.next_button.text = "Creation Failed"
        finally:
            # Re-enable buttons
            await asyncio.sleep(2)
            self.next_button.enabled = True
            self.back_button.enabled = True
            self.next_button.text = "Create Another"
            self._show_step(0)
    
    async def _safe_callback(self, callback: Callable, *args):
        """Safely execute callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")


class AgentDashboard:
    """
    Dashboard for viewing and managing active agents.
    """
    
    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        agent_runtime: Optional[AgentRuntime] = None
    ):
        """Initialize agent dashboard."""
        self.registry = agent_registry
        self.runtime = agent_runtime
        
        # UI components
        self.container = None
        self.agent_list_container = None
        self.detail_container = None
        self.selected_agent = None
        
        self._build_ui()
    
    def _build_ui(self) -> toga.Box:
        """Build the dashboard UI."""
        self.container = toga.Box(style=Pack(
            direction=ROW,
            background_color="#f8f9fa"
        ))
        
        # Left panel - Agent list
        left_panel = toga.Box(style=Pack(
            direction=COLUMN,
            width=300,
            padding=10,
            background_color="#ffffff"
        ))
        
        # Header
        header = toga.Label(
            "Your Agents",
            style=Pack(
                font_size=18,
                font_weight="bold",
                margin_bottom=15
            )
        )
        left_panel.add(header)
        
        # Filter/search
        self.search_input = toga.TextInput(
            placeholder="Search agents...",
            on_change=self._on_search_change,
            style=Pack(
                margin_bottom=10
            )
        )
        left_panel.add(self.search_input)
        
        # Agent list
        self.agent_list_container = toga.Box(style=Pack(
            direction=COLUMN
        ))
        
        self.agent_scroll = toga.ScrollContainer(
            content=self.agent_list_container,
            style=Pack(
                flex=1
            )
        )
        left_panel.add(self.agent_scroll)
        
        self.container.add(left_panel)
        
        # Right panel - Agent details
        self.detail_container = toga.Box(style=Pack(
            direction=COLUMN,
            flex=1,
            padding=20,
            background_color="#ffffff"
        ))
        
        # Placeholder when no agent selected
        placeholder = toga.Label(
            "Select an agent to view details",
            style=Pack(
                font_size=14,
                color="#6c757d"
            )
        )
        self.detail_container.add(placeholder)
        
        self.container.add(self.detail_container)
        
        # Load agents
        asyncio.create_task(self._load_agents())
        
        return self.container
    
    async def _load_agents(self):
        """Load agents from registry."""
        if not self.registry:
            return
        
        try:
            # Get all agents
            agents = await self.registry.list_agents()
            
            # Clear existing
            for child in list(self.agent_list_container.children):
                self.agent_list_container.remove(child)
            
            # Add agent cards
            for agent_record in agents:
                self._add_agent_card(agent_record)
                
        except Exception as e:
            logger.error(f"Error loading agents: {e}")
    
    def _add_agent_card(self, agent_record):
        """Add an agent card to the list."""
        spec = agent_record.specification
        
        # Card container
        card = toga.Box(style=Pack(
            direction=COLUMN,
            background_color="#f8f9fa",
            padding=10,
            margin_bottom=10
        ))
        
        # Top row - name and status
        top_row = toga.Box(style=Pack(
            direction=ROW,
            align_items="center",
            margin_bottom=5
        ))
        
        # Status indicator
        status_color = {
            AgentStatus.ACTIVE: "🟢",
            AgentStatus.INACTIVE: "⚪",
            AgentStatus.ERROR: "🔴",
            AgentStatus.SUSPENDED: "🟡"
        }.get(agent_record.status, "⚪")
        
        status_label = toga.Label(
            status_color,
            style=Pack(
                font_size=12,
                margin_right=5
            )
        )
        top_row.add(status_label)
        
        # Name
        name_label = toga.Label(
            spec.name,
            style=Pack(
                font_weight="bold",
                flex=1
            )
        )
        top_row.add(name_label)
        
        card.add(top_row)
        
        # Description
        desc_label = toga.Label(
            spec.description[:50] + "..." if len(spec.description) > 50 else spec.description,
            style=Pack(
                font_size=11,
                color="#6c757d",
                margin_bottom=5
            )
        )
        card.add(desc_label)
        
        # Metrics row
        metrics_row = toga.Box(style=Pack(
            direction=ROW,
            align_items="center"
        ))
        
        # Execution count
        exec_label = toga.Label(
            f"🔄 {agent_record.metrics.total_executions}",
            style=Pack(
                font_size=10,
                margin_right=10
            )
        )
        metrics_row.add(exec_label)
        
        # Success rate
        success_rate = (
            agent_record.metrics.successful_executions / 
            agent_record.metrics.total_executions * 100
            if agent_record.metrics.total_executions > 0 else 0
        )
        success_label = toga.Label(
            f"✓ {success_rate:.0f}%",
            style=Pack(
                font_size=10
            )
        )
        metrics_row.add(success_label)
        
        card.add(metrics_row)
        
        # Make clickable
        # Note: In real Toga, we'd use a button or custom widget
        view_btn = toga.Button(
            "View Details",
            on_press=lambda w, ar=agent_record: asyncio.create_task(self._show_agent_details(ar)),
            style=Pack(
                width=100,
                font_size=11,
                margin_top=5
            )
        )
        card.add(view_btn)
        
        self.agent_list_container.add(card)
    
    async def _show_agent_details(self, agent_record):
        """Show detailed view of an agent."""
        self.selected_agent = agent_record
        
        # Clear detail container
        for child in list(self.detail_container.children):
            self.detail_container.remove(child)
        
        spec = agent_record.specification
        
        # Header
        header_box = toga.Box(style=Pack(
            direction=ROW,
            align_items="center",
            margin_bottom=20
        ))
        
        name_label = toga.Label(
            spec.name,
            style=Pack(
                font_size=24,
                font_weight="bold",
                flex=1
            )
        )
        header_box.add(name_label)
        
        # Action buttons
        if self.runtime:
            if agent_record.status == AgentStatus.INACTIVE:
                start_btn = toga.Button(
                    "▶️ Start",
                    on_press=lambda w: asyncio.create_task(self._start_agent(spec.id)),
                    style=Pack(
                        background_color="#28a745",
                        color="#ffffff",
                        margin_right=10
                    )
                )
                header_box.add(start_btn)
            else:
                stop_btn = toga.Button(
                    "⏹️ Stop",
                    on_press=lambda w: asyncio.create_task(self._stop_agent(spec.id)),
                    style=Pack(
                        background_color="#dc3545",
                        color="#ffffff",
                        margin_right=10
                    )
                )
                header_box.add(stop_btn)
        
        self.detail_container.add(header_box)
        
        # Info sections
        sections = [
            ("Description", spec.description),
            ("Goal", spec.goal),
            ("Type", spec.type.value),
            ("Trigger", spec.trigger_type),
            ("Status", agent_record.status.value),
            ("Created", agent_record.created_at.strftime("%Y-%m-%d %H:%M"))
        ]
        
        for label, value in sections:
            self._add_info_row(label, value)
        
        # Metrics section
        metrics_header = toga.Label(
            "Performance Metrics",
            style=Pack(
                font_size=16,
                font_weight="bold",
                margin_top=20,
                margin_bottom=10
            )
        )
        self.detail_container.add(metrics_header)
        
        metrics = agent_record.metrics
        metric_items = [
            ("Total Executions", str(metrics.total_executions)),
            ("Success Rate", f"{metrics.successful_executions / metrics.total_executions * 100:.1f}%" if metrics.total_executions > 0 else "N/A"),
            ("Average Runtime", f"{metrics.average_runtime_seconds:.2f}s"),
            ("Last Execution", metrics.last_execution_time.strftime("%Y-%m-%d %H:%M") if metrics.last_execution_time else "Never")
        ]
        
        for label, value in metric_items:
            self._add_info_row(label, value)
        
        # Recent executions
        if self.registry:
            history_header = toga.Label(
                "Recent Executions",
                style=Pack(
                    font_size=16,
                    font_weight="bold",
                    margin_top=20,
                    margin_bottom=10
                )
            )
            self.detail_container.add(history_header)
            
            # Load history
            history = await self.registry.get_agent_history(spec.id, limit=5)
            
            for execution in history:
                self._add_execution_row(execution)
    
    def _add_info_row(self, label: str, value: str):
        """Add an info row to details."""
        row = toga.Box(style=Pack(
            direction=ROW,
            margin_bottom=8
        ))
        
        label_widget = toga.Label(
            f"{label}:",
            style=Pack(
                width=120,
                font_weight="bold",
                font_size=12
            )
        )
        row.add(label_widget)
        
        value_widget = toga.Label(
            value,
            style=Pack(
                flex=1,
                font_size=12
            )
        )
        row.add(value_widget)
        
        self.detail_container.add(row)
    
    def _add_execution_row(self, execution: Dict[str, Any]):
        """Add execution history row."""
        row = toga.Box(style=Pack(
            direction=ROW,
            background_color="#f8f9fa",
            padding=8,
            margin_bottom=5,
            align_items="center"
        ))
        
        # Status icon
        status_icon = "✅" if execution['success'] else "❌"
        icon_label = toga.Label(
            status_icon,
            style=Pack(
                margin_right=10
            )
        )
        row.add(icon_label)
        
        # Time
        time_label = toga.Label(
            datetime.fromisoformat(execution['started_at']).strftime("%m/%d %H:%M"),
            style=Pack(
                font_size=11,
                width=100
            )
        )
        row.add(time_label)
        
        # Result or error
        if execution['success'] and execution.get('output'):
            result_text = str(execution['output'])[:50] + "..."
        elif execution.get('error'):
            result_text = f"Error: {execution['error'][:50]}..."
        else:
            result_text = "No output"
        
        result_label = toga.Label(
            result_text,
            style=Pack(
                flex=1,
                font_size=11,
                color="#6c757d"
            )
        )
        row.add(result_label)
        
        self.detail_container.add(row)
    
    async def _start_agent(self, agent_id: str):
        """Start an agent."""
        if self.runtime and self.registry:
            try:
                # Get agent spec
                agent_record = await self.registry.get_agent(agent_id)
                if agent_record:
                    # Deploy agent
                    await self.runtime.deploy_agent(agent_record.specification)
                    
                    # Update status
                    await self.registry.update_agent(
                        agent_id,
                        status=AgentStatus.ACTIVE
                    )
                    
                    # Refresh display
                    await self._load_agents()
                    await self._show_agent_details(agent_record)
                    
            except Exception as e:
                logger.error(f"Error starting agent: {e}")
    
    async def _stop_agent(self, agent_id: str):
        """Stop an agent."""
        if self.runtime and self.registry:
            try:
                # Stop agent
                await self.runtime.stop_agent(agent_id)
                
                # Update status
                await self.registry.update_agent(
                    agent_id,
                    status=AgentStatus.INACTIVE
                )
                
                # Refresh display
                await self._load_agents()
                agent_record = await self.registry.get_agent(agent_id)
                if agent_record:
                    await self._show_agent_details(agent_record)
                    
            except Exception as e:
                logger.error(f"Error stopping agent: {e}")
    
    async def _on_search_change(self, widget):
        """Handle search input change."""
        query = widget.value.lower()
        
        # Filter agents
        # Simple implementation - in production would be more sophisticated
        for child in list(self.agent_list_container.children):
            # Check if agent matches query
            # This is simplified - would need proper widget tracking
            pass


class AgentPanel:
    """
    Main agent management panel that combines all agent UI components.
    """
    
    def __init__(
        self,
        agent_builder: Optional[AgentBuilder] = None,
        agent_runtime: Optional[AgentRuntime] = None,
        agent_registry: Optional[AgentRegistry] = None
    ):
        """Initialize agent panel."""
        self.agent_builder = agent_builder
        self.agent_runtime = agent_runtime or AgentRuntime()
        self.agent_registry = agent_registry or AgentRegistry()
        
        # UI components
        self.container = None
        self.creation_wizard = None
        self.dashboard = None
        
        self._build_ui()
    
    def _build_ui(self) -> toga.Box:
        """Build the main agent panel UI."""
        self.container = toga.Box(style=Pack(
            direction=COLUMN
        ))
        
        # Tab selector
        self.tab_selector = toga.Box(style=Pack(
            direction=ROW,
            margin_bottom=10,
            background_color="#e9ecef",
            padding=5
        ))
        
        self.dashboard_tab = toga.Button(
            "Dashboard",
            on_press=lambda w: self._show_tab('dashboard'),
            style=Pack(
                margin_right=10,
                background_color="#007bff",
                color="#ffffff"
            )
        )
        self.tab_selector.add(self.dashboard_tab)
        
        self.create_tab = toga.Button(
            "Create Agent",
            on_press=lambda w: self._show_tab('create'),
            style=Pack(
                margin_right=10
            )
        )
        self.tab_selector.add(self.create_tab)
        
        self.container.add(self.tab_selector)
        
        # Content area
        self.content_area = toga.Box(style=Pack(
            flex=1
        ))
        self.container.add(self.content_area)
        
        # Initialize with dashboard
        self._show_tab('dashboard')
        
        return self.container
    
    def _show_tab(self, tab_name: str):
        """Show the specified tab."""
        # Clear content
        for child in list(self.content_area.children):
            self.content_area.remove(child)
        
        # Update tab styles
        if tab_name == 'dashboard':
            self.dashboard_tab.style.background_color = "#007bff"
            self.dashboard_tab.style.color = "#ffffff"
            self.create_tab.style.background_color = None
            self.create_tab.style.color = None
            
            # Show dashboard
            if not self.dashboard:
                self.dashboard = AgentDashboard(
                    self.agent_registry,
                    self.agent_runtime
                )
            self.content_area.add(self.dashboard.container)
            
        elif tab_name == 'create':
            self.create_tab.style.background_color = "#007bff"
            self.create_tab.style.color = "#ffffff"
            self.dashboard_tab.style.background_color = None
            self.dashboard_tab.style.color = None
            
            # Show creation wizard
            if not self.creation_wizard:
                self.creation_wizard = AgentCreationWizard(
                    on_agent_created=self._on_agent_created,
                    agent_builder=self.agent_builder
                )
            self.content_area.add(self.creation_wizard.container)
    
    async def _on_agent_created(self, spec):
        """Handle agent creation."""
        try:
            # Register agent
            agent_id = await self.agent_registry.register_agent(spec)
            
            # Deploy if manual trigger
            if spec.trigger_type == 'manual':
                await self.agent_runtime.deploy_agent(spec)
                await self.agent_registry.update_agent(
                    agent_id,
                    status=AgentStatus.ACTIVE
                )
            
            # Switch to dashboard
            self._show_tab('dashboard')
            
            # Refresh dashboard
            if self.dashboard:
                await self.dashboard._load_agents()
                
        except Exception as e:
            logger.error(f"Error handling agent creation: {e}")