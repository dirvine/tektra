"""
Agent Panel - AI Agent Management Interface

This module provides the interface for creating and managing AI agents.
Currently provides basic functionality with room for future expansion.
"""

import asyncio
from typing import Optional

import toga
from loguru import logger
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .animations.animation_manager import AnimationManager


class AgentPanel:
    """
    Agent management interface for Tektra AI Assistant.
    
    This provides a basic interface for agent creation and management.
    Future versions will include more sophisticated agent building capabilities.
    """

    def __init__(self, agent_builder=None, agent_runtime=None, agent_registry=None, animation_manager: Optional[AnimationManager] = None):
        """
        Initialize the agent panel.
        
        Args:
            agent_builder: The agent builder service
            agent_runtime: The agent runtime service
            agent_registry: The agent registry service
            animation_manager: Animation manager for micro-interactions
        """
        self.agent_builder = agent_builder
        self.agent_runtime = agent_runtime
        self.agent_registry = agent_registry
        self.animation_manager = animation_manager or AnimationManager()
        
        # Current state
        self.current_tab = "create"
        self.agents = []
        
        # Track interactive elements for micro-interactions
        self.interactive_elements = {}
        
        # Build the UI
        self.container = self._build_interface()
        
        logger.info("Agent panel initialized with micro-interactions")

    def _build_interface(self) -> toga.Box:
        """Build the main agent interface."""
        main_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                flex=1,
                padding=20,
                background_color="#f8f9fa"
            )
        )
        
        # Header
        header = toga.Label(
            "AI Agent Management",
            style=Pack(
                font_size=18,
                font_weight="bold",
                margin_bottom=20,
                color="#1976d2"
            )
        )
        main_container.add(header)
        
        # Tab navigation
        tab_container = self._build_tab_navigation()
        main_container.add(tab_container)
        
        # Content area
        self.content_area = toga.Box(
            style=Pack(
                direction=COLUMN,
                flex=1,
                margin_top=20,
                background_color="#ffffff",
                padding=20
            )
        )
        main_container.add(self.content_area)
        
        # Show initial tab
        self._show_tab("create")
        
        return main_container

    def _build_tab_navigation(self) -> toga.Box:
        """Build the tab navigation."""
        tab_container = toga.Box(
            style=Pack(
                direction=ROW,
                margin_bottom=10
            )
        )
        
        # Create tab button
        self.create_tab_button = toga.Button(
            "Create Agent",
            on_press=lambda x: self._show_tab("create"),
            style=Pack(
                width=120,
                height=40,
                margin_right=10,
                background_color="#1976d2",
                color="#ffffff"
            )
        )
        
        # Set up micro-interactions for create tab button
        self._setup_button_micro_interactions(
            self.create_tab_button, 
            "create_tab_button",
            {
                "hover_scale": 1.03,
                "press_scale": 0.97,
                "hover_duration": 0.15,
                "press_duration": 0.1,
                "spring_back_duration": 0.15
            }
        )
        
        tab_container.add(self.create_tab_button)
        
        # Dashboard tab button
        self.dashboard_tab_button = toga.Button(
            "Dashboard",
            on_press=lambda x: self._show_tab("dashboard"),
            style=Pack(
                width=120,
                height=40,
                margin_right=10,
                background_color="#eeeeee",
                color="#333333"
            )
        )
        
        # Set up micro-interactions for dashboard tab button
        self._setup_button_micro_interactions(
            self.dashboard_tab_button, 
            "dashboard_tab_button",
            {
                "hover_scale": 1.03,
                "press_scale": 0.97,
                "hover_duration": 0.15,
                "press_duration": 0.1,
                "spring_back_duration": 0.15
            }
        )
        
        tab_container.add(self.dashboard_tab_button)
        
        return tab_container

    def _setup_button_micro_interactions(self, button: toga.Button, button_id: str, config: dict = None):
        """Set up micro-interactions for a button."""
        try:
            micro_manager = self.animation_manager.micro_interaction_manager
            element_id = micro_manager.setup_button_interactions(
                button,
                button_id=button_id,
                interaction_config=config
            )
            self.interactive_elements[button_id] = element_id
            logger.debug(f"Set up micro-interactions for button: {button_id}")
        except Exception as e:
            logger.debug(f"Could not set up micro-interactions for {button_id}: {e}")

    def _show_tab(self, tab_name: str):
        """Show a specific tab."""
        self.current_tab = tab_name
        
        # Update button styles
        if tab_name == "create":
            self.create_tab_button.style.background_color = "#1976d2"
            self.create_tab_button.style.color = "#ffffff"
            self.dashboard_tab_button.style.background_color = "#eeeeee"
            self.dashboard_tab_button.style.color = "#333333"
        else:
            self.create_tab_button.style.background_color = "#eeeeee"
            self.create_tab_button.style.color = "#333333"
            self.dashboard_tab_button.style.background_color = "#1976d2"
            self.dashboard_tab_button.style.color = "#ffffff"
        
        # Clear content area
        self.content_area.clear()
        
        # Show appropriate content
        if tab_name == "create":
            self._show_create_tab()
        elif tab_name == "dashboard":
            self._show_dashboard_tab()

    def _show_create_tab(self):
        """Show the agent creation tab."""
        # Title
        title = toga.Label(
            "Create New AI Agent",
            style=Pack(
                font_size=16,
                font_weight="bold",
                margin_bottom=20,
                color="#333333"
            )
        )
        self.content_area.add(title)
        
        # Description input
        desc_label = toga.Label(
            "Describe what your agent should do:",
            style=Pack(
                font_size=14,
                margin_bottom=10,
                color="#666666"
            )
        )
        self.content_area.add(desc_label)
        
        self.agent_description = toga.MultilineTextInput(
            placeholder="Example: Create a data analysis agent that can process CSV files, generate charts, and provide insights about the data patterns.",
            style=Pack(
                width=600,
                height=100,
                margin_bottom=20,
                padding=10,
                font_size=14
            )
        )
        self.content_area.add(self.agent_description)
        
        # Agent type selection
        type_label = toga.Label(
            "Agent Type:",
            style=Pack(
                font_size=14,
                margin_bottom=10,
                color="#666666"
            )
        )
        self.content_area.add(type_label)
        
        self.agent_type = toga.Selection(
            items=["CODE", "TOOL_CALLING", "HYBRID"],
            style=Pack(
                width=200,
                margin_bottom=20
            )
        )
        self.content_area.add(self.agent_type)
        
        # Create button
        create_button = toga.Button(
            "Create Agent",
            on_press=self._create_agent,
            style=Pack(
                width=150,
                height=40,
                background_color="#4caf50",
                color="#ffffff",
                margin_top=20
            )
        )
        
        # Set up micro-interactions for create button
        self._setup_button_micro_interactions(
            create_button, 
            "create_agent_button",
            {
                "hover_scale": 1.05,
                "press_scale": 0.95,
                "hover_duration": 0.2,
                "press_duration": 0.1,
                "spring_back_duration": 0.2,
                "enable_spring_back": True
            }
        )
        
        self.content_area.add(create_button)
        
        # Status area
        self.create_status = toga.Label(
            "",
            style=Pack(
                margin_top=20,
                color="#666666"
            )
        )
        self.content_area.add(self.create_status)

    def _show_dashboard_tab(self):
        """Show the agent dashboard tab."""
        # Title
        title = toga.Label(
            "Agent Dashboard",
            style=Pack(
                font_size=16,
                font_weight="bold",
                margin_bottom=20,
                color="#333333"
            )
        )
        self.content_area.add(title)
        
        # Agent list
        if not self.agents:
            no_agents_label = toga.Label(
                "No agents created yet. Use the 'Create Agent' tab to build your first agent.",
                style=Pack(
                    font_size=14,
                    color="#666666",
                    margin_top=50,
                    text_align="center"
                )
            )
            self.content_area.add(no_agents_label)
        else:
            # Show agent list
            for i, agent in enumerate(self.agents):
                agent_item = self._build_agent_item(agent, i)
                self.content_area.add(agent_item)

    def _build_agent_item(self, agent: dict, index: int) -> toga.Box:
        """Build a single agent item for the dashboard."""
        item_container = toga.Box(
            style=Pack(
                direction=ROW,
                margin_bottom=10,
                padding=15,
                background_color="#f8f9fa"
            )
        )
        
        # Agent info
        info_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                flex=1
            )
        )
        
        name_label = toga.Label(
            agent.get("name", f"Agent {index + 1}"),
            style=Pack(
                font_size=14,
                font_weight="bold",
                color="#333333"
            )
        )
        info_container.add(name_label)
        
        desc_label = toga.Label(
            agent.get("description", "No description")[:80] + "...",
            style=Pack(
                font_size=12,
                color="#666666"
            )
        )
        info_container.add(desc_label)
        
        item_container.add(info_container)
        
        # Status
        status_label = toga.Label(
            agent.get("status", "Unknown"),
            style=Pack(
                font_size=12,
                color="#4caf50" if agent.get("status") == "Running" else "#666666",
                width=80
            )
        )
        item_container.add(status_label)
        
        # Actions
        actions_container = toga.Box(
            style=Pack(direction=ROW)
        )
        
        stop_button = toga.Button(
            "Stop",
            on_press=lambda x: self._stop_agent(index),
            style=Pack(
                width=60,
                height=30,
                background_color="#f44336",
                color="#ffffff",
                margin_left=10
            )
        )
        
        # Set up micro-interactions for stop button
        self._setup_button_micro_interactions(
            stop_button, 
            f"stop_agent_button_{index}",
            {
                "hover_scale": 1.05,
                "press_scale": 0.95,
                "hover_duration": 0.15,
                "press_duration": 0.08,
                "spring_back_duration": 0.15,
                "enable_spring_back": True
            }
        )
        
        actions_container.add(stop_button)
        
        item_container.add(actions_container)
        
        return item_container

    def _create_agent(self, widget):
        """Handle agent creation."""
        description = self.agent_description.value.strip()
        if not description:
            self.create_status.text = "Please enter a description for your agent."
            return
        
        self.create_status.text = "Creating agent..."
        
        # For now, just add a mock agent
        agent = {
            "name": f"Agent {len(self.agents) + 1}",
            "description": description,
            "type": self.agent_type.value,
            "status": "Running",
            "created_at": "Now"
        }
        
        self.agents.append(agent)
        
        self.create_status.text = "✅ Agent created successfully!"
        
        # Clear the form
        self.agent_description.value = ""
        
        logger.info(f"Mock agent created: {agent}")

    def _stop_agent(self, index: int):
        """Stop an agent."""
        if 0 <= index < len(self.agents):
            self.agents[index]["status"] = "Stopped"
            # Refresh the dashboard
            if self.current_tab == "dashboard":
                self._show_dashboard_tab()
            logger.info(f"Agent {index} stopped")

    def get_agent_count(self) -> int:
        """Get the number of agents."""
        return len(self.agents)

    def get_running_agents(self) -> list:
        """Get list of running agents."""
        return [agent for agent in self.agents if agent.get("status") == "Running"]