"""
Progressive Feature Discovery System

This module implements a smart feature discovery system that introduces
users to Tektra's capabilities progressively and contextually.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable

import toga
from loguru import logger
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .animations.animation_manager import AnimationManager


class DiscoveryTrigger(Enum):
    """Triggers for feature discovery."""
    APP_START = "app_start"
    FIRST_MESSAGE = "first_message"
    VOICE_AVAILABLE = "voice_available"
    FILE_UPLOAD = "file_upload"
    AGENT_CREATION = "agent_creation"
    MULTIMODAL_AVAILABLE = "multimodal_available"
    MEMORY_INTEGRATION = "memory_integration"
    PERFORMANCE_MILESTONE = "performance_milestone"


class TutorialType(Enum):
    """Types of tutorials available."""
    TOOLTIP = "tooltip"
    OVERLAY = "overlay"
    GUIDED_TOUR = "guided_tour"
    CONTEXTUAL_HINT = "contextual_hint"
    FEATURE_SPOTLIGHT = "feature_spotlight"


@dataclass
class FeatureInfo:
    """Information about a discoverable feature."""
    name: str
    title: str
    description: str
    tutorial_type: TutorialType
    trigger: DiscoveryTrigger
    prerequisites: List[str]
    benefits: List[str]
    tutorial_content: str
    estimated_time: int  # seconds
    priority: int = 1  # 1=high, 5=low
    requires_confirmation: bool = False
    
    
@dataclass
class UserDiscoveryState:
    """User's feature discovery progress."""
    discovered_features: List[str]
    completed_tutorials: List[str]
    skipped_features: List[str]
    tutorial_preferences: Dict[str, bool]
    last_activity: datetime
    total_features_used: int
    discovery_score: float
    first_app_launch: datetime
    
    @classmethod
    def default(cls):
        """Create default discovery state for new users."""
        return cls(
            discovered_features=[],
            completed_tutorials=[],
            skipped_features=[],
            tutorial_preferences={
                "show_tooltips": True,
                "show_guided_tours": True,
                "auto_discover": True,
                "advanced_features": False
            },
            last_activity=datetime.now(),
            total_features_used=0,
            discovery_score=0.0,
            first_app_launch=datetime.now()
        )


class FeatureDiscoveryManager:
    """
    Manages progressive feature discovery and user onboarding.
    
    Features:
    - Context-aware feature introduction
    - User behavior analytics
    - Customizable tutorial experiences
    - Progress tracking and optimization
    """
    
    def __init__(self, config_dir: Path, app_instance, animation_manager: Optional[AnimationManager] = None):
        """
        Initialize feature discovery manager.
        
        Args:
            config_dir: Directory to store user discovery state
            app_instance: Reference to main app for UI integration
            animation_manager: Animation manager for micro-interactions
        """
        self.config_dir = config_dir
        self.app = app_instance
        self.state_file = config_dir / "discovery_state.json"
        self.animation_manager = animation_manager or AnimationManager()
        
        # Track interactive elements for micro-interactions
        self.interactive_elements = {}
        
        # Load or create user state
        self.user_state = self._load_user_state()
        
        # Feature registry
        self.features = self._initialize_features()
        
        # Discovery callbacks
        self.discovery_callbacks: Dict[str, List[Callable]] = {}
        
        # Tutorial UI components
        self.current_tutorial = None
        self.tutorial_overlay = None
        
        logger.info("Feature discovery manager initialized with micro-interactions")
    
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
    
    def _initialize_features(self) -> Dict[str, FeatureInfo]:
        """Initialize the feature registry."""
        return {
            "basic_chat": FeatureInfo(
                name="basic_chat",
                title="Chat with AI",
                description="Start conversations with Tektra's AI assistant",
                tutorial_type=TutorialType.GUIDED_TOUR,
                trigger=DiscoveryTrigger.APP_START,
                prerequisites=[],
                benefits=[
                    "Natural language conversations",
                    "Real-time AI responses",
                    "Context-aware assistance"
                ],
                tutorial_content="""
                Welcome to Tektra! Let's start with the basics:
                
                1. Type your message in the input box at the bottom
                2. Press Enter or click Send to talk to the AI
                3. Tektra will respond in real-time
                
                Try asking: "Hello, what can you help me with?"
                """,
                estimated_time=30,
                priority=1
            ),
            
            "voice_interaction": FeatureInfo(
                name="voice_interaction",
                title="Voice Conversations",
                description="Talk to Tektra using your voice",
                tutorial_type=TutorialType.FEATURE_SPOTLIGHT,
                trigger=DiscoveryTrigger.VOICE_AVAILABLE,
                prerequisites=["basic_chat"],
                benefits=[
                    "Hands-free interaction",
                    "Natural voice conversations",
                    "Voice-optimized responses"
                ],
                tutorial_content="""
                🎤 Voice Mode Now Available!
                
                Click the microphone button or say "Hey Tektra" to start:
                • Natural voice conversations
                • Hands-free operation
                • Voice-optimized responses
                
                Perfect for when your hands are busy!
                """,
                estimated_time=20,
                priority=2
            ),
            
            "file_analysis": FeatureInfo(
                name="file_analysis",
                title="File Upload & Analysis",
                description="Upload documents and images for AI analysis",
                tutorial_type=TutorialType.CONTEXTUAL_HINT,
                trigger=DiscoveryTrigger.FILE_UPLOAD,
                prerequisites=["basic_chat"],
                benefits=[
                    "Document summarization",
                    "Image analysis and description",
                    "Multi-format support"
                ],
                tutorial_content="""
                📎 File Analysis Available!
                
                Upload files for AI analysis:
                • Documents: PDF, Word, Text, Markdown
                • Images: JPEG, PNG, GIF
                • Data: JSON, CSV
                
                Just click the paperclip button and select your file!
                """,
                estimated_time=15,
                priority=2
            ),
            
            "ai_agents": FeatureInfo(
                name="ai_agents",
                title="AI Agent Creation",
                description="Create specialized AI agents for specific tasks",
                tutorial_type=TutorialType.GUIDED_TOUR,
                trigger=DiscoveryTrigger.AGENT_CREATION,
                prerequisites=["basic_chat", "file_analysis"],
                benefits=[
                    "Task-specific AI assistants",
                    "Code execution capabilities",
                    "Persistent agent memory"
                ],
                tutorial_content="""
                🤖 Create Your Own AI Agents!
                
                Specialized agents can help with specific tasks:
                • Data analysis agents
                • Code writing agents  
                • Research assistants
                • Custom workflows
                
                Try saying: "Create an agent that can analyze data files"
                """,
                estimated_time=45,
                priority=3
            ),
            
            "memory_system": FeatureInfo(
                name="memory_system",
                title="Conversation Memory",
                description="Tektra remembers context across conversations",
                tutorial_type=TutorialType.TOOLTIP,
                trigger=DiscoveryTrigger.MEMORY_INTEGRATION,
                prerequisites=["basic_chat"],
                benefits=[
                    "Context retention across sessions",
                    "Personalized responses",
                    "Learning from interactions"
                ],
                tutorial_content="""
                🧠 Memory System Active!
                
                Tektra now remembers our conversations:
                • Context carries between sessions
                • Personalized responses improve over time
                • Reference previous discussions naturally
                
                Try referencing something from an earlier conversation!
                """,
                estimated_time=10,
                priority=3
            ),
            
            "advanced_features": FeatureInfo(
                name="advanced_features",
                title="Advanced Capabilities",
                description="Multimodal processing and performance optimization",
                tutorial_type=TutorialType.OVERLAY,
                trigger=DiscoveryTrigger.PERFORMANCE_MILESTONE,
                prerequisites=["voice_interaction", "file_analysis", "ai_agents"],
                benefits=[
                    "Vision and image understanding",
                    "Performance analytics",
                    "Model customization"
                ],
                tutorial_content="""
                🚀 Advanced Features Unlocked!
                
                You're now a power user! Access to:
                • Advanced multimodal processing
                • Performance analytics and tuning
                • Model selection and customization
                • Beta feature access
                
                Check the Settings panel for advanced options.
                """,
                estimated_time=60,
                priority=4
            )
        }
    
    def _load_user_state(self) -> UserDiscoveryState:
        """Load user discovery state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    data['last_activity'] = datetime.fromisoformat(data['last_activity'])
                    data['first_app_launch'] = datetime.fromisoformat(data['first_app_launch'])
                    return UserDiscoveryState(**data)
        except Exception as e:
            logger.warning(f"Failed to load discovery state: {e}")
        
        return UserDiscoveryState.default()
    
    def _save_user_state(self):
        """Save user discovery state to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            data = asdict(self.user_state)
            data['last_activity'] = self.user_state.last_activity.isoformat()
            data['first_app_launch'] = self.user_state.first_app_launch.isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save discovery state: {e}")
    
    async def trigger_discovery(self, trigger: DiscoveryTrigger, context: Dict = None):
        """
        Trigger feature discovery based on user actions.
        
        Args:
            trigger: The discovery trigger event
            context: Additional context for the discovery
        """
        context = context or {}
        
        # Find features that match this trigger
        eligible_features = [
            feature for feature in self.features.values()
            if (feature.trigger == trigger and 
                feature.name not in self.user_state.discovered_features and
                feature.name not in self.user_state.skipped_features and
                self._check_prerequisites(feature))
        ]
        
        if not eligible_features:
            return
        
        # Sort by priority and select the most important
        eligible_features.sort(key=lambda f: f.priority)
        feature_to_show = eligible_features[0]
        
        # Check user preferences
        if not self._should_show_feature(feature_to_show):
            return
        
        logger.info(f"Triggering discovery for feature: {feature_to_show.name}")
        await self._show_feature_discovery(feature_to_show, context)
    
    def _check_prerequisites(self, feature: FeatureInfo) -> bool:
        """Check if feature prerequisites are met."""
        return all(
            prereq in self.user_state.discovered_features 
            for prereq in feature.prerequisites
        )
    
    def _should_show_feature(self, feature: FeatureInfo) -> bool:
        """Determine if feature should be shown based on user preferences."""
        # Check if user wants tutorials
        if not self.user_state.tutorial_preferences.get("auto_discover", True):
            return False
        
        # Check if it's too soon since last activity
        time_since_last = datetime.now() - self.user_state.last_activity
        if time_since_last < timedelta(minutes=1):  # Avoid overwhelming user
            return False
        
        # Check advanced features preference
        if (feature.priority >= 4 and 
            not self.user_state.tutorial_preferences.get("advanced_features", False)):
            return False
        
        return True
    
    async def _show_feature_discovery(self, feature: FeatureInfo, context: Dict):
        """Show feature discovery UI."""
        try:
            if feature.tutorial_type == TutorialType.TOOLTIP:
                await self._show_tooltip(feature, context)
            elif feature.tutorial_type == TutorialType.OVERLAY:
                await self._show_overlay(feature, context)
            elif feature.tutorial_type == TutorialType.GUIDED_TOUR:
                await self._show_guided_tour(feature, context)
            elif feature.tutorial_type == TutorialType.CONTEXTUAL_HINT:
                await self._show_contextual_hint(feature, context)
            elif feature.tutorial_type == TutorialType.FEATURE_SPOTLIGHT:
                await self._show_feature_spotlight(feature, context)
                
        except Exception as e:
            logger.error(f"Error showing feature discovery: {e}")
    
    async def _show_tooltip(self, feature: FeatureInfo, context: Dict):
        """Show a simple tooltip for the feature."""
        # Add system message to chat
        if hasattr(self.app, 'chat_panel'):
            message = f"💡 **{feature.title}** - {feature.description}"
            self.app.chat_panel.add_message("system", message)
    
    async def _show_contextual_hint(self, feature: FeatureInfo, context: Dict):
        """Show a contextual hint in the chat."""
        if hasattr(self.app, 'chat_panel'):
            message = f"""
🌟 **New Feature Available: {feature.title}**

{feature.description}

**Benefits:**
{chr(10).join(f"• {benefit}" for benefit in feature.benefits)}

{feature.tutorial_content}

*This hint won't show again. Enable it when you're ready!*
            """.strip()
            
            self.app.chat_panel.add_message("system", message)
    
    async def _show_feature_spotlight(self, feature: FeatureInfo, context: Dict):
        """Show a feature spotlight notification."""
        if hasattr(self.app, 'chat_panel'):
            message = f"""
✨ **{feature.title}** ✨

{feature.tutorial_content}

**Benefits:**
{chr(10).join(f"• {benefit}" for benefit in feature.benefits)}

*Ready to try it out?*
            """.strip()
            
            self.app.chat_panel.add_message("system", message)
    
    async def _show_overlay(self, feature: FeatureInfo, context: Dict):
        """Show an overlay tutorial dialog."""
        try:
            # Create tutorial dialog
            self.tutorial_overlay = toga.Window(
                title=f"Tutorial: {feature.title}",
                size=(500, 400),
                resizable=False
            )
            
            main_box = toga.Box(
                style=Pack(
                    direction=COLUMN,
                    padding=20,
                    background_color="#f8f9fa"
                )
            )
            
            # Title
            title_label = toga.Label(
                feature.title,
                style=Pack(
                    font_size=18,
                    font_weight="bold",
                    margin_bottom=10,
                    color="#1976d2"
                )
            )
            main_box.add(title_label)
            
            # Description
            desc_label = toga.Label(
                feature.description,
                style=Pack(
                    font_size=14,
                    margin_bottom=15,
                    color="#666666"
                )
            )
            main_box.add(desc_label)
            
            # Benefits
            benefits_label = toga.Label(
                "Benefits:",
                style=Pack(
                    font_weight="bold",
                    margin_bottom=5
                )
            )
            main_box.add(benefits_label)
            
            for benefit in feature.benefits:
                benefit_label = toga.Label(
                    f"• {benefit}",
                    style=Pack(
                        margin_left=10,
                        margin_bottom=3,
                        color="#333333"
                    )
                )
                main_box.add(benefit_label)
            
            # Tutorial content
            content_box = toga.ScrollContainer(
                content=toga.Label(
                    feature.tutorial_content,
                    style=Pack(
                        font_size=12,
                        color="#333333",
                        text_align="left"
                    )
                ),
                style=Pack(
                    flex=1,
                    margin_top=15,
                    margin_bottom=15,
                    background_color="#ffffff",
                    padding=10
                )
            )
            main_box.add(content_box)
            
            # Buttons
            button_box = toga.Box(
                style=Pack(
                    direction=ROW,
                    alignment="center"
                )
            )
            
            skip_button = toga.Button(
                "Skip",
                on_press=lambda w: self._skip_tutorial(feature),
                style=Pack(
                    margin_right=10,
                    background_color="#eeeeee"
                )
            )
            
            # Set up micro-interactions for skip button
            self._setup_button_micro_interactions(
                skip_button,
                f"skip_button_{feature.name}",
                {
                    "hover_scale": 1.03,
                    "press_scale": 0.97,
                    "hover_duration": 0.15,
                    "press_duration": 0.1,
                    "spring_back_duration": 0.15
                }
            )
            
            button_box.add(skip_button)
            
            complete_button = toga.Button(
                "Got it!",
                on_press=lambda w: self._complete_tutorial(feature),
                style=Pack(
                    background_color="#1976d2",
                    color="#ffffff"
                )
            )
            
            # Set up micro-interactions for complete button
            self._setup_button_micro_interactions(
                complete_button,
                f"complete_button_{feature.name}",
                {
                    "hover_scale": 1.05,
                    "press_scale": 0.95,
                    "hover_duration": 0.2,
                    "press_duration": 0.1,
                    "spring_back_duration": 0.2,
                    "enable_spring_back": True
                }
            )
            
            button_box.add(complete_button)
            
            main_box.add(button_box)
            
            self.tutorial_overlay.content = main_box
            self.tutorial_overlay.show()
            
        except Exception as e:
            logger.error(f"Error showing overlay tutorial: {e}")
            # Fallback to simple message
            await self._show_contextual_hint(feature, context)
    
    async def _show_guided_tour(self, feature: FeatureInfo, context: Dict):
        """Show a guided tour for the feature."""
        # For now, show as enhanced contextual hint
        # TODO: Implement step-by-step guided tour
        await self._show_overlay(feature, context)
    
    def _skip_tutorial(self, feature: FeatureInfo):
        """Mark feature as skipped."""
        self.user_state.skipped_features.append(feature.name)
        self._close_tutorial()
        self._save_user_state()
        logger.info(f"User skipped tutorial for: {feature.name}")
    
    def _complete_tutorial(self, feature: FeatureInfo):
        """Mark feature as discovered and tutorial as completed."""
        self.user_state.discovered_features.append(feature.name)
        self.user_state.completed_tutorials.append(feature.name)
        self.user_state.total_features_used += 1
        self.user_state.discovery_score += 1.0 / len(self.features)
        self.user_state.last_activity = datetime.now()
        
        self._close_tutorial()
        self._save_user_state()
        
        # Trigger callbacks
        self._notify_discovery_callbacks(feature.name)
        
        logger.info(f"User completed tutorial for: {feature.name}")
    
    def _close_tutorial(self):
        """Close the current tutorial overlay."""
        if self.tutorial_overlay:
            self.tutorial_overlay.close()
            self.tutorial_overlay = None
    
    def _notify_discovery_callbacks(self, feature_name: str):
        """Notify registered callbacks about feature discovery."""
        if feature_name in self.discovery_callbacks:
            for callback in self.discovery_callbacks[feature_name]:
                try:
                    callback(feature_name)
                except Exception as e:
                    logger.error(f"Error in discovery callback: {e}")
    
    def register_discovery_callback(self, feature_name: str, callback: Callable):
        """Register a callback for when a feature is discovered."""
        if feature_name not in self.discovery_callbacks:
            self.discovery_callbacks[feature_name] = []
        self.discovery_callbacks[feature_name].append(callback)
    
    def is_feature_discovered(self, feature_name: str) -> bool:
        """Check if a feature has been discovered."""
        return feature_name in self.user_state.discovered_features
    
    def get_discovery_progress(self) -> Dict:
        """Get user's discovery progress."""
        total_features = len(self.features)
        discovered_count = len(self.user_state.discovered_features)
        completed_tutorials = len(self.user_state.completed_tutorials)
        
        return {
            "total_features": total_features,
            "discovered_count": discovered_count,
            "completed_tutorials": completed_tutorials,
            "discovery_percentage": (discovered_count / total_features) * 100,
            "tutorial_completion_rate": (completed_tutorials / max(1, discovered_count)) * 100,
            "discovery_score": self.user_state.discovery_score,
            "days_since_first_launch": (datetime.now() - self.user_state.first_app_launch).days
        }
    
    def reset_discovery_state(self):
        """Reset user's discovery state (for testing or user request)."""
        self.user_state = UserDiscoveryState.default()
        self._save_user_state()
        logger.info("Discovery state reset")
    
    def update_tutorial_preferences(self, preferences: Dict[str, bool]):
        """Update user's tutorial preferences."""
        self.user_state.tutorial_preferences.update(preferences)
        self._save_user_state()
        logger.info(f"Tutorial preferences updated: {preferences}")


# Global discovery manager instance
_discovery_manager = None

def get_discovery_manager() -> Optional[FeatureDiscoveryManager]:
    """Get the global discovery manager instance."""
    return _discovery_manager

def initialize_discovery_manager(config_dir: Path, app_instance) -> FeatureDiscoveryManager:
    """Initialize the global discovery manager."""
    global _discovery_manager
    _discovery_manager = FeatureDiscoveryManager(config_dir, app_instance)
    return _discovery_manager