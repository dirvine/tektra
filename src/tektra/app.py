"""
Tektra AI Assistant - Main Application

This module contains the main TektraApp class built with Briefcase and Toga,
providing a native desktop experience for the AI assistant.
"""

import asyncio
import platform
from pathlib import Path

import toga
from loguru import logger
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .agents import AgentBuilder, AgentRegistry
from .agents.simple_runtime import SimpleAgentRuntime
from .ai.multimodal import MultimodalProcessor

# Import Tektra components
from .ai.simple_llm import SimpleLLM
from .ai.smart_router import SmartRouter
from .data.storage import DataStorage
from .gui.agent_panel import AgentPanel
from .gui.chat_panel import ChatManager, ChatPanel
from .gui.feature_discovery import (
    initialize_discovery_manager, 
    get_discovery_manager,
    DiscoveryTrigger
)
from .gui.progress_dialog import ProgressDialog, ProgressTracker
from .gui.progress_overlay import ProgressOverlay
from .gui.startup_dialog import StartupDialog
from .gui.themes import theme_manager
from .memory.memos_integration import TektraMemOSIntegration
from .models.model_interface import default_registry, default_factory, ModelConfig
from .models.model_manager import ModelManager
from .models.model_updater import ModelUpdateManager
from .utils.config import AppConfig
try:
    from .voice.pipeline_embedded import EmbeddedVoicePipeline
    VOICE_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Voice pipeline not available: {e}")
    VOICE_PIPELINE_AVAILABLE = False
    EmbeddedVoicePipeline = None

try:
    from .voice.unmute_embedded import EmbeddedUnmute
    UNMUTE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unmute not available: {e}")
    UNMUTE_AVAILABLE = False
    EmbeddedUnmute = None


class TektraApp(toga.App):
    """
    Main Tektra AI Assistant application.

    Provides a native desktop interface for voice-interactive AI assistance
    with hybrid Unmute + Qwen architecture.
    """

    def startup(self):
        """
        Initialize the application.

        This method is called by Toga when the app starts.
        """
        logger.info("Starting Tektra AI Assistant...")

        # Initialize configuration
        self.config = AppConfig()

        # Initialize data storage
        self.data_storage = DataStorage()
        
        # Progress components
        self.progress_dialog = None
        self.progress_overlay = None
        self.main_container = None

        # Backend components (initialized later)
        self.simple_llm = None
        self.voice_pipeline = None
        self.smart_router = None
        self.multimodal_processor = None
        self.conversation_memory = None
        
        # Embedded voice components
        self.model_manager = None
        self.unmute = None

        # Agent system components
        self.agent_builder = None
        self.agent_runtime = None
        self.agent_registry = None
        self.agent_panel = None

        # GUI components
        self.chat_panel = None
        self.chat_manager = None
        
        # Feature discovery system
        self.discovery_manager = None
        
        # Enhanced model management
        self.model_registry = default_registry
        self.model_factory = default_factory
        self.model_updater = None

        # Application state
        self.is_initialized = False
        self.is_voice_active = False
        self.is_camera_active = False
        self.initialization_progress = 0.0
        self.initialization_status = "Starting up..."
        self.startup_mode = "full"  # Can be "full" or "api"

        # Create main window
        self.main_window = toga.MainWindow(title="Tektra AI Assistant")

        # Build the main interface
        self.build_interface()

        # Show the window
        self.main_window.show()

        # Skip startup dialog - go directly to full mode with progress
        self.startup_mode = "full"
        
        # Start backend initialization immediately
        asyncio.create_task(self._handle_startup())

        logger.info("Tektra app startup complete")
    
    async def _handle_startup(self):
        """Handle startup mode selection."""
        # Wait a moment for dialog to be ready
        await asyncio.sleep(0.1)
        
        # In a real app, we'd wait for the dialog result
        # For now, check if user has a saved preference
        saved_mode = self.config.get("startup_mode", None)
        
        if saved_mode:
            self.startup_mode = saved_mode
            # Close startup dialog if still open
            await self.initialize_backend_systems()
        else:
            # Wait for dialog completion (simplified for demo)
            await asyncio.sleep(2)
            # Default to full mode
            self.startup_mode = "full"
            await self.initialize_backend_systems()

    def build_interface(self):
        """Build the main user interface."""
        # Get theme
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        # Create main container with theme background
        self.main_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                background_color=colors.background
            )
        )

        # Header with title and status
        header = self.build_header()
        self.main_container.add(header)

        # Main content area
        content_area = self.build_content_area()
        self.main_container.add(content_area)

        # Status bar
        status_bar = self.build_status_bar()
        self.main_container.add(status_bar)

        # Set as main window content
        self.main_window.content = self.main_container
        
        # Create progress overlay (hidden by default)
        self.progress_overlay = ProgressOverlay(self.main_container)

        # Create app menu
        self.create_menu()

    def update_status_indicator(self, indicator_name: str, status: str):
        """Update a status indicator with the given status."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        # Get the indicator components
        if indicator_name == "model":
            dot = self.model_status_dot
            indicator = self.model_status_indicator
        elif indicator_name == "voice":
            dot = self.voice_status_dot
            indicator = self.voice_status_indicator
        elif indicator_name == "connection":
            dot = self.connection_status_dot
            indicator = self.connection_status_indicator
        else:
            return
            
        # Update dot color based on status
        if status == "active" or status == "ready":
            dot.style.background_color = colors.success
        elif status == "loading" or status == "working":
            dot.style.background_color = colors.warning
        elif status == "error" or status == "offline":
            dot.style.background_color = colors.error
        else:  # inactive/disabled
            dot.style.background_color = colors.text_disabled
    
    def build_header(self) -> toga.Box:
        """Build the application header."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        header = toga.Box(
            style=Pack(
                direction=ROW,
                padding=(spacing["md"], spacing["lg"]),
                background_color=colors.surface,
            )
        )

        # App title and logo
        title_box = toga.Box(style=Pack(direction=COLUMN, flex=1))

        self.app_title = toga.Label(
            "Tektra AI Assistant",
            style=Pack(
                font_size=theme.typography["heading2"]["size"],
                font_weight=theme.typography["heading2"]["weight"],
                color=colors.text_primary,
                margin_bottom=spacing["xs"]
            ),
        )

        self.app_subtitle = toga.Label(
            "Voice-Interactive AI with Embedded Models",
            style=Pack(
                font_size=theme.typography["caption"]["size"],
                color=colors.text_secondary
            ),
        )

        title_box.add(self.app_title)
        title_box.add(self.app_subtitle)
        header.add(title_box)

        # Status indicators
        status_box = toga.Box(style=Pack(direction=ROW))

        # Create modern pill-style status indicators
        self.model_status_indicator = toga.Box(
            style=Pack(
                direction=ROW,
                margin_right=spacing["sm"],
                padding=(spacing["xs"], spacing["md"]),
                background_color=colors.surface
            )
        )
        self.model_status_dot = toga.Box(
            style=Pack(
                width=8,
                height=8,
                background_color=colors.text_disabled,
                margin_right=spacing["xs"]
            )
        )
        self.model_status_text = toga.Label(
            "Models",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["caption"]["size"],
                font_weight="normal"
            )
        )
        self.model_status_indicator.add(self.model_status_dot)
        self.model_status_indicator.add(self.model_status_text)

        self.voice_status_indicator = toga.Box(
            style=Pack(
                direction=ROW,
                margin_right=spacing["sm"],
                padding=(spacing["xs"], spacing["md"]),
                background_color=colors.surface
            )
        )
        self.voice_status_dot = toga.Box(
            style=Pack(
                width=8,
                height=8,
                background_color=colors.text_disabled,
                margin_right=spacing["xs"]
            )
        )
        self.voice_status_text = toga.Label(
            "Voice",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["caption"]["size"],
                font_weight="normal"
            )
        )
        self.voice_status_indicator.add(self.voice_status_dot)
        self.voice_status_indicator.add(self.voice_status_text)

        self.connection_status_indicator = toga.Box(
            style=Pack(
                direction=ROW,
                padding=(spacing["xs"], spacing["md"]),
                background_color=colors.surface
            )
        )
        self.connection_status_dot = toga.Box(
            style=Pack(
                width=8,
                height=8,
                background_color=colors.text_disabled,
                margin_right=spacing["xs"]
            )
        )
        self.connection_status_text = toga.Label(
            "Services",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["caption"]["size"],
                font_weight="normal"
            )
        )
        self.connection_status_indicator.add(self.connection_status_dot)
        self.connection_status_indicator.add(self.connection_status_text)

        status_box.add(self.model_status_indicator)
        status_box.add(self.voice_status_indicator)
        status_box.add(self.connection_status_indicator)

        header.add(status_box)

        return header

    def build_content_area(self) -> toga.Box:
        """Build the main content area."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        content = toga.Box(
            style=Pack(
                direction=ROW,
                flex=1,
                background_color=colors.background
            )
        )

        # Left sidebar
        left_sidebar = self.build_left_sidebar()
        content.add(left_sidebar)

        # Center chat area
        chat_area = self.build_chat_area()
        content.add(chat_area)

        # Right sidebar
        right_sidebar = self.build_right_sidebar()
        content.add(right_sidebar)

        return content

    def build_left_sidebar(self) -> toga.Box:
        """Build the left sidebar with controls."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        sidebar = toga.Box(
            style=Pack(
                direction=COLUMN,
                width=280,
                padding=spacing["lg"],
                background_color=colors.surface
            )
        )

        # Models section
        models_label = toga.Label(
            "AI Models",
            style=Pack(
                font_size=theme.typography["heading3"]["size"],
                font_weight=theme.typography["heading3"]["weight"],
                color=colors.text_primary,
                margin_bottom=spacing["sm"]
            ),
        )
        sidebar.add(models_label)

        self.model_info_label = toga.Label(
            "Initializing...",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["caption"]["size"],
                margin_bottom=spacing["lg"]
            ),
        )
        sidebar.add(self.model_info_label)

        # Voice controls section
        voice_label = toga.Label(
            "Voice Controls",
            style=Pack(
                font_size=theme.typography["heading3"]["size"],
                font_weight=theme.typography["heading3"]["weight"],
                color=colors.text_primary,
                margin_bottom=spacing["sm"]
            ),
        )
        sidebar.add(voice_label)

        self.voice_toggle_btn = toga.Button(
            "Start Voice Mode",
            on_press=self.toggle_voice_mode,
            style=Pack(
                width=240,
                margin_bottom=spacing["sm"],
                background_color=colors.primary,
                color="#ffffff",
                padding=(spacing["sm"], spacing["md"]),
                font_size=theme.typography["button"]["size"],
                font_weight=theme.typography["button"]["weight"]
            ),
            enabled=False,
        )
        sidebar.add(self.voice_toggle_btn)

        self.push_to_talk_btn = toga.Button(
            "Push to Talk",
            style=Pack(
                width=240,
                margin_bottom=spacing["lg"],
                background_color=colors.surface,
                color=colors.primary,
                padding=(spacing["sm"], spacing["md"]),
                font_size=theme.typography["button"]["size"],
                font_weight=theme.typography["button"]["weight"]
            ),
            enabled=False
        )
        sidebar.add(self.push_to_talk_btn)

        # Camera controls section
        camera_label = toga.Label(
            "Vision", style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5)
        )
        sidebar.add(camera_label)

        self.camera_toggle_btn = toga.Button(
            "Enable Camera",
            on_press=self.toggle_camera,
            style=Pack(width=200, margin_bottom=15),
            enabled=False,
        )
        sidebar.add(self.camera_toggle_btn)

        # File upload section
        files_label = toga.Label(
            "File Analysis",
            style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5),
        )
        sidebar.add(files_label)

        self.upload_file_btn = toga.Button(
            "Upload File",
            on_press=self.upload_file,
            style=Pack(width=200, margin_bottom=5),
            enabled=False,
        )
        sidebar.add(self.upload_file_btn)

        # Settings section
        settings_label = toga.Label(
            "Settings", style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5)
        )
        sidebar.add(settings_label)

        self.settings_btn = toga.Button(
            "Preferences",
            on_press=self.show_settings,
            style=Pack(width=200, margin_bottom=5),
        )
        sidebar.add(self.settings_btn)

        return sidebar

    def build_chat_area(self) -> toga.Box:
        """Build the central chat interface using ChatPanel."""
        # Initialize chat panel with callbacks
        self.chat_panel = ChatPanel(
            on_message_send=self.handle_chat_message,
            on_voice_toggle=self.toggle_voice_mode,
            on_file_upload=self.upload_file,
        )

        # Initialize chat manager
        self.chat_manager = ChatManager(self.chat_panel)
        
        # Trigger app start discovery after UI is ready
        if self.discovery_manager:
            asyncio.create_task(
                self.discovery_manager.trigger_discovery(DiscoveryTrigger.APP_START)
            )

        return self.chat_panel.widget

    def build_right_sidebar(self) -> toga.Box:
        """Build the right sidebar with analytics and status."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        sidebar = toga.Box(
            style=Pack(
                direction=COLUMN,
                width=240,
                padding=spacing["lg"],
                background_color=colors.surface
            )
        )

        # Conversation stats
        stats_label = toga.Label(
            "Session Stats",
            style=Pack(
                font_size=theme.typography["heading3"]["size"],
                font_weight=theme.typography["heading3"]["weight"],
                color=colors.text_primary,
                margin_bottom=spacing["sm"]
            ),
        )
        sidebar.add(stats_label)

        self.message_count_label = toga.Label(
            "Messages: 0",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["body2"]["size"],
                margin_bottom=spacing["xs"]
            )
        )
        sidebar.add(self.message_count_label)

        self.routing_stats_label = toga.Label(
            "Routing: -",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["body2"]["size"],
                margin_bottom=spacing["lg"]
            )
        )
        sidebar.add(self.routing_stats_label)

        # System performance
        performance_label = toga.Label(
            "Performance",
            style=Pack(
                font_size=theme.typography["heading3"]["size"],
                font_weight=theme.typography["heading3"]["weight"],
                color=colors.text_primary,
                margin_bottom=spacing["sm"]
            ),
        )
        sidebar.add(performance_label)

        self.memory_usage_label = toga.Label(
            "Memory: -",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["body2"]["size"],
                margin_bottom=spacing["xs"]
            )
        )
        sidebar.add(self.memory_usage_label)

        self.response_time_label = toga.Label(
            "Response: -",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["body2"]["size"],
                margin_bottom=spacing["lg"]
            )
        )
        sidebar.add(self.response_time_label)

        return sidebar

    def build_status_bar(self) -> toga.Box:
        """Build the bottom status bar."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        status_bar = toga.Box(
            style=Pack(
                direction=ROW,
                padding=(spacing["sm"], spacing["lg"]),
                background_color=colors.surface,
            )
        )

        # Current status
        self.status_label = toga.Label(
            "Initializing Tektra AI Assistant...",
            style=Pack(
                flex=1,
                color=colors.text_primary,
                font_size=theme.typography["caption"]["size"]
            ),
        )
        status_bar.add(self.status_label)

        # Progress indicator
        self.progress_label = toga.Label(
            "0%",
            style=Pack(
                color=colors.text_secondary,
                font_size=theme.typography["caption"]["size"],
                font_weight="normal"
            )
        )
        status_bar.add(self.progress_label)

        return status_bar

    def create_menu(self):
        """Create the application menu."""
        # File menu
        file_menu = toga.Group("File")
        self.main_window.app.commands.add(
            toga.Command(
                self.upload_file,
                text="Upload File",
                tooltip="Upload a file for analysis",
                group=file_menu,
                section=0,
            )
        )

        # Voice menu
        voice_menu = toga.Group("Voice")
        self.main_window.app.commands.add(
            toga.Command(
                self.toggle_voice_mode,
                text="Toggle Voice Mode",
                tooltip="Start or stop voice conversation",
                group=voice_menu,
                section=0,
            )
        )

        # Help menu
        help_menu = toga.Group("Help")
        self.main_window.app.commands.add(
            toga.Command(
                self.show_about,
                text="About Tektra",
                tooltip="About this application",
                group=help_menu,
                section=0,
            )
        )

    async def initialize_backend_systems(self):
        """Initialize all backend AI systems."""
        try:
            logger.info("Initializing backend systems...")
            
            # Show progress overlay instead of separate dialog
            if self.progress_overlay:
                self.progress_overlay.show(
                    title="Initializing Tektra AI Assistant",
                    cancellable=True
                )
            
            # Create progress tracker for overlay
            tracker = ProgressTracker(self.progress_overlay if self.progress_overlay else None)
            tracker.add_step("discovery", 5)
            tracker.add_step("models", 10)
            tracker.add_step("memory", 10)
            tracker.add_step("llm", 40)
            tracker.add_step("voice", 20)
            tracker.add_step("finalize", 15)

            # Update status
            await self.update_status("Initializing AI models...", 0)
            tracker.update_step("discovery", 0, "Initializing feature discovery...")

            # Get app data directory
            app_data_dir = Path.home() / ".tektra"
            app_data_dir.mkdir(exist_ok=True)
            
            # Initialize feature discovery manager
            await self.update_status("Initializing feature discovery...", 5)
            self.discovery_manager = initialize_discovery_manager(app_data_dir, self)
            tracker.complete_step("discovery")
            
            # Initialize model manager and updater
            self.model_manager = ModelManager(app_data_dir)
            self.model_updater = ModelUpdateManager(
                registry=self.model_registry,
                factory=self.model_factory,
                models_dir=app_data_dir / "models"
            )
            await self.model_updater.start()
            await self.update_status("Model management initialized", 15)
            tracker.complete_step("models")

            # Initialize multimodal processor
            self.multimodal_processor = MultimodalProcessor()
            await self.update_status("Multimodal processor ready", 20)

            # Initialize conversation memory with MemOS
            tracker.update_step("memory", 0, "Initializing conversation memory...")
            await self.update_status("Initializing conversation memory...", 25)
            try:
                self.conversation_memory = TektraMemOSIntegration(
                    memory_dir=app_data_dir / "memory",
                    user_id="default_user"  # Could be made configurable
                )
                await self.update_status("Conversation memory ready", 27)
                logger.info("MemOS conversation memory initialized")
            except Exception as e:
                logger.warning(f"MemOS initialization failed, using fallback: {e}")
                await self.update_status("Memory system ready (fallback)", 27)
            tracker.complete_step("memory")

            # Check if we should skip model loading
            startup_mode = getattr(self, 'startup_mode', 'full')
            
            # Initialize Simple LLM (skip in API mode)
            if startup_mode == "full":
                # Use a small, fast model that actually works
                model_name = self.config.get("llm_model_name", "microsoft/Phi-3-mini-4k-instruct")
                
                # Create progress callback that updates overlay
                async def model_progress_callback(progress: float, status: str, bytes_downloaded: int = 0, total_bytes: int = 0):
                    if self.progress_overlay and self.progress_overlay.cancelled:
                        raise Exception("Model loading cancelled")
                    
                    # Update progress tracker
                    tracker.update_step("llm", progress * 100, status)
                    
                    # Calculate overall progress (llm is 30% to 60% of total)
                    overall_progress = 30 + (progress * 30)
                    
                    # Update overlay with progress
                    if self.progress_overlay:
                        self.progress_overlay.update_progress(
                            progress=overall_progress,
                            operation=status,
                            bytes_downloaded=bytes_downloaded,
                            total_bytes=total_bytes
                        )
                    
                    await self.update_status(f"Loading model: {status}", overall_progress)
                
                self.simple_llm = SimpleLLM(model_name=model_name)
                llm_success = await self.simple_llm.initialize(model_progress_callback)
                tracker.complete_step("llm")
            else:
                # API mode - skip local model loading
                await self.update_status("Configuring API mode...", 50)
                tracker.update_step("llm", 100, "API mode configured")
                tracker.complete_step("llm")
                llm_success = True
                # TODO: Initialize API client here

            if llm_success:
                self.model_status_indicator.text = "🟢 Models"
                await self.update_status("Language model loaded successfully", 60)
            else:
                self.model_status_indicator.text = "🔴 Models"
                await self.update_status("Failed to load language model", 60)

            # Initialize embedded voice system (skip in API mode)
            if startup_mode == "full":
                tracker.update_step("voice", 0, "Initializing embedded voice system...")
                await self.update_status("Initializing embedded voice system...", 70)
                
                # Pass progress dialog to voice initialization
                voice_success = await self.initialize_embedded_voice_system(self.progress_dialog)
                tracker.complete_step("voice")
            else:
                # API mode - skip voice system
                await self.update_status("Voice features disabled in API mode", 70)
                tracker.update_step("voice", 100, "Voice disabled in API mode")
                tracker.complete_step("voice")
                voice_success = False

            if voice_success:
                self.update_status_indicator("voice", "ready")
                await self.update_status("Embedded voice system ready", 80)
                progress_dialog.update("Embedded voice system ready", 80)
            else:
                self.update_status_indicator("voice", "error")
                await self.update_status("Embedded voice system unavailable", 80)
                progress_dialog.update("Embedded voice system unavailable", 80)

            # Initialize smart router
            if self.simple_llm and self.voice_pipeline:
                self.smart_router = SmartRouter(
                    llm_backend=self.simple_llm,
                    voice_pipeline=self.voice_pipeline,
                    multimodal_processor=self.multimodal_processor,
                    conversation_memory=self.conversation_memory,
                )
                await self.update_status("Smart router initialized", 85)

            # Initialize agent system
            await self.update_status("Initializing agent system...", 90)
            try:
                # Create agent registry
                self.agent_registry = AgentRegistry()

                # Create agent builder with Simple LLM backend
                if self.simple_llm:
                    self.agent_builder = AgentBuilder(self.simple_llm)

                # Create simple agent runtime
                self.agent_runtime = SimpleAgentRuntime(
                    llm_backend=self.simple_llm,
                    memory_manager=getattr(self.data_storage, 'memory_manager', None)
                )

                # Create agent panel for UI
                self.agent_panel = AgentPanel(
                    agent_builder=self.agent_builder,
                    agent_runtime=self.agent_runtime,
                    agent_registry=self.agent_registry,
                )

                await self.update_status("Agent system ready", 95)
                logger.info("Agent system initialized successfully")

            except Exception as e:
                logger.warning(f"Agent system initialization failed: {e}")
                # Non-critical failure - continue without agent system

            # Final initialization
            tracker.update_step("finalize", 50, "Finalizing setup...")
            self.is_initialized = True
            await self.update_status("Tektra AI Assistant ready!", 100)
            tracker.complete_step("finalize")
            
            if self.progress_overlay:
                self.progress_overlay.update_progress(100, "Tektra AI Assistant ready!")
                # Hide progress overlay after brief pause
                await asyncio.sleep(0.5)  # Brief pause to show completion
                self.progress_overlay.hide()

            # Enable controls
            self.enable_controls()

            logger.success("Backend initialization complete!")

        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            await self.update_status(f"Initialization failed: {e}", 0)
            
            # Hide progress overlay on error
            if self.progress_overlay:
                self.progress_overlay.hide()
            
            self.show_error(
                "Initialization Error", f"Failed to initialize backend systems:\n\n{e}"
            )

    async def update_status(self, status: str, progress: float):
        """Update the application status and progress."""
        self.initialization_status = status
        self.initialization_progress = progress

        # Update UI elements
        self.status_label.text = status
        self.progress_label.text = f"{progress:.0f}%"

        # Update model info
        if self.simple_llm:
            model_info = self.simple_llm.get_model_info()
            model_text = f"LLM: {model_info['loading_status'][:30]}"
            self.model_info_label.text = model_text
        elif self.unmute:
            model_info = self.unmute.get_model_info()
            memory_usage = model_info.get('memory_usage', {}).get('total', 0)
            model_text = f"Unmute: {memory_usage:.0f}MB loaded"
            self.model_info_label.text = model_text

        logger.info(f"Status: {status} ({progress:.0f}%)")
        
    async def initialize_embedded_voice_system(self, progress_dialog=None) -> bool:
        """Initialize the embedded voice system with model downloads."""
        try:
            # Check which models need downloading
            await self.update_status("Checking for AI models...", 25)
            if progress_dialog:
                progress_dialog.update("Checking for AI models...", 25, "Scanning model directory...")
            
            required_models = ["unmute_stt", "unmute_llm", "unmute_tts"]
            total_size = self.model_manager.get_total_model_size(required_models)
            
            missing_models = self.model_manager.get_missing_models(required_models)
            
            if missing_models:
                # Show download dialog
                download_size = self.model_manager.get_total_model_size(missing_models)
                
                should_download = await self.main_window.confirm_dialog(
                    "Model Download Required",
                    f"Tektra needs to download {download_size}MB of AI models "
                    f"for voice conversation. This is a one-time download.\n\n"
                    f"Download now?"
                )
                
                if not should_download:
                    await self.update_status("Running without voice features", 25)
                    return False
                    
            # Download missing models
            await self.update_status("Downloading AI models...", 30)
            
            success = await self.model_manager.ensure_models_available(
                models=required_models,
                progress_callback=self.on_model_download_progress
            )
            
            if not success:
                await self.update_status("Model download failed", 30)
                return False
                
            # Initialize embedded Unmute (if available)
            if UNMUTE_AVAILABLE and EmbeddedUnmute:
                await self.update_status("Loading AI models...", 50)
                
                device = "cuda" if self.config.get("use_cuda", True) else "cpu"
                self.unmute = EmbeddedUnmute(
                    model_dir=self.model_manager.model_dir,
                    device=device,
                    memory_limit_gb=self.config.get("max_memory_gb", 4.0)
                )
                
                model_loaded = await self.unmute.initialize_models(
                    progress_callback=self.on_model_load_progress
                )
                
                if not model_loaded:
                    await self.update_status("Failed to load models", 50)
                    return False
            else:
                await self.update_status("Voice features disabled (dependencies unavailable)", 50)
                self.unmute = None
                
            # Initialize voice pipeline (if available)
            if VOICE_PIPELINE_AVAILABLE and EmbeddedVoicePipeline and self.unmute:
                await self.update_status("Initializing voice pipeline...", 65)
                
                self.voice_pipeline = EmbeddedVoicePipeline(
                    unmute=self.unmute,
                    on_transcription=self.on_voice_transcription,
                    on_response=self.on_voice_response,
                    on_audio_response=self.on_audio_response,
                    on_status_change=self.on_voice_status_change
                )
                
                voice_ready = await self.voice_pipeline.initialize()
            else:
                await self.update_status("Voice pipeline disabled", 65)
                self.voice_pipeline = None
                voice_ready = False
            
            if voice_ready and self.unmute:
                # Log memory usage
                memory_info = self.unmute.get_memory_usage()
                logger.info(f"Model memory usage: {memory_info}")
                return True
            else:
                await self.update_status("Voice pipeline initialization failed", 65)
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize embedded voice system: {e}")
            await self.update_status(f"Voice initialization failed: {e}", 25)
            return False
            
    async def on_model_download_progress(self, status: str, progress: float):
        """Handle model download progress."""
        # Map to overall progress (30-50%)
        overall_progress = 30 + (progress * 20)
        await self.update_status(status, overall_progress)
        
    async def on_model_load_progress(self, status: str, progress: float):
        """Handle model loading progress."""
        # Map to overall progress (50-65%)
        overall_progress = 50 + (progress * 15)
        await self.update_status(status, overall_progress)

    async def on_model_progress(self, progress: float, status: str):
        """Handle model loading progress updates."""
        # Map model progress to overall progress (30-60%)
        overall_progress = 30 + (progress * 0.3)
        await self.update_status(f"Loading model: {status}", overall_progress)

    def enable_controls(self):
        """Enable user interface controls after initialization."""
        # Enable main controls
        self.upload_file_btn.enabled = True

        # Enable chat panel features
        if self.chat_panel:
            self.chat_panel.enable_file_features(True)
            
            # Trigger file upload feature discovery
            if self.discovery_manager:
                asyncio.create_task(
                    self.discovery_manager.trigger_discovery(DiscoveryTrigger.FILE_UPLOAD)
                )

            if self.voice_pipeline and self.voice_pipeline.is_initialized:
                self.chat_panel.enable_voice_features(True)
                
                # Trigger voice feature discovery
                if self.discovery_manager:
                    asyncio.create_task(
                        self.discovery_manager.trigger_discovery(DiscoveryTrigger.VOICE_AVAILABLE)
                    )

        # Enable voice controls
        if self.voice_pipeline and self.voice_pipeline.is_initialized:
            self.voice_toggle_btn.enabled = True
            self.push_to_talk_btn.enabled = True

        if self.simple_llm and self.simple_llm.is_initialized:
            self.camera_toggle_btn.enabled = True
            
            # Trigger multimodal feature discovery
            if self.discovery_manager:
                asyncio.create_task(
                    self.discovery_manager.trigger_discovery(DiscoveryTrigger.MULTIMODAL_AVAILABLE)
                )
        
        # Trigger memory integration discovery
        if self.conversation_memory and self.discovery_manager:
            asyncio.create_task(
                self.discovery_manager.trigger_discovery(DiscoveryTrigger.MEMORY_INTEGRATION)
            )

    def update_conversation_stats(self):
        """Update conversation statistics display."""
        message_count = 0
        if self.chat_panel:
            message_count = self.chat_panel.get_message_count()

        self.message_count_label.text = f"Messages: {message_count}"

        if self.smart_router:
            stats = self.smart_router.get_router_stats()
            unmute_pct = stats.get("unmute_percentage", 0)
            llm_pct = stats.get("llm_percentage", 0)
            self.routing_stats_label.text = (
                f"Voice: {unmute_pct:.0f}% | LLM: {llm_pct:.0f}%"
            )

    # Event handlers
    async def handle_chat_message(self, message: str):
        """Handle message from chat panel."""
        if not self.is_initialized or not self.smart_router:
            return

        try:
            # Trigger first message discovery
            if self.discovery_manager:
                asyncio.create_task(
                    self.discovery_manager.trigger_discovery(DiscoveryTrigger.FIRST_MESSAGE)
                )
                
                # Check for agent creation keywords
                agent_keywords = ["create agent", "make agent", "build agent", "new agent"]
                if any(keyword in message.lower() for keyword in agent_keywords):
                    asyncio.create_task(
                        self.discovery_manager.trigger_discovery(DiscoveryTrigger.AGENT_CREATION)
                    )

            # Update conversation stats
            self.update_conversation_stats()

            # Check if this is an agent-related query
            if await self.handle_agent_query(message):
                # Handled as agent query, no need to process further
                return

            # Process through chat manager
            await self.chat_manager.process_user_message(
                message, self.smart_router, context={"session_id": "main_session"}
            )

        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            if self.chat_panel:
                self.chat_panel.add_message(
                    "assistant", f"Sorry, I encountered an error: {e}"
                )
        finally:
            # Update stats after processing
            self.update_conversation_stats()

    def on_input_change(self, widget):
        """Handle input text changes (legacy - now handled by ChatPanel)."""
        logger.debug("Legacy input change handler called - functionality moved to ChatPanel")

    async def toggle_voice_mode(self, widget=None):
        """Toggle voice conversation mode."""
        if not self.voice_pipeline:
            return

        if not self.is_voice_active:
            # Start voice mode
            success = await self.voice_pipeline.start_listening()
            if success:
                self.is_voice_active = True
                self.voice_toggle_btn.text = "Stop Voice Mode"
                self.voice_status_indicator.text = "🟡 Voice (Listening)"

                # Update chat panel
                if self.chat_panel:
                    self.chat_panel.update_voice_status(True)
        else:
            # Stop voice mode
            success = await self.voice_pipeline.stop_listening()
            if success:
                self.is_voice_active = False
                self.voice_toggle_btn.text = "Start Voice Mode"
                self.update_status_indicator("voice", "ready")

                # Update chat panel
                if self.chat_panel:
                    self.chat_panel.update_voice_status(False)

    async def toggle_camera(self, widget):
        """Toggle camera for vision tasks."""
        try:
            if not hasattr(self, 'camera_manager'):
                # Initialize camera manager on first use
                from .vision import CameraManager
                self.camera_manager = CameraManager(
                    on_error=lambda msg: self.chat_panel.add_message("assistant", f"Camera: {msg}") if self.chat_panel else None
                )
                await self.camera_manager.initialize()
            
            if not self.is_camera_active:
                # Start camera
                success = await self.camera_manager.start_capture()
                if success:
                    self.is_camera_active = True
                    self.camera_toggle_btn.text = "Disable Camera"
                    if hasattr(self, 'camera_status_indicator'):
                        self.camera_status_indicator.text = "📹 Camera Active"
                    
                    if self.chat_panel:
                        self.chat_panel.add_message("system", "📹 Camera activated (Note: Live camera feed coming in future update)")
                    logger.info("Camera activated")
                else:
                    if self.chat_panel:
                        self.chat_panel.add_message("assistant", "Failed to start camera. Using file upload for vision analysis.")
            else:
                # Stop camera
                success = await self.camera_manager.stop_capture()
                if success:
                    self.is_camera_active = False
                    self.camera_toggle_btn.text = "Enable Camera"
                    if hasattr(self, 'camera_status_indicator'):
                        self.camera_status_indicator.text = "📷 Camera"
                    
                    if self.chat_panel:
                        self.chat_panel.add_message("system", "📷 Camera deactivated")
                    logger.info("Camera deactivated")
                    
        except Exception as e:
            logger.error(f"Error toggling camera: {e}")
            if self.chat_panel:
                self.chat_panel.add_message("assistant", f"Camera error: {e}. Please use file upload for vision analysis.")
            
            # Reset state on error
            self.is_camera_active = False
            self.camera_toggle_btn.text = "Enable Camera"

    async def start_voice_input(self, widget):
        """Start voice input (push-to-talk)."""
        if not self.voice_pipeline or not self.voice_pipeline.is_initialized:
            if self.chat_panel:
                self.chat_panel.add_message("assistant", "Voice pipeline not available. Please check voice settings.")
            return

        # Start recording for push-to-talk
        if not hasattr(self, 'is_push_to_talk_active'):
            self.is_push_to_talk_active = False

        if not self.is_push_to_talk_active:
            # Start push-to-talk recording
            success = await self.voice_pipeline.start_listening()
            if success:
                self.is_push_to_talk_active = True
                # Update UI to show recording state
                if hasattr(self, 'voice_status_indicator'):
                    self.voice_status_indicator.text = "🔴 Recording (Release to send)"
                if self.chat_panel:
                    self.chat_panel.add_message("system", "🎤 Recording... (Release button to send)")
                logger.info("Push-to-talk recording started")
            else:
                if self.chat_panel:
                    self.chat_panel.add_message("assistant", "Failed to start voice recording. Please try again.")
        else:
            # Stop push-to-talk recording and process
            success = await self.voice_pipeline.stop_listening()
            if success:
                self.is_push_to_talk_active = False
                # Update UI to show processing state
                if hasattr(self, 'voice_status_indicator'):
                    self.voice_status_indicator.text = "🟡 Processing voice input..."
                if self.chat_panel:
                    self.chat_panel.add_message("system", "🔄 Processing audio...")
                logger.info("Push-to-talk recording stopped, processing audio")
            else:
                if self.chat_panel:
                    self.chat_panel.add_message("assistant", "Failed to stop voice recording. Please try again.")
                self.is_push_to_talk_active = False

    async def upload_file(self, widget):
        """Upload a file for analysis."""
        if not self.is_initialized:
            return

        try:
            # Open file dialog
            file_path = await self.main_window.open_file_dialog(
                title="Select file for analysis",
                file_types=["txt", "md", "json", "pdf", "docx", "png", "jpg", "jpeg"],
            )

            if file_path:
                await self.process_uploaded_file(file_path)

        except Exception as e:
            logger.error(f"File upload error: {e}")
            self.show_error("Upload Error", f"Failed to upload file: {e}")

    async def process_uploaded_file(self, file_path: str):
        """Process an uploaded file."""
        try:
            self.status_label.text = "Processing file..."

            # Process file with multimodal processor
            result = await self.multimodal_processor.process_file(file_path)

            filename = Path(file_path).name

            if result["processing_status"] == "success":
                # Use chat manager to handle file processing
                if self.chat_manager:
                    await self.chat_manager.handle_file_processed(filename, "success")

                # Ask user what they want to know
                if self.chat_panel:
                    self.chat_panel.add_message(
                        "assistant",
                        f"I've processed the file '{filename}'. What would you like to know about it?",
                    )
            else:
                error = result.get("error", "Unknown error")
                if self.chat_manager:
                    await self.chat_manager.handle_file_processed(
                        filename, f"error: {error}"
                    )

                if self.chat_panel:
                    self.chat_panel.add_message(
                        "assistant", f"Sorry, I couldn't process that file: {error}"
                    )

            self.status_label.text = "Ready"

            # Update conversation stats
            self.update_conversation_stats()

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            if self.chat_panel:
                self.chat_panel.add_message("assistant", f"Error processing file: {e}")

    async def show_settings(self, widget):
        """Show application settings."""
        try:
            # Create settings dialog with key configuration options
            settings_text = f"""Current Configuration:

🤖 AI Model Settings:
• Model: {self.config.get('qwen_model_name', 'Not set')}
• Memory Limit: {self.config.get('max_memory_gb', 8.0)} GB
• Quantization: {self.config.get('qwen_quantization', 8)} bit

🎤 Voice Settings:
• Voice Enabled: {self.config.get('voice_enabled', True)}
• Auto-start Voice: {self.config.get('auto_start_voice', False)}
• Sample Rate: {self.config.get('audio_sample_rate', 16000)} Hz

🖥️ Interface Settings:
• Window Size: {self.config.get('window_width', 1200)}x{self.config.get('window_height', 800)}
• Show Performance: {self.config.get('show_performance_stats', True)}
• Debug Mode: {self.config.get('debug_mode', False)}

📁 File Processing:
• Max File Size: {self.config.get('max_file_size_mb', 100)} MB
• Supported Images: {len(self.config.get('supported_image_formats', []))} formats
• Supported Docs: {len(self.config.get('supported_document_formats', []))} formats

⚙️ Smart Router:
• Confidence Threshold: {self.config.get('confidence_threshold', 0.6)}
• Voice Bias: {self.config.get('voice_bias', 0.1)}
• Hybrid Routing: {self.config.get('enable_hybrid_routing', True)}

Configuration file: {self.config.config_file}
Total settings: {len(self.config.config_data)}

Note: To modify settings, edit the configuration file or restart the application 
with environment variables. Advanced settings UI coming in future release."""

            # Show configuration in dialog
            self.main_window.show_info_dialog(
                title="Tektra Settings", 
                message=settings_text
            )
            
            logger.info("Settings dialog displayed")
            
        except Exception as e:
            logger.error(f"Error showing settings dialog: {e}")
            self.main_window.show_error_dialog(
                title="Settings Error",
                message=f"Failed to load settings: {e}"
            )

    async def show_about(self, widget):
        """Show about dialog."""
        about_text = f"""Tektra AI Assistant v{self.app.version}

A voice-interactive AI assistant with multimodal capabilities.

Architecture:
• Kyutai Unmute for voice conversations
• Qwen 2.5-VL for complex reasoning and vision
• Smart routing for optimal AI selection
• Native desktop experience with Briefcase

Platform: {platform.system()} {platform.release()}
Python: {platform.python_version()}

Built with ❤️ using Python and Briefcase"""

        self.main_window.info_dialog("About Tektra AI Assistant", about_text)

    def show_error(self, title: str, message: str):
        """Show an error dialog."""
        self.main_window.error_dialog(title, message)

    # Voice pipeline callbacks
    async def on_voice_transcription(self, text: str):
        """Handle voice transcription from Unmute STT."""
        if self.chat_manager:
            await self.chat_manager.handle_voice_transcription(text)

        # Update conversation stats
        self.update_conversation_stats()

    async def on_voice_response(self, text: str):
        """Handle voice response from Unmute LLM."""
        if self.chat_manager:
            await self.chat_manager.handle_voice_response(text)

        # Update conversation stats
        self.update_conversation_stats()

    async def on_audio_response(self, audio_data: bytes):
        """Handle audio response from Unmute TTS."""
        # Audio is played automatically by Unmute
        logger.debug(f"Received audio response: {len(audio_data)} bytes")

    async def on_voice_status_change(self, status: str):
        """Handle voice status changes."""
        logger.debug(f"Voice status: {status}")

        if "Listening" in status:
            self.update_status_indicator("voice", "working")
            self.voice_status_text.text = "Voice (Listening)"
        elif "Speaking" in status:
            self.update_status_indicator("voice", "working")
            self.voice_status_text.text = "Voice (Speaking)"
        elif "Ready" in status:
            self.update_status_indicator("voice", "ready")
            self.voice_status_text.text = "Voice"
        elif "Error" in status:
            self.update_status_indicator("voice", "error")

    # Application lifecycle
    def on_exit(self):
        """Handle application exit."""
        logger.info("Tektra AI Assistant shutting down...")

        # Cleanup backend systems
        if self.voice_pipeline:
            asyncio.create_task(self.voice_pipeline.cleanup())

        if self.simple_llm:
            asyncio.create_task(self.simple_llm.cleanup())
            
        if self.unmute:
            asyncio.create_task(self.unmute.cleanup())

        logger.info("Tektra AI Assistant shutdown complete")

    async def show_agent_creator(self, widget):
        """Show the agent creation interface."""
        if not self.agent_panel:
            self.show_info("Agent System", "Agent system is not initialized yet.")
            return

        # Create a new window for agent creation
        agent_window = toga.Window(title="Create AI Agent", size=(800, 600))
        agent_window.content = self.agent_panel.container
        self.agent_panel._show_tab("create")
        agent_window.show()

    async def show_agent_dashboard(self, widget):
        """Show the agent management dashboard."""
        if not self.agent_panel:
            self.show_info("Agent System", "Agent system is not initialized yet.")
            return

        # Create a new window for agent dashboard
        dashboard_window = toga.Window(title="Agent Dashboard", size=(1000, 700))
        dashboard_window.content = self.agent_panel.container
        self.agent_panel._show_tab("dashboard")
        dashboard_window.show()

    async def handle_agent_query(self, message: str) -> bool:
        """
        Check if the message is agent-related and handle it.

        Returns:
            bool: True if handled as agent query, False otherwise
        """
        if not self.agent_builder:
            return False

        # Check for agent creation patterns
        agent_patterns = [
            "create an agent",
            "build an agent",
            "make an agent",
            "i need an agent",
            "deploy an agent",
        ]

        message_lower = message.lower()

        for pattern in agent_patterns:
            if pattern in message_lower:
                # Extract the agent description
                description = message
                for p in agent_patterns:
                    description = description.lower().replace(p, "").strip()

                # Show agent creation in chat
                self.chat_panel.add_message(
                    "assistant",
                    "I'll help you create an agent. Let me analyze your requirements...",
                )

                try:
                    # Create agent from description
                    spec = await self.agent_builder.create_agent_from_description(
                        description
                    )

                    # Register and deploy
                    await self.agent_registry.register_agent(spec)
                    agent_id = await self.agent_runtime.deploy_agent(spec)

                    # Show success
                    response = f"""✅ Agent Created Successfully!

**Name:** {spec.name}
**Type:** {spec.type.value}
**Description:** {spec.description}

The agent is now ready to execute tasks. You can ask it to do things like:
- "Execute code to calculate something"
- "Analyze data"
- "Create a simple script"

Agent ID: {agent_id}"""

                    self.chat_panel.add_message("assistant", response)

                except Exception as e:
                    logger.error(f"Error creating agent: {e}")
                    self.chat_panel.add_message(
                        "assistant", f"I encountered an error creating the agent: {e}"
                    )

                return True

        # Check for agent management queries
        if any(
            phrase in message_lower
            for phrase in ["list agents", "show agents", "my agents"]
        ):
            if self.agent_runtime:
                agents = self.agent_runtime.list_agents()
                if agents:
                    response = "Here are your agents:\n\n"
                    for agent in agents[:5]:  # Show first 5
                        response += (
                            f"• **{agent['name']}** - {agent['state']}\n"
                            f"  Description: {agent['description'][:50]}...\n"
                            f"  Created: {agent['created_at'][:16]}\n\n"
                        )
                    response += f"Total: {len(agents)} agents"
                else:
                    response = "You don't have any agents yet. Try creating one by saying 'Create an agent that...'"

                self.chat_panel.add_message("assistant", response)
                return True

        return False

    async def cleanup(self):
        """Clean up resources when the app is closing."""
        logger.info("Cleaning up Tektra AI Assistant...")
        
        try:
            # Stop model updater
            if self.model_updater:
                await self.model_updater.stop()
                logger.info("Model updater stopped")
            
            # Clean up voice pipeline
            if self.voice_pipeline:
                await self.voice_pipeline.cleanup()
                logger.info("Voice pipeline cleaned up")
            
            # Clean up Simple LLM
            if self.simple_llm:
                await self.simple_llm.cleanup()
                logger.info("Simple LLM cleaned up")
            
            # Clean up smart router
            if self.smart_router:
                await self.smart_router.cleanup()
                logger.info("Smart router cleaned up")
            
            logger.info("Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for the application."""
    return TektraApp()


if __name__ == "__main__":
    app = main()
    app.main_loop()
