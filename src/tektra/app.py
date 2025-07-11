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

from .agents import AgentBuilder, AgentRegistry, AgentRuntime, SandboxType
from .ai.multimodal import MultimodalProcessor

# Import Tektra components
from .ai.qwen_backend import QwenBackend, QwenModelConfig
from .ai.smart_router import SmartRouter
from .data.storage import DataStorage
from .gui.agent_panel import AgentPanel
from .gui.chat_panel import ChatManager, ChatPanel
from .utils.config import AppConfig
from .voice import VoiceConversationPipeline


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

        # Backend components (initialized later)
        self.qwen_backend = None
        self.voice_pipeline = None
        self.smart_router = None
        self.multimodal_processor = None

        # Agent system components
        self.agent_builder = None
        self.agent_runtime = None
        self.agent_registry = None
        self.agent_panel = None

        # GUI components
        self.chat_panel = None
        self.chat_manager = None

        # Application state
        self.is_initialized = False
        self.is_voice_active = False
        self.is_camera_active = False
        self.initialization_progress = 0.0
        self.initialization_status = "Starting up..."

        # Create main window
        self.main_window = toga.MainWindow(title="Tektra AI Assistant")

        # Build the main interface
        self.build_interface()

        # Show the window
        self.main_window.show()

        # Start background initialization
        asyncio.create_task(self.initialize_backend_systems())

        logger.info("Tektra app startup complete")

    def build_interface(self):
        """Build the main user interface."""
        # Create main container
        main_container = toga.Box(style=Pack(direction=COLUMN))

        # Header with title and status
        header = self.build_header()
        main_container.add(header)

        # Main content area
        content_area = self.build_content_area()
        main_container.add(content_area)

        # Status bar
        status_bar = self.build_status_bar()
        main_container.add(status_bar)

        # Set as main window content
        self.main_window.content = main_container

        # Create app menu
        self.create_menu()

    def build_header(self) -> toga.Box:
        """Build the application header."""
        header = toga.Box(
            style=Pack(
                direction=ROW,
                align_items="center",
                margin=(10, 15),
                background_color="#2c3e50",
            )
        )

        # App title and logo
        title_box = toga.Box(style=Pack(direction=COLUMN, flex=1))

        self.app_title = toga.Label(
            "Tektra AI Assistant",
            style=Pack(
                font_size=18, font_weight="bold", color="#ecf0f1", margin_bottom=2
            ),
        )

        self.app_subtitle = toga.Label(
            "Voice-Interactive AI with Unmute + Qwen",
            style=Pack(font_size=12, color="#95a5a6", font_style="italic"),
        )

        title_box.add(self.app_title)
        title_box.add(self.app_subtitle)
        header.add(title_box)

        # Status indicators
        status_box = toga.Box(style=Pack(direction=ROW, align_items="center"))

        self.model_status_indicator = toga.Label(
            "⚪ Models", style=Pack(margin_right=10, color="#95a5a6", font_size=12)
        )

        self.voice_status_indicator = toga.Label(
            "⚪ Voice", style=Pack(margin_right=10, color="#95a5a6", font_size=12)
        )

        self.connection_status_indicator = toga.Label(
            "⚪ Services", style=Pack(color="#95a5a6", font_size=12)
        )

        status_box.add(self.model_status_indicator)
        status_box.add(self.voice_status_indicator)
        status_box.add(self.connection_status_indicator)

        header.add(status_box)

        return header

    def build_content_area(self) -> toga.Box:
        """Build the main content area."""
        content = toga.Box(style=Pack(direction=ROW, flex=1))

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
        sidebar = toga.Box(
            style=Pack(
                direction=COLUMN, width=250, margin=10, background_color="#34495e"
            )
        )

        # Models section
        models_label = toga.Label(
            "AI Models",
            style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5),
        )
        sidebar.add(models_label)

        self.model_info_label = toga.Label(
            "Initializing...",
            style=Pack(color="#95a5a6", font_size=11, margin_bottom=15),
        )
        sidebar.add(self.model_info_label)

        # Voice controls section
        voice_label = toga.Label(
            "Voice Controls",
            style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5),
        )
        sidebar.add(voice_label)

        self.voice_toggle_btn = toga.Button(
            "Start Voice Mode",
            on_press=self.toggle_voice_mode,
            style=Pack(width=200, margin_bottom=5),
            enabled=False,
        )
        sidebar.add(self.voice_toggle_btn)

        self.push_to_talk_btn = toga.Button(
            "Push to Talk", style=Pack(width=200, margin_bottom=15), enabled=False
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

        return self.chat_panel.widget

    def build_right_sidebar(self) -> toga.Box:
        """Build the right sidebar with analytics and status."""
        sidebar = toga.Box(
            style=Pack(
                direction=COLUMN, width=200, margin=10, background_color="#2c3e50"
            )
        )

        # Conversation stats
        stats_label = toga.Label(
            "Session Stats",
            style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5),
        )
        sidebar.add(stats_label)

        self.message_count_label = toga.Label(
            "Messages: 0", style=Pack(color="#95a5a6", font_size=11, margin_bottom=5)
        )
        sidebar.add(self.message_count_label)

        self.routing_stats_label = toga.Label(
            "Routing: -", style=Pack(color="#95a5a6", font_size=11, margin_bottom=15)
        )
        sidebar.add(self.routing_stats_label)

        # System performance
        performance_label = toga.Label(
            "Performance",
            style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5),
        )
        sidebar.add(performance_label)

        self.memory_usage_label = toga.Label(
            "Memory: -", style=Pack(color="#95a5a6", font_size=11, margin_bottom=5)
        )
        sidebar.add(self.memory_usage_label)

        self.response_time_label = toga.Label(
            "Response: -", style=Pack(color="#95a5a6", font_size=11, margin_bottom=15)
        )
        sidebar.add(self.response_time_label)

        return sidebar

    def build_status_bar(self) -> toga.Box:
        """Build the bottom status bar."""
        status_bar = toga.Box(
            style=Pack(
                direction=ROW,
                margin=(5, 15),
                background_color="#34495e",
                align_items="center",
            )
        )

        # Current status
        self.status_label = toga.Label(
            "Initializing Tektra AI Assistant...",
            style=Pack(flex=1, color="#ecf0f1", font_size=12),
        )
        status_bar.add(self.status_label)

        # Progress indicator
        self.progress_label = toga.Label(
            "0%", style=Pack(color="#95a5a6", font_size=12)
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

            # Update status
            await self.update_status("Initializing AI models...", 10)

            # Find unmute path (parent directory)
            unmute_path = Path(__file__).parent.parent.parent.parent / "unmute"
            logger.info(f"Unmute path: {unmute_path}")

            # Initialize multimodal processor
            self.multimodal_processor = MultimodalProcessor()
            await self.update_status("Multimodal processor ready", 20)

            # Initialize Qwen backend
            await self.update_status("Loading Qwen AI model...", 30)
            qwen_config = QwenModelConfig(
                model_name=self.config.get(
                    "qwen_model_name", "Qwen/Qwen2.5-VL-7B-Instruct"
                ),
                quantization_bits=self.config.get("qwen_quantization", 8),
                max_memory_gb=self.config.get("max_memory_gb", 8.0),
            )

            self.qwen_backend = QwenBackend(qwen_config)
            qwen_success = await self.qwen_backend.initialize(self.on_model_progress)

            if qwen_success:
                self.model_status_indicator.text = "🟢 Models"
                await self.update_status("Qwen model loaded successfully", 60)
            else:
                self.model_status_indicator.text = "🔴 Models"
                await self.update_status("Failed to load Qwen model", 60)

            # Initialize voice pipeline
            await self.update_status("Initializing voice services...", 70)
            service_config = self.config.get_service_config()
            self.voice_pipeline = VoiceConversationPipeline(
                unmute_path=unmute_path,
                on_transcription=self.on_voice_transcription,
                on_response=self.on_voice_response,
                on_audio_response=self.on_audio_response,
                on_status_change=self.on_voice_status_change,
                service_config=service_config,
            )

            voice_success = await self.voice_pipeline.initialize()

            if voice_success:
                self.voice_status_indicator.text = "🟢 Voice"
                await self.update_status("Voice services ready", 80)
            else:
                self.voice_status_indicator.text = "🔴 Voice"
                await self.update_status("Voice services unavailable", 80)

            # Initialize smart router
            if self.qwen_backend and self.voice_pipeline:
                self.smart_router = SmartRouter(
                    qwen_backend=self.qwen_backend,
                    voice_pipeline=self.voice_pipeline,
                    multimodal_processor=self.multimodal_processor,
                )
                await self.update_status("Smart router initialized", 85)

            # Initialize agent system
            await self.update_status("Initializing agent system...", 90)
            try:
                # Create agent registry
                self.agent_registry = AgentRegistry()

                # Create agent builder with Qwen backend
                if self.qwen_backend:
                    self.agent_builder = AgentBuilder(self.qwen_backend)

                # Create agent runtime with appropriate sandbox
                sandbox_type = SandboxType.PROCESS  # Default to process isolation
                if self.config.get("use_docker_sandbox", False):
                    sandbox_type = SandboxType.DOCKER

                self.agent_runtime = AgentRuntime(
                    sandbox_type=sandbox_type,
                    memory_manager=self.data_storage.memory_manager,
                    qwen_backend=self.qwen_backend
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
            self.is_initialized = True
            await self.update_status("Tektra AI Assistant ready!", 100)

            # Enable controls
            self.enable_controls()

            logger.success("Backend initialization complete!")

        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            await self.update_status(f"Initialization failed: {e}", 0)
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
        if self.qwen_backend:
            model_info = self.qwen_backend.get_model_info()
            model_text = f"Qwen: {model_info['loading_status'][:30]}"
            self.model_info_label.text = model_text

        logger.info(f"Status: {status} ({progress:.0f}%)")

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

            if self.voice_pipeline and self.voice_pipeline.is_initialized:
                self.chat_panel.enable_voice_features(True)

        # Enable voice controls
        if self.voice_pipeline and self.voice_pipeline.is_initialized:
            self.voice_toggle_btn.enabled = True
            self.push_to_talk_btn.enabled = True

        if self.qwen_backend and self.qwen_backend.is_initialized:
            self.camera_toggle_btn.enabled = True

    def update_conversation_stats(self):
        """Update conversation statistics display."""
        message_count = 0
        if self.chat_panel:
            message_count = self.chat_panel.get_message_count()

        self.message_count_label.text = f"Messages: {message_count}"

        if self.smart_router:
            stats = self.smart_router.get_router_stats()
            unmute_pct = stats.get("unmute_percentage", 0)
            qwen_pct = stats.get("qwen_percentage", 0)
            self.routing_stats_label.text = (
                f"Unmute: {unmute_pct:.0f}% | Qwen: {qwen_pct:.0f}%"
            )

    # Event handlers
    async def handle_chat_message(self, message: str):
        """Handle message from chat panel."""
        if not self.is_initialized or not self.smart_router:
            return

        try:
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
                self.voice_status_indicator.text = "🟢 Voice"

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
            self.voice_status_indicator.text = "🟡 Voice (Listening)"
        elif "Speaking" in status:
            self.voice_status_indicator.text = "🟡 Voice (Speaking)"
        elif "Ready" in status:
            self.voice_status_indicator.text = "🟢 Voice"
        elif "Error" in status:
            self.voice_status_indicator.text = "🔴 Voice"

    # Application lifecycle
    def on_exit(self):
        """Handle application exit."""
        logger.info("Tektra AI Assistant shutting down...")

        # Cleanup backend systems
        if self.voice_pipeline:
            asyncio.create_task(self.voice_pipeline.cleanup())

        if self.qwen_backend:
            asyncio.create_task(self.qwen_backend.cleanup())

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
                    await self.agent_runtime.deploy_agent(spec)

                    # Show success
                    response = f"""✅ Agent Created Successfully!

**Name:** {spec.name}
**Type:** {spec.type.value}
**Goal:** {spec.goal}

The agent is now running and will {spec.description.lower()}.

You can manage your agents by clicking 'Manage Agents' in the sidebar."""

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
            if self.agent_registry:
                agents = await self.agent_registry.list_agents()
                if agents:
                    response = "Here are your active agents:\n\n"
                    for agent in agents[:5]:  # Show first 5
                        response += (
                            f"• **{agent.specification.name}** - {agent.status.value}\n"
                        )
                    response += f"\nTotal: {len(agents)} agents"
                else:
                    response = "You don't have any agents yet. Try creating one by saying 'Create an agent that...'"

                self.chat_panel.add_message("assistant", response)
                return True

        return False


def main():
    """Main entry point for the application."""
    return TektraApp()


if __name__ == "__main__":
    app = main()
    app.main_loop()
