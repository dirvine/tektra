#!/usr/bin/env python3
"""
Tektra AI Assistant - Demo Version

A simplified demonstration of the Tektra application architecture
without heavy AI/ML dependencies for testing the GUI.
"""

import asyncio
import sys
import platform
from pathlib import Path

# UV script dependencies for standalone execution
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "toga>=0.4.0",
# ]
# ///

try:
    import toga
    from toga.style import Pack
    from toga.style.pack import COLUMN, ROW
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run with: uv run demo.py")
    sys.exit(1)

# Simple logging replacement
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def success(self, msg): print(f"SUCCESS: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

logger = SimpleLogger()


class MockBackend:
    """Mock AI backend for demonstration."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Mock initialization."""
        logger.info(f"Initializing {self.name}...")
        await asyncio.sleep(1)  # Simulate loading time
        self.is_initialized = True
        logger.success(f"{self.name} initialized successfully!")
        return True
    
    async def process_query(self, query: str) -> str:
        """Mock query processing."""
        await asyncio.sleep(0.5)  # Simulate processing time
        return f"[{self.name}] Mock response to: '{query}'"


class MockSmartRouter:
    """Mock smart router for demonstration."""
    
    def __init__(self, qwen_backend, unmute_backend):
        self.qwen_backend = qwen_backend
        self.unmute_backend = unmute_backend
        self.query_count = 0
    
    async def route_query(self, query: str) -> dict:
        """Mock query routing."""
        self.query_count += 1
        
        # Simple routing logic for demo
        if "analyze" in query.lower() or "complex" in query.lower():
            backend = self.qwen_backend
            route = "analytical"
        else:
            backend = self.unmute_backend
            route = "conversational"
        
        response = await backend.process_query(query)
        
        return {
            "routing_decision": {"route": route, "confidence": 0.85},
            "response": response,
            "processing_time": 0.5
        }
    
    def get_router_stats(self) -> dict:
        """Get routing statistics."""
        return {
            "total_queries": self.query_count,
            "unmute_percentage": 60,
            "qwen_percentage": 40
        }


class TektraDemoApp(toga.App):
    """Tektra AI Assistant Demo Application."""
    
    def startup(self):
        """Initialize the demo application."""
        logger.info("Starting Tektra AI Assistant Demo...")
        
        # Initialize mock backends
        self.qwen_backend = MockBackend("Qwen Analytical")
        self.unmute_backend = MockBackend("Unmute Voice")
        self.smart_router = None
        
        # Application state
        self.is_initialized = False
        self.messages = []
        
        # Create main window
        self.main_window = toga.MainWindow(title="Tektra AI Assistant - Demo")
        
        # Build interface
        self.build_interface()
        
        # Show window
        self.main_window.show()
        
        # Start async initialization
        asyncio.create_task(self.initialize_backends())
        
        logger.info("Demo app startup complete")
    
    def build_interface(self):
        """Build the main user interface."""
        main_container = toga.Box(style=Pack(direction=COLUMN))
        
        # Header
        header = self.build_header()
        main_container.add(header)
        
        # Status bar (build first to initialize labels)
        status_bar = self.build_status_bar()
        
        # Content area
        content = toga.Box(style=Pack(direction=ROW, flex=1))
        
        # Left sidebar
        left_sidebar = self.build_sidebar()
        content.add(left_sidebar)
        
        # Chat area
        chat_area = self.build_chat_area()
        content.add(chat_area)
        
        main_container.add(content)
        
        # Add status bar at the bottom
        main_container.add(status_bar)
        
        self.main_window.content = main_container
    
    def build_header(self) -> toga.Box:
        """Build application header."""
        header = toga.Box(style=Pack(
            direction=ROW,
            align_items="center",
            margin=10,
            background_color="#2c3e50"
        ))
        
        title = toga.Label(
            "Tektra AI Assistant - Demo",
            style=Pack(
                font_size=18,
                font_weight="bold",
                color="#ecf0f1",
                flex=1
            )
        )
        
        self.status_indicator = toga.Label(
            "⚪ Initializing...",
            style=Pack(color="#95a5a6", font_size=12)
        )
        
        header.add(title)
        header.add(self.status_indicator)
        
        return header
    
    def build_sidebar(self) -> toga.Box:
        """Build left sidebar."""
        sidebar = toga.Box(style=Pack(
            direction=COLUMN,
            width=200,
            margin=10,
            background_color="#34495e"
        ))
        
        # Demo info
        demo_label = toga.Label(
            "DEMO MODE",
            style=Pack(
                font_weight="bold",
                color="#e74c3c",
                margin_bottom=10
            )
        )
        sidebar.add(demo_label)
        
        info_label = toga.Label(
            "This is a demonstration of the Tektra GUI without AI dependencies.",
            style=Pack(
                color="#ecf0f1",
                font_size=10,
                margin_bottom=15,
                text_align="left"
            )
        )
        sidebar.add(info_label)
        
        # Architecture info
        arch_label = toga.Label(
            "Architecture:",
            style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5)
        )
        sidebar.add(arch_label)
        
        arch_info = toga.Label(
            "• Briefcase + Toga GUI\n• Smart Router\n• Qwen Backend (Mock)\n• Unmute Pipeline (Mock)\n• Async Architecture",
            style=Pack(color="#95a5a6", font_size=10, margin_bottom=15)
        )
        sidebar.add(arch_info)
        
        # Demo controls
        controls_label = toga.Label(
            "Demo Controls:",
            style=Pack(font_weight="bold", color="#ecf0f1", margin_bottom=5)
        )
        sidebar.add(controls_label)
        
        self.reset_btn = toga.Button(
            "Reset Chat",
            on_press=self.reset_chat,
            style=Pack(width=180, margin_bottom=10)
        )
        sidebar.add(self.reset_btn)
        
        return sidebar
    
    def build_chat_area(self) -> toga.Box:
        """Build chat interface."""
        chat_container = toga.Box(style=Pack(
            direction=COLUMN,
            flex=1,
            margin=10
        ))
        
        # Chat display
        self.chat_display = toga.ScrollContainer(style=Pack(
            flex=1,
            background_color="#ffffff",
            margin=10
        ))
        
        self.messages_container = toga.Box(style=Pack(direction=COLUMN))
        self.chat_display.content = self.messages_container
        
        chat_container.add(self.chat_display)
        
        # Input area
        input_area = toga.Box(style=Pack(
            direction=ROW,
            margin_top=10,
            align_items="center"
        ))
        
        self.message_input = toga.TextInput(
            placeholder="Type your message here... (try 'analyze this' or 'hello')",
            style=Pack(flex=1, margin_right=10)
        )
        input_area.add(self.message_input)
        
        self.send_btn = toga.Button(
            "Send",
            on_press=self.send_message,
            style=Pack(width=80),
            enabled=False
        )
        input_area.add(self.send_btn)
        
        chat_container.add(input_area)
        
        # Add welcome message
        self.add_welcome_message()
        
        return chat_container
    
    def build_status_bar(self) -> toga.Box:
        """Build status bar."""
        status_bar = toga.Box(style=Pack(
            direction=ROW,
            margin=5,
            background_color="#34495e",
            align_items="center"
        ))
        
        self.status_label = toga.Label(
            "Initializing backends...",
            style=Pack(flex=1, color="#ecf0f1", font_size=12)
        )
        
        self.stats_label = toga.Label(
            "Messages: 0",
            style=Pack(color="#95a5a6", font_size=12)
        )
        
        status_bar.add(self.status_label)
        status_bar.add(self.stats_label)
        
        return status_bar
    
    def add_welcome_message(self):
        """Add welcome message."""
        welcome_text = """Welcome to Tektra AI Assistant Demo! 🤖

This demonstration showcases:
• Native Python desktop app with Briefcase + Toga
• Hybrid AI architecture (Unmute + Qwen)
• Smart query routing
• Async backend processing

Try typing messages like:
• "Hello, how are you?" (routes to Unmute)
• "Analyze this complex problem" (routes to Qwen)

The backends are mocked for this demo."""
        
        self.add_message("assistant", welcome_text)
    
    def add_message(self, role: str, content: str):
        """Add message to chat display."""
        message_box = toga.Box(style=Pack(
            direction=ROW,
            margin=(5, 0),
            align_items="start" if role == "assistant" else "end"
        ))
        
        # Message styling
        bg_color = "#e3f2fd" if role == "assistant" else "#2196f3"
        text_color = "#1976d2" if role == "assistant" else "#ffffff"
        
        message_label = toga.Label(
            content,
            style=Pack(
                margin=10,
                background_color=bg_color,
                color=text_color
            )
        )
        
        if role == "user":
            # Right-align user messages
            spacer = toga.Box(style=Pack(flex=1))
            message_box.add(spacer)
        
        message_box.add(message_label)
        
        if role == "assistant":
            # Left-align assistant messages
            spacer = toga.Box(style=Pack(flex=1))
            message_box.add(spacer)
        
        self.messages_container.add(message_box)
        self.messages.append({"role": role, "content": content})
        
        # Update stats
        self.update_stats()
    
    async def initialize_backends(self):
        """Initialize mock backends."""
        try:
            # Initialize Qwen backend
            self.status_label.text = "Initializing Qwen backend..."
            self.status_indicator.text = "🟡 Loading..."
            await self.qwen_backend.initialize()
            
            # Initialize Unmute backend
            self.status_label.text = "Initializing Unmute backend..."
            await self.unmute_backend.initialize()
            
            # Initialize smart router
            self.status_label.text = "Initializing smart router..."
            await asyncio.sleep(0.5)
            self.smart_router = MockSmartRouter(self.qwen_backend, self.unmute_backend)
            
            # Complete initialization
            self.is_initialized = True
            self.status_label.text = "Tektra AI Assistant ready!"
            self.status_indicator.text = "🟢 Ready"
            self.send_btn.enabled = True
            
            logger.success("All backends initialized successfully!")
            
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            self.status_label.text = f"Initialization failed: {e}"
            self.status_indicator.text = "🔴 Error"
    
    async def send_message(self, widget):
        """Send message through smart router."""
        message = self.message_input.value.strip()
        if not message or not self.is_initialized:
            return
        
        # Add user message
        self.add_message("user", message)
        
        # Clear input
        self.message_input.value = ""
        
        # Show processing
        self.status_label.text = "Processing message..."
        
        try:
            # Route through smart router
            result = await self.smart_router.route_query(message)
            
            # Add response
            response = result["response"]
            route = result["routing_decision"]["route"]
            self.add_message("assistant", f"{response}\n\n[Routed to: {route}]")
            
            self.status_label.text = "Ready"
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.add_message("assistant", f"Error: {e}")
            self.status_label.text = "Ready"
    
    def reset_chat(self, widget):
        """Reset chat messages."""
        # Clear messages container
        for message_widget in list(self.messages_container.children):
            self.messages_container.remove(message_widget)
        
        self.messages.clear()
        self.add_welcome_message()
        logger.info("Chat reset")
    
    def update_stats(self):
        """Update statistics display."""
        message_count = len(self.messages)
        self.stats_label.text = f"Messages: {message_count}"
        
        if self.smart_router:
            stats = self.smart_router.get_router_stats()
            unmute_pct = stats.get("unmute_percentage", 0)
            qwen_pct = stats.get("qwen_percentage", 0)
            self.stats_label.text = f"Messages: {message_count} | Unmute: {unmute_pct}% | Qwen: {qwen_pct}%"


def main():
    """Main entry point."""
    logger.info(f"Starting Tektra Demo on {platform.system()} {platform.release()}")
    app = TektraDemoApp("Tektra AI Assistant Demo", "org.tektra.demo")
    return app.main_loop()


if __name__ == "__main__":
    sys.exit(main())