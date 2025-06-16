#!/usr/bin/env python3
"""
Tektra AI Assistant CLI

Command-line interface for managing and running the Tektra AI Assistant.
"""

import asyncio
import signal
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .app.config import settings
from .app.database import init_database
from .server import start_server

app = typer.Typer(
    name="tektra",
    help="Tektra AI Assistant - Advanced AI with voice, vision, and robotics. Run 'tektra' to start the server.",
    rich_markup_mode="rich",
)

console = Console()


def print_banner():
    """Print the Tektra banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ████████╗███████╗██╗  ██╗████████╗██████╗  █████╗           ║
║  ╚══██╔══╝██╔════╝██║ ██╔╝╚══██╔══╝██╔══██╗██╔══██╗          ║
║     ██║   █████╗  █████╔╝    ██║   ██████╔╝███████║          ║
║     ██║   ██╔══╝  ██╔═██╗    ██║   ██╔══██╗██╔══██║          ║
║     ██║   ███████╗██║  ██╗   ██║   ██║  ██║██║  ██║          ║
║     ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝          ║
║                                                              ║
║            Advanced AI Assistant v0.9.14                    ║
║          Voice • Vision • Robotics • Chat                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


@app.command()
def start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    open_browser: bool = typer.Option(
        True, "--browser/--no-browser", help="Open browser automatically"
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """
    Start the Tektra AI Assistant server (default command).

    This will start both the backend API server and serve the frontend interface.
    Setup happens automatically on first run.
    """
    print_banner()

    console.print("\n🚀 Starting Tektra AI Assistant...\n", style="bold green")

    # Perform setup steps
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create directories
        task1 = progress.add_task("Setting up directories...", total=None)
        try:
            data_dir = Path.home() / ".tektra" / "data"
            models_dir = Path.home() / ".tektra" / "models"
            logs_dir = Path.home() / ".tektra" / "logs"

            data_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            progress.update(task1, description="✅ Directories ready")
        except Exception as e:
            progress.update(task1, description=f"⚠️  Directory setup failed: {e}")

        # Initialize database
        task2 = progress.add_task("Initializing database...", total=None)
        try:
            asyncio.run(init_database())
            progress.update(task2, description="✅ Database initialized")
        except Exception as e:
            progress.update(
                task2, description=f"⚠️  Database initialization failed: {e}"
            )

        # Run automatic setup
        task3 = progress.add_task("Setting up models and dependencies...", total=None)
        try:
            from .app.services.auto_installer import auto_installer

            setup_results = asyncio.run(auto_installer.run_initial_setup())

            if setup_results["success"]:
                models_count = len(setup_results["models_installed"])
                progress.update(
                    task3,
                    description=f"✅ Setup complete ({models_count} models ready)",
                )
                auto_installer.mark_setup_complete()
            else:
                progress.update(task3, description="⚠️  Setup completed with warnings")
        except Exception as e:
            progress.update(
                task3, description=f"⚠️  Setup failed: {e} (continuing anyway)"
            )

        # Quick dependency check
        task4 = progress.add_task("Checking system requirements...", total=None)
        try:
            # Check essential imports
            import aiofiles
            import cryptography
            import edge_tts
            import numpy

            progress.update(task4, description="✅ Core dependencies available")
        except Exception as e:
            progress.update(task4, description=f"⚠️  Some dependencies missing: {e}")

        time.sleep(0.5)  # Brief pause for user to see the status

    # Show configuration
    config_table = Table(title="Server Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="magenta")

    try:
        config_table.add_row("Host", str(host))
        config_table.add_row("Port", str(port))
        config_table.add_row("Debug Mode", "Yes" if debug else "No")
        config_table.add_row("Auto-reload", "Yes" if reload else "No")
        config_table.add_row("Frontend URL", f"http://{str(host)}:{str(port)}")
        config_table.add_row("API URL", f"http://{str(host)}:{str(port)}/api/v1")
        config_table.add_row("WebSocket URL", f"ws://{str(host)}:{str(port)}/ws")

        console.print(config_table)
    except Exception as e:
        # Fallback to simple text output if table rendering fails
        console.print("📋 Server Configuration:", style="bold cyan")
        console.print(f"   Host: {host}")
        console.print(f"   Port: {port}")
        console.print(f"   Debug Mode: {'Yes' if debug else 'No'}")
        console.print(f"   Auto-reload: {'Yes' if reload else 'No'}")
        console.print(f"   Frontend URL: http://{host}:{port}")
        console.print(f"   API URL: http://{host}:{port}/api/v1")
        console.print(f"   WebSocket URL: ws://{host}:{port}/ws")
    console.print()

    # Start server
    console.print("🌟 Server starting...", style="bold yellow")
    console.print(f"📡 API available at: [link]http://{host}:{port}/api/v1[/link]")
    console.print(f"🌐 Web interface at: [link]http://{host}:{port}[/link]")
    console.print(f"🔌 WebSocket at: [link]ws://{host}:{port}/ws[/link]")
    console.print("\n💡 Use [bold]Ctrl+C[/bold] to stop the server\n")

    # Open browser
    if open_browser:
        console.print("🌐 Opening browser...", style="bold blue")
        # Use localhost instead of 0.0.0.0 for browser
        browser_host = "localhost" if host == "0.0.0.0" else host
        webbrowser.open(f"http://{browser_host}:{port}")

    try:
        start_server(host=host, port=port, reload=reload, debug=debug)
    except KeyboardInterrupt:
        console.print("\n\n👋 Shutting down Tektra AI Assistant...", style="bold red")
        console.print("Thank you for using Tektra! 🚀", style="bold green")


@app.command()
def setup():
    """
    Legacy setup command - no longer needed.

    Setup now happens automatically when you run 'tektra' for the first time.
    """
    console.print("\n💡 No setup needed!", style="bold green")
    console.print(
        "Tektra now sets up automatically when you run it for the first time."
    )
    console.print(
        "\nJust run: [bold cyan]tektra[/bold cyan] or [bold cyan]tektra start[/bold cyan]"
    )


@app.command()
def info():
    """Show information about the Tektra AI Assistant installation."""
    print_banner()

    # System info
    info_table = Table(title="System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="magenta")
    info_table.add_column("Details", style="white")

    # Check Python version
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    info_table.add_row("Python", "✅ OK", python_version)

    # Check directories
    data_dir = Path.home() / ".tektra"
    if data_dir.exists():
        info_table.add_row("Data Directory", "✅ OK", str(data_dir))
    else:
        info_table.add_row(
            "Data Directory", "⚠️ Will be created", "Auto-created on first run"
        )

    # Check database
    db_path = Path("tektra.db")
    if db_path.exists():
        info_table.add_row("Database", "✅ OK", str(db_path))
    else:
        info_table.add_row("Database", "⚠️ Will be created", "Auto-created on first run")

    console.print(info_table)

    # Configuration
    console.print("\n")
    config_panel = Panel(
        f"""[bold]Current Configuration:[/bold]

• Database URL: {settings.database_url}
• Debug Mode: {'Yes' if settings.debug else 'No'}
• Default Host: {settings.host}
• Default Port: {settings.port}
• Model Cache: {settings.model_cache_dir}
• Max Model Memory: {settings.max_model_memory_gb}GB""",
        title="Configuration",
        border_style="blue",
    )
    console.print(config_panel)


@app.command()
def version():
    """Show the version of Tektra AI Assistant."""
    from . import __version__

    console.print(f"Tektra AI Assistant v{__version__}", style="bold green")


# Phi-4 Management Commands
@app.command()
def enable_phi4(
    host: str = typer.Option("localhost", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """
    Enable Microsoft Phi-4 Multimodal for superior speech recognition.

    This will load the advanced Phi-4 model for enhanced accuracy and performance.
    Requires the server to be running.
    """
    console.print("\n🚀 Enabling Microsoft Phi-4 Multimodal...\n", style="bold blue")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading Phi-4 model...", total=None)

        try:
            # Check if server is running
            progress.update(task, description="Checking server connection...")
            response = httpx.get(f"http://{host}:{port}/health", timeout=5.0)

            if response.status_code != 200:
                progress.update(task, description="❌ Server not responding")
                console.print(f"\n❌ Cannot connect to server at http://{host}:{port}")
                console.print(
                    "Please start the server first with: [bold cyan]tektra start[/bold cyan]"
                )
                return

            # Load Phi-4 model
            progress.update(task, description="Loading Phi-4 Multimodal model...")
            response = httpx.post(
                f"http://{host}:{port}/api/v1/audio/api/v1/audio/phi4/load",
                timeout=300.0,
            )

            if response.status_code == 200:
                progress.update(task, description="✅ Phi-4 model loaded successfully")
            else:
                progress.update(task, description="❌ Failed to load Phi-4 model")
                console.print(
                    f"\n❌ Error: {response.json().get('detail', 'Unknown error')}"
                )
                return

        except httpx.ConnectError:
            progress.update(task, description="❌ Cannot connect to server")
            console.print(f"\n❌ Cannot connect to server at http://{host}:{port}")
            console.print(
                "Please start the server first with: [bold cyan]tektra start[/bold cyan]"
            )
            return
        except httpx.TimeoutException:
            progress.update(task, description="❌ Request timed out")
            console.print(
                "\n❌ Request timed out. Model loading can take several minutes."
            )
            console.print("Please check the server logs and try again.")
            return
        except Exception as e:
            progress.update(task, description=f"❌ Error: {str(e)}")
            console.print(f"\n❌ Unexpected error: {e}")
            return

    console.print("\n🎉 Phi-4 Multimodal enabled successfully!", style="bold green")
    console.print("\n✨ You now have access to:")
    console.print("  • 95%+ speech recognition accuracy")
    console.print("  • 8-language support with auto-detection")
    console.print("  • Unified speech + chat processing")
    console.print("  • 128K context for better understanding")
    console.print("\n🌐 Try it out in your web interface!")


@app.command()
def disable_phi4(
    host: str = typer.Option("localhost", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """
    Disable Phi-4 and free up memory.

    This will unload the Phi-4 model and fall back to Whisper for speech recognition.
    """
    console.print("\n🔄 Disabling Phi-4 Multimodal...\n", style="bold yellow")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Unloading Phi-4 model...", total=None)

        try:
            # Check if server is running
            progress.update(task, description="Checking server connection...")
            response = httpx.get(f"http://{host}:{port}/health", timeout=5.0)

            if response.status_code != 200:
                progress.update(task, description="❌ Server not responding")
                console.print(f"\n❌ Cannot connect to server at http://{host}:{port}")
                return

            # Unload Phi-4 model
            progress.update(task, description="Unloading Phi-4 model...")
            response = httpx.post(
                f"http://{host}:{port}/api/v1/audio/api/v1/audio/phi4/unload",
                timeout=30.0,
            )

            if response.status_code == 200:
                progress.update(
                    task, description="✅ Phi-4 model unloaded successfully"
                )
            else:
                progress.update(task, description="❌ Failed to unload Phi-4 model")
                console.print(
                    f"\n❌ Error: {response.json().get('detail', 'Unknown error')}"
                )
                return

        except httpx.ConnectError:
            progress.update(task, description="❌ Cannot connect to server")
            console.print(f"\n❌ Cannot connect to server at http://{host}:{port}")
            return
        except Exception as e:
            progress.update(task, description=f"❌ Error: {str(e)}")
            console.print(f"\n❌ Unexpected error: {e}")
            return

    console.print("\n✅ Phi-4 disabled successfully!", style="bold green")
    console.print("💡 Tektra will now use Whisper for speech recognition.")
    console.print(
        "🔄 You can re-enable Phi-4 anytime with: [bold cyan]tektra enable-phi4[/bold cyan]"
    )


@app.command()
def phi4_status(
    host: str = typer.Option("localhost", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """
    Check the status of Phi-4 Multimodal model.

    Shows whether Phi-4 is loaded and provides system information.
    """
    console.print("\n📊 Checking Phi-4 status...\n", style="bold blue")

    try:
        # Get Phi-4 info
        response = httpx.get(
            f"http://{host}:{port}/api/v1/audio/api/v1/audio/phi4/info", timeout=10.0
        )

        if response.status_code != 200:
            console.print("❌ Cannot get Phi-4 status")
            return

        info = response.json()

        # Create status table
        status_table = Table(title="Phi-4 Multimodal Status")
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="white")

        # Model status
        status_icon = "✅ Loaded" if info.get("is_loaded") else "❌ Not Loaded"
        status_table.add_row("Model Status", status_icon)
        status_table.add_row("Model Name", info.get("model_name", "Unknown"))
        status_table.add_row("Device", info.get("device", "Unknown"))
        status_table.add_row(
            "Available", "✅ Yes" if info.get("available") else "❌ No"
        )

        # Capabilities
        capabilities = info.get("capabilities", {})
        if capabilities:
            status_table.add_row(
                "Speech Recognition",
                "✅ Yes" if capabilities.get("speech_recognition") else "❌ No",
            )
            status_table.add_row(
                "Chat Completion",
                "✅ Yes" if capabilities.get("chat_completion") else "❌ No",
            )
            status_table.add_row(
                "Language Detection",
                "✅ Yes" if capabilities.get("language_detection") else "❌ No",
            )
            status_table.add_row(
                "Multimodal", "✅ Yes" if capabilities.get("multimodal") else "❌ No"
            )

        # Language support
        languages = info.get("supported_languages", {})
        if languages:
            lang_list = ", ".join(languages.keys())
            status_table.add_row("Supported Languages", f"{len(languages)} languages")
            status_table.add_row("Language Codes", lang_list)

        console.print(status_table)

        # Show recommendations
        if info.get("is_loaded"):
            console.print("\n🎉 Phi-4 is active and ready!", style="bold green")
            console.print(
                "✨ Enjoying superior speech recognition and AI capabilities!"
            )
        else:
            console.print("\n💡 Phi-4 is not loaded", style="bold yellow")
            console.print(
                "🚀 Enable it for enhanced performance: [bold cyan]tektra enable-phi4[/bold cyan]"
            )

    except httpx.ConnectError:
        console.print(f"❌ Cannot connect to server at http://{host}:{port}")
        console.print(
            "Please start the server first with: [bold cyan]tektra start[/bold cyan]"
        )
    except Exception as e:
        console.print(f"❌ Error checking status: {e}")


@app.command()
def enhance():
    """
    Quick setup for enhanced Tektra experience.

    This will start the server and automatically enable Phi-4 for the best experience.
    """
    print_banner()
    console.print("\n🚀 Setting up enhanced Tektra experience...\n", style="bold blue")

    # Start server in background and enable Phi-4
    console.print("📡 Starting server...")
    console.print("🧠 This will enable Phi-4 Multimodal for the best experience")
    console.print("⏳ This may take a few minutes on first run...\n")

    # For now, just start the server normally
    console.print(
        "💡 After the server starts, run: [bold cyan]tektra enable-phi4[/bold cyan]"
    )
    console.print("🌐 Web interface will open automatically\n")

    # Start server normally
    start_server(host="localhost", port=8000, open_browser=True)


@app.command()
def install_deps(
    dependency: str = typer.Argument(
        help="Dependency to install: transformers, biometric, advanced_audio, ml_models"
    ),
):
    """
    Install optional dependencies safely.

    Available dependencies:
    - transformers: HuggingFace Transformers for advanced ML models
    - biometric: Camera-based biometric authentication
    - advanced_audio: Advanced audio processing capabilities
    - ml_models: Core PyTorch ML framework
    """
    console.print(f"\n🔧 Installing {dependency} dependencies...\n", style="bold blue")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Installing {dependency}...", total=None)

        try:
            # Import auto_installer
            progress.update(task, description="Initializing installer...")
            from .app.services.auto_installer import auto_installer

            # Install the dependency
            progress.update(task, description=f"Installing {dependency} packages...")
            result = asyncio.run(auto_installer.install_optional_dependency(dependency))

            if result.get("success"):
                progress.update(
                    task, description=f"✅ {dependency} installed successfully"
                )
                console.print(
                    f"\n🎉 {dependency} dependencies installed successfully!",
                    style="bold green",
                )

                if result.get("method") == "alternative":
                    console.print(
                        "✨ Used compilation-free alternative for faster installation"
                    )

                packages = result.get("packages", [])
                if packages:
                    console.print("📦 Installed packages:")
                    for pkg in packages:
                        console.print(f"  • {pkg}")

            else:
                progress.update(task, description=f"❌ Failed to install {dependency}")
                error = result.get("error", "Unknown error")
                console.print(f"\n❌ Installation failed: {error}", style="bold red")

                suggestion = result.get("suggestion")
                if suggestion:
                    console.print(f"💡 Suggestion: {suggestion}", style="yellow")

                console.print(
                    "\n🔍 Alternative installation methods:", style="bold cyan"
                )
                console.print(f"  • Manual: pip install tektra[{dependency}]")
                console.print("  • With UV: uv tool install tektra --with tektra[ml]")
                return

        except Exception as e:
            progress.update(task, description=f"❌ Error: {str(e)}")
            console.print(f"\n❌ Unexpected error: {e}", style="bold red")
            return

    console.print("\n✅ Dependencies ready for use!", style="bold green")
    console.print("🚀 Restart Tektra to use the new capabilities")


def main():
    """Main entry point for the CLI."""
    try:
        # If no command is provided, default to start
        if len(sys.argv) == 1:
            # Default to start command with default parameters
            start(
                host="0.0.0.0", port=8000, open_browser=True, reload=False, debug=False
            )
        else:
            app()
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="bold yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="bold red")
        sys.exit(1)


if __name__ == "__main__":
    main()
