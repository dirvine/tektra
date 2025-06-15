#!/usr/bin/env python3
"""
Tektra AI Assistant CLI

Command-line interface for managing and running the Tektra AI Assistant.
"""

import typer
import asyncio
import webbrowser
import time
import signal
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

from .server import start_server
from .app.database import init_database
from .app.config import settings

app = typer.Typer(
    name="tektra",
    help="Tektra AI Assistant - Advanced AI with voice, vision, and robotics",
    rich_markup_mode="rich"
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
║            Advanced AI Assistant v0.2.2                     ║
║          Voice • Vision • Robotics • Chat                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


@app.command()
def start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    open_browser: bool = typer.Option(True, "--browser/--no-browser", help="Open browser automatically"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """
    Start the Tektra AI Assistant server.
    
    This will start both the backend API server and serve the frontend interface.
    """
    print_banner()
    
    console.print("\n🚀 Starting Tektra AI Assistant...\n", style="bold green")
    
    # Show configuration
    config_table = Table(title="Server Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="magenta")
    
    config_table.add_row("Host", host)
    config_table.add_row("Port", str(port))
    config_table.add_row("Debug Mode", "Yes" if debug else "No")
    config_table.add_row("Auto-reload", "Yes" if reload else "No")
    config_table.add_row("Frontend URL", f"http://{host}:{port}")
    config_table.add_row("API URL", f"http://{host}:{port}/api/v1")
    config_table.add_row("WebSocket URL", f"ws://{host}:{port}/ws")
    
    console.print(config_table)
    console.print()
    
    # Initialize database
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing database...", total=None)
        try:
            asyncio.run(init_database())
            progress.update(task, description="✅ Database initialized")
        except Exception as e:
            progress.update(task, description=f"⚠️  Database initialization failed: {e}")
        
        time.sleep(0.5)  # Brief pause for user to see the status
    
    # Start server
    console.print("🌟 Server starting...", style="bold yellow")
    console.print(f"📡 API available at: [link]http://{host}:{port}/api/v1[/link]")
    console.print(f"🌐 Web interface at: [link]http://{host}:{port}[/link]")
    console.print(f"🔌 WebSocket at: [link]ws://{host}:{port}/ws[/link]")
    console.print("\n💡 Use [bold]Ctrl+C[/bold] to stop the server\n")
    
    # Open browser
    if open_browser:
        console.print("🌐 Opening browser...", style="bold blue")
        webbrowser.open(f"http://{host}:{port}")
    
    try:
        start_server(host=host, port=port, reload=reload, debug=debug)
    except KeyboardInterrupt:
        console.print("\n\n👋 Shutting down Tektra AI Assistant...", style="bold red")
        console.print("Thank you for using Tektra! 🚀", style="bold green")


@app.command()
def setup():
    """
    Set up Tektra AI Assistant for first use.
    
    This will initialize the database and create necessary directories.
    """
    print_banner()
    
    console.print("\n🔧 Setting up Tektra AI Assistant...\n", style="bold blue")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create directories
        task1 = progress.add_task("Creating directories...", total=None)
        try:
            data_dir = Path.home() / ".tektra" / "data"
            models_dir = Path.home() / ".tektra" / "models"
            logs_dir = Path.home() / ".tektra" / "logs"
            
            data_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            progress.update(task1, description="✅ Directories created")
        except Exception as e:
            progress.update(task1, description=f"❌ Directory creation failed: {e}")
            return
        
        # Initialize database
        task2 = progress.add_task("Initializing database...", total=None)
        try:
            asyncio.run(init_database())
            progress.update(task2, description="✅ Database initialized")
        except Exception as e:
            progress.update(task2, description=f"❌ Database initialization failed: {e}")
            return
        
        time.sleep(1)  # Brief pause for user to see the status
    
    console.print("\n🎉 Setup complete!", style="bold green")
    console.print("\nYou can now start Tektra with: [bold cyan]tektra start[/bold cyan]")


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
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info_table.add_row("Python", "✅ OK", python_version)
    
    # Check directories
    data_dir = Path.home() / ".tektra"
    if data_dir.exists():
        info_table.add_row("Data Directory", "✅ OK", str(data_dir))
    else:
        info_table.add_row("Data Directory", "❌ Missing", "Run 'tektra setup'")
    
    # Check database
    db_path = Path("tektra.db")
    if db_path.exists():
        info_table.add_row("Database", "✅ OK", str(db_path))
    else:
        info_table.add_row("Database", "❌ Missing", "Run 'tektra setup'")
    
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
        border_style="blue"
    )
    console.print(config_panel)


@app.command()
def version():
    """Show the version of Tektra AI Assistant."""
    from . import __version__
    console.print(f"Tektra AI Assistant v{__version__}", style="bold green")


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="bold yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="bold red")
        sys.exit(1)


if __name__ == "__main__":
    main()