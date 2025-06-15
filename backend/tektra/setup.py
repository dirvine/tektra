#!/usr/bin/env python3
"""
Tektra AI Assistant Setup

Setup and initialization utilities.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .app.database import init_database

console = Console()


async def setup_tektra():
    """
    Set up Tektra AI Assistant for first use.
    """
    console.print("🔧 Setting up Tektra AI Assistant...", style="bold blue")
    
    # Create directories
    data_dir = Path.home() / ".tektra" / "data"
    models_dir = Path.home() / ".tektra" / "models" 
    logs_dir = Path.home() / ".tektra" / "logs"
    
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        console.print("✅ Created directories", style="green")
    except Exception as e:
        console.print(f"❌ Failed to create directories: {e}", style="red")
        return False
    
    # Initialize database
    try:
        await init_database()
        console.print("✅ Database initialized", style="green")
    except Exception as e:
        console.print(f"❌ Database initialization failed: {e}", style="red")
        return False
    
    console.print("🎉 Setup complete!", style="bold green")
    return True


def main():
    """Main entry point for setup."""
    try:
        success = asyncio.run(setup_tektra())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n❌ Setup cancelled", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Setup failed: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()