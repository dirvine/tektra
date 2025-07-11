#!/usr/bin/env python3
"""
Tektra AI Assistant - Main Entry Point

This module provides the main entry point for the Tektra AI Assistant application.
"""

import sys
from pathlib import Path

# Add the source directory to Python path for development
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tektra.app import TektraApp


def main():
    """Main entry point for the Tektra AI Assistant application."""
    try:
        app = TektraApp(formal_name="Tektra AI Assistant", app_id="com.maidsafe.tektra")
        return app.main_loop()
    except KeyboardInterrupt:
        print("\nTektra AI Assistant stopped by user.")
        return 0
    except Exception as e:
        print(f"Fatal error starting Tektra AI Assistant: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
