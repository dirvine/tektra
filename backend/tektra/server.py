#!/usr/bin/env python3
"""
Tektra AI Assistant Server

Server management and startup logic.
"""

import os
from pathlib import Path

import uvicorn

from .app.config import settings
from .app.main import app as fastapi_app


def start_server(
    host: str = "0.0.0.0", port: int = 8000, reload: bool = False, debug: bool = False
):
    """
    Start the Tektra AI Assistant server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        debug: Enable debug mode
    """
    # Update settings
    settings.host = host
    settings.port = port
    settings.debug = debug
    settings.reload = reload

    # Configure uvicorn
    uvicorn_config = {
        "app": "tektra.app.main:app",
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": "debug" if debug else "info",
        "access_log": debug,
        "use_colors": True,
    }

    # Add static file serving for frontend
    if not reload:
        # In production mode, serve the built frontend
        frontend_dir = Path(__file__).parent / "frontend" / "dist"
        if frontend_dir.exists():
            uvicorn_config["app"] = fastapi_app

    # Start the server
    uvicorn.run(**uvicorn_config)


def main():
    """Main entry point for the server command."""
    start_server(
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        debug=settings.debug,
    )


if __name__ == "__main__":
    main()
