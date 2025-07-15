"""
Docker Utilities - Standalone Mode

This module provides empty stubs for Docker functionality since Tektra
now runs in standalone mode with embedded AI models.
"""

from typing import Any

from loguru import logger


class DockerUtils:
    """Docker utilities stub for standalone mode."""

    def __init__(self):
        """Initialize Docker utilities (standalone mode)."""
        self.docker_available = False
        logger.info("Docker utilities disabled - running in standalone mode")

    def is_docker_available(self) -> bool:
        """Check if Docker is available (always False in standalone mode)."""
        return False

    def get_docker_info(self) -> dict[str, Any]:
        """Get Docker system information (not available in standalone mode)."""
        return {
            "available": False, 
            "error": "Docker not available in standalone mode"
        }

    def list_containers(self, all_containers: bool = False) -> list[dict[str, Any]]:
        """List Docker containers (empty in standalone mode)."""
        return []

    def get_container_logs(self, container_name: str, lines: int = 50) -> str:
        """Get logs from a specific container (not available in standalone mode)."""
        return "Docker not available in standalone mode"

    def is_container_running(self, container_name: str) -> bool:
        """Check if a specific container is running (always False in standalone mode)."""
        return False

    def start_container(self, container_name: str) -> bool:
        """Start a specific container (not available in standalone mode)."""
        return False

    def stop_container(self, container_name: str) -> bool:
        """Stop a specific container (not available in standalone mode)."""
        return False

    def health_check_service(self, service_url: str, timeout: int = 5) -> bool:
        """Check if a service is healthy (not available in standalone mode)."""
        return False

    async def wait_for_service_health(
        self, service_url: str, max_attempts: int = 30, delay: float = 2.0
    ) -> bool:
        """Wait for a service to become healthy (not available in standalone mode)."""
        return False

    def cleanup_containers(self, filter_labels: dict[str, str] | None = None) -> int:
        """Clean up stopped containers (not available in standalone mode)."""
        return 0

    def get_system_resources(self) -> dict[str, Any]:
        """Get system resource usage (not available in standalone mode)."""
        return {"available": False, "error": "Docker not available in standalone mode"}