"""
Unmute Service Manager

This module manages the Kyutai Unmute services via Docker Compose, providing:
- Automatic service startup and shutdown
- Health monitoring and status checking
- Service configuration and management
- Integration with the main Tektra application
"""

import asyncio
import time
from pathlib import Path

import aiohttp
import docker
from loguru import logger


class UnmuteServiceManager:
    """
    Manages Kyutai Unmute services using Docker Compose.

    This class handles the lifecycle of Unmute services including:
    - Starting and stopping services via Docker Compose
    - Health monitoring and status checking
    - Service configuration and port management
    - Integration with the main application
    """

    def __init__(
        self, unmute_path: Path, base_url: str | None = None, config: dict | None = None
    ):
        """
        Initialize the Unmute Service Manager.

        Args:
            unmute_path: Path to the Unmute directory (git submodule)
            base_url: Base URL for service communication (deprecated, use config)
            config: Service configuration dictionary
        """
        self.unmute_path = Path(unmute_path)
        # Support legacy base_url parameter for backward compatibility
        if config:
            self.base_url = config.get("base_url", "http://localhost")
            self.service_ports = {
                "stt": config.get("stt_port", 8001),
                "tts": config.get("tts_port", 8002),
                "llm": config.get("llm_port", 8000),
                "backend": config.get("backend_port", 8000),
            }
            self.websocket_base_url = config.get("websocket_url", "ws://localhost:8000")
        elif base_url:
            self.base_url = base_url
            self.service_ports = {
                "stt": 8001,
                "tts": 8002,
                "llm": 8000,
                "backend": 8000,
            }
            # Extract host from base_url and construct websocket URL
            host = base_url.replace("http://", "").replace("https://", "")
            self.websocket_base_url = f"ws://{host}:{self.service_ports['backend']}"
        else:
            self.base_url = "http://localhost"
            self.service_ports = {
                "stt": 8001,
                "tts": 8002,
                "llm": 8000,
                "backend": 8000,
            }
            self.websocket_base_url = "ws://localhost:8000"
        self.services_running = False

        # Docker client for direct container management
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None

        # Validate unmute path
        self.compose_file = self.unmute_path / "docker-compose.yml"
        if not self.compose_file.exists():
            logger.warning(
                f"Unmute docker-compose.yml not found at {self.compose_file}"
            )

    async def setup_unmute_services(self) -> bool:
        """
        Start Unmute services via Docker Compose.

        Returns:
            bool: True if services started successfully, False otherwise
        """
        if not self.compose_file.exists():
            logger.error(f"Cannot start services: {self.compose_file} not found")
            return False

        logger.info("Starting Unmute services via Docker Compose...")

        try:
            # First, stop any existing services
            await self._stop_compose_services()

            # Start services
            cmd = ["docker-compose", "-f", str(self.compose_file), "up", "-d"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.unmute_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Failed to start Unmute services: {error_msg}")
                return False

            logger.info("Docker Compose services started successfully")
            self.services_running = True

            # Wait for services to be ready
            logger.info("Waiting for services to become healthy...")
            healthy = await self.wait_for_services(timeout=120)

            if healthy:
                logger.success("All Unmute services are healthy and ready!")
                return True
            else:
                logger.error("Services started but failed health checks")
                return False

        except Exception as e:
            logger.error(f"Exception starting Unmute services: {e}")
            return False

    async def stop_unmute_services(self) -> bool:
        """
        Stop Unmute services.

        Returns:
            bool: True if services stopped successfully, False otherwise
        """
        logger.info("Stopping Unmute services...")

        try:
            result = await self._stop_compose_services()
            if result:
                self.services_running = False
                logger.info("Unmute services stopped successfully")
            return result
        except Exception as e:
            logger.error(f"Error stopping Unmute services: {e}")
            return False

    async def _stop_compose_services(self) -> bool:
        """Stop services via Docker Compose."""
        if not self.compose_file.exists():
            return True

        cmd = ["docker-compose", "-f", str(self.compose_file), "down"]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.unmute_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        return process.returncode == 0

    async def check_service_health(self) -> dict[str, bool]:
        """
        Check health of all Unmute services.

        Returns:
            Dict mapping service names to health status (True = healthy)
        """
        health_status = {}

        # Check each service endpoint
        services_to_check = [
            ("backend", f"{self.base_url}:{self.service_ports['backend']}/health"),
            ("stt", f"{self.base_url}:{self.service_ports['stt']}/health"),
            ("tts", f"{self.base_url}:{self.service_ports['tts']}/health"),
        ]

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            for service_name, url in services_to_check:
                try:
                    async with session.get(url) as response:
                        health_status[service_name] = response.status == 200
                        if response.status == 200:
                            logger.debug(f"Service {service_name} is healthy")
                        else:
                            logger.warning(
                                f"Service {service_name} returned status {response.status}"
                            )
                except Exception as e:
                    health_status[service_name] = False
                    logger.debug(f"Service {service_name} health check failed: {e}")

        return health_status

    async def wait_for_services(self, timeout: int = 120) -> bool:
        """
        Wait for all services to become healthy.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if all services became healthy within timeout
        """
        start_time = time.time()
        check_interval = 5

        logger.info(f"Waiting up to {timeout}s for services to become ready...")

        while time.time() - start_time < timeout:
            health = await self.check_service_health()

            healthy_services = [name for name, status in health.items() if status]
            total_services = len(health)

            logger.info(
                f"Healthy services: {len(healthy_services)}/{total_services} "
                f"({', '.join(healthy_services)})"
            )

            if all(health.values()) and len(health) > 0:
                logger.success("All services are healthy!")
                return True

            # Log which services are still not ready
            unhealthy = [name for name, status in health.items() if not status]
            if unhealthy:
                logger.debug(f"Waiting for services: {', '.join(unhealthy)}")

            await asyncio.sleep(check_interval)

        # Final check and detailed error reporting
        final_health = await self.check_service_health()
        unhealthy_services = [
            name for name, status in final_health.items() if not status
        ]

        if unhealthy_services:
            logger.error(
                f"Services not ready after {timeout}s: {', '.join(unhealthy_services)}"
            )

            # Try to get more detailed error information
            await self._log_service_details()

        return len(unhealthy_services) == 0

    async def _log_service_details(self):
        """Log detailed information about service status for debugging."""
        if not self.docker_client:
            return

        try:
            # Get container information
            containers = self.docker_client.containers.list(all=True)
            unmute_containers = [c for c in containers if "unmute" in c.name.lower()]

            logger.info("Docker container status:")
            for container in unmute_containers:
                logger.info(f"  {container.name}: {container.status}")

                # Get recent logs
                try:
                    logs = container.logs(tail=10).decode("utf-8", errors="ignore")
                    if logs.strip():
                        logger.debug(f"Recent logs for {container.name}:\n{logs}")
                except Exception as e:
                    logger.debug(f"Could not get logs for {container.name}: {e}")

        except Exception as e:
            logger.debug(f"Error getting container details: {e}")

    async def get_service_status(self) -> dict[str, any]:
        """
        Get comprehensive status of Unmute services.

        Returns:
            Dict containing service status information
        """
        health = await self.check_service_health()

        status = {
            "services_running": self.services_running,
            "compose_file_exists": self.compose_file.exists(),
            "unmute_path": str(self.unmute_path),
            "service_health": health,
            "healthy_count": sum(1 for status in health.values() if status),
            "total_services": len(health),
            "all_healthy": all(health.values()) and len(health) > 0,
            "service_urls": {
                name: f"{self.base_url}:{port}"
                for name, port in self.service_ports.items()
            },
        }

        return status

    async def restart_service(self, service_name: str) -> bool:
        """
        Restart a specific service.

        Args:
            service_name: Name of the service to restart

        Returns:
            bool: True if restart was successful
        """
        if not self.docker_client:
            logger.error("Docker client not available for service restart")
            return False

        try:
            containers = self.docker_client.containers.list(all=True)
            target_container = None

            for container in containers:
                if service_name.lower() in container.name.lower():
                    target_container = container
                    break

            if not target_container:
                logger.error(f"Container for service {service_name} not found")
                return False

            logger.info(f"Restarting service: {service_name}")
            target_container.restart()

            # Wait a moment then check health
            await asyncio.sleep(5)
            health = await self.check_service_health()

            return health.get(service_name, False)

        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {e}")
            return False

    def get_websocket_url(self) -> str:
        """
        Get the WebSocket URL for Unmute backend communication.

        Returns:
            str: WebSocket URL for the Unmute backend
        """
        # Use configured websocket URL, appending /ws if not present
        if self.websocket_base_url.endswith("/ws"):
            return self.websocket_base_url
        else:
            return f"{self.websocket_base_url}/ws"

    def get_service_url(self, service: str) -> str:
        """
        Get the URL for a specific service.

        Args:
            service: Service name (stt, tts, llm, backend)

        Returns:
            str: Service URL
        """
        if service not in self.service_ports:
            raise ValueError(f"Unknown service: {service}")

        return f"{self.base_url}:{self.service_ports[service]}"

    async def cleanup(self):
        """Cleanup resources and stop services."""
        await self.stop_unmute_services()

        if self.docker_client:
            try:
                self.docker_client.close()
            except Exception as e:
                logger.debug(f"Error closing Docker client: {e}")
