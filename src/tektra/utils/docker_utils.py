"""
Docker Utilities

This module provides Docker management utilities for Tektra AI Assistant,
including container management and service orchestration.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

try:
    import docker
    from docker.errors import DockerException
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logger.warning("Docker library not available - using mock implementation")


class DockerUtils:
    """Docker management utilities for Tektra AI Assistant."""
    
    def __init__(self):
        """Initialize Docker utilities."""
        self.client = None
        self.docker_available = DOCKER_AVAILABLE
        
        if self.docker_available:
            try:
                self.client = docker.from_env()
                self.client.ping()
                logger.info("Docker client initialized successfully")
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
                self.docker_available = False
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        if not self.docker_available:
            return False
        
        try:
            self.client.ping()
            return True
        except Exception:
            return False
    
    def get_docker_info(self) -> Dict[str, Any]:
        """Get Docker system information."""
        if not self.is_docker_available():
            return {"available": False, "error": "Docker not available"}
        
        try:
            info = self.client.info()
            return {
                "available": True,
                "version": info.get("ServerVersion", "Unknown"),
                "containers": info.get("Containers", 0),
                "images": info.get("Images", 0),
                "memory": info.get("MemTotal", 0),
                "cpus": info.get("NCPU", 0)
            }
        except Exception as e:
            logger.error(f"Error getting Docker info: {e}")
            return {"available": False, "error": str(e)}
    
    def list_containers(self, all_containers: bool = False) -> List[Dict[str, Any]]:
        """List Docker containers."""
        if not self.is_docker_available():
            return []
        
        try:
            containers = self.client.containers.list(all=all_containers)
            return [
                {
                    "id": container.id[:12],
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "Unknown",
                    "status": container.status,
                    "created": container.attrs.get("Created", "Unknown"),
                    "ports": container.attrs.get("NetworkSettings", {}).get("Ports", {})
                }
                for container in containers
            ]
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []
    
    def get_container_logs(self, container_name: str, lines: int = 50) -> str:
        """Get logs from a specific container."""
        if not self.is_docker_available():
            return "Docker not available"
        
        try:
            container = self.client.containers.get(container_name)
            logs = container.logs(tail=lines).decode('utf-8')
            return logs
        except Exception as e:
            logger.error(f"Error getting container logs: {e}")
            return f"Error: {e}"
    
    def is_container_running(self, container_name: str) -> bool:
        """Check if a specific container is running."""
        if not self.is_docker_available():
            return False
        
        try:
            container = self.client.containers.get(container_name)
            return container.status == "running"
        except Exception:
            return False
    
    def start_container(self, container_name: str) -> bool:
        """Start a specific container."""
        if not self.is_docker_available():
            return False
        
        try:
            container = self.client.containers.get(container_name)
            container.start()
            logger.info(f"Container {container_name} started")
            return True
        except Exception as e:
            logger.error(f"Error starting container {container_name}: {e}")
            return False
    
    def stop_container(self, container_name: str) -> bool:
        """Stop a specific container."""
        if not self.is_docker_available():
            return False
        
        try:
            container = self.client.containers.get(container_name)
            container.stop()
            logger.info(f"Container {container_name} stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping container {container_name}: {e}")
            return False
    
    def run_docker_compose(self, 
                          compose_file: Path, 
                          command: str = "up -d",
                          working_dir: Optional[Path] = None) -> Tuple[bool, str]:
        """Run docker-compose command."""
        if working_dir is None:
            working_dir = compose_file.parent
        
        try:
            cmd = ["docker-compose", "-f", str(compose_file)] + command.split()
            
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Docker Compose command succeeded: {' '.join(cmd)}")
                return True, result.stdout
            else:
                logger.error(f"Docker Compose command failed: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error("Docker Compose command timed out")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"Error running Docker Compose: {e}")
            return False, str(e)
    
    def get_compose_services_status(self, compose_file: Path) -> Dict[str, str]:
        """Get status of services in a docker-compose file."""
        working_dir = compose_file.parent
        
        try:
            cmd = ["docker-compose", "-f", str(compose_file), "ps", "--format", "json"]
            
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse JSON output
                import json
                services = {}
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            service_info = json.loads(line)
                            services[service_info['Service']] = service_info['State']
                        except json.JSONDecodeError:
                            continue
                return services
            else:
                logger.error(f"Failed to get compose services status: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting compose services status: {e}")
            return {}
    
    def health_check_service(self, service_url: str, timeout: int = 5) -> bool:
        """Check if a service is healthy by making an HTTP request."""
        try:
            import httpx
            
            response = httpx.get(service_url, timeout=timeout)
            return response.status_code == 200
            
        except Exception as e:
            logger.debug(f"Health check failed for {service_url}: {e}")
            return False
    
    async def wait_for_service_health(self, 
                                    service_url: str, 
                                    max_attempts: int = 30,
                                    delay: float = 2.0) -> bool:
        """Wait for a service to become healthy."""
        for attempt in range(max_attempts):
            if self.health_check_service(service_url):
                logger.info(f"Service {service_url} is healthy")
                return True
            
            logger.debug(f"Service {service_url} not ready, attempt {attempt + 1}/{max_attempts}")
            await asyncio.sleep(delay)
        
        logger.error(f"Service {service_url} failed to become healthy after {max_attempts} attempts")
        return False
    
    def cleanup_containers(self, filter_labels: Optional[Dict[str, str]] = None) -> int:
        """Clean up stopped containers."""
        if not self.is_docker_available():
            return 0
        
        try:
            containers = self.client.containers.list(all=True, filters={"status": "exited"})
            
            if filter_labels:
                containers = [c for c in containers if all(
                    c.labels.get(k) == v for k, v in filter_labels.items()
                )]
            
            cleaned_count = 0
            for container in containers:
                try:
                    container.remove()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove container {container.name}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} containers")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up containers: {e}")
            return 0
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get system resource usage."""
        if not self.is_docker_available():
            return {"available": False}
        
        try:
            # Get overall system info
            info = self.client.info()
            
            # Get running containers stats
            containers = self.client.containers.list()
            total_memory = 0
            total_cpu = 0
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    memory_stats = stats.get('memory_stats', {})
                    cpu_stats = stats.get('cpu_stats', {})
                    
                    # Memory usage
                    if 'usage' in memory_stats:
                        total_memory += memory_stats['usage']
                    
                    # CPU usage (simplified)
                    if 'cpu_usage' in cpu_stats:
                        total_cpu += cpu_stats['cpu_usage'].get('total_usage', 0)
                        
                except Exception as e:
                    logger.debug(f"Error getting stats for container {container.name}: {e}")
            
            return {
                "available": True,
                "total_memory": info.get("MemTotal", 0),
                "used_memory": total_memory,
                "containers_running": len(containers),
                "system_time": info.get("SystemTime", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {"available": False, "error": str(e)}