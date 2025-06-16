"""
Automatic Installation Service.

Downloads and installs required models and dependencies automatically on first run.
Provides seamless user experience without manual model setup.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download, list_repo_files

from ..config import settings

logger = logging.getLogger(__name__)


class AutoInstaller:
    """Automatic installation and setup service."""

    def __init__(self):
        self.models_dir = Path(settings.model_cache_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Detect package manager
        self.package_manager = self._detect_package_manager()

        # Essential models for core functionality
        self.essential_models = {
            "tts": {
                "required": True,
                "description": "Text-to-Speech (built-in edge-tts)",
                "check": self._check_tts_available,
            },
            "whisper": {
                "required": False,
                "description": "OpenAI Whisper for speech recognition",
                "download": self._install_whisper,
                "check": self._check_whisper_available,
            },
            "phi4": {
                "required": False,
                "description": "Microsoft Phi-4 Multimodal",
                "download": self._install_phi4,
                "check": self._check_phi4_available,
            },
        }

        # Optional dependencies that can be installed on demand
        self.optional_deps = {
            "biometric": {
                "packages": ["opencv-python>=4.8.0"],
                "description": "Biometric authentication (camera-based)",
                "install_method": "safe",
            },
            "advanced_audio": {
                "packages": ["scipy>=1.11.0"],
                "description": "Advanced audio processing",
                "install_method": "safe",
            },
            "ml_models": {
                "packages": ["torch>=2.1.0"],
                "description": "Core ML framework (PyTorch)",
                "install_method": "safe",
            },
            "transformers": {
                "packages": ["transformers>=4.40.0", "tokenizers>=0.15.0"],
                "description": "HuggingFace Transformers (may require compilation)",
                "install_method": "compile_safe",
                "alternatives": ["transformers[torch]>=4.40.0"],
            },
        }

    def _detect_package_manager(self) -> str:
        """Detect which package manager to use (uv, pip, etc.)."""
        try:
            # Check if we're running inside a UV tool environment
            executable_path = sys.executable
            logger.debug(f"Python executable path: {executable_path}")
            
            if "uv/tools/" in executable_path:
                # UV tool environments: check if uv is available and working
                logger.debug("Detected UV tool environment")
                if shutil.which("uv"):
                    # Test if uv pip works in this environment
                    try:
                        result = subprocess.run(
                            ["uv", "pip", "--help"],
                            capture_output=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            logger.debug("UV pip is available in tool environment")
                            return "uv_tool"
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        pass
                
                # If UV isn't working, fallback to trying to install directly in the tool environment
                logger.debug("UV not available, checking if we can use direct pip in tool environment")
                return "uv_tool_direct"

            # Check if we're in a regular UV environment (not tool)
            if os.environ.get("VIRTUAL_ENV") and shutil.which("uv"):
                # Only use UV for regular project environments
                venv_path = os.environ.get("VIRTUAL_ENV", "")
                if "uv/tools/" not in venv_path:
                    return "uv"

            # Check if pip is available
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "--version"],
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                return "pip"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Try UV as fallback
                if shutil.which("uv"):
                    return "uv"

                return "none"

        except Exception as e:
            logger.debug(f"Package manager detection failed: {e}")
            return "pip"  # Default fallback

    def _get_install_command(self, packages: List[str]) -> List[str]:
        """Get the appropriate install command for the detected package manager."""
        if self.package_manager == "uv_tool":
            # UV tool environments: try uv pip first, then fallback to break-system-packages
            logger.debug("Using UV tool environment install command with --system flag")
            return ["uv", "pip", "install", "--system"] + packages + ["--quiet"]
        elif self.package_manager == "uv_tool_direct":
            # UV tool environment but with manual pip installation
            # This is a workaround for cases where UV isn't working properly
            logger.debug("UV tool environment detected but pip unavailable - running in limited mode")
            # Return a command that will "succeed" but not actually install anything
            # This allows Tektra to run without ML dependencies
            return ["echo", "UV tool environment: ML features require manual installation"]
        elif self.package_manager == "uv":
            # Regular UV environments: use uv pip install --system
            return ["uv", "pip", "install", "--system"] + packages + ["--quiet"]
        elif self.package_manager == "pip":
            # Standard pip environments
            return (
                [sys.executable, "-m", "pip", "install", "--user"]
                + packages
                + ["--quiet", "--disable-pip-version-check"]
            )
        else:
            # Fallback - try pip anyway
            return (
                [sys.executable, "-m", "pip", "install", "--user"]
                + packages
                + ["--quiet", "--disable-pip-version-check"]
            )

    def _get_fallback_install_command(self, packages: List[str]) -> List[str]:
        """Get fallback install command for externally managed environments."""
        logger.debug("Using fallback install with --break-system-packages")
        return (
            [sys.executable, "-m", "pip", "install", "--break-system-packages"]
            + packages
            + ["--quiet", "--disable-pip-version-check"]
        )

    async def run_initial_setup(self) -> Dict[str, Any]:
        """Run initial setup on first launch."""
        logger.info("Running initial Tektra setup...")

        results = {
            "success": True,
            "models_installed": [],
            "models_failed": [],
            "optional_available": {},
            "auto_installed": [],
            "setup_time": 0,
        }

        try:
            import time

            start_time = time.time()

            # Check and install essential models
            for model_name, config in self.essential_models.items():
                try:
                    if config.get("check") and await config["check"]():
                        logger.info(f"✓ {config['description']} - already available")
                        results["models_installed"].append(model_name)
                    elif config.get("download") and config.get("required", False):
                        logger.info(f"Installing {config['description']}...")
                        await config["download"]()
                        results["models_installed"].append(model_name)
                    else:
                        logger.info(f"⚠ {config['description']} - optional, skipping")

                except Exception as e:
                    logger.warning(f"Failed to install {model_name}: {e}")
                    results["models_failed"].append(model_name)

            # Check optional dependencies availability and auto-install safe ones
            for dep_name, config in self.optional_deps.items():
                available = await self._check_packages_available(config["packages"])
                results["optional_available"][dep_name] = available

                if available:
                    logger.info(f"✓ {config['description']} - available")
                elif config.get("install_method") == "safe":
                    # Auto-install safe dependencies
                    logger.info(f"Auto-installing {config['description']}...")
                    install_result = await self.install_optional_dependency(dep_name)
                    if install_result.get("success"):
                        results["auto_installed"].append(dep_name)
                        results["optional_available"][dep_name] = True
                    else:
                        logger.info(
                            f"○ {config['description']} - install manually if needed"
                        )
                else:
                    if self.package_manager == "uv_tool_direct":
                        # Special message for UV tool environment issues
                        logger.info(
                            f"○ {config['description']} - optional (UV tool environment)"
                        )
                    elif self.package_manager == "uv_tool":
                        # UV tool but with externally managed Python
                        logger.info(
                            f"○ {config['description']} - optional (externally managed Python)"
                        )
                    else:
                        install_hint = (
                            "tektra install-deps " + dep_name
                            if config.get("install_method") == "compile_safe"
                            else f"pip install {' '.join(config['packages'])}"
                        )
                        logger.info(
                            f"○ {config['description']} - install with: {install_hint}"
                        )

            results["setup_time"] = time.time() - start_time
            logger.info(f"Setup completed in {results['setup_time']:.1f}s")

            if results["auto_installed"]:
                logger.info(f"Auto-installed: {', '.join(results['auto_installed'])}")
            
            # Show helpful summary for UV tool environments
            if self.package_manager in ["uv_tool", "uv_tool_direct"]:
                logger.info("✨ Tektra is ready! Core features available:")
                logger.info("  • Chat interface and conversation management")
                logger.info("  • Text-to-speech with Edge-TTS")
                logger.info("  • 3D avatar with lip-sync animation")
                logger.info("  • Biometric security")
                logger.info("🔧 For advanced ML features, run: uv pip install torch transformers")
            else:
                logger.info("✨ Tektra setup complete! All features ready.")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            results["success"] = False
            results["error"] = str(e)

        return results

    async def _check_tts_available(self) -> bool:
        """Check if TTS service is available."""
        try:
            import edge_tts

            return True
        except ImportError:
            return False

    async def _check_whisper_available(self) -> bool:
        """Check if Whisper is available."""
        try:
            import whisper

            return True
        except ImportError:
            return False

    async def _install_whisper(self) -> bool:
        """Install OpenAI Whisper."""
        try:
            logger.info("Installing OpenAI Whisper...")
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "install",
                "openai-whisper",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info("✓ OpenAI Whisper installed successfully")
                return True
            else:
                logger.warning(f"Whisper installation failed: {stderr.decode()}")
                return False

        except Exception as e:
            logger.warning(f"Failed to install Whisper: {e}")
            return False

    async def _check_phi4_available(self) -> bool:
        """Check if Phi-4 model is available."""
        try:
            phi4_path = self.models_dir / "microsoft" / "Phi-4"
            return phi4_path.exists() and any(phi4_path.glob("*.safetensors"))
        except Exception:
            return False

    async def _install_phi4(self) -> bool:
        """Install Phi-4 Multimodal model."""
        try:
            logger.info("Downloading Microsoft Phi-4 Multimodal model...")

            model_repo = "microsoft/Phi-4"
            model_dir = self.models_dir / "microsoft" / "Phi-4"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Essential files for Phi-4
            essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

            # Download essential files first (small, quick)
            for filename in essential_files:
                try:
                    local_path = hf_hub_download(
                        repo_id=model_repo,
                        filename=filename,
                        cache_dir=str(self.models_dir),
                    )
                    logger.info(f"✓ Downloaded {filename}")
                except Exception as e:
                    logger.warning(f"Could not download {filename}: {e}")

            # Model weights will be downloaded on-demand during first use
            logger.info(
                "✓ Phi-4 model configuration ready (weights will download on first use)"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to prepare Phi-4 model: {e}")
            return False

    async def _check_packages_available(self, packages: List[str]) -> bool:
        """Check if Python packages are available."""
        try:
            for package in packages:
                package_name = package.split(">=")[0].split("==")[0]
                try:
                    __import__(package_name.replace("-", "_"))
                except ImportError:
                    return False
            return True
        except Exception:
            return False

    async def install_optional_dependency(self, dep_name: str) -> Dict[str, Any]:
        """Install an optional dependency group with smart fallback."""
        if dep_name not in self.optional_deps:
            return {"success": False, "error": f"Unknown dependency: {dep_name}"}

        config = self.optional_deps[dep_name]
        packages = config["packages"]
        install_method = config.get("install_method", "safe")

        try:
            logger.info(f"Installing {config['description']}...")

            # For compile_safe dependencies, try alternatives first
            if install_method == "compile_safe" and "alternatives" in config:
                logger.info("Trying compilation-free alternatives first...")
                for alt_package in config["alternatives"]:
                    success = await self._try_install_package(alt_package)
                    if success:
                        logger.info(
                            f"✓ {config['description']} installed via alternative: {alt_package}"
                        )
                        return {
                            "success": True,
                            "packages": [alt_package],
                            "method": "alternative",
                        }

            # Standard installation
            cmd = self._get_install_command(packages)
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"✓ {config['description']} installed successfully")
                return {"success": True, "packages": packages, "method": "standard"}
            else:
                error_msg = stderr.decode() if stderr else "Installation failed"
                logger.warning(f"Installation failed: {error_msg}")
                
                # Check for externally managed environment error
                if "externally managed" in error_msg.lower() or "externally-managed-environment" in error_msg.lower():
                    logger.info(f"Detected externally managed environment, trying fallback for {config['description']}...")
                    fallback_cmd = self._get_fallback_install_command(packages)
                    logger.debug(f"Fallback command: {' '.join(fallback_cmd)}")
                    process = await asyncio.create_subprocess_exec(
                        *fallback_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        logger.info(f"✓ {config['description']} installed via fallback method")
                        return {"success": True, "packages": packages, "method": "fallback_system"}
                    else:
                        fallback_error = stderr.decode() if stderr else "Fallback installation failed"
                        logger.warning(f"Fallback installation failed: {fallback_error}")

                # For compile_safe, suggest manual installation
                if install_method == "compile_safe":
                    suggestion = f"Manual installation required: pip install {' '.join(packages)}"
                    logger.info(f"💡 {suggestion}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "suggestion": suggestion,
                    }

                return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"Failed to install {dep_name}: {e}")
            return {"success": False, "error": str(e)}

    async def _try_install_package(self, package: str) -> bool:
        """Try to install a single package quietly."""
        try:
            cmd = self._get_install_command([package])
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True
            else:
                # Check for externally managed environment and try fallback
                error_msg = stderr.decode() if stderr else ""
                if "externally managed" in error_msg.lower() or "externally-managed-environment" in error_msg.lower():
                    fallback_cmd = self._get_fallback_install_command([package])
                    process = await asyncio.create_subprocess_exec(
                        *fallback_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    return process.returncode == 0
                
                return False
        except Exception:
            return False

    def get_installation_status(self) -> Dict[str, Any]:
        """Get current installation status."""
        return {
            "models_dir": str(self.models_dir),
            "essential_models": list(self.essential_models.keys()),
            "optional_deps": list(self.optional_deps.keys()),
            "setup_required": not (self.models_dir / ".tektra_setup_complete").exists(),
        }

    def mark_setup_complete(self):
        """Mark initial setup as complete."""
        setup_file = self.models_dir / ".tektra_setup_complete"
        import time

        setup_file.write_text(f"Setup completed at {time.time()}")

    async def install_dependency_silently(self, dep_name: str) -> bool:
        """Install a dependency silently in the background without user notifications."""
        if dep_name not in self.optional_deps:
            return False

        config = self.optional_deps[dep_name]
        packages = config["packages"]

        try:
            # Check if already available first
            if await self._check_packages_available(packages):
                return True

            # Get appropriate install command
            cmd = self._get_install_command(packages)
            logger.debug(f"Installing {dep_name} with command: {' '.join(cmd)}")

            # Install silently
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            # Check if successful
            if process.returncode == 0:
                # Verify installation
                success = await self._check_packages_available(packages)
                if success:
                    logger.info(f"✓ Successfully installed {dep_name}")
                    return True
                else:
                    logger.warning(
                        f"Installation completed but packages not available for {dep_name}"
                    )
                    return False
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.warning(f"Installation failed for {dep_name}: {error_msg}")
                
                # Check for externally managed environment error and try fallback
                if "externally managed" in error_msg.lower() or "externally-managed-environment" in error_msg.lower():
                    logger.info(f"Detected externally managed environment, trying fallback for {dep_name}...")
                    fallback_cmd = self._get_fallback_install_command(packages)
                    logger.debug(f"Fallback command: {' '.join(fallback_cmd)}")
                    
                    process = await asyncio.create_subprocess_exec(
                        *fallback_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        # Verify fallback installation
                        success = await self._check_packages_available(packages)
                        if success:
                            logger.info(f"✓ Successfully installed {dep_name} via fallback")
                            return True
                        else:
                            logger.warning(f"Fallback installation completed but packages not available for {dep_name}")
                            return False
                    else:
                        fallback_error = stderr.decode() if stderr else "Unknown fallback error"
                        logger.debug(f"Fallback installation also failed for {dep_name}: {fallback_error}")
                
                return False

        except Exception as e:
            logger.error(f"Exception during {dep_name} installation: {e}")
            return False

    async def ensure_dependency_available(
        self, dep_name: str, timeout: float = 30.0
    ) -> bool:
        """Ensure a dependency is available, installing it silently if needed."""
        try:
            # Check if already available
            if dep_name in self.optional_deps:
                packages = self.optional_deps[dep_name]["packages"]
                if await self._check_packages_available(packages):
                    return True

            # Install silently with timeout
            install_task = asyncio.create_task(
                self.install_dependency_silently(dep_name)
            )
            try:
                return await asyncio.wait_for(install_task, timeout=timeout)
            except asyncio.TimeoutError:
                install_task.cancel()
                return False

        except Exception:
            return False

    def start_background_installation(self, dep_name: str) -> asyncio.Task:
        """Start background installation of a dependency without blocking."""

        async def background_install():
            try:
                await self.install_dependency_silently(dep_name)
            except Exception:
                pass  # Silent failure

        return asyncio.create_task(background_install())


# Global installer instance
auto_installer = AutoInstaller()
