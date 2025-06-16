"""
Automatic Installation Service.

Downloads and installs required models and dependencies automatically on first run.
Provides seamless user experience without manual model setup.
"""

import logging
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import sys

from huggingface_hub import hf_hub_download, list_repo_files
from ..config import settings

logger = logging.getLogger(__name__)


class AutoInstaller:
    """Automatic installation and setup service."""
    
    def __init__(self):
        self.models_dir = Path(settings.model_cache_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Essential models for core functionality
        self.essential_models = {
            "tts": {
                "required": True,
                "description": "Text-to-Speech (built-in edge-tts)",
                "check": self._check_tts_available
            },
            "whisper": {
                "required": False,
                "description": "OpenAI Whisper for speech recognition",
                "download": self._install_whisper,
                "check": self._check_whisper_available
            },
            "phi4": {
                "required": False, 
                "description": "Microsoft Phi-4 Multimodal",
                "download": self._install_phi4,
                "check": self._check_phi4_available
            }
        }
        
        # Optional dependencies that can be installed on demand
        self.optional_deps = {
            "biometric": {
                "packages": ["opencv-python>=4.8.0", "face-recognition>=1.3.0"],
                "description": "Biometric authentication (face recognition)"
            },
            "advanced_audio": {
                "packages": ["speechbrain>=0.5.16", "scipy>=1.11.0"],
                "description": "Advanced voice recognition"
            },
            "ml_models": {
                "packages": ["torch>=2.1.0", "transformers>=4.40.0"],
                "description": "Machine learning model support"
            }
        }
    
    async def run_initial_setup(self) -> Dict[str, Any]:
        """Run initial setup on first launch."""
        logger.info("Running initial Tektra setup...")
        
        results = {
            "success": True,
            "models_installed": [],
            "models_failed": [],
            "optional_available": {},
            "setup_time": 0
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
            
            # Check optional dependencies availability
            for dep_name, config in self.optional_deps.items():
                available = await self._check_packages_available(config["packages"])
                results["optional_available"][dep_name] = available
                if available:
                    logger.info(f"✓ {config['description']} - available")
                else:
                    logger.info(f"○ {config['description']} - install with: pip install {' '.join(config['packages'])}")
            
            results["setup_time"] = time.time() - start_time
            logger.info(f"Setup completed in {results['setup_time']:.1f}s")
            
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
                sys.executable, "-m", "pip", "install", "openai-whisper",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
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
            essential_files = [
                "config.json",
                "tokenizer.json", 
                "tokenizer_config.json"
            ]
            
            # Download essential files first (small, quick)
            for filename in essential_files:
                try:
                    local_path = hf_hub_download(
                        repo_id=model_repo,
                        filename=filename,
                        cache_dir=str(self.models_dir)
                    )
                    logger.info(f"✓ Downloaded {filename}")
                except Exception as e:
                    logger.warning(f"Could not download {filename}: {e}")
            
            # Model weights will be downloaded on-demand during first use
            logger.info("✓ Phi-4 model configuration ready (weights will download on first use)")
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
        """Install an optional dependency group."""
        if dep_name not in self.optional_deps:
            return {"success": False, "error": f"Unknown dependency: {dep_name}"}
        
        config = self.optional_deps[dep_name]
        packages = config["packages"]
        
        try:
            logger.info(f"Installing {config['description']}...")
            
            # Install packages
            cmd = [sys.executable, "-m", "pip", "install"] + packages
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✓ {config['description']} installed successfully")
                return {"success": True, "packages": packages}
            else:
                error_msg = stderr.decode() if stderr else "Installation failed"
                logger.warning(f"Installation failed: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Failed to install {dep_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_installation_status(self) -> Dict[str, Any]:
        """Get current installation status."""
        return {
            "models_dir": str(self.models_dir),
            "essential_models": list(self.essential_models.keys()),
            "optional_deps": list(self.optional_deps.keys()),
            "setup_required": not (self.models_dir / ".tektra_setup_complete").exists()
        }
    
    def mark_setup_complete(self):
        """Mark initial setup as complete."""
        setup_file = self.models_dir / ".tektra_setup_complete"
        setup_file.write_text(f"Setup completed at {asyncio.get_event_loop().time()}")


# Global installer instance
auto_installer = AutoInstaller()