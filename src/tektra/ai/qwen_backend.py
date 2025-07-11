"""
Qwen Backend for Analytical and Vision Tasks

This module provides comprehensive Qwen model integration for:
- Complex reasoning and analytical queries
- Vision analysis with Qwen-VL models
- Multimodal processing (text + image)
- High-performance inference with optimization
- Memory-efficient model management
"""

import asyncio
import gc
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import psutil
import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    pipeline,
)

# Import memory system
from ..memory import MemoryContext, TektraMemoryManager


class QwenModelConfig:
    """Configuration for Qwen model loading and optimization."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_bits: int = 8,
        max_memory_gb: float = 8.0,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        use_flash_attention: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.max_memory_gb = max_memory_gb
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_flash_attention = use_flash_attention
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    def get_quantization_config(self) -> BitsAndBytesConfig | None:
        """
        Get quantization configuration for memory optimization.
        
        Attempts to configure quantization with robust fallback handling.
        Returns None if quantization not supported or fails validation.
        """
        try:
            # Check if quantization is enabled in configuration
            if self.quantization_bits not in [4, 8]:
                logger.info(f"Quantization disabled: unsupported bits={self.quantization_bits}")
                return None
            
            # Verify bitsandbytes availability and compatibility
            if not self._validate_quantization_support():
                logger.warning("Quantization not supported on this system, using full precision")
                return None
            
            # Check available memory
            available_memory = self._get_available_memory_gb()
            if available_memory < self._get_required_memory_gb(with_quantization=True):
                logger.warning(f"Insufficient memory for quantization: {available_memory:.1f}GB available")
                return None
            
            # Configure 8-bit quantization
            if self.quantization_bits == 8:
                logger.info("Configuring 8-bit quantization for memory optimization")
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_threshold=6.0  # Outlier threshold
                )
            
            # Configure 4-bit quantization
            elif self.quantization_bits == 4:
                logger.info("Configuring 4-bit quantization for maximum memory efficiency")
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,  # Use explicit dtype
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            return None
            
        except ImportError as e:
            logger.warning(f"BitsAndBytesConfig not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error configuring quantization: {e}")
            return None

    def _validate_quantization_support(self) -> bool:
        """
        Validate that quantization is supported on this system.
        
        Returns:
            bool: True if quantization is supported
        """
        try:
            # Check if bitsandbytes is available
            import bitsandbytes as bnb
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.info("CUDA not available, quantization requires GPU")
                return False
            
            # Check PyTorch version compatibility
            torch_version = torch.__version__.split('+')[0]  # Remove CUDA suffix
            major, minor = map(int, torch_version.split('.')[:2])
            
            if major < 2 or (major == 2 and minor < 0):
                logger.warning(f"PyTorch {torch_version} may not support latest quantization features")
            
            # Test bitsandbytes functionality
            try:
                # Simple test to verify bitsandbytes works
                test_tensor = torch.randn(10, 10).cuda()
                _ = bnb.nn.Linear8bitLt(10, 5)
                logger.debug("BitsAndBytesConfig validation successful")
                return True
                
            except Exception as e:
                logger.warning(f"BitsAndBytesConfig test failed: {e}")
                return False
                
        except ImportError:
            logger.info("BitsAndBytesConfig not installed, using full precision")
            return False
        except Exception as e:
            logger.warning(f"Quantization validation error: {e}")
            return False

    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        try:
            # Check GPU memory if CUDA available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_free = torch.cuda.memory_stats()['reserved_bytes.all.freed'] / (1024**3)
                logger.debug(f"GPU memory: {gpu_memory:.1f}GB total, ~{gpu_free:.1f}GB available")
                return gpu_memory * 0.8  # Use 80% of GPU memory as safety margin
            
            # Fallback to system RAM
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            logger.debug(f"System RAM: {available_gb:.1f}GB available")
            return available_gb
            
        except Exception as e:
            logger.warning(f"Could not determine available memory: {e}")
            return 8.0  # Conservative fallback

    def _get_required_memory_gb(self, with_quantization: bool = True) -> float:
        """
        Estimate required memory for model based on configuration.
        
        Args:
            with_quantization: Whether quantization will be used
            
        Returns:
            float: Estimated memory requirement in GB
        """
        # Base model size estimates for common Qwen models
        model_sizes = {
            "qwen/qwen2.5-vl-7b": 7.0,  # 7B parameters
            "qwen/qwen2.5-vl-3b": 3.0,  # 3B parameters
            "qwen/qwen2.5-vl-1.5b": 1.5, # 1.5B parameters
        }
        
        # Get base model size
        model_name_lower = self.model_name.lower()
        base_size = 7.0  # Default assumption for 7B model
        
        for name, size in model_sizes.items():
            if name in model_name_lower:
                base_size = size
                break
        
        # Calculate memory requirements
        if with_quantization:
            if self.quantization_bits == 4:
                # 4-bit: ~0.5-0.7 GB per billion parameters
                return base_size * 0.6 + 2.0  # Add overhead
            elif self.quantization_bits == 8:
                # 8-bit: ~1.0-1.2 GB per billion parameters  
                return base_size * 1.1 + 2.0  # Add overhead
        
        # Full precision: ~2.0-2.5 GB per billion parameters
        return base_size * 2.2 + 3.0  # Add overhead

    def get_quantization_status(self) -> dict[str, Any]:
        """
        Get detailed quantization status and recommendations.
        
        Returns:
            dict: Quantization status information for user feedback
        """
        status = {
            "enabled": False,
            "method": None,
            "memory_savings": "0%",
            "performance_impact": "none",
            "recommendation": "",
            "system_support": False,
            "memory_available": self._get_available_memory_gb(),
            "memory_required_full": self._get_required_memory_gb(with_quantization=False),
            "memory_required_quantized": self._get_required_memory_gb(with_quantization=True)
        }
        
        # Check system support
        status["system_support"] = self._validate_quantization_support()
        
        # Check if quantization is configured
        if self.quantization_bits in [4, 8]:
            quantization_config = self.get_quantization_config()
            
            if quantization_config is not None:
                status["enabled"] = True
                status["method"] = f"{self.quantization_bits}-bit"
                
                # Calculate memory savings
                full_memory = status["memory_required_full"]
                quantized_memory = status["memory_required_quantized"]
                savings_percent = int((1 - quantized_memory / full_memory) * 100)
                status["memory_savings"] = f"{savings_percent}%"
                
                # Performance impact
                if self.quantization_bits == 4:
                    status["performance_impact"] = "minimal (4-bit)"
                    status["recommendation"] = "Maximum memory efficiency with minimal quality loss"
                elif self.quantization_bits == 8:
                    status["performance_impact"] = "negligible (8-bit)"
                    status["recommendation"] = "Good balance of memory savings and quality"
            else:
                status["recommendation"] = self._get_quantization_recommendation()
        else:
            status["recommendation"] = "Full precision selected - highest quality but requires more memory"
        
        return status

    def _get_quantization_recommendation(self) -> str:
        """Get recommendation for quantization configuration."""
        available = self._get_available_memory_gb()
        required_full = self._get_required_memory_gb(with_quantization=False)
        required_8bit = self._get_required_memory_gb(with_quantization=True)
        
        if not self._validate_quantization_support():
            return "Quantization not supported - install bitsandbytes and ensure CUDA is available"
        
        if available < required_8bit:
            return f"Insufficient memory ({available:.1f}GB available, {required_8bit:.1f}GB needed for 8-bit)"
        elif available < required_full:
            return f"Recommend 8-bit quantization ({required_8bit:.1f}GB vs {required_full:.1f}GB full precision)"
        else:
            return f"Sufficient memory available - quantization optional for {available:.1f}GB system"

    def get_generation_config(self) -> dict[str, Any]:
        """Get text generation configuration."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "pad_token_id": None,  # Will be set after tokenizer loading
            "eos_token_id": None,  # Will be set after tokenizer loading
        }


class QwenPerformanceMonitor:
    """Monitor performance metrics for Qwen inference."""

    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.token_counts = []
        self.last_inference_time = 0
        self.total_inferences = 0

    def record_inference(self, duration: float, input_tokens: int, output_tokens: int):
        """Record inference performance metrics."""
        self.inference_times.append(duration)
        self.token_counts.append({"input": input_tokens, "output": output_tokens})
        self.last_inference_time = duration
        self.total_inferences += 1

        # Record memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)

        # Keep only recent metrics (last 100 inferences)
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
            self.memory_usage = self.memory_usage[-100:]
            self.token_counts = self.token_counts[-100:]

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        if not self.inference_times:
            return {"total_inferences": 0}

        avg_time = sum(self.inference_times) / len(self.inference_times)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)

        total_input_tokens = sum(t["input"] for t in self.token_counts)
        total_output_tokens = sum(t["output"] for t in self.token_counts)

        tokens_per_second = (
            total_output_tokens / sum(self.inference_times)
            if self.inference_times
            else 0
        )

        return {
            "total_inferences": self.total_inferences,
            "avg_inference_time": avg_time,
            "last_inference_time": self.last_inference_time,
            "avg_memory_mb": avg_memory,
            "current_memory_mb": self.memory_usage[-1] if self.memory_usage else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tokens_per_second": tokens_per_second,
            "min_inference_time": min(self.inference_times),
            "max_inference_time": max(self.inference_times),
        }


class QwenBackend:
    """
    Comprehensive Qwen backend for analytical and vision tasks.

    Provides high-performance inference for:
    - Complex analytical queries with Qwen text models
    - Vision analysis with Qwen-VL multimodal models
    - Optimized memory usage with quantization
    - Async processing for GUI integration
    """

    def __init__(self, config: QwenModelConfig | None = None):
        """
        Initialize Qwen backend.

        Args:
            config: Model configuration, uses defaults if None
        """
        self.config = config or QwenModelConfig()

        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None  # For vision models
        self.device = None

        # State management
        self.is_initialized = False
        self.is_vision_model = False
        self.model_loading_progress = 0.0
        self.loading_status = "Not started"

        # Performance monitoring
        self.performance_monitor = QwenPerformanceMonitor()

        # Generation pipeline
        self.text_generator = None

        # Memory management
        self.max_memory_bytes = int(self.config.max_memory_gb * 1024 * 1024 * 1024)

        # Memory system integration
        self.memory_manager: TektraMemoryManager | None = None
        self.memory_enabled = False

        logger.info(f"Qwen backend initialized with model: {self.config.model_name}")

    async def initialize(self, progress_callback: Callable | None = None) -> bool:
        """
        Initialize the Qwen model asynchronously.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized:
            logger.info("Qwen backend already initialized")
            return True

        try:
            logger.info(f"Initializing Qwen model: {self.config.model_name}")

            # Update progress
            await self._update_progress(
                0, "Starting model initialization...", progress_callback
            )

            # Detect device
            self.device = self._detect_optimal_device()
            logger.info(f"Using device: {self.device}")

            # Check if this is a vision model
            self.is_vision_model = (
                "VL" in self.config.model_name
                or "vision" in self.config.model_name.lower()
            )

            if self.is_vision_model:
                success = await self._initialize_vision_model(progress_callback)
            else:
                success = await self._initialize_text_model(progress_callback)

            if success:
                self.is_initialized = True
                await self._update_progress(
                    100, "Model initialization complete!", progress_callback
                )
                logger.success("Qwen model initialized successfully")

                # Log memory usage
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"Current memory usage: {memory_mb:.1f} MB")

            return success

        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {e}")
            await self._update_progress(
                0, f"Initialization failed: {e}", progress_callback
            )
            return False

    async def _initialize_text_model(
        self, progress_callback: Callable | None = None
    ) -> bool:
        """Initialize text-only Qwen model."""
        try:
            await self._update_progress(20, "Loading tokenizer...", progress_callback)

            # Use a simpler text-only model name if we're falling back from vision
            text_model_name = self.config.model_name
            if "VL" in text_model_name:
                # Try to use a text-only version
                text_model_name = text_model_name.replace("-VL", "")
                logger.info(f"Using text-only model: {text_model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                text_model_name, trust_remote_code=True, use_fast=True
            )

            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            await self._update_progress(50, "Loading model...", progress_callback)

            # Configure quantization
            quantization_config = self.config.get_quantization_config()

            # Load model with optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                text_model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                attn_implementation=(
                    "flash_attention_2" if self.config.use_flash_attention else None
                ),
                low_cpu_mem_usage=True,
            )

            await self._update_progress(
                80, "Setting up generation pipeline...", progress_callback
            )

            # Create generation pipeline
            generation_config = self.config.get_generation_config()
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            generation_config["eos_token_id"] = self.tokenizer.eos_token_id

            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                **generation_config,
            )

            # Mark as text-only mode
            self.is_vision_model = False
            logger.info("Successfully initialized text-only Qwen model")

            return True

        except Exception as e:
            logger.error(f"Error initializing text model: {e}")
            return False

    async def _initialize_vision_model(
        self, progress_callback: Callable | None = None
    ) -> bool:
        """Initialize vision-capable Qwen-VL model."""
        try:
            await self._update_progress(
                20, "Loading vision processor...", progress_callback
            )

            # Check for torchvision dependency
            try:
                import torchvision  # noqa: F401

                logger.info("Torchvision available for vision processing")
            except ImportError:
                logger.warning(
                    "Torchvision not available - falling back to text-only mode"
                )
                return await self._initialize_text_model(progress_callback)

            # Load processor for vision model
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )

            await self._update_progress(
                50, "Loading vision model...", progress_callback
            )

            # Configure quantization
            quantization_config = self.config.get_quantization_config()

            # Load vision model - use correct class based on model version
            if "2.5-VL" in self.config.model_name:
                # Use Qwen2_5_VLForConditionalGeneration for Qwen2.5-VL models
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    torch_dtype=self.config.torch_dtype,
                    device_map=self.config.device_map,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                # Use Qwen2VLForConditionalGeneration for Qwen2-VL models
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    torch_dtype=self.config.torch_dtype,
                    device_map=self.config.device_map,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            await self._update_progress(
                80, "Setting up vision pipeline...", progress_callback
            )

            # The processor already includes the tokenizer for vision models
            self.tokenizer = self.processor.tokenizer

            return True

        except Exception as e:
            logger.error(f"Error initializing vision model: {e}")
            # Try to fall back to text-only mode
            logger.info("Attempting fallback to text-only mode")
            return await self._initialize_text_model(progress_callback)

    def _detect_optimal_device(self) -> str:
        """Detect the optimal device for model inference."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA available, GPU memory: {gpu_memory:.1f} GB")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
        else:
            logger.info("Using CPU for inference")
            return "cpu"

    async def enable_memory(self, memory_manager: TektraMemoryManager):
        """Enable memory support for context-aware generation."""
        self.memory_manager = memory_manager
        self.memory_enabled = True
        logger.info("Memory-enhanced generation enabled")

    async def _get_memory_context(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Get relevant memory context for the prompt."""
        if not self.memory_enabled or not self.memory_manager:
            return ""

        try:
            # Extract context information
            user_id = context.get("user_id") if context else None
            agent_id = context.get("agent_id") if context else None
            session_id = context.get("session_id") if context else None

            # Search for relevant memories
            memory_context = MemoryContext(
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                query=prompt,
                max_results=5,
                min_relevance=0.3,
                time_window_hours=24,  # Last 24 hours
            )

            search_result = await self.memory_manager.search_memories(memory_context)

            if search_result.entries:
                # Format memory context
                memory_text = "\n--- Relevant Context ---\n"
                for entry in search_result.entries:
                    memory_text += f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M')}] {entry.content}\n"
                memory_text += "--- End Context ---\n"

                logger.debug(
                    f"Added {len(search_result.entries)} memory entries to context"
                )
                return memory_text

        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")

        return ""

    async def _save_to_memory(
        self, prompt: str, response: str, context: dict[str, Any] | None = None
    ):
        """Save conversation to memory."""
        if not self.memory_enabled or not self.memory_manager:
            return

        try:
            # Extract context information
            user_id = context.get("user_id") if context else None
            agent_id = context.get("agent_id") if context else None
            session_id = context.get("session_id") if context else None

            # Save conversation to memory
            await self.memory_manager.add_conversation(
                user_message=prompt,
                assistant_response=response,
                user_id=user_id or "unknown",
                session_id=session_id or "default",
                agent_id=agent_id,
            )

            logger.debug("Saved conversation to memory")

        except Exception as e:
            logger.warning(f"Failed to save conversation to memory: {e}")

    async def generate_response(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """
        Generate text response for analytical queries.

        Args:
            prompt: Input text prompt
            context: Optional context information

        Returns:
            str: Generated response text
        """
        if not self.is_initialized:
            raise RuntimeError("Qwen backend not initialized")

        if self.is_vision_model:
            logger.warning(
                "Using vision model for text-only query - consider using process_vision_query"
            )

        start_time = time.time()

        try:
            # Get memory context first
            memory_context = await self._get_memory_context(prompt, context)

            # Prepare prompt with context if provided
            formatted_prompt = self._format_prompt(prompt, context, memory_context)

            # Count input tokens
            input_tokens = len(self.tokenizer.encode(formatted_prompt))

            logger.debug(f"Generating response for prompt: {formatted_prompt[:100]}...")

            if self.is_vision_model:
                # Use processor for vision model
                response = await self._generate_with_vision_model(formatted_prompt)
            else:
                # Use text generator pipeline
                response = await self._generate_with_text_model(formatted_prompt)

            # Count output tokens
            output_tokens = len(self.tokenizer.encode(response))

            # Record performance
            duration = time.time() - start_time
            self.performance_monitor.record_inference(
                duration, input_tokens, output_tokens
            )

            # Save to memory if enabled
            await self._save_to_memory(prompt, response, context)

            logger.info(
                f"Generated response in {duration:.2f}s ({output_tokens} tokens)"
            )
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def process_vision_query(
        self, prompt: str, image: Image.Image | np.ndarray | str
    ) -> str:
        """
        Process vision query with image input.

        Args:
            prompt: Text prompt describing the task
            image: Image data (PIL Image, numpy array, or file path)

        Returns:
            str: Analysis result
        """
        if not self.is_initialized:
            raise RuntimeError("Qwen backend not initialized")

        if not self.is_vision_model:
            raise RuntimeError("Vision processing requires a vision-capable model")

        start_time = time.time()

        try:
            # Process image input
            processed_image = self._process_image_input(image)

            # Format prompt for vision task
            vision_prompt = self._format_vision_prompt(prompt)

            logger.debug(f"Processing vision query: {vision_prompt[:100]}...")

            # Prepare inputs
            inputs = self.processor(
                text=[vision_prompt],
                images=[processed_image],
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            if self.device != "cpu":
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract only the new generated text
            response = response[len(vision_prompt) :].strip()

            # Record performance
            duration = time.time() - start_time
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            self.performance_monitor.record_inference(
                duration, input_tokens, output_tokens
            )

            logger.info(f"Processed vision query in {duration:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error processing vision query: {e}")
            raise

    async def _generate_with_text_model(self, prompt: str) -> str:
        """Generate response using text model pipeline."""
        try:
            # Use pipeline for generation
            result = self.text_generator(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
            )

            return result[0]["generated_text"].strip()

        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            raise

    async def _generate_with_vision_model(self, prompt: str) -> str:
        """Generate response using vision model (text-only mode)."""
        try:
            # Tokenize input
            inputs = self.processor(text=[prompt], return_tensors="pt", padding=True)

            # Move to device
            if self.device != "cpu":
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract only the new generated text
            response = response[len(prompt) :].strip()

            return response

        except Exception as e:
            logger.error(f"Error in vision model text generation: {e}")
            raise

    def _process_image_input(
        self, image: Image.Image | np.ndarray | str
    ) -> Image.Image:
        """Process various image input formats to PIL Image."""
        if isinstance(image, str):
            # File path
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Numpy array
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # PIL Image
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _format_prompt(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        memory_context: str = "",
    ) -> str:
        """Format prompt with context, memory, and instructions."""
        formatted_prompt = prompt

        # Add memory context if available
        if memory_context:
            formatted_prompt = f"{memory_context}\n{formatted_prompt}"

        # Add regular context if provided
        if context:
            context_str = "\n".join(
                [
                    f"{k}: {v}"
                    for k, v in context.items()
                    if v and k not in ["user_id", "agent_id", "session_id"]
                ]
            )
            if context_str:
                formatted_prompt = (
                    f"Context:\n{context_str}\n\nQuery: {formatted_prompt}"
                )

        # Add instruction prefix for better performance
        return f"<|im_start|>system\nYou are a helpful AI assistant that provides detailed, accurate, and analytical responses.<|im_end|>\n<|im_start|>user\n{formatted_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _format_vision_prompt(self, prompt: str) -> str:
        """Format prompt for vision tasks."""
        return f"<|im_start|>system\nYou are a helpful AI assistant with vision capabilities. Analyze the provided image and respond to the user's request.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    async def _update_progress(
        self, progress: float, status: str, callback: Callable | None
    ):
        """Update loading progress."""
        self.model_loading_progress = progress
        self.loading_status = status

        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress, status)
            else:
                callback(progress, status)

    def optimize_memory(self):
        """Optimize memory usage by clearing caches and garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        logger.debug("Memory optimization completed")

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            "model_name": self.config.model_name,
            "is_initialized": self.is_initialized,
            "is_vision_model": self.is_vision_model,
            "device": str(self.device),
            "quantization_bits": self.config.quantization_bits,
            "loading_progress": self.model_loading_progress,
            "loading_status": self.loading_status,
            "performance_stats": self.performance_monitor.get_stats(),
        }

        if self.is_initialized:
            # Add memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            info["current_memory_mb"] = memory_mb

            # Add model parameter count if available
            if self.model:
                try:
                    param_count = sum(p.numel() for p in self.model.parameters())
                    info["parameter_count"] = param_count
                    info["parameter_count_millions"] = param_count / 1_000_000
                except Exception as e:
                    logger.debug(f"Could not retrieve model parameter count: {e}")

        return info

    async def cleanup(self):
        """Cleanup model resources."""
        logger.info("Cleaning up Qwen backend...")

        # Clear model references
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.text_generator = None

        # Clear GPU memory
        self.optimize_memory()

        self.is_initialized = False
        logger.info("Qwen backend cleanup complete")
