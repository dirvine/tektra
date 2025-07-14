"""
Simple LLM Backend for Tektra AI Assistant

This module provides a lightweight, working LLM implementation using a small,
fast model that can actually generate responses. This replaces the complex
Qwen backend with something that works reliably.
"""

import asyncio
import concurrent.futures
import gc
import time
from typing import Any, Optional, List, Dict, Callable

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline
)

from ..utils.transformers_progress import TransformersProgressMonitor
from ..utils.simple_progress import SimpleProgressIndicator


class SimpleLLM:
    """
    Simple, working LLM backend using a small, fast model.
    
    This uses Microsoft's Phi-3-mini model which is:
    - Small enough to run on most hardware (2.4GB)
    - Fast inference
    - Good quality responses
    - Reliable and well-supported
    """

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize the Simple LLM.
        
        Args:
            model_name: The model to use (default: Phi-3-mini)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 2048
        self.generation_config = None
        
        logger.info(f"Simple LLM initialized with model: {model_name}")
        logger.info(f"Device: {self.device}")

    async def initialize(self, progress_callback: Optional[callable] = None) -> bool:
        """
        Initialize the model and tokenizer.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing Simple LLM...")
            
            if progress_callback:
                await progress_callback(0.1, "Loading tokenizer...")
            
            # Load tokenizer in background thread
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.tokenizer = await loop.run_in_executor(
                    executor,
                    lambda: AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        padding_side="left"
                    )
                )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Tokenizer loaded successfully")
            
            if progress_callback:
                await progress_callback(0.3, "Preparing to download model...")
            
            # Check if model is already cached
            from pathlib import Path
            import os
            
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            model_cache = cache_dir / f"models--{self.model_name.replace('/', '--')}"
            is_cached = model_cache.exists() and any(model_cache.glob("blobs/*"))
            
            # Use simple progress indicator for better UX
            progress_indicator = SimpleProgressIndicator(progress_callback)
            
            # Start progress in background
            initial_message = "Loading model from cache" if is_cached else "Downloading model files"
            progress_task = asyncio.create_task(
                progress_indicator.run_with_progress(
                    initial_message,
                    duration=300,
                    is_cached=is_cached
                )
            )
            
            try:
                # Load model with appropriate settings
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "device_map": "auto" if self.device == "cuda" else None,
                    "low_cpu_mem_usage": True,
                }
                
                # Show we're downloading
                logger.info("Starting model download - this may take a few minutes...")
                
                # Load model in background thread to avoid blocking UI
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    self.model = await loop.run_in_executor(
                        executor,
                        lambda: AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **model_kwargs
                        )
                    )
                    
                logger.info("Model download and loading complete!")
                    
            finally:
                # Stop progress
                progress_indicator.stop()
                try:
                    await asyncio.wait_for(progress_task, timeout=1.0)
                except asyncio.TimeoutError:
                    progress_task.cancel()
            
            if progress_callback:
                await progress_callback(0.7, "Setting up generation config...")
            
            # Configure generation settings
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
            
            if progress_callback:
                await progress_callback(0.9, "Creating pipeline...")
            
            # Create pipeline for easier inference in background thread
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.pipeline = await loop.run_in_executor(
                    executor,
                    lambda: pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0 if self.device == "cuda" else -1,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        trust_remote_code=True,
                        generation_config=self.generation_config
                    )
                )
            
            if progress_callback:
                await progress_callback(1.0, "Model ready!")
            
            self.is_initialized = True
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
            logger.success("Simple LLM initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Simple LLM: {e}")
            self.is_initialized = False
            return False

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            context: Optional conversation context
            
        Returns:
            str: The generated response
        """
        if not self.is_initialized:
            return "Sorry, I'm not initialized yet. Please wait a moment."
        
        try:
            # Format prompt with context if provided
            formatted_prompt = self._format_prompt(prompt, context)
            
            logger.debug(f"Generating response for prompt: {formatted_prompt[:100]}...")
            
            # Generate response
            start_time = time.time()
            
            # Update generation config for this request
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
            
            # Generate with pipeline
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                generation_config=generation_config,
                return_full_text=False  # Only return new tokens
            )
            
            response = outputs[0]["generated_text"].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            generation_time = time.time() - start_time
            tokens_per_second = max_tokens / generation_time if generation_time > 0 else 0
            
            logger.debug(f"Response generated in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
            logger.debug(f"Response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error generating a response: {e}"

    def _format_prompt(self, prompt: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format the prompt with context for better responses.
        
        Args:
            prompt: The user's prompt
            context: Optional conversation context
            
        Returns:
            str: The formatted prompt
        """
        # System message
        system_message = (
            "You are Tektra, a helpful AI assistant. "
            "You are knowledgeable, friendly, and concise. "
            "Provide helpful and accurate responses."
        )
        
        # Build conversation context
        conversation = [f"System: {system_message}"]
        
        if context:
            # Add recent conversation history (last 5 messages)
            for message in context[-5:]:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user":
                    conversation.append(f"User: {content}")
                elif role == "assistant":
                    conversation.append(f"Assistant: {content}")
        
        # Add current prompt
        conversation.append(f"User: {prompt}")
        conversation.append("Assistant:")
        
        return "\n".join(conversation)

    def _clean_response(self, response: str) -> str:
        """
        Clean up the generated response.
        
        Args:
            response: The raw response from the model
            
        Returns:
            str: The cleaned response
        """
        # Remove any system tokens or formatting
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|im_start|>", "")
        response = response.replace("Assistant:", "")
        response = response.replace("User:", "")
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        # Ensure response ends properly
        if response and not response.endswith(('.', '!', '?')):
            # Find the last complete sentence
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response.strip()

    async def process_message(
        self,
        message: str,
        context: Optional[dict] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process a message and generate a response.
        
        Args:
            message: The user's message
            context: Optional context information
            conversation_history: Optional conversation history
            
        Returns:
            str: The response
        """
        try:
            # Extract generation parameters from context
            temperature = 0.7
            max_tokens = 512
            
            if context:
                temperature = context.get("temperature", 0.7)
                max_tokens = context.get("max_tokens", 512)
            
            # Generate response
            response = await self.generate_response(
                prompt=message,
                max_tokens=max_tokens,
                temperature=temperature,
                context=conversation_history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error processing your message: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "max_length": self.max_length,
        }
        
        if self.is_initialized:
            info["loading_status"] = "Ready"
            
            # Add memory info if available
            if torch.cuda.is_available():
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
                info["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1024**3
        else:
            info["loading_status"] = "Not initialized"
        
        return info

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.pipeline:
                del self.pipeline
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.is_initialized = False
            logger.info("Simple LLM cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.is_initialized and self.model is not None

    async def test_generation(self) -> bool:
        """
        Test if the model can generate responses.
        
        Returns:
            bool: True if test successful
        """
        if not self.is_initialized:
            return False
        
        try:
            test_prompt = "Hello, how are you?"
            response = await self.generate_response(test_prompt, max_tokens=50)
            return len(response) > 0
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return False