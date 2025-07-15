"""
Tektra Model Management System

This module provides comprehensive model management capabilities including
downloading, loading, optimization, performance monitoring, and easy model updating.
"""

from .model_manager import ModelManager
from .model_interface import (
    ModelInterface,
    ModelMetadata,
    ModelConfig,
    ModelRegistry,
    ModelFactory,
    ModelType,
    ModelStatus,
    QuantizationType,
    default_registry,
    default_factory
)
from .model_updater import (
    ModelUpdateManager,
    ModelUpdate,
    UpdateStatus,
    UpdatePriority,
    UpdateProgress
)

__all__ = [
    "ModelManager",
    "ModelInterface",
    "ModelMetadata",
    "ModelConfig",
    "ModelRegistry",
    "ModelFactory",
    "ModelType",
    "ModelStatus",
    "QuantizationType",
    "default_registry",
    "default_factory",
    "ModelUpdateManager",
    "ModelUpdate",
    "UpdateStatus",
    "UpdatePriority",
    "UpdateProgress",
]