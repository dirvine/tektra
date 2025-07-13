#!/usr/bin/env python3
"""
Performance Optimization System

Automatic performance optimization using profiling data, resource usage patterns,
and machine learning-based tuning.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with scikit-learn,numpy,scipy,loguru python optimizer.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn>=1.3.0",
#     "numpy>=1.24.0",
#     "scipy>=1.11.0",
#     "loguru>=0.7.0",
# ]
# ///

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from loguru import logger


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    CONSERVATIVE = "conservative"    # Minimal changes, high stability
    BALANCED = "balanced"            # Balance performance and stability
    AGGRESSIVE = "aggressive"        # Maximum performance, may impact stability
    ADAPTIVE = "adaptive"            # Learn and adapt over time
    CUSTOM = "custom"                # User-defined rules


@dataclass
class OptimizationParameter:
    """A tunable optimization parameter."""
    
    name: str
    current_value: float
    min_value: float
    max_value: float
    
    # Optimization properties
    step_size: float = 0.1           # How much to change per iteration
    is_integer: bool = False         # Whether parameter must be integer
    scale: str = "linear"            # linear, log, exponential
    
    # History
    value_history: deque = field(default_factory=lambda: deque(maxlen=100))
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Constraints
    constraints: List[Callable[[float], bool]] = field(default_factory=list)
    
    def is_valid(self, value: float) -> bool:
        """Check if a value is valid for this parameter."""
        if value < self.min_value or value > self.max_value:
            return False
        
        for constraint in self.constraints:
            if not constraint(value):
                return False
        
        return True
    
    def adjust_value(self, delta: float) -> float:
        """Adjust parameter value by delta."""
        new_value = self.current_value + delta
        
        # Apply bounds
        new_value = max(self.min_value, min(self.max_value, new_value))
        
        # Apply integer constraint
        if self.is_integer:
            new_value = round(new_value)
        
        return new_value
    
    def record_performance(self, value: float, performance: float) -> None:
        """Record a value and its performance."""
        self.value_history.append(value)
        self.performance_history.append(performance)


@dataclass
class OptimizationRule:
    """A rule for optimization decisions."""
    
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Dict[str, float]]
    priority: int = 0
    enabled: bool = True
    
    # Tracking
    times_applied: int = 0
    last_applied: Optional[float] = None
    
    def apply(self, context: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Apply rule if condition is met."""
        if not self.enabled:
            return None
        
        if self.condition(context):
            self.times_applied += 1
            self.last_applied = time.time()
            return self.action(context)
        
        return None


@dataclass
class OptimizationResult:
    """Results from an optimization run."""
    
    timestamp: float
    strategy: OptimizationStrategy
    
    # Parameter changes
    parameter_changes: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # name -> (old, new)
    
    # Performance impact
    performance_before: Dict[str, float] = field(default_factory=dict)
    performance_after: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)
    
    # Success metrics
    improvement_percent: float = 0.0
    stability_score: float = 1.0  # 0-1, higher is more stable
    
    @property
    def was_successful(self) -> bool:
        """Check if optimization was successful."""
        return self.improvement_percent > 0 and self.stability_score > 0.7


class PerformanceOptimizer:
    """
    Intelligent performance optimization system.
    
    Features:
    - Automatic parameter tuning
    - Rule-based optimization
    - Machine learning predictions
    - A/B testing support
    - Rollback capabilities
    - Performance regression detection
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.2,
        history_size: int = 1000,
        enable_ml: bool = True
    ):
        """
        Initialize performance optimizer.
        
        Args:
            strategy: Default optimization strategy
            learning_rate: How quickly to adjust parameters
            exploration_rate: Probability of exploring vs exploiting
            history_size: Size of performance history to maintain
            enable_ml: Enable machine learning predictions
        """
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.history_size = history_size
        self.enable_ml = enable_ml
        
        # Thread safety (must be initialized first)
        self._lock = threading.RLock()
        
        # Parameters to optimize
        self.parameters: Dict[str, OptimizationParameter] = {}
        
        # Optimization rules
        self.rules: List[OptimizationRule] = []
        self._setup_default_rules()
        
        # Performance metrics tracking
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # Optimization history
        self.optimization_results: List[OptimizationResult] = []
        
        # ML models for prediction
        self.ml_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # State management
        self.current_state: Dict[str, float] = {}
        self.best_state: Dict[str, float] = {}
        self.best_performance: float = float('-inf')
        
        logger.info(f"Performance optimizer initialized with {strategy.value} strategy")
    
    def _setup_default_rules(self) -> None:
        """Set up default optimization rules."""
        # CPU optimization rules
        self.add_rule(OptimizationRule(
            name="high_cpu_reduce_threads",
            condition=lambda ctx: ctx.get("cpu_usage", 0) > 90,
            action=lambda ctx: {"thread_pool_size": -2, "worker_count": -1},
            priority=10
        ))
        
        self.add_rule(OptimizationRule(
            name="low_cpu_increase_threads",
            condition=lambda ctx: ctx.get("cpu_usage", 0) < 30 and ctx.get("queue_size", 0) > 10,
            action=lambda ctx: {"thread_pool_size": 2, "worker_count": 1},
            priority=5
        ))
        
        # Memory optimization rules
        self.add_rule(OptimizationRule(
            name="high_memory_reduce_cache",
            condition=lambda ctx: ctx.get("memory_usage", 0) > 85,
            action=lambda ctx: {"cache_size": -0.2, "pool_size": -0.1},
            priority=10
        ))
        
        self.add_rule(OptimizationRule(
            name="memory_pressure_gc",
            condition=lambda ctx: ctx.get("memory_usage", 0) > 95,
            action=lambda ctx: {"gc_threshold": -0.5, "cache_ttl": -0.3},
            priority=15
        ))
        
        # Response time rules
        self.add_rule(OptimizationRule(
            name="slow_response_optimize",
            condition=lambda ctx: ctx.get("avg_response_time", 0) > 1.0,
            action=lambda ctx: {"batch_size": -0.1, "timeout": 0.1},
            priority=8
        ))
        
        # Throughput rules
        self.add_rule(OptimizationRule(
            name="low_throughput_scale",
            condition=lambda ctx: ctx.get("throughput", float('inf')) < ctx.get("target_throughput", 100),
            action=lambda ctx: {"worker_count": 1, "batch_size": 0.1},
            priority=7
        ))
    
    def register_parameter(
        self,
        name: str,
        current_value: float,
        min_value: float,
        max_value: float,
        **kwargs
    ) -> None:
        """
        Register a parameter for optimization.
        
        Args:
            name: Parameter name
            current_value: Current value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            **kwargs: Additional parameter properties
        """
        with self._lock:
            param = OptimizationParameter(
                name=name,
                current_value=current_value,
                min_value=min_value,
                max_value=max_value,
                **kwargs
            )
            
            self.parameters[name] = param
            self.current_state[name] = current_value
            
            logger.info(f"Registered parameter: {name} (range: {min_value}-{max_value})")
    
    def add_rule(self, rule: OptimizationRule) -> None:
        """Add an optimization rule."""
        with self._lock:
            self.rules.append(rule)
            self.rules.sort(key=lambda r: r.priority, reverse=True)
            logger.debug(f"Added optimization rule: {rule.name}")
    
    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Record performance metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        with self._lock:
            timestamp = time.time()
            
            for name, value in metrics.items():
                self.metrics_history[name].append((timestamp, value))
            
            # Update parameter performance if optimization is running
            overall_performance = self._calculate_overall_performance(metrics)
            
            for param in self.parameters.values():
                if param.value_history:
                    param.record_performance(
                        param.current_value,
                        overall_performance
                    )
            
            # Check if this is the best performance
            if overall_performance > self.best_performance:
                self.best_performance = overall_performance
                self.best_state = self.current_state.copy()
                logger.info(f"New best performance: {overall_performance:.3f}")
    
    def _calculate_overall_performance(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score from metrics."""
        # Simple weighted average - can be customized
        weights = {
            "throughput": 1.0,
            "avg_response_time": -0.5,  # Lower is better
            "error_rate": -2.0,          # Lower is better
            "cpu_usage": -0.1,           # Penalty for high CPU
            "memory_usage": -0.1,        # Penalty for high memory
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += abs(weight)
        
        return score / total_weight if total_weight > 0 else 0.0
    
    async def optimize(self, context: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Run optimization based on current metrics and context.
        
        Args:
            context: Additional context for optimization
            
        Returns:
            Optimization result
        """
        with self._lock:
            result = OptimizationResult(
                timestamp=time.time(),
                strategy=self.strategy
            )
            
            # Get current metrics
            current_metrics = self._get_current_metrics()
            result.performance_before = current_metrics.copy()
            
            # Add context
            if context:
                current_metrics.update(context)
            
            # Apply strategy
            if self.strategy == OptimizationStrategy.CONSERVATIVE:
                changes = self._optimize_conservative(current_metrics)
            elif self.strategy == OptimizationStrategy.AGGRESSIVE:
                changes = self._optimize_aggressive(current_metrics)
            elif self.strategy == OptimizationStrategy.ADAPTIVE:
                changes = await self._optimize_adaptive(current_metrics)
            else:  # BALANCED or CUSTOM
                changes = self._optimize_balanced(current_metrics)
            
            # Apply changes
            for param_name, delta in changes.items():
                if param_name in self.parameters:
                    param = self.parameters[param_name]
                    old_value = param.current_value
                    new_value = param.adjust_value(delta)
                    
                    if new_value != old_value:
                        param.current_value = new_value
                        self.current_state[param_name] = new_value
                        result.parameter_changes[param_name] = (old_value, new_value)
                        
                        logger.info(f"Adjusted {param_name}: {old_value} -> {new_value}")
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(current_metrics)
            
            # Store result
            self.optimization_results.append(result)
            
            return result
    
    def _optimize_conservative(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Conservative optimization - small, safe changes."""
        changes = {}
        
        # Apply only highest priority rules
        for rule in self.rules[:3]:  # Top 3 rules
            rule_changes = rule.apply(metrics)
            if rule_changes:
                for param, delta in rule_changes.items():
                    # Reduce change magnitude for conservative approach
                    changes[param] = delta * 0.5
                break  # Apply only one rule
        
        return changes
    
    def _optimize_balanced(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Balanced optimization - apply rules with moderation."""
        changes = defaultdict(float)
        applied_count = 0
        max_rules = 3
        
        # Apply rules
        for rule in self.rules:
            if applied_count >= max_rules:
                break
            
            rule_changes = rule.apply(metrics)
            if rule_changes:
                for param, delta in rule_changes.items():
                    changes[param] += delta * self.learning_rate
                applied_count += 1
        
        return dict(changes)
    
    def _optimize_aggressive(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggressive optimization - larger changes for faster convergence."""
        changes = defaultdict(float)
        
        # Apply all applicable rules
        for rule in self.rules:
            rule_changes = rule.apply(metrics)
            if rule_changes:
                for param, delta in rule_changes.items():
                    changes[param] += delta * 2.0  # Amplify changes
        
        # Also use gradient ascent on performance
        for param_name, param in self.parameters.items():
            if len(param.performance_history) >= 3:
                # Estimate gradient
                recent_values = list(param.value_history)[-3:]
                recent_performance = list(param.performance_history)[-3:]
                
                if len(set(recent_values)) > 1:  # Need variation
                    gradient = np.polyfit(recent_values, recent_performance, 1)[0]
                    changes[param_name] += gradient * param.step_size * 2.0
        
        return dict(changes)
    
    async def _optimize_adaptive(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Adaptive optimization using ML predictions."""
        changes = {}
        
        # Start with rule-based changes
        rule_changes = self._optimize_balanced(metrics)
        changes.update(rule_changes)
        
        if not self.enable_ml:
            return changes
        
        # Use ML predictions if we have enough data
        for param_name, param in self.parameters.items():
            if len(param.value_history) < 50:
                continue  # Not enough data
            
            try:
                # Prepare training data
                X = np.array(param.value_history).reshape(-1, 1)
                y = np.array(param.performance_history)
                
                # Train or update model
                if param_name not in self.ml_models:
                    self.ml_models[param_name] = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=5,
                        random_state=42
                    )
                    self.scalers[param_name] = StandardScaler()
                
                # Fit model
                X_scaled = self.scalers[param_name].fit_transform(X)
                self.ml_models[param_name].fit(X_scaled, y)
                
                # Predict performance for different values
                test_values = np.linspace(
                    param.min_value,
                    param.max_value,
                    20
                ).reshape(-1, 1)
                
                test_scaled = self.scalers[param_name].transform(test_values)
                predictions = self.ml_models[param_name].predict(test_scaled)
                
                # Find best predicted value
                best_idx = np.argmax(predictions)
                best_value = test_values[best_idx, 0]
                
                # Calculate change needed
                delta = best_value - param.current_value
                
                # Apply exploration vs exploitation
                if np.random.random() < self.exploration_rate:
                    # Explore: add some randomness
                    delta += np.random.normal(0, param.step_size)
                
                changes[param_name] = changes.get(param_name, 0) + delta * self.learning_rate
                
            except Exception as e:
                logger.warning(f"ML optimization failed for {param_name}: {e}")
        
        return changes
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # CPU recommendations
        cpu_usage = metrics.get("cpu_usage", 0)
        if cpu_usage > 80:
            recommendations.append(
                f"High CPU usage ({cpu_usage:.1f}%). Consider scaling horizontally or optimizing algorithms."
            )
        elif cpu_usage < 20:
            recommendations.append(
                f"Low CPU usage ({cpu_usage:.1f}%). Resources may be over-provisioned."
            )
        
        # Memory recommendations
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > 80:
            recommendations.append(
                f"High memory usage ({memory_usage:.1f}%). Consider increasing cache eviction or reducing pool sizes."
            )
        
        # Response time recommendations
        avg_response = metrics.get("avg_response_time", 0)
        if avg_response > 1.0:
            recommendations.append(
                f"Slow average response time ({avg_response:.2f}s). Consider caching or query optimization."
            )
        
        # Error rate recommendations
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.05:
            recommendations.append(
                f"High error rate ({error_rate*100:.1f}%). Investigate error patterns and add retries."
            )
        
        return recommendations
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        current = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                # Get average of recent values
                recent = [value for _, value in list(history)[-10:]]
                current[metric_name] = sum(recent) / len(recent)
        
        return current
    
    def rollback(self) -> bool:
        """
        Rollback to the best known state.
        
        Returns:
            True if rollback was successful
        """
        with self._lock:
            if not self.best_state:
                logger.warning("No best state to rollback to")
                return False
            
            # Apply best state
            for param_name, value in self.best_state.items():
                if param_name in self.parameters:
                    self.parameters[param_name].current_value = value
                    self.current_state[param_name] = value
            
            logger.info(f"Rolled back to best state (performance: {self.best_performance:.3f})")
            return True
    
    def start_ab_test(
        self,
        test_name: str,
        parameter: str,
        variant_a: float,
        variant_b: float,
        duration_seconds: float = 300.0
    ) -> None:
        """
        Start an A/B test for a parameter.
        
        Args:
            test_name: Name of the test
            parameter: Parameter to test
            variant_a: Value for variant A
            variant_b: Value for variant B
            duration_seconds: Test duration
        """
        with self._lock:
            if parameter not in self.parameters:
                raise ValueError(f"Unknown parameter: {parameter}")
            
            self.ab_tests[test_name] = {
                "parameter": parameter,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "start_time": time.time(),
                "duration": duration_seconds,
                "results_a": [],
                "results_b": [],
                "current_variant": "A"
            }
            
            logger.info(f"Started A/B test '{test_name}' for {parameter}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        with self._lock:
            recent_results = self.optimization_results[-10:] if self.optimization_results else []
            
            # Calculate statistics
            total_optimizations = len(self.optimization_results)
            successful_optimizations = sum(
                1 for r in self.optimization_results if r.was_successful
            )
            
            # Parameter statistics
            param_stats = {}
            for name, param in self.parameters.items():
                if param.performance_history:
                    param_stats[name] = {
                        "current_value": param.current_value,
                        "best_value": param.value_history[
                            np.argmax(param.performance_history)
                        ] if param.performance_history else param.current_value,
                        "average_performance": np.mean(param.performance_history) if param.performance_history else 0,
                        "value_range_explored": (
                            min(param.value_history) if param.value_history else param.min_value,
                            max(param.value_history) if param.value_history else param.max_value
                        )
                    }
            
            # Rule statistics
            rule_stats = [
                {
                    "name": rule.name,
                    "times_applied": rule.times_applied,
                    "last_applied": rule.last_applied,
                    "enabled": rule.enabled
                }
                for rule in self.rules
            ]
            
            return {
                "total_optimizations": total_optimizations,
                "successful_optimizations": successful_optimizations,
                "success_rate": successful_optimizations / total_optimizations if total_optimizations > 0 else 0,
                "current_performance": self._calculate_overall_performance(self._get_current_metrics()),
                "best_performance": self.best_performance,
                "parameter_stats": param_stats,
                "rule_stats": sorted(rule_stats, key=lambda r: r["times_applied"], reverse=True),
                "recent_results": [
                    {
                        "timestamp": r.timestamp,
                        "strategy": r.strategy.value,
                        "changes": r.parameter_changes,
                        "improvement": r.improvement_percent,
                        "stability": r.stability_score
                    }
                    for r in recent_results
                ],
                "active_ab_tests": list(self.ab_tests.keys())
            }


def create_performance_optimizer(**kwargs) -> PerformanceOptimizer:
    """
    Create a performance optimizer with the given configuration.
    
    Args:
        **kwargs: Optimizer configuration
        
    Returns:
        Configured performance optimizer
    """
    return PerformanceOptimizer(**kwargs)


if __name__ == "__main__":
    import random
    
    async def demo_performance_optimizer():
        """Demonstrate performance optimization functionality."""
        print("🎯 Performance Optimizer Demo")
        print("=" * 40)
        
        # Create optimizer
        optimizer = create_performance_optimizer(
            strategy=OptimizationStrategy.ADAPTIVE,
            learning_rate=0.2,
            exploration_rate=0.3
        )
        
        # Register parameters
        optimizer.register_parameter(
            "thread_pool_size",
            current_value=10,
            min_value=1,
            max_value=50,
            is_integer=True,
            step_size=2
        )
        
        optimizer.register_parameter(
            "cache_size",
            current_value=100,
            min_value=10,
            max_value=1000,
            step_size=10
        )
        
        optimizer.register_parameter(
            "batch_size",
            current_value=32,
            min_value=1,
            max_value=128,
            is_integer=True,
            step_size=8
        )
        
        optimizer.register_parameter(
            "gc_threshold",
            current_value=0.8,
            min_value=0.1,
            max_value=1.0,
            step_size=0.1
        )
        
        print(f"Registered {len(optimizer.parameters)} parameters")
        
        # Simulate optimization loop
        print("\nRunning optimization simulation...")
        
        for iteration in range(20):
            # Simulate metrics based on current parameters
            thread_pool = optimizer.current_state["thread_pool_size"]
            cache_size = optimizer.current_state["cache_size"]
            batch_size = optimizer.current_state["batch_size"]
            gc_threshold = optimizer.current_state["gc_threshold"]
            
            # Generate synthetic metrics
            throughput = (
                thread_pool * 10 +
                cache_size * 0.1 +
                batch_size * 0.5 +
                random.uniform(-5, 5)
            )
            
            response_time = (
                1.0 / (thread_pool * 0.1 + 1) +
                0.5 / (cache_size * 0.01 + 1) +
                batch_size * 0.001 +
                random.uniform(0, 0.2)
            )
            
            cpu_usage = min(100, (
                thread_pool * 2 +
                batch_size * 0.5 +
                random.uniform(0, 20)
            ))
            
            memory_usage = min(100, (
                cache_size * 0.05 +
                batch_size * 0.2 +
                (1 - gc_threshold) * 20 +
                random.uniform(0, 10)
            ))
            
            error_rate = max(0, (
                0.01 * (1 if thread_pool > 40 else 0) +
                0.02 * (1 if memory_usage > 90 else 0) +
                random.uniform(0, 0.02)
            ))
            
            metrics = {
                "throughput": throughput,
                "avg_response_time": response_time,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "error_rate": error_rate,
                "queue_size": random.randint(0, 50)
            }
            
            # Record metrics
            optimizer.record_metrics(metrics)
            
            # Run optimization every 5 iterations
            if iteration % 5 == 4:
                print(f"\nIteration {iteration + 1}:")
                print(f"   Current metrics:")
                for name, value in metrics.items():
                    print(f"      {name}: {value:.2f}")
                
                result = await optimizer.optimize({"target_throughput": 150})
                
                if result.parameter_changes:
                    print(f"   Parameter changes:")
                    for param, (old, new) in result.parameter_changes.items():
                        print(f"      {param}: {old} -> {new}")
                
                if result.recommendations:
                    print(f"   Recommendations:")
                    for rec in result.recommendations:
                        print(f"      - {rec}")
            
            # Small delay
            await asyncio.sleep(0.1)
        
        # Start A/B test
        print("\nStarting A/B test...")
        optimizer.start_ab_test(
            "thread_pool_experiment",
            "thread_pool_size",
            variant_a=optimizer.current_state["thread_pool_size"],
            variant_b=optimizer.current_state["thread_pool_size"] + 5,
            duration_seconds=60
        )
        
        # Get optimization report
        print("\nOptimization Report:")
        report = optimizer.get_optimization_report()
        
        print(f"   Total optimizations: {report['total_optimizations']}")
        print(f"   Success rate: {report['success_rate']:.2%}")
        print(f"   Current performance: {report['current_performance']:.2f}")
        print(f"   Best performance: {report['best_performance']:.2f}")
        
        print("\n   Parameter Statistics:")
        for param, stats in report['parameter_stats'].items():
            print(f"      {param}:")
            print(f"         Current: {stats['current_value']}")
            print(f"         Best: {stats['best_value']}")
            print(f"         Avg performance: {stats['average_performance']:.2f}")
        
        print("\n   Most Used Rules:")
        for rule in report['rule_stats'][:3]:
            if rule['times_applied'] > 0:
                print(f"      {rule['name']}: applied {rule['times_applied']} times")
        
        # Demonstrate rollback
        print("\nDemonstrating rollback...")
        print(f"   Current state: {optimizer.current_state}")
        
        if optimizer.rollback():
            print(f"   Rolled back to: {optimizer.current_state}")
        
        print("\n🎯 Performance Optimizer Demo Complete")
    
    # Run demo
    asyncio.run(demo_performance_optimizer())