#!/usr/bin/env python3
"""
Performance Monitoring and Optimization Demo

This demo showcases the comprehensive performance monitoring and optimization
system for UI animations, including automatic quality adjustment and fallback modes.
"""

import asyncio
import time
import random
from typing import List, Dict, Any

# Mock Toga for demonstration
class MockWidget:
    def __init__(self, name: str):
        self.name = name
        self.id = id(self)
    
    def __str__(self):
        return f"MockWidget({self.name})"

# Import our performance monitoring system
from src.tektra.gui.animations.performance_monitor import UIPerformanceMonitor, PerformanceThresholds
from src.tektra.gui.animations.performance_optimizer import PerformanceOptimizer
from src.tektra.gui.animations.animation_config import AnimationType, AnimationConfig


class PerformanceMonitoringDemo:
    """Demo class for performance monitoring and optimization."""
    
    def __init__(self):
        """Initialize the demo."""
        # Create performance monitor with custom thresholds
        self.thresholds = PerformanceThresholds(
            target_fps=60.0,
            minimum_fps=30.0,
            critical_fps=15.0,
            optimization_check_interval=2.0
        )
        
        self.performance_monitor = UIPerformanceMonitor(self.thresholds)
        self.performance_optimizer = PerformanceOptimizer(self.performance_monitor)
        
        # Demo widgets
        self.widgets = [
            MockWidget("message_1"),
            MockWidget("message_2"),
            MockWidget("button_1"),
            MockWidget("input_field"),
            MockWidget("typing_indicator")
        ]
        
        print("🚀 Performance Monitoring Demo Initialized")
        print(f"📊 Target FPS: {self.thresholds.target_fps}")
        print(f"⚠️  Minimum FPS: {self.thresholds.minimum_fps}")
        print(f"🔴 Critical FPS: {self.thresholds.critical_fps}")
        print()
    
    def simulate_frame_times(self, scenario: str) -> List[float]:
        """Simulate different performance scenarios."""
        if scenario == "excellent":
            # 60 FPS - 16.67ms per frame
            return [16.67 + random.uniform(-1, 1) for _ in range(20)]
        elif scenario == "good":
            # 45 FPS - 22.22ms per frame
            return [22.22 + random.uniform(-2, 2) for _ in range(20)]
        elif scenario == "poor":
            # 20 FPS - 50ms per frame
            return [50.0 + random.uniform(-5, 10) for _ in range(20)]
        elif scenario == "critical":
            # 10 FPS - 100ms per frame
            return [100.0 + random.uniform(-10, 20) for _ in range(20)]
        else:
            return [16.67] * 20
    
    def demonstrate_performance_monitoring(self):
        """Demonstrate basic performance monitoring."""
        print("📈 PERFORMANCE MONITORING DEMONSTRATION")
        print("=" * 50)
        
        scenarios = ["excellent", "good", "poor", "critical"]
        
        for scenario in scenarios:
            print(f"\n🎯 Testing {scenario.upper()} performance scenario:")
            
            # Reset metrics for clean test
            self.performance_monitor.reset_metrics()
            
            # Simulate frame times
            frame_times = self.simulate_frame_times(scenario)
            for frame_time in frame_times:
                self.performance_monitor.record_frame_time(frame_time)
            
            # Get performance summary
            summary = self.performance_monitor.get_detailed_performance_summary()
            
            print(f"   FPS: {summary['fps_metrics']['current_fps']:.1f}")
            print(f"   Avg Frame Time: {summary['fps_metrics']['average_frame_time']:.2f}ms")
            print(f"   Performance Level: {summary['performance_level']}")
            print(f"   Should Reduce Animations: {summary['optimization']['should_reduce_animations']}")
            
            # Show recommendations
            recommendations = summary['optimization']['recommendations']
            if recommendations:
                print("   Recommendations:")
                for rec in recommendations[:3]:  # Show first 3
                    print(f"     • {rec}")
    
    def demonstrate_animation_optimization(self):
        """Demonstrate animation configuration optimization."""
        print("\n\n🎨 ANIMATION OPTIMIZATION DEMONSTRATION")
        print("=" * 50)
        
        # Base animation configuration
        base_config = AnimationConfig(
            duration=0.5,
            delay=0.1
        )
        
        print(f"📋 Base Configuration:")
        print(f"   Duration: {base_config.duration}s")
        print(f"   Delay: {base_config.delay}s")
        print(f"   Performance Mode: {base_config.performance_mode}")
        
        # Test optimization under different performance levels
        scenarios = [
            ("excellent", "high"),
            ("good", "medium"), 
            ("poor", "low"),
            ("critical", "critical")
        ]
        
        for scenario_name, expected_level in scenarios:
            print(f"\n🔧 Optimizing for {scenario_name.upper()} performance:")
            
            # Simulate performance scenario
            self.performance_monitor.reset_metrics()
            frame_times = self.simulate_frame_times(scenario_name)
            for frame_time in frame_times:
                self.performance_monitor.record_frame_time(frame_time)
            
            # Get optimized configuration
            optimized_config = self.performance_optimizer.optimize_animation_config(base_config)
            
            print(f"   Optimized Duration: {optimized_config.duration:.3f}s")
            print(f"   Optimized Delay: {optimized_config.delay:.3f}s")
            print(f"   Performance Mode: {optimized_config.performance_mode}")
            print(f"   Reduced Motion: {optimized_config.reduced_motion}")
    
    def demonstrate_profile_switching(self):
        """Demonstrate optimization profile switching."""
        print("\n\n⚙️  OPTIMIZATION PROFILE DEMONSTRATION")
        print("=" * 50)
        
        profiles = ["ultra", "high", "balanced", "performance", "minimal", "fallback"]
        
        for profile_name in profiles:
            self.performance_optimizer.switch_to_profile(profile_name)
            profile = self.performance_optimizer.current_profile
            
            print(f"\n📊 Profile: {profile_name.upper()}")
            print(f"   Max Concurrent Animations: {profile.max_concurrent_animations}")
            print(f"   Quality Multiplier: {profile.animation_quality_multiplier:.1f}")
            print(f"   Complex Easing: {profile.enable_complex_easing}")
            print(f"   Micro Interactions: {profile.enable_micro_interactions}")
            print(f"   Frame Rate Target: {profile.frame_rate_target:.0f} FPS")
            print(f"   Memory Limit: {profile.memory_limit_mb:.0f} MB")
    
    def demonstrate_animation_skipping(self):
        """Demonstrate animation skipping logic."""
        print("\n\n⏭️  ANIMATION SKIPPING DEMONSTRATION")
        print("=" * 50)
        
        animation_types = [
            AnimationType.FADE_IN,
            AnimationType.SLIDE_IN,
            AnimationType.BUTTON_PRESS,
            AnimationType.TYPING_INDICATOR
        ]
        
        # Test with different profiles
        test_profiles = ["high", "performance", "fallback"]
        
        for profile_name in test_profiles:
            print(f"\n🎯 Testing with {profile_name.upper()} profile:")
            self.performance_optimizer.switch_to_profile(profile_name)
            
            for animation_type in animation_types:
                should_skip = self.performance_optimizer.should_skip_animation(animation_type)
                status = "SKIP" if should_skip else "ALLOW"
                print(f"   {animation_type.value}: {status}")
    
    async def demonstrate_animation_queuing(self):
        """Demonstrate animation queuing system."""
        print("\n\n📋 ANIMATION QUEUING DEMONSTRATION")
        print("=" * 50)
        
        # Switch to a restrictive profile
        self.performance_optimizer.switch_to_profile("minimal")  # Max 3 concurrent
        
        # Simulate high animation load
        self.performance_monitor.metrics.animation_count = 5  # Above limit
        
        print(f"Current active animations: {self.performance_monitor.metrics.animation_count}")
        print(f"Profile limit: {self.performance_optimizer.current_profile.max_concurrent_animations}")
        
        # Queue some animations
        queued_animations = []
        for i in range(5):
            animation_data = {
                "type": f"fade_in_{i}",
                "widget": self.widgets[i % len(self.widgets)],
                "callback": self.create_mock_animation_callback(f"animation_{i}")
            }
            
            success = self.performance_optimizer.queue_animation(animation_data)
            if success:
                queued_animations.append(animation_data)
                print(f"✅ Queued animation_{i}")
            else:
                print(f"❌ Failed to queue animation_{i}")
        
        print(f"\nQueued animations: {len(self.performance_optimizer.animation_queue)}")
        
        # Simulate performance improvement
        print("\n🔄 Simulating performance improvement...")
        self.performance_monitor.metrics.animation_count = 1  # Reduce load
        
        # Process queue
        await self.performance_optimizer.process_animation_queue()
        
        print(f"Remaining queued animations: {len(self.performance_optimizer.animation_queue)}")
    
    def create_mock_animation_callback(self, name: str):
        """Create a mock animation callback."""
        async def callback():
            print(f"   🎬 Executing {name}")
            await asyncio.sleep(0.1)  # Simulate animation time
        return callback
    
    def demonstrate_comprehensive_summary(self):
        """Demonstrate comprehensive performance summary."""
        print("\n\n📊 COMPREHENSIVE PERFORMANCE SUMMARY")
        print("=" * 50)
        
        # Simulate some activity
        self.performance_monitor.record_animation_start("demo_anim_1", "fade_in")
        self.performance_monitor.record_animation_start("demo_anim_2", "slide_in")
        
        frame_times = self.simulate_frame_times("good")
        for frame_time in frame_times:
            self.performance_monitor.record_frame_time(frame_time)
        
        # Get comprehensive summary
        monitor_summary = self.performance_monitor.get_detailed_performance_summary()
        optimizer_summary = self.performance_optimizer.get_optimization_summary()
        
        print("🔍 Performance Monitor Summary:")
        print(f"   Performance Level: {monitor_summary['performance_level']}")
        print(f"   Current FPS: {monitor_summary['fps_metrics']['current_fps']:.1f}")
        print(f"   Active Animations: {monitor_summary['animation_metrics']['active_animations']}")
        print(f"   Total Started: {monitor_summary['animation_metrics']['total_started']}")
        print(f"   CPU Usage: {monitor_summary['system_metrics']['cpu_usage']:.1f}%")
        
        print("\n⚙️  Performance Optimizer Summary:")
        print(f"   Current Profile: {optimizer_summary['current_profile']['name']}")
        print(f"   Quality Multiplier: {optimizer_summary['current_profile']['quality_multiplier']:.1f}")
        print(f"   Fallback Mode: {optimizer_summary['optimization_state']['fallback_mode_active']}")
        print(f"   Memory Usage: {optimizer_summary['optimization_state']['memory_usage_mb']:.1f} MB")
        
        # Show recommendations
        recommendations = monitor_summary['optimization']['recommendations']
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations[:5]:
                print(f"   • {rec}")
    
    async def run_full_demo(self):
        """Run the complete demonstration."""
        print("🎭 PERFORMANCE MONITORING & OPTIMIZATION DEMO")
        print("=" * 60)
        print("This demo showcases the comprehensive performance monitoring")
        print("and optimization system for UI animations.\n")
        
        # Run all demonstrations
        self.demonstrate_performance_monitoring()
        self.demonstrate_animation_optimization()
        self.demonstrate_profile_switching()
        self.demonstrate_animation_skipping()
        await self.demonstrate_animation_queuing()
        self.demonstrate_comprehensive_summary()
        
        print("\n\n✨ DEMO COMPLETE")
        print("=" * 60)
        print("The performance monitoring system provides:")
        print("• Real-time FPS and frame time tracking")
        print("• Automatic animation quality adjustment")
        print("• System resource monitoring")
        print("• Performance-based fallback modes")
        print("• Animation queuing for overloaded systems")
        print("• Comprehensive performance analytics")
        print("\nThis ensures smooth UI performance across all system capabilities! 🚀")


async def main():
    """Run the performance monitoring demo."""
    demo = PerformanceMonitoringDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())