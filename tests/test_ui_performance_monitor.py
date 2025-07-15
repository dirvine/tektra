"""
Tests for UI Performance Monitor

Tests the comprehensive performance monitoring system including frame rate tracking,
system resource monitoring, and automatic optimization.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

from src.tektra.gui.animations.performance_monitor import (
    UIPerformanceMonitor, 
    PerformanceThresholds,
    SystemPerformanceMetrics,
    AnimationPerformanceMetrics
)
from src.tektra.gui.animations.animation_config import AnimationConfig, EasingFunction


class TestUIPerformanceMonitor:
    """Test suite for UIPerformanceMonitor."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor for testing."""
        thresholds = PerformanceThresholds(
            target_fps=60.0,
            minimum_fps=30.0,
            critical_fps=15.0,
            optimization_check_interval=1.0
        )
        return UIPerformanceMonitor(thresholds)
    
    def test_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.monitoring_enabled is True
        assert performance_monitor.auto_optimization_enabled is True
        assert performance_monitor.current_performance_level == "high"
        assert len(performance_monitor.optimization_callbacks) == 0
    
    def test_frame_time_recording(self, performance_monitor):
        """Test frame time recording and FPS calculation."""
        # Record some frame times
        frame_times = [16.67, 16.67, 16.67, 16.67, 16.67]  # 60 FPS
        for frame_time in frame_times:
            performance_monitor.record_frame_time(frame_time)
        
        # Check FPS calculation
        assert abs(performance_monitor.metrics.fps - 60.0) < 1.0
        assert performance_monitor.metrics.average_frame_time == 16.67
        assert len(performance_monitor.frame_time_history) == 5
    
    def test_animation_tracking(self, performance_monitor):
        """Test animation start/end tracking."""
        # Start some animations
        performance_monitor.record_animation_start("anim1", "fade_in")
        performance_monitor.record_animation_start("anim2", "slide_in")
        
        assert performance_monitor.metrics.animation_count == 2
        assert performance_monitor.animation_metrics.total_animations_started == 2
        assert "fade_in" in performance_monitor.animation_metrics.animation_types_count
        assert "slide_in" in performance_monitor.animation_metrics.animation_types_count
        
        # End animations
        performance_monitor.record_animation_end("anim1", completed=True)
        performance_monitor.record_animation_end("anim2", completed=False)
        
        assert performance_monitor.metrics.animation_count == 0
        assert performance_monitor.animation_metrics.total_animations_completed == 1
        assert performance_monitor.animation_metrics.total_animations_cancelled == 1
    
    def test_performance_level_determination(self, performance_monitor):
        """Test performance level determination based on metrics."""
        # Test high performance (good FPS, low resource usage)
        with patch('psutil.cpu_percent', return_value=20.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 30.0
            
            # Record good frame times (60 FPS)
            for _ in range(10):
                performance_monitor.record_frame_time(16.67)
            
            level = performance_monitor._determine_performance_level()
            assert level == "high"
    
    def test_performance_level_degradation(self, performance_monitor):
        """Test performance level degradation detection."""
        # Test critical performance (poor FPS, high resource usage)
        with patch('psutil.cpu_percent', return_value=90.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90.0
            
            # Record poor frame times (10 FPS)
            for _ in range(10):
                performance_monitor.record_frame_time(100.0)
            
            level = performance_monitor._determine_performance_level()
            assert level == "critical"
    
    def test_should_reduce_animations(self, performance_monitor):
        """Test animation reduction recommendation."""
        # Test with good performance
        with patch('psutil.cpu_percent', return_value=20.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 30.0
            
            for _ in range(10):
                performance_monitor.record_frame_time(16.67)
            
            assert not performance_monitor.should_reduce_animations()
        
        # Test with poor performance
        with patch('psutil.cpu_percent', return_value=90.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90.0
            
            for _ in range(10):
                performance_monitor.record_frame_time(100.0)
            
            assert performance_monitor.should_reduce_animations()
    
    def test_performance_recommendations(self, performance_monitor):
        """Test performance optimization recommendations."""
        # Test with poor performance
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 88.0
            
            # Record poor frame times
            for _ in range(20):
                performance_monitor.record_frame_time(50.0)  # 20 FPS
            
            recommendations = performance_monitor.get_performance_recommendations()
            
            assert len(recommendations) > 0
            assert any("CPU usage" in rec for rec in recommendations)
            assert any("memory usage" in rec for rec in recommendations)
    
    def test_optimized_config_generation(self, performance_monitor):
        """Test generation of optimized animation configurations."""
        base_config = AnimationConfig(
            duration=0.5,
            easing=EasingFunction.EASE_OUT,
            delay=0.1
        )
        
        # Test with good performance (should return original config)
        with patch('psutil.cpu_percent', return_value=20.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 30.0
            
            for _ in range(10):
                performance_monitor.record_frame_time(16.67)
            
            optimized = performance_monitor.get_optimized_config(base_config)
            assert optimized.duration == base_config.duration
        
        # Test with poor performance (should return optimized config)
        with patch('psutil.cpu_percent', return_value=90.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90.0
            
            for _ in range(10):
                performance_monitor.record_frame_time(100.0)
            
            optimized = performance_monitor.get_optimized_config(base_config)
            assert optimized.duration < base_config.duration
            assert optimized.delay == 0.0  # Delays removed for performance
    
    @patch('psutil.sensors_battery')
    @patch('psutil.net_io_counters')
    @patch('psutil.disk_io_counters')
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_system_metrics_update(self, mock_cpu, mock_memory, mock_disk, mock_net, mock_battery, performance_monitor):
        """Test system metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value.percent = 60.0
        mock_memory.return_value.available = 4 * 1024 * 1024 * 1024  # 4GB
        
        mock_disk_io = Mock()
        mock_disk_io.read_bytes = 1024 * 1024 * 100  # 100MB
        mock_disk_io.write_bytes = 1024 * 1024 * 50   # 50MB
        mock_disk.return_value = mock_disk_io
        
        mock_net_io = Mock()
        mock_net_io.bytes_sent = 1024 * 1024 * 10     # 10MB
        mock_net_io.bytes_recv = 1024 * 1024 * 20     # 20MB
        mock_net.return_value = mock_net_io
        
        mock_battery_info = Mock()
        mock_battery_info.percent = 75.0
        mock_battery_info.power_plugged = True
        mock_battery.return_value = mock_battery_info
        
        # Update metrics
        performance_monitor._update_system_metrics()
        
        # Verify metrics were collected
        assert performance_monitor.system_metrics.cpu_usage == 45.0
        assert performance_monitor.system_metrics.memory_usage == 60.0
        assert performance_monitor.system_metrics.battery_level == 75.0
        assert performance_monitor.system_metrics.power_plugged is True
    
    def test_optimization_callbacks(self, performance_monitor):
        """Test optimization callback system."""
        callback_called = False
        callback_args = None
        
        def test_callback():
            nonlocal callback_called, callback_args
            callback_called = True
        
        # Add callback
        performance_monitor.add_optimization_callback(test_callback)
        assert len(performance_monitor.optimization_callbacks) == 1
        
        # Trigger optimization by simulating poor performance
        with patch('psutil.cpu_percent', return_value=90.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90.0
            
            for _ in range(10):
                performance_monitor.record_frame_time(100.0)
            
            # Force optimization check
            performance_monitor._check_performance_optimization()
            
            assert callback_called
        
        # Remove callback
        performance_monitor.remove_optimization_callback(test_callback)
        assert len(performance_monitor.optimization_callbacks) == 0
    
    def test_detailed_performance_summary(self, performance_monitor):
        """Test comprehensive performance summary generation."""
        # Add some test data
        performance_monitor.record_animation_start("test1", "fade_in")
        performance_monitor.record_frame_time(20.0)
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 40.0
            mock_memory.return_value.available = 2 * 1024 * 1024 * 1024  # 2GB
            
            summary = performance_monitor.get_detailed_performance_summary()
            
            assert "performance_level" in summary
            assert "fps_metrics" in summary
            assert "animation_metrics" in summary
            assert "system_metrics" in summary
            assert "optimization" in summary
            
            # Check specific values
            assert summary["fps_metrics"]["current_fps"] == 50.0  # 1000/20
            assert summary["animation_metrics"]["active_animations"] == 1
            assert summary["system_metrics"]["cpu_usage"] == 50.0
    
    def test_background_monitoring(self, performance_monitor):
        """Test background monitoring thread."""
        # Start background monitoring
        performance_monitor.start_background_monitoring()
        
        # Wait a short time for thread to start
        time.sleep(0.1)
        
        assert performance_monitor._monitoring_thread is not None
        assert performance_monitor._monitoring_thread.is_alive()
        
        # Stop monitoring
        performance_monitor.stop_background_monitoring()
        
        # Wait for thread to stop
        time.sleep(0.1)
        
        assert not performance_monitor._monitoring_thread.is_alive()
    
    @pytest.mark.asyncio
    async def test_async_monitoring_loop(self, performance_monitor):
        """Test async monitoring loop."""
        # Create a task for the monitoring loop
        monitoring_task = asyncio.create_task(performance_monitor.start_monitoring_loop())
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop monitoring and cancel task
        performance_monitor.monitoring_enabled = False
        monitoring_task.cancel()
        
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass  # Expected when cancelling
    
    def test_performance_history_maintenance(self, performance_monitor):
        """Test that performance history is properly maintained."""
        # Add more entries than the history limit
        for i in range(performance_monitor.thresholds.performance_history_length + 50):
            performance_monitor.record_frame_time(16.67)
        
        # Check that history doesn't exceed limit
        assert len(performance_monitor.frame_time_history) <= performance_monitor.thresholds.performance_history_length
    
    def test_auto_optimization_toggle(self, performance_monitor):
        """Test enabling/disabling auto optimization."""
        assert performance_monitor.auto_optimization_enabled is True
        
        performance_monitor.enable_auto_optimization(False)
        assert performance_monitor.auto_optimization_enabled is False
        
        performance_monitor.enable_auto_optimization(True)
        assert performance_monitor.auto_optimization_enabled is True
    
    def test_metrics_reset(self, performance_monitor):
        """Test resetting performance metrics."""
        # Add some data
        performance_monitor.record_frame_time(20.0)
        performance_monitor.record_animation_start("test", "fade")
        
        assert len(performance_monitor.metrics.frame_times) > 0
        assert performance_monitor.metrics.animation_count > 0
        
        # Reset metrics
        performance_monitor.reset_metrics()
        
        assert len(performance_monitor.metrics.frame_times) == 0
        assert performance_monitor.metrics.animation_count == 0