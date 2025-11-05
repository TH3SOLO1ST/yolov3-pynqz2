"""
Performance monitoring utilities for YOLOv3 realtime system
"""

import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import logging

class PerformanceMonitor:
    """Real-time performance monitoring system"""

    def __init__(self, window_size: int = 100):
        """Initialize performance monitor"""
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)

        # Performance metrics
        self.fps_history = deque(maxlen=window_size)
        self.capture_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.display_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)

        # System metrics
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.memory_percent = deque(maxlen=window_size)

        # Frame statistics
        self.total_frames = 0
        self.dropped_frames = 0
        self.start_time = time.time()

        # Threading
        self.monitoring_thread = None
        self.monitoring_active = False

        self.logger.info("Performance monitor initialized")

    def start_monitoring(self, interval: float = 1.0):
        """Start background system monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop background system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        self.logger.info("System monitoring stopped")

    def _monitor_system(self, interval: float):
        """Background system monitoring thread"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.used / (1024**3))  # GB
                self.memory_percent.append(memory.percent)

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                time.sleep(interval)

    def log_capture_time(self, capture_time: float):
        """Log camera capture time"""
        self.capture_times.append(capture_time)

    def log_inference_time(self, inference_time: float):
        """Log DPU inference time"""
        self.inference_times.append(inference_time)

    def log_display_time(self, display_time: float):
        """Log display render time"""
        self.display_times.append(display_time)

    def log_total_time(self, total_time: float):
        """Log total processing time"""
        self.total_times.append(total_time)

    def update_fps(self, fps: float):
        """Update FPS measurement"""
        self.fps_history.append(fps)
        self.total_frames += int(fps)

    def increment_dropped_frames(self, count: int = 1):
        """Increment dropped frame counter"""
        self.dropped_frames += count

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time

        stats = {
            "uptime_seconds": uptime,
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "drop_rate_percent": (self.dropped_frames / max(1, self.total_frames)) * 100,
        }

        # FPS statistics
        if self.fps_history:
            fps_array = np.array(self.fps_history)
            stats.update({
                "fps_current": self.fps_history[-1] if self.fps_history else 0,
                "fps_avg": np.mean(fps_array),
                "fps_min": np.min(fps_array),
                "fps_max": np.max(fps_array),
                "fps_std": np.std(fps_array)
            })

        # Timing statistics
        for name, times in [
            ("capture", self.capture_times),
            ("inference", self.inference_times),
            ("display", self.display_times),
            ("total", self.total_times)
        ]:
            if times:
                times_array = np.array(times) * 1000  # Convert to milliseconds
                stats.update({
                    f"{name}_time_avg_ms": np.mean(times_array),
                    f"{name}_time_min_ms": np.min(times_array),
                    f"{name}_time_max_ms": np.max(times_array),
                    f"{name}_time_std_ms": np.std(times_array)
                })

        # System statistics
        if self.cpu_usage:
            cpu_array = np.array(self.cpu_usage)
            stats.update({
                "cpu_usage_percent": cpu_array[-1] if self.cpu_usage else 0,
                "cpu_usage_avg": np.mean(cpu_array),
                "cpu_usage_max": np.max(cpu_array)
            })

        if self.memory_usage:
            memory_array = np.array(self.memory_usage)
            stats.update({
                "memory_usage_gb": memory_array[-1] if self.memory_usage else 0,
                "memory_usage_avg_gb": np.mean(memory_array),
                "memory_usage_max_gb": np.max(memory_array)
            })

        if self.memory_percent:
            memory_percent_array = np.array(self.memory_percent)
            stats.update({
                "memory_percent": memory_percent_array[-1] if self.memory_percent else 0,
                "memory_percent_avg": np.mean(memory_percent_array),
                "memory_percent_max": np.max(memory_percent_array)
            })

        # Calculate pipeline efficiency
        if self.inference_times and self.capture_times and self.display_times:
            pipeline_time = (np.mean(self.inference_times) +
                           np.mean(self.capture_times) +
                           np.mean(self.display_times))
            ideal_frame_time = 1.0 / stats.get("fps_avg", 1.0)
            efficiency = min(100, (ideal_frame_time / pipeline_time) * 100) if pipeline_time > 0 else 0
            stats["pipeline_efficiency_percent"] = efficiency

        return stats

    def get_summary_string(self) -> str:
        """Get formatted performance summary"""
        stats = self.get_stats()

        summary = [
            f"Performance Summary (Uptime: {stats['uptime_seconds']:.1f}s)",
            f"FPS: {stats.get('fps_current', 0):.1f} (avg: {stats.get('fps_avg', 0):.1f})",
            f"Frames: {stats['total_frames']} (dropped: {stats['dropped_frames']}, {stats['drop_rate_percent']:.1f}%)",
            f"Inference: {stats.get('inference_time_avg_ms', 0):.1f}ms avg",
            f"CPU: {stats.get('cpu_usage_percent', 0):.1f}%, Memory: {stats.get('memory_percent', 0):.1f}%"
        ]

        return " | ".join(summary)

    def reset_stats(self):
        """Reset all performance statistics"""
        self.fps_history.clear()
        self.capture_times.clear()
        self.inference_times.clear()
        self.display_times.clear()
        self.total_times.clear()
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.memory_percent.clear()

        self.total_frames = 0
        self.dropped_frames = 0
        self.start_time = time.time()

        self.logger.info("Performance statistics reset")

    def export_stats(self, filename: str) -> bool:
        """Export performance statistics to file"""
        try:
            stats = self.get_stats()

            # Convert numpy arrays to lists for JSON serialization
            export_data = {}
            for key, value in stats.items():
                if hasattr(value, 'tolist'):
                    export_data[key] = value.tolist()
                else:
                    export_data[key] = value

            # Add raw data arrays
            export_data['raw_data'] = {
                'fps_history': list(self.fps_history),
                'capture_times': list(self.capture_times),
                'inference_times': list(self.inference_times),
                'display_times': list(self.display_times),
                'total_times': list(self.total_times),
                'cpu_usage': list(self.cpu_usage),
                'memory_usage': list(self.memory_usage),
                'memory_percent': list(self.memory_percent)
            }

            import json
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Performance statistics exported to {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting statistics: {e}")
            return False

    def check_performance_alerts(self) -> List[str]:
        """Check for performance issues and return alerts"""
        alerts = []
        stats = self.get_stats()

        # FPS alerts
        current_fps = stats.get('fps_current', 0)
        if current_fps < 15:
            alerts.append(f"Low FPS: {current_fps:.1f} (target: 15+)")
        elif current_fps < 10:
            alerts.append(f"Very Low FPS: {current_fps:.1f}")

        # Drop rate alerts
        drop_rate = stats.get('drop_rate_percent', 0)
        if drop_rate > 10:
            alerts.append(f"High frame drop rate: {drop_rate:.1f}%")
        elif drop_rate > 5:
            alerts.append(f"Elevated frame drop rate: {drop_rate:.1f}%")

        # Inference time alerts
        inference_time = stats.get('inference_time_avg_ms', 0)
        if inference_time > 100:
            alerts.append(f"High inference time: {inference_time:.1f}ms")
        elif inference_time > 50:
            alerts.append(f"Elevated inference time: {inference_time:.1f}ms")

        # CPU alerts
        cpu_usage = stats.get('cpu_usage_percent', 0)
        if cpu_usage > 90:
            alerts.append(f"Very high CPU usage: {cpu_usage:.1f}%")
        elif cpu_usage > 80:
            alerts.append(f"High CPU usage: {cpu_usage:.1f}%")

        # Memory alerts
        memory_percent = stats.get('memory_percent', 0)
        if memory_percent > 90:
            alerts.append(f"Very high memory usage: {memory_percent:.1f}%")
        elif memory_percent > 80:
            alerts.append(f"High memory usage: {memory_percent:.1f}%")

        return alerts

class FrameTimer:
    """Context manager for timing frame processing stages"""

    def __init__(self, monitor: PerformanceMonitor, stage: str):
        self.monitor = monitor
        self.stage = stage
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            if self.stage == "capture":
                self.monitor.log_capture_time(duration)
            elif self.stage == "inference":
                self.monitor.log_inference_time(duration)
            elif self.stage == "display":
                self.monitor.log_display_time(duration)
            elif self.stage == "total":
                self.monitor.log_total_time(duration)

class LatencyProfiler:
    """Profiler for measuring end-to-end latency"""

    def __init__(self):
        self.timestamps = {}

    def mark(self, event: str):
        """Mark a timestamp for an event"""
        self.timestamps[event] = time.time()

    def get_latency(self, start_event: str, end_event: str) -> Optional[float]:
        """Get latency between two events"""
        if start_event in self.timestamps and end_event in self.timestamps:
            return self.timestamps[end_event] - self.timestamps[start_event]
        return None

    def get_all_latencies(self) -> Dict[str, float]:
        """Get all measured latencies"""
        latencies = {}
        events = list(self.timestamps.keys())

        for i in range(len(events) - 1):
            start = events[i]
            end = events[i + 1]
            latency = self.get_latency(start, end)
            if latency is not None:
                latencies[f"{start}_to_{end}"] = latency

        return latencies

    def reset(self):
        """Reset all timestamps"""
        self.timestamps.clear()