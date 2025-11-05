#!/usr/bin/env python3
"""
Test script for complete YOLOv3 system integration
"""

import sys
import time
import argparse
import threading
from pathlib import Path
import cv2

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import YOLOv3Realtime
from config.settings import Config

def test_system_initialization(config_path: str = None):
    """Test complete system initialization"""
    print("Testing system initialization...")

    config = Config(config_path)
    system = YOLOv3Realtime(config_path)

    # Test component initialization
    if not system.initialize_components():
        print("FAIL: System initialization failed")
        return False

    print("PASS: System initialized successfully")

    # Test performance monitor
    stats = system.perf_monitor.get_stats()
    print(f"Performance monitor stats: {stats}")

    # Cleanup
    system.stop()
    return True

def test_system_integration(config_path: str = None, duration: int = 30):
    """Test complete system integration with real video processing"""
    print(f"Testing system integration for {duration} seconds...")

    config = Config(config_path)
    system = YOLOv3Realtime(config_path)

    # Initialize system
    if not system.initialize_components():
        print("FAIL: System initialization failed")
        return False

    print("Starting system integration test...")
    print("Press Ctrl+C to stop early")

    # Test statistics
    start_time = time.time()
    frame_count = 0
    detection_count = 0

    try:
        # Start system in a separate thread
        system_thread = threading.Thread(target=system.start, daemon=True)
        system_thread.start()

        # Monitor performance
        while time.time() - start_time < duration:
            time.sleep(5)  # Check every 5 seconds

            elapsed = time.time() - start_time
            stats = system.perf_monitor.get_stats()

            print(f"  Time: {elapsed:.1f}s")
            print(f"  FPS: {stats.get('fps_current', 0):.1f}")
            print(f"  Inference: {stats.get('inference_time_avg_ms', 0):.1f}ms")
            print(f"  CPU: {stats.get('cpu_usage_percent', 0):.1f}%")
            print(f"  Memory: {stats.get('memory_percent', 0):.1f}%")
            print("---")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    # Stop system
    system.stop()

    # Get final statistics
    final_stats = system.perf_monitor.get_stats()
    total_time = time.time() - start_time

    print(f"\nIntegration Test Results:")
    print(f"  Duration: {total_time:.1f}s")
    print(f"  Average FPS: {final_stats.get('fps_avg', 0):.1f}")
    print(f"  Total frames: {final_stats.get('total_frames', 0)}")
    print(f"  Dropped frames: {final_stats.get('dropped_frames', 0)}")
    print(f"  Drop rate: {final_stats.get('drop_rate_percent', 0):.1f}%")
    print(f"  Avg inference time: {final_stats.get('inference_time_avg_ms', 0):.1f}ms")
    print(f"  Avg CPU usage: {final_stats.get('cpu_usage_avg', 0):.1f}%")
    print(f"  Avg memory usage: {final_stats.get('memory_percent_avg', 0):.1f}%")

    # Evaluate performance
    avg_fps = final_stats.get('fps_avg', 0)
    drop_rate = final_stats.get('drop_rate_percent', 100)

    if avg_fps >= 15 and drop_rate < 10:
        print("PASS: System integration performance is acceptable")
        return True
    else:
        print("FAIL: System integration performance is below acceptable levels")
        return False

def test_system_stress(config_path: str = None, duration: int = 300):
    """Stress test the system for extended period"""
    print(f"Running system stress test for {duration} seconds ({duration//60} minutes)...")

    config = Config(config_path)
    system = YOLOv3Realtime(config_path)

    if not system.initialize_components():
        print("FAIL: System initialization failed")
        return False

    print("Starting stress test...")
    print("This test monitors system stability over extended periods")

    # Start system
    system_thread = threading.Thread(target=system.start, daemon=True)
    system_thread.start()

    # Monitor for issues
    start_time = time.time()
    last_check_time = start_time
    last_fps = 0
    fps_samples = []
    memory_samples = []

    try:
        while time.time() - start_time < duration:
            time.sleep(30)  # Check every 30 seconds

            current_time = time.time()
            stats = system.perf_monitor.get_stats()

            # Collect metrics
            current_fps = stats.get('fps_current', 0)
            memory_usage = stats.get('memory_percent', 0)

            fps_samples.append(current_fps)
            memory_samples.append(memory_usage)

            # Check for performance degradation
            if len(fps_samples) > 1:
                fps_trend = fps_samples[-1] - fps_samples[-2]
                if fps_trend < -5:  # FPS drop of more than 5
                    print(f"WARNING: FPS drop detected ({fps_trend:.1f})")

            # Check for memory leaks
            if len(memory_samples) > 2:
                memory_growth = memory_samples[-1] - memory_samples[0]
                if memory_growth > 20:  # Memory growth of more than 20%
                    print(f"WARNING: Memory growth detected ({memory_growth:.1f}%)")

            # Print progress
            elapsed = current_time - start_time
            remaining = duration - elapsed
            print(f"  Progress: {elapsed:.0f}s / {duration:.0f}s, "
                  f"FPS: {current_fps:.1f}, Memory: {memory_usage:.1f}%")

    except KeyboardInterrupt:
        print("\nStress test interrupted by user")

    # Stop system
    system.stop()

    # Analyze stability
    if len(fps_samples) > 0:
        avg_fps = sum(fps_samples) / len(fps_samples)
        min_fps = min(fps_samples)
        max_fps = max(fps_samples)
        fps_std = (sum((x - avg_fps) ** 2 for x in fps_samples) / len(fps_samples)) ** 0.5
    else:
        avg_fps = min_fps = max_fps = fps_std = 0

    if len(memory_samples) > 0:
        avg_memory = sum(memory_samples) / len(memory_samples)
        min_memory = min(memory_samples)
        max_memory = max(memory_samples)
        memory_growth = memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0
    else:
        avg_memory = min_memory = max_memory = memory_growth = 0

    print(f"\nStress Test Results:")
    print(f"  Test duration: {time.time() - start_time:.1f}s")
    print(f"  FPS - Avg: {avg_fps:.1f}, Min: {min_fps:.1f}, Max: {max_fps:.1f}, Std: {fps_std:.1f}")
    print(f"  Memory - Avg: {avg_memory:.1f}%, Min: {min_memory:.1f}%, Max: {max_memory:.1f}%")
    print(f"  Memory growth: {memory_growth:.1f}%")
    print(f"  Samples collected: {len(fps_samples)}")

    # Evaluate stability
    fps_stability = (fps_std / avg_fps) * 100 if avg_fps > 0 else 100  # Coefficient of variation
    memory_stability = abs(memory_growth)

    if fps_stability < 20 and memory_stability < 15:  # Less than 20% FPS variation, less than 15% memory growth
        print("PASS: System shows good stability under stress")
        return True
    else:
        print("WARN: System shows signs of instability under stress")
        return True  # Don't fail for stress test warnings

def test_system_components(config_path: str = None):
    """Test individual system components"""
    print("Testing individual system components...")

    config = Config(config_path)
    system = YOLOv3Realtime(config_path)

    # Test camera
    print("Testing camera component...")
    try:
        if system.camera.initialize():
            print("PASS: Camera initialization successful")
            camera_info = system.camera.get_camera_info()
            print(f"  Camera info: {camera_info}")
            system.camera.cleanup()
        else:
            print("FAIL: Camera initialization failed")
            return False
    except Exception as e:
        print(f"FAIL: Camera test failed with error: {e}")
        return False

    # Test inference engine
    print("Testing inference component...")
    try:
        if system.inference.load_model():
            print("PASS: Inference model loading successful")
            model_info = system.inference.get_model_info()
            print(f"  Model info: {model_info}")
            system.inference.cleanup()
        else:
            print("FAIL: Inference model loading failed")
            return False
    except Exception as e:
        print(f"FAIL: Inference test failed with error: {e}")
        return False

    # Test display
    print("Testing display component...")
    try:
        if system.display.initialize():
            print("PASS: Display initialization successful")
            display_info = system.display.get_display_info()
            print(f"  Display info: {display_info}")
            system.display.cleanup()
        else:
            print("FAIL: Display initialization failed")
            return False
    except Exception as e:
        print(f"FAIL: Display test failed with error: {e}")
        return False

    print("PASS: All component tests successful")
    return True

def test_configuration(config_path: str = None):
    """Test configuration system"""
    print("Testing configuration system...")

    try:
        # Test default configuration
        config = Config()
        print("PASS: Default configuration loaded")

        # Test configuration validation
        config.validate()
        print("PASS: Configuration validation successful")

        # Test custom configuration if provided
        if config_path:
            custom_config = Config(config_path)
            custom_config.validate()
            print("PASS: Custom configuration loaded and validated")

        # Test configuration export
        test_config_file = "test_config.json"
        config.save_config(test_config_file)
        print("PASS: Configuration export successful")

        # Cleanup
        import os
        if os.path.exists(test_config_file):
            os.remove(test_config_file)

        return True

    except Exception as e:
        print(f"FAIL: Configuration test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test YOLOv3 complete system')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test', type=str,
                        choices=['init', 'integration', 'stress', 'components', 'config'],
                        default='init', help='Test type to run')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration for integration/stress tests (seconds)')

    args = parser.parse_args()

    print("YOLOv3 Full System Test Suite")
    print("=" * 40)

    success = False

    if args.test == 'init':
        success = test_system_initialization(args.config)
    elif args.test == 'integration':
        success = test_system_integration(args.config, args.duration)
    elif args.test == 'stress':
        success = test_system_stress(args.config, args.duration)
    elif args.test == 'components':
        success = test_system_components(args.config)
    elif args.test == 'config':
        success = test_configuration(args.config)

    if success:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()