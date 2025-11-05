#!/usr/bin/env python3
"""
Test script for USB camera functionality
"""

import sys
import time
import argparse
from pathlib import Path
import cv2

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from camera.usb_camera import USBCamera
from config.settings import Config

def test_camera_basic(config_path: str = None):
    """Test basic camera functionality"""
    print("Testing basic camera functionality...")

    config = Config(config_path)
    camera = USBCamera(config)

    # Initialize camera
    if not camera.initialize():
        print("FAIL: Camera initialization failed")
        return False

    print("PASS: Camera initialized successfully")

    # Get camera info
    info = camera.get_camera_info()
    print(f"Camera info: {info}")

    # Test frame capture
    print("Testing frame capture...")
    frame_count = 0
    start_time = time.time()

    for i in range(10):
        frame = camera.capture_frame()
        if frame is not None:
            frame_count += 1
            print(f"  Frame {i+1}: {frame.shape}")
        else:
            print(f"  Frame {i+1}: Failed to capture")

    capture_time = time.time() - start_time
    fps = frame_count / capture_time if capture_time > 0 else 0

    print(f"Captured {frame_count}/10 frames in {capture_time:.2f}s ({fps:.1f} FPS)")

    if frame_count >= 8:  # Allow for some frame drops
        print("PASS: Frame capture test successful")
    else:
        print("FAIL: Too many frame drops")
        return False

    # Test camera settings
    print("Testing camera settings...")
    original_brightness = camera.get_camera_property(cv2.CAP_PROP_BRIGHTNESS)
    camera.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    new_brightness = camera.get_camera_property(cv2.CAP_PROP_BRIGHTNESS)

    if abs(new_brightness - 0.5) < 0.1:
        print("PASS: Camera settings adjustment successful")
    else:
        print("WARN: Camera settings may not be adjustable")

    # Cleanup
    camera.cleanup()
    print("Camera test completed")
    return True

def test_camera_performance(config_path: str = None, duration: int = 30):
    """Test camera performance over extended period"""
    print(f"Testing camera performance for {duration} seconds...")

    config = Config(config_path)
    camera = USBCamera(config)

    if not camera.initialize():
        print("FAIL: Camera initialization failed")
        return False

    # Performance metrics
    frame_count = 0
    drop_count = 0
    start_time = time.time()

    print("Starting performance test...")
    print("Press Ctrl+C to stop early")

    try:
        while time.time() - start_time < duration:
            frame = camera.capture_frame()
            if frame is not None:
                frame_count += 1
            else:
                drop_count += 1

            # Print progress every 5 seconds
            if frame_count % (config.CAMERA_FPS * 5) == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                drop_rate = (drop_count / (frame_count + drop_count)) * 100 if (frame_count + drop_count) > 0 else 0
                print(f"  Time: {elapsed:.1f}s, FPS: {current_fps:.1f}, Drops: {drop_count} ({drop_rate:.1f}%)")

            time.sleep(1.0 / config.CAMERA_FPS)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    # Calculate final metrics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    drop_rate = (drop_count / (frame_count + drop_count)) * 100 if (frame_count + drop_count) > 0 else 0

    print(f"\nPerformance Test Results:")
    print(f"  Duration: {total_time:.1f}s")
    print(f"  Frames captured: {frame_count}")
    print(f"  Frames dropped: {drop_count}")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Drop rate: {drop_rate:.1f}%")

    # Evaluate performance
    if avg_fps >= config.CAMERA_FPS * 0.8 and drop_rate < 5:
        print("PASS: Camera performance is acceptable")
        success = True
    else:
        print("FAIL: Camera performance is below acceptable levels")
        success = False

    camera.cleanup()
    return success

def test_camera_list():
    """Test camera enumeration"""
    print("Testing camera enumeration...")

    config = Config()
    camera = USBCamera(config)

    available_cameras = camera.list_available_cameras()

    print(f"Found {len(available_cameras)} cameras:")
    for i, cam in enumerate(available_cameras):
        print(f"  Camera {i}: Index {cam['index']}, {cam['width']}x{cam['height']}, Backend: {cam['backend']}")

    if len(available_cameras) > 0:
        print("PASS: Camera enumeration successful")
        return True
    else:
        print("WARN: No cameras found")
        return False

def test_camera_preview(config_path: str = None, duration: int = 10):
    """Test camera with live preview"""
    print(f"Testing camera preview for {duration} seconds...")

    config = Config(config_path)
    camera = USBCamera(config)

    if not camera.initialize():
        print("FAIL: Camera initialization failed")
        return False

    print("Displaying camera preview (press 'q' to quit)")

    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < duration:
            frame = camera.capture_frame()
            if frame is not None:
                frame_count += 1
                cv2.imshow('Camera Preview', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Preview stopped by user")
                    break
            else:
                print("Failed to capture frame")
                break

    except KeyboardInterrupt:
        print("\nPreview interrupted by user")

    cv2.destroyAllWindows()
    camera.cleanup()

    actual_duration = time.time() - start_time
    avg_fps = frame_count / actual_duration if actual_duration > 0 else 0

    print(f"Preview Results:")
    print(f"  Duration: {actual_duration:.1f}s")
    print(f"  Frames displayed: {frame_count}")
    print(f"  Average FPS: {avg_fps:.1f}")

    if frame_count > 0:
        print("PASS: Camera preview successful")
        return True
    else:
        print("FAIL: No frames displayed")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test USB camera functionality')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test', type=str, choices=['basic', 'performance', 'list', 'preview'],
                        default='basic', help='Test type to run')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration for performance/preview tests (seconds)')

    args = parser.parse_args()

    print("YOLOv3 Camera Test Suite")
    print("=" * 40)

    success = False

    if args.test == 'basic':
        success = test_camera_basic(args.config)
    elif args.test == 'performance':
        success = test_camera_performance(args.config, args.duration)
    elif args.test == 'list':
        success = test_camera_list()
    elif args.test == 'preview':
        success = test_camera_preview(args.config, args.duration)

    if success:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()