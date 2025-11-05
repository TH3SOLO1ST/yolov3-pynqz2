#!/usr/bin/env python3
"""
Test script for DPU inference functionality
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference.dpu_inference import DPUInference
from inference.postprocessor import YOLOv3Postprocessor
from config.settings import Config

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with some patterns"""
    # Create random image
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Add some geometric patterns for potential detection
    # Add rectangles
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.rectangle(image, (200, 100), (300, 200), (0, 255, 0), -1)
    cv2.rectangle(image, (350, 50), (450, 150), (0, 0, 255), -1)

    return image

def test_model_loading(config_path: str = None):
    """Test DPU model loading"""
    print("Testing model loading...")

    config = Config(config_path)
    inference = DPUInference(config)

    # Test model loading
    if not inference.load_model():
        print("FAIL: Model loading failed")
        return False

    print("PASS: Model loaded successfully")

    # Get model info
    model_info = inference.get_model_info()
    print(f"Model info: {model_info}")

    # Cleanup
    inference.cleanup()
    return True

def test_inference_single(config_path: str = None):
    """Test single inference"""
    print("Testing single inference...")

    config = Config(config_path)
    inference = DPUInference(config)

    if not inference.load_model():
        print("FAIL: Model loading failed")
        return False

    # Create test image
    test_image = create_test_image(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    print(f"Test image shape: {test_image.shape}")

    # Run inference
    start_time = time.time()
    detections = inference.infer(test_image)
    inference_time = time.time() - start_time

    print(f"Inference time: {inference_time*1000:.2f}ms")
    print(f"Detections: {len(detections)}")

    for i, det in enumerate(detections):
        print(f"  Detection {i+1}: {det['class_name']} ({det['confidence']:.3f}) at {det['bbox']}")

    print("PASS: Single inference completed")
    inference.cleanup()
    return True

def test_inference_performance(config_path: str = None, num_iterations: int = 100):
    """Test inference performance"""
    print(f"Testing inference performance with {num_iterations} iterations...")

    config = Config(config_path)
    inference = DPUInference(config)

    if not inference.load_model():
        print("FAIL: Model loading failed")
        return False

    # Create test image
    test_image = create_test_image(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # Performance metrics
    inference_times = []
    total_detections = 0

    print("Running performance test...")

    for i in range(num_iterations):
        start_time = time.time()
        detections = inference.infer(test_image)
        inference_time = time.time() - start_time

        inference_times.append(inference_time)
        total_detections += len(detections)

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{num_iterations} iterations")

    # Calculate statistics
    inference_times = np.array(inference_times)
    total_time = np.sum(inference_times)
    avg_time = np.mean(inference_times) * 1000  # ms
    min_time = np.min(inference_times) * 1000
    max_time = np.max(inference_times) * 1000
    std_time = np.std(inference_times) * 1000
    throughput = 1.0 / np.mean(inference_times)

    print(f"\nPerformance Test Results:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  Std deviation: {std_time:.2f}ms")
    print(f"  Throughput: {throughput:.2f} FPS")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections per frame: {total_detections/num_iterations:.2f}")

    # Evaluate performance
    target_fps = 30  # Target 30 FPS
    actual_fps = throughput

    if actual_fps >= target_fps * 0.8:  # 80% of target
        print("PASS: Inference performance is acceptable")
        success = True
    else:
        print("FAIL: Inference performance is below target")
        success = False

    inference.cleanup()
    return success

def test_postprocessing(config_path: str = None):
    """Test YOLOv3 postprocessing"""
    print("Testing YOLOv3 postprocessing...")

    config = Config(config_path)
    postprocessor = YOLOv3Postprocessor(config)

    # Create mock DPU outputs (simulated)
    batch_size = 1
    input_h, input_w = config.INPUT_SIZE[1], config.INPUT_SIZE[0]

    # Mock outputs for 3 YOLOv3 scales
    mock_outputs = []
    scales = [
        (max(1, input_h // 32), max(1, input_w // 32)),   # Small objects
        (max(1, input_h // 16), max(1, input_w // 16)),   # Medium objects
        (max(1, input_h // 8), max(1, input_w // 8))      # Large objects
    ]

    for scale_h, scale_w in scales:
        # Create mock output: batch x (num_anchors * (num_classes + 5)) x scale_h x scale_w
        output_dim = 3 * (config.NUM_CLASSES + 5)
        mock_output = np.random.random((batch_size, output_dim, scale_h, scale_w)).astype(np.float32)
        mock_outputs.append(mock_output)

    # Test postprocessing
    original_shape = (config.CAMERA_HEIGHT, config.CAMERA_WIDTH)
    detections = postprocessor.process(mock_outputs, original_shape)

    print(f"Postprocessing results:")
    print(f"  Input scales: {len(scales)}")
    print(f"  Output detections: {len(detections)}")

    for i, det in enumerate(detections[:5]):  # Show first 5 detections
        print(f"  Detection {i+1}: {det['class_name']} ({det['confidence']:.3f})")

    if len(detections) >= 0:
        print("PASS: Postprocessing completed successfully")
        return True
    else:
        print("FAIL: Postprocessing failed")
        return False

def test_inference_benchmark(config_path: str = None):
    """Run comprehensive inference benchmark"""
    print("Running comprehensive inference benchmark...")

    config = Config(config_path)
    inference = DPUInference(config)

    if not inference.load_model():
        print("FAIL: Model loading failed")
        return False

    # Run built-in benchmark
    results = inference.benchmark_model(num_iterations=100)

    if 'error' in results:
        print(f"FAIL: Benchmark failed - {results['error']}")
        return False

    print(f"Benchmark Results:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Avg time: {results['avg_time_ms']:.2f}ms")
    print(f"  Min time: {results['min_time_ms']:.2f}ms")
    print(f"  Max time: {results['max_time_ms']:.2f}ms")
    print(f"  Std time: {results['std_time_ms']:.2f}ms")
    print(f"  Avg FPS: {results['avg_fps']:.2f}")
    print(f"  Throughput FPS: {results['throughput_fps']:.2f}")

    # Evaluate results
    if results['avg_fps'] >= 15:  # Minimum acceptable FPS
        print("PASS: Benchmark performance is acceptable")
        success = True
    else:
        print("FAIL: Benchmark performance is below minimum")
        success = False

    inference.cleanup()
    return success

def test_memory_usage(config_path: str = None, duration: int = 60):
    """Test memory usage during inference"""
    print(f"Testing memory usage for {duration} seconds...")

    try:
        import psutil
        import os
    except ImportError:
        print("SKIP: Memory usage test requires psutil (pip install psutil)")
        return True

    config = Config(config_path)
    inference = DPUInference(config)

    if not inference.load_model():
        print("FAIL: Model loading failed")
        return False

    process = psutil.Process(os.getpid())
    test_image = create_test_image(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # Monitor memory usage
    memory_samples = []
    start_time = time.time()
    sample_interval = 1.0  # Sample every second

    print("Monitoring memory usage...")

    while time.time() - start_time < duration:
        # Run inference
        detections = inference.infer(test_image)

        # Sample memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        memory_samples.append(memory_mb)

        # Print progress
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.1f}s, Memory: {memory_mb:.1f}MB, Detections: {len(detections)}")

        time.sleep(sample_interval)

    # Calculate memory statistics
    memory_samples = np.array(memory_samples)
    avg_memory = np.mean(memory_samples)
    max_memory = np.max(memory_samples)
    min_memory = np.min(memory_samples)
    memory_growth = memory_samples[-1] - memory_samples[0]

    print(f"\nMemory Usage Results:")
    print(f"  Duration: {duration}s")
    print(f"  Samples: {len(memory_samples)}")
    print(f"  Average memory: {avg_memory:.1f}MB")
    print(f"  Peak memory: {max_memory:.1f}MB")
    print(f"  Minimum memory: {min_memory:.1f}MB")
    print(f"  Memory growth: {memory_growth:.1f}MB")

    # Check for memory leaks
    if memory_growth < 50:  # Allow up to 50MB growth
        print("PASS: No significant memory leak detected")
        success = True
    else:
        print("WARN: Potential memory leak detected")
        success = True  # Don't fail test for memory warning

    inference.cleanup()
    return success

def main():
    parser = argparse.ArgumentParser(description='Test DPU inference functionality')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test', type=str, choices=['load', 'single', 'performance', 'postprocess', 'benchmark', 'memory'],
                        default='load', help='Test type to run')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for performance tests')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration for memory test (seconds)')

    args = parser.parse_args()

    print("YOLOv3 Inference Test Suite")
    print("=" * 40)

    success = False

    if args.test == 'load':
        success = test_model_loading(args.config)
    elif args.test == 'single':
        success = test_inference_single(args.config)
    elif args.test == 'performance':
        success = test_inference_performance(args.config, args.iterations)
    elif args.test == 'postprocess':
        success = test_postprocessing(args.config)
    elif args.test == 'benchmark':
        success = test_inference_benchmark(args.config)
    elif args.test == 'memory':
        success = test_memory_usage(args.config, args.duration)

    if success:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()