"""
DPU inference engine for YOLOv3 realtime system
Handles DPU model loading, preprocessing, and inference execution
"""

import numpy as np
import cv2
import time
import logging
import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# PYNQ imports for DPU
try:
    import pynq
    from pynq import Overlay
    from pynq.dpu import DPU
except ImportError:
    print("Warning: PYNQ DPU libraries not available. Using simulation mode.")
    DPU = None

from .postprocessor import YOLOv3Postprocessor

class DPUInference:
    """DPU inference engine for YOLOv3"""

    def __init__(self, config):
        """Initialize DPU inference engine"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # DPU-related attributes
        self.overlay = None
        self.dpu = None
        self.model_loaded = False
        self.simulation_mode = False

        # Model attributes
        self.model_path = config.MODEL_PATH
        self.input_shape = (config.INPUT_SIZE[1], config.INPUT_SIZE[0])  # H, W
        self.input_size = config.INPUT_SIZE  # W, H
        self.num_classes = config.NUM_CLASSES

        # Postprocessor
        self.postprocessor = YOLOv3Postprocessor(config)

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0

        self.logger.info(f"DPU inference engine initialized for model: {self.model_path}")

    def load_model(self) -> bool:
        """Load DPU model"""
        try:
            self.logger.info("Loading DPU model...")

            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Try to load PYNQ overlay and DPU
            if DPU is not None:
                # Load overlay with DPU
                overlay_path = os.path.splitext(self.model_path)[0] + ".hwh"
                if os.path.exists(overlay_path):
                    self.overlay = Overlay(overlay_path)
                    self.logger.info(f"Loaded overlay: {overlay_path}")

                    # Load DPU
                    self.dpu = self.overlay.dpu
                    self.logger.info("DPU loaded successfully")

                    # Configure DPU
                    self._configure_dpu()
                else:
                    self.logger.warning(f"Overlay file not found: {overlay_path}")
                    self.simulation_mode = True
            else:
                self.simulation_mode = True

            if self.simulation_mode:
                self.logger.warning("Running in simulation mode - DPU hardware not available")
                self._setup_simulation()

            self.model_loaded = True
            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.simulation_mode = True
            self._setup_simulation()
            return False

    def _configure_dpu(self):
        """Configure DPU settings"""
        if self.dpu is None:
            return

        try:
            # Get DPU information
            self.logger.info(f"DPU Info: {self.dpu}")

            # Check input shape compatibility
            # Note: Actual DPU configuration depends on the compiled model
            self.logger.info(f"Expected input shape: {self.input_shape}")

        except Exception as e:
            self.logger.error(f"Error configuring DPU: {e}")

    def _setup_simulation(self):
        """Setup simulation mode for testing without DPU hardware"""
        self.logger.info("Setting up simulation mode")
        # In simulation mode, we'll use a mock inference result
        self.simulation_detections = []

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for DPU inference"""
        try:
            # Resize frame to input size
            resized = cv2.resize(frame, self.input_size)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1] range
            normalized = rgb_frame.astype(np.float32) / 255.0

            # Convert to model input format (CHW)
            # Input should be: (batch_size, channels, height, width)
            input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension

            return input_tensor

        except Exception as e:
            self.logger.error(f"Error preprocessing frame: {e}")
            return np.array([])

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run DPU inference on frame"""
        if not self.model_loaded:
            self.logger.error("Model not loaded")
            return []

        start_time = time.time()

        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            if input_tensor.size == 0:
                return []

            # Run inference
            if self.simulation_mode:
                raw_outputs = self._simulate_inference(input_tensor)
            else:
                raw_outputs = self._run_dpu_inference(input_tensor)

            # Postprocess results
            detections = self.postprocessor.process(raw_outputs, frame.shape[:2])

            # Update performance metrics
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time)

            return detections

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return []

    def _run_dpu_inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference on actual DPU hardware"""
        try:
            if self.dpu is None:
                raise RuntimeError("DPU not available")

            # Run DPU inference
            # Note: Actual DPU API call depends on PYNQ version and DPU configuration
            # This is a general template that may need adjustment

            # Prepare input buffer
            input_buffer = input_tensor.astype(np.float32)

            # Run inference
            outputs = self.dpu.inference(input_buffer)

            # Process outputs
            # YOLOv3 typically has 3 output tensors for different scales
            processed_outputs = []
            for output in outputs:
                processed_outputs.append(output)

            return processed_outputs

        except Exception as e:
            self.logger.error(f"DPU inference error: {e}")
            return []

    def _simulate_inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Simulate DPU inference for testing"""
        try:
            # Create mock output tensors based on YOLOv3 architecture
            batch_size = input_tensor.shape[0]

            # YOLOv3 output shapes for different scales
            # Scale 1 (small objects): batch x 255 x 13 x 13 (for 416x416 input)
            # Scale 2 (medium objects): batch x 255 x 26 x 26
            # Scale 3 (large objects): batch x 255 x 52 x 52

            h, w = self.input_shape
            output_scales = [
                (max(1, h // 32), max(1, w // 32)),   # Small objects
                (max(1, h // 16), max(1, w // 16)),   # Medium objects
                (max(1, h // 8), max(1, w // 8))      # Large objects
            ]

            outputs = []
            for scale_h, scale_w in output_scales:
                # Create mock output: batch x (num_anchors * (num_classes + 5)) x scale_h x scale_w
                # For YOLOv3: 3 anchors per scale, (classes + 5) values per anchor
                output_dim = 3 * (self.num_classes + 5)
                mock_output = np.random.random((batch_size, output_dim, scale_h, scale_w)).astype(np.float32)
                outputs.append(mock_output)

            # Simulate some processing time
            time.sleep(0.02)  # 20ms simulated inference time

            return outputs

        except Exception as e:
            self.logger.error(f"Simulation inference error: {e}")
            return []

    def _update_performance_stats(self, inference_time: float):
        """Update inference performance statistics"""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.avg_inference_time = self.total_inference_time / self.inference_count

        # Log performance every 100 inferences
        if self.inference_count % 100 == 0:
            fps = 1.0 / self.avg_inference_time if self.avg_inference_time > 0 else 0
            self.logger.info(f"Inference performance: {self.avg_inference_time*1000:.2f}ms avg, {fps:.2f} FPS")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics"""
        return {
            "inference_count": self.inference_count,
            "avg_inference_time_ms": self.avg_inference_time * 1000,
            "avg_fps": 1.0 / self.avg_inference_time if self.avg_inference_time > 0 else 0,
            "simulation_mode": self.simulation_mode,
            "model_loaded": self.model_loaded
        }

    def benchmark_model(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model performance"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}

        self.logger.info(f"Running benchmark with {num_iterations} iterations...")

        # Create random input tensor
        input_tensor = np.random.random((1, 3, self.input_shape[0], self.input_shape[1])).astype(np.float32)

        inference_times = []

        try:
            for i in range(num_iterations):
                start_time = time.time()

                if self.simulation_mode:
                    self._simulate_inference(input_tensor)
                else:
                    self._run_dpu_inference(input_tensor)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                if (i + 1) % 20 == 0:
                    self.logger.info(f"Benchmark progress: {i + 1}/{num_iterations}")

            # Calculate statistics
            inference_times = np.array(inference_times)
            stats = {
                "iterations": num_iterations,
                "avg_time_ms": np.mean(inference_times) * 1000,
                "min_time_ms": np.min(inference_times) * 1000,
                "max_time_ms": np.max(inference_times) * 1000,
                "std_time_ms": np.std(inference_times) * 1000,
                "avg_fps": 1.0 / np.mean(inference_times),
                "throughput_fps": num_iterations / np.sum(inference_times)
            }

            self.logger.info(f"Benchmark completed: {stats['avg_time_ms']:.2f}ms avg, {stats['avg_fps']:.2f} FPS")
            return stats

        except Exception as e:
            self.logger.error(f"Benchmark error: {e}")
            return {"error": str(e)}

    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "input_shape": self.input_shape,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "model_loaded": self.model_loaded,
            "simulation_mode": self.simulation_mode,
            "dpu_available": DPU is not None,
            "overlay_loaded": self.overlay is not None
        }

    def cleanup(self):
        """Cleanup DPU resources"""
        try:
            if self.dpu is not None:
                # DPU cleanup handled by overlay
                pass

            if self.overlay is not None:
                self.overlay.free()
                self.overlay = None

            self.model_loaded = False
            self.logger.info("DPU resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error during DPU cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()