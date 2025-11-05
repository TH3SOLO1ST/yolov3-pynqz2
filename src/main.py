"""
Realtime YOLOv3 on PYNQ-Z2 using DPU acceleration
Main application entry point for USB webcam capture, DPU inference, and HDMI output
"""

import sys
import os
import time
import threading
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from config.settings import Config
from camera.usb_camera import USBCamera
from inference.dpu_inference import DPUInference
from display.hdmi_output import HDMIOutput
from utils.logger import setup_logger
from utils.performance import PerformanceMonitor

class YOLOv3Realtime:
    def __init__(self, config_path=None):
        """Initialize the YOLOv3 realtime detection system"""
        self.config = Config(config_path)
        self.logger = setup_logger(__name__)

        # Initialize performance monitor
        self.perf_monitor = PerformanceMonitor()

        # Initialize components
        self.camera = None
        self.inference = None
        self.display = None

        # Threading controls
        self.running = False
        self.capture_thread = None
        self.inference_thread = None
        self.display_thread = None

        # Frame buffers for pipeline
        self.frame_buffer = []
        self.result_buffer = []
        self.buffer_lock = threading.Lock()

        self.logger.info("YOLOv3 Realtime System initialized")

    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")

            # Initialize USB camera
            self.camera = USBCamera(self.config)
            self.camera.initialize()
            self.logger.info("USB camera initialized")

            # Initialize DPU inference engine
            self.inference = DPUInference(self.config)
            self.inference.load_model()
            self.logger.info("DPU inference engine initialized")

            # Initialize HDMI display
            self.display = HDMIOutput(self.config)
            self.display.initialize()
            self.logger.info("HDMI display initialized")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False

    def capture_worker(self):
        """Camera capture thread worker"""
        while self.running:
            try:
                start_time = time.time()

                # Capture frame from USB camera
                frame = self.camera.capture_frame()
                if frame is not None:
                    with self.buffer_lock:
                        self.frame_buffer.append(frame)
                        # Limit buffer size to prevent memory issues
                        if len(self.frame_buffer) > self.config.BUFFER_SIZE:
                            self.frame_buffer.pop(0)

                # Log capture performance
                capture_time = time.time() - start_time
                self.perf_monitor.log_capture_time(capture_time)

            except Exception as e:
                self.logger.error(f"Camera capture error: {e}")
                time.sleep(0.01)  # Prevent tight loop on error

    def inference_worker(self):
        """DPU inference thread worker"""
        while self.running:
            try:
                if len(self.frame_buffer) > 0:
                    start_time = time.time()

                    # Get frame for inference
                    with self.buffer_lock:
                        if len(self.frame_buffer) > 0:
                            frame = self.frame_buffer.pop(0)
                        else:
                            continue

                    # Run DPU inference
                    detections = self.inference.infer(frame)

                    # Store results
                    with self.buffer_lock:
                        self.result_buffer.append((frame, detections))
                        # Limit result buffer
                        if len(self.result_buffer) > self.config.BUFFER_SIZE:
                            self.result_buffer.pop(0)

                    # Log inference performance
                    inference_time = time.time() - start_time
                    self.perf_monitor.log_inference_time(inference_time)

                else:
                    time.sleep(0.001)  # Short sleep when no frames

            except Exception as e:
                self.logger.error(f"Inference error: {e}")
                time.sleep(0.01)

    def display_worker(self):
        """Display output thread worker"""
        last_fps_update = time.time()
        frame_count = 0

        while self.running:
            try:
                if len(self.result_buffer) > 0:
                    start_time = time.time()

                    # Get results for display
                    with self.buffer_lock:
                        if len(self.result_buffer) > 0:
                            frame, detections = self.result_buffer.pop(0)
                        else:
                            continue

                    # Display frame with detections
                    self.display.display_frame(frame, detections)

                    # Update FPS counter
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_fps_update >= 1.0:
                        fps = frame_count / (current_time - last_fps_update)
                        self.perf_monitor.update_fps(fps)
                        frame_count = 0
                        last_fps_update = current_time

                    # Log display performance
                    display_time = time.time() - start_time
                    self.perf_monitor.log_display_time(display_time)

                else:
                    time.sleep(0.001)  # Short sleep when no results

            except Exception as e:
                self.logger.error(f"Display error: {e}")
                time.sleep(0.01)

    def start(self):
        """Start the realtime detection system"""
        if not self.initialize_components():
            return False

        self.logger.info("Starting YOLOv3 realtime detection...")
        self.running = True

        try:
            # Start worker threads
            self.capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
            self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
            self.display_thread = threading.Thread(target=self.display_worker, daemon=True)

            self.capture_thread.start()
            self.inference_thread.start()
            self.display_thread.start()

            self.logger.info("All threads started successfully")

            # Main monitoring loop
            try:
                while self.running:
                    # Print performance statistics every 10 seconds
                    time.sleep(10)
                    stats = self.perf_monitor.get_stats()
                    self.logger.info(f"Performance: {stats}")

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                self.stop()

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop()
            return False

        return True

    def stop(self):
        """Stop the realtime detection system"""
        self.logger.info("Stopping YOLOv3 realtime detection...")
        self.running = False

        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.inference_thread:
            self.inference_thread.join(timeout=2)
        if self.display_thread:
            self.display_thread.join(timeout=2)

        # Cleanup components
        if self.camera:
            self.camera.cleanup()
        if self.inference:
            self.inference.cleanup()
        if self.display:
            self.display.cleanup()

        self.logger.info("System stopped successfully")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Realtime YOLOv3 on PYNQ-Z2')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Create and start system
    system = YOLOv3Realtime(args.config)

    try:
        success = system.start()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()