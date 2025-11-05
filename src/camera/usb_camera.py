"""
USB Webcam capture module for YOLOv3 realtime system
Handles video capture from USB webcams using OpenCV
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
import logging

class USBCamera:
    """USB webcam capture class"""

    def __init__(self, config):
        """Initialize USB camera with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.camera = None
        self.is_initialized = False

        # Camera settings
        self.camera_index = config.CAMERA_INDEX
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT
        self.fps = config.CAMERA_FPS
        self.format = config.CAMERA_FORMAT

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0.0

        self.logger.info(f"USB Camera initialized with index {self.camera_index}")

    def initialize(self) -> bool:
        """Initialize the USB camera"""
        try:
            self.logger.info(f"Initializing USB camera {self.camera_index}...")

            # Create VideoCapture object
            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            # Try to set pixel format if supported
            if self.format.upper() == "MJPEG":
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            elif self.format.upper() == "YUYV":
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps:.2f}fps")

            # Test capture
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture test frame")

            self.logger.info(f"Test frame captured: {frame.shape}")

            self.is_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            self.cleanup()
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the USB camera"""
        if not self.is_initialized or self.camera is None:
            self.logger.error("Camera not initialized")
            return None

        try:
            # Read frame from camera
            ret, frame = self.camera.read()

            if not ret or frame is None:
                self.logger.warning("Failed to capture frame")
                return None

            # Update FPS counter
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.actual_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

            return frame

        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None

    def get_camera_info(self) -> dict:
        """Get camera information and current settings"""
        if not self.is_initialized or self.camera is None:
            return {"status": "not_initialized"}

        try:
            info = {
                "status": "initialized",
                "camera_index": self.camera_index,
                "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.camera.get(cv2.CAP_PROP_FPS),
                "actual_fps": self.actual_fps,
                "brightness": self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.camera.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.camera.get(cv2.CAP_PROP_SATURATION),
                "exposure": self.camera.get(cv2.CAP_PROP_EXPOSURE),
                "auto_exposure": self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            }

            # Get backend name
            backend = self.camera.getBackendName()
            info["backend"] = backend

            return info

        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            return {"status": "error", "error": str(e)}

    def set_camera_property(self, property_id: int, value: float) -> bool:
        """Set camera property"""
        if not self.is_initialized or self.camera is None:
            return False

        try:
            return self.camera.set(property_id, value)
        except Exception as e:
            self.logger.error(f"Error setting camera property {property_id}: {e}")
            return False

    def get_camera_property(self, property_id: int) -> float:
        """Get camera property value"""
        if not self.is_initialized or self.camera is None:
            return 0.0

        try:
            return self.camera.get(property_id)
        except Exception as e:
            self.logger.error(f"Error getting camera property {property_id}: {e}")
            return 0.0

    def auto_adjust_exposure(self, frame: np.ndarray) -> bool:
        """Automatically adjust camera exposure based on frame brightness"""
        if not self.is_initialized:
            return False

        try:
            # Calculate average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)

            # Target brightness (middle of range)
            target_brightness = 128

            # Adjust exposure if auto exposure is available
            auto_exposure = self.get_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE)

            if auto_exposure == 1.0:  # Auto exposure is enabled
                # Let camera handle it
                return True
            else:
                # Manual exposure adjustment
                current_exposure = self.get_camera_property(cv2.CAP_PROP_EXPOSURE)

                # Simple proportional control
                brightness_error = target_brightness - avg_brightness
                exposure_adjustment = brightness_error * 0.1

                new_exposure = current_exposure + exposure_adjustment
                new_exposure = max(0.1, min(1.0, new_exposure))  # Clamp to valid range

                return self.set_camera_property(cv2.CAP_PROP_EXPOSURE, new_exposure)

        except Exception as e:
            self.logger.error(f"Error in auto exposure adjustment: {e}")
            return False

    def reset_camera(self) -> bool:
        """Reset camera to default settings"""
        if not self.is_initialized:
            return False

        try:
            # Reset camera by releasing and re-initializing
            self.camera.release()
            time.sleep(0.1)

            self.camera = cv2.VideoCapture(self.camera_index)
            self.is_initialized = self.camera.isOpened()

            if self.is_initialized:
                # Re-apply settings
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            return self.is_initialized

        except Exception as e:
            self.logger.error(f"Error resetting camera: {e}")
            return False

    def list_available_cameras(self) -> list:
        """List available USB cameras"""
        available_cameras = []

        # Try camera indices 0-9
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to get some info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if width > 0 and height > 0:
                        available_cameras.append({
                            "index": i,
                            "width": width,
                            "height": height,
                            "backend": cap.getBackendName()
                        })

                    cap.release()

            except Exception:
                continue

        return available_cameras

    def validate_camera(self, camera_index: int) -> bool:
        """Validate if a camera index is working"""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False

            # Try to read a frame
            ret, frame = cap.read()
            cap.release()

            return ret and frame is not None

        except Exception:
            return False

    def cleanup(self):
        """Cleanup camera resources"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

            self.is_initialized = False
            self.logger.info("Camera resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error during camera cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()