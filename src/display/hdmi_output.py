"""
HDMI display output module for YOLOv3 realtime system
Handles HDMI display with detection overlays and performance metrics
"""

import cv2
import numpy as np
import time
import logging
import os
from typing import List, Dict, Any, Optional, Tuple

# PYNQ display imports
try:
    import pynq
    from pynq.lib.video import HDMIOut
    from pynq import Overlay
except ImportError:
    print("Warning: PYNQ HDMI libraries not available. Using OpenCV display.")
    HDMIOut = None

class HDMIOutput:
    """HDMI display output with detection overlays"""

    def __init__(self, config):
        """Initialize HDMI output with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Display settings
        self.output_width = config.OUTPUT_WIDTH
        self.output_height = config.OUTPUT_HEIGHT
        self.show_fps = config.SHOW_FPS
        self.show_performance = config.SHOW_PERFORMANCE
        self.fullscreen = config.FULLSCREEN

        # Display state
        self.hdmi_out = None
        self.overlay_display = None
        self.is_initialized = False
        self.simulation_mode = False

        # Performance metrics
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.display_times = []

        # Font settings for overlays
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.line_thickness = 2

        self.logger.info(f"HDMI output initialized: {self.output_width}x{self.output_height}")

    def initialize(self) -> bool:
        """Initialize HDMI display"""
        try:
            self.logger.info("Initializing HDMI display...")

            # Try to initialize PYNQ HDMI output
            if HDMIOut is not None:
                try:
                    # Look for HDMI output
                    if hasattr(pynq, 'Video') and hasattr(pynq.Video, 'HDMI_out'):
                        self.hdmi_out = pynq.Video.HDMI_out
                        self.hdmi_out.configure(self.output_width, self.output_height)
                        self.logger.info("PYNQ HDMI output configured")
                    else:
                        self.simulation_mode = True
                        self.logger.warning("PYNQ HDMI not available, using simulation mode")
                except Exception as e:
                    self.logger.warning(f"PYNQ HDMI initialization failed: {e}")
                    self.simulation_mode = True
            else:
                self.simulation_mode = True

            if self.simulation_mode:
                self.logger.info("Using OpenCV window display mode")
                cv2.namedWindow('YOLOv3 Detection', cv2.WINDOW_NORMAL)
                if self.fullscreen:
                    cv2.setWindowProperty('YOLOv3 Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            self.is_initialized = True
            self.logger.info("HDMI display initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize HDMI display: {e}")
            return False

    def display_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> bool:
        """Display frame with detection overlays"""
        if not self.is_initialized:
            self.logger.error("Display not initialized")
            return False

        try:
            start_time = time.time()

            # Resize frame to output resolution
            display_frame = self._resize_frame(frame)

            # Draw detection overlays
            display_frame = self._draw_detections(display_frame, detections)

            # Draw performance overlays
            if self.show_fps or self.show_performance:
                display_frame = self._draw_performance_info(display_frame)

            # Display frame
            if self.simulation_mode:
                self._display_opencv(display_frame)
            else:
                self._display_hdmi(display_frame)

            # Update performance metrics
            display_time = time.time() - start_time
            self._update_display_performance(display_time)

            return True

        except Exception as e:
            self.logger.error(f"Error displaying frame: {e}")
            return False

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to output resolution"""
        try:
            # Calculate scaling to maintain aspect ratio
            frame_h, frame_w = frame.shape[:2]
            target_h, target_w = self.output_height, self.output_width

            # Calculate scaling factor
            scale = min(target_w / frame_w, target_h / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)

            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h))

            # Create output frame with letterboxing if needed
            output_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Calculate position to center the resized frame
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            # Place resized frame in center
            output_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            return output_frame

        except Exception as e:
            self.logger.error(f"Error resizing frame: {e}")
            return np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes and labels on frame"""
        try:
            display_frame = frame.copy()

            # Calculate scaling factors for coordinate conversion
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / self.config.CAMERA_WIDTH
            scale_y = frame_h / self.config.CAMERA_HEIGHT

            for det in detections:
                bbox = det['bbox']
                confidence = det['confidence']
                class_name = det['class_name']
                class_id = det['class_id']

                # Scale bounding box coordinates
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)

                # Get color for this class
                color = self.config.get_color_for_class(class_id)

                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, self.line_thickness)

                # Prepare label
                label = f"{class_name}: {confidence:.2f}"

                # Calculate label background size
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, self.font, self.font_scale, self.font_thickness
                )

                # Ensure label is within frame bounds
                label_y = max(y1, label_h + 10)

                # Draw label background
                cv2.rectangle(
                    display_frame,
                    (x1, label_y - label_h - baseline - 5),
                    (x1 + label_w, label_y + baseline + 5),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    display_frame, label,
                    (x1, label_y - baseline - 2),
                    self.font, self.font_scale,
                    (255, 255, 255),
                    self.font_thickness
                )

            return display_frame

        except Exception as e:
            self.logger.error(f"Error drawing detections: {e}")
            return frame

    def _draw_performance_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS and performance information on frame"""
        try:
            display_frame = frame.copy()
            y_offset = 30
            line_height = 25

            # Background for performance info
            overlay_height = 100 if self.show_performance else 30
            cv2.rectangle(
                display_frame,
                (10, 10),
                (300, overlay_height),
                (0, 0, 0),
                -1
            )

            # Add semi-transparent overlay
            overlay = display_frame.copy()
            cv2.rectangle(
                overlay,
                (10, 10),
                (300, overlay_height),
                (0, 0, 0),
                -1
            )
            alpha = 0.7
            display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)

            # Draw FPS
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                display_frame, fps_text,
                (20, y_offset),
                self.font, self.font_scale,
                (0, 255, 0),
                self.font_thickness
            )

            if self.show_performance and len(self.display_times) > 0:
                y_offset += line_height

                # Draw average display time
                avg_display_time = np.mean(self.display_times[-10:])  # Last 10 frames
                display_time_text = f"Display: {avg_display_time*1000:.1f}ms"
                cv2.putText(
                    display_frame, display_time_text,
                    (20, y_offset),
                    self.font, self.font_scale,
                    (0, 255, 255),
                    self.font_thickness
                )

                y_offset += line_height

                # Draw detection count
                det_count_text = f"Detections: {len(getattr(self, '_last_detections', []))}"
                cv2.putText(
                    display_frame, det_count_text,
                    (20, y_offset),
                    self.font, self.font_scale,
                    (255, 255, 0),
                    self.font_thickness
                )

            return display_frame

        except Exception as e:
            self.logger.error(f"Error drawing performance info: {e}")
            return frame

    def _display_hdmi(self, frame: np.ndarray):
        """Display frame using PYNQ HDMI output"""
        try:
            if self.hdmi_out is not None:
                # Convert frame to RGB format if needed
                if frame.shape[2] == 3:  # BGR
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame

                # Display frame via HDMI
                self.hdmi_out.writeframe(frame_rgb)

        except Exception as e:
            self.logger.error(f"HDMI display error: {e}")

    def _display_opencv(self, frame: np.ndarray):
        """Display frame using OpenCV window"""
        try:
            cv2.imshow('YOLOv3 Detection', frame)

            # Check for key press (ESC to exit)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                raise KeyboardInterrupt("ESC pressed")

        except Exception as e:
            self.logger.error(f"OpenCV display error: {e}")

    def _update_display_performance(self, display_time: float):
        """Update display performance metrics"""
        self.frame_count += 1
        self.display_times.append(display_time)

        # Keep only last 100 display times
        if len(self.display_times) > 100:
            self.display_times.pop(0)

        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def get_display_info(self) -> Dict[str, Any]:
        """Get display information and statistics"""
        return {
            "resolution": (self.output_width, self.output_height),
            "fps": self.fps,
            "initialized": self.is_initialized,
            "simulation_mode": self.simulation_mode,
            "show_fps": self.show_fps,
            "show_performance": self.show_performance,
            "avg_display_time_ms": np.mean(self.display_times) * 1000 if self.display_times else 0,
            "total_frames": self.frame_count
        }

    def set_display_mode(self, show_fps: Optional[bool] = None,
                        show_performance: Optional[bool] = None,
                        fullscreen: Optional[bool] = None):
        """Update display mode settings"""
        if show_fps is not None:
            self.show_fps = show_fps

        if show_performance is not None:
            self.show_performance = show_performance

        if fullscreen is not None:
            self.fullscreen = fullscreen
            if self.simulation_mode:
                try:
                    if self.fullscreen:
                        cv2.setWindowProperty('YOLOv3 Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty('YOLOv3 Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                except Exception as e:
                    self.logger.error(f"Error setting fullscreen mode: {e}")

    def clear_display(self):
        """Clear the display"""
        try:
            if self.simulation_mode:
                # Create blank frame
                blank_frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
                cv2.imshow('YOLOv3 Detection', blank_frame)
                cv2.waitKey(1)
            elif self.hdmi_out is not None:
                # Clear HDMI display
                blank_frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
                self._display_hdmi(blank_frame)

        except Exception as e:
            self.logger.error(f"Error clearing display: {e}")

    def save_screenshot(self, filename: str) -> bool:
        """Save current display frame as screenshot"""
        try:
            # Get current frame (this would need to be stored during display_frame)
            # For now, we'll create a placeholder
            screenshot = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            cv2.putText(
                screenshot, "YOLOv3 Screenshot",
                (self.output_width // 2 - 100, self.output_height // 2),
                self.font, 1.0,
                (255, 255, 255),
                2
            )

            cv2.imwrite(filename, screenshot)
            self.logger.info(f"Screenshot saved: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving screenshot: {e}")
            return False

    def cleanup(self):
        """Cleanup display resources"""
        try:
            if self.simulation_mode:
                cv2.destroyAllWindows()
            elif self.hdmi_out is not None:
                # Turn off HDMI output
                if hasattr(self.hdmi_out, 'close'):
                    self.hdmi_out.close()

            self.is_initialized = False
            self.logger.info("Display resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error during display cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()