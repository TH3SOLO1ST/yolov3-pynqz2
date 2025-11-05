"""
Configuration settings for YOLOv3 realtime system
"""

import os
import json
from pathlib import Path
from typing import Tuple, List

class Config:
    """Configuration management for YOLOv3 realtime system"""

    def __init__(self, config_path=None):
        """Initialize configuration with default values"""
        # Set default paths
        self.project_root = Path(__file__).parent.parent.parent

        # Model settings
        self.MODEL_PATH = str(self.project_root / "model" / "compiled" / "model_compiled.dpu")
        self.CLASSES_FILE = str(self.project_root / "model" / "dataset" / "classes.txt")
        self.INPUT_SIZE = (416, 416)  # Must match trained model
        self.CONF_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.4
        self.NUM_CLASSES = 80  # Default COCO classes

        # Camera settings
        self.CAMERA_INDEX = 0
        self.CAMERA_WIDTH = 1280
        self.CAMERA_HEIGHT = 720
        self.CAMERA_FPS = 30
        self.CAMERA_FORMAT = "YUYV"  # or "MJPEG"

        # Display settings
        self.OUTPUT_WIDTH = 1920
        self.OUTPUT_HEIGHT = 1080
        self.SHOW_FPS = True
        self.SHOW_PERFORMANCE = True
        self.FULLSCREEN = False

        # Performance tuning
        self.BUFFER_SIZE = 3  # Number of frames in pipeline
        self.INFERENCE_THREADS = 1  # DPU is single-threaded
        self.MAX_DETECTIONS = 100

        # YOLOv3 specific settings
        self.ANCHORS = [
            (10, 13), (16, 30), (33, 23),  # Small objects
            (30, 61), (62, 45), (59, 119),  # Medium objects
            (116, 90), (156, 198), (373, 326)  # Large objects
        ]
        self.STRIDES = [8, 16, 32]  # YOLOv3 output strides

        # Colors for bounding boxes (BGR format for OpenCV)
        self.CLASS_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (192, 192, 192), (64, 64, 64), (128, 128, 128), (255, 165, 0)
        ]

        # DPU specific settings
        self.DPU_MEM_BASE = 0x4F000000
        self.DPU_MEM_SIZE = 16 * 1024 * 1024  # 16MB

        # Logging settings
        self.LOG_LEVEL = "INFO"
        self.LOG_FILE = str(self.project_root / "yolo_pynq.log")

        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

        # Load class names if file exists
        self.load_class_names()

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)

            # Update configuration with custom values
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Unknown config key '{key}' ignored")

        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")

    def load_class_names(self):
        """Load class names from file"""
        self.class_names = []

        if os.path.exists(self.CLASSES_FILE):
            try:
                with open(self.CLASSES_FILE, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                self.NUM_CLASSES = len(self.class_names)
            except Exception as e:
                print(f"Error loading classes file: {e}")
                self.class_names = [f"class_{i}" for i in range(self.NUM_CLASSES)]
        else:
            # Use default COCO class names
            self.class_names = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
            ]

    def get_anchors_for_scale(self, scale_idx):
        """Get anchors for specific YOLOv3 output scale"""
        if scale_idx == 0:  # Small objects
            return self.ANCHORS[6:9]
        elif scale_idx == 1:  # Medium objects
            return self.ANCHORS[3:6]
        else:  # Large objects (scale_idx == 2)
            return self.ANCHORS[0:3]

    def get_color_for_class(self, class_id):
        """Get color for specific class ID"""
        if class_id < len(self.CLASS_COLORS):
            return self.CLASS_COLORS[class_id]
        else:
            # Generate color based on class ID if not predefined
            hue = (class_id * 137) % 256  # Golden angle approximation
            return (hue, (hue + 85) % 256, (hue + 170) % 256)

    def validate(self):
        """Validate configuration settings"""
        errors = []

        # Check model file exists
        if not os.path.exists(self.MODEL_PATH):
            errors.append(f"Model file not found: {self.MODEL_PATH}")

        # Check input dimensions
        if self.INPUT_SIZE[0] % 32 != 0 or self.INPUT_SIZE[1] % 32 != 0:
            errors.append("Input dimensions must be multiples of 32")

        # Check camera dimensions
        if self.CAMERA_WIDTH <= 0 or self.CAMERA_HEIGHT <= 0:
            errors.append("Camera dimensions must be positive")

        # Check thresholds
        if not 0 <= self.CONF_THRESHOLD <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        if not 0 <= self.NMS_THRESHOLD <= 1:
            errors.append("NMS threshold must be between 0 and 1")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    def save_config(self, output_path):
        """Save current configuration to JSON file"""
        config_dict = {}

        # Include all public attributes
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                if not isinstance(value, (Path, type(None))):
                    config_dict[attr] = value

        try:
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        except Exception as e:
            raise ValueError(f"Failed to save config: {e}")

    def __str__(self):
        """String representation of configuration"""
        return f"""YOLOv3 Configuration:
Model: {self.MODEL_PATH}
Classes: {self.NUM_CLASSES}
Input Size: {self.INPUT_SIZE}
Camera: {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT} @ {self.CAMERA_FPS}fps
Display: {self.OUTPUT_WIDTH}x{self.OUTPUT_HEIGHT}
Confidence Threshold: {self.CONF_THRESHOLD}
NMS Threshold: {self.NMS_THRESHOLD}
Buffer Size: {self.BUFFER_SIZE}"""