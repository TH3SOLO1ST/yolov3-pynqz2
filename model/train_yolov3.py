#!/usr/bin/env python3
"""
YOLOv3 training script for custom datasets
Supports both Darknet and PyTorch implementations
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from typing import List, Dict, Any

# Try to import YOLOv3 implementations
try:
    # Try YOLOv5/YOLOv8 (PyTorch) - preferred for modern training
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("Using Ultralytics YOLO implementation")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics YOLO not available")

try:
    # Try to import Darknet for original YOLOv3
    import darknet
    DARKNET_AVAILABLE = True
    print("Darknet available for YOLOv3")
except ImportError:
    DARKNET_AVAILABLE = False
    print("Darknet not available")

class YOLOv3Trainer:
    """YOLOv3 training utility"""

    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file) if config_file else self.default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'model': 'yolov3-tiny.pt',  # Start with YOLOv3-tiny for faster training
            'data': 'model/dataset/data.yaml',
            'epochs': 100,
            'batch_size': 16,
            'img_size': 416,
            'lr': 0.001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'workers': 8,
            'project': 'runs/train',
            'name': 'yolov3_custom',
            'save_period': 10,
            'patience': 50,
            'device': str(self.device),
            'optimizer': 'Adam',
            'lr_scheduler': 'cosine'
        }

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    return json.load(f)
                elif config_file.endswith(('.yml', '.yaml')):
                    return yaml.safe_load(f)
                else:
                    raise ValueError("Config file must be JSON or YAML")
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config()

    def train_ultralytics(self) -> bool:
        """Train using Ultralytics YOLO implementation"""
        try:
            print("Training with Ultralytics YOLO...")

            # Initialize model
            if self.config['model'].endswith('.pt'):
                model = YOLO(self.config['model'])
            else:
                # Create new YOLOv3 model (if supported)
                model = YOLO('yolov3.yaml')

            # Train model
            results = model.train(
                data=self.config['data'],
                epochs=self.config['epochs'],
                imgsz=self.config['img_size'],
                batch=self.config['batch_size'],
                lr0=self.config['lr'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay'],
                warmup_epochs=self.config['warmup_epochs'],
                project=self.config['project'],
                name=self.config['name'],
                save_period=self.config['save_period'],
                patience=self.config['patience'],
                device=self.config['device'],
                optimizer=self.config['optimizer'],
                lr_scheduler=self.config['lr_scheduler'],
                workers=self.config['workers']
            )

            print("Training completed!")
            print(f"Best model saved to: {results.save_dir}")

            # Export to ONNX for potential conversion
            try:
                model.export(format='onnx')
                print("Model exported to ONNX format")
            except Exception as e:
                print(f"ONNX export failed: {e}")

            return True

        except Exception as e:
            print(f"Ultralytics training failed: {e}")
            return False

    def train_darknet(self) -> bool:
        """Train using original Darknet YOLOv3"""
        try:
            print("Training with Darknet YOLOv3...")

            # Create Darknet configuration files
            self.create_darknet_config()

            # Build Darknet command
            cmd = [
                './darknet',
                'detector',
                'train',
                'data/darknet.data',
                'cfg/yolov3_custom.cfg',
                '-gpus', '0',  # GPU index
                '-clear'
            ]

            # Execute training
            import subprocess
            result = subprocess.run(cmd, cwd='.')

            return result.returncode == 0

        except Exception as e:
            print(f"Darknet training failed: {e}")
            return False

    def create_darknet_config(self):
        """Create Darknet configuration files"""
        try:
            # Create .data file
            data_content = f"""classes = {len(self.config.get('classes', []))}
train = {self.config['data'].replace('data.yaml', 'images/train')}
valid = {self.config['data'].replace('data.yaml', 'images/val')}
names = {self.config['data'].replace('data.yaml', 'classes.txt')}
backup = backup/
eval=coco
"""

            with open('data/darknet.data', 'w') as f:
                f.write(data_content)

            # Create .cfg file based on YOLOv3
            cfg_content = f"""[net]
# Testing
batch=1
subdivisions=1
# Training
batch={self.config['batch_size']}
subdivisions={self.config['batch_size']}
width={self.config['img_size']}
height={self.config['img_size']}
channels=3
momentum={self.config['momentum']}
decay={self.config['weight_decay']}
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate={self.config['lr']}
burn_in=1000
max_batches = {self.config['epochs'] * 1000}
policy=steps
steps={int(self.config['epochs'] * 800)}, {int(self.config['epochs'] * 900)}
scales=.1,.1

[convolutional]
size=1
stride=1
pad=1
filters={len(self.config.get('classes', [])) + 5}
activation=linear

[region]
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
bias_match=1
classes={len(self.config.get('classes', []))}
coords=4
num=9
softmax=1
jitter=.3
rescore=1

object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1

absolute=1
thresh = .6
random=1
"""

            with open('cfg/yolov3_custom.cfg', 'w') as f:
                f.write(cfg_content)

            print("Darknet configuration files created")

        except Exception as e:
            print(f"Error creating Darknet config: {e}")

    def prepare_for_dnndk(self, model_path: str) -> bool:
        """Prepare trained model for DNNDK compilation"""
        try:
            print("Preparing model for DNNDK compilation...")

            # This would involve:
            # 1. Convert PyTorch model to TensorFlow/ONNX
            # 2. Quantize to 8-bit integers
            # 3. Compile with DNNDK tools

            print("Note: DNNDK preparation requires Xilinx Vitis AI tools")
            print("This step should be performed on the development machine with DNNDK installed")

            # Placeholder for DNNDK preparation steps
            steps = [
                "1. Export PyTorch model to ONNX",
                "2. Convert ONNX to TensorFlow",
                "3. Quantize model to 8-bit using DNNDK tools",
                "4. Compile with DNNC compiler",
                "5. Generate .elf file for PYNQ deployment"
            ]

            for step in steps:
                print(f"   {step}")

            return True

        except Exception as e:
            print(f"Error preparing for DNNDK: {e}")
            return False

    def validate_model(self, model_path: str, data_path: str) -> Dict[str, float]:
        """Validate trained model performance"""
        try:
            print("Validating model performance...")

            if ULTRALYTICS_AVAILABLE:
                model = YOLO(model_path)
                results = model.val(data=data_path)

                metrics = {
                    'map50': results.box.map50,
                    'map50_95': results.box.map,
                    'precision': results.box.mp,
                    'recall': results.box.mr
                }

                print(f"Validation Results:")
                print(f"  mAP@0.5: {metrics['map50']:.4f}")
                print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")

                return metrics

            else:
                print("Validation requires Ultralytics YOLO")
                return {}

        except Exception as e:
            print(f"Error validating model: {e}")
            return {}

    def train(self) -> bool:
        """Main training function"""
        try:
            print("Starting YOLOv3 training...")
            print(f"Configuration: {self.config}")

            # Check dataset
            if not os.path.exists(self.config['data']):
                print(f"Error: Dataset file not found: {self.config['data']}")
                return False

            # Choose training method
            if ULTRALYTICS_AVAILABLE:
                return self.train_ultralytics()
            elif DARKNET_AVAILABLE:
                return self.train_darknet()
            else:
                print("Error: No YOLO implementation available")
                print("Install ultralytics: pip install ultralytics")
                return False

        except Exception as e:
            print(f"Training failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv3 model')
    parser.add_argument('--config', type=str, help='Training configuration file')
    parser.add_argument('--data', type=str, default='model/dataset/data.yaml',
                        help='Dataset configuration file')
    parser.add_argument('--model', type=str, default='yolov3-tiny.pt',
                        help='Base model or config')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=416,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Training device (cpu/cuda/auto)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='yolov3_custom',
                        help='Experiment name')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing model')

    args = parser.parse_args()

    # Create trainer
    trainer = YOLOv3Trainer(args.config)

    # Override config with command line arguments
    if args.data:
        trainer.config['data'] = args.data
    if args.model:
        trainer.config['model'] = args.model
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['batch_size'] = args.batch_size
    if args.img_size:
        trainer.config['img_size'] = args.img_size
    if args.lr:
        trainer.config['lr'] = args.lr
    if args.device != 'auto':
        trainer.config['device'] = args.device
    if args.project:
        trainer.config['project'] = args.project
    if args.name:
        trainer.config['name'] = args.name

    # Validate or train
    if args.validate_only:
        if args.resume:
            trainer.validate_model(args.resume, args.data)
        else:
            print("Error: --resume required for validation")
    else:
        success = trainer.train()
        if success:
            print("Training completed successfully!")

            # Prepare for DNNDK if requested
            if hasattr(args, 'prepare_dnndk') and args.prepare_dnndk:
                trainer.prepare_for_dnndk(trainer.config['model'])
        else:
            print("Training failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()