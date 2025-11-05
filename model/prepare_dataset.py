#!/usr/bin/env python3
"""
Dataset preparation script for YOLOv3 training
Handles dataset downloading, annotation conversion, and train/validation split
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict
import yaml

class DatasetPreparer:
    """YOLOv3 dataset preparation utility"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        self.classes_file = self.dataset_dir / "classes.txt"
        self.data_yaml = self.dataset_dir / "data.yaml"

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"Dataset directory: {self.dataset_dir}")

    def create_coco_subset(self, coco_dir: str, target_classes: List[str],
                          samples_per_class: int = 1000) -> bool:
        """Create a subset of COCO dataset with specified classes"""
        try:
            print(f"Creating COCO subset with classes: {target_classes}")

            # This would require COCO API (pycocotools)
            # For now, we'll create a placeholder structure
            print("Note: COCO subset creation requires pycocotools")
            print("Install with: pip install pycocotools")

            # Create sample directory structure
            for split in ['train', 'val']:
                (self.images_dir / split).mkdir(exist_ok=True)
                (self.labels_dir / split).mkdir(exist_ok=True)

            return True

        except Exception as e:
            print(f"Error creating COCO subset: {e}")
            return False

    def create_custom_dataset(self, image_dir: str, annotation_dir: str = None) -> bool:
        """Create custom dataset from images and optional annotations"""
        try:
            image_path = Path(image_dir)
            if not image_path.exists():
                print(f"Error: Image directory not found: {image_dir}")
                return False

            print(f"Processing images from: {image_dir}")

            # Copy images
            for image_file in image_path.glob("*"):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(image_file, self.images_dir / image_file.name)

            # Copy annotations if provided
            if annotation_dir:
                annotation_path = Path(annotation_dir)
                if annotation_path.exists():
                    for ann_file in annotation_path.glob("*"):
                        if ann_file.suffix.lower() in ['.txt', '.json', '.xml']:
                            shutil.copy2(ann_file, self.labels_dir / ann_file.name)

            print(f"Copied {len(list(self.images_dir.glob('*')))} images")
            return True

        except Exception as e:
            print(f"Error creating custom dataset: {e}")
            return False

    def convert_voc_to_yolo(self, voc_dir: str, classes_file: str) -> bool:
        """Convert VOC XML annotations to YOLO format"""
        try:
            import xml.etree.ElementTree as ET

            print("Converting VOC annotations to YOLO format...")

            # Read classes
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            voc_path = Path(voc_dir)
            if not voc_path.exists():
                print(f"Error: VOC directory not found: {voc_dir}")
                return False

            # Convert each XML file
            for xml_file in voc_path.glob("*.xml"):
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image dimensions
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)

                # Create YOLO annotation file
                yolo_file = self.labels_dir / f"{xml_file.stem}.txt"

                with open(yolo_file, 'w') as yolo_f:
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        if class_name not in classes:
                            continue

                        class_id = classes.index(class_name)

                        # Get bounding box
                        bbox = obj.find('bndbox')
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)

                        # Convert to YOLO format (normalized)
                        x_center = (xmin + xmax) / 2.0 / img_width
                        y_center = (ymin + ymax) / 2.0 / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height

                        yolo_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # Copy corresponding image
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_file = xml_file.parent / f"{xml_file.stem}{ext}"
                    if img_file.exists():
                        shutil.copy2(img_file, self.images_dir / img_file.name)
                        break

            print(f"Converted {len(list(voc_path.glob('*.xml')))} annotations")
            return True

        except Exception as e:
            print(f"Error converting VOC annotations: {e}")
            return False

    def create_train_val_split(self, val_ratio: float = 0.2) -> bool:
        """Create train/validation split"""
        try:
            print(f"Creating train/validation split (val_ratio={val_ratio})")

            # Get all image files
            image_files = list(self.images_dir.glob("*.jpg")) + \
                         list(self.images_dir.glob("*.jpeg")) + \
                         list(self.images_dir.glob("*.png"))

            if not image_files:
                print("No image files found")
                return False

            # Shuffle files
            random.shuffle(image_files)

            # Calculate split
            val_count = int(len(image_files) * val_ratio)
            train_count = len(image_files) - val_count

            # Create split directories
            train_img_dir = self.images_dir / "train"
            val_img_dir = self.images_dir / "val"
            train_lbl_dir = self.labels_dir / "train"
            val_lbl_dir = self.labels_dir / "val"

            for dir_path in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
                dir_path.mkdir(exist_ok=True)

            # Split files
            train_files = image_files[:train_count]
            val_files = image_files[train_count:]

            # Copy files to appropriate directories
            for img_file in train_files:
                # Copy image
                shutil.move(str(img_file), str(train_img_dir / img_file.name))
                # Move corresponding label if exists
                label_file = self.labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.move(str(label_file), str(train_lbl_dir / label_file.name))

            for img_file in val_files:
                # Copy image
                shutil.move(str(img_file), str(val_img_dir / img_file.name))
                # Move corresponding label if exists
                label_file = self.labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.move(str(label_file), str(val_lbl_dir / label_file.name))

            print(f"Train set: {len(train_files)} images")
            print(f"Validation set: {len(val_files)} images")
            return True

        except Exception as e:
            print(f"Error creating train/val split: {e}")
            return False

    def create_classes_file(self, classes: List[str]) -> bool:
        """Create classes.txt file"""
        try:
            print(f"Creating classes file with {len(classes)} classes")
            with open(self.classes_file, 'w') as f:
                for class_name in classes:
                    f.write(f"{class_name}\n")
            return True

        except Exception as e:
            print(f"Error creating classes file: {e}")
            return False

    def create_data_yaml(self, train_path: str, val_path: str,
                        nc: int, names: List[str]) -> bool:
        """Create data.yaml for YOLOv5/YOLOv8 training"""
        try:
            data_config = {
                'train': str(train_path),
                'val': str(val_path),
                'nc': nc,
                'names': names
            }

            with open(self.data_yaml, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)

            print(f"Created data.yaml: {self.data_yaml}")
            return True

        except Exception as e:
            print(f"Error creating data.yaml: {e}")
            return False

    def validate_dataset(self) -> Dict[str, any]:
        """Validate dataset and return statistics"""
        try:
            print("Validating dataset...")

            stats = {
                'total_images': 0,
                'total_labels': 0,
                'train_images': 0,
                'val_images': 0,
                'missing_labels': 0,
                'empty_labels': 0,
                'classes': []
            }

            # Read classes
            if self.classes_file.exists():
                with open(self.classes_file, 'r') as f:
                    stats['classes'] = [line.strip() for line in f.readlines()]

            # Count files
            for split in ['train', 'val']:
                img_dir = self.images_dir / split
                lbl_dir = self.labels_dir / split

                if img_dir.exists():
                    images = list(img_dir.glob("*.jpg")) + \
                            list(img_dir.glob("*.jpeg")) + \
                            list(img_dir.glob("*.png"))
                    stats[f'{split}_images'] = len(images)
                    stats['total_images'] += len(images)

                if lbl_dir.exists():
                    labels = list(lbl_dir.glob("*.txt"))
                    stats['total_labels'] += len(labels)

                    # Check for empty labels
                    for label_file in labels:
                        if label_file.stat().st_size == 0:
                            stats['empty_labels'] += 1

            # Check for missing labels
            stats['missing_labels'] = stats['total_images'] - stats['total_labels']

            # Print statistics
            print(f"Dataset Statistics:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Total labels: {stats['total_labels']}")
            print(f"  Train images: {stats['train_images']}")
            print(f"  Validation images: {stats['val_images']}")
            print(f"  Missing labels: {stats['missing_labels']}")
            print(f"  Empty labels: {stats['empty_labels']}")
            print(f"  Classes: {len(stats['classes'])}")

            return stats

        except Exception as e:
            print(f"Error validating dataset: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Prepare YOLOv3 dataset')
    parser.add_argument('--dataset-dir', type=str, default='model/dataset',
                        help='Output dataset directory')
    parser.add_argument('--image-dir', type=str, help='Input image directory')
    parser.add_argument('--annotation-dir', type=str, help='Input annotation directory')
    parser.add_argument('--voc-dir', type=str, help='VOC format annotation directory')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['person', 'car', 'bicycle'],
                        help='Class names')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--create-coco-subset', action='store_true',
                        help='Create COCO subset (requires pycocotools)')

    args = parser.parse_args()

    # Create dataset preparer
    preparer = DatasetPreparer(args.dataset_dir)

    # Create classes file
    preparer.create_classes_file(args.classes)

    # Process different input types
    if args.create_coco_subset:
        preparer.create_coco_subset(args.dataset_dir, args.classes)
    elif args.voc_dir:
        # Convert VOC annotations
        if preparer.convert_voc_to_yolo(args.voc_dir, args.classes):
            print("VOC conversion completed")
    elif args.image_dir:
        # Create custom dataset
        preparer.create_custom_dataset(args.image_dir, args.annotation_dir)

    # Create train/validation split
    preparer.create_train_val_split(args.val_ratio)

    # Create data.yaml for training
    train_path = preparer.images_dir / "train"
    val_path = preparer.images_dir / "val"
    preparer.create_data_yaml(train_path, val_path, len(args.classes), args.classes)

    # Validate dataset
    stats = preparer.validate_dataset()

    print(f"\nDataset preparation completed!")
    print(f"Dataset location: {args.dataset_dir}")

if __name__ == "__main__":
    main()