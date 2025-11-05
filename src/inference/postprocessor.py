"""
YOLOv3 postprocessor for DPU inference output
Handles NMS, coordinate conversion, and confidence thresholding
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Tuple

class YOLOv3Postprocessor:
    """YOLOv3 postprocessor for handling DPU output"""

    def __init__(self, config):
        """Initialize postprocessor with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # YOLOv3 configuration
        self.num_classes = config.NUM_CLASSES
        self.conf_threshold = config.CONF_THRESHOLD
        self.nms_threshold = config.NMS_THRESHOLD
        self.anchors = config.ANCHORS
        self.strides = config.STRIDES

        # Output scales
        self.num_scales = len(self.strides)
        self.num_anchors_per_scale = 3
        self.output_dim_per_anchor = self.num_classes + 5  # 4 bbox + 1 objectness + classes

        self.logger.info(f"YOLOv3 postprocessor initialized: {self.num_classes} classes")

    def process(self, raw_outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Process raw DPU outputs to get final detections

        Args:
            raw_outputs: List of output tensors from DPU
            original_shape: Original image shape (height, width)

        Returns:
            List of detection dictionaries
        """
        try:
            if not raw_outputs or len(raw_outputs) != self.num_scales:
                self.logger.warning(f"Expected {self.num_scales} outputs, got {len(raw_outputs)}")
                return []

            # Process each scale output
            all_detections = []
            input_h, input_w = self.config.INPUT_SIZE[1], self.config.INPUT_SIZE[0]

            for scale_idx, output in enumerate(raw_outputs):
                if output.ndim != 4:
                    self.logger.warning(f"Unexpected output shape at scale {scale_idx}: {output.shape}")
                    continue

                # Extract scale detections
                scale_detections = self._process_scale_output(
                    output, scale_idx, (input_h, input_w), original_shape
                )
                all_detections.extend(scale_detections)

            # Apply NMS across all detections
            final_detections = self._apply_nms(all_detections)

            return final_detections

        except Exception as e:
            self.logger.error(f"Error in postprocessing: {e}")
            return []

    def _process_scale_output(self, output: np.ndarray, scale_idx: int,
                            input_shape: Tuple[int, int],
                            original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Process output from a single YOLOv3 scale"""
        try:
            batch_size, channels, grid_h, grid_w = output.shape

            if batch_size != 1:
                self.logger.warning(f"Expected batch size 1, got {batch_size}")

            # Get anchors for this scale
            scale_anchors = self._get_scale_anchors(scale_idx)

            # Reshape output: (grid_h, grid_w, num_anchors, output_dim)
            output = output.transpose(0, 2, 3, 1)  # (1, grid_h, grid_w, channels)
            output = output.reshape(grid_h, grid_w, self.num_anchors_per_scale, self.output_dim_per_anchor)

            detections = []

            for grid_y in range(grid_h):
                for grid_x in range(grid_w):
                    for anchor_idx in range(self.num_anchors_per_scale):
                        # Extract prediction for this cell and anchor
                        prediction = output[grid_y, grid_x, anchor_idx]

                        # Extract components
                        tx, ty, tw, th = prediction[:4]  # Box coordinates
                        objectness = prediction[4]       # Objectness score
                        class_probs = prediction[5:]     # Class probabilities

                        # Apply sigmoid to objectness and class probabilities
                        objectness = self._sigmoid(objectness)
                        class_probs = self._sigmoid(class_probs)

                        # Find best class
                        class_conf = np.max(class_probs)
                        class_id = np.argmax(class_probs)

                        # Combine objectness and class confidence
                        confidence = objectness * class_conf

                        # Apply confidence threshold
                        if confidence < self.conf_threshold:
                            continue

                        # Convert to actual box coordinates
                        box = self._decode_box(
                            tx, ty, tw, th, grid_x, grid_y, grid_w, grid_h,
                            scale_anchors[anchor_idx], input_shape, original_shape
                        )

                        detection = {
                            'bbox': box,  # [x1, y1, x2, y2] in original image coordinates
                            'confidence': float(confidence),
                            'class_id': int(class_id),
                            'class_name': self._get_class_name(class_id)
                        }

                        detections.append(detection)

            return detections

        except Exception as e:
            self.logger.error(f"Error processing scale {scale_idx}: {e}")
            return []

    def _decode_box(self, tx: float, ty: float, tw: float, th: float,
                   grid_x: int, grid_y: int, grid_w: int, grid_h: int,
                   anchor: Tuple[int, int], input_shape: Tuple[int, int],
                   original_shape: Tuple[int, int]) -> List[float]:
        """Decode YOLOv3 box prediction to actual coordinates"""
        try:
            # Apply sigmoid to center coordinates
            bx = (self._sigmoid(tx) + grid_x) / grid_w
            by = (self._sigmoid(ty) + grid_y) / grid_h

            # Apply exponential to width/height and apply anchor
            bw = (np.exp(tw) * anchor[0]) / input_shape[1]
            bh = (np.exp(th) * anchor[1]) / input_shape[0]

            # Convert center coordinates to corner coordinates
            x1 = (bx - bw / 2) * original_shape[1]  # Width
            y1 = (by - bh / 2) * original_shape[0]  # Height
            x2 = (bx + bw / 2) * original_shape[1]
            y2 = (by + bh / 2) * original_shape[0]

            # Clamp to image boundaries
            x1 = max(0, min(original_shape[1] - 1, x1))
            y1 = max(0, min(original_shape[0] - 1, y1))
            x2 = max(0, min(original_shape[1] - 1, x2))
            y2 = max(0, min(original_shape[0] - 1, y2))

            return [x1, y1, x2, y2]

        except Exception as e:
            self.logger.error(f"Error decoding box: {e}")
            return [0, 0, 0, 0]

    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to detections"""
        try:
            if not detections:
                return []

            # Group detections by class
            class_detections = {}
            for det in detections:
                class_id = det['class_id']
                if class_id not in class_detections:
                    class_detections[class_id] = []
                class_detections[class_id].append(det)

            # Apply NMS per class
            final_detections = []
            for class_id, class_dets in class_detections.items():
                if len(class_dets) == 0:
                    continue

                # Extract boxes and scores
                boxes = np.array([det['bbox'] for det in class_dets])
                scores = np.array([det['confidence'] for det in class_dets])

                # Convert boxes to (x, y, w, h) format for NMS
                boxes_xywh = np.zeros_like(boxes)
                boxes_xywh[:, 0] = boxes[:, 0]  # x
                boxes_xywh[:, 1] = boxes[:, 1]  # y
                boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
                boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

                # Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes_xywh.tolist(), scores.tolist(),
                    self.conf_threshold, self.nms_threshold
                )

                # Keep selected detections
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                elif isinstance(indices, tuple):
                    indices = indices[0]

                for idx in indices:
                    final_detections.append(class_dets[idx])

            # Sort by confidence (highest first)
            final_detections.sort(key=lambda x: x['confidence'], reverse=True)

            # Limit number of detections
            max_detections = self.config.MAX_DETECTIONS
            if len(final_detections) > max_detections:
                final_detections = final_detections[:max_detections]

            return final_detections

        except Exception as e:
            self.logger.error(f"Error applying NMS: {e}")
            return detections[:self.config.MAX_DETECTIONS]

    def _get_scale_anchors(self, scale_idx: int) -> List[Tuple[int, int]]:
        """Get anchor boxes for specific scale"""
        if scale_idx == 0:  # Large objects (high resolution)
            return self.anchors[6:9]
        elif scale_idx == 1:  # Medium objects
            return self.anchors[3:6]
        else:  # Small objects (low resolution)
            return self.anchors[0:3]

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-x))

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        if class_id < len(self.config.class_names):
            return self.config.class_names[class_id]
        else:
            return f"class_{class_id}"

    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Visualize detections on image (for debugging)"""
        vis_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            class_id = det['class_id']

            # Get color for this class
            color = self.config.get_color_for_class(class_id)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                vis_image, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return vis_image

    def get_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detection statistics"""
        if not detections:
            return {
                'total_detections': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0
            }

        class_counts = {}
        confidences = []

        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)

        return {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences)
        }