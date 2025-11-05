"""Inference module for YOLOv3 realtime system"""

from .dpu_inference import DPUInference
from .postprocessor import YOLOv3Postprocessor

__all__ = ['DPUInference', 'YOLOv3Postprocessor']