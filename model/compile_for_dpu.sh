#!/bin/bash
# Compile YOLOv3 model for DPU deployment using DNNDK

set -e

echo "Compiling YOLOv3 model for DPU..."

# Check if DNNDK is available
if [ -z "$DNNDK" ]; then
    echo "Error: DNNDK environment variable not set"
    echo "Please install DNNDK v3.1 and set DNNDK environment variable"
    echo "Example: export DNNDK=/opt/DNNDK"
    exit 1
fi

if [ ! -d "$DNNDK" ]; then
    echo "Error: DNNDK directory not found: $DNNDK"
    exit 1
fi

# Set paths
MODEL_DIR=$(pwd)/model
TRAINED_MODEL=$MODEL_DIR/training/yolov3_custom.pt  # Adjust path as needed
COMPILED_DIR=$MODEL_DIR/compiled

echo "Model directory: $MODEL_DIR"
echo "Trained model: $TRAINED_MODEL"
echo "Compiled output: $COMPILED_DIR"

# Create compiled directory
mkdir -p "$COMPILED_DIR"

# Check if trained model exists
if [ ! -f "$TRAINED_MODEL" ]; then
    echo "Error: Trained model not found: $TRAINED_MODEL"
    echo "Please train the model first using: python model/train_yolov3.py"
    exit 1
fi

# DNNDK compilation process
echo "Starting DNNDK compilation process..."

# Step 1: Convert PyTorch model to ONNX
echo "Step 1: Converting PyTorch model to ONNX..."
cd "$MODEL_DIR"
python3 -c "
import torch
import torch.onnx
from ultralytics import YOLO

# Load trained model
model = YOLO('$TRAINED_MODEL')

# Create dummy input
dummy_input = torch.randn(1, 3, 416, 416)

# Export to ONNX
torch.onnx.export(
    model.model,
    dummy_input,
    'training/yolov3_custom.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print('Model exported to ONNX format')
"

# Step 2: Convert ONNX to TensorFlow (using tf2onnx)
echo "Step 2: Converting ONNX to TensorFlow..."
python3 -c "
import tf2onnx
import onnx
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load('training/yolov3_custom.onnx')

# Convert to TensorFlow
tf_rep = tf2onnx.tfonnx.process_tf_graph(onnx_model,
    opset=11,
    input_names=['input'],
    output_names=['output']
)

# Save TensorFlow model
with tf.Graph().as_default():
    tf.import_graph_def(tf_rep.graph_def, name='')
    with tf.Session() as sess:
        # Save frozen graph
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            ['output']
        )

        with tf.gfile.GFile('training/yolov3_custom.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())

print('Model converted to TensorFlow format')
"

# Step 3: Quantize model using DNNDK
echo "Step 3: Quantizing model using DNNDK..."
cd "$COMPILED_DIR"

# Create calibration dataset (use sample images)
echo "Creating calibration dataset..."
mkdir -p calibration_images
python3 -c "
import cv2
import numpy as np
import os
from pathlib import Path

# Find sample images from dataset
dataset_path = Path('../dataset/images/train')
image_files = list(dataset_path.glob('*.jpg'))[:100]  # Use 100 images for calibration

if not image_files:
    print('No calibration images found, creating synthetic data')
    # Create synthetic calibration data
    for i in range(100):
        synthetic_img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        cv2.imwrite(f'calibration_images/calib_{i:03d}.jpg', synthetic_img)
else:
    # Copy and resize real images
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is not None:
            img_resized = cv2.resize(img, (416, 416))
            cv2.imwrite(f'calibration_images/calib_{i:03d}.jpg', img_resized)

print(f'Created {len(list(Path(\"calibration_images\").glob(\"*.jpg\")))} calibration images')
"

# Run DNNDK quantization
echo "Running DNNDK quantization..."
"$DNNDK/dnnc" \
    --model="../training/yolov3_custom.pb" \
    --output="yolov3_custom_quantized.pb" \
    --input-shape="1,3,416,416" \
    --input-node="input" \
    --output-node="output" \
    --calib-images="calibration_images/" \
    --data-type="int8" \
    --bit-width=8 \
    --scale-type="per-channel" \
    --zeropoint-type="per-channel"

# Step 4: Compile quantized model for DPU
echo "Step 4: Compiling quantized model for DPU..."
"$DNNDK/dnc" \
    --model="yolov3_custom_quantized.pb" \
    --output="yolov3_custom.elf" \
    --target="DPUCAHX8H" \
    --arch="B1152" \
    --cpu_arch="arm64" \
    --net_name="yolov3_custom"

# Step 5: Generate DPU instructions
echo "Step 5: Generating DPU instructions..."
"$DNNDK/dnnc" \
    --elf="yolov3_custom.elf" \
    --output="yolov3_custom.dpu" \
    --mode="dpu"

# Create model metadata file
echo "Creating model metadata..."
cat > model_info.json << EOF
{
    "model_name": "YOLOv3 Custom",
    "input_shape": [1, 3, 416, 416],
    "output_shape": "YOLOv3 outputs (3 scales)",
    "quantization": "int8",
    "target_architecture": "DPUCAHX8H",
    "dpu_config": "B1152",
    "compilation_date": "$(date)",
    "dnndk_version": "v3.1",
    "original_model": "$TRAINED_MODEL",
    "compiled_files": {
        "quantized_pb": "yolov3_custom_quantized.pb",
        "elf": "yolov3_custom.elf",
        "dpu_instructions": "yolov3_custom.dpu"
    }
}
EOF

# Copy compiled files to expected locations
echo "Copying files to deployment directory..."
mkdir -p ../deploy
cp yolov3_custom.elf ../deploy/
cp yolov3_custom.dpu ../deploy/
cp model_info.json ../deploy/

# Display results
echo ""
echo "DPU compilation completed successfully!"
echo ""
echo "Generated files:"
echo "- Quantized model: yolov3_custom_quantized.pb"
echo "- DPU executable: yolov3_custom.elf"
echo "- DPU instructions: yolov3_custom.dpu"
echo "- Model metadata: model_info.json"
echo ""
echo "Files copied to: ../deploy/"
echo ""
echo "File sizes:"
ls -lh *.elf *.dpu *.pb 2>/dev/null || echo "Some files may not exist"
echo ""
echo "Next steps:"
echo "1. Copy .elf and .dpu files to PYNQ-Z2"
echo "2. Update config file with model paths"
echo "3. Test inference on PYNQ-Z2"
echo ""