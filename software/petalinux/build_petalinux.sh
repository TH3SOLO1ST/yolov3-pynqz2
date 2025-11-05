#!/bin/bash
# Build PetaLinux for YOLOv3 on PYNQ-Z2

set -e

echo "Building PetaLinux for YOLOv3..."

# Check if PetaLinux is available
if ! command -v petalinux-build &> /dev/null; then
    echo "Error: PetaLinux not found in PATH"
    echo "Please source PetaLinux settings.sh first"
    exit 1
fi

# Set project directory
PROJECT_DIR=$(pwd)/software/petalinux/pynq_z2_dpu

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: PetaLinux project not found: $PROJECT_DIR"
    echo "Please run: ./software/petalinux/create_petalinux_project.sh"
    exit 1
fi

echo "Building PetaLinux project in: $PROJECT_DIR"

# Clean previous builds (optional)
if [ "$1" == "clean" ]; then
    echo "Cleaning previous build..."
    cd "$PROJECT_DIR"
    petalinux-build -x clean
fi

# Build the system
echo "Starting PetaLinux build..."
cd "$PROJECT_DIR"

# Set parallel build jobs based on CPU cores
BUILD_JOBS=$(nproc)
echo "Using $BUILD_JOBS parallel build jobs"

# Build the system
petalinux-build -j $BUILD_JOBS

# Check if build was successful
if [ ! -f "images/linux/image.ub" ]; then
    echo "Error: Build failed - image.ub not found"
    exit 1
fi

if [ ! -f "images/linux/zynq_fsbl.elf" ]; then
    echo "Error: Build failed - FSBL not found"
    exit 1
fi

# Create boot files
echo "Creating boot files..."
BIT_FILE="../system_wrapper.bit"

if [ ! -f "$BIT_FILE" ]; then
    echo "Error: Bitstream not found: $BIT_FILE"
    echo "Please ensure hardware build completed successfully"
    exit 1
fi

petalinux-package --boot --fsbl images/linux/zynq_fsbl.elf \
    --fpga "$BIT_FILE" --u-boot

# Check if BOOT.bin was created
if [ ! -f "images/linux/BOOT.bin" ]; then
    echo "Error: BOOT.bin creation failed"
    exit 1
fi

# Create SD card image (optional)
echo "Creating SD card image..."
petalinux-package --wic --disk-size "8G" --images-dir images/linux/

# Copy important files to parent directory
echo "Copying build artifacts..."
mkdir -p ../deploy
cp images/linux/BOOT.bin ../deploy/
cp images/linux/image.ub ../deploy/
cp "$BIT_FILE" ../deploy/

if [ -f "images/wic/rootfs.wic" ]; then
    cp images/wic/rootfs.wic ../deploy/
fi

# Show build summary
echo ""
echo "PetaLinux build completed successfully!"
echo ""
echo "Build artifacts:"
echo "- BOOT.bin: $(ls -lh images/linux/BOOT.bin | awk '{print $5}')"
echo "- image.ub: $(ls -lh images/linux/image.ub | awk '{print $5}')"
echo "- Bitstream: $(ls -lh "$BIT_FILE" | awk '{print $5}')"
echo ""
echo "Files copied to: ../deploy/"
echo ""
echo "Next steps:"
echo "1. Copy BOOT.bin and image.ub to SD card boot partition"
echo "2. Copy rootfs to SD card root partition"
echo "3. Insert SD card into PYNQ-Z2 and boot"
echo "4. Login and deploy YOLOv3 application"
echo ""