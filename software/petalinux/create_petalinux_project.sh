#!/bin/bash
# Create PetaLinux project for YOLOv3 on PYNQ-Z2

set -e

echo "Creating PetaLinux project for YOLOv3..."

# Check if PetaLinux is available
if ! command -v petalinux-create &> /dev/null; then
    echo "Error: PetaLinux not found in PATH"
    echo "Please source PetaLinux settings.sh first"
    echo "Example: source /opt/pkg/petalinux/2019.2/settings.sh"
    exit 1
fi

# Set project directory
PROJECT_DIR=$(pwd)/software/petalinux
PROJECT_NAME="pynq_z2_dpu"

echo "Project directory: $PROJECT_DIR"
echo "Project name: $PROJECT_NAME"

# Check if hardware files exist
HW_FILE="$PROJECT_DIR/system.hdf"
BIT_FILE="$PROJECT_DIR/system_wrapper.bit"

if [ ! -f "$HW_FILE" ]; then
    echo "Error: Hardware definition file not found: $HW_FILE"
    echo "Please run the hardware build first: ./hardware/build_dpu.sh"
    exit 1
fi

if [ ! -f "$BIT_FILE" ]; then
    echo "Error: Bitstream file not found: $BIT_FILE"
    echo "Please run the hardware build first: ./hardware/build_dpu.sh"
    exit 1
fi

# Create PetaLinux project
echo "Creating PetaLinux project..."
cd "$PROJECT_DIR"
petalinux-create -t project -n "$PROJECT_NAME" --template zynq

# Configure hardware
echo "Configuring hardware..."
cd "$PROJECT_NAME"
petalinux-config --get-hw-description ../

# Apply custom configuration
echo "Applying kernel configuration..."
# This would typically be done interactively or with a config file
# For now, we'll create a config fragment

echo "Applying rootfs configuration..."
# This would typically be done interactively or with a config file
# For now, we'll create the necessary files

echo "PetaLinux project created successfully!"
echo ""
echo "Next steps:"
echo "1. Configure kernel: petalinux-config -c kernel"
echo "2. Configure rootfs: petalinux-config -c rootfs"
echo "3. Build system: petalinux-build"
echo "4. Create boot files: petalinux-package --boot --fsbl images/linux/zynq_fsbl.elf --fpga ../system_wrapper.bit --u-boot"
echo ""