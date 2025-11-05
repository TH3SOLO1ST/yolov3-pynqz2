#!/bin/bash
# Build script for DPU hardware design

set -e

echo "Building DPU hardware design for PYNQ-Z2..."

# Check if Vivado is available
if ! command -v vivado &> /dev/null; then
    echo "Error: Vivado not found in PATH"
    echo "Please source Vivado settings64.sh first"
    exit 1
fi

# Set project directory
PROJ_DIR=$(pwd)/hardware/pynq_z2_dpu
SCRIPT_DIR=$(pwd)/hardware

echo "Project directory: $PROJ_DIR"

# Create project directory if it doesn't exist
mkdir -p "$PROJ_DIR"

# Run Vivado script to create project
echo "Creating Vivado project..."
cd "$PROJ_DIR"
vivado -mode batch -source "$SCRIPT_DIR/create_vivado_project.tcl"

# Check if project was created
if [ ! -f "pynq_z2_dpu.xpr" ]; then
    echo "Error: Failed to create Vivado project"
    exit 1
fi

# Build the project (non-interactive)
echo "Building bitstream..."
vivado -mode batch -source "$SCRIPT_DIR/build_bitstream.tcl"

# Check if bitstream was generated
BITSTREAM="$PROJ_DIR/pynq_z2_dpu.runs/impl_1/system_wrapper.bit"
if [ ! -f "$BITSTREAM" ]; then
    echo "Error: Bitstream generation failed"
    exit 1
fi

# Generate hardware definition file
echo "Exporting hardware..."
vivado -mode batch -source "$SCRIPT_DIR/export_hardware.tcl"

# Check if hardware definition was exported
HWDEF="$PROJ_DIR/system_wrapper.hdf"
if [ ! -f "$HWDEF" ]; then
    echo "Error: Hardware export failed"
    exit 1
fi

# Copy files to expected locations
echo "Copying generated files..."
mkdir -p ../software/petalinux/
cp "$BITSTREAM" ../software/petalinux/system_wrapper.bit
cp "$HWDEF" ../software/petalinux/system.hdf

# Generate hardware handoff file for PYNQ
if [ -f "$PROJ_DIR/pynq_z2_dpu.srcs/sources_1/bd/system/hw_handoff/system.hwh" ]; then
    cp "$PROJ_DIR/pynq_z2_dpu.srcs/sources_1/bd/system/hw_handoff/system.hwh" ../software/petalinux/system.hwh
fi

echo ""
echo "DPU hardware build completed successfully!"
echo ""
echo "Generated files:"
echo "- Bitstream: $BITSTREAM"
echo "- Hardware definition: $HWDEF"
echo "- Hardware handoff: ${HWDEF%.hdf}.hwh"
echo ""
echo "Files copied to: ../software/petalinux/"
echo ""
echo "Next steps:"
echo "1. Build PetaLinux using the exported hardware"
echo "2. Run: cd ../software/petalinux && petalinux-build"
echo ""