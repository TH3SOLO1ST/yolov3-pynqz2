#!/bin/bash
# Development environment setup script for YOLOv3 on PYNQ-Z2

set -e

echo "Setting up YOLOv3 PYNQ-Z2 development environment..."

# Check if running on supported platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected - proceeding with setup"
else
    echo "Warning: This setup is designed for Linux systems"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r config/requirements.txt

# Set up environment variables
echo "Setting up environment variables..."

# Check for Xilinx tools
if [ -d "/opt/Xilinx" ]; then
    echo "Xilinx tools detected"
    if [ -d "/opt/Xilinx/Vivado/2020.1" ]; then
        export VIVADO=/opt/Xilinx/Vivado/2020.1
        echo "export VIVADO=/opt/Xilinx/Vivado/2020.1" >> ~/.bashrc
    fi

    if [ -d "/opt/pkg/petalinux/2019.2" ]; then
        export PETALINUX=/opt/pkg/petalinux/2019.2
        echo "export PETALINUX=/opt/pkg/petalinux/2019.2" >> ~/.bashrc
    fi
else
    echo "Xilinx tools not found in standard locations"
    echo "Please install Vivado 2020.1 and PetaLinux 2019.2"
fi

# Check for DNNDK
if [ -d "/opt/DNNDK" ]; then
    export DNNDK=/opt/DNNDK
    echo "export DNNDK=/opt/DNNDK" >> ~/.bashrc
else
    echo "DNNDK not found in /opt/DNNDK"
    echo "Please install DNNDK v3.1"
fi

# Source Xilinx settings if available
if [ -n "$VIVADO" ] && [ -f "$VIVADO/settings64.sh" ]; then
    echo "source $VIVADO/settings64.sh" >> ~/.bashrc
    source "$VIVADO/settings64.sh"
fi

if [ -n "$PETALINUX" ] && [ -f "$PETALINUX/settings.sh" ]; then
    echo "source $PETALINUX/settings.sh" >> ~/.bashrc
    source "$PETALINUX/settings.sh"
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p logs
mkdir -p data/test_images
mkdir -p data/test_videos

# Download sample classes file if not exists
if [ ! -f "model/dataset/classes.txt" ]; then
    echo "Creating default COCO classes file..."
    mkdir -p model/dataset
    cat > model/dataset/classes.txt << EOF
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF
fi

# Set up git hooks if git repository
if [ -d ".git" ]; then
    echo "Setting up git hooks..."
    cp scripts/git-hooks/* .git/hooks/ 2>/dev/null || true
    chmod +x .git/hooks/* 2>/dev/null || true
fi

# Create startup script
echo "Creating startup script..."
cat > start_yolo.sh << 'EOF'
#!/bin/bash
# Startup script for YOLOv3 realtime system

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run the application
cd src
python3 main.py --config ../config/default_config.json "$@"
EOF

chmod +x start_yolo.sh

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Install Vivado 2020.1 (if not already installed)"
echo "2. Install PetaLinux 2019.2 (if not already installed)"
echo "3. Install DNNDK v3.1 (if not already installed)"
echo "4. Run: source ~/.bashrc (or restart terminal)"
echo "5. Run: ./start_yolo.sh --help"
echo ""