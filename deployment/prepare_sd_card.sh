#!/bin/bash
# Prepare SD card for PYNQ-Z2 YOLOv3 deployment

set -e

echo "Preparing SD card for PYNQ-Z2 YOLOv3 deployment..."

# Check if running as root for SD card operations
if [ "$EUID" -ne 0 ]; then
    echo "Note: Some operations may require sudo privileges"
fi

# Set paths
PROJECT_ROOT=$(pwd)
SOFTWARE_DIR="$PROJECT_ROOT/software/petalinux"
DEPLOY_DIR="$PROJECT_ROOT/deployment"

# Check if required files exist
BOOT_BIN="$SOFTWARE_DIR/pynq_z2_dpu/images/linux/BOOT.bin"
IMAGE_UB="$SOFTWARE_DIR/pynq_z2_dpu/images/linux/image.ub"
ROOTFS_TAR="$SOFTWARE_DIR/pynq_z2_dpu/images/linux/rootfs.tar.gz"

if [ ! -f "$BOOT_BIN" ]; then
    echo "Error: BOOT.bin not found: $BOOT_BIN"
    echo "Please run PetaLinux build first: ./software/petalinux/build_petalinux.sh"
    exit 1
fi

if [ ! -f "$IMAGE_UB" ]; then
    echo "Error: image.ub not found: $IMAGE_UB"
    echo "Please run PetaLinux build first: ./software/petalinux/build_petalinux.sh"
    exit 1
fi

if [ ! -f "$ROOTFS_TAR" ]; then
    echo "Error: rootfs.tar.gz not found: $ROOTFS_TAR"
    echo "Please run PetaLinux build first: ./software/petalinux/build_petalinux.sh"
    exit 1
fi

# Create deployment directory
mkdir -p "$DEPLOY_DIR/sd_card"
SD_CARD_DIR="$DEPLOY_DIR/sd_card"

# Create boot directory structure
mkdir -p "$SD_CARD_DIR/boot"
mkdir -p "$SD_CARD_DIR/root"

echo "Copying boot files..."
cp "$BOOT_BIN" "$SD_CARD_DIR/boot/"
cp "$IMAGE_UB" "$SD_CARD_DIR/boot/"

# Create boot.scr (U-Boot script)
echo "Creating boot script..."
cat > "$SD_CARD_DIR/boot/boot.scr" << 'EOF'
# U-Boot script for YOLOv3 PYNQ-Z2

# Set boot arguments
setenv bootargs 'console=ttyPS0,115200 root=/dev/mmcblk0p2 rw rootwait earlyprintk'

# Load kernel and device tree
fatload mmc 0:1 0x3000000 image.ub
fatload mmc 0:1 0x2000000 BOOT.bin

# Boot the system
bootm 0x3000000
EOF

# Create extlinux configuration (alternative boot method)
mkdir -p "$SD_CARD_DIR/boot/extlinux"
cat > "$SD_CARD_DIR/boot/extlinux/extlinux.conf" << 'EOF'
default YOLOv3-PYNQ-Z2

label YOLOv3-PYNQ-Z2
    kernel /image.ub
    devicetree /devicetree.dtb
    append console=ttyPS0,115200 root=/dev/mmcblk0p2 rw rootwait earlyprintk
    fdtdir /boot/
EOF

echo "Extracting root filesystem..."
if [ -f "$ROOTFS_TAR" ]; then
    cd "$SD_CARD_DIR/root"
    tar -xzf "$ROOTFS_TAR"
    cd "$PROJECT_ROOT"
else
    echo "Warning: rootfs.tar.gz not found, you'll need to extract it manually"
fi

# Copy application files
echo "Copying YOLOv3 application..."
mkdir -p "$SD_CARD_DIR/root/home/yolo"
cp -r "$PROJECT_ROOT/src" "$SD_CARD_DIR/root/home/yolo/"
cp -r "$PROJECT_ROOT/model" "$SD_CARD_DIR/root/home/yolo/"
cp -r "$PROJECT_ROOT/config" "$SD_CARD_DIR/root/home/yolo/"
cp "$PROJECT_ROOT/start_yolo.sh" "$SD_CARD_DIR/root/home/yolo/"

# Create startup script
echo "Creating startup script..."
cat > "$SD_CARD_DIR/root/home/yolo/start_yolo_pynq.sh" << 'EOF'
#!/bin/bash
# Startup script for YOLOv3 on PYNQ-Z2

echo "Starting YOLOv3 on PYNQ-Z2..."

# Set environment variables
export PYTHONPATH="/home/yolo/src:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/lib:$LD_LIBRARY_PATH"

# Change to application directory
cd /home/yolo

# Check if DPU bitstream is loaded
if [ ! -d "/sys/class/dpu" ]; then
    echo "Loading DPU bitstream..."
    # Load bitstream (this may need adjustment based on your PYNQ version)
    mkdir -p /lib/firmware
    cp /boot/system_wrapper.bit /lib/firmware/
    echo 0 > /sys/class/firmware_loader/firmware/loading
    echo -n "system_wrapper.bit" > /sys/class/firmware_loader/firmware/data
    echo 1 > /sys/class/firmware_loader/firmware/loading
fi

# Start YOLOv3 application
echo "Starting YOLOv3 realtime detection..."
cd src
python3 main.py --config ../config/default_config.json

echo "YOLOv3 application stopped"
EOF

chmod +x "$SD_CARD_DIR/root/home/yolo/start_yolo_pynq.sh"

# Create systemd service for auto-start
echo "Creating systemd service..."
mkdir -p "$SD_CARD_DIR/root/etc/systemd/system"
cat > "$SD_CARD_DIR/root/etc/systemd/system/yolo-pynq.service" << 'EOF'
[Unit]
Description=YOLOv3 Realtime Detection on PYNQ-Z2
After=network.target

[Service]
Type=simple
User=yolo
WorkingDirectory=/home/yolo
ExecStart=/home/yolo/start_yolo_pynq.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
mkdir -p "$SD_CARD_DIR/root/etc/systemd/system/multi-user.target.wants"
ln -s "/etc/systemd/system/yolo-pynq.service" "$SD_CARD_DIR/root/etc/systemd/system/multi-user.target.wants/"

# Create network configuration
echo "Creating network configuration..."
mkdir -p "$SD_CARD_DIR/root/etc/systemd/network"
cat > "$SD_CARD_DIR/root/etc/systemd/network/20-wired.network" << 'EOF'
[Match]
Name=eth0

[Network]
DHCP=yes
EOF

# Create yolo user if not exists
echo "Creating yolo user..."
cat >> "$SD_CARD_DIR/root/etc/passwd" << 'EOF'
yolo:x:1000:1000:YOLOv3 User:/home/yolo:/bin/bash
EOF

cat >> "$SD_CARD_DIR/root/etc/shadow" << 'EOF'
yolo:$6$rounds=4096$saltsalt$hash:18600:0:99999:7:::
EOF

cat >> "$SD_CARD_DIR/root/etc/group" << 'EOF'
yolo:x:1000:
EOF

# Create sudoers entry for yolo user
mkdir -p "$SD_CARD_DIR/root/etc/sudoers.d"
echo "yolo ALL=(ALL) NOPASSWD:ALL" > "$SD_CARD_DIR/root/etc/sudoers.d/yolo"

# Create fstab for proper mount points
echo "Creating fstab..."
cat > "$SD_CARD_DIR/root/etc/fstab" << 'EOF'
# <file system> <mount point>   <type>  <options>       <dump>  <pass>
/dev/mmcblk0p1  /boot           vfat    defaults        0       2
/dev/mmcblk0p2  /               ext4    errors=remount-ro 0       1
proc            /proc           proc    defaults        0       0
tmpfs           /tmp            tmpfs   defaults        0       0
tmpfs           /var/log        tmpfs   defaults        0       0
EOF

# Create hostname file
echo "yolo-pynq-z2" > "$SD_CARD_DIR/root/etc/hostname"

# Create hosts file
cat > "$SD_CARD_DIR/root/etc/hosts" << 'EOF'
127.0.0.1   localhost
127.0.1.1   yolo-pynq-z2
EOF

# Copy additional configuration files
echo "Copying additional configuration..."

# Copy compiled model if exists
if [ -f "$PROJECT_ROOT/model/compiled/yolov3_custom.elf" ]; then
    mkdir -p "$SD_CARD_DIR/root/home/yolo/model/compiled"
    cp "$PROJECT_ROOT/model/compiled/"*.elf "$SD_CARD_DIR/root/home/yolo/model/compiled/" 2>/dev/null || true
    cp "$PROJECT_ROOT/model/compiled/"*.dpu "$SD_CARD_DIR/root/home/yolo/model/compiled/" 2>/dev/null || true
fi

# Create log directory
mkdir -p "$SD_CARD_DIR/root/var/log/yolo"

# Create performance monitoring script
cat > "$SD_CARD_DIR/root/home/yolo/monitor_performance.sh" << 'EOF'
#!/bin/bash
# Performance monitoring script for YOLOv3 on PYNQ-Z2

LOG_FILE="/var/log/yolo/performance.log"
INTERVAL=30

echo "Starting performance monitoring..."

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Get system stats
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    TEMP=$(cat /sys/class/hwmon/hwmon0/temp1_input 2>/dev/null | cut -c1-2 || echo "N/A")

    # Log stats
    echo "$TIMESTAMP,CPU:$CPU_USAGE%,MEM:$MEMORY_USAGE%,TEMP:$TEMP" >> $LOG_FILE

    # Check if YOLOv3 process is running
    if pgrep -f "python3 main.py" > /dev/null; then
        echo "$TIMESTAMP,STATUS:Running" >> $LOG_FILE
    else
        echo "$TIMESTAMP,STATUS:Stopped" >> $LOG_FILE
    fi

    sleep $INTERVAL
done
EOF

chmod +x "$SD_CARD_DIR/root/home/yolo/monitor_performance.sh"

# Create system monitoring script
cat > "$SD_CARD_DIR/root/home/yolo/system_status.sh" << 'EOF'
#!/bin/bash
# System status script

echo "=== YOLOv3 PYNQ-Z2 System Status ==="
echo "Time: $(date)"
echo "Uptime: $(uptime -p)"
echo ""

echo "=== CPU and Memory ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "Temperature: $(cat /sys/class/hwmon/hwmon0/temp1_input 2>/dev/null | cut -c1-2 || echo "N/A")°C"
echo ""

echo "=== Storage ==="
df -h | grep -E "(Filesystem|/dev/mmc)"
echo ""

echo "=== Network ==="
ip addr show eth0 | grep "inet " | awk '{print "IP: " $2}' || echo "No network connection"
echo ""

echo "=== YOLOv3 Process ==="
if pgrep -f "python3 main.py" > /dev/null; then
    echo "YOLOv3: Running (PID: $(pgrep -f "python3 main.py"))"
else
    echo "YOLOv3: Not running"
fi
echo ""

echo "=== Recent Logs ==="
tail -10 /var/log/yolo/performance.log 2>/dev/null || echo "No performance logs available"
EOF

chmod +x "$SD_CARD_DIR/root/home/yolo/system_status.sh"

# Calculate directory sizes
echo ""
echo "SD Card Preparation Summary:"
echo "==========================="
echo "Boot partition: $(du -sh "$SD_CARD_DIR/boot" | cut -f1)"
echo "Root partition: $(du -sh "$SD_CARD_DIR/root" | cut -f1)"
echo "Total size: $(du -sh "$SD_CARD_DIR" | cut -f1)"

# Create SD card flashing instructions
cat > "$DEPLOY_DIR/FLASH_SD_CARD.md" << 'EOF'
# SD Card Flashing Instructions

## Required Materials
- MicroSD card (32GB Class 10 recommended)
- SD card reader
- Computer with Linux/macOS/Windows

## Flashing Instructions

### On Linux
1. Insert SD card into reader
2. Identify SD card device: `lsblk`
3. Unmount if auto-mounted: `sudo umount /dev/sdX*`
4. Flash using script: `sudo ./deployment/flash_sd_card.sh /dev/sdX`

### On macOS
1. Insert SD card into reader
2. Identify SD card: `diskutil list`
3. Unmount: `diskutil unmountDisk /dev/diskX`
4. Flash using script: `sudo ./deployment/flash_sd_card.sh /dev/diskX`

### On Windows
1. Use Rufus or Etcher
2. Select SD card device
3. Flash using provided image files

## Manual Flashing (if script fails)
1. Create two partitions on SD card:
   - Partition 1: 100MB, FAT32 (bootable)
   - Partition 2: Remaining space, ext4

2. Mount boot partition and copy files from `deployment/sd_card/boot/`

3. Mount root partition and copy files from `deployment/sd_card/root/`

## Post-Flash Setup
1. Insert SD card into PYNQ-Z2
2. Connect USB webcam and HDMI monitor
3. Apply power
4. Wait for boot (2-3 minutes)
5. SSH into PYNQ-Z2 (default: root/root)
6. Verify YOLOv3 is running: `ps aux | grep python3`

## Troubleshooting
- If boot fails: Check SD card integrity and try re-flashing
- If no video: Verify HDMI connection and check camera compatibility
- For issues: See docs/TROUBLESHOOTING.md
EOF

echo ""
echo "SD card preparation completed!"
echo ""
echo "Files prepared in: $SD_CARD_DIR"
echo "Flash instructions: $DEPLOY_DIR/FLASH_SD_CARD.md"
echo ""
echo "Next steps:"
echo "1. Flash SD card using: sudo ./deployment/flash_sd_card.sh /dev/sdX"
echo "2. Insert into PYNQ-Z2 and boot"
echo "3. Connect to SSH: ssh root@192.168.x.x"
echo "4. Check status: /home/yolo/system_status.sh"
echo ""