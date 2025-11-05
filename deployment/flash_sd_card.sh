#!/bin/bash
# Flash prepared SD card image to physical SD card

set -e

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <sd_device>"
    echo "Example: $0 /dev/sdb"
    echo ""
    echo "WARNING: This will completely erase the specified device!"
    echo "Please double-check the device path before proceeding."
    exit 1
fi

SD_DEVICE=$1
PROJECT_ROOT=$(pwd)
SD_CARD_DIR="$PROJECT_ROOT/deployment/sd_card"

# Verify SD card directory exists
if [ ! -d "$SD_CARD_DIR" ]; then
    echo "Error: SD card preparation directory not found: $SD_CARD_DIR"
    echo "Please run: ./deployment/prepare_sd_card.sh"
    exit 1
fi

# Safety checks
echo "WARNING: This will completely erase all data on $SD_DEVICE"
echo "This operation is irreversible!"
echo ""

# Verify device exists and is a block device
if [ ! -b "$SD_DEVICE" ]; then
    echo "Error: $SD_DEVICE is not a block device"
    exit 1
fi

# Check if device is mounted
if mount | grep -q "$SD_DEVICE"; then
    echo "Error: $SD_DEVICE has mounted partitions"
    echo "Please unmount all partitions first:"
    echo "  sudo umount ${SD_DEVICE}*"
    exit 1
fi

# Show device information
echo "Device information:"
lsblk "$SD_DEVICE"
echo ""

# Get device size
DEVICE_SIZE=$(lsblk -b -d -n -o SIZE "$SD_DEVICE")
DEVICE_SIZE_GB=$((DEVICE_SIZE / 1024 / 1024 / 1024))

echo "Device size: ${DEVICE_SIZE_GB}GB"

if [ $DEVICE_SIZE_GB -lt 8 ]; then
    echo "Error: SD card is too small (minimum 8GB required)"
    exit 1
fi

# Final confirmation
echo "This will flash YOLOv3 PYNQ-Z2 system to $SD_DEVICE"
echo "All existing data will be permanently destroyed!"
echo ""
read -p "Type 'YES' to continue: " confirmation

if [ "$confirmation" != "YES" ]; then
    echo "Operation cancelled"
    exit 1
fi

echo ""
echo "Starting SD card flash process..."

# Create partition table
echo "Creating partition table..."
parted "$SD_DEVICE" --script -- mklabel msdos
parted "$SD_DEVICE" --script -- mkpart primary fat32 1MiB 101MiB
parted "$SD_DEVICE" --script -- mkpart primary ext4 101MiB 100%
parted "$SD_DEVICE" --script -- set 1 boot on

# Wait for partition table to be recognized
sleep 2

# Format partitions
echo "Formatting partitions..."

# Determine partition names based on device
if [[ $SD_DEVICE == *mmcblk* ]]; then
    BOOT_PARTITION="${SD_DEVICE}p1"
    ROOT_PARTITION="${SD_DEVICE}p2"
else
    BOOT_PARTITION="${SD_DEVICE}1"
    ROOT_PARTITION="${SD_DEVICE}2"
fi

echo "Boot partition: $BOOT_PARTITION"
echo "Root partition: $ROOT_PARTITION"

# Format boot partition
mkfs.vfat -F 32 -n "BOOT" "$BOOT_PARTITION"

# Format root partition
mkfs.ext4 -F -L "ROOTFS" "$ROOT_PARTITION"

# Create mount points
MOUNT_BOOT="/tmp/sd_boot"
MOUNT_ROOT="/tmp/sd_root"

mkdir -p "$MOUNT_BOOT" "$MOUNT_ROOT"

# Mount partitions
echo "Mounting partitions..."
mount "$BOOT_PARTITION" "$MOUNT_BOOT"
mount "$ROOT_PARTITION" "$MOUNT_ROOT"

# Copy files
echo "Copying files to SD card..."

# Copy boot files
echo "  Copying boot files..."
rsync -av "$SD_CARD_DIR/boot/" "$MOUNT_BOOT/"

# Copy root filesystem
echo "  Copying root filesystem..."
rsync -av "$SD_CARD_DIR/root/" "$MOUNT_ROOT/"

# Set correct permissions
echo "Setting permissions..."
chmod 755 "$MOUNT_ROOT"
chmod 755 "$MOUNT_ROOT/home"
chmod 755 "$MOUNT_ROOT/home/yolo"

# Sync filesystems
echo "Syncing filesystems..."
sync

# Unmount partitions
echo "Unmounting partitions..."
umount "$MOUNT_BOOT"
umount "$MOUNT_ROOT"

# Clean up mount points
rmdir "$MOUNT_BOOT" "$MOUNT_ROOT"

# Run filesystem check
echo "Running filesystem checks..."
fsck.vfat -a "$BOOT_PARTITION" || true
fsck.ext4 -f -y "$ROOT_PARTITION" || true

echo ""
echo "SD card flash completed successfully!"
echo ""
echo "Summary:"
echo "  Boot partition: $(lsblk -n -o FSTYPE,SIZE "$BOOT_PARTITION")"
echo "  Root partition: $(lsblk -n -o FSTYPE,SIZE "$ROOT_PARTITION")"
echo "  Boot label: $(blkid -o value -s LABEL "$BOOT_PARTITION")"
echo "  Root label: $(blkid -o value -s LABEL "$ROOT_PARTITION")"
echo ""
echo "Next steps:"
echo "1. Remove SD card from computer"
echo "2. Insert SD card into PYNQ-Z2"
echo "3. Connect USB webcam and HDMI monitor"
echo "4. Apply power to PYNQ-Z2"
echo "5. Wait for boot (2-3 minutes)"
echo "6. Check for YOLOv3 startup on HDMI output"
echo ""
echo "To monitor system after boot:"
echo "  Find PYNQ-Z2 IP address (check your router)"
echo "  SSH: ssh root@<IP_ADDRESS>"
echo "  Status: /home/yolo/system_status.sh"
echo ""