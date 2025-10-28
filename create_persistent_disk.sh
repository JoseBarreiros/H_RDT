#!/bin/bash
# create_persistent_disk.sh
# Creates persistent disk and copies preprocessed data from current instance

set -e

echo "=========================================="
echo "Creating Persistent Disk for H-RDT Data"
echo "=========================================="

# Configuration
DISK_NAME="hrdt-data-cache"
DISK_SIZE=2000  # 2TB (adjust if needed)
ZONE=$(gcloud config get-value compute/zone 2>/dev/null || echo "us-central1-a")
INSTANCE_NAME=$(hostname -s | cut -d'-' -f1)

echo "Configuration:"
echo "  Instance: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  Disk: $DISK_NAME ($DISK_SIZE GB)"
echo ""

# Check if disk already exists
if gcloud compute disks describe "$DISK_NAME" --zone="$ZONE" >/dev/null 2>&1; then
    echo "⚠️  Disk $DISK_NAME already exists!"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Create disk
echo ""
echo "📦 Step 1: Creating persistent disk..."
gcloud compute disks create "$DISK_NAME" \
    --size="$DISK_SIZE" \
    --type=pd-ssd \
    --zone="$ZONE" \
    --description="H-RDT preprocessed EgoDex dataset cache"

echo "✅ Disk created"

# Step 2: Attach to instance
echo ""
echo "🔗 Step 2: Attaching disk to instance..."
gcloud compute instances attach-disk "$INSTANCE_NAME" \
    --disk "$DISK_NAME" \
    --device-name hrdt-data \
    --zone="$ZONE"

echo "✅ Disk attached"

# Step 3: Format and mount (on this instance)
echo ""
echo "💾 Step 3: Formatting and mounting disk..."
echo "This will format the disk (erases all data). Continue?"
read -p "Type 'yes' to continue: " -r
echo
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Aborted."
    exit 1
fi

# Wait for disk to be available
sleep 3

# Format
sudo mkfs.ext4 -F /dev/disk/by-id/google-hrdt-data-cache

# Mount
sudo mkdir -p /mnt/disks/hrdt-data
sudo mount -o discard,defaults /dev/disk/by-id/google-hrdt-data-cache /mnt/disks/hrdt-data
sudo chown -R $USER:$USER /mnt/disks/hrdt-data

# Make permanent
if ! grep -q "hrdt-data" /etc/fstab; then
    echo UUID=$(sudo blkid -s UUID -o value /dev/disk/by-id/google-hrdt-data-cache) \
        /mnt/disks/hrdt-data ext4 discard,defaults,nofail 0 2 | \
        sudo tee -a /etc/fstab
fi

echo "✅ Disk mounted at /mnt/disks/hrdt-data"

# Step 4: Copy data
echo ""
echo "📋 Step 4: Copying preprocessed data..."
echo "This will copy ~1.8TB of data. This may take 1-2 hours."
echo ""

DATA_SOURCE="/home/jose-barreiros/egodex/organized"

if [ ! -d "$DATA_SOURCE" ]; then
    echo "❌ Data source not found: $DATA_SOURCE"
    echo "Please update DATA_SOURCE in this script"
    exit 1
fi

# Estimate time
echo "Source: $DATA_SOURCE"
du -sh "$DATA_SOURCE"
echo ""
read -p "Continue with copy? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo "Starting rsync..."
rsync -avh --progress "$DATA_SOURCE/" /mnt/disks/hrdt-data/egodex/

echo "✅ Data copy complete"

# Step 5: Verify
echo ""
echo "🔍 Step 5: Verifying data..."
du -sh /mnt/disks/hrdt-data/egodex/
ls -lh /mnt/disks/hrdt-data/egodex/ | head -10

# Step 6: Create snapshot
echo ""
echo "📸 Step 6: Creating snapshot..."
SNAPSHOT_NAME="hrdt-preprocessed-$(date +%Y%m%d)"
gcloud compute disks snapshot "$DISK_NAME" \
    --snapshot-names "$SNAPSHOT_NAME" \
    --zone="$ZONE" \
    --description="Complete preprocessed EgoDex dataset for H-RDT"

echo "✅ Snapshot created: $SNAPSHOT_NAME"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Data location: /mnt/disks/hrdt-data/egodex/"
echo "Snapshot: $SNAPSHOT_NAME"
echo ""
echo "Next steps:"
echo "1. Update config to use this data:"
echo "   export HRDT_OUTPUT_DIR=\"/mnt/disks/hrdt-data/egodex\""
echo ""
echo "2. Create new instances with this disk:"
echo "   gcloud compute instances create instance-name \\"
echo "       --disk=\"name=hrdt-data-instance1,source-snapshot=$SNAPSHOT_NAME\""
echo ""
echo "3. Or use INSTANCE_SETUP.md for detailed instructions"
echo "=========================================="

