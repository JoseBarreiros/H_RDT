# Persistent Disk Setup for Existing Preprocessed Data

This guide explains how to create a persistent disk and move your preprocessed EgoDex data to it on your current instance.

## Overview

**What this does:**
- Creates a 2TB persistent disk for the complete dataset
- Attaches it to your current instance
- Copies your complete data (MP4 videos + preprocessed .hdf5 + .pt files)
- Creates a snapshot for sharing with other instances

**Why:**
- Avoid re-preprocessing on each instance
- Have complete dataset available (videos + preprocessed data)
- Share data across multiple training instances
- Enable parallel scaling law experiments

**Data location:**
- **EgoDex Data**: Currently at `~/egodex/organized/` (~1.8TB total)
  - Will be at: `/mnt/disks/hrdt-data/egodex/organized/`
  - Includes:
    - MP4 videos: ~850GB
    - Preprocessed .hdf5 files: ~477GB
    - Preprocessed .pt files: ~477GB
    - Total: ~1.8TB

- **RobotWin2 Table 8 Data**: Currently at `/mnt/disks/hrdt-data/robotwin2/table8_tasks/` (~17GB total)
  - Already on persistent disk: `/mnt/disks/hrdt-data/robotwin2/table8_tasks/`
  - Includes:
    - Compressed zip files: ~6.6GB
    - Extracted HDF5 data: ~10.4GB
    - Language embeddings: Pre-computed in repository
    - Total: ~17GB

## Prerequisites

- Access to your current GCP instance
- Preprocessed data at `~/egodex/organized/`
- `gcloud` CLI installed and configured

## Step-by-Step Instructions

### Step 0: Verify RobotWin2 Table 8 Data (Optional)

If you want to verify your Table 8 data is ready for fine-tuning:

```bash
# Verify Table 8 data integrity and training compatibility
python verify_robotwin2_dataset.py --data_root /mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted

# Expected output: "🎉 All Table 8 tasks verified successfully!"
```

### Step 1: Create Persistent Disk

Create a 2TB SSD persistent disk:

```bash
gcloud compute disks create hrdt-data-cache \
    --size=2000 \
    --type=pd-ssd \
    --zone=us-central1-a \
    --description="H-RDT preprocessed EgoDex dataset"
```

**Note:** Using 2TB disk because your data is ~1.8TB. Adjust `--zone` to match your instance's zone.

### Step 2: Get Instance Name and Attach Disk

First, get your instance name:

```bash
hostname
```

Then attach the disk (replace `INSTANCE-NAME` with the output):

```bash
gcloud compute instances attach-disk INSTANCE-NAME \
    --disk hrdt-data-cache \
    --device-name hrdt-data \
    --zone=us-central1-a
```

**Note:** If you're unsure of your zone, run:
```bash
gcloud config get-value compute/zone
```

### Step 3: Format and Mount Disk

**⚠️ WARNING:** Formatting erases all data on the disk. This is safe for a new disk.

```bash
# Format the disk
sudo mkfs.ext4 -F /dev/disk/by-id/google-hrdt-data-cache

# Create mount point
sudo mkdir -p /mnt/disks/hrdt-data

# Mount disk
sudo mount -o discard,defaults /dev/disk/by-id/google-hrdt-data-cache /mnt/disks/hrdt-data

# Fix ownership
sudo chown -R $USER:$USER /mnt/disks/hrdt-data
```

**Make mount permanent** (survives reboots):

```bash
# Add to /etc/fstab
echo UUID=$(sudo blkid -s UUID -o value /dev/disk/by-id/google-hrdt-data-cache) \
    /mnt/disks/hrdt-data ext4 discard,defaults,nofail 0 2 | \
    sudo tee -a /etc/fstab
```

Verify the mount:

```bash
df -h /mnt/disks/hrdt-data
```

You should see the 2TB disk mounted.

### Step 4: Copy Preprocessed Data

This will copy ~1.8TB of data. **Expected time: 1-2 hours**

```bash
mkdir -p /mnt/disks/hrdt-data/egodex/organized
rsync -avh --progress ~/egodex/organized/ /mnt/disks/hrdt-data/egodex/organized/
```

**Monitoring progress:**
- Open another terminal and run:
```bash
watch -n 5 du -sh /mnt/disks/hrdt-data/
```

**If interrupted:**
- rsync will resume where it left off when you run it again

**Verify copy completed:**
```bash
du -sh /mnt/disks/hrdt-data/egodex/
# Should show ~1.8TB
```

### Step 5: Create Snapshot

After the copy completes, create a snapshot. This allows you to create new instances with this data:

```bash
gcloud compute disks snapshot hrdt-data-cache \
    --snapshot-names hrdt-preprocessed-$(date +%Y%m%d) \
    --zone=us-central1-a \
    --description="Complete preprocessed EgoDex dataset for H-RDT scaling law experiments"
```

**Verify snapshot:**
```bash
gcloud compute snapshots list --filter="name:hrdt-preprocessed"
```

## After Setup

### ✅ Your Data Is Now Available At

**EgoDex Data:**
```bash
/mnt/disks/hrdt-data/egodex/organized
```

**RobotWin2 Table 8 Data:**
```bash
/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted
```

### ✅ Verify Your Data

**Verify EgoDex data:**
```bash
python verify_dataset.py --data_root /mnt/disks/hrdt-data/egodex/organized
```

**Verify RobotWin2 Table 8 data:**
```bash
python verify_robotwin2_dataset.py --data_root /mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted
```

### ✅ Update Training Configuration

Update your training scripts to use the persistent disk data:

**For EgoDex pre-training:**
```bash
# In ~/.config/hrdt/config.sh or your training script
export EGODEX_DATA_ROOT="/mnt/disks/hrdt-data/egodex/organized"
export HRDT_OUTPUT_DIR="/mnt/disks/hrdt-data/egodex/organized"
```

**For RobotWin2 Table 8 fine-tuning:**
```bash
# Set RobotWin2 data path
export ROBOTWIN_DATA_ROOT="/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted"

# Run fine-tuning
accelerate launch main.py \
    --dataset_name="robotwin_agilex" \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --config_path configs/hrdt_finetune.yaml \
    --output_dir ./checkpoints/table8_finetune \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --dataset_type finetune \
    --report_to wandb
```

### ✅ Start Training

```bash
source hrdt_env/bin/activate
export HRDT_OUTPUT_DIR="/mnt/disks/hrdt-data/egodex/organized"
bash pretrain.sh
```

### ✅ Create New Instances with This Data

See `INSTANCE_SETUP.md` Step 2 for creating new training instances with this disk.

## Cost Estimate

**Persistent Disk:**
- 2TB SSD: ~$320/month
- 2TB Standard: ~$80/month (slower, but fine for this use)

**Recommendation:** Use standard disk if cost is a concern. Performance difference is minimal for reading preprocessed data.

## Troubleshooting

### "Disk already attached"
```bash
# Detach first
gcloud compute instances detach-disk INSTANCE-NAME \
    --disk hrdt-data-cache --zone us-central1-a
```

### "Permission denied"
```bash
sudo chown -R $USER:$USER /mnt/disks/hrdt-data
```

### "Cannot mount: no such file or directory"
```bash
# Wait a few seconds after attach command, then try again
sleep 5
sudo mount /dev/disk/by-id/google-hrdt-data-cache /mnt/disks/hrdt-data
```

### "Out of disk space"
```bash
# Resize disk
gcloud compute disks resize hrdt-data-cache --size=2500 --zone us-central1-a

# Expand filesystem
sudo resize2fs /dev/disk/by-id/google-hrdt-data-cache
```

### Check disk status
```bash
# List disks
gcloud compute disks list | grep hrdt

# Check snapshots
gcloud compute snapshots list | grep hrdt

# Check mount
df -h | grep hrdt
```

## Next Steps

1. ✅ You now have persistent disk with all preprocessed data
2. 📖 See `INSTANCE_SETUP.md` to create training instances
3. 📖 See `SCALING_LAW_GUIDE.md` to run experiments with different data percentages
4. 🚀 Start training!

## Alternative: Automated Script

If you prefer an automated approach, run:

```bash
bash create_persistent_disk.sh
```

This performs all steps automatically with prompts for confirmation.

