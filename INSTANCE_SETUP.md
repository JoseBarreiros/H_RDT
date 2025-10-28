# Multi-Instance Setup Guide: H-RDT Scaling Law Experiments

This guide explains how to run H-RDT scaling law experiments across multiple GCP instances efficiently using a shared persistent disk.

## Strategy Overview

**The Problem:**
- Preprocessing takes 12-24 hours
- Dataset is ~1.8TB (includes MP4 videos + preprocessed files)
- You want to run multiple experiments (different data percentages)
- Don't want to duplicate preprocessing or data

**The Solution:**
1. Preprocess once → save to persistent disk
2. Create snapshots → attach to training instances
3. Each instance uses same data, different `--data_percentage`
4. No duplication, fast startup

## Architecture

```
┌──────────────────────────────────────┐
│   Preprocessing Instance              │
│   - Preprocesses EgoDex once         │
│   - Saves to /mnt/disks/hrdt-data/   │
└──────────────────┬───────────────────┘
                    │ Creates snapshot
                    ↓
        ┌──────────────────────────┐
        │  Persistent Disk (2TB)   │
        │  /mnt/disks/hrdt-data/   │
        │  - MP4 videos            │
        │  - HDF5 files            │
        │  - T5 embeddings (.pt)   │
        │  - Statistics            │
        └──────┬─────────┬─────────┘
               │         │
        Instance 1    Instance 2
       (p=10%)       (p=25%)
```

## Step-by-Step Setup

### Step 1: Preprocessing Instance

**📖 If you already have preprocessed data on an existing instance, see [PERSISTENT_DISK_SETUP.md](PERSISTENT_DISK_SETUP.md) for moving it to a persistent disk.**

**1.1 Create and Attach Disk**

```bash
# Create 2TB persistent disk for complete dataset (one-time)
# Note: Complete dataset is ~1.8TB (MP4 videos + preprocessed files)
gcloud compute disks create hrdt-data-cache \
    --size=2000 --type=pd-ssd \
    --zone=us-central1-a \
    --description="H-RDT complete EgoDex dataset cache (includes MP4s)"

# Attach to preprocessing instance
gcloud compute instances attach-disk YOUR_PREPROCESSING_INSTANCE \
    --disk hrdt-data-cache \
    --device-name hrdt-data \
    --zone us-central1-a
```

**1.2 Format and Mount Disk (on preprocessing instance)**

```bash
# SSH into preprocessing instance
gcloud compute ssh YOUR_PREPROCESSING_INSTANCE --zone us-central1-a

# Format disk (ONLY ONCE - this erases data!)
sudo mkfs.ext4 -F /dev/disk/by-id/google-hrdt-data-cache

# Create mount point
sudo mkdir -p /mnt/disks/hrdt-data

# Mount disk
sudo mount -o discard,defaults /dev/disk/by-id/google-hrdt-data-cache /mnt/disks/hrdt-data

# Change ownership
sudo chown jose-barreiros:jose-barreiros /mnt/disks/hrdt-data

# Make mount permanent (survives reboots)
echo UUID=$(sudo blkid -s UUID -o value /dev/disk/by-id/google-hrdt-data-cache) \
    /mnt/disks/hrdt-data ext4 discard,defaults,nofail 0 2 | \
    sudo tee -a /etc/fstab
```

**1.3 Setup Instance**

```bash
# On preprocessing instance
cd ~
git clone <YOUR_REPO> H_RDT  # or copy from existing instance
cd H_RDT

# Run setup script
bash setup_instance.sh

# Activate environment
source ~/.config/hrdt/activate.sh

# Set data paths
export EGODEX_DATA_ROOT="/home/jose-barreiros/egodex/organized"
export HRDT_OUTPUT_DIR="/mnt/disks/hrdt-data/processed"
```

**1.4 Run Preprocessing**

```bash
# Preprocess all data (once)
source datasets/pretrain/setup_pretrain.sh
bash datasets/pretrain/run_pretrain_pipeline.sh

# This creates:
# /mnt/disks/hrdt-data/processed/*.hdf5
# /mnt/disks/hrdt-data/processed/*.pt
# /mnt/disks/hrdt-data/processed/egodex_stat.json

# Verify
python verify_dataset.py --data_root "$EGODEX_DATA_ROOT"
```

**1.5 Create Snapshot**

```bash
# After preprocessing completes, create snapshot
gcloud compute disks snapshot hrdt-data-cache \
    --snapshot-names hrdt-preprocessed-$(date +%Y%m%d) \
    --zone us-central1-a \
    --description="Preprocessed EgoDex dataset for H-RDT scaling law experiments"

# Verify snapshot created
gcloud compute snapshots list --filter="name:hrdt-preprocessed"
```

### Step 2: Training Instances

For each training instance (running different `data_percentage`):

**2.1 Create and Attach Disk (per instance)**

```bash
# Create new disk from snapshot (for each instance)
for i in 1 2 3 4; do
    gcloud compute disks create hrdt-data-instance${i} \
        --source-snapshot hrdt-preprocessed-YYYYMMDD \
        --size=2000 --type=pd-ssd --zone us-central1-a
done

# Attach to instance
gcloud compute instances attach-disk YOUR_TRAINING_INSTANCE1 \
    --disk hrdt-data-instance1 \
    --device-name hrdt-data \
    --zone us-central1-a
```

**2.2 Mount Disk (on training instance)**

```bash
# SSH into training instance
gcloud compute ssh YOUR_TRAINING_INSTANCE1 --zone us-central1-a

# Create mount point
sudo mkdir -p /mnt/disks/hrdt-data

# Mount (NOT formatting - using existing data!)
sudo mount /dev/disk/by-id/google-hrdt-data-instance1 /mnt/disks/hrdt-data

# verify the disk exists
sudo blkid -s UUID -o value /dev/disk/by-id/google-hrdt-data-instance1

# Make permanent
echo UUID=$(sudo blkid -s UUID -o value /dev/disk/by-id/google-hrdt-data-instance1) \
    /mnt/disks/hrdt-data ext4 discard,defaults,nofail 0 2 | \
    sudo tee -a /etc/fstab

# Verify data is there
ls -lh /mnt/disks/hrdt-data/
```

**2.3 Setup Instance**

```bash
# On training instance
cd ~
git clone <YOUR_REPO> H_RDT
cd H_RDT
bash setup_instance.sh

source ~/.config/hrdt/activate.sh

# Set paths
export EGODEX_DATA_ROOT="/mnt/disks/hrdt-data/egodex/organized"  # Raw data location
export HRDT_OUTPUT_DIR="/mnt/disks/hrdt-data/processed"  # Use cached data
```

**2.4 Start Training**

```bash
# Train with 10% of data
accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed configs/zero1.json \
    --config_path configs/hrdt_pretrain.yaml \
    --output_dir ./checkpoints/scaling_p{10} \
    --train_batch_size 48 \
    --sample_batch_size 32 \
    --max_train_steps 100000 \
    --learning_rate 1e-4 \
    --data_percentage {0.1} \
    --seed 42 \
    --checkpointing_period 10000 \
    --checkpoints_total_limit 40 \
    --sample_period 1000 \
    --lr_scheduler=constant_with_warmup \
    --mixed_precision bf16 \
    --dataloader_num_workers 32 \
    --dataset_type pretrain \
    --report_to wandb \
    --upsample_rate 3 \
    --image_aug \
    --gradient_checkpointing \
    --precomp_lang_embed \
    --training_mode lang \
    --mode pretrain

# Change --data_percentage for other instances:
# Instance 2: --data_percentage 0.25
# Instance 3: --data_percentage 0.5
# Instance 4: --data_percentage 1.0
```

## Cost Estimates

**Storage (Monthly):**
```
2TB SSD: ~$320/month
Snapshots: Incremental, very cheap (~$5/month)
```

**Alternative: Use 1TB Standard Disk (~$80/month) for preprocessed data only**

**Compute (per experiment):**
```
A100 1x: $3.67/hour
100K steps: ~100 hours = ~$370 per experiment
4 experiments: ~$1,480
```

**Total: ~$1,500-2,000 for complete scaling law study**

## Alternative: If Instances Are in Different Zones

If you need to share across zones, you have 2 options:

### Option A: Copy Disk to Other Zone

```bash
# Create disk in new zone from snapshot
gcloud compute disks create hrdt-data-zone2 \
    --source-snapshot hrdt-preprocessed-YYYYMMDD \
    --size=500 --type=pd-ssd \
    --zone us-west1-a  # Different zone
```

### Option B: Use GCS as Intermediary (One-Time)

```bash
# Upload from preprocessing instance
gsutil -m cp -r /mnt/disks/hrdt-data/processed/* \
    gs://your-bucket-name/hrdt-processed/

# Download to training instance (takes ~10 hours)
gsutil -m rsync -r gs://your-bucket-name/hrdt-processed/ \
    /mnt/disks/hrdt-data/

# Then detach from GCS, use local disk
```

## Common Issues & Solutions

### "Disk already attached"
```bash
gcloud compute instances detach-disk INSTANCE_NAME \
    --disk hrdt-data-cache --zone us-central1-a
```

### "Cannot mount: already mounted"
```bash
# Unmount first
sudo umount /mnt/disks/hrdt-data
# Then mount again
sudo mount /dev/disk/by-id/google-hrdt-data /mnt/disks/hrdt-data
```

### "Permission denied"
```bash
# Fix ownership
sudo chown -R $USER:$USER /mnt/disks/hrdt-data
```

### "Out of disk space"
```bash
# Check usage
df -h /mnt/disks/hrdt-data

# Clean old checkpoints
find ~/H_RDT/checkpoints -name "checkpoint-*" -mtime +7 -delete

# Or increase disk size
gcloud compute disks resize hrdt-data-cache --size=1000 --zone us-central1-a
```

## Quick Reference

### Check Disk Status
```bash
# List disks
gcloud compute disks list | grep hrdt

# Check snapshots
gcloud compute snapshots list | grep hrdt

# Check instance disks
gcloud compute instances describe INSTANCE_NAME | grep disk
```

### Monitor Training
```bash
# Check instance
gcloud compute instances list | grep scaling

# SSH into instance
gcloud compute ssh INSTANCE_NAME --zone us-central1-a

# Watch logs
tail -f ~/H_RDT/checkpoints/scaling_p10/train.log

# Check WandB
wandb dashboard
```

## Next Steps

1. ✅ Follow Step 1 to preprocess data once
2. ✅ Create snapshot
3. ✅ For each data percentage, create instance with disk from snapshot
4. ✅ Run training with different `--data_percentage`
5. ✅ Monitor with WandB

For detailed information about scaling law experiments, see [SCALING_LAW_GUIDE.md](SCALING_LAW_GUIDE.md).

