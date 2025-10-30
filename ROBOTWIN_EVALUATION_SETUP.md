# RobotWin Evaluation Setup Guide

This guide explains how to evaluate your H-RDT fine-tuned checkpoint (`table8_finetune_pretrain0618/checkpoint-30`) in the RobotWin simulation environment.

## Overview

The evaluation process involves:
1. Converting DeepSpeed ZeRO-3 checkpoint to consolidated format
2. Setting up RobotWin environment and dependencies
3. Copying checkpoint and assets to RobotWin
4. Running evaluation

## Prerequisites

- ✅ H-RDT fine-tuned checkpoint (`checkpoints/table8_finetune_pretrain0618/checkpoint-30`)
- ✅ RobotWin repository cloned at `~/RoboTwin`
- ✅ Python virtual environment with H-RDT dependencies installed

## Step 1: Convert DeepSpeed Checkpoint

DeepSpeed saves checkpoints in sharded format (ZeRO-3). We need to convert it to a single consolidated file for evaluation.

### Check Checkpoint Format

```bash
# Check if checkpoint is in DeepSpeed format
ls -lh ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/

# If you see a `pytorch_model/` directory with multiple shard files, it's DeepSpeed format
# The consolidated `pytorch_model.bin` file will be small (~3MB) if it's just metadata
```

### Convert Checkpoint

```bash
cd ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30

# Activate environment
source ~/.config/hrdt/activate.sh

# Run conversion script (uses absolute paths to avoid directory issues)
python ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/zero_to_fp32.py \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin
```

**Expected output:**
```
Processing zero checkpoint '.../pytorch_model'
Detected checkpoint of type zero stage 3, world_size: 4
Parsing checkpoint created by deepspeed==0.15.1
Reconstructed Trainable fp32 state dict with 547 params 2062417166 elements
Saving fp32 state dict to .../pytorch_model_consolidated.bin
```

**Note:** The consolidated file will be ~7.7GB. This may take several minutes.

### Verify Conversion

```bash
ls -lh ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin
# Should be ~7.7GB
```

## Step 2: Setup RobotWin Environment

### Install RobotWin Dependencies

```bash
cd ~/RoboTwin

# Activate H-RDT environment (contains required packages)
source ~/.config/hrdt/activate.sh

# Install RobotWin requirements
pip install -r ~/RoboTwin/script/requirements.txt

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Run installation script (fixes SAPIEN and mplib code)
bash ~/RoboTwin/script/_install.sh
```

### Download RobotWin Assets

```bash
cd ~/RoboTwin/assets

# Download assets from HuggingFace
python ~/RoboTwin/assets/_download.py

# Extract zip files
unzip background_texture.zip
unzip embodiments.zip
unzip objects.zip

# Clean up
rm -f *.zip

# Configure paths
cd ~/RoboTwin
python ./script/update_embodiment_config_path.py
```

**Note:** Asset download is ~15GB and may take 20-30 minutes.

### Install System Dependencies

```bash
# Install ffmpeg (required for video recording)
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Step 3: Prepare Checkpoint for RobotWin

### Copy Checkpoint Files

```bash
# Create checkpoint directory
mkdir -p ~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30

# Copy consolidated checkpoint
cp ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin \
   ~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/pytorch_model.bin

# Copy config (remove pretrained_backbone_path if present)
cp ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/config.json \
   ~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/config.json

# Remove pretrained_backbone_path from config (not needed for fine-tuned checkpoint)
cd ~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30
python3 << EOF
import json
with open('config.json', 'r') as f:
    config = json.load(f)
config.pop('pretrained_backbone_path', None)
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
EOF
```

### Copy Required Files

```bash
# Copy inference code
cp -r ~/H_RDT/inference/robotwin2_example/H-RDT/* \
      ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/

# Copy language embeddings
cp -r ~/H_RDT/datasets/robotwin2/lang_embeddings/*.pt \
      ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/

# Copy config and stats
cp ~/H_RDT/configs/hrdt_finetune.yaml \
   ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/hrdt.yaml

cp ~/H_RDT/datasets/robotwin2/stats.json \
   ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/stats.json

# Copy vision encoder models
cp -r ~/H_RDT/bak \
      ~/RoboTwin/policy/H-RDT/
```

### Setup Policy Module

```bash
# Copy deploy_policy.py to policy directory
cp ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.py \
   ~/RoboTwin/policy/H-RDT/deploy_policy.py

# Create __init__.py to expose get_model function
cat > ~/RoboTwin/policy/H-RDT/__init__.py << 'EOF'
# H-RDT Policy Module
from .deploy_policy import get_model, eval, reset_model, encode_obs

__all__ = ['get_model', 'eval', 'reset_model', 'encode_obs']
EOF

# Copy deploy_policy.yml
cp ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.yml \
   ~/RoboTwin/policy/H-RDT/deploy_policy.yml

# Link or copy required directories (models, utils, bak)
# These should already be in place from previous steps
```

## Step 4: Configure Evaluation Script

Edit `~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh`:

```bash
policy_name="H-RDT"
task_name="grab_roller"              # Change for each task
task_config="demo_randomized"        # "demo_randomized" (Hard) or "demo_clean" (Easy)
ckpt_setting="checkpoints/table8_checkpoint30"
seed="42"
gpu_id="0"
```

## Step 5: Run Evaluation

```bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh
bash eval.sh
```

## Evaluating All Table 8 Tasks

To evaluate all 13 tasks, create a script:

```bash
#!/bin/bash
# evaluate_all_table8_tasks.sh

cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh

tasks=(
    "grab_roller"
    "handover_mic"
    "lift_pot"
    "move_can_pot"
    "open_laptop"
    "pick_dual_bottles"
    "place_dual_shoes"
    "place_object_basket"
    "place_phone_stand"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "stack_blocks_two"
    "stack_bowls_two"
)

for task in "${tasks[@]}"; do
    echo "=========================================="
    echo "Evaluating task: $task"
    echo "=========================================="
    
    # Update eval.sh
    sed -i "s/task_name=\".*\"/task_name=\"$task\"/" eval.sh
    
    # Run evaluation
    bash eval.sh
    
    echo ""
done
```

## Expected Results

Based on Table 8 from the H-RDT paper:
- **Easy mode** (`demo_clean`): ~68.7% average success rate
- **Hard mode** (`demo_randomized`): ~25.6% average success rate

**Current Results:**
- Task: `grab_roller`
- Mode: Hard (`demo_randomized`)
- Success Rate: ~12.5% (2/16 trials)

## Troubleshooting

### Checkpoint Loading Issues

**Problem:** `RuntimeError: size mismatch` when loading checkpoint
- **Solution:** Ensure you're using the consolidated checkpoint (`pytorch_model_consolidated.bin`), not the small metadata file

**Problem:** `FileNotFoundError: pretrained_backbone_path`
- **Solution:** Remove `pretrained_backbone_path` from `config.json` (see Step 3)

### Module Import Issues

**Problem:** `ModuleNotFoundError: No module named 'sapien'`
- **Solution:** Install RobotWin dependencies (see Step 2)

**Problem:** `AttributeError: module 'H-RDT' has no attribute 'get_model'`
- **Solution:** Ensure `__init__.py` exists and exports `get_model` (see Step 3)

### Path Issues

**Problem:** Script can't find files
- **Solution:** Use absolute paths in scripts, or ensure you're in the correct directory

**Problem:** `eval.sh` changes directory incorrectly
- **Solution:** The script now uses absolute paths calculated from script location

### Asset Issues

**Problem:** `FileNotFoundError: assets/objects/objaverse/list.json`
- **Solution:** Download and extract RobotWin assets (see Step 2)

## Files Modified/Created

1. **Checkpoint Conversion:**
   - `~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin` (7.7GB)

2. **RobotWin Setup:**
   - `~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/pytorch_model.bin`
   - `~/RoboTwin/policy/H-RDT/deploy_policy.py`
   - `~/RoboTwin/policy/H-RDT/__init__.py`
   - `~/RoboTwin/policy/H-RDT/deploy_policy.yml`
   - `~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh` (updated)

3. **Documentation:**
   - `ROBOTWIN_EVALUATION_SETUP.md` (this file)

## Quick Reference

```bash
# Convert checkpoint
cd ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30
source ~/.config/hrdt/activate.sh
python zero_to_fp32.py ~(pwd) ~(pwd)/pytorch_model_consolidated.bin

# Copy to RobotWin
cp pytorch_model_consolidated.bin ~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/pytorch_model.bin

# Evaluate
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh
bash eval.sh
```

