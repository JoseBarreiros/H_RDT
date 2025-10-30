# H-RDT RobotWin Evaluation Guide

Complete guide for evaluating H-RDT fine-tuned checkpoints in RobotWin simulation.

## Table of Contents

1. [One-Time Setup](#one-time-setup) ⚙️ **Do this once**
2. [Per-Checkpoint Evaluation](#per-checkpoint-evaluation) 🔄 **Do this for each checkpoint**
3. [Evaluation](#evaluation)
4. [Troubleshooting](#troubleshooting)

## One-Time Setup ⚙️

**These steps only need to be done once per machine/environment.**

### Prerequisites

1. **RobotWin Repository**
   - Location: `~/RoboTwin`
   - Installation: See [RobotWin Installation Guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

2. **System Requirements**
   - Python 3.10+
   - CUDA-capable GPU
   - ~30GB free disk space (for assets)

### Step 1: Install RobotWin Dependencies

```bash
cd ~/RoboTwin
source ~/.config/hrdt/activate.sh

# Install RobotWin requirements
pip install -r script/requirements.txt

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Run installation script (fixes SAPIEN and mplib code)
bash script/_install.sh

# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### Step 2: Download RobotWin Assets

```bash
cd ~/RoboTwin/assets
source ~/.config/hrdt/activate.sh
python _download.py

# Extract assets (may take time)
unzip background_texture.zip
unzip embodiments.zip
unzip objects.zip

# Clean up
rm -f *.zip

# Configure paths
cd ~/RoboTwin
python script/update_embodiment_config_path.py
```

**Note:** Asset download is ~15GB and takes 20-30 minutes. **Only needed once.**

### Step 3: Copy Code Structure to RobotWin

**This is a one-time setup** - copy the inference code and language embeddings:

```bash
# Copy inference code structure
cp -r ~/H_RDT/inference/robotwin2_example/H-RDT/* \
      ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/

# Copy language embeddings (one-time, shared across checkpoints)
cp -r ~/H_RDT/datasets/robotwin2/lang_embeddings/*.pt \
      ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/

# Copy configs and stats (one-time)
cp ~/H_RDT/configs/hrdt_finetune.yaml \
   ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/hrdt.yaml

cp ~/H_RDT/datasets/robotwin2/stats.json \
   ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/stats.json

# Copy vision encoder models (one-time)
cp -r ~/H_RDT/bak \
      ~/RoboTwin/policy/H-RDT/

# Setup policy module structure (one-time)
mkdir -p ~/RoboTwin/policy/H-RDT
cp ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.py \
   ~/RoboTwin/policy/H-RDT/deploy_policy.py

cat > ~/RoboTwin/policy/H-RDT/__init__.py << 'EOF'
# H-RDT Policy Module
from .deploy_policy import get_model, eval, reset_model, encode_obs

__all__ = ['get_model', 'eval', 'reset_model', 'encode_obs']
EOF

cp ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.yml \
   ~/RoboTwin/policy/H-RDT/deploy_policy.yml
```

**✅ After this, you're ready to evaluate any checkpoint!**

## Per-Checkpoint Evaluation 🔄

**These steps are repeated for each checkpoint you want to evaluate.**

### Step 1: Convert DeepSpeed Checkpoint

If your checkpoint is in DeepSpeed ZeRO-3 format (has `pytorch_model/` directory), convert it:

```bash
cd ~/H_RDT

# Using the helper script (recommended)
bash convert_deepspeed_checkpoint.sh \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin

# Or manually:
cd ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30
source ~/.config/hrdt/activate.sh
python zero_to_fp32.py ~(pwd) ~(pwd)/pytorch_model_consolidated.bin
```

**Expected output:** ~7.7GB consolidated checkpoint file

**Note:** If checkpoint is already consolidated (single `.bin` file), skip this step.

### Step 2: Copy Checkpoint to RobotWin

```bash
# Create checkpoint directory (use descriptive name)
CHECKPOINT_NAME="table8_checkpoint30"  # Change this for each checkpoint
mkdir -p ~/RoboTwin/policy/H-RDT/checkpoints/$CHECKPOINT_NAME

# Copy consolidated checkpoint
cp ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin \
   ~/RoboTwin/policy/H-RDT/checkpoints/$CHECKPOINT_NAME/pytorch_model.bin

# Copy config
cp ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/config.json \
   ~/RoboTwin/policy/H-RDT/checkpoints/$CHECKPOINT_NAME/config.json

# Remove pretrained_backbone_path from config (not needed for fine-tuned checkpoint)
cd ~/RoboTwin/policy/H-RDT/checkpoints/$CHECKPOINT_NAME
python3 << EOF
import json
with open('config.json', 'r') as f:
    config = json.load(f)
config.pop('pretrained_backbone_path', None)
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
EOF
```

**Or use the helper script:**

```bash
bash prepare_robotwin_evaluation.sh \
    table8_checkpoint30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30
```

**Note:** The helper script also copies code/files, but you can skip those if already done in one-time setup.

## Evaluation

### Configure Evaluation Script

Edit `~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh`:

```bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
nano eval.sh  # or your preferred editor
```

Set these variables:
```bash
task_name="grab_roller"         # Task to evaluate
task_config="demo_randomized"   # "demo_randomized" (Hard) or "demo_clean" (Easy)
ckpt_setting="checkpoints/table8_checkpoint30"  # Change this for each checkpoint
gpu_id="0"                      # GPU to use
```

### Run Evaluation

```bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh
bash eval.sh
```

**Note:** Change `ckpt_setting` in `eval.sh` to evaluate different checkpoints.

### Evaluate All Table 8 Tasks

Create `evaluate_all_tasks.sh`:

```bash
#!/bin/bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh

tasks=(
    "grab_roller" "handover_mic" "lift_pot" "move_can_pot"
    "open_laptop" "pick_dual_bottles" "place_dual_shoes"
    "place_object_basket" "place_phone_stand" "put_bottles_dustbin"
    "put_object_cabinet" "stack_blocks_two" "stack_bowls_two"
)

for task in "${tasks[@]}"; do
    echo "Evaluating: $task"
    sed -i "s/task_name=\".*\"/task_name=\"$task\"/" eval.sh
    bash eval.sh
done
```

### Expected Results

Based on H-RDT paper Table 8:

| Mode | Expected Success Rate |
|------|----------------------|
| Easy (`demo_clean`) | ~68.7% |
| Hard (`demo_randomized`) | ~25.6% |

**Current Results (checkpoint-30, Hard mode):**
- `grab_roller`: ~12.5% (2/16 trials)

## Troubleshooting

### Checkpoint Issues

**Problem:** `RuntimeError: size mismatch` when loading checkpoint
- **Solution:** Ensure you're using the consolidated checkpoint (`pytorch_model_consolidated.bin`), not the small metadata file

**Problem:** `FileNotFoundError: pretrained_backbone_path`
- **Solution:** The preparation script removes this automatically. If manual, edit `config.json` and remove `pretrained_backbone_path`.

### Module Import Issues

**Problem:** `ModuleNotFoundError: No module named 'sapien'`
- **Solution:** Install RobotWin dependencies (Step 2)

**Problem:** `AttributeError: module 'H-RDT' has no attribute 'get_model'`
- **Solution:** Ensure `~/RoboTwin/policy/H-RDT/__init__.py` exists and exports `get_model`

### Path Issues

**Problem:** Script can't find checkpoint or files
- **Solution:** Use absolute paths or ensure you're in the correct directory

**Problem:** `eval.sh` changes directory incorrectly
- **Solution:** The script uses absolute paths calculated from script location

### Asset Issues

**Problem:** `FileNotFoundError: assets/objects/objaverse/list.json`
- **Solution:** Download and extract RobotWin assets (Step 3)

### Evaluation Issues

**Problem:** Low success rate
- **Normal:** Hard mode has much lower success rate than Easy mode
- **Check:** Ensure language embeddings are loaded correctly
- **Check:** Verify checkpoint is from correct training step

## Files Structure

```
~/H_RDT/
├── checkpoints/
│   └── table8_finetune_pretrain0618/
│       └── checkpoint-30/
│           ├── pytorch_model/              # DeepSpeed shards
│           ├── pytorch_model_consolidated.bin  # Converted checkpoint (7.7GB)
│           └── config.json
├── scripts/
│   ├── convert_deepspeed_checkpoint.sh
│   └── prepare_robotwin_evaluation.sh
└── docs/
    ├── ROBOTWIN_EVALUATION_SETUP.md      # Detailed setup guide
    └── EVALUATION_QUICK_REFERENCE.md     # Quick commands

~/RoboTwin/
└── policy/
    └── H-RDT/
        ├── checkpoints/
        │   └── table8_checkpoint30/
        │       ├── pytorch_model.bin       # Consolidated checkpoint
        │       └── config.json
        ├── inference/
        │   └── robotwin2_example/
        │       └── H-RDT/
        │           ├── eval.sh            # Evaluation script
        │           ├── deploy_policy.py
        │           └── utils/
        │               ├── lang_embeddings/  # Task embeddings
        │               ├── hrdt.yaml
        │               └── stats.json
        ├── deploy_policy.py
        ├── deploy_policy.yml
        └── __init__.py                   # Module exports
```

## Additional Resources

- **RobotWin Documentation:** https://robotwin-platform.github.io/doc/
- **H-RDT Paper:** Check paper for Table 8 details
- **Detailed Setup:** See `ROBOTWIN_EVALUATION_SETUP.md`
- **Quick Reference:** See `EVALUATION_QUICK_REFERENCE.md`

## Support

For issues:
1. Check troubleshooting section above
2. Verify all prerequisites are met
3. Check RobotWin documentation
4. Review error messages carefully

