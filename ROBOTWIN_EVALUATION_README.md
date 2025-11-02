# H-RDT RobotWin Evaluation Guide

Complete guide for evaluating H-RDT fine-tuned checkpoints in RobotWin simulation.

## Table of Contents

1. [One-Time Setup](#one-time-setup) ⚙️ **Do this once**
2. [Per-Checkpoint Evaluation](#per-checkpoint-evaluation) 🔄 **Do this for each checkpoint**
3. [Evaluation](#evaluation)
4. [Troubleshooting](#troubleshooting)

## One-Time Setup ⚙️

**These steps only need to be done once per machine/environment.**

## Prerequisites

1. **RobotWin Repository**
   - Location: `~/RoboTwin`
   - Installation: See [RobotWin Installation Guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

2. **H-RDT Environment**
   - Python virtual environment should be set up at `~/H_RDT/hrdt_env`
   - If you don't have `~/.config/hrdt/activate.sh`, you can create it by running `bash setup_instance.sh` from `~/H_RDT`, or manually activate the environment

3. **System Requirements**
   - Python 3.10+
   - CUDA-capable GPU
   - ~30GB free disk space (for assets)

### Step 1: Install RobotWin Dependencies

**Note:** If you don't have `~/.config/hrdt/activate.sh`, you can either:
1. Run `bash setup_instance.sh` from the H_RDT directory to create it, OR
2. Manually activate the environment (see alternative below)

```bash
cd ~/RoboTwin

# Option 1: Use activate.sh (if you have it)
source ~/.config/hrdt/activate.sh

# Option 2: Manual activation (if activate.sh doesn't exist)
cd ~/H_RDT
source hrdt_env/bin/activate
export HRDT_PROJECT_ROOT="$HOME/H_RDT"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"
cd ~/RoboTwin

# Verify virtual environment is active
which python
# Should show: /home/username/H_RDT/hrdt_env/bin/python

# Verify torch is installed (install if missing)
python -c "import torch; print('✅ torch version:', torch.__version__)" || pip install torch==2.4.1 torchvision

# Install RobotWin requirements
pip install transforms3d==0.4.2 sapien==3.0.0b1 scipy==1.10.1 mplib==0.2.1 \
    gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic \
    zarr openai huggingface_hub==0.25.0 h5py pyglet wandb moviepy termcolor av matplotlib

# Note: scipy version conflict warnings are safe to ignore

# Fix SAPIEN code
echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER

# Fix mplib code
echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

# Install Curobo (optional, but recommended)
echo "Installing Curobo ..."
cd envs
git clone https://github.com/NVlabs/curobo.git || echo "Curobo already cloned"
cd curobo
pip install -e . --no-build-isolation
cd ../..

# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg
```

**Note:** If you see permission errors, ensure you're using the virtual environment (not system Python). The `which python` command should show a path containing `hrdt_env`.

### Step 2: Download RobotWin Assets

```bash
cd ~/RoboTwin/assets

# Option 1: Use activate.sh (if you have it)
source ~/.config/hrdt/activate.sh

# Option 2: Manual activation (if activate.sh doesn't exist)
cd ~/H_RDT
source hrdt_env/bin/activate
export HRDT_PROJECT_ROOT="$HOME/H_RDT"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"
cd ~/RoboTwin/assets

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
# Create directory structure first (if it doesn't exist)
mkdir -p ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

# Copy inference code structure
cp -r ~/H_RDT/inference/robotwin2_example/H-RDT/* \
      ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/

# Copy language embeddings (one-time, shared across checkpoints)
mkdir -p ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings
cp -r ~/H_RDT/datasets/robotwin2/lang_embeddings/*.pt \
      ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/

# Copy configs and stats (one-time)
# Note: deploy_policy.py expects utils/ in the same directory as deploy_policy.py
mkdir -p ~/RoboTwin/policy/H-RDT/utils
cp ~/H_RDT/configs/hrdt_finetune.yaml \
   ~/RoboTwin/policy/H-RDT/utils/hrdt.yaml

cp ~/H_RDT/datasets/robotwin2/stats.json \
   ~/RoboTwin/policy/H-RDT/utils/stats.json

# Also copy language embeddings to utils (needed by deploy_policy.py)
mkdir -p ~/RoboTwin/policy/H-RDT/utils/lang_embeddings
cp -r ~/H_RDT/datasets/robotwin2/lang_embeddings/*.pt \
      ~/RoboTwin/policy/H-RDT/utils/lang_embeddings/

# Copy vision encoder models (one-time)
# Note: deploy_policy.py expects bak/ in the same directory as deploy_policy.py
mkdir -p ~/RoboTwin/policy/H-RDT/bak
cp -r ~/H_RDT/bak/* \
      ~/RoboTwin/policy/H-RDT/bak/

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

#### Check Checkpoint Format

```bash
CHECKPOINT_NAME="table8_handover_mic/checkpoint-10000"  # Change this for each checkpoint

# Check if checkpoint is in DeepSpeed format
ls -lh ~/H_RDT/checkpoints/$CHECKPOINT_NAME/

# If you see a `pytorch_model/` directory with multiple shard files, it's DeepSpeed format
# The consolidated `pytorch_model.bin` file will be small (~3MB) if it's just metadata
```

#### Convert Checkpoint

```bash
cd ~/H_RDT

# Using the helper script (recommended)
bash convert_deepspeed_checkpoint.sh \
    ~/H_RDT/checkpoints/$CHECKPOINT_NAME \
    ~/H_RDT/checkpoints/$CHECKPOINT_NAME/pytorch_model_consolidated.bin

# Or manually:
cd ~/H_RDT/checkpoints/$CHECKPOINT_NAME

# Option 1: Use activate.sh (if you have it)
source ~/.config/hrdt/activate.sh

# Option 2: Manual activation (if activate.sh doesn't exist)
cd ~/H_RDT
source hrdt_env/bin/activate
export HRDT_PROJECT_ROOT="$HOME/H_RDT"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

cd ~/H_RDT/checkpoints/$CHECKPOINT_NAME

python zero_to_fp32.py $(pwd) $(pwd)/pytorch_model_consolidated.bin
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

#### Verify Conversion

```bash
ls -lh ~/H_RDT/checkpoints/$CHECKPOINT_NAME/pytorch_model_consolidated.bin
# Should be ~7.7GB
```

**Note:** If checkpoint is already consolidated (single `.bin` file), skip this step.

### Step 2: Copy Checkpoint to RobotWin

```bash
# Create checkpoint directory (use descriptive name)

mkdir -p ~/RoboTwin/policy/H-RDT/checkpoints/$CHECKPOINT_NAME

# Copy consolidated checkpoint
cp ~/H_RDT/checkpoints/$CHECKPOINT_NAME/pytorch_model_consolidated.bin \
   ~/RoboTwin/policy/H-RDT/checkpoints/$CHECKPOINT_NAME/pytorch_model.bin

# Copy config
cp ~/H_RDT/checkpoints/$CHECKPOINT_NAME/config.json \
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

**Important:** Ensure `eval.sh` contains the correct path calculation code. The script should look like this:

```bash
#!/bin/bash

policy_name="H-RDT"
task_name="grab_roller"         # Change this for each task
task_config="demo_randomized"   # "demo_randomized" (Hard) or "demo_clean" (Easy)
ckpt_setting="checkpoints/table8_handover_mic/checkpoint-10000"  # Change this for each checkpoint
seed="42"
gpu_id="0"

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Calculate RoboTwin root directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$ROBOTWIN_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
```

**Set these variables:**
- `task_name`: Task to evaluate (e.g., `"grab_roller"`, `"handover_mic"`, etc.)
- `task_config`: `"demo_randomized"` (Hard mode) or `"demo_clean"` (Easy mode)
- `ckpt_setting`: Checkpoint path relative to policy directory (e.g., `"checkpoints/table8_checkpoint30"`)
- `gpu_id`: GPU to use (e.g., `"0"`)

**Note:** The path calculation code (lines 14-17) is essential - it ensures the script finds `script/eval_policy.py` regardless of where you run the script from.

### Run Evaluation

```bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

# Option 1: Use activate.sh (if you have it)
source ~/.config/hrdt/activate.sh
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

# Option 2: Manual activation (if activate.sh doesn't exist)
cd ~/H_RDT
source hrdt_env/bin/activate
export HRDT_PROJECT_ROOT="$HOME/H_RDT"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

bash eval.sh
```

**Note:** Change `ckpt_setting` in `eval.sh` to evaluate different checkpoints.

### Finding Evaluation Results

After running evaluation, the output videos and results can be found in:

```
~/RoboTwin/eval_result/{task_name}/{policy_name}/{task_config}/checkpoints/{checkpoint_path}/{timestamp}/
```

For example, for `handover_mic` task with `demo_clean` config and checkpoint `table8_handover_mic/checkpoint-10000`:

```bash
~/RoboTwin/eval_result/handover_mic/H-RDT/demo_clean/checkpoints/table8_handover_mic/checkpoint-10000/2025-11-01 22:35:05/
```

This directory will contain:
- `episode0.mp4`, `episode1.mp4`, `episode2.mp4`, ... - Video recordings of each evaluation trial
- Other evaluation artifacts (logs, metrics, etc.)

The timestamp directory is created automatically for each evaluation run, allowing you to track multiple evaluation runs over time.

### Copying Videos to Local Machine

To copy evaluation videos from your GCP instance to your local machine:

**Using RSYNC (Recommended):**

```bash
# Set your instance IP and SSH key (if needed)
INSTANCE_IP="34.56.106.17"
SSH_KEY="~/.ssh/gcp_key"  # Omit -e flag if using default SSH key
CHECKPOINT="table8_handover_mic/checkpoint-10000"
EP_TIMESTAMP='2025-11-01 22:35:05'

# Option 1: Copy entire checkpoint directory (RECOMMENDED - avoids escaping issues)
rsync -avhP -e "ssh -i $SSH_KEY" \
  jose-barreiros@$INSTANCE_IP:~/RoboTwin/eval_result/handover_mic/H-RDT/demo_clean/checkpoints/$CHECKPOINT/ \
  ./checkpoint_videos/$CHECKPOINT/
# This copies all timestamp subdirectories to ./checkpoint_videos/$CHECKPOINT/
# Then navigate locally: cd "./checkpoint_videos/$CHECKPOINT/$EP_TIMESTAMP"

# Option 2: Copy specific timestamp directory using absolute path
# Note: Create parent directories first (since $CHECKPOINT contains a slash)
# Or use --mkpath flag if rsync version >= 3.2.4
mkdir -p "./videos/$CHECKPOINT/$EP_TIMESTAMP"

rsync -avhP -e "ssh -i $SSH_KEY" \
  "jose-barreiros@$INSTANCE_IP:/home/jose-barreiros/RoboTwin/eval_result/handover_mic/H-RDT/demo_clean/checkpoints/$CHECKPOINT/$EP_TIMESTAMP/" \
  "./videos/$CHECKPOINT/$EP_TIMESTAMP/"

# Alternative: Use --mkpath to auto-create parent directories (rsync 3.2.4+)
# rsync -avhP --mkpath -e "ssh -i $SSH_KEY" \
#   "jose-barreiros@$INSTANCE_IP:/home/jose-barreiros/RoboTwin/eval_result/handover_mic/H-RDT/demo_clean/checkpoints/$CHECKPOINT/$EP_TIMESTAMP/" \
#   "./videos/$CHECKPOINT/$EP_TIMESTAMP/"

# Copy only MP4 files (faster)
mkdir -p "./videos/$CHECKPOINT/$EP_TIMESTAMP"
rsync -avhP -e "ssh -i $SSH_KEY" --include="*.mp4" --exclude="*" \
  "jose-barreiros@$INSTANCE_IP:/home/jose-barreiros/RoboTwin/eval_result/handover_mic/H-RDT/demo_clean/checkpoints/$CHECKPOINT/$EP_TIMESTAMP/" \
  "./videos/$CHECKPOINT/$EP_TIMESTAMP/"

# Copy all evaluation results for a task
rsync -avhP -e "ssh -i $SSH_KEY" \
  jose-barreiros@$INSTANCE_IP:~/RoboTwin/eval_result/handover_mic/ ./eval_results/
```

**Note:** If you don't need to specify an SSH key (using default `~/.ssh/id_rsa`), omit the `-e "ssh -i $SSH_KEY"` part.

**RSYNC flags:**
- `-a`: Archive mode (preserves permissions, timestamps)
- `-v`: Verbose output
- `-h`: Human-readable file sizes
- `-P`: Progress indicator + partial (resumes interrupted transfers)

**Finding your instance IP:**
- GCP Console: Compute Engine > VM instances > External IP
- From instance: `curl -s ifconfig.me`

**Note:** Use quotes or escape spaces in directory names (like timestamps). RSYNC is recommended for large files as it can resume interrupted transfers.

**Troubleshooting rsync/scp errors:**

**Problem:** `rsync: change_dir failed: No such file or directory`
- **Solution 1:** First verify the path exists on remote:
  ```bash
  ssh -i $SSH_KEY jose-barreiros@$INSTANCE_IP \
    'ls -d ~/RoboTwin/eval_result/handover_mic/H-RDT/demo_clean/checkpoints/table8_handover_mic/checkpoint-10000/*'
  ```
- **Solution 2:** Use absolute path instead of `~` expansion:
  ```bash
  # Use /home/jose-barreiros instead of ~
  rsync -avhP -e "ssh -i $SSH_KEY" \
    jose-barreiros@$INSTANCE_IP:/home/jose-barreiros/RoboTwin/eval_result/.../checkpoint-10000/'2025-11-01 22:35:05'/ \
    ./videos/
  ```
- **Solution 3:** Copy the parent directory (no timestamp escaping needed):
  ```bash
  rsync -avhP -e "ssh -i $SSH_KEY" \
    jose-barreiros@$INSTANCE_IP:~/RoboTwin/eval_result/handover_mic/H-RDT/demo_clean/checkpoints/table8_handover_mic/checkpoint-10000/ \
    ./checkpoint_videos/
  ```

**Problem:** `rsync: mkdir failed: No such file or directory`
- **Solution:** Create parent directories first (since `$CHECKPOINT` contains a slash):
  ```bash
  mkdir -p "./videos/$CHECKPOINT/$EP_TIMESTAMP"
  ```
  Or use `--mkpath` flag if rsync version >= 3.2.4:
  ```bash
  rsync -avhP --mkpath -e "ssh -i $SSH_KEY" ...
  ```
  Check rsync version: `rsync --version`

**Problem:** Authentication fails or "Permission denied"
- **Solution:** Specify SSH key explicitly:
  ```bash
  rsync -avhP -e "ssh -i ~/.ssh/your_gcp_key" ...
  scp -i ~/.ssh/your_gcp_key ...
  ```

### Evaluate All Table 8 Tasks

Create `evaluate_all_tasks.sh`:

```bash
#!/bin/bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

# Option 1: Use activate.sh (if you have it)
source ~/.config/hrdt/activate.sh

# Option 2: Manual activation (if activate.sh doesn't exist)
cd ~/H_RDT
source hrdt_env/bin/activate
export HRDT_PROJECT_ROOT="$HOME/H_RDT"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

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

**Note:** 
- If you trained in **multi-task mode**: Evaluate on all tasks with the same checkpoint
- If you trained in **single-task mode**: Each task has its own checkpoint, evaluate each task with its corresponding checkpoint:
  ```bash
  # For single-task checkpoint:
  ckpt_setting="checkpoints/table8_grab_roller_checkpoint30"
  task_name="grab_roller"
  ```

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
- **Solution:** The script uses absolute paths calculated from script location. If you see path errors, ensure `eval.sh` uses the path calculation shown in the guide (lines 14-17).

### Asset Issues

**Problem:** `FileNotFoundError: assets/objects/objaverse/list.json`
- **Solution:** Download and extract RobotWin assets (Step 3)

### Evaluation Issues

**Problem:** Low success rate
- **Normal:** Hard mode has much lower success rate than Easy mode
- **Check:** Ensure language embeddings are loaded correctly
- **Check:** Verify checkpoint is from correct training step

## Quick Reference

### One-Time Setup Checklist

```bash
# Check dependencies installed
python -c "import sapien; print('✅ SAPIEN')"

# Check assets downloaded
test -f ~/RoboTwin/assets/objects/objaverse/list.json && echo "✅ Assets" || echo "❌ Missing assets"

# Check code structure copied
test -f ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.py && echo "✅ Code" || echo "❌ Missing code"

# Check language embeddings
ls ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/*.pt | wc -l
# Should be 49
```

### Per-Checkpoint Checklist

```bash
# Check consolidated checkpoint
ls -lh ~/H_RDT/checkpoints/YOUR_CHECKPOINT/pytorch_model_consolidated.bin
# Should be ~7.7GB

# Check RobotWin checkpoint
ls -lh ~/RoboTwin/policy/H-RDT/checkpoints/YOUR_CHECKPOINT_NAME/pytorch_model.bin
# Should be ~7.7GB

# Check eval.sh configured
grep "ckpt_setting=" ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh
# Should point to your checkpoint
```

## Files Structure

```
~/H_RDT/
├── checkpoints/
│   └── table8_finetune_pretrain0618/
│       └── checkpoint-30/
│           ├── pytorch_model/              # DeepSpeed shards
│           ├── pytorch_model_consolidated.bin  # Converted checkpoint (7.7GB)
│           └── config.json
├── convert_deepspeed_checkpoint.sh
├── prepare_robotwin_evaluation.sh
└── ...

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

## Support

For issues:
1. Check troubleshooting section above
2. Verify all prerequisites are met
3. Check RobotWin documentation
4. Review error messages carefully

