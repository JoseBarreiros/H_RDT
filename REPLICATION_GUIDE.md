# Replication Guide

This guide ensures you can replicate the RobotWin evaluation setup.

## 🔄 Setup Types

### ⚙️ One-Time Setup (Do Once Per Machine)
- Install RobotWin dependencies
- Download RobotWin assets (~15GB)
- Copy code structure to RobotWin

### 🔄 Per-Checkpoint Steps (Repeat for Each Checkpoint)
- Convert DeepSpeed checkpoint (if needed)
- Copy checkpoint to RobotWin
- Update `ckpt_setting` in `eval.sh`
- Run evaluation

## Prerequisites Checklist

- [ ] H-RDT fine-tuned checkpoint exists
- [ ] RobotWin repository cloned at `~/RoboTwin`
- [ ] Python virtual environment with H-RDT dependencies
- [ ] ~30GB free disk space
- [ ] CUDA-capable GPU

## Step-by-Step Replication

### ⚙️ One-Time Setup Steps

#### Step 1: Install RobotWin Dependencies

```bash
cd ~/RoboTwin
source ~/.config/hrdt/activate.sh

# Install RobotWin requirements
pip install -r script/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
bash script/_install.sh
sudo apt-get install -y ffmpeg
```

#### Step 2: Download RobotWin Assets

```bash
cd ~/RoboTwin/assets
source ~/.config/hrdt/activate.sh
python _download.py

# Extract (may take time)
unzip background_texture.zip
unzip embodiments.zip
unzip objects.zip

# Clean up
rm -f *.zip

# Configure paths
cd ~/RoboTwin
python script/update_embodiment_config_path.py
```

#### Step 3: Copy Code Structure

```bash
# Copy inference code (one-time)
cp -r ~/H_RDT/inference/robotwin2_example/H-RDT/* \
      ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/

# Copy language embeddings (one-time, shared)
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

# Setup policy module (one-time)
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

**✅ One-time setup complete!**

---

### 🔄 Per-Checkpoint Steps

#### Step 1: Verify Checkpoint Format

```bash
# Check if checkpoint is DeepSpeed format
ls -lh ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/

# Should see:
# - pytorch_model/ directory (with shard files) - needs conversion
# - OR single pytorch_model.bin file - already consolidated
```

#### Step 2: Convert Checkpoint (If Needed)

```bash
cd ~/H_RDT
bash convert_deepspeed_checkpoint.sh \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin

# Verify output
ls -lh ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin
# Should be ~7.7GB
```

#### Step 3: Copy Checkpoint to RobotWin

```bash
cd ~/H_RDT
bash prepare_robotwin_evaluation.sh \
    table8_checkpoint30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30
```

#### Step 4: Update Evaluation Script

Edit `~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh`:

```bash
ckpt_setting="checkpoints/table8_checkpoint30"  # Change for each checkpoint
task_name="grab_roller"                         # Change task as needed
task_config="demo_randomized"                   # "demo_randomized" (Hard) or "demo_clean" (Easy)
```

#### Step 5: Run Evaluation

```bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh
bash eval.sh
```

## Verification Commands

### Check One-Time Setup

```bash
# Check dependencies installed
python -c "import sapien; print('✅ SAPIEN')"

# Check assets downloaded
test -f ~/RoboTwin/assets/objects/objaverse/list.json && echo "✅ Assets" || echo "❌ Missing assets"

# Check code structure
test -f ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.py && echo "✅ Code" || echo "❌ Missing code"

# Check language embeddings
ls ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/*.pt | wc -l
# Should be 49
```

### Check Per-Checkpoint Setup

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

## Expected Output

Evaluation should show:
- Model loading messages
- Language embedding loading
- Step-by-step evaluation progress
- Success/failure for each trial
- Final success rate

Example:
```
grab_roller | H-RDT | demo_randomized | checkpoints/table8_checkpoint30
Success rate: 2/16 => 12.5%
```

## Troubleshooting

### Checkpoint Issues

**Problem:** Conversion fails
- Check `zero_to_fp32.py` exists in checkpoint directory
- Check `latest` file exists and points to `pytorch_model`
- Ensure sufficient disk space (~10GB free)

**Problem:** Checkpoint loading fails
- Verify using consolidated checkpoint, not sharded format
- Check config.json doesn't have `pretrained_backbone_path`

### Environment Issues

**Problem:** Module import errors
- Reinstall RobotWin dependencies (one-time setup)
- Check virtual environment is activated
- Verify `__init__.py` exists in policy directory (one-time setup)

**Problem:** Asset errors
- Re-download assets if corrupted (one-time setup)
- Check paths are configured correctly

## Files to Keep

Keep these files for replication:
- `convert_deepspeed_checkpoint.sh`
- `prepare_robotwin_evaluation.sh`
- `ROBOTWIN_EVALUATION_SETUP.md`
- `ROBOTWIN_EVALUATION_README.md`
- `ROBOTWIN_SETUP_TYPES.md` - NEW: Explains one-time vs per-checkpoint
- Consolidated checkpoint file

## Next Steps

After successful evaluation:
1. Evaluate other tasks by changing `task_name` in `eval.sh`
2. Try Easy mode by setting `task_config="demo_clean"`
3. Evaluate all 13 tasks using the loop script
4. Compare results with Table 8 in the paper

## Documentation

- **Setup Types:** `ROBOTWIN_SETUP_TYPES.md` - One-time vs per-checkpoint breakdown
- **Main Guide:** `ROBOTWIN_EVALUATION_README.md` - Complete guide
- **Quick Reference:** `EVALUATION_QUICK_REFERENCE.md` - Quick commands
