# RobotWin Evaluation: One-Time vs Per-Checkpoint Setup

## 🔄 Setup Types

### ⚙️ One-Time Setup (Do Once Per Machine)

These steps only need to be done once when setting up RobotWin evaluation:

1. **Install RobotWin Dependencies**
   - SAPIEN, mplib, pytorch3d, curobo, etc.
   - System dependencies (ffmpeg)

2. **Download RobotWin Assets**
   - ~15GB of assets (background textures, embodiments, objects)
   - Takes 20-30 minutes to download

3. **Copy Code Structure**
   - Inference code (`deploy_policy.py`, etc.)
   - Language embeddings (49 task embeddings)
   - Config files (`hrdt.yaml`, `stats.json`)
   - Vision encoder models (`bak/` folder)
   - Policy module structure (`__init__.py`, etc.)

**After completing one-time setup, you can evaluate any checkpoint!**

### 🔄 Per-Checkpoint Steps (Repeat for Each Checkpoint)

These steps are repeated for each checkpoint you want to evaluate:

1. **Convert Checkpoint** (if DeepSpeed format)
   - Convert ZeRO-3 sharded checkpoint to consolidated format
   - Only needed if checkpoint has `pytorch_model/` directory

2. **Copy Checkpoint**
   - Copy consolidated `.bin` file to RobotWin
   - Copy and clean `config.json`
   - Create checkpoint directory with descriptive name

3. **Update Evaluation Script**
   - Change `ckpt_setting` in `eval.sh` to point to new checkpoint
   - Optionally change `task_name` and `task_config`

4. **Run Evaluation**
   - Execute `eval.sh`
   - Results are saved automatically

---

## 📋 Quick Checklist

### ✅ One-Time Setup (Check Once)

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

### 🔄 Per-Checkpoint Checklist

```bash
# Check checkpoint converted (if needed)
ls -lh ~/H_RDT/checkpoints/YOUR_CHECKPOINT/pytorch_model_consolidated.bin
# Should be ~7.7GB

# Check checkpoint copied to RobotWin
ls -lh ~/RoboTwin/policy/H-RDT/checkpoints/YOUR_CHECKPOINT_NAME/pytorch_model.bin
# Should be ~7.7GB

# Check eval.sh configured
grep "ckpt_setting=" ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh
# Should point to your checkpoint
```

---

## 🚀 Workflow Examples

### First Time Setup

```bash
# 1. One-time: Install dependencies
cd ~/RoboTwin && source ~/.config/hrdt/activate.sh
pip install -r script/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
bash script/_install.sh
sudo apt-get install -y ffmpeg

# 2. One-time: Download assets
cd ~/RoboTwin/assets
python _download.py
unzip *.zip && rm -f *.zip
cd ~/RoboTwin && python script/update_embodiment_config_path.py

# 3. One-time: Copy code structure
# (Use prepare_robotwin_evaluation.sh or manual copy)
```

### Evaluating Checkpoint #1

```bash
# 1. Per-checkpoint: Convert (if needed)
bash convert_deepspeed_checkpoint.sh \
    ~/H_RDT/checkpoints/checkpoint-1 \
    ~/H_RDT/checkpoints/checkpoint-1/pytorch_model_consolidated.bin

# 2. Per-checkpoint: Copy to RobotWin
bash prepare_robotwin_evaluation.sh \
    checkpoint1 \
    ~/H_RDT/checkpoints/checkpoint-1

# 3. Per-checkpoint: Update eval.sh
# Edit ckpt_setting="checkpoints/checkpoint1"

# 4. Per-checkpoint: Run evaluation
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
bash eval.sh
```

### Evaluating Checkpoint #2 (Same Machine)

```bash
# Skip one-time setup - already done!

# 1. Per-checkpoint: Convert (if needed)
bash convert_deepspeed_checkpoint.sh \
    ~/H_RDT/checkpoints/checkpoint-2 \
    ~/H_RDT/checkpoints/checkpoint-2/pytorch_model_consolidated.bin

# 2. Per-checkpoint: Copy to RobotWin
bash prepare_robotwin_evaluation.sh \
    checkpoint2 \
    ~/H_RDT/checkpoints/checkpoint-2

# 3. Per-checkpoint: Update eval.sh
# Edit ckpt_setting="checkpoints/checkpoint2"

# 4. Per-checkpoint: Run evaluation
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
bash eval.sh
```

---

## 📁 File Locations

### One-Time Files (Shared Across Checkpoints)

```
~/RoboTwin/policy/H-RDT/
├── inference/robotwin2_example/H-RDT/
│   ├── deploy_policy.py          # Evaluation code
│   ├── eval.sh                    # Evaluation script
│   └── utils/
│       ├── lang_embeddings/      # 49 task embeddings (shared)
│       ├── hrdt.yaml             # Config (shared)
│       └── stats.json            # Stats (shared)
├── bak/                          # Vision encoder models (shared)
├── deploy_policy.py              # Policy module entry
├── deploy_policy.yml
└── __init__.py                   # Module exports
```

### Per-Checkpoint Files (Each Checkpoint Has Its Own)

```
~/RoboTwin/policy/H-RDT/checkpoints/
├── checkpoint1/
│   ├── pytorch_model.bin         # Checkpoint-specific
│   └── config.json               # Checkpoint-specific
├── checkpoint2/
│   ├── pytorch_model.bin
│   └── config.json
└── checkpoint3/
    ├── pytorch_model.bin
    └── config.json
```

---

## 💡 Tips

1. **Keep Checkpoint Names Descriptive:**
   - Use names like `table8_checkpoint30`, `table8_checkpoint50`, etc.
   - Makes it easier to track which checkpoint you're evaluating

2. **Reuse Language Embeddings:**
   - Language embeddings are shared across all checkpoints
   - Only copy once in one-time setup

3. **Convert Checkpoints as Needed:**
   - Only convert if checkpoint is in DeepSpeed format
   - Already consolidated checkpoints can be copied directly

4. **Quick Checkpoint Switch:**
   - To evaluate different checkpoint, just update `ckpt_setting` in `eval.sh`
   - No need to re-run one-time setup

5. **Batch Evaluation:**
   - Set up multiple checkpoint directories
   - Create a loop script to evaluate all checkpoints sequentially

