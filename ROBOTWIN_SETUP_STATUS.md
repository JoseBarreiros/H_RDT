# RobotWin Evaluation Setup Status

## ✅ Completed Steps

1. **Checkpoint Preparation**
   - Checkpoint copied to: `~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/`
   - Language embeddings copied (49 files)
   - Config and stats files copied
   - Vision encoder models (bak folder) copied
   - eval.sh configured with checkpoint path

2. **Dependencies Installed**
   - ✅ SAPIEN 3.0.0b1 installed
   - ✅ RobotWin requirements installed (mplib, gymnasium, open3d, etc.)
   - ✅ PyTorch3D installed
   - ✅ Curobo installed
   - ✅ Code fixes applied (SAPIEN and mplib adjustments)

## ⚠️ Remaining Step: Download Assets

RobotWin requires assets to be downloaded from HuggingFace. This is a **large download** (several GB).

### To Download Assets:

```bash
cd ~/RoboTwin
source ~/.config/hrdt/activate.sh

# Download assets from HuggingFace
cd assets
python _download.py

# Extract zip files
unzip background_texture.zip
unzip embodiments.zip  
unzip objects.zip

# Remove zip files
rm -rf *.zip

# Configure paths
cd ..
python ./script/update_embodiment_config_path.py
```

### Or use the convenience script:

```bash
cd ~/RoboTwin
source ~/.config/hrdt/activate.sh
bash script/_download_assets.sh
```

**Note:** This download may take 20-30 minutes depending on your internet connection.

## 🚀 After Assets Are Downloaded

Once assets are downloaded, you can evaluate your checkpoint:

```bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

# Edit eval.sh to change task_name and task_config as needed
# Current settings:
#   task_name="grab_roller"
#   task_config="demo_randomized"  (Hard mode)
#   task_config="demo_clean"       (Easy mode)

source ~/.config/hrdt/activate.sh
bash eval.sh
```

## 📋 Table 8 Tasks

To evaluate all 13 Table 8 tasks, edit `task_name` in eval.sh for each task:
- `grab_roller`
- `handover_mic`
- `lift_pot`
- `move_can_pot`
- `open_laptop`
- `pick_dual_bottles`
- `place_dual_shoes`
- `place_object_basket`
- `place_phone_stand`
- `put_bottles_dustbin`
- `put_object_cabinet`
- `stack_blocks_two`
- `stack_bowls_two`

## Expected Results

Based on Table 8 from the H-RDT paper:
- **Easy mode** (`demo_clean`): ~68.7% average success rate
- **Hard mode** (`demo_randomized`): ~25.6% average success rate

