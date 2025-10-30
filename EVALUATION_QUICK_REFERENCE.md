# Checkpoint Evaluation Quick Reference

This document provides quick reference commands for evaluating H-RDT checkpoints in RobotWin simulation.

## 🔄 Setup Types

### ⚙️ One-Time Setup (Do Once)

- [ ] RobotWin dependencies installed
- [ ] RobotWin assets downloaded (~15GB)
- [ ] Code structure copied to RobotWin
- [ ] Policy module structure set up

### 🔄 Per-Checkpoint Steps (Repeat for Each Checkpoint)

- [ ] DeepSpeed checkpoint converted (if needed)
- [ ] Checkpoint copied to RobotWin
- [ ] `ckpt_setting` updated in `eval.sh`
- [ ] Evaluation run

## Quick Commands

### ⚙️ One-Time Setup

```bash
# Install dependencies
cd ~/RoboTwin && source ~/.config/hrdt/activate.sh
pip install -r script/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
bash script/_install.sh
sudo apt-get install -y ffmpeg

# Download assets
cd ~/RoboTwin/assets
python _download.py
unzip *.zip && rm -f *.zip
cd ~/RoboTwin && python script/update_embodiment_config_path.py

# Copy code structure (see ROBOTWIN_SETUP_TYPES.md for details)
```

### 🔄 Per-Checkpoint: Convert Checkpoint

```bash
cd ~/H_RDT
bash convert_deepspeed_checkpoint.sh \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin
```

### 🔄 Per-Checkpoint: Prepare for RobotWin

```bash
cd ~/H_RDT
bash prepare_robotwin_evaluation.sh \
    table8_checkpoint30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30
```

### 🔄 Per-Checkpoint: Run Evaluation

```bash
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh

# Edit eval.sh: ckpt_setting="checkpoints/table8_checkpoint30"
# Then run:
bash eval.sh
```

## Table 8 Tasks

| Task Name | Description |
|-----------|-------------|
| `grab_roller` | Grab a roller object |
| `handover_mic` | Hand over a microphone |
| `lift_pot` | Lift a pot |
| `move_can_pot` | Move can to pot |
| `open_laptop` | Open laptop lid |
| `pick_dual_bottles` | Pick up two bottles |
| `place_dual_shoes` | Place two shoes |
| `place_object_basket` | Place object in basket |
| `place_phone_stand` | Place phone on stand |
| `put_bottles_dustbin` | Put bottles in dustbin |
| `put_object_cabinet` | Put object in cabinet |
| `stack_blocks_two` | Stack two blocks |
| `stack_bowls_two` | Stack two bowls |

## Evaluation Modes

- **Easy Mode:** `task_config="demo_clean"` - Clean backgrounds, minimal randomization
- **Hard Mode:** `task_config="demo_randomized"` - Messy table, random backgrounds, lighting variations

## Current Results

- **Task:** `grab_roller`
- **Mode:** Hard (`demo_randomized`)
- **Success Rate:** ~12.5% (2/16 trials)
- **Checkpoint:** `table8_finetune_pretrain0618/checkpoint-30`

## File Locations

### One-Time Files (Shared)

- **Language embeddings:** `~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/` (49 files)
- **Config files:** `~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/`
- **Vision encoder models:** `~/RoboTwin/policy/H-RDT/bak/`

### Per-Checkpoint Files

- **Consolidated checkpoint:** `~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin` (7.7GB)
- **RobotWin checkpoint:** `~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/pytorch_model.bin`
- **Evaluation script:** `~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh`
- **Policy module:** `~/RoboTwin/policy/H-RDT/deploy_policy.py`

## Troubleshooting

See `ROBOTWIN_EVALUATION_SETUP.md` for detailed troubleshooting guide.

## Documentation

- **Setup Types:** `ROBOTWIN_SETUP_TYPES.md` - One-time vs per-checkpoint breakdown
- **Main Guide:** `ROBOTWIN_EVALUATION_README.md` - Complete evaluation guide
- **Detailed Setup:** `ROBOTWIN_EVALUATION_SETUP.md` - Step-by-step instructions
