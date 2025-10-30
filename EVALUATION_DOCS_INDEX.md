# H-RDT Evaluation Documentation Index

## 📚 Documentation Files

### Main Guides

1. **`ROBOTWIN_EVALUATION_README.md`** ⭐ **START HERE**
   - Complete guide for RobotWin evaluation
   - Quick start and detailed setup
   - Troubleshooting section
   - File structure reference

2. **`ROBOTWIN_EVALUATION_SETUP.md`**
   - Detailed step-by-step setup instructions
   - DeepSpeed checkpoint conversion details
   - RobotWin environment setup
   - Asset download and configuration

### Quick References

3. **`EVALUATION_QUICK_REFERENCE.md`**
   - Quick command reference
   - Table 8 task list
   - Evaluation modes
   - Current results

4. **`EVALUATION_SETUP_SUMMARY.md`**
   - Summary of completed setup
   - Verification checklist
   - Next steps

### Related Guides

5. **`ROBOTWIN_TABLE8_SETUP.md`**
   - Table 8 fine-tuning setup
   - Data download and extraction
   - Single-task vs multi-task training

6. **`ROBOTWIN2_FINETUNING_GUIDE.md`**
   - RobotWin2 fine-tuning guide

## 🛠️ Helper Scripts

### `convert_deepspeed_checkpoint.sh`

Converts DeepSpeed ZeRO-3 checkpoint to consolidated format.

**Usage:**
```bash
bash convert_deepspeed_checkpoint.sh [checkpoint_dir] [output_file]
```

**Example:**
```bash
bash convert_deepspeed_checkpoint.sh \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin
```

**What it does:**
- Checks checkpoint format
- Creates `latest` file if needed
- Runs `zero_to_fp32.py` conversion
- Verifies output file

### `prepare_robotwin_evaluation.sh`

Prepares checkpoint for RobotWin evaluation.

**Usage:**
```bash
bash prepare_robotwin_evaluation.sh [checkpoint_name] [source_checkpoint]
```

**Example:**
```bash
bash prepare_robotwin_evaluation.sh \
    table8_checkpoint30 \
    ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30
```

**What it does:**
- Copies consolidated checkpoint to RobotWin
- Copies language embeddings
- Copies configs and stats
- Sets up policy module structure
- Removes `pretrained_backbone_path` from config

## 📋 Quick Start

```bash
# 1. Convert checkpoint
bash convert_deepspeed_checkpoint.sh

# 2. Prepare for RobotWin
bash prepare_robotwin_evaluation.sh

# 3. Run evaluation
cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
source ~/.config/hrdt/activate.sh
bash eval.sh
```

## 🔍 Finding Information

- **New to evaluation?** → Start with `ROBOTWIN_EVALUATION_README.md`
- **Need detailed setup?** → See `ROBOTWIN_EVALUATION_SETUP.md`
- **Quick commands?** → See `EVALUATION_QUICK_REFERENCE.md`
- **Troubleshooting?** → See troubleshooting section in `ROBOTWIN_EVALUATION_SETUP.md`

## 📝 Notes

- All scripts use absolute paths to avoid directory issues
- Scripts include error checking and helpful messages
- Documentation is designed to be reproducible
- All paths are relative to `~/H_RDT` and `~/RoboTwin`

