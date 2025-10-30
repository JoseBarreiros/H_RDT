# Evaluation Setup Summary

## ✅ Completed Setup

Your H-RDT checkpoint (`table8_finetune_pretrain0618/checkpoint-30`) is now set up for RobotWin evaluation:

### What Was Done

1. **Checkpoint Conversion**
   - ✅ Converted DeepSpeed ZeRO-3 checkpoint to consolidated format
   - ✅ Output: `~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin` (7.7GB)

2. **RobotWin Setup**
   - ✅ Installed RobotWin dependencies (SAPIEN, mplib, pytorch3d, curobo)
   - ✅ Downloaded and extracted RobotWin assets (~15GB)
   - ✅ Installed system dependencies (ffmpeg)

3. **Checkpoint Preparation**
   - ✅ Copied consolidated checkpoint to RobotWin
   - ✅ Copied language embeddings (49 files)
   - ✅ Copied configs and stats
   - ✅ Set up policy module structure
   - ✅ Removed `pretrained_backbone_path` from config

4. **Evaluation Configuration**
   - ✅ Configured `eval.sh` script
   - ✅ Fixed path issues in evaluation script

### Current Status

**Evaluation Running Successfully:**
- Task: `grab_roller`
- Mode: Hard (`demo_randomized`)
- Success Rate: ~12.5% (2/16 trials so far)
- Status: ✅ Working correctly

### Files Created

**Documentation:**
- `ROBOTWIN_EVALUATION_SETUP.md` - Detailed setup guide
- `ROBOTWIN_EVALUATION_README.md` - Main evaluation guide
- `EVALUATION_QUICK_REFERENCE.md` - Quick command reference

**Scripts:**
- `convert_deepspeed_checkpoint.sh` - Checkpoint conversion helper
- `prepare_robotwin_evaluation.sh` - RobotWin preparation helper

**Checkpoints:**
- `~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin` (7.7GB)
- `~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/pytorch_model.bin` (7.7GB)

### How to Replicate

1. **Convert checkpoint:**
   ```bash
   bash convert_deepspeed_checkpoint.sh
   ```

2. **Prepare for RobotWin:**
   ```bash
   bash prepare_robotwin_evaluation.sh
   ```

3. **Run evaluation:**
   ```bash
   cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
   source ~/.config/hrdt/activate.sh
   bash eval.sh
   ```

### Next Steps

To evaluate other tasks or modes:

1. **Edit `eval.sh`:**
   ```bash
   cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
   nano eval.sh
   ```
   
   Change:
   - `task_name="grab_roller"` → any Table 8 task name
   - `task_config="demo_randomized"` → `"demo_clean"` for Easy mode

2. **Run evaluation:**
   ```bash
   bash eval.sh
   ```

### Documentation

- **Main Guide:** `ROBOTWIN_EVALUATION_README.md`
- **Detailed Setup:** `ROBOTWIN_EVALUATION_SETUP.md`
- **Quick Reference:** `EVALUATION_QUICK_REFERENCE.md`

### Verification

To verify everything is set up correctly:

```bash
# Check consolidated checkpoint exists
ls -lh ~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model_consolidated.bin
# Should be ~7.7GB

# Check RobotWin checkpoint exists
ls -lh ~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/pytorch_model.bin
# Should be ~7.7GB

# Check language embeddings
ls ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/*.pt | wc -l
# Should be 49 files

# Check assets
ls ~/RoboTwin/assets/objects/objaverse/list.json
# Should exist
```

### Troubleshooting

See `ROBOTWIN_EVALUATION_SETUP.md` for detailed troubleshooting guide.

Common issues:
- **Checkpoint loading errors:** Ensure using consolidated checkpoint
- **Module import errors:** Check `__init__.py` exists in policy directory
- **Asset errors:** Verify assets are downloaded and extracted

