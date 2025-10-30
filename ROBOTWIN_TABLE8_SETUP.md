# H-RDT Table 8 Replication Progress Summary

## 🎉 Current Status: Ready for Fine-tuning

We have successfully completed all the setup steps needed to replicate Table 8 from the H-RDT paper!

## ✅ Completed Tasks

### 1. **Table 8 Tasks Identified**
- Extracted the complete Table 8 from the H-RDT paper
- Identified all 13 tasks needed for replication
- Confirmed expected results: 68.7% (Easy) / 25.6% (Hard) average success rate

### 2. **Data Downloaded**
- Downloaded all 13 Table 8 tasks from [RoboTwin2.0 Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)
- Source: `aloha-agilex_clean_50` data (clean trajectories)
- Total compressed size: 6.6 GB

### 3. **Data Extracted**
- Extracted all zip files to organized directory structure
- Total extracted size: 10.4 GB
- 650 episodes total (50 per task × 13 tasks)
- Location: `/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted/`

### 4. **Code Updates**
- Added `--dataset_name` argument to `main.py` for dynamic dataset selection
- Added `--robotwin_mode`, `--robotwin_task_name`, `--robotwin_hdf5_folder` arguments for single-task training
- Updated `train/train.py` to use the dataset name argument
- Updated `datasets/dataset.py` to support both single-task and multi-task modes
- Created automated download and extraction scripts

### 5. **Documentation Updated**
- Updated `README.md` with Table 8 tasks information
- Updated `datasets/robotwin2/README.md` with extraction details
- Created comprehensive summary documents

### 6. **Data Verification Created**
- Created `verify_robotwin2_dataset.py` for comprehensive data validation
- Verifies HDF5 structure, language embeddings, and training compatibility
- Confirms all 650 episodes across 13 tasks are ready for fine-tuning

## 📁 Files Created/Modified

### New Scripts:
- `download_table8_tasks.py` - Downloads Table 8 tasks from Hugging Face
- `extract_table8_data.py` - Extracts and organizes the data
- `verify_robotwin2_dataset.py` - Comprehensive verification of Table 8 data and training compatibility
- `install_nvtop.sh` - Automated nvtop installation
- `download_vision_models.py` - Downloads DINO/SigLIP models

### Documentation:
- `TABLE8_DOWNLOAD_SUMMARY.md` - Download results summary
- `TABLE8_EXTRACTION_SUMMARY.md` - Extraction results summary

### Modified Files:
- `main.py` - Added `--dataset_name` argument
- `train/train.py` - Updated to use dynamic dataset name
- `setup_instance.sh` - Added nvtop installation and vision model download
- `README.md` - Updated with Table 8 information
- `datasets/robotwin2/README.md` - Updated with extraction details

## 🚀 Next Steps: Ready to Fine-tune

### 1. **Set Environment Variables**
```bash
export ROBOTWIN2_DATA_ROOT="/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted"
```

### 2. **Choose Training Mode**

#### Option A: **Multi-Task Training** (Default)
Train on all 13 tasks simultaneously with balanced sampling:

```bash
accelerate launch --main_process_port 29500 main.py \
  --pretrained_vision_encoder_name_or_path="dino-siglip" \
  --deepspeed="./configs/zero1.json" \
  --config_path "configs/hrdt_finetune.yaml" \
  --output_dir "./checkpoints/table8_finetune_pretrain0618" \
  --train_batch_size 16 \
  --sample_batch_size 16 \
  --max_train_steps 10000 \
  --checkpointing_period 1000 \
  --sample_period 500 \
  --checkpoints_total_limit 10 \
  --lr_scheduler "constant_with_warmup" \
  --learning_rate 1e-4 \
  --mixed_precision "bf16" \
  --dataloader_num_workers 24 \
  --dataset_type "finetune" \
  --dataset_name "robotwin_agilex" \
  --robotwin_mode "multi_task" \
  --report_to wandb \
  --upsample_rate 3 \
  --image_aug \
  --gradient_checkpointing \
  --precomp_lang_embed \
  --training_mode "lang" \
  --mode "finetune" \
  --pretrained_backbone_path "./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin"
```

#### Option B: **Single-Task Training** (For Table 8 Replication)
Train on one task at a time. Run separate fine-tuning for each of the 13 tasks:

```bash
# Example: Fine-tune on "grab_roller" task
accelerate launch --main_process_port 29500 main.py \
  --pretrained_vision_encoder_name_or_path="dino-siglip" \
  --deepspeed="./configs/zero1.json" \
  --config_path "configs/hrdt_finetune.yaml" \
  --output_dir "./checkpoints/table8_grab_roller" \
  --train_batch_size 16 \
  --sample_batch_size 16 \
  --max_train_steps 10000 \
  --checkpointing_period 1000 \
  --sample_period 500 \
  --checkpoints_total_limit 10 \
  --lr_scheduler "constant_with_warmup" \
  --learning_rate 1e-4 \
  --mixed_precision "bf16" \
  --dataloader_num_workers 24 \
  --dataset_type "finetune" \
  --dataset_name "robotwin_agilex" \
  --robotwin_mode "single_task" \
  --robotwin_task_name "grab_roller" \
  --robotwin_hdf5_folder "aloha-agilex_clean_50/data" \
  --report_to wandb \
  --upsample_rate 3 \
  --image_aug \
  --gradient_checkpointing \
  --precomp_lang_embed \
  --training_mode "lang" \
  --mode "finetune" \
  --pretrained_backbone_path "./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin"
```

**Repeat for all 13 tasks:**
- `grab_roller`, `handover_mic`, `lift_pot`, `move_can_pot`, `open_laptop`
- `pick_dual_bottles`, `place_dual_shoes`, `place_object_basket`, `place_phone_stand`
- `put_bottles_dustbin`, `put_object_cabinet`, `stack_blocks_two`, `stack_bowls_two`

### 3. **New RobotWin Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--robotwin_mode` | `"multi_task"` | Dataset mode: `"single_task"` or `"multi_task"` |
| `--robotwin_task_name` | `None` | Task name for single-task mode (required if `robotwin_mode="single_task"`) |
| `--robotwin_hdf5_folder` | `"aloha-agilex_clean_50/data"` | HDF5 folder path within task directory |

**Data Path Structure:**
```
ROBOTWIN2_DATA_ROOT/
├── {task_name}/
│   └── {robotwin_hdf5_folder}/
│       ├── episode0.hdf5
│       └── ...
```

Example: `/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted/grab_roller/aloha-agilex_clean_50/data/`

### 4. **Expected Results**
Based on Table 8 from the H-RDT paper:
- **Easy mode**: ~68.7% average success rate
- **Hard mode**: ~25.6% average success rate

## 📊 Data Summary

| Task | Episodes | Size | Description |
|------|----------|------|-------------|
| `grab_roller` | 50 | 289.6 MB | Use both arms to grab the roller |
| `handover_mic` | 50 | 642.9 MB | Handover microphone between arms |
| `lift_pot` | 50 | 393.2 MB | Use both arms to lift the pot |
| `move_can_pot` | 50 | 398.6 MB | Move can beside the pot |
| `open_laptop` | 50 | 549.8 MB | Open laptop with one arm |
| `pick_dual_bottles` | 50 | 448.2 MB | Pick up bottles with both arms |
| `place_dual_shoes` | 50 | 899.2 MB | Place shoes in shoebox |
| `place_object_basket` | 50 | 974.2 MB | Place object in basket |
| `place_phone_stand` | 50 | 338.6 MB | Place phone on stand |
| `put_bottles_dustbin` | 50 | 2.7 GB | Put bottles in dustbin |
| `put_object_cabinet` | 50 | 947.6 MB | Put object in cabinet drawer |
| `stack_blocks_two` | 50 | 873.2 MB | Stack green block on red block |
| `stack_bowls_two` | 50 | 942.4 MB | Stack two bowls |

**Total**: 650 episodes, 10.4 GB extracted data

## 🎯 Ready for Table 8 Replication!

All setup is complete. You can now run fine-tuning to replicate the Table 8 results from the H-RDT paper. The data is properly organized, the code is updated, and comprehensive verification confirms everything is ready.

### Final Verification:
```bash
# Verify everything is ready
python verify_robotwin2_dataset.py --data_root /mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted
```

Expected output: `🎉 All Table 8 tasks verified successfully! ✅ 13 tasks ready for fine-tuning`

---

**Status**: ✅ **READY FOR FINE-TUNING** 🚀
