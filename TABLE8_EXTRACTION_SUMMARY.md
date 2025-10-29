# Table 8 Tasks Data Extraction Summary

## ✅ Successfully Extracted All 13 Tasks

**Total Size**: 17 GB (6.6 GB compressed → 10.4 GB extracted)  
**Location**: `/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted/`  
**Tasks**: 13 tasks × 50 episodes each = 650 total episodes

## 📁 Data Structure

Each task follows this structure:
```
task_name/
└── aloha-agilex_clean_50/
    ├── data/                    # HDF5 trajectory files (50 episodes)
    │   ├── episode0.hdf5
    │   ├── episode1.hdf5
    │   └── ... (50 episodes total)
    ├── _traj_data/             # Additional trajectory data
    ├── instructions/           # Task instructions
    ├── video/                 # Video recordings
    ├── scene_info.json        # Scene configuration
    └── seed.txt              # Random seed used
```

## 🎯 Extracted Tasks

| Task | Episodes | Extracted Size | Description |
|------|----------|----------------|-------------|
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

## 📊 HDF5 Data Structure

Each episode contains:

### 🤖 Actions (14-dimensional)
- `joint_action/left_arm`: (93, 6) - Left arm joint actions
- `joint_action/left_gripper`: (93,) - Left gripper actions  
- `joint_action/right_arm`: (93, 6) - Right arm joint actions
- `joint_action/right_gripper`: (93,) - Right gripper actions
- `joint_action/vector`: (93, 14) - Combined action vector

### 📷 Observations (Multi-camera)
- `observation/front_camera/rgb`: RGB images from front camera
- `observation/head_camera/rgb`: RGB images from head camera  
- `observation/left_camera/rgb`: RGB images from left camera
- `observation/right_camera/rgb`: RGB images from right camera
- Camera intrinsics/extrinsics for each view

### 🎯 End Poses
- `endpose/left_endpose`: (93, 7) - Left arm end-effector poses
- `endpose/right_endpose`: (93, 7) - Right arm end-effector poses
- `endpose/left_gripper`: (93,) - Left gripper states
- `endpose/right_gripper`: (93,) - Right gripper states

## 🚀 Usage for H-RDT Fine-tuning

### 1. Update Dataset Configuration
Set the data root path in your configuration:
```bash
export ROBOTWIN_DATA_ROOT="/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted"
```

### 2. Run Fine-tuning
```bash
accelerate launch main.py \
    --dataset_name="robotwin_agilex" \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --config_path configs/hrdt_finetune.yaml \
    --output_dir ./checkpoints/table8_finetune \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --dataset_type finetune \
    --report_to wandb
```

### 3. Expected Results
Based on Table 8 from the H-RDT paper:
- **Easy mode**: ~68.7% average success rate
- **Hard mode**: ~25.6% average success rate

## 📋 Data Verification

✅ **All 13 tasks extracted successfully**  
✅ **50 episodes per task (650 total episodes)**  
✅ **Multi-camera RGB observations**  
✅ **14-dimensional dual-arm actions**  
✅ **Proper HDF5 structure maintained**  
✅ **Ready for H-RDT fine-tuning**

---

**🎉 Data extraction complete! Ready for Table 8 replication experiments.**
