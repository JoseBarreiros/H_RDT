# Table 8 Tasks Download Summary

## Successfully Downloaded 13 Tasks for Table 8 Replication

**Source**: [RoboTwin2.0 Dataset on Hugging Face](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)

**Platform**: Aloha-Agilex (dual-arm robot)  
**Data Type**: Clean trajectories (50 per task)  
**Total Size**: 6.6 GB  
**Location**: `/mnt/disks/hrdt-data/robotwin2/table8_tasks/dataset/`

## Downloaded Tasks

| Task | File Size | Description |
|------|-----------|-------------|
| `grab_roller` | 212.1 MB | Use both arms to grab the roller on the table |
| `handover_mic` | 393.4 MB | Use one arm to grasp the microphone and handover it to the other arm |
| `lift_pot` | 267.3 MB | Use BOTH arms to lift the pot |
| `move_can_pot` | 298.1 MB | Pick up the can and move it to beside the pot |
| `open_laptop` | 399.2 MB | Use one arm to open the laptop |
| `pick_dual_bottles` | 305.8 MB | Pick up one bottle with one arm, and pick up another bottle with the other arm |
| `place_dual_shoes` | 598.8 MB | Use both arms to pick up the two shoes and put them in the shoebox |
| `place_object_basket` | 605.8 MB | Grab the target object and put it in the basket, then move the basket slightly away |
| `place_phone_stand` | 243.7 MB | Pick up the phone and put it on the phone stand |
| `put_bottles_dustbin` | 1.6 GB | Grab the bottles and put them into the dustbin to the left of the table |
| `put_object_cabinet` | 678.7 MB | Open the cabinet's drawer and put the object in the drawer |
| `stack_blocks_two` | 573.8 MB | Stack the green block on the red block |
| `stack_bowls_two` | 593.4 MB | Stack the two bowls on top of each other |

## Table 8 Performance Results

These tasks correspond to the **13 tasks evaluated in Table 8** of the H-RDT paper, where:

- **H-RDT achieved the highest average performance**: 68.7% (Easy mode) and 25.6% (Hard mode)
- **Evaluation modes**: Easy (clean scenes) and Hard (domain randomization)
- **Platform**: Aloha-Agilex-1.0 dual-arm robot
- **Training**: 10k steps, 4 H100 GPUs, batch size 16 per GPU

## Next Steps

1. **Extract the zip files** when ready to use the data
2. **Configure the dataset path** in your H-RDT training configuration
3. **Run fine-tuning** using these specific tasks to replicate Table 8 results

## File Structure

```
/mnt/disks/hrdt-data/robotwin2/table8_tasks/dataset/
├── grab_roller/aloha-agilex_clean_50.zip
├── handover_mic/aloha-agilex_clean_50.zip
├── lift_pot/aloha-agilex_clean_50.zip
├── move_can_pot/aloha-agilex_clean_50.zip
├── open_laptop/aloha-agilex_clean_50.zip
├── pick_dual_bottles/aloha-agilex_clean_50.zip
├── place_dual_shoes/aloha-agilex_clean_50.zip
├── place_object_basket/aloha-agilex_clean_50.zip
├── place_phone_stand/aloha-agilex_clean_50.zip
├── put_bottles_dustbin/aloha-agilex_clean_50.zip
├── put_object_cabinet/aloha-agilex_clean_50.zip
├── stack_blocks_two/aloha-agilex_clean_50.zip
└── stack_bowls_two/aloha-agilex_clean_50.zip
```

## Usage

To use these tasks for fine-tuning H-RDT:

```bash
# Set dataset path
export ROBOTWIN_DATA_ROOT="/mnt/disks/hrdt-data/robotwin2/table8_tasks/dataset"

# Run fine-tuning with robotwin_agilex dataset
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

---

**Download completed successfully!** 🎉
