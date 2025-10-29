# H-RDT RobotWin2 Data Processing

This directory contains scripts for processing RobotWin2 dataset for H-RDT fine-tuning.

## 🎯 Table 8 Tasks Ready for Fine-tuning

**✅ Table 8 Tasks Downloaded and Extracted!**

We have successfully downloaded and extracted the **13 tasks** needed to replicate Table 8 from the H-RDT paper:

- **Location**: `/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted/`
- **Tasks**: 13 tasks × 50 episodes each = 650 total episodes
- **Size**: 10.4 GB extracted data
- **Platform**: Aloha-Agilex dual-arm robot
- **Data Type**: Clean trajectories (aloha-agilex_clean_50)

### Table 8 Tasks Available:
1. `grab_roller` - Use both arms to grab the roller
2. `handover_mic` - Handover microphone between arms  
3. `lift_pot` - Use both arms to lift the pot
4. `move_can_pot` - Move can beside the pot
5. `open_laptop` - Open laptop with one arm
6. `pick_dual_bottles` - Pick up bottles with both arms
7. `place_dual_shoes` - Place shoes in shoebox
8. `place_object_basket` - Place object in basket
9. `place_phone_stand` - Place phone on stand
10. `put_bottles_dustbin` - Put bottles in dustbin
11. `put_object_cabinet` - Put object in cabinet drawer
12. `stack_blocks_two` - Stack green block on red block
13. `stack_bowls_two` - Stack two bowls

### Expected Results (Table 8):
- **Easy mode**: 68.7% average success rate
- **Hard mode**: 25.6% average success rate

## 🔍 Data Verification

**Verify your Table 8 data before training:**

```bash
# Verify data integrity and training compatibility
python verify_robotwin2_dataset.py --data_root /mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted

# For detailed output
python verify_robotwin2_dataset.py --data_root /mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted --verbose
```

The verification script checks:
- ✅ **HDF5 Structure**: Dual-arm actions (14D), multi-camera observations (4 cameras)
- ✅ **Language Embeddings**: T5-XXL embeddings (4096D) for all 13 tasks
- ✅ **Training Compatibility**: Config file compatibility with data structure
- ✅ **Data Integrity**: All 650 episodes across 13 tasks
- ✅ **Action Dimensions**: 14D (6+1+6+1 for left_arm+left_gripper+right_arm+right_gripper)

## Quick Start

### 1. Setup Environment

Edit the paths in `setup_robotwin2.sh` according to your environment:

```bash
# Edit the script with your paths
nano datasets/robotwin2/setup_robotwin2.sh

# Then source it to set up environment variables
source datasets/robotwin2/setup_robotwin2.sh
```

**Required paths to configure (only if processing):**
- `ROBOTWIN2_DATA_ROOT`: Path to your RobotWin2 dataset (set to `/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted/` for Table 8 tasks)
- `T5_MODEL_PATH`: Path to your T5-v1_1-xxl model

### 2. Run Pipeline (Not Required)

**Since pre-computed language embeddings are already provided in the repository, you do NOT need to run the pipeline.**

The pipeline is only needed if you want to regenerate files:

```bash
# Default: Skip all processing steps (recommended - files already provided)
./datasets/robotwin2/run_robotwin2_pipeline.sh

# Only run if you need to regenerate specific files:
ENABLE_STATS_CALCULATION=true ./datasets/robotwin2/run_robotwin2_pipeline.sh
ENABLE_LANGUAGE_ENCODING=true ./datasets/robotwin2/run_robotwin2_pipeline.sh
```

### 3. Run Individual Steps (Not Required)

**These steps are not required since pre-computed files are already provided:**

```bash
# Step 1: Calculate statistics (not needed, stats.json already provided)
python datasets/robotwin2/calc_stat.py

# Step 2: Encode language embeddings (not needed, lang_embeddings/ already provided)
python datasets/robotwin2/encode_lang_batch.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROBOTWIN2_DATA_ROOT` | RobotWin2 dataset root directory | Required |
| `T5_MODEL_PATH` | T5-v1_1-xxl model path | Required |
| `HRDT_PROJECT_ROOT` | H-RDT project root | Auto-detected |
| `HRDT_CONFIG_PATH` | Config file path | `configs/hrdt_finetune.yaml` |
| `HRDT_OUTPUT_DIR` | Output directory | `datasets/robotwin2` |
| `NUM_PROCESSES` | Number of processes | 8 |
| `NUM_GPUS` | Number of GPUs | 8 |
| `PROCESSES_PER_GPU` | Processes per GPU | 4 |

### Command Line Arguments

Scripts accept command line arguments that override environment variables:

```bash
# Calculate statistics with custom settings
python datasets/robotwin2/calc_stat.py

# Encode language embeddings (uses environment variables)  
python datasets/robotwin2/encode_lang_batch.py
```

## Directory Structure

```
datasets/robotwin2/
├── setup_robotwin2.sh              # Environment setup script
├── run_robotwin2_pipeline.sh       # Complete pipeline runner
├── calc_stat.py                    # Step 1: Calculate statistics
├── encode_lang_batch.py            # Step 2: Batch encode language embeddings
├── robotwin_agilex_dataset.py      # RobotWin2 dataset loader
├── stats.json                     # Pre-computed statistics file
├── task_instructions.csv          # Task instruction mapping
├── lang_embeddings/               # Pre-computed language embeddings
│   ├── adjust_bottle.pt
│   ├── beat_block_hammer.pt
│   ├── ...                        # All task embeddings
│   └── turn_switch.pt
└── README.md                      # This file
```

## Expected Dataset Structure

### Table 8 Tasks Structure (Ready to Use)
The Table 8 tasks are already extracted and ready at `/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted/`:

```
/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted/
├── grab_roller/
│   └── aloha-agilex_clean_50/
│       ├── data/
│       │   ├── episode0.hdf5
│       │   ├── episode1.hdf5
│       │   └── ... (50 episodes)
│       ├── instructions/
│       ├── video/
│       ├── scene_info.json
│       └── seed.txt
├── handover_mic/
│   └── aloha-agilex_clean_50/
│       └── ... (same structure)
└── ... (11 more tasks)
```

### General RobotWin2 Structure (for other tasks)
Your RobotWin2 dataset should be organized as (only needed if running processing steps):

```
$ROBOTWIN2_DATA_ROOT/
├── task1/
│   ├── demo_clean/data/
│   │   ├── episode0.hdf5
│   │   ├── episode1.hdf5
│   │   └── ...
│   └── ...
├── task2/
├── task3/
└── ...
```

## Available Files

The repository includes pre-computed files:

1. **Statistics**: `stats.json` with min/max values for action normalization (Note: Not used in RobotWin2 training)
2. **Language Embeddings**: `lang_embeddings/*.pt` files with T5 embeddings for all tasks
3. **Task Instructions**: `task_instructions.csv` with task name to instruction mapping

## Language Embedding Usage

The dataset loader automatically reads language embeddings from the centralized `lang_embeddings/` directory. Each task's embedding is stored as `{task_name}.pt` containing:

```python
{
    "instruction": "task instruction text", 
    "embeddings": torch.Tensor  # T5 embeddings [1, seq_len, 4096]
}
```

The embeddings are loaded using relative paths, making the system portable across different environments.

## 🔧 Verification Script

The `verify_robotwin2_dataset.py` script provides comprehensive verification:

### Features:
- **Data Structure Validation**: Checks HDF5 files for proper dual-arm action structure
- **Language Embedding Verification**: Validates T5-XXL embeddings format and dimensions
- **Training Compatibility**: Ensures config file matches data requirements
- **Multi-threaded Processing**: Fast verification of all 650 episodes
- **Detailed Reporting**: Categorizes issues by type (HDF5, language, scene info)

### Usage Examples:
```bash
# Basic verification
python verify_robotwin2_dataset.py --data_root /path/to/extracted/data

# Detailed output with issue descriptions
python verify_robotwin2_dataset.py --data_root /path/to/extracted/data --verbose

# Custom number of workers
python verify_robotwin2_dataset.py --data_root /path/to/extracted/data --num_workers 4
```

### Expected Output:
```
🎉 All Table 8 tasks verified successfully!
✅ 13 tasks ready for fine-tuning
✅ Training configuration is compatible
```