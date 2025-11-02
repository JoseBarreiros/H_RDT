# H-RDT Pretrain Data Processing

This directory contains scripts for processing EgoDex dataset for H-RDT pretraining.

## Overview

The pretraining data processing consists of three steps:
1. **Precompute 48D Actions** - Convert hand transforms to 48-dimensional action vectors
2. **Calculate Statistics** - Compute min/max values for action normalization  
3. **Encode Language** - Generate T5 embeddings for language instructions

## EgoDex Dataset Information

### Dataset Statistics

**Total Episodes**: 318,082

| Split | Episodes | Percentage |
|-------|----------|------------|
| Training | 314,839 | 99.0% |
| Test | 3,243 | 1.0% |
| **Total** | **318,082** | **100%** |

**Additional Details:**
- **Tasks**: 111 different tasks
- **Format**: Each episode = 1 HDF5 file (with corresponding MP4 video and `.pt` language embedding file)
- **Action Dimensions**: 48-dimensional hand actions
- **Preprocessing Status**: All episodes have been processed with:
  - 48D actions precomputed
  - Language embeddings generated (T5)
  - Statistics calculated

### Scaling Law Experiments

When using `--data_percentage` for scaling law experiments:

- The `data_percentage` parameter only affects the **training set**
- The **test set** (3,243 episodes) is always used at 100% for validation
- **Sampling**: Done at the episode/file level (not timestep level)
  - Each selected episode can be sampled multiple times at different timesteps during training
  - This provides diversity through both episode selection and timestep sampling

**Examples:**
- `data_percentage=0.1` (10%): ~31,484 training episodes + 3,243 test episodes
- `data_percentage=0.5` (50%): ~157,420 training episodes + 3,243 test episodes
- `data_percentage=1.0` (100%): 314,839 training episodes + 3,243 test episodes

## Quick Start

### 1. Setup Environment

Edit the paths in `setup_pretrain.sh` according to your environment:

```bash
# Edit the script with your paths
nano datasets/pretrain/setup_pretrain.sh

# Then source it to set up environment variables
source datasets/pretrain/setup_pretrain.sh
```

**Required paths to configure:**
- `EGODEX_DATA_ROOT`: Path to your EgoDex dataset
- `T5_MODEL_PATH`: Path to your T5-v1_1-xxl model

### 2. Run Complete Pipeline

```bash
# Run all three steps automatically
./datasets/pretrain/run_pretrain_pipeline.sh
```

### 3. Run Individual Steps (Optional)

```bash
# Step 1: Precompute 48D actions
python datasets/pretrain/precompute_48d_actions.py

# Step 2: Calculate statistics
python datasets/pretrain/calc_stat.py

# Step 3: Encode language embeddings
python datasets/pretrain/encode_lang_batch.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EGODEX_DATA_ROOT` | EgoDex dataset root directory | Required |
| `T5_MODEL_PATH` | T5-v1_1-xxl model path | Required |
| `HRDT_PROJECT_ROOT` | H-RDT project root | Auto-detected |
| `HRDT_CONFIG_PATH` | Config file path | `configs/hrdt_pretrain.yaml` |
| `HRDT_OUTPUT_DIR` | Output directory | `datasets/pretrain` |
| `NUM_PROCESSES` | Number of processes | 8 |
| `NUM_GPUS` | Number of GPUs | 8 |
| `PROCESSES_PER_GPU` | Processes per GPU | 4 |
| `FORCE_OVERWRITE` | Force overwrite existing data | true |

### Command Line Arguments

Each script also accepts command line arguments that override environment variables:

```bash
# Precompute 48D actions with custom settings
python datasets/pretrain/precompute_48d_actions.py \
    --data_root /path/to/egodex \
    --num_processes 16 \
    --force_overwrite \
    --test_mode

# Calculate statistics (no additional args needed)
python datasets/pretrain/calc_stat.py

# Encode language embeddings (uses environment variables)
python datasets/pretrain/encode_lang_batch.py
```

## Directory Structure

```
datasets/pretrain/
├── setup_pretrain.sh              # Environment setup script
├── run_pretrain_pipeline.sh       # Complete pipeline runner
├── run_test_subset.sh             # Test subset processing script
├── precompute_48d_actions.py     # Step 1: Precompute actions
├── calc_stat.py                   # Step 2: Calculate statistics
├── encode_lang_batch.py           # Step 3: Encode language
├── egodex_dataset.py              # EgoDex dataset loader
├── egodex_stat.json                # Generated statistics file
├── egodex_large_values.txt        # Outlier detection log
├── test_subset_output/            # Test subset output directory
│   ├── egodex_stat.json           # Test statistics
│   └── egodex_large_values.txt    # Test large values log
└── README.md                      # This file
```

## Expected Dataset Structure

Your EgoDex dataset should be organized as:

```
$EGODEX_DATA_ROOT/
├── train/
│   ├── task1/
│   │   ├── 0.hdf5
│   │   ├── 0.mp4
│   │   ├── 0.pt
│   │   ├── 1.hdf5
│   │   └── ...
│   └── task2/
└── test/
    ├── task1/
    └── task2/
```

## Output Files

After processing, you'll have:

1. **48D Action Data**: Added as `actions_48d` key in all HDF5 files
2. **Statistics**: `egodex_stat.json` with min/max values for normalization + metadata (file count, timestamp, dimensions)
3. **Language Embeddings**: `.pt` files alongside each HDF5 file
4. **Log Files**: `egodex_large_values.txt` with outlier information

### Statistics File Format

The `egodex_stat.json` file now includes metadata:
```json
{
    "egodex": {
        "min": [...],
        "max": [...]
    },
    "metadata": {
        "files_processed": 318082,
        "timestamp": "2025-10-27T16:06:09.534163",
        "data_root": "/path/to/egodex/organized",
        "action_dims": 48,
        "large_values_count": 0,
        "error_count": 0
    }
}
```

## Testing with Test Subset

For quick testing without processing the full dataset:

```bash
# Create and process test subset (won't overwrite full dataset stats)
./datasets/pretrain/run_test_subset.sh
```

This script:
- Creates separate output directory (`test_subset_output/`)
- Processes only the test subset
- Generates its own statistics file
- Runs verification at the end