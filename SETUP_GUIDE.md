# H-RDT Setup Guide

This guide documents the complete setup process for running H-RDT pretraining with EgoDex data, including virtual environment setup and data preprocessing.

## 📋 Prerequisites

- Python 3.10+
- CUDA-capable GPU(s) (tested with 4x NVIDIA A100-SXM4-80GB)
- Sufficient disk space for:
  - EgoDex dataset (~400GB)
  - Virtual environment (~10GB)
  - Model checkpoints and processed data (~50GB)

## 🚀 Installation

### Option 1: Virtual Environment (Recommended)

1. **Create and activate virtual environment:**
   ```bash
   cd /path/to/H_RDT
   python3 -m venv hrdt_env
   source hrdt_env/bin/activate
   pip install --upgrade pip
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Conda Environment (Alternative)

1. **Create conda environment:**
   ```bash
   conda create -n hrdt python=3.10
   conda activate hrdt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Setup

### EgoDex Dataset Structure

Ensure your EgoDex dataset is organized as follows:
```
~/egodex/organized/
├── train/
│   ├── task1/
│   │   ├── 0.hdf5
│   │   ├── 1.hdf5
│   │   └── ...
│   └── task2/
├── test/
│   ├── task1/
│   │   ├── 0.hdf5
│   │   ├── 1.hdf5
│   │   └── ...
│   └── task2/
└── additional/
```

### T5 Model Setup

The T5-v1_1-xxl model will be automatically downloaded from HuggingFace when first used. No manual download required.

## 🔧 Data Preprocessing

### 1. Configure Environment

Edit the setup script with your paths:
```bash
# Edit datasets/pretrain/setup_pretrain.sh
export EGODEX_DATA_ROOT="/home/jose-barreiros/egodex/organized"
export T5_MODEL_PATH="google/t5-v1_1-xxl"
```

### 2. Run Preprocessing Pipeline

The preprocessing consists of three steps:
1. **48D Actions**: Convert hand transforms to 48-dimensional action vectors
2. **Statistics**: Compute min/max values for action normalization
3. **Language Embeddings**: Generate T5 embeddings for language instructions

```bash
# Activate environment
source hrdt_env/bin/activate  # or conda activate hrdt

# Setup preprocessing environment
source datasets/pretrain/setup_pretrain.sh

# Run complete pipeline
./datasets/pretrain/run_pretrain_pipeline.sh
```

### 3. Verify Preprocessing Results

After successful preprocessing, you should have:

1. **48D Action Data**: Added as `actions_48d` key in all HDF5 files
2. **Statistics File**: `datasets/pretrain/egodex_stat.json`
3. **Language Embeddings**: `.pt` files alongside each HDF5 file
4. **Log Files**: `datasets/pretrain/egodex_large_values.txt`

#### Automated Verification (Recommended)

Use the comprehensive verification script:
```bash
# Run the verification script
python verify_dataset.py --data_root /path/to/your/egodex/dataset

# For detailed output showing specific issues
python verify_dataset.py --data_root /path/to/your/egodex/dataset --verbose
```

The verification script will check:
- ✅ Presence of HDF5, MP4, and language encoding (.pt) files
- ✅ Correct structure of HDF5 files (actions_48d, transforms, etc.)
- ✅ Data consistency across file types
- ✅ Language encoding validity

**Expected Output**: All files should pass verification. Language encoding length mismatches are expected and not errors (instruction-level vs frame-level data).

#### Manual Verification

```bash
# Check statistics file
ls -la datasets/pretrain/egodex_stat.json

# Check for language embeddings
find ~/egodex/organized -name "*.pt" | head -5

# Check a sample HDF5 file for 48D actions
python -c "
import h5py
with h5py.File('~/egodex/organized/test/slot_batteries/0.hdf5', 'r') as f:
    print('Keys:', list(f.keys()))
    if 'actions_48d' in f:
        print('48D actions shape:', f['actions_48d'].shape)
"
```

## 🧪 Testing Setup (Optional)

### Creating a Test Subset

For debugging or initial testing, create a small subset of the dataset:

```bash
# Create test subset (adjust paths as needed)
mkdir -p ~/egodex/test_subset/{train,test}

# Copy a few task directories for testing
cp -r ~/egodex/organized/train/dry_hands ~/egodex/test_subset/train/
cp -r ~/egodex/organized/test/dry_hands ~/egodex/test_subset/test/

# Update setup script for test subset
# Edit datasets/pretrain/setup_pretrain.sh:
export EGODEX_DATA_ROOT="/home/jose-barreiros/egodex/test_subset"

# Run preprocessing on test subset
source datasets/pretrain/setup_pretrain.sh
python datasets/pretrain/precompute_48d_actions.py --data_root /home/jose-barreiros/egodex/test_subset --num_processes 8 --force_overwrite
python datasets/pretrain/encode_lang_batch.py

# Verify test subset
python verify_dataset.py --data_root /home/jose-barreiros/egodex/test_subset
```

This approach allows you to:
- Test the complete pipeline quickly
- Debug issues without processing the full dataset
- Validate setup before running on the complete dataset

## 🎯 Starting Pretraining

### 1. Configure Dataset

Ensure the dataset is configured for EgoDex:
```python
# In datasets/dataset.py, line ~45
self.dataset_name = "egodex"
```

### 2. Run Pretraining

```bash
# Activate environment
source hrdt_env/bin/activate  # or conda activate hrdt

# Start pretraining
source pretrain.sh
```

The pretraining script will:
- Use 4 GPUs with DeepSpeed ZeRO-1 optimization
- Train for 1,000,000 steps with checkpointing every 5,000 steps
- Save checkpoints to `./checkpoints/pretrain/`
- Log to Weights & Biases (configure `WANDB_PROJECT` if needed)

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Device Errors**: Ensure you have the correct number of GPUs available
   ```bash
   nvidia-smi --list-gpus
   ```

2. **T5 Model Loading**: The model is automatically downloaded from HuggingFace. Ensure internet connectivity.

3. **Memory Issues**: Reduce batch size or use gradient checkpointing (already enabled in pretrain.sh)

4. **Permission Errors**: Ensure write permissions for checkpoint directory
   ```bash
   mkdir -p ./checkpoints/pretrain
   ```

5. **Missing Language Encodings**: If verification shows missing `.pt` files:
   ```bash
   # Re-run language encoding with correct GPU count
   source datasets/pretrain/setup_pretrain.sh
   python datasets/pretrain/encode_lang_batch.py
   ```

6. **Missing 48D Actions**: If verification shows missing `actions_48d` in HDF5 files:
   ```bash
   # Re-run 48D actions preprocessing
   source datasets/pretrain/setup_pretrain.sh
   python datasets/pretrain/precompute_48d_actions.py --data_root /path/to/dataset --num_processes 8 --force_overwrite
   ```

7. **Dataset Loading Errors**: If training fails with "Missing precomputed actions_48d data":
   - Verify preprocessing completed successfully using the verification script
   - Check that all required files exist (HDF5, MP4, .pt)
   - Ensure dataset configuration points to correct dataset name ("egodex")

8. **NCCL Communication Errors**: If you see NCCL warnings:
   - The current configuration in `pretrain.sh` should handle most cases
   - For specific network interfaces, adjust `NCCL_SOCKET_IFNAME` in `pretrain.sh`

### Verification Commands

```bash
# Check GPU availability
nvidia-smi

# Verify Python environment
source hrdt_env/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test T5 model loading
python -c "
import os
os.environ['HRDT_PROJECT_ROOT'] = '/home/jose-barreiros/H_RDT'
import sys
sys.path.append('/home/jose-barreiros/H_RDT')
from models.encoder.t5_encoder import T5Embedder
embedder = T5Embedder(device='cuda:0', from_pretrained='google/t5-v1_1-xxl')
print('T5 model loaded successfully!')
"
```

## 📈 Monitoring Training

### Checkpoints
- Checkpoints saved every 5,000 steps in `./checkpoints/pretrain/`
- Model files: `pytorch_model.bin`, `config.json`, `training_args.bin`

### Logs
- Training logs: Console output and Weights & Biases
- Preprocessing logs: `datasets/pretrain/egodex_large_values.txt`

### Resume Training
To resume from a checkpoint:
```bash
# Edit pretrain.sh and add:
--resume_from_checkpoint="checkpoint-50000" \
```

## 🎛️ Configuration Options

### Key Parameters in pretrain.sh:
- `--train_batch_size=32`: Batch size per GPU
- `--max_train_steps=1000000`: Total training steps
- `--learning_rate=1e-4`: Learning rate
- `--checkpointing_period=5000`: Checkpoint frequency
- `--mixed_precision="bf16"`: Mixed precision training

### GPU Configuration:
- `CUDA_VISIBLE_DEVICES=0,1,2,3`: Use 4 GPUs
- DeepSpeed ZeRO-1 for memory optimization
- Gradient checkpointing enabled

## 📚 Additional Resources

- [Original README](README.md): Main project documentation
- [Paper](https://arxiv.org/abs/2507.23523): Technical details
- [Project Page](https://embodiedfoundation.github.io/hrdt): Project overview
- [Model on HuggingFace](https://huggingface.co/embodiedfoundation/H-RDT): Pre-trained models

## 🆘 Support

For issues and questions:
1. Check this setup guide and troubleshooting section
2. Review the main README.md
3. Check project Issues on GitHub
4. Join the WeChat discussion group (QR code in main README)

---

*This guide was created based on successful setup and testing with EgoDex data on a 4-GPU system.*
