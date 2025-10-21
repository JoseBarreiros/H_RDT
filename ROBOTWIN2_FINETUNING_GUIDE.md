# H-RDT RobotWin2 Fine-tuning Guide

This guide will help you set up and run fine-tuning of H-RDT on the RobotWin2 dataset.

## Prerequisites

### 1. Dataset Requirements

You need to obtain the RobotWin2 dataset. The dataset should have the following structure:

```
robotwin2_dataset/
├── aloha-agilex/
│   ├── data/
│   │   ├── task1/
│   │   │   ├── episode_000000.hdf5
│   │   │   ├── episode_000001.hdf5
│   │   │   └── ...
│   │   ├── task2/
│   │   └── ...
│   └── ...
```

### 2. T5 Model

You need the T5-v1_1-xxl model for language encoding. Download it from HuggingFace:

```bash
# Download T5 model
huggingface-cli download google/t5-v1_1-xxl --local-dir ./t5-v1_1-xxl
```

## Setup Process

### Step 1: Set Environment Variables

```bash
# Set the path to your RobotWin2 dataset
export ROBOTWIN2_DATA_ROOT="/path/to/your/robotwin2/dataset/aloha-agilex"

# Set the path to your T5 model
export T5_MODEL_PATH="/path/to/your/t5-v1_1-xxl"

# Run the setup script
source setup_robotwin2_finetune.sh
```

### Step 2: Activate Virtual Environment

```bash
source hrdt_env/bin/activate
```

### Step 3: Run Fine-tuning

```bash
# For single GPU training
bash finetune_single_gpu.sh

# Or for multi-GPU training (if you have multiple GPUs)
bash finetune.sh
```

## Configuration Details

### Model Configuration

The fine-tuning uses the following key configurations:

- **Action Dimension**: 14 (RobotWin2) vs 48 (pretrained)
- **State Dimension**: 14 (RobotWin2) vs 48 (pretrained)
- **Number of Cameras**: 3 (RobotWin2) vs 1 (pretrained)
- **Image Patches**: 196 (RobotWin2) vs 729 (pretrained)

### Training Parameters

- **Batch Size**: 8 (single GPU) / 32 (multi-GPU)
- **Learning Rate**: 1e-4
- **Max Steps**: 10,000 (single GPU) / 1,000,000 (multi-GPU)
- **Checkpointing**: Every 1,000 steps
- **Mixed Precision**: bfloat16

## Expected Output

The fine-tuning process will:

1. Load the pretrained H-RDT model (48D actions)
2. Initialize new action encoders/decoders for 14D actions
3. Keep the shared backbone (vision, language, transformer)
4. Train on RobotWin2 data
5. Save checkpoints to `./checkpoints/robotwin2/`

## Monitoring Training

### Using Weights & Biases

```bash
# Login to wandb (optional)
wandb login

# Training will automatically log to wandb
```

### Check Training Progress

```bash
# Check checkpoint directory
ls -la checkpoints/robotwin2/

# View training logs
tail -f checkpoints/robotwin2/training.log
```

## Evaluation

After fine-tuning, you can evaluate the model using the provided evaluation scripts:

```bash
# The fine-tuned model will be in checkpoints/robotwin2/
# Use it with the RobotWin2 evaluation scripts
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `finetune_single_gpu.sh`
2. **Dataset Not Found**: Check `ROBOTWIN2_DATA_ROOT` path
3. **T5 Model Not Found**: Check `T5_MODEL_PATH` path
4. **Permission Denied**: Make scripts executable: `chmod +x *.sh`

### Memory Requirements

- **Minimum**: 16GB GPU memory for single GPU training
- **Recommended**: 24GB+ GPU memory or multiple GPUs
- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ recommended

## Next Steps

After successful fine-tuning:

1. Test the fine-tuned model on RobotWin2 tasks
2. Compare performance with the paper results
3. Experiment with different hyperparameters if needed
4. Use the fine-tuned model for your specific robot tasks


