# Scaling Law Experiments with EgoDex

This guide explains how to run scaling law experiments to study the relationship between dataset size and model performance.

## 🎯 Overview

Scaling law experiments train models with different percentages of the full dataset to understand:
- **Data efficiency**: How well the model performs with limited data
- **Scaling behavior**: The relationship between dataset size and performance
- **Diminishing returns**: At what point adding more data provides minimal benefit

## 📊 Typical Experiment Design

Train models with the following data percentages:
- 1% (~3,181 files)
- 5% (~15,904 files)
- 10% (~31,808 files)
- 25% (~79,521 files)
- 50% (~159,041 files)
- 75% (~238,562 files)
- 100% (~318,082 files)

## 🚀 Running Experiments

### Quick Start: Automated Script

Run all experiments automatically:

```bash
./run_scaling_law_experiments.sh
```

This will:
1. Train models at each percentage level
2. Save results to separate directories
3. Use fixed seed (42) for reproducibility
4. Produce checkpoints every 10k steps

### Manual Method: Single Percentage

Train with a specific percentage:

```bash
source hrdt_env/bin/activate

accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed configs/zero1.json \
    --config_path configs/hrdt_pretrain.yaml \
    --output_dir ./checkpoints/scaling_p10 \
    --train_batch_size 32 \
    --sample_batch_size 32 \
    --max_train_steps 1000000 \
    --learning_rate 1e-4 \
    --data_percentage 0.1 \
    --seed 42 \
    --checkpointing_period 5000 \
    --checkpoints_total_limit 40 \
    --sample_period 500 \
    --lr_scheduler="constant_with_warmup" \
    --mixed_precision bf16 \
    --dataloader_num_workers 32 \
    --dataset_type pretrain \
    --report_to wandb \
    --upsample_rate 3 \
    --image_aug \
    --gradient_checkpointing \
    --precomp_lang_embed \
    --training_mode lang \
    --mode pretrain
```

### Example: Training with 10% of Data

```bash
# 10% = 0.1 - Minimal command for quick testing
accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed configs/zero1.json \
    --config_path configs/hrdt_pretrain.yaml \
    --output_dir ./checkpoints/scaling_p10 \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --data_percentage 0.1 \
    --seed 42 \
    --precomp_lang_embed \
    --mixed_precision bf16
```

## Upload the checkpoint to GCS

```bash
gsutil -m cp -r ./checkpoints/scaling_p10/checkpoint-20000 gs://cortex-hrdt/scaling_p10
```

## 📈 Understanding the Results

### Subsampling Behavior

When you set `--data_percentage 0.1`:
- The dataset loader randomly samples 10% of all files
- Uses a fixed seed (default: 42) for reproducibility
- Same files are sampled each time with the same seed
- Works before data loading, so it's efficient

### Training Characteristics

Expected behavior as percentage increases:

| Data % | # Files | Training Time | Expected Performance |
|--------|---------|---------------|---------------------|
| 1%     | ~3K     | Fast          | Lower               |
| 5%     | ~16K    | Medium        | Improving           |
| 10%    | ~32K    | Medium        | Noticeable gains    |
| 25%    | ~80K    | Long          | Significant gains   |
| 50%    | ~159K   | Very Long     | Strong performance  |
| 100%   | ~318K   | Full Time     | Best performance    |

## 🔬 Experimental Tips

### 1. Reproducibility

Always use the same seed across experiments:
```bash
--seed 42  # Fixed seed ensures consistent subsampling
```

### 2. Efficient Experiment Design

- **Start with smaller percentages** to validate setup
- **Use shorter training** for initial experiments:
  ```bash
  --max_train_steps 1000  # Quick test
  ```
- **Scale up** after confirming results

### 3. Correct Command Format

**⚠️ Important:** Always use this format based on `pretrain.sh`:

- Use `main.py` (not `train/train.py`)
- Include `--pretrained_vision_encoder_name_or_path="dino-siglip"`  
- Use `configs/zero1.json` (not `configs/deepspeed_config.json`)
- Include all parameters from `pretrain.sh` for compatibility

See [CORRECT_COMMAND_FORMAT.md](CORRECT_COMMAND_FORMAT.md) for details.

### 3. Validation Set

The validation set always uses 100% of test data (not subsampled):
- This ensures fair comparison across experiments
- Validation performance reflects true model capability

## 📊 Analysis and Plotting

After running experiments, analyze:

### 1. Dataset Sizes
```bash
# Check actual file counts for each percentage
cat scaling_experiments/p10/logs/*.txt | grep "Loaded.*data files"
```

### 2. Training Progress
Compare loss curves across percentages to see:
- How quickly models converge with different data sizes
- Whether more data improves generalization

### 3. Create Scaling Plots

Plot data size vs performance:
- X-axis: Number of training files (or percentage)
- Y-axis: Success rate / performance metric
- Identify the "sweet spot" for data efficiency

## 🎛️ Advanced Usage

### Custom Seed

For different random subsamples:
```bash
--seed 123  # Different seed = different random subset
```

### Training with Fixed Subset

After initial exploration, train on a fixed small percentage to save compute:

```bash
# Train on 10% for faster iteration
accelerate launch train/train.py \
    --data_percentage 0.1 \
    --seed 42 \
    # ... other args
```

## 💡 Use Cases

1. **Data Efficiency Studies**: Find minimum data needed for good performance
2. **Budget Constraints**: Determine if fewer examples work
3. **Progressive Training**: Start small, add more data as needed
4. **Performance Scaling**: Understand model scaling with data size
5. **Publication-Ready Figures**: Generate scaling law plots for papers

## 📝 Notes

- **Seed matters**: Same seed = same random subset
- **Fixed seed recommended**: Ensures fair comparison
- **Validation unaffected**: Always uses full test set
- **Memory efficient**: Subsampling happens once during initialization
- **Fast iteration**: Test ideas quickly with small percentages

## 🔄 Combining with Test Subset

You can also test on a small subset first:

```bash
# Use test subset for fast iteration
export EGODEX_DATA_ROOT="/home/jose-barreiros/egodex/test_subset"

# Then run scaling experiments
./run_scaling_law_experiments.sh
```

This allows you to:
1. Test your code quickly on small data
2. Validate preprocessing is working
3. Then scale up to full dataset percentages

