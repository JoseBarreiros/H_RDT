# Correct Command Format for Scaling Experiments

## Issue Found

The original commands in the documentation were missing required parameters and using incorrect file paths.

## ✅ Correct Command Format

```bash
accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed configs/zero1.json \
    --config_path configs/hrdt_pretrain.yaml \
    --output_dir ./checkpoints/scaling_p10 \
    --train_batch_size 32 \
    --sample_batch_size 32 \
    --max_train_steps 100000 \
    --learning_rate 1e-4 \
    --data_percentage 0.1 \
    --seed 42 \
    --checkpointing_period 10000 \
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

## Key Changes

1. **Use `main.py`** instead of `train/train.py`
2. **Add `--pretrained_vision_encoder_name_or_path="dino-siglip"`** (required)
3. **Use `configs/zero1.json`** instead of `configs/deepspeed_config.json`
4. **Include all parameters** from `pretrain.sh` for full compatibility

## Original Issues

### ❌ Incorrect Commands (Old)

```bash
# Missing --pretrained_vision_encoder_name_or_path
accelerate launch train/train.py \
    --config_path configs/hrdt_pretrain.yaml \
    --deepspeed configs/deepspeed_config.json \  # Wrong file
    --data_percentage 0.1 \
    ...
```

**Problems:**
1. Missing `--pretrained_vision_encoder_name_or_path="dino-siglip"` → Causes `KeyError: None`
2. Using `configs/deepspeed_config.json` → File doesn't exist (should be `zero1.json` or `zero2.json`)
3. Using `train/train.py` instead of `main.py` → Wrong entry point

## Quick Reference

Based on `pretrain.sh`, here's the template for scaling experiments:

```bash
# Key differences from full training
--data_percentage 0.1 \        # Add this for scaling
--seed 42 \                     # Add this for reproducibility
--max_train_steps 100000 \      # Adjust as needed

# Everything else matches pretrain.sh
```

## Files Updated

- ✅ `SCALING_LAW_GUIDE.md` - Updated command examples
- ✅ `README.md` - Updated scaling law section
- ✅ `run_scaling_law_experiments.sh` - Fixed automated script
- ✅ `CORRECT_COMMAND_FORMAT.md` - This file with full details

