#!/bin/bash

# Quick test script to verify training works with data_percentage

echo "Testing training with 1% of EgoDex data..."
echo ""

source hrdt_env/bin/activate

# Run a very short training test
accelerate launch train/train.py \
    --config_path configs/hrdt_pretrain.yaml \
    --deepspeed configs/deepspeed_config.json \
    --output_dir ./checkpoints/test_scaling_1pct \
    --train_batch_size 4 \
    --sample_batch_size 2 \
    --max_train_steps 10 \
    --data_percentage 0.01 \
    --seed 42 \
    --checkpointing_period 100000 \
    --precomp_lang_embed \
    --mixed_precision bf16 \
    --report_to "none"

echo ""
echo "✅ Training test completed!"
echo "Check ./checkpoints/test_scaling_1pct for results"

