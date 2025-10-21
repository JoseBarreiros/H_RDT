#!/bin/bash

# H-RDT Fine-tuning Script for Single GPU
# This script is adapted for single GPU training

# Disable NCCL for single GPU
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0

# Set up environment
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

# Project settings
export WANDB_PROJECT="hrdt-robotwin2"
export OUTPUT_DIR="./checkpoints/robotwin2"

export VISION_ENCODER_NAME="dino-siglip"

# Create output directory
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Single GPU training with accelerate
accelerate launch main.py \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --config_path="configs/hrdt_finetune.yaml" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=8 \
    --max_train_steps=10000 \
    --checkpointing_period=1000 \
    --sample_period=100 \
    --checkpoints_total_limit=5 \
    --lr_scheduler="constant_with_warmup" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --dataset_type="finetune" \
    --upsample_rate=3 \
    --image_aug \
    --gradient_checkpointing \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="finetune" \
    --pretrained_backbone_path="./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin"

echo "Fine-tuning completed!"
