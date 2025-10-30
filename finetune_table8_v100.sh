#!/bin/bash
# Fine-tuning script optimized for V100 GPUs (16GB) with ZeRO-3 CPU offload
# This config uses ZeRO-3 to shard parameters, gradients, and optimizer states
# CPU offload reduces GPU memory usage significantly

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3

export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="${CUTLASS_PATH:-/usr/local/cutlass}"

# Set PyTorch memory allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_PROJECT="h-rdt"
export ROBOTWIN2_DATA_ROOT="/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted"

# Activate environment
source ~/.config/hrdt/activate.sh

accelerate launch --num_processes 4 --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed="./configs/zero3_cpu_offload.json" \
    --config_path="configs/hrdt_finetune.yaml" \
    --output_dir="./checkpoints/table8_finetune_pretrain0618" \
    --train_batch_size=4 \
    --sample_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=10000 \
    --checkpointing_period=10 \
    --sample_period=100 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=500 \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --dataset_type="finetune" \
    --dataset_name="robotwin_agilex" \
    --report_to=wandb \
    --upsample_rate=3 \
    --image_aug \
    --gradient_checkpointing \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="finetune" \
    --pretrained_backbone_path="./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin"

