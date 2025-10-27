#!/bin/bash

# H-RDT Scaling Law Experiments
# Train models with different percentages of data to plot scaling curves

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}H-RDT Scaling Law Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
EXPERIMENT_ROOT="/home/jose-barreiros/H_RDT/scaling_experiments"
SEED=42
DATA_ROOT="/home/jose-barreiros/egodex/organized"

# Data percentages to test (for scaling law plots)
PERCENTAGES=(0.01 0.05 0.1 0.25 0.5 0.75 1.0)

echo -e "${YELLOW}Configuration:${NC}"
echo "  Data Root: $DATA_ROOT"
echo "  Experiment Root: $EXPERIMENT_ROOT"
echo "  Seed: $SEED"
echo "  Percentages to test: ${PERCENTAGES[@]}"
echo ""

# Activate environment
source /home/jose-barreiros/H_RDT/hrdt_env/bin/activate

# Create experiment directory
mkdir -p "$EXPERIMENT_ROOT"

# Run experiments for each percentage
for pct in "${PERCENTAGES[@]}"; do
    pct_str=$(printf "%.0f" $(echo "$pct * 100" | bc))
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running experiment: ${pct_str}% of data${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    OUTPUT_DIR="${EXPERIMENT_ROOT}/p${pct_str}"
    mkdir -p "$OUTPUT_DIR"
    
    # Run training with this percentage
    accelerate launch --main_process_port 29500 main.py \
        --pretrained_vision_encoder_name_or_path="dino-siglip" \
        --deepspeed configs/zero1.json \
        --config_path configs/hrdt_pretrain.yaml \
        --output_dir "$OUTPUT_DIR" \
        --train_batch_size 32 \
        --sample_batch_size 32 \
        --max_train_steps 100000 \
        --learning_rate 1e-4 \
        --data_percentage "$pct" \
        --seed "$SEED" \
        --checkpointing_period 5000 \
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
    
    echo ""
    echo -e "${YELLOW}Completed experiment for ${pct_str}%${NC}"
    echo -e "${YELLOW}Results saved to: $OUTPUT_DIR${NC}"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All Scaling Law Experiments Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Experiment results saved in: $EXPERIMENT_ROOT"
echo ""
echo "To analyze results:"
echo "  - Compare model performance across different data percentages"
echo "  - Plot scaling curve: data size vs success rate"
echo "  - Use this to understand data efficiency and scaling behavior"

