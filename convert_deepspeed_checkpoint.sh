#!/bin/bash
# convert_deepspeed_checkpoint.sh
# Converts DeepSpeed ZeRO-3 checkpoint to consolidated format for evaluation

set -e

# Configuration
CHECKPOINT_DIR="${1:-~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30}"
OUTPUT_FILE="${2:-}"

# Expand tilde if present
CHECKPOINT_DIR="${CHECKPOINT_DIR/#\~/$HOME}"
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="${CHECKPOINT_DIR}/pytorch_model_consolidated.bin"
fi
OUTPUT_FILE="${OUTPUT_FILE/#\~/$HOME}"

echo "=========================================="
echo "DeepSpeed Checkpoint Conversion"
echo "=========================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output file: $OUTPUT_FILE"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Check if pytorch_model directory exists (DeepSpeed format)
if [ ! -d "$CHECKPOINT_DIR/pytorch_model" ]; then
    echo "Error: DeepSpeed checkpoint not found. Expected: $CHECKPOINT_DIR/pytorch_model/"
    echo "This script is for converting DeepSpeed ZeRO-3 checkpoints."
    exit 1
fi

# Check if latest file exists
if [ ! -f "$CHECKPOINT_DIR/latest" ]; then
    echo "Creating 'latest' file pointing to pytorch_model..."
    echo "pytorch_model" > "$CHECKPOINT_DIR/latest"
fi

# Activate environment
echo "Activating H-RDT environment..."
source ~/.config/hrdt/activate.sh

# Check if zero_to_fp32.py exists
ZERO_TO_FP32_SCRIPT="$CHECKPOINT_DIR/zero_to_fp32.py"
if [ ! -f "$ZERO_TO_FP32_SCRIPT" ]; then
    echo "Error: zero_to_fp32.py not found in checkpoint directory"
    echo "Expected: $ZERO_TO_FP32_SCRIPT"
    exit 1
fi

# Run conversion
echo ""
echo "Starting conversion..."
echo "This may take several minutes (~24GB checkpoint)..."
echo ""

python "$ZERO_TO_FP32_SCRIPT" \
    "$CHECKPOINT_DIR" \
    "$OUTPUT_FILE"

# Verify output
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "✅ Conversion successful!"
    echo "Output file: $OUTPUT_FILE"
    echo "File size: $FILE_SIZE"
    echo ""
    echo "Next steps:"
    echo "  1. Copy to RobotWin: cp $OUTPUT_FILE ~/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/pytorch_model.bin"
    echo "  2. Run evaluation: cd ~/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT && bash eval.sh"
else
    echo "❌ Error: Output file not created"
    exit 1
fi

