#!/bin/bash
# prepare_checkpoint_for_robotwin.sh
# Prepares your checkpoint for RobotWin simulation evaluation

set -e

CHECKPOINT_PATH="checkpoints/table8_finetune_pretrain0618/checkpoint-30"
ROBOTWIN_BASE="${ROBOTWIN_BASE:-$HOME/RoboTwin}"  # Default to ~/RoboTwin, can override with env var

# Expand tilde if present
ROBOTWIN_BASE="${ROBOTWIN_BASE/#\~/$HOME}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "Preparing checkpoint for RobotWin evaluation..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "RobotWin base: $ROBOTWIN_BASE"

# Check if RobotWin path exists
if [ ! -d "$ROBOTWIN_BASE" ]; then
    echo "⚠️  Warning: RobotWin base path not found: $ROBOTWIN_BASE"
    echo ""
    echo "Please either:"
    echo "  1. Set ROBOTWIN_BASE environment variable:"
    echo "     export ROBOTWIN_BASE=/path/to/RoboTwin"
    echo "     bash prepare_checkpoint_for_robotwin.sh"
    echo ""
    echo "  2. Or update ROBOTWIN_BASE in this script directly"
    echo ""
    echo "Searching for RobotWin installation..."
    find "$HOME" -maxdepth 3 -type d -name "RoboTwin" 2>/dev/null | head -3
    exit 1
fi

H_RDT_IN_ROBOTWIN="$ROBOTWIN_BASE/policy/H-RDT"
CHECKPOINT_NAME="table8_checkpoint30"
TARGET_CHECKPOINT_DIR="$H_RDT_IN_ROBOTWIN/checkpoints/$CHECKPOINT_NAME"

echo ""
echo "1. Copying checkpoint files..."
mkdir -p "$TARGET_CHECKPOINT_DIR"
cp "$CHECKPOINT_PATH/config.json" "$TARGET_CHECKPOINT_DIR/"
cp "$CHECKPOINT_PATH/pytorch_model.bin" "$TARGET_CHECKPOINT_DIR/"
echo "✅ Checkpoint copied to: $TARGET_CHECKPOINT_DIR"

echo ""
echo "2. Verifying language embeddings..."
LANG_EMBED_DIR="$H_RDT_IN_ROBOTWIN/inference/robotwin2_example/H-RDT/utils/lang_embeddings"
if [ -d "$LANG_EMBED_DIR" ]; then
    echo "✅ Language embeddings directory exists"
    echo "   Location: $LANG_EMBED_DIR"
else
    echo "⚠️  Warning: Language embeddings directory not found"
    echo "   Expected: $LANG_EMBED_DIR"
    echo "   Copy from: datasets/robotwin2/lang_embeddings/"
fi

echo ""
echo "3. Next steps:"
echo "   a) Edit eval.sh in RobotWin:"
echo "      cd $H_RDT_IN_ROBOTWIN/inference/robotwin2_example/H-RDT"
echo "      # Set: ckpt_setting=\"checkpoints/$CHECKPOINT_NAME\""
echo "      # Set: task_name=\"grab_roller\" (or other task)"
echo ""
echo "   b) Run evaluation:"
echo "      bash eval.sh"
echo ""
echo "✅ Checkpoint preparation complete!"

