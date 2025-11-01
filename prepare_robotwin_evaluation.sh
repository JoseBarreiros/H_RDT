#!/bin/bash
# prepare_robotwin_evaluation.sh
# Prepares checkpoint for RobotWin evaluation
# 
# This script handles PER-CHECKPOINT steps:
# - Copies checkpoint files
# - Cleans config.json
#
# ONE-TIME setup (dependencies, assets, code) should be done separately.
# See ROBOTWIN_EVALUATION_README.md for details.

set -e

# Configuration
CHECKPOINT_NAME="${1:-table8_checkpoint30}"
SOURCE_CHECKPOINT="${2:-~/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30}"
ROBOTWIN_BASE="${ROBOTWIN_BASE:-$HOME/RoboTwin}"

# Expand tildes
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT/#\~/$HOME}"
ROBOTWIN_BASE="${ROBOTWIN_BASE/#\~/$HOME}"

echo "=========================================="
echo "Preparing Checkpoint for RobotWin Evaluation"
echo "=========================================="
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Source checkpoint: $SOURCE_CHECKPOINT"
echo "RobotWin base: $ROBOTWIN_BASE"
echo ""

# Check if source checkpoint exists
if [ ! -d "$SOURCE_CHECKPOINT" ]; then
    echo "Error: Source checkpoint not found: $SOURCE_CHECKPOINT"
    exit 1
fi

# Check if RobotWin exists
if [ ! -d "$ROBOTWIN_BASE" ]; then
    echo "Error: RobotWin base directory not found: $ROBOTWIN_BASE"
    exit 1
fi

# Check if consolidated checkpoint exists
CONSOLIDATED_CHECKPOINT="${SOURCE_CHECKPOINT}/pytorch_model_consolidated.bin"
if [ ! -f "$CONSOLIDATED_CHECKPOINT" ]; then
    echo "⚠️  Warning: Consolidated checkpoint not found: $CONSOLIDATED_CHECKPOINT"
    echo "Please run convert_deepspeed_checkpoint.sh first"
    exit 1
fi

# Create directories
TARGET_CHECKPOINT_DIR="$ROBOTWIN_BASE/policy/H-RDT/checkpoints/$CHECKPOINT_NAME"
mkdir -p "$TARGET_CHECKPOINT_DIR"
mkdir -p "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings"

echo "1. Copying checkpoint files..."
cp "$CONSOLIDATED_CHECKPOINT" "$TARGET_CHECKPOINT_DIR/pytorch_model.bin"
cp "$SOURCE_CHECKPOINT/config.json" "$TARGET_CHECKPOINT_DIR/config.json"

# Remove pretrained_backbone_path from config
echo "2. Cleaning config.json..."
python3 << EOF
import json
import sys
config_path = "$TARGET_CHECKPOINT_DIR/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
if 'pretrained_backbone_path' in config:
    del config['pretrained_backbone_path']
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("   Removed pretrained_backbone_path from config")
else:
    print("   Config already clean")
EOF

echo "3. Checking one-time setup files..."
# Note: These are one-time setup files, but we check/copy them if missing
# to make the script more robust

H_RDT_ROOT="$(cd "$(dirname "$SOURCE_CHECKPOINT")/../.." && pwd)"

# Check if code structure exists, copy if missing
if [ ! -f "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.py" ]; then
    echo "   ⚠️  Code structure missing, copying now (one-time setup)"
    cp -r "$H_RDT_ROOT/inference/robotwin2_example/H-RDT/"* \
          "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/" 2>/dev/null || true
else
    echo "   ✅ Code structure already exists (one-time setup done)"
fi

# Check language embeddings
if [ ! -d "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings" ] || \
   [ $(ls "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/"*.pt 2>/dev/null | wc -l) -lt 49 ]; then
    echo "   ⚠️  Language embeddings missing, copying now (one-time setup)"
    mkdir -p "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings"
    if [ -d "$H_RDT_ROOT/datasets/robotwin2/lang_embeddings" ]; then
        cp "$H_RDT_ROOT/datasets/robotwin2/lang_embeddings/"*.pt \
           "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/" 2>/dev/null || true
    fi
else
    echo "   ✅ Language embeddings already exist (one-time setup done)"
fi

# Check configs
if [ ! -f "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/hrdt.yaml" ]; then
    echo "   ⚠️  Config missing, copying now (one-time setup)"
    if [ -f "$H_RDT_ROOT/configs/hrdt_finetune.yaml" ]; then
        cp "$H_RDT_ROOT/configs/hrdt_finetune.yaml" \
           "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/hrdt.yaml"
    fi
fi

if [ ! -f "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/stats.json" ]; then
    echo "   ⚠️  Stats missing, copying now (one-time setup)"
    if [ -f "$H_RDT_ROOT/datasets/robotwin2/stats.json" ]; then
        cp "$H_RDT_ROOT/datasets/robotwin2/stats.json" \
           "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/stats.json"
    fi
fi

# Check vision encoder models
if [ ! -d "$ROBOTWIN_BASE/policy/H-RDT/bak" ]; then
    echo "   ⚠️  Vision encoder models missing, copying now (one-time setup)"
    if [ -d "$H_RDT_ROOT/bak" ]; then
        cp -r "$H_RDT_ROOT/bak" \
              "$ROBOTWIN_BASE/policy/H-RDT/" 2>/dev/null || true
    fi
else
    echo "   ✅ Vision encoder models already exist (one-time setup done)"
fi

# Check policy module structure
if [ ! -f "$ROBOTWIN_BASE/policy/H-RDT/__init__.py" ]; then
    echo "   ⚠️  Policy module missing, setting up now (one-time setup)"
    mkdir -p "$ROBOTWIN_BASE/policy/H-RDT"
    cp "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.py" \
       "$ROBOTWIN_BASE/policy/H-RDT/deploy_policy.py" 2>/dev/null || true

    cat > "$ROBOTWIN_BASE/policy/H-RDT/__init__.py" << 'EOF'
# H-RDT Policy Module
from .deploy_policy import get_model, eval, reset_model, encode_obs

__all__ = ['get_model', 'eval', 'reset_model', 'encode_obs']
EOF

    cp "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.yml" \
       "$ROBOTWIN_BASE/policy/H-RDT/deploy_policy.yml" 2>/dev/null || true
else
    echo "   ✅ Policy module already exists (one-time setup done)"
fi

echo ""
echo "Note: One-time setup files checked/copied if missing."
echo "      For faster repeated checkpoints, ensure one-time setup is complete."
echo ""

echo "4. Per-checkpoint: Setting up checkpoint..."
mkdir -p "$ROBOTWIN_BASE/policy/H-RDT"
cp "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.py" \
   "$ROBOTWIN_BASE/policy/H-RDT/deploy_policy.py" 2>/dev/null || true

cat > "$ROBOTWIN_BASE/policy/H-RDT/__init__.py" << 'EOF'
# H-RDT Policy Module
from .deploy_policy import get_model, eval, reset_model, encode_obs

__all__ = ['get_model', 'eval', 'reset_model', 'encode_obs']
EOF

cp "$ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT/deploy_policy.yml" \
   "$ROBOTWIN_BASE/policy/H-RDT/deploy_policy.yml" 2>/dev/null || true

echo ""
echo "✅ Checkpoint preparation complete!"
echo ""
echo "Checkpoint location: $TARGET_CHECKPOINT_DIR"
echo ""
echo "Next steps:"
echo "  1. Update eval.sh: ckpt_setting=\"checkpoints/$CHECKPOINT_NAME\""
echo "  2. Run evaluation: cd $ROBOTWIN_BASE/policy/H-RDT/inference/robotwin2_example/H-RDT && bash eval.sh"
echo ""
echo "Note: One-time setup (dependencies, assets, code) should be done separately."
echo "      See ROBOTWIN_EVALUATION_README.md for details."

