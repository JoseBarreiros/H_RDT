#!/bin/bash

# H-RDT RobotWin2 Fine-tuning Setup
# Configure paths for RobotWin2 fine-tuning

# Get the project root directory
export HRDT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

echo "H-RDT RobotWin2 Fine-tuning Setup"
echo "=================================="
echo "Project Root: $HRDT_PROJECT_ROOT"

# Check if RobotWin2 dataset path is provided
if [ -z "$ROBOTWIN2_DATA_ROOT" ]; then
    echo ""
    echo "⚠️  ROBOTWIN2_DATA_ROOT is not set!"
    echo "Please set the path to your RobotWin2 dataset:"
    echo "export ROBOTWIN2_DATA_ROOT=\"/path/to/your/robotwin2/dataset\""
    echo ""
    echo "For example:"
    echo "export ROBOTWIN2_DATA_ROOT=\"/home/user/datasets/robotwin2/aloha-agilex\""
    echo ""
    echo "Then run this script again:"
    echo "source setup_robotwin2_finetune.sh"
    echo ""
    exit 1
fi

# Check if T5 model path is provided
if [ -z "$T5_MODEL_PATH" ]; then
    echo ""
    echo "⚠️  T5_MODEL_PATH is not set!"
    echo "Please set the path to your T5 model:"
    echo "export T5_MODEL_PATH=\"/path/to/your/t5-v1_1-xxl\""
    echo ""
    echo "For example:"
    echo "export T5_MODEL_PATH=\"/home/user/models/t5-v1_1-xxl\""
    echo ""
    echo "Then run this script again:"
    echo "source setup_robotwin2_finetune.sh"
    echo ""
    exit 1
fi

# Verify paths exist
if [ ! -d "$ROBOTWIN2_DATA_ROOT" ]; then
    echo "❌ RobotWin2 dataset directory not found: $ROBOTWIN2_DATA_ROOT"
    exit 1
fi

if [ ! -d "$T5_MODEL_PATH" ]; then
    echo "❌ T5 model directory not found: $T5_MODEL_PATH"
    exit 1
fi

echo "✅ RobotWin2 dataset found: $ROBOTWIN2_DATA_ROOT"
echo "✅ T5 model found: $T5_MODEL_PATH"

# Set up environment variables
export HRDT_CONFIG_PATH="${HRDT_PROJECT_ROOT}/configs/hrdt_finetune.yaml"
export HRDT_OUTPUT_DIR="${HRDT_PROJECT_ROOT}/datasets/robotwin2"

# Processing parameters (reduced for single GPU)
export NUM_PROCESSES=8
export NUM_GPUS=1
export PROCESSES_PER_GPU=8

# Create output directory
mkdir -p "$HRDT_OUTPUT_DIR"

echo ""
echo "Environment variables set:"
echo "  ROBOTWIN2_DATA_ROOT: $ROBOTWIN2_DATA_ROOT"
echo "  T5_MODEL_PATH: $T5_MODEL_PATH"
echo "  HRDT_CONFIG_PATH: $HRDT_CONFIG_PATH"
echo "  HRDT_OUTPUT_DIR: $HRDT_OUTPUT_DIR"
echo "  NUM_PROCESSES: $NUM_PROCESSES"
echo ""

# Update the dataset configuration
echo "Updating dataset configuration..."

# Create a temporary Python script to update the dataset paths
python3 << EOF
import yaml
import os

# Load the dataset configuration
config_path = "${HRDT_PROJECT_ROOT}/datasets/robotwin2/setup_robotwin2.sh"

# Read the current setup script
with open(config_path, 'r') as f:
    content = f.read()

# Update the paths
content = content.replace('export ROBOTWIN2_DATA_ROOT="/share/hongzhe/datasets/robotwin2/dataset/aloha-agilex"', f'export ROBOTWIN2_DATA_ROOT="${ROBOTWIN2_DATA_ROOT}"')
content = content.replace('export T5_MODEL_PATH="/data/lingxuan/weights/t5-v1_1-xxl"', f'export T5_MODEL_PATH="${T5_MODEL_PATH}"')

# Write back the updated content
with open(config_path, 'w') as f:
    f.write(content)

print("✅ Dataset configuration updated")
EOF

echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source hrdt_env/bin/activate"
echo ""
echo "2. Run the fine-tuning:"
echo "   bash finetune_single_gpu.sh"
echo ""
echo "3. Monitor training with wandb (optional):"
echo "   wandb login"


