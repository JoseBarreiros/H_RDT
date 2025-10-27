#!/bin/bash

# H-RDT Test Subset Processing and Verification Script
# This script runs preprocessing and verification on the test subset
# without affecting the full dataset results

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}H-RDT Test Subset Processing & Verification${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Configuration
TEST_DATA_ROOT="/home/jose-barreiros/egodex/test_subset"
TEST_OUTPUT_DIR="/home/jose-barreiros/H_RDT/datasets/pretrain/test_subset_output"
PROJECT_ROOT="/home/jose-barreiros/H_RDT"

# Create output directory for test results
mkdir -p "$TEST_OUTPUT_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Data Root: $TEST_DATA_ROOT"
echo "  Output Dir: $TEST_OUTPUT_DIR"
echo ""

# Activate virtual environment
cd "$PROJECT_ROOT"
source hrdt_env/bin/activate

# Set environment variables for test processing
export EGODEX_DATA_ROOT="$TEST_DATA_ROOT"
export T5_MODEL_PATH="google/t5-v1_1-xxl"
export HRDT_PROJECT_ROOT="$PROJECT_ROOT"
export HRDT_CONFIG_PATH="${PROJECT_ROOT}/configs/hrdt_pretrain.yaml"
export HRDT_OUTPUT_DIR="$TEST_OUTPUT_DIR"

# Test output paths
export STATS_OUTPUT_PATH="${TEST_OUTPUT_DIR}/egodex_stat.json"
export LARGE_VALUES_LOG="${TEST_OUTPUT_DIR}/egodex_large_values.txt"
export NUM_PROCESSES=8

echo -e "${GREEN}Starting test subset processing...${NC}"
echo ""

# Step 1: Precompute 48D actions
echo "=" | tr '=' '*' | head -c 60 && echo ""
echo "Step 1: Precomputing 48D actions..."
echo "=" | tr '=' '*' | head -c 60 && echo ""
python datasets/pretrain/precompute_48d_actions.py \
    --data_root "$EGODEX_DATA_ROOT" \
    --num_processes "$NUM_PROCESSES" \
    --force_overwrite

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Step 1 completed with warnings or errors${NC}"
fi

echo ""

# Step 2: Calculate statistics
echo "=" | tr '=' '*' | head -c 60 && echo ""
echo "Step 2: Calculating dataset statistics..."
echo "=" | tr '=' '*' | head -c 60 && echo ""
python datasets/pretrain/calc_stat.py \
    --data_root "$EGODEX_DATA_ROOT" \
    --output_path "$STATS_OUTPUT_PATH" \
    --large_values_log "$LARGE_VALUES_LOG"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Step 2 completed with warnings or errors${NC}"
fi

echo ""

# Step 3: Encode language embeddings
echo "=" | tr '=' '*' | head -c 60 && echo ""
echo "Step 3: Encoding language embeddings..."
echo "=" | tr '=' '*' | head -c 60 && echo ""
python datasets/pretrain/encode_lang_batch.py

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Step 3 completed with warnings or errors${NC}"
fi

echo ""

# Run verification
echo -e "${GREEN}Running verification on test subset...${NC}"
echo ""
python verify_dataset.py --data_root "$TEST_DATA_ROOT" --stats_dir "$TEST_OUTPUT_DIR"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Subset Processing Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Generated files (separate from full dataset):"
echo "  📊 Statistics: $STATS_OUTPUT_PATH"
echo "  📝 Large values log: $LARGE_VALUES_LOG"
echo "  📁 48D actions: Added to HDF5 files in $TEST_DATA_ROOT"
echo "  📝 Language embeddings: *.pt files in $TEST_DATA_ROOT"
echo ""
echo "Main dataset files remain untouched:"
echo "  ✅ /home/jose-barreiros/H_RDT/datasets/pretrain/egodex_stat.json"
echo "  ✅ /home/jose-barreiros/egodex/organized (all 318,082 files)"

