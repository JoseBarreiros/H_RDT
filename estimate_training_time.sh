#!/bin/bash
# Quick script to estimate training time

echo "Training Time Estimation for 500K steps on 4x A100"
echo "=================================================="
echo ""

# Configuration
TOTAL_STEPS=500000
BATCH_SIZE_PER_GPU=32
NUM_GPUS=4

# Estimate steps per second (conservative to optimistic)
echo "Estimated Throughput: 0.8 - 2.0 steps/second"
echo ""

# Conservative estimate
CONSERVATIVE_SPS=1.0
CONSERVATIVE_TIME=$((500000 / CONSERVATIVE_SPS))
CONSERVATIVE_HOURS=$((CONSERVATIVE_TIME / 3600))
CONSERVATIVE_DAYS=$(echo "scale=1; $CONSERVATIVE_HOURS / 24" | bc)

# Optimistic estimate
OPTIMISTIC_SPS=2.0
OPTIMISTIC_TIME=$((500000 / OPTIMISTIC_SPS))
OPTIMISTIC_HOURS=$((OPTIMISTIC_TIME / 3600))
OPTIMISTIC_DAYS=$(echo "scale=1; $OPTIMISTIC_HOURS / 24" | bc)

echo "At 1.0 steps/sec:   $CONSERVATIVE_HOURS hours (~$CONSERVATIVE_DAYS days)"
echo "At 2.0 steps/sec:   $OPTIMISTIC_HOURS hours (~$OPTIMISTIC_DAYS days)"
echo ""
echo "Recommended: Monitor actual throughput in first 100 steps"
echo "Then use: actual_sps * 500000 = total_seconds"
