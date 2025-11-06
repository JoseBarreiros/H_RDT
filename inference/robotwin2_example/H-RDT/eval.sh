#!/bin/bash

# Usage: ./eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [seed]
# Example: ./eval.sh handover_mic demo_clean checkpoints/table8_handover_mic/checkpoint-10000 0
# Example: ./eval.sh handover_mic demo_clean checkpoints/table8_handover_mic/checkpoint-10000 0 42

if [ $# -lt 4 ]; then
    echo "Usage: $0 <task_name> <task_config> <ckpt_setting> <gpu_id> [seed]"
    echo ""
    echo "Arguments:"
    echo "  task_name      : Task name (e.g., handover_mic, grab_roller)"
    echo "  task_config    : Task config (demo_clean or demo_randomized)"
    echo "  ckpt_setting   : Checkpoint path (e.g., checkpoints/table8_handover_mic/checkpoint-10000)"
    echo "  gpu_id         : GPU ID to use (e.g., 0)"
    echo "  seed           : Random seed (optional, default: 42)"
    echo ""
    echo "Examples:"
    echo "  $0 handover_mic demo_clean checkpoints/table8_handover_mic/checkpoint-10000 0"
    echo "  $0 grab_roller demo_randomized checkpoints/table8_grab_roller/checkpoint-10000 1 42"
    exit 1
fi

policy_name="H-RDT"
task_name="$1"
task_config="$2"
ckpt_setting="$3"
gpu_id="$4"
seed="${5:-42}"  # Default to 42 if not provided

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mGPU ID (to use): ${gpu_id}\033[0m"
echo -e "\033[33mTask: ${task_name}, Config: ${task_config}, Checkpoint: ${ckpt_setting}, Seed: ${seed}\033[0m"

# Calculate RoboTwin root directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$ROBOTWIN_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
