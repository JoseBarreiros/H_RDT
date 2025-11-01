#!/bin/bash

policy_name="H-RDT"
task_name="handover_mic"
task_config="demo_clean"  # demo_randomized or demo_clean
ckpt_setting=""  # e.g. checkpoints/table8_handover_mic/checkpoint-10000
seed="42"
gpu_id="0"
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

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
    # [TODO] add parameters here
