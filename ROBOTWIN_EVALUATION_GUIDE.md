# Evaluating Checkpoint in RobotWin Simulation

This guide explains how to evaluate your H-RDT checkpoint (`table8_finetune_pretrain0618/checkpoint-30`) in the RobotWin simulation environment.

## Prerequisites

1. **RobotWin environment installed** - You need access to the RobotWin simulation environment
2. **Checkpoint ready** - Your checkpoint at `checkpoints/table8_finetune_pretrain0618/checkpoint-30/`

## Setup Steps

### 1. Copy H-RDT to RobotWin Policy Directory

```bash
# Assuming RobotWin is installed at /path/to/RoboTwin
cp -r /home/jose-barreiros/H_RDT /path/to/RoboTwin/policy/
```

### 2. Copy Vision Encoder Models (bak folder)

```bash
# Copy the bak folder containing vision encoder models
cp -r /home/jose-barreiros/H_RDT/bak /path/to/RoboTwin/policy/H-RDT/
```

### 3. Copy Your Checkpoint

```bash
# Create checkpoint directory in RobotWin
mkdir -p /path/to/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/

# Copy checkpoint files
cp /home/jose-barreiros/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/config.json \
   /path/to/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/

cp /home/jose-barreiros/H_RDT/checkpoints/table8_finetune_pretrain0618/checkpoint-30/pytorch_model.bin \
   /path/to/RoboTwin/policy/H-RDT/checkpoints/table8_checkpoint30/
```

### 4. Verify Language Embeddings Exist

The RobotWin evaluation requires language embeddings for each task. Check that they exist:

```bash
ls /path/to/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/utils/lang_embeddings/
```

All 13 Table 8 tasks should have `.pt` files:
- `grab_roller.pt`
- `handover_mic.pt`
- `lift_pot.pt`
- `move_can_pot.pt`
- `open_laptop.pt`
- `pick_dual_bottles.pt`
- `place_dual_shoes.pt`
- `place_object_basket.pt`
- `place_phone_stand.pt`
- `put_bottles_dustbin.pt`
- `put_object_cabinet.pt`
- `stack_blocks_two.pt`
- `stack_bowls_two.pt`

### 5. Configure eval.sh

Edit `/path/to/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT/eval.sh`:

```bash
policy_name="H-RDT"
task_name="grab_roller"  # Change this for each task
task_config="demo_randomized"  # or "demo_clean" for Easy mode
ckpt_setting="checkpoints/table8_checkpoint30"  # Your checkpoint path
seed="42"
gpu_id="0"
```

**Important:** 
- `task_config="demo_randomized"` = **Hard mode** (default)
- `task_config="demo_clean"` = **Easy mode**
- Change `task_name` for each of the 13 tasks

### 6. Run Evaluation

```bash
cd /path/to/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT
bash eval.sh
```

## Evaluating All Table 8 Tasks

To evaluate all 13 tasks, create a loop script:

```bash
#!/bin/bash
# evaluate_all_table8_tasks.sh

cd /path/to/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

tasks=(
    "grab_roller"
    "handover_mic"
    "lift_pot"
    "move_can_pot"
    "open_laptop"
    "pick_dual_bottles"
    "place_dual_shoes"
    "place_object_basket"
    "place_phone_stand"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "stack_blocks_two"
    "stack_bowls_two"
)

for task in "${tasks[@]}"; do
    echo "Evaluating task: $task"
    
    # Update eval.sh task_name
    sed -i "s/task_name=\".*\"/task_name=\"$task\"/" eval.sh
    
    # Run evaluation
    bash eval.sh
    
    echo "Completed: $task"
    echo "---"
done
```

## Expected Results

Based on Table 8 from the H-RDT paper:
- **Easy mode** (`demo_clean`): ~68.7% average success rate
- **Hard mode** (`demo_randomized`): ~25.6% average success rate

## Notes

- If you trained in **multi-task mode**: Evaluate on all tasks with the same checkpoint
- If you trained in **single-task mode**: Each task has its own checkpoint, evaluate each task with its corresponding checkpoint:
  ```bash
  # For single-task checkpoint:
  ckpt_setting="checkpoints/table8_grab_roller_checkpoint30"
  task_name="grab_roller"
  ```

## Troubleshooting

1. **Checkpoint not found**: Verify the checkpoint path is relative to `H-RDT/inference/robotwin2_example/H-RDT/`
2. **Language embedding missing**: Copy from `/home/jose-barreiros/H_RDT/datasets/robotwin2/lang_embeddings/` to RobotWin's lang_embeddings folder
3. **Config mismatch**: Ensure `utils/hrdt.yaml` matches your training config (`configs/hrdt_finetune.yaml`)

