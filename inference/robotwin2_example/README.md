# RoboTwin2 Inference Setup

## Setup Steps

1. Copy H-RDT folder to RoboTwin/policy/
```bash
cp -r H-RDT /path/to/RoboTwin/policy/
```

2. Copy bak folder to H-RDT/
```bash
cp -r H-RDT/bak /path/to/RoboTwin/policy/H-RDT/
```

3. Create checkpoints directory and copy model files
```bash
mkdir -p /path/to/RoboTwin/policy/H-RDT/checkpoints/folder_name/
cp H-RDT/checkpoints/*/config.json /path/to/RoboTwin/policy/H-RDT/checkpoints/folder_name/
cp H-RDT/checkpoints/*/pytorch_model.bin /path/to/RoboTwin/policy/H-RDT/checkpoints/folder_name/
```

## Run Inference

The `eval.sh` script accepts command-line arguments, so no file editing is needed:

```bash
cd /path/to/RoboTwin/policy/H-RDT/inference/robotwin2_example/H-RDT

# Usage: ./eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [seed]
./eval.sh handover_mic demo_clean checkpoints/folder_name 0

# Example with all arguments:
./eval.sh grab_roller demo_randomized checkpoints/table8_grab_roller/checkpoint-10000 0 42
```

**Arguments:**
- `task_name`: Task to evaluate (e.g., `handover_mic`, `grab_roller`)
- `task_config`: `demo_clean` (Easy) or `demo_randomized` (Hard)
- `ckpt_setting`: Checkpoint path (e.g., `checkpoints/folder_name`)
- `gpu_id`: GPU ID to use (e.g., `0`)
- `seed`: Random seed (optional, defaults to `42`)

**Note:** Run `./eval.sh` without arguments to see the usage message.
