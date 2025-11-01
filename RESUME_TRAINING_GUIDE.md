# Resume Training from Checkpoint Guide

This guide explains how to resume training from a checkpoint for both pretraining and fine-tuning scenarios.

## 📋 Quick Summary

- **Pretrain Resume**: Add `--resume_from_checkpoint checkpoint-XXXXX` with same `--output_dir`
- **Finetune Resume**: Use `--mode="pretrain"` (not `"finetune"`) + `--resume_from_checkpoint`
- **Key Requirement**: Checkpoint must exist in `--output_dir` directory

## 🔄 Resuming Pretraining

### Basic Command

```bash
accelerate launch --main_process_port 29500 main.py \
    --resume_from_checkpoint checkpoint-60000 \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed configs/zero1.json \
    --config_path configs/hrdt_pretrain.yaml \
    --output_dir ./checkpoints/scaling_p100 \
    --train_batch_size 48 \
    --sample_batch_size 32 \
    --max_train_steps 100000 \
    --learning_rate 1e-4 \
    --data_percentage 1.0 \
    --seed 42 \
    --checkpointing_period 10000 \
    --checkpoints_total_limit 40 \
    --sample_period 1000 \
    --lr_scheduler=constant_with_warmup \
    --mixed_precision bf16 \
    --dataloader_num_workers 32 \
    --dataset_type pretrain \
    --report_to wandb \
    --upsample_rate 3 \
    --image_aug \
    --gradient_checkpointing \
    --precomp_lang_embed \
    --training_mode lang \
    --mode pretrain
```

### Important Requirements

1. **Same Output Directory**: Use the same `--output_dir` as original training
   - ✅ Correct: `--output_dir ./checkpoints/scaling_p100` (checkpoint exists here)
   - ❌ Wrong: `--output_dir ./checkpoints/new_dir` (checkpoint doesn't exist here)

2. **Relative Checkpoint Name**: Use relative name, not full path
   - ✅ Correct: `--resume_from_checkpoint checkpoint-60000`
   - ❌ Wrong: `--resume_from_checkpoint ./checkpoints/scaling_p100/checkpoint-60000`

3. **Same Learning Rate Scheduler**: Keep `--lr_scheduler` argument the same
   - The scheduler state is saved in checkpoint (`scheduler.bin`)
   - Using a different scheduler type may cause issues

4. **Keep Hyperparameters Same**: Maintain all training hyperparameters
   - Batch size, learning rate, optimizer settings should match
   - Different settings may cause training instability

### Using "latest" Checkpoint

Automatically resume from the most recent checkpoint:

```bash
--resume_from_checkpoint latest
```

This finds the highest numbered checkpoint in `output_dir`.

### Resuming in Different Output Directory

If you must use a different output directory:

1. **Copy the checkpoint first:**
   ```bash
   mkdir -p ./checkpoints/new_output_dir
   cp -r ./checkpoints/original_dir/checkpoint-60000 ./checkpoints/new_output_dir/
   ```

2. **Then resume with new output_dir:**
   ```bash
   accelerate launch ... \
       --resume_from_checkpoint checkpoint-60000 \
       --output_dir ./checkpoints/new_output_dir \
       ...
   ```

## 🔄 Resuming Fine-tuning

### Basic Command

```bash
accelerate launch main.py \
    --resume_from_checkpoint checkpoint-5000 \
    --dataset_name="robotwin_agilex" \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --config_path configs/hrdt_finetune.yaml \
    --output_dir ./checkpoints/table8_finetune \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --dataset_type finetune \
    --mode pretrain \
    --pretrained_backbone_path "./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin" \
    --report_to wandb
```

### Critical Difference: Mode Parameter

**⚠️ IMPORTANT:** When resuming fine-tuning, use `--mode="pretrain"` (NOT `"finetune"`)

- `--mode="finetune"`: Used for **starting** fine-tuning (loads pretrained backbone)
- `--mode="pretrain"`: Used for **resuming** any training (pretrain or finetune)

### Why Mode Matters

- `"finetune"` mode initializes action encoder/decoder from scratch
- `"pretrain"` mode loads everything from checkpoint (including action layers)

## 🔍 What Gets Restored from Checkpoint

When resuming, the following are restored:

1. **Model Weights**: `pytorch_model.bin`
2. **Optimizer State**: Optimizer parameters and momentum
3. **Learning Rate Scheduler**: Scheduler state from `scheduler.bin`
4. **Random States**: For reproducibility (`random_states_*.pkl`)
5. **Global Step**: Training continues from checkpoint step number
6. **Accelerator State**: Distributed training state (if using multi-GPU)

## ⚠️ Common Issues and Solutions

### Issue 1: Checkpoint Not Found

**Error:**
```
ValueError: Tried to find ./checkpoints/new_dir/checkpoint-60000 but folder does not exist
```

**Cause:** Checkpoint doesn't exist in the specified `--output_dir`

**Solutions:**
1. Use the same `--output_dir` as original training
2. Or copy checkpoint to new directory before resuming:
   ```bash
   cp -r ./checkpoints/original_dir/checkpoint-60000 ./checkpoints/new_dir/
   ```

### Issue 2: Scheduler State Mismatch

**Error:** Learning rate behaves unexpectedly or errors occur

**Cause:** Different `--lr_scheduler` argument than original training

**Solution:** Use the same `--lr_scheduler` argument as original training
- Check your original training command
- Match the scheduler type exactly

### Issue 3: Training Starts from Step 0

**Symptom:** Training logs show step 0 instead of resuming from checkpoint step

**Causes:**
- Checkpoint not found in `--output_dir`
- Wrong checkpoint name format
- Full path used instead of relative name

**Solutions:**
- Verify checkpoint exists: `ls ./checkpoints/your_dir/checkpoint-60000/`
- Use relative name: `checkpoint-60000` not `./checkpoints/.../checkpoint-60000`
- Check that `--output_dir` matches checkpoint location

### Issue 4: WandB Run Continuity

**Symptom:** WandB creates a new run instead of continuing previous run

**Cause:** WandB creates new runs by default when resuming

**Solutions:**
- This is expected behavior
- WandB will log from the resumed step
- If you need to continue the same run, you may need to specify `--wandb_run_id` manually

### Issue 5: Checkpoint Contains Multiple Shards (DeepSpeed)

**Symptom:** Checkpoint directory has `pytorch_model/` subdirectory with sharded files

**Cause:** DeepSpeed ZeRO-3 saves checkpoints in sharded format

**Note:** This is fine for resuming training! The `accelerator.load_state()` handles this automatically.

**Only convert to consolidated format for:**
- Model inference
- Sharing checkpoints
- Loading in different frameworks

## 📝 Best Practices

1. **Always Use Same Output Directory**
   - Simplest and most reliable approach
   - Avoids checkpoint copying issues

2. **Match Hyperparameters**
   - Keep batch size, learning rate, scheduler type identical
   - Changing these mid-training can cause instability

3. **Verify Checkpoint Exists Before Training**
   ```bash
   ls ./checkpoints/your_dir/checkpoint-XXXXX/
   # Should show: pytorch_model.bin, scheduler.bin, config.json, etc.
   ```

4. **Document Checkpoint Information**
   - Note the step number when resuming
   - Track hyperparameters used in original training
   - Save training commands for reference

5. **Use Descriptive Checkpoint Names**
   - When copying checkpoints, use descriptive names
   - Example: `checkpoint-60000-resumed` or `checkpoint-60000-2`

## 📊 Resume Training Checklist

Before resuming training, verify:

- [ ] Checkpoint directory exists in `--output_dir`
- [ ] Using same `--lr_scheduler` as original training
- [ ] Using same `--output_dir` as original (or copied checkpoint)
- [ ] All hyperparameters match original training
- [ ] For fine-tuning: using `--mode="pretrain"` (not `"finetune"`)
- [ ] Checkpoint name is relative (e.g., `checkpoint-60000`, not full path)

## 🔗 Related Documentation

- [README.md](README.md) - Main documentation with quick resume examples
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Initial setup and configuration
- [ROBOTWIN_TABLE8_SETUP.md](ROBOTWIN_TABLE8_SETUP.md) - Fine-tuning setup guide

---

**Need Help?** Check the troubleshooting section above or review the main README for additional examples.

