# Comparing Models Trained with Different Data Percentages

This guide explains how to fairly compare models pretrained with different percentages of EgoDex data (e.g., 1% vs 100%).

## 🎯 Two Fair Comparison Approaches

When comparing models trained with different data percentages, there are **two valid approaches**, each measuring different things:

### **Option 1: Compare by Same Number of Steps** (Training Efficiency)
- **What it measures**: Performance given the same computational budget
- **Use case**: When you want to know "Which performs better with the same training time/steps?"
- **Trade-off**: The smaller dataset model sees each example many more times

### **Option 2: Compare by Same Number of Epochs** (Data Efficiency) ⭐ **Most Fair**
- **What it measures**: Performance given the same number of passes over each example
- **Use case**: When you want to know "How much does more data help?"
- **Trade-off**: Requires training the larger dataset model for much longer

## 📊 Your Specific Case

You have:
- **Model A**: Pretrained with 1% EgoDex data for 80k steps
- **Model B**: Pretrained with 100% EgoDex data for 80k steps

**Dataset Sizes:**
- 1% dataset: ~3,181 files
- 100% dataset: 318,082 files

**Epoch Calculations** (assuming batch_size=48, 4 GPUs, gradient_accumulation=1):
- 100% model at 80k steps ≈ **48 epochs**
- 1% model at 80k steps ≈ **1,212 epochs** (100x more epochs!)

## 🔬 Option 1: Compare by Same Number of Steps

### Comparison Setup

Compare both models at `checkpoint-80000`:

```
Model A (1%):  ./checkpoints/scaling_p1/checkpoint-80000
Model B (100%): ./checkpoints/scaling_p100/checkpoint-80000
```

### What This Measures

- **Training efficiency**: Which model performs better given the same computational budget?
- **Practical performance**: What can you achieve with the same training time?
- **Convergence speed**: Which model converges faster in terms of steps?

### Considerations

✅ **Advantages:**
- Easy to implement (use same checkpoint number)
- Fair computational comparison
- Practical for real-world deployment decisions

❌ **Disadvantages:**
- **Unfair data exposure**: 1% model has seen each example ~25x more than 100% model
- 1% model: ~1,212 epochs vs 100% model: ~48 epochs
- Doesn't answer "How much does more data help?"

### Evaluation Protocol

**Step 1: Fine-tune Both Models**

Fine-tune both at checkpoint-80000 on the same downstream task (e.g., RobotWin Table 8 tasks):

**Model A (1% pretraining) → Fine-tuning:**
```bash
accelerate launch main.py \
    --dataset_name="robotwin_agilex" \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --config_path configs/hrdt_finetune.yaml \
    --output_dir ./checkpoints/ft_from_1pct_80ksteps \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --dataset_type finetune \
    --mode finetune \
    --pretrained_backbone_path "./checkpoints/scaling_p1/checkpoint-80000/pytorch_model.bin" \
    --report_to wandb
```

**Model B (100% pretraining) → Fine-tuning:**
```bash
accelerate launch main.py \
    --dataset_name="robotwin_agilex" \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --config_path configs/hrdt_finetune.yaml \
    --output_dir ./checkpoints/ft_from_100pct_80ksteps \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --dataset_type finetune \
    --mode finetune \
    --pretrained_backbone_path "./checkpoints/scaling_p100/checkpoint-80000/pytorch_model.bin" \
    --report_to wandb
```

**⚠️ Critical:** Use **identical** fine-tuning hyperparameters for both models!

## 🔬 Option 2: Compare by Same Number of Epochs ⭐ **Most Fair**

### Comparison Setup

Compare models at the **same number of epochs**, which requires different step numbers:

**If 100% model has 48 epochs at 80k steps:**
```
Model A (1%):  ./checkpoints/scaling_p1/checkpoint-800  (48 epochs, ~800 steps)
Model B (100%): ./checkpoints/scaling_p100/checkpoint-80000  (48 epochs, ~80k steps)
```

### What This Measures

- **Data efficiency**: How much does more data help when each example is seen the same number of times?
- **True scaling behavior**: Fair comparison of dataset size impact
- **Data quality**: Does diversity matter more than repetition?

### Considerations

✅ **Advantages:**
- **Most fair comparison**: Each example seen the same number of times
- Answers "How much does more data help?"
- Standard for scaling law research

❌ **Disadvantages:**
- Requires training the 100% model for much longer (8M steps for 1,212 epochs)
- May not be practical if compute budget is limited

### Epoch-to-Steps Conversion

**For 100% model (48 epochs):**
- Dataset size: 318,082 files
- Effective batch size: 48 × 4 GPUs × 1 grad_accum = 192
- Steps per epoch: ~1,667 steps
- Total steps for 48 epochs: **80,000 steps**

**For 1% model (48 epochs):**
- Dataset size: ~3,181 files
- Effective batch size: 48 × 4 GPUs × 1 grad_accum = 192
- Steps per epoch: ~16.67 steps
- Total steps for 48 epochs: **~800 steps**

### Training the 1% Model to Match Epochs

If you need to train the 1% model to match 48 epochs:

```bash
accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed configs/zero1.json \
    --config_path configs/hrdt_pretrain.yaml \
    --output_dir ./checkpoints/scaling_p1_48epochs \
    --train_batch_size 48 \
    --sample_batch_size 32 \
    --max_train_steps 800 \
    --learning_rate 1e-4 \
    --data_percentage 0.01 \
    --seed 42 \
    --checkpointing_period 200 \
    --checkpoints_total_limit 10 \
    --sample_period 100 \
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

This will train the 1% model to 48 epochs (~800 steps) to match the 100% model at 48 epochs.

### Evaluation Protocol

**Step 1: Fine-tune Both Models at Same Epoch Count**

Fine-tune both at 48 epochs on the same downstream task:

**Model A (1% pretraining, 48 epochs) → Fine-tuning:**
```bash
accelerate launch main.py \
    --dataset_name="robotwin_agilex" \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --config_path configs/hrdt_finetune.yaml \
    --output_dir ./checkpoints/ft_from_1pct_48epochs \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --dataset_type finetune \
    --mode finetune \
    --pretrained_backbone_path "./checkpoints/scaling_p1/checkpoint-800/pytorch_model.bin" \
    --report_to wandb
```

**Model B (100% pretraining, 48 epochs) → Fine-tuning:**
```bash
accelerate launch main.py \
    --dataset_name="robotwin_agilex" \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --config_path configs/hrdt_finetune.yaml \
    --output_dir ./checkpoints/ft_from_100pct_48epochs \
    --train_batch_size 32 \
    --max_train_steps 10000 \
    --learning_rate 1e-4 \
    --dataset_type finetune \
    --mode finetune \
    --pretrained_backbone_path "./checkpoints/scaling_p100/checkpoint-80000/pytorch_model.bin" \
    --report_to wandb
```

## 📊 Which Comparison Should You Use?

### Use **Option 1 (Same Steps)** if:
- ✅ You have a fixed computational budget
- ✅ You want to know practical performance with same training time
- ✅ You're comparing training efficiency
- ✅ You can't train the larger dataset model longer

### Use **Option 2 (Same Epochs)** if:
- ✅ You want to measure true data scaling behavior
- ✅ You want a fair comparison of dataset size impact
- ✅ You're doing scaling law research
- ✅ You have compute budget to train the larger model longer
- ✅ You want to answer "How much does more data help?"

## 🔍 Additional Comparison Points

### Compare Intermediate Checkpoints

For Option 1 (same steps), compare at:
- `checkpoint-40000`: Mid-training comparison
- `checkpoint-60000`: Late-training comparison
- `checkpoint-80000`: Final comparison

For Option 2 (same epochs), compare at:
- Model A: checkpoint-200, checkpoint-400, checkpoint-800
- Model B: checkpoint-20000, checkpoint-40000, checkpoint-80000
- (Matching same epoch counts)

### Compare Training Dynamics

Compare pretraining metrics:
- **Training loss curves** (from WandB or logs)
- **Validation loss** (if available)
- **Convergence speed** (steps vs epochs)
- **Sample efficiency** (how many examples needed per improvement)

## 📈 Expected Findings

Based on scaling law research:

### Option 1 (Same Steps) Results:
1. **1% model may perform surprisingly well**: Has seen each example many times
2. **100% model shows better generalization**: More diverse data seen
3. **Diminishing returns**: 100% model likely better but not 100x better
4. **Early convergence**: 1% model may converge faster in steps

### Option 2 (Same Epochs) Results:
1. **100% model should clearly outperform**: True benefit of more data
2. **Data efficiency**: Shows how much diversity matters vs repetition
3. **Scaling behavior**: More linear relationship between data and performance
4. **Transfer learning**: Larger dataset should transfer better to downstream tasks

## 📝 Comparison Checklist

### For Option 1 (Same Steps):
- [ ] Both models at checkpoint-80000
- [ ] Both models use same pretraining hyperparameters
- [ ] Both fine-tuned with identical hyperparameters
- [ ] Both evaluated on same RobotWin tasks
- [ ] Note: 1% model has seen ~1,212 epochs vs 100% model ~48 epochs

### For Option 2 (Same Epochs):
- [ ] Both models trained to same epoch count (e.g., 48 epochs)
- [ ] Calculate correct step numbers for each model
- [ ] Both models use same pretraining hyperparameters
- [ ] Both fine-tuned with identical hyperparameters
- [ ] Both evaluated on same RobotWin tasks
- [ ] Note: 1% model at ~800 steps vs 100% model at ~80k steps

## 🎯 Recommended Evaluation Protocol

### For Both Options:

1. **Fine-tune both models** on RobotWin Table 8 tasks
   - Use checkpoint matching your comparison method
   - Same fine-tuning setup
   
2. **Evaluate on all 13 Table 8 tasks**
   - Easy mode: `demo_fixed`
   - Hard mode: `demo_randomized`
   
3. **Compare average success rates**
   - Overall average
   - Per-task comparison
   - Statistical significance (if needed)

4. **Analyze training dynamics**
   - Fine-tuning loss curves
   - Convergence speed
   - Final performance
   - **For Option 1**: Steps to convergence
   - **For Option 2**: Epochs to convergence

## 📊 Quick Reference: Epoch-to-Steps Mapping

**For 100% EgoDex (318,082 files) with batch_size=48, 4 GPUs:**

| Epochs | Steps |
|--------|-------|
| 1      | ~1,667 |
| 10     | ~16,667 |
| 24     | ~40,000 |
| 48     | ~80,000 |
| 100    | ~166,667 |

**For 1% EgoDex (~3,181 files) with batch_size=48, 4 GPUs:**

| Epochs | Steps |
|--------|-------|
| 1      | ~17 |
| 10     | ~167 |
| 24     | ~400 |
| 48     | ~800 |
| 100    | ~1,667 |
| 1,212  | ~80,000 |

## 🔗 Related Documentation

- [SCALING_LAW_GUIDE.md](SCALING_LAW_GUIDE.md) - Running scaling law experiments
- [ROBOTWIN_TABLE8_SETUP.md](ROBOTWIN_TABLE8_SETUP.md) - Fine-tuning setup
- [ROBOTWIN_EVALUATION_README.md](ROBOTWIN_EVALUATION_README.md) - Evaluation guide

---

**Summary:** Choose your comparison method based on what you want to measure. Option 1 (same steps) measures training efficiency, while Option 2 (same epochs) measures true data scaling and is more fair for answering "How much does more data help?"
