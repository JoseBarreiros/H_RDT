# Testing the Scaling Law Feature

This document explains how to test the data percentage feature for scaling law experiments.

## ✅ Test Results

The scaling feature has been successfully implemented and tested. Here's what was verified:

### Dataset Loading Tests
- ✅ Loads correctly with different percentages (1%, 5%, 10%, 25%, 50%, 100%)
- ✅ Reproducible with same seed
- ✅ Different seeds produce different subsets
- ✅ Percentage scaling is accurate

### File Count Verification
- Full dataset (100%): 314,839 files
- 50%: 157,419 files
- 25%: 78,709 files
- 10%: 31,483 files
- 5%: 15,741 files
- 1%: 3,148 files

## 🧪 How to Test

### 1. Quick Unit Test (Already Passed)

```bash
source hrdt_env/bin/activate
python test_scaling_feature.py
```

This tests:
- Dataset loading with different percentages
- Reproducibility with same seed
- Different seeds produce different samples
- Percentage calculations are correct

**Result**: ✅ All tests passed!

### 2. End-to-End Training Test

Test that training works with subsampled data:

```bash
./test_training_with_scaling.sh
```

Or run manually:

```bash
source hrdt_env/bin/activate

accelerate launch train/train.py \
    --config_path configs/hrdt_pretrain.yaml \
    --deepspeed configs/deepspeed_config.json \
    --output_dir ./checkpoints/test_scaling \
    --data_percentage 0.01 \
    --max_train_steps 50 \
    --seed 42 \
    --precomp_lang_embed \
    --mixed_precision bf16
```

This will:
- Train with only 1% of data (~3,148 files)
- Run for just 50 steps (quick test)
- Verify training pipeline works end-to-end

### 3. Quick Validation

Check the logs to see:
- Dataset size matches expected percentage
- Training starts successfully
- No errors with data loading

```bash
# Check training output
cat checkpoints/test_scaling/logs/*.txt | grep "Loaded"
```

Should show:
```
Subsampled to 3148/314800 files (1.0% of data)
Loaded 3148 train data files
```

## 📊 Real Scaling Experiments

After testing, run full scaling law experiments:

```bash
# Run all percentages (1%, 5%, 10%, 25%, 50%, 100%)
./run_scaling_law_experiments.sh
```

Or run specific percentages:

```bash
# Train with 10% for 10k steps
accelerate launch train/train.py \
    --config_path configs/hrdt_pretrain.yaml \
    --deepspeed configs/deepspeed_config.json \
    --output_dir ./checkpoints/scaling_p10 \
    --data_percentage 0.1 \
    --max_train_steps 10000 \
    --seed 42 \
    --precomp_lang_embed
```

## 🎯 What to Verify

### 1. Dataset Loading
- [x] Different percentages load correct number of files
- [x] Same seed = same files (reproducible)
- [x] Different seed = different files

### 2. Training Pipeline
- [ ] Training starts without errors
- [ ] Loss decreases during training
- [ ] Model checkpoints are saved

### 3. Results Comparison
- [ ] Smaller percentages train faster
- [ ] Performance improves with more data
- [ ] Scaling curve shows expected behavior

## 🔍 Troubleshooting

### Error: "data_percentage not recognized"
Make sure you're using the updated code:
```bash
cd /home/jose-barreiros/H_RDT
git status  # Check for uncommitted changes
```

### Error: "Dataset size mismatch"
The percentage affects training data only, not validation:
- Training uses the specified percentage
- Validation always uses 100% of test data
- This is intentional for fair comparison

### Memory Issues
For large experiments, reduce batch size:
```bash
--train_batch_size 16  # Instead of 32
```

## 📝 Expected Behavior

### Dataset Output
You should see messages like:
```
Subsampled to 31483/314830 files (10.0% of data)
Loaded 31483 train data files
```

### Training Output
```
Start training...
Train dataset size: 31483
...
Step 0: loss = X.XX
Step 100: loss = X.XX
...
```

### Checkpoints
Stored in:
```
checkpoints/scaling_p{X}/checkpoint-{step}/
```

## 🚀 Next Steps After Testing

1. **Validate preprocessing**: Ensure all data files exist
   ```bash
   python verify_dataset.py --data_root /home/jose-barreiros/egodex/organized
   ```

2. **Run quick test**: Validate with 1% for 100 steps
   ```bash
   ./test_training_with_scaling.sh
   ```

3. **Run scaling experiments**: Train at all percentages
   ```bash
   ./run_scaling_law_experiments.sh
   ```

4. **Analyze results**: Plot data size vs performance
   - Use logs from each experiment
   - Compare final performance metrics
   - Identify scaling trends

## 📚 More Information

- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Scaling Law Experiments**: [SCALING_LAW_GUIDE.md](SCALING_LAW_GUIDE.md)
- **Preprocessing**: [PREPROCESSING_SUMMARY.md](PREPROCESSING_SUMMARY.md)

