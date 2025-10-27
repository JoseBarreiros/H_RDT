# H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation
### 📝[Paper](https://arxiv.org/abs/2507.23523) | 🌍[Project Page](https://embodiedfoundation.github.io/hrdt) | 🤗[Model](https://huggingface.co/embodiedfoundation/H-RDT) | 💬[WeChat Contact](#-contact-us) 

![H-RDT](assets/h-rdt.jpg)

## 📰 News
• **[2025.8.12]** Updated RoboTwin2 inference code

H-RDT (**H**uman to **R**obotics **D**iffusion **T**ransformer) is a novel approach that leverages **large-scale egocentric human manipulation data** to enhance robot manipulation capabilities. Our key insight is that large-scale egocentric human manipulation videos with paired 3D hand pose annotations provide rich behavioral priors that capture natural manipulation strategies and can benefit robotic policy learning.

## 🚀 Installation

### Quick Start (Recommended)

For a complete setup guide with EgoDex data preprocessing, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

1. **Create virtual environment:**
   ```bash
   python3 -m venv hrdt_env
   source hrdt_env/bin/activate
   pip install --upgrade pip
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models (optional):**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download --resume-download embodiedfoundation/H-RDT --local-dir ./
   ```

### Alternative: Conda Installation

1. **Create conda environment:**
   ```bash
   conda create -n hrdt python=3.10
   conda activate hrdt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Usage

### Stage 1: Human Data Pre-training (EgoDex)

> **📖 For detailed setup instructions with troubleshooting, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

> **📊 For scaling law experiments, see [SCALING_LAW_GUIDE.md](SCALING_LAW_GUIDE.md)**

#### Data Preprocessing
Before training, preprocess the EgoDex dataset:

1. **Configure paths:**
   ```bash
   # Edit datasets/pretrain/setup_pretrain.sh with your paths
   nano datasets/pretrain/setup_pretrain.sh
   
   # Set your EgoDex dataset and T5 model paths:
   export EGODEX_DATA_ROOT="/path/to/your/egodex/dataset"
   export T5_MODEL_PATH="google/t5-v1_1-xxl"  # Uses HuggingFace model
   ```

2. **Setup environment:**
   ```bash
   source hrdt_env/bin/activate  # or conda activate hrdt
   source datasets/pretrain/setup_pretrain.sh
   ```

3. **Run data processing pipeline:**
   ```bash
   # Automatically runs: precompute_48d_actions.py → calc_stat.py → encode_lang_batch.py
   ./datasets/pretrain/run_pretrain_pipeline.sh
   ```

4. **Verify preprocessing results:**
   ```bash
   # Run comprehensive verification
   python verify_dataset.py --data_root /path/to/your/egodex/dataset
   
   # For detailed output
   python verify_dataset.py --data_root /path/to/your/egodex/dataset --verbose
   ```

#### Start Pre-training
After data preprocessing is complete:

**1. EgoDex Pretrain (fresh start):**
1. Dataset is already configured for EgoDex (default)
2. Run training:
   ```bash
   source hrdt_env/bin/activate  # or conda activate hrdt
   bash pretrain.sh
   ```
   
   **Optional:** To customize dataset configuration, edit `datasets/dataset.py` line ~45:
   ```python
   self.dataset_name = "egodex"  # Already set by default
   ```

**2. Pretrain Resume:**
Edit `pretrain.sh`, add this line:
```bash
--resume_from_checkpoint="checkpoint-450000" \
```

**3. Scaling Law Experiments:**
Train with different percentages of EgoDex data to study data efficiency and performance scaling:
```bash
# Train with 10% of data for faster iteration
source hrdt_env/bin/activate

accelerate launch --main_process_port 29500 main.py \
    --pretrained_vision_encoder_name_or_path="dino-siglip" \
    --deepspeed configs/zero1.json \
    --config_path configs/hrdt_pretrain.yaml \
    --output_dir ./checkpoints/scaling_p10 \
    --train_batch_size 32 \
    --sample_batch_size 32 \
    --max_train_steps 100000 \
    --learning_rate 1e-4 \
    --data_percentage 0.1 \
    --seed 42 \
    --checkpointing_period 10000 \
    --precomp_lang_embed \
    --mixed_precision bf16 \
    --dataloader_num_workers 32 \
    --dataset_type pretrain \
    --upsample_rate 3 \
    --image_aug \
    --gradient_checkpointing \
    --training_mode lang \
    --mode pretrain

# Or run all percentages automatically
./run_scaling_law_experiments.sh
```
📖 **For detailed scaling law guide, see [SCALING_LAW_GUIDE.md](SCALING_LAW_GUIDE.md)**

### Stage 2: Cross-Embodiment Fine-tuning

#### Data Preprocessing (for RobotWin2)
**Pre-computed language embeddings are already provided - no preprocessing needed!**

1. **Setup environment:**
   ```bash
   # Edit datasets/robotwin2/setup_robotwin2.sh if needed (only for regenerating files)
   source datasets/robotwin2/setup_robotwin2.sh
   ```

2. **Data processing pipeline (Not Required):**
   ```bash
   # Not needed - lang_embeddings/ already provided in repository
   # Only run if you want to regenerate files:
   # ./datasets/robotwin2/run_robotwin2_pipeline.sh
   ```

#### Robot Fine-tuning (load human pre-trained backbone):
1. Configure dataset:
   ```python
   # Edit datasets/dataset.py line ~45
   self.dataset_name = "robotwin_agilex"  # or your robot name
   
   # Add your dataset initialization if not exists:
   elif self.dataset_name == "your_robot":
       self.hdf5_dataset = YourRobotDataset(config=config)
   ```
2. Run training:
   ```bash
   bash finetune.sh  # Already configured with pretrained_backbone_path
   ```

#### Finetune Resume:
Edit your current finetune script, make these changes:
```bash
# Change this line:
--mode="finetune" \
# To:
--mode="pretrain" \

# And add:
--resume_from_checkpoint="checkpoint-5000" \
```

## 🎯 Training Modes

| Training Scenario | Base Script | Required Shell Script Modifications | Mode & Key Parameters |
|-------------------|-------------|-------------------------------------|----------------------|
| **Human Pretrain (Fresh)** | `pretrain.sh` | `--mode="pretrain"` | Start pretraining on EgoDex human data |
| **Human Pretrain Resume** | `pretrain.sh` | Add: `--resume_from_checkpoint="checkpoint-450000" \` | `--mode="pretrain"` |
| **Robot Fine-tuning** | `finetune.sh` | Change: `--mode="finetune" \`<br>Add: `--pretrained_backbone_path="./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin" \`<br>Change: `--config_path="configs/hrdt_finetune.yaml" \` | Load human pre-trained backbone, fresh action layers |
| **Robot Finetune Resume** | Your finetune script | Change: `--mode="finetune"` → `--mode="pretrain"`<br>Add: `--resume_from_checkpoint="checkpoint-5000" \` | Continue robot fine-tuning |

### Dataset Configuration

Before training, you need to configure the dataset in `datasets/dataset.py`:

#### For Human Pre-training (EgoDex):
```python
# In datasets/dataset.py, line ~45
self.dataset_name = "egodex"

# The EgoDexDataset will be automatically initialized
```

#### For Robot Fine-tuning:
```python
# In datasets/dataset.py, line ~45  
self.dataset_name = "your_robot_name"  # e.g., "robotwin_agilex"

# Add your dataset to the initialization logic:
elif self.dataset_name == "your_robot_name":
    self.hdf5_dataset = YourRobotDataset(
        config=config,
        # your dataset parameters
    )
```

#### Adding New Robot Datasets:
1. Create your dataset folder: `datasets/your_robot/`
2. Implement your dataset class (see `datasets/robotwin2/` as example)
3. Create data processing scripts (see `datasets/pretrain/` or `datasets/robotwin2/` as examples)
4. Import in `datasets/dataset.py`
5. Add initialization logic in `VLAConsumerDataset.__init__`

### Key Configuration Files
- `configs/hrdt_pretrain.yaml`: Human pre-training configuration (buffer path updated)
- `configs/hrdt_finetune.yaml`: Robot fine-tuning configuration (buffer path updated)
- `datasets/dataset.py`: Dataset selection and initialization
- Modify `state_dim`, `action_dim`, `output_size` for your robot

**Note:** All hardcoded user-specific paths have been removed and replaced with generic paths or environment variables. See [LINGXUAN_CLEANUP.md](LINGXUAN_CLEANUP.md) for details.

## 📞 Contact Us

### WeChat Discussion Group
Join our WeChat group to discuss H-RDT related technical issues:

<div align="center">
<img src="assets/wechat_group_qr.jpg" width="200" alt="WeChat Group QR Code">
<p><em>WeChat Group QR Code</em></p>
</div>

### Personal WeChat
For other questions or collaboration opportunities, please add personal WeChat:

<div align="center">
<img src="assets/personal_wechat_qr.jpg" width="200" alt="Personal WeChat QR Code">
<p><em>Personal WeChat QR Code</em></p>
</div>

---

*Note: If the QR code expires, please contact us through project Issues for the latest contact information.*
