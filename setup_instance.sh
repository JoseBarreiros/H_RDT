#!/bin/bash
# setup_instance.sh
# Quick setup script for H-RDT instance
# Run this on each new GCP instance

set -e

echo "=========================================="
echo "H-RDT Instance Setup"
echo "=========================================="

# Configuration
PROJECT_ROOT="$HOME/H_RDT"
ENV_NAME="hrdt_env"

# 1. Install basic dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git python3.10-venv python3-pip htop ncdu || true

# 2. Check if repo exists
if [ -d "$PROJECT_ROOT" ]; then
    echo ""
    echo "📥 Repository found at $PROJECT_ROOT"
    echo "   Run 'git pull' to update if needed"
else
    echo ""
    echo "⚠️  Repository not found at $PROJECT_ROOT"
    echo ""
    echo "To set up:"
    echo "  git clone <YOUR_REPO_URL> H_RDT"
    echo "  cd H_RDT"
    echo "  bash setup_instance.sh"
    echo ""
    exit 1
fi

# 3. Create virtual environment
echo ""
echo "🐍 Setting up Python environment..."
cd "$PROJECT_ROOT"

if [ -d "$ENV_NAME" ]; then
    echo "   Virtual environment already exists"
else
    python3.10 -m venv "$ENV_NAME"
    echo "   Created virtual environment"
fi

# 4. Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
source "$ENV_NAME/bin/activate"
pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "   Dependencies installed"
else
    echo "   ⚠️  requirements.txt not found, skipping package installation"
fi

# 4b. Download vision encoder models if not already present
echo ""
if [ ! -f "bak/dino-siglip/vit_large_patch14_reg4_dinov2.lvd142m/pytorch_model.bin" ]; then
    echo "📥 Downloading vision encoder models..."
    python download_vision_models.py
else
    echo "✅ Vision encoder models already present"
fi

# 5. Create configuration file
echo ""
echo "⚙️  Creating configuration file..."

mkdir -p ~/.config/hrdt

cat > ~/.config/hrdt/config.sh << 'EOF'
#!/bin/bash
# H-RDT Instance Configuration
# Source with: source ~/.config/hrdt/config.sh

export HRDT_PROJECT_ROOT="$HOME/H_RDT"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

# Update these paths for your setup
export EGODEX_DATA_ROOT="/mnt/disks/hrdt-data/egodex/organized"   # Raw + preprocessed data on disk
export HRDT_OUTPUT_DIR="/mnt/disks/hrdt-data/egodex/organized"    # Same location for preprocessed data

# Model paths
export T5_MODEL_PATH="google/t5-v1_1-xxl"

# Training config
export WANDB_PROJECT="h-rdt"
export WANDB_ENTITY="jose-barreiros-879-jb"

# CUDA/CUTLASS
export CUDA_HOME=/usr/local/cuda
export CUTLASS_PATH="${CUTLASS_PATH:-/usr/local/cutlass}"

echo "✅ H-RDT environment configured"
echo "   Project: $HRDT_PROJECT_ROOT"
echo "   Data Cache: $HRDT_OUTPUT_DIR"
EOF

chmod +x ~/.config/hrdt/config.sh

# 6. Create activation script
cat > ~/.config/hrdt/activate.sh << 'EOF'
#!/bin/bash
# Quick activation: source ~/.config/hrdt/activate.sh

export HRDT_PROJECT_ROOT="$HOME/H_RDT"
cd "$HRDT_PROJECT_ROOT"

source hrdt_env/bin/activate
source ~/.config/hrdt/config.sh

echo "🚀 H-RDT environment activated!"
echo "   Project: $HRDT_PROJECT_ROOT"
echo "   Python: $(which python)"
EOF

chmod +x ~/.config/hrdt/activate.sh

# 7. Check for data directory
echo ""
echo "🔍 Checking data directory..."
if [ -d "/mnt/disks/hrdt-data" ]; then
    if [ -d "/mnt/disks/hrdt-data/egodex/organized" ]; then
        echo "   ✅ Data found at /mnt/disks/hrdt-data/egodex/organized"
        du -sh /mnt/disks/hrdt-data/egodex/organized/ | head -1
        echo ""
        echo "   Checking for preprocessed files..."
        find /mnt/disks/hrdt-data/egodex/organized -name "*.pt" | wc -l | xargs echo "   Language embeddings (.pt files):"
        if [ -f "datasets/pretrain/egodex_stat.json" ]; then
            echo "   ✅ Statistics file found in local repo"
        fi
    else
        echo "   ⚠️  Disk mounted but no data found at /mnt/disks/hrdt-data/egodex/organized"
        echo "      Follow INSTANCE_SETUP.md to set up the persistent disk"
    fi
else
    echo "   ⚠️  Persistent disk not mounted"
    echo "      Follow INSTANCE_SETUP.md to mount the disk"
fi

# 8. Check GPU availability
echo ""
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv
    
    # Install nvtop for GPU monitoring (optional)
    echo ""
    echo "🔧 GPU monitoring tool installation..."
    if command -v nvtop &> /dev/null; then
        echo "   ✅ nvtop already installed"
    else
        read -p "   Install nvtop for GPU monitoring? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "   Installing nvtop..."
            if [ -f "install_nvtop.sh" ]; then
                bash install_nvtop.sh
            else
                echo "   ⚠️  install_nvtop.sh not found, skipping"
            fi
        else
            echo "   Skipping nvtop installation"
        fi
    fi
else
    echo "   ⚠️  nvidia-smi not found - GPUs may not be accessible"
    echo "      For GCP instances with GPUs, the NVIDIA driver should be auto-installed"
    echo "      If training fails, check GPU driver installation"
fi

# 9. Final instructions
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "1. Activate environment:"
echo "   source ~/.config/hrdt/activate.sh"
echo ""
echo "2. Monitor GPUs (if installed):"
echo "   nvtop              # Interactive GPU monitor"
echo ""
echo "3. Verify GPU access (if using GPUs):"
echo "   python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
echo "4. Verify dataset:"
echo "   python verify_dataset.py --data_root \$EGODEX_DATA_ROOT"
echo ""
echo "5. Start training:"
echo "   bash pretrain.sh"
echo ""
echo "💡 For scaling law experiments, see SCALING_LAW_GUIDE.md"
echo "💡 For multi-instance setup, see INSTANCE_SETUP.md"
echo "=========================================="

