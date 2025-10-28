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
export EGODEX_DATA_ROOT="/home/jose-barreiros/egodex/organized"  # Raw data
export HRDT_OUTPUT_DIR="/mnt/disks/hrdt-data/processed"           # Preprocessed data

# Model paths
export T5_MODEL_PATH="google/t5-v1_1-xxl"

# Training config
export WANDB_PROJECT="hrdt-scaling-laws"
export WANDB_ENTITY="your-entity"  # Update this

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
    if [ -f "/mnt/disks/hrdt-data/processed/egodex_stat.json" ]; then
        echo "   ✅ Preprocessed data found at /mnt/disks/hrdt-data"
        ls -lh /mnt/disks/hrdt-data/processed/ | head -5
    else
        echo "   ⚠️  Disk mounted but no preprocessed data found"
        echo "      Update HRDT_OUTPUT_DIR in ~/.config/hrdt/config.sh"
    fi
else
    echo "   ⚠️  Persistent disk not mounted"
    echo "      If using persistent disk, follow INSTANCE_SETUP.md to mount it"
fi

# 8. Final instructions
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "1. Activate environment:"
echo "   source ~/.config/hrdt/activate.sh"
echo ""
echo "2. Update paths in config if needed:"
echo "   nano ~/.config/hrdt/config.sh"
echo ""
echo "3. Verify dataset:"
echo "   python verify_dataset.py --data_root \$EGODEX_DATA_ROOT"
echo ""
echo "4. Start training:"
echo "   bash pretrain.sh"
echo ""
echo "💡 For multi-instance setup, see INSTANCE_SETUP.md"
echo "=========================================="

