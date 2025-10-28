#!/bin/bash
# install_nvtop.sh
# Automated installation of nvtop (NVIDIA GPU monitoring tool)

set -e

echo "=========================================="
echo "Installing nvtop (NVIDIA GPU Monitor)"
echo "=========================================="

# Install dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libdrm-dev \
    cmake \
    build-essential \
    git \
    libncurses5-dev \
    libncursesw5-dev

# Create nvtop directory in user's home
NVTOP_DIR="$HOME/nvtop"

echo ""
echo "📥 Cloning nvtop repository..."
if [ -d "$NVTOP_DIR" ]; then
    echo "   nvtop directory already exists, updating..."
    cd "$NVTOP_DIR"
    git pull
else
    git clone https://github.com/Syllo/nvtop.git "$NVTOP_DIR"
    cd "$NVTOP_DIR"
fi

# Build directory
echo ""
echo "🔨 Building nvtop..."
rm -rf build
mkdir -p build && cd build

# Configure with NVIDIA support only
cmake .. \
    -DNVIDIA_SUPPORT=ON \
    -DAMDGPU_SUPPORT=OFF \
    -DINTEL_SUPPORT=OFF \
    -DV3D_SUPPORT=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build with all available cores
echo ""
echo "🔨 Compiling (this may take a few minutes)..."
make -j$(nproc)

# Install
echo ""
echo "📦 Installing nvtop..."
sudo make install

# Verify installation
echo ""
echo "=========================================="
echo "✅ nvtop installation complete!"
echo "=========================================="
echo ""
echo "Verify installation:"
echo "  nvtop --version"
echo ""
echo "Usage:"
echo "  nvtop              # Launch GPU monitor"
echo "  nvtop --help       # Show help"
echo ""

# Clean up build directory (optional)
read -p "Remove build directory to save space? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning up build directory..."
    cd "$NVTOP_DIR"
    rm -rf build
    echo "✅ Build directory removed"
else
    echo "💡 Keeping build directory. Remove it later with: rm -rf $NVTOP_DIR/build"
fi

echo ""
echo "=========================================="
echo "nvtop is ready to use!"
echo "=========================================="

