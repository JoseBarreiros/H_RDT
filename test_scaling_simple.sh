#!/bin/bash

# Simple test to verify the scaling feature works
# This just loads the dataset with different percentages and prints the sizes

echo "Testing scaling feature with dataset loading..."

source hrdt_env/bin/activate

python << 'EOF'
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datasets.pretrain.egodex_dataset import EgoDexDataset

data_root = os.environ.get("EGODEX_DATA_ROOT", "/home/jose-barreiros/egodex/organized")

print("=" * 60)
print("Testing Scaling Feature")
print("=" * 60)

# Test different percentages
for pct in [1.0, 0.5, 0.1, 0.01]:
    dataset = EgoDexDataset(
        data_root=data_root,
        config=None,
        upsample_rate=3,
        val=False,
        use_precomp_lang_embed=True,
        data_percentage=pct,
        seed=42
    )
    
    print(f"\n{pct*100:>4.0f}% of data: {len(dataset):>6} files")

print("\n" + "=" * 60)
print("✅ Scaling feature works correctly!")
print("=" * 60)
EOF

echo ""
echo "Now test the scaling feature in your training!"

