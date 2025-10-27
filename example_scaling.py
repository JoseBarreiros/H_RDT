#!/usr/bin/env python3
"""
Example script demonstrating how to use data_percentage for scaling law experiments.

This script shows how to load the dataset with different percentages of data
and demonstrates reproducible subsampling with seeds.
"""

import os
import json
from datasets.dataset import VLAConsumerDataset
from torchvision import transforms
import yaml

def load_config(config_path="configs/hrdt_pretrain.yaml"):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_image_transform():
    """Create image transform pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])

def main():
    print("=" * 60)
    print("H-RDT Scaling Law Example")
    print("=" * 60)
    print()
    
    # Load configuration
    config = load_config()
    image_transform = create_image_transform()
    
    # Test different percentages
    percentages = [1.0, 0.5, 0.1, 0.05, 0.01]
    
    print("Testing different data percentages...")
    print()
    
    for pct in percentages:
        print(f"📊 Loading {pct*100:.0f}% of data (seed=42):")
        print("-" * 40)
        
        # Create dataset with this percentage
        dataset = VLAConsumerDataset(
            config=config,
            image_transform=image_transform,
            num_cameras=config["common"]["num_cameras"],
            dataset_type="pretrain",
            image_aug=False,
            upsample_rate=3,
            val=False,
            use_precomp_lang_embed=True,
            dataset_name="egodex",
            data_percentage=pct,
            seed=42,  # Fixed seed for reproducibility
        )
        
        print(f"  Total files in dataset: {len(dataset)}")
        print(f"  This is {pct*100:.1f}% of the full dataset")
        print()
    
    print("=" * 60)
    print("✅ Scaling law example complete!")
    print("=" * 60)
    print()
    print("Key points:")
    print("  • Same seed (42) = same random subset")
    print("  • Different percentages = different dataset sizes")
    print("  • Use --data_percentage flag when training")
    print()
    print("Example training commands:")
    print("  # Train with 10% of data:")
    print("  accelerate launch train/train.py \\")
    print("      --config_path configs/hrdt_pretrain.yaml \\")
    print("      --data_percentage 0.1 \\")
    print("      --seed 42")
    print()
    print("  # Or run automated scaling experiments:")
    print("  ./run_scaling_law_experiments.sh")

if __name__ == "__main__":
    main()

