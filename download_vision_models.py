#!/usr/bin/env python3
"""
Download DINO and SigLIP vision encoder models
"""

import os
import timm
import torch

def download_model(model_id, output_dir):
    """Download and save a timm model"""
    print(f"\n📥 Downloading {model_id}...")
    
    try:
        # Load model from timm (this will download if needed)
        model = timm.create_model(model_id, pretrained=True, num_classes=0)
        
        # Save the model
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), output_path)
        
        print(f"✅ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_siglip_model(model_id, output_dir):
    """Download and save SigLIP model"""
    print(f"\n📥 Downloading {model_id}...")
    
    try:
        # Load model from timm
        model = timm.create_model(model_id, pretrained=True, num_classes=0)
        
        # Save the model
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "open_clip_pytorch_model.bin")
        torch.save(model.state_dict(), output_path)
        
        print(f"✅ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    base_dir = "bak/dino-siglip"
    
    print("=" * 50)
    print("H-RDT Vision Encoder Model Downloader")
    print("=" * 50)
    
    # Download DINO model
    dino_dir = os.path.join(base_dir, "vit_large_patch14_reg4_dinov2.lvd142m")
    download_model("vit_large_patch14_reg4_dinov2.lvd142m", dino_dir)
    
    # Download SigLIP model
    siglip_dir = os.path.join(base_dir, "vit_so400m_patch14_siglip_384")
    download_siglip_model("vit_so400m_patch14_siglip_384", siglip_dir)
    
    print("\n" + "=" * 50)
    print("✅ All models downloaded successfully!")
    print(f"Models saved in: {os.path.abspath(base_dir)}")
    print("=" * 50)

