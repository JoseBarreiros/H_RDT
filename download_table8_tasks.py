#!/usr/bin/env python3
"""
Download only the 13 tasks needed for Table 8 replication from RoboTwin2.0 dataset.
Focuses on aloha-agilex_clean_50 data.
"""

import os
import subprocess
from huggingface_hub import hf_hub_download, list_repo_files

# Table 8 tasks from the paper (exact names as they appear in the dataset)
TABLE8_TASKS = [
    "grab_roller",
    "handover_mic", 
    "lift_pot",
    "move_can_pot",
    "open_laptop",
    "pick_dual_bottles",
    "place_object_basket",
    "place_dual_shoes",
    "place_phone_stand",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "stack_blocks_two",
    "stack_bowls_two"
]

# Dataset repository
REPO_ID = "TianxingChen/RoboTwin2.0"

def find_table8_files():
    """Find the specific aloha-agilex_clean_50 files for Table 8 tasks"""
    print("🔍 Finding Table 8 task files for aloha-agilex_clean_50...")
    
    try:
        files = list_repo_files(REPO_ID, repo_type="dataset")
        
        table8_files = []
        for file in files:
            if file.startswith("dataset/") and file.endswith(".zip"):
                # Extract task name and platform info
                parts = file.split("/")
                task_path = parts[1]  # e.g., "grab_roller"
                filename = parts[2]   # e.g., "aloha-agilex_clean_50.zip"
                
                # Check if this is a Table 8 task and aloha-agilex_clean_50
                if task_path in TABLE8_TASKS and filename == "aloha-agilex_clean_50.zip":
                    table8_files.append((task_path, file))
                    print(f"   ✅ Found: {task_path} -> {file}")
        
        print(f"\n🎯 Found {len(table8_files)} Table 8 tasks for aloha-agilex_clean_50")
        return table8_files
        
    except Exception as e:
        print(f"❌ Error finding files: {e}")
        return []

def download_task(task_name, file_path, output_dir):
    """Download a specific task file"""
    print(f"📥 Downloading {task_name}...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"   ✅ Downloaded: {os.path.basename(downloaded_path)}")
        return downloaded_path
        
    except Exception as e:
        print(f"   ❌ Error downloading {task_name}: {e}")
        return None

def main():
    print("=" * 60)
    print("RoboTwin2.0 Table 8 Tasks Downloader")
    print("Downloading aloha-agilex_clean_50 data only")
    print("=" * 60)
    
    # Set output directory (local first, then we'll move to mounted disk)
    output_dir = "/home/jose-barreiros/H_RDT/table8_tasks"
    
    print(f"📁 Output directory: {output_dir}")
    
    # Find Table 8 files
    table8_files = find_table8_files()
    
    if not table8_files:
        print("❌ No Table 8 tasks found in the dataset")
        return
    
    # Download each task
    print(f"\n🚀 Starting download of {len(table8_files)} tasks...")
    
    downloaded_count = 0
    total_size = 0
    
    for task_name, file_path in table8_files:
        result = download_task(task_name, file_path, output_dir)
        if result:
            downloaded_count += 1
            # Get file size
            size_mb = os.path.getsize(result) / (1024 * 1024)
            total_size += size_mb
            print(f"   📊 Size: {size_mb:.1f} MB")
    
    print(f"\n✅ Successfully downloaded {downloaded_count}/{len(table8_files)} tasks")
    print(f"💾 Total size: {total_size:.1f} MB")
    print(f"📁 Files saved to: {output_dir}")
    
    # List downloaded files
    if os.path.exists(output_dir):
        print(f"\n📋 Downloaded files:")
        for file in sorted(os.listdir(output_dir)):
            if file.endswith('.zip'):
                file_path = os.path.join(output_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   - {file} ({size_mb:.1f} MB)")
    
    print(f"\n🎉 Download complete! You now have the 13 Table 8 tasks for aloha-agilex_clean_50")
    print(f"   These correspond to the tasks evaluated in Table 8 of the H-RDT paper")

if __name__ == "__main__":
    main()