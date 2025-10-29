#!/usr/bin/env python3
"""
Extract all Table 8 task zip files and organize the data structure.
"""

import os
import zipfile
import shutil
from pathlib import Path

def extract_all_tasks():
    """Extract all Table 8 task zip files"""
    
    # Source and destination directories
    source_dir = "/mnt/disks/hrdt-data/robotwin2/table8_tasks/dataset"
    dest_dir = "/mnt/disks/hrdt-data/robotwin2/table8_tasks/extracted"
    
    print("=" * 60)
    print("Table 8 Tasks Data Extraction")
    print("=" * 60)
    
    print(f"📁 Source directory: {source_dir}")
    print(f"📁 Destination directory: {dest_dir}")
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    task_dirs.sort()
    
    print(f"\n🔍 Found {len(task_dirs)} task directories:")
    for task_dir in task_dirs:
        print(f"   - {task_dir}")
    
    print(f"\n🚀 Starting extraction...")
    
    extracted_count = 0
    total_size = 0
    
    for task_dir in task_dirs:
        task_path = os.path.join(source_dir, task_dir)
        zip_file = os.path.join(task_path, "aloha-agilex_clean_50.zip")
        
        if os.path.exists(zip_file):
            print(f"\n📦 Extracting {task_dir}...")
            
            try:
                # Create task directory in destination
                task_dest = os.path.join(dest_dir, task_dir)
                os.makedirs(task_dest, exist_ok=True)
                
                # Extract zip file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(task_dest)
                
                # Get extracted size
                extracted_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                   for dirpath, dirnames, filenames in os.walk(task_dest)
                                   for filename in filenames)
                extracted_size_mb = extracted_size / (1024 * 1024)
                total_size += extracted_size_mb
                
                print(f"   ✅ Extracted: {extracted_size_mb:.1f} MB")
                extracted_count += 1
                
                # List extracted contents
                extracted_files = []
                for root, dirs, files in os.walk(task_dest):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), task_dest)
                        extracted_files.append(rel_path)
                
                print(f"   📋 Files: {len(extracted_files)} files")
                if len(extracted_files) <= 10:  # Show files if not too many
                    for file in extracted_files[:5]:
                        print(f"      - {file}")
                    if len(extracted_files) > 5:
                        print(f"      ... and {len(extracted_files) - 5} more files")
                
            except Exception as e:
                print(f"   ❌ Error extracting {task_dir}: {e}")
        else:
            print(f"   ⚠️  Zip file not found for {task_dir}")
    
    print(f"\n✅ Successfully extracted {extracted_count}/{len(task_dirs)} tasks")
    print(f"💾 Total extracted size: {total_size:.1f} MB")
    print(f"📁 Extracted data location: {dest_dir}")
    
    # Show final structure
    print(f"\n📋 Final directory structure:")
    for task_dir in sorted(os.listdir(dest_dir)):
        task_path = os.path.join(dest_dir, task_dir)
        if os.path.isdir(task_path):
            file_count = sum(len(files) for _, _, files in os.walk(task_path))
            print(f"   📁 {task_dir}/ ({file_count} files)")
    
    return dest_dir

def main():
    extracted_dir = extract_all_tasks()
    
    print(f"\n🎉 Extraction complete!")
    print(f"📁 All Table 8 task data extracted to: {extracted_dir}")
    print(f"\n💡 Next steps:")
    print(f"   1. Verify the extracted data structure")
    print(f"   2. Update your dataset configuration to point to: {extracted_dir}")
    print(f"   3. Run fine-tuning with the robotwin_agilex dataset")

if __name__ == "__main__":
    main()
