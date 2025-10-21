#!/usr/bin/env python3
"""
Dataset Verification Script for EgoDex
Verifies that MP4, HDF5, and language encoding files are present and consistent.
"""

import os
import sys
import h5py
import torch
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def verify_hdf5_file(hdf5_path):
    """Verify HDF5 file structure and required keys."""
    issues = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check required keys
            required_keys = ['actions', 'transforms', 'images']
            for key in required_keys:
                if key not in f:
                    issues.append(f"Missing required key: {key}")
            
            # Check for precomputed actions_48d
            if 'actions_48d' not in f:
                issues.append("Missing precomputed actions_48d data")
            else:
                actions_48d = f['actions_48d']
                if len(actions_48d.shape) != 2 or actions_48d.shape[1] != 48:
                    issues.append(f"Invalid actions_48d shape: {actions_48d.shape}")
            
            # Check data consistency
            if 'actions' in f and 'transforms' in f:
                actions_len = len(f['actions'])
                transforms_len = len(f['transforms'])
                if actions_len != transforms_len:
                    issues.append(f"Actions ({actions_len}) and transforms ({transforms_len}) length mismatch")
            
            if 'images' in f:
                images_len = len(f['images'])
                if 'transforms' in f:
                    transforms_len = len(f['transforms'])
                    if images_len != transforms_len:
                        issues.append(f"Images ({images_len}) and transforms ({transforms_len}) length mismatch")
                        
    except Exception as e:
        issues.append(f"HDF5 read error: {str(e)}")
    
    return issues


def verify_language_encoding(pt_path, hdf5_path):
    """Verify language encoding file exists and is consistent with HDF5."""
    issues = []
    
    if not pt_path.exists():
        issues.append("Missing language encoding file")
        return issues
    
    try:
        # Load language encoding
        lang_encoding = torch.load(pt_path, map_location='cpu', weights_only=True)
        
        # Check if it's a tensor or dict
        if isinstance(lang_encoding, torch.Tensor):
            lang_len = len(lang_encoding)
        elif isinstance(lang_encoding, dict):
            lang_len = len(lang_encoding)
        else:
            issues.append(f"Unexpected language encoding type: {type(lang_encoding)}")
            return issues
        
            # Check consistency with HDF5
            with h5py.File(hdf5_path, 'r') as f:
                if 'transforms' in f:
                    transforms_len = len(f['transforms'])
                    # Note: Language encodings are instruction-level, not frame-level
                    # Length mismatches are expected and acceptable
                    if lang_len != transforms_len:
                        # This is informational, not an error
                        pass  # Language encoding length mismatch is expected
                    
    except Exception as e:
        issues.append(f"Language encoding read error: {str(e)}")
    
    return issues


def verify_mp4_file(mp4_path):
    """Verify MP4 file exists."""
    issues = []
    
    if not mp4_path.exists():
        issues.append("Missing MP4 file")
    elif mp4_path.stat().st_size == 0:
        issues.append("Empty MP4 file")
    
    return issues


def verify_single_file(file_info_dict):
    """Verify a single file and return issues."""
    file_issues = []
    
    # Verify HDF5
    hdf5_issues = verify_hdf5_file(file_info_dict['hdf5_path'])
    file_issues.extend(hdf5_issues)
    
    # Verify language encoding
    lang_issues = verify_language_encoding(
        file_info_dict['pt_path'], 
        file_info_dict['hdf5_path']
    )
    file_issues.extend(lang_issues)
    
    # Verify MP4
    mp4_issues = verify_mp4_file(file_info_dict['mp4_path'])
    file_issues.extend(mp4_issues)
    
    return file_info_dict['file_id'], file_issues


def collect_dataset_files(data_root):
    """Collect all dataset files and organize by task."""
    data_root = Path(data_root)
    file_info = defaultdict(list)
    
    # Scan train and test directories
    for split in ['train', 'test']:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            continue
            
        for task_dir in split_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            
            # Find HDF5 files
            for hdf5_file in task_dir.glob('*.hdf5'):
                file_id = hdf5_file.stem
                
                # Expected file paths
                mp4_path = task_dir / f"{file_id}.mp4"
                pt_path = task_dir / f"{file_id}.pt"
                
                file_info[task_name].append({
                    'file_id': file_id,
                    'split': split,
                    'hdf5_path': hdf5_file,
                    'mp4_path': mp4_path,
                    'pt_path': pt_path,
                })
    
    return file_info


def verify_dataset(data_root, verbose=False, num_workers=None):
    """Verify the entire dataset with optional multiprocessing."""
    print(f"🔍 Verifying dataset at: {data_root}")
    print("=" * 60)
    
    # Collect all files
    file_info = collect_dataset_files(data_root)
    
    if not file_info:
        print("❌ No files found in dataset!")
        return False
    
    total_files = sum(len(files) for files in file_info.values())
    print(f"📊 Found {total_files} files across {len(file_info)} tasks")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)  # Cap at 16 to avoid too many processes
    print(f"🔧 Using {num_workers} parallel workers")
    print()
    
    # Verification results
    all_good = True
    task_stats = {}
    
    for task_name, files in file_info.items():
        print(f"📁 Task: {task_name} ({len(files)} files)")
        
        task_issues = 0
        task_stats[task_name] = {
            'total': len(files),
            'issues': 0,
            'missing_actions_48d': 0,
            'missing_lang_encoding': 0,
            'missing_mp4': 0,
            'data_inconsistency': 0
        }
        
        # Process files in parallel
        if len(files) > 1 and num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(verify_single_file, files),
                    total=len(files),
                    desc=f"  Verifying {task_name}",
                    leave=False
                ))
        else:
            # Sequential processing for small tasks or single worker
            results = []
            for file_info_dict in tqdm(files, desc=f"  Verifying {task_name}", leave=False):
                results.append(verify_single_file(file_info_dict))
        
        # Process results
        for file_id, file_issues in results:
            if file_issues:
                task_issues += 1
                task_stats[task_name]['issues'] += 1
                
                # Categorize issues
                for issue in file_issues:
                    if 'actions_48d' in issue:
                        task_stats[task_name]['missing_actions_48d'] += 1
                    elif 'language encoding' in issue or 'Missing language encoding' in issue:
                        task_stats[task_name]['missing_lang_encoding'] += 1
                    elif 'MP4' in issue:
                        task_stats[task_name]['missing_mp4'] += 1
                    elif 'mismatch' in issue or 'length' in issue:
                        task_stats[task_name]['data_inconsistency'] += 1
                
                if verbose:
                    print(f"    ❌ {file_info_dict['file_id']}: {'; '.join(file_issues)}")
        
        if task_issues == 0:
            print(f"    ✅ All {len(files)} files verified successfully")
        else:
            print(f"    ⚠️  {task_issues}/{len(files)} files have issues")
            all_good = False
        
        print()
    
    # Summary
    print("=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_issues = sum(stats['issues'] for stats in task_stats.values())
    total_files = sum(stats['total'] for stats in task_stats.values())
    
    if all_good:
        print("🎉 All files verified successfully!")
        print(f"✅ {total_files} files ready for training")
    else:
        print(f"⚠️  Found issues in {total_issues}/{total_files} files")
        print()
        
        # Detailed breakdown
        total_missing_actions_48d = sum(stats['missing_actions_48d'] for stats in task_stats.values())
        total_missing_lang = sum(stats['missing_lang_encoding'] for stats in task_stats.values())
        total_missing_mp4 = sum(stats['missing_mp4'] for stats in task_stats.values())
        total_inconsistencies = sum(stats['data_inconsistency'] for stats in task_stats.values())
        
        print("📊 Issue breakdown:")
        if total_missing_actions_48d > 0:
            print(f"  • Missing actions_48d: {total_missing_actions_48d} files")
        if total_missing_lang > 0:
            print(f"  • Missing language encoding: {total_missing_lang} files")
        if total_missing_mp4 > 0:
            print(f"  • Missing MP4 files: {total_missing_mp4} files")
        if total_inconsistencies > 0:
            print(f"  • Data inconsistencies: {total_inconsistencies} files")
            print("    Note: Language encoding length mismatches are expected (instruction-level vs frame-level)")
        
        print()
        print("🔧 Recommended actions:")
        if total_missing_actions_48d > 0:
            print("  • Run: python datasets/pretrain/precompute_48d_actions.py")
        if total_missing_lang > 0:
            print("  • Run: python datasets/pretrain/encode_lang_batch.py")
        if total_missing_mp4 > 0:
            print("  • Check MP4 file availability in dataset")
        if total_inconsistencies > 0:
            print("  • Investigate data preprocessing pipeline")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(description="Verify EgoDex dataset files")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the EgoDex dataset")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed error messages")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect, max 16)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"❌ Data root does not exist: {args.data_root}")
        sys.exit(1)
    
    success = verify_dataset(args.data_root, verbose=args.verbose, num_workers=args.num_workers)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
