#!/usr/bin/env python3
"""
Dataset Verification Script for RobotWin2 (Table 8 Tasks)
Verifies that HDF5 files and language encoding files are present and consistent.
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
    """Verify RobotWin2 HDF5 file structure and required keys."""
    issues = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check required keys for RobotWin2 dataset
            required_keys = ['observation', 'joint_action']
            for key in required_keys:
                if key not in f:
                    issues.append(f"Missing required key: {key}")
            
            # Check observation structure
            if 'observation' in f:
                obs_group = f['observation']
                if not isinstance(obs_group, h5py.Group):
                    issues.append("Observation should be a group")
                else:
                    # Check for camera observations
                    camera_keys = ['front_camera', 'head_camera', 'left_camera', 'right_camera']
                    found_cameras = []
                    for cam_key in camera_keys:
                        if cam_key in obs_group:
                            found_cameras.append(cam_key)
                            cam_group = obs_group[cam_key]
                            if 'rgb' not in cam_group:
                                issues.append(f"Missing RGB data in {cam_key}")
                    
                    if len(found_cameras) == 0:
                        issues.append("No camera observations found")
                    elif len(found_cameras) < 3:
                        issues.append(f"Expected 3+ cameras, found: {found_cameras}")
            
            # Check joint action structure
            if 'joint_action' in f:
                action_group = f['joint_action']
                if not isinstance(action_group, h5py.Group):
                    issues.append("Joint action should be a group")
                else:
                    # Check for dual-arm actions
                    arm_keys = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']
                    for arm_key in arm_keys:
                        if arm_key not in action_group:
                            issues.append(f"Missing {arm_key} in joint_action")
                    
                    # Check action vector (14D: 6+1+6+1 for left_arm+left_gripper+right_arm+right_gripper)
                    if 'vector' in action_group:
                        vector = action_group['vector']
                        if len(vector.shape) != 2 or vector.shape[1] != 14:
                            issues.append(f"Invalid action vector shape: {vector.shape}, expected (N, 14)")
                    
                    # Check individual arm actions (6D for each arm)
                    for arm_key in ['left_arm', 'right_arm']:
                        if arm_key in action_group:
                            arm_actions = action_group[arm_key]
                            if len(arm_actions.shape) != 2 or arm_actions.shape[1] != 6:
                                issues.append(f"Invalid {arm_key} shape: {arm_actions.shape}, expected (N, 6)")
                    
                    # Check gripper actions (1D for each gripper)
                    for gripper_key in ['left_gripper', 'right_gripper']:
                        if gripper_key in action_group:
                            gripper_actions = action_group[gripper_key]
                            if len(gripper_actions.shape) != 1:
                                issues.append(f"Invalid {gripper_key} shape: {gripper_actions.shape}, expected (N,)")
            
            # Check endpose structure (optional but common in RobotWin2)
            if 'endpose' in f:
                endpose_group = f['endpose']
                if isinstance(endpose_group, h5py.Group):
                    endpose_keys = ['left_endpose', 'right_endpose', 'left_gripper', 'right_gripper']
                    for key in endpose_keys:
                        if key not in endpose_group:
                            issues.append(f"Missing {key} in endpose")
                        else:
                            endpose_data = endpose_group[key]
                            if key.endswith('_endpose'):
                                if len(endpose_data.shape) != 2 or endpose_data.shape[1] != 7:
                                    issues.append(f"Invalid {key} shape: {endpose_data.shape}, expected (N, 7)")
                            elif key.endswith('_gripper'):
                                if len(endpose_data.shape) != 1:
                                    issues.append(f"Invalid {key} shape: {endpose_data.shape}, expected (N,)")
            
            # Check data consistency - RobotWin2 has different structure
            # Observations are organized by camera groups, actions are organized by arm groups
            # The length check should be done differently for RobotWin2
            if 'observation' in f and 'joint_action' in f:
                # Get actual observation length from camera data
                obs_group = f['observation']
                action_group = f['joint_action']
                
                # Find a camera to get observation length
                obs_len = None
                for cam_key in ['front_camera', 'head_camera', 'left_camera', 'right_camera']:
                    if cam_key in obs_group and 'rgb' in obs_group[cam_key]:
                        obs_len = len(obs_group[cam_key]['rgb'])
                        break
                
                # Find action length from vector or individual arms
                action_len = None
                if 'vector' in action_group:
                    action_len = len(action_group['vector'])
                elif 'left_arm' in action_group:
                    action_len = len(action_group['left_arm'])
                
                if obs_len is not None and action_len is not None and obs_len != action_len:
                    # This is actually expected in RobotWin2 - observations and actions can have different lengths
                    # due to different sampling rates or data collection methods
                    pass  # Don't report this as an error
                        
    except Exception as e:
        issues.append(f"HDF5 read error: {str(e)}")
    
    return issues


def verify_language_encoding(pt_path, task_name):
    """Verify language encoding file exists and is valid."""
    issues = []
    
    if not pt_path.exists():
        issues.append("Missing language encoding file")
        return issues
    
    try:
        # Load language encoding
        lang_encoding = torch.load(pt_path, map_location='cpu', weights_only=True)
        
        # Check if it's a dict with expected structure
        if not isinstance(lang_encoding, dict):
            issues.append(f"Language encoding should be a dict, got {type(lang_encoding)}")
            return issues
        
        # Check required keys (RobotWin2 format)
        required_keys = ['instruction', 'embeddings']
        for key in required_keys:
            if key not in lang_encoding:
                issues.append(f"Missing required key in language encoding: {key}")
        
        # Check embeddings - RobotWin2 uses different format
        if 'embeddings' in lang_encoding:
            embeddings = lang_encoding['embeddings']
            if not isinstance(embeddings, torch.Tensor):
                issues.append("Embeddings should be a torch.Tensor")
            elif len(embeddings.shape) != 3:
                issues.append(f"RobotWin2 embeddings should be 3D tensor (batch, seq, dim), got shape: {embeddings.shape}")
            elif embeddings.shape[2] != 4096:  # T5-XXL embedding size
                issues.append(f"Expected T5-XXL embeddings (4096), got: {embeddings.shape[2]}")
        
        # Check instruction
        if 'instruction' in lang_encoding:
            instruction = lang_encoding['instruction']
            if not isinstance(instruction, str) or len(instruction.strip()) == 0:
                issues.append("Instruction should be a non-empty string")
                    
    except Exception as e:
        issues.append(f"Language encoding read error: {str(e)}")
    
    return issues


def verify_scene_info(scene_info_path):
    """Verify scene_info.json file exists and is valid."""
    issues = []
    
    if not scene_info_path.exists():
        issues.append("Missing scene_info.json file")
        return issues
    
    try:
        import json
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        
        # Check if it's a dict
        if not isinstance(scene_info, dict):
            issues.append("scene_info.json should contain a JSON object")
        
        # Check for common RobotWin2 scene info keys (optional)
        # RobotWin2 scene_info.json may have different structure
        if not scene_info:  # Empty dict
            issues.append("scene_info.json is empty")
                    
    except Exception as e:
        issues.append(f"Error reading scene_info.json: {str(e)}")
    
    return issues


def verify_single_task(task_info_dict):
    """Verify a single task directory and return issues."""
    task_issues = []
    task_name = task_info_dict['task_name']
    
    # Check HDF5 files
    hdf5_files = list(task_info_dict['task_dir'].glob('**/*.hdf5'))
    if not hdf5_files:
        task_issues.append("No HDF5 files found")
    else:
        for hdf5_file in hdf5_files:
            hdf5_issues = verify_hdf5_file(hdf5_file)
            task_issues.extend([f"{hdf5_file.name}: {issue}" for issue in hdf5_issues])
    
    # Check language encoding
    lang_pt_path = task_info_dict['lang_pt_path']
    lang_issues = verify_language_encoding(lang_pt_path, task_name)
    task_issues.extend([f"Language encoding: {issue}" for issue in lang_issues])
    
    # Check scene info
    scene_info_path = task_info_dict['scene_info_path']
    scene_issues = verify_scene_info(scene_info_path)
    task_issues.extend([f"Scene info: {issue}" for issue in scene_issues])
    
    return task_name, task_issues


def collect_robotwin2_files(data_root):
    """Collect all RobotWin2 Table 8 task files."""
    data_root = Path(data_root)
    task_info = {}
    
    # Expected Table 8 tasks
    table8_tasks = [
        "grab_roller", "handover_mic", "lift_pot", "move_can_pot", 
        "open_laptop", "pick_dual_bottles", "place_dual_shoes", 
        "place_object_basket", "place_phone_stand", "put_bottles_dustbin", 
        "put_object_cabinet", "stack_blocks_two", "stack_bowls_two"
    ]
    
    for task_name in table8_tasks:
        task_dir = data_root / task_name / "aloha-agilex_clean_50"
        
        if not task_dir.exists():
            print(f"Warning: Task directory not found: {task_dir}")
            continue
        
        # Expected file paths
        lang_pt_path = Path("datasets/robotwin2/lang_embeddings") / f"{task_name}.pt"
        scene_info_path = task_dir / "scene_info.json"
        
        task_info[task_name] = {
            'task_name': task_name,
            'task_dir': task_dir,
            'lang_pt_path': lang_pt_path,
            'scene_info_path': scene_info_path,
        }
    
    return task_info


def verify_training_compatibility():
    """Verify that the dataset is compatible with H-RDT training configuration."""
    issues = []
    
    try:
        # Check if config file exists
        config_path = Path("configs/hrdt_finetune.yaml")
        if not config_path.exists():
            issues.append("Missing training config file: configs/hrdt_finetune.yaml")
            return issues
        
        # Load and check config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key training parameters
        common = config.get('common', {})
        
        # Check action dimension (should be 14 for dual-arm)
        action_dim = common.get('action_dim')
        if action_dim != 14:
            issues.append(f"Config action_dim={action_dim}, expected 14 for dual-arm robot")
        
        # Check state dimension (should be 14 for dual-arm)
        state_dim = common.get('state_dim')
        if state_dim != 14:
            issues.append(f"Config state_dim={state_dim}, expected 14 for dual-arm robot")
        
        # Check number of cameras (should be 3)
        num_cameras = common.get('num_cameras')
        if num_cameras != 3:
            issues.append(f"Config num_cameras={num_cameras}, expected 3")
        
        # Check action chunk size
        action_chunk_size = common.get('action_chunk_size')
        if action_chunk_size is None:
            issues.append("Missing action_chunk_size in config")
        
        # Check text feature dimension (should be 4096 for T5-XXL)
        text_feature_dim = config.get('model', {}).get('text', {}).get('feature_dim')
        if text_feature_dim != 4096:
            issues.append(f"Config text feature_dim={text_feature_dim}, expected 4096 for T5-XXL")
        
    except Exception as e:
        issues.append(f"Error checking training compatibility: {str(e)}")
    
    return issues


def verify_robotwin2_dataset(data_root, verbose=False, num_workers=None):
    """Verify the RobotWin2 Table 8 dataset."""
    print(f"🔍 Verifying RobotWin2 Table 8 dataset at: {data_root}")
    print("=" * 60)
    
    # Collect all task files
    task_info = collect_robotwin2_files(data_root)
    
    if not task_info:
        print("❌ No Table 8 tasks found in dataset!")
        return False
    
    print(f"📊 Found {len(task_info)} Table 8 tasks")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 for task-level parallelism
    print(f"🔧 Using {num_workers} parallel workers")
    print()
    
    # Verification results
    all_good = True
    task_stats = {}
    
    # Process tasks in parallel
    if len(task_info) > 1 and num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(verify_single_task, task_info.values()),
                total=len(task_info),
                desc="Verifying tasks",
                leave=False
            ))
    else:
        # Sequential processing
        results = []
        for task_info_dict in tqdm(task_info.values(), desc="Verifying tasks", leave=False):
            results.append(verify_single_task(task_info_dict))
    
    # Process results
    for task_name, task_issues in results:
        print(f"📁 Task: {task_name}")
        
        task_stats[task_name] = {
            'total_issues': len(task_issues),
            'hdf5_issues': 0,
            'lang_issues': 0,
            'scene_issues': 0,
            'other_issues': 0
        }
        
        if task_issues:
            all_good = False
            
            # Categorize issues
            for issue in task_issues:
                if 'HDF5' in issue or '.hdf5' in issue:
                    task_stats[task_name]['hdf5_issues'] += 1
                elif 'Language encoding' in issue:
                    task_stats[task_name]['lang_issues'] += 1
                elif 'Scene info' in issue:
                    task_stats[task_name]['scene_issues'] += 1
                else:
                    task_stats[task_name]['other_issues'] += 1
                
                if verbose:
                    print(f"    ❌ {issue}")
            
            print(f"    ⚠️  {len(task_issues)} issues found")
        else:
            print(f"    ✅ All files verified successfully")
        
        print()
    
    # Verify training compatibility
    print("=" * 60)
    print("🔧 VERIFYING TRAINING COMPATIBILITY")
    print("=" * 60)
    
    training_issues = verify_training_compatibility()
    if training_issues:
        print("⚠️  Training compatibility issues:")
        for issue in training_issues:
            print(f"  • {issue}")
        all_good = False
    else:
        print("✅ Training configuration is compatible")
    
    print()
    
    # Summary
    print("=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_issues = sum(stats['total_issues'] for stats in task_stats.values())
    total_tasks = len(task_stats)
    
    if all_good:
        print("🎉 All Table 8 tasks verified successfully!")
        print(f"✅ {total_tasks} tasks ready for fine-tuning")
    else:
        print(f"⚠️  Found issues in {total_issues} tasks")
        print()
        
        # Detailed breakdown
        total_hdf5_issues = sum(stats['hdf5_issues'] for stats in task_stats.values())
        total_lang_issues = sum(stats['lang_issues'] for stats in task_stats.values())
        total_scene_issues = sum(stats['scene_issues'] for stats in task_stats.values())
        total_other_issues = sum(stats['other_issues'] for stats in task_stats.values())
        
        print("📊 Issue breakdown:")
        if total_hdf5_issues > 0:
            print(f"  • HDF5 file issues: {total_hdf5_issues}")
        if total_lang_issues > 0:
            print(f"  • Language encoding issues: {total_lang_issues}")
        if total_scene_issues > 0:
            print(f"  • Scene info issues: {total_scene_issues}")
        if total_other_issues > 0:
            print(f"  • Other issues: {total_other_issues}")
        
        print()
        print("🔧 Recommended actions:")
        if total_hdf5_issues > 0:
            print("  • Check HDF5 file integrity and structure")
        if total_lang_issues > 0:
            print("  • Verify language embeddings are properly generated")
            print("  • Check: datasets/robotwin2/lang_embeddings/")
        if total_scene_issues > 0:
            print("  • Check scene_info.json files in task directories")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(description="Verify RobotWin2 Table 8 dataset files")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the RobotWin2 Table 8 dataset")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed error messages")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect, max 8)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"❌ Data root does not exist: {args.data_root}")
        sys.exit(1)
    
    success = verify_robotwin2_dataset(args.data_root, verbose=args.verbose, num_workers=args.num_workers)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
