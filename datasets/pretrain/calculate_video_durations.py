#!/usr/bin/env python3
"""
Calculate average video duration for EgoDex dataset
Uses either MP4 files directly (if ffprobe available) or estimates from HDF5 frame counts
"""

import os
import json
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
import argparse

def get_video_duration_ffprobe(video_path):
    """Get video duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None

def get_frame_count_hdf5(hdf5_path):
    """Get frame count from HDF5 file"""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'actions_48d' in f:
                return len(f['actions_48d'])
            elif 'transforms' in f:
                # Check transforms group
                transforms = f['transforms']
                if isinstance(transforms, h5py.Group):
                    # Get length from any body part
                    for key in ['leftHand', 'rightHand']:
                        if key in transforms:
                            return len(transforms[key])
                else:
                    return len(transforms)
    except Exception:
        pass
    return None

def estimate_duration_from_frames(num_frames, fps=30):
    """Estimate duration from frame count assuming constant FPS"""
    return num_frames / fps

def calculate_video_stats(data_root, output_path=None, use_ffprobe=True, sample_size=None):
    """
    Calculate average video duration for EgoDex dataset
    
    Args:
        data_root: Root directory of EgoDex dataset
        output_path: Output JSON file path (optional)
        use_ffprobe: Whether to use ffprobe for direct video analysis
        sample_size: Number of videos to sample (None = all)
    """
    root_path = Path(data_root)
    durations = []
    frame_counts = []
    fps_list = []
    
    # Find all MP4 files
    mp4_files = []
    for split in ['train', 'test']:
        split_dir = root_path / split
        if split_dir.exists():
            mp4_files.extend(list(split_dir.rglob('*.mp4')))
    
    print(f"Found {len(mp4_files)} MP4 files")
    
    if sample_size:
        import random
        random.seed(42)
        mp4_files = random.sample(mp4_files, min(sample_size, len(mp4_files)))
        print(f"Sampling {len(mp4_files)} videos")
    
    # Check if ffprobe is available
    ffprobe_available = use_ffprobe and subprocess.run(['which', 'ffprobe'], 
                                                       capture_output=True).returncode == 0
    
    if not ffprobe_available and use_ffprobe:
        print("Warning: ffprobe not found, will use HDF5 frame counts instead")
        ffprobe_available = False
    
    # Process videos
    for mp4_path in tqdm(mp4_files, desc="Processing videos"):
        duration = None
        
        # Try ffprobe first
        if ffprobe_available:
            duration = get_video_duration_ffprobe(mp4_path)
            if duration:
                durations.append(duration)
                # Try to get FPS
                try:
                    cmd = [
                        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=r_frame_rate', '-of',
                        'default=noprint_wrappers=1:nokey=1', str(mp4_path)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        fps_str = result.stdout.strip()
                        if '/' in fps_str:
                            num, den = map(int, fps_str.split('/'))
                            fps = num / den if den > 0 else 30
                            fps_list.append(fps)
                except:
                    pass
        
        # Fallback: use HDF5 frame count
        if duration is None:
            hdf5_path = mp4_path.with_suffix('.hdf5')
            if hdf5_path.exists():
                frame_count = get_frame_count_hdf5(hdf5_path)
                if frame_count:
                    frame_counts.append(frame_count)
                    # Estimate duration assuming 30 FPS (common for EgoDex)
                    duration = estimate_duration_from_frames(frame_count, fps=30)
                    durations.append(duration)
    
    if not durations:
        print("Error: No durations found!")
        return None
    
    # Calculate statistics
    durations = np.array(durations)
    stats = {
        "total_videos": len(durations),
        "average_duration_seconds": float(np.mean(durations)),
        "median_duration_seconds": float(np.median(durations)),
        "min_duration_seconds": float(np.min(durations)),
        "max_duration_seconds": float(np.max(durations)),
        "std_duration_seconds": float(np.std(durations)),
        "total_duration_hours": float(np.sum(durations) / 3600),
    }
    
    if frame_counts:
        frame_counts = np.array(frame_counts)
        stats["average_frames"] = float(np.mean(frame_counts))
        stats["median_frames"] = float(np.median(frame_counts))
        stats["total_frames"] = int(np.sum(frame_counts))
    
    if fps_list:
        fps_list = np.array(fps_list)
        stats["average_fps"] = float(np.mean(fps_list))
        stats["fps_samples"] = len(fps_list)
    
    # Print results
    print("\n" + "="*60)
    print("EgoDex Video Duration Statistics")
    print("="*60)
    print(f"Total videos analyzed: {stats['total_videos']:,}")
    print(f"\nDuration Statistics:")
    print(f"  Average: {stats['average_duration_seconds']:.2f} seconds ({stats['average_duration_seconds']/60:.2f} minutes)")
    print(f"  Median:  {stats['median_duration_seconds']:.2f} seconds ({stats['median_duration_seconds']/60:.2f} minutes)")
    print(f"  Min:     {stats['min_duration_seconds']:.2f} seconds")
    print(f"  Max:     {stats['max_duration_seconds']:.2f} seconds ({stats['max_duration_seconds']/60:.2f} minutes)")
    print(f"  Std Dev: {stats['std_duration_seconds']:.2f} seconds")
    print(f"\nTotal duration: {stats['total_duration_hours']:.2f} hours")
    
    if 'average_frames' in stats:
        print(f"\nFrame Statistics:")
        print(f"  Average frames: {stats['average_frames']:.1f}")
        print(f"  Median frames:  {stats['median_frames']:.1f}")
        print(f"  Total frames:   {stats['total_frames']:,}")
    
    if 'average_fps' in stats:
        print(f"\nFPS Statistics:")
        print(f"  Average FPS: {stats['average_fps']:.2f} (from {stats['fps_samples']} samples)")
    
    print("="*60)
    
    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {output_path}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Calculate EgoDex video durations")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("EGODEX_DATA_ROOT", "/home/jose-barreiros/egodex/organized"),
        help="EgoDex dataset root directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON file path (optional)"
    )
    parser.add_argument(
        "--no-ffprobe",
        action="store_true",
        help="Don't use ffprobe, only use HDF5 frame counts"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size (for faster analysis, None = all videos)"
    )
    
    args = parser.parse_args()
    
    stats = calculate_video_stats(
        data_root=args.data_root,
        output_path=args.output_path,
        use_ffprobe=not args.no_ffprobe,
        sample_size=args.sample_size
    )
    
    if stats is None:
        return 1
    return 0

if __name__ == "__main__":
    exit(main())

