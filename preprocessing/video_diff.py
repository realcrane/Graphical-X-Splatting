import os
import argparse
import sys
from pathlib import Path
import tempfile
import shutil
import cv2
import numpy as np

from _common import do_system

def extract_frames(video_path, output_dir):
    """Extract all frames from a video to a directory using ffmpeg"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = f'ffmpeg -i "{video_path}" -q:v 1 "{output_dir}/frame_%06d.png"'
    do_system(cmd)
    
    # Return list of extracted frame files
    frame_files = sorted(list(output_dir.glob("frame_*.png")))
    return frame_files

def frames_to_video(frame_dir, output_video_path, fps=30):
    """Convert frames back to video using ffmpeg"""
    frame_pattern = f"{frame_dir}/frame_%06d.png"
    cmd = f'ffmpeg -r {fps} -i "{frame_pattern}" -c:v libx264 -crf 18 -pix_fmt yuv420p -y "{output_video_path}"'
    do_system(cmd)

def get_video_fps(video_path):
    """Get FPS of a video using ffprobe"""
    import subprocess
    import json
    
    try:
        cmd = f'ffprobe -v quiet -print_format json -show_streams "{video_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                fps_str = stream['r_frame_rate']
                num, den = map(int, fps_str.split('/'))
                return num / den
        return 30  # default fallback
    except:
        return 30  # default fallback

def process_frame_conditional(frame_a_path, frame_b_path, output_path, threshold=0.01):
    """
    Process two frames with conditional logic:
    If |A - B| > threshold, use A pixel value, otherwise use 0 (black).
    """
    # Read frames
    frame_a = cv2.imread(str(frame_a_path))
    frame_b = cv2.imread(str(frame_b_path))
    
    if frame_a is None or frame_b is None:
        print(f"Error reading frames: {frame_a_path}, {frame_b_path}")
        return False
    
    # Ensure frames have same dimensions
    if frame_a.shape != frame_b.shape:
        print(f"Frame dimension mismatch: {frame_a.shape} vs {frame_b.shape}")
        return False
    
    # Convert to float for calculations
    frame_a_float = frame_a.astype(np.float32) / 255.0
    frame_b_float = frame_b.astype(np.float32) / 255.0
    
    # Compute absolute difference
    diff = np.abs(frame_a_float - frame_b_float)
    
    # Create mask: True where any channel difference > threshold
    mask = np.any(diff > threshold, axis=2)
    
    # Create output frame
    output_frame = np.zeros_like(frame_a)
    
    # Where mask is True, use original A frame pixels
    output_frame[mask] = frame_a[mask]
    
    # Save result
    cv2.imwrite(str(output_path), output_frame)
    return True

def compute_video_difference_conditional(video_a_path, video_b_path, output_path, threshold=0.01):
    """
    Compute conditional difference between two videos by processing frames individually.
    """
    print(f"Processing: {os.path.basename(video_a_path)} - {os.path.basename(video_b_path)} (threshold: {threshold})")
    
    # Get video FPS
    fps = get_video_fps(video_a_path)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create subdirectories for frames
        frames_a_dir = tmp_path / "frames_a"
        frames_b_dir = tmp_path / "frames_b"
        frames_out_dir = tmp_path / "frames_out"
        
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        
        print("  Extracting frames from video A...")
        frames_a = extract_frames(video_a_path, frames_a_dir)
        
        print("  Extracting frames from video B...")
        frames_b = extract_frames(video_b_path, frames_b_dir)
        
        if len(frames_a) != len(frames_b):
            print(f"  Warning: Frame count mismatch - A: {len(frames_a)}, B: {len(frames_b)}")
            min_frames = min(len(frames_a), len(frames_b))
            frames_a = frames_a[:min_frames]
            frames_b = frames_b[:min_frames]
        
        print(f"  Processing {len(frames_a)} frames...")
        
        # Process each frame pair
        for i, (frame_a, frame_b) in enumerate(zip(frames_a, frames_b)):
            output_frame = frames_out_dir / f"frame_{i+1:06d}.png"
            
            success = process_frame_conditional(frame_a, frame_b, output_frame, threshold)
            if not success:
                print(f"  Failed to process frame {i+1}")
                return False
            
            if (i + 1) % 30 == 0:
                print(f"  Processed {i+1}/{len(frames_a)} frames")
        
        print("  Converting frames back to video...")
        frames_to_video(frames_out_dir, output_path, fps)
        
    return True

def process_video_directories(dir_a, dir_b, output_dir, threshold=0.01):
    """
    Process all corresponding videos in two directories.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get video files from directory A
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.MP4', '.AVI', '.MOV'}
    dir_a_path = Path(dir_a)
    dir_b_path = Path(dir_b)
    
    video_files_a = [f for f in dir_a_path.iterdir() 
                     if f.is_file() and f.suffix in video_extensions]
    
    if not video_files_a:
        print(f"No video files found in {dir_a}")
        return
    
    processed_count = 0
    
    for video_file_a in sorted(video_files_a):
        # Look for corresponding file in directory B
        video_file_b = dir_b_path / video_file_a.name
        
        if not video_file_b.exists():
            print(f"Warning: Corresponding video {video_file_b} not found, skipping {video_file_a.name}")
            continue
        
        # Create output path
        output_file = Path(output_dir) / video_file_a.name

        # Compute conditional difference
        success = compute_video_difference_conditional(
            str(video_file_a), 
            str(video_file_b), 
            str(output_file),
            threshold
        )
        
        if success:
            processed_count += 1
        else:
            print(f"Failed to process {video_file_a.name}")
    
    print(f"\nCompleted! Processed {processed_count} video pairs.")

def main():
    parser = argparse.ArgumentParser(description="Compute conditional frame-wise difference between corresponding videos in two directories")
    parser.add_argument("dir_a", help="Path to first video directory (A)")
    parser.add_argument("dir_b", help="Path to second video directory (B)")
    parser.add_argument("output_dir", help="Path to output directory (O)")
    parser.add_argument("--threshold", type=float, default=0.01, 
                       help="Threshold for difference comparison (default: 0.01)")
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.isdir(args.dir_a):
        print(f"Error: Directory A '{args.dir_a}' does not exist")
        return
    
    if not os.path.isdir(args.dir_b):
        print(f"Error: Directory B '{args.dir_b}' does not exist")
        return
    
    # Validate threshold
    if args.threshold < 0 or args.threshold > 1:
        print(f"Error: Threshold must be between 0 and 1, got {args.threshold}")
        return
    
    print(f"Computing conditional differences between videos in:")
    print(f"  Directory A: {args.dir_a}")
    print(f"  Directory B: {args.dir_b}")
    print(f"  Output: {args.output_dir}")
    print(f"  Threshold: {args.threshold}")
    print()
    
    process_video_directories(args.dir_a, args.dir_b, args.output_dir, args.threshold)

if __name__ == "__main__":
    main()