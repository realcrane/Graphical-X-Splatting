"""Generate one point cloud per frame from a Google Immersive capture.

Expected layout::

    base_dir/colmap_<frame>/manual/{cameras.txt, images.txt}
    base_dir/colmap_<frame>/images/camera_XXXX.png

For each frame the camera views (``camera_XXXX.png``) are matched pairwise with
ROMA and triangulated into a PLY. See pcd_from_images_n3dv.py for the N3DV
(``camXX_YYYY.png``) variant. Shared geometry/IO helpers live in _common.py.
"""

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree

from romatch import roma_indoor as matcher

from _common import (storePly, read_cameras_txt, read_images_txt, get_camera_matrix,
                     qvec2rotmat, triangulate_points, get_image_colors)


def organize_images_by_camera_immersive(images_dir, frame_id):
    """Find ``camera_XXXX.{png,jpg}`` files in a frame -> {camera_id: path}."""
    cameras = {}
    images_dir = Path(images_dir)
    for image_file in sorted(images_dir.glob("camera_*")):
        if image_file.is_file():
            match = re.match(r'camera_(\d+)\.(png|jpg)', image_file.name)
            if match:
                cameras[int(match.group(1))] = image_file
    return cameras


def generate_point_cloud_per_frame_immersive(base_dir, output_dir, device,
                                              frames_to_process=None, min_certainty=0.5,
                                              match_threshold=1000, max_views=None, max_points=None,
                                              outlier_threshold=None, search_radius=0.1, min_neighbors=5):
    """Generate one point cloud per frame using the Immersive folder structure."""
    base_dir = Path(base_dir)

    # Find all colmap folders
    colmap_folders = sorted([f for f in base_dir.glob("colmap_*") if f.is_dir()])

    if len(colmap_folders) == 0:
        raise RuntimeError(f"No colmap_X folders found in {base_dir}")

    # Extract frame numbers
    frame_numbers = []
    for folder in colmap_folders:
        match = re.match(r'colmap_(\d+)', folder.name)
        if match:
            frame_numbers.append(int(match.group(1)))

    # Filter frames if specified
    if frames_to_process is not None:
        frame_numbers = [f for f in frame_numbers if f in frames_to_process]

    print(f"Found {len(frame_numbers)} frames to process: {sorted(frame_numbers)}")

    # Initialize ROMA model
    print("Loading ROMA model...")
    roma_model = matcher(device=device)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    print("\nProcessing frames...")
    for frame_id in tqdm(sorted(frame_numbers), ncols=80):
        # Check if point cloud already exists for this frame
        output_file = output_dir / f"frame_{frame_id:04d}.ply"
        if output_file.exists():
            print(f"\nFrame {frame_id}: Point cloud already exists, skipping...")
            continue

        # Get paths for this frame
        colmap_dir = base_dir / f"colmap_{frame_id}"
        cameras_file = colmap_dir / "manual" / "cameras.txt"
        images_file = colmap_dir / "manual" / "images.txt"
        images_dir = colmap_dir / "images"

        if not cameras_file.exists() or not images_file.exists():
            print(f"\nFrame {frame_id}: COLMAP files not found in {colmap_dir}, skipping...")
            continue

        # Read COLMAP files for this frame (poses keyed by the camera_XXXX number).
        cameras = read_cameras_txt(cameras_file)
        camera_poses = read_images_txt(
            images_file,
            lambda name: int(re.search(r'camera_(\d+)', name).group(1))
            if re.search(r'camera_(\d+)', name) else None)

        print(f"\nFrame {frame_id}: Found {len(cameras)} COLMAP cameras, {len(camera_poses)} camera poses")

        # Organize images by camera for this frame
        frame_cameras = organize_images_by_camera_immersive(images_dir, frame_id)

        print(f"Frame {frame_id}: Found {len(frame_cameras)} camera images in {images_dir}")
        if len(frame_cameras) == 0:
            print(f"Frame {frame_id}: No camera images found! Checking directory structure...")
            # Debug: list what's actually in the directory
            if images_dir.exists():
                image_files = list(images_dir.glob("camera_*"))
                print(f"  Found {len(image_files)} camera_* files")
                if image_files:
                    print(f"  Sample files: {[f.name for f in image_files[:5]]}")
            else:
                print(f"  Images directory does not exist: {images_dir}")
            continue

        camera_ids = sorted(frame_cameras.keys())

        # Limit number of views if specified
        if max_views is not None and len(camera_ids) > max_views:
            camera_ids = camera_ids[:max_views]
            print(f"\nFrame {frame_id}: Limited to {max_views} cameras out of {len(frame_cameras)}")

        if len(camera_ids) < 2:
            print(f"\nFrame {frame_id}: Only {len(camera_ids)} camera(s), skipping...")
            continue

        print(f"\nFrame {frame_id}: Processing {len(camera_ids)} cameras")

        # Storage for this frame's 3D points
        frame_points = []
        frame_colors = []

        # Process all camera pairs for this frame
        for i, cam1_id in enumerate(camera_ids):
            img1_path = frame_cameras[cam1_id]

            if cam1_id not in camera_poses:
                print(f"  Warning: Camera {cam1_id} not found in COLMAP poses, skipping...")
                continue

            # Get camera 1 info
            cam1_pose = camera_poses[cam1_id]
            cam1 = cameras[cam1_pose['colmap_camera_id']]
            K1 = get_camera_matrix(cam1)
            R1 = qvec2rotmat(cam1_pose['qvec'])
            t1 = cam1_pose['tvec']

            # Match with other cameras in this frame
            for cam2_id in camera_ids[i+1:]:
                img2_path = frame_cameras[cam2_id]

                if cam2_id not in camera_poses:
                    continue

                # Get camera 2 info
                cam2_pose = camera_poses[cam2_id]
                cam2 = cameras[cam2_pose['colmap_camera_id']]
                K2 = get_camera_matrix(cam2)
                R2 = qvec2rotmat(cam2_pose['qvec'])
                t2 = cam2_pose['tvec']

                try:
                    # Match images using ROMA
                    W_A, H_A = Image.open(img1_path).size
                    W_B, H_B = Image.open(img2_path).size

                    warp, certainty = roma_model.match(str(img1_path), str(img2_path), device=device)

                    # Sample matches
                    matches, certainty_sampled = roma_model.sample(warp, certainty, num=match_threshold*2)

                    # Filter by certainty
                    high_cert_mask = certainty_sampled > min_certainty
                    if high_cert_mask.sum() < match_threshold:
                        print(f"  Pair cam{cam1_id:04d}-cam{cam2_id:04d}: Insufficient matches ({high_cert_mask.sum()})")
                        continue

                    matches = matches[high_cert_mask]
                    certainty_sampled = certainty_sampled[high_cert_mask]

                    # Convert to pixel coordinates
                    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                    kpts1 = kpts1.cpu().numpy()
                    kpts2 = kpts2.cpu().numpy()

                    # Triangulate 3D points
                    points_3d, valid_mask = triangulate_points(kpts1, kpts2, K1, K2, R1, t1, R2, t2)

                    if valid_mask.sum() == 0:
                        print(f"  Pair cam{cam1_id:04d}-cam{cam2_id:04d}: No valid triangulations")
                        continue

                    # Get valid points and their colors
                    valid_points = points_3d[valid_mask]
                    valid_kpts1 = kpts1[valid_mask]
                    colors = get_image_colors(valid_kpts1, img1_path, H_A, W_A)

                    frame_points.append(valid_points)
                    frame_colors.append(colors)

                    print(f"  Pair cam{cam1_id:04d}-cam{cam2_id:04d}: {len(valid_points)} points")

                except Exception as e:
                    print(f"  Error processing pair cam{cam1_id:04d}-cam{cam2_id:04d}: {e}")
                    traceback.print_exc()
                    continue

        # Combine all points for this frame
        if len(frame_points) == 0:
            print(f"Frame {frame_id}: No valid points generated!")
            continue

        frame_points = np.vstack(frame_points)
        frame_colors = np.vstack(frame_colors)

        print(f"Frame {frame_id}: Total {len(frame_points)} 3D points")

        # Remove duplicate points
        print(f"Frame {frame_id}: Removing duplicates...")
        tree = cKDTree(frame_points)

        unique_indices = []
        used = set()
        for i in range(len(frame_points)):
            if i not in used:
                neighbors = tree.query_ball_point(frame_points[i], r=0.01)
                unique_indices.append(i)
                used.update(neighbors)

        frame_points = frame_points[unique_indices]
        frame_colors = frame_colors[unique_indices]

        print(f"Frame {frame_id}: {len(frame_points)} points after deduplication")

        # Remove outliers if specified
        if outlier_threshold is not None and len(frame_points) > 10:
            print(f"Frame {frame_id}: Removing outliers...")

            # Compute scene center and standard deviation per axis
            scene_center = frame_points.mean(axis=0)
            scene_std = frame_points.std(axis=0)

            # Compute deviation from center for each axis
            deviations = np.abs(frame_points - scene_center) / (scene_std + 1e-6)

            # Keep points where ALL axes are within threshold std devs from center
            inlier_mask = np.all(deviations < outlier_threshold, axis=1)

            filtered_count = len(frame_points) - inlier_mask.sum()
            print(f"Frame {frame_id}: Removed {filtered_count} points beyond {outlier_threshold}σ from scene center")

            frame_points = frame_points[inlier_mask]
            frame_colors = frame_colors[inlier_mask]

            # Remove isolated points
            if len(frame_points) > 10:
                tree = cKDTree(frame_points)
                neighbor_counts = tree.query_ball_point(frame_points, r=search_radius, return_length=True)
                density_mask = neighbor_counts >= min_neighbors

                isolated_count = len(frame_points) - density_mask.sum()
                if isolated_count > 0:
                    print(f"Frame {frame_id}: Removed {isolated_count} isolated points (< {min_neighbors} neighbors within radius {search_radius})")
                    frame_points = frame_points[density_mask]
                    frame_colors = frame_colors[density_mask]

            print(f"Frame {frame_id}: {len(frame_points)} points after outlier removal")

        # Limit number of points if specified
        if max_points is not None and len(frame_points) > max_points:
            print(f"Frame {frame_id}: Sampling {max_points} points from {len(frame_points)}")
            sample_indices = np.random.choice(len(frame_points), max_points, replace=False)
            frame_points = frame_points[sample_indices]
            frame_colors = frame_colors[sample_indices]

        # Save point cloud for this frame
        storePly(str(output_file), frame_points, frame_colors)
        print(f"Saved point cloud with {len(frame_points)} points to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate point clouds per frame from Immersive dataset using ROMA matcher. '
                    'Expected structure: base_dir/colmap_X/images/camera_XXXX.png'
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing colmap_X folders (e.g., /path/to/01_Welder_undist)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for PLY files (one per frame)')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='Specific frame IDs to process (default: all frames)')
    parser.add_argument('--min_certainty', type=float, default=0.8,
                        help='Minimum certainty threshold for matches (default: 0.8)')
    parser.add_argument('--match_threshold', type=int, default=2000,
                        help='Minimum number of matches required per pair (default: 2000)')
    parser.add_argument('--max_views', type=int, default=None,
                        help='Maximum number of camera views to use per frame (default: all available)')
    parser.add_argument('--max_points', type=int, default=100000,
                        help='Maximum number of points to keep per frame (default: 100000)')
    parser.add_argument('--outlier_threshold', type=float, default=2.0,
                        help='Outlier removal threshold in standard deviations (default: 2.0)')
    parser.add_argument('--search_radius', type=float, default=0.3,
                        help='Radius for local density check (default: 0.3)')
    parser.add_argument('--min_neighbors', type=int, default=100,
                        help='Minimum number of neighbors within search_radius (default: 100)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to use (cuda/cpu/mps, default: cuda)')

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            device = torch.device('mps')

    print(f"Using device: {device}")

    # Generate point clouds
    generate_point_cloud_per_frame_immersive(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        device=device,
        frames_to_process=args.frames,
        min_certainty=args.min_certainty,
        match_threshold=args.match_threshold,
        max_views=args.max_views,
        max_points=args.max_points,
        outlier_threshold=args.outlier_threshold,
        search_radius=args.search_radius,
        min_neighbors=args.min_neighbors,
    )


if __name__ == "__main__":
    main()
