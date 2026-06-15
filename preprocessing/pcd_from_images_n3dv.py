"""Generate one point cloud per frame from an N3DV-style multi-camera capture.

Image layout: a flat folder of ``camXX_YYYY.png`` files (XX = camera id,
YYYY = frame id) plus COLMAP ``cameras.txt`` / ``images.txt``. For each frame,
all camera views are matched pairwise with ROMA and triangulated into a PLY.

See pcd_from_images_immersive.py for the Google Immersive (``camera_XXXX.png``)
variant. Shared geometry/IO helpers live in _common.py.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree

from romatch import roma_indoor as matcher

from _common import (storePly, read_cameras_txt, read_images_txt, get_camera_matrix,
                     qvec2rotmat, triangulate_points, get_image_colors)


def parse_image_filename(filename):
    """Parse ``camXX_YYYY.ext`` -> (camera_id, frame_id), or None if no match."""
    match = re.match(r'cam(\d+)_(\d+)\.\w+', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def organize_images_by_frame(image_dir):
    """Organize ``camXX_YYYY.ext`` images into {frame_id: {camera_id: path}}."""
    frames = defaultdict(dict)
    image_dir = Path(image_dir)
    for img_path in image_dir.glob("cam*_*.*"):
        parsed = parse_image_filename(img_path.name)
        if parsed:
            camera_id, frame_id = parsed
            frames[frame_id][camera_id] = img_path
    return dict(frames)


def generate_point_cloud_per_frame(colmap_dir, image_dir, output_dir, device,
                                    frames_to_process=None, max_frames=None, min_certainty=0.5,
                                    match_threshold=1000, max_views=None, max_points=None,
                                    outlier_threshold=None, search_radius=0.1, min_neighbors=5):
    """Generate one point cloud per frame using all camera views at that frame."""
    # Read COLMAP files
    cameras_file = Path(colmap_dir) / "cameras.txt"
    images_file = Path(colmap_dir) / "images.txt"

    print("Reading COLMAP files...")
    cameras = read_cameras_txt(cameras_file)
    # Images are keyed by the camera id parsed from the camXX_YYYY filename.
    camera_poses = read_images_txt(
        images_file, lambda name: (parse_image_filename(name) or (None,))[0])

    print(f"Loaded {len(cameras)} cameras and {len(camera_poses)} camera poses")

    # Organize images by frame
    print("Organizing images by frame...")
    frames = organize_images_by_frame(image_dir)

    print(f"Found {len(frames)} frames")

    # Filter frames if specified
    if frames_to_process is not None:
        frames = {fid: fdata for fid, fdata in frames.items() if fid in frames_to_process}
        print(f"Processing {len(frames)} specified frames")

    # Limit number of frames if specified
    if max_frames is not None and len(frames) > max_frames:
        sorted_frame_ids = sorted(frames.keys())[:max_frames]
        frames = {fid: frames[fid] for fid in sorted_frame_ids}
        print(f"Limited to first {max_frames} frames (frame IDs: {sorted_frame_ids[0]}-{sorted_frame_ids[-1]})")

    # Initialize ROMA model
    print("Loading ROMA model...")
    roma_model = matcher(device=device)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    print("\nProcessing frames...")
    for frame_id in tqdm(sorted(frames.keys()), ncols=80):
        # Check if point cloud already exists for this frame
        output_file = output_dir / f"frame_{frame_id:04d}.ply"
        if output_file.exists():
            print(f"\nFrame {frame_id}: Point cloud already exists, skipping...")
            continue

        frame_cameras = frames[frame_id]
        camera_ids = sorted(frame_cameras.keys())

        # Limit number of views if specified
        if max_views is not None and len(camera_ids) > max_views:
            camera_ids = camera_ids[:max_views]
            print(f"\nFrame {frame_id}: Limited to {max_views} cameras out of {len(frame_cameras)}")

        if len(camera_ids) < 2:
            print(f"Frame {frame_id}: Only {len(camera_ids)} camera(s), skipping...")
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
                        print(f"  Pair cam{cam1_id:02d}-cam{cam2_id:02d}: Insufficient matches ({high_cert_mask.sum()})")
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
                        print(f"  Pair cam{cam1_id:02d}-cam{cam2_id:02d}: No valid triangulations")
                        continue

                    # Get valid points and their colors
                    valid_points = points_3d[valid_mask]
                    valid_kpts1 = kpts1[valid_mask]
                    colors = get_image_colors(valid_kpts1, img1_path, H_A, W_A)

                    frame_points.append(valid_points)
                    frame_colors.append(colors)

                    print(f"  Pair cam{cam1_id:02d}-cam{cam2_id:02d}: {len(valid_points)} points")

                except Exception as e:
                    print(f"  Error processing pair cam{cam1_id:02d}-cam{cam2_id:02d}: {e}")
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
            scene_center = frame_points.mean(axis=0)  # [x_mean, y_mean, z_mean]
            scene_std = frame_points.std(axis=0)      # [x_std, y_std, z_std]

            # Compute deviation from center for each axis
            deviations = np.abs(frame_points - scene_center) / (scene_std + 1e-6)

            # Keep points where ALL axes are within threshold std devs from center
            inlier_mask = np.all(deviations < outlier_threshold, axis=1)

            filtered_count = len(frame_points) - inlier_mask.sum()
            print(f"Frame {frame_id}: Removed {filtered_count} points beyond {outlier_threshold}σ from scene center")

            frame_points = frame_points[inlier_mask]
            frame_colors = frame_colors[inlier_mask]

            # Remove isolated points (points with few neighbors in local region)
            if len(frame_points) > 10:
                tree = cKDTree(frame_points)

                # Count neighbors within a local radius
                neighbor_counts = tree.query_ball_point(frame_points, r=search_radius, return_length=True)

                # Minimum number of neighbors required (including the point itself)
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
            # Randomly sample points
            sample_indices = np.random.choice(len(frame_points), max_points, replace=False)
            frame_points = frame_points[sample_indices]
            frame_colors = frame_colors[sample_indices]

        # Save point cloud for this frame
        storePly(str(output_file), frame_points, frame_colors)
        print(f"Saved point cloud with {len(frame_points)} points to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate point clouds per frame from multi-camera setup using ROMA matcher'
    )
    parser.add_argument('--colmap_dir', type=str, default='./N3DV/N3DV/cook_spinach/',
                        help='Directory containing cameras.txt and images.txt (e.g., colmap/sparse/0)')
    parser.add_argument('--image_dir', type=str, default='./N3DV/N3DV/cook_spinach/images',
                        help='Directory containing image files named as camXX_YYYY.png')
    parser.add_argument('--output_dir', type=str, default='./point_clouds',
                        help='Output directory for PLY files (one per frame)')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='Specific frame IDs to process (default: all frames)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process, starting from frame 0 (default: all frames)')
    parser.add_argument('--min_certainty', type=float, default=0.8,
                        help='Minimum certainty threshold for matches (default: 0.8)')
    parser.add_argument('--match_threshold', type=int, default=2000,
                        help='Minimum number of matches required per pair (default: 2000)')
    parser.add_argument('--max_views', type=int, default=12,
                        help='Maximum number of camera views to use per frame (default: all available)')
    parser.add_argument('--max_points', type=int, default=100000,
                        help='Maximum number of points to keep per frame (default: all points)')
    parser.add_argument('--outlier_threshold', type=float, default=2.0,
                        help='Outlier removal threshold in standard deviations from scene center per axis. '
                             'Points beyond this on ANY axis are removed. Lower = stricter.')
    parser.add_argument('--search_radius', type=float, default=0.3,
                        help='Radius for local density check to remove isolated points. Adjust based on scene scale.')
    parser.add_argument('--min_neighbors', type=int, default=100,
                        help='Minimum number of neighbors within search_radius required to keep a point.')
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
    generate_point_cloud_per_frame(
        colmap_dir=args.colmap_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=device,
        frames_to_process=args.frames,
        max_frames=args.max_frames,
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
