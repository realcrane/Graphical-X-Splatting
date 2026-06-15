"""Merge per-frame point clouds into a single temporal (4D) point cloud.

Reads ``frame_XXXX.ply`` files (as produced by pcd_from_images*.py), estimates a
forward per-point velocity between consecutive frames, deduplicates stationary
points across time, and writes a ``points4D.ply`` (xyz + rgb + time + velocity)
plus velocity-magnitude debug visualizations. Shared PLY I/O lives in _common.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from matplotlib import cm

# Repo root on path for the utils import below.
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.graphics_utils import BasicPointCloud, AdvancedPointCloud

from _common import fetchPly, storePly, store4DPly


def save_velocity_debug_frames(input_dir, points_per_frame, colors_per_frame, velocities_per_frame, frame_numbers):
    """Save per-frame debug PLY files with velocity-magnitude (inferno) coloring."""
    debug_dir = Path(input_dir) / "velocity_debug"
    debug_dir.mkdir(exist_ok=True)

    print(f"\nSaving per-frame velocity debug visualizations to {debug_dir}...")

    # Get inferno colormap
    inferno = cm.get_cmap('inferno')

    for i, (points, velocity, frame_num) in enumerate(zip(points_per_frame, velocities_per_frame, frame_numbers)):
        # Compute velocity magnitudes for this frame
        velocity_magnitudes = np.linalg.norm(velocity, axis=1)

        # Normalize to [0, 1] for colormap
        v_min = velocity_magnitudes.min()
        v_max = velocity_magnitudes.max()
        if v_max > v_min:
            normalized_velocities = (velocity_magnitudes - v_min) / (v_max - v_min)
        else:
            normalized_velocities = np.zeros_like(velocity_magnitudes)

        # Apply inferno colormap
        velocity_colors = inferno(normalized_velocities)[:, :3]  # RGB only, drop alpha
        velocity_colors_uint8 = (velocity_colors * 255).astype(np.uint8)

        # Save frame PLY
        debug_file = debug_dir / f"frame_{frame_num:04d}_velocity.ply"
        storePly(debug_file, points, velocity_colors_uint8)

        if i == 0 or i == len(points_per_frame) - 1 or i % 10 == 0:
            print(f"  Frame {frame_num}: {len(points)} points, velocity range: {v_min:.4f} to {v_max:.4f}")

    print(f"Saved {len(points_per_frame)} velocity debug frames to {debug_dir}")


def compute_velocity_forward(points_curr, points_next, region_radius=0.1, min_neighbors=1):
    """
    Simplified velocity: k=1 nearest neighbor in next frame.
    Velocity = translation between matched pairs; distances > region_radius -> zero.
    """
    if points_next is None or len(points_next) == 0:
        return np.zeros_like(points_curr)
    tree_next = cKDTree(points_next)
    # For each point in the current frame, gather ALL neighbors within region_radius in the next frame
    neighbor_lists = tree_next.query_ball_point(points_curr, r=region_radius)
    velocities = np.zeros_like(points_curr)
    # Average displacement over all neighbors within radius; zero if insufficient neighbors
    for i, nbrs in enumerate(neighbor_lists):
        if len(nbrs) >= min_neighbors:
            mean_next_location = points_next[nbrs].mean(axis=0)
            velocities[i] = mean_next_location - points_curr[i]
    return velocities


def compute_velocity_forward_centers(points_curr, points_next, region_radius=0.1, min_neighbors=1):
    if points_next is None or len(points_next) == 0:
        return np.zeros_like(points_curr)

    tree_next = cKDTree(points_next)
    tree_curr = cKDTree(points_curr)

    next_neighbor_lists = tree_next.query_ball_point(points_curr, r=region_radius)
    curr_neighbor_lists = tree_curr.query_ball_point(points_curr, r=region_radius)

    velocities = np.zeros_like(points_curr)
    for i, (next_nbrs, curr_nbrs) in enumerate(zip(next_neighbor_lists, curr_neighbor_lists)):
        if len(next_nbrs) >= min_neighbors and len(curr_nbrs) >= min_neighbors:
            centroid_next = points_next[next_nbrs].mean(axis=0)
            centroid_curr = points_curr[curr_nbrs].mean(axis=0)
            velocities[i] = centroid_next - centroid_curr
    return velocities


def merge_point_clouds(input_dir, output_file, k_neighbors=1, max_velocity=None, max_points=None, max_points_per_frame=None, motion_weight=2.0, region_radius=0.1, min_neighbors=5, frame_stride=1):
    """
    Merge per-frame point clouds into a single temporal point cloud.

    Args:
        input_dir: Directory containing frame_XXXX.ply files
        output_file: Output PLY file path
        k_neighbors: Number of nearest neighbors for velocity computation
        max_velocity: Maximum velocity magnitude (for outlier filtering)
        max_points: Maximum number of points in final merged cloud (None for all)
        max_points_per_frame: Maximum number of points per frame (None for no limit)
        motion_weight: Weight multiplier for t>0 frames during sampling (default: 2.0)
                      Higher values allocate more points to motion regions
        region_radius: maximum distance for valid NN match
        min_neighbors: (unused)
        frame_stride: number of frames to look ahead for NN matching (default: 1)
    """
    input_dir = Path(input_dir)

    # Find all frame files
    frame_files = sorted(input_dir.glob("frame_*.ply"))

    if len(frame_files) == 0:
        print(f"No frame files found in {input_dir}")
        return

    print(f"Found {len(frame_files)} frame files")

    # Extract frame numbers
    frame_numbers = []
    for f in frame_files:
        frame_num = int(f.stem.split('_')[1])
        frame_numbers.append(frame_num)

    # Sort by frame number
    sorted_indices = np.argsort(frame_numbers)
    frame_files = [frame_files[i] for i in sorted_indices]
    frame_numbers = [frame_numbers[i] for i in sorted_indices]

    print(f"Processing frames {frame_numbers[0]} to {frame_numbers[-1]}")

    # Normalize time values
    min_frame = frame_numbers[0]
    max_frame = frame_numbers[-1]
    if max_frame == min_frame:
        # Only one frame
        normalized_times = [0.0]
    else:
        normalized_times = [(fn - min_frame) / (max_frame - min_frame) for fn in frame_numbers]

    # Storage for merged point cloud
    all_points = []
    all_colors = []
    all_normals = []
    all_times = []
    all_velocities = []

    # Storage for per-frame debug output
    points_per_frame = []
    colors_per_frame = []
    velocities_per_frame = []

    # Pre-load all point clouds for forward velocity calculation
    print("\nLoading all point clouds...")
    all_pcds = []
    for i, frame_file in enumerate(tqdm(frame_files, ncols=80)):
        print(f"\nLoading frame {frame_numbers[i]} from {frame_file}...")
        pcd = fetchPly(frame_file)

        # Limit points per frame if specified
        if max_points_per_frame is not None and len(pcd.points) > max_points_per_frame:
            original_count = len(pcd.points)
            indices = np.random.choice(len(pcd.points), max_points_per_frame, replace=False)
            pcd = BasicPointCloud(
                points=pcd.points[indices],
                colors=pcd.colors[indices],
                normals=pcd.normals[indices]
            )
            print(f"  Frame {frame_numbers[i]}: Sampled {max_points_per_frame} from {original_count} points")

        all_pcds.append(pcd)

    print("\nComputing velocities and merging (forward kNN)...")
    for i in tqdm(range(len(all_pcds)), ncols=80):
        pcd = all_pcds[i]
        frame_time = normalized_times[i]
        # Use configurable stride to select the matching frame
        next_pcd = all_pcds[i + frame_stride] if i + frame_stride < len(all_pcds) else None
        next_points = next_pcd.points if next_pcd is not None else None

        print(f"\nProcessing frame {frame_numbers[i]} (t={frame_time:.3f}) with {len(pcd.points)} points")

        # Compute velocity: forward NN only
        velocity = compute_velocity_forward_centers(pcd.points, next_points, region_radius=region_radius, min_neighbors=min_neighbors)

        # If any velocity magnitude exceeds max, normalize those vectors to unit length
        if max_velocity is not None:
            velocity_magnitude = np.linalg.norm(velocity, axis=1, keepdims=True)
            exceeds = (velocity_magnitude.squeeze() > max_velocity)
            if np.any(exceeds):
                # Normalize exceeding velocities to unit vectors preserving direction
                velocity[exceeds] = velocity[exceeds] / velocity_magnitude[exceeds]
                print(f"  Frame {frame_numbers[i]}: Normalized {exceeds.sum()} velocity vectors to unit length (threshold {max_velocity})")

        points = pcd.points
        colors = pcd.colors
        normals = pcd.normals

        # Create time array for this frame
        times = np.full(len(points), frame_time)

        # Store per-frame data for debug visualization
        points_per_frame.append(points.copy())
        colors_per_frame.append(colors.copy())
        velocities_per_frame.append(velocity.copy())

        # Append to merged cloud
        all_points.append(points)
        all_colors.append(colors)
        all_normals.append(normals)
        all_times.append(times)
        all_velocities.append(velocity)

        velocity_magnitudes = np.linalg.norm(velocity, axis=1)
        zero_velocity_count = (velocity_magnitudes < 1e-6).sum()
        zero_velocity_pct = 100 * zero_velocity_count / len(velocity_magnitudes)
        print(f"  Frame {frame_numbers[i]} (t={frame_time:.3f}): {len(points)} points, "
              f"avg velocity: {velocity_magnitudes.mean():.4f}, max: {velocity_magnitudes.max():.4f}, "
              f"zero-vel: {zero_velocity_count} ({zero_velocity_pct:.1f}%)")

    # Concatenate all data
    print("\nCombining all frames...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    all_normals = np.vstack(all_normals)
    all_times = np.concatenate(all_times)
    all_velocities = np.vstack(all_velocities)

    # Convert colors from [0-1] to [0-255] for store4DPly
    all_colors_uint8 = (all_colors * 255).astype(np.uint8)

    print(f"Total points before deduplication: {len(all_points)}")
    print(f"Time range: [{all_times.min():.3f}, {all_times.max():.3f}]")
    print(f"Velocity statistics (before processing):")
    print(f"  Mean magnitude: {np.linalg.norm(all_velocities, axis=1).mean():.4f}")
    print(f"  Max magnitude: {np.linalg.norm(all_velocities, axis=1).max():.4f}")
    print(f"  Min magnitude: {np.linalg.norm(all_velocities, axis=1).min():.4f}")

    # Post-merge: normalize any vectors exceeding max magnitude to unit length
    if max_velocity is not None:
        print(f"\nNormalizing velocity vectors exceeding {max_velocity} to unit length...")
        velocity_magnitudes = np.linalg.norm(all_velocities, axis=1, keepdims=True)
        exceeds_mask = velocity_magnitudes.squeeze() > max_velocity
        if np.any(exceeds_mask):
            all_velocities[exceeds_mask] = all_velocities[exceeds_mask] / velocity_magnitudes[exceeds_mask]
            print(f"  Normalized {exceeds_mask.sum()} velocity vectors to unit length")

    # Velocity smoothing removed

    print(f"\nVelocity statistics (after processing):")
    print(f"  Mean magnitude: {np.linalg.norm(all_velocities, axis=1).mean():.4f}")
    print(f"  Max magnitude: {np.linalg.norm(all_velocities, axis=1).max():.4f}")
    print(f"  Min magnitude: {np.linalg.norm(all_velocities, axis=1).min():.4f}")

    # Smart deduplication: remove redundant stationary points across time
    print(f"\nSmart deduplication: removing redundant stationary points...")
    velocity_magnitudes = np.linalg.norm(all_velocities, axis=1)
    stationary_mask = velocity_magnitudes < 1e-4  # Nearly zero velocity

    # Build 4D positions (x, y, z, t) for spatial-temporal clustering
    xyzwt = np.column_stack([all_points, all_times])

    # Keep all moving points
    moving_indices = np.where(~stationary_mask)[0]
    stationary_indices = np.where(stationary_mask)[0]

    print(f"  Moving points: {len(moving_indices)}")
    print(f"  Stationary points: {len(stationary_indices)}")

    if len(stationary_indices) > 0:
        # For stationary points, cluster in spatial-temporal space
        # Keep only one representative per cluster
        stat_xyzwt = xyzwt[stationary_indices]
        # Scale time dimension to match spatial importance (adjust weight as needed)
        time_weight = 0.5  # Lower = more temporal merging, higher = keep more temporal variation
        stat_xyzwt_scaled = stat_xyzwt.copy()
        stat_xyzwt_scaled[:, 3] *= time_weight

        # Build tree and find duplicates within radius
        tree = cKDTree(stat_xyzwt_scaled)
        merge_radius = region_radius * 0.3  # Smaller = keep more points, larger = more aggressive merging

        # Greedy clustering: keep first point in each cluster
        kept_stat_indices = []
        remaining = set(range(len(stationary_indices)))

        for i in range(len(stationary_indices)):
            if i not in remaining:
                continue
            # Find all neighbors (including self)
            neighbors = tree.query_ball_point(stat_xyzwt_scaled[i], r=merge_radius)
            # Remove all neighbors from remaining set
            remaining -= set(neighbors)
            # Keep this representative point
            kept_stat_indices.append(stationary_indices[i])

        kept_stat_indices = np.array(kept_stat_indices)
        print(f"  Stationary points after clustering: {len(kept_stat_indices)} "
              f"(removed {len(stationary_indices) - len(kept_stat_indices)})")

        # Combine moving + deduplicated stationary points
        final_indices = np.concatenate([moving_indices, kept_stat_indices])
    else:
        final_indices = moving_indices

    # Apply deduplication
    all_points = all_points[final_indices]
    all_colors_uint8 = all_colors_uint8[final_indices]
    all_normals = all_normals[final_indices]
    all_times = all_times[final_indices]
    all_velocities = all_velocities[final_indices]

    print(f"Total points after deduplication: {len(all_points)}")

    # Reduce to max_points if specified
    if max_points is not None and len(all_points) > max_points:
        print(f"\nReducing to {max_points} points from {len(all_points)}...")

        # Compute sampling weights: favor t>0 (motion regions) over t=0
        weights = np.ones(len(all_points))
        motion_mask = all_times > 0
        weights[motion_mask] *= motion_weight

        # Normalize weights to probabilities
        weights = weights / weights.sum()

        # Weighted sampling
        sample_indices = np.random.choice(len(all_points), max_points, replace=False, p=weights)
        all_points = all_points[sample_indices]
        all_colors_uint8 = all_colors_uint8[sample_indices]
        all_normals = all_normals[sample_indices]
        all_times = all_times[sample_indices]
        all_velocities = all_velocities[sample_indices]

        t0_count = (all_times == 0).sum()
        motion_count = (all_times > 0).sum()
        print(f"Final point count: {len(all_points)} (t=0: {t0_count}, t>0: {motion_count})")

    # Save merged point cloud
    print(f"\nSaving to {output_file}...")
    store4DPly(output_file, all_points, all_colors_uint8, all_times, all_velocities)
    print(f"Saved 4D point cloud with {len(all_points)} points to {output_file}")

    # Save 4D debug PLY with velocity magnitude visualization
    debug_4d_file = str(Path(output_file).parent / (Path(output_file).stem + "_velocity_debug.ply"))
    print(f"\nSaving 4D velocity debug visualization to {debug_4d_file}...")

    # Compute velocity magnitudes for the final merged cloud
    final_velocity_magnitudes = np.linalg.norm(all_velocities, axis=1)

    # Normalize to [0, 1] for colormap
    v_min = final_velocity_magnitudes.min()
    v_max = final_velocity_magnitudes.max()
    if v_max > v_min:
        normalized_velocities = (final_velocity_magnitudes - v_min) / (v_max - v_min)
    else:
        normalized_velocities = np.zeros_like(final_velocity_magnitudes)

    # Apply inferno colormap
    inferno = cm.get_cmap('inferno')
    velocity_colors = inferno(normalized_velocities)[:, :3]  # RGB only, drop alpha
    velocity_colors_uint8 = (velocity_colors * 255).astype(np.uint8)

    store4DPly(debug_4d_file, all_points, velocity_colors_uint8, all_times, all_velocities)
    print(f"Saved 4D velocity debug PLY (velocity range: {v_min:.4f} to {v_max:.4f})")

    # Save per-frame debug PLY files with velocity magnitude visualization
    save_velocity_debug_frames(input_dir, points_per_frame, colors_per_frame, velocities_per_frame, frame_numbers)

    return AdvancedPointCloud(
        points=all_points,
        colors=all_colors,
        normals=all_normals,
        time=all_times,
        velocity=all_velocities
    )


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-frame point clouds into temporal point cloud with velocity'
    )
    parser.add_argument('--input_dir', type=str, default='./point_clouds',
                        help='Directory containing frame_XXXX.ply files')
    parser.add_argument('--output', type=str, default='points4D.ply',
                        help='Output PLY file path')
    parser.add_argument('--k_neighbors', type=int, default=1,
                        help='Number of nearest neighbors for velocity computation (default: 1)')
    parser.add_argument('--max_velocity', type=float, default=1.0,
                        help='Maximum velocity magnitude for outlier filtering (default: 1.0)')
    parser.add_argument('--max_points', type=int, default=2000000,
                        help='Maximum number of points in final merged cloud (default: 2,000,000)')
    parser.add_argument('--motion_weight', type=float, default=2.0,
                        help='Weight multiplier for t>0 frames during sampling (default: 2.0). '
                             'Higher values allocate more points to motion regions.')
    parser.add_argument('--velocity_smoothing_neighbors', type=int, default=0,
                        help='Velocity smoothing disabled in simplified mode.')
    parser.add_argument('--region_radius', type=float, default=0.3,
                        help='Maximum distance for valid NN match between frames.')
    parser.add_argument('--min_neighbors', type=int, default=50,
                        help='K for NN (kept for API, unused beyond k=1).')
    parser.add_argument('--max_points_per_frame', type=int, default=10000,
                        help='Maximum number of points per frame (default: 10000). '
                             'Useful to limit memory and speed up processing.')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Number of frames to look ahead for NN matching (default: 1). '
                             'Increase to capture slower motion.')

    args = parser.parse_args()

    merge_point_clouds(
        input_dir=args.input_dir,
        output_file=args.output,
        k_neighbors=args.k_neighbors,
        max_velocity=args.max_velocity,
        max_points=args.max_points,
        max_points_per_frame=args.max_points_per_frame,
        motion_weight=args.motion_weight,
        region_radius=args.region_radius,
        min_neighbors=args.min_neighbors,
        frame_stride=args.frame_stride
    )


if __name__ == "__main__":
    main()
