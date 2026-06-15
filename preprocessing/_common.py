"""Shared helpers for the scripts in this folder.

Holds code that used to be copy-pasted across the point-cloud / COLMAP / video
tools: PLY I/O, COLMAP text parsing, camera geometry, and a shell helper.

Heavy or optional third-party imports (plyfile, cv2, PIL, and the torch-backed
``utils.graphics_utils``) are done lazily inside the functions that need them, so
``import _common`` stays cheap for the lightweight video tools that only use
``do_system``.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so the lazy ``from utils...`` imports below
# resolve regardless of which script imported this module.
sys.path.insert(0, str(Path(__file__).parent.parent))


# --------------------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------------------
def do_system(arg):
    """Run a shell command, aborting the process if it fails."""
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


# --------------------------------------------------------------------------------------
# PLY I/O
# --------------------------------------------------------------------------------------
def fetchPly(path):
    """Load a point cloud from a PLY file as a BasicPointCloud."""
    from plyfile import PlyData
    from utils.graphics_utils import BasicPointCloud
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    """Save a basic point cloud (xyz + rgb) to a PLY file."""
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def store4DPly(path, xyz, rgb, time, velocity):
    """Save a 4D point cloud (xyz + rgb + time + velocity) to a PLY file."""
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('time', 'f4'),
             ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, time.reshape(-1, 1), velocity), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


# --------------------------------------------------------------------------------------
# COLMAP text parsing
# --------------------------------------------------------------------------------------
def read_cameras_txt(cameras_file):
    """Read a COLMAP cameras.txt -> {camera_id: {model, width, height, params}}."""
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cameras[int(parts[0])] = {
                'model': parts[1],
                'width': int(float(parts[2])),
                'height': int(float(parts[3])),
                'params': [float(p) for p in parts[4:]],
            }
    return cameras


def read_images_txt(images_file, key_from_name):
    """Read a COLMAP images.txt, keyed by a caller-supplied function.

    ``key_from_name(name)`` maps an image filename to the dict key to store the
    pose under (e.g. a camera id), or returns None to skip that image. This is
    the only part that differs between dataset layouts.

    Returns {key: {'qvec', 'tvec', 'colmap_camera_id', 'name'}}.
    """
    poses = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        colmap_camera_id = int(parts[8])
        name = ' '.join(parts[9:])
        key = key_from_name(name)
        if key is not None:
            poses[key] = {
                'qvec': qvec,
                'tvec': tvec,
                'colmap_camera_id': colmap_camera_id,
                'name': name,
            }
        # Skip the POINTS2D line that follows each image line.
        i += 2
    return poses


# --------------------------------------------------------------------------------------
# Camera geometry
# --------------------------------------------------------------------------------------
def qvec2rotmat(qvec):
    """Convert a quaternion (w, x, y, z) to a rotation matrix."""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def get_camera_matrix(camera):
    """Build the 3x3 intrinsic matrix K from a COLMAP camera dict."""
    model = camera['model']
    params = camera['params']
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'):
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    if model in ('PINHOLE', 'RADIAL', 'OPENCV'):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise ValueError(f"Unsupported camera model: {model}")


def triangulate_points(kpts1, kpts2, K1, K2, R1, t1, R2, t2):
    """Triangulate 3D points from corresponding 2D keypoints in two views.

    Returns (points_3d, valid_mask), keeping points in front of both cameras
    and closer than 1000 units.
    """
    import cv2
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    points_4d = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    valid_mask = np.ones(len(points_3d), dtype=bool)
    for P, R, t in [(P1, R1, t1), (P2, R2, t2)]:
        points_cam = (R @ points_3d.T + t.reshape(3, 1)).T
        valid_mask &= (points_cam[:, 2] > 0)
        valid_mask &= (points_cam[:, 2] < 1000)
    return points_3d, valid_mask


def get_image_colors(kpts, image_path, H, W):
    """Sample RGB colors [0-255] at keypoint locations from an image."""
    from PIL import Image
    img = Image.open(image_path).resize((W, H))
    img_array = np.array(img)
    kpts_int = np.round(kpts).astype(int)
    kpts_int[:, 0] = np.clip(kpts_int[:, 0], 0, W - 1)
    kpts_int[:, 1] = np.clip(kpts_int[:, 1], 0, H - 1)
    return img_array[kpts_int[:, 1], kpts_int[:, 0]]
