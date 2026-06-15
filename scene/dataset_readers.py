#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This file is a derivative work of the original software.
# Modifications and additions:
# 2026, Doga Yilmaz (doga.yilmaz@ucl.ac.uk)
# Virtual Environments and Computer Graphics Lab, UCL
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For original inquiries contact  george.drettakis@inria.fr
# For modification inquiries contact doga.yilmaz@ucl.ac.uk
#

import os
import re
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from utils.sh_utils import SH2RGB
from scene.nt_model import BasicPointCloud, AdvancedPointCloud
from multiprocessing.pool import ThreadPool


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0
    cam_id: int = -1
    frame_id: int = -1
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetch4DPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    time = np.array(vertices['time'])
    velocity = np.vstack([vertices['vx'], vertices['vy'], vertices['vz']]).T
    return AdvancedPointCloud(points=positions, colors=colors, normals=normals, time=time, velocity=velocity)

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def store4DPly(path, xyz, rgb, time, velocity):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('time', 'f4'),
            ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, time.reshape(-1,1), velocity), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readDynamicCamerasFromColmap(cam_extrinsics, cam_intrinsics, images_folder, max_dynamic_frames=None, time_duration=None, timestamp_type="normalized"):
    cam_infos = []
    global_uid = 0

    image_fnames = sorted(os.listdir(images_folder))

    if not max_dynamic_frames:
        max_frame_id = 0
        for image_fname in image_fnames:
            image_name = os.path.splitext(image_fname)[0]
            frame_id = int(image_name.split('_')[1])
            max_frame_id = max(max_frame_id, frame_id)
    else:
        max_frame_id = max_dynamic_frames - 1

    for image_fname in image_fnames:
        image_name = os.path.splitext(image_fname)[0]
        cam_id = int(image_name.split('_')[0].replace('cam', ''))
        frame_id = int(image_name.split('_')[1])

        try:
            cam_key = next(key for key, extr in cam_extrinsics.items() if int(extr.name.split('.')[0].split('_')[0].replace('cam', '')) == cam_id)
        except StopIteration:
            raise RuntimeError(f"Camera matching failed for {image_fname}")

        extr = cam_extrinsics[cam_key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = global_uid
        global_uid += 1
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        if time_duration is not None:
            if max_frame_id == 0:
                timestamp = 0.0
            else:
                if timestamp_type == "frame_id":
                    timestamp = float(frame_id)
                elif timestamp_type == "normalized":
                    timestamp = (float(frame_id) / float(max_frame_id)) * time_duration
        else:
            if max_frame_id == 0:
                timestamp = 0.0
            else:
                if timestamp_type == "frame_id":
                    timestamp = float(frame_id)
                elif timestamp_type == "normalized":
                    timestamp = float(frame_id) / float(max_frame_id)

        if max_dynamic_frames is not None and frame_id >= max_dynamic_frames:
            continue

        image_path = os.path.join(images_folder, image_fname)
        image = None
        if intr.model == "PINHOLE" and len(intr.params) >= 4:
            fl_x = intr.params[0]
            fl_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                timestamp=timestamp, cam_id=cam_id, frame_id=frame_id,
                                fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
        else:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                timestamp=timestamp, cam_id=cam_id, frame_id=frame_id)
        
        cam_infos.append(cam_info)
    
    return cam_infos

def readDynamicImmersiveSceneInfo(path, images, eval, eval_count=1, omit_cams_frames=None, init_type="sfm", max_cap=None, num_pts=100000, shrink=1, max_dynamic_frames=None, time_duration=None, timestamp_type="normalized"):
    """
    Read dynamic scene info from Immersive dataset format where data is split across multiple colmap_X folders.
    Each colmap_X folder represents a frame in time.
    
    Uses the Immersive fisheye pipeline:
        - Images are loaded from raw distorted frames: {path}/{camera_name}/{frame_idx}.png
        - Camera intrinsics come from manual/ (matching the distorted images)
        - The training loop will warp 2x-rendered images to distorted space for comparison
    """
    # Find all colmap folders in the path
    colmap_folders = sorted([f for f in os.listdir(path) if f.startswith("colmap_") and os.path.isdir(os.path.join(path, f))])
    
    if len(colmap_folders) == 0:
        raise RuntimeError(f"No colmap folders found in {path}")
    
    # Extract frame numbers from folder names
    frame_numbers = []
    for folder in colmap_folders:
        try:
            frame_num = int(folder.split("_")[1])
            frame_numbers.append(frame_num)
        except:
            continue
    
    if len(frame_numbers) == 0:
        raise RuntimeError(f"Could not parse frame numbers from colmap folders in {path}")
    
    starttime = min(frame_numbers)
    max_frame_id = max(frame_numbers)
    duration = max_frame_id - starttime + 1
    
    if max_dynamic_frames is not None:
        duration = min(duration, max_dynamic_frames)
        max_frame_id = starttime + duration - 1
    
    print(f"Reading Dynamic Immersive Scene: frames {starttime} to {max_frame_id} (duration: {duration})")
    
    reading_dir = "images" if images == None else images
    
    # Read cameras from all colmap folders
    cam_infos_unsorted = []
    global_uid = 0
    
    for frame_idx in range(starttime, starttime + duration):
        colmap_folder = os.path.join(path, f"colmap_{frame_idx}")
        
        if not os.path.exists(colmap_folder):
            print(f"Warning: colmap folder {colmap_folder} not found, skipping frame {frame_idx}")
            continue
        
        # Intrinsics from manual/ (matching raw distorted images)
        # Extrinsics from manual/ (poses from models.json)
        try:
            cameras_intrinsic_file = os.path.join(colmap_folder, "manual", "cameras.txt")
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        except:
            # Fallback to sparse/0
            try:
                cameras_intrinsic_file = os.path.join(colmap_folder, "sparse/0", "cameras.bin")
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            except:
                cameras_intrinsic_file = os.path.join(colmap_folder, "sparse/0", "cameras.txt")
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
        # Load extrinsics: prefer manual (models.json poses) for consistent orientation,
        # fallback to sparse/0
        cameras_loaded = False
        try:
            cameras_extrinsic_file = os.path.join(colmap_folder, "manual", "images.txt")
            if os.path.exists(cameras_extrinsic_file):
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cameras_loaded = True
        except:
            pass
        
        if not cameras_loaded:
            try:
                cameras_extrinsic_file = os.path.join(colmap_folder, "sparse/0", "images.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            except:
                cameras_extrinsic_file = os.path.join(colmap_folder, "sparse/0", "images.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        
        images_folder = os.path.join(colmap_folder, reading_dir)
        
        # Calculate timestamp for this frame
        if duration == 1:
            timestamp = 0.0
        else:
            if timestamp_type == "frame_id":
                timestamp = float(frame_idx - starttime)
            elif timestamp_type == "normalized":
                if time_duration is not None:
                    timestamp = (float(frame_idx - starttime) / float(duration - 1)) * time_duration
                else:
                    timestamp = float(frame_idx - starttime) / float(duration - 1)
        
        # Read cameras for this frame
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width
            
            uid = global_uid
            global_uid += 1
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            
            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            
            # Resolve image path based on mode
            image_name = extr.name
            image_name_base = os.path.splitext(image_name)[0]  # e.g., "camera_0001"
            
            # Load raw distorted frames from {path}/{camera_name}/{frame_idx}.png
            raw_image_path = os.path.join(path, image_name_base, f"{frame_idx}.png")
            if os.path.exists(raw_image_path):
                image_path = raw_image_path
            else:
                raise FileNotFoundError(
                    f"Raw distorted image not found: {raw_image_path}\n"
                    f"Run preprocessing/colmap_from_video_immersive.py first to extract and symlink raw frames.")
            
            # Try to extract cam_id from the image name
            # Assuming format like "camera_0001.png" or "cam_01.png"
            try:
                if "camera_" in image_name:
                    cam_id = int(image_name.split("camera_")[1].split(".")[0].split("_")[0])
                elif "cam" in image_name.lower():
                    match = re.search(r'cam[_]?(\d+)', image_name.lower())
                    if match:
                        cam_id = int(match.group(1))
                    else:
                        cam_id = idx
                else:
                    cam_id = idx
            except:
                cam_id = idx
            
            frame_id = frame_idx - starttime
            
            image = None
            
            if intr.model == "PINHOLE" and len(intr.params) >= 4:
                fl_x = intr.params[0]
                fl_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, width=width, height=height,
                                    timestamp=timestamp, cam_id=cam_id, frame_id=frame_id,
                                    fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, width=width, height=height,
                                    timestamp=timestamp, cam_id=cam_id, frame_id=frame_id)
            
            cam_infos_unsorted.append(cam_info)
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: (x.cam_id, x.frame_id))
    
    nerf_normalization = getNerfppNorm(cam_infos)
    
    cam_infos = groupCameras(cam_infos)
    
    if eval:
        all_cam_frame_ids = {cam_id: sorted(cam_infos[cam_id].keys()) for cam_id in sorted(cam_infos.keys())}
        test_cam_frame_ids = {cam_id: sorted(cam_infos[cam_id].keys()) for cam_id in sorted(list(cam_infos.keys())[:eval_count])}
        train_cam_frame_ids = {cam_id: sorted(cam_infos[cam_id].keys()) for cam_id in sorted(list(cam_infos.keys())[eval_count:])}
        
        if omit_cams_frames is not None and len(omit_cams_frames.keys()) > 0:
            print("removing specified cameras/frames from training set:")
            remove_cams_frames(test_cam_frame_ids, train_cam_frame_ids, omit_cams_frames)
    else:
        all_cam_frame_ids = {cam_id: sorted(cam_infos[cam_id].keys()) for cam_id in sorted(cam_infos.keys())}
        test_cam_frame_ids = {}
        train_cam_frame_ids = all_cam_frame_ids
    
    test_cam_infos = {cam_id: {frame_id: cam_infos[cam_id][frame_id] for frame_id in test_cam_frame_ids[cam_id]} for cam_id in test_cam_frame_ids.keys()}
    train_cam_infos = {cam_id: {frame_id: cam_infos[cam_id][frame_id] for frame_id in train_cam_frame_ids[cam_id]} for cam_id in train_cam_frame_ids.keys()}
    
    print(f"Total frames loaded: {sum(len(frames) for frames in cam_infos.values())}")
    print(f"Training frames: {sum(len(frames) for frames in train_cam_infos.values())}")
    print(f"Test frames: {sum(len(frames) for frames in test_cam_infos.values())}")
    
    # Handle point cloud initialization
    if init_type == "sfm":
        # Combine point clouds from all colmap folders with timestamps
        totalply_path = os.path.join(path, f"points3D_total_{duration}.ply")
        
        if not os.path.exists(totalply_path):
            print("Converting point3D from all frames to combined .ply with timestamps...")
            totalxyz = []
            totalrgb = []
            totaltime = []
            
            for frame_idx in range(starttime, starttime + duration):
                colmap_folder = os.path.join(path, f"colmap_{frame_idx}")
                bin_path = os.path.join(colmap_folder, "sparse/0/points3D.bin")
                txt_path = os.path.join(colmap_folder, "sparse/0/points3D.txt")
                
                try:
                    if os.path.exists(bin_path):
                        xyz, rgb, _ = read_points3D_binary(bin_path)
                    elif os.path.exists(txt_path):
                        xyz, rgb, _ = read_points3D_text(txt_path)
                    else:
                        print(f"Warning: No points3D file found in {colmap_folder}, skipping")
                        continue
                    
                    totalxyz.append(xyz)
                    totalrgb.append(rgb)
                    
                    # Normalize time to [0, 1] or use time_duration
                    if duration == 1:
                        time_val = 0.0
                    else:
                        time_val = float(frame_idx - starttime) / float(duration - 1)
                        if time_duration is not None:
                            time_val *= time_duration
                    
                    totaltime.append(np.ones((xyz.shape[0], 1)) * time_val)
                except Exception as e:
                    print(f"Error reading points3D from {colmap_folder}: {e}")
                    continue
            
            if len(totalxyz) > 0:
                xyz = np.concatenate(totalxyz, axis=0)
                rgb = np.concatenate(totalrgb, axis=0)
                totaltime = np.concatenate(totaltime, axis=0)
                assert xyz.shape[0] == rgb.shape[0]
                # Use store4DPly for point clouds with time information
                velocity = np.zeros_like(xyz)  # No velocity information available
                store4DPly(totalply_path, xyz, rgb, totaltime.squeeze(), velocity)
            else:
                print("Warning: No point clouds found, creating empty point cloud")
                xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 * 2 - (nerf_normalization["radius"] * 3)
                rgb = np.random.random((num_pts, 3)) * 255
                totaltime = np.zeros((num_pts,))
                velocity = np.zeros_like(xyz)
                store4DPly(totalply_path, xyz, rgb, totaltime, velocity)
        
        ply_path = totalply_path
    elif init_type == "sfm_4D":
        ply_path = os.path.join(path, "points4D.ply")
        if not os.path.exists(ply_path):
            raise RuntimeError("No points4D.ply file detected, please provide a valid 4D point cloud file or use init_type sfm.")
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 * 2 - (nerf_normalization["radius"] * 3)
        rgb = np.random.random((num_pts, 3)) * 255
        
        storePly(ply_path, xyz, rgb)
    else:
        raise ValueError("Please specify a correct init_type: random, sfm, or sfm_4D")
    
    try:
        if init_type == "sfm_4D":
            pcd = fetch4DPly(ply_path)
        else:
            pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        pcd = None
    
    if pcd is not None and max_cap is not None and pcd.points.shape[0] > max_cap:
        print(f"Shrinking point cloud to {max_cap} points")
        indices = np.random.choice(pcd.points.shape[0], max_cap, replace=False)
        if init_type == "sfm_4D":
            pcd = AdvancedPointCloud(points=pcd.points[indices],
                                     colors=pcd.colors[indices],
                                     time=pcd.time[indices],
                                     velocity=pcd.velocity[indices],
                                     normals=pcd.normals[indices])
        else:
            pcd = BasicPointCloud(points=pcd.points[indices],
                                  colors=pcd.colors[indices],
                                  normals=pcd.normals[indices])
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readDynamicColmapSceneInfo(path, images, eval, eval_count=1, omit_cams_frames=None, init_type="sfm", max_cap=None, num_pts=100000, shrink=1, max_dynamic_frames=None, time_duration=None, timestamp_type="normalized"):
    try:
        cameras_extrinsic_file = os.path.join(path, "images.bin")
        cameras_intrinsic_file = os.path.join(path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "images.txt")
        cameras_intrinsic_file = os.path.join(path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    print("Reading Dynamic Training Cameras")
    cam_infos_unsorted = readDynamicCamerasFromColmap(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                                     images_folder=os.path.join(path, reading_dir),
                                                     max_dynamic_frames=max_dynamic_frames, time_duration=time_duration, timestamp_type=timestamp_type)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : (x.cam_id, x.frame_id))

    nerf_normalization = getNerfppNorm(cam_infos)

    cam_infos = groupCameras(cam_infos)

    if eval:
        all_cam_frame_ids = {cam_id: sorted(cam_infos[cam_id].keys()) for cam_id in sorted(cam_infos.keys())}
        test_cam_frame_ids = {cam_id: sorted(cam_infos[cam_id].keys()) for cam_id in sorted(list(cam_infos.keys())[:eval_count])}
        train_cam_frame_ids = {cam_id: sorted(cam_infos[cam_id].keys()) for cam_id in sorted(list(cam_infos.keys())[eval_count:])}

        if omit_cams_frames is not None and len(omit_cams_frames.keys()) > 0:
            print("removing specified cameras/frames from training set:")
            remove_cams_frames(test_cam_frame_ids, train_cam_frame_ids, omit_cams_frames)
    else:
        test_cam_frame_ids = {}
        train_cam_frame_ids = all_cam_frame_ids

    test_cam_infos = {cam_id: {frame_id: cam_infos[cam_id][frame_id] for frame_id in test_cam_frame_ids[cam_id]} for cam_id in test_cam_frame_ids.keys()}
    train_cam_infos = {cam_id: {frame_id: cam_infos[cam_id][frame_id] for frame_id in train_cam_frame_ids[cam_id]} for cam_id in train_cam_frame_ids.keys()}

    print(f"Total frames loaded: {sum(len(frames) for frames in cam_infos.values())}")
    print(f"Training frames: {sum(len(frames) for frames in train_cam_infos.values())}")
    print(f"Test frames: {sum(len(frames) for frames in test_cam_infos.values())}")

    if init_type == "sfm":
        ply_path = os.path.join(path, "points3D.ply")
        bin_path = os.path.join(path, "points3D.bin")
        txt_path = os.path.join(path, "points3D.txt")
        if not os.path.exists(ply_path):
            print("[WARNING] No points3D.ply file detected, creating one from sparse points3D.txt.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    elif init_type == "sfm_4D":
        ply_path = os.path.join(path, "points4D.ply")
        if not os.path.exists(ply_path):
            raise RuntimeError("No points4D.ply file detected, please provide a valid 4D point cloud file or use init_type sfm.")
    elif init_type == "random":
        num_pts = 1000
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)
        
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        raise ValueError("Please specify a correct init_type: random or sfm")

    if init_type == "sfm_4D":
        pcd = fetch4DPly(ply_path)
    else:
        pcd = fetchPly(ply_path)

    if max_cap is not None and pcd.points.shape[0] > max_cap:
        print(f"Shrinking point cloud to {max_cap} points")
        indices = np.random.choice(pcd.points.shape[0], max_cap, replace=False)
        if init_type == "sfm_4D":
            pcd = AdvancedPointCloud(points=pcd.points[indices],
                                     colors=pcd.colors[indices],
                                     time=pcd.time[indices],
                                     velocity=pcd.velocity[indices],
                                     normals=pcd.normals[indices])
        else:
            pcd = BasicPointCloud(points=pcd.points[indices],
                                  colors=pcd.colors[indices],
                                  normals=pcd.normals[indices])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def remove_cams_frames(test_cam_frame_ids, train_cam_frame_ids, omit_cams_frames):
    for cam_id in list(train_cam_frame_ids.keys()):
        if str(cam_id) in omit_cams_frames.keys():
            if -1 in omit_cams_frames[str(cam_id)]:
                train_cam_frame_ids.pop(cam_id)
            else:
                for frame_id in omit_cams_frames[str(cam_id)]:
                    if frame_id in train_cam_frame_ids[cam_id]:
                        train_cam_frame_ids[cam_id].remove(frame_id)

def groupCameras(cam_infos):
    """This function groups cameras by their camera ID and Frame."""
    grouped_cameras = {}
    for cam_info in cam_infos:
        if cam_info.cam_id not in grouped_cameras:
            grouped_cameras[cam_info.cam_id] = {}
        grouped_cameras[cam_info.cam_id][cam_info.frame_id] = cam_info
    return grouped_cameras

def readDynamicCamerasFromTransforms(path, transformsfile, white_background, extension=".png", max_dynamic_frames=None, time_duration=None, timestamp_type="normalized"):
    cam_infos = []
    global_uid = 0

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        
        max_frame_id = 0
        if timestamp_type == "normalized" or max_dynamic_frames is None:
            for frame in frames:
                image_name = Path(frame["file_path"]).stem
                name_parts = image_name.split('_')
                if len(name_parts) >= 2:
                    frame_id = int(name_parts[1])
                    max_frame_id = max(max_frame_id, frame_id)
        
        for idx, frame in enumerate(frames):
            file_path = frame["file_path"].lstrip('./')
            
            if not extension in file_path:
                cam_name = os.path.join(path, file_path + extension)
            else:
                cam_name = os.path.join(path, file_path)

            image_name = Path(file_path).stem
            name_parts = image_name.split('_')
            
            if len(name_parts) >= 2:
                if name_parts[0].startswith('cam'):
                    cam_id = int(name_parts[0].replace('cam', ''))
                elif name_parts[0].replace('-', '').isdigit():
                    cam_id = int(name_parts[0])
                else:
                    cam_id = 0
                
                if name_parts[1].replace('-', '').isdigit():
                    frame_id = int(name_parts[1])
                else:
                    frame_id = idx
            else:
                if image_name.replace('-', '').isdigit():
                    cam_id = 0
                    frame_id = int(image_name)
                else:
                    cam_id = idx
                    frame_id = 0
            
            if max_dynamic_frames is not None and frame_id >= max_dynamic_frames:
                continue
            
            if time_duration is not None:
                if max_frame_id == 0:
                    timestamp = 0.0
                else:
                    if timestamp_type == "frame_id":
                        timestamp = float(frame_id)
                    elif timestamp_type == "normalized":
                        timestamp = (float(frame_id) / float(max_frame_id)) * time_duration
            else:
                if max_frame_id == 0:
                    timestamp = 0.0
                else:
                    if timestamp_type == "frame_id":
                        timestamp = float(frame_id)
                    elif timestamp_type == "normalized":
                        timestamp = float(frame_id) / float(max_frame_id)

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)

            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
                image = Image.fromarray(im_data, "RGBA")
                width, height = image.size[0], image.size[1]

            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx

            uid = global_uid
            global_uid += 1

            cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=width,
                                        height=height, timestamp=timestamp, cam_id=cam_id, frame_id=frame_id))

    return cam_infos

def readDynamicBlenderSceneInfo(path, white_background, eval, extension=".png", omit_cams_frames=None, init_type="random", max_cap=None, num_pts=100000, shrink=1, max_dynamic_frames=None, time_duration=None, timestamp_type="normalized"):
    print("Reading Training Transforms")
    train_cam_infos = readDynamicCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension, max_dynamic_frames, time_duration, timestamp_type)
    print("Reading Test Transforms")
    test_cam_infos = readDynamicCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension, max_dynamic_frames, time_duration, timestamp_type)

    train_cam_infos_sorted = sorted(train_cam_infos, key=lambda x: (x.cam_id, x.frame_id))
    test_cam_infos_sorted = sorted(test_cam_infos, key=lambda x: (x.cam_id, x.frame_id))

    all_cam_infos = train_cam_infos + test_cam_infos
    nerf_normalization = getNerfppNorm(all_cam_infos)

    train_cam_infos = groupCameras(train_cam_infos_sorted)
    test_cam_infos = groupCameras(test_cam_infos_sorted)

    if not eval:
        for cam_id, frames in test_cam_infos.items():
            if cam_id not in train_cam_infos:
                train_cam_infos[cam_id] = {}
            train_cam_infos[cam_id].update(frames)
        test_cam_infos = {}
        print(f"Merged test into train. Total: {len(train_cam_infos.keys())} cameras, {sum(len(frames) for frames in train_cam_infos.values())} frames")

    print(f"Total frames loaded: {sum(len(frames) for frames in train_cam_infos.values()) + sum(len(frames) for frames in test_cam_infos.values())}")

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if max_cap is not None and pcd.points.shape[0] > max_cap:
        print(f"Shrinking point cloud to {max_cap} points")
        indices = np.random.choice(pcd.points.shape[0], max_cap, replace=False)
        pcd = BasicPointCloud(points=pcd.points[indices],
                              colors=pcd.colors[indices],
                              normals=pcd.normals[indices])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "DynamicColmap": readDynamicColmapSceneInfo,
    "DynamicBlender": readDynamicBlenderSceneInfo,
    "DynamicImmersive": readDynamicImmersiveSceneInfo
}