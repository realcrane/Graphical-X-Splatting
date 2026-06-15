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

from scene.cameras import Camera, CameraImmersive
import numpy as np
from utils.image_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_immersive=False):
    if cam_info.image is None:
        with Image.open(cam_info.image_path) as temp_img:
            orig_w, orig_h = temp_img.size
    else:
        orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    CameraClass = CameraImmersive if is_immersive else Camera

    if cam_info.image is not None:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
        
        return CameraClass(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=gt_image, gt_alpha_mask=loaded_mask,
                      image_name=cam_info.image_name, uid=id, timestamp=cam_info.timestamp, 
                      data_device=args.data_device, args=args,
                      image_path=cam_info.image_path, resolution=resolution,
                      image_width=resolution[0], image_height=resolution[1],
                      cam_id=cam_info.cam_id, frame_id=cam_info.frame_id,
                      fl_x=cam_info.fl_x if cam_info.fl_x > 0 else None,
                      fl_y=cam_info.fl_y if cam_info.fl_y > 0 else None,
                      cx=cam_info.cx if cam_info.cx > 0 else None,
                      cy=cam_info.cy if cam_info.cy > 0 else None,
                      orig_width=orig_w, orig_height=orig_h)
    else:
        return CameraClass(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=None, gt_alpha_mask=None,
                      image_name=cam_info.image_name, uid=id, timestamp=cam_info.timestamp, 
                      data_device=args.data_device, args=args,
                      image_path=cam_info.image_path, resolution=resolution,
                      image_width=resolution[0], image_height=resolution[1],
                      cam_id=cam_info.cam_id, frame_id=cam_info.frame_id,
                      fl_x=cam_info.fl_x if cam_info.fl_x > 0 else None,
                      fl_y=cam_info.fl_y if cam_info.fl_y > 0 else None,
                      cx=cam_info.cx if cam_info.cx > 0 else None,
                      cy=cam_info.cy if cam_info.cy > 0 else None,
                      orig_width=orig_w, orig_height=orig_h)

def cameraDict_from_camInfos(cam_infos, resolution_scale, args, is_immersive=False):
    camera_dict = {}

    for cam_id, cam_info_dict in cam_infos.items():
        camera_dict[cam_id] = {frame_id: loadCam(args, frame.uid, frame, resolution_scale, is_immersive=is_immersive) 
                                for frame_id, frame in cam_info_dict.items()}

    return camera_dict

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width),
        'cx': camera.cx,
        'cy': camera.cy,
        'fl_x': camera.fl_x,
        'fl_y': camera.fl_y,
        'cam_id' : camera.cam_id,
        'frame_id' : camera.frame_id
    }
    return camera_entry
