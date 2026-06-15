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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
from PIL import Image
from utils.image_utils import PILtoTorch

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", args=None,
                 timestamp = 0.0,
                 image_path=None, resolution=None, image_width=None, image_height=None,
                 cam_id=None, frame_id=None,
                 fl_x=None, fl_y=None, cx=None, cy=None,
                 orig_width=None, orig_height=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        
        # cam_id / frame_id are always provided by the dynamic scene readers.
        self.cam_id = cam_id
        self.frame_id = frame_id if frame_id is not None else 0

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self._image_path = image_path
        self._resolution = resolution
        self._gt_alpha_mask = gt_alpha_mask
        self._original_image = None  # Will be loaded on-demand
        
        if image is not None:
            self._original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self._original_image.shape[2]
            self.image_height = self._original_image.shape[1]

            if gt_alpha_mask is not None:
                self._original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self._original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            self.image_width = image_width
            self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        
        # Use center-shift projection when principal point is available and off-center
        if fl_x is not None and fl_y is not None and cx is not None and cy is not None and fl_x > 0 and fl_y > 0 and cx > 0 and cy > 0:
            # Projection matrix is scale-invariant: scaling all pixel quantities by the same 
            # factor gives the identical matrix. Use original (unscaled) values.
            proj_w = orig_width if orig_width is not None else self.image_width
            proj_h = orig_height if orig_height is not None else self.image_height
            self.projection_matrix = getProjectionMatrixCenterShift(
                znear=self.znear, zfar=self.zfar, 
                cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y, 
                w=proj_w, h=proj_h
            ).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    @property
    def original_image(self):
        if self._original_image is None:
            if self._image_path is None:
                raise RuntimeError(f"Camera {self.uid} ({self.image_name}): Cannot load image - image_path is None and image was not pre-loaded")

            pil_image = Image.open(self._image_path)
            resized_image_rgb = PILtoTorch(pil_image, self._resolution)
            
            if resized_image_rgb.shape[0] == 4:
                self._gt_alpha_mask = resized_image_rgb[3:4, ...]
            
            gt_image = resized_image_rgb[:3, ...]
            self._original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            
            if self._gt_alpha_mask is not None:
                self._original_image *= self._gt_alpha_mask.to(self.data_device)
            else:
                self._original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            
            pil_image.close()
            
        return self._original_image
    
    def unload_image(self):
        """Explicitly unload the image from GPU memory"""
        if self._original_image is not None:
            del self._original_image
            self._original_image = None

        if self._gt_alpha_mask is not None:
            del self._gt_alpha_mask
            self._gt_alpha_mask = None
    
    def __str__(self):
        return f"Camera {self.uid} (colmap_id: {self.colmap_id}, cam_id: {self.cam_id}, frame_id: {self.frame_id}, timestamp: {self.timestamp})"

class CameraImmersive(Camera):
    """
    Immersive fisheye camera with 2× super-sampled rendering.
    
    Renders at 2× the GT resolution. During training, the 2× rendered image is
    warped through a precomputed distortion flow map (from undistorted space back
    to the original fisheye space) using grid_sample, which also implicitly 
    downsamples to the GT resolution. The result is compared against the raw
    distorted ground truth image — avoiding undistortion artifacts entirely.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", args=None,
                 timestamp=0.0,
                 image_path=None, resolution=None, image_width=None, image_height=None,
                 cam_id=None, frame_id=None,
                 fl_x=None, fl_y=None, cx=None, cy=None,
                 orig_width=None, orig_height=None
                 ):
        # Initialize parent Camera with the GT resolution
        super().__init__(
            colmap_id=colmap_id, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
            image=image, gt_alpha_mask=gt_alpha_mask,
            image_name=image_name, uid=uid,
            trans=trans, scale=scale, data_device=data_device, args=args,
            timestamp=timestamp,
            image_path=image_path, resolution=resolution,
            image_width=image_width, image_height=image_height,
            cam_id=cam_id, frame_id=frame_id,
            fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy,
            orig_width=orig_width, orig_height=orig_height
        )
        
        # Store GT dimensions before doubling
        self.gt_width = self.image_width
        self.gt_height = self.image_height
        
        # Double the render resolution — the rasterizer uses image_width/image_height
        self.image_width = 2 * self.gt_width
        self.image_height = 2 * self.gt_height
        
        # Distortion flow map: set externally after construction
        # Shape: [1, gt_height, gt_width, 2] with values in [-1, 1]
        self.fisheyemapper = None

    @property
    def original_image(self):
        """Override to use gt_height/gt_width (not the doubled render dimensions)."""
        if self._original_image is None:
            if self._image_path is None:
                raise RuntimeError(f"Camera {self.uid} ({self.image_name}): Cannot load image - image_path is None and image was not pre-loaded")

            pil_image = Image.open(self._image_path)
            resized_image_rgb = PILtoTorch(pil_image, self._resolution)

            if resized_image_rgb.shape[0] == 4:
                self._gt_alpha_mask = resized_image_rgb[3:4, ...]

            gt_image = resized_image_rgb[:3, ...]
            self._original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

            if self._gt_alpha_mask is not None:
                self._original_image *= self._gt_alpha_mask.to(self.data_device)
            else:
                self._original_image *= torch.ones((1, self.gt_height, self.gt_width), device=self.data_device)

            pil_image.close()

        return self._original_image


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

