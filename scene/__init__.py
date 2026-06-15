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
import random
import torch
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
from utils.camera_utils import cameraDict_from_camInfos, camera_to_JSON
from utils.fisheye_utils import getfisheyemapper
from scene.cameras import CameraImmersive
from scene.nt_model import NTModel
class Scene_nt:

    primitives : NTModel

    def __init__(self, args : ModelParams, primitives : NTModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.args = args
        self.model_path = self.args.model_path
        self.loaded_iter = None
        self.primitives = primitives
        
        if load_iteration:
            self.loaded_iter = searchForMaxIteration(self.model_path)
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self._is_immersive = False

        print("Loading scene from path: {}".format(args.source_path))

        if args.max_dynamic_frames >= 1 and os.path.exists(os.path.join(args.source_path, "models.json")):
            print("Found models.json, assuming Dynamic Immersive dataset!")
            self._is_immersive = True
            scene_info = sceneLoadTypeCallbacks["DynamicImmersive"](args.source_path, args.images, args.eval, omit_cams_frames=args.omit_cams_frames, init_type=args.init_type, max_cap=args.cap_max, max_dynamic_frames=args.max_dynamic_frames, time_duration=None, timestamp_type=args.timestamp_type)
        elif args.max_dynamic_frames >= 1 and os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found poses_bounds.npy file, assuming Dynamic Colmap dataset!")
            scene_info = sceneLoadTypeCallbacks["DynamicColmap"](args.source_path, args.images, args.eval, omit_cams_frames=args.omit_cams_frames, init_type=args.init_type, max_cap=args.cap_max, max_dynamic_frames=args.max_dynamic_frames, time_duration=None, timestamp_type=args.timestamp_type)
        elif args.max_dynamic_frames >= 1 and os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Dynamic Blender dataset!")
            scene_info = sceneLoadTypeCallbacks["DynamicBlender"](args.source_path, args.images, args.eval, omit_cams_frames=args.omit_cams_frames, init_type=args.init_type, max_cap=args.cap_max, max_dynamic_frames=args.max_dynamic_frames, time_duration=None, timestamp_type=args.timestamp_type)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                for cam_id, cam_info_dict in scene_info.test_cameras.items():
                    camlist.extend(cam_info_dict.values())
            if scene_info.train_cameras:
                for cam_id, cam_info_dict in scene_info.train_cameras.items():
                    camlist.extend(cam_info_dict.values())
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print(f"Loading Training Cameras (num cameras: {len(scene_info.train_cameras)}), (num frames: {sum(len(frames) for frames in scene_info.train_cameras.values())})")
            self.train_cameras[resolution_scale] = cameraDict_from_camInfos(scene_info.train_cameras, resolution_scale, args, is_immersive=self._is_immersive)
            print(f"Loading Test Cameras (num cameras: {len(scene_info.test_cameras)}), (num frames: {sum(len(frames) for frames in scene_info.test_cameras.values())})")
            self.test_cameras[resolution_scale] = cameraDict_from_camInfos(scene_info.test_cameras, resolution_scale, args, is_immersive=self._is_immersive)

            # Attach fisheye distortion flow maps to CameraImmersive objects
            # Flow maps are stored on CPU and shared across all frames of the same camera
            if self._is_immersive:
                flow_cache = {}  # camera_name -> CPU tensor, shared across frames
                for cam_dict in [self.train_cameras[resolution_scale], self.test_cameras[resolution_scale]]:
                    for cam_id, cam_frames in cam_dict.items():
                        for frame_id, cam in cam_frames.items():
                            if isinstance(cam, CameraImmersive) and cam.fisheyemapper is None:
                                camera_name = os.path.splitext(cam.image_name)[0]
                                if camera_name not in flow_cache:
                                    flow_cache[camera_name] = getfisheyemapper(args.source_path, camera_name)
                                cam.fisheyemapper = flow_cache[camera_name]
                print(f"[Immersive] Loaded {len(flow_cache)} fisheye flow maps (CPU, shared across frames)")

        if not self.loaded_iter:
            self.primitives.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.cap_max)

    def save(self, iteration, is_best=None):
        if is_best is None:
            torch.save((self.primitives.capture(),
                        iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")
        elif is_best == "test":
            torch.save((self.primitives.capture(), 
                        iteration), self.model_path + "/chkpnt_best_test.pth")
        elif is_best == "train":
            torch.save((self.primitives.capture(), 
                        iteration), self.model_path + "/chkpnt_best_train.pth")
        else:
            torch.save((self.primitives.capture(), 
                        iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")
            print("[WARNING] Model saved, but is_best must be None, 'test', or 'train'.")

    def load(self, checkpoint, is_train=True):
        (primitives_model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        if is_train:
            self.primitives.restore(primitives_model_params, self.primitives.args)
        else:
            self.primitives.restore(primitives_model_params, None)
        return first_iter

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]