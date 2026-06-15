import torch
import numpy as np
from typing import Union
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, inverse_translated_sigmoid, get_step_lr_func
from torch import nn
import os
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, AdvancedPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from utils.reloc_utils import compute_relocation_graphixs_cuda
from utils.sghmc import AdamSGHMC
from torch.optim import Adam

class NTModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        # NOTE: make sure freedom of degree is always >= 1.0f
        self.nu_degree_activation = nn.Hardtanh(1, 10000)

        # NOTE: make sure duration is always positive
        self.duration_activation = lambda x: torch.nn.functional.softplus(x, beta=1)
        self.duration_inverse_activation = lambda x: torch.log(torch.expm1(x))

        self.sigmoid_function = torch.sigmoid
        self.inverse_translated_sigmoid = inverse_translated_sigmoid

    def __init__(self, sh_degree, nu_degree, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._base_opacity = torch.empty(0)
        self.likelihoods = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.max_dynamic_frames = args.max_dynamic_frames
        self.setup_functions()

        # NOTE: degree of freedom
        self._degree = torch.empty(0)
        self.nu_degree = nu_degree

        # NOTE: Dynamic scene parameters
        self.order = args.order
        self._origin_time = torch.empty(0)
        self._duration = torch.empty(0)
        self._velocity = torch.empty(0)
        self._acceleration = torch.empty(0)
        self._jerk = torch.empty(0)
        self._snap = torch.empty(0)
        self.learn_motion_from_iter = 0
        
    def capture(self):
        print("[CAPTURE] Capturing NTModel parameters")
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._base_opacity,
            self.likelihoods,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._degree,
            self.order,
            self._origin_time,
            self._duration,
            self._velocity,
            self._acceleration,
            self._jerk,
            self._snap,
        )
    
    def restore(self, model_args, training_args):
        print("[RESTORE] Restoring NTModel parameters")
        (self.active_sh_degree, 
         self._xyz, 
         self._features_dc, 
         self._features_rest,
         self._scaling, 
         self._rotation, 
         self._base_opacity,
         self.likelihoods,
         self.max_radii2D, 
         xyz_gradient_accum, 
         denom,
         opt_dict, 
         self.spatial_lr_scale,
         self._degree,
         self.order,
         self._origin_time,
         self._duration,
         self._velocity,
         self._acceleration,
         self._jerk,
         self._snap) = model_args

        if training_args is not None:
            self.training_setup(training_args)
    
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_base_opacity(self):
        return self.opacity_activation(self._base_opacity)
        
    @property
    def get_likelihoods(self):
        return self.likelihoods

    @property
    def get_nu_degree(self):
        return self.nu_degree_activation(self._degree)

    @property
    def get_origin_time(self):
        return self._origin_time
    
    @property
    def get_duration(self):
        return self.duration_activation(self._duration)
    
    @property
    def get_velocity(self):
        return self._velocity
    
    @property
    def get_acceleration(self):
        return self._acceleration
    
    @property
    def get_jerk(self):
        return self._jerk
    
    @property
    def get_snap(self):
        return self._snap
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : Union[BasicPointCloud, AdvancedPointCloud], spatial_lr_scale : float, cap_max: int=2000000):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        pcd_points_for_dist = np.asarray(pcd.points)

        if isinstance(pcd, AdvancedPointCloud):
            print("AdvancedPointCloud detected. Initializing dynamic scene parameters.")
            times = torch.tensor(np.asarray(pcd.time)).float().cuda()
            velocities = torch.tensor(np.asarray(pcd.velocity)).float().cuda()
            if times.dim() == 1:
                times = times.unsqueeze(-1)
            if velocities.dim() == 1:
                velocities = velocities.unsqueeze(-1)
            
            unique_times = torch.unique(times)
            unique_times_sorted = torch.sort(unique_times)[0]
            
            if len(unique_times_sorted) > self.max_dynamic_frames:
                print(f"Filtering points: Found {len(unique_times_sorted)} frames, keeping first {self.max_dynamic_frames}")
                max_time_threshold = unique_times_sorted[self.max_dynamic_frames - 1]
                
                valid_mask = (times <= max_time_threshold).squeeze()
                valid_mask_cpu = valid_mask.cpu().numpy()
                
                fused_point_cloud = fused_point_cloud[valid_mask]
                fused_color = fused_color[valid_mask]
                features = features[valid_mask]
                times = times[valid_mask]
                velocities = velocities[valid_mask]
                pcd_points_for_dist = pcd_points_for_dist[valid_mask_cpu]
                
                print(f"Filtered from {valid_mask.shape[0]} to {fused_point_cloud.shape[0]} points")
            
            if times.numel() > 0:
                time_min = times.min()
                time_max = times.max()
                if time_max > time_min:
                    times = (times - time_min) / (time_max - time_min)
                    print(f"Rescaled time values from [{time_min:.3f}, {time_max:.3f}] to [0, 1]")
                else:
                    times = torch.zeros_like(times)
                    print("All time values are the same, setting to 0")
            
            accelerations = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            time_bins = 30
            time_indices = (times * (time_bins - 1)).long().clamp(0, time_bins - 1)
                
            for bin_idx in range(time_bins):
                mask = (time_indices == bin_idx).squeeze()
                if mask.sum() > 1:
                    avg_velocity = velocities[mask].mean(dim=0, keepdim=True)
                    velocity_deviation = velocities[mask] - avg_velocity
                    accelerations[mask] = velocity_deviation * 0.1
            
            print(f"Initialized acceleration from velocity data. Mean acceleration magnitude: {torch.norm(accelerations, dim=-1).mean():.3f}, Max magnitude: {torch.norm(accelerations, dim=-1).max():.3f}")
            
            jerks = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            for bin_idx in range(time_bins):
                mask = (time_indices == bin_idx).squeeze()
                if mask.sum() > 1:
                    avg_acceleration = accelerations[mask].mean(dim=0, keepdim=True)
                    acceleration_deviation = accelerations[mask] - avg_acceleration
                    jerks[mask] = acceleration_deviation * 0.1
            
            print(f"Initialized jerk from acceleration data. Mean jerk magnitude: {torch.norm(jerks, dim=-1).mean():.3f}, Max magnitude: {torch.norm(jerks, dim=-1).max():.3f}")
            
            snaps = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            for bin_idx in range(time_bins):
                mask = (time_indices == bin_idx).squeeze()
                if mask.sum() > 1:
                    avg_jerk = jerks[mask].mean(dim=0, keepdim=True)
                    jerk_deviation = jerks[mask] - avg_jerk
                    snaps[mask] = jerk_deviation * 0.1
            
            print(f"Initialized snap from jerk data. Mean snap magnitude: {torch.norm(snaps, dim=-1).mean():.3f}, Max magnitude: {torch.norm(snaps, dim=-1).max():.3f}")

        else:
            print("BasicPointCloud detected. Initialized time, velocity, acceleration, jerk, and snap to zero.")
            times = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            velocities = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
            accelerations = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            jerks = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            snaps = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(pcd_points_for_dist).float().cuda()), 0.0000001)
        if torch.any(dist2.isinf()):
            dist2 = torch.where(dist2.isinf(), 10., dist2)

        scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        degrees = torch.full_like(opacities, self.nu_degree)

        velocity_magnitude = torch.norm(velocities, dim=-1, keepdim=True)
        velocity_mag_max = velocity_magnitude.max()
        velocity_mag_normalized = velocity_magnitude / velocity_mag_max

        duration_max = 8.0 / self.max_dynamic_frames
        duration_min = 2.0 / self.max_dynamic_frames
        duration_values = duration_max - (duration_max - duration_min) * velocity_mag_normalized

        duration_values = torch.clamp(duration_values, min=1e-6)
        durations = self.duration_inverse_activation(duration_values)
        print(f"Initialized duration inversely proportional to velocity magnitude (frames={self.max_dynamic_frames}. Max duration: {duration_values.max():.6f}, Min duration: {duration_values.min():.6f}")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._base_opacity = nn.Parameter(opacities.requires_grad_(True))

        self._degree = nn.Parameter(degrees.requires_grad_(True))

        self._origin_time = nn.Parameter(times.requires_grad_(True))
        self._duration = nn.Parameter(durations.requires_grad_(True))
        self._velocity = nn.Parameter(velocities.requires_grad_(True))
        self._acceleration = nn.Parameter(accelerations.requires_grad_(True))
        self._jerk = nn.Parameter(jerks.requires_grad_(True))
        self._snap = nn.Parameter(snaps.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.likelihoods = torch.zeros((self.get_xyz.shape[0]), requires_grad=False, device="cuda")

        print(f"Shape of _xyz: {self._xyz.shape}")
        print(f"Shape of _origin_time: {self._origin_time.shape}")
        print(f"Shape of _velocity: {self._velocity.shape}")
        print(f"Shape of _acceleration: {self._acceleration.shape}")
        print(f"Shape of _jerk: {self._jerk.shape}")
        print(f"Shape of _snap: {self._snap.shape}")
            
    def training_setup(self, training_args, C_burnin=5e3, C=1.3e2, burnin_iterations=7000, noise_scale=1.0, optimizer_type="sghmc"):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.likelihoods = torch.zeros((self.get_xyz.shape[0]), requires_grad=False, device="cuda")
        self.optimizer_type = optimizer_type
        self.learn_motion_from_iter = training_args.learn_motion_from_iter

        if self.order < 4:
            training_args.snap_lr_init = 0.0
            training_args.snap_lr_final = 0.0
            self._snap.data.zero_()
        if self.order < 3:
            training_args.jerk_lr_init = 0.0
            training_args.jerk_lr_final = 0.0
            self._jerk.data.zero_()
        if self.order < 2:
            training_args.acceleration_lr_init = 0.0
            training_args.acceleration_lr_final = 0.0
            self._acceleration.data.zero_()
        if self.order < 1:
            training_args.velocity_lr_init = 0.0
            training_args.velocity_lr_final = 0.0
            self._velocity.data.zero_()
            
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._base_opacity], 'lr': training_args.base_opacity_lr, "name": "base_opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._degree], 'lr': training_args.degree_lr, "name": "degree"},
            {'params': [self._origin_time], 'lr': training_args.origin_time_lr_init , "name": "origin_time"},
            {'params': [self._duration], 'lr': training_args.duration_lr_init , "name": "duration"},
            {'params': [self._velocity], 'lr': training_args.velocity_lr_init , "name": "velocity"},
            {'params': [self._acceleration], 'lr': training_args.acceleration_lr_init , "name": "acceleration"},
            {'params': [self._jerk], 'lr': training_args.jerk_lr_init , "name": "jerk"},
            {'params': [self._snap], 'lr': training_args.snap_lr_init , "name": "snap"},
        ]

        if optimizer_type.lower() == "adam":
            print("[INFO] Using Adam optimizer")
            self.optimizer = Adam(l, eps=1e-15)
        elif optimizer_type.lower() == "sghmc":
            print(f"[INFO] Using SGHMC optimizer with noise scale: {noise_scale}")
            self.optimizer = AdamSGHMC(params=l, eps=1e-15, mdecay=C, scale_grad=1.0, mdecay_burnin=C_burnin, burnin_iterations=burnin_iterations, noise_scale=noise_scale)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Supported types are 'sghmc' and 'adam'.")
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.velocity_scheduler_args = get_expon_lr_func(lr_init=training_args.velocity_lr_init,
                                                    lr_final=training_args.velocity_lr_final,
                                                    max_steps=training_args.velocity_lr_max_steps)
        self.acceleration_scheduler_args = get_expon_lr_func(lr_init=training_args.acceleration_lr_init,
                                                    lr_final=training_args.acceleration_lr_final,
                                                    max_steps=training_args.acceleration_lr_max_steps)
        self.jerk_scheduler_args = get_expon_lr_func(lr_init=training_args.jerk_lr_init,
                                                    lr_final=training_args.jerk_lr_final,
                                                    max_steps=training_args.jerk_lr_max_steps)
        self.snap_scheduler_args = get_expon_lr_func(lr_init=training_args.snap_lr_init,
                                                    lr_final=training_args.snap_lr_final,
                                                    max_steps=training_args.snap_lr_max_steps)
        self.time_scheduler_args = get_expon_lr_func(lr_init=training_args.origin_time_lr_init,
                                                    lr_final=training_args.origin_time_lr_final,
                                                    max_steps=training_args.origin_time_lr_max_steps)
        self.duration_scheduler_args = get_expon_lr_func(lr_init=training_args.duration_lr_init,
                                                    lr_final=training_args.duration_lr_final,
                                                    max_steps=training_args.duration_lr_max_steps)
        self.degree_scheduler_args = get_step_lr_func(lr_init=training_args.degree_lr, step_size=5000, gamma=0.5)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "degree":
                lr = self.degree_scheduler_args(iteration)
                param_group['lr'] = lr
                break
    
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                break
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "velocity":
                if iteration < self.learn_motion_from_iter:
                    param_group['lr'] = 0.0
                else:
                    lr = self.velocity_scheduler_args(iteration)
                    param_group['lr'] = lr
                break 
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "acceleration":
                if iteration < self.learn_motion_from_iter:
                    param_group['lr'] = 0.0
                else:
                    lr = self.acceleration_scheduler_args(iteration)
                    param_group['lr'] = lr
                break
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "jerk":
                if iteration < self.learn_motion_from_iter:
                    param_group['lr'] = 0.0
                else:
                    lr = self.jerk_scheduler_args(iteration)
                    param_group['lr'] = lr
                break

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "snap":
                if iteration < self.learn_motion_from_iter:
                    param_group['lr'] = 0.0
                else:
                    lr = self.snap_scheduler_args(iteration)
                    param_group['lr'] = lr
                break

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "origin_time":
                lr = self.time_scheduler_args(iteration)
                param_group['lr'] = lr
                break
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "duration":
                lr = self.duration_scheduler_args(iteration)
                param_group['lr'] = lr
                break

    def cat_tensors_to_optimizer(self, tensors_dict, inds, opt):
        optimizable_tensors = {}
        for group in opt.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = opt.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                
                if self.optimizer_type == "sghmc":
                    stored_state["momentum"] = torch.cat((stored_state["momentum"], torch.zeros_like(extension_tensor)), dim=0)

                del opt.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                opt.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_likelihood, new_scaling, new_rotation, new_degree, new_time, new_duration, new_velocity, new_acceleration, new_jerk, new_snap, indices=None, reset_params=False, reset_likelihoods=True):
        print(f"Total gaussians before densification: {self.get_xyz.shape[0]}")
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "base_opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "degree" : new_degree,
        "origin_time": new_time,
        "duration": new_duration,
        "velocity": new_velocity,
        "acceleration": new_acceleration,
        "jerk": new_jerk,
        "snap": new_snap}

        optimizable_tensors = self.cat_tensors_to_optimizer(d, indices, self.optimizer)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._base_opacity = optimizable_tensors["base_opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._degree = optimizable_tensors["degree"]

        self._origin_time = optimizable_tensors["origin_time"]
        self._duration = optimizable_tensors["duration"]
        self._velocity = optimizable_tensors["velocity"]
        self._acceleration = optimizable_tensors["acceleration"]
        self._jerk = optimizable_tensors["jerk"]
        self._snap = optimizable_tensors["snap"]
        if reset_likelihoods:
            self.likelihoods = torch.zeros((self.get_xyz.shape[0]), requires_grad=False, device=self.get_xyz.device)
        else:
            self.likelihoods = torch.cat((self.likelihoods, new_likelihood), dim=0)

        print(f"Total gaussians after densification: {self.get_xyz.shape[0]}")

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device)
        else:
            num_new = new_xyz.shape[0]
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((num_new, 1), device=self.get_xyz.device)], dim=0)
            self.denom = torch.cat([self.denom, torch.zeros((num_new, 1), device=self.get_xyz.device)], dim=0)
            self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((num_new), device=self.get_xyz.device)], dim=0)

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "base_opacity": self._base_opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation,
            "degree" : self._degree,
            "origin_time": self._origin_time,
            "duration": self._duration,
            "velocity": self._velocity,
            "acceleration": self._acceleration,
            "jerk": self._jerk,
            "snap": self._snap
            }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._base_opacity = optimizable_tensors["base_opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._degree = optimizable_tensors["degree"]

        self._origin_time = optimizable_tensors["origin_time"]
        self._duration = optimizable_tensors["duration"]
        self._velocity = optimizable_tensors["velocity"]
        self._acceleration = optimizable_tensors["acceleration"]
        self._jerk = optimizable_tensors["jerk"]
        self._snap = optimizable_tensors["snap"]

        return optimizable_tensors

    def replace_tensors_to_optimizer_momentum(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "base_opacity": self._base_opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation,
            "degree" : self._degree,
            "origin_time": self._origin_time,
            "duration": self._duration,
            "velocity": self._velocity,
            "acceleration": self._acceleration,
            "jerk": self._jerk,
            "snap": self._snap
            }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                if inds is not None:
                    # NOTE: do not reset Adam momentum to avoid overlap (like adding noise)
                    
                    if self.optimizer_type == "sghmc":
                        stored_state["momentum"][inds] = 0

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._base_opacity = optimizable_tensors["base_opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._degree = optimizable_tensors["degree"]

        self._origin_time = optimizable_tensors["origin_time"]
        self._duration = optimizable_tensors["duration"]
        self._velocity = optimizable_tensors["velocity"]
        self._acceleration = optimizable_tensors["acceleration"]
        self._jerk = optimizable_tensors["jerk"]
        self._snap = optimizable_tensors["snap"]

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_graphixs_cuda(
            opacity_old = self.get_base_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            nu_degree = self.get_nu_degree[idxs, 0],
            N=ratio[idxs, 0] + 1
        )

        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max = 1.0 - torch.finfo(torch.float32).eps, min = 0.005)

        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return (
            self._xyz[idxs],
            self._features_dc[idxs],
            self._features_rest[idxs],
            new_opacity,
            self.likelihoods[idxs],
            new_scaling,
            self._rotation[idxs],
            self._degree[idxs],
            self._origin_time[idxs],
            self._duration[idxs],
            self._velocity[idxs],
            self._acceleration[idxs],
            self._jerk[idxs],
            self._snap[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    
    def add_densification_stats(self, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            self._xyz.grad[update_filter, :], 
            dim=-1, 
            keepdim=True
        )
        self.denom[update_filter] += 1

    def recycle_components_temporal(self, dead_mask=None, lambda_g=0.5, lambda_o=0.5):
        if dead_mask is None or dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        # Only recycle 5% of all components at max
        if dead_mask.sum() > int(0.05 * self.get_base_opacity.shape[0]):
            sorted_vals, indices = torch.sort(self.get_base_opacity.squeeze(-1))
            dead_indices = indices[0: int(0.05 * self.get_base_opacity.shape[0])]

        if alive_indices.shape[0] <= 0:
            return

        avg_gradient = self.xyz_gradient_accum[alive_indices] / (self.denom[alive_indices] + 1e-8)
        gradient_norm = avg_gradient / (avg_gradient.max() + torch.finfo(torch.float32).eps)
        opacity_alive = self.get_base_opacity[alive_indices]
        
        # Combined sampling score: gradient + opacity
        sampling_score = lambda_g * gradient_norm + lambda_o * opacity_alive
        sampling_score = sampling_score.squeeze(-1)
        probs = sampling_score / (sampling_score.sum() + torch.finfo(torch.float32).eps)
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        # Update dead Gaussians with parameters from high-score regions
        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._base_opacity[dead_indices],
            self.likelihoods[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
            self._degree[dead_indices],
            self._origin_time[dead_indices],
            self._duration[dead_indices],
            self._velocity[dead_indices],
            self._acceleration[dead_indices],
            self._jerk[dead_indices],
            self._snap[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._base_opacity[reinit_idx] = self._base_opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)
        self.replace_tensors_to_optimizer_momentum(inds=dead_indices)
        
    def add_components_temporal(self, cap_max, lambda_g=0.5, lambda_o=0.5):
        current_num_points = self._base_opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0
        
        avg_gradient = self.xyz_gradient_accum / (self.denom + 1e-8)
        gradient_norm = avg_gradient / (avg_gradient.max() + torch.finfo(torch.float32).eps)
        opacity_values = self.get_base_opacity
        
        # Combined sampling score: gradient + opacity
        sampling_score = lambda_g * gradient_norm + lambda_o * opacity_values
        sampling_score = sampling_score.squeeze(-1)
        probs = sampling_score / (sampling_score.sum() + torch.finfo(torch.float32).eps)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_likelihood,
            new_scaling,
            new_rotation,
            new_degree,
            new_time,
            new_duration,
            new_velocity,
            new_acceleration,
            new_jerk,
            new_snap,
        ) = self._update_params(add_idx, ratio=ratio)

        self._base_opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_likelihood, new_scaling, new_rotation, new_degree, new_time, new_duration, new_velocity, new_acceleration, new_jerk, new_snap, add_idx, reset_params=False, reset_likelihoods=True)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs