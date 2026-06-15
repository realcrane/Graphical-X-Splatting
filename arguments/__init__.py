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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            original_value = value
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    if original_value:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_false")
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if original_value:
                        group.add_argument("--" + key, default=value, action="store_false")
                    else:
                        group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        # Also set defaults for any attributes in self that weren't in args
        for key in vars(self):
            attr_name = key[1:] if key.startswith("_") else key
            if not hasattr(group, attr_name):
                setattr(group, attr_name, getattr(self, key))
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.timestamp_type = "normalized"
        self.test_frames = [0]
        self.omit_cams_frames = None
        self.max_dynamic_frames = 1

        self.data_device = "cuda"
        self.eval = True

        self.nu_degree = 10.0
        self.order = 4

        self.cap_max = -1
        self.init_type = "sfm_4D"

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        # default
        self.batch_size = 2

        # optimizer hyperparameters
        self.optimizer_type = "sghmc"  # options: "sghmc", "adam"
        self.optimizer_noise_scale = 1.0
        self.C_burnin = 5e5
        self.C = 1e2
        self.burnin_iterations = 7000
        self.opacity_threshold = 0.005

        self.position_lr_init = 0.001
        self.position_lr_final = 0.00001
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.base_opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.origin_time_lr_init = 0.001
        self.origin_time_lr_final = 0.00001
        self.origin_time_lr_max_steps = 30_000
        self.duration_lr_init = 0.01
        self.duration_lr_final = 0.00001
        self.duration_lr_max_steps = 30_000
        self.velocity_lr_init = 0.01
        self.velocity_lr_final = 0.0001
        self.velocity_lr_max_steps = 30_000
        self.acceleration_lr_init = 0.01
        self.acceleration_lr_final = 0.0001
        self.acceleration_lr_max_steps = 30_000
        self.jerk_lr_init = 0.01
        self.jerk_lr_final = 0.0001
        self.jerk_lr_max_steps = 30_000
        self.snap_lr_init = 0.01
        self.snap_lr_final = 0.0001
        self.snap_lr_max_steps = 30_000
        self.dynamics_noise_std = 0.0001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.densify_from_iter = 100
        self.densify_until_iter = 25_000
        self.learn_motion_from_iter = 800
        self.random_background = False

        self.degree_lr = 10.0

        self.scale_reg = 0.01
        self.velocity_reg = 0.001
        self.acceleration_reg = 0.001
        self.jerk_reg = 0.001
        self.snap_reg = 0.001
        self.temporal_opacity_reg = 0.01
        # L_img regularizers (eq. imageDis): opacity-L1 (eps_o) and sqrt-eigenvalue-L1 (eps_Sigma).
        # Paper gives no values; small defaults, tune per scene.
        self.opacity_l1_reg = 1e-3
        self.eigenvalue_l1_reg = 1e-3
        self.likelihood_reg = 1e-8
        self.likelihood_epsilon = 1e-8
        self.recycle_lambda_g = 0.5
        self.recycle_lambda_o = 0.5

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)