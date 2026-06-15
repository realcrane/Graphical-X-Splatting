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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from pytorch_msssim import ms_ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def get_image_data(renders_dir, gt_dir):
    render_paths = []
    gt_paths = []
    image_names = []

    for cam_id in os.listdir(renders_dir):
        for fname in os.listdir(renders_dir / cam_id):
            if not fname.endswith('.png') and not fname.endswith('.jpg'):
                continue
            render_paths.append(renders_dir / cam_id / fname)
            gt_paths.append(gt_dir / cam_id / fname)
            image_names.append(cam_id + "_" + fname)

    return render_paths, gt_paths, image_names

def evaluate(model_paths, split):

    full_dict = {}
    print("")

    for scene_dir in model_paths:

        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}

        data_root = Path(scene_dir) / split

        for method in os.listdir(data_root):
            print("Method:", method)

            full_dict[scene_dir][method] = {}

            method_dir = data_root / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            render_paths, gt_paths, image_names = get_image_data(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss_vgg = []
            lpipss_alex = []
            ms_ssims = []
            dssims = []

            for idx in tqdm(range(len(render_paths)), desc="Metrics", ncols=80):
                render = Image.open(render_paths[idx])
                gt = Image.open(gt_paths[idx])
                render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

                ssim_val = ssim(render_tensor, gt_tensor)
                psnr_val = psnr(render_tensor, gt_tensor)
                lpips_vgg_val = lpips(render_tensor, gt_tensor, net_type='vgg')
                lpips_alex_val = lpips(render_tensor, gt_tensor, net_type='alex')
                ms_ssims_val = ms_ssim(render_tensor, gt_tensor, data_range=1, size_average=True)
                dssims_val = (1 - ms_ssims_val) / 2

                del render, gt, render_tensor, gt_tensor
                torch.cuda.empty_cache()

                ssims.append(ssim_val * (1 / len(render_paths)))
                psnrs.append(psnr_val * (1 / len(render_paths)))
                lpipss_vgg.append(lpips_vgg_val * (1 / len(render_paths)))
                lpipss_alex.append(lpips_alex_val * (1 / len(render_paths)))
                ms_ssims.append(ms_ssims_val * (1 / len(render_paths)))
                dssims.append(dssims_val * (1 / len(render_paths)))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).sum(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).sum(), ".5"))
            print("  LPIPS_VGG: {:>12.7f}".format(torch.tensor(lpipss_vgg).sum(), ".5"))
            print("  LPIPS_ALEX: {:>12.7f}".format(torch.tensor(lpipss_alex).sum(), ".5"))
            print("  MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).sum(), ".5"))
            print("  DSSIM: {:>12.7f}".format(torch.tensor(dssims).sum(), ".5"))
            print("")

            full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).sum().item(),
                    "PSNR": torch.tensor(psnrs).sum().item(),
                    "LPIPS_VGG": torch.tensor(lpipss_vgg).sum().item(),
                    "LPIPS_ALEX": torch.tensor(lpipss_alex).sum().item(),
                    "MS-SSIM": torch.tensor(ms_ssims).sum().item(),
                    "DSSIM": torch.tensor(dssims).sum().item(),
                })
        with open(scene_dir + f"/{split}_results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        evaluate(args.model_paths, split="train")

    if not args.skip_test:
        evaluate(args.model_paths, split="test")