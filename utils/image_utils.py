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
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def easy_cmap(x: torch.Tensor):
    x_rgb = torch.zeros((3, x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    x_max, x_min = x.max(), x.min()
    x_normalize = (x - x_min) / (x_max - x_min)
    x_rgb[0] = torch.clamp(x_normalize, 0, 1)
    x_rgb[1] = torch.clamp(x_normalize, 0, 1)
    x_rgb[2] = torch.clamp(x_normalize, 0, 1)
    return x_rgb

def percentile_magnitude_cmap(x: torch.Tensor, colormap='turbo', lower_percentile=1, upper_percentile=99, gamma=0.5):    
    x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute percentiles
    vmin = torch.quantile(x_clean, lower_percentile / 100.0).item()
    vmax = torch.quantile(x_clean, upper_percentile / 100.0).item()
    
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    
    # Clip and normalize
    x_clipped = torch.clamp(x_clean, vmin, vmax)
    x_norm = (x_clipped - vmin) / (vmax - vmin)
    
    # Apply gamma for better visibility
    x_norm = torch.pow(x_norm, gamma)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    x_colored = cmap(x_norm.cpu().numpy())
    x_rgb = torch.from_numpy(x_colored[..., :3]).permute(2, 0, 1).float().to(x.device)
    
    return x_rgb

def make_grid_w_props(images: list, labels: list):
    if len(images) < 2:
        raise ValueError("Need at least 2 images (GT and Render)")
    
    if len(labels) != len(images):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of images ({len(images)})")
    
    C, H, W = images[0].shape
    first_row_images = images[:2]
    second_row_images = images[2:]
    N = len(second_row_images)
    
    if N == 0:
        grid = add_labels_to_row([first_row_images[0], first_row_images[1]], labels[:2])
        return torch.cat(grid, dim=2)

    first_row_width = 2 * W
    second_row_single_width = first_row_width // N
    second_row_single_height = int(H * (second_row_single_width / W))
    
    second_row_resized = []
    for img in second_row_images:
        resized = F.interpolate(
            img.unsqueeze(0), 
            size=(second_row_single_height, second_row_single_width), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        second_row_resized.append(resized)
    
    first_row_with_labels = add_labels_to_row([first_row_images[0], first_row_images[1]], labels[:2])
    second_row_with_labels = add_labels_to_row(second_row_resized, labels[2:])
    
    first_row_concat = torch.cat(first_row_with_labels, dim=2)
    
    second_row_concat = torch.cat(second_row_with_labels, dim=2)
    
    if second_row_concat.shape[2] != first_row_concat.shape[2]:
        second_row_concat = F.interpolate(
            second_row_concat.unsqueeze(0),
            size=(second_row_concat.shape[1], first_row_concat.shape[2]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    grid = torch.cat([first_row_concat, second_row_concat], dim=1)
    
    return grid

def add_labels_to_row(images: list, labels: list):
    labeled_images = []
    for img, label in zip(images, labels):
        img_with_label = img.clone()
        img_clamped = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        img_np = (img_clamped.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = 4
        bg_box = [0, 0, text_width + 2*padding, text_height + 2*padding]
        draw.rectangle(bg_box, fill=(0, 0, 0, 180))      
        draw.text((padding, padding), label, fill=(255, 255, 255), font=font)
        img_with_label = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        img_with_label = img_with_label.to(img.device)
        labeled_images.append(img_with_label)
    
    return labeled_images
    
