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

from errno import EEXIST
from os import makedirs, path
import os
import sys
import random
from datetime import datetime
import numpy as np
import torch

def safe_state(silent, seed=0):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.cuda.set_device(torch.device("cuda:0"))

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    checkpoint_numbers = [0]
    for fname in os.listdir(folder):
        if fname.startswith("chkpnt"):
            # Skip special checkpoint names like chkpnt_best_train or chkpnt_best_test
            if "best_train" in fname or "best_test" in fname:
                continue
            iter_num = int(fname.split("chkpnt")[-1].split(".")[0])
            checkpoint_numbers.append(iter_num)

    return max(checkpoint_numbers)
