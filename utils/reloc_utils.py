from diff_graphixs_rasterization import compute_relocation_graphixs
import torch
import math

N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n+1):
        binoms[n, k] = math.comb(n, k)

# NOTE: this is for Student's t
def compute_relocation_graphixs_cuda(opacity_old, scale_old, nu_degree, N):
    N.clamp_(min=1, max=N_max)
    return compute_relocation_graphixs(opacity_old, scale_old, nu_degree, N, binoms, N_max)